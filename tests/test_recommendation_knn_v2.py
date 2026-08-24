import unittest

import pandas as pd

from src.grade_scale import GradeScale
from src.recommendation import Recommender


def _grade_scale() -> GradeScale:
    rows = [
        ("985.111", 0, 49, 0.00, "F", "F"),
        ("984.111", 50, 54, 1.50, "P", "D"),
        ("983.111", 55, 59, 1.75, "P", "D+"),
        ("982.111", 60, 64, 2.00, "P", "C-"),
        ("981.111", 65, 69, 2.25, "P", "C"),
        ("980.111", 70, 74, 2.50, "P", "C+"),
        ("979.111", 75, 79, 2.75, "P", "B-"),
        ("978.111", 80, 84, 3.00, "P", "B"),
        ("977.111", 85, 89, 3.25, "P", "B+"),
        ("976.111", 90, 94, 3.50, "P", "A-"),
        ("975.111", 95, 97, 3.75, "P", "A"),
        ("974.111", 98, 100, 4.00, "P", "A+"),
    ]
    return GradeScale(
        "3.111",
        pd.DataFrame(
            rows,
            columns=[
                "grade_id",
                "from_percent",
                "to_percent",
                "points",
                "finish_status",
                "grade_show",
            ],
        ),
    )


class _FakeScorer:
    def extract_snapshot(self, df_history: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "student_id": "current-student",
                "start_agpa_points": 2.40,
                "start_level_ord": 2,
                "diploma_gpa": 84.0,
            }
        )

    def score(self, *, candidate_course_ids, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "pred_mark": [75.0 for _ in candidate_course_ids],
                "pass_prob": [0.90 for _ in candidate_course_ids],
                "course_credits": [3.0 for _ in candidate_course_ids],
            },
            index=candidate_course_ids,
        )

    def score_plan(self, *, plan_course_ids, **kwargs) -> pd.DataFrame:
        return self.score(candidate_course_ids=plan_course_ids)


class _FakeKNNV2Level:
    def __init__(self, *, return_empty: bool = False) -> None:
        self.query = None
        self.prediction_query = None
        self.return_empty = return_empty

    def find_nearest_gpa(self, **kwargs) -> pd.DataFrame:
        self.query = kwargs
        if self.return_empty:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "university_id": "111",
                    "student_id": "neighbour-1",
                    "degree_id": "degree-1",
                    "part_id": "20241",
                    "matched_gpa": 2.68,
                    "gpa_distance": 0.02,
                    "knn_route": "returning_degree_level_cumulative_gpa",
                    "term_gpa": 2.8,
                    "term_gpa_delta": 0.1,
                    "cumulative_gpa_delta": 0.05,
                    "term_gpa_improved": 1,
                    "cumulative_gpa_improved": 1,
                    "any_course_failed": 0,
                    "all_courses_passed": 1,
                }
            ]
        )

    def predict(self, **kwargs) -> dict:
        self.prediction_query = kwargs
        if self.return_empty:
            return {
                "covered": False,
                "support": 0,
                "knn_route": None,
                "predicted_any_course_failed": None,
                "failure_probability": None,
                "predicted_term_gpa": None,
                "predicted_semester_average_mark": None,
            }
        return {
            "covered": True,
            "support": 1,
            "knn_route": "returning_degree_level_cumulative_gpa",
            "predicted_any_course_failed": 0,
            "failure_probability": 0.0,
            "predicted_term_gpa": 2.8,
            "predicted_semester_average_mark": 76.0,
        }

    def summarize(self, neighbours: pd.DataFrame) -> dict:
        if neighbours.empty:
            return {
                "support": 0,
                "knn_route": None,
                "mean_matched_gpa": None,
                "median_gpa_distance": None,
                "mean_term_gpa": None,
                "mean_term_gpa_delta": None,
                "mean_cumulative_gpa_delta": None,
                "pct_term_gpa_improved": None,
                "pct_cumulative_gpa_improved": None,
                "pct_any_course_failed": None,
            }
        return {
            "support": 1,
            "knn_route": "returning_degree_level_cumulative_gpa",
            "mean_matched_gpa": 2.68,
            "median_gpa_distance": 0.02,
            "mean_term_gpa": 2.8,
            "mean_term_gpa_delta": 0.1,
            "mean_cumulative_gpa_delta": 0.05,
            "pct_term_gpa_improved": 1.0,
            "pct_cumulative_gpa_improved": 1.0,
            "pct_any_course_failed": 0.0,
        }

    def courses_for_neighbours(self, neighbours: pd.DataFrame) -> pd.DataFrame:
        if neighbours.empty:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "university_id": "111",
                    "student_id": "neighbour-1",
                    "degree_id": "degree-1",
                    "part_id": "20241",
                    "course_id": "course-1",
                    "final_mark": 76.0,
                    "is_passed": 1,
                }
            ]
        )


class RecommendationKNNV2IntegrationTest(unittest.TestCase):
    def test_recommender_rejects_a_non_3_111_grade_scale(self):
        scale = _grade_scale()
        scale.grade_version_id = "4.111"

        with self.assertRaisesRegex(ValueError, "3.111"):
            Recommender(_FakeScorer(), _FakeKNNV2Level(), scale)

    def test_recommend_queries_level_knn_with_latest_official_state(self):
        history = pd.DataFrame(
            [
                {
                    "student_id": "current-student",
                    "part_id": "20231",
                    "end_agpa_points": 2.50,
                    "end_level_name_short": 2,
                },
                {
                    "student_id": "current-student",
                    "part_id": "20232",
                    "end_agpa_points": 2.70,
                    "end_level_name_short": 3,
                },
            ]
        )
        knn = _FakeKNNV2Level()
        recommender = Recommender(_FakeScorer(), knn, _grade_scale())

        plans = recommender.recommend(
            df_history=history,
            candidate_course_ids=["course-1"],
            target_part_id="20241",
            degree_id="degree-1",
            n_plans=1,
            top_k=1,
        )

        self.assertEqual(knn.query["degree_id"], "degree-1")
        self.assertEqual(knn.query["academic_level"], 3)
        self.assertEqual(knn.query["gpa"], 2.70)
        self.assertFalse(knn.query["cold_start"])
        self.assertEqual(knn.query["k"], 20)
        self.assertEqual(knn.query["exclude_student_id"], "current-student")
        self.assertEqual(knn.prediction_query["degree_id"], "degree-1")
        self.assertEqual(knn.prediction_query["academic_level"], 3)
        self.assertEqual(plans[0]["knn_support"], 1)
        self.assertEqual(plans[0]["knn_avg_pass_rate"], 1.0)
        self.assertEqual(plans[0]["knn_failure_probability"], 0.0)
        self.assertEqual(plans[0]["knn_predicted_term_gpa"], 2.8)
        self.assertEqual(plans[0]["knn_similar_plan_avg_mark"], 76.0)
        self.assertEqual(plans[0]["expected_agpa"], 2.75)

    def test_explicit_cold_start_uses_diploma_gpa(self):
        history = pd.DataFrame(
            [{"student_id": "new-student", "part_id": "20241", "diploma_gpa": 84.0}]
        )
        knn = _FakeKNNV2Level()
        recommender = Recommender(_FakeScorer(), knn, _grade_scale())

        recommender.recommend(
            df_history=history,
            candidate_course_ids=["course-1"],
            target_part_id="20241",
            degree_id="degree-1",
            n_plans=1,
            top_k=1,
            cold_start=True,
            academic_level=1,
        )

        self.assertTrue(knn.query["cold_start"])
        self.assertEqual(knn.query["gpa"], 84.0)
        self.assertEqual(knn.query["academic_level"], 1)

    def test_no_exact_degree_level_neighbours_keeps_recommendation_available(self):
        history = pd.DataFrame(
            [
                {
                    "student_id": "current-student",
                    "part_id": "20232",
                    "end_agpa_points": 2.70,
                    "end_level_name_short": 6,
                }
            ]
        )
        recommender = Recommender(
            _FakeScorer(), _FakeKNNV2Level(return_empty=True), _grade_scale()
        )

        plans = recommender.recommend(
            df_history=history,
            candidate_course_ids=["course-1"],
            target_part_id="20241",
            degree_id="degree-without-level-history",
            n_plans=1,
            top_k=1,
        )

        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0]["knn_support"], 0)
        self.assertIsNone(plans[0]["knn_route"])


if __name__ == "__main__":
    unittest.main()
