import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.knn_advisor_v2 import (
    COLD_START_ROUTE,
    KNNAdvisorV2,
    RETURNING_ROUTE,
)


def _outcome(student_id, degree_id, part_id, agpa, diploma, first):
    return {
        "university_id": "111",
        "student_id": student_id,
        "degree_id": degree_id,
        "part_id": part_id,
        "cumulative_gpa_before": agpa,
        "diploma_gpa": diploma,
        "is_first_active_semester": first,
        "semester_average_mark": 70.0,
        "term_gpa": 2.5,
        "term_gpa_delta": 0.2,
        "cumulative_gpa_delta": 0.1,
        "term_gpa_improved": 1,
        "cumulative_gpa_improved": 1,
        "any_course_failed": 0,
        "all_courses_passed": 1,
    }


def _course(student_id, degree_id, part_id):
    return {
        "student_course_id": f"{student_id}-{part_id}",
        "university_id": "111",
        "student_id": student_id,
        "degree_id": degree_id,
        "part_id": part_id,
        "course_id": "course-1",
        "course_credits": 3.0,
        "attempt_number": 1,
        "final_mark": 70,
        "finish_status": "P",
        "is_passed": 1,
        "is_failed": 0,
    }


class KNNAdvisorV2Test(unittest.TestCase):
    def setUp(self):
        outcomes = pd.DataFrame(
            [
                _outcome("s1", "d1", "20231", 2.40, 80.0, 0),
                _outcome("s2", "d1", "20232", 2.55, 70.0, 0),
                _outcome("s3", "d1", "20233", 0.00, 83.0, 1),
                _outcome("s4", "d2", "20231", 2.49, 82.0, 0),
            ]
        )
        courses = pd.DataFrame(
            [
                _course(row.student_id, row.degree_id, row.part_id)
                for row in outcomes.itertuples(index=False)
            ]
        )
        self.advisor = KNNAdvisorV2.build(outcomes, courses)

    def test_returning_finds_nearest_gpa_only_inside_degree(self):
        result = self.advisor.find_nearest_gpa(
            degree_id="d1", gpa=2.50, cold_start=False, k=2
        )

        self.assertEqual(list(result["student_id"]), ["s2", "s1"])
        self.assertEqual(set(result["degree_id"]), {"d1"})
        self.assertEqual(set(result["knn_route"]), {RETURNING_ROUTE})

    def test_cold_start_uses_diploma_gpa(self):
        result = self.advisor.find_nearest_gpa(
            degree_id="d1", gpa=82.5, cold_start=True, k=5
        )

        self.assertEqual(list(result["student_id"]), ["s3"])
        self.assertEqual(result["matched_gpa"].iloc[0], 83.0)
        self.assertEqual(result["knn_route"].iloc[0], COLD_START_ROUTE)

    def test_student_can_be_excluded_and_artifact_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "knn.pkl"
            self.advisor.save(path)
            loaded = KNNAdvisorV2.load(path)
            result = loaded.find_nearest_gpa(
                degree_id="d1",
                gpa=2.40,
                k=2,
                exclude_student_id="s1",
            )

        self.assertEqual(list(result["student_id"]), ["s2"])
        neighbour_courses = loaded.courses_for_neighbours(result)
        self.assertEqual(list(neighbour_courses["student_id"]), ["s2"])


if __name__ == "__main__":
    unittest.main()
