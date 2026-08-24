import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.knn_advisor_v2 import KNNAdvisorV2Level


def _outcome(student_id, part_id, agpa, diploma, first, level):
    return {
        "university_id": "111",
        "student_id": student_id,
        "degree_id": "d1",
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
        "academic_level_before": level,
        "academic_level_after": level,
        "academic_level_delta": 0,
        "academic_level_advanced": 0,
    }


def _course(student_id, part_id):
    return {
        "student_course_id": f"{student_id}-{part_id}",
        "university_id": "111",
        "student_id": student_id,
        "degree_id": "d1",
        "part_id": part_id,
        "course_id": "course-1",
        "course_credits": 3.0,
        "attempt_number": 1,
        "final_mark": 70,
        "finish_status": "P",
        "is_passed": 1,
        "is_failed": 0,
    }


class KNNAdvisorV2LevelTest(unittest.TestCase):
    def setUp(self):
        outcomes = pd.DataFrame(
            [
                _outcome("s1", "20231", 2.40, 80.0, 0, 2),
                _outcome("s1", "20232", 2.41, 80.0, 0, 2),
                _outcome("s2", "20232", 2.55, 70.0, 0, 2),
                _outcome("s3", "20231", 2.42, 80.0, 0, 3),
                _outcome("cs1", "20231", 0.00, 83.0, 1, 1),
            ]
        )
        courses = pd.DataFrame(
            [
                _course(row.student_id, row.part_id)
                for row in outcomes.itertuples(index=False)
            ]
        )
        self.advisor = KNNAdvisorV2Level.build(outcomes, courses)

    def test_exact_level_and_one_snapshot_per_student(self):
        result = self.advisor.find_nearest_gpa(
            degree_id="d1",
            academic_level=2,
            gpa=2.42,
            k=10,
        )

        self.assertEqual(list(result["student_id"]), ["s1", "s2"])
        self.assertEqual(result["student_id"].nunique(), len(result))
        self.assertEqual(set(result["matched_academic_level"]), {2})
        self.assertNotIn("s3", set(result["student_id"]))
        self.assertEqual(set(result["level_fallback_used"]), {0})

    def test_cold_start_uses_exact_start_level_and_diploma_gpa(self):
        result = self.advisor.find_nearest_gpa(
            degree_id="d1",
            academic_level=1,
            gpa=82.5,
            cold_start=True,
            k=5,
        )

        self.assertEqual(list(result["student_id"]), ["cs1"])
        self.assertEqual(result["matched_gpa"].iloc[0], 83.0)

    def test_level_aware_artifact_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "knn-level.pkl"
            self.advisor.save(path)
            loaded = KNNAdvisorV2Level.load(path)
            result = loaded.find_nearest_gpa(
                degree_id="d1",
                academic_level=3,
                gpa=2.42,
                k=5,
            )

        self.assertEqual(list(result["student_id"]), ["s3"])
        self.assertEqual(loaded.metadata["level_fallback"], "none_exact_level_only")


if __name__ == "__main__":
    unittest.main()
