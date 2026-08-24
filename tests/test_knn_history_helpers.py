import unittest

import pandas as pd

from src.knn_history_helpers import (
    attach_official_references,
    build_student_semester_courses,
    build_student_semester_outcomes,
)


def _row(part_id, attempt_number, mark, *, grade_id, status_id):
    return {
        "student_course_id": f"sc-{part_id}",
        "student_status_id": status_id,
        "university_id": "111",
        "student_id": "student-1",
        "degree_id": "degree-1",
        "part_id": part_id,
        "course_id": "course-1",
        "grade_id": grade_id,
        "final_mark": mark,
        "course_credits": 3.0,
        "attempt_number": attempt_number,
        "prev_gpa_points": 2.0,
        "gpa_points": 2.5,
        "start_agpa_points": 2.2,
        "start_total_in_courses": 10.0,
        "start_total_in_credits": 30.0,
        "semester_reg_credits": 3.0,
        "semester_reg_courses": 1.0,
        "semester_pass_credits": 3.0 if mark >= 50 else 0.0,
        "total_fail_credits": 3.0,
        "reg_total_semesters": float(attempt_number),
        "start_level_ord": 2,
        "is_first_active_semester": 0,
        "model_prev_gpa": 2.0,
        "last_valid_gpa_before_current_semester": 2.0,
        "fail_credit_ratio_capped": 0.1,
        "prior_interruption_count": 0,
        "consecutive_interruption_count": 0,
        "prev_semester_was_interruption": 0,
        "part_semester": int(part_id[-1]),
        "diploma_gpa": 80.0,
        "diploma_type_bucket": 15,
    }


class KNNHistoryHelpersTest(unittest.TestCase):
    def setUp(self):
        train = pd.DataFrame(
            [
                _row("20231", 1, 48, grade_id="2.111", status_id="11.111"),
                _row("20232", 2, 67, grade_id="1.111", status_id="12.111"),
                _row("20233", 3, 55, grade_id="1.111", status_id="13.111"),
            ]
        )
        status = pd.DataFrame(
            {
                "student_status_id": [11.111, 12.111, 13.111],
                "end_agpa_points": [2.0, 2.3, 2.4],
                "end_total_in_courses": [10, 11, 12],
                "end_total_in_credits": [30, 33, 36],
                "start_level_name_short": [2, 2, 2],
                "end_level_name_short": [2, 3, 3],
            }
        )
        grades = pd.DataFrame(
            {
                "grade_id": [1.111, 2.111],
                "finish_status": ["P", "F"],
                "grade_show": ["C", "F"],
            }
        )
        self.enriched = attach_official_references(train, status, grades)

    def test_course_history_uses_only_prior_attempts(self):
        courses = build_student_semester_courses(self.enriched)
        third = courses.loc[courses["part_id"].eq("20233")].iloc[0]

        self.assertEqual(third["course_attempts_prior"], 2)
        self.assertEqual(third["course_max_attempt_number_prior"], 2)
        self.assertEqual(third["course_last_mark_prior"], 67)
        self.assertEqual(third["course_best_mark_prior"], 67)
        self.assertEqual(third["course_mean_mark_prior"], 57.5)
        self.assertEqual(third["course_failures_prior"], 1)
        self.assertEqual(third["course_last_status_prior"], "passed")
        self.assertEqual(third["course_last_attempt_part"], "20232")
        self.assertEqual(third["is_retake"], 1)

    def test_first_attempt_has_empty_prior_history(self):
        courses = build_student_semester_courses(self.enriched)
        first = courses.loc[courses["part_id"].eq("20231")].iloc[0]

        self.assertEqual(first["course_attempts_prior"], 0)
        self.assertEqual(first["course_failures_prior"], 0)
        self.assertTrue(pd.isna(first["course_last_mark_prior"]))
        self.assertEqual(first["is_retake"], 0)
        self.assertEqual(first["outcome_status"], "failed")

    def test_semester_outcome_uses_official_end_agpa(self):
        semesters = build_student_semester_outcomes(self.enriched)
        second = semesters.loc[semesters["part_id"].eq("20232")].iloc[0]

        self.assertEqual(second["cumulative_gpa_before"], 2.2)
        self.assertEqual(second["cumulative_gpa_after"], 2.3)
        self.assertAlmostEqual(second["cumulative_gpa_delta"], 0.1)
        self.assertEqual(second["academic_level_before"], 2)
        self.assertEqual(second["academic_level_after"], 3)
        self.assertEqual(second["academic_level_delta"], 1)
        self.assertEqual(second["academic_level_advanced"], 1)
        self.assertEqual(second["failed_course_count"], 0)
        self.assertEqual(second["all_courses_passed"], 1)


if __name__ == "__main__":
    unittest.main()
