"""Regression tests for feature-engineering contracts that protect model safety.

The cases here focus on timeline grouping, previous-GPA fallback order,
policy exclusions, and leakage prevention because those behaviours directly
affect training reliability and inference correctness.
"""

import unittest

import numpy as np
import pandas as pd

from src.feature_engineering import run_feature_engineering_job


def _base_row(**overrides):
    # Keep a complete minimal course-row fixture so each test can override only
    # the field involved in the behaviour it protects.
    row = {
        "university_id": "111",
        "student_id": "student",
        "degree_id": "degree",
        "part_id": "20201",
        "course_id": "course",
        "course_credits": 3,
        "semester_reg_credits": 12,
        "semester_reg_courses": 4,
        "semester_pass_credits": 12,
        "gpa_points": 2.5,
        "start_total_in_courses": 0,
        "start_total_in_credits": 0,
        "start_agpa_points": 0,
        "prev_gpa_points": np.nan,
        "total_fail_credits": 0,
        "start_part_id": "20201",
        "start_level_name_pl": "first year",
        "requirement_type_id": 1,
        "degree_requirement_credits_count": 12,
    }
    row.update(overrides)
    return row


class FeatureEngineeringJobTest(unittest.TestCase):
    # These tests are intentionally behaviour-level: they treat the public
    # feature-engineering job as the contract instead of testing private helpers.

    def test_over_policy_semester_remains_in_gpa_history_before_split(self):
        df = pd.DataFrame(
            [
                _base_row(
                    student_id="s1",
                    part_id="20201",
                    course_id="c1",
                    semester_reg_credits=30,
                    semester_reg_courses=9,
                    gpa_points=3.4,
                ),
                _base_row(
                    student_id="s1",
                    part_id="20202",
                    course_id="c2",
                    semester_reg_credits=12,
                    semester_reg_courses=4,
                    gpa_points=2.7,
                    start_total_in_courses=4,
                    start_total_in_credits=12,
                ),
            ]
        )

        result = run_feature_engineering_job(df)
        primary = result["df_primary"]

        self.assertEqual(len(result["df_excluded_over_policy"]), 1)
        self.assertEqual(len(primary), 1)
        self.assertAlmostEqual(
            primary["last_valid_gpa_before_current_semester"].iloc[0],
            3.4,
        )
        self.assertEqual(
            primary["prev_gpa_fill_source"].iloc[0],
            "last_valid_gpa_before_current_semester",
        )
        self.assertAlmostEqual(primary["prev_gpa_points_clean"].iloc[0], 3.4)

    def test_interruption_semester_does_not_become_valid_previous_gpa(self):
        df = pd.DataFrame(
            [
                _base_row(student_id="s2", part_id="20201", course_id="c1", gpa_points=2.0),
                _base_row(
                    student_id="s2",
                    part_id="20202",
                    course_id="c2",
                    semester_reg_credits=12,
                    semester_pass_credits=0,
                    gpa_points=0,
                    start_total_in_courses=4,
                    start_total_in_credits=12,
                ),
                _base_row(
                    student_id="s2",
                    part_id="20203",
                    course_id="c3",
                    gpa_points=2.8,
                    start_total_in_courses=4,
                    start_total_in_credits=12,
                ),
            ]
        )

        result = run_feature_engineering_job(df)
        audit = result["df_model_audit"].sort_values("part_id").reset_index(drop=True)
        current = audit.loc[audit["part_id"].eq("20203")].iloc[0]

        self.assertEqual(int(audit.loc[audit["part_id"].eq("20202"), "is_interruption_semester"].iloc[0]), 1)
        self.assertEqual(int(current["prev_semester_was_interruption"]), 1)
        self.assertEqual(int(current["prior_interruption_count"]), 1)
        self.assertEqual(int(current["consecutive_interruption_count"]), 1)
        self.assertAlmostEqual(current["last_valid_gpa_before_current_semester"], 2.0)
        self.assertAlmostEqual(current["prev_gpa_points_clean"], 2.0)

    def test_timeline_grouping_is_university_aware(self):
        df = pd.DataFrame(
            [
                _base_row(
                    university_id="111",
                    student_id="same_student",
                    degree_id="same_degree",
                    part_id="20201",
                    course_id="c1",
                    gpa_points=3.7,
                ),
                _base_row(
                    university_id="222",
                    student_id="same_student",
                    degree_id="same_degree",
                    part_id="20202",
                    course_id="c2",
                    gpa_points=2.0,
                ),
            ]
        )

        result = run_feature_engineering_job(df)
        row_222 = result["df_model_audit"].loc[
            result["df_model_audit"]["university_id"].eq("222")
        ].iloc[0]

        self.assertTrue(pd.isna(row_222["last_valid_gpa_before_current_semester"]))
        self.assertEqual(row_222["prev_gpa_fill_source"], "zero_fallback")
        self.assertEqual(float(row_222["prev_gpa_points_clean"]), 0.0)

    def test_university_id_is_derived_from_dotted_id_suffix_when_missing(self):
        df = pd.DataFrame(
            [
                _base_row(
                    student_id="673.111",
                    degree_id="3.111",
                    course_id="101.111",
                )
            ]
        ).drop(columns=["university_id"])

        result = run_feature_engineering_job(df)
        audit = result["df_model_audit"]

        self.assertIn("university_id", audit.columns)
        self.assertEqual(audit["university_id"].iloc[0], "111")
        self.assertEqual(
            result["diagnostics"]["university_id_source"],
            "derived_from_id_suffix",
        )

    def test_keeps_distinct_first_semester_concepts_and_reports_mismatch(self):
        df = pd.DataFrame(
            [
                _base_row(
                    student_id="s3",
                    part_id="20202",
                    start_total_in_courses=3,
                    start_total_in_credits=9,
                )
            ]
        )

        result = run_feature_engineering_job(df)
        row = result["df_model_audit"].iloc[0]

        self.assertEqual(int(row["is_first_active_semester"]), 0)
        self.assertEqual(int(row["is_first_row_in_timeline"]), 1)
        self.assertEqual(
            result["diagnostics"]["timeline_diagnostics"]["first_semester_concept_mismatch_count"],
            1,
        )

    def test_structural_zero_switch_adds_model_prev_gpa_without_changing_audit_clean_value(self):
        df = pd.DataFrame([_base_row(student_id="s4", gpa_points=2.0)])

        result = run_feature_engineering_job(df, structural_zero_as_nan=True)
        audit = result["df_model_audit"]

        self.assertIn("model_prev_gpa", audit.columns)
        self.assertEqual(float(audit["prev_gpa_points_clean"].iloc[0]), 0.0)
        self.assertTrue(pd.isna(audit["model_prev_gpa"].iloc[0]))


    def test_transfer_cold_start_not_suspicious_is_advanced_standing(self):
        df = pd.DataFrame(
            [
                _base_row(
                    student_id="transfer",
                    start_level_name_pl="third year",
                    start_total_in_courses=0,
                    start_total_in_credits=0,
                    prev_gpa_points=np.nan,
                    start_agpa_points=0.0,
                    gpa_points=2.0,
                )
            ]
        )
        result = run_feature_engineering_job(df)
        diag = result["diagnostics"]
        self.assertEqual(diag["suspicious_zero_fallback_count"], 0)
        self.assertEqual(diag["advanced_standing_zero_fallback_count"], 1)

    def test_returning_student_zero_fallback_is_suspicious(self):
        df = pd.DataFrame(
            [
                _base_row(
                    student_id="returning",
                    start_total_in_courses=10,
                    start_total_in_credits=30,
                    prev_gpa_points=np.nan,
                    start_agpa_points=0.0,
                    gpa_points=2.0,
                )
            ]
        )
        result = run_feature_engineering_job(df)
        self.assertEqual(result["diagnostics"]["suspicious_zero_fallback_count"], 1)

    def test_start_level_ord_parses_decimal_level_and_arabic_indic(self):
        for label, expected in [("3.0", 3), ("level 3", 3), ("٣", 3)]:
            with self.subTest(label=label):
                df = pd.DataFrame([_base_row(start_level_name_pl=label)])
                result = run_feature_engineering_job(df)
                actual = int(result["df_model_audit"]["start_level_ord"].iloc[0])
                self.assertEqual(actual, expected)

    def test_assert_no_leakage_columns_raises_on_gpa_points(self):
        from src.feature_engineering import assert_no_leakage_columns

        X = pd.DataFrame({"gpa_points": [1.0], "prev_gpa_points_clean": [2.0]})
        with self.assertRaises(ValueError):
            assert_no_leakage_columns(X)


if __name__ == "__main__":
    unittest.main()
