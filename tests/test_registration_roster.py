import unittest

import pandas as pd

from src.feature_engineering import SEMESTER_KEY
from src.registration_roster import (
    KNOWN_OUTCOME_COLUMNS,
    REGISTRATION_ROSTER_FEATURE_INPUT_COLUMNS,
    ROW_AUDIT_ONLY_COLUMNS,
    assert_model_facing_roster_columns,
    assert_unique_peer_membership,
    build_registration_roster,
    model_facing_roster,
    roster_row_audit,
)


def _raw_row(
    student_course_id,
    course_id,
    *,
    student_id="1.111",
    degree_id="10.111",
    part_id="20241",
    faculty_id="5.111",
    course_credits=2,
    active="A",
    register_status="R",
    finish_status="P",
):
    return {
        "student_course_id": student_course_id,
        "student_id": student_id,
        "course_id": course_id,
        "part_id": part_id,
        "degree_id": degree_id,
        "faculty_id": faculty_id,
        "course_credits": course_credits,
        "active": active,
        "register_status": register_status,
        "finish_status": finish_status,
        # These source outcomes must never be propagated to the roster.
        "final_mark": 75,
        "points": 2.75,
        "grade_id": "9.111",
        "course_outcome_status": "passed",
        "gpa_points": 2.5,
        "semester_pass_credits": 12,
    }


def _target_row(
    student_course_id="100.111",
    course_id="10.111",
    *,
    student_id="1.111",
    degree_id="10.111",
    part_id="20241",
    semester_reg_courses=1,
):
    return {
        "university_id": 111.0,
        "student_course_id": student_course_id,
        "student_id": student_id,
        "course_id": course_id,
        "part_id": part_id,
        "degree_id": degree_id,
        "faculty_id": "9.111",
        "course_credits": 3,
        "requirement_type_id": 7,
        "degree_requirement_credits_count": 44,
        "degree_course_key": "trusted-target-key",
        "semester_reg_courses": semester_reg_courses,
    }


def _acd_row(
    degree_id,
    course_id,
    requirement_type_id,
    *,
    credits_count=30,
    course_credits=3,
):
    course_base = str(course_id).split(".")[0]
    degree_base = str(degree_id).split(".")[0]
    return {
        "degree_course_id": f"{degree_base}{course_base}.111",
        "degree_id": degree_id,
        "course_id": course_id,
        "requirement_type_id": requirement_type_id,
        "requirement_type_sl": f"req-{requirement_type_id}",
        "credits_count": credits_count,
        "course_credits": course_credits,
        "course_name_sl": f"course-{course_id}",
        "degree_name_sl": f"degree-{degree_id}",
    }


class TestRegistrationRoster(unittest.TestCase):
    def test_withdrawn_and_all_other_finish_statuses_remain_in_roster(self):
        target = pd.DataFrame(
            [_target_row(semester_reg_courses=4)]
        )
        raw = pd.DataFrame(
            [
                _raw_row(" 100.111 ", " 10.111 ", finish_status="P"),
                _raw_row("101.111", "20.111", finish_status="W"),
                _raw_row(
                    "102.111", "30.111", register_status=" e ", finish_status=None
                ),
                _raw_row("103.111", "40.111", finish_status="UNFINISHED"),
                _raw_row(
                    "104.111",
                    "50.111",
                    active="I",
                    register_status="R",
                    finish_status="W",
                ),
                _raw_row(
                    "105.111",
                    "60.111",
                    register_status="C",
                    finish_status="P",
                ),
                _raw_row(
                    "200.111",
                    "70.111",
                    student_id="2.111",
                    finish_status="W",
                ),
            ]
        )
        # Exercise source-column normalization as well as value normalization.
        raw.columns = [column.upper() for column in raw.columns]
        acd = pd.DataFrame(
            [
                _acd_row("10.111", "10.111", 1),
                _acd_row("10.111", "20.111", 2),
                _acd_row("10.111", "30.111", 3),
                _acd_row("10.111", "40.111", 4),
            ]
        )

        result = build_registration_roster(raw, target, acd)
        roster = result.roster

        self.assertEqual(
            set(roster["student_course_id"]),
            {"100.111", "101.111", "102.111", "103.111"},
        )
        finish_by_occurrence = roster.set_index("student_course_id")[
            "finish_status_audit"
        ].to_dict()
        self.assertEqual(finish_by_occurrence["101.111"], "W")
        self.assertTrue(pd.isna(finish_by_occurrence["102.111"]))
        self.assertEqual(finish_by_occurrence["103.111"], "UNFINISHED")
        self.assertFalse(result.diagnostics["filters"]["finish_status_filter_applied"])

        # Target values, not disagreeing raw/ACD values, own target metadata.
        target_occurrence = roster.loc[
            roster["student_course_id"].eq("100.111")
        ].iloc[0]
        self.assertEqual(target_occurrence["faculty_id"], "9.111")
        self.assertEqual(target_occurrence["course_credits"], 3)
        self.assertEqual(target_occurrence["requirement_type_id"], 7)
        self.assertEqual(
            target_occurrence["degree_course_key"], "trusted-target-key"
        )
        self.assertEqual(target_occurrence["faculty_id_source"], "target_occurrence")
        self.assertEqual(
            target_occurrence["course_credits_source"], "target_occurrence"
        )

        self.assertTrue(
            set(REGISTRATION_ROSTER_FEATURE_INPUT_COLUMNS).issubset(roster.columns)
        )
        self.assertNotIn(
            "finish_status_audit", REGISTRATION_ROSTER_FEATURE_INPUT_COLUMNS
        )
        self.assertFalse(KNOWN_OUTCOME_COLUMNS.intersection(roster.columns))
        self.assertEqual(
            result.diagnostics["outcome_columns_present_in_roster"], []
        )
        self.assertEqual(
            result.diagnostics["excluded_register_status_counts"], {"C": 1}
        )
        self.assertEqual(result.diagnostics["row_counts"]["roster_only"], 3)

    def test_exact_unique_fallback_ambiguous_and_absent_acd_metadata(self):
        target = pd.DataFrame([_target_row(semester_reg_courses=5)])
        raw = pd.DataFrame(
            [
                _raw_row("100.111", "10.111"),
                _raw_row("101.111", "20.111"),
                _raw_row("102.111", "30.111"),
                _raw_row("103.111", "40.111"),
                _raw_row("104.111", "50.111"),
            ]
        )
        acd = pd.DataFrame(
            [
                _acd_row("10.111", "10.111", 1, credits_count=10),
                _acd_row("10.111", "20.111", 2, credits_count=20),
                # Course 30 belongs to one ACD degree, so degree 10 may safely
                # use the current pipeline's unique-course-degree fallback.
                _acd_row("55.111", "30.111", 3, credits_count=30),
                # Course 40 belongs to two other degrees: fallback is ambiguous.
                _acd_row("55.111", "40.111", 4, credits_count=40),
                _acd_row("56.111", "40.111", 5, credits_count=50),
            ]
        )

        roster = build_registration_roster(raw, target, acd).roster.set_index(
            "course_id"
        )

        self.assertEqual(roster.loc["10.111", "requirement_type_id"], 7)
        self.assertEqual(
            roster.loc["10.111", "requirement_metadata_source"],
            "target_occurrence",
        )
        self.assertEqual(roster.loc["20.111", "requirement_type_id"], 2)
        self.assertEqual(
            roster.loc["20.111", "requirement_metadata_source"],
            "acd_exact_degree_course_match",
        )
        self.assertEqual(roster.loc["20.111", "degree_requirement_credits_count"], 20)
        self.assertEqual(roster.loc["30.111", "requirement_type_id"], 3)
        self.assertEqual(
            roster.loc["30.111", "requirement_metadata_source"],
            "acd_unique_course_degree_fallback",
        )
        self.assertEqual(roster.loc["30.111", "acd_resolved_degree_id"], "55.111")
        # The fallback is metadata-only: it must not rewrite student degree/key.
        self.assertEqual(
            roster.loc["30.111", "degree_course_key"], "10.111__30.111"
        )

        self.assertTrue(pd.isna(roster.loc["40.111", "requirement_type_id"]))
        self.assertEqual(
            roster.loc["40.111", "requirement_metadata_source"],
            "acd_ambiguous_multiple_degrees",
        )
        self.assertTrue(pd.isna(roster.loc["50.111", "requirement_type_id"]))
        self.assertEqual(
            roster.loc["50.111", "requirement_metadata_source"],
            "acd_course_absent",
        )

    def test_duplicate_course_occurrences_collapse_to_one_peer(self):
        # One course registered twice in one semester is ONE peer, not two.
        target = pd.DataFrame([_target_row(semester_reg_courses=2)])
        raw = pd.DataFrame(
            [
                _raw_row("100.111", "10.111"),
                _raw_row("101.111", "10.111", finish_status="W"),
            ]
        )
        acd = pd.DataFrame([_acd_row("10.111", "10.111", 1)])

        result = build_registration_roster(raw, target, acd)
        roster = result.roster
        self.assertEqual(len(roster), 1)
        self.assertEqual(roster["course_id"].nunique(), 1)
        # The target occurrence is kept as the course's representative, so
        # exact target matching by student_course_id still resolves.
        self.assertEqual(roster["student_course_id"].iloc[0], "100.111")
        self.assertTrue(roster["is_target_occurrence"].all())
        self.assertTrue(
            (roster["registration_roster_occurrence_count"] == 1).all()
        )
        self.assertTrue(
            (roster["registration_roster_unique_course_count"] == 1).all()
        )
        # The published roster is unique on the membership key by construction.
        self.assertEqual(
            result.diagnostics["duplicate_course_occurrence_rows"], 0
        )
        collapse = result.diagnostics["peer_membership_collapse"]
        self.assertEqual(collapse["occurrence_rows"], 2)
        self.assertEqual(collapse["peer_rows"], 1)
        self.assertEqual(collapse["collapsed_duplicate_occurrence_rows"], 1)
        self.assertEqual(collapse["courses_with_multiple_occurrences"], 1)
        assert_unique_peer_membership(roster)

    def test_conflicting_roster_only_duplicates_fail_loudly(self):
        # Two ROSTER-ONLY occurrences of one course with different registration
        # credits are equal-authority candidates, so collapsing would be an
        # arbitrary pick and the build must refuse.
        target = pd.DataFrame([_target_row(semester_reg_courses=3)])
        raw = pd.DataFrame(
            [
                _raw_row("100.111", "10.111"),
                _raw_row("101.111", "20.111", course_credits=2),
                _raw_row("102.111", "20.111", course_credits=5),
            ]
        )
        acd = pd.DataFrame(
            [_acd_row("10.111", "10.111", 1), _acd_row("10.111", "20.111", 2)]
        )

        with self.assertRaisesRegex(ValueError, "disagree on 'course_credits'"):
            build_registration_roster(raw, target, acd)

    def test_target_occurrence_wins_the_representative_slot(self):
        # A target occurrence and a roster-only occurrence of the SAME course
        # differ on faculty_id only because the target override is
        # authoritative. That is resolved precedence, not a conflict.
        target = pd.DataFrame([_target_row(semester_reg_courses=2)])
        raw = pd.DataFrame(
            [
                _raw_row("100.111", "10.111", faculty_id="5.111"),
                _raw_row("101.111", "10.111", faculty_id="5.111",
                         finish_status="W"),
            ]
        )
        acd = pd.DataFrame([_acd_row("10.111", "10.111", 1)])

        roster = build_registration_roster(raw, target, acd).roster
        self.assertEqual(len(roster), 1)
        self.assertEqual(roster["student_course_id"].iloc[0], "100.111")
        # The surviving row carries the target's authoritative faculty_id.
        self.assertEqual(roster["faculty_id"].iloc[0], "9.111")
        self.assertEqual(roster["faculty_id_source"].iloc[0], "target_occurrence")

    def test_model_facing_roster_drops_outcome_and_target_proxy_columns(self):
        target = pd.DataFrame([_target_row(semester_reg_courses=2)])
        raw = pd.DataFrame(
            [
                _raw_row("100.111", "10.111"),
                _raw_row("101.111", "20.111", finish_status="W"),
            ]
        )
        acd = pd.DataFrame(
            [_acd_row("10.111", "10.111", 1), _acd_row("10.111", "20.111", 2)]
        )

        roster = build_registration_roster(raw, target, acd).roster
        for column in ROW_AUDIT_ONLY_COLUMNS:
            self.assertIn(column, roster.columns)

        model_facing = model_facing_roster(roster)
        for column in ROW_AUDIT_ONLY_COLUMNS:
            self.assertNotIn(column, model_facing.columns)
        self.assertEqual(len(model_facing), len(roster))
        # The historical difficulty names stay allow-listed, not pattern-banned.
        allowed = model_facing.assign(course_pass_rate_historical=0.5)
        self.assertEqual(assert_model_facing_roster_columns(allowed), [])
        with self.assertRaisesRegex(AssertionError, "final_mark"):
            assert_model_facing_roster_columns(
                model_facing.assign(final_mark=1)
            )

        audit = roster_row_audit(roster)
        self.assertEqual(len(audit), len(roster))
        for column in ROW_AUDIT_ONLY_COLUMNS:
            self.assertIn(column, audit.columns)
        self.assertEqual(int(audit["is_target_occurrence"].sum()), 1)

    def test_target_requires_exact_semester_occurrence_coverage(self):
        target = pd.DataFrame([_target_row()])
        # The stable occurrence ID exists, but its semester is different.
        raw = pd.DataFrame(
            [_raw_row("100.111", "10.111", part_id="20242")]
        )
        acd = pd.DataFrame([_acd_row("10.111", "10.111", 1)])

        with self.assertRaisesRegex(
            ValueError, "exact semester keys and course_id"
        ):
            build_registration_roster(raw, target, acd)

    def test_ambiguous_dotted_university_ids_fail_loudly(self):
        target = pd.DataFrame([_target_row()])
        raw = pd.DataFrame(
            [
                _raw_row(
                    "100.111",
                    "10.111",
                    student_id="1.111",
                    degree_id="10.222",
                )
            ]
        )
        acd = pd.DataFrame([_acd_row("10.111", "10.111", 1)])

        with self.assertRaisesRegex(ValueError, "ambiguous university suffixes"):
            build_registration_roster(raw, target, acd)

    def test_normalized_ids_match_and_registration_count_mismatch_is_explained(self):
        target = pd.DataFrame(
            [
                _target_row(
                    student_course_id="100.111",
                    course_id="10.111",
                    part_id=20241,
                    semester_reg_courses=3,
                )
            ]
        )
        raw = pd.DataFrame(
            [
                _raw_row(" 100.111 ", "10.111", part_id="20241.0"),
                _raw_row("101.111", "20.111", part_id=20241, finish_status="W"),
                _raw_row(
                    "102.111",
                    "30.111",
                    part_id=20241,
                    register_status=" cancelled ",
                ),
            ]
        )
        acd = pd.DataFrame(
            [
                _acd_row("10.111", "10.111", 1),
                _acd_row("10.111", "20.111", 2),
            ]
        )

        result = build_registration_roster(raw, target, acd)
        roster = result.roster
        self.assertEqual(roster["university_id"].unique().tolist(), ["111"])
        self.assertEqual(str(roster["part_id"].dtype), "Int64")
        self.assertEqual(
            roster["student_course_id"].tolist(), ["100.111", "101.111"]
        )

        comparison = result.semester_count_comparison.iloc[0]
        self.assertEqual(comparison["registration_roster_occurrence_count"], 2)
        self.assertEqual(comparison["semester_reg_courses"], 3)
        self.assertFalse(comparison["registration_count_matches"])
        self.assertEqual(
            comparison["registration_count_mismatch_reason"],
            "roster_shortfall_with_excluded_register_status_rows",
        )
        self.assertEqual(comparison["excluded_register_status_count"], 1)
        self.assertEqual(comparison["register_status_breakdown"], "CANCELLED:1")

        excluded = result.excluded_registration_status_summary
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded.iloc[0]["register_status"], "CANCELLED")
        self.assertEqual(excluded.iloc[0]["excluded_register_status_count"], 1)
        self.assertEqual(len(result.mismatch_examples), 1)

    def test_multiple_raw_rows_with_one_occurrence_id_are_rejected(self):
        target = pd.DataFrame([_target_row(semester_reg_courses=2)])
        raw = pd.DataFrame(
            [
                _raw_row("100.111", "10.111"),
                _raw_row("100.111", "20.111"),
            ]
        )
        acd = pd.DataFrame(
            [
                _acd_row("10.111", "10.111", 1),
                _acd_row("10.111", "20.111", 2),
            ]
        )

        with self.assertRaisesRegex(ValueError, "student_course_id is ambiguous"):
            build_registration_roster(raw, target, acd)

    def test_all_target_rows_are_covered_once_across_multiple_groups(self):
        target = pd.DataFrame(
            [
                _target_row(semester_reg_courses=1),
                _target_row(
                    student_course_id="200.111",
                    course_id="20.111",
                    student_id="2.111",
                    semester_reg_courses=1,
                ),
            ]
        )
        raw = pd.DataFrame(
            [
                _raw_row("100.111", "10.111"),
                _raw_row(
                    "200.111", "20.111", student_id="2.111", finish_status="F"
                ),
            ]
        )
        acd = pd.DataFrame(
            [
                _acd_row("10.111", "10.111", 1),
                _acd_row("10.111", "20.111", 2),
            ]
        )

        result = build_registration_roster(raw, target, acd)
        self.assertEqual(
            result.diagnostics["row_counts"]["target_occurrences_matched"], 2
        )
        self.assertEqual(
            set(result.roster.loc[result.roster["is_target_occurrence"], "target_row_position"]),
            {0, 1},
        )
        self.assertEqual(
            result.roster.groupby(SEMESTER_KEY, dropna=False).size().tolist(),
            [1, 1],
        )


if __name__ == "__main__":
    unittest.main()
