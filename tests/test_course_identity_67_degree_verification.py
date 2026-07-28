import ast
import json
import unittest
from pathlib import Path

import pandas as pd

from scripts.course_identity_67_degree_verification import (
    NOT_AVAILABLE,
    classify_conclusion,
    compare_degree_sets,
    dotted_university_id,
    values_for,
)
from src.cleaning_utils import normalize_id_to_string


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "course_identity_67_degree_verification.py"
ALL_PATH = ROOT / "models" / "runs" / "COURSE_IDENTITY_67_DEGREE_VERIFICATION.csv"
BEST_PATH = ROOT / "models" / "runs" / "COURSE_IDENTITY_67_BEST_MATCH_PER_COURSE.csv"
REVIEW_PATH = ROOT / "models" / "runs" / "COURSE_IDENTITY_67_HUMAN_REVIEW.csv"
JSON_PATH = ROOT / "models" / "runs" / "COURSE_IDENTITY_67_DEGREE_VERIFICATION.json"
INPUT_PATH = ROOT / "models" / "runs" / "COURSE_IDENTITY_CANDIDATES.csv"


class CourseIdentity67DegreeVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all_pairs = pd.read_csv(ALL_PATH, dtype=str, keep_default_na=False)
        cls.best = pd.read_csv(BEST_PATH, dtype=str, keep_default_na=False)
        cls.review = pd.read_csv(REVIEW_PATH, dtype=str, keep_default_na=False)
        cls.payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    def test_exactly_67_new_candidate_courses_are_processed(self):
        source = pd.read_csv(INPUT_PATH, dtype=str, keep_default_na=False)
        source = source.loc[
            source["diagnostic_status"].eq("likely_renumbered_needs_review")
        ]
        self.assertEqual(source["new_course_id"].nunique(), 67)
        self.assertEqual(self.all_pairs["new_course_id"].nunique(), 67)
        self.assertEqual(len(self.all_pairs), len(source))
        self.assertEqual(len(self.best), 67)
        self.assertEqual(len(self.review), 67)
        self.assertEqual(
            set(zip(source["new_course_id"], source["candidate_old_course_id"])),
            set(zip(self.all_pairs["new_course_id"], self.all_pairs["old_course_id"])),
        )

    def test_ids_remain_strings_and_dotted_suffixes_are_preserved(self):
        self.assertEqual(normalize_id_to_string("1423.111"), "1423.111")
        self.assertEqual(dotted_university_id("1423.111"), "111")
        for column in ("new_course_id", "old_course_id"):
            self.assertTrue(
                self.all_pairs[column].map(lambda value: isinstance(value, str)).all()
            )
            self.assertTrue(self.all_pairs[column].str.contains(".", regex=False).all())

    def test_same_degree_compares_actual_normalized_ids(self):
        exact = compare_degree_sets(
            {"26.111"}, {"26.111"}, {"111"}, {"111"}
        )
        different = compare_degree_sets(
            {"26.111"}, {"27.111"}, {"111"}, {"111"}
        )
        self.assertTrue(exact["same_degree"])
        self.assertEqual(exact["shared"], {"26.111"})
        self.assertFalse(different["same_degree"])
        self.assertEqual(
            different["degree_relationship"],
            "SAME_UNIVERSITY_DIFFERENT_DEGREE",
        )

    def test_same_name_does_not_imply_same_degree(self):
        conclusion, _ = classify_conclusion(
            {
                "degree_relationship": "SAME_UNIVERSITY_DIFFERENT_DEGREE",
                "name_similarity": 1.0,
            }
        )
        self.assertEqual(conclusion, "SAME_UNIVERSITY_DIFFERENT_DEGREE")

    def test_same_faculty_does_not_imply_same_degree(self):
        conclusion, _ = classify_conclusion(
            {
                "degree_relationship": "SAME_UNIVERSITY_DIFFERENT_DEGREE",
                "same_faculty": True,
                "name_similarity": 1.0,
            }
        )
        self.assertEqual(conclusion, "SAME_UNIVERSITY_DIFFERENT_DEGREE")

    def test_multiple_catalog_rows_are_not_silently_collapsed(self):
        rows = pd.DataFrame(
            {
                "course_id": ["10.111", "10.111", "10.111"],
                "degree_id": ["1.111", "2.111", "3.111"],
            }
        )
        degrees = values_for(rows, "degree_id", id_field=True)
        self.assertEqual(len(rows), 3)
        self.assertEqual(degrees, {"1.111", "2.111", "3.111"})

    def test_shared_and_nonshared_degree_sets_are_exact(self):
        result = compare_degree_sets(
            {"1.111", "2.111"},
            {"2.111", "3.111"},
            {"111"},
            {"111"},
        )
        self.assertEqual(result["shared"], {"2.111"})
        self.assertEqual(result["old_only"], {"1.111"})
        self.assertEqual(result["new_only"], {"3.111"})
        self.assertEqual(result["degree_relationship"], "PARTIAL_DEGREE_OVERLAP")

    def test_similarity_never_creates_confirmed_equivalent(self):
        evidence = {
            "degree_relationship": "SAME_SINGLE_DEGREE",
            "same_faculty": True,
            "credits_match": True,
            "course_type_match": True,
            "requirement_type_match": True,
            "planned_year_match": True,
            "planned_semester_match": True,
            "different_course_id": True,
            "name_similarity": 1.0,
            "temporal_replacement_signal": "DIRECT_REPLACEMENT",
        }
        conclusion, _ = classify_conclusion(evidence)
        self.assertEqual(
            conclusion, "STRICT_SAME_DEGREE_RENUMBERING_CANDIDATE"
        )
        self.assertNotEqual(conclusion, "confirmed_equivalent")
        self.assertNotIn(
            "confirmed_equivalent",
            set(self.all_pairs["diagnostic_conclusion"]),
        )

    def test_script_never_constructs_a_test_parquet_path(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        string_literals = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        self.assertFalse(
            any("df_test_final.parquet" in value for value in string_literals)
        )
        self.assertNotIn("df_test", source.casefold())

    def test_inputs_are_hash_verified_unchanged(self):
        integrity = self.payload["source_integrity"]
        self.assertTrue(integrity)
        self.assertTrue(all(item["unchanged"] for item in integrity.values()))
        self.assertFalse(self.payload["scope"]["dataset_or_source_changed"])
        self.assertFalse(self.payload["scope"]["test_read"])
        self.assertEqual(self.payload["scope"]["test_policy"], "closed_not_read")

    def test_missing_comparison_is_not_treated_as_match(self):
        self.assertEqual(
            self.all_pairs.loc[
                self.all_pairs["same_faculty"].eq(NOT_AVAILABLE),
                "same_faculty",
            ].unique().tolist(),
            [NOT_AVAILABLE],
        )

    def test_human_review_decision_fields_are_blank(self):
        for column in (
            "review_decision",
            "reviewer_name",
            "review_date",
            "review_notes",
        ):
            self.assertTrue(self.review[column].eq("").all())


if __name__ == "__main__":
    unittest.main()
