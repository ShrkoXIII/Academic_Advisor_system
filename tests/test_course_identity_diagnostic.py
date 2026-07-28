"""Focused guardrail tests for the course-identity diagnostic."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

from scripts.course_identity_diagnostic import (
    DATASET_SPLITS,
    classify_candidate,
    normalize_course_name,
    temporal_replacement_evidence,
)
from src.cleaning_utils import normalize_id_to_string


class CourseIdentityDiagnosticTests(unittest.TestCase):
    def test_name_normalization_is_deterministic(self):
        value = "  مقدمة،   في البرمجة! "
        expected = "مقدمة في البرمجة"
        self.assertEqual(normalize_course_name(value), expected)
        self.assertEqual(normalize_course_name(value), expected)

    def test_numeric_levels_are_preserved(self):
        self.assertNotEqual(
            normalize_course_name("الفيزياء 1"),
            normalize_course_name("الفيزياء 2"),
        )
        self.assertEqual(normalize_course_name("برمجة-3"), "برمجة 3")

    def test_similarity_never_confirms_without_official_evidence(self):
        status = classify_candidate(
            official_mapping_evidence="",
            similarity=1.0,
            candidate_score=100.0,
            structural_match_count=7,
        )
        self.assertEqual(status, "likely_renumbered_needs_review")
        confirmed = classify_candidate(
            official_mapping_evidence="registrar equivalence table row 12",
            similarity=0.0,
            candidate_score=0.0,
            structural_match_count=0,
        )
        self.assertEqual(confirmed, "confirmed_equivalent")

    def test_temporal_replacement_requires_disappearance(self):
        self.assertTrue(
            temporal_replacement_evidence("20213", "20221", 100, 0)
        )
        self.assertFalse(
            temporal_replacement_evidence("20213", "20221", 100, 1)
        )
        self.assertFalse(
            temporal_replacement_evidence("20212", "20221", 100, 0)
        )
        self.assertTrue(
            temporal_replacement_evidence("20164", "20171", 100, 0)
        )

    def test_dotted_id_suffix_is_preserved(self):
        self.assertEqual(normalize_id_to_string("1423.111"), "1423.111")
        self.assertEqual(normalize_id_to_string("1423.0"), "1423")

    def test_no_test_split_access(self):
        self.assertEqual(DATASET_SPLITS, ("train", "valid"))
        source_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "course_identity_diagnostic.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        string_literals = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        forbidden = [
            value
            for value in string_literals
            if "df_test" in value.casefold()
        ]
        self.assertEqual(forbidden, [])


if __name__ == "__main__":
    unittest.main()
