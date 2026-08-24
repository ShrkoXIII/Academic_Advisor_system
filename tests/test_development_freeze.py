"""Regression tests for the development V0 safety freeze."""

from __future__ import annotations

import unittest

from scripts.verify_development_freeze import (
    verify_artifact_fingerprints,
    verify_contract_and_rows,
    verify_golden_predictions,
)


class DevelopmentFreezeTests(unittest.TestCase):
    def test_required_artifact_fingerprints(self) -> None:
        result = verify_artifact_fingerprints()
        self.assertEqual(result["checked"], 15)
        self.assertEqual(result["recorded_only_skipped"], 5)

    def test_contract_model_widths_and_row_counts(self) -> None:
        result = verify_contract_and_rows()
        self.assertEqual(result["contract"], "baseline_41")
        self.assertEqual(result["feature_count"], 41)
        self.assertEqual(result["train_rows"], 603_068)
        self.assertEqual(result["valid_rows"], 75_155)

    def test_golden_valid_predictions(self) -> None:
        result = verify_golden_predictions()
        self.assertEqual(result["case_count"], 12)
        self.assertEqual(
            set(result["segments"]),
            {
                "cold_start_start_part_equals_part",
                "returning",
                "difficulty_fallback",
                "retake",
            },
        )


if __name__ == "__main__":
    unittest.main()
