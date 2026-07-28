"""Focused tests for the locked R2 coverage decision implementation."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

from scripts.r2_coverage_rescore import direction


ROOT = Path(__file__).resolve().parents[1]


class R2CoverageRescoreTests(unittest.TestCase):
    def test_metric_directions_are_locked(self):
        self.assertEqual(direction("roc_auc", 0.1), "beneficial")
        self.assertEqual(direction("roc_auc", -0.1), "harmful")
        self.assertEqual(direction("brier", -0.1), "beneficial")
        self.assertEqual(direction("brier", 0.1), "harmful")
        self.assertEqual(direction("fail_average_precision", 0), "zero")

    def test_persisted_decision_keeps_incumbent_when_rule_not_fully_met(self):
        report = json.loads(
            (
                ROOT
                / "models"
                / "runs"
                / "R2_COVERED_UNCOVERED_FIVE_SEED_REPORT.json"
            ).read_text(encoding="utf-8")
        )
        evaluation = report["locked_rule_evaluation"]
        self.assertFalse(evaluation["all_clauses_satisfied"])
        self.assertEqual(
            evaluation["decision"], "KEEP_DEFAULT_127_FOR_M1"
        )
        self.assertEqual(report["scope"]["m1_contract"], "baseline_41")
        self.assertFalse(report["scope"]["m2_changed"])

    def test_all_frozen_pairs_passed_before_prediction_loading(self):
        report = json.loads(
            (
                ROOT
                / "models"
                / "runs"
                / "R2_COVERED_UNCOVERED_FIVE_SEED_REPORT.json"
            ).read_text(encoding="utf-8")
        )
        parity = report["parity"]
        self.assertTrue(parity["all_pairs_passed"])
        self.assertTrue(
            parity["prediction_loading_started_only_after_this_gate"]
        )
        self.assertTrue(
            all(pair["all_passed"] for pair in parity["pairs"].values())
        )

    def test_rescore_script_constructs_no_test_split_path(self):
        source = (
            ROOT / "scripts" / "r2_coverage_rescore.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        literals = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        self.assertFalse(
            any("df_test" in value.casefold() for value in literals)
        )


if __name__ == "__main__":
    unittest.main()
