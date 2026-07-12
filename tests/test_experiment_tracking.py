import csv
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src import experiment_tracking as tracking


class ExperimentTrackingTests(unittest.TestCase):
    def test_normalize_case_name(self):
        self.assertEqual(tracking.normalize_case_name(" Add Diploma Signals! "), "add-diploma-signals")
        with self.assertRaises(ValueError):
            tracking.normalize_case_name("---")

    def test_persistent_collisions_receive_suffixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime(2026, 7, 12, 9, 30, tzinfo=timezone.utc)
            first = tracking.resolve_output(root, "baseline 39f", "", None, now=now)
            second = tracking.resolve_output(root, "baseline 39f", "", None, now=now)
            self.assertTrue(first.persistent)
            self.assertEqual(first.run_id, "2026-07-12_0930__baseline-39f")
            self.assertEqual(second.run_id, "2026-07-12_0930__baseline-39f__02")

    def test_quick_run_and_compare_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = tracking.resolve_output(Path(tmp), None, "", None)
            self.assertEqual(context.output_dir, Path(tmp) / "quick" / "latest")
            with self.assertRaises(ValueError):
                tracking.resolve_output(Path(tmp), None, "", "baseline")

    def test_leaderboard_appends_without_removing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "leaderboard.csv"
            base = {field: "" for field in tracking.LEADERBOARD_FIELDS}
            base.update({"run_id": "first", "case_name": "first"})
            tracking.append_leaderboard_row(path, base)
            base.update({"run_id": "second", "case_name": "second"})
            tracking.append_leaderboard_row(path, base)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["run_id"] for row in rows], ["first", "second"])
