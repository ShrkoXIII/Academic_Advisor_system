"""Isolated Phase 1 tests for ``2026-08_temporal_rebuild_v1``.

Synthetic fixtures only. Nothing here reads or writes project data: every test
builds its own frame in memory or under the OS temporary directory, so the
suite is safe to run against a populated data root.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load(module_name: str, filename: str):
    """Import a rebuild script by path without executing its ``main``."""
    path = PROJECT_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SPLIT = _load("_rebuild_split", "rebuild_2026_08_phase1_split.py")
GATE = _load("_rebuild_gate", "rebuild_2026_08_gate15_lineage_materiality.py")

LIVE_BASENAMES = frozenset(
    f"df_{split}_{generation}.parquet"
    for split in ("train", "valid", "test")
    for generation in ("base", "difficulty", "concurrent", "final")
)


def synthetic_source() -> pd.DataFrame:
    """One row per part_id of interest, plus a duplicate-free key column."""
    parts = [
        "20051", "20134", "20164", "20213", "20233",  # -> train
        "20241", "20242", "20243",                     # -> valid (whole of 2024)
        "20251",                                       # -> test
        "20252",                                       # -> excluded (partial)
    ]
    return pd.DataFrame(
        {
            "student_course_id": [f"sc{i}" for i in range(len(parts))],
            "student_id": [f"st{i%3}" for i in range(len(parts))],
            "course_id": [f"c{i%4}" for i in range(len(parts))],
            "degree_id": ["d1"] * len(parts),
            "part_id": parts,
        }
    )


class TemporalBoundaryTests(unittest.TestCase):
    def test_assignment_matches_declared_boundaries(self) -> None:
        df = synthetic_source()
        got = SPLIT.assign_split(df["part_id"])
        expected = [
            "train", "train", "train", "train", "train",
            "valid", "valid", "valid",
            "test",
            "excluded",
        ]
        self.assertEqual(list(got), expected)

    def test_train_admits_every_semester_suffix_through_20233(self) -> None:
        # Suffixes 3 and 4 exist in the real data; lexicographic comparison must
        # not silently drop them from TRAIN.
        parts = pd.Series(["20133", "20134", "20143", "20144", "20233"])
        self.assertTrue((SPLIT.assign_split(parts) == "train").all())

    def test_whole_of_academic_year_2024_is_valid(self) -> None:
        # Amendment 2 Correction 1: VALID is all of 2024, including 20243.
        parts = pd.Series(["20241", "20242", "20243"])
        self.assertTrue((SPLIT.assign_split(parts) == "valid").all())

    def test_2025_never_enters_valid(self) -> None:
        parts = pd.Series(["20251", "20252"])
        self.assertEqual(list(SPLIT.assign_split(parts)), ["test", "excluded"])

    def test_no_row_enters_two_splits(self) -> None:
        df = synthetic_source()
        assignment = SPLIT.assign_split(df["part_id"])
        keys = {
            name: set(df.loc[(assignment == name).values, "student_course_id"])
            for name in ("train", "valid", "test")
        }
        self.assertEqual(keys["train"] & keys["valid"], set())
        self.assertEqual(keys["train"] & keys["test"], set())
        self.assertEqual(keys["valid"] & keys["test"], set())

    def test_assignment_is_deterministic(self) -> None:
        df = synthetic_source()
        first = list(SPLIT.assign_split(df["part_id"]))
        shuffled = df.sample(frac=1.0, random_state=7)
        second = SPLIT.assign_split(shuffled["part_id"]).reindex(df.index)
        self.assertEqual(first, list(second))

    def test_row_counts_reconcile(self) -> None:
        df = synthetic_source()
        assignment = SPLIT.assign_split(df["part_id"])
        total = sum(int((assignment == n).sum()) for n in ("train", "valid", "test"))
        total += int((assignment == "excluded").sum())
        self.assertEqual(total, len(df))


class PartIdConventionTests(unittest.TestCase):
    def test_equal_length_numeric_ids_order_lexicographically(self) -> None:
        values = ["20051", "20134", "20213", "20233", "20241", "20251", "20252"]
        self.assertEqual(sorted(values), values)
        self.assertTrue(all(len(v) == 5 and v.isdigit() for v in values))

    def test_unequal_length_would_break_ordering(self) -> None:
        # Guards the reason the length check exists: mixed widths make
        # lexicographic order non-chronological.
        self.assertGreater("9999", "20241")


class ExclusionReasonTests(unittest.TestCase):
    def test_every_excluded_row_has_an_explicit_reason(self) -> None:
        df = synthetic_source()
        assignment = SPLIT.assign_split(df["part_id"])
        reason = SPLIT.exclusion_reason(df["part_id"], assignment)
        excluded = assignment == "excluded"
        self.assertTrue((reason[excluded.values] != "").all())
        self.assertTrue((reason[(~excluded).values] == "").all())

    def test_20252_carries_its_specific_reason(self) -> None:
        parts = pd.Series(["20252", "29999"])
        assignment = SPLIT.assign_split(parts)
        reason = list(SPLIT.exclusion_reason(parts, assignment))
        self.assertEqual(reason[0], "20252_PARTIAL_FOUND_EXCLUDED")
        self.assertEqual(reason[1], "unassigned_part_id_outside_declared_boundaries")

    def test_void_20243_reason_is_never_emitted(self) -> None:
        # The reason is void per Amendment 2 Correction 1; 20243 is now VALID,
        # so no row may carry it.
        parts = pd.Series(["20241", "20242", "20243", "20251", "20252"])
        assignment = SPLIT.assign_split(parts)
        reasons = set(SPLIT.exclusion_reason(parts, assignment))
        self.assertNotIn(
            "year_2024_semester_3_outside_declared_valid_enumeration", reasons
        )


class CandidateFilenameSafetyTests(unittest.TestCase):
    def test_candidate_basenames_never_match_a_live_basename(self) -> None:
        for path in (SPLIT.TRAIN_OUT, SPLIT.VALID_OUT, SPLIT.TEST_OUT):
            self.assertNotIn(path.name.lower(), LIVE_BASENAMES)

    def test_candidate_outputs_stay_inside_the_rebuild_version(self) -> None:
        for path in (SPLIT.TRAIN_OUT, SPLIT.VALID_OUT, SPLIT.TEST_OUT):
            self.assertIn(SPLIT.REBUILD_VERSION, path.parts)


class DeterminismTests(unittest.TestCase):
    def test_fingerprint_is_row_order_independent(self) -> None:
        df = synthetic_source()
        self.assertEqual(
            SPLIT.frame_fingerprint(df),
            SPLIT.frame_fingerprint(df.sample(frac=1.0, random_state=3)),
        )

    def test_fingerprint_changes_when_a_value_changes(self) -> None:
        df = synthetic_source()
        mutated = df.copy()
        mutated.loc[0, "course_id"] = "CHANGED"
        self.assertNotEqual(
            SPLIT.frame_fingerprint(df), SPLIT.frame_fingerprint(mutated)
        )

    def test_fingerprint_survives_parquet_round_trip(self) -> None:
        df = synthetic_source()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.parquet"
            df.to_parquet(path, index=False)
            self.assertEqual(
                SPLIT.frame_fingerprint(df),
                SPLIT.frame_fingerprint(pd.read_parquet(path)),
            )


class OverwriteSafetyTests(unittest.TestCase):
    def test_baseline_manifest_is_never_regenerated(self) -> None:
        preflight = _load("_rebuild_preflight", "rebuild_2026_08_preflight.py")
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "current_artifacts_baseline_manifest.csv"
            existing.write_text("absolute_path\n", encoding="utf-8")
            original = preflight.MANIFEST_CSV
            try:
                preflight.MANIFEST_CSV = existing
                with self.assertRaises(SystemExit):
                    preflight.main()
            finally:
                preflight.MANIFEST_CSV = original


class FrozenThresholdTests(unittest.TestCase):
    def test_threshold_formula_matches_the_frozen_rule(self) -> None:
        for eligible in (0, 1, 500, 67_307, 99_999, 156_097, 1_000_000):
            self.assertEqual(
                SPLIT_threshold(eligible), max(1000, math.ceil(0.01 * eligible))
            )

    def test_floor_dominates_below_one_hundred_thousand_rows(self) -> None:
        self.assertEqual(SPLIT_threshold(67_307), 1000)

    def test_one_percent_dominates_above_one_hundred_thousand_rows(self) -> None:
        self.assertEqual(SPLIT_threshold(156_097), 1561)

    def test_decision_is_inclusive_at_the_threshold(self) -> None:
        self.assertEqual(GATE.summarise(67_307, 1000)["phase_2_decision"], "PROCEED")
        self.assertEqual(
            GATE.summarise(67_307, 999)["phase_2_decision"],
            "DEFERRED_NO_MATERIAL_NEED",
        )

    def test_reported_share_matches_counts(self) -> None:
        figures = GATE.summarise(67_307, 1034)
        self.assertAlmostEqual(figures["affected_share"], 1034 / 67_307, places=6)


def SPLIT_threshold(eligible: int) -> int:
    return GATE.summarise(eligible, 0)["materiality_threshold"]


class GateIndependenceFromValidOutcomesTests(unittest.TestCase):
    def test_outcome_columns_are_stripped_before_identity_resolution(self) -> None:
        frame = pd.DataFrame(
            {
                "student_course_id": ["a"],
                "course_id": ["c1"],
                "final_mark": [88.0],
                "grade_id": ["g1"],
                "gpa_points": [4.0],
                "points": [3.0],
            }
        )
        stripped = GATE.strip_outcomes(frame)
        for column in GATE.VALID_OUTCOME_COLUMNS:
            self.assertNotIn(column, stripped.columns)
        self.assertIn("course_id", stripped.columns)

    def test_row_level_output_columns_contain_no_outcome(self) -> None:
        for column in GATE.VALID_OUTCOME_COLUMNS:
            self.assertNotIn(column, GATE.IDENTITY_OUTPUT_COLUMNS)


class GateMatchesCourseDifficultyTests(unittest.TestCase):
    """The gate's never-in-TRAIN rule must equal production's `course_is_new`."""

    def _frame(self, rows: list[tuple[str, str, str]]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "student_course_id": [f"k{i}" for i in range(len(rows))],
                "degree_id": [r[0] for r in rows],
                "course_id": [r[1] for r in rows],
                "faculty_id": [r[2] for r in rows],
                "degree_course_key": [f"{r[0]}__{r[1]}" for r in rows],
                "part_id": ["20211"] * len(rows),
                "requirement_type_id": [1] * len(rows),
                "course_credits": [3.0] * len(rows),
                "attempt_number": [1] * len(rows),
                "final_mark": [70.0] * len(rows),
            }
        )

    def test_identity_only_path_equals_production_course_is_new(self) -> None:
        from src.course_difficulty import (
            DifficultyConfig,
            apply_difficulty_state,
            fit_difficulty_state,
        )

        train = self._frame(
            [("d1", "c1", "f1"), ("d1", "c2", "f1"), ("d2", "c1", "f1")]
        )
        valid = self._frame(
            [
                ("d1", "c1", "f1"),  # seen at level 1
                ("d9", "c2", "f1"),  # unseen degree, course seen at level 2
                ("d9", "c9", "f1"),  # never in train -> course_is_new
            ]
        )
        state = fit_difficulty_state(train, DifficultyConfig())
        production = apply_difficulty_state(
            GATE.strip_outcomes(valid), state, include_source=True
        )["course_is_new"].tolist()

        _, figures = GATE.measure_identity_only(train, valid)
        identity_only = figures["affected_rows"]

        self.assertEqual(production, [0, 0, 1])
        self.assertEqual(identity_only, sum(production))


if __name__ == "__main__":
    unittest.main()
