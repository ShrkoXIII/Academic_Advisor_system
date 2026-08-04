"""Tests for the extracted diploma bucketing rule.

The decisive test re-fits the map on the superseded TRAIN split and asserts the
result reproduces the live artifact field for field. That is what makes the
extraction from 03_diploma_type_bucketing.ipynb checkable: if the notebook's
rule had been altered while being lifted out, this test fails.

It is skipped, not failed, when the superseded TRAIN split is not on disk, so
the suite stays runnable on a machine without the project data.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diploma_bucketing import (  # noqa: E402
    DIPLOMA_TYPE_COLUMN,
    RARE_BUCKET_LABEL,
    UNSEEN_BUCKET_LABEL,
    assert_no_reserved_label_collision,
    bucket_map_state,
    fit_diploma_bucket_map,
    load_bucket_map,
    save_bucket_map,
)
from src.paths import DIPLOMA_TYPE_BUCKET_MAP_PATH, MODEL_DATA_DIR  # noqa: E402

# The split the live map records as its fit source.
LEGACY_TRAIN_PATH = MODEL_DATA_DIR / "df_train_difficulty.parquet"


class RefitReproducesTheLiveMapTests(unittest.TestCase):
    """The extraction did not change behaviour."""

    @classmethod
    def setUpClass(cls) -> None:
        if not LEGACY_TRAIN_PATH.exists():
            raise unittest.SkipTest(f"superseded TRAIN not on disk: {LEGACY_TRAIN_PATH}")
        if not DIPLOMA_TYPE_BUCKET_MAP_PATH.exists():
            raise unittest.SkipTest("live diploma bucket map not on disk")
        cls.live = json.loads(
            DIPLOMA_TYPE_BUCKET_MAP_PATH.read_text(encoding="utf-8")
        )
        cls.train = pd.read_parquet(
            LEGACY_TRAIN_PATH, columns=[DIPLOMA_TYPE_COLUMN]
        )
        cls.refitted = fit_diploma_bucket_map(cls.train)

    def test_fit_source_row_count_matches_the_recorded_one(self) -> None:
        self.assertEqual(len(self.train), self.live["fit_source"]["rows"])

    def test_top_codes_match_in_order(self) -> None:
        self.assertEqual(list(self.refitted.top_codes), self.live["top_codes"])

    def test_code_to_bucket_map_matches(self) -> None:
        refitted = {
            str(int(code)): int(bucket)
            for code, bucket in self.refitted.code_to_bucket.items()
        }
        self.assertEqual(refitted, self.live["code_to_bucket_map"])

    def test_reserved_labels_and_categories_match(self) -> None:
        self.assertEqual(
            self.refitted.rare_bucket_label, self.live["rare_bucket_label"]
        )
        self.assertEqual(
            self.refitted.unseen_bucket_label, self.live["unseen_bucket_label"]
        )
        self.assertEqual(list(self.refitted.categories), self.live["categories"])

    def test_persisted_state_matches_the_live_artifact_field_for_field(self) -> None:
        state = bucket_map_state(
            self.refitted,
            fit_source_path=self.live["fit_source"]["path"],
            fit_rows=self.live["fit_source"]["rows"],
            created=self.live["created"],
            git_commit=self.live["git_commit"],
        )
        self.assertEqual(state, self.live)
        self.assertEqual(list(state), list(self.live))


class FittingRuleTests(unittest.TestCase):
    """The rule itself, on frames small enough to reason about by hand."""

    def setUp(self) -> None:
        # 5 codes above the rest, plus two rare ones and a null.
        codes = (
            [15] * 9 + [16] * 8 + [13] * 7 + [19] * 6 + [26] * 5 + [32] * 2 + [9] * 1
        )
        self.train = pd.DataFrame(
            {DIPLOMA_TYPE_COLUMN: codes + [None] * 4}, dtype="object"
        )
        self.train[DIPLOMA_TYPE_COLUMN] = pd.to_numeric(
            self.train[DIPLOMA_TYPE_COLUMN], errors="coerce"
        )

    def test_top_five_keep_their_raw_codes(self) -> None:
        fitted = fit_diploma_bucket_map(self.train)
        self.assertEqual(list(fitted.top_codes), [15, 16, 13, 19, 26])
        for code in (15, 16, 13, 19, 26):
            self.assertEqual(fitted.code_to_bucket[code], code)

    def test_remaining_train_codes_go_to_the_rare_bucket(self) -> None:
        fitted = fit_diploma_bucket_map(self.train)
        self.assertEqual(fitted.code_to_bucket[32], RARE_BUCKET_LABEL)
        self.assertEqual(fitted.code_to_bucket[9], RARE_BUCKET_LABEL)

    def test_nulls_are_excluded_from_the_ranking(self) -> None:
        # 4 nulls beat code 9's single row, but must not take a top-five slot.
        fitted = fit_diploma_bucket_map(self.train)
        self.assertEqual(len(fitted.top_codes), 5)
        self.assertNotIn(UNSEEN_BUCKET_LABEL, fitted.top_codes)

    def test_nulls_and_unseen_codes_map_to_the_unseen_bucket(self) -> None:
        fitted = fit_diploma_bucket_map(self.train)
        applied = fitted.apply(pd.Series([15.0, 32.0, 999.0, None]))
        self.assertEqual(
            list(applied), [15, RARE_BUCKET_LABEL, UNSEEN_BUCKET_LABEL, UNSEEN_BUCKET_LABEL]
        )

    def test_categories_are_the_top_five_plus_both_reserved_labels(self) -> None:
        fitted = fit_diploma_bucket_map(self.train)
        self.assertEqual(
            list(fitted.categories), [13, 15, 16, 19, 26, RARE_BUCKET_LABEL, UNSEEN_BUCKET_LABEL]
        )
        categorical = fitted.as_categorical(pd.Series([15.0, None]))
        self.assertEqual(list(categorical.categories), list(fitted.categories))
        self.assertEqual(int(categorical[1]), UNSEEN_BUCKET_LABEL)

    def test_a_real_code_equal_to_the_rare_label_is_refused(self) -> None:
        colliding = pd.DataFrame({DIPLOMA_TYPE_COLUMN: [RARE_BUCKET_LABEL, 15.0]})
        with self.assertRaises(AssertionError):
            assert_no_reserved_label_collision(
                {"train": self.train, "valid": colliding}
            )
        # No collision in the clean frames.
        assert_no_reserved_label_collision({"train": self.train})

    def test_a_split_without_the_column_is_refused(self) -> None:
        with self.assertRaises(AssertionError):
            fit_diploma_bucket_map(pd.DataFrame({"something_else": [1, 2]}))


class PersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="diploma_map_test_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.fitted = fit_diploma_bucket_map(
            pd.DataFrame({DIPLOMA_TYPE_COLUMN: [15.0] * 3 + [16.0] * 2 + [32.0]})
        )

    def test_round_trip_preserves_the_map(self) -> None:
        path = self.root / "diploma_type_bucket_map.json"
        save_bucket_map(
            bucket_map_state(self.fitted, fit_source_path="synthetic", fit_rows=6),
            path,
        )
        reloaded = load_bucket_map(path)
        self.assertEqual(reloaded, self.fitted)

    def test_existing_map_is_never_overwritten(self) -> None:
        path = self.root / "diploma_type_bucket_map.json"
        state = bucket_map_state(self.fitted, fit_source_path="synthetic", fit_rows=6)
        save_bucket_map(state, path)
        with self.assertRaises(FileExistsError):
            save_bucket_map(state, path)


if __name__ == "__main__":
    unittest.main()
