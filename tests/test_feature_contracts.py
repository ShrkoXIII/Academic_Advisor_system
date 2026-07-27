"""Named feature-contract tests for the baseline_41 vs concurrent_44 experiment.

Covers the binding contract rules: exact membership and ordering, the
set-difference identity, GPA-trend retention, audit-column exclusion, the
locked reporting threshold, contract-driven column loading, mandatory contract
selection for persistent runs, and TEST staying closed by default.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import model_training as mt
from src.concurrent_group_features import AUDIT_CONCURRENT_FEATURES
from src.model_training import (
    BASELINE_41_FEATURES,
    CONCURRENT_44_FEATURES,
    CONCURRENT_MODEL_FEATURES,
    FEATURE_CONTRACTS,
    REPORTING_THRESHOLD,
    learn_categorical_levels,
    prepare_X_y,
    resolve_feature_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION_DIR = (
    PROJECT_ROOT
    / "data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent"
)


def _synthetic_split(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A frame carrying every column the concurrent_44 contract can ask for."""
    rng = np.random.default_rng(seed)
    contract = resolve_feature_contract("concurrent_44")
    frame = pd.DataFrame(index=range(n))
    for column in contract.training_data_columns:
        frame[column] = rng.normal(size=n)
    # Columns that need real, non-Gaussian values.
    frame["requirement_size_bucket"] = rng.choice(
        list(mt.REQUIREMENT_BUCKET_ORD), size=n
    )
    frame["requirement_type_id"] = rng.choice([1, 2, 3], size=n)
    frame["diploma_type_bucket"] = rng.choice([11, 13], size=n)
    frame["difficulty_fallback_level"] = rng.choice([1, 2, 3], size=n)
    frame["is_first_active_semester"] = rng.choice([0, 1], size=n)
    frame["no_previous_progress"] = rng.choice([0, 1], size=n)
    frame["attempt_number"] = rng.choice([1, 2], size=n)
    frame["diploma_gpa"] = rng.uniform(60, 95, size=n)
    # Both classes must be present for M1.
    frame[mt.TARGET_GRADE] = rng.uniform(0, 100, size=n).round(2)
    return frame


class ContractMembershipTest(unittest.TestCase):
    def test_01_baseline_41_has_41_unique_ordered_features(self):
        self.assertEqual(len(BASELINE_41_FEATURES), 41)
        self.assertEqual(len(set(BASELINE_41_FEATURES)), 41)
        self.assertEqual(
            list(FEATURE_CONTRACTS["baseline_41"].features), BASELINE_41_FEATURES
        )

    def test_02_concurrent_44_has_44_unique_ordered_features(self):
        self.assertEqual(len(CONCURRENT_44_FEATURES), 44)
        self.assertEqual(len(set(CONCURRENT_44_FEATURES)), 44)
        self.assertEqual(
            list(FEATURE_CONTRACTS["concurrent_44"].features), CONCURRENT_44_FEATURES
        )

    def test_03_set_difference_is_exactly_the_three_concurrent_features(self):
        self.assertEqual(
            set(CONCURRENT_44_FEATURES) - set(BASELINE_41_FEATURES),
            set(CONCURRENT_MODEL_FEATURES),
        )
        self.assertEqual(set(BASELINE_41_FEATURES) - set(CONCURRENT_44_FEATURES), set())
        # Order-preserving form: dropping the three reproduces baseline_41 exactly.
        self.assertEqual(
            [f for f in CONCURRENT_44_FEATURES if f not in set(CONCURRENT_MODEL_FEATURES)],
            BASELINE_41_FEATURES,
        )
        # The three keep their accepted positions in the 44 contract.
        self.assertEqual(
            [CONCURRENT_44_FEATURES.index(f) for f in CONCURRENT_MODEL_FEATURES],
            [33, 34, 35],
        )

    def test_04_both_contracts_keep_the_gpa_trend_features(self):
        for features in (BASELINE_41_FEATURES, CONCURRENT_44_FEATURES):
            self.assertIn("gpa_trend_delta", features)
            self.assertIn("gpa_trend_missing", features)

    def test_05_concurrent_audit_columns_are_in_neither_contract(self):
        self.assertEqual(len(AUDIT_CONCURRENT_FEATURES), 5)
        for column in AUDIT_CONCURRENT_FEATURES:
            self.assertNotIn(column, BASELINE_41_FEATURES)
            self.assertNotIn(column, CONCURRENT_44_FEATURES)

    def test_14_contracts_share_every_non_feature_setting(self):
        baseline = FEATURE_CONTRACTS["baseline_41"]
        concurrent = FEATURE_CONTRACTS["concurrent_44"]
        self.assertEqual(baseline.categorical_features, concurrent.categorical_features)
        self.assertEqual(
            dict(baseline.derived_feature_sources),
            dict(concurrent.derived_feature_sources),
        )
        self.assertEqual(baseline.reporting_threshold, concurrent.reporting_threshold)
        self.assertEqual(baseline.reporting_threshold, REPORTING_THRESHOLD)
        self.assertEqual(REPORTING_THRESHOLD, 0.80)
        # Only the concurrent arm needs plan context at serve time.
        self.assertFalse(baseline.requires_concurrent_plan_context)
        self.assertTrue(concurrent.requires_concurrent_plan_context)

    def test_resolver_rejects_an_unknown_contract_name(self):
        with self.assertRaisesRegex(ValueError, "Unknown feature contract"):
            resolve_feature_contract("baseline_39")


class PrepareXYContractTest(unittest.TestCase):
    def setUp(self):
        self.df = _synthetic_split()

    def test_06_prepare_x_y_returns_exactly_the_contract_columns_in_order(self):
        for name, expected in (
            ("baseline_41", BASELINE_41_FEATURES),
            ("concurrent_44", CONCURRENT_44_FEATURES),
        ):
            with self.subTest(contract=name):
                levels = learn_categorical_levels(self.df, name)
                X, y = prepare_X_y(self.df, "pass", levels, name)
                self.assertEqual(list(X.columns), expected)
                self.assertEqual(X.shape[1], len(expected))
                self.assertEqual(len(y), len(self.df))

    def test_08_concurrent_contract_fails_loudly_on_a_missing_required_column(self):
        levels = learn_categorical_levels(self.df, "concurrent_44")
        for column in CONCURRENT_MODEL_FEATURES:
            with self.subTest(missing=column):
                broken = self.df.drop(columns=[column])
                with self.assertRaises(ValueError) as caught:
                    prepare_X_y(broken, "pass", levels, "concurrent_44")
                self.assertIn(column, str(caught.exception))
                self.assertIn("concurrent_44", str(caught.exception))
                # The same frame is perfectly valid for baseline_41.
                X, _ = prepare_X_y(broken, "pass", levels, "baseline_41")
                self.assertEqual(list(X.columns), BASELINE_41_FEATURES)

    def test_09_categorical_levels_come_from_train_only(self):
        train = self.df.iloc[:200].copy()
        valid = self.df.iloc[200:].copy()
        train["diploma_type_bucket"] = 11
        valid["diploma_type_bucket"] = 99  # unseen in train
        for name in ("baseline_41", "concurrent_44"):
            with self.subTest(contract=name):
                levels = learn_categorical_levels(train, name)
                self.assertEqual(levels["diploma_type_bucket"], [11])
                X_valid, _ = prepare_X_y(valid, "pass", levels, name)
                mapped = X_valid["diploma_type_bucket"].astype(int).unique().tolist()
                self.assertEqual(mapped, [mt.UNKNOWN_CATEGORY])

    def test_10_unknown_and_missing_categoricals_map_identically_in_both(self):
        frame = self.df.copy()
        frame.loc[frame.index[:5], "diploma_type_bucket"] = np.nan   # missing
        frame.loc[frame.index[5:10], "diploma_type_bucket"] = 999    # unseen
        train = self.df.copy()
        results = {}
        for name in ("baseline_41", "concurrent_44"):
            levels = learn_categorical_levels(train, name)
            X, _ = prepare_X_y(frame, "pass", levels, name)
            results[name] = X["diploma_type_bucket"].astype(int).to_numpy()
        np.testing.assert_array_equal(results["baseline_41"], results["concurrent_44"])
        # Both missing and unseen collapse onto the single unknown code.
        self.assertTrue((results["baseline_41"][:10] == mt.UNKNOWN_CATEGORY).all())


@unittest.skipUnless(
    (VERSION_DIR / "df_valid_final.parquet").is_file(),
    "versioned final splits not present (data/ is gitignored)",
)
class VersionedSplitReadTest(unittest.TestCase):
    """Contract-driven reading of the real 84-column VALID file."""

    # Deliberately use VALID: controlled experiments must never read TEST.
    PATH = VERSION_DIR / "df_valid_final.parquet"

    def test_07_baseline_reads_versioned_file_and_ignores_concurrent_columns(self):
        on_disk = set(pq.ParquetFile(self.PATH).schema_arrow.names)
        # The file really does carry all eight concurrent columns.
        for column in CONCURRENT_MODEL_FEATURES + AUDIT_CONCURRENT_FEATURES:
            self.assertIn(column, on_disk)

        baseline = resolve_feature_contract("baseline_41")
        self.assertFalse(
            [c for c in baseline.training_data_columns if c.startswith("concurrent")]
        )
        df = mt._read_existing_split(self.PATH, baseline)
        self.assertFalse([c for c in df.columns if c.startswith("concurrent")])

        levels = learn_categorical_levels(df, baseline)
        X, _ = prepare_X_y(df, "pass", levels, baseline)
        self.assertEqual(list(X.columns), BASELINE_41_FEATURES)

    def test_07b_concurrent_contract_reads_its_three_columns_from_the_same_file(self):
        concurrent = resolve_feature_contract("concurrent_44")
        df = mt._read_existing_split(self.PATH, concurrent)
        for column in CONCURRENT_MODEL_FEATURES:
            self.assertIn(column, df.columns)
        # The five audit columns are still left on disk.
        for column in AUDIT_CONCURRENT_FEATURES:
            self.assertNotIn(column, df.columns)


class CliAndRunArtifactTest(unittest.TestCase):
    """End-to-end main() on tiny synthetic data: contract required, TEST closed."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.train_path = root / "train.parquet"
        cls.valid_path = root / "valid.parquet"
        _synthetic_split(400, seed=1).to_parquet(cls.train_path)
        _synthetic_split(200, seed=2).to_parquet(cls.valid_path)
        # Deliberately nonexistent: reading TEST would raise FileNotFoundError.
        cls.absent_test_path = root / "test_MUST_NOT_BE_READ.parquet"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _run(self, contract_name: str, out_dir: Path, extra=()):
        mt.main([
            "--train", str(self.train_path),
            "--valid", str(self.valid_path),
            "--test", str(self.absent_test_path),
            "--out", str(out_dir),
            "--run-name", f"unit-{contract_name}",
            "--feature-contract", contract_name,
            "--num-threads", "1",
            *extra,
        ])
        runs = sorted((out_dir / "runs").glob("*__unit-*"))
        self.assertEqual(len(runs), 1, runs)
        return runs[0]

    def test_11_persistent_run_requires_an_explicit_feature_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as caught:
                mt.main([
                    "--train", str(self.train_path),
                    "--valid", str(self.valid_path),
                    "--out", tmp,
                    "--run-name", "no-contract",
                ])
            self.assertNotEqual(caught.exception.code, 0)

    def test_12_and_13_test_stays_closed_and_contract_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run_dir = self._run(
                "baseline_41", out, extra=("--seed", "52")
            )

            # (12) The test path does not exist. The run completing at all
            # proves TEST was never read.
            self.assertFalse(self.absent_test_path.exists())

            contract = json.loads(
                (run_dir / "feature_contract.json").read_text(encoding="utf-8")
            )
            # (13) contract identity + locked threshold are recorded.
            self.assertEqual(contract["contract_name"], "baseline_41")
            self.assertEqual(contract["feature_count"], 41)
            self.assertEqual(contract["ordered_features"], BASELINE_41_FEATURES)
            self.assertEqual(contract["reporting_threshold"], 0.80)
            self.assertEqual(contract["test_policy"], "closed_not_read")
            self.assertEqual(contract["random_seed"], 52)
            self.assertEqual(contract["effective_seed_settings"]["seed"], 52)
            self.assertEqual(
                set(contract["effective_seed_settings"]),
                set(mt.EFFECTIVE_SEED_PARAM_NAMES),
            )
            for value in contract["effective_seed_settings"].values():
                self.assertIsInstance(value, int)
            self.assertEqual(contract["categorical_levels_learned_from"], "train_only")
            self.assertIn("git", contract)
            self.assertIn("lightgbm_params", contract)

            metrics = json.loads(
                (run_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["run_settings"]["feature_contract"], "baseline_41")
            self.assertEqual(metrics["run_settings"]["reporting_threshold"], 0.80)
            self.assertEqual(metrics["run_settings"]["test_policy"], "closed_not_read")
            self.assertEqual(metrics["run_settings"]["random_seed"], 52)
            self.assertEqual(
                metrics["run_settings"]["effective_seed_settings"],
                contract["effective_seed_settings"],
            )
            # metrics.json now carries the threshold each P/R/F1 was cut at.
            self.assertEqual(
                metrics["m1_pass_classifier"]["valid"]["reporting_threshold"], 0.80
            )
            self.assertIsNone(metrics["m1_pass_classifier"]["test"])
            self.assertIsNone(metrics["m2_grade_regressor"]["test"])

    def test_14b_both_contracts_produce_identical_non_feature_settings(self):
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            a = json.loads(
                (
                    self._run(
                        "baseline_41", Path(tmp_a), extra=("--seed", "62")
                    )
                    / "feature_contract.json"
                )
                .read_text(encoding="utf-8")
            )
            b = json.loads(
                (
                    self._run(
                        "concurrent_44", Path(tmp_b), extra=("--seed", "62")
                    )
                    / "feature_contract.json"
                )
                .read_text(encoding="utf-8")
            )
        shared_keys = [
            "categorical_features",
            "categorical_levels",
            "unknown_category_code",
            "derived_feature_sources",
            "dropped_feature_guard",
            "target_m1_classifier",
            "target_m2_regressor",
            "reporting_threshold",
            "random_seed",
            "effective_seed_settings",
            "lightgbm_params",
            "training_control",
            "diploma_gpa_handling",
            "data_rows",
            "test_policy",
            "train_path",
            "valid_path",
        ]
        for key in shared_keys:
            with self.subTest(key=key):
                self.assertEqual(a[key], b[key])
        # Only the contract identity and its feature list may differ.
        self.assertNotEqual(a["contract_name"], b["contract_name"])
        self.assertEqual(b["feature_count"] - a["feature_count"], 3)
        self.assertEqual(
            set(b["ordered_features"]) - set(a["ordered_features"]),
            set(CONCURRENT_MODEL_FEATURES),
        )

    def test_15_changing_seed_does_not_change_the_selected_feature_contract(self):
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            seed_52 = json.loads(
                (
                    self._run(
                        "baseline_41", Path(tmp_a), extra=("--seed", "52")
                    )
                    / "feature_contract.json"
                ).read_text(encoding="utf-8")
            )
            seed_62 = json.loads(
                (
                    self._run(
                        "baseline_41", Path(tmp_b), extra=("--seed", "62")
                    )
                    / "feature_contract.json"
                ).read_text(encoding="utf-8")
            )
        self.assertEqual(seed_52["contract_name"], seed_62["contract_name"])
        self.assertEqual(seed_52["ordered_features"], seed_62["ordered_features"])
        self.assertNotEqual(
            seed_52["effective_seed_settings"],
            seed_62["effective_seed_settings"],
        )


if __name__ == "__main__":
    unittest.main()
