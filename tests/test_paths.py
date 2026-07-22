from pathlib import Path
import unittest

from src.paths import (
    ARTIFACTS_DIR,
    DIPLOMA_TYPE_BUCKET_MAP_PATH,
    FEATURE_ENGINEERED_PRIMARY_PATH,
    FEATURES_DIR,
    GPA_TREND_REPORTS_DIR,
    MODEL_DATA_DIR,
    MODEL_DATA_VERSIONS_DIR,
    MODEL_RUNS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    SELECTED_MODEL_POPULATION_PATH,
    model_split_path,
)


class PathContractTests(unittest.TestCase):
    def test_canonical_feature_paths_are_centralized(self) -> None:
        self.assertEqual(
            SELECTED_MODEL_POPULATION_PATH,
            FEATURES_DIR / "selected_model_population.parquet",
        )
        self.assertEqual(
            FEATURE_ENGINEERED_PRIMARY_PATH,
            FEATURES_DIR / "feature_engineered_primary.parquet",
        )
        self.assertEqual(MODEL_DATA_VERSIONS_DIR, MODEL_DATA_DIR / "versions")
        self.assertEqual(
            DIPLOMA_TYPE_BUCKET_MAP_PATH,
            ARTIFACTS_DIR / "diploma_type_bucket_map.json",
        )
        self.assertEqual(MODEL_RUNS_DIR, MODELS_DIR / "runs")
        self.assertEqual(GPA_TREND_REPORTS_DIR, REPORTS_DIR / "gpa_trend")

    def test_model_split_path_defaults_and_supports_version_roots(self) -> None:
        self.assertEqual(
            model_split_path("TRAIN", "FINAL"),
            MODEL_DATA_DIR / "df_train_final.parquet",
        )
        custom_root = Path("version-root")
        self.assertEqual(
            model_split_path("valid", "difficulty", custom_root),
            custom_root / "df_valid_difficulty.parquet",
        )

    def test_model_split_path_rejects_unknown_contract_values(self) -> None:
        for split, generation in (("holdout", "final"), ("train", "raw")):
            with self.subTest(split=split, generation=generation):
                with self.assertRaises(ValueError):
                    model_split_path(split, generation)


if __name__ == "__main__":
    unittest.main()
