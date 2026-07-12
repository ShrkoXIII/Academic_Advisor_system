import unittest

import pandas as pd

from src.model_training import (
    CATEGORICAL_FEATURES,
    EXPECTED_FEATURE_COUNT,
    MODEL_FEATURES,
    learn_categorical_levels,
    prepare_X_y,
)


class ModelTrainingContractTests(unittest.TestCase):
    def _frame(self):
        row = {feature: 0 for feature in MODEL_FEATURES}
        row.update(
            {
                "final_mark": 70.0,
                "requirement_size_bucket": "none_or_unknown",
                "requirement_type_id": 1,
                "diploma_type_bucket": 13,
                "diploma_gpa": 85.0,
            }
        )
        return pd.DataFrame([row, {**row, "final_mark": 40.0}])

    def test_diploma_features_extend_the_contract_without_raw_type_id(self):
        self.assertEqual(len(MODEL_FEATURES), EXPECTED_FEATURE_COUNT)
        self.assertEqual(EXPECTED_FEATURE_COUNT, 39)
        self.assertIn("diploma_gpa", MODEL_FEATURES)
        self.assertIn("diploma_type_bucket", MODEL_FEATURES)
        self.assertEqual(
            CATEGORICAL_FEATURES,
            ["requirement_type_id", "diploma_type_bucket"],
        )
        self.assertNotIn("start_level_missing", MODEL_FEATURES)
        self.assertNotIn("difficulty_fallback_level", MODEL_FEATURES)
        self.assertNotIn("diploma_type_id", MODEL_FEATURES)

    def test_diploma_type_bucket_is_train_fitted_categorical(self):
        train = self._frame()
        levels = learn_categorical_levels(train)
        valid = self._frame()
        valid.loc[0, "diploma_type_bucket"] = 999
        valid.loc[1, "diploma_type_bucket"] = pd.NA
        X, _ = prepare_X_y(valid, "pass", levels)
        self.assertEqual(str(X["diploma_type_bucket"].dtype), "category")
        self.assertEqual(X["diploma_type_bucket"].astype(int).tolist(), [-1, -1])
        self.assertEqual(str(X["diploma_gpa"].dtype), "float64")
