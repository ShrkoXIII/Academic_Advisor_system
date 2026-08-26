"""Train LightGBM pass/fail (M1) and final-mark (M2) models — V1 baseline.

Model naming — LOCKED (do not reverse):
    M1 = pass/fail CLASSIFIER   target = (final_mark >= 50).astype(int)
    M2 = final_mark REGRESSOR    target = final_mark   (raw 0-100, no transform)

Artifact mapping:
    persistent --run-name runs -> MODELS_DIR/runs/<timestamp>__<case>/
    no --run-name quick runs  -> MODELS_DIR/quick/latest/

V1 scope: LightGBM only. No sample weights. No scale_pos_weight. No XGBoost.

Usage
-----
python -m src.model_training

All arguments default to the canonical final-generation splits returned by
``src.paths.model_split_path`` (written by 03_diploma_type_bucketing) and
MODELS_DIR. Pass --train/--valid/--test/--out only to override. From a
notebook, call main([]) so Jupyter's own argv is not parsed.

This module never builds the parquet splits. Every invocation trains both
models from scratch; loading saved weights for prediction is handled by the
inference/analysis code instead.

Compatibility imports for the training CLI
------------------------------------------

The implementation lives in three focused modules:

* ``feature_contracts`` defines model inputs and targets.
* ``data_prep`` turns saved splits into model matrices.
* ``train_evaluate`` owns training, evaluation, and the CLI workflow.

This module remains the stable ``python -m src.model_training`` entry point and
keeps the names used by notebooks, tests, and report scripts available.  The
imports are explicit so readers can see that this file adds no training logic.
"""

from src.data_prep import (
    _read_existing_split,
    learn_categorical_levels,
    prepare_X_y,
    run_pre_training_diagnostics,
)
from src.feature_contracts import (
    BASELINE_41_CONTRACT,
    BASELINE_41_FEATURES,
    CATEGORICAL_FEATURES,
    CONCURRENT_43_CONTRACT,
    CONCURRENT_43_FEATURES,
    CONCURRENT_44_CONTRACT,
    CONCURRENT_44_FEATURES,
    CONCURRENT_MODEL_FEATURES,
    DEFAULT_FEATURE_CONTRACT,
    DERIVED_FEATURE_SOURCES,
    DROPPED_FEATURES,
    EXPECTED_FEATURE_COUNT,
    FEATURE_CONTRACTS,
    MODEL_FEATURES,
    REPORTING_THRESHOLD,
    REQUIREMENT_BUCKET_ORD,
    SEGMENT_ONLY_COLUMNS,
    SERVING_LIMITATION_NOTE,
    TARGET_GRADE,
    TRAINING_DATA_COLUMNS,
    UNKNOWN_CATEGORY,
    FeatureContract,
    resolve_feature_contract,
)
from src.train_evaluate import (
    EARLY_STOPPING_ROUNDS,
    EFFECTIVE_SEED_PARAM_NAMES,
    M1_METRIC,
    M1_OBJECTIVE,
    M2_METRIC,
    M2_OBJECTIVE,
    NUM_BOOST_ROUND,
    TUNABLE_PARAM_NAMES,
    _SHARED_PARAMS,
    _THRESHOLDS,
    _effective_seed_settings,
    build_run_contract,
    collect_segment_auc,
    effective_lgbm_params,
    evaluate_grade,
    evaluate_pass,
    main,
    train_grade_model,
    train_pass_model,
)


__all__ = [
    "BASELINE_41_CONTRACT",
    "BASELINE_41_FEATURES",
    "CATEGORICAL_FEATURES",
    "CONCURRENT_43_CONTRACT",
    "CONCURRENT_43_FEATURES",
    "CONCURRENT_44_CONTRACT",
    "CONCURRENT_44_FEATURES",
    "CONCURRENT_MODEL_FEATURES",
    "DEFAULT_FEATURE_CONTRACT",
    "DERIVED_FEATURE_SOURCES",
    "DROPPED_FEATURES",
    "EARLY_STOPPING_ROUNDS",
    "EFFECTIVE_SEED_PARAM_NAMES",
    "EXPECTED_FEATURE_COUNT",
    "FEATURE_CONTRACTS",
    "FeatureContract",
    "M1_METRIC",
    "M1_OBJECTIVE",
    "M2_METRIC",
    "M2_OBJECTIVE",
    "MODEL_FEATURES",
    "NUM_BOOST_ROUND",
    "REPORTING_THRESHOLD",
    "REQUIREMENT_BUCKET_ORD",
    "SEGMENT_ONLY_COLUMNS",
    "SERVING_LIMITATION_NOTE",
    "TARGET_GRADE",
    "TRAINING_DATA_COLUMNS",
    "TUNABLE_PARAM_NAMES",
    "UNKNOWN_CATEGORY",
    "_SHARED_PARAMS",
    "_THRESHOLDS",
    "_effective_seed_settings",
    "_read_existing_split",
    "build_run_contract",
    "collect_segment_auc",
    "effective_lgbm_params",
    "evaluate_grade",
    "evaluate_pass",
    "learn_categorical_levels",
    "main",
    "prepare_X_y",
    "resolve_feature_contract",
    "run_pre_training_diagnostics",
    "train_grade_model",
    "train_pass_model",
]

if __name__ == "__main__":
    main()
