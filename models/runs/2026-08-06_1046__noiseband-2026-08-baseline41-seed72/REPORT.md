# noiseband-2026-08-baseline41-seed72

**Run ID:** 2026-08-06_1046__noiseband-2026-08-baseline41-seed72
**Date:** 2026-08-06T10:46:46+03:00
**Features:** 41
**Compared with:** none

## What changed

- Noise-band measurement on 2026-08_temporal_rebuild_v1; seed 72; M1 locked baseline_41; no tuning.

## Why

Record this isolated training experiment for reproducible comparison.

## Main result

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

- M1 valid fail-class Average Precision: 0.2526 (no baseline)
- M1 valid AUC: 0.8048 (no baseline)
- M1 valid Brier: 0.0651 (no baseline)
- M1 train-valid AUC gap: 0.0723 (no baseline)

## M2 regressor

- M2 valid MAE: 9.8399 (no baseline)
- M2 valid RMSE: 13.4857 (no baseline)
- M2 valid R2: 0.3108 (no baseline)

## Fail-class at the locked reporting threshold (VALID)

- M1 valid fail precision: 0.2523 (no baseline)
- M1 valid fail recall: 0.4224 (no baseline)
- M1 valid fail F1: 0.3159 (no baseline)

## Segment result (VALID; existing segment definitions only)

- cold_start_gpa valid AUC (n=7162): 0.6517 (no baseline)
- first_semester valid AUC (n=7162): 0.6517 (no baseline)
- low_difficulty_support valid AUC (n=1203): 0.7305 (no baseline)
- retake_attempt valid AUC (n=6701): 0.6802 (no baseline)
- **NOTE:** `first_semester` and `cold_start_gpa` have identical population size (n=7162) and identical AUC — treat them as one segment until the definitions are separated.

## Run settings

- data_rows: `{'train': 606562, 'valid': 75380}`
- dataset_version: `2026-08_temporal_rebuild_v1`
- diploma_gpa_handling: `{'method': 'train_median_fill', 'learned_from': 'train_only', 'fill_value': 85.42, 'null_counts': {'train': {'missing_before': 31, 'missing_after': 0}, 'valid': {'missing_before': 9, 'missing_after': 0}}}`
- effective_seed_settings: `{'seed': 72, 'data_random_seed': 273, 'feature_fraction_seed': 31059, 'bagging_seed': 27940, 'drop_seed': 29506}`
- feature_contract: `baseline_41`
- feature_count: `41`
- lightgbm_params: `{'m1_pass_classifier': {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'num_leaves': 127, 'min_child_samples': 50, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'histogram_pool_size': 256, 'force_col_wise': True, 'num_threads': 4, 'verbose': -1, 'seed': 72}, 'm2_grade_regressor': {'objective': 'regression_l1', 'metric': 'mae', 'learning_rate': 0.05, 'num_leaves': 127, 'min_child_samples': 50, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'histogram_pool_size': 256, 'force_col_wise': True, 'num_threads': 4, 'verbose': -1, 'seed': 72}, 'differing_keys': ['metric', 'objective'], 'tuned_off_default': {}}`
- num_threads: `4`
- random_seed: `72`
- reporting_threshold: `0.8`
- test_policy: `closed_not_read`
- train_path: `data/model_data/versions/2026-08_temporal_rebuild_v1/05_dataset/train_dataset_candidate.parquet`
- training_control: `{'num_boost_round': 2000, 'early_stopping_rounds': 50, 'early_stopping_selection_split': 'valid_only'}`
- valid_path: `data/model_data/versions/2026-08_temporal_rebuild_v1/05_dataset/valid_dataset_candidate.parquet`

## Important flags

- diploma_gpa has no dedicated missing-value indicator; source-level median fill remains unchanged and unmatched diploma records may be null.
