# rebuild-v2-baseline41-s62

**Run ID:** 2026-08-18_1643__rebuild-v2-baseline41-s62
**Date:** 2026-08-18T16:43:14+03:00
**Features:** 41
**Compared with:** none

## What changed

- baseline_41 on 2026-08_temporal_rebuild_v2, seed 62, patience 120

## Why

Record this isolated training experiment for reproducible comparison.

## Main result

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

- M1 valid fail-class Average Precision: 0.2392 (no baseline)
- M1 valid AUC: 0.8069 (no baseline)
- M1 valid Brier: 0.0636 (no baseline)
- M1 train-valid AUC gap: 0.0729 (no baseline)

## M2 regressor

- M2 valid MAE: 9.6444 (no baseline)
- M2 valid RMSE: 12.8542 (no baseline)
- M2 valid R2: 0.3354 (no baseline)

## Fail-class at the locked reporting threshold (VALID)

- M1 valid fail precision: 0.2411 (no baseline)
- M1 valid fail recall: 0.4250 (no baseline)
- M1 valid fail F1: 0.3076 (no baseline)

## Segment result (VALID; existing segment definitions only)

- cold_start_gpa valid AUC (n=7148): 0.6586 (no baseline)
- first_semester valid AUC (n=7148): 0.6586 (no baseline)
- low_difficulty_support valid AUC (n=1202): 0.7423 (no baseline)
- retake_attempt valid AUC (n=6626): 0.6729 (no baseline)
- **NOTE:** `first_semester` and `cold_start_gpa` have identical population size (n=7148) and identical AUC — treat them as one segment until the definitions are separated.

## Run settings

- data_rows: `{'train': 603068, 'valid': 75155}`
- dataset_version: `2026-08_temporal_rebuild_v2`
- diploma_gpa_handling: `{'method': 'train_median_fill', 'learned_from': 'train_only', 'fill_value': 85.48, 'null_counts': {'train': {'missing_before': 0, 'missing_after': 0}, 'valid': {'missing_before': 0, 'missing_after': 0}}}`
- effective_seed_settings: `{'seed': 62, 'data_random_seed': 241, 'feature_fraction_seed': 19802, 'bagging_seed': 18760, 'drop_seed': 14704}`
- feature_contract: `baseline_41`
- feature_count: `41`
- lightgbm_params: `{'m1_pass_classifier': {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'num_leaves': 127, 'min_child_samples': 50, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'histogram_pool_size': 256, 'force_col_wise': True, 'num_threads': 4, 'verbose': -1, 'seed': 62}, 'm2_grade_regressor': {'objective': 'regression_l1', 'metric': 'mae', 'learning_rate': 0.05, 'num_leaves': 127, 'min_child_samples': 50, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'histogram_pool_size': 256, 'force_col_wise': True, 'num_threads': 4, 'verbose': -1, 'seed': 62}, 'differing_keys': ['metric', 'objective'], 'tuned_off_default': {}}`
- num_threads: `4`
- random_seed: `62`
- reporting_threshold: `0.8`
- test_policy: `closed_not_read`
- train_path: `data/model_data/versions/2026-08_temporal_rebuild_v2/05_dataset/train_dataset_candidate.parquet`
- training_control: `{'num_boost_round': 2000, 'early_stopping_rounds': 120, 'early_stopping_selection_split': 'valid_only'}`
- valid_path: `data/model_data/versions/2026-08_temporal_rebuild_v2/05_dataset/valid_dataset_candidate.parquet`

## Important flags

- diploma_gpa has no dedicated missing-value indicator; source-level median fill remains unchanged and unmatched diploma records may be null.
