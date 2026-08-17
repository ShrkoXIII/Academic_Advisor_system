# rebuild-v1-concurrent43-s42

**Run ID:** 2026-08-16_1417__rebuild-v1-concurrent43-s42
**Date:** 2026-08-16T14:17:48+03:00
**Features:** 43
**Compared with:** 2026-08-16_1349__rebuild-v1-baseline41-s42

## What changed

- concurrent_43 vs baseline_41, same seed, 2026-08_temporal_rebuild_v1

## Why

Record this isolated training experiment for reproducible comparison.

## Main result

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

- M1 valid fail-class Average Precision: 0.2512 -> 0.2508 (-0.0003)
- M1 valid AUC: 0.8054 -> 0.8050 (-0.0003)
- M1 valid Brier: 0.0652 -> 0.0652 (+0.0000)
- M1 train-valid AUC gap: 0.0779 -> 0.0744 (-0.0036)

## M2 regressor

- M2 valid MAE: 9.8777 -> 9.8821 (+0.0044)
- M2 valid RMSE: 13.5239 -> 13.5050 (-0.0189)
- M2 valid R2: 0.3071 -> 0.3091 (+0.0020)

## Fail-class at the locked reporting threshold (VALID)

- M1 valid fail precision: 0.2536 -> 0.2491 (-0.0045)
- M1 valid fail recall: 0.4260 -> 0.4362 (+0.0102)
- M1 valid fail F1: 0.3179 -> 0.3171 (-0.0008)

## Segment result (VALID; existing segment definitions only)

- cold_start_gpa valid AUC (n=7162): 0.6559 -> 0.6444 (-0.0115)
- first_semester valid AUC (n=7162): 0.6559 -> 0.6444 (-0.0115)
- low_difficulty_support valid AUC (n=1203): 0.7361 -> 0.7273 (-0.0088)
- retake_attempt valid AUC (n=6700): 0.6745 -> 0.6762 (+0.0017)
- **NOTE:** `first_semester` and `cold_start_gpa` have identical population size (n=7162) and identical AUC — treat them as one segment until the definitions are separated.

## Run settings

- data_rows: `{'train': 606563, 'valid': 75383}`
- dataset_version: `2026-08_temporal_rebuild_v1`
- diploma_gpa_handling: `{'method': 'train_median_fill', 'learned_from': 'train_only', 'fill_value': 85.42, 'null_counts': {'train': {'missing_before': 0, 'missing_after': 0}, 'valid': {'missing_before': 0, 'missing_after': 0}}}`
- effective_seed_settings: `{'seed': 42, 'data_random_seed': 175, 'feature_fraction_seed': 30056, 'bagging_seed': 400, 'drop_seed': 17869}`
- feature_contract: `baseline_41` -> `concurrent_43`  **DIFFERS FROM BASELINE**
- feature_count: `41` -> `43`  **DIFFERS FROM BASELINE**
- lightgbm_params: `{'m1_pass_classifier': {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'num_leaves': 127, 'min_child_samples': 50, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'histogram_pool_size': 256, 'force_col_wise': True, 'num_threads': 4, 'verbose': -1, 'seed': 42}, 'm2_grade_regressor': {'objective': 'regression_l1', 'metric': 'mae', 'learning_rate': 0.05, 'num_leaves': 127, 'min_child_samples': 50, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'histogram_pool_size': 256, 'force_col_wise': True, 'num_threads': 4, 'verbose': -1, 'seed': 42}, 'differing_keys': ['metric', 'objective'], 'tuned_off_default': {}}`
- num_threads: `4`
- random_seed: `42`
- reporting_threshold: `0.8`
- test_policy: `closed_not_read`
- train_path: `data/model_data/versions/2026-08_temporal_rebuild_v1/05_dataset/train_dataset_candidate.parquet`
- training_control: `{'num_boost_round': 2000, 'early_stopping_rounds': 50, 'early_stopping_selection_split': 'valid_only'}`
- valid_path: `data/model_data/versions/2026-08_temporal_rebuild_v1/05_dataset/valid_dataset_candidate.parquet`

- Intended delta: ['feature_contract', 'feature_count']; unintended differences: none

## Important flags

- diploma_gpa has no dedicated missing-value indicator; source-level median fill remains unchanged and unmatched diploma records may be null.
