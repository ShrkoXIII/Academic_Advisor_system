# seed62-regr2-leaves31-concurrent-43

**Run ID:** 2026-07-27_1819__seed62-regr2-leaves31-concurrent-43
**Date:** 2026-07-27T18:19:04+03:00
**Features:** 43
**Compared with:** 2026-07-27_1329__seed62-concurrent-43-drop-dead-missing-flag

## What changed

- R2 five-seed confirmation: num_leaves 127->31, seed 62, concurrent_43, single lever

## Why

Record this isolated training experiment for reproducible comparison.

## Main result

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

- M1 valid fail-class Average Precision: 0.3250 -> 0.3216 (-0.0035)
- M1 valid AUC: 0.8112 -> 0.8092 (-0.0020)
- M1 valid Brier: 0.0805 -> 0.0807 (+0.0003)
- M1 train-valid AUC gap: 0.0609 -> 0.0436 (-0.0172)

## M2 regressor

- M2 valid MAE: 9.5607 -> 9.5849 (+0.0242)
- M2 valid RMSE: 12.8347 -> 12.8780 (+0.0433)
- M2 valid R2: 0.3539 -> 0.3496 (-0.0043)

## Fail-class at the locked reporting threshold (VALID)

- M1 valid fail precision: 0.3268 -> 0.3257 (-0.0011)
- M1 valid fail recall: 0.4473 -> 0.4335 (-0.0138)
- M1 valid fail F1: 0.3777 -> 0.3719 (-0.0058)

## Segment result (VALID; existing segment definitions only)

- cold_start_gpa valid AUC (n=14732): 0.7343 -> 0.7346 (+0.0003)
- first_semester valid AUC (n=14732): 0.7343 -> 0.7346 (+0.0003)
- low_difficulty_support valid AUC (n=25627): 0.7712 -> 0.7671 (-0.0041)
- retake_attempt valid AUC (n=17958): 0.6773 -> 0.6749 (-0.0024)
- **NOTE:** `first_semester` and `cold_start_gpa` have identical population size (n=14732) and identical AUC — treat them as one segment until the definitions are separated.

## Run settings

- data_rows: `{'train': 450465, 'valid': 156097}`
- dataset_version: `2026-07-26_batched_fixes__registration_roster_concurrent`
- diploma_gpa_handling: `{'method': 'train_median_fill', 'learned_from': 'train_only', 'fill_value': 84.58, 'null_counts': {'train': {'missing_before': 31, 'missing_after': 0}, 'valid': {'missing_before': 0, 'missing_after': 0}}}`
- effective_seed_settings: `{'seed': 62, 'data_random_seed': 241, 'feature_fraction_seed': 19802, 'bagging_seed': 18760, 'drop_seed': 14704}`
- feature_contract: `concurrent_43`
- feature_count: `43`
- lightgbm_params: `None` -> `{'m1_pass_classifier': {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 50, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'histogram_pool_size': 256, 'force_col_wise': True, 'num_threads': 4, 'verbose': -1, 'seed': 62}, 'm2_grade_regressor': {'objective': 'regression_l1', 'metric': 'mae', 'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 50, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'histogram_pool_size': 256, 'force_col_wise': True, 'num_threads': 4, 'verbose': -1, 'seed': 62}, 'differing_keys': ['metric', 'objective'], 'tuned_off_default': {'num_leaves': {'default': 127, 'this_run': 31}}}`  **DIFFERS FROM BASELINE**
- num_threads: `4`
- random_seed: `62`
- reporting_threshold: `0.8`
- test_policy: `closed_not_read`
- train_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_train_final.parquet`
- training_control: `{'num_boost_round': 2000, 'early_stopping_rounds': 50, 'early_stopping_selection_split': 'valid_only'}`
- valid_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_valid_final.parquet`

- Intended delta: none; unintended differences: none

## Important flags

- diploma_gpa has no dedicated missing-value indicator; source-level median fill remains unchanged and unmatched diploma records may be null.
