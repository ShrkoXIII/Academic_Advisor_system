# reg-r4-lambda10-baseline-41

**Run ID:** 2026-07-27_1459__reg-r4-lambda10-baseline-41
**Date:** 2026-07-27T14:59:34+03:00
**Features:** 41
**Compared with:** 2026-07-26_1551__baseline-41-gpa-trend-control

## What changed

- R4 screening: reg_lambda 1.0->10.0, seed 42, single lever

## Why

Record this isolated training experiment for reproducible comparison.

## Main result

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

- M1 valid fail-class Average Precision: 0.3220 -> 0.3206 (-0.0014)
- M1 valid AUC: 0.8092 -> 0.8091 (-0.0001)
- M1 valid Brier: 0.0808 -> 0.0809 (+0.0001)
- M1 train-valid AUC gap: 0.0554 -> 0.0547 (-0.0007)

## M2 regressor

- M2 valid MAE: 9.5667 -> 9.5964 (+0.0297)
- M2 valid RMSE: 12.8549 -> 12.9027 (+0.0478)
- M2 valid R2: 0.3519 -> 0.3471 (-0.0048)

## Fail-class at the locked reporting threshold (VALID)

- M1 valid fail precision: 0.3307 -> 0.3302 (-0.0005)
- M1 valid fail recall: 0.4195 -> 0.4121 (-0.0074)
- M1 valid fail F1: 0.3698 -> 0.3666 (-0.0032)

## Segment result (VALID; existing segment definitions only)

- cold_start_gpa valid AUC (n=14732): 0.7329 -> 0.7271 (-0.0058)
- first_semester valid AUC (n=14732): 0.7329 -> 0.7271 (-0.0058)
- low_difficulty_support valid AUC (n=25627): 0.7644 -> 0.7643 (-0.0001)
- retake_attempt valid AUC (n=17958): 0.6768 -> 0.6761 (-0.0007)
- **NOTE:** `first_semester` and `cold_start_gpa` have identical population size (n=14732) and identical AUC — treat them as one segment until the definitions are separated.

## Run settings

- data_rows: `None` -> `{'train': 450465, 'valid': 156097}`  **DIFFERS FROM BASELINE**
- dataset_version: `2026-07-26_batched_fixes__registration_roster_concurrent`
- diploma_gpa_handling: `None` -> `{'method': 'train_median_fill', 'learned_from': 'train_only', 'fill_value': 84.58, 'null_counts': {'train': {'missing_before': 31, 'missing_after': 0}, 'valid': {'missing_before': 0, 'missing_after': 0}}}`  **DIFFERS FROM BASELINE**
- effective_seed_settings: `None` -> `{'seed': 42, 'data_random_seed': 175, 'feature_fraction_seed': 30056, 'bagging_seed': 400, 'drop_seed': 17869}`  **DIFFERS FROM BASELINE**
- feature_contract: `baseline_41`
- feature_count: `41`
- lightgbm_params: `None` -> `{'m1_pass_classifier': {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'num_leaves': 127, 'min_child_samples': 50, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 10.0, 'histogram_pool_size': 256, 'force_col_wise': True, 'num_threads': 4, 'verbose': -1, 'seed': 42}, 'm2_grade_regressor': {'objective': 'regression_l1', 'metric': 'mae', 'learning_rate': 0.05, 'num_leaves': 127, 'min_child_samples': 50, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 10.0, 'histogram_pool_size': 256, 'force_col_wise': True, 'num_threads': 4, 'verbose': -1, 'seed': 42}, 'differing_keys': ['metric', 'objective'], 'tuned_off_default': {'reg_lambda': {'default': 1.0, 'this_run': 10.0}}}`  **DIFFERS FROM BASELINE**
- num_threads: `4`
- random_seed: `42`
- reporting_threshold: `0.8`
- test_policy: `closed_not_read`
- train_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_train_final.parquet`
- training_control: `None` -> `{'num_boost_round': 2000, 'early_stopping_rounds': 50, 'early_stopping_selection_split': 'valid_only'}`  **DIFFERS FROM BASELINE**
- valid_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_valid_final.parquet`

- Intended delta: none; unintended differences: none

## Important flags

- diploma_gpa has no dedicated missing-value indicator; source-level median fill remains unchanged and unmatched diploma records may be null.
