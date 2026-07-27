# seed52-regr2-leaves31-baseline-41

**Run ID:** 2026-07-27_1600__seed52-regr2-leaves31-baseline-41
**Date:** 2026-07-27T16:00:14+03:00
**Features:** 41
**Compared with:** 2026-07-27_1027__seed52-baseline-41-gpa-trend-control

## What changed

- R2 five-seed confirmation: num_leaves 127->31, seed 52, baseline_41, single lever

## Why

Record this isolated training experiment for reproducible comparison.

## Main result

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

- M1 valid fail-class Average Precision: 0.3224 -> 0.3232 (+0.0009)
- M1 valid AUC: 0.8100 -> 0.8092 (-0.0007)
- M1 valid Brier: 0.0807 -> 0.0807 (+0.0000)
- M1 train-valid AUC gap: 0.0687 -> 0.0357 (-0.0330)

## M2 regressor

- M2 valid MAE: 9.5715 -> 9.5851 (+0.0136)
- M2 valid RMSE: 12.8552 -> 12.8685 (+0.0133)
- M2 valid R2: 0.3519 -> 0.3505 (-0.0014)

## Fail-class at the locked reporting threshold (VALID)

- M1 valid fail precision: 0.3210 -> 0.3252 (+0.0042)
- M1 valid fail recall: 0.4431 -> 0.4256 (-0.0175)
- M1 valid fail F1: 0.3723 -> 0.3687 (-0.0036)

## Segment result (VALID; existing segment definitions only)

- cold_start_gpa valid AUC (n=14732): 0.7357 -> 0.7403 (+0.0046)
- first_semester valid AUC (n=14732): 0.7357 -> 0.7403 (+0.0046)
- low_difficulty_support valid AUC (n=25627): 0.7653 -> 0.7676 (+0.0023)
- retake_attempt valid AUC (n=17958): 0.6734 -> 0.6746 (+0.0012)
- **NOTE:** `first_semester` and `cold_start_gpa` have identical population size (n=14732) and identical AUC — treat them as one segment until the definitions are separated.

## Run settings

- data_rows: `{'train': 450465, 'valid': 156097}`
- dataset_version: `2026-07-26_batched_fixes__registration_roster_concurrent`
- diploma_gpa_handling: `{'method': 'train_median_fill', 'learned_from': 'train_only', 'fill_value': 84.58, 'null_counts': {'train': {'missing_before': 31, 'missing_after': 0}, 'valid': {'missing_before': 0, 'missing_after': 0}}}`
- effective_seed_settings: `{'seed': 52, 'data_random_seed': 208, 'feature_fraction_seed': 8545, 'bagging_seed': 9580, 'drop_seed': 32671}`
- feature_contract: `baseline_41`
- feature_count: `41`
- lightgbm_params: `None` -> `{'m1_pass_classifier': {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 50, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'histogram_pool_size': 256, 'force_col_wise': True, 'num_threads': 4, 'verbose': -1, 'seed': 52}, 'm2_grade_regressor': {'objective': 'regression_l1', 'metric': 'mae', 'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 50, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'histogram_pool_size': 256, 'force_col_wise': True, 'num_threads': 4, 'verbose': -1, 'seed': 52}, 'differing_keys': ['metric', 'objective'], 'tuned_off_default': {'num_leaves': {'default': 127, 'this_run': 31}}}`  **DIFFERS FROM BASELINE**
- num_threads: `4`
- random_seed: `52`
- reporting_threshold: `0.8`
- test_policy: `closed_not_read`
- train_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_train_final.parquet`
- training_control: `{'num_boost_round': 2000, 'early_stopping_rounds': 50, 'early_stopping_selection_split': 'valid_only'}`
- valid_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_valid_final.parquet`

- Intended delta: none; unintended differences: none

## Important flags

- diploma_gpa has no dedicated missing-value indicator; source-level median fill remains unchanged and unmatched diploma records may be null.
