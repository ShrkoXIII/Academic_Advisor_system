# seed52-regr2-leaves31-concurrent-43

**Run ID:** 2026-07-27_1601__seed52-regr2-leaves31-concurrent-43
**Date:** 2026-07-27T16:01:10+03:00
**Features:** 43
**Compared with:** 2026-07-27_1328__seed52-concurrent-43-drop-dead-missing-flag

## What changed

- R2 five-seed confirmation: num_leaves 127->31, seed 52, concurrent_43, single lever

## Why

Record this isolated training experiment for reproducible comparison.

## Main result

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

- M1 valid fail-class Average Precision: 0.3217 -> 0.3227 (+0.0010)
- M1 valid AUC: 0.8101 -> 0.8096 (-0.0005)
- M1 valid Brier: 0.0807 -> 0.0807 (+0.0001)
- M1 train-valid AUC gap: 0.0659 -> 0.0444 (-0.0216)

## M2 regressor

- M2 valid MAE: 9.5667 -> 9.5743 (+0.0076)
- M2 valid RMSE: 12.8535 -> 12.8603 (+0.0068)
- M2 valid R2: 0.3521 -> 0.3514 (-0.0007)

## Fail-class at the locked reporting threshold (VALID)

- M1 valid fail precision: 0.3236 -> 0.3299 (+0.0063)
- M1 valid fail recall: 0.4443 -> 0.4244 (-0.0199)
- M1 valid fail F1: 0.3745 -> 0.3712 (-0.0033)

## Segment result (VALID; existing segment definitions only)

- cold_start_gpa valid AUC (n=14732): 0.7346 -> 0.7335 (-0.0011)
- first_semester valid AUC (n=14732): 0.7346 -> 0.7335 (-0.0011)
- low_difficulty_support valid AUC (n=25627): 0.7675 -> 0.7665 (-0.0010)
- retake_attempt valid AUC (n=17958): 0.6769 -> 0.6762 (-0.0007)
- **NOTE:** `first_semester` and `cold_start_gpa` have identical population size (n=14732) and identical AUC — treat them as one segment until the definitions are separated.

## Run settings

- data_rows: `{'train': 450465, 'valid': 156097}`
- dataset_version: `2026-07-26_batched_fixes__registration_roster_concurrent`
- diploma_gpa_handling: `{'method': 'train_median_fill', 'learned_from': 'train_only', 'fill_value': 84.58, 'null_counts': {'train': {'missing_before': 31, 'missing_after': 0}, 'valid': {'missing_before': 0, 'missing_after': 0}}}`
- effective_seed_settings: `{'seed': 52, 'data_random_seed': 208, 'feature_fraction_seed': 8545, 'bagging_seed': 9580, 'drop_seed': 32671}`
- feature_contract: `concurrent_43`
- feature_count: `43`
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
