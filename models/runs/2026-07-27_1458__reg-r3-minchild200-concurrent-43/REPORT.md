# reg-r3-minchild200-concurrent-43

**Run ID:** 2026-07-27_1458__reg-r3-minchild200-concurrent-43
**Date:** 2026-07-27T14:58:42+03:00
**Features:** 43
**Compared with:** 2026-07-27_1327__seed42-concurrent-43-drop-dead-missing-flag

## What changed

- R3 screening: min_child_samples 50->200, seed 42, single lever

## Why

Record this isolated training experiment for reproducible comparison.

## Main result

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

- M1 valid fail-class Average Precision: 0.3233 -> 0.3214 (-0.0019)
- M1 valid AUC: 0.8100 -> 0.8101 (+0.0001)
- M1 valid Brier: 0.0807 -> 0.0807 (+0.0001)
- M1 train-valid AUC gap: 0.0591 -> 0.0695 (+0.0104)

## M2 regressor

- M2 valid MAE: 9.5784 -> 9.5734 (-0.0050)
- M2 valid RMSE: 12.8621 -> 12.8762 (+0.0141)
- M2 valid R2: 0.3512 -> 0.3498 (-0.0014)

## Fail-class at the locked reporting threshold (VALID)

- M1 valid fail precision: 0.3273 -> 0.3224 (-0.0049)
- M1 valid fail recall: 0.4296 -> 0.4358 (+0.0062)
- M1 valid fail F1: 0.3715 -> 0.3707 (-0.0008)

## Segment result (VALID; existing segment definitions only)

- cold_start_gpa valid AUC (n=14732): 0.7325 -> 0.7340 (+0.0015)
- first_semester valid AUC (n=14732): 0.7325 -> 0.7340 (+0.0015)
- low_difficulty_support valid AUC (n=25627): 0.7673 -> 0.7669 (-0.0004)
- retake_attempt valid AUC (n=17958): 0.6785 -> 0.6738 (-0.0047)
- **NOTE:** `first_semester` and `cold_start_gpa` have identical population size (n=14732) and identical AUC — treat them as one segment until the definitions are separated.

## Run settings

- data_rows: `{'train': 450465, 'valid': 156097}`
- dataset_version: `2026-07-26_batched_fixes__registration_roster_concurrent`
- diploma_gpa_handling: `{'method': 'train_median_fill', 'learned_from': 'train_only', 'fill_value': 84.58, 'null_counts': {'train': {'missing_before': 31, 'missing_after': 0}, 'valid': {'missing_before': 0, 'missing_after': 0}}}`
- effective_seed_settings: `{'seed': 42, 'data_random_seed': 175, 'feature_fraction_seed': 30056, 'bagging_seed': 400, 'drop_seed': 17869}`
- feature_contract: `concurrent_43`
- feature_count: `43`
- lightgbm_params: `None` -> `{'m1_pass_classifier': {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'num_leaves': 127, 'min_child_samples': 200, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'histogram_pool_size': 256, 'force_col_wise': True, 'num_threads': 4, 'verbose': -1, 'seed': 42}, 'm2_grade_regressor': {'objective': 'regression_l1', 'metric': 'mae', 'learning_rate': 0.05, 'num_leaves': 127, 'min_child_samples': 200, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'histogram_pool_size': 256, 'force_col_wise': True, 'num_threads': 4, 'verbose': -1, 'seed': 42}, 'differing_keys': ['metric', 'objective'], 'tuned_off_default': {'min_child_samples': {'default': 50, 'this_run': 200}}}`  **DIFFERS FROM BASELINE**
- num_threads: `4`
- random_seed: `42`
- reporting_threshold: `0.8`
- test_policy: `closed_not_read`
- train_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_train_final.parquet`
- training_control: `{'num_boost_round': 2000, 'early_stopping_rounds': 50, 'early_stopping_selection_split': 'valid_only'}`
- valid_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_valid_final.parquet`

- Intended delta: none; unintended differences: none

## Important flags

- diploma_gpa has no dedicated missing-value indicator; source-level median fill remains unchanged and unmatched diploma records may be null.
