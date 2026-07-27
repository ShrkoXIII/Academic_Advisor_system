# seed82-regr2-leaves31-concurrent-43

**Run ID:** 2026-07-27_1825__seed82-regr2-leaves31-concurrent-43
**Date:** 2026-07-27T18:25:10+03:00
**Features:** 43
**Compared with:** 2026-07-27_1331__seed82-concurrent-43-drop-dead-missing-flag

## What changed

- R2 five-seed confirmation: num_leaves 127->31, seed 82, concurrent_43, single lever

## Why

Record this isolated training experiment for reproducible comparison.

## Main result

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

- M1 valid fail-class Average Precision: 0.3231 -> 0.3212 (-0.0019)
- M1 valid AUC: 0.8108 -> 0.8091 (-0.0018)
- M1 valid Brier: 0.0806 -> 0.0808 (+0.0002)
- M1 train-valid AUC gap: 0.0588 -> 0.0363 (-0.0225)

## M2 regressor

- M2 valid MAE: 9.5540 -> 9.5992 (+0.0452)
- M2 valid RMSE: 12.8374 -> 12.8857 (+0.0483)
- M2 valid R2: 0.3537 -> 0.3488 (-0.0049)

## Fail-class at the locked reporting threshold (VALID)

- M1 valid fail precision: 0.3278 -> 0.3270 (-0.0008)
- M1 valid fail recall: 0.4309 -> 0.4322 (+0.0013)
- M1 valid fail F1: 0.3723 -> 0.3723 (+0.0000)

## Segment result (VALID; existing segment definitions only)

- cold_start_gpa valid AUC (n=14732): 0.7331 -> 0.7348 (+0.0017)
- first_semester valid AUC (n=14732): 0.7331 -> 0.7348 (+0.0017)
- low_difficulty_support valid AUC (n=25627): 0.7696 -> 0.7671 (-0.0025)
- retake_attempt valid AUC (n=17958): 0.6791 -> 0.6745 (-0.0046)
- **NOTE:** `first_semester` and `cold_start_gpa` have identical population size (n=14732) and identical AUC — treat them as one segment until the definitions are separated.

## Run settings

- data_rows: `{'train': 450465, 'valid': 156097}`
- dataset_version: `2026-07-26_batched_fixes__registration_roster_concurrent`
- diploma_gpa_handling: `{'method': 'train_median_fill', 'learned_from': 'train_only', 'fill_value': 84.58, 'null_counts': {'train': {'missing_before': 31, 'missing_after': 0}, 'valid': {'missing_before': 0, 'missing_after': 0}}}`
- effective_seed_settings: `{'seed': 82, 'data_random_seed': 306, 'feature_fraction_seed': 9548, 'bagging_seed': 4352, 'drop_seed': 11540}`
- feature_contract: `concurrent_43`
- feature_count: `43`
- lightgbm_params: `None` -> `{'m1_pass_classifier': {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 50, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'histogram_pool_size': 256, 'force_col_wise': True, 'num_threads': 4, 'verbose': -1, 'seed': 82}, 'm2_grade_regressor': {'objective': 'regression_l1', 'metric': 'mae', 'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 50, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'histogram_pool_size': 256, 'force_col_wise': True, 'num_threads': 4, 'verbose': -1, 'seed': 82}, 'differing_keys': ['metric', 'objective'], 'tuned_off_default': {'num_leaves': {'default': 127, 'this_run': 31}}}`  **DIFFERS FROM BASELINE**
- num_threads: `4`
- random_seed: `82`
- reporting_threshold: `0.8`
- test_policy: `closed_not_read`
- train_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_train_final.parquet`
- training_control: `{'num_boost_round': 2000, 'early_stopping_rounds': 50, 'early_stopping_selection_split': 'valid_only'}`
- valid_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_valid_final.parquet`

- Intended delta: none; unintended differences: none

## Important flags

- diploma_gpa has no dedicated missing-value indicator; source-level median fill remains unchanged and unmatched diploma records may be null.
