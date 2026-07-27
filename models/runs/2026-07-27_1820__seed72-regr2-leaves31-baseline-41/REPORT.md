# seed72-regr2-leaves31-baseline-41

**Run ID:** 2026-07-27_1820__seed72-regr2-leaves31-baseline-41
**Date:** 2026-07-27T18:20:39+03:00
**Features:** 41
**Compared with:** 2026-07-27_1035__seed72-baseline-41-gpa-trend-control

## What changed

- R2 five-seed confirmation: num_leaves 127->31, seed 72, baseline_41, single lever

## Why

Record this isolated training experiment for reproducible comparison.

## Main result

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

- M1 valid fail-class Average Precision: 0.3221 -> 0.3208 (-0.0013)
- M1 valid AUC: 0.8095 -> 0.8090 (-0.0006)
- M1 valid Brier: 0.0807 -> 0.0809 (+0.0002)
- M1 train-valid AUC gap: 0.0582 -> 0.0371 (-0.0211)

## M2 regressor

- M2 valid MAE: 9.5492 -> 9.5761 (+0.0269)
- M2 valid RMSE: 12.8356 -> 12.8698 (+0.0342)
- M2 valid R2: 0.3539 -> 0.3504 (-0.0035)

## Fail-class at the locked reporting threshold (VALID)

- M1 valid fail precision: 0.3341 -> 0.3309 (-0.0032)
- M1 valid fail recall: 0.4252 -> 0.4109 (-0.0143)
- M1 valid fail F1: 0.3742 -> 0.3666 (-0.0076)

## Segment result (VALID; existing segment definitions only)

- cold_start_gpa valid AUC (n=14732): 0.7354 -> 0.7363 (+0.0009)
- first_semester valid AUC (n=14732): 0.7354 -> 0.7363 (+0.0009)
- low_difficulty_support valid AUC (n=25627): 0.7689 -> 0.7666 (-0.0023)
- retake_attempt valid AUC (n=17958): 0.6753 -> 0.6777 (+0.0024)
- **NOTE:** `first_semester` and `cold_start_gpa` have identical population size (n=14732) and identical AUC — treat them as one segment until the definitions are separated.

## Run settings

- data_rows: `{'train': 450465, 'valid': 156097}`
- dataset_version: `2026-07-26_batched_fixes__registration_roster_concurrent`
- diploma_gpa_handling: `{'method': 'train_median_fill', 'learned_from': 'train_only', 'fill_value': 84.58, 'null_counts': {'train': {'missing_before': 31, 'missing_after': 0}, 'valid': {'missing_before': 0, 'missing_after': 0}}}`
- effective_seed_settings: `{'seed': 72, 'data_random_seed': 273, 'feature_fraction_seed': 31059, 'bagging_seed': 27940, 'drop_seed': 29506}`
- feature_contract: `baseline_41`
- feature_count: `41`
- lightgbm_params: `None` -> `{'m1_pass_classifier': {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 50, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'histogram_pool_size': 256, 'force_col_wise': True, 'num_threads': 4, 'verbose': -1, 'seed': 72}, 'm2_grade_regressor': {'objective': 'regression_l1', 'metric': 'mae', 'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 50, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'histogram_pool_size': 256, 'force_col_wise': True, 'num_threads': 4, 'verbose': -1, 'seed': 72}, 'differing_keys': ['metric', 'objective'], 'tuned_off_default': {'num_leaves': {'default': 127, 'this_run': 31}}}`  **DIFFERS FROM BASELINE**
- num_threads: `4`
- random_seed: `72`
- reporting_threshold: `0.8`
- test_policy: `closed_not_read`
- train_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_train_final.parquet`
- training_control: `{'num_boost_round': 2000, 'early_stopping_rounds': 50, 'early_stopping_selection_split': 'valid_only'}`
- valid_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_valid_final.parquet`

- Intended delta: none; unintended differences: none

## Important flags

- diploma_gpa has no dedicated missing-value indicator; source-level median fill remains unchanged and unmatched diploma records may be null.
