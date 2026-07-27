# seed82-regr2-leaves31-baseline-41

**Run ID:** 2026-07-27_1823__seed82-regr2-leaves31-baseline-41
**Date:** 2026-07-27T18:23:38+03:00
**Features:** 41
**Compared with:** 2026-07-27_1038__seed82-baseline-41-gpa-trend-control

## What changed

- R2 five-seed confirmation: num_leaves 127->31, seed 82, baseline_41, single lever

## Why

Record this isolated training experiment for reproducible comparison.

## Main result

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

- M1 valid fail-class Average Precision: 0.3238 -> 0.3209 (-0.0029)
- M1 valid AUC: 0.8097 -> 0.8096 (-0.0002)
- M1 valid Brier: 0.0808 -> 0.0810 (+0.0002)
- M1 train-valid AUC gap: 0.0464 -> 0.0448 (-0.0016)

## M2 regressor

- M2 valid MAE: 9.6008 -> 9.5771 (-0.0237)
- M2 valid RMSE: 12.8950 -> 12.8725 (-0.0225)
- M2 valid R2: 0.3479 -> 0.3501 (+0.0022)

## Fail-class at the locked reporting threshold (VALID)

- M1 valid fail precision: 0.3368 -> 0.3373 (+0.0005)
- M1 valid fail recall: 0.4037 -> 0.3953 (-0.0084)
- M1 valid fail F1: 0.3673 -> 0.3640 (-0.0033)

## Segment result (VALID; existing segment definitions only)

- cold_start_gpa valid AUC (n=14732): 0.7357 -> 0.7356 (-0.0001)
- first_semester valid AUC (n=14732): 0.7357 -> 0.7356 (-0.0001)
- low_difficulty_support valid AUC (n=25627): 0.7709 -> 0.7641 (-0.0068)
- retake_attempt valid AUC (n=17958): 0.6806 -> 0.6739 (-0.0067)
- **NOTE:** `first_semester` and `cold_start_gpa` have identical population size (n=14732) and identical AUC — treat them as one segment until the definitions are separated.

## Run settings

- data_rows: `{'train': 450465, 'valid': 156097}`
- dataset_version: `2026-07-26_batched_fixes__registration_roster_concurrent`
- diploma_gpa_handling: `{'method': 'train_median_fill', 'learned_from': 'train_only', 'fill_value': 84.58, 'null_counts': {'train': {'missing_before': 31, 'missing_after': 0}, 'valid': {'missing_before': 0, 'missing_after': 0}}}`
- effective_seed_settings: `{'seed': 82, 'data_random_seed': 306, 'feature_fraction_seed': 9548, 'bagging_seed': 4352, 'drop_seed': 11540}`
- feature_contract: `baseline_41`
- feature_count: `41`
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
