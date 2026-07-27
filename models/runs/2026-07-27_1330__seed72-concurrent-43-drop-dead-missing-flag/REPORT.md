# seed72-concurrent-43-drop-dead-missing-flag

**Run ID:** 2026-07-27_1330__seed72-concurrent-43-drop-dead-missing-flag
**Date:** 2026-07-27T13:30:55+03:00
**Features:** 43
**Compared with:** 2026-07-27_1036__seed72-concurrent-44-registration-roster-candidate

## What changed

- seed72_concurrent_43_drop_dead_missing_flag_multiseed_TEST_closed_not_read

## Why

Record this isolated training experiment for reproducible comparison.

## Main result

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

- M1 valid fail-class Average Precision: 0.3237 -> 0.3242 (+0.0005)
- M1 valid AUC: 0.8106 -> 0.8101 (-0.0004)
- M1 valid Brier: 0.0807 -> 0.0806 (-0.0001)
- M1 train-valid AUC gap: 0.0523 -> 0.0533 (+0.0010)

## M2 regressor

- M2 valid MAE: 9.5366 -> 9.5555 (+0.0189)
- M2 valid RMSE: 12.8137 -> 12.8495 (+0.0358)
- M2 valid R2: 0.3561 -> 0.3525 (-0.0036)

## Fail-class at the locked reporting threshold (VALID)

- M1 valid fail precision: 0.3368 -> 0.3265 (-0.0103)
- M1 valid fail recall: 0.4169 -> 0.4302 (+0.0133)
- M1 valid fail F1: 0.3726 -> 0.3712 (-0.0014)

## Segment result (VALID; existing segment definitions only)

- cold_start_gpa valid AUC (n=14732): 0.7389 -> 0.7353 (-0.0036)
- first_semester valid AUC (n=14732): 0.7389 -> 0.7353 (-0.0036)
- low_difficulty_support valid AUC (n=25627): 0.7709 -> 0.7700 (-0.0009)
- retake_attempt valid AUC (n=17958): 0.6786 -> 0.6786 (+0.0000)
- **NOTE:** `first_semester` and `cold_start_gpa` have identical population size (n=14732) and identical AUC — treat them as one segment until the definitions are separated.

## Run settings

- data_rows: `{'train': 450465, 'valid': 156097}`
- dataset_version: `2026-07-26_batched_fixes__registration_roster_concurrent`
- diploma_gpa_handling: `{'method': 'train_median_fill', 'learned_from': 'train_only', 'fill_value': 84.58, 'null_counts': {'train': {'missing_before': 31, 'missing_after': 0}, 'valid': {'missing_before': 0, 'missing_after': 0}}}`
- effective_seed_settings: `{'seed': 72, 'data_random_seed': 273, 'feature_fraction_seed': 31059, 'bagging_seed': 27940, 'drop_seed': 29506}`
- feature_contract: `concurrent_44` -> `concurrent_43`  **DIFFERS FROM BASELINE**
- feature_count: `44` -> `43`  **DIFFERS FROM BASELINE**
- num_threads: `4`
- random_seed: `72`
- reporting_threshold: `0.8`
- test_policy: `closed_not_read`
- train_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_train_final.parquet`
- training_control: `{'num_boost_round': 2000, 'early_stopping_rounds': 50, 'early_stopping_selection_split': 'valid_only'}`
- valid_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_valid_final.parquet`

- Intended delta: ['feature_contract', 'feature_count']; unintended differences: none

## Important flags

- diploma_gpa has no dedicated missing-value indicator; source-level median fill remains unchanged and unmatched diploma records may be null.
