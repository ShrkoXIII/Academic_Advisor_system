# seed82-concurrent-43-drop-dead-missing-flag

**Run ID:** 2026-07-27_1331__seed82-concurrent-43-drop-dead-missing-flag
**Date:** 2026-07-27T13:31:59+03:00
**Features:** 43
**Compared with:** 2026-07-27_1039__seed82-concurrent-44-registration-roster-candidate

## What changed

- seed82_concurrent_43_drop_dead_missing_flag_multiseed_TEST_closed_not_read

## Why

Record this isolated training experiment for reproducible comparison.

## Main result

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

- M1 valid fail-class Average Precision: 0.3219 -> 0.3231 (+0.0012)
- M1 valid AUC: 0.8099 -> 0.8108 (+0.0010)
- M1 valid Brier: 0.0807 -> 0.0806 (-0.0001)
- M1 train-valid AUC gap: 0.0731 -> 0.0588 (-0.0144)

## M2 regressor

- M2 valid MAE: 9.5504 -> 9.5540 (+0.0036)
- M2 valid RMSE: 12.8275 -> 12.8374 (+0.0099)
- M2 valid R2: 0.3547 -> 0.3537 (-0.0010)

## Fail-class at the locked reporting threshold (VALID)

- M1 valid fail precision: 0.3227 -> 0.3278 (+0.0051)
- M1 valid fail recall: 0.4307 -> 0.4309 (+0.0002)
- M1 valid fail F1: 0.3690 -> 0.3723 (+0.0033)

## Segment result (VALID; existing segment definitions only)

- cold_start_gpa valid AUC (n=14732): 0.7313 -> 0.7331 (+0.0018)
- first_semester valid AUC (n=14732): 0.7313 -> 0.7331 (+0.0018)
- low_difficulty_support valid AUC (n=25627): 0.7643 -> 0.7696 (+0.0053)
- retake_attempt valid AUC (n=17958): 0.6740 -> 0.6791 (+0.0051)
- **NOTE:** `first_semester` and `cold_start_gpa` have identical population size (n=14732) and identical AUC — treat them as one segment until the definitions are separated.

## Run settings

- data_rows: `{'train': 450465, 'valid': 156097}`
- dataset_version: `2026-07-26_batched_fixes__registration_roster_concurrent`
- diploma_gpa_handling: `{'method': 'train_median_fill', 'learned_from': 'train_only', 'fill_value': 84.58, 'null_counts': {'train': {'missing_before': 31, 'missing_after': 0}, 'valid': {'missing_before': 0, 'missing_after': 0}}}`
- effective_seed_settings: `{'seed': 82, 'data_random_seed': 306, 'feature_fraction_seed': 9548, 'bagging_seed': 4352, 'drop_seed': 11540}`
- feature_contract: `concurrent_44` -> `concurrent_43`  **DIFFERS FROM BASELINE**
- feature_count: `44` -> `43`  **DIFFERS FROM BASELINE**
- num_threads: `4`
- random_seed: `82`
- reporting_threshold: `0.8`
- test_policy: `closed_not_read`
- train_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_train_final.parquet`
- training_control: `{'num_boost_round': 2000, 'early_stopping_rounds': 50, 'early_stopping_selection_split': 'valid_only'}`
- valid_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_valid_final.parquet`

- Intended delta: ['feature_contract', 'feature_count']; unintended differences: none

## Important flags

- diploma_gpa has no dedicated missing-value indicator; source-level median fill remains unchanged and unmatched diploma records may be null.
