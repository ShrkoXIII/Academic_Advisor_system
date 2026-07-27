# seed62-concurrent-43-drop-dead-missing-flag

**Run ID:** 2026-07-27_1329__seed62-concurrent-43-drop-dead-missing-flag
**Date:** 2026-07-27T13:29:58+03:00
**Features:** 43
**Compared with:** 2026-07-27_1033__seed62-concurrent-44-registration-roster-candidate

## What changed

- seed62_concurrent_43_drop_dead_missing_flag_multiseed_TEST_closed_not_read

## Why

Record this isolated training experiment for reproducible comparison.

## Main result

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

- M1 valid fail-class Average Precision: 0.3223 -> 0.3250 (+0.0028)
- M1 valid AUC: 0.8102 -> 0.8112 (+0.0010)
- M1 valid Brier: 0.0806 -> 0.0805 (-0.0001)
- M1 train-valid AUC gap: 0.0745 -> 0.0609 (-0.0137)

## M2 regressor

- M2 valid MAE: 9.5852 -> 9.5607 (-0.0245)
- M2 valid RMSE: 12.8861 -> 12.8347 (-0.0514)
- M2 valid R2: 0.3488 -> 0.3539 (+0.0051)

## Fail-class at the locked reporting threshold (VALID)

- M1 valid fail precision: 0.3230 -> 0.3268 (+0.0038)
- M1 valid fail recall: 0.4581 -> 0.4473 (-0.0108)
- M1 valid fail F1: 0.3789 -> 0.3777 (-0.0012)

## Segment result (VALID; existing segment definitions only)

- cold_start_gpa valid AUC (n=14732): 0.7348 -> 0.7343 (-0.0005)
- first_semester valid AUC (n=14732): 0.7348 -> 0.7343 (-0.0005)
- low_difficulty_support valid AUC (n=25627): 0.7710 -> 0.7712 (+0.0002)
- retake_attempt valid AUC (n=17958): 0.6757 -> 0.6773 (+0.0016)
- **NOTE:** `first_semester` and `cold_start_gpa` have identical population size (n=14732) and identical AUC — treat them as one segment until the definitions are separated.

## Run settings

- data_rows: `{'train': 450465, 'valid': 156097}`
- dataset_version: `2026-07-26_batched_fixes__registration_roster_concurrent`
- diploma_gpa_handling: `{'method': 'train_median_fill', 'learned_from': 'train_only', 'fill_value': 84.58, 'null_counts': {'train': {'missing_before': 31, 'missing_after': 0}, 'valid': {'missing_before': 0, 'missing_after': 0}}}`
- effective_seed_settings: `{'seed': 62, 'data_random_seed': 241, 'feature_fraction_seed': 19802, 'bagging_seed': 18760, 'drop_seed': 14704}`
- feature_contract: `concurrent_44` -> `concurrent_43`  **DIFFERS FROM BASELINE**
- feature_count: `44` -> `43`  **DIFFERS FROM BASELINE**
- num_threads: `4`
- random_seed: `62`
- reporting_threshold: `0.8`
- test_policy: `closed_not_read`
- train_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_train_final.parquet`
- training_control: `{'num_boost_round': 2000, 'early_stopping_rounds': 50, 'early_stopping_selection_split': 'valid_only'}`
- valid_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_valid_final.parquet`

- Intended delta: ['feature_contract', 'feature_count']; unintended differences: none

## Important flags

- diploma_gpa has no dedicated missing-value indicator; source-level median fill remains unchanged and unmatched diploma records may be null.
