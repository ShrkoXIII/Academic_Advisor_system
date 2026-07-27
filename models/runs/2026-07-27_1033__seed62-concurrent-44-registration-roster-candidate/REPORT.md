# seed62-concurrent-44-registration-roster-candidate

**Run ID:** 2026-07-27_1033__seed62-concurrent-44-registration-roster-candidate
**Date:** 2026-07-27T10:33:26+03:00
**Features:** 44
**Compared with:** 2026-07-27_1031__seed62-baseline-41-gpa-trend-control

## What changed

- seed62_concurrent_44_controlled_multiseed_TEST_closed_not_read

## Why

Record this isolated training experiment for reproducible comparison.

## Main result

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

- M1 valid fail-class Average Precision: 0.3233 -> 0.3223 (-0.0010)
- M1 valid AUC: 0.8092 -> 0.8102 (+0.0010)
- M1 valid Brier: 0.0807 -> 0.0806 (-0.0001)
- M1 train-valid AUC gap: 0.0616 -> 0.0745 (+0.0130)

## M2 regressor

- M2 valid MAE: 9.5387 -> 9.5852 (+0.0465)
- M2 valid RMSE: 12.8081 -> 12.8861 (+0.0780)
- M2 valid R2: 0.3566 -> 0.3488 (-0.0078)

## Fail-class at the locked reporting threshold (VALID)

- M1 valid fail precision: 0.3283 -> 0.3230 (-0.0053)
- M1 valid fail recall: 0.4200 -> 0.4581 (+0.0381)
- M1 valid fail F1: 0.3685 -> 0.3789 (+0.0104)

## Segment result (VALID; existing segment definitions only)

- cold_start_gpa valid AUC (n=14732): 0.7284 -> 0.7348 (+0.0064)
- first_semester valid AUC (n=14732): 0.7284 -> 0.7348 (+0.0064)
- low_difficulty_support valid AUC (n=25627): 0.7624 -> 0.7710 (+0.0086)
- retake_attempt valid AUC (n=17958): 0.6760 -> 0.6757 (-0.0003)
- **NOTE:** `first_semester` and `cold_start_gpa` have identical population size (n=14732) and identical AUC — treat them as one segment until the definitions are separated.

## Run settings

- data_rows: `{'train': 450465, 'valid': 156097}`
- dataset_version: `2026-07-26_batched_fixes__registration_roster_concurrent`
- diploma_gpa_handling: `{'method': 'train_median_fill', 'learned_from': 'train_only', 'fill_value': 84.58, 'null_counts': {'train': {'missing_before': 31, 'missing_after': 0}, 'valid': {'missing_before': 0, 'missing_after': 0}}}`
- effective_seed_settings: `{'seed': 62, 'data_random_seed': 241, 'feature_fraction_seed': 19802, 'bagging_seed': 18760, 'drop_seed': 14704}`
- feature_contract: `baseline_41` -> `concurrent_44`  **DIFFERS FROM BASELINE**
- feature_count: `41` -> `44`  **DIFFERS FROM BASELINE**
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
