# seed42-concurrent-43-drop-dead-missing-flag

**Run ID:** 2026-07-27_1327__seed42-concurrent-43-drop-dead-missing-flag
**Date:** 2026-07-27T13:27:41+03:00
**Features:** 43
**Compared with:** 2026-07-26_1554__concurrent-44-registration-roster-candidate

## What changed

- seed42_concurrent_43_drop_dead_missing_flag_multiseed_TEST_closed_not_read

## Why

Record this isolated training experiment for reproducible comparison.

## Main result

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

- M1 valid fail-class Average Precision: 0.3228 -> 0.3233 (+0.0004)
- M1 valid AUC: 0.8098 -> 0.8100 (+0.0002)
- M1 valid Brier: 0.0807 -> 0.0807 (-0.0000)
- M1 train-valid AUC gap: 0.0568 -> 0.0591 (+0.0022)

## M2 regressor

- M2 valid MAE: 9.5293 -> 9.5784 (+0.0491)
- M2 valid RMSE: 12.8140 -> 12.8621 (+0.0481)
- M2 valid R2: 0.3560 -> 0.3512 (-0.0048)

## Fail-class at the locked reporting threshold (VALID)

- M1 valid fail precision: 0.3298 -> 0.3273 (-0.0025)
- M1 valid fail recall: 0.4224 -> 0.4296 (+0.0072)
- M1 valid fail F1: 0.3704 -> 0.3715 (+0.0011)

## Segment result (VALID; existing segment definitions only)

- cold_start_gpa valid AUC (n=14732): 0.7411 -> 0.7325 (-0.0086)
- first_semester valid AUC (n=14732): 0.7411 -> 0.7325 (-0.0086)
- low_difficulty_support valid AUC (n=25627): 0.7711 -> 0.7673 (-0.0038)
- retake_attempt valid AUC (n=17958): 0.6765 -> 0.6785 (+0.0020)
- **NOTE:** `first_semester` and `cold_start_gpa` have identical population size (n=14732) and identical AUC — treat them as one segment until the definitions are separated.

## Run settings

- data_rows: `None` -> `{'train': 450465, 'valid': 156097}`  **DIFFERS FROM BASELINE**
- dataset_version: `2026-07-26_batched_fixes__registration_roster_concurrent`
- diploma_gpa_handling: `None` -> `{'method': 'train_median_fill', 'learned_from': 'train_only', 'fill_value': 84.58, 'null_counts': {'train': {'missing_before': 31, 'missing_after': 0}, 'valid': {'missing_before': 0, 'missing_after': 0}}}`  **DIFFERS FROM BASELINE**
- effective_seed_settings: `None` -> `{'seed': 42, 'data_random_seed': 175, 'feature_fraction_seed': 30056, 'bagging_seed': 400, 'drop_seed': 17869}`  **DIFFERS FROM BASELINE**
- feature_contract: `concurrent_44` -> `concurrent_43`  **DIFFERS FROM BASELINE**
- feature_count: `44` -> `43`  **DIFFERS FROM BASELINE**
- num_threads: `4`
- random_seed: `42`
- reporting_threshold: `0.8`
- test_policy: `closed_not_read`
- train_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_train_final.parquet`
- training_control: `None` -> `{'num_boost_round': 2000, 'early_stopping_rounds': 50, 'early_stopping_selection_split': 'valid_only'}`  **DIFFERS FROM BASELINE**
- valid_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_valid_final.parquet`

- Intended delta: ['feature_contract', 'feature_count']; unintended differences: none

## Important flags

- diploma_gpa has no dedicated missing-value indicator; source-level median fill remains unchanged and unmatched diploma records may be null.
