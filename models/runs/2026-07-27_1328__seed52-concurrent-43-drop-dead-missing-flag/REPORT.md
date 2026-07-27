# seed52-concurrent-43-drop-dead-missing-flag

**Run ID:** 2026-07-27_1328__seed52-concurrent-43-drop-dead-missing-flag
**Date:** 2026-07-27T13:28:53+03:00
**Features:** 43
**Compared with:** 2026-07-27_1028__seed52-concurrent-44-registration-roster-candidate

## What changed

- seed52_concurrent_43_drop_dead_missing_flag_multiseed_TEST_closed_not_read

## Why

Record this isolated training experiment for reproducible comparison.

## Main result

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

- M1 valid fail-class Average Precision: 0.3203 -> 0.3217 (+0.0013)
- M1 valid AUC: 0.8096 -> 0.8101 (+0.0005)
- M1 valid Brier: 0.0808 -> 0.0807 (-0.0001)
- M1 train-valid AUC gap: 0.0731 -> 0.0659 (-0.0071)

## M2 regressor

- M2 valid MAE: 9.5549 -> 9.5667 (+0.0118)
- M2 valid RMSE: 12.8475 -> 12.8535 (+0.0060)
- M2 valid R2: 0.3527 -> 0.3521 (-0.0006)

## Fail-class at the locked reporting threshold (VALID)

- M1 valid fail precision: 0.3192 -> 0.3236 (+0.0044)
- M1 valid fail recall: 0.4346 -> 0.4443 (+0.0097)
- M1 valid fail F1: 0.3681 -> 0.3745 (+0.0064)

## Segment result (VALID; existing segment definitions only)

- cold_start_gpa valid AUC (n=14732): 0.7241 -> 0.7346 (+0.0105)
- first_semester valid AUC (n=14732): 0.7241 -> 0.7346 (+0.0105)
- low_difficulty_support valid AUC (n=25627): 0.7611 -> 0.7675 (+0.0064)
- retake_attempt valid AUC (n=17958): 0.6754 -> 0.6769 (+0.0015)
- **NOTE:** `first_semester` and `cold_start_gpa` have identical population size (n=14732) and identical AUC — treat them as one segment until the definitions are separated.

## Run settings

- data_rows: `{'train': 450465, 'valid': 156097}`
- dataset_version: `2026-07-26_batched_fixes__registration_roster_concurrent`
- diploma_gpa_handling: `{'method': 'train_median_fill', 'learned_from': 'train_only', 'fill_value': 84.58, 'null_counts': {'train': {'missing_before': 31, 'missing_after': 0}, 'valid': {'missing_before': 0, 'missing_after': 0}}}`
- effective_seed_settings: `{'seed': 52, 'data_random_seed': 208, 'feature_fraction_seed': 8545, 'bagging_seed': 9580, 'drop_seed': 32671}`
- feature_contract: `concurrent_44` -> `concurrent_43`  **DIFFERS FROM BASELINE**
- feature_count: `44` -> `43`  **DIFFERS FROM BASELINE**
- num_threads: `4`
- random_seed: `52`
- reporting_threshold: `0.8`
- test_policy: `closed_not_read`
- train_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_train_final.parquet`
- training_control: `{'num_boost_round': 2000, 'early_stopping_rounds': 50, 'early_stopping_selection_split': 'valid_only'}`
- valid_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_valid_final.parquet`

- Intended delta: ['feature_contract', 'feature_count']; unintended differences: none

## Important flags

- diploma_gpa has no dedicated missing-value indicator; source-level median fill remains unchanged and unmatched diploma records may be null.
