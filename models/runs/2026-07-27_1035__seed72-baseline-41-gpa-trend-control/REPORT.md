# seed72-baseline-41-gpa-trend-control

**Run ID:** 2026-07-27_1035__seed72-baseline-41-gpa-trend-control
**Date:** 2026-07-27T10:35:18+03:00
**Features:** 41
**Compared with:** none

## What changed

- seed72_baseline_41_controlled_multiseed_TEST_closed_not_read

## Why

Record this isolated training experiment for reproducible comparison.

## Main result

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

- M1 valid fail-class Average Precision: 0.3221 (no baseline)
- M1 valid AUC: 0.8095 (no baseline)
- M1 valid Brier: 0.0807 (no baseline)
- M1 train-valid AUC gap: 0.0582 (no baseline)

## M2 regressor

- M2 valid MAE: 9.5492 (no baseline)
- M2 valid RMSE: 12.8356 (no baseline)
- M2 valid R2: 0.3539 (no baseline)

## Fail-class at the locked reporting threshold (VALID)

- M1 valid fail precision: 0.3341 (no baseline)
- M1 valid fail recall: 0.4252 (no baseline)
- M1 valid fail F1: 0.3742 (no baseline)

## Segment result (VALID; existing segment definitions only)

- cold_start_gpa valid AUC (n=14732): 0.7354 (no baseline)
- first_semester valid AUC (n=14732): 0.7354 (no baseline)
- low_difficulty_support valid AUC (n=25627): 0.7689 (no baseline)
- retake_attempt valid AUC (n=17958): 0.6753 (no baseline)
- **NOTE:** `first_semester` and `cold_start_gpa` have identical population size (n=14732) and identical AUC — treat them as one segment until the definitions are separated.

## Run settings

- data_rows: `{'train': 450465, 'valid': 156097}`
- dataset_version: `2026-07-26_batched_fixes__registration_roster_concurrent`
- diploma_gpa_handling: `{'method': 'train_median_fill', 'learned_from': 'train_only', 'fill_value': 84.58, 'null_counts': {'train': {'missing_before': 31, 'missing_after': 0}, 'valid': {'missing_before': 0, 'missing_after': 0}}}`
- effective_seed_settings: `{'seed': 72, 'data_random_seed': 273, 'feature_fraction_seed': 31059, 'bagging_seed': 27940, 'drop_seed': 29506}`
- feature_contract: `baseline_41`
- feature_count: `41`
- num_threads: `4`
- random_seed: `72`
- reporting_threshold: `0.8`
- test_policy: `closed_not_read`
- train_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_train_final.parquet`
- training_control: `{'num_boost_round': 2000, 'early_stopping_rounds': 50, 'early_stopping_selection_split': 'valid_only'}`
- valid_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_valid_final.parquet`

## Important flags

- diploma_gpa has no dedicated missing-value indicator; source-level median fill remains unchanged and unmatched diploma records may be null.
