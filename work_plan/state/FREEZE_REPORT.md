# Development V0 Freeze Report

**Freeze ID:** `development-v0-2026-08-22`  
**Status:** development safety reference; **not** a model/dataset promotion.  
**TEST:** not read by this freeze.

## Frozen reference

- Dataset: `2026-08_temporal_rebuild_v2`.
- TRAIN rows: 603,068.
- VALID rows: 75,155.
- Reference run: `2026-08-18_1647__rebuild-v2-baseline41-s82`.
- Development-reference M1: `m1_pass_model.lgbm`.
- Development-reference M2: `m2_grade_model.lgbm`.
- Contract used by both frozen runnable models: `baseline_41`, 41 features.
- Serving difficulty state: v2 `03_features/difficulty_state/`.

The v2 dataset metadata names `concurrent_43` as the intended M2 contract, but
there is no v2 concurrent-43 run on disk. The runnable reference therefore
freezes the seed-82 v2 baseline-41 artifa cts for both models for regression
protection only. This does not decide the eventual M2 production contract.

## Test baseline

The pre-freeze suite ran 252 tests: 249 passed and 3 were skipped. The first
run had one environment-only error because `git ls-files` rejected the sandbox
user as an unsafe repository owner. The failing test passed with a process-local
`safe.directory` setting, and the full suite then exited 0 with no failures or
errors. No global Git configuration or repository state was changed.

After adding the three freeze regression tests, the complete suite ran 255
tests in 44.279 seconds: 252 passed, 3 skipped, 0 failures, and 0 errors.

## Golden predictions

Twelve VALID rows were fixed by zero-based row position in the hash-frozen
parquet. The fixture stores no direct student identifiers and covers:

- three current-definition cold-start rows (`start_part_id == part_id`);
- three returning rows;
- three difficulty-fallback rows;
- three retake rows.

For every case the verifier pins:

- selected scalar audit fields;
- the prepared 41-feature vector SHA-256;
- M1 pass probability;
- M2 predicted mark.

## Serving limitations discovered during the freeze

The batch-model reference is reproducible, but a complete recommendation
response cannot yet be frozen:

1. `StudentScorer.load()` does not accept or recover the stored contract, and
   `Recommender.load()` cannot explicitly pass `baseline_41`.
2. A real `score_plan` attempt with the reference models stops with LightGBM's
   `train and valid dataset categorical_feature do not match` error. Current
   unit tests use dummy models and do not expose this real-artifact mismatch.
3. No complete production KNN artifact exists.
4. Official GPA, eligibility, offered-course, and set-level backtest layers are
   still missing.

These are recorded defects for later scoped tasks. They were not fixed inside
the freeze because the purpose of this step is to preserve and measure the
current state.

## Verification commands

Default immutable-artifact and golden-prediction verification:

```powershell
& .venv/Scripts/python.exe scripts/verify_development_freeze.py
```

Also verify the recorded pre-development source-code snapshot:

```powershell
& .venv/Scripts/python.exe scripts/verify_development_freeze.py --strict-code
```

The default mode intentionally does not enforce source hashes because KNN and
recommendation source files are expected to change during the product build.
The immutable data/model/difficulty hashes and golden predictions remain hard
gates.
