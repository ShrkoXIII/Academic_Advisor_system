# Noise band measurement protocol — `2026-08_temporal_rebuild_v1`

**Status: PRE-REGISTERED. Zero results in this file.** Committed before any
training on this split begins, per governance: a clause that appears in the
same commit as results is not pre-registered.

## Why

`models/runs/NOISE_BAND.md` was measured on the split
`2026-07-26_batched_fixes__registration_roster_concurrent`. The band is a
function of VALID's size and composition. VALID is now
`20241 + 20242 + 20243` at 75,380 rows (was 156,097) on
`2026-08_temporal_rebuild_v1`. Until a new band exists on this split, no
difference between any two runs on this split can be called meaningful.

## Fixed design

- **Seeds:** 42, 52, 62, 72, 82 — fixed and complete. No seed may be dropped
  for any reason, including looking like an outlier.
- **Models / contracts (locked, per `CLAUDE.md`):**
  - M1 = `baseline_41`
  - M2 = `concurrent_43`
  - `concurrent_44` is archived and forbidden in any new run.
- **Split version:** `2026-08_temporal_rebuild_v1`
  - TRAIN: `data/model_data/versions/2026-08_temporal_rebuild_v1/05_dataset/train_dataset_candidate.parquet`
  - VALID: `data/model_data/versions/2026-08_temporal_rebuild_v1/05_dataset/valid_dataset_candidate.parquet`
  - Row counts confirmed in Phase 0: TRAIN 606,562 / VALID 75,380. If the
    actual resolved row counts at run time differ, Phase 2 stops immediately.
  - TEST (`test_provisional_dataset_candidate.parquet`) stays closed —
    `--evaluate-test` is never passed, and every run additionally points
    `--test` at a nonexistent path as the proof mechanism, per `CLAUDE.md` §5.
- **Evaluation partition:** VALID only.
- **Hyperparameters:** locked at `_SHARED_PARAMS` defaults
  (`src/model_training.py`), including `num_leaves=127`. Nothing is tuned.
  No `scale_pos_weight`, no SMOTE, no fail-class weighting.
- **Run count:** 10 total = 2 models × 5 seeds. The seed is the only thing
  that varies between runs of the same model/contract. Each CLI invocation
  trains both the M1 pass-classifier and the M2 grade-regressor internally
  (`src/model_training.py` always trains both), but only the locked-contract
  model's metrics are used per run:
  - 5 runs at `--feature-contract baseline_41` → M1 metrics used
  - 5 runs at `--feature-contract concurrent_43` → M2 metrics used

## Metrics (identified in Phase 0, unchanged from `NOISE_BAND.md`)

The same ten metrics, read from the same places:

| Metric | Source path in `metrics.json` |
|---|---|
| M1 VALID AUC | `m1_pass_classifier.valid.auc` |
| M1 VALID fail-class AP | `m1_pass_classifier.valid.fail_avg_precision` |
| M1 VALID Brier | `m1_pass_classifier.valid.brier` |
| M1 train-valid AUC gap | `m1_pass_classifier.valid.train_valid_auc_gap` |
| M2 VALID MAE | `m2_grade_regressor.valid.mae` |
| M2 VALID RMSE | `m2_grade_regressor.valid.rmse` |
| M2 VALID R2 | `m2_grade_regressor.valid.r2` |
| Cold-start AUC (`first_semester`) | `segments.valid.first_semester.auc` |
| Low-difficulty-support AUC | `segments.valid.low_difficulty_support.auc` |
| Level-1-difficulty AUC | not persisted by `model_training.py`; computed the same way the original multiseed report scripts did (mask `difficulty_fallback_level == 1` on VALID, scored with the saved M1 booster's predict_proba) |

**Stated adaptation, not a formula change.** The original band paired two
contracts at each seed (`delta = candidate − baseline`) because that
measurement had two experimental arms per model. This measurement has only
one locked arm per model (M1 is always `baseline_41`; M2 is always
`concurrent_43` — there is no second contract to pair against). The
aggregation mechanic is unchanged — mean, median, SD, min, max computed
across the 5 seed-level values, and the band is the `[min, max]` range — it
is applied directly to the **raw VALID metric value at each seed** instead
of to a paired delta, because pairing is structurally unavailable under
locked single contracts. This is not a change to how the range/SD statistics
are computed; it is the same statistic applied to the only inputs this
design produces. Flagged explicitly here and to be repeated in the final
report so the new band is never read as directly comparable to the old
one's numbers.

## Band formula (identical mechanic to `NOISE_BAND.md`)

Quoted from `models/runs/NOISE_BAND.md` lines 30-33:

> "A new candidate delta that falls within `[min, max]` above (or has the
> same order of magnitude as the SD) is noise, not signal, for that metric."

Applied here: for each metric, across the 5 seed values,
compute mean, median, SD, min, and max. The band is `[min, max]`; SD is
reported alongside as a secondary same-order-of-magnitude check, exactly as
in the original file. No multiplier is applied to SD in either version.

## Reporting threshold

0.80 (locked, per `CLAUDE.md` §6 — not the product threshold).

## Out of scope for this pass

- No comparison to the old `NOISE_BAND.md` numbers as if they were on the
  same footing.
- No promotion, no defect fixes, no hyperparameter changes.
- No entries drafted for `Decisions_Log.md`.
