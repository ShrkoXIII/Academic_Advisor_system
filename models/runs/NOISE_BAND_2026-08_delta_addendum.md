# Paired-delta noise band — `2026-08_temporal_rebuild_v1` (addendum)

## This is a post-hoc computation

**The ten runs analysed here were trained on 2026-08-06, before this analysis
was scoped.** Nothing in this file was pre-registered. The numbers were
already on disk and already committed (`35b3588`) when the decision to
compute paired deltas was made, so the formula was chosen with the run
artifacts available. Do not cite this file as a pre-registered result, and
do not read the protocol file `noise_band_2026-08_protocol.md` as covering
it — that protocol pre-registered the raw-value measurement, not this one.

Two things do constrain the researcher freedom here, and they are the only
two claimed:

1. **The formula is not a new choice.** It is quoted verbatim from
   `models/runs/NOISE_BAND.md` (lines 32–33) and applied unchanged:

   > "A new candidate delta that falls within `[min, max]` above (or has the
   > same order of magnitude as the SD) is noise, not signal, for that metric."

   Band = `[min, max]` of the five paired deltas. No SD multiplier, then or now.

2. **The seed set is complete and fixed.** All five canonical seeds — 42, 52,
   62, 72, 82 — are present. No seed was dropped, excluded, or re-run.

## Why this addendum exists

`NOISE_BAND_2026-08_temporal_rebuild_v1.md` measured the spread of **raw
VALID metric values** across seeds. A noise band measures the spread of a
**paired delta** between two arms at the same seed — a different and
generally smaller quantity, because paired arms share their data-subsampling
and feature-fraction draws, so much of the seed noise cancels. That report
states the limitation itself in its closing section.

The paired data was never missing. `src/model_training.py` trains **both**
M1 and M2 in every invocation, so the five `baseline_41` runs also produced
M2 metrics under `baseline_41`, and the five `concurrent_43` runs also
produced M1 metrics under `concurrent_43`. That is a complete 2×5 paired
design for both models, already on disk. This addendum reads it; it trains
nothing and re-runs nothing.

## Pairing validation

`delta = concurrent_43 − baseline_41`, same seed both arms.

| Seed | Arm A (`baseline_41`) | Arm B (`concurrent_43`) |
|---:|---|---|
| 42 | `2026-08-06_1043__noiseband-2026-08-baseline41-seed42` | `2026-08-06_1048__noiseband-2026-08-concurrent43-seed42` |
| 52 | `2026-08-06_1044__noiseband-2026-08-baseline41-seed52` | `2026-08-06_1049__noiseband-2026-08-concurrent43-seed52` |
| 62 | `2026-08-06_1045__noiseband-2026-08-baseline41-seed62` | `2026-08-06_1050__noiseband-2026-08-concurrent43-seed62` |
| 72 | `2026-08-06_1046__noiseband-2026-08-baseline41-seed72` | `2026-08-06_1051__noiseband-2026-08-concurrent43-seed72` |
| 82 | `2026-08-06_1047__noiseband-2026-08-baseline41-seed82` | `2026-08-06_1052__noiseband-2026-08-concurrent43-seed82` |

Every field below was read from each run's `metrics.json` → `run_settings`
and compared across the two arms of each pair. **All five pairs passed every
check.**

| Checked field | Result across all 5 pairs |
|---|---|
| `random_seed` | matched within each pair (42/52/62/72/82) |
| `effective_seed_settings` (`seed`, `data_random_seed`, `feature_fraction_seed`, `bagging_seed`, `drop_seed`) | matched within each pair, all five derived seeds |
| `data_rows.train` | 606,562 — identical in all 10 runs |
| `data_rows.valid` | 75,380 — identical in all 10 runs |
| `lightgbm_params.m1_pass_classifier` | byte-identical within each pair (all keys) |
| `lightgbm_params.m2_grade_regressor` | byte-identical within each pair (all keys) |
| `lightgbm_params.tuned_off_default` | `{}` in all 10 runs — nothing tuned |
| `reporting_threshold` | 0.8 in all 10 runs |
| `dataset_version` | `2026-08_temporal_rebuild_v1` in all 10 runs |
| `test_policy` | `closed_not_read` in all 10 runs |
| `train_path` / `valid_path` | identical strings in all 10 runs |
| `num_threads` | 4 in all 10 runs |
| `training_control` | 2000-round cap, 50-round early stopping, `valid_only` selection — identical in all 10 |
| `diploma_gpa_handling` | train-median fill, value 85.42, identical null counts — identical in all 10 |

**Only permitted difference, observed:** `feature_contract` =
`baseline_41` (41 features) vs `concurrent_43` (43 features). No other field
differed in any pair.

## Paired deltas and the band

`delta = concurrent_43 − baseline_41`. "Dir" is the improving direction.
The band is `[min, max]` — the original formula, no SD multiplier.
"Impr." counts seeds whose delta moved in the improving direction.

| Metric | Dir | seed 42 | seed 52 | seed 62 | seed 72 | seed 82 | Mean | Median | SD | **Band [min, max]** | Impr. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 VALID AUC | higher | +0.001340 | +0.002048 | +0.002656 | -0.000610 | +0.001608 | +0.001408 | +0.001608 | 0.001233 | **[-0.000610, +0.002656]** | 4/5 |
| M1 VALID fail-class AP | higher | +0.001903 | +0.004899 | +0.002450 | -0.002620 | -0.001636 | +0.000999 | +0.001903 | 0.003089 | **[-0.002620, +0.004899]** | 3/5 |
| M1 VALID Brier | lower | -0.000153 | -0.000315 | +0.000198 | -0.000117 | +0.000393 | +0.000001 | -0.000117 | 0.000287 | **[-0.000315, +0.000393]** | 3/5 |
| M1 train-valid AUC gap | lower | -0.011955 | -0.003347 | +0.018402 | -0.015282 | +0.015486 | +0.000661 | -0.003347 | 0.015524 | **[-0.015282, +0.018402]** | 3/5 |
| M2 VALID MAE | lower | -0.000600 | +0.001200 | +0.007100 | +0.008100 | +0.000000 | +0.003160 | +0.001200 | 0.004120 | **[-0.000600, +0.008100]** | 1/5 |
| M2 VALID RMSE | lower | -0.024200 | -0.024400 | -0.022200 | +0.000400 | -0.026900 | -0.019460 | -0.024200 | 0.011227 | **[-0.026900, +0.000400]** | 4/5 |
| M2 VALID R2 | higher | +0.002400 | +0.002500 | +0.002300 | -0.000100 | +0.002800 | +0.001980 | +0.002400 | 0.001178 | **[-0.000100, +0.002800]** | 4/5 |
| Segment AUC `first_semester` | higher | -0.011066 | +0.002399 | -0.011275 | -0.011643 | -0.012050 | -0.008727 | -0.011275 | 0.006231 | **[-0.012050, +0.002399]** | 1/5 |
| Segment AUC `low_difficulty_support` | higher | -0.007175 | -0.007446 | -0.002214 | -0.000422 | -0.010300 | -0.005511 | -0.007175 | 0.004069 | **[-0.010300, -0.000422]** | 0/5 |
| Segment AUC `cold_start_gpa` | higher | -0.011066 | +0.002399 | -0.011275 | -0.011643 | -0.012050 | -0.008727 | -0.011275 | 0.006231 | **[-0.012050, +0.002399]** | 1/5 |

### Threshold dependence at 0.80

| Metric | Threshold-dependent at 0.80? |
|---|---|
| M1 VALID AUC | No — rank-based |
| M1 VALID fail-class AP | No — averaged across all thresholds |
| M1 VALID Brier | No — computed from raw probabilities |
| M1 train-valid AUC gap | No — difference of two rank-based AUCs |
| M2 VALID MAE / RMSE / R2 | No — regression; threshold does not apply |
| All three segment AUCs | No — rank-based within the segment mask |

**Every metric in this band is threshold-independent, so the whole band is
usable for model selection regardless of the 0.80 reporting cut.** The
threshold-dependent family that `evaluate_pass` also emits —
`fail_precision`, `fail_recall`, `fail_f1`, `precision`, `recall`, `f1`,
`accuracy`, `confusion_matrix` — is **not** in this band and acquires no
band from this addendum. Any future comparison on those must not borrow
these ranges.

### `first_semester` and `cold_start_gpa` are one piece of evidence, not two

Their deltas are identical to every decimal place because the two segments
are still the same population (n=7,162 at every seed) — the open defect in
`CLAUDE.md` §9. Per `CLAUDE.md` §6 they count as **one** segment result.
The table lists both rows only so neither is silently dropped.

### `Level-1-difficulty AUC` — deliberately skipped

Level 1 covers 72,018 of 75,380 VALID rows (95.5%) on this split, so its AUC
is close to a restatement of overall VALID AUC rather than an independent
segment; no separate band is reported for it.

### Precision floor on the M2 rows

`evaluate_grade` stores `mae`, `rmse`, and `r2` rounded to 4 decimals
(`src/model_training.py:841`), so every M2 delta is quantised to 0.0001 —
the seed-72 MAE delta of `+0.000000` means "below the recorded resolution",
not "exactly zero". M1 `auc`, `fail_avg_precision`, and `brier` are stored
unrounded, and the segment AUCs were read from `auc_unrounded`, so those
rows carry full precision.

## Scope

**This band applies to `2026-08_temporal_rebuild_v1` only, and its numbers
are not comparable to `models/runs/NOISE_BAND.md`'s** — that band was
measured on a 156,097-row VALID from a different split and against
`concurrent_44`, whereas this one is a 75,380-row VALID against
`concurrent_43`. Same formula, different measurement; the ranges must never
be placed side by side as if one superseded the other numerically.

## Observation recorded without interpretation

M1's AUC delta and M2's MAE delta run in opposite directions: M1 VALID AUC
improves in 4/5 seeds (mean +0.001408) while M2 VALID MAE worsens in 3/5
seeds (mean +0.003160, one seed below recorded resolution). M2's own metrics
also disagree internally — RMSE and R² both improve in 4/5 seeds while MAE
does not, and MAE is M2's training objective (`regression_l1`).

**These runs were not designed as a contract comparison and must not be read
as one.** They were produced to measure seed spread under locked contracts;
the arms differ in feature contract only as a by-product of that design.
Nothing about contract choice, model assignment, or defect status follows
from the numbers above, and none is asserted here.

## Push

Nothing was pushed. Owner-run command:

```
git push origin main
```
