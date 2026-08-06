# Noise band — `2026-08_temporal_rebuild_v1` (five-seed, single locked arm per model)

Protocol: [`noise_band_2026-08_protocol.md`](noise_band_2026-08_protocol.md)
(pre-registered at commit `a2cd549af02bfb4209aac4d9c591a8fe8005dc3f`, before
any run in this file was trained).

Seeds 42, 52, 62, 72, 82; VALID only; TEST `closed_not_read` in every run.
M1 = `baseline_41` (5 runs), M2 = `concurrent_43` (5 runs). 10 runs total,
seed the only thing that varies within each model's five runs.

**This band applies to the `2026-08_temporal_rebuild_v1` split only.** It is
not directly comparable to `models/runs/NOISE_BAND.md` (the
`2026-07-26_batched_fixes__registration_roster_concurrent` split) — see
"Not comparable to the old band" below.

## Run inventory

All ten runs trained both M1 and M2 internally (the CLI always does); only
the locked-contract model's metrics from each run are used below. All ten
resolved TRAIN/VALID to the same paths, with the row counts pre-registered
as the stop condition — none triggered it:

| Run ID | Contract | Seed | Resolved train path | Resolved valid path | TRAIN rows | VALID rows | TEST policy |
|---|---|---|---|---|---:|---:|---|
| `2026-08-06_1043__noiseband-2026-08-baseline41-seed42` | baseline_41 (M1) | 42 | `data/model_data/versions/2026-08_temporal_rebuild_v1/05_dataset/train_dataset_candidate.parquet` | `.../valid_dataset_candidate.parquet` | 606,562 | 75,380 | closed_not_read |
| `2026-08-06_1044__noiseband-2026-08-baseline41-seed52` | baseline_41 (M1) | 52 | same | same | 606,562 | 75,380 | closed_not_read |
| `2026-08-06_1045__noiseband-2026-08-baseline41-seed62` | baseline_41 (M1) | 62 | same | same | 606,562 | 75,380 | closed_not_read |
| `2026-08-06_1046__noiseband-2026-08-baseline41-seed72` | baseline_41 (M1) | 72 | same | same | 606,562 | 75,380 | closed_not_read |
| `2026-08-06_1047__noiseband-2026-08-baseline41-seed82` | baseline_41 (M1) | 82 | same | same | 606,562 | 75,380 | closed_not_read |
| `2026-08-06_1048__noiseband-2026-08-concurrent43-seed42` | concurrent_43 (M2) | 42 | same | same | 606,562 | 75,380 | closed_not_read |
| `2026-08-06_1049__noiseband-2026-08-concurrent43-seed52` | concurrent_43 (M2) | 52 | same | same | 606,562 | 75,380 | closed_not_read |
| `2026-08-06_1050__noiseband-2026-08-concurrent43-seed62` | concurrent_43 (M2) | 62 | same | same | 606,562 | 75,380 | closed_not_read |
| `2026-08-06_1051__noiseband-2026-08-concurrent43-seed72` | concurrent_43 (M2) | 72 | same | same | 606,562 | 75,380 | closed_not_read |
| `2026-08-06_1052__noiseband-2026-08-concurrent43-seed82` | concurrent_43 (M2) | 82 | same | same | 606,562 | 75,380 | closed_not_read |

Hyperparameters recorded identically in every run's `metrics.json` →
`run_settings.lightgbm_params`: `num_leaves=127`, `min_child_samples=50`,
`reg_lambda=1.0`, `reg_alpha=0.1`, `learning_rate=0.05`, `feature_fraction=0.8`,
`bagging_fraction=0.8`, `bagging_freq=5`, `num_threads=4`,
`force_col_wise=True` — nothing tuned. `reporting_threshold=0.8` in every
run. `dataset_version` recorded as `2026-08_temporal_rebuild_v1` in every
run's `run_settings`.

**Early stopping fired in every run** — no run reached the 2000-round cap:

| Seed | M1 best_iteration | M2 best_iteration |
|---:|---:|---:|
| 42 | 315 | 101 |
| 52 | 275 | 82 |
| 62 | 118 | 86 |
| 72 | 255 | 89 |
| 82 | 129 | 93 |

## Metrics — five seed values and the band

Band = `[min, max]` across the five seed values, per the pre-registered
formula (mean/median/SD reported alongside as the same-order-of-magnitude
secondary check, exactly as `NOISE_BAND.md` does — no multiplier on SD).
**These are raw VALID values per seed, not paired deltas** — see the stated
adaptation in the protocol and repeated below.

| Metric | seed 42 | seed 52 | seed 62 | seed 72 | seed 82 | Mean | Median | SD | Band (min, max) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 VALID AUC | 0.805055 | 0.804973 | 0.803844 | 0.804838 | 0.804848 | 0.804712 | 0.804848 | 0.000493 | [0.803844, 0.805055] |
| M1 VALID fail-class AP | 0.254592 | 0.250543 | 0.250604 | 0.252594 | 0.252561 | 0.252179 | 0.252561 | 0.001681 | [0.250543, 0.254592] |
| M1 VALID Brier | 0.065049 | 0.065232 | 0.064837 | 0.065083 | 0.064736 | 0.064988 | 0.065049 | 0.000199 | [0.064736, 0.065232] |
| M1 train-valid AUC gap | 0.078986 | 0.074412 | 0.052617 | 0.072319 | 0.053949 | 0.066456 | 0.072319 | 0.012274 | [0.052617, 0.078986] |
| M2 VALID MAE | 9.8391 | 9.8424 | 9.8474 | 9.8480 | 9.8490 | 9.84518 | 9.84740 | 0.004248 | [9.8391, 9.8490] |
| M2 VALID RMSE | 13.4691 | 13.4644 | 13.4802 | 13.4861 | 13.4795 | 13.47586 | 13.47950 | 0.008860 | [13.4644, 13.4861] |
| M2 VALID R2 | 0.3124 | 0.3129 | 0.3113 | 0.3107 | 0.3114 | 0.31174 | 0.31140 | 0.000891 | [0.3107, 0.3129] |
| Cold-start AUC (`first_semester`) | 0.6567 | 0.6465 | 0.6591 | 0.6517 | 0.6607 | 0.654940 | 0.656700 | 0.005816 | [0.6465, 0.6607] |
| Low-difficulty-support AUC | 0.7328 | 0.7307 | 0.7255 | 0.7305 | 0.7303 | 0.729960 | 0.730500 | 0.002688 | [0.7255, 0.7328] |
| Level-1-difficulty AUC | 0.805856 | 0.805855 | 0.804574 | 0.805611 | 0.805641 | 0.805507 | 0.805641 | 0.000534 | [0.804574, 0.805856] |

Sign convention unchanged from `NOISE_BAND.md`: for Brier, AUC gap, MAE, and
RMSE, lower is the improving direction; for AUC, AP, and R2, higher is
improving. That convention is about reading a future *delta*, not about
these raw values themselves.

`Level-1-difficulty AUC` was not read from `metrics.json` — `model_training.py`
does not persist this segment. It was computed by loading each seed's saved
`m1_pass_model.lgbm`, scoring VALID, and masking `difficulty_fallback_level
== 1`, mirroring the same treatment used in
`scripts/generate_multiseed_baseline41_vs_concurrent44_report.py`'s
`segment_masks()`/`segment_metrics()`. n=72,018 in every seed (VALID is
fixed across seeds; only the model differs).

## `first_semester` / `cold_start_gpa` — same known defect, confirmed on this split

Both segments returned identical n (7,162) and identical AUC at every seed
in this run set — the populations are still the same, as documented in
`CLAUDE.md` §9. Per governance, these count as **one piece of evidence**,
not two independent segment checks, on this split as on the last one.

## Difficulty-level composition on this split (context, not a defect claim)

Unlike the prior split — where `difficulty_fallback_level` was constant at
1 in TRAIN (`src/model_training.py:212`, "train is intentionally all Level
1") — TRAIN on `2026-08_temporal_rebuild_v1` is **not** constant:
Level 1 = 566,960 / 606,562 rows (93.5%), with the remainder spread across
levels 2–6. VALID is 72,018 / 75,380 (95.5%) Level 1. Stated as observed
composition only; no action taken.

## Stated adaptation — not a formula change

`NOISE_BAND.md`'s band was the range of five **paired deltas**
(`candidate_metric(seed) − baseline_metric(seed)`), because that
measurement had two experimental arms per model at each seed. This
measurement has exactly one locked arm per model — M1 is always
`baseline_41`, M2 is always `concurrent_43` — so there is no second
contract to pair against. The statistical treatment is identical (mean,
median, SD, min, max across five seed-level numbers; band = `[min, max]`;
no SD multiplier), applied here to the raw VALID metric at each seed instead
of to a delta, because pairing is structurally unavailable under locked
single contracts. This was pre-registered in the protocol before any run
was trained, not decided after seeing results.

## Not comparable to the old band

This band's numbers (e.g. M1 VALID AUC range [0.8038, 0.8051]) must never
be read side-by-side with `NOISE_BAND.md`'s numbers (e.g. M1 VALID AUC delta
range [-0.000382, +0.001042]) as if they measured the same thing:

- Old band = spread of a **candidate-minus-baseline delta**, on a 156,097-row
  VALID.
- New band = spread of a **raw metric**, on a 75,380-row VALID, composed
  differently (`20241+20242+20243` vs. the prior window).

Going forward on `2026-08_temporal_rebuild_v1`: a future run's raw VALID
metric falling inside the `[min, max]` band above is not distinguishable
from seed noise on this split. This band says nothing about deltas between
two contracts, because that measurement no longer exists under the locked
single-contract design.

## Reproducibility caveat (carried from Phase 0)

`_SHARED_PARAMS` sets `force_col_wise=True` and `num_threads=4` but not
`deterministic=True`. The derived sub-seeds
(`data_random_seed`/`feature_fraction_seed`/`bagging_seed`/`drop_seed`) are
recorded per run and confirmed identical between each run's M1 and M2
booster, so the seed **configuration** is verified reproducible. Whether
re-running the same seed a second time reproduces byte-identical output was
not tested here (out of scope for a measurement pass) and nothing in this
repository proves it either way.

## Push

Nothing here was pushed. Owner-run command:

```
git push origin main
```
