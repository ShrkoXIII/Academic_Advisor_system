# Noise band — baseline_41 vs concurrent_44 (five-seed)

Source: [`MULTISEED_BASELINE41_VS_CONCURRENT44_REPORT.md`](MULTISEED_BASELINE41_VS_CONCURRENT44_REPORT.md)
(seeds 42, 52, 62, 72, 82; VALID only; TEST closed_not_read).

This is the acceptance yardstick for every future comparison against
`baseline_41` / `concurrent_44` on this dataset version. A candidate whose
paired VALID delta falls inside the ranges below is **not evidence** of a
real effect — it is indistinguishable from seed noise observed here. A
claim of improvement must clear this band, and ideally be checked across
more than one seed itself.

| Metric | Baseline mean | Candidate mean | Mean delta | Median delta | SD delta | Min delta | Max delta | Improved | Worsened |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 VALID AUC | 0.809523 | 0.810002 | +0.000479 | +0.000587 | 0.000608 | -0.000382 | +0.001042 | 4 | 1 |
| M1 VALID fail-class AP | 0.322717 | 0.322213 | -0.000505 | -0.000986 | 0.001629 | -0.002045 | +0.001544 | 2 | 3 |
| M1 VALID Brier | 0.080749 | 0.080706 | -0.0000425 | -0.0000809 | 0.0000927 | -0.000108 | +0.000119 | 4 | 1 |
| M1 train-valid AUC gap | 0.058059 | 0.065982 | +0.007922 | +0.004329 | 0.012487 | -0.005873 | +0.026720 | 1 | 4 |
| M2 VALID MAE | 9.565374 | 9.551282 | -0.014092 | -0.016550 | 0.037233 | -0.050423 | +0.046520 | 4 | 1 |
| M2 VALID RMSE | 12.849731 | 12.837753 | -0.011978 | -0.021834 | 0.055090 | -0.067477 | +0.078050 | 4 | 1 |
| M2 VALID R2 | 0.352428 | 0.353635 | +0.001207 | +0.002196 | 0.005553 | -0.007865 | +0.006807 | 4 | 1 |
| Cold-start AUC (`first_semester`) | 0.733628 | 0.734043 | +0.000415 | +0.003499 | 0.008271 | -0.011618 | +0.008190 | 3 | 2 |
| Low-difficulty-support AUC | 0.766385 | 0.767682 | +0.001297 | +0.002060 | 0.006613 | -0.006657 | +0.008522 | 3 | 2 |
| Level-1-difficulty AUC | 0.821019 | 0.821267 | +0.000248 | +0.000334 | 0.000624 | -0.000538 | +0.001140 | 3 | 2 |

Sign convention: raw delta = candidate − baseline. For Brier, AUC gap, MAE,
and RMSE, lower is better, so a *negative* delta is the improving
direction; for AUC, AP, and R2, a *positive* delta is improving.

## How to use this band

- A new candidate delta that falls within `[min, max]` above (or has the
  same order of magnitude as the SD) is noise, not signal, for that metric.
- `m1_train_valid_auc_gap` has the widest spread (SD 0.0125, range
  0.0326) and worsened in 4/5 seeds for the candidate — this is the metric
  most likely to produce a false "improvement" from a single seed.
- Segment AUCs (cold-start, low-difficulty-support, level-1) all have a
  2-vs-3 or 3-vs-2 improved/worsened split — none of them independently
  confirms a direction across seeds.
- This band is specific to this dataset version
  (`2026-07-26_batched_fixes__registration_roster_concurrent`), this pair
  of contracts, and this LightGBM configuration. It does not transfer to a
  different dataset version, a different contract pair, or a different
  training configuration without re-deriving it.
