# Regularization screening — seed 42, four single-lever configurations

Screening only: seed 42, four configurations, both arms = 8 runs. Seeds 52/62/72/82 were NOT run; confirmation is a separate approved task.

Acceptance rule pre-registered in [`docs/EXPERIMENT_REGULARIZATION_PLAN.md`](../../docs/EXPERIMENT_REGULARIZATION_PLAN.md), committed before any of these runs existed. No threshold was invented after seeing results.

Paired delta = **configuration minus same-contract control**, both at seed 42.

**Stated limitation.** `NOISE_BAND.md` was measured from CONTRACT-change deltas across five seeds, not from HYPERPARAMETER-change deltas. It is the best available yardstick for this pass, not an exact one. Do not treat it as precise.

## Verdicts

- **R1 (num_leaves 127 -> 63): FAIL** — failing: clause 1 (PRIMARY: M1 train-valid AUC gap outside band, beneficial, in BOTH arms).
- **R2 (num_leaves 127 -> 31): PASS** — all three pre-registered clauses satisfied in both arms.
- **R3 (min_child_samples 50 -> 200): FAIL** — failing: clause 1 (PRIMARY: M1 train-valid AUC gap outside band, beneficial, in BOTH arms).
- **R4 (reg_lambda 1.0 -> 10.0): FAIL** — failing: clause 1 (PRIMARY: M1 train-valid AUC gap outside band, beneficial, in BOTH arms); clause 2 (GUARDRAIL M1).

Candidate(s) for five-seed confirmation: R2.

## How to read these verdicts

This is ONE seed. Nothing here is confirmed; screening selects what is worth spending five seeds on, and nothing more. The caveats below are reported because they are true, not to reopen the locked rule — no verdict above was adjusted after the numbers were seen.

**Out-of-band harmful movement on metrics the rule does not score.** The pre-registered clauses cover the M1 gap, the three M1 VALID guardrails and the three M2 VALID guardrails. Segment AUCs are reported but are explicitly NOT clauses, so the following did not and must not change any verdict — they are flagged for the confirmation task to watch:

- R2 · concurrent_43 · `level_1_auc` delta -0.000705 (band -0.000538 … +0.001140) — configuration verdict remains **PASS**.

**R2: what actually moved.** A gap shrinks either because VALID improved or because TRAIN came down. Guardrail 2 exists to reject the case where the gap closed by a VALID collapse; the decomposition per arm:

| Arm | TRAIN AUC delta | VALID AUC delta | Gap delta | Mechanism |
|---|---:|---:|---:|---|
| baseline_41 | -0.005573 | +0.001520 | -0.007093 | TRAIN down **and** VALID up — genuine generalization gain |
| concurrent_43 | -0.022905 | -0.000299 | -0.022606 | mostly TRAIN coming down; VALID roughly held (no collapse, so guardrail 2 is not breached) |

Tightest M1 guardrail margin: `m1_valid_brier` in concurrent_43, delta +0.000071, only 0.000048 inside the harmful edge. Inside the band is inside the band — but this is close enough that a second seed could land the other side of it.

M2 VALID MAE worsened in: baseline_41, concurrent_43 (inside the band, so guardrail 3 is not breached). Because `_SHARED_PARAMS` is shared, this configuration cannot help M1 without also moving M2. Per B3 that is a finding to report, not a licence to split the parameters per model — that architectural decision is not made here.

## Controls (not retrained)

- `baseline_41` seed 42: `models/runs/2026-07-26_1551__baseline-41-gpa-trend-control`
- `concurrent_43` seed 42: `models/runs/2026-07-27_1327__seed42-concurrent-43-drop-dead-missing-flag`

## The eight screening runs

| Config | Lever | Arm | Run path |
|---|---|---|---|
| R1 | num_leaves 127 -> 63 | baseline_41 | `models/runs/2026-07-27_1454__reg-r1-leaves63-baseline-41` |
| R1 | num_leaves 127 -> 63 | concurrent_43 | `models/runs/2026-07-27_1455__reg-r1-leaves63-concurrent-43` |
| R2 | num_leaves 127 -> 31 | baseline_41 | `models/runs/2026-07-27_1456__reg-r2-leaves31-baseline-41` |
| R2 | num_leaves 127 -> 31 | concurrent_43 | `models/runs/2026-07-27_1457__reg-r2-leaves31-concurrent-43` |
| R3 | min_child_samples 50 -> 200 | baseline_41 | `models/runs/2026-07-27_1458__reg-r3-minchild200-baseline-41` |
| R3 | min_child_samples 50 -> 200 | concurrent_43 | `models/runs/2026-07-27_1458__reg-r3-minchild200-concurrent-43` |
| R4 | reg_lambda 1.0 -> 10.0 | baseline_41 | `models/runs/2026-07-27_1459__reg-r4-lambda10-baseline-41` |
| R4 | reg_lambda 1.0 -> 10.0 | concurrent_43 | `models/runs/2026-07-27_1500__reg-r4-lambda10-concurrent-43` |

## Control-parity verification

Each run checked against its same-contract control: contract identity and ordered features, categorical levels, reporting threshold, test policy, dataset version and TRAIN/VALID SHA-256, row counts, effective seeds (read from the serialized models), M1/M2 seed equality, the full serialized LightGBM parameter block for both models, boost-round cap, threads, early stopping, and diploma-GPA fill.

| Config | Arm | Checks | Result | Failed |
|---|---|---:|:---:|---|
| R1 | baseline_41 | 22 | PASS | none |
| R1 | concurrent_43 | 22 | PASS | none |
| R2 | baseline_41 | 22 | PASS | none |
| R2 | concurrent_43 | 22 | PASS | none |
| R3 | baseline_41 | 22 | PASS | none |
| R3 | concurrent_43 | 22 | PASS | none |
| R4 | baseline_41 | 22 | PASS | none |
| R4 | concurrent_43 | 22 | PASS | none |

The only serialized LightGBM parameter that differs between any run and its control is that run's single lever — verified independently for M1 and M2 in every one of the eight runs.

Two checks are satisfied by inference rather than by direct JSON equality, because the seed-42 `baseline_41` control predates the `data_rows`, `training_control` and `diploma_gpa_handling` fields (the provenance caveat already recorded in `Decisions_Log.md`). Row counts follow from identical TRAIN/VALID SHA-256; the diploma fill is the TRAIN median of an identically hashed TRAIN file and is therefore deterministic and equal. Both are labelled with a note in the JSON.

## Best iteration (round cap = 2000)

| Config | Arm | M1 best_iter | M1 vs control | M2 best_iter | M2 vs control | Hit cap |
|---|---|---:|---:|---:|---:|:---:|
| R1 | baseline_41 | 455 | +318 | 430 | -8 | no |
| R1 | concurrent_43 | 199 | +44 | 252 | -48 | no |
| R2 | baseline_41 | 456 | +319 | 680 | +242 | no |
| R2 | concurrent_43 | 242 | +87 | 480 | +180 | no |
| R3 | baseline_41 | 152 | +15 | 258 | -180 | no |
| R3 | concurrent_43 | 232 | +77 | 277 | -23 | no |
| R4 | baseline_41 | 152 | +15 | 420 | -18 | no |
| R4 | concurrent_43 | 160 | +5 | 540 | +240 | no |
| control | baseline_41 | 137 | — | 438 | — | no |
| control | concurrent_43 | 155 | — | 300 | — | no |

No run reached the 2000-round cap: early stopping fired in all eight screening runs and both controls, so every comparison is between converged models.

## R1 — num_leaves 127 -> 63

Verdict: **FAIL** — failing clause 1 (PRIMARY: M1 train-valid AUC gap outside band, beneficial, in BOTH arms)

### R1 · baseline_41

M1 TRAIN AUC is shown beside the gap so a gap that shrank only because TRAIN collapsed is visible.

| Metric | Control | Config | Delta | Band min | Band max | Judgment | B6 role |
|---|---:|---:|---:|---:|---:|:---|:---|
| M1 TRAIN AUC | 0.864572 | 0.877924 | +0.013352 | — | — | context | reported, not a clause |
| m1_train_valid_auc_gap | 0.055384 | 0.066863 | +0.011479 | -0.005873 | +0.026720 | inside_band | **PRIMARY (clause 1)** |
| m1_valid_auc | 0.809189 | 0.811061 | +0.001873 | -0.000382 | +0.001042 | outside_band_beneficial | guardrail (clause 2) |
| m1_valid_fail_ap | 0.321983 | 0.323924 | +0.001941 | -0.002045 | +0.001544 | outside_band_beneficial | guardrail (clause 2) |
| m1_valid_brier | 0.080778 | 0.080640 | -0.000139 | -0.000108 | +0.000119 | outside_band_beneficial | guardrail (clause 2) |
| m2_valid_mae | 9.566710 | 9.591951 | +0.025242 | -0.050423 | +0.046520 | inside_band | guardrail (clause 3) |
| m2_valid_rmse | 12.854909 | 12.900379 | +0.045470 | -0.067477 | +0.078050 | inside_band | guardrail (clause 3) |
| m2_valid_r2 | 0.351909 | 0.347316 | -0.004593 | -0.007865 | +0.006807 | inside_band | guardrail (clause 3) |
| cold_start_auc | 0.732931 | 0.739576 | +0.006645 | -0.011618 | +0.008190 | inside_band | reported, not a clause |
| low_difficulty_support_auc | 0.764405 | 0.767556 | +0.003151 | -0.006657 | +0.008522 | inside_band | reported, not a clause |
| level_1_auc | 0.820962 | 0.822689 | +0.001727 | -0.000538 | +0.001140 | outside_band_beneficial | reported, not a clause |

M2 TRAIN MAE 8.756643 -> 9.156897 (+0.400254); TRAIN RMSE 12.466085 -> 12.851654; TRAIN R2 0.491136 -> 0.459171.

M1/M2 direction: M1 VALID AUC improved, M2 VALID MAE worsened — **OPPOSED**.

### R1 · concurrent_43

M1 TRAIN AUC is shown beside the gap so a gap that shrank only because TRAIN collapsed is visible.

| Metric | Control | Config | Delta | Band min | Band max | Judgment | B6 role |
|---|---:|---:|---:|---:|---:|:---|:---|
| M1 TRAIN AUC | 0.869047 | 0.857534 | -0.011513 | — | — | context | reported, not a clause |
| m1_train_valid_auc_gap | 0.059061 | 0.047220 | -0.011840 | -0.005873 | +0.026720 | outside_band_beneficial | **PRIMARY (clause 1)** |
| m1_valid_auc | 0.809987 | 0.810314 | +0.000327 | -0.000382 | +0.001042 | inside_band | guardrail (clause 2) |
| m1_valid_fail_ap | 0.323284 | 0.323734 | +0.000450 | -0.002045 | +0.001544 | inside_band | guardrail (clause 2) |
| m1_valid_brier | 0.080660 | 0.080607 | -0.000053 | -0.000108 | +0.000119 | inside_band | guardrail (clause 2) |
| m2_valid_mae | 9.578376 | 9.597331 | +0.018956 | -0.050423 | +0.046520 | inside_band | guardrail (clause 3) |
| m2_valid_rmse | 12.862120 | 12.889290 | +0.027170 | -0.067477 | +0.078050 | inside_band | guardrail (clause 3) |
| m2_valid_r2 | 0.351182 | 0.348438 | -0.002744 | -0.007865 | +0.006807 | inside_band | guardrail (clause 3) |
| cold_start_auc | 0.732509 | 0.735455 | +0.002946 | -0.011618 | +0.008190 | inside_band | reported, not a clause |
| low_difficulty_support_auc | 0.767322 | 0.766909 | -0.000414 | -0.006657 | +0.008522 | inside_band | reported, not a clause |
| level_1_auc | 0.821214 | 0.821872 | +0.000658 | -0.000538 | +0.001140 | inside_band | reported, not a clause |

M2 TRAIN MAE 8.968135 -> 9.411695 (+0.443559); TRAIN RMSE 12.661945 -> 13.088970; TRAIN R2 0.475020 -> 0.439013.

M1/M2 direction: M1 VALID AUC improved, M2 VALID MAE worsened — **OPPOSED**.

### R1 — segment AUCs (VALID)

`first_semester` and `cold_start_gpa` are the SAME population (n=14,732, open defect) — ONE piece of evidence, not two.

| Arm | Segment | n | Control AUC | Config AUC | Delta |
|---|---|---:|---:|---:|---:|
| baseline_41 | first_semester | 14732 | 0.732931 | 0.739576 | +0.006645 |
| baseline_41 | cold_start_gpa | 14732 | 0.732931 | 0.739576 | +0.006645 |
| baseline_41 | retake_attempt | 17958 | 0.676827 | 0.674095 | -0.002732 |
| baseline_41 | low_difficulty_support | 25627 | 0.764405 | 0.767556 | +0.003151 |
| baseline_41 | level_1_difficulty | 120858 | 0.820962 | 0.822689 | +0.001727 |
| concurrent_43 | first_semester | 14732 | 0.732509 | 0.735455 | +0.002946 |
| concurrent_43 | cold_start_gpa | 14732 | 0.732509 | 0.735455 | +0.002946 |
| concurrent_43 | retake_attempt | 17958 | 0.678540 | 0.675254 | -0.003286 |
| concurrent_43 | low_difficulty_support | 25627 | 0.767322 | 0.766909 | -0.000414 |
| concurrent_43 | level_1_difficulty | 120858 | 0.821214 | 0.821872 | +0.000658 |

## R2 — num_leaves 127 -> 31

Verdict: **PASS**

### R2 · baseline_41

M1 TRAIN AUC is shown beside the gap so a gap that shrank only because TRAIN collapsed is visible.

| Metric | Control | Config | Delta | Band min | Band max | Judgment | B6 role |
|---|---:|---:|---:|---:|---:|:---|:---|
| M1 TRAIN AUC | 0.864572 | 0.858999 | -0.005573 | — | — | context | reported, not a clause |
| m1_train_valid_auc_gap | 0.055384 | 0.048291 | -0.007093 | -0.005873 | +0.026720 | outside_band_beneficial | **PRIMARY (clause 1)** |
| m1_valid_auc | 0.809189 | 0.810708 | +0.001520 | -0.000382 | +0.001042 | outside_band_beneficial | guardrail (clause 2) |
| m1_valid_fail_ap | 0.321983 | 0.324293 | +0.002310 | -0.002045 | +0.001544 | outside_band_beneficial | guardrail (clause 2) |
| m1_valid_brier | 0.080778 | 0.080741 | -0.000037 | -0.000108 | +0.000119 | inside_band | guardrail (clause 2) |
| m2_valid_mae | 9.566710 | 9.598180 | +0.031470 | -0.050423 | +0.046520 | inside_band | guardrail (clause 3) |
| m2_valid_rmse | 12.854909 | 12.881185 | +0.026276 | -0.067477 | +0.078050 | inside_band | guardrail (clause 3) |
| m2_valid_r2 | 0.351909 | 0.349257 | -0.002652 | -0.007865 | +0.006807 | inside_band | guardrail (clause 3) |
| cold_start_auc | 0.732931 | 0.735834 | +0.002902 | -0.011618 | +0.008190 | inside_band | reported, not a clause |
| low_difficulty_support_auc | 0.764405 | 0.765806 | +0.001401 | -0.006657 | +0.008522 | inside_band | reported, not a clause |
| level_1_auc | 0.820962 | 0.822428 | +0.001466 | -0.000538 | +0.001140 | outside_band_beneficial | reported, not a clause |

M2 TRAIN MAE 8.756643 -> 9.318166 (+0.561524); TRAIN RMSE 12.466085 -> 13.004004; TRAIN R2 0.491136 -> 0.446273.

M1/M2 direction: M1 VALID AUC improved, M2 VALID MAE worsened — **OPPOSED**.

### R2 · concurrent_43

M1 TRAIN AUC is shown beside the gap so a gap that shrank only because TRAIN collapsed is visible.

| Metric | Control | Config | Delta | Band min | Band max | Judgment | B6 role |
|---|---:|---:|---:|---:|---:|:---|:---|
| M1 TRAIN AUC | 0.869047 | 0.846143 | -0.022905 | — | — | context | reported, not a clause |
| m1_train_valid_auc_gap | 0.059061 | 0.036455 | -0.022606 | -0.005873 | +0.026720 | outside_band_beneficial | **PRIMARY (clause 1)** |
| m1_valid_auc | 0.809987 | 0.809688 | -0.000299 | -0.000382 | +0.001042 | inside_band | guardrail (clause 2) |
| m1_valid_fail_ap | 0.323284 | 0.322956 | -0.000328 | -0.002045 | +0.001544 | inside_band | guardrail (clause 2) |
| m1_valid_brier | 0.080660 | 0.080731 | +0.000071 | -0.000108 | +0.000119 | inside_band | guardrail (clause 2) |
| m2_valid_mae | 9.578376 | 9.607860 | +0.029485 | -0.050423 | +0.046520 | inside_band | guardrail (clause 3) |
| m2_valid_rmse | 12.862120 | 12.898573 | +0.036453 | -0.067477 | +0.078050 | inside_band | guardrail (clause 3) |
| m2_valid_r2 | 0.351182 | 0.347499 | -0.003683 | -0.007865 | +0.006807 | inside_band | guardrail (clause 3) |
| cold_start_auc | 0.732509 | 0.737105 | +0.004596 | -0.011618 | +0.008190 | inside_band | reported, not a clause |
| low_difficulty_support_auc | 0.767322 | 0.769094 | +0.001771 | -0.006657 | +0.008522 | inside_band | reported, not a clause |
| level_1_auc | 0.821214 | 0.820509 | -0.000705 | -0.000538 | +0.001140 | outside_band_harmful | reported, not a clause |

M2 TRAIN MAE 8.968135 -> 9.463626 (+0.495490); TRAIN RMSE 12.661945 -> 13.137578; TRAIN R2 0.475020 -> 0.434839.

M1/M2 direction: M1 VALID AUC worsened, M2 VALID MAE worsened — **same direction**.

### R2 — segment AUCs (VALID)

`first_semester` and `cold_start_gpa` are the SAME population (n=14,732, open defect) — ONE piece of evidence, not two.

| Arm | Segment | n | Control AUC | Config AUC | Delta |
|---|---|---:|---:|---:|---:|
| baseline_41 | first_semester | 14732 | 0.732931 | 0.735834 | +0.002902 |
| baseline_41 | cold_start_gpa | 14732 | 0.732931 | 0.735834 | +0.002902 |
| baseline_41 | retake_attempt | 17958 | 0.676827 | 0.679111 | +0.002284 |
| baseline_41 | low_difficulty_support | 25627 | 0.764405 | 0.765806 | +0.001401 |
| baseline_41 | level_1_difficulty | 120858 | 0.820962 | 0.822428 | +0.001466 |
| concurrent_43 | first_semester | 14732 | 0.732509 | 0.737105 | +0.004596 |
| concurrent_43 | cold_start_gpa | 14732 | 0.732509 | 0.737105 | +0.004596 |
| concurrent_43 | retake_attempt | 17958 | 0.678540 | 0.676522 | -0.002017 |
| concurrent_43 | low_difficulty_support | 25627 | 0.767322 | 0.769094 | +0.001771 |
| concurrent_43 | level_1_difficulty | 120858 | 0.821214 | 0.820509 | -0.000705 |

## R3 — min_child_samples 50 -> 200

Verdict: **FAIL** — failing clause 1 (PRIMARY: M1 train-valid AUC gap outside band, beneficial, in BOTH arms)

### R3 · baseline_41

M1 TRAIN AUC is shown beside the gap so a gap that shrank only because TRAIN collapsed is visible.

| Metric | Control | Config | Delta | Band min | Band max | Judgment | B6 role |
|---|---:|---:|---:|---:|---:|:---|:---|
| M1 TRAIN AUC | 0.864572 | 0.865493 | +0.000920 | — | — | context | reported, not a clause |
| m1_train_valid_auc_gap | 0.055384 | 0.055776 | +0.000392 | -0.005873 | +0.026720 | inside_band | **PRIMARY (clause 1)** |
| m1_valid_auc | 0.809189 | 0.809717 | +0.000529 | -0.000382 | +0.001042 | inside_band | guardrail (clause 2) |
| m1_valid_fail_ap | 0.321983 | 0.322350 | +0.000367 | -0.002045 | +0.001544 | inside_band | guardrail (clause 2) |
| m1_valid_brier | 0.080778 | 0.080779 | +0.000001 | -0.000108 | +0.000119 | inside_band | guardrail (clause 2) |
| m2_valid_mae | 9.566710 | 9.579480 | +0.012771 | -0.050423 | +0.046520 | inside_band | guardrail (clause 3) |
| m2_valid_rmse | 12.854909 | 12.876191 | +0.021282 | -0.067477 | +0.078050 | inside_band | guardrail (clause 3) |
| m2_valid_r2 | 0.351909 | 0.349762 | -0.002148 | -0.007865 | +0.006807 | inside_band | guardrail (clause 3) |
| cold_start_auc | 0.732931 | 0.733080 | +0.000149 | -0.011618 | +0.008190 | inside_band | reported, not a clause |
| low_difficulty_support_auc | 0.764405 | 0.768570 | +0.004164 | -0.006657 | +0.008522 | inside_band | reported, not a clause |
| level_1_auc | 0.820962 | 0.821057 | +0.000096 | -0.000538 | +0.001140 | inside_band | reported, not a clause |

M2 TRAIN MAE 8.756643 -> 9.081850 (+0.325207); TRAIN RMSE 12.466085 -> 12.790839; TRAIN R2 0.491136 -> 0.464278.

M1/M2 direction: M1 VALID AUC improved, M2 VALID MAE worsened — **OPPOSED**.

### R3 · concurrent_43

M1 TRAIN AUC is shown beside the gap so a gap that shrank only because TRAIN collapsed is visible.

| Metric | Control | Config | Delta | Band min | Band max | Judgment | B6 role |
|---|---:|---:|---:|---:|---:|:---|:---|
| M1 TRAIN AUC | 0.869047 | 0.879569 | +0.010522 | — | — | context | reported, not a clause |
| m1_train_valid_auc_gap | 0.059061 | 0.069464 | +0.010403 | -0.005873 | +0.026720 | inside_band | **PRIMARY (clause 1)** |
| m1_valid_auc | 0.809987 | 0.810105 | +0.000119 | -0.000382 | +0.001042 | inside_band | guardrail (clause 2) |
| m1_valid_fail_ap | 0.323284 | 0.321385 | -0.001900 | -0.002045 | +0.001544 | inside_band | guardrail (clause 2) |
| m1_valid_brier | 0.080660 | 0.080721 | +0.000061 | -0.000108 | +0.000119 | inside_band | guardrail (clause 2) |
| m2_valid_mae | 9.578376 | 9.573396 | -0.004980 | -0.050423 | +0.046520 | inside_band | guardrail (clause 3) |
| m2_valid_rmse | 12.862120 | 12.876215 | +0.014095 | -0.067477 | +0.078050 | inside_band | guardrail (clause 3) |
| m2_valid_r2 | 0.351182 | 0.349759 | -0.001423 | -0.007865 | +0.006807 | inside_band | guardrail (clause 3) |
| cold_start_auc | 0.732509 | 0.734005 | +0.001496 | -0.011618 | +0.008190 | inside_band | reported, not a clause |
| low_difficulty_support_auc | 0.767322 | 0.766921 | -0.000402 | -0.006657 | +0.008522 | inside_band | reported, not a clause |
| level_1_auc | 0.821214 | 0.821151 | -0.000063 | -0.000538 | +0.001140 | inside_band | reported, not a clause |

M2 TRAIN MAE 8.968135 -> 9.036397 (+0.068262); TRAIN RMSE 12.661945 -> 12.737441; TRAIN R2 0.475020 -> 0.468741.

M1/M2 direction: M1 VALID AUC improved, M2 VALID MAE improved — **same direction**.

### R3 — segment AUCs (VALID)

`first_semester` and `cold_start_gpa` are the SAME population (n=14,732, open defect) — ONE piece of evidence, not two.

| Arm | Segment | n | Control AUC | Config AUC | Delta |
|---|---|---:|---:|---:|---:|
| baseline_41 | first_semester | 14732 | 0.732931 | 0.733080 | +0.000149 |
| baseline_41 | cold_start_gpa | 14732 | 0.732931 | 0.733080 | +0.000149 |
| baseline_41 | retake_attempt | 17958 | 0.676827 | 0.677661 | +0.000833 |
| baseline_41 | low_difficulty_support | 25627 | 0.764405 | 0.768570 | +0.004164 |
| baseline_41 | level_1_difficulty | 120858 | 0.820962 | 0.821057 | +0.000096 |
| concurrent_43 | first_semester | 14732 | 0.732509 | 0.734005 | +0.001496 |
| concurrent_43 | cold_start_gpa | 14732 | 0.732509 | 0.734005 | +0.001496 |
| concurrent_43 | retake_attempt | 17958 | 0.678540 | 0.673832 | -0.004707 |
| concurrent_43 | low_difficulty_support | 25627 | 0.767322 | 0.766921 | -0.000402 |
| concurrent_43 | level_1_difficulty | 120858 | 0.821214 | 0.821151 | -0.000063 |

## R4 — reg_lambda 1.0 -> 10.0

Verdict: **FAIL** — failing clause 1 (PRIMARY: M1 train-valid AUC gap outside band, beneficial, in BOTH arms); clause 2 (GUARDRAIL M1)

### R4 · baseline_41

M1 TRAIN AUC is shown beside the gap so a gap that shrank only because TRAIN collapsed is visible.

| Metric | Control | Config | Delta | Band min | Band max | Judgment | B6 role |
|---|---:|---:|---:|---:|---:|:---|:---|
| M1 TRAIN AUC | 0.864572 | 0.863797 | -0.000776 | — | — | context | reported, not a clause |
| m1_train_valid_auc_gap | 0.055384 | 0.054699 | -0.000684 | -0.005873 | +0.026720 | inside_band | **PRIMARY (clause 1)** |
| m1_valid_auc | 0.809189 | 0.809097 | -0.000091 | -0.000382 | +0.001042 | inside_band | guardrail (clause 2) |
| m1_valid_fail_ap | 0.321983 | 0.320587 | -0.001396 | -0.002045 | +0.001544 | inside_band | guardrail (clause 2) |
| m1_valid_brier | 0.080778 | 0.080901 | +0.000122 | -0.000108 | +0.000119 | outside_band_harmful | guardrail (clause 2) |
| m2_valid_mae | 9.566710 | 9.596394 | +0.029684 | -0.050423 | +0.046520 | inside_band | guardrail (clause 3) |
| m2_valid_rmse | 12.854909 | 12.902725 | +0.047816 | -0.067477 | +0.078050 | inside_band | guardrail (clause 3) |
| m2_valid_r2 | 0.351909 | 0.347079 | -0.004830 | -0.007865 | +0.006807 | inside_band | guardrail (clause 3) |
| cold_start_auc | 0.732931 | 0.727130 | -0.005801 | -0.011618 | +0.008190 | inside_band | reported, not a clause |
| low_difficulty_support_auc | 0.764405 | 0.764307 | -0.000098 | -0.006657 | +0.008522 | inside_band | reported, not a clause |
| level_1_auc | 0.820962 | 0.820743 | -0.000218 | -0.000538 | +0.001140 | inside_band | reported, not a clause |

M2 TRAIN MAE 8.756643 -> 8.776698 (+0.020055); TRAIN RMSE 12.466085 -> 12.493619; TRAIN R2 0.491136 -> 0.488886.

M1/M2 direction: M1 VALID AUC worsened, M2 VALID MAE worsened — **same direction**.

### R4 · concurrent_43

M1 TRAIN AUC is shown beside the gap so a gap that shrank only because TRAIN collapsed is visible.

| Metric | Control | Config | Delta | Band min | Band max | Judgment | B6 role |
|---|---:|---:|---:|---:|---:|:---|:---|
| M1 TRAIN AUC | 0.869047 | 0.865975 | -0.003072 | — | — | context | reported, not a clause |
| m1_train_valid_auc_gap | 0.059061 | 0.056179 | -0.002881 | -0.005873 | +0.026720 | inside_band | **PRIMARY (clause 1)** |
| m1_valid_auc | 0.809987 | 0.809796 | -0.000191 | -0.000382 | +0.001042 | inside_band | guardrail (clause 2) |
| m1_valid_fail_ap | 0.323284 | 0.322759 | -0.000525 | -0.002045 | +0.001544 | inside_band | guardrail (clause 2) |
| m1_valid_brier | 0.080660 | 0.080718 | +0.000058 | -0.000108 | +0.000119 | inside_band | guardrail (clause 2) |
| m2_valid_mae | 9.578376 | 9.538479 | -0.039897 | -0.050423 | +0.046520 | inside_band | guardrail (clause 3) |
| m2_valid_rmse | 12.862120 | 12.804047 | -0.058073 | -0.067477 | +0.078050 | inside_band | guardrail (clause 3) |
| m2_valid_r2 | 0.351182 | 0.357028 | +0.005846 | -0.007865 | +0.006807 | inside_band | guardrail (clause 3) |
| cold_start_auc | 0.732509 | 0.731831 | -0.000678 | -0.011618 | +0.008190 | inside_band | reported, not a clause |
| low_difficulty_support_auc | 0.767322 | 0.766977 | -0.000345 | -0.006657 | +0.008522 | inside_band | reported, not a clause |
| level_1_auc | 0.821214 | 0.821022 | -0.000192 | -0.000538 | +0.001140 | inside_band | reported, not a clause |

M2 TRAIN MAE 8.968135 -> 8.607275 (-0.360860); TRAIN RMSE 12.661945 -> 12.326690; TRAIN R2 0.475020 -> 0.502452.

M1/M2 direction: M1 VALID AUC worsened, M2 VALID MAE improved — **OPPOSED**.

### R4 — segment AUCs (VALID)

`first_semester` and `cold_start_gpa` are the SAME population (n=14,732, open defect) — ONE piece of evidence, not two.

| Arm | Segment | n | Control AUC | Config AUC | Delta |
|---|---|---:|---:|---:|---:|
| baseline_41 | first_semester | 14732 | 0.732931 | 0.727130 | -0.005801 |
| baseline_41 | cold_start_gpa | 14732 | 0.732931 | 0.727130 | -0.005801 |
| baseline_41 | retake_attempt | 17958 | 0.676827 | 0.676053 | -0.000774 |
| baseline_41 | low_difficulty_support | 25627 | 0.764405 | 0.764307 | -0.000098 |
| baseline_41 | level_1_difficulty | 120858 | 0.820962 | 0.820743 | -0.000218 |
| concurrent_43 | first_semester | 14732 | 0.732509 | 0.731831 | -0.000678 |
| concurrent_43 | cold_start_gpa | 14732 | 0.732509 | 0.731831 | -0.000678 |
| concurrent_43 | retake_attempt | 17958 | 0.678540 | 0.676763 | -0.001777 |
| concurrent_43 | low_difficulty_support | 25627 | 0.767322 | 0.766977 | -0.000345 |
| concurrent_43 | level_1_difficulty | 120858 | 0.821214 | 0.821022 | -0.000192 |

## Did M1 and M2 move together?

| Config | Arm | M1 VALID AUC | M2 VALID MAE | Same direction |
|---|---|:---|:---|:---:|
| R1 | baseline_41 | improved | worsened | **NO** |
| R1 | concurrent_43 | improved | worsened | **NO** |
| R2 | baseline_41 | improved | worsened | **NO** |
| R2 | concurrent_43 | worsened | worsened | yes |
| R3 | baseline_41 | improved | worsened | **NO** |
| R3 | concurrent_43 | improved | improved | yes |
| R4 | baseline_41 | worsened | worsened | yes |
| R4 | concurrent_43 | worsened | improved | **NO** |

Every configuration moves `_SHARED_PARAMS`, so M1 and M2 always change together. Where they move in opposite directions, that is the finding B3 asks to report: per-model parameters would be a new architectural divergence and that decision is not made here.

## Integrity confirmations

- TEST is `closed_not_read` in all eight runs; every M1/M2 `test` metric field is null; each run passed a NONEXISTENT `--test` path (`data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_test_CLOSED_DO_NOT_READ.parquet`, exists=False), so completing at all proves TEST was never opened. `--evaluate-test` was never passed.
- TRAIN SHA-256 `8aaff32aeac5b375…`, VALID SHA-256 `228719fa492da84b…`, identical across all ten runs (eight screening + two controls).
- Dataset version `2026-07-26_batched_fixes__registration_roster_concurrent`; TRAIN 450,465 rows, VALID 156,097 rows. No dataset was copied or moved.
- Both controls were reused unchanged; nothing was retrained.
- Every metric above was recomputed by re-scoring the saved models against TRAIN/VALID. Only `best_iteration` is read from each run's `metrics.json`. `level_1_difficulty` is not stored in `metrics.json` and exists only because it is recomputed here.
- No `CURRENT_VERSION.txt`, promotion marker, live model artifact, default parameter, inference wiring, or recommendation wiring was changed.

Generated at commit `a6ec653841ebafe8978ac9a6edec4b7371029df6` (working tree clean: False).
