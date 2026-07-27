# R2 (`num_leaves` 31) — five-seed confirmation

Eight new runs (seeds 52, 62, 72, 82 × two arms), plus the seed-42 R2 pair reused from screening, each compared against its same-seed same-contract DEFAULT-parameter control. Ten controls reused unchanged; nothing was retrained.

Analysis rule pre-registered in [`docs/EXPERIMENT_R2_CONFIRMATION_PLAN.md`](../../docs/EXPERIMENT_R2_CONFIRMATION_PLAN.md), committed before any confirmation run existed. Every judgement phrase it contains has an exact definition fixed there; none was chosen after seeing results.

Paired delta = **R2 run minus same-seed same-contract default control**.

**Stated limitation.** `NOISE_BAND.md` was measured from CONTRACT-change deltas across seeds, not from HYPERPARAMETER-change deltas. It is the best available yardstick for this pass, not an exact one. Do not treat it as precise.

No statistical significance is claimed from five seeds. Language here is deliberately limited to stable direction across seeds, mixed direction, inside observed seed variability, and consistent but small.

## Findings

- **M1 baseline_41: CONFIRMED** — clause 3.2.1 and clause 3.2.2 both satisfied.
- **M1 concurrent_43: NOT CONFIRMED** — clause 3.2.2 breached (M1 VALID guardrail).
- **Mechanism:** the arm-dependent split does NOT repeat — `baseline_41` classified `generalization_gain` in 2/5 seeds, `concurrent_43` classified `train_collapse` in 4/5 seeds.
- **M2 impact: HARMED_WITHIN_NOISE** — VALID MAE worsened in >=4 of 5 seeds in both arms; every five-seed mean is inside the band.

The two arms are reported separately and are never merged into one verdict.

## How to read these findings

**`baseline_41` CONFIRMED, but the seed-42 MECHANISM did not hold.** The gap shrank in 5/5 seeds and survives leave-one-seed-out, and no M1 VALID guardrail is breached — that is what the pre-registered rule asks, and it is met. But the reason the gap closed is not the one seed 42 suggested: `generalization_gain` in only 2/5 seeds, with 2/5 `train_collapse` and 1/5 `mixed`. Seed 42 was not representative of how this arm behaves. CONFIRMED here means the clauses were met, not that R2 buys a clean generalization gain.

**`concurrent_43` NOT CONFIRMED on the guardrails, not on the gap.** Its gap improved in 5/5 seeds — the largest and most stable gap reduction of either arm. It fails because VALID quality paid for it: VALID AUC mean -0.001041 and VALID Brier mean +0.000174 are both outside the band on the harmful side, each with two seeds beyond twice the harmful edge. This is precisely the failure mode the guardrail clause exists to catch, and the seed-42 Brier margin flagged at screening (0.000048 inside the edge) turned out to be the early warning.

**A shrinking gap was never the goal in itself.** Both arms shrink the gap in 5/5 seeds; they differ in what it cost. Read the mechanism table and the guardrail table together, not the gap column alone.

## Run environment and state

- Git commit at report time: `235a1db21be1ff4d43e2eab3153b624cc5b2a317`; working tree clean: False.
- Test suite: `python -m unittest discover -s tests -t .` — 117 tests, OK.
- Memory: 15.7 GB physical, 2.63 GB available at report time (83% load); commit charge 5.43 GB available of 27.7 GB.
- Pagefile: `C:\pagefile.sys` — active: True.
- One LightGBM training at a time throughout; `--num-threads 4` on every run, control and confirmation alike.

## All eighteen run paths

### Eight NEW confirmation runs (`--num-leaves 31`)

| Seed | Arm | Run path |
|---:|---|---|
| 52 | baseline_41 | `models/runs/2026-07-27_1600__seed52-regr2-leaves31-baseline-41` |
| 52 | concurrent_43 | `models/runs/2026-07-27_1601__seed52-regr2-leaves31-concurrent-43` |
| 62 | baseline_41 | `models/runs/2026-07-27_1817__seed62-regr2-leaves31-baseline-41` |
| 62 | concurrent_43 | `models/runs/2026-07-27_1819__seed62-regr2-leaves31-concurrent-43` |
| 72 | baseline_41 | `models/runs/2026-07-27_1820__seed72-regr2-leaves31-baseline-41` |
| 72 | concurrent_43 | `models/runs/2026-07-27_1822__seed72-regr2-leaves31-concurrent-43` |
| 82 | baseline_41 | `models/runs/2026-07-27_1823__seed82-regr2-leaves31-baseline-41` |
| 82 | concurrent_43 | `models/runs/2026-07-27_1825__seed82-regr2-leaves31-concurrent-43` |

### Two REUSED seed-42 R2 runs (from screening, not retrained)

| Seed | Arm | Run path |
|---:|---|---|
| 42 | baseline_41 | `models/runs/2026-07-27_1456__reg-r2-leaves31-baseline-41` |
| 42 | concurrent_43 | `models/runs/2026-07-27_1457__reg-r2-leaves31-concurrent-43` |

Verified before reuse to match this protocol exactly: single lever `num_leaves`=31, identical TRAIN/VALID SHA-256, threshold 0.80, seed and derived seed fields, four threads, 2000-round cap with 50-round VALID-only early stopping, train-only diploma-GPA median fill, and identical contract definitions and categorical levels. No material difference, so no rerun.

### Ten DEFAULT-parameter controls (reused unchanged, never retrained)

| Seed | Arm | Control path |
|---:|---|---|
| 42 | baseline_41 | `models/runs/2026-07-26_1551__baseline-41-gpa-trend-control` |
| 42 | concurrent_43 | `models/runs/2026-07-27_1327__seed42-concurrent-43-drop-dead-missing-flag` |
| 52 | baseline_41 | `models/runs/2026-07-27_1027__seed52-baseline-41-gpa-trend-control` |
| 52 | concurrent_43 | `models/runs/2026-07-27_1328__seed52-concurrent-43-drop-dead-missing-flag` |
| 62 | baseline_41 | `models/runs/2026-07-27_1031__seed62-baseline-41-gpa-trend-control` |
| 62 | concurrent_43 | `models/runs/2026-07-27_1329__seed62-concurrent-43-drop-dead-missing-flag` |
| 72 | baseline_41 | `models/runs/2026-07-27_1035__seed72-baseline-41-gpa-trend-control` |
| 72 | concurrent_43 | `models/runs/2026-07-27_1330__seed72-concurrent-43-drop-dead-missing-flag` |
| 82 | baseline_41 | `models/runs/2026-07-27_1038__seed82-baseline-41-gpa-trend-control` |
| 82 | concurrent_43 | `models/runs/2026-07-27_1331__seed82-concurrent-43-drop-dead-missing-flag` |

## Parity verification — one lever, nothing else

Each R2 run checked against its same-seed same-contract control: contract identity and ordered features, categorical levels, threshold 0.80, test policy, dataset version and TRAIN/VALID SHA-256, row counts, effective seeds read out of the serialized models, M1/M2 seed equality, the complete serialized LightGBM parameter block for BOTH models, the 2000-round cap, four threads, early stopping, and diploma-GPA fill.

| Seed | Arm | Checks | Result | Failed |
|---:|---|---:|:---:|---|
| 42 | baseline_41 | 22 | PASS | none |
| 42 | concurrent_43 | 22 | PASS | none |
| 52 | baseline_41 | 22 | PASS | none |
| 52 | concurrent_43 | 22 | PASS | none |
| 62 | baseline_41 | 22 | PASS | none |
| 62 | concurrent_43 | 22 | PASS | none |
| 72 | baseline_41 | 22 | PASS | none |
| 72 | concurrent_43 | 22 | PASS | none |
| 82 | baseline_41 | 22 | PASS | none |
| 82 | concurrent_43 | 22 | PASS | none |

In every pair the ONLY differing serialized LightGBM parameter is `num_leaves` (127 → 31), verified independently for M1 and M2.

## best_iteration and the 2000-round cap

| Seed | Arm | M1 control | M1 R2 | M1 shift | M2 control | M2 R2 | M2 shift | Hit cap |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| 42 | baseline_41 | 137 | 456 | +319 | 438 | 680 | +242 | no |
| 42 | concurrent_43 | 155 | 242 | +87 | 300 | 480 | +180 | no |
| 52 | baseline_41 | 227 | 235 | +8 | 308 | 680 | +372 | no |
| 52 | concurrent_43 | 199 | 353 | +154 | 360 | 648 | +288 | no |
| 62 | baseline_41 | 172 | 423 | +251 | 379 | 461 | +82 | no |
| 62 | concurrent_43 | 173 | 344 | +171 | 247 | 865 | +618 | no |
| 72 | baseline_41 | 151 | 250 | +99 | 576 | 716 | +140 | no |
| 72 | concurrent_43 | 128 | 444 | +316 | 443 | 426 | -17 | no |
| 82 | baseline_41 | 100 | 375 | +275 | 223 | 665 | +442 | no |
| 82 | concurrent_43 | 158 | 235 | +77 | 612 | 471 | -141 | no |

No confirmation run reached the 2000-round cap: early stopping fired in all ten R2 runs and all ten controls, so every comparison is between converged models. At 31 leaves this was a real risk and it did not materialise.

## baseline_41 — M1 verdict: CONFIRMED

clause 3.2.1 and clause 3.2.2 both satisfied.

### baseline_41 — exact per-seed metrics (unrounded)

| Seed | Arm/kind | TRAIN AUC | VALID AUC | VALID fail AP | VALID Brier | AUC gap | M1 iter |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | control | 0.8645724829519178 | 0.80918853274907998 | 0.3219831593412697 | 0.08077845438389275 | 0.055383950202837817 | 137 |
| 42 | r2 | 0.85899922096738801 | 0.81070810154286976 | 0.32429284670986108 | 0.080741315318703877 | 0.048291119424518247 | 456 |
| 52 | control | 0.8787079850192796 | 0.80997253661778768 | 0.32238769665892147 | 0.080684004794831787 | 0.068735448401491928 | 227 |
| 52 | r2 | 0.84497886995973093 | 0.80924455010994134 | 0.32323899865329253 | 0.080713205102490776 | 0.035734319849789586 | 235 |
| 62 | control | 0.87075819630425255 | 0.8091809346395975 | 0.32327512466144837 | 0.080739139118575012 | 0.061577261664655047 | 172 |
| 62 | r2 | 0.85692328284470876 | 0.80984737203541246 | 0.32257739376770694 | 0.080734739486564122 | 0.047075910809296295 | 423 |
| 72 | control | 0.86774611911737154 | 0.80954652606314181 | 0.32210829879484304 | 0.080742190735292763 | 0.058199593054229726 | 151 |
| 72 | r2 | 0.846015348653398 | 0.80896405352989831 | 0.32081079648103267 | 0.080922745393997964 | 0.037051295123499695 | 250 |
| 82 | control | 0.85612696942789923 | 0.80972709467034232 | 0.32383240525549156 | 0.080799918448565516 | 0.046399874757556914 | 100 |
| 82 | r2 | 0.85435659524332164 | 0.80957696983366889 | 0.32088917642504589 | 0.081027250638388448 | 0.044779625409652746 | 375 |

| Seed | Arm/kind | TRAIN MAE | VALID MAE | VALID RMSE | VALID R2 | M2 iter |
|---:|---|---:|---:|---:|---:|---:|
| 42 | control | 8.7566428180207385 | 9.5667097306309596 | 12.854908853104188 | 0.35190917004813094 | 438 |
| 42 | r2 | 9.3181664035385623 | 9.5981801001125682 | 12.881184622339692 | 0.34925703315109546 | 680 |
| 52 | control | 8.9432660180277139 | 9.5714505170826882 | 12.855157806946629 | 0.35188406737855193 | 308 |
| 52 | r2 | 9.3134787476566228 | 9.5850601173152263 | 12.868481203747649 | 0.35053992509893817 | 680 |
| 62 | control | 8.8442814451531664 | 9.5387162884294021 | 12.808063845401405 | 0.35662402297236351 | 379 |
| 62 | r2 | 9.492683473435271 | 9.5908319847766688 | 12.869649035247059 | 0.35042204104771491 | 461 |
| 72 | control | 8.5762227261173045 | 9.5491759528971389 | 12.835566407271592 | 0.35385803345502753 | 576 |
| 72 | r2 | 9.284806345637401 | 9.5761328649938129 | 12.869791461872522 | 0.35040766338934171 | 716 |
| 82 | control | 9.1388574685523363 | 9.6008191316346618 | 12.894956155591352 | 0.34786484458287481 | 223 |
| 82 | r2 | 9.3312135594448566 | 9.5771206366487966 | 12.872485345844076 | 0.35013569168961245 | 665 |

### baseline_41 — per-seed deltas (R2 minus control)

| Seed | m1_train_valid_auc_gap | m1_valid_auc | m1_valid_fail_ap | m1_valid_brier | m2_valid_mae | m2_valid_rmse | m2_valid_r2 | cold_start_auc | low_difficulty_support_auc | level_1_auc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | -0.007093 | +0.001520 | +0.002310 | -0.000037 | +0.031470 | +0.026276 | -0.002652 | +0.002902 | +0.001401 | +0.001466 |
| 52 | -0.033001 | -0.000728 | +0.000851 | +0.000029 | +0.013610 | +0.013323 | -0.001344 | +0.004592 | +0.002300 | -0.001461 |
| 62 | -0.014501 | +0.000666 | -0.000698 | -0.000004 | +0.052116 | +0.061585 | -0.006202 | +0.013487 | +0.007385 | -0.000510 |
| 72 | -0.021148 | -0.000582 | -0.001298 | +0.000181 | +0.026957 | +0.034225 | -0.003450 | +0.000888 | -0.002316 | -0.000964 |
| 82 | -0.001620 | -0.000150 | -0.002943 | +0.000227 | -0.023698 | -0.022471 | +0.002271 | -0.000038 | -0.006820 | +0.000851 |

### baseline_41 — five-seed summary vs the band

| Metric | Mean | Median | SD | Min | Max | Improved | Worsened | Band min | Band max | Mean judgment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| m1_train_valid_auc_gap | -0.015473 | -0.014501 | 0.012273 | -0.033001 | -0.001620 | 5 | 0 | -0.005873 | +0.026720 | outside_band_beneficial |
| m1_valid_auc | +0.000145 | -0.000150 | 0.000941 | -0.000728 | +0.001520 | 2 | 3 | -0.000382 | +0.001042 | inside_band |
| m1_valid_fail_ap | -0.000355 | -0.000698 | 0.002016 | -0.002943 | +0.002310 | 2 | 3 | -0.002045 | +0.001544 | inside_band |
| m1_valid_brier | +0.000079 | +0.000029 | 0.000118 | -0.000037 | +0.000227 | 2 | 3 | -0.000108 | +0.000119 | inside_band |
| m2_valid_mae | +0.020091 | +0.026957 | 0.028115 | -0.023698 | +0.052116 | 1 | 4 | -0.050423 | +0.046520 | inside_band |
| m2_valid_rmse | +0.022588 | +0.026276 | 0.030765 | -0.022471 | +0.061585 | 1 | 4 | -0.067477 | +0.078050 | inside_band |
| m2_valid_r2 | -0.002276 | -0.002652 | 0.003101 | -0.006202 | +0.002271 | 1 | 4 | -0.007865 | +0.006807 | inside_band |
| cold_start_auc | +0.004366 | +0.002902 | 0.005405 | -0.000038 | +0.013487 | 4 | 1 | -0.011618 | +0.008190 | inside_band |
| low_difficulty_support_auc | +0.000390 | +0.001401 | 0.005313 | -0.006820 | +0.007385 | 3 | 2 | -0.006657 | +0.008522 | inside_band |
| level_1_auc | -0.000123 | -0.000510 | 0.001237 | -0.001461 | +0.001466 | 2 | 3 | -0.000538 | +0.001140 | inside_band |

Per-seed band judgments, for the metrics the rule scores:

| Metric | seed 42 | seed 52 | seed 62 | seed 72 | seed 82 |
|---|---|---|---|---|---|
| m1_train_valid_auc_gap | outside_band_beneficial | outside_band_beneficial | outside_band_beneficial | outside_band_beneficial | inside_band |
| m1_valid_auc | outside_band_beneficial | outside_band_harmful | inside_band | outside_band_harmful | inside_band |
| m1_valid_fail_ap | outside_band_beneficial | inside_band | inside_band | inside_band | outside_band_harmful |
| m1_valid_brier | inside_band | inside_band | inside_band | outside_band_harmful | outside_band_harmful |
| m2_valid_mae | inside_band | inside_band | outside_band_harmful | inside_band | inside_band |
| m2_valid_rmse | inside_band | inside_band | inside_band | inside_band | inside_band |
| m2_valid_r2 | inside_band | inside_band | inside_band | inside_band | inside_band |

### baseline_41 — clause 3.2.1 (gap improvement, not carried by one seed)

- Gap improved (shrank) in **5 of 5** seeds — requirement ≥ 4: True.
- Leave-one-seed-out means (each recomputed with one seed dropped): -0.017568, -0.011091, -0.015716, -0.014054, -0.018936.
- All five leave-one-out means still improving: True.
- **Clause 3.2.1 satisfied: True**

### baseline_41 — clause 3.2.2 (M1 VALID guardrails)

| Metric | Mean delta | Mean judgment | Harmful edge | Large-outlier edge (2x) | Large harmful outliers |
|---|---:|:---|---:|---:|---|
| m1_valid_auc | +0.000145 | inside_band | -0.000382 | -0.000764 | none |
| m1_valid_fail_ap | -0.000355 | inside_band | -0.002045 | -0.004090 | none |
| m1_valid_brier | +0.000079 | inside_band | +0.000119 | +0.000238 | none |

- **Clause 3.2.2 satisfied: True**

## concurrent_43 — M1 verdict: NOT CONFIRMED

clause 3.2.2 breached (M1 VALID guardrail).

### concurrent_43 — exact per-seed metrics (unrounded)

| Seed | Arm/kind | TRAIN AUC | VALID AUC | VALID fail AP | VALID Brier | AUC gap | M1 iter |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | control | 0.86904731661822021 | 0.80998671304389847 | 0.32328432378749011 | 0.080659720768416765 | 0.059060603574321746 | 155 |
| 42 | r2 | 0.84614278883501193 | 0.80968782257808103 | 0.32295586340348192 | 0.080730847199650874 | 0.036454966256930899 | 242 |
| 52 | control | 0.87598416628910791 | 0.81006061782365579 | 0.32168750555421266 | 0.080674761991739216 | 0.06592354846545212 | 199 |
| 52 | r2 | 0.85392862246991319 | 0.8095586162174323 | 0.32265129423943162 | 0.080739959439325354 | 0.044370006252480887 | 353 |
| 62 | control | 0.87207479189936721 | 0.8111789396158946 | 0.32504066432223122 | 0.0804833010533101 | 0.060895852283472607 | 173 |
| 62 | r2 | 0.85281097190717359 | 0.80916477150920474 | 0.32158096931082158 | 0.080746621387272663 | 0.043646200397968848 | 344 |
| 72 | control | 0.86343574574589343 | 0.81014039907952684 | 0.32415001218075062 | 0.080611800014190882 | 0.053295346666366594 | 128 |
| 72 | r2 | 0.85799285436541317 | 0.80953951849979244 | 0.32012147818552883 | 0.080924847359530178 | 0.048453335865620728 | 444 |
| 82 | control | 0.86961268874233943 | 0.81084677921028914 | 0.32314535918433407 | 0.080648458480657381 | 0.058765909532050298 | 158 |
| 82 | r2 | 0.84535076316333346 | 0.80905906043247322 | 0.32121544044148082 | 0.080804255344554585 | 0.036291702730860242 | 235 |

| Seed | Arm/kind | TRAIN MAE | VALID MAE | VALID RMSE | VALID R2 | M2 iter |
|---:|---|---:|---:|---:|---:|---:|
| 42 | control | 8.9681354029364009 | 9.5783755788310003 | 12.862119935222113 | 0.35118186080742175 | 300 |
| 42 | r2 | 9.4636258427017097 | 9.607860119520689 | 12.898573357436732 | 0.34749892859049125 | 480 |
| 52 | control | 8.865228283360457 | 9.566677569057136 | 12.853481140005309 | 0.35205312072018002 | 360 |
| 52 | r2 | 9.3204024217040615 | 9.5742784541919708 | 12.860253425400032 | 0.35137015598202315 | 648 |
| 62 | control | 9.0696004471459926 | 9.5606993674037426 | 12.834704632815408 | 0.3539447939347059 | 247 |
| 62 | r2 | 9.1999065997174032 | 9.5848765312701172 | 12.877966241670295 | 0.34958217047503626 | 865 |
| 72 | control | 8.7500550033601865 | 9.5555115107053119 | 12.849506951860375 | 0.35245373821582437 | 443 |
| 72 | r2 | 9.5154074591383306 | 9.5826248740539057 | 12.866012431926107 | 0.35078909435310057 | 426 |
| 82 | control | 8.5402298214549415 | 9.5539611231306498 | 12.837399896924117 | 0.3536734246601112 | 612 |
| 82 | r2 | 9.4740824458332451 | 9.5991601403975206 | 12.885663431714812 | 0.3488044256489482 | 471 |

### concurrent_43 — per-seed deltas (R2 minus control)

| Seed | m1_train_valid_auc_gap | m1_valid_auc | m1_valid_fail_ap | m1_valid_brier | m2_valid_mae | m2_valid_rmse | m2_valid_r2 | cold_start_auc | low_difficulty_support_auc | level_1_auc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | -0.022606 | -0.000299 | -0.000328 | +0.000071 | +0.029485 | +0.036453 | -0.003683 | +0.004596 | +0.001771 | -0.000705 |
| 52 | -0.021554 | -0.000502 | +0.000964 | +0.000065 | +0.007601 | +0.006772 | -0.000683 | -0.001055 | -0.001001 | -0.000522 |
| 62 | -0.017250 | -0.002014 | -0.003460 | +0.000263 | +0.024177 | +0.043262 | -0.004363 | +0.000299 | -0.004032 | -0.002225 |
| 72 | -0.004842 | -0.000601 | -0.004029 | +0.000313 | +0.027113 | +0.016505 | -0.001665 | +0.002674 | -0.001184 | -0.000574 |
| 82 | -0.022474 | -0.001788 | -0.001930 | +0.000156 | +0.045199 | +0.048264 | -0.004869 | +0.001638 | -0.002491 | -0.001688 |

### concurrent_43 — five-seed summary vs the band

| Metric | Mean | Median | SD | Min | Max | Improved | Worsened | Band min | Band max | Mean judgment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| m1_train_valid_auc_gap | -0.017745 | -0.021554 | 0.007537 | -0.022606 | -0.004842 | 5 | 0 | -0.005873 | +0.026720 | outside_band_beneficial |
| m1_valid_auc | -0.001041 | -0.000601 | 0.000797 | -0.002014 | -0.000299 | 0 | 5 | -0.000382 | +0.001042 | outside_band_harmful |
| m1_valid_fail_ap | -0.001757 | -0.001930 | 0.002094 | -0.004029 | +0.000964 | 1 | 4 | -0.002045 | +0.001544 | inside_band |
| m1_valid_brier | +0.000174 | +0.000156 | 0.000112 | +0.000065 | +0.000313 | 0 | 5 | -0.000108 | +0.000119 | outside_band_harmful |
| m2_valid_mae | +0.026715 | +0.027113 | 0.013428 | +0.007601 | +0.045199 | 0 | 5 | -0.050423 | +0.046520 | inside_band |
| m2_valid_rmse | +0.030251 | +0.036453 | 0.017835 | +0.006772 | +0.048264 | 0 | 5 | -0.067477 | +0.078050 | inside_band |
| m2_valid_r2 | -0.003052 | -0.003683 | 0.001800 | -0.004869 | -0.000683 | 0 | 5 | -0.007865 | +0.006807 | inside_band |
| cold_start_auc | +0.001630 | +0.001638 | 0.002172 | -0.001055 | +0.004596 | 4 | 1 | -0.011618 | +0.008190 | inside_band |
| low_difficulty_support_auc | -0.001387 | -0.001184 | 0.002143 | -0.004032 | +0.001771 | 1 | 4 | -0.006657 | +0.008522 | inside_band |
| level_1_auc | -0.001143 | -0.000705 | 0.000770 | -0.002225 | -0.000522 | 0 | 5 | -0.000538 | +0.001140 | outside_band_harmful |

Per-seed band judgments, for the metrics the rule scores:

| Metric | seed 42 | seed 52 | seed 62 | seed 72 | seed 82 |
|---|---|---|---|---|---|
| m1_train_valid_auc_gap | outside_band_beneficial | outside_band_beneficial | outside_band_beneficial | inside_band | outside_band_beneficial |
| m1_valid_auc | inside_band | outside_band_harmful | outside_band_harmful | outside_band_harmful | outside_band_harmful |
| m1_valid_fail_ap | inside_band | inside_band | outside_band_harmful | outside_band_harmful | inside_band |
| m1_valid_brier | inside_band | inside_band | outside_band_harmful | outside_band_harmful | outside_band_harmful |
| m2_valid_mae | inside_band | inside_band | inside_band | inside_band | inside_band |
| m2_valid_rmse | inside_band | inside_band | inside_band | inside_band | inside_band |
| m2_valid_r2 | inside_band | inside_band | inside_band | inside_band | inside_band |

### concurrent_43 — clause 3.2.1 (gap improvement, not carried by one seed)

- Gap improved (shrank) in **5 of 5** seeds — requirement ≥ 4: True.
- Leave-one-seed-out means (each recomputed with one seed dropped): -0.016530, -0.016793, -0.017869, -0.020971, -0.016563.
- All five leave-one-out means still improving: True.
- **Clause 3.2.1 satisfied: True**

### concurrent_43 — clause 3.2.2 (M1 VALID guardrails)

| Metric | Mean delta | Mean judgment | Harmful edge | Large-outlier edge (2x) | Large harmful outliers |
|---|---:|:---|---:|---:|---|
| m1_valid_auc | -0.001041 | outside_band_harmful | -0.000382 | -0.000764 | seed 62 (-0.002014), seed 82 (-0.001788) |
| m1_valid_fail_ap | -0.001757 | inside_band | -0.002045 | -0.004090 | none |
| m1_valid_brier | +0.000174 | outside_band_harmful | +0.000119 | +0.000238 | seed 62 (+0.000263), seed 72 (+0.000313) |

- **Clause 3.2.2 satisfied: False**

## Mechanism test (section 3.3)

Pre-registered thresholds: TRAIN "fell sharply" at a drop ≥ 0.01; VALID "flat" means not degraded beyond the band floor (-0.000382). Classification order and the explicit `mixed` category are fixed in the plan.

| Seed | Arm | TRAIN AUC delta | VALID AUC delta | TRAIN drop | drop/|VALID change| | Classification |
|---:|---|---:|---:|---:|---:|:---|
| 42 | baseline_41 | -0.005573 | +0.001520 | +0.005573 | 3.7 | `generalization_gain` |
| 42 | concurrent_43 | -0.022905 | -0.000299 | +0.022905 | 76.6 | `train_collapse` |
| 52 | baseline_41 | -0.033729 | -0.000728 | +0.033729 | 46.3 | `train_collapse` |
| 52 | concurrent_43 | -0.022056 | -0.000502 | +0.022056 | 43.9 | `train_collapse` |
| 62 | baseline_41 | -0.013835 | +0.000666 | +0.013835 | 20.8 | `mixed` |
| 62 | concurrent_43 | -0.019264 | -0.002014 | +0.019264 | 9.6 | `train_collapse` |
| 72 | baseline_41 | -0.021731 | -0.000582 | +0.021731 | 37.3 | `train_collapse` |
| 72 | concurrent_43 | -0.005443 | -0.000601 | +0.005443 | 9.1 | `mixed` |
| 82 | baseline_41 | -0.001770 | -0.000150 | +0.001770 | 11.8 | `generalization_gain` |
| 82 | concurrent_43 | -0.024262 | -0.001788 | +0.024262 | 13.6 | `train_collapse` |

- `baseline_41`: 2× `generalization_gain`, 1× `mixed`, 2× `train_collapse`.
- `concurrent_43`: 1× `mixed`, 4× `train_collapse`.

**Does the seed-42 arm-dependent split repeat across all five seeds?** NO. Rule: repeats only if baseline_41 is generalization_gain in >=4/5 AND concurrent_43 is train_collapse in >=4/5. Observed: `baseline_41` `generalization_gain` in 2/5, `concurrent_43` `train_collapse` in 4/5.

Per the plan, a `train_collapse` classification does NOT by itself fail clause 3.2 — the clauses stand as written. It is reported as a separate finding.

## M2 impact (section 3.4)

**Status: HARMED_WITHIN_NOISE** — VALID MAE worsened in >=4 of 5 seeds in both arms; every five-seed mean is inside the band.

| Seed | Arm | VALID MAE delta | VALID RMSE delta | VALID R2 delta | MAE judgment |
|---:|---|---:|---:|---:|:---|
| 42 | baseline_41 | +0.031470 | +0.026276 | -0.002652 | inside_band |
| 42 | concurrent_43 | +0.029485 | +0.036453 | -0.003683 | inside_band |
| 52 | baseline_41 | +0.013610 | +0.013323 | -0.001344 | inside_band |
| 52 | concurrent_43 | +0.007601 | +0.006772 | -0.000683 | inside_band |
| 62 | baseline_41 | +0.052116 | +0.061585 | -0.006202 | outside_band_harmful |
| 62 | concurrent_43 | +0.024177 | +0.043262 | -0.004363 | inside_band |
| 72 | baseline_41 | +0.026957 | +0.034225 | -0.003450 | inside_band |
| 72 | concurrent_43 | +0.027113 | +0.016505 | -0.001665 | inside_band |
| 82 | baseline_41 | -0.023698 | -0.022471 | +0.002271 | inside_band |
| 82 | concurrent_43 | +0.045199 | +0.048264 | -0.004869 | inside_band |

| Arm | MAE worsened seeds | MAE improved seeds | MAE mean | RMSE mean | R2 mean |
|---|---:|---:|---:|---:|---:|
| baseline_41 | 4 | 1 | +0.020091 | +0.022588 | -0.002276 |
| concurrent_43 | 5 | 0 | +0.026715 | +0.030251 | -0.003052 |

`_SHARED_PARAMS` is shared by M1 and M2, so R2 cannot move one without the other. Per-model parameters are **not** implemented and **not** recommended here; that decision belongs to the user. This section reports the evidence only.

## Watch items flagged at screening

Watch items 1 and 3 are pre-registered as **reported-only**: segment AUCs and best_iteration are not clauses of the confirmation rule, so they did not and must not change any verdict above. Inventing a clause after seeing results is what pre-registration prevents.

Watch item 2 is different and must not be misread as reported-only: VALID Brier **is** a clause-3.2.2 guardrail metric. What screening flagged was how narrow its margin was, and the margin is what is reported below — but the metric itself was scored by the pre-registered clause exactly as written, and in `concurrent_43` it breached.

### Watch item 1 — `level_1_difficulty` AUC (reported, not scored)

Out-of-band harmful in R2·concurrent_43 at seed 42 (−0.000705 against a −0.000538 floor). Across all five seeds:

| Seed | Arm | Control AUC | R2 AUC | Delta | Judgment |
|---:|---|---:|---:|---:|:---|
| 42 | baseline_41 | 0.820962 | 0.822428 | +0.001466 | outside_band_beneficial |
| 42 | concurrent_43 | 0.821214 | 0.820509 | -0.000705 | outside_band_harmful |
| 52 | baseline_41 | 0.821731 | 0.820270 | -0.001461 | outside_band_harmful |
| 52 | concurrent_43 | 0.821119 | 0.820597 | -0.000522 | inside_band |
| 62 | baseline_41 | 0.821100 | 0.820590 | -0.000510 | inside_band |
| 62 | concurrent_43 | 0.822170 | 0.819945 | -0.002225 | outside_band_harmful |
| 72 | baseline_41 | 0.820866 | 0.819902 | -0.000964 | outside_band_harmful |
| 72 | concurrent_43 | 0.821004 | 0.820430 | -0.000574 | outside_band_harmful |
| 82 | baseline_41 | 0.820437 | 0.821288 | +0.000851 | inside_band |
| 82 | concurrent_43 | 0.821821 | 0.820133 | -0.001688 | outside_band_harmful |

- `baseline_41`: mean -0.000123 (inside_band), outside-band-harmful in 2/5 seeds.
- `concurrent_43`: mean -0.001143 (outside_band_harmful), outside-band-harmful in 4/5 seeds.

### Watch item 2 — VALID Brier margin to its harmful edge (SCORED metric)

At seed 42 R2·concurrent_43 sat only 0.000048 inside the harmful edge (+0.000119). Margin = harmful edge minus delta; negative means it crossed. This metric is a clause-3.2.2 guardrail — the margin is reported here, and the metric was scored by the clause.

| Seed | Arm | Brier delta | Margin to harmful edge | Judgment |
|---:|---|---:|---:|:---|
| 42 | baseline_41 | -0.000037 | +0.000156 | inside_band |
| 42 | concurrent_43 | +0.000071 | +0.000048 | inside_band |
| 52 | baseline_41 | +0.000029 | +0.000090 | inside_band |
| 52 | concurrent_43 | +0.000065 | +0.000054 | inside_band |
| 62 | baseline_41 | -0.000004 | +0.000123 | inside_band |
| 62 | concurrent_43 | +0.000263 | -0.000144 | outside_band_harmful |
| 72 | baseline_41 | +0.000181 | -0.000062 | outside_band_harmful |
| 72 | concurrent_43 | +0.000313 | -0.000194 | outside_band_harmful |
| 82 | baseline_41 | +0.000227 | -0.000108 | outside_band_harmful |
| 82 | concurrent_43 | +0.000156 | -0.000037 | outside_band_harmful |

### Watch item 3 — best_iteration and the round cap (reported, not scored)

Covered in the best_iteration table above. No run reached the 2000-round cap.

## Segment stability (VALID)

`first_semester` and `cold_start_gpa` are the SAME population (n=14,732, open defect) — ONE piece of evidence, not two.

| Seed | Arm | Segment | n | Control AUC | R2 AUC | Delta |
|---:|---|---|---:|---:|---:|---:|
| 42 | baseline_41 | first_semester | 14732 | 0.732931 | 0.735834 | +0.002902 |
| 42 | baseline_41 | cold_start_gpa | 14732 | 0.732931 | 0.735834 | +0.002902 |
| 42 | baseline_41 | retake_attempt | 17958 | 0.676827 | 0.679111 | +0.002284 |
| 42 | baseline_41 | low_difficulty_support | 25627 | 0.764405 | 0.765806 | +0.001401 |
| 42 | baseline_41 | level_1_difficulty | 120858 | 0.820962 | 0.822428 | +0.001466 |
| 42 | concurrent_43 | first_semester | 14732 | 0.732509 | 0.737105 | +0.004596 |
| 42 | concurrent_43 | cold_start_gpa | 14732 | 0.732509 | 0.737105 | +0.004596 |
| 42 | concurrent_43 | retake_attempt | 17958 | 0.678540 | 0.676522 | -0.002017 |
| 42 | concurrent_43 | low_difficulty_support | 25627 | 0.767322 | 0.769094 | +0.001771 |
| 42 | concurrent_43 | level_1_difficulty | 120858 | 0.821214 | 0.820509 | -0.000705 |
| 52 | baseline_41 | first_semester | 14732 | 0.735737 | 0.740329 | +0.004592 |
| 52 | baseline_41 | cold_start_gpa | 14732 | 0.735737 | 0.740329 | +0.004592 |
| 52 | baseline_41 | retake_attempt | 17958 | 0.673375 | 0.674622 | +0.001247 |
| 52 | baseline_41 | low_difficulty_support | 25627 | 0.765295 | 0.767596 | +0.002300 |
| 52 | baseline_41 | level_1_difficulty | 120858 | 0.821731 | 0.820270 | -0.001461 |
| 52 | concurrent_43 | first_semester | 14732 | 0.734561 | 0.733506 | -0.001055 |
| 52 | concurrent_43 | cold_start_gpa | 14732 | 0.734561 | 0.733506 | -0.001055 |
| 52 | concurrent_43 | retake_attempt | 17958 | 0.676898 | 0.676182 | -0.000716 |
| 52 | concurrent_43 | low_difficulty_support | 25627 | 0.767508 | 0.766507 | -0.001001 |
| 52 | concurrent_43 | level_1_difficulty | 120858 | 0.821119 | 0.820597 | -0.000522 |
| 62 | baseline_41 | first_semester | 14732 | 0.728398 | 0.741884 | +0.013487 |
| 62 | baseline_41 | cold_start_gpa | 14732 | 0.728398 | 0.741884 | +0.013487 |
| 62 | baseline_41 | retake_attempt | 17958 | 0.676007 | 0.676698 | +0.000691 |
| 62 | baseline_41 | low_difficulty_support | 25627 | 0.762430 | 0.769815 | +0.007385 |
| 62 | baseline_41 | level_1_difficulty | 120858 | 0.821100 | 0.820590 | -0.000510 |
| 62 | concurrent_43 | first_semester | 14732 | 0.734328 | 0.734627 | +0.000299 |
| 62 | concurrent_43 | cold_start_gpa | 14732 | 0.734328 | 0.734627 | +0.000299 |
| 62 | concurrent_43 | retake_attempt | 17958 | 0.677329 | 0.674912 | -0.002418 |
| 62 | concurrent_43 | low_difficulty_support | 25627 | 0.771169 | 0.767138 | -0.004032 |
| 62 | concurrent_43 | level_1_difficulty | 120858 | 0.822170 | 0.819945 | -0.002225 |
| 72 | baseline_41 | first_semester | 14732 | 0.735388 | 0.736277 | +0.000888 |
| 72 | baseline_41 | cold_start_gpa | 14732 | 0.735388 | 0.736277 | +0.000888 |
| 72 | baseline_41 | retake_attempt | 17958 | 0.675334 | 0.677718 | +0.002384 |
| 72 | baseline_41 | low_difficulty_support | 25627 | 0.768869 | 0.766553 | -0.002316 |
| 72 | baseline_41 | level_1_difficulty | 120858 | 0.820866 | 0.819902 | -0.000964 |
| 72 | concurrent_43 | first_semester | 14732 | 0.735331 | 0.738004 | +0.002674 |
| 72 | concurrent_43 | cold_start_gpa | 14732 | 0.735331 | 0.738004 | +0.002674 |
| 72 | concurrent_43 | retake_attempt | 17958 | 0.678634 | 0.674840 | -0.003794 |
| 72 | concurrent_43 | low_difficulty_support | 25627 | 0.770007 | 0.768823 | -0.001184 |
| 72 | concurrent_43 | level_1_difficulty | 120858 | 0.821004 | 0.820430 | -0.000574 |
| 82 | baseline_41 | first_semester | 14732 | 0.735686 | 0.735648 | -0.000038 |
| 82 | baseline_41 | cold_start_gpa | 14732 | 0.735686 | 0.735648 | -0.000038 |
| 82 | baseline_41 | retake_attempt | 17958 | 0.680578 | 0.673903 | -0.006675 |
| 82 | baseline_41 | low_difficulty_support | 25627 | 0.770927 | 0.764107 | -0.006820 |
| 82 | baseline_41 | level_1_difficulty | 120858 | 0.820437 | 0.821288 | +0.000851 |
| 82 | concurrent_43 | first_semester | 14732 | 0.733142 | 0.734780 | +0.001638 |
| 82 | concurrent_43 | cold_start_gpa | 14732 | 0.733142 | 0.734780 | +0.001638 |
| 82 | concurrent_43 | retake_attempt | 17958 | 0.679142 | 0.674534 | -0.004607 |
| 82 | concurrent_43 | low_difficulty_support | 25627 | 0.769626 | 0.767135 | -0.002491 |
| 82 | concurrent_43 | level_1_difficulty | 120858 | 0.821821 | 0.820133 | -0.001688 |

## Integrity confirmations

- TEST is `closed_not_read` in all eight new runs; every M1/M2 `test` metric field is null in every run and control; each run passed a NONEXISTENT `--test` path (`data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_test_CLOSED_DO_NOT_READ.parquet`, exists=False), so completing at all proves TEST was never opened. `--evaluate-test` was never passed.
- TRAIN SHA-256 `8aaff32aeac5b375…`, VALID SHA-256 `228719fa492da84b…`, identical across all twenty runs (ten R2 + ten controls).
- Dataset version `2026-07-26_batched_fixes__registration_roster_concurrent`; TRAIN 450,465 rows, VALID 156,097 rows. No dataset was copied or moved.
- The ten controls were reused unchanged and the seed-42 R2 pair was reused unchanged; only the eight new confirmation runs were trained.
- `_SHARED_PARAMS` defaults are UNCHANGED: `num_leaves` is still 127, `min_child_samples` 50, `reg_lambda` 1.0. R2 was applied per run via `--num-leaves 31`, never by editing a default.
- `CURRENT_VERSION.txt` unchanged (`2026-07-21_gpa_trend_feature`). No promotion marker, live model artifact, inference wiring, or recommendation wiring was touched. M1 was not frozen. Per-model parameters were not implemented.
- Every metric above was recomputed by re-scoring the saved models against TRAIN/VALID. Only `best_iteration` is read from each run's `metrics.json`. `level_1_difficulty` is not stored in `metrics.json` and exists only because it is recomputed here.

