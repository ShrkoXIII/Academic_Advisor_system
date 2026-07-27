# Pre-registered plan — R2 (`num_leaves` 31) five-seed confirmation

**Status: PRE-REGISTERED. Committed before any confirmation run was trained.**
This file is the analysis rule for the R2 confirmation. It may not be revised
after results are seen. If it turns out to be a bad rule, the correct response
is to record that in `Decisions_Log.md` and design a new, separately
pre-registered experiment — not to edit this file.

Context: seed-42 screening
([`models/runs/REGULARIZATION_SCREENING_SEED42_REPORT.md`](../models/runs/REGULARIZATION_SCREENING_SEED42_REPORT.md))
passed R2 (`num_leaves` 127→31) under the rule pre-registered in
[`EXPERIMENT_REGULARIZATION_PLAN.md`](EXPERIMENT_REGULARIZATION_PLAN.md), and
failed R1, R3, R4. This pass asks whether the R2 result repeats across seeds
52/62/72/82, with separate findings for M1, for M2, and for the MECHANISM.

Sections 3.1–3.4 below are the task specification, reproduced verbatim.

---

## 3.1 Per-seed deltas

delta = R2 run minus its same-seed same-contract default control.
Computed separately for baseline_41 and concurrent_43.

Collect per seed and arm:

- M1: TRAIN AUC, VALID AUC, VALID fail AP, VALID Brier, train-valid AUC gap,
  best_iteration
- M2: TRAIN and VALID MAE, RMSE, R2, best_iteration
- the five existing segment AUCs (first_semester and cold_start_gpa are ONE
  piece of evidence, n=14,732)

Yardstick: `models/runs/NOISE_BAND.md`. Restate its limitation in the report —
the band was measured from contract-change deltas across seeds, not from
hyperparameter-change deltas; it is the best available yardstick, not an exact
one.

## 3.2 M1 confirmation rule

R2 is CONFIRMED for an arm only if, across the five seeds:

1. the train-valid AUC gap improves (shrinks) in at least 4 of 5 seeds, and the
   mean improvement is not produced by a single outlier seed; and
2. VALID AUC, VALID fail AP, and VALID Brier do not degrade beyond the band in
   the five-seed mean, and no single seed shows a large harmful outlier.

Otherwise NOT CONFIRMED, or INCONCLUSIVE if the direction is mixed.
Report the two arms separately. Do not merge them into one verdict.

## 3.3 Mechanism test — pre-registered, not a post-hoc observation

For every seed and arm, classify HOW the gap closed:

- `generalization_gain` — VALID AUC improved or was flat, TRAIN fell modestly
- `train_collapse` — the gap closed mainly because TRAIN AUC fell sharply while
  VALID AUC was flat or fell

Report, per arm, the TRAIN AUC delta and VALID AUC delta side by side, and the
ratio of TRAIN drop to VALID change, so the classification is auditable rather
than asserted.

Screening at seed 42 showed baseline_41 = generalization_gain
(TRAIN −0.0056, VALID +0.0015) and concurrent_43 = train_collapse
(TRAIN −0.0229, VALID −0.0003). State explicitly whether this arm-dependent
split repeats across all five seeds.

A `train_collapse` classification does NOT by itself fail clause 3.2 —
the clauses stand as written. It is reported as a separate finding.

## 3.4 M2 impact rule

`_SHARED_PARAMS` is shared, so R2 moves M2 too. Report separately:

- M2 VALID MAE, RMSE, R2 deltas per seed and arm
- whether M2 degrades in a consistent direction across seeds
- whether any M2 degradation exceeds the band

Return an M2 status: HARMED_CONSISTENTLY | HARMED_WITHIN_NOISE | UNAFFECTED |
IMPROVED.

Do NOT implement per-model parameters, and do not recommend a specific split.
That decision belongs to the user. Report the evidence only.

---

# Operational definitions — fixed now, before any result exists

The rules above contain judgement phrases ("single outlier seed", "large
harmful outlier", "fell sharply", "consistent direction"). Left undefined they
could be bent after the fact, which is exactly what pre-registration prevents.
Each is given an exact, auditable meaning here.

## Band values (transcribed from `models/runs/NOISE_BAND.md`)

`NOISE_BAND.md` stays the source of truth; if these disagree with it, it wins
and the disagreement is a defect to report.

| Metric | Band min | Band max | Improving direction | Harmful edge |
|---|---:|---:|:---:|---:|
| M1 train–valid AUC gap | -0.005873 | +0.026720 | negative | > +0.026720 |
| M1 VALID AUC | -0.000382 | +0.001042 | positive | < -0.000382 |
| M1 VALID fail-class AP | -0.002045 | +0.001544 | positive | < -0.002045 |
| M1 VALID Brier | -0.000108 | +0.000119 | negative | > +0.000119 |
| M2 VALID MAE | -0.050423 | +0.046520 | negative | > +0.046520 |
| M2 VALID RMSE | -0.067477 | +0.078050 | negative | > +0.078050 |
| M2 VALID R² | -0.007865 | +0.006807 | positive | < -0.007865 |
| Cold-start AUC (`first_semester`) | -0.011618 | +0.008190 | positive | < -0.011618 |
| Low-difficulty-support AUC | -0.006657 | +0.008522 | positive | < -0.006657 |
| Level-1-difficulty AUC | -0.000538 | +0.001140 | positive | < -0.000538 |

**Limitation, to be repeated in the report.** This band was measured from
CONTRACT-change deltas across seeds, not from HYPERPARAMETER-change deltas. It
is the best available yardstick for this pass, not an exact one. It must not be
treated as precise.

## Judgment vocabulary

Every metric is labelled with exactly one of `inside_band`,
`outside_band_beneficial`, or `outside_band_harmful`, defined as in the
screening plan: inside `[min, max]` is `inside_band`; outside it in the
improving direction for that metric is beneficial; outside it in the worsening
direction is harmful.

## Clause 3.2.1 — "not produced by a single outlier seed"

Operationalised as **leave-one-seed-out stability**: recompute the mean gap
delta five times, each time dropping one seed. The clause is satisfied only if
**all five** leave-one-out means still show improvement (mean gap delta < 0).
If dropping any single seed flips the mean to non-improving, the mean was
carried by that seed and the clause fails.

Both parts must hold: gap improves in ≥ 4 of 5 seeds **and** leave-one-out
stability holds.

## Clause 3.2.2 — "no single seed shows a large harmful outlier"

Two parts, both required, for each of VALID AUC, VALID fail AP, VALID Brier:

- the five-seed **mean** delta is not `outside_band_harmful`; and
- **no single seed** exceeds the harmful edge by more than **2×** its distance
  from zero. Concretely, a seed is a "large harmful outlier" if its delta is
  beyond twice the harmful edge: VALID AUC < -0.000764, fail AP < -0.004090,
  Brier > +0.000238.

A single seed landing just outside the band on the harmful side is therefore
NOT automatically disqualifying — that is ordinary seed variability. Landing
at twice the edge is.

## Verdict assignment (per arm, reported separately)

- **CONFIRMED** — clause 3.2.1 and clause 3.2.2 both satisfied.
- **NOT CONFIRMED** — the gap improved in ≤ 2 of 5 seeds, or a clause-3.2.2
  guardrail is breached (mean harmful, or a large harmful outlier).
- **INCONCLUSIVE** — anything else: direction mixed (gap improves in exactly 3
  of 5 seeds), or the gap improves in ≥ 4 of 5 but leave-one-out stability
  fails without any guardrail breach.

The two arms get two verdicts. They are never merged.

## Section 3.3 — mechanism thresholds

Per seed and arm, using M1 AUC deltas versus the same-seed control:

- `train_drop` = −(TRAIN AUC delta), positive when TRAIN fell.
- `ratio` = `train_drop` / |VALID AUC delta|, reported as `inf` when the VALID
  delta is exactly 0.

**"Fell sharply" is fixed at a TRAIN drop ≥ 0.010 AUC.** This threshold is
chosen to sit between the two mechanisms already observed at seed 42 and named
in the task text (0.0056 modest versus 0.0229 sharp). It is set now, before
seeds 52–82 exist, and is not revisited afterwards.

Classification, evaluated in this order:

1. `generalization_gain` — VALID AUC delta > 0 **and** `train_drop` < 0.010.
2. `train_collapse` — `train_drop` ≥ 0.010 **and** VALID AUC delta ≤ 0.
3. `generalization_gain` — VALID AUC delta ≥ -0.000382 (improved or flat,
   i.e. not degraded beyond the band floor) **and** `train_drop` < 0.010.
4. `mixed` — anything else. Reported as `mixed`, never silently folded into
   either category.

The arm-dependent split "repeats" only if `baseline_41` classifies as
`generalization_gain` in ≥ 4 of 5 seeds **and** `concurrent_43` classifies as
`train_collapse` in ≥ 4 of 5 seeds. Any other outcome is reported as the split
not repeating, with the per-seed table shown.

## Section 3.4 — M2 status assignment

Evaluated on the five-seed means and per-seed counts, both arms considered:

1. **HARMED_CONSISTENTLY** — any of M2 VALID MAE, RMSE, R² has a five-seed
   mean delta that is `outside_band_harmful` in either arm.
2. **HARMED_WITHIN_NOISE** — not the above, and VALID MAE worsened (delta > 0)
   in ≥ 4 of 5 seeds in **both** arms.
3. **IMPROVED** — not the above, and VALID MAE improved (delta < 0) in ≥ 4 of 5
   seeds in **both** arms.
4. **UNAFFECTED** — anything else (direction mixed across seeds or arms, all
   means inside the band).

Exactly one status is returned. Per-model parameter splitting is NOT
implemented and NOT recommended by this pass; the evidence is reported and the
decision belongs to the user.

## Watch items — reported, never scored

Pre-registered as reported-only. Inventing a clause after seeing results is
what pre-registration exists to prevent, so none of these may change any
verdict above:

- `level_1_difficulty` AUC, all five seeds, both arms. It was
  `outside_band_harmful` in R2·concurrent_43 at seed 42 (−0.000705 against a
  −0.000538 floor).
- R2·concurrent_43 VALID Brier margin to its harmful edge, all five seeds. At
  seed 42 it sat only 0.000048 inside.
- `best_iteration` for every run, with an explicit FLAG on any run reaching the
  2000-round cap — early stopping never fired there, so that run is truncated,
  not converged, and its comparison is not like-for-like. This is a real risk
  at 31 leaves, where seed 42 already needed ~3× the rounds of the default.

## Out of scope — named so it cannot drift in

- Implementing or recommending per-model (M1 vs M2) parameters.
- Changing `_SHARED_PARAMS` defaults.
- Freezing M1, promotion, `CURRENT_VERSION.txt`, or any inference/recommendation
  wiring.
- Opening TEST. TEST stays `closed_not_read`; every run passes a nonexistent
  `--test` path and `--evaluate-test` is forbidden.
- Any configuration other than R2, and any contract other than `baseline_41`
  and `concurrent_43`. `concurrent_44` is archived and must never be run.
- Claims of statistical significance from five seeds. Permitted language:
  stable direction across seeds, mixed direction, inside observed seed
  variability, consistent but small.
