# Pre-registered plan — regularization pass (M1 train–valid AUC gap)

**Status: PRE-REGISTERED. Committed before any regularization run was
trained.** This file is the acceptance rule for the pass. It may not be
revised after results are seen. If it turns out to be a bad rule, the
correct response is to record that in `Decisions_Log.md` and design a new,
separately pre-registered experiment — not to edit this file.

Sections B1–B6 below are the task specification, reproduced verbatim.

---

## B1. Purpose and scope

Goal: find whether stronger regularization narrows M1's train-valid AUC gap
without paying more VALID performance than the documented noise band allows.

Two arms, always both: `baseline_41` and `concurrent_43`.
`concurrent_44` is archived — never run it.

This task runs SCREENING ONLY: seed 42, four single-lever configurations, both
arms = 8 runs. It then STOPS. Confirmation across seeds 52/62/72/82 is a
separate approved task.

## B2. Controls already on disk — do NOT retrain them

Control = the existing default-parameter runs at seed 42:

```text
models/runs/2026-07-26_1551__baseline-41-gpa-trend-control
models/runs/<the seed42 concurrent_43 run created in the previous task>
```

Locate the second path from the concurrent_43 report and state it explicitly
before running anything.

## B3. Current parameters and the four configurations

Current shared parameters (verify against `_SHARED_PARAMS` in
src/model_training.py and report any mismatch before proceeding):

```text
learning_rate 0.05 · num_leaves 127 · min_child_samples 50
feature_fraction 0.8 · bagging_fraction 0.8 · bagging_freq 5
reg_alpha 0.1 · reg_lambda 1.0 · histogram_pool_size 256
force_col_wise true · num_threads 4 · seed 42
```

Four configurations, ONE lever each (single-lever keeps attribution clean;
bundles would make a win unattributable):

```text
R1: num_leaves        127 -> 63
R2: num_leaves        127 -> 31
R3: min_child_samples  50 -> 200
R4: reg_lambda        1.0 -> 10.0
```

Everything else stays exactly as above, including num_threads=4,
histogram_pool_size=256, the 2000-round cap, 50-round VALID-only early
stopping, threshold 0.80, categorical levels, diploma fill, and the dataset.

`_SHARED_PARAMS` is shared by M1 and M2, so every configuration changes BOTH
models. This is intended for screening. If a configuration helps M1 and hurts
M2, that is a finding to report — per-model parameters would be a new
architectural divergence and that decision is NOT yours to make here.

## B4. CLI change

Inspect whether the CLI can already set these levers per run. If not, make the
smallest change adding explicit typed flags:

```text
--num-leaves --min-child-samples --reg-lambda
```

Do not add a free-form params argument — explicit flags stay auditable.
Defaults must reproduce today's values exactly.

`metrics.json` must record the COMPLETE effective LightGBM parameter dict for
M1 and M2 separately, so any future comparison can verify equality
programmatically.

Add tests proving: each flag reaches the effective LightGBM parameters; omitting
them reproduces current defaults byte-for-byte; the seed derivation is
unchanged; TEST stays unread. Run the full suite; stop if not green.

## B5. Runs

Eight runs, seed 42, one at a time, never concurrent:

```text
reg_R1_leaves63_baseline_41      reg_R1_leaves63_concurrent_43
reg_R2_leaves31_baseline_41      reg_R2_leaves31_concurrent_43
reg_R3_minchild200_baseline_41   reg_R3_minchild200_concurrent_43
reg_R4_lambda10_baseline_41      reg_R4_lambda10_concurrent_43
```

Explicit --train and --valid paths from the immutable version
`2026-07-26_batched_fixes__registration_roster_concurrent`.
Nonexistent --test path. Never pass --evaluate-test.

After each run verify against its control that everything matches except the
single lever: dataset hashes, row counts, categorical levels, threshold, seed
and derived seeds, threads, early stopping, diploma fill, contract.

Report best_iteration for every run and FLAG any run that reaches the 2000-round
cap — early stopping never fired there and that run's comparison is truncated,
not converged.

## B6. Pre-registered acceptance rule — locked before results

Deltas are computed against the same-contract control at seed 42.
The yardstick is `models/runs/NOISE_BAND.md`.

Stated limitation, to be repeated in the report: the band was measured from
contract-change deltas across seeds, not from hyperparameter-change deltas. It
is the best available yardstick, not an exact one. Do not treat it as precise.

A configuration PASSES screening only if, in BOTH arms:

1. PRIMARY — the M1 train-valid AUC gap improves (shrinks) beyond the band's
   observed range, i.e. it is outside the band in the beneficial direction; and
2. GUARDRAIL M1 — VALID AUC, VALID fail-class AP, and VALID Brier do not
   degrade beyond the band; and
3. GUARDRAIL M2 — VALID MAE, RMSE, and R2 do not degrade beyond the band.

A gap that shrinks because TRAIN performance collapsed is not a pass — that is
what guardrail 2 exists to catch. Report TRAIN AUC alongside the gap so this is
visible.

Do not invent a new threshold after seeing results. Do not rank on
gap-improvement alone. If no configuration passes, report exactly that: the
current parameters stand and M1 freezes as-is. That is a legitimate result.

---

## Operative band values (transcribed from `models/runs/NOISE_BAND.md`)

Transcribed here so the rule is self-contained and auditable at the moment
of pre-registration. `NOISE_BAND.md` remains the source of truth; if these
ever disagree, `NOISE_BAND.md` wins and the disagreement is a defect to
report.

Sign convention (from `NOISE_BAND.md`): delta = candidate − control. For
Brier, AUC gap, MAE and RMSE, lower is better, so a **negative** delta
improves; for AUC, AP and R², a **positive** delta improves.

| Metric | Band min | Band max | Improving direction | Role in B6 |
|---|---:|---:|:---:|---|
| M1 train–valid AUC gap | -0.005873 | +0.026720 | negative | **PRIMARY** (clause 1) |
| M1 VALID AUC | -0.000382 | +0.001042 | positive | guardrail (clause 2) |
| M1 VALID fail-class AP | -0.002045 | +0.001544 | positive | guardrail (clause 2) |
| M1 VALID Brier | -0.000108 | +0.000119 | negative | guardrail (clause 2) |
| M2 VALID MAE | -0.050423 | +0.046520 | negative | guardrail (clause 3) |
| M2 VALID RMSE | -0.067477 | +0.078050 | negative | guardrail (clause 3) |
| M2 VALID R² | -0.007865 | +0.006807 | positive | guardrail (clause 3) |
| Cold-start AUC (`first_semester`) | -0.011618 | +0.008190 | positive | reported, not a clause |
| Low-difficulty-support AUC | -0.006657 | +0.008522 | positive | reported, not a clause |
| Level-1-difficulty AUC | -0.000538 | +0.001140 | positive | reported, not a clause |

**How each clause reads against these numbers, fixed now:**

- **Clause 1 (PRIMARY, must be satisfied in BOTH arms):** the M1 train–valid
  AUC gap delta must be **< -0.005873**. A delta inside `[-0.005873,
  +0.026720]` is inside the band and is **not** a pass, however much it is
  liked.
- **Clause 2 (GUARDRAIL M1):** a metric fails if its delta is outside the
  band in the **harmful** direction — VALID AUC < -0.000382, fail-class AP
  < -0.002045, or Brier > +0.000119. Inside-band movement in either
  direction does not fail a guardrail; out-of-band movement in the
  beneficial direction does not fail a guardrail either.
- **Clause 3 (GUARDRAIL M2):** a metric fails if MAE > +0.046520, RMSE >
  +0.078050, or R² < -0.007865.

The three segment AUCs are reported per B7 but are **not** clauses of the
pass/fail rule. `first_semester` and `cold_start_gpa` are the same VALID
population (n=14,732; open defect) and count as ONE piece of evidence.

## Judgment vocabulary, fixed now

Every metric in the screening report is labelled with exactly one of:

- `inside_band` — delta within `[band min, band max]`. Not evidence.
- `outside_band_beneficial` — delta outside the band, in the improving
  direction for that metric.
- `outside_band_harmful` — delta outside the band, in the worsening
  direction for that metric.

A configuration is `PASS` only when clause 1 is `outside_band_beneficial`
in **both** arms and no clause-2 or clause-3 metric is
`outside_band_harmful` in **either** arm. Otherwise `FAIL`, and the report
names the failing clause. "No configuration passes" is a legitimate,
pre-accepted outcome: the current parameters stand and M1 freezes as-is.

## Out of scope for this pass — named so it cannot drift in

- Seeds 52/62/72/82 (confirmation is a separate, separately approved task).
- Changing `_SHARED_PARAMS` defaults.
- Per-model (M1-vs-M2) parameter divergence.
- Freezing, promotion, `CURRENT_VERSION.txt`, or any inference/recommendation
  wiring.
- Opening TEST. TEST is `closed_not_read`; every run passes a nonexistent
  `--test` path and `--evaluate-test` is forbidden.
- Multi-lever ("bundle") configurations. Single-lever only, so any win is
  attributable.
