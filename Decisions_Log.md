# Decisions Log

This file is a new, git-tracked decision log for the Academic Advisor repo.
Earlier project decisions (D1–D7, the 2026-07-07 governance freeze, etc.)
were recorded in `obsidian_vault/Decisions_Log.md`, which lives outside
this git checkout and is gitignored. That history is not duplicated here;
this file starts a fresh, in-repo record going forward.

---

## 2026-07-27 — Multi-seed stability experiment: baseline_41 vs concurrent_44

**Scope.** Five paired seeds (42, 52, 62, 72, 82), each seed training both
`baseline_41` (41 features) and `concurrent_44` (44 features, adds
`concurrent_peer_difficulty_mean/max/missing`) on the same VALID-only
selection split. TEST stayed `closed_not_read` in all ten runs. Full
evidence: [`models/runs/MULTISEED_BASELINE41_VS_CONCURRENT44_REPORT.md`](models/runs/MULTISEED_BASELINE41_VS_CONCURRENT44_REPORT.md)
and its noise-band summary, [`models/runs/NOISE_BAND.md`](models/runs/NOISE_BAND.md).

**M1 (pass/fail classifier): stays on `baseline_41`. Status: INCONCLUSIVE.**
Mean VALID AUC delta +0.00048 (candidate better), but the train-valid AUC
gap worsened in 4/5 seeds (mean +0.0079) and fail-class AP was mixed
(improved in only 2/5 seeds). The AUC improvement is within the observed
seed-to-seed noise band and does not clear it on the metrics that matter
operationally (fail-class AP, generalization gap).

**M2 (grade regressor): concurrent features SUPPORTED in principle.**
Mean VALID MAE delta -0.0141 (SD 0.0372), direction stable in 4/5 seeds;
effect is small but consistent. Not wired into production and not
promoted by this task — see "Two-contract direction" below.

**RETRACTED: the cold-start justification from the earlier single-seed
experiment.** Cold-start AUC improved in only 3/5 seeds under the
five-seed run; the SD of paired deltas (0.0083) is the same order of
magnitude as the original single-seed effect (0.0082). Do not cite
cold-start improvement as evidence for the concurrent features again
without a wider or repeated-seed check.

**`concurrent_peer_difficulty_missing`: effectively dead.** Used by any
model in only 2 of 5 seeds (M1 seed 52 once; M2 seed 82 once), under
0.001% of total gain in both cases, zero splits everywhere else. Treated
as dead weight. Removal is deferred to the future `concurrent_43`
contract task — not done here (out of scope for this close-out).

**Residual assumption — seed-42 pair reused, not rerun.** The seed-42
baseline/candidate pair predates the `--seed` CLI flag and was reused from
the earlier single-seed experiment rather than rerun under the current
seeded code path. Their `feature_contract.json`/`metrics.json` do not
contain the `effective_seed_settings` field the other four seed pairs
have. Verification for this task read the four resolved LightGBM sub-seeds
directly out of the serialized `.lgbm` model files instead (see
`_effective_seed_settings`, `src/model_training.py`); they match the
report's table (`data_random_seed=175, feature_fraction_seed=30056,
bagging_seed=400, drop_seed=17869`) and are identical between the M1/M2
models and between the baseline/candidate arms. This is a verified
artifact-level equality, not a claim that the seed-42 pair was produced by
literally the same code path as seeds 52–82 — log it as an assumption
about equivalence, not a verified process equality.

**Seed-52 provenance repair.** The seed-52 baseline run's
`feature_contract.json` git block (`git.commit`, `git.branch`,
`git.working_tree_clean`, `git.dirty_paths`) was edited after the fact, at
2026-07-27T10:31 (its file mtime), roughly 3–4 minutes after the run's
`metrics.json` and both `.lgbm` models were written (10:27–10:28,
unchanged since). The repair made the baseline's git block byte-identical
to the candidate's git block from the same seed pair (commit
`5928aaa1485bf6fc9d930eb8f49e3146498c846d`, branch `main`,
`working_tree_clean: false`, same two dirty paths) — this current,
post-repair state is verified correct.

The **pre-repair values are not recoverable**: `models/` (including
`models/runs/`) is gitignored, so there is no git history for the file; no
backup copy exists on disk; and the run's stdout/stderr logs do not
capture the git-metadata step. The stated cause is that "the first
launcher failed to inject Git safe-directory state." Reading
`_git_state()` (`src/model_training.py:920-942`): it shells out to
`git rev-parse HEAD` / `git status --porcelain` with `check=False` and
silently converts any non-zero exit (e.g. a `dubious ownership` failure
from a missing `safe.directory` config) into `commit=None, branch=None,
working_tree_clean=None, dirty_paths=[]`. That is a plausible
reconstruction of what the pre-repair block likely looked like, consistent
with the stated failure mode — but it is an inference from reading the
code, not a recovered fact, and is recorded here as such. No metric,
model, or data artifact was touched by the repair; only this one JSON
file's `git` block changed.

**Two-contract direction (M1=41, M2=43) — intended direction only, not
wired.** Given the M1/M2 split above, running M1 on `baseline_41` and M2
on a future 43-feature contract (`concurrent_44` minus the dead
`concurrent_peer_difficulty_missing`) is a plausible direction. This is
**not implemented or decided** — no `concurrent_43` contract exists, no
promotion has happened, and no inference/recommendation wiring was
touched. The decision is deferred until after a regularization pass that
must run both arms (M1 and M2) under both contracts before any promotion
is considered.

**Integrity confirmations (verified against on-disk artifacts, not just
the report's prose):** all ten runs' `test_policy` = `closed_not_read`;
every M1/M2 `test` metric field is `null`; no run's JSON contains a
readable TEST path string; `df_test_final.parquet` for the dataset version
used (`2026-07-26_batched_fixes__registration_roster_concurrent`) has an
mtime (2026-07-26 12:34) that predates the entire experiment window
(2026-07-26 15:51 through 2026-07-27 10:39); train/valid SHA-256 hashes
are identical across all ten runs; row counts are 450465 train / 156097
valid in all ten. No dataset, live model artifact, `CURRENT_VERSION.txt`,
promotion marker, recommendation wiring, or inference wiring was changed
by this experiment or by this close-out task. No training was run as part
of this close-out; all numbers above were recomputed from existing
artifacts (including an independent reload of two trained models to
recompute the `level_1_difficulty` segment AUC, which is not stored in
`metrics.json`).

**Next action.** Keep M1 on `baseline_41`. Treat the M2 evidence for
`concurrent_44` as supported-but-small; no promotion or wiring change
follows from this task. Await explicit human review before acting on the
two-contract direction.

---

## 2026-07-27 — Define `concurrent_43`; re-base the position gate; five-seed validation against `concurrent_44`

**Scope.** Defined `concurrent_43` = `concurrent_44` minus
`concurrent_peer_difficulty_missing` (the feature the entry above calls
effectively dead), order of the remaining 43 features preserved exactly.
`baseline_41` and `concurrent_44` are untouched and remain selectable;
`DEFAULT_FEATURE_CONTRACT` stays `concurrent_44`. Trained `concurrent_43`
on the same immutable dataset version
(`data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent`)
at the same five seeds (42, 52, 62, 72, 82) already used for the
`baseline_41` vs `concurrent_44` experiment above; TEST stayed
`closed_not_read` in all five new runs (nonexistent `--test` path). The
five existing `concurrent_44` runs from that earlier experiment were reused
unchanged as the comparison arm — nothing was retrained except
`concurrent_43`. Full evidence:
[`models/runs/MULTISEED_CONCURRENT43_VS_CONCURRENT44_REPORT.md`](models/runs/MULTISEED_CONCURRENT43_VS_CONCURRENT44_REPORT.md),
verified pairwise against
[`models/runs/CONCURRENT43_VS_CONCURRENT44_VERIFICATION.json`](models/runs/CONCURRENT43_VS_CONCURRENT44_VERIFICATION.json).
Acceptance yardstick: [`models/runs/NOISE_BAND.md`](models/runs/NOISE_BAND.md)
(derived from the `baseline_41` vs `concurrent_44` paired deltas above).

**Position-gate re-base
(`scripts/build_concurrent_group_features.py:_assert_contract`).** The
dataset builder's `EXPECTED_LEGACY_MODEL_POSITION = 35` gate previously read
`MODEL_FEATURES` / `EXPECTED_FEATURE_COUNT`, deprecated globals in
`src/model_training.py` that are aliases for "whichever list a maintainer
last pointed them at" (in practice, always `CONCURRENT_44_FEATURES`, since
nothing had ever repointed them). Before: `legacy in MODEL_FEATURES and
MODEL_FEATURES.index(legacy) == 35`. After: the same check against the
explicit, named `CONCURRENT_44_FEATURES` list imported directly, so the gate
is pinned to `concurrent_44` specifically rather than to an ambiguous
global. Same gate values today (44 features, legacy indicator at index 35,
all seven gates pass); the dataset's column layout is unaffected — the
builder still writes all 8 concurrent columns (3 model + 5 audit) regardless
of which named model contract is selected for training. The re-base matters
going forward: now that `concurrent_43` is a real, selectable contract that
legitimately excludes this column, a gate still keyed off the deprecated
globals would risk breaking (or silently no-op) the moment anyone repointed
those globals — the re-based gate cannot break that way, because it no
longer reads them.

**Validation outcome — mixed by model, matching the M1/M2 split above.**

- **M2 (grade regressor): EQUIVALENT.** All three primary VALID deltas
  (MAE mean +0.0118, RMSE mean +0.0097, R2 mean -0.0010) fall inside the
  noise band, including the full per-seed range for RMSE and R2. `concurrent_43`
  and `concurrent_44` are indistinguishable for M2 on this evidence.
- **M1 (pass/fail classifier): INCONCLUSIVE.** VALID AUC, fail-class AP, and
  Brier all landed inside the noise band (mean deltas +0.00044, +0.00125,
  -0.00009 respectively — all improving or flat). The train-valid AUC gap
  improved on average (mean -0.0064) and in 3/5 seeds by more than the
  largest improvement recorded in the noise band (band min -0.005873); no
  seed worsened the gap beyond the band's harmful edge (+0.02672). This is
  an out-of-band result on the **beneficial** side, not evidence of harm —
  but per the band-is-the-bar rule, a metric outside the band in either
  direction blocks a clean EQUIVALENT verdict rather than being folded into
  one after the fact. Best-iteration also dropped materially for M1 (mean
  -42.4 rounds, -21% of the `concurrent_44` mean, in 3/5 seeds), directionally
  consistent with the smaller AUC gap; M2's best-iteration shift was noisy
  and not systematic (mixed direction, no band-relevant effect).

**No blanket "`concurrent_43` replaces `concurrent_44`" — split by model,
direction only, nothing wired.** M2's evidence supports treating
`concurrent_43` as the concurrent arm for M2 in future comparisons. M1's
evidence does not yet support that swap; `concurrent_44` remains the M1
concurrent arm pending more seeds or a repeated check specifically on the
train-valid AUC gap. This mirrors the M1=`baseline_41`/M2=candidate split
already recorded above, and is likewise **not implemented or wired** — no
promotion, no inference/recommendation change, no contract default change.
`baseline_41` and `concurrent_44` are untouched by this task; `concurrent_43`
is simply now available for the future regularization pass to select
explicitly.

**No production contract changed; nothing promoted.** `DEFAULT_FEATURE_CONTRACT`
is still `concurrent_44`. No dataset, live `MODEL_DATA_DIR`,
`CURRENT_VERSION.txt`, promotion marker, inference wiring, or recommendation
wiring was touched. TEST was not read by any of the five new runs.

**Next action (named, not started).** Regularization pass, two arms:
`baseline_41` vs `concurrent_43` — direction only; do not start it without
separate explicit approval.

---

## 2026-07-27 — CORRECTION to the `concurrent_43` entry above

This log is append-only: the entry above is left in place as written, and
this entry supersedes the part of it that is wrong. Specifically, the
sentence "`concurrent_44` remains the M1 concurrent arm pending more seeds
or a repeated check specifically on the train-valid AUC gap" is **wrong**
and is corrected here.

```text
Correction to the concurrent_43 entry (2026-07-27):
concurrent_44 has NO remaining role. M1's contract is baseline_41, decided by
the 5-seed baseline_41 vs concurrent_44 experiment (concurrent features were
rejected for M1 because the train-valid AUC gap worsened in 4/5 seeds).
concurrent_43 is the only concurrent arm going forward: M2's contract, and the
concurrent arm of the regularization pass. concurrent_44 is ARCHIVED — it must
not be used in any new run; it survives only inside past reports.
```

**What this changes in practice.**

- M1's contract is `baseline_41`. It is decided, not pending. The
  "INCONCLUSIVE" wording in the first entry above describes the strength
  of the evidence, not an open question about which contract M1 uses.
- `concurrent_43` is the sole concurrent arm from here on — M2's contract,
  and the concurrent arm of the regularization pass.
- `concurrent_44` is ARCHIVED. It must not appear in any new run. It
  survives only inside past reports
  (`MULTISEED_BASELINE41_VS_CONCURRENT44_REPORT.md`,
  `MULTISEED_CONCURRENT43_VS_CONCURRENT44_REPORT.md`,
  `CONCURRENT43_VS_CONCURRENT44_VERIFICATION.json`) and inside the
  archived run folders those reports cite. Those artifacts are not edited.

**Not changed by this correction.** `DEFAULT_FEATURE_CONTRACT` in
`src/model_training.py` is still `concurrent_44` — stale, and it affects
quick runs only, because a persistent run (`--run-name`) is test-enforced
to pass `--feature-contract` explicitly and is never silently defaulted.
Repointing the default is wiring, deferred until after the M1/M2 freeze.
`CONCURRENT_44_FEATURES` also stays in the code as the named list the
dataset builder's position gate is pinned to
(`scripts/build_concurrent_group_features.py:_assert_contract`); archived
means "not selectable for new runs", not "deleted". No dataset, live model
artifact, `CURRENT_VERSION.txt`, promotion marker, inference wiring, or
recommendation wiring is touched by this correction — it is a
documentation-only entry.

---

## 2026-07-27 — Regularization pass: criterion pre-registered before any run

**Scope.** Created [`docs/EXPERIMENT_REGULARIZATION_PLAN.md`](docs/EXPERIMENT_REGULARIZATION_PLAN.md),
the pre-registered plan and acceptance rule for the regularization pass
(`baseline_41` vs `concurrent_43`, both arms always). It is committed
**before any regularization run is trained**, so the success criterion is
locked before any result exists. It may not be revised after results are
seen.

**Why it is a separate committed file.** The acceptance rule is only
meaningful if it demonstrably predates the numbers it judges. Committing
it first makes that ordering verifiable from git history rather than
asserted in prose.

**Stated limitation, carried into the plan and every report that uses
it.** [`models/runs/NOISE_BAND.md`](models/runs/NOISE_BAND.md) was
measured from contract-change deltas across seeds, not from
hyperparameter-change deltas. It is the best available yardstick, not an
exact one for this pass, and must not be treated as precise.

**Next action.** Seed-42 screening only: four single-lever configurations
× two arms = 8 runs, then STOP. Confirmation across seeds 52/62/72/82 is a
separate task requiring separate explicit approval.

---

## 2026-07-27 — Regularization screening, seed 42: R2 (`num_leaves` 31) is the only candidate

**Scope.** Eight training runs at seed 42 — four single-lever
configurations × two arms (`baseline_41`, `concurrent_43`), one at a time,
never concurrent. `concurrent_44` was not run. The two seed-42
default-parameter controls were reused unchanged, not retrained:
`models/runs/2026-07-26_1551__baseline-41-gpa-trend-control` and
`models/runs/2026-07-27_1327__seed42-concurrent-43-drop-dead-missing-flag`.
TEST stayed `closed_not_read` in all eight runs (nonexistent `--test`
path; `--evaluate-test` never passed). Full evidence:
[`models/runs/REGULARIZATION_SCREENING_SEED42_REPORT.md`](models/runs/REGULARIZATION_SCREENING_SEED42_REPORT.md)
and its JSON. Acceptance rule: the pre-registered
[`docs/EXPERIMENT_REGULARIZATION_PLAN.md`](docs/EXPERIMENT_REGULARIZATION_PLAN.md),
committed before any of these runs existed.

**Outcome — one of four passes.**

| Config | Lever | Verdict | Deciding clause |
|---|---|---|---|
| R1 | `num_leaves` 127→63 | FAIL | clause 1: gap inside band in `baseline_41` (+0.0115) |
| R2 | `num_leaves` 127→31 | **PASS** | all three clauses satisfied in both arms |
| R3 | `min_child_samples` 50→200 | FAIL | clause 1: gap inside band in both arms |
| R4 | `reg_lambda` 1.0→10.0 | FAIL | clause 1 (gap inside band) **and** clause 2 (Brier +0.000122, out-of-band harmful in `baseline_41`) |

**R2 detail.** Gap delta −0.0071 (`baseline_41`) and −0.0226
(`concurrent_43`), both beyond the band's beneficial edge (−0.005873). The
two arms got there differently, and this matters: in `baseline_41` TRAIN
fell 0.0056 while VALID AUC *rose* 0.0015 — a genuine generalization gain.
In `concurrent_43` TRAIN fell 0.0229 while VALID fell 0.0003 — the gap
closed mostly by TRAIN coming down, with VALID holding. Guardrail 2 exists
to reject a gap that closed via a VALID collapse; there is no collapse
here, so R2 passes as written.

**Caveats recorded now, before any confirmation run — none of them changes
the verdict.**

- **One seed.** Screening selects what is worth five seeds; it confirms
  nothing. R2 is a candidate, not a result.
- **`level_1_auc` moved out of band on the harmful side** in R2 ·
  `concurrent_43` (−0.000705 against a band floor of −0.000538). Segment
  AUCs are explicitly NOT clauses of the pre-registered rule, so this did
  not and must not change the PASS. It is flagged for the confirmation
  task to watch.
- **Tightest guardrail margin is small.** R2 · `concurrent_43` VALID Brier
  is +0.000071 against a harmful edge of +0.000119 — inside the band, but
  close enough that another seed could land the other side.
- **M2 got worse under R2 in both arms** (VALID MAE +0.0315 / +0.0295,
  both inside the band). `_SHARED_PARAMS` is shared, so a configuration
  cannot move M1 without moving M2. Per the plan's B3 this is a finding to
  report, **not** a licence to split parameters per model — that
  architectural decision is not made here.
- **The band is not an exact yardstick for this pass.** `NOISE_BAND.md`
  was measured from contract-change deltas across seeds, not from
  hyperparameter-change deltas.

**Fitting behaviour.** No run reached the 2000-round cap, so every
comparison is between converged models, not truncated ones. R2's
best_iteration rose materially (M1 137→456 in `baseline_41`, 155→242 in
`concurrent_43`), which is the expected cost of smaller trees.

**Verification.** Every screening run was checked against its
same-contract control on 22 points — contract identity and ordered
features, categorical levels, threshold 0.80, test policy, dataset version
and TRAIN/VALID SHA-256, row counts, effective seeds read out of the
serialized models, M1/M2 seed equality, the complete serialized LightGBM
parameter block for both models, the 2000-round cap, four threads, early
stopping, and the diploma-GPA fill. All 22 passed in all eight runs: the
only serialized LightGBM parameter differing between a run and its control
is that run's single lever. Two checks are satisfied by inference rather
than JSON equality because the seed-42 `baseline_41` control predates the
`data_rows` / `training_control` / `diploma_gpa_handling` fields — the
provenance caveat already recorded above; both are labelled in the JSON.
All metrics were recomputed by re-scoring the saved models against
TRAIN/VALID rather than trusting stored values (only `best_iteration` is
read from `metrics.json`).

**CLI change that made this possible.** `src/model_training.py` gained
three explicit typed flags — `--num-leaves`, `--min-child-samples`,
`--reg-lambda` — with defaults read from `_SHARED_PARAMS`, so omitting
them reproduces previous behaviour exactly. No free-form params argument:
every deviation stays auditable. `metrics.json` now records the complete
effective LightGBM parameter dict for M1 and M2 separately, verified
against each trained booster's own parameters before being written.
Committed BEFORE the runs so every run recorded a clean tree.

**Nothing was changed, frozen, or promoted.** `_SHARED_PARAMS` defaults
are untouched — R2 is a screening candidate, not a new default. M1 is not
frozen. No `CURRENT_VERSION.txt`, promotion marker, live model artifact,
inference wiring, or recommendation wiring was touched. TEST was not read.

**Next action (named, NOT started).** Five-seed confirmation of R2 across
seeds 52/62/72/82 in both arms, watching `level_1_auc` and VALID Brier in
`concurrent_43` specifically. Requires separate explicit approval — do not
start it, and do not treat this screening PASS as a decision.

---

## 2026-07-27 — R2 five-seed confirmation: CONFIRMED for `baseline_41`, NOT CONFIRMED for `concurrent_43`

**Scope.** Eight new runs (seeds 52, 62, 72, 82 × two arms), each
`--num-leaves 31` with everything else at defaults, compared against its
same-seed same-contract DEFAULT-parameter control. The seed-42 R2 pair from
screening was reused after verifying it matches this protocol exactly; the
ten controls were reused unchanged. Nothing was retrained. `concurrent_44`
was not run. TEST stayed `closed_not_read` throughout. Full evidence:
[`models/runs/R2_CONFIRMATION_5SEED_REPORT.md`](models/runs/R2_CONFIRMATION_5SEED_REPORT.md)
and its JSON. Rule: the pre-registered
[`docs/EXPERIMENT_R2_CONFIRMATION_PLAN.md`](docs/EXPERIMENT_R2_CONFIRMATION_PLAN.md),
committed before any confirmation run existed.

**M1 `baseline_41`: CONFIRMED.** Gap improved in 5/5 seeds (mean −0.01547),
stable under leave-one-seed-out (all five LOO means between −0.0111 and
−0.0189). No M1 VALID guardrail breached: VALID AUC +0.000145, fail AP
−0.000355, Brier +0.000079 — all inside the band, no seed beyond twice a
harmful edge.

**M1 `concurrent_43`: NOT CONFIRMED.** It fails on the guardrails, not on
the gap. Its gap improved in 5/5 seeds and by more than either arm (mean
−0.01775), but VALID quality paid for it: VALID AUC mean −0.001041 and
VALID Brier mean +0.000174 are both **outside the band on the harmful
side**, each with two seeds beyond twice the harmful edge (AUC: seeds 62
−0.002014, 82 −0.001788; Brier: seeds 62 +0.000263, 72 +0.000313). This is
exactly the failure mode clause 3.2.2 exists to catch.

**The seed-42 MECHANISM story did NOT hold — this is the most important
finding.** Screening reported `baseline_41` = `generalization_gain` and
`concurrent_43` = `train_collapse`. Across five seeds the split does **not**
repeat: `baseline_41` classifies `generalization_gain` in only **2/5** seeds
(42, 82), with 2/5 `train_collapse` (52, 72) and 1/5 `mixed` (62).
`concurrent_43` is `train_collapse` in 4/5. Seed 42 was not representative
of `baseline_41`. **CONFIRMED for `baseline_41` therefore means the
pre-registered clauses were met — not that R2 buys a clean generalization
gain.** A shrinking gap was never the goal in itself; both arms shrink it in
5/5 seeds and differ only in what it cost.

**M2 impact: HARMED_WITHIN_NOISE.** VALID MAE worsened in 4/5 seeds
(`baseline_41`, mean +0.0201) and 5/5 seeds (`concurrent_43`, mean +0.0267);
RMSE and R² move the same way. Every M2 five-seed mean is inside the band,
so the degradation is consistent in direction but small — inside observed
seed variability. `_SHARED_PARAMS` is shared by M1 and M2, so R2 cannot move
one without the other. **Per-model parameters were NOT implemented and are
NOT recommended here** — that decision belongs to the user; this pass
reports evidence only.

**Watch items.**

- `level_1_difficulty` AUC (reported, not scored — segments are not clauses):
  the seed-42 flag was not a fluke and got worse. `concurrent_43` mean
  −0.001143, outside the band harmful, harmful in **4/5** seeds.
  `baseline_41` mean −0.000123 (inside band), harmful in 2/5.
- VALID Brier margin: this metric **is** a clause-3.2.2 guardrail, so unlike
  the other watch items it was scored. The seed-42 margin of 0.000048 was an
  early warning — in `concurrent_43` it crossed the harmful edge in 3/5
  seeds (62, 72, 82).
- Round cap: **no run reached the 2000-round cap** (max best_iteration
  observed M1 456, M2 865), so every comparison is between converged models,
  not truncated ones. At 31 leaves this was a real risk and did not
  materialise.

**Verification.** All ten R2/control pairs passed 22 parity checks each, run
inline after every training and again in the report from one shared
implementation (`scripts/r2_parity.py`). In every pair the only differing
serialized LightGBM parameter is `num_leaves` (127→31), verified
independently for M1 and M2. All metrics were recomputed by re-scoring the
saved models against TRAIN/VALID; only `best_iteration` is read from
`metrics.json`.

**No statistical significance is claimed from five seeds.**

**Nothing changed, frozen, or promoted.** `_SHARED_PARAMS` defaults are
untouched (`num_leaves` still 127) — R2 was applied per run via
`--num-leaves 31`, never by editing a default. M1 is not frozen. No
`CURRENT_VERSION.txt`, promotion marker, live model artifact, inference
wiring, or recommendation wiring was touched. TEST was not read.

**Next action — a decision for the user, not for a session.** The evidence
now supports several distinct readings and the project rule is that the
accept/reject call is made by the human. The open question: whether a
gap reduction that is only sometimes a generalization gain, and that costs
M2 a small consistent amount, is worth adopting for `baseline_41` — and
separately, whether `concurrent_43`'s guardrail failure ends the "unify both
models on one contract" direction. Do not start further runs without
explicit approval.

## 2026-07-28 — Difficulty-coverage decay diagnostic (read-only; no decision)

**Scope.** Completed the requested TRAIN/VALID-only diagnostic against dataset
`2026-07-26_batched_fixes__registration_roster_concurrent`. Reused the frozen
seed-42 `baseline_41` control M1 and frozen seed-42 `concurrent_43` M2. Neither
run was retrained or re-tuned. The project TEST split was never read; its
recorded 44.7% Level-1 coverage remains context only. No dataset, `src/` file,
default, `CURRENT_VERSION.txt`, promotion marker, or inference/recommendation
wiring changed.

**Definitions and discrepancy.** The inherited coverage figures are Level-1
membership (`difficulty_fallback_level == 1`), not the model-facing
confidence flag. On the specified parquets, Level-1 coverage recomputes to
94.37% TRAIN and 77.42% VALID, versus inherited 93.6% and 76.2%. For product
and accuracy splits the diagnostic uses `course_difficulty_missing == 0`,
which is the current model-facing definition: a Level-1/Level-2 known course
with at least 20 historical rows. That coverage is 89.61% TRAIN and 82.78%
VALID.

**Main evidence.** VALID has 26,882 uncovered rows: 25,627 (95.33%) are
`never_in_train`, 1,255 (4.67%) are `thin_history`, and 0 are `other`. At the
exact plan grain (`university_id`, `student_id`, `degree_id`, `part_id`),
8,076 of 34,293 student-semesters (23.55%) contain at least one uncovered
course; 73.45% of affected student-semesters have a majority uncovered.
Coverage drops 9.28 percentage points at the TRAIN/VALID boundary and another
12.04 points from VALID 20223 to 20231, so the decline is cliff-like rather
than gradual.

Frozen-model scoring shows a materially different uncovered population. For
M1, uncovered minus covered is −0.052361 ROC AUC, +0.011697 fail AP, and
+0.033873 Brier; fail rate is 14.19% versus 9.54%. For M2, uncovered minus
covered is +2.136785 MAE, +2.434020 RMSE, and −0.097175 R2; mean final mark is
1.453236 points lower. Base rates are reported to prevent treating every
metric gap as causal model failure.

**Cutoff estimate.** Moving one/two/three VALID semesters into TRAIN would
make 8,939/9,715/9,384 later currently-uncovered rows covered, respectively,
while absorbing 3,613/7,495/8,610 current uncovered rows into TRAIN and
shrinking VALID to 122,177/90,201/79,156 rows. This is evidence only. Any such
move makes existing runs, dataset hashes, and `models/runs/NOISE_BAND.md`
non-comparable.

**PART B.** `first_semester` and `cold_start_gpa` each independently select
14,732 VALID rows with zero mismatches. Their masks reference different
columns, but preprocessing assigns both columns from the same
`no_previous_progress` boolean series. Diagnosis only; no proposal or fix.

**Artifacts and verification.** Full evidence is in
`models/runs/DIFFICULTY_COVERAGE_DIAGNOSTIC.md` and its JSON, generated by
`scripts/difficulty_coverage_diagnostic.py`. The full native suite
(`python -m unittest discover -s tests -t .`) passed 117 tests. Existing unit
tests train synthetic toy models under the OS temporary directory; they did
not retrain either frozen run, read the project TEST split, or write a project
model/dataset artifact. No remedy is recommended or implemented.
