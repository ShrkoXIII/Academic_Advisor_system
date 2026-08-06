# `models/runs/` index — 2026-08-06

Read-only inventory. Nothing here trains, retrains, promotes, or decides
anything. TEST was not opened: no `.parquet` under any `versions/*/` path was
read, and no `.lgbm` internals were opened (directory listings and
`metrics.json` / `REPORT.md` only, per the task's cheap-and-read-only rule).

**Scope of verification.** Every number in this file was recomputed from
on-disk artifacts. Where a claim could not be verified from an artifact, it is
labelled as unverified rather than repeated from a filename or a report.

---

## 1. Is `leaderboard.csv` trustworthy as the index?

The task asked for a 5-run spot-check. The full check was cheap, so all 49 rows
were compared instead.

| Check | Result |
|---|---|
| Leaderboard rows | 49 |
| Rows whose `metrics.json` exists | 49 / 49 |
| Rows where `m1_valid_auc`, `m1_valid_fail_precision`, `m1_valid_fail_recall`, `m1_valid_fail_f1`, `m2_valid_mae`, `m2_valid_r2` all match `metrics.json` to 1e-9 | **49 / 49** |
| Leaderboard rows with no run directory | 0 |

The requested five-directory spot-check, spanning the oldest baseline, the
difficulty change, a multiseed contract run, the R2 confirmation, and the
current rebuild, was:

| Run | Leaderboard AUC | `metrics.json` AUC | Leaderboard MAE | `metrics.json` MAE |
|---|---:|---:|---:|---:|
| `2026-07-12_1208__baseline-39f` | 0.7807 | 0.7807 | 10.018 | 10.018 |
| `2026-07-16_1025__new-difficulty-logic` | 0.8085576195021211 | 0.8085576195021211 | 9.5956 | 9.5956 |
| `2026-07-27_1033__seed62-concurrent-44-registration-roster-candidate` | 0.8102227571432905 | 0.8102227571432905 | 9.5852 | 9.5852 |
| `2026-07-27_1825__seed82-regr2-leaves31-concurrent-43` | 0.8090590604324732 | 0.8090590604324732 | 9.5992 | 9.5992 |
| `2026-08-06_1043__noiseband-2026-08-baseline41-seed42` | 0.8050550056393688 | 0.8050550056393688 | 9.8397 | 9.8397 |

Method: for each `run_id`, load `models/runs/<run_id>/metrics.json` and compare
`m1_pass_classifier.valid.{auc,fail_precision,fail_recall,fail_f1}` and
`m2_grade_regressor.valid.{mae,r2}` against the corresponding CSV columns.

**Verdict: `leaderboard.csv` is accurate for the six metric columns it carries,
for every row it carries.** Its limitation is coverage, not correctness — see
§4.

`leaderboard.csv` does **not** carry: feature contract, seed, dataset version,
reporting threshold, train–valid gap, or Brier. Named contracts and run settings
live in the run artifacts (`feature_contract.json` and, for controlled runs,
`metrics.json.run_settings`). The pre-2026-07-26 runs preserve ordered but
unnamed v1 contracts; they do not preserve seeds, dataset versions, or reporting
thresholds (§3).

---

## 2. Run groups

61 run-shaped directories exist (plus 5 non-run aggregate directories, listed in
§5). Groups below are inferred from run-name and note patterns actually present
in the data, then confirmed against each run's `run_settings`.

### G1 — Legacy migration / pre-governance baselines

| | |
|---|---|
| Runs | 4 |
| Dates | 2026-07-12 |
| Feature contract | Unnamed ordered v1 contracts in `feature_contract.json`: original 39-feature baseline (2 runs); 41-feature `+ diploma_gpa,diploma_type_bucket` (1); 39-feature removal of `start_level_missing,difficulty_fallback_level` while retaining the diploma pair (1) |
| Seeds | **none recorded** |
| Report | none |

`2026-07-12_1208__baseline-39f`, `…__02`, `…_1215__add-diploma-signals`,
`…_1513__remove-dead-const`.

Question answered: establishes the 39-feature starting point and the effect of
adding the two diploma signals (VALID AUC 0.7807 → 0.7820, M2 MAE 10.018 →
9.9064). The first two were not trained here — their `one_line_change` reads
"Migrated existing root-level 39-feature baseline artifacts", which matches
`scripts/migrate_legacy_baseline.py` (copies root-level artifacts into a
hash-verified persistent run).

Caveat: `1208__baseline-39f` and `1208__baseline-39f__02` carry byte-identical
metrics and both are present; `1215` and `1513` are recorded against `__02` as
`baseline_run_id`.

### G2 — Six-level temporal course-difficulty logic

| | |
|---|---|
| Runs | 5 directories — **3 populated, 2 empty** |
| Dates | 2026-07-16 |
| Feature contract | Unnamed ordered v1 39-feature contract in `feature_contract.json` (the post-dead-constant contract with the two diploma features) |
| Seeds | none recorded |
| Report | none in `models/runs/` |

Populated: `1008__new-difficulty-logic`, `1025__new-difficulty-logic`,
`1439__new-difficulty-logic-0-85`. Empty: `1424__`, `1433__` (§4).

Question answered: does the 6-level difficulty fallback chain move M1? VALID AUC
0.7820 → 0.808558.

Verified detail worth recording: all three populated runs share the **identical**
VALID AUC `0.8085576195021211` and identical M2 MAE `9.5956`. They differ only
in fail-class precision/recall/F1 (`1008`/`1025`: 0.5779 / 0.0393; `1439`:
0.2764 / 0.5848). That pattern is consistent with one model reported at two
different cuts. **However, `reporting_threshold` is absent from all three
`metrics.json` files, and neither `REPORT.md` mentions a threshold** — so the
`-0-85` in the directory name is a filename claim I could not confirm from any
artifact. Per CLAUDE.md §6 these are not P/R/F1-comparable to the 0.80 runs
regardless.

### G3 — GPA trend feature

| | |
|---|---|
| Runs | 1 (`2026-07-21_1224__gpa-trend-feature`) |
| Dates | 2026-07-21 |
| Feature contract | Unnamed ordered v1 41-feature contract in `feature_contract.json` (the preceding 39 plus `gpa_trend_delta` and `gpa_trend_missing`) |
| Seeds | none recorded |
| Report | none in `models/runs/`; plan at `docs/plans/2026-07-21_gpa_trend_feature_plan.md` |

Question answered: isolated GPA-trend delta + missing indicator. VALID AUC
0.80918853274908, M2 MAE 9.5667. This generation was promoted to live —
`CURRENT_VERSION.txt` → `2026-07-21_gpa_trend_feature` (commit `13f5cc1`,
"Promote GPA trend feature to live (39 -> 41 features)").

### G4 — `baseline_41` vs `concurrent_44`, five-seed contract comparison

| | |
|---|---|
| Runs | 10 |
| Dates | 2026-07-26 15:51 → 2026-07-27 10:39 |
| Feature contracts | `baseline_41` (5), `concurrent_44` (5) |
| Seeds | 42, 52, 62, 72, 82 (paired) |
| Dataset | `2026-07-26_batched_fixes__registration_roster_concurrent` |
| Report | `models/runs/MULTISEED_BASELINE41_VS_CONCURRENT44_REPORT.md` |

Question answered: do the concurrent peer-difficulty features earn their place?
Report verdict, quoted: "**M1 verdict: INCONCLUSIVE**"; "**M2 verdict:
SUPPORTED**".

These ten runs are also the measurement source for `models/runs/NOISE_BAND.md`,
which names this report as its source and is the acceptance yardstick for the
`2026-07-26_…` split.

First group where `run_settings.test_policy = closed_not_read` appears.

### G5 — `concurrent_43`: drop the dead missing-flag

| | |
|---|---|
| Runs | 5 |
| Dates | 2026-07-27 13:27 → 13:31 |
| Feature contract | `concurrent_43` |
| Seeds | 42, 52, 62, 72, 82 |
| Report | `models/runs/MULTISEED_CONCURRENT43_VS_CONCURRENT44_REPORT.md` |

Question answered: does removing `concurrent_peer_difficulty_missing` (dead —
zero splits in most seeds) change anything? Each run's `baseline_run_id` points
at its same-seed `concurrent_44` counterpart in G4.

A separate programmatic check exists:
`scripts/verify_concurrent_43_vs_concurrent_44.py` →
`models/runs/CONCURRENT43_VS_CONCURRENT44_VERIFICATION.json`.

### G6 — Regularization screening, seed 42, four single levers

| | |
|---|---|
| Runs | 8 (4 configs × 2 arms) |
| Dates | 2026-07-27 14:54 → 15:00 |
| Feature contracts | `baseline_41` (4), `concurrent_43` (4) |
| Seeds | 42 only |
| Report | `models/runs/REGULARIZATION_SCREENING_SEED42_REPORT.md` |

Levers confirmed from each run's `run_settings.lightgbm_params.m1_pass_classifier`
(not from the run name):

| Config | Verified parameter change |
|---|---|
| R1 | `num_leaves` 127 → **63** |
| R2 | `num_leaves` 127 → **31** |
| R3 | `min_child_samples` 50 → **200** |
| R4 | `reg_lambda` 1.0 → **10.0** |

Every other listed parameter is identical across all eight. Question answered:
which single lever shrinks M1's train–valid AUC gap without paying too much
VALID? Report verdicts, quoted: "**R1 … FAIL**", "**R2 … PASS**".

### G7 — R2 (`num_leaves` 31) five-seed confirmation

| | |
|---|---|
| Runs | 8 (seeds 52/62/72/82 × 2 arms) |
| Dates | 2026-07-27 16:00 → 18:25 |
| Feature contracts | `baseline_41` (4), `concurrent_43` (4) |
| Seeds | 52, 62, 72, 82 — the seed-42 R2 pair is reused from G6, not retrained |
| Reports | `R2_CONFIRMATION_5SEED_REPORT.md`, then `R2_COVERED_UNCOVERED_FIVE_SEED_REPORT.md` |

Question answered: does R2 hold across seeds? The follow-up coverage-segment
rescore states the decision verbatim: "**Final decision:
`KEEP_DEFAULT_127_FOR_M1`.**" and "This is a read-only rescore of existing
frozen `baseline_41` M1 binaries. No model was trained" — i.e. that second
report added no run directories.

### G8 — Predecessor-prior pilot (**not trainings**)

| | |
|---|---|
| Directories | 10 (5 seeds × `baseline` / `withprior`) |
| Dates | 2026-07-30 |
| Feature contracts | `baseline_41` and `concurrent_43`, per `metrics.json` |
| Seeds | 42, 52, 62, 72, 82 |
| Report | `models/runs/phase3_predecessor_prior_pilot/PHASE3_PILOT_REPORT.md` |

**These are evaluations, not training runs, and that is why they are correctly
absent from `leaderboard.csv`.** Evidence: each `metrics.json` carries
`"model_artifact_reused_from": "2026-07-26_1551__baseline-41-gpa-trend-control"`
plus a `model_sha256`, and each directory contains only
`m1_pass_classifier_valid_predictions.npy`,
`m2_grade_regressor_valid_predictions.npy`, and `metrics.json` — no `.lgbm`, no
`feature_contract.json`, no `training_curves.json`.

Every one is stamped `"status": "PILOT — PENDING/UNREVIEWED MAPPINGS — NOT FOR
PROMOTION"`.

Outcome, from `Decisions_Log.md`, quoted verbatim:

> the 2026-07-30 predecessor-prior pilot tested this exact mechanism against
> 16,269 of the then-25,627 affected rows and found it **harmful**: M1 AUC fell
> in 5/5 seeds, M2 MAE rose in 5/5 at roughly 4× the published noise band

and

```text
phase_2_decision = NOT_AUTHORISED_BY_OWNER_DESPITE_PROCEED
```

### G9 — Noise band on `2026-08_temporal_rebuild_v1`

| | |
|---|---|
| Runs | 10 |
| Dates | 2026-08-06 10:43 → 10:52 |
| Feature contracts | `baseline_41` ×5 (M1 arm), `concurrent_43` ×5 (M2 arm) |
| Seeds | 42, 52, 62, 72, 82 |
| Dataset | `2026-08_temporal_rebuild_v1` (TRAIN 606,562 / VALID 75,380) |
| Reports | `NOISE_BAND_2026-08_temporal_rebuild_v1.md`; protocol `noise_band_2026-08_protocol.md`; `NOISE_BAND_2026-08_delta_addendum.md` |

Question answered: what is seed-only variation on the *new* split? Confirmed
from `run_settings`: all ten share `num_leaves=127`, `min_child_samples=50`,
`reg_lambda=1.0` (no tuning), `test_policy=closed_not_read`, and resolve to
`…/05_dataset/{train,valid}_dataset_candidate.parquet`.

The protocol file is explicit that it was pre-registered ("Committed before any
training on this split begins"), and the addendum is equally explicit that it
was **not**: "**The ten runs analysed here were trained on 2026-08-06, before
this analysis was scoped.** Nothing in this file was pre-registered."

---

## 3. A cross-group fact worth recording: TEST metrics on disk

Checked every run's `m1_pass_classifier.test` field:

- **8 runs have non-null TEST metrics stored on disk.** All of them are in
  G1–G3: the four 2026-07-12 runs, the three 2026-07-16 runs, and
  `2026-07-21_1224__gpa-trend-feature`.
- **41 runs have `test: null`.** Every training run from `2026-07-26_1551`
  onward.
- **10 directories have no `m1_pass_classifier.test` key at all** — the G8
  pilot evaluations. Their `metrics.json` uses a different shape: no `valid`
  or `test` blocks, and the per-arm fields (`model_artifact_reused_from`,
  `reporting_threshold`, `test_policy`, `segments`) sit nested *inside*
  `m1_pass_classifier` rather than under a top-level `run_settings`.

8 + 41 + 10 = 59, which is the 61 run-shaped directories minus the 2 empty
ones (§4a). Corrected 2026-08-06: an earlier draft of this file reported "51
runs have `test: null`", which silently merged the 41 genuine nulls with the
10 key-absent pilot directories.

This is consistent with the `closed_not_read` policy having taken effect at the
2026-07-26 controlled-baseline run — the first run whose `metrics.json` carries
a `run_settings.test_policy` field at all. It is recorded here as an inventory
fact, not raised as a defect: those eight runs predate the policy. No TEST
parquet was read in producing this document.

Also absent from all 8 pre-2026-07-26 runs: `run_settings` entirely — so no
**named** feature contract, no seed, no dataset version, no reporting threshold,
and no train–valid gap. Their ordered feature lists do exist in
`feature_contract.json`, but the missing run settings still make them
non-comparable to later controlled runs.

---

## 4. Run directories not represented in `leaderboard.csv`

12 directories. They fall into two distinct cases, and the distinction matters.

### 4a. Empty directories — 2 (flagged)

| Directory | Contents | Git |
|---|---|---|
| `2026-07-16_1424__new-difficulty-logic-0-85` | **0 files** | untracked (0 files under it in `git ls-files`) |
| `2026-07-16_1433__new-difficulty-logic-0-85` | **0 files** | untracked |

No `metrics.json`, no model, no report — nothing to reconstruct what they were.
They sit between `1008`/`1025` and the populated `1439` run of the same name,
which suggests abandoned attempts, but **there is no artifact that establishes
this**, so it is a hypothesis, not a finding. Flagged for the owner; not
touched.

### 4b. Legitimately absent — 10

The ten `2026-07-30_1138__predecessor_prior_pilot_*` directories (G8). These are
paired evaluations of a reused model artifact, not trainings, so a leaderboard
row would misrepresent them. Their absence is correct, not an omission.

---

## 5. Non-run directories under `models/runs/`

Five directories are aggregate outputs, not runs:

| Directory | Holds |
|---|---|
| `phase2_human_review` | Phase 2 review tables |
| `phase2_lineage_scope_fix` | Phase 2S outputs |
| `phase2_link_corrections` | Phase 2T outputs |
| `phase2_train_membership_revision` | Phase 2R outputs |
| `phase3_predecessor_prior_pilot` | `PHASE3_PILOT_REPORT.md`, `PHASE3_PILOT_CLAUSES.json`, `phase3_pilot_evaluation.json`, `audit_tables/` |

The Phase 2 lineage work these four hold is closed — see the
`NOT_AUTHORISED_BY_OWNER_DESPITE_PROCEED` decision quoted in §2 G8.

---

## 6. Reconciliation

```
49  leaderboard rows      (G1 4 + G2 3 + G3 1 + G4 10 + G5 5 + G6 8 + G7 8 + G9 10)
+12 unlisted directories  (2 empty + 10 pilot evaluations)
———
 61 run-shaped directories
+ 5 aggregate directories
———
 66 directories under models/runs/
```
