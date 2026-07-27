# CLAUDE.md — Academic Advisor · Models Phase

> **Last updated:** 2026-07-27. This version SUPERSEDES the old "paths/naming
> cleanup" objective — that work is now a deferred debt item (section 9).
>
> **Scope of this file:** the ML modeling workstream only — from the current
> state (regularization pass pending) to the M1/M2 freeze. The product phase
> is summarized in section 10 as direction only; it is NOT in scope until
> explicitly opened.
>
> If a task prompt conflicts with this file, follow the prompt but flag the
> conflict explicitly in your report (as was correctly done on 2026-07-27).

## 1. What this system is

Academic advisor and course-recommendation system for a credit-hour
university. Two LightGBM models are the engines:

- **M1 — pass/fail classifier.** Binary target `final_mark >= 50`.
  Withdrawal rows are removed from the modeling population upstream.
- **M2 — grade regressor.** Target: raw `final_mark` (0–100).

Their outputs feed a decision layer that ranks candidate course sets by
expected AGPA gain and fail risk. M2 feeds deterministic AGPA math, so
**calibrated probabilities are non-negotiable**: no SMOTE, no
`scale_pos_weight`, no fail-class sample weighting — ever.

## 2. Environment

- Windows, Python 3.11, venv. Repo: `D:\AI\Real projects\Academic_Advisor`.
- Data root: env var `ACADEMIC_ADVISOR_DATA_DIR` →
  `D:\AI\data_clean_academic_advisor\data\`.
- 16 GB RAM. Pagefile ACTIVE (verified 2026-07-27 after reboot:
  `C:\pagefile.sys` in `Win32_PageFileUsage`).
- Consequences: **one LightGBM training at a time**, `--num-threads 4`
  (default), and never change memory-relevant parameters to rescue only one
  arm of a comparison.

## 3. Current objective — the only active workstream

**Regularization pass. Two arms: `baseline_41` vs `concurrent_43`.**

- Goal: shrink M1's train–valid AUC gap without losing more VALID
  performance than the noise band allows.
- The design is pre-registered in
  `docs/EXPERIMENT_REGULARIZATION_PLAN.md` — parameter configurations, run
  budget, and the locked success criterion. It was committed BEFORE any
  regularization run was trained and **may not be revised after results are
  seen**.
- Approved so far: **seed-42 screening only** — four single-lever
  configurations × two arms = 8 runs, then STOP. Confirmation across seeds
  52/62/72/82 is a separate task needing separate explicit approval; so is
  any change to `_SHARED_PARAMS` defaults.
- Open question the pass answers: does stronger regularization tame the gap
  enough for M1 to accept the concurrent features? If yes, both models may
  unify on `concurrent_43`; if not, M1 freezes on `baseline_41`.
- Sequence after the pass: freeze M1 → finalize M2's contract → product
  phase (section 10).

## 4. Feature contracts

| Contract | Status | Role |
|---|---|---|
| `baseline_41` | **final for M1** | decided by the 5-seed experiment |
| `concurrent_43` | active candidate | M2's concurrent arm; = `concurrent_44` minus the dead `concurrent_peer_difficulty_missing` |
| `concurrent_44` | **archived** | never use in new runs; exists only inside past reports |

Evidence on record:

- 5-seed `baseline_41` vs `concurrent_44`: M1 mean VALID AUC delta +0.00048
  while the train–valid gap worsened in 4/5 seeds (mean +0.0079) and fail-AP
  was mixed (2/5) → concurrent features **rejected for M1**. M2 improved
  MAE/RMSE/R² together in 4/5 seeds (mean MAE −0.0141, SD 0.0372) →
  **supported for M2**.
- 5-seed `concurrent_43` vs `concurrent_44`: M2 **EQUIVALENT** (all deltas
  inside the noise band). M1 INCONCLUSIVE-positive: the gap narrowed beyond
  the band on the beneficial side in 3/5 seeds and best-iteration dropped
  ~21% — an input for the regularization design, not a contract change.
- **RETRACTED:** the cold-start justification from the single-seed
  experiment (survived in only 3/5 seeds; SD of paired deltas 0.0083 ≈ the
  original 0.0082 effect). Never cite cold-start as evidence for the
  concurrent features.
- Persistent runs MUST pass `--feature-contract` explicitly (test-enforced;
  a persistent run is never silently defaulted). The code default
  `DEFAULT_FEATURE_CONTRACT` is still `concurrent_44` — stale, affects quick
  runs only; changing defaults is wiring and stays deferred until after
  freeze.

## 5. Data

- Immutable version for ALL runs:
  `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent`
  — `df_train_final.parquet` (450,465 rows), `df_valid_final.parquet`
  (156,097 rows); 84 columns = 76 source + 8 concurrent.
- **TEST is `closed_not_read`.** Never pass `--evaluate-test`. Every
  training command passes a NONEXISTENT `--test` path — that is the proof
  mechanism (reading it would raise); keep it.
- The CLI defaults for `--train/--valid/--test` point at the LIVE
  `model_split_path(...)` — **never rely on them**. Every run passes
  explicit paths.
- Never copy datasets into new folders. Two sources of truth break the
  frozen-version hash guarantees. Promotion later = flipping
  `CURRENT_VERSION.txt`, not moving files.
- Live `MODEL_DATA_DIR` untouched; `CURRENT_VERSION.txt` →
  `2026-07-21_gpa_trend_feature`. **No promotion before freeze.** Before any
  promotion: quarantine legacy versions `2026-07-25_150014` and
  `2026-07-23_160509` (the latter is still `PRIOR_BUILD_ID`).

## 6. Run protocol

- CLI (`src/model_training.py`): `--feature-contract`, `--seed`, explicit
  `--train/--valid/--test`, `--run-name` (persistent), `--note`,
  `--compare-to`, `--num-threads` (default 4). Forbidden before freeze:
  `--evaluate-test`.
- `--seed` is the master seed; LightGBM deterministically derives
  `data_random_seed`, `feature_fraction_seed`, `bagging_seed`, `drop_seed`
  from it. Both arms of any pair share one seed. Unseeded runs are not
  comparable to seeded runs.
- Canonical seed set: **42, 52, 62, 72, 82**.
- Reporting threshold **0.80, fixed** — cross-run comparability only. It is
  NOT the product threshold; that is a post-freeze business decision
  (possibly segment-specific). Runs cut at other thresholds (0.5, 0.85
  history) are not P/R/F1-comparable.
- Selection on VALID only. Higher-better: AUC, AP, R², precision/recall/F1.
  Lower-better: Brier, train–valid gap, MAE, RMSE.
- **`models/runs/NOISE_BAND.md` is the acceptance yardstick.** A delta
  inside the band is not evidence — liked or not. Never invent acceptance
  thresholds after seeing results.
- One change per run. Fixed LightGBM mechanics across any compared pair:
  2000-round cap, 50-round early stopping on VALID, train-only diploma-GPA
  median fill, identical categorical levels.
- Every run writes `feature_contract.json` and `metrics.json` including the
  threshold, effective seed fields, and test policy.
- Segments: `first_semester` and `cold_start_gpa` are currently the SAME
  population (n=14,732) — count them as ONE piece of evidence (open defect,
  section 9).

## 7. Git discipline

- Commit code BEFORE training so runs record a clean tree; commit run
  provenance (json/csv/md only) after.
- **Never push. Never promote.** Print the push command and stop; the user
  pushes.
- Never stage: `.lgbm`, `.parquet`, `.log`;
  `note_books/training_notebooks/inspect_train.ipynb`; the 7 untracked
  historical run folders — unless explicitly instructed.
- Verification is evidence-first: recompute metrics from on-disk artifacts
  (`metrics.json`, models, data). Never trust a report's numbers — including
  your own previous reports.

## 8. Role of Claude Code in this repo

Mechanical implementer. Experiment design, acceptance criteria, and
accept/reject decisions are made by the human outside these sessions.
Execute the scoped prompt, verify from artifacts, report, STOP at the gates,
and wait. When in doubt, stop and ask — do not improvise, and do not make
the project decision.

## 9. Known open defects — do not silently fix inside other tasks

- `first_semester` == `cold_start_gpa` identical populations (n=14,732) —
  definition/masking bug; blocks segment-based decisions until fixed.
- Difficulty-coverage decay: Level-1 coverage 93.6% train → 76.2% valid →
  44.7% test; drives the 0.186/0.134 difficulty shift; needs its own
  investigation before TEST is ever opened.
- `01_train_lightgbm.ipynb` trains on live defaults regardless of its own
  path variables — experiments go through the CLI only, never the notebook.
- `parity_check.py` uses cwd-relative paths; `merge_diploma.py` contains
  dead code.
- Path-governance debt (~45 `model_split_path` conversion sites, 15 missing
  constants) — deferred workstream, NOT active. This was this file's
  previous "current objective"; it is superseded, not cancelled.

## 10. After the freeze — direction only, NOT current scope

Do not start any of this without an explicit new instruction: real AGPA
engine replacing the `_mark_to_gpa_points` approximation; eligibility engine
(currently empty); degree-progress constraints (easy-course bias is
unbounded given the 150/230 pool structure); real course credits from
`V_SCH_COURSE_OFFER` (3.0 fallback today); plan-generation fixes (caps ~5
courses while students take 6; ranking biases small plans, 0.20 vs 0.10);
set-level backtesting — the real product metric; V1 candidate course sets
come from advisors, not auto-generation; KNN is an explanation/evidence
tool, not a second predictor; the LLM layer explains results only; product
threshold(s), possibly per segment, as a business decision.

## 11. Authoritative documents

- `Decisions_Log.md` (repo root) — decision history used by tasks. If it
  conflicts with the Obsidian vault, stop and ask — do not pick a winner.
- `models/runs/NOISE_BAND.md` — the noise band (acceptance yardstick).
  Measured from contract-change deltas across seeds, NOT from
  hyperparameter-change deltas — the best available yardstick for the
  regularization pass, not an exact one. Never treat it as precise.
- `docs/EXPERIMENT_REGULARIZATION_PLAN.md` — pre-registered plan and locked
  acceptance rule for the regularization pass. Frozen once committed.
- `docs/pipeline_rules.md` — CLI and pipeline rules.
- Latest reports:
  `models/runs/MULTISEED_BASELINE41_VS_CONCURRENT44_REPORT.md`,
  `models/runs/MULTISEED_CONCURRENT43_VS_CONCURRENT44_REPORT.md`.