# Academic Advisor — Full Pipeline Rules (PAUSED reference)

> These are the full project rules for the LOGIC job (targets, leakage,
> merges, ML metrics). They are PAUSED while the paths-and-names cleanup
> runs. When cleanup is done, move the sections you need back into
> `CLAUDE.md` and shrink the paths section there to one line.
>
> To use these during the logic job, tell Claude Code: "read
> docs/pipeline_rules.md before this task".

---

## Project Goal

Course outcome prediction and semester course-plan recommendation for a
credit-hour university.

* M1 = pass probability classifier. Probabilities MUST be calibrated —
  downstream AGPA math and plan scoring depend on probability quality.
* M2 = final mark regressor, target range 0–100.
* Full flow: course predictions → plan candidate generation → KNN
  similar-student evidence → plan scoring (AGPA / risk / workload /
  graduation progress) → ranked recommendations.

## Pipeline order

> Updated 2026-07-07 (governance decision D1/D2 — see `docs/data_governance_plan.md`):
> the diploma merge moved upstream, and the split enrichment stages write
> distinct generations instead of rewriting the split files in place.

clean → merge CRG+ADD+ACD (`01_merge_crg_add_acd`) → diploma merge
(`02_merge_diploma`, distinct `merged_with_diploma.parquet`) → feature
selection (`select`) → feature engineering (`handle_gpa`) →
`split_diagnostics.ipynb` (base splits) → `course_difficulty.ipynb`
(difficulty generation) → `diploma_type_bucketing.ipynb` (final generation) →
training → KNN index → recommendation.

## src/ module ownership

* `cleaning_utils.py` — single owner of ID normalization rules.
* `feature_engineering.py` — builds features, owns `after_fet_eng.parquet`.
* `merge_diploma.py` — **SUPERSEDED (D1, 2026-07-07; neutralized in Phase 7a).**
  Historically extended `after_fet_eng.parquet` with `diploma_gpa` /
  `diploma_type_id` as a documented exception. The diploma join now happens
  upstream in the `02_merge_diploma` notebook; `after_fet_eng.parquet` has a
  single writer — the feature-engineering stage. The script is kept for
  history and raises on execution.
* `model_training.py` — trains M1/M2.
* `inference.py` — `StudentScorer` (course-level scoring at inference time).
* `knn_advisor.py` — KNN index build/query over train-only snapshots.
* `recommendation.py` — plan generation, scoring, ranking.

## Target Definitions

* M1 target MUST be derived from the approved academic outcome definition
  based on `finish_status`.
* DO NOT define pass/fail as `final_mark >= 50` anywhere — not in the model
  target, not in KNN evidence aggregates, not in evaluation metrics.
* There must be exactly ONE pass-definition helper function, imported by
  every module that needs it.
* M2 target = `final_mark`. Never change target definitions without
  explicit approval.

> KNOWN VIOLATION to fix in the logic job: `knn_advisor.py` computes
> `sem_pass_rate` with `(final_mark >= 50)`, and `recommendation.py`
> `_mark_to_gpa_points` bakes in the same 50 cutoff. Both must move to the
> single pass-definition helper.

## Artifact Ownership

* One notebook/script owns each saved artifact. An owner may rewrite ONLY
  the artifacts it owns; nothing else may touch them.
* `split_diagnostics.ipynb` exclusively owns `df_train` / `df_valid` /
  `df_test`. Downstream notebooks read them, never recreate them.
* `diploma_type_bucketing.ipynb` may rewrite the split parquets in place
  (idempotent, documented exception).
* Before changing any artifact, identify its owner first.

## Locked Decisions (do not change without explicit approval)

* Temporal split: Train 2005–2021 / Validation 2022–2023 / Test 2024 + 2025 S1.
* Feature set = explicit allowlist in `feature_contract.json`. Never a
  denylist. Never modify the contract without approval.
* Leakage control: ALL learned statistics come from TRAIN ONLY — course
  difficulty, diploma bucketing, historical rates, aggregates, thresholds,
  learned mappings, KNN index, scalers, and imputation medians.
  Validation/test outcomes must never influence any of them.
* Unseen/missing categories get explicit documented fallback codes
  (`-1`, bucket `6`, fallback level `6`). Never silently drop them.
* Class imbalance: no `scale_pos_weight`, no SMOTE. Calibration quality is
  a hard requirement.
* One unified model unless evaluation evidence proves segmentation is needed.

## Merge Rules

For every merge:

* State merge keys explicitly.
* Use `validate="many_to_one"` where the expected relationship is many-to-one.
* Assert row count before and after. Print unmatched counts.
* Never silently drop rows. Never silently deduplicate to make a merge pass.

### ACD Degree-Course Rule

ACD curriculum metadata is degree-specific. Never borrow
`requirement_type_id`, degree-specific requirement credits, or other
curriculum metadata from another `degree_id` merely because the same
`course_id` exists there. Unmatched rows stay as explicit missing/fallback
cases and are audited, unless a fallback rule is explicitly approved.

## Row-Change Rules

Any operation that can change row count prints: before count, after count,
delta, and the reason. No silent drops.

## Working Rules for Claude

* Make minimal changes. Do not refactor unrelated code.
* Inspect the current implementation before proposing edits. Never infer
  behavior from a filename or a notebook title.
* If a docstring/comment and the code disagree, trust the executed code and
  report the inconsistency.
* Never modify split logic, saved split parquets, target definitions, or
  `feature_contract.json` without explicit approval.
* Do not overwrite artifacts you do not own unless explicitly requested.
* Preserve existing audit outputs unless explicitly asked to remove them.
* Before writing code, explain the idea in plain language.
* When uncertain about a data or ML decision, stop and report the
  uncertainty instead of inventing a rule.
* If an existing decision is statistically unsafe, leakage-prone, or
  logically inconsistent, warn explicitly rather than following it silently.
* After refactoring any pipeline stage, run `scripts/parity_check.py` and
  report the diff before declaring the work done.

## ML Validation Priorities

For M1: report at minimum Log Loss, Brier Score, calibration
curve/reliability, and a discrimination metric (ROC-AUC or PR-AUC).
ROC-AUC alone is not sufficient.

For M2: report at minimum MAE, RMSE, and error broken down by important
segments (cold-start vs returning students, level, course difficulty band).

Always compare train, validation, and untouched test performance separately.

## Known issues to address in the logic job

* Pass-definition violation in `knn_advisor.py` and `recommendation.py`
  (see Target Definitions above).
* `recommendation.py` composite-score weights: docstring says knn=0.20 but
  code applies 0.20 to workload and adds knn as a separate unweighted bonus.
  Decide intended design, then make code and docstring agree.
* `merge_diploma.py` uses hardcoded paths and imports the private
  `_normalize_key_series` from `feature_engineering`. Move that helper to
  `cleaning_utils.py` as a public function.
* `knn_advisor.py` stores imputation medians as an attribute on the sklearn
  scaler (`scaler.median_fill_`). Move them to a top-level key in the pickle.
* `_build_snapshot_df` assumes semester-level features are constant across
  course rows. Add an assertion to verify.
* `_mark_to_gpa_points` is hand-coded. Replace with a lookup from the
  `v_acs_grade` source so the grade scale has one authority.
* Legacy data-root references in docstrings
  (`D:/AI/data_clean_academic_advisor/...`) must be removed.

## Final Rule

Reproducibility, leakage prevention, calibrated probabilities, and
academic-rule correctness take priority over small metric improvements.