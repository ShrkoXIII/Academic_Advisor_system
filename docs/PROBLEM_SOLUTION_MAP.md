# Problem → Solution Map — Academic Advisor

Generated: 2026-08-09 | Commit: `3d73cbe` (+1 untracked file: `scripts/logical_parity_check.py`, not yet committed) | Dataset version: **three live candidates, not unified** — `CURRENT_VERSION.txt` → `2026-07-21_gpa_trend_feature` (the actually-promoted live version); `2026-07-26_batched_fixes__registration_roster_concurrent` (the frozen basis CLAUDE.md §5 names for all seeded experiments, and for the committed `NOISE_BAND.md`); `2026-08_temporal_rebuild_v1` (the newest rebuild, `version_promoted: false` in its own Phase 3 report). See PS-52.

Basis: delta from `docs/manifests/freeze_phase9_2026-07-08.md`. `claude_code_prompt_project_inventory.md` does not exist in this repo; its evident successors — `docs/manifests/codebase_map_2026-08.md`, `docs/manifests/models_runs_index_2026-08.md`, `docs/manifests/project_pipeline_routes_2026-08.md`, and the five `docs/map/*.md` stage documents (all committed `28fb623`, 2026-08-06) — were used as the inventory basis instead, per rule 7 (cite recorded evidence, don't re-derive it).

## How to read this file

Every entry is `PS-XX`: a problem, the solution built for it, the file it lives in today, and how we know it worked — or `NO VERIFICATION FOUND` / `UNVERIFIED` where that evidence doesn't exist. `Implemented in` cites a notebook by cell number where the solution is notebook-based (this codebase is a script/notebook hybrid) and a `file.py:LINE-LINE` range where it's a module. `CONTRADICTS BRIEF` marks a place where the repo says something different from what this task's brief assumed.

## Summary table

| ID | Stage | Problem (5 words) | Status |
|----|-------|-------------------|--------|
| PS-01 | 0 Extract | Extractor SQL misaliases a column | PLANNED |
| PS-02 | 0 Extract | Unset data root drifted writes | LIVE |
| PS-03 | 0 Extract | Ad-hoc parquet writes, no guard | LIVE |
| PS-04 | 1 Clean | Three parallel ID-normalizer implementations | LIVE |
| PS-05 | 1 Clean | Dotted IDs corruptible by float casts | LIVE |
| PS-06 | 1 Clean | student_id float64 blocked all merges | LIVE |
| PS-07 | 2 Merge | Diploma merge double-wrote one artifact | SUPERSEDED |
| PS-08 | 2 Merge | diploma_gpa fill uses future cohorts | ABANDONED |
| PS-09 | 3 Population | M1 needed one explicit binary target | LIVE |
| PS-10 | 3 Population | No single pass-definition helper exists | PLANNED |
| PS-11 | 4 Leakage | Fitted stats could leak future rows | LIVE |
| PS-12 | 4 Leakage | Inference could silently refit difficulty | LIVE |
| PS-13 | 5 Features | Two cold-start segments are identical | PLANNED |
| PS-14 | 5 Features | No leakage-safe GPA-trend signal | LIVE |
| PS-15 | 5 Features | Plan caps were unnamed magic numbers | LIVE |
| PS-16 | 6 Difficulty | Difficulty coverage decays by split | LIVE |
| PS-17 | 6 Difficulty | Flat stats fail sparse/new courses | LIVE |
| PS-18a | 6 Difficulty | In-place difficulty rewrite was unsafe | SUPERSEDED |
| PS-18b | 6 Difficulty | Default notebook route now broken | PLANNED |
| PS-19 | 7 Concurrent | Target excludes withdrawn peers | LIVE |
| PS-20 | 7 Concurrent | Dead feature inflated the contract | LIVE |
| PS-21 | 7 Concurrent | Position gate pinned to ambiguous global | LIVE |
| PS-22 | 8 Identity | 182 course IDs have zero TRAIN history | PLANNED |
| PS-23 | 8 Identity | Does a predecessor prior help? | ABANDONED |
| PS-24 | 8 Identity | Phase 2 remap never authorised | ABANDONED |
| PS-25 | 9 Split | Old TEST was the only 2024 data | LIVE |
| PS-26 | 9 Split | Boundary change's coverage effect unmeasured | LIVE |
| PS-27 | 9 Split | One degree, two faculties, fit crashed | LIVE |
| PS-28 | 9 Split | Streaming assumed a stored pandas index | LIVE |
| PS-29 | 9 Split | Diploma map fit on superseded split | LIVE |
| PS-30 | 9 Split | No version-scoped path resolver existed | LIVE |
| PS-31 | 9 Split | No script for final split/assembly | LIVE |
| PS-32 | 10 Training | Calibration must survive class imbalance | LIVE |
| PS-33 | 10 Training | Three contracts, one archived, ambiguity risk | LIVE |
| PS-34 | 10 Training | Runs at different thresholds aren't comparable | LIVE |
| PS-35 | 10 Training | Stale default could silently mis-train | LIVE |
| PS-36 | 10 Training | Shrink M1's gap without hurting VALID | PLANNED |
| PS-37 | 10 Training | Two contracts, one candidate dataset | LIVE |
| PS-38 | 11 Evaluation | One seed is not a stable effect | LIVE |
| PS-39 | 11 Evaluation | Noise band invalid on new split | LIVE/SUPERSEDED |
| PS-40 | 11 Evaluation | Acceptance rules must predate results | LIVE |
| PS-41 | 12 Repro | Leaderboard numbers could go stale | LIVE |
| PS-42 | 12 Repro | A rerun could drift off its lever | LIVE |
| PS-43 | 12 Repro | Byte-parity is the wrong rebuild bar | LIVE |
| PS-44 | 13 Inference | Trained models need a scoring bridge | LIVE |
| PS-45 | 14 Recommend | GPA scale has no single authority | PLANNED |
| PS-46 | 14 Recommend | Plan course-count cap value | LIVE |
| PS-47 | 14 Recommend | Composite score docstring ≠ code | PLANNED |
| PS-48 | 14 Recommend | Pass threshold hardcoded five times | PLANNED |
| PS-49 | 14 Recommend | No eligibility/prerequisite engine exists | PLANNED |
| PS-50 | 14 Recommend | KNN index has no producer decision | PLANNED |
| PS-51 | 16 Governance | Set-level backtesting does not exist | PLANNED |
| PS-52 | 16 Governance | CLAUDE.md's objective section is stale | LIVE |
| PS-53 | 16 Governance | A governance log misstated its authorship | LIVE |

---

## Stage 0 — Raw extraction from Oracle

### PS-01 — Extractor SQL bug misaligns a raw column

- **Stage:** 0 Raw extraction
- **Problem:** In the `V_ADD_ACADEMIC_INFO` query inside the raw extractor, a missing comma makes `DIPLOMA_TYPE_SL` get aliased as `ACTIVE`; the real `ACTIVE` column is never selected (the `WHERE ACTIVE='A'` filter still applies upstream in Oracle).
- **Why it mattered:** The saved raw "active" column actually contains diploma-type text, and `diploma_type_sl` is silently unavailable downstream — currently harmless only because nothing reads that column today.
- **Evidence of the problem:** `docs/data_governance_plan.md:305` ("Latent extractor SQL bug (flag only — logic fix out of scope)").
- **Solution:** None. Flagged explicitly as a data-logic change requiring its own approval; excluded from the paths/naming job that found it.
- **Implemented in:** NOT IMPLEMENTED.
- **How we know it worked:** NO VERIFICATION FOUND — the defect is undisturbed.
- **Status:** PLANNED

### PS-02 — Unset/misconfigured data root silently drifted writes into the wrong tree

- **Stage:** 0 Raw extraction
- **Problem:** `src/paths.py`'s import-time `ensure_dir` loop materializes missing folders, which can mask an unset or mistyped `ACADEMIC_ADVISOR_DATA_DIR` by silently creating and writing into an empty shell tree instead of the real data root — the "twin-copy drift" risk.
- **Why it mattered:** This was not theoretical. Two out-of-band commits (`54eed08`, `4b0d3b1`) re-executed `clean_v_acd_degree_course.ipynb` with the guard absent: the first run failed (wrong path, no write), the second succeeded but wrote into the repo-shell tree instead of the active root.
- **Evidence of the problem:** `docs/manifests/freeze_phase9_2026-07-08.md:9-15` (the incident, sha256-verified as not having touched governed data).
- **Solution:** `assert_data_root()` — refuses to run if the resolved `DATA_DIR` didn't pre-exist before import (distinguishing "populated root" from "root `ensure_dir` just created"), and asserts required input artifacts exist and are non-empty. Wired into every writer notebook entry point that lacked equivalent protection (`00_extract_raw_tables`, the five per-table cleaners, `01_select_model_population`).
- **Implemented in:** `src/paths.py:83-101`
- **How we know it worked:** `scripts/parity_check.py` re-run at 51/51 PASS and the native unit suite at 10/10 after the guard was wired in (`docs/manifests/freeze_phase9_2026-07-08.md:17`).
- **Status:** LIVE
- **Caveat, not re-opened here:** training/inference entry points (`01_train_lightgbm.ipynb`, `src/inference.py`) are not wired with this guard — recorded as "partial wiring" (`docs/data_governance_plan.md:75`), carried forward, non-blocking.

### PS-03 — Ad-hoc `to_parquet()` calls had no guaranteed parent directory or shared write behaviour

- **Stage:** 0 Raw extraction
- **Problem:** Every notebook called `df.to_parquet(PATH, ...)` directly, with no shared parent-directory creation or write contract.
- **Why it mattered:** A refactor touching every save call site in the pipeline is exactly the kind of change that can silently alter behaviour at one call site while looking correct everywhere else.
- **Evidence of the problem:** Implicit in the fact that a single-helper refactor was worth doing at all; verified retroactively, line-by-line, during the Phase 9 freeze audit rather than pre-declared as a defect.
- **Solution:** `save_parquet()` centralizes `ensure_parent_dir()` + `df.to_parquet(output_path, index=index, **kwargs)`; every notebook call site was rewritten to call it.
- **Implemented in:** `src/io_utils.py:8-27`
- **How we know it worked:** Diffed line-by-line during the Phase 9 audit — every call site confirmed semantically identical to what it replaced (`docs/manifests/freeze_phase9_2026-07-08.md:9`).
- **Status:** LIVE

## Stage 1 — Per-table cleaning

### PS-04 — Three parallel ID-normalizer implementations risked divergent dotted-ID handling

- **Stage:** 1 Per-table cleaning
- **Problem:** ID normalization existed in three places at once — the `cleaning_utils` family, `feature_engineering._normalize_key_series`, and ad-hoc notebook casts — with no single canonical implementation.
- **Why it mattered:** Dotted university-suffix IDs (e.g. `15.111`) are identity, not decimals; a second, subtly different normalizer is exactly how a join key silently corrupts.
- **Evidence of the problem:** `docs/governance_contracts.md:9-21` (Contract 1: "the Phase 2 finding of three parallel implementations is debt, not license").
- **Solution:** Contract 1 designates `cleaning_utils.normalize_id_series` / `normalize_id_columns` (element rule `normalize_id_to_string`) as the canonical normalizer for every ID class, applied at source-cleaning time, never patched at merge time.
- **Implemented in:** `src/cleaning_utils.py:62-122` (`normalize_id_to_string`, `normalize_id_series`, `normalize_id_columns`)
- **How we know it worked:** Codebase-map census confirms live imports at `src/concurrent_group_features.py:76` and `src/registration_roster.py:39`; pipeline join paths verified conforming as of commit `c6069bc` (`docs/governance_contracts.md:21`).
- **Status:** LIVE
- **Superseded by / caveat:** `feature_engineering._normalize_key_series` remains a second live normalizer — explicitly logged as post-freeze consolidation debt, not swapped during this job.

### PS-05 — Dotted-suffix IDs are silently corruptible by float casts

- **Stage:** 1 Per-table cleaning
- **Problem:** Raw `diploma_type_id` is float64 with a dotted suffix at source. A `to_numeric`/`astype(float)` anywhere on a dotted ID silently breaks it; the cleaning notebook even carried a dead `fillna(6.111)` on `diploma_type_id` — a float literal encoding a dotted ID, the project's own worked example of the failure mode.
- **Why it mattered:** A join on a corrupted ID fails silently (wrong match or no match), not loudly.
- **Evidence of the problem:** `docs/data_governance_plan.md:112` (item 12, cell 7 of `Academic_info_clean.ipynb`), `:299` ("Dotted-ID corruption" risk).
- **Solution:** Per-table ID-class registry (`src/schemas.py`) drives `src/id_casting.py::normalize_ids`, applied at source-cleaning time in every current cleaner notebook; `diploma_type_id` specifically: suffix-consistency check → strip → `Int64` cast at cleaning time (contract 1 table).
- **Implemented in:** `src/id_casting.py:16` (`normalize_ids`), `src/schemas.py:16-142` (`IdColumnRule`, `ID_COLUMN_RULES`, `TABLE_ID_COLUMNS`)
- **How we know it worked:** `docs/map/01-extract-and-clean.md:82-83` confirms `normalize_ids` is imported and applied in all five current cleaner notebooks at named cell indices.
- **Status:** LIVE

### PS-06 — Unnormalized `student_id` blocked the diploma merge from ever materializing

- **Stage:** 1 Per-table cleaning
- **Problem:** The on-disk cleaned diploma artifact had `student_id` as float64 even though normalization code existed — the artifact predated the code. The diploma merge's own dtype guard correctly blocked every run against it.
- **Why it mattered:** `merged_with_diploma.parquet` had literally never been materialized before this was fixed — a whole downstream chain (select → feature-eng → splits) was blocked on one unnormalized column.
- **Evidence of the problem:** `docs/data_governance_plan.md:175` (Phase 2, "Confirmed CRITICAL: student_id float64-sourced/unnormalized on the diploma join path"), `:304`.
- **Solution:** `student_id` normalized with `cleaning_utils.normalize_id_columns` at diploma-source cleaning time, guarded by row-count/unique-key/dtype asserts.
- **Implemented in:** `note_books/pre_processing/V_ADD_ACADEMIC_INFO/clean_v_add_academic_info.ipynb` (cell 7 area, per `docs/data_governance_plan.md:112`)
- **How we know it worked:** Chain A stage 1 executed successfully in the 7c rebuild for the first time; `docs/manifests/validation_7c_2026-07-08.md` records the full validation.
- **Status:** LIVE

## Stage 2 — Merging tables to one grain

### PS-07 — The diploma merge extended a single-owner artifact in place, creating a second writer

- **Stage:** 2 Merging tables
- **Problem:** The legacy `src/merge_diploma.py` joined `diploma_gpa`/`diploma_type_id` onto `after_fet_eng.parquet` **in place** — the same artifact the feature-engineering stage owned and wrote — violating one-owner-per-artifact.
- **Why it mattered:** Two writers of one path is exactly the ownership conflict `docs/data_governance_plan.md` was written to close (conflict C1); a second run could silently overwrite the other's work.
- **Evidence of the problem:** `docs/governance_contracts.md:32` (Contract 2, "C1 enforcement pending"); `docs/data_governance_plan.md:54-56` (D1).
- **Solution:** The diploma join moved upstream into a dedicated `02_merge_diploma` notebook that reads the CRG+ADD+ACD merge output plus the cleaned diploma source and writes a **distinct** artifact (`merged_with_diploma.parquet`); `src/merge_diploma.py` was neutralized with a module-level `raise RuntimeError` before any import or I/O.
- **Implemented in:** `note_books/merge/02_merge_diploma.ipynb` (cells 1, 2, 7, 10 per `docs/map/02-merge-and-features.md:86`); guard at `src/merge_diploma.py:12`
- **How we know it worked:** "The D1 chain ran end-to-end in 7c with runtime guards passing" (`docs/data_governance_plan.md:70`); repo-wide grep confirms `merge_diploma.py` has no live caller (`docs/manifests/codebase_map_2026-08.md:110`).
- **Status:** SUPERSEDED (superseded by `note_books/merge/02_merge_diploma.ipynb`)

### PS-08 — `diploma_gpa`'s null fill is computed over the full source population, including future cohorts

- **Stage:** 2 Merging tables
- **Problem:** `Academic_info_clean.ipynb`'s median fill for `diploma_gpa` is computed over the **entire** diploma source population at cleaning time — upstream of the split, so the statistic is fit on data that includes future validation/test cohorts. This conflicts with the project's own train-only leakage rule.
- **Why it mattered:** If left silently encoded as compliant, it would have been a genuine, undisclosed leakage path into a feature used by both models.
- **Evidence of the problem:** `docs/data_governance_plan.md:24` ("D6"), `:112` (cell 7).
- **Solution:** Not a code fix. The owner reviewed the conflict and explicitly accepted it as a **logged exception** for this cycle: the affected population is small (~40 rows), the values are pre-admission facts available before any target semester, and the exception does not generalize to any future fitted statistic (contracts 14–15 still apply to everything else).
- **Implemented in:** Decision recorded at `docs/governance_contracts.md:99` (Contract 13) and `docs/data_governance_plan.md:317` (D6 resolution) — no code changed.
- **How we know it worked:** N/A by design — the "solution" is the explicit decision not to remediate, not a code change to verify.
- **Status:** ABANDONED
- **Abandoned because:** User decision 2026-07-07, logged in the (out-of-repo) Obsidian `Decisions_Log.md` per `docs/data_governance_plan.md:317`; rationale = small affected count + pre-admission-fact status.

## Stage 3 — Population selection & target definition

### PS-09 — M1 needed one explicit, unambiguous binary target computed only over completed attempts

- **Stage:** 3 Population & target
- **Problem:** A course-recommendation model needs a single, precisely defined pass/fail label, and it must not be computed over registrations that never resulted in a grade.
- **Why it mattered:** An ambiguous or leaky target definition would invalidate every downstream metric and the AGPA math built on top of it.
- **Evidence of the problem:** N/A — this is a design requirement, not a discovered defect; stated directly in `CLAUDE.md:19-20`.
- **Solution:** `TARGET_M1_DEFINITION = "(final_mark >= 50).astype(int)"`, applied only to the modeling population after withdrawal rows are removed upstream.
- **Implemented in:** `src/model_training.py:246` (constant), `:502` (`y = (df[TARGET_GRADE] >= 50).astype(int)` inside `prepare_X_y`, `429-512`)
- **How we know it worked:** Every one of the 49 leaderboard-recorded training runs applies this target identically; independently recomputed to 1e-9 against `metrics.json` for all 49 (`docs/manifests/models_runs_index_2026-08.md:14-23`).
- **Status:** LIVE

### PS-10 — No single pass-definition helper exists, so `>= 50` is hardcoded independently in at least five places

- **Stage:** 3 Population & target
- **Problem:** `docs/pipeline_rules.md` requires exactly one pass-definition helper function, imported everywhere it's needed. That helper does not exist. `final_mark >= 50` is hardcoded independently in `src/course_difficulty.py:251,329`, `src/knn_advisor.py:75`, `src/model_training.py:502`, and inside `src/recommendation.py:57-71`'s GPA-point conversion.
- **Why it mattered:** If the pass threshold or its logic ever needs to change, only *some* of five independent sites would be updated, silently desynchronizing the training target from the KNN evidence and the recommendation-layer GPA conversion.
- **Evidence of the problem:** `docs/pipeline_rules.md:56-67` ("There must be exactly ONE pass-definition helper function... KNOWN VIOLATION to fix in the logic job") — confirmed still present by direct grep of current code (five hardcoded sites, zero shared helper).
- **Solution:** None implemented.
- **Implemented in:** NOT IMPLEMENTED.
- **How we know it worked:** NO VERIFICATION FOUND.
- **Status:** PLANNED

## Stage 4 — Leakage control

### PS-11 — Fitted statistics could leak validation/test rows into training-time knowledge

- **Stage:** 4 Leakage control
- **Problem:** Course-difficulty stats, diploma-type buckets, categorical level sets, GPA-trend deltas, and any other learned statistic must be fit on TRAIN only — a single mis-scoped `.fit()` on the full frame is a silent, hard-to-detect leak.
- **Why it mattered:** M2 feeds deterministic AGPA math; a leaked statistic doesn't just hurt a metric, it corrupts the product's core arithmetic.
- **Evidence of the problem:** `docs/pipeline_rules.md:84-87` (locked decision, stated as a rule rather than a discovered bug).
- **Solution:** Contracts 14–15 (`docs/governance_contracts.md:101-115`) require every train-fitted transformation to persist its fitted state as a versioned artifact and forbid refitting at inference; enforced in code for diploma bucketing (`fit_diploma_bucket_map`), course difficulty (`fit_difficulty_state`), and categorical levels (`learn_categorical_levels`).
- **Implemented in:** `src/diploma_bucketing.py:101-148`, `src/course_difficulty.py:447-556`, `src/model_training.py:381-394`
- **How we know it worked:** Phase 3 rebuild report states diploma bucketing was verified fitted on TRAIN only (606,562 rows through `20233`); course difficulty verified incremental on TRAIN by replay (`Decisions_Log.md:1683-1685`).
- **Status:** LIVE

### PS-12 — Inference-time preprocessing could silently re-derive state instead of reusing the frozen train-time version

- **Stage:** 4 Leakage control
- **Problem:** Applying course-difficulty or diploma-bucket logic at VALID/TEST/inference time by recomputing (rather than loading) the fitted state would use information those rows shouldn't have.
- **Why it mattered:** This is the leakage risk contract 15 exists to close, one level below "fit on the right split" — even a *correctly train-fitted* statistic can leak if it's silently re-fit at apply time.
- **Evidence of the problem:** `docs/governance_contracts.md:113-115` (Contract 15).
- **Solution:** `apply_difficulty_state` never updates from its input; it only applies a loaded, frozen lookup.
- **Implemented in:** `src/course_difficulty.py:557` (`apply_difficulty_state`)
- **How we know it worked:** "Verified by replay: reloading the persisted state and re-applying it reproduces all nine difficulty columns exactly" (`Decisions_Log.md:1680-1682`).
- **Status:** LIVE

## Stage 5 — Feature engineering — student timeline

### PS-13 — `first_semester` and `cold_start_gpa` are reported as two segments but are definitionally identical

- **Stage:** 5 Student-timeline features
- **Problem:** Both segment columns are cast from the exact same boolean Series in the same function, one line apart.
- **Why it mattered:** Any report that shows both as independent evidence points is silently double-counting one signal — this is required entry #8, confirmed by direct code inspection.
- **Evidence of the problem:** `src/feature_engineering.py:402-403`:
  ```python
  semester_df["no_previous_progress"] = no_previous_progress.astype(int)
  semester_df["is_first_active_semester"] = no_previous_progress.astype(int)
  ```
  consumed identically at `src/model_training.py:850,853`: `"first_semester": df["is_first_active_semester"] == 1` and `"cold_start_gpa": df["no_previous_progress"] == 1`.
- **Solution:** None — code-level fix not implemented. Governance workaround only: both segments are treated as **one** piece of evidence, never two, in every report that uses them.
- **Implemented in:** NOT IMPLEMENTED (workaround documented at `src/experiment_tracking.py:176-181`, `CLAUDE.md:134-136`).
- **How we know it worked:** Confirmed still true on the newest split too — both segments return identical n=7,162 and identical AUC at every seed (`models/runs/NOISE_BAND_2026-08_delta_addendum.md:118-123`).
- **Status:** PLANNED

### PS-14 — No leakage-safe signal captured whether a student's GPA was trending up or down

- **Stage:** 5 Student-timeline features
- **Problem:** The feature set had absolute GPA snapshots but nothing describing trajectory, and any trend feature risks looking at a future semester's GPA relative to the current row.
- **Why it mattered:** Trend is plausibly predictive of near-term risk, but only if computed strictly from information available before the target semester.
- **Evidence of the problem:** `docs/plans/2026-07-21_gpa_trend_feature_plan.md` (pre-registered design).
- **Solution:** `gpa_trend_delta` = delta of the last two **valid prior-semester** GPAs, with an explicit `gpa_trend_missing` indicator; reconstructed and audited by a dedicated script.
- **Implemented in:** `src/feature_engineering.py` (feature computation); audited by `scripts/audit_gpa_trend.py` (`SUPPORT`, imported at `scripts/build_gpa_trend_dataset.py:38-43`)
- **How we know it worked:** Promoted to live 39→41 features (commit `13f5cc1`); isolated single-run VALID AUC moved 0.80920→0.80919 while M2 MAE improved 9.5956→9.5667 (`docs/manifests/models_runs_index_2026-08.md:121-124`), and the feature survived every later multiseed contract experiment as part of `baseline_41`.
- **Status:** LIVE

### PS-15 — Plan-generation semester limits were unnamed magic numbers

- **Stage:** 5 Student-timeline features
- **Problem:** Constraining a generated course plan to a realistic credit/course load needs named, single-sourced limits, not inline literals scattered across the recommendation layer.
- **Why it mattered:** An unnamed limit invites drift between the value used to prune candidate plans and the value used to score workload.
- **Evidence of the problem:** N/A — architectural necessity, not a discovered bug.
- **Solution:** `MAX_ALLOWED_SEMESTER_CREDITS` and `MAX_ALLOWED_SEMESTER_COURSES` defined once and imported everywhere they're enforced.
- **Implemented in:** `src/feature_engineering.py:18-19`
- **How we know it worked:** Both constants are imported and used identically in `src/recommendation.py:46-49,78,117,134,158,189` — single source, no local redefinition found.
- **Status:** LIVE
- **See also:** PS-46 for the actual value and a brief-contradiction flag.

## Stage 6 — Course difficulty

### PS-16 — Course-difficulty coverage decays sharply across the split boundary

- **Stage:** 6 Course difficulty
- **Problem:** On the (then-current) split, Level-1 difficulty coverage fell 93.6% TRAIN → 76.2% VALID → 44.7% TEST — a cliff, not a gradual decline (9.28 points at the TRAIN/VALID boundary alone, another 12.04 from VALID 20223→20231).
- **Why it mattered:** Uncovered rows score measurably worse (M1: −0.052 AUC, +0.034 Brier; M2: +2.14 MAE) — a fifth of VALID student-semesters contain at least one uncovered course, so this isn't a tail-case footnote.
- **Evidence of the problem:** `Decisions_Log.md:461-514` (2026-07-28 diagnostic), `models/runs/DIFFICULTY_COVERAGE_DIAGNOSTIC.md`.
- **Solution:** Diagnosis only at the time; the actual fix is the temporal-boundary redesign (PS-25/PS-26) undertaken specifically to shrink this gap.
- **Implemented in:** `scripts/difficulty_coverage_diagnostic.py` (diagnostic, not a fix)
- **How we know it worked:** The redesign's measured effect: Level-1 coverage 93.5% TRAIN → **95.5% VALID** → **83.1% TEST** on `2026-08_temporal_rebuild_v1`, against the old 93.6/76.2/44.7 (`Decisions_Log.md:1697-1705`, "This is the effect the rebuild was undertaken to obtain").
- **Status:** LIVE

### PS-17 — A single flat difficulty statistic can't serve both well-observed and sparse/new courses

- **Stage:** 6 Course difficulty
- **Problem:** Course difficulty (pass rate, average mark) needs enough historical rows per course to be trustworthy; many courses have thin or zero TRAIN history.
- **Why it mattered:** Falling back to a global average for a thin-history course throws away real signal; using the thin course's own noisy statistic overfits.
- **Evidence of the problem:** `[[project_course_difficulty_6level]]` (project memory) — fully implemented 2026-06-25.
- **Solution:** Six-level hierarchical fallback chain, fit train-only with leave-one-out on TRAIN, later rebuilt as a persisted, versioned, temporal state.
- **Implemented in:** `src/course_difficulty.py:447-556` (`fit_difficulty_state`), state I/O at `:792-829` per `docs/manifests/codebase_map_2026-08.md:100`
- **How we know it worked:** `assert no leackage in course diffculty` (commit `decf675`); reused unchanged as the difficulty backbone of every subsequent contract and the 2026-08 rebuild.
- **Status:** LIVE

### PS-18a — In-place difficulty rewrite of shared split files was an ownership hazard

- **Stage:** 6 Course difficulty
- **Problem:** The legacy `02_course_difficulty.ipynb` read the three shared split paths and **saved its enriched frames back onto the same three paths** — an in-place rewrite of another stage's artifact (governance conflicts C2–C4).
- **Why it mattered:** A stale reader of the bare split name would silently get unenriched data after any rebuild; two different notebooks were both writing to one shared path.
- **Evidence of the problem:** `docs/data_governance_plan.md:108` (item 8 of the 2026-07-07 verification table).
- **Solution:** D2 — distinct split generations (base → difficulty → final), each with exactly one owner; the legacy notebook was then hard-guarded: `raise RuntimeError("WRITE DISABLED: this legacy notebook was superseded by scripts/build_b2_temporal_course_stats.py...")` before its write loop.
- **Implemented in:** `scripts/build_b2_temporal_course_stats.py` (`main()` per `docs/manifests/codebase_map_2026-08.md:56`, lines 905-940); guard entered commit `decf675` per `docs/map/05-off-route.md:108`
- **How we know it worked:** All ten 2026-08-06 controlled runs resolve `05_dataset` paths built through this route, never the legacy one (`docs/map/05-off-route.md:53`).
- **Status:** SUPERSEDED (superseded by `scripts/build_b2_temporal_course_stats.py`)

### PS-18b — Guarding the legacy notebook broke the maintained default training route as a side effect

- **Stage:** 6 Course difficulty
- **Problem:** `02_course_difficulty.ipynb` now raises before writing `df_*_difficulty.parquet`, but `03_diploma_type_bucketing.ipynb` still reads those three files to produce `df_*_final.parquet` — which is exactly what `01_train_lightgbm.ipynb` trains on by default. `df_*_difficulty.parquet` has no current producer.
- **Why it mattered:** The notebook's default entrance to training is broken end-to-end; only stale on-disk residue from before the guard makes it look like it might still work.
- **Evidence of the problem:** `docs/map/05-off-route.md:22-51` (diagram + quoted `RuntimeError` text).
- **Solution:** None. "Recorded here as an observation only. No fix was made and none is proposed" (`docs/map/05-off-route.md:55`).
- **Implemented in:** NOT IMPLEMENTED.
- **How we know it worked:** N/A — not fixed. Confirmed the controlled (2026-08-06) runs are unaffected because they pass explicit `05_dataset` paths and never resolve these defaults (`docs/map/05-off-route.md:53`).
- **Status:** PLANNED

## Stage 7 — Concurrent peer features

### PS-19 — The completed-outcome model target invisibly excludes withdrawn students from any peer group built from it

- **Stage:** 7 Concurrent peer features
- **Problem:** A "who else was taking this course with me" feature needs the true registered class, but the model target contains only completed, post-filter course occurrences — a withdrawn or otherwise unfinished registration is absent from the target while still having been part of the student's actual registered load.
- **Why it mattered:** Building "concurrent peer difficulty" from the target population would silently undercount every course's peer group by however many students withdrew from it — the exact bias the feature is trying to avoid, required entry #1 of this task.
- **Evidence of the problem:** `src/registration_roster.py:1-6` (module docstring, stated as the module's own reason to exist).
- **Solution:** Reconstruct registration-time membership from the **raw** CRG student-course table, filtering only on the two registration-time eligibility rules (`active == "A"`, `register_status in {"R", "E"}`) — `finish_status`/outcome columns are deliberately never used as a filter and never copied into the roster.
- **Implemented in:** `src/registration_roster.py:1-28,43-44` (rules), `:966-976` (filter applied)
- **How we know it worked:** Rebuild match rate 99.220% TRAIN / 100% VALID / 100% TEST against raw CRG, with rosters carrying **more** rows than the completed-target population (TRAIN +37,810, VALID +5,042, TEST +2,632) — direct, quantified proof the roster captures registrations the target misses (`Decisions_Log.md:1687-1693`).
- **Status:** LIVE

### PS-20 — A near-zero-usage feature inflated the concurrent contract for no measurable benefit

- **Stage:** 7 Concurrent peer features
- **Problem:** `concurrent_peer_difficulty_missing` was used by any model in only 2 of 5 seeds, contributing under 0.001% of total gain in both cases, zero splits everywhere else.
- **Why it mattered:** A dead feature in a locked contract is pure noise risk with zero offsetting benefit — worth removing before the contract is used for real decisions.
- **Evidence of the problem:** `Decisions_Log.md:39-43` (2026-07-27 multiseed entry).
- **Solution:** Defined `concurrent_43` = `concurrent_44` minus `concurrent_peer_difficulty_missing`, order of the remaining 43 features preserved exactly.
- **Implemented in:** `src/model_training.py:183-205` (`CONCURRENT_43_FEATURES` + its asserted exclusion)
- **How we know it worked:** `scripts/verify_concurrent_43_vs_concurrent_44.py` independently verifies the exclusion; Phase 3 rebuild report confirms `concurrent_peer_difficulty_missing` "verified absent from all three datasets as a drop, not a never-built" (`Decisions_Log.md:1653-1654`).
- **Status:** LIVE

### PS-21 — The dataset builder's contract gate was pinned to a deprecated, ambiguous global

- **Stage:** 7 Concurrent peer features
- **Problem:** `scripts/build_concurrent_group_features.py`'s position gate checked `legacy in MODEL_FEATURES and MODEL_FEATURES.index(legacy) == 35`, where `MODEL_FEATURES` is a deprecated alias for "whichever feature list a maintainer last pointed it at" — in practice always `CONCURRENT_44_FEATURES`, but not by name.
- **Why it mattered:** The moment anyone repointed the deprecated global (plausible now that `concurrent_43` is a real, legitimately-shorter contract), the gate would silently break or no-op instead of failing loudly.
- **Evidence of the problem:** `Decisions_Log.md:141-159` (2026-07-27 entry, "Position-gate re-base").
- **Solution:** Re-based `_assert_contract` to check the same position against the explicit, named `CONCURRENT_44_FEATURES` list directly, removing the dependency on the ambiguous global.
- **Implemented in:** `scripts/build_concurrent_group_features.py:106,273-297` (`EXPECTED_LEGACY_MODEL_POSITION`, `_assert_contract`)
- **How we know it worked:** "Same gate values today (44 features, legacy indicator at index 35, all seven gates pass)" — verified unchanged behaviourally while closing the future-breakage risk (`Decisions_Log.md:151-152`).
- **Status:** LIVE

## Stage 8 — Course identity / lineage (Phase 1 & 2)

### PS-22 — 182 VALID-only course IDs have zero TRAIN history: renumbering, or genuinely new curriculum?

- **Stage:** 8 Course identity / lineage
- **Problem:** 182 distinct `course_id`s appear only in VALID, carrying 25,627 of VALID's 26,882 uncovered rows (95.3%) — is this systematic renumbering (fixable by a mapping) or a real curriculum change (not fixable by any mapping)?
- **Why it mattered:** If fixable, it would recover up to 13,686–18,060 uncovered VALID rows' worth of difficulty signal; if not, chasing a mapping wastes registrar review effort on courses that are genuinely new.
- **Evidence of the problem:** `Decisions_Log.md:516-596` (investigation), `:600-682` (2025-history reconciliation), `:683-724` (stricter official-evidence rule).
- **Solution:** Multi-pass investigation (identifier-structure check, disappearance-vs-appearance volume analysis, degree-lineage + structural-attribute classification, extended 2025 history, then reclassification under an official-evidence-only rule) — never a mapping applied.
- **Implemented in:** `scripts/course_identity_investigation.py`, `scripts/course_identity_reconciliation_2025.py`, `scripts/course_identity_diagnostic.py`, `scripts/course_identity_67_degree_verification.py` (all `HISTORICAL`)
- **How we know it worked:** Under the strictest (official-evidence-only) rule the result is 0 confirmed, 67 `likely_renumbered_needs_review` (13,686 rows), 11 genuinely new, 104 unresolved; verified at exact `degree_id` grain, 0/67 share a normalized catalog degree (`Decisions_Log.md:776-783`). Freeze gate `MODEL_FREEZE_BLOCKED_BY_COURSE_IDENTITY` remains open pending registrar review.
- **Status:** PLANNED

### PS-23 — Does importing a proposed predecessor course's difficulty actually help the uncovered rows it targets?

- **Stage:** 8 Course identity / lineage
- **Problem:** Before spending registrar-review effort on the 182-course lineage question, does the mechanism it would unlock (borrowing a predecessor course's difficulty prior) even help?
- **Why it mattered:** If harmful, authorizing Phase 2 human review would be spending real effort on a mechanism that makes both models worse on exactly the rows it touches.
- **Evidence of the problem:** N/A — this was tested directly, not discovered as a defect.
- **Solution:** A pilot applied the predecessor prior to 16,269 of the then-25,627 affected rows and paired-evaluated it against frozen models across 5 seeds.
- **Implemented in:** `scripts/phase3_predecessor_prior_pilot_build.py`, `scripts/phase3_predecessor_prior_pilot_evaluate.py`, `scripts/phase3_predecessor_prior_pilot_report.py`
- **How we know it worked:** It didn't. M1 AUC fell in 5/5 seeds (mean −0.00383, 4/5 beyond the noise band); M2 MAE rose in 5/5 seeds at ~4× the band's worst noise excursion. Model-free check: actual pass rate on the affected rows is 0.8474; the existing Level-4/5 fallback prior already estimates 0.8485 (near-exact); the predecessor prior moves the estimate to 0.8164, away from the truth (`Decisions_Log.md:1199-1219`).
- **Status:** ABANDONED
- **Abandoned because:** Verdict `MIXED` (3/6 clauses passed, but both *improvement* clauses failed with the sign inverted in 5/5 seeds) — `Decisions_Log.md:1113-1152`. The pilot's own acceptance clauses were also flagged as not properly pre-registered (committed in the same commit as the results), so the result is retained as evidence at reduced weight, not as a confirmed experimental outcome (`Decisions_Log.md:1242-1267`).

### PS-24 — Phase 2 lineage remapping cleared its statistical gate but was still not authorized

- **Stage:** 8 Course identity / lineage
- **Problem:** The pre-registered materiality gate (`affected_rows >= max(1000, 1% of eligible VALID)`) returned `PROCEED` on the rebuilt split — 1,203 affected rows against a 1,000 floor, a 203-row margin.
- **Why it mattered:** A `PROCEED` result only authorizes candidate generation and human review, not application — and PS-23's pilot had already shown the underlying mechanism was harmful on 87% of the *old, larger* affected population.
- **Evidence of the problem:** `Decisions_Log.md:1450-1454` (gate result), `1458-1465` (owner's stated reasoning).
- **Solution:** Owner decision to not authorize Phase 2, overriding the gate's `PROCEED`, citing PS-23's pilot evidence directly rather than re-litigating it.
- **Implemented in:** Decision recorded at `Decisions_Log.md:1470-1471` (`phase_2_decision = NOT_AUTHORISED_BY_OWNER_DESPITE_PROCEED`), reaffirmed at `:1560-1568` (Amendment 3) after the gate figures were corrected (1,034→1,203 rows)
- **How we know it worked:** Phase 3 of the rebuild proceeded from the Phase 1 split using original `degree_id`/`course_id`, `lineage_applied = false`, verified unchanged on read-back (`Decisions_Log.md:1664-1669`, `docs/map/03-rebuild-2026-08.md:114`).
- **Status:** ABANDONED
- **Abandoned because:** Owner decision, not the gate — see evidence above.

## Stage 9 — Temporal splitting TRAIN/VALID/TEST

### PS-25 — The old TEST partition was the only place 2024 data could live, but the project needed VALID data through 2024

- **Stage:** 9 Temporal split
- **Problem:** Under the old split (`TRAIN` through 20213 / `VALID` 2022-2023 / `TEST` 2024+2025S1), academic year 2024 could only be observed as TEST — but the difficulty-coverage decay (PS-16) needed a boundary fix that required 2024 in VALID.
- **Why it mattered:** This consumes the existing TEST holdout entirely — a deliberate, one-way, logged decision, not an accident.
- **Evidence of the problem:** `Decisions_Log.md:807-825` (Declaration 1) — the decision is the evidence; it states its own rationale.
- **Solution:** Single-split-family redesign: `TRAIN` through 20233, `VALID` = all of academic year 2024 (`20241+20242+20243`, corrected in Amendment 2 to include the previously-missed third semester), `TEST` = academic year 2025, provisional (`20251` only; `20252` found present but excluded as incomplete — 29% completion, abnormal semester ratio).
- **Implemented in:** `scripts/rebuild_2026_08_phase1_split.py`
- **How we know it worked:** Split reconciliation confirmed exactly: 606,562 + 75,380 + 34,628 + 11,282(excluded) = 727,852 (`Decisions_Log.md:1558`).
- **Status:** LIVE

### PS-26 — Did the boundary change actually fix the coverage decay it was undertaken for?

- **Stage:** 9 Temporal split
- **Problem:** PS-25's redesign was motivated by PS-16's coverage decay; that motivation needed to be checked against the rebuilt split, not assumed.
- **Why it mattered:** A boundary change is expensive (consumes the TEST holdout, invalidates every prior seeded experiment) and needed to actually deliver the coverage improvement it was justified by.
- **Evidence of the problem:** `CLAUDE.md:163-165` (the decay figures the rebuild targeted).
- **Solution:** Measure Level-1 coverage on the new split the same way it was measured on the old one.
- **Implemented in:** `scripts/rebuild_2026_08_phase3_assemble.py` (downstream measurement point) via `03_features/*_difficulty_candidate.parquet`
- **How we know it worked:** Level-1 coverage 93.5% TRAIN → 95.5% VALID → 83.1% TEST, against the old 93.6/76.2/44.7 (`Decisions_Log.md:1697-1705`, "This is the effect the rebuild was undertaken to obtain").
- **Status:** LIVE

### PS-27 — One degree mapped to two faculties once 2022+ rows entered TRAIN, and the difficulty fit crashed

- **Stage:** 9 Temporal split
- **Problem:** `fit_difficulty_state(TRAIN)` assumed each `degree_id` maps to exactly one `faculty_id`. Under the new boundary, four degrees map to two faculties each (the reassignment rows are all 2022+, which the old TRAIN excluded).
- **Why it mattered:** Without a fix, `difficulty_fallback_level`, `course_low_support`, `course_history_count`, and `difficulty_group_support_count` could not be computed at all on the new split — a hard blocker, not a quality issue.
- **Evidence of the problem:** `Decisions_Log.md:1490-1499` (exact error message quoted, degree IDs named).
- **Solution:** Resolve the degree→faculty ambiguity by modal frequency within the Level-3 parent.
- **Implemented in:** commit `8fda96b` ("Fix: resolve degree->faculty ambiguity by modal frequency in the Level-3 parent")
- **How we know it worked:** The Phase 3 feature reconstruction ran to completion afterward with 75 structural + 12 temporal checks passing and a full byte-identical scratch-root re-run (`Decisions_Log.md:1707-1713`).
- **Status:** LIVE

### PS-28 — Streaming batch helpers assumed a stored pandas index that Phase-1 split candidates don't have

- **Stage:** 9 Temporal split
- **Problem:** The difficulty and GPA-trend streaming wrappers assumed the source parquet stores a pandas index. Live legacy splits do; the new Phase 1 rebuild candidates don't — every batch after the first returned a `RangeIndex` restarting at zero, and the guard correctly raised rather than silently misaligning rows.
- **Why it mattered:** The unsafe alternative (`preserve_index=True`) would have written an index counter restarting every 25,000 rows — a silent, catastrophic misalignment bug if the guard hadn't caught it first.
- **Evidence of the problem:** `Decisions_Log.md:1717-1721` (root cause stated exactly).
- **Solution:** Keep the exact index-equality check when an index is present; otherwise assert the batch is positionally untouched before aligning by position.
- **Implemented in:** the difficulty and trend streaming wrappers inside `scripts/build_b2_temporal_course_stats.py` / `scripts/build_gpa_trend_dataset.py` (exact line range not independently re-read this pass)
- **How we know it worked:** "Legacy re-verified after the fix: still byte-identical to `2026-07-21_gpa_trend_feature`" (`Decisions_Log.md:1725-1726`).
- **Status:** LIVE

### PS-29 — The diploma bucket map's fitted state pointed at a superseded split

- **Stage:** 9 Temporal split
- **Problem:** `diploma_type_bucket_map.json` records `fit_source = df_train_difficulty.parquet` from the pre-rebuild split. Reusing it on the new split would carry a TRAIN-fitted encoding from a population the rebuild no longer uses — violating contract 14/15 in spirit even though the file itself is unchanged.
- **Why it mattered:** A "frozen fitted state" contract is only as good as knowing which TRAIN it was frozen against; silently reusing a stale one is exactly the failure mode contracts 14-15 exist to prevent.
- **Evidence of the problem:** `Decisions_Log.md:1572-1574` (Amendment 3).
- **Solution:** Refit the map on the new TRAIN (606,562 rows through `20233`) and persist it version-locally rather than reusing or overwriting the live one.
- **Implemented in:** `scripts/rebuild_2026_08_fit_diploma_bucket_map.py`; fitting logic in `src/diploma_bucketing.py:101-148` (`fit_diploma_bucket_map`)
- **How we know it worked:** Fitting rule verified unchanged (`top_codes=[15,16,13,19,26]`, `rare_bucket_label=6`, `unseen_bucket_label=-1`); output persisted at `data/model_data/versions/2026-08_temporal_rebuild_v1/diploma_type_bucket_map.json`, live map at `data/artifacts/` left untouched (`Decisions_Log.md:1596-1618`).
- **Status:** LIVE

### PS-30 — No path resolver existed for version-scoped rebuild artifacts

- **Stage:** 9 Temporal split
- **Problem:** The live `model_split_path()` system resolves exactly one current generation; the rebuild needed to address multiple named split generations inside a versioned, non-live directory tree without duplicating datasets (`CLAUDE.md:101-103` forbids copying datasets into new folders).
- **Why it mattered:** Without a dedicated resolver, rebuild scripts would either hardcode version paths (fragile, unauditable) or risk colliding with the live path system.
- **Evidence of the problem:** N/A — architectural gap, addressed proactively rather than discovered as a failure.
- **Solution:** `src/rebuild_paths.py` resolves and containment-checks every named split generation under a version root.
- **Implemented in:** `src/rebuild_paths.py:108-235` (path resolvers, per `docs/manifests/codebase_map_2026-08.md:113`)
- **How we know it worked:** Imported and used by every live rebuild script, e.g. `scripts/rebuild_2026_08_phase3_assemble.py:64-69`.
- **Status:** LIVE

### PS-31 — No script existed for the final split generation or final dataset assembly

- **Stage:** 9 Temporal split
- **Problem:** The "final" generation (difficulty + diploma bucket) and the final model-facing assembly step each had no committed script — only a notebook that writes to live (non-versioned) paths.
- **Why it mattered:** The concurrent-features builder requires a final-generation input to run against; without a script, the versioned rebuild chain could not run at all.
- **Evidence of the problem:** `Decisions_Log.md:1728-1736` ("This confirms the earlier architecture audit's finding of no single current final-assembly orchestrator").
- **Solution:** Two new scripts written, scoped to the version root.
- **Implemented in:** `scripts/rebuild_2026_08_phase3_diploma_bucket_apply.py`, `scripts/rebuild_2026_08_phase3_assemble.py`
- **How we know it worked:** Phase 3 report on disk: TRAIN 606,562 / VALID 75,380 / TEST 34,628 rows, 85-86 columns, `feature_manifest.csv` 87 rows, `test_outcomes_read=false`, `model_trained=false` (`docs/map/03-rebuild-2026-08.md:113-115`).
- **Status:** LIVE

## Stage 10 — Model training M1/M2 + contracts

### PS-32 — Calibrated probabilities must survive a class-imbalanced pass/fail target

- **Stage:** 10 Model training
- **Problem:** M1 is a pass/fail classifier on an imbalanced target; the standard remedies (`scale_pos_weight`, SMOTE, fail-class sample weighting) all distort probability calibration.
- **Why it mattered:** M2's predictions feed deterministic AGPA math, and M1's probabilities feed a decision layer that ranks course sets — required entry #2. Distorted calibration would corrupt every downstream ranking, not just accuracy.
- **Evidence of the problem:** N/A — this is a locked design rule, not a discovered failure; stated in `CLAUDE.md:24-26`.
- **Solution:** Never apply `scale_pos_weight`, SMOTE, or fail-class weighting; confirmed absent by direct grep of `src/` and `scripts/` (only two comment-level mentions, both stating the ban).
- **Implemented in:** `src/model_training.py:11` (module docstring: "No sample weights. No scale_pos_weight. No XGBoost."), `:701` (`train_pass_model`, "NO scale_pos_weight in V1")
- **How we know it worked:** Zero occurrences of `scale_pos_weight=`/`SMOTE` as actual parameters anywhere in `src/` or `scripts/` (repo-wide grep, this task).
- **Status:** LIVE

### PS-33 — Two models needed potentially different feature sets, and an old contract needed retiring without deleting its evidence

- **Stage:** 10 Model training
- **Problem:** After a 5-seed comparison, M1 and M2 landed on different feature sets, and one contract (`concurrent_44`) needed to be retired from future use while its historical evidence stayed intact.
- **Why it mattered:** Required entry #3. Silent reuse of `concurrent_44` in any new run would compare against a contract the project explicitly rejected.
- **Evidence of the problem:** `Decisions_Log.md:11-117` (baseline_41 vs concurrent_44), `:204-247` (correction: concurrent_44 archived, not "pending").
- **Solution:** `baseline_41` (41 features) = M1's contract; `concurrent_43` (43 features) = M2's contract; `concurrent_44` archived — kept in code (named list, not deleted) but excluded from `FEATURE_CONTRACTS` selection.
- **Implemented in:** `src/model_training.py:92-205` (`CONCURRENT_44_FEATURES`, `BASELINE_41_FEATURES`, `CONCURRENT_43_FEATURES`, with asserted set relationships)
- **How we know it worked:** `resolve_feature_contract("concurrent_44")` — need to confirm it's still resolvable (archived ≠ deleted, per `Decisions_Log.md:241-244`) but persistent runs are test-enforced to name a contract explicitly (`tests/test_feature_contracts.py:252`, `test_11_persistent_run_requires_an_explicit_feature_contract`).
- **Status:** LIVE

### PS-34 — Runs evaluated at different thresholds aren't precision/recall/F1-comparable

- **Stage:** 10 Model training
- **Problem:** Early runs (`0.5`, `0.85` history per CLAUDE.md) used different reporting cuts; without a fixed cut, no two runs' fail-precision/recall/F1 can be honestly compared.
- **Why it mattered:** Required entry #4. A varying threshold would make every cross-run comparison in this project's history unreliable on exactly the metrics that matter for a fail-catching classifier.
- **Evidence of the problem:** `CLAUDE.md:120-123` (explicit statement that other-threshold runs aren't comparable).
- **Solution:** `REPORTING_THRESHOLD = 0.80`, fixed, explicitly not the eventual product threshold (that's a post-freeze business decision).
- **Implemented in:** `src/model_training.py:244` (constant), applied at `:777,906,1290` and recorded in every run's `metrics.json`
- **How we know it worked:** Present as `reporting_threshold: 0.8` in all ten `2026-08-06` runs' `run_settings` (`models/runs/NOISE_BAND_2026-08_temporal_rebuild_v1.md:40`).
- **Status:** LIVE

### PS-35 — The code default still points at an archived contract, and nothing stopped a persistent run from silently inheriting it

- **Stage:** 10 Model training
- **Problem:** `DEFAULT_FEATURE_CONTRACT` remained `concurrent_44` (archived, per PS-33) after the archival decision — a stale default that, unguarded, could let a persistent run silently train against a rejected contract.
- **Why it mattered:** Repointing the default is deferred wiring work, but a persistent run silently defaulting to it would be a real correctness bug, not just staleness.
- **Evidence of the problem:** `Decisions_Log.md:236-240` (the correction entry itself names this as a known, accepted gap).
- **Solution:** Persistent runs (`--run-name`) are test-enforced to pass `--feature-contract` explicitly; only quick/throwaway runs may silently use the stale default.
- **Implemented in:** `src/model_training.py:353` (`DEFAULT_FEATURE_CONTRACT = CONCURRENT_44_CONTRACT.name`), enforcement at `:1283` (CLI arg resolution) and `:1205`
- **How we know it worked:** `tests/test_feature_contracts.py:252` (`test_11_persistent_run_requires_an_explicit_feature_contract`).
- **Status:** LIVE
- **Follow-up:** Repointing the default itself is explicitly deferred until after the M1/M2 freeze (`CLAUDE.md:83-87`) — tracked here as still open, not re-opened as a new problem.

### PS-36 — Can stronger regularization shrink M1's train-valid AUC gap without paying too much VALID performance?

- **Stage:** 10 Model training
- **Problem:** M1 showed a persistent train-valid AUC gap; the open question was whether a single hyperparameter lever could shrink it within an acceptance rule locked before any run existed.
- **Why it mattered:** This decision determines whether M1 and M2 could ever unify on one feature contract (`concurrent_43`) or must stay split (`baseline_41`/`concurrent_43`).
- **Evidence of the problem:** `docs/EXPERIMENT_REGULARIZATION_PLAN.md` (pre-registered before any run, per `Decisions_Log.md:251-274`).
- **Solution:** Seed-42 screening of four single-lever configs (`num_leaves`, `min_child_samples`, `reg_lambda`) × two arms; the one PASS (`num_leaves` 127→31, "R2") was then confirmed across seeds 52/62/72/82.
- **Implemented in:** `src/model_training.py:612` (`effective_lgbm_params`) + `--num-leaves`/`--min-child-samples`/`--reg-lambda` CLI flags (commit `a6ec653`)
- **How we know it worked:** R2 **CONFIRMED** for `baseline_41` (gap improved 5/5 seeds, no VALID guardrail breached) but **NOT CONFIRMED** for `concurrent_43` (VALID AUC and Brier both outside the band on the harmful side, 2 seeds each beyond twice the harmful edge) — `Decisions_Log.md:374-419`. The seed-42 "mechanism" story (generalization gain vs. train collapse) did not replicate across seeds either.
- **Status:** PLANNED
- **Open decision (not made by any session):** whether to adopt R2 for `baseline_41` alone, given it costs M2 a small consistent amount and the gap shrink is "only sometimes a generalization gain" (`Decisions_Log.md:452-459`, explicitly deferred to the human).

### PS-37 — Two active contracts diverge per model, and the rebuild dataset must serve both without choosing

- **Stage:** 10 Model training
- **Problem:** `baseline_41` and `concurrent_43` are different column sets; a single rebuilt dataset needs to support training either model's contract without duplicating data or picking a winner.
- **Why it mattered:** Required entry #3 continuation — the rebuild is not itself a contract decision and must not look like one.
- **Evidence of the problem:** `Decisions_Log.md:931-935` (Declaration 3: "The rebuild therefore produces one candidate dataset per split containing the union of both contracts... does not choose between the contracts").
- **Solution:** Assemble the union of both contracts (43 columns) plus a per-column manifest stating which contract(s) consume each column.
- **Implemented in:** `scripts/rebuild_2026_08_phase3_assemble.py`
- **How we know it worked:** "The union of the two active contracts is `concurrent_43`; `baseline_41` was verified — not assumed — to be a strict subset" (`Decisions_Log.md:1649-1650`); `feature_manifest.csv` records all 87 columns with contract membership.
- **Status:** LIVE

## Stage 11 — Evaluation, seeds, NOISE_BAND

### PS-38 — A single training seed is not evidence of a stable effect

- **Stage:** 11 Evaluation, seeds, NOISE_BAND
- **Problem:** Early feature/contract comparisons ran at one seed; a metric delta at one seed can't distinguish a real effect from training-noise.
- **Why it mattered:** The project's own retraction (see PS-40) shows this happened for real — a cold-start improvement cited from a single-seed experiment did not hold at 5 seeds.
- **Evidence of the problem:** `docs/pipeline_rules.md:174-178` (locked rule, citing the retraction as its own justification).
- **Solution:** Canonical 5-seed protocol (42, 52, 62, 72, 82); `--seed` deterministically derives LightGBM's sub-seeds; both arms of any paired comparison must share one seed.
- **Implemented in:** `src/model_training.py` (seed CLI, commit `81e00f6`), `_effective_seed_settings` at `:661-678`
- **How we know it worked:** Used consistently across every multiseed comparison from `baseline_41 vs concurrent_44` onward (`Decisions_Log.md:11-450`), and the retracted cold-start claim is explicitly logged as the counter-example that motivated the rule (`CLAUDE.md:79-82`).
- **Status:** LIVE

### PS-39 — The committed noise band is measured on a dataset version the newest work no longer uses

- **Stage:** 11 Evaluation, seeds, NOISE_BAND
- **Problem:** `models/runs/NOISE_BAND.md` was measured from `baseline_41` vs `concurrent_44` paired deltas on `2026-07-26_batched_fixes__registration_roster_concurrent` (156,097-row VALID). The newest work runs on `2026-08_temporal_rebuild_v1` (75,380-row VALID, different composition, different contract pairing).
- **Why it mattered:** Required entry #9. The band is explicitly a function of VALID size and composition; reading the old band against the new split's deltas would compare two different measurements as if they were one.
- **Evidence of the problem:** `Decisions_Log.md:963-966` (Declaration 4: "`NOISE_BAND.md` in particular is invalid as a yardstick for the new split... A new band must be measured before any delta on the new split is interpreted") — this is the repo **stating its own contradiction** with any assumption that one band covers both.
- **Solution:** A second, independent noise band measured directly on `2026-08_temporal_rebuild_v1` — as raw per-seed VALID values first (single locked arm per model, no second contract to pair against), then a post-hoc paired-delta addendum once the pairing was recognized as available from the same ten runs.
- **Implemented in:** `models/runs/NOISE_BAND_2026-08_temporal_rebuild_v1.md`, `models/runs/NOISE_BAND_2026-08_delta_addendum.md`
- **How we know it worked:** Ten runs (5 seeds × 2 locked arms) with `dataset_version=2026-08_temporal_rebuild_v1` verified identically in every run's `run_settings`; the addendum explicitly self-flags as post-hoc, not pre-registered (`models/runs/NOISE_BAND_2026-08_delta_addendum.md:5-11`) — an honesty disclosure, not a defect that was hidden.
- **Status:** LIVE (for `2026-08_temporal_rebuild_v1`) / SUPERSEDED (the old `NOISE_BAND.md`, for that purpose only — it remains valid for the `2026-07-26...` split it was measured on)
- **CONTRADICTS BRIEF:** the task brief's required-entry #9 phrasing ("the old NOISE_BAND.md is invalid on the current split") is confirmed true, but "the current split" is itself ambiguous in this repo — see PS-52. The old band is not invalid in general, only for the newer split.

### PS-40 — Acceptance rules decided after seeing results can be quietly rewritten to fit them

- **Stage:** 11 Evaluation, seeds, NOISE_BAND
- **Problem:** An acceptance rule is only meaningful evidence if it demonstrably predates the numbers it judges.
- **Why it mattered:** The project has one clean example of getting this right and one clear example of getting it wrong, in its own history — worth recording both.
- **Evidence of the problem:** `Decisions_Log.md:1242-1267` (the pilot's clauses were introduced in the same commit as its results — "the same commit that contains the results").
- **Solution:** Commit the acceptance rule as its own file, before training any run that will be judged by it, so ordering is verifiable from git history instead of asserted in prose.
- **Implemented in:** `docs/EXPERIMENT_REGULARIZATION_PLAN.md`, `docs/EXPERIMENT_R2_CONFIRMATION_PLAN.md`, `docs/EXPERIMENT_R2_COVERAGE_DECISION_PLAN.md` — each committed before its runs, per `Decisions_Log.md:251-263,384-385,727-728`
- **How we know it worked:** Git history itself is the proof — commit `6fa053e` ("Pre-register R2 coverage decision rule") predates `0ce2160` ("Resolve M1 R2 coverage decision"). Contrast case (what NOT pre-registering looks like) is PS-23.
- **Status:** LIVE

## Stage 12 — Reproducibility & experiment tracking

### PS-41 — Leaderboard numbers could silently drift from the run artifacts they summarize

- **Stage:** 12 Reproducibility & experiment tracking
- **Problem:** A hand-maintained or loosely-appended leaderboard CSV can drift from the `metrics.json` files it's supposed to summarize, and a schema change could corrupt historical rows silently.
- **Why it mattered:** Every cross-run comparison in this project's history depends on the leaderboard being trustworthy.
- **Evidence of the problem:** N/A — addressed by design, not a discovered corruption.
- **Solution:** `append_leaderboard_row` re-reads the existing CSV header and raises `ValueError` on any mismatch before appending, so the schema cannot drift silently.
- **Implemented in:** `src/experiment_tracking.py:304-363` (`finalize_persistent_run`, `append_leaderboard_row`)
- **How we know it worked:** All 49 leaderboard rows independently recomputed against their own `metrics.json`: 49/49 match to 1e-9 on all six metric columns, every row has a matching run directory (`docs/manifests/models_runs_index_2026-08.md:14-24`).
- **Status:** LIVE

### PS-42 — Two runs meant to differ by exactly one hyperparameter could drift on something else unnoticed

- **Stage:** 12 Reproducibility & experiment tracking
- **Problem:** A "single lever" comparison (e.g. R1-R4 regularization screening) is only valid if every *other* setting is provably identical between the two runs — eyeballing JSON diffs doesn't scale and doesn't get re-checked later.
- **Why it mattered:** A silently-different second parameter would misattribute a metric delta to the wrong cause.
- **Evidence of the problem:** N/A — a structural risk in any paired-comparison design, addressed proactively.
- **Solution:** A shared 22-point parity checker (contract identity, ordered features, categorical levels, threshold, test policy, dataset hashes, row counts, effective seeds, full serialized LightGBM parameter blocks, round cap, early stopping, diploma-fill) run inline after training and again independently in report generation.
- **Implemented in:** `scripts/r2_parity.py` (imported by `scripts/generate_r2_confirmation_report.py:41` and `scripts/r2_coverage_rescore.py:36`)
- **How we know it worked:** "All 22 passed in all eight [screening] runs: the only serialized LightGBM parameter differing between a run and its control is that run's single lever" (`Decisions_Log.md:337-345`); repeated for all ten R2/control confirmation pairs (`Decisions_Log.md:436-442`).
- **Status:** LIVE

### PS-43 — Byte-identity is the wrong bar for validating a rerun of a data pipeline

- **Stage:** 12 Reproducibility & experiment tracking
- **Problem:** Parquet is not byte-stable across writes (compression, row-group ordering, page layout can all differ between two writes of logically identical data), so a byte-hash comparison between two pipeline runs proves nothing about whether the *data* matches.
- **Why it mattered:** Validating any rebuild or rerun (e.g. the 2026-08 temporal rebuild's own re-run-into-a-scratch-root check) needs a comparison that's actually sensitive to real data differences and insensitive to harmless write-format differences.
- **Evidence of the problem:** `docs/data_governance_plan.md:146-148` (§4, the "Logical parity" validation type definition this script implements).
- **Solution:** A general-purpose two-parquet logical-parity CLI: outer-merge on a declared natural key (row order irrelevant in either input), compares row counts, unique keys/duplicates, schema/dtypes, sort-normalized values, null patterns, distributions, with a declared float tolerance.
- **Implemented in:** `scripts/logical_parity_check.py:1-38` (module docstring + design rationale)
- **How we know it worked:** NO VERIFICATION FOUND — this file is new and **not yet committed** (`git status`: `?? scripts/logical_parity_check.py`), postdates the `codebase_map_2026-08.md` census, and has no recorded caller or run yet.
- **Status:** LIVE
- **Caveat:** "LIVE" here means the code exists and is complete, not that it has been exercised — flagged honestly rather than upgraded on the strength of its docstring alone.

## Stage 13 — Inference / student scorer

### PS-44 — Trained M1/M2 models produce predictions, but nothing turns those into per-course scores for a candidate plan on its own

- **Stage:** 13 Inference / student scorer
- **Problem:** `src/model_training.py` produces `.lgbm` model files; something has to load them plus the frozen difficulty state, build a course-level feature row for a specific student/semester/course, and score it — a distinct responsibility from training.
- **Why it mattered:** Without this bridge, a trained model is just a file; nothing downstream (KNN evidence, plan generation) can consume its predictions.
- **Evidence of the problem:** N/A — architectural necessity, not a discovered defect.
- **Solution:** `StudentScorer` loads M1/M2 plus difficulty state and exposes `extract_snapshot`/`score`/`score_plan` for the recommendation layer to call.
- **Implemented in:** `src/inference.py:126` (`class StudentScorer`), `.load` per `docs/manifests/codebase_map_2026-08.md:107` at lines 159-175
- **How we know it worked:** Exercised by `tests/test_inference_score_plan.py`, and is the sole scoring path `src/recommendation.py:254` (`Recommender.load`) composes against.
- **Status:** LIVE
- **Reachability caveat:** Complete and tested, but the repo-wide caller census finds only `src/recommendation.py` and tests — no current pipeline entry point reaches it (`docs/manifests/codebase_map_2026-08.md:107`). This is a governance fact (nothing currently invokes the recommendation stack in production), not a claim that the code is broken.

## Stage 14 — Plan generation, AGPA, KNN, ranking

### PS-45 — The GPA scale used for plan scoring has no single source of truth

- **Stage:** 14 Plan generation, AGPA, KNN, ranking
- **Problem:** `_mark_to_gpa_points` hand-codes five breakpoints (90/80/70/60/50 → 4.0/3.5/3.0/2.5/2.0/0.0) to convert a predicted `final_mark` into a GPA-point value for plan scoring.
- **Why it mattered:** Required entry #5. This is an approximation of whatever the university's actual grade scale is (sourced from `v_acs_grade` per `docs/pipeline_rules.md:194`), not the scale itself — every AGPA number the recommendation layer produces inherits this approximation's error.
- **Evidence of the problem:** `docs/pipeline_rules.md:194-195` ("`_mark_to_gpa_points` is hand-coded. Replace with a lookup from the `v_acs_grade` source so the grade scale has one authority"); `CLAUDE.md:176-177` (post-freeze direction: "real AGPA engine replacing the `_mark_to_gpa_points` approximation").
- **Solution:** None implemented — explicitly named as post-freeze, not-yet-started work.
- **Implemented in:** current approximation at `src/recommendation.py:57-71`; replacement NOT IMPLEMENTED.
- **How we know it worked:** NO VERIFICATION FOUND for a replacement — none exists yet.
- **Status:** PLANNED

### PS-46 — What is the actual plan-generation course-count cap, and does it match real registration behaviour?

- **Stage:** 14 Plan generation, AGPA, KNN, ranking
- **Problem:** Required entry #7. `CLAUDE.md §10` states plan generation "caps ~5 courses while students take 6."
- **Why it mattered:** If the code cap is materially different from what CLAUDE.md claims, any reasoning built on "~5 vs 6" is working from the wrong number.
- **Evidence of the problem:** `CLAUDE.md:180-181`.
- **Solution:** `MAX_ALLOWED_SEMESTER_COURSES = 8`, `MAX_ALLOWED_SEMESTER_CREDITS = 25` — both enforced identically wherever plans are generated or scored.
- **Implemented in:** `src/feature_engineering.py:18-19`; enforced in `src/recommendation.py:78,112,117,134,158,189`
- **How we know it worked:** Read directly from source (`src/feature_engineering.py:18-19`) and confirmed as the actual default used by `_generate_candidate_plans` (`src/recommendation.py:78`).
- **Status:** LIVE
- **CONTRADICTS BRIEF:** The code's actual cap is **8 courses / 25 credits**, not "~5." CLAUDE.md §10's "0.20 vs 0.10" clause in the same sentence *is* confirmed by code (workload_ratio weighted 0.20 vs. graduation_progress weighted 0.10 in the composite score, `src/recommendation.py:204-209` — see PS-47), but the specific "~5 courses" figure is not found anywhere in code and does not match `MAX_ALLOWED_SEMESTER_COURSES`. This map reports the verified code value; it does not resolve which framing CLAUDE.md intended.

### PS-47 — The composite plan score's documented weights don't match what the code computes

- **Stage:** 14 Plan generation, AGPA, KNN, ranking
- **Problem:** `_score_plan`'s docstring states "Weights are: AGPA=0.40, risk=0.30, knn=0.20, grad=0.10." The code applies `0.40` to AGPA, `0.30` to risk, **`0.20` to workload** (not KNN), `0.10` to graduation progress, and adds a separate **unweighted** `knn_bonus` term on top.
- **Why it mattered:** Anyone reading the docstring to understand why one plan outranked another would draw the wrong conclusion about how much KNN evidence actually influences ranking.
- **Evidence of the problem:** `docs/pipeline_rules.md:185-186` ("composite-score weights: docstring says knn=0.20 but code applies 0.20 to workload and adds knn as a separate unweighted bonus. Decide intended design, then make code and docstring agree") — confirmed still present, unchanged, by direct read of current code.
- **Solution:** None implemented.
- **Implemented in:** docstring at `src/recommendation.py:163`; code at `:204-210` (`knn_bonus` computed separately at `:198-201`)
- **How we know it worked:** NO VERIFICATION FOUND — not fixed.
- **Status:** PLANNED

### PS-48 — See PS-10: the missing pass-definition helper also reaches the recommendation layer

- **Stage:** 14 Plan generation, AGPA, KNN, ranking
- **Problem:** `src/knn_advisor.py:75` computes `sem_pass_rate=("final_mark", lambda x: (x >= 50).mean())` and `src/recommendation.py:69` uses the same `50` cutoff inside `_mark_to_gpa_points` — both independent hardcodes of the same threshold PS-10 already flags as missing a shared helper.
- **Why it mattered:** KNN-derived "similar students' pass rate" evidence and the AGPA scoring layer both silently depend on a literal that only training's `TARGET_M1_DEFINITION` currently documents as canonical.
- **Evidence of the problem:** `docs/pipeline_rules.md:64-67` (same violation named directly).
- **Solution:** None implemented — same open item as PS-10, cited separately here because it's the concrete instance living in the recommendation stage.
- **Implemented in:** NOT IMPLEMENTED.
- **How we know it worked:** NO VERIFICATION FOUND.
- **Status:** PLANNED

### PS-49 — No eligibility/prerequisite engine exists

- **Stage:** 14 Plan generation, AGPA, KNN, ranking
- **Problem:** Required entry #6. Plan generation can enumerate any combination of candidate courses under a pass-probability floor and credit/course caps — nothing checks prerequisites, degree-requirement rules, or course-offering eligibility.
- **Why it mattered:** A ranked "recommendation" that isn't actually a legal course selection for the student is not usable as-is; `CLAUDE.md §10` names this directly as empty and as not-yet-started.
- **Evidence of the problem:** `CLAUDE.md:177` ("eligibility engine (currently empty)"). Confirmed by repo-wide search: zero matches for an eligibility/prerequisite engine anywhere in `src/` or `scripts/` — the only "eligib*" hits are unrelated (registration-time *eligibility rules* in `registration_roster.py`, and *eligible row* counts in the lineage-gate governance math).
- **Solution:** None implemented.
- **Implemented in:** NOT IMPLEMENTED.
- **How we know it worked:** NO VERIFICATION FOUND.
- **Status:** PLANNED

### PS-50 — The KNN index has no confirmed producer and its architectural role is still an open decision

- **Stage:** 14 Plan generation, AGPA, KNN, ranking
- **Problem:** `knn_index.pkl` is a live-consumed artifact (`src/knn_advisor.py` builds/loads/queries it) with no reproducible committed producer script, and its role in the live architecture was put on hold rather than decided.
- **Why it mattered:** Building a producer for an artifact whose role isn't settled risks locking in an architecture decision (e.g., "KNN is a second predictor" vs. "KNN is explanation-only evidence," per `CLAUDE.md:183-184`) by accident, through tooling rather than through an explicit decision.
- **Evidence of the problem:** `docs/data_governance_plan.md:298,315,318` ("D7 — knn_index.pkl role in the live architecture — STILL OPEN. HOLD stands").
- **Solution:** None implemented — deliberately: "No 7-group may touch it before D7 is decided."
- **Implemented in:** NOT IMPLEMENTED (existing artifact/module: `src/knn_advisor.py:87-206`, build/save/load at `:105-155`)
- **How we know it worked:** N/A — this is a HOLD, not a fix; verified still HOLD as of the Phase 9 freeze (`docs/manifests/freeze_phase9_2026-07-08.md:50`, "D7 (KNN index role)... carried forward").
- **Status:** PLANNED

## Stage 15 — Explanation layer (LLM, output-only)

NO EVIDENCE FOUND IN REPO. `CLAUDE.md §10` names an "LLM layer [that] explains results only" as post-freeze direction, not current scope. A repo-wide search for LLM/GPT/large-language-model/chatbot implementation, imports, or wiring across `src/`, `scripts/`, and notebooks returned zero implementation hits — every match was either this task's own governance documents describing the *future* role, or unrelated substrings inside notebook output cells. No entry is forced for this stage; there is no problem-in-code to reconstruct yet.

## Stage 16 — Governance & freeze discipline

### PS-51 — Set-level backtesting — the metric that would actually validate a *recommended plan*, not just a per-course prediction — does not exist

- **Stage:** 16 Governance & freeze discipline
- **Problem:** Required entry #10. Every evaluation in this project (M1 AUC/Brier, M2 MAE/RMSE/R², segment AUCs, noise bands) scores individual row-level or segment-level predictions. None scores a *ranked plan* — e.g., "did the top-ranked plan this system would have generated actually outperform the alternatives, for real students who took real course sets?"
- **Why it mattered:** `CLAUDE.md §10` names this directly as "the real product metric" — everything upstream of it (M1, M2, KNN, ranking) is validated in isolation, never end-to-end as the product the user actually experiences.
- **Evidence of the problem:** `CLAUDE.md:182-183` ("set-level backtesting — the real product metric"). Confirmed by repo-wide search: zero matches for backtest/back-test/back_test anywhere in `src/` or `scripts/` — the only hit outside this map's own evidence trail is `CLAUDE.md` itself.
- **Solution:** None implemented.
- **Implemented in:** NOT IMPLEMENTED.
- **How we know it worked:** NO VERIFICATION FOUND.
- **Status:** PLANNED

### PS-52 — `CLAUDE.md`'s "current objective" section describes a workstream that has already finished, and a newer one the file never mentions

- **Stage:** 16 Governance & freeze discipline
- **Problem:** `CLAUDE.md §3` ("Current objective") describes the regularization pass as in-progress, gated at "seed-42 screening only... then STOP," dated "Last updated: 2026-07-27." Git history shows that pass completed (5-seed R2 confirmation, `Decisions_Log.md:374-459`) and was followed by an entirely new dataset rebuild (`2026-08_temporal_rebuild_v1`: Phase 0 preflight through Phase 3 assembly, a new lineage-materiality gate, and a **second** noise band) — all dated 2026-08-02 through 2026-08-06, after the file's own timestamp.
- **Why it mattered:** A task prompt that trusts `CLAUDE.md §3` at face value would re-litigate a finished decision (R2) and miss an entire dataset generation (`2026-08_temporal_rebuild_v1`) that the newest training runs actually use.
- **Evidence of the problem:** `CLAUDE.md:3,39-58` (the stale section) against `git log --oneline` commits `df03477`…`9a4a11a` (2026-08-02 to 2026-08-06) and `Decisions_Log.md:798-1757` (the entire rebuild governance trail those commits implement).
- **Solution:** None implemented — this map is itself read-only and cannot edit `CLAUDE.md`; the file's own header says a task should "follow the prompt but flag the conflict explicitly" (`CLAUDE.md:11-12`), which this entry does.
- **Implemented in:** NOT IMPLEMENTED — flagging only, per this task's own read-only mandate.
- **How we know it worked:** N/A.
- **Status:** LIVE — this describes the current, verified state of the discrepancy, not a fix.
- **CONTRADICTS BRIEF:** `CLAUDE.md` presents §3 as the sole active workstream; the repo's own commit history and `Decisions_Log.md` show it closed and superseded by unlogged-in-CLAUDE.md work. This is the same ambiguity behind this map's header "Dataset version" field.

### PS-53 — A governance log incorrectly claimed no AI assistant had authored its own wording

- **Stage:** 16 Governance & freeze discipline
- **Problem:** The committed governance-declarations entry for `2026-08_temporal_rebuild_v1` opened with "They are recorded here by the project owner. No agent authored, edited, or appended them" — which was not accurate; an AI assistant drafted the wording in an interactive planning conversation before the owner reviewed and committed it.
- **Why it mattered:** A governance log's entire value is that its claims can be trusted at face value; an inaccurate self-description undermines that even when the substantive content (the owner's decisions) was correct.
- **Evidence of the problem:** `Decisions_Log.md:988-999` (the amendment quotes the false sentence verbatim before correcting it).
- **Solution:** A same-day append-only amendment correcting the provenance claim, distinguishing clearly between "content decided by the owner" (unchanged) and "wording drafted by an assistant, then reviewed" (the correction) — and reaffirming the implementation agent had no authoring role and may only read/quote the log verbatim.
- **Implemented in:** `Decisions_Log.md:975-1021` (Amendment 1)
- **How we know it worked:** The append-only discipline is externally checkable: the original entry (`df03477c`) is still present unedited, and the correction is a separate, later commit — exactly the ordering the log's own rules require (`Decisions_Log.md:1012-1016`, the materiality-threshold ordering argument applies identically here).
- **Status:** LIVE

---

## Open problems with no solution yet

| ID | Title |
|----|-------|
| PS-01 | Extractor SQL bug misaligns `DIPLOMA_TYPE_SL`/`ACTIVE` |
| PS-10 | No single pass-definition helper (5 independent `>= 50` hardcodes) |
| PS-13 | `first_semester` / `cold_start_gpa` are the same population |
| PS-18b | Maintained notebook training route is broken by design |
| PS-22 | 182 course-identity candidates await registrar review |
| PS-36 | Regularization (R2) adoption for `baseline_41` is an open human decision |
| PS-45 | `_mark_to_gpa_points` needs a real GPA-scale authority |
| PS-47 | Composite plan-score docstring doesn't match code |
| PS-48 | Pass threshold hardcoded outside the recommendation layer too |
| PS-49 | No eligibility/prerequisite engine |
| PS-50 | KNN index producer/role decision (D7) still on HOLD |
| PS-51 | No set-level backtesting exists |
| Stage 15 | Explanation layer (LLM) — no implementation exists at all |

## Contradictions found

- **PS-39 / PS-52 (required entry #9):** the committed `models/runs/NOISE_BAND.md` is valid only for `2026-07-26_batched_fixes__registration_roster_concurrent`; the newest dataset (`2026-08_temporal_rebuild_v1`) has its own, separately-measured band. Neither the task brief nor `CLAUDE.md` names a single "current split" — the repo genuinely has three candidate answers to "what is the current dataset version," reconciled in this map's header and at PS-52.
- **PS-46 (required entry #7):** `CLAUDE.md §10` states plan generation "caps ~5 courses"; the code's actual constant is `MAX_ALLOWED_SEMESTER_COURSES = 8`. The same sentence's second clause ("ranking biases small plans, 0.20 vs 0.10") is confirmed accurate by code (PS-47).
- **PS-52:** `CLAUDE.md §3` presents the regularization pass as the sole active, in-progress workstream; git history and `Decisions_Log.md` show it finished and superseded by an entire unlogged-in-CLAUDE.md dataset rebuild.

## Coverage report

- Files under `src/` referenced by at least one entry: **19 / 19**
  (`__init__.py` — referenced only in prose, PS-44's stage intro / codebase-map citation, not as a primary "Implemented in" target, since it is a 1-line package marker with no logic of its own.)
- Files under `src/` referenced by NO entry: none.
- Entries with `NO VERIFICATION FOUND`: **8** (PS-01, PS-10, PS-43, PS-45, PS-47, PS-48, PS-49, PS-51). Three further entries (PS-08, PS-18b, PS-50) use `N/A` rather than that literal phrase because there is no code "fix" to verify by design (an accepted exception, a deliberately-unresolved break, and a governance HOLD, respectively) — their *problem* evidence is solid; only the "how do we know a fix worked" question doesn't apply. PS-13's segment-identity *defect* is unfixed (would qualify) but its governance *workaround* (count as one segment) is independently verified, so it is not counted here.
- Entries tagged `UNVERIFIED` (evidence too thin to place elsewhere): **0** — every entry in this map was placeable as LIVE/SUPERSEDED/ABANDONED/PLANNED with direct evidence; no stage required more than 3 such thin entries, so no `UNVERIFIED`-triggered stop condition was hit.
