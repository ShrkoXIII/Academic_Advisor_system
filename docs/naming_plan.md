# Naming & Numbering Plan — Phase 6 Deliverable (APPROVED)

Status: **APPROVED by user 2026-07-07**, with the open decision rows resolved the same day: D5 → outlier branch exploratory/inactive (`handeling_outliers.ipynb` → `explore_outlier_removal.ipynb` in 7b; `without_outliers.parquet` frozen exploratory); `all.ipynb` → ARCHIVE as dead prototype (not renamed into the pipeline); root pre-drop ADD artifact → ARCHIVE after a no-live-consumer check; `course_difficulty_lookup.parquet` → producer ADOPTED (lands in 7c); `knn_index.pkl` → HOLD pending D7. Produced 2026-07-07 under `docs/data_governance_plan.md` (Revision 4, corrected) and `docs/governance_contracts.md` (approved 2026-07-07; contracts 7–9 fix the principles this plan applies). Phase 6 is **planning only**: nothing here renames, moves, or edits any file. Every rename executes in its Phase 7b group — notebooks via `git mv` (+ in-file reference fixes in the same group), data artifacts via copy → repoint readers → validate (byte parity for pure rename/move, §4B logical parity where a rerun is involved) → retire preserved original, per contracts 10–11. Every proposal below was grounded by reading the actual code, not the filename.

Naming rules honored throughout (CLAUDE.md + contract 7–9): names say what the file does; one merge notebook per stage; the early merge (CRG+ADD+ACD) and the diploma merge stay separate stages; no two files whose names differ only by letter case; numbered = pipeline order, unnumbered = diagnostics/exploration.

Priorities: **P1** = safety or truth (name lies about behavior, or naming enables a silent-wrong-read); **P2** = consistency; **P3** = optional, churn-sensitive — may be deferred or rejected without affecting P1/P2.

---

## 6A — Notebooks (names + numbering)

Numbering scheme: numbers order execution **within a stage folder** (`00_`, `01_`, …); cross-folder order is fixed by the stage-boundary contract 5. Per-table cleaning notebooks are deliberately **unnumbered** (mutually independent; all run after `00_extract_raw_tables`, before the merges). Unnumbered pipeline-adjacent notebooks use verb prefixes: `explore_` (read-only inspection), `judge_`/`*_diagnostic` (validation), `trace_` (debugging).

### note_books/merge/

| Current | Proposed | Prio | Why |
|---|---|---|---|
| `01_merge_crg_add_acd.ipynb` | KEEP | — | Correct pattern (CLAUDE.md's own example). |
| `02_final_merged_with_dimploma.ipynb` | `02_merge_diploma.ipynb` | P1 | Fixes the residual "dimploma" typo; matches the plan's logical stage name (D1). "mergerd" was already fixed out-of-band in `c6069bc`. |

One merge notebook per stage confirmed: exactly two notebooks, two stages, kept separate. No dead merge prototypes remain in `merge/`.

### note_books/pre_processing/

| Current | Proposed | Prio | Why |
|---|---|---|---|
| `extact_all_row_tables.ipynb` | `00_extract_raw_tables.ipynb` | P1 | Two typos ("extact", "row"→"raw"); it is the sole raw writer and the true pipeline entry — deserves number 00. |
| `all.ipynb` | **USER DECISION** — recommended: archive as dead prototype; else `explore_cleaned_tables.ipynb` | P1 | Verified read-only (loads 4 cleaned tables, no merges, no writes). `docs/paths_audit.md` already flagged it as a dead-prototype deletion candidate. Per CLAUDE.md: move to a repo `note_books/archive/` first; permanent deletion needs separate approval. |
| `ACS_GRADE/` (folder) | `V_ACS_GRADE/` | P2 | Only table folder missing the `V_` prefix; matches the data-tree folder `V_ACS_GRADE`. |
| `ACS_GRADE/clean_ACS_grade.ipynb` | `clean_v_acs_grade.ipynb` | P2 | Case + pattern consistency; matches its output `clean_v_acs_grade.parquet`. |
| `V_ACADEMIC_INFO/` (folder) | `V_ADD_ACADEMIC_INFO/` | P2 | The source view is `RAS_USER.V_ADD_ACADEMIC_INFO`; folder name should not drop the `ADD_`. |
| `V_ACADEMIC_INFO/Academic_info_clean.ipynb` | `clean_v_add_academic_info.ipynb` | P2 | Verb-first `clean_<view>` pattern like its siblings; removes the stray capital. |
| `V_ACD_DEGREE_COURSE/load_preprocessing.ipynb` | `clean_v_acd_degree_course.ipynb` | P1 | Current name says nothing ("load_preprocessing" describes no action on no object); it cleans and writes the ACD table. |
| `V_ADD_STUDENT_DEGREE_STATUS/add_student_degree_status_clean.ipynb` | `clean_v_add_student_degree_status.ipynb` | P2 | Verb-first consistency ("add_…" reads as the verb *add*). |
| `V_CRG_STUDENT_COURSE/clean_v_crg_student_course.ipynb` | KEEP | — | The canonical pattern (CLAUDE.md example). |
| `V_CRG_STUDENT_COURSE/mark_finish_status_disagreement_diagnostic.ipynb` | KEEP | — | Descriptive, correctly unnumbered. (Its live hardcoded path is a 7a code fix, not a naming issue.) |
| `V_CRG_STD_COR_TEMP_REQ/read.ipynb` | `explore_v_crg_std_cor_temp_request.ipynb` | P1 | Verified read-only exploration of the raw table. Folder → `V_CRG_STD_COR_TEMP_REQUEST/` (P3; true view name). |
| `V_CRG_STUDENT_PASSED/read.ipynb` | `explore_v_crg_student_passed_credit.ipynb` | P1 | Verified read-only. Folder → `V_CRG_STUDENT_PASSED_CREDIT/` (P3). |
| `V_SCH_COURSE_OFFER/read.ipynb` | `explore_v_sch_course_offer.ipynb` | P1 | Verified read-only. |

### note_books/feature_eng/

| Current | Proposed | Prio | Why |
|---|---|---|---|
| `select.ipynb` | `01_select_model_population.ipynb` | P2 | Stage number + says what it selects (columns AND the `start_agpa_points` row filter). Minimal alternative: `01_select.ipynb`. |
| `handle_gpa.ipynb` | `02_feature_engineering.ipynb` | P1 | Name lies: it runs the **entire** `run_feature_engineering_job` and owns `after_fet_eng.parquet`; GPA repair is one sub-step. |
| `handeling_outliers.ipynb` | **D5-dependent** — if exploratory (recommendation on record): `explore_outlier_removal.ipynb`; if adopted into training: numbered stage name assigned then | P2 | Typo "handeling" fixed either way. Do not rename before D5 is decided. |
| `pipeline_run_judge_test.ipynb` | KEEP (optional P3: `judge_feature_engineering.ipynb`) | P3 | Descriptive enough; correctly unnumbered. |

### note_books/model_eng/

| Current | Proposed | Prio | Why |
|---|---|---|---|
| `split_diagnostics.ipynb` | `01_split_diagnostics.ipynb` | P2 | Number reflects it runs first in this folder. Name kept — it is CLAUDE.md's own example — despite the known tension that it *owns* the base splits, not merely diagnoses them (alternative `01_split_base.ipynb` if you prefer ownership-true naming; flagging, not deciding). |
| `course_difficulty.ipynb` | `02_course_difficulty.ipynb` | P2 | Number only. |
| `diploma_type_bucketing.ipynb` | `03_diploma_type_bucketing.ipynb` | P2 | Number only. |
| `course_difficulty_fallback_diagnostic.ipynb` | KEEP | — | Correctly unnumbered diagnostic. |
| `read.ipynb` | `explore_train_split.ipynb` | P1 | Verified read-only inspection of `df_train`. |

### note_books/training_notebooks/

| Current | Proposed | Prio | Why |
|---|---|---|---|
| `light_gbm.ipynb` | `01_train_lightgbm.ipynb` | P2 | Verb + order (it drives `src/model_training.py`). |
| `results_analysis.ipynb` | `02_results_analysis.ipynb` | P2 | Runs after training. |

### note_books/debug/ and project root

| Current | Proposed | Prio | Why |
|---|---|---|---|
| `debug/trace_student.ipynb` | KEEP | — | Correct verb-prefixed diagnostic name. |
| root `read.ipynb` | move to `note_books/debug/` + rename `explore_raw_add_student_degree_status.ipynb` | P1 | Verified read-only exploration of the raw ADD table; a loose, meaningless-named notebook at the project root violates both placement and naming rules. |

After all 6A renames: zero `read.ipynb` files remain (five today), no `all.ipynb`, no case-only conflicts anywhere.

**Deferred folder-level items (P3, recommend against for now):** `note_books/` → `notebooks/` and any restructuring of `feature_eng`/`model_eng`/`training_notebooks` — high churn, breaks documented references (`paths_audit.md`, Decisions_Log, this plan), no safety gain.

---

## 6B — DataFrame naming

**Principles (contract 9).** Pipeline notebooks use semantic names: `df_<domain>[_<stage>]` (e.g. `df_crg`, `df_crg_add`, `df_selected_model`, `df_primary`). Positional names (`df`, `df1`, `df2`, `d`, `a`) are forbidden in pipeline notebooks and tolerated only in throwaway diagnostic cells. Loop variables (`_df_split`) and job-result keys (`df_primary`, `df_model_audit`, `df_excluded_over_policy`) are canonical as-is. A conforming in-repo example: `df_add_clean_step` in `add_student_degree_status_clean.ipynb`.

**Worst offenders (concrete renames, applied only inside each notebook's 7b group; rename-only edits must pass §4B logical parity — identical outputs):**

| Notebook | Current | Proposed |
|---|---|---|
| `01_merge_crg_add_acd` | `df_merge_test` (it becomes the real merged frame) | `df_crg_add` |
| `01_merge_crg_add_acd` | `dfcrg_acd_add` (missing underscore, scrambled) | `df_crg_add_acd` |
| `select` | `df` → `df1` → `df_feature` → `df_feature2` chain | `df_merged_diploma` → `df_selected_base` → `df_selected_narrow` → `df_selected_model` |
| `select` | `d`, `d1`, `a` (raw-table diagnostic cells) | `df_raw_add`, `df_raw_crg`, or drop the cells (drop = separate 7b decision) |
| `handle_gpa` | `df` (input) | `df_selected_model` (matches upstream artifact) |
| `handle_gpa` | `df1` (5-row debug head feeding `merged.csv`) | removed together with the debug write (already a 7a item) |
| `Academic_info_clean` | `df` → `df1` | `df_academic_info_raw` → `df_academic_info_clean` |

---

## 6C — Saved artifacts

### RAW_DIR (owner: `00_extract_raw_tables`)

| Current | Proposed | Prio | Repoint |
|---|---|---|---|
| `v_add_adcademic_info.parquet` | `v_add_academic_info.parquet` | **P1 — required, fixes a broken read** | writer: extractor cell 5; reader: `Academic_info_clean` cell 0 (already reads the corrected name — **the file it reads does not exist**; see risk flag in plan §10). One coordinated 7b group: rename the on-disk artifact (copy+preserve), confirm extractor and cleaner both use the corrected name. |
| `v_crg_student_course_raw.parquet` | KEEP (documented exception: `_raw` suffix redundant inside `raw/`) | P3 | — |
| all other raw parquets | KEEP | — | Names match their source views. |

### PREPROCESSED_DIR (owners: the per-table cleaners)

| Current | Proposed | Prio | Repoint |
|---|---|---|---|
| `V_add_academic_info/v_add_adcademic_info_cleaned.parquet` | `V_ADD_ACADEMIC_INFO/clean_v_add_academic_info.parquet` | P1 | writer: diploma cleaner; readers: `02_merge_diploma` cell 1; stale docstring ref in retired `merge_diploma.py` (update text in 7a). Fixes typo + folder case + suffix-vs-prefix inconsistency. |
| `V_ACD_DEGREE_COURSE/v_acd_degree_course.parquet` | `clean_v_acd_degree_course.parquet` | P2 | readers: `01_merge_crg_add_acd` cell 3, `all.ipynb` (pending its disposition), `trace_student`. Only cleaned artifact without the `clean_` prefix. |
| root stray `v_add_student_degree_status_clean.parquet` (pre-drop twin, loose in `preprocessed/`) | **USER DECISION** — recommended: ARCHIVE (no pipeline reader; only `trace_student` lists it) | P2 | Known quirk (paths_audit §Risks 5): the ADD cleaner wrote its output twice. **Producer side already resolved out-of-band 2026-07-07** — the user removed the second write from the cleaner; only the stale on-disk artifact's disposition remains. |
| `clean_v_crg_student_course.parquet`, `clean_v_add_student_degree_status.parquet`, `clean_v_acs_grade.parquet` | KEEP | — | Already conform. |

### MERGE_DIR (owners: the two merge notebooks)

| Current | Proposed | Prio |
|---|---|---|
| `merge_crg_add.parquet` | KEEP | — |
| `merged_add_acd_crg.parquet` | KEEP (P3 optional: `merge_crg_add_acd.parquet` for verb+order consistency — churn outweighs benefit; recommend reject) | P3 |
| `merged_with_diploma.parquet` | KEEP (D1-blessed name, already implemented and read by `select`) | — |
| `merge_crg_add_unmatched_add_snapshot.csv` | KEEP name; **relocate to `AUDIT_DIR`** (contract 6 tension, 7b move) | P2 |

### FEATURES_DIR (7b relocation targets out of AUDIT_DIR — plan Phase 4 actions)

| Current | Proposed | Prio | Repoint |
|---|---|---|---|
| `AUDIT_DIR/df_crg_add_acd.parquet` (select output; name neither says "selected" nor belongs in audit) | `FEATURES_DIR/selected_model_population.parquet` | P1 | writer: `select` last cell; readers: `handle_gpa`, `handeling_outliers`, `pipeline_run_judge_test`, `trace_student`. Byte-parity move+rename. |
| `AUDIT_DIR/after_fet_eng.parquet` | `FEATURES_DIR/feature_engineered_primary.parquet` | P1 | writer: `handle_gpa` (single writer — C1 enforcement precondition applies before the move); readers: `split_diagnostics`; stale refs in retired `merge_diploma.py` + bucketing error text (7a text fixes). It saves `df_primary` — the name says so. |

### MODEL_DATA_DIR — D2 split generations (the deferred key deliverable)

> **Status update (7a, 2026-07-07):** the generation filenames below are now implemented in
> code — each of the three split-stage notebooks reads the previous generation and writes its
> own. The files themselves do not exist on disk until the gated 7c rebuild; the bare-named
> on-disk splits remain the live training inputs, and the training-side reader repointing
> stays 7b/7c as planned.

Generation-suffixed names; owner per generation (contracts 2–3):

| Generation | Files | Owner |
|---|---|---|
| base | `df_train_base.parquet`, `df_valid_base.parquet`, `df_test_base.parquet` | `split_diagnostics` |
| difficulty-enriched | `df_train_difficulty.parquet`, `df_valid_difficulty.parquet`, `df_test_difficulty.parquet` | `course_difficulty` |
| final (model-facing) | `df_train_final.parquet`, `df_valid_final.parquet`, `df_test_final.parquet` | `diploma_type_bucketing` |

**The bare names `df_train.parquet` / `df_valid.parquet` / `df_test.parquet` are RETIRED** — after the 7c migration no file keeps a bare name (originals preserved per contract 11, then retired). Rationale (raised as a risk flag in plan §10): four live readers today read the bare names (`light_gbm`, `results_analysis`, `course_difficulty_fallback_diagnostic`, `model_eng/read`); if base splits kept the bare name, training would silently read unenriched data after the rebuild. Retiring the name makes every stale reader fail loudly. Reader repointing (7b/7c): training + results-analysis + KNN build → `*_final`; fallback diagnostic → `*_difficulty` (it inspects difficulty columns); exploration → any, explicitly.

### ARTIFACTS_DIR — fitted state (contract 14)

| Artifact | Name | Notes |
|---|---|---|
| diploma bucketing fit state | `diploma_type_bucket_map.json` | New; written by `diploma_type_bucketing` during the D2 rebuild; contents per contract 14. |
| course difficulty lookups | `course_difficulty_lookup.parquet` | Existing name kept; gains a committed producer if you adopt one (open USER DECISION). |
| KNN index | `knn_index.pkl` | Existing name kept; same producer decision. |
| categorical levels | `categorical_levels.json` | New; from `learn_categorical_levels`. |

Version identity lives **inside** each artifact (fit-source commit + train-manifest reference), not in filename version suffixes.

### Other

- `FINAL_DIR/without_outliers.parquet` — **D5-dependent**; defer naming/disposition until D5 is decided.
- Data-root residues `merged.csv`, `after_feature_eng_run.csv` — archive per Phase 4 (no rename); `merged.csv`'s live producer write is removed in 7a.
- `models/grade_model.lgbm`, `pass_model.lgbm`, `metrics.json` — KEEP (conform).

---

## Execution constraints (bind 7b)

1. Nothing renames now. Each rename belongs to a 7b group with: preserve → rename/move → repoint every reader listed above → validate (byte parity for pure move/rename; §4B parity when a rerun is involved) → user approval → retire preserved original.
2. Notebook renames are `git mv` + same-group fixes of in-repo textual references (e.g. `02_merge_diploma`'s error text citing "Read.ipynb"; bucketing's `merge_diploma.py` failure message — already 7a items).
3. The `v_add_academic_info` raw rename group is **blocking for 7c** (Chain A stage 1 cannot run until writer and reader agree on one name — see plan §10 flag).
4. Ordering: folder renames and the file renames inside them land in one group per folder to avoid double-repoints.
5. D5 gates two items only (`handeling_outliers` notebook, `without_outliers.parquet`); everything else is decidable now.

**Gate:** this draft awaits Phase 6 approval. Approving it approves names only — every physical change still runs behind its Phase 7 group gate.
