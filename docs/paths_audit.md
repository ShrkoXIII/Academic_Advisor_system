# Paths Task — Phase 1 Audit & Conversion Plan

## Context

CLAUDE.md's active job: every notebook and script must get file paths from `src/paths.py`; no hardcoded absolute paths anywhere (code, docstrings, markdown usage examples). This document is the read-only Phase 1 audit (nothing has been modified, executed, renamed, moved, or deleted) plus a proposed conversion order. Scope is PATHS ONLY: no data logic, ML logic, targets, features, thresholds, or model code are touched or proposed. Design parameters from the user's earlier answers: **(1)** `DATA_DIR` derives from env var `ACADEMIC_ADVISOR_DATA_DIR` with fallback `PROJECT_ROOT / "data"`; **(2)** legacy-root split reads are to be repointed to `MODEL_DATA_DIR` — still to be re-confirmed at conversion time per CLAUDE.md (report, never silently repoint); **(3)** all constants point at current real on-disk locations — no file moves in this job.

## Audit results — hardcoded absolute paths

**74 source occurrences total: 64 in notebooks (21 of 25), 10 in `src/*.py`.**
Classifications: **CANON** = active canonical project data path (`D:\AI\Real projects\Academic_Advisor\data\...`) → replace with `paths.py` constant; **LEGACY** = `D:/AI/data_clean_academic_advisor/...` → bug, repoint/rewrite explicitly. All confidences High unless noted — every path was read in the actual cell/line, not inferred from filenames.

### Python scripts (`src/`) — 10 occurrences

| File / location | Where | Exact path (abbreviated) | Class | Replacement |
|---|---|---|---|---|
| `src/merge_diploma.py` L17 | **executable code** | `r"D:\...\data\preprocessed\V_add_academic_info\v_add_adcademic_info_cleaned.parquet"` (+ legacy root in trailing comment) | CANON (+LEGACY comment) | `PREPROCESSED_DIR / "V_add_academic_info" / "v_add_adcademic_info_cleaned.parquet"` |
| `src/merge_diploma.py` L20 | **executable code** | `"D:/.../data/audit/after_fet_eng.parquet"` | CANON | `AUDIT_DIR / "after_fet_eng.parquet"` |
| `src/model_training.py` L16–18 | module docstring (CLI example) | `D:/AI/data_clean_academic_advisor/data/model_data/df_{train,valid,test}.parquet` | **LEGACY** | example rewritten to reference `paths.MODEL_DATA_DIR` |
| `src/inference.py` L16 | module docstring (usage example) | `D:/AI/data_clean_academic_advisor/data/artifacts/course_difficulty_lookup.parquet` | **LEGACY** | `ARTIFACTS_DIR / "course_difficulty_lookup.parquet"` |
| `src/knn_advisor.py` L16, L19 | module docstring (usage example) | `D:/AI/data_clean_academic_advisor/data/artifacts/knn_index.pkl` ×2 | **LEGACY** | `ARTIFACTS_DIR / "knn_index.pkl"` |
| `src/recommendation.py` L21–22 | module docstring (usage example) | `D:/AI/.../course_difficulty_lookup.parquet`, `D:/AI/.../knn_index.pkl` (elided legacy) | **LEGACY** | `ARTIFACTS_DIR / ...` |

Also non-absolute but violating "paths from paths.py": CWD-relative `'models/...'` defaults in `inference.py`/`recommendation.py` docstrings and `model_training.py` argparse `--out` default `"models"` (L629) → `MODELS_DIR`.

### Notebooks — 64 occurrences (code + markdown cells)

| Notebook | Cells (type) | Count | Paths under `data/` | Class | Replacement |
|---|---|---|---|---|---|
| `note_books/merge/01_merge_crg_add_acd.ipynb` | c3 (code), c21 (code), c0 (md) | 3 | reads `preprocessed/V_ACD_DEGREE_COURSE/v_acd_degree_course.parquet`; writes `features/merged_add_acd_crg.parquet` | CANON | `PREPROCESSED_DIR / ...`, `FEATURES_DIR / ...` (cell 2 already derives other paths via `find_project_root()` — replace that helper with `paths.py` imports) |
| `note_books/training_notebooks/light_gbm.ipynb` | c2 (code), c9 (md ×3) | 4 | `DATA_DIR = "D:/AI/data_clean_academic_advisor/data/model_data"` + CLI example | **LEGACY — the wrong-folder bug** | `MODEL_DATA_DIR` (repoint only with explicit approval) |
| `note_books/training_notebooks/results_analysis.ipynb` | c1 (code) | 1 | `Path("D:\AI\Real projects\...\data\model_data")` — **non-raw string**, works only by escape-sequence luck | CANON (fragile) | `MODEL_DATA_DIR`; also `MODELS_DIR = Path("models")` → `paths.MODELS_DIR` |
| `note_books/model_eng/split_diagnostics.ipynb` | c3, c5 (code) | 2 | `data/model_data` (SPLIT_DATA_DIR), `data/audit/after_fet_eng.parquet` | CANON | `MODEL_DATA_DIR`, `AUDIT_DIR / "after_fet_eng.parquet"` |
| `note_books/model_eng/course_difficulty.ipynb` | c4 (code) | 1 | `data/model_data` | CANON | `MODEL_DATA_DIR` |
| `note_books/model_eng/course_difficulty_fallback_diagnostic.ipynb` | c2 (code, L35 + L44) | 2 | SPLIT_DATA_DIR default `data/model_data` **+ silent fallback to legacy root** | CANON + **LEGACY** | `MODEL_DATA_DIR`; legacy fallback block flagged for removal (needs explicit approval) |
| `note_books/model_eng/diploma_type_bucketing.ipynb` | c4 (code) | 1 | `data/model_data` | CANON | `MODEL_DATA_DIR` |
| `note_books/model_eng/read.ipynb` | c3 (code), c0 (md) | 2 | `data/model_data/df_train.parquet` | CANON | `MODEL_DATA_DIR / "df_train.parquet"` |
| `note_books/feature_eng/select.ipynb` | c3, c52, c60, c62 (code), c0 (md ×4) | 8 | reads `features/merged_add_acd_crg.parquet`, `raw/...` ×2; writes `audit/df_crg_add_acd.parquet` | CANON | `FEATURES_DIR`, `RAW_DIR`, `AUDIT_DIR` |
| `note_books/feature_eng/handle_gpa.ipynb` | c4, c21, c27 (code), c0 (md ×3) | 6 | reads `audit/df_crg_add_acd.parquet`; writes `data/merged.csv` (data root!), `audit/after_fet_eng.parquet` | CANON | `AUDIT_DIR`, `DATA_DIR / "merged.csv"` |
| `note_books/feature_eng/handeling_outliers.ipynb` | c3, c15 (code) | 2 | reads `audit/df_crg_add_acd.parquet`; writes `final/without_outliers.parquet` | CANON | `AUDIT_DIR`, `FINAL_DIR` |
| `note_books/feature_eng/pipeline_run_judge_test.ipynb` | c4 (code) | 1 | `audit/df_crg_add_acd.parquet` (candidate list entry) | CANON | `AUDIT_DIR / ...` |
| `note_books/debug/trace_student.ipynb` | c2 (code) | 1 | `ROOT = r"D:\...\data"` (10 artifacts joined onto it) | CANON | `ROOT = DATA_DIR` (keeps the relative joins) |
| `note_books/pre_processing/all.ipynb` | c3 (code ×4), c0 (md ×4) | 8 | reads 4 cleaned tables under `preprocessed/` | CANON | `PREPROCESSED_DIR / ...` |
| `note_books/pre_processing/ACS_GRADE/clean_ACS_grade.ipynb` | c6, c37 (code), c0 (md) | 3 | `raw/v_acs_grade.parquet`; writes `preprocessed/V_ACS_GRADE/clean_v_acs_grade.parquet` | CANON | `RAW_DIR`, `PREPROCESSED_DIR` |
| `note_books/pre_processing/V_CRG_STUDENT_COURSE/clean_v_crg_student_course.ipynb` | c4, c63 (code), c0 (md) | 3 | `raw/v_crg_student_course_raw.parquet`; writes `preprocessed/V_CRG_STUDENT_COURSE/...` | CANON | `RAW_DIR`, `PREPROCESSED_DIR` |
| `note_books/pre_processing/V_ADD_STUDENT_DEGREE_STATUS/add_student_degree_status_clean.ipynb` | c34, c39, c47 (code), c0 (md ×3) | 6 | writes clean file **twice**: `preprocessed/v_add_student_degree_status_clean.parquet` (root) AND `preprocessed/V_ADD_STUDENT_DEGREE_STATUS/clean_...parquet`; reads CRG clean | CANON | `PREPROCESSED_DIR / ...` (keep both writes — behavior-preserving) |
| `note_books/pre_processing/V_ACD_DEGREE_COURSE/load_preprocessing.ipynb` | c15 (code), c0 (md) | 2 | writes `preprocessed/V_ACD_DEGREE_COURSE/v_acd_degree_course.parquet` (reads already use `RAW_DIR`) | CANON | `PREPROCESSED_DIR / ...` |
| `note_books/pre_processing/V_ACADEMIC_INFO/Read.ipynb` | c0, c12, c14 (code) | 3 | `raw/v_add_adcademic_info.parquet`; `preprocessed/V_add_academic_info/...` ×2 | CANON | `RAW_DIR`, `PREPROCESSED_DIR` (keep lower-case folder name as-is; renaming is task 2) |
| `note_books/pre_processing/V_SCH_COURSE_OFFER/read.ipynb` | c2 (code), c0 (md) | 2 | `raw/v_sch_course_offers.parquet` | CANON | `RAW_DIR / ...` |
| `note_books/pre_processing/V_CRG_STD_COR_TEMP_REQ/read.ipynb` | c2 (code), c0 (md) | 2 | `raw/v_crg_std_cor_temp_request.parquet` | CANON | `RAW_DIR / ...` |

**Clean files (no absolute paths, already use `paths.py`):** root `read.ipynb`, `pre_processing/extact_all_row_tables.ipynb`, `pre_processing/V_CRG_STUDENT_PASSED/read.ipynb` (all import `RAW_DIR`), and `src/{paths,cleaning_utils,feature_engineering,db_connect,__init__}.py`, `tests/`.
**Output-cell residue** (stale execution echoes, not source; cleared naturally on re-run): 9 notebooks, largest `clean_ACS_grade.ipynb` (34).
**`archive/` folder:** does not exist anywhere in the project — nothing to audit separately.

## `src/paths.py` gap analysis

Present: `PROJECT_ROOT`, `DATA_DIR`, `RAW_DIR`, `CLEAN_DIR`, `PREPROCESSED_DIR` (alias of `CLEAN_DIR` — duplicate naming, keep `PREPROCESSED_DIR` as the primary going forward), `FEATURES_DIR`, `FINAL_DIR`, `REPORTS_DIR`, `ensure_dir`, `ensure_parent`.

Problems / gaps:
1. **`DATA_DIR = PROJECT_ROOT / "data"` hardcoded ([paths.py:12](../src/paths.py#L12)) — bug** under the external-data design. Proposed fix: `DATA_DIR = Path(os.environ.get("ACADEMIC_ADVISOR_DATA_DIR", PROJECT_ROOT / "data"))`.
2. **Missing constants** (all point at current real locations): `AUDIT_DIR = DATA_DIR/"audit"`, `MODEL_DATA_DIR = DATA_DIR/"model_data"`, `ARTIFACTS_DIR = DATA_DIR/"artifacts"`, `MODELS_DIR = PROJECT_ROOT/"models"` (models live at project root, NOT under data/), `MERGE_DIR = PREPROCESSED_DIR/"merge"` (where the merge notebook actually writes; `data/merged/` does not exist), `ARCHIVE_DIR = DATA_DIR/"archive"`.
3. **Non-canonical but real**: `FINAL_DIR` (`data/final/` exists and is written by `handeling_outliers.ipynb`) — keep; `REPORTS_DIR` — keep, unused by notebooks.
4. **Import-time side effect**: the module `mkdir`s six folders on import. With an env-derived `DATA_DIR`, a typo'd env var would silently create and use a wrong tree. Add new dirs to this list conservatively (audit/model_data/artifacts/models only — they all exist), and keep behavior otherwise.
5. No env/`os` import today — `import os` must be added.

## Conversion order (one file at a time; after each: verify identical resolved paths, then run per CLAUDE.md)

**Step 0 — `src/paths.py`** (prerequisite): env-var `DATA_DIR` + missing constants above. Verify: import it, print all constants, confirm each equals the audit table's current location byte-for-byte (with env var unset).

Then, priority order (wrong-folder bug first):
1. `note_books/merge/01_merge_crg_add_acd.ipynb` — early CRG+ADD+ACD merge
2. Training: `light_gbm.ipynb` (**legacy repoint — reported as a bug; repoint only on explicit approval**), `src/model_training.py` (docstring + `--out` default), `results_analysis.ipynb` (fragile non-raw string + `MODELS_DIR`)
3. Split consumers: `split_diagnostics.ipynb`, `model_eng/read.ipynb`, `course_difficulty_fallback_diagnostic.ipynb` (**legacy fallback block — reported; removal needs explicit approval**)
4. `course_difficulty.ipynb`
5. Diploma stage: `diploma_type_bucketing.ipynb`, `src/merge_diploma.py` (keep its "must match split_diagnostics" invariant — both become `AUDIT_DIR / "after_fet_eng.parquet"`)
6. Feature eng: `select.ipynb`, `handle_gpa.ipynb`, `handeling_outliers.ipynb`, `pipeline_run_judge_test.ipynb`; docstring examples in `src/inference.py`, `src/knn_advisor.py`, `src/recommendation.py`
7. Cleaning: `clean_v_crg_student_course.ipynb`, `clean_ACS_grade.ipynb`, `add_student_degree_status_clean.ipynb`, `load_preprocessing.ipynb`, `V_ACADEMIC_INFO/Read.ipynb`
8. Diagnostics/inspection: `trace_student.ipynb`, `mark_finish_status_disagreement_diagnostic.ipynb` (no absolute path — only confirm), `V_SCH_COURSE_OFFER/read.ipynb`, `V_CRG_STD_COR_TEMP_REQ/read.ipynb`, `all.ipynb`
9. Archived files: none exist.

Notebook edit mechanics: use NotebookEdit on the specific cells listed in the audit table; touch only path strings/imports, never logic. Markdown usage examples get the same constant-based rewrite as their code cells.

## Merge notebooks (task 3 — identified only, NOT touched in this job)

- **Early merge (CRG+ADD+ACD), maintained:** `note_books/merge/01_merge_crg_add_acd.ipynb` — real merges, writes `preprocessed/merge/merge_crg_add.parquet` and `features/merged_add_acd_crg.parquet`.
- **Dead prototype candidate:** `note_books/pre_processing/all.ipynb` — loads the same 4 cleaned tables, performs **no** merges and **no** writes ("broad inspection or manual join checks"). Deletion candidate for task 3, pending approval.
- **Later diploma merge (separate stage, keep separate):** `src/merge_diploma.py` (script, no notebook duplicate). `diploma_type_bucketing.ipynb` is a feature builder, not a merge.

## Risks & ambiguities

1. **`light_gbm.ipynb` cell 2 is a SyntaxError**: literally `import subproc  ess, sys` (verified in raw JSON). The training notebook cannot run today. Fixing the typo is a (trivial) code edit beyond pure paths — flagged here for explicit approval; without it the notebook can't be run-verified.
2. **`.gitignore` ignores `src/`** — the source modules are untracked and invisible to gitignore-respecting tools. Secrets check on `src/db_connect.py` came back CLEAR (no credentials defined anywhere in the project; no `.env` present), so tracking `src/` is safe. Un-ignoring `src/` + a safety commit is a manual step the user performs before any path edit.
3. **Legacy root still live on disk** with byte-identical split copies (same Jul 2 timestamps); after repointing, the legacy copies become dead weight and will silently diverge if anything regenerates — recommend archiving/deleting later (separate job).
4. Memory note `project_data_location.md` (data lives at legacy root) is now outdated — update after this plan is approved.
5. `add_student_degree_status_clean.ipynb` writes its clean output to two locations (pre-drop root copy + subfolder). Known quirk (documented in `trace_student.ipynb`); both writes preserved.
6. `preprocessed/V_add_academic_info/` casing is inconsistent with sibling folders — left as-is (renames are task 2).
7. `handle_gpa.ipynb` writes `merged.csv` at the data root; preserved as `DATA_DIR / "merged.csv"`.
8. **Student PII in tracked notebooks**: executed output cells in ≥9 notebooks contain row-level student data, and `trace_student.ipynb` hardcodes a real student ID. Already in git history; flag before pushing to any remote (separate decision, e.g. `nbstripout`).

## Verification

Nothing is executed in Phase 1. When conversion is explicitly approved, per file and in order: (a) static check — script imports `src.paths`, resolves the new expressions, asserts string-equality with the old absolute paths from this audit; (b) only then, run the notebook/script per CLAUDE.md and confirm it reads/writes the SAME files (compare file lists + sizes before/after); if any output differs, stop and report. `paths.py` itself: verify with env var unset (fallback) and set to a temp dir (override), confirming no directory is created in the wrong place.
