# Notebook Review

This index summarizes the notebook layer after adding leading architectural notes to each `.ipynb`. The notebooks were not executed; the review is based on their code structure, imports, reads, writes, and existing markdown.

## Pipeline Map

1. Raw extraction: `note_books/pre_processing/extact_all_row_tables.ipynb`
2. Source-table cleaning: notebooks under `note_books/pre_processing/V_*` and `ACS_GRADE`
3. Merge/pre-feature dataset assembly: `note_books/pre_processing/merge/*`, `note_books/pre_processing/all.ipynb`
4. Feature engineering and audits: `note_books/feature_eng/*`
5. Model split and difficulty engineering: `note_books/model_eng/*`
6. Debugging and spot checks: `note_books/debug/trace_student.ipynb`, root `read.ipynb`, small `read.ipynb` notebooks

## Notebook Inventory

| Notebook | Role | Persisted Outputs |
|---|---|---|
| `read.ipynb` | Root scratchpad for raw ADD table inspection. | None detected. |
| `note_books/debug/trace_student.ipynb` | Trace one student's records across artifacts. | None detected. |
| `note_books/feature_eng/handeling_outliers.ipynb` | Outlier investigation and filtered feature output. | `data/final/without_outliers.parquet` |
| `note_books/feature_eng/handle_gpa.ipynb` | Runs `run_feature_engineering_job` and exports engineered data. | `data/merged.csv`, `data/audit/after_fet_eng.parquet` |
| `note_books/feature_eng/pipeline_run_judge_test.ipynb` | Diagnostic review of feature-engineering behaviour. | None detected. |
| `note_books/feature_eng/select.ipynb` | Selects and prepares columns for feature engineering. | `data/audit/df_crg_add_acd.parquet` |
| `note_books/model_eng/course_difficulty.ipynb` | Computes course-difficulty features and enriches splits. | Updated train/valid/test parquet splits. |
| `note_books/model_eng/course_difficulty_fallback_diagnostic.ipynb` | Audits difficulty fallback coverage. | None detected. |
| `note_books/model_eng/read.ipynb` | Quick training-split inspection. | None detected. |
| `note_books/model_eng/split_diagnostics.ipynb` | Creates and validates train/valid/test splits. | `df_train.parquet`, `df_valid.parquet`, `df_test.parquet` |
| `note_books/pre_processing/ACS_GRADE/clean_ACS_grade.ipynb` | Cleans grade lookup data. | `clean_v_acs_grade.parquet` |
| `note_books/pre_processing/all.ipynb` | Loads cleaned source tables for broad inspection. | None detected. |
| `note_books/pre_processing/extact_all_row_tables.ipynb` | Extracts Oracle source views to raw parquet snapshots. | Raw parquet files under `data/raw`. |
| `note_books/pre_processing/merge/i.ipynb` | Scratch merge exploration for ACD and CRG tables. | None detected. |
| `note_books/pre_processing/merge/merge_crg_add.ipynb` | Prototype CRG + ADD merge. | None detected. |
| `note_books/pre_processing/merge/mergecrgadd.ipynb` | Maintained CRG + ADD + ACD merge. | Merged feature parquet and unmatched CSV report. |
| `note_books/pre_processing/V_ACD_DEGREE_COURSE/load_preprocessing.ipynb` | Cleans curriculum degree-course metadata. | `v_acd_degree_course.parquet` |
| `note_books/pre_processing/V_ADD_STUDENT_DEGREE_STATUS/add_student_degree_status_clean.ipynb` | Cleans student degree-status snapshots. | Clean ADD status parquet outputs. |
| `note_books/pre_processing/V_CRG_STD_COR_TEMP_REQ/read.ipynb` | Inspects temporary course-request data. | None detected. |
| `note_books/pre_processing/V_CRG_STUDENT_COURSE/clean_v_crg_student_course.ipynb` | Cleans core student-course attempt records. | `clean_v_crg_student_course.parquet` |
| `note_books/pre_processing/V_CRG_STUDENT_COURSE/mark_finish_status_disagreement_diagnostic.ipynb` | Diagnoses mark/status disagreement. | None detected. |
| `note_books/pre_processing/V_CRG_STUDENT_PASSED/read.ipynb` | Inspects passed-credit source table. | None detected. |
| `note_books/pre_processing/V_SCH_COURSE_OFFER/read.ipynb` | Inspects course-offering data for candidate generation. | None detected. |

## Cross-Cutting Maintainability Notes

- Many notebooks use absolute `D:\AI\...` paths. Prefer `src.paths` constants so work can move across machines.
- Important pipeline stages still live only in notebooks: raw extraction, table cleaning, merge assembly, split creation, and course-difficulty enrichment. These should eventually become scripts or source modules with tests.
- Several notebooks write directly over model or feature artifacts. Add explicit output versioning before comparing experiments.
- The split and difficulty notebooks affect model validity; their leakage assumptions should be covered by regression tests.
- Scratch notebooks (`read.ipynb`, `merge/i.ipynb`, small table `read.ipynb` files) are useful for exploration but should not become hidden production dependencies.
