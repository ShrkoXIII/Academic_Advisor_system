# Academic Advisor Architecture Review

## High-Level Summary

This project builds an academic course-planning recommender. The production path is centered on `src/feature_engineering.py`, which converts raw student-course-semester rows into leak-safe features. Those features train grade and pass-risk models in `src/model_training.py`; `src/inference.py` loads trained artifacts and scores candidate courses; `src/recommendation.py` generates and ranks course plans using model predictions plus similar-student evidence from `src/knn_advisor.py`.

Notebooks under `note_books/` appear to be exploration, preprocessing, diagnostics, and model-engineering workspaces. The source package under `src/` is the reusable application layer. Generated artifacts live under `models/`, egg-info directories, and notebook outputs.

For a notebook-by-notebook index, see `NOTEBOOK_REVIEW.md`.

## Main Inputs, Outputs, and Side Effects

### `src.feature_engineering.run_feature_engineering_job`

Inputs:
- `df`: pandas DataFrame at course-row grain. Expected keys include `university_id` or dotted ID suffixes, `student_id`, `degree_id`, `part_id`, and academic fields such as GPA, credits, course IDs, requirement metadata, and start-level data.
- `structural_zero_as_nan`: optional boolean that adds `model_prev_gpa` with structural zero fallback converted to `NaN`.

Outputs:
- `df_model_audit`: full feature/audit DataFrame, including over-policy rows.
- `df_primary`: training-ready rows after over-policy semester exclusion.
- `df_excluded_over_policy`: excluded overload rows retained for audit.
- `diagnostics`: dictionary of conflict reports, row counts, fallback counts, merge checks, and feature schema changes.

Side effects:
- Prints diagnostics to stdout.
- Does not write files.

### `src.model_training`

Inputs:
- `prepare_X_y(df, target)`: feature-engineered DataFrame and target type `grade` or `pass`.
- `train_grade_model(df_train, df_valid, ...)`: train and validation DataFrames.
- `train_pass_model(df_train, df_valid, ...)`: train and validation DataFrames.
- CLI `main()`: parquet paths for `--train`, `--valid`, `--test`, and output directory `--out`.

Outputs:
- LightGBM `Booster` objects.
- Metric dictionaries for evaluation.
- CLI writes `grade_model.lgbm`, `pass_model.lgbm`, and `metrics.json`.

Side effects:
- Reads parquet splits from disk in CLI mode.
- Writes model artifacts and metrics.
- Prints training/evaluation logs.

### `src.inference.StudentScorer`

Inputs:
- `load(...)`: trained LightGBM model paths and course-difficulty lookup parquet path.
- `score(...)`: historical course rows for one student, candidate course IDs, target part ID, degree ID, optional expected workload, optional precomputed snapshot.
- `score_plan(...)`: historical rows plus a concrete plan list.

Outputs:
- Candidate-course DataFrame indexed by `course_id` with `pred_mark`, `pass_prob`, attempt number, credits, historical course stats, and difficulty-missing flag.

Side effects:
- Reads model and parquet artifacts during `load`.
- Suppresses feature-engineering stdout while building snapshots.

### `src.recommendation.Recommender`

Inputs:
- `load(...)`: grade model, pass model, difficulty lookup, and KNN index artifact paths.
- `recommend(...)`: student history, candidate course IDs, target semester, degree ID, optional remaining credits, plan count, and top-k count.

Outputs:
- List of ranked plan dictionaries with courses, total credits, expected AGPA, risk, workload, graduation progress, KNN evidence, and composite score.

Side effects:
- Reads artifacts through `StudentScorer.load` and `KNNAdvisor.load`.

### `src.knn_advisor.KNNAdvisor`

Inputs:
- `build(df_train, ...)`: feature-engineered training DataFrame.
- `find_similar(snapshot, k)`: one student snapshot.
- `summarize_evidence(neighbours, plan_course_ids)`: KNN neighbour rows and optional plan.

Outputs:
- Persistable KNN advisor.
- Similar-neighbour DataFrame with `_knn_distance`.
- Evidence dictionary with pass-rate, marks, GPA, failure-rate, and plan-overlap summaries.

Side effects:
- `save(path)` writes a pickle artifact.
- `load(path)` reads a pickle artifact.

### Utility Modules

`src.cleaning_utils.py` normalizes ID-like columns and appends audit reasons. It returns DataFrames or mutates the selected reason column in-place for `append_reason`.

`src.paths.py` centralizes data/report directories and creates expected folders at import time.

`src.db_connect.py` returns an Oracle connection using credential constants expected from local configuration or environment injection.

## Logic Flow Pseudo-Code

### End-to-End Recommendation

```text
load trained grade model
load trained pass model
load course difficulty lookup
load KNN similar-student index

given student history and candidate courses:
    build one current student snapshot from history
    score every candidate course with grade and pass models
    remove candidates below minimum pass-probability threshold
    generate valid course-plan combinations under credit/course limits
    find similar historical student-semesters once
    for each candidate plan:
        rescore plan with exact total credits and course count
        summarize KNN evidence for that plan
        calculate expected AGPA, risk, workload, graduation progress
        blend those axes into composite score
    return top-k plans sorted by composite score
```

### Feature Engineering

```text
validate df is a pandas DataFrame
copy df into full audit frame
ensure university_id exists or derive it from dotted ID suffixes
normalize semester timeline keys
add policy flags and capped fail-history fields

build semester history:
    detect conflicting semester-level values
    aggregate course rows to one row per university/student/degree/part
    sort each student-degree timeline by part_id
    detect interruption semesters
    compute prior interruption counts
    compute first-active and first-row timeline flags
    compute last valid GPA before current semester with forward-fill then shift

merge semester history back to course rows:
    normalize merge keys on both sides
    validate many-to-one merge coverage
    copy prefixed semester features to canonical feature names

repair previous GPA:
    try raw previous GPA
    else last valid GPA before current semester
    else start AGPA
    else structural zero fallback
    record source and missing/invalid-zero flags

add row-wise features:
    split part IDs into year and semester
    parse start-level ordinal
    build requirement features
    compute fail-credit ratio
    build degree-course key

report suspicious zero fallback rows
check semester-stability contracts
split df_model_audit into df_primary and df_excluded_over_policy
return frames and diagnostics
```

### Model Training

```text
read train, valid, and test parquet splits
for each split:
    encode requirement_size_bucket
    select explicit MODEL_FEATURES allow-list
    assert leakage columns are absent

train grade regression model with validation early stopping
evaluate grade model on valid and test
save grade model

train pass/fail classifier with class weighting
evaluate pass model on valid and test
run stratified pass-model diagnostics
save pass model
write metrics.json
```

## Code Annotation Status

The Python source files now include intent-focused comments before major logical blocks. Because the project is Python, annotations use `#` comments instead of `//` comments.

Annotated files:
- `src/feature_engineering.py`
- `src/model_training.py`
- `src/inference.py`
- `src/recommendation.py`
- `src/knn_advisor.py`
- `src/cleaning_utils.py`
- `src/paths.py`
- `src/db_connect.py`
- `tests/test_feature_engineering.py`

Notebook status:
- Every `.ipynb` in the repository now has a leading generated `Architectural Notes` markdown cell.
- The notes summarize each notebook's purpose, inputs/data sources, outputs/side effects, logic flow, and maintainability risks.
- Notebook review details are summarized in `NOTEBOOK_REVIEW.md`.

Generated artifacts were not rewritten.

## Complexity and Maintainability Notes

- `feature_engineering.py` is the largest module and mixes transformation logic, diagnostics, printing, and policy rules. It would be easier to maintain if diagnostics were routed through a logger or structured reporter instead of direct `print` calls.
- The feature schema is duplicated across feature engineering, model training, and inference. `MODEL_FEATURES`, difficulty fields, and row assembly should eventually be governed by one schema object or contract test.
- `StudentScorer.score()` manually constructs candidate rows with many `snapshot.get(...)` and `diff_row.get(...)` calls. This is a drift risk whenever training features change.
- `KNNAdvisor.save()` uses pickle, which is convenient but not portable or safe for untrusted artifacts. A structured artifact format would be safer for production.
- `src.db_connect.py` references credential constants that are not defined in the module. That is acceptable for local notebooks only if a config injection convention exists; otherwise it should load from environment variables or an ignored settings file.
- `src.paths.py` creates directories at import time. This is convenient for notebooks but can surprise library callers and tests.
- Several files contain mojibake/encoding artifacts in existing comments and docs. Normalizing file encoding would improve readability but should be handled as a separate cleanup to avoid mixing behavior-neutral annotation with text-repair churn.
