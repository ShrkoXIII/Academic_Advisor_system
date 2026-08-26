"""Feature engineering pipeline for academic course-outcome prediction.

This module converts raw student-course-semester rows into leak-safe model
features, audit frames, and diagnostics. The core design is to compute timeline
features at semester granularity first, then merge them back to course rows so
each candidate course carries the correct registration-time student state.
"""

import re

import numpy as np
import pandas as pd


# Domain caps and policy thresholds are centralized so model training,
# inference, and tests use one shared definition of extreme or invalid loads.
HIGH_COURSE_CREDITS_VALUE = 24
MAX_ALLOWED_SEMESTER_CREDITS = 25
MAX_ALLOWED_SEMESTER_COURSES = 8
FAIL_CREDITS_CAP = 120
STRUCTURAL_ZERO_AS_NAN = False

MISSING_KEY_SENTINEL = "__MISSING__"

# Translates Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩) to ASCII digits for numeric level parsing.
_ARABIC_INDIC_TRANS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Keys define the natural grain for semester-level history and student-degree
# timelines; university_id is included to avoid cross-campus collisions.
SEMESTER_KEY = [
    "university_id",
    "student_id",
    "degree_id",
    "part_id",
]
STUDENT_DEGREE_KEY = [
    "university_id",
    "student_id",
    "degree_id",
]

# Semester aggregation columns are collapsed from course rows before timeline
# features are calculated, preventing duplicate course rows from distorting
# prior-history features.
SEMESTER_AGGREGATION_COLUMNS = [
    "gpa_points",
    "semester_reg_credits",
    "semester_reg_courses",
    "semester_pass_credits",
    "start_total",
    "start_total_in_courses",
    "start_total_in_credits",
    "start_agpa_points",
    "prev_gpa_points",
    "over_policy_semester_credits",
    "over_policy_semester_courses",
    "exclude_over_policy_semester",
]

# These columns are computed at semester grain and later merged back to every
# course row in the same semester.
SEMESTER_FEATURE_COLUMNS = [
    "is_interruption_semester",
    "prev_semester_was_interruption",
    "prior_interruption_count",
    "consecutive_interruption_count",
    "no_previous_progress",
    "is_first_active_semester",
    "is_first_row_in_timeline",
    "last_valid_gpa_before_current_semester",
    "gpa_trend_delta",
]

# Public feature catalog used by diagnostics to report what this job added or
# repaired. The model itself still selects a narrower allow-list later.
FEATURE_COLUMNS = [
    "university_id",
    "is_high_credit_course",
    "over_policy_semester_credits",
    "over_policy_semester_courses",
    "exclude_over_policy_semester",
    "is_extreme_fail_history",
    "total_fail_credits_capped",
    "fail_credit_ratio_capped",
    "is_interruption_semester",
    "prev_semester_was_interruption",
    "prior_interruption_count",
    "consecutive_interruption_count",
    "no_previous_progress",
    "is_first_active_semester",
    "is_first_row_in_timeline",
    "last_valid_gpa_before_current_semester",
    "gpa_trend_delta",
    "gpa_trend_missing",
    "prev_gpa_points_missing",
    "prev_gpa_points_zero",
    "prev_gpa_invalid_zero_case",
    "prev_gpa_points_clean",
    "prev_gpa_fill_source",
    "prev_gpa_replaced_due_to_invalid_zero",
    "prev_gpa_actual_zero_performance",
    "model_prev_gpa",
    "part_year",
    "part_semester",
    "start_year",
    "start_semester",
    "start_level_missing",
    "start_level_ord",
    "requirement_type_missing",
    "degree_requirement_credits_count_missing",
    "course_share_of_requirement",
    "requirement_size_bucket",
    "degree_course_key",
]

# Columns that must NEVER enter the model matrix X.
# gpa_points and semester_pass_credits are used ONLY to derive shifted, past-only
# history features inside this job; they are leakage only if placed in X.
LEAKAGE_COLUMNS = [
    "final_mark",
    "gpa_points",
    "semester_pass_credits",
    "grade_id",
    "part_id",
    "start_part_id",
    "student_course_id",
    "student_id",
    "course_id",
    "degree_id",
    "faculty_id",
    "student_status_id",
    "degree_course_key",
    "prev_gpa_points",
    "total_fail_credits",
]


def safe_display(obj, max_rows=25):
    """Display in notebooks and fall back to compact console output."""
    try:
        from IPython.display import display

        display(obj)
        return
    except Exception:
        pass

    if isinstance(obj, pd.DataFrame):
        print(obj.head(max_rows).to_string())
    elif isinstance(obj, pd.Series):
        print(obj.head(max_rows).to_string())
    else:
        print(obj)


def assert_no_leakage_columns(X, extra=None):
    """Raise ValueError listing any LEAKAGE_COLUMNS present in X's columns.

    Call this before passing X to a model. Do NOT call inside
    run_feature_engineering_job — X is assembled in a later step.
    """
    forbidden = set(LEAKAGE_COLUMNS)
    if extra:
        forbidden.update(extra)
    found = [c for c in X.columns if c in forbidden]
    if found:
        raise ValueError(f"Leakage columns found in feature matrix X: {found}")


def _normalize_key_series(series):
    """Normalize merge and timeline keys to stripped strings with a sentinel."""
    normalized = series.astype("string").str.strip()
    normalized = normalized.mask(normalized.fillna("").eq(""))
    return normalized.fillna(MISSING_KEY_SENTINEL)


def _numeric_series(df, column, default=np.nan):
    """Return a numeric series for a possibly missing column."""
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _boolean_series(df, column, default=False):
    """Return a boolean series for a possibly missing column."""
    if column in df.columns:
        return df[column].fillna(default).astype(bool)
    return pd.Series(default, index=df.index, dtype="bool")


def _make_temp_names(existing_columns, labels, prefix):
    """Build temporary column names that do not collide with existing columns."""
    used = set(existing_columns)
    result = {}
    for label in labels:
        base = f"{prefix}{label}"
        name = base
        counter = 1
        while name in used:
            name = f"{base}_{counter}"
            counter += 1
        used.add(name)
        result[label] = name
    return result


def _derive_university_id(df):
    """Derive university_id from the dotted suffix of available ID columns."""
    # Prefer deriving from any stable dotted ID suffix so older datasets without
    # an explicit university column can still be grouped safely.
    candidates = [
        "student_id",
        "degree_id",
        "course_id",
        "faculty_id",
        "grade_id",
        "student_course_id",
    ]
    derived = pd.Series(pd.NA, index=df.index, dtype="string")
    for column in candidates:
        if column not in df.columns:
            continue
        values = _normalize_key_series(df[column]).replace(MISSING_KEY_SENTINEL, pd.NA)
        suffix = values.str.extract(r"\.([^.]+)$", expand=False)
        derived = derived.fillna(suffix)
    return derived.fillna(MISSING_KEY_SENTINEL).astype("string").str.strip()


def ensure_university_id(df, diagnostics):
    """Add or normalize university_id for university-aware grouping."""
    # If the upstream pipeline already supplied university_id, trust it after
    # normalization and record that no derivation was needed.
    if "university_id" in df.columns:
        df["university_id"] = _normalize_key_series(df["university_id"])
        diagnostics["university_id_source"] = "existing"
        return df

    # Fall back to ID suffix derivation to avoid mixing identical student or
    # degree IDs from different universities in timeline calculations.
    df["university_id"] = _derive_university_id(df)
    diagnostics["university_id_source"] = "derived_from_id_suffix"
    missing_count = int(df["university_id"].eq(MISSING_KEY_SENTINEL).sum())
    diagnostics["derived_university_id_missing_count"] = missing_count
    if missing_count:
        print(
            "WARNING: university_id could not be derived for "
            f"{missing_count} rows; using {MISSING_KEY_SENTINEL!r}."
        )

    # Suffix-consistency diagnostic: warn when two or more present ID columns
    # give different dotted suffixes (future multi-university safety check).
    _SUFFIX_SOURCE_COLS = [
        "student_id", "degree_id", "course_id",
        "faculty_id", "grade_id", "student_course_id",
    ]
    present_suffix_cols = [c for c in _SUFFIX_SOURCE_COLS if c in df.columns]
    if len(present_suffix_cols) >= 2:
        suffix_frame = pd.DataFrame(index=df.index)
        for col in present_suffix_cols:
            vals = _normalize_key_series(df[col]).replace(MISSING_KEY_SENTINEL, pd.NA)
            suffix_frame[col] = vals.str.extract(r"\.([^.]+)$", expand=False)

        def _row_has_conflict(row):
            present = row.dropna()
            return len(present) >= 2 and present.nunique() > 1

        conflict_mask = suffix_frame.apply(_row_has_conflict, axis=1)
        conflict_count = int(conflict_mask.sum())
        diagnostics["suffix_consistency_conflict_count"] = conflict_count
        if conflict_count:
            print(
                f"WARNING: {conflict_count} rows have inconsistent ID suffixes "
                "(possible multi-university data)."
            )
            example_cols = [c for c in ["student_id", "degree_id", "course_id"] if c in df.columns]
            safe_display(df.loc[conflict_mask, example_cols].head(25))
        else:
            print("Suffix consistency check: no conflicts detected.")
    else:
        diagnostics["suffix_consistency_conflict_count"] = 0

    return df


def normalize_timeline_keys(df):
    """Normalize key columns used for grouping and timeline sorting."""
    # Timeline features are only valid when every required grouping key exists.
    missing = [column for column in SEMESTER_KEY if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required semester key columns: {missing}")

    # Normalize before any groupby/merge so string, numeric, and blank variants
    # of the same ID resolve consistently.
    for column in SEMESTER_KEY:
        df[column] = _normalize_key_series(df[column])
    return df


def add_policy_and_fail_flags(df, diagnostics):
    """Build policy flags and capped fail-history features."""
    # Policy and fail-history flags are created before the training split so
    # excluded rows remain visible in df_model_audit diagnostics.
    course_credits = _numeric_series(df, "course_credits", default=np.nan)
    semester_reg_credits = _numeric_series(df, "semester_reg_credits", default=np.nan)

    df["is_high_credit_course"] = course_credits.eq(HIGH_COURSE_CREDITS_VALUE).fillna(False).astype(int)
    df["over_policy_semester_credits"] = semester_reg_credits.gt(MAX_ALLOWED_SEMESTER_CREDITS).fillna(False)

    if "semester_reg_courses" in df.columns:
        semester_reg_courses = _numeric_series(df, "semester_reg_courses", default=np.nan)
        df["over_policy_semester_courses"] = semester_reg_courses.gt(MAX_ALLOWED_SEMESTER_COURSES).fillna(False)
    else:
        print("WARNING: semester_reg_courses is missing; course-count overload check was skipped.")
        diagnostics.setdefault("warnings", []).append("semester_reg_courses missing")
        df["over_policy_semester_courses"] = False

    # The exclusion flag is retained as an audit feature while df_primary drops
    # these rows from normal training later.
    df["exclude_over_policy_semester"] = (
        df["over_policy_semester_credits"] | df["over_policy_semester_courses"]
    ).astype(bool)

    # Cap extreme fail credits to reduce outlier leverage while preserving an
    # explicit indicator that the raw value was unusually high.
    total_fail_credits = _numeric_series(df, "total_fail_credits", default=0).fillna(0)
    df["is_extreme_fail_history"] = total_fail_credits.gt(FAIL_CREDITS_CAP).astype(int)
    df["total_fail_credits_capped"] = total_fail_credits.clip(lower=0, upper=FAIL_CREDITS_CAP)

    return df


def _build_semester_conflict_report(df, semester_columns):
    """Find conflicting values within student-degree-semester groups."""
    conflict_frames = []
    for column in semester_columns:
        conflicts = (
            df.groupby(SEMESTER_KEY, dropna=False)[column]
            .nunique(dropna=False)
            .reset_index(name="unique_value_count")
        )
        conflicts = conflicts[conflicts["unique_value_count"].gt(1)].copy()
        if not conflicts.empty:
            conflicts.insert(0, "column", column)
            conflict_frames.append(conflicts)

    if conflict_frames:
        return pd.concat(conflict_frames, ignore_index=True)

    return pd.DataFrame(columns=["column"] + SEMESTER_KEY + ["unique_value_count"])


def _prior_consecutive_interruptions(flags):
    """Count immediately prior interruption semesters in one timeline."""
    counts = []
    current_run = 0
    for flag in flags.fillna(0).astype(int):
        counts.append(current_run)
        current_run = current_run + 1 if flag == 1 else 0
    return pd.Series(counts, index=flags.index)


def _sort_semester_frame(semester_df):
    """Sort timelines with numeric part_id plus text tiebreaker."""
    temp_names = _make_temp_names(
        semester_df.columns,
        ["part_sort_key", "part_sort_text"],
        "__fe_sort_",
    )
    part_sort_key = temp_names["part_sort_key"]
    part_sort_text = temp_names["part_sort_text"]

    semester_df[part_sort_key] = pd.to_numeric(semester_df["part_id"], errors="coerce")
    semester_df[part_sort_text] = semester_df["part_id"].astype("string")

    sorted_df = semester_df.sort_values(
        ["university_id", "student_id", "degree_id", part_sort_key, part_sort_text],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    sorted_df = sorted_df.drop(columns=[part_sort_key, part_sort_text])
    return sorted_df


def _active_semester_flags(semester_df):
    """Compute first-active-semester and no-previous-progress flags."""
    if "start_total" in semester_df.columns:
        start_total = _numeric_series(semester_df, "start_total", default=np.nan).fillna(0)
        no_previous_progress = start_total.eq(0)
    elif {"start_total_in_courses", "start_total_in_credits"}.issubset(semester_df.columns):
        start_courses = _numeric_series(semester_df, "start_total_in_courses", default=np.nan).fillna(0)
        start_credits = _numeric_series(semester_df, "start_total_in_credits", default=np.nan).fillna(0)
        no_previous_progress = start_courses.eq(0) & start_credits.eq(0)
    elif "start_total_in_courses" in semester_df.columns:
        start_courses = _numeric_series(semester_df, "start_total_in_courses", default=np.nan).fillna(0)
        no_previous_progress = start_courses.eq(0)
    elif "start_total_in_credits" in semester_df.columns:
        start_credits = _numeric_series(semester_df, "start_total_in_credits", default=np.nan).fillna(0)
        no_previous_progress = start_credits.eq(0)
    else:
        no_previous_progress = pd.Series(False, index=semester_df.index)

    semester_df["no_previous_progress"] = no_previous_progress.astype(int)
    semester_df["is_first_active_semester"] = no_previous_progress.astype(int)
    return semester_df


def add_valid_gpa_history_features(semester_df, gpa_points=None):
    """Add strictly-prior last-valid GPA and two-valid-observation trend."""
    if gpa_points is None:
        gpa_points = _numeric_series(semester_df, "gpa_points", default=np.nan)
    valid_gpa_for_history = gpa_points.where(
        gpa_points.gt(0).fillna(False) & semester_df["is_interruption_semester"].eq(0)
    )
    valid_gpa_group = valid_gpa_for_history.groupby(
        [semester_df[column] for column in STUDENT_DEGREE_KEY],
        sort=False,
        dropna=False,
    )
    last_valid_gpa = valid_gpa_group.transform(lambda values: values.ffill().shift(1))
    semester_df["last_valid_gpa_before_current_semester"] = last_valid_gpa

    # Build trend from the valid-observation subsequence, not from a twice-shifted
    # forward-filled series. At each valid row, last_valid_gpa is the preceding
    # valid observation. Keeping it only on valid rows, then forward-filling and
    # shifting once more, propagates the second-to-last valid observation while
    # remaining strictly before the current semester.
    previous_valid_gpa_at_valid_row = last_valid_gpa.where(valid_gpa_for_history.notna())
    second_last_valid_gpa = previous_valid_gpa_at_valid_row.groupby(
        [semester_df[column] for column in STUDENT_DEGREE_KEY],
        sort=False,
        dropna=False,
    ).transform(lambda values: values.ffill().shift(1))
    semester_df["gpa_trend_delta"] = last_valid_gpa - second_last_valid_gpa
    return semester_df


def build_semester_history(df_model_audit, diagnostics):
    """Build full-audit semester history before policy exclusions."""
    # Build semester history from the full audit frame so over-policy semesters
    # can still inform future GPA history before they are excluded from training.
    semester_columns = [column for column in SEMESTER_AGGREGATION_COLUMNS if column in df_model_audit.columns]
    conflict_report = _build_semester_conflict_report(df_model_audit, semester_columns)
    diagnostics["semester_conflict_report"] = conflict_report

    if conflict_report.empty:
        print("No semester-level conflicts detected before aggregation.")
    else:
        print(
            "WARNING: semester-level columns have conflicting values. "
            "Aggregation will use numeric max where appropriate."
        )
        safe_display(conflict_report.head(25))

    # Convert candidate semester-level columns to numeric before aggregation so
    # duplicate course rows collapse deterministically.
    semester_source = df_model_audit[SEMESTER_KEY].copy()
    for column in semester_columns:
        semester_source[column] = _numeric_series(df_model_audit, column, default=np.nan)

    # Use max for repeated semester values because these columns should be
    # constant per semester; conflicts are already reported above.
    agg_spec = {column: "max" for column in semester_columns}
    if agg_spec:
        semester_df = semester_source.groupby(
            SEMESTER_KEY,
            dropna=False,
            as_index=False,
            sort=False,
        ).agg(agg_spec)
    else:
        semester_df = (
            semester_source.groupby(SEMESTER_KEY, dropna=False, as_index=False, sort=False)
            .size()
            .drop(columns=["size"])
        )

    # Keep row counts for auditing how many course rows contributed to each
    # semester snapshot.
    semester_row_counts = (
        semester_source.groupby(SEMESTER_KEY, dropna=False, sort=False)
        .size()
        .reset_index(name="semester_row_count")
    )
    semester_df = semester_df.merge(
        semester_row_counts,
        on=SEMESTER_KEY,
        how="left",
        validate="one_to_one",
    )

    # Assert uniqueness before timeline features, because duplicated semester
    # keys would make previous-semester values ambiguous.
    duplicated_keys = semester_df.duplicated(SEMESTER_KEY, keep=False)
    assert not duplicated_keys.any(), "semester-level aggregation is not unique on semester_key"

    semester_df = _sort_semester_frame(semester_df)

    # Identify interruption semesters at semester grain so every course row in
    # that semester receives the same interruption context later.
    reg_credits = _numeric_series(semester_df, "semester_reg_credits", default=0).fillna(0)
    pass_credits = _numeric_series(semester_df, "semester_pass_credits", default=0).fillna(0)
    gpa_points = _numeric_series(semester_df, "gpa_points", default=np.nan)

    semester_df["is_interruption_semester"] = (
        reg_credits.gt(0) & pass_credits.eq(0) & gpa_points.eq(0).fillna(False)
    ).astype(int)

    # Derive prior interruption features by walking each student-degree timeline
    # after sorting by academic part_id.
    semester_group = semester_df.groupby(STUDENT_DEGREE_KEY, dropna=False, sort=False)
    semester_df["prev_semester_was_interruption"] = (
        semester_group["is_interruption_semester"].shift(1).fillna(0).astype(int)
    )
    semester_df["prior_interruption_count"] = (
        semester_group["is_interruption_semester"].cumsum() - semester_df["is_interruption_semester"]
    ).astype(int)
    semester_df["consecutive_interruption_count"] = (
        semester_group["is_interruption_semester"]
        .transform(_prior_consecutive_interruptions)
        .astype(int)
    )

    # Preserve both "no previous progress" and "first row in timeline" because
    # transfer/advanced-standing cases can make those concepts diverge.
    semester_df = _active_semester_flags(semester_df)
    semester_df["is_first_row_in_timeline"] = semester_group.cumcount().eq(0).astype(int)

    # Shift valid GPA history so the current semester's GPA never leaks into
    # features used to predict that same semester.
    semester_df = add_valid_gpa_history_features(semester_df, gpa_points)

    # Store timeline diagnostics because most data quality bugs in this pipeline
    # show up as grouping, sorting, or first-semester concept issues.
    timeline_diagnostics = {
        "student_degree_timeline_count": int(
            semester_df[STUDENT_DEGREE_KEY].drop_duplicates().shape[0]
        ),
        "part_sort_key_min": float(pd.to_numeric(semester_df["part_id"], errors="coerce").min())
        if pd.to_numeric(semester_df["part_id"], errors="coerce").notna().any()
        else np.nan,
        "part_sort_key_max": float(pd.to_numeric(semester_df["part_id"], errors="coerce").max())
        if pd.to_numeric(semester_df["part_id"], errors="coerce").notna().any()
        else np.nan,
        "first_semester_concept_mismatch_count": int(
            semester_df["is_first_active_semester"]
            .ne(semester_df["is_first_row_in_timeline"])
            .sum()
        ),
    }
    diagnostics["timeline_diagnostics"] = timeline_diagnostics

    print("Timeline diagnostics:")
    safe_display(pd.Series(timeline_diagnostics, name="value").to_frame())

    return semester_df


def merge_semester_history(df_model_audit, semester_df, diagnostics):
    """Merge semester-level features back with normalized many-to-one keys."""
    # Use temporary normalized keys so the merge is robust without overwriting
    # caller-visible ID columns.
    merge_labels = ["university", "student", "degree", "part"]
    temp_names = _make_temp_names(
        list(df_model_audit.columns) + list(semester_df.columns),
        merge_labels,
        "__fe_merge_",
    )
    left_merge_columns = [temp_names[label] for label in merge_labels]
    right_merge_columns = left_merge_columns

    key_map = dict(zip(merge_labels, SEMESTER_KEY))
    left = df_model_audit.copy()
    left_row_count = len(left)
    right = semester_df.copy()
    for label in merge_labels:
        key_column = key_map[label]
        temp_column = temp_names[label]
        left[temp_column] = _normalize_key_series(left[key_column])
        right[temp_column] = _normalize_key_series(right[key_column])

    # Prefix right-side features to avoid accidental collisions while validating
    # the merge shape.
    feature_rename = {column: f"__fe_semester_{column}" for column in SEMESTER_FEATURE_COLUMNS}
    right_features = right[right_merge_columns + SEMESTER_FEATURE_COLUMNS].rename(columns=feature_rename)

    assert not right_features.duplicated(right_merge_columns).any(), (
        "semester feature merge keys are not unique"
    )

    # Preview merge coverage before applying it so diagnostics can reveal missing
    # semester feature matches.
    merge_preview = left[left_merge_columns].merge(
        right_features[right_merge_columns],
        on=left_merge_columns,
        how="left",
        indicator=True,
        validate="many_to_one",
    )
    merge_counts = merge_preview["_merge"].value_counts(dropna=False).to_frame("row_count")
    diagnostics["semester_feature_merge_counts"] = merge_counts
    print("Semester feature merge check:")
    safe_display(merge_counts)

    left_only_keys = merge_preview.loc[
        merge_preview["_merge"].eq("left_only"),
        left_merge_columns,
    ].drop_duplicates()
    diagnostics["semester_feature_left_only_keys"] = left_only_keys
    if not left_only_keys.empty:
        print("WARNING: Some df_model_audit rows did not match semester_df. Example keys:")
        safe_display(left_only_keys.head(25))

    # Apply the many-to-one merge only after the right-side keys are confirmed
    # unique.
    merged = left.merge(
        right_features,
        on=left_merge_columns,
        how="left",
        validate="many_to_one",
        indicator="__fe_semester_feature_merge",
    )
    assert len(merged) == left_row_count, (
        "semester feature merge changed course-row count: "
        f"before={left_row_count}, after={len(merged)}"
    )

    # Move prefixed columns back to their canonical feature names.
    for column in SEMESTER_FEATURE_COLUMNS:
        merged[column] = merged[feature_rename[column]]

    # Trend remains NaN until two valid prior semester GPAs exist. Preserve
    # that distinction from a real flat trend (delta == 0.0) with a paired flag.
    merged["gpa_trend_missing"] = merged["gpa_trend_delta"].isna().astype(int)

    # Missing integer flags should be neutral zero; GPA history values remain nullable.
    nullable_features = {
        "last_valid_gpa_before_current_semester",
        "gpa_trend_delta",
    }
    integer_features = [
        column for column in SEMESTER_FEATURE_COLUMNS if column not in nullable_features
    ]
    for column in integer_features:
        merged[column] = merged[column].fillna(0).astype(int)

    temp_drop_columns = (
        left_merge_columns
        + list(feature_rename.values())
        + ["__fe_semester_feature_merge"]
    )
    merged = merged.drop(columns=temp_drop_columns, errors="ignore")
    return merged


def repair_previous_gpa_chain(df, diagnostics, structural_zero_as_nan=STRUCTURAL_ZERO_AS_NAN):
    """Repair previous-GPA features using the required fallback order."""
    # Keep the raw inputs separate from repaired outputs so diagnostics can
    # explain exactly which fallback supplied each row's previous GPA.
    raw_prev_gpa = _numeric_series(df, "prev_gpa_points", default=np.nan)
    last_valid_prev_gpa = _numeric_series(
        df,
        "last_valid_gpa_before_current_semester",
        default=np.nan,
    )
    start_agpa_points = _numeric_series(df, "start_agpa_points", default=np.nan)

    # Data-quality flags preserve whether the raw previous GPA was absent,
    # zero-valued, or suspicious for a non-first semester.
    df["prev_gpa_points_missing"] = raw_prev_gpa.isna().astype(int)
    df["prev_gpa_points_zero"] = raw_prev_gpa.eq(0).fillna(False).astype(int)
    df["prev_gpa_invalid_zero_case"] = (
        raw_prev_gpa.eq(0).fillna(False) & df["is_first_active_semester"].ne(1)
    ).astype(int)

    # Fill one source at a time in priority order so each row records the first
    # trustworthy GPA source available.
    clean = pd.Series(np.nan, index=df.index, dtype="float64")
    source = pd.Series(pd.NA, index=df.index, dtype="string")

    fallback_sources = [
        ("raw_prev_gpa", raw_prev_gpa),
        ("last_valid_gpa_before_current_semester", last_valid_prev_gpa),
        ("start_agpa_points", start_agpa_points),
    ]
    for source_name, source_values in fallback_sources:
        needs_value = clean.isna()
        source_has_value = source_values.gt(0).fillna(False)
        use_source = needs_value & source_has_value
        clean.loc[use_source] = source_values.loc[use_source]
        source.loc[use_source] = source_name

    # Zero fallback is a structural cold-start value; it is tracked separately
    # from true raw GPA values.
    needs_zero_fallback = clean.isna()
    clean.loc[needs_zero_fallback] = 0.0
    source.loc[needs_zero_fallback] = "zero_fallback"

    df["prev_gpa_points_clean"] = clean
    df["prev_gpa_fill_source"] = source.astype("string")
    df["prev_gpa_replaced_due_to_invalid_zero"] = (
        df["prev_gpa_invalid_zero_case"].eq(1)
        & df["prev_gpa_fill_source"].ne("raw_prev_gpa")
    ).astype(int)
    # placeholder, not implemented — must NOT be used as a model feature
    df["prev_gpa_actual_zero_performance"] = 0

    # Some models prefer structural cold-start zeros as missing values while the
    # audit column remains explicitly filled.
    if structural_zero_as_nan:
        df["model_prev_gpa"] = df["prev_gpa_points_clean"].where(
            df["prev_gpa_fill_source"].ne("zero_fallback"),
            np.nan,
        )

    # Hard diagnostics ensure the fallback chain never returns null model inputs.
    source_counts = df["prev_gpa_fill_source"].value_counts(dropna=False).to_frame("row_count")
    diagnostics["prev_gpa_fill_source_counts"] = source_counts
    diagnostics["prev_gpa_fill_source_null_count"] = int(df["prev_gpa_fill_source"].isna().sum())
    diagnostics["prev_gpa_points_clean_nan_count"] = int(df["prev_gpa_points_clean"].isna().sum())

    print("Previous-GPA chain diagnostics:")
    safe_display(source_counts)
    print("prev_gpa_fill_source null count:", diagnostics["prev_gpa_fill_source_null_count"])
    print("prev_gpa_points_clean NaN count:", diagnostics["prev_gpa_points_clean_nan_count"])

    assert diagnostics["prev_gpa_fill_source_null_count"] == 0, "prev_gpa_fill_source contains nulls"
    assert diagnostics["prev_gpa_points_clean_nan_count"] == 0, "prev_gpa_points_clean contains NaN"

    return df


def add_part_split_features(df):
    """Split part identifiers into year and semester components."""
    # Split compact academic term IDs into numeric fields that tree models can
    # compare directly.
    part_text = df["part_id"].astype("string").str.extract(r"(\d{4})\D*(\d+)")
    df["part_year"] = pd.to_numeric(part_text[0], errors="coerce")
    df["part_semester"] = pd.to_numeric(part_text[1], errors="coerce")

    if "start_part_id" in df.columns:
        start_part_text = df["start_part_id"].astype("string").str.extract(r"(\d{4})\D*(\d+)")
    else:
        start_part_text = pd.DataFrame(
            {0: pd.Series(pd.NA, index=df.index), 1: pd.Series(pd.NA, index=df.index)}
        )
    df["start_year"] = pd.to_numeric(start_part_text[0], errors="coerce")
    df["start_semester"] = pd.to_numeric(start_part_text[1], errors="coerce")
    return df


def add_start_level_features(df, diagnostics=None):
    """Convert start level labels to an ordinal year number.

    Fallback order per row:
    1. Numeric extraction from start_level_name_pl — handles plain digits ("3"),
       decimals ("3.0"), labels with digits ("level 3"), and Arabic-Indic digits (٣).
    2. English text map (first year / year 1 / third year / etc.).
    3. start_level_id numeric value (clipped to >= 0).
    4. 0 with start_level_missing = 1.
    """
    # Use a small controlled map after numeric extraction to support common
    # English labels without treating arbitrary text as level information.
    _LEVEL_TEXT_MAP = {
        "first year": 1, "year 1": 1, "1": 1,
        "second year": 2, "year 2": 2, "2": 2,
        "third year": 3, "year 3": 3, "3": 3,
        "fourth year": 4, "year 4": 4, "4": 4,
        "fifth year": 5, "year 5": 5, "5": 5,
        "sixth year": 6, "year 6": 6, "6": 6,
    }

    def _extract_level_int(val):
        """Extract first integer in 1..6 after translating Arabic-Indic digits."""
        if pd.isna(val):
            return np.nan
        translated = str(val).translate(_ARABIC_INDIC_TRANS)
        m = re.search(r"\d+", translated)
        if m:
            n = int(m.group())
            if 1 <= n <= 6:
                return float(n)
        return np.nan

    name_derived = None
    id_derived = None

    # Prefer the human-readable level name when present because it often carries
    # the clearest transferred/advanced-standing signal.
    if "start_level_name_pl" in df.columns:
        start_level_raw = df["start_level_name_pl"].astype("string").str.strip()
        start_level_clean = start_level_raw.str.lower()
        start_level_clean = start_level_clean.mask(start_level_clean.fillna("").eq(""))

        numeric_extracted = start_level_clean.map(_extract_level_int)
        text_mapped = pd.to_numeric(start_level_clean.map(_LEVEL_TEXT_MAP), errors="coerce")
        name_derived = numeric_extracted.where(numeric_extracted.notna(), text_mapped)

    # Fall back to the numeric level ID when text is unavailable or unparsable.
    if "start_level_id" in df.columns:
        id_series = _numeric_series(df, "start_level_id", default=np.nan)
        id_derived = id_series.clip(lower=0).where(id_series.notna())

    derived = (
        name_derived
        if name_derived is not None
        else pd.Series(np.nan, index=df.index, dtype="float64")
    )
    if id_derived is not None:
        derived = derived.where(derived.notna(), id_derived)

    df["start_level_missing"] = derived.isna().astype(int)
    df["start_level_ord"] = derived.fillna(0).astype(int)

    # Record parsing diagnostics so new level labels can be added deliberately.
    if diagnostics is not None:
        ord_counts = df["start_level_ord"].value_counts(dropna=False).sort_index()
        diagnostics["start_level_ord_counts"] = ord_counts
        print("start_level_ord distribution:")
        safe_display(ord_counts.to_frame("count"))

        if "start_level_name_pl" in df.columns:
            zero_mask = df["start_level_ord"].eq(0)
            if zero_mask.any():
                zero_examples = (
                    df.loc[zero_mask, "start_level_name_pl"]
                    .dropna()
                    .drop_duplicates()
                    .head(25)
                )
                diagnostics["start_level_zero_examples"] = zero_examples
                print("Raw start_level_name_pl values that mapped to 0 (up to 25 unique):")
                safe_display(zero_examples.to_frame())

        if name_derived is not None and id_derived is not None:
            both_valid = name_derived.notna() & id_derived.notna()
            mismatch_count = int((both_valid & name_derived.ne(id_derived)).sum())
            diagnostics["start_level_name_id_mismatch_count"] = mismatch_count
            print(f"start_level name-derived vs id-derived mismatch count: {mismatch_count}")

    return df


def add_requirement_features(df):
    """Build requirement missing, share, and size-bucket features."""
    # Requirement metadata gives the model curriculum context while preserving
    # missingness as an explicit signal.
    if "requirement_type_id" in df.columns:
        df["requirement_type_missing"] = df["requirement_type_id"].isna().astype(int)
    else:
        df["requirement_type_missing"] = 1

    requirement_credits = _numeric_series(
        df,
        "degree_requirement_credits_count",
        default=np.nan,
    )
    course_credits = _numeric_series(df, "course_credits", default=0).fillna(0)

    df["degree_requirement_credits_count_missing"] = requirement_credits.isna().astype(int)
    requirement_credits_filled = requirement_credits.fillna(0)

    # The share feature normalizes course credits by requirement size, avoiding
    # raw-credit comparisons across different degree structures.
    df["course_share_of_requirement"] = 0.0
    valid_requirement = requirement_credits_filled.gt(0)
    df.loc[valid_requirement, "course_share_of_requirement"] = (
        course_credits.loc[valid_requirement]
        / requirement_credits_filled.loc[valid_requirement]
    )

    df["requirement_size_bucket"] = (
        pd.cut(
            requirement_credits_filled,
            bins=[-0.01, 0, 12, 24, 60, np.inf],
            labels=["none_or_unknown", "small", "medium", "large", "very_large"],
        )
        .astype("string")
        .fillna("unknown")
    )
    return df


def add_fail_ratio_feature(df):
    """Build capped fail-credit ratio from prior credits plus fails."""
    # Ratio form makes fail history comparable across students with different
    # completed-credit totals.
    start_credits = _numeric_series(df, "start_total_in_credits", default=0).fillna(0)
    fail_credits = _numeric_series(df, "total_fail_credits_capped", default=0).fillna(0)
    denominator = start_credits + fail_credits

    df["fail_credit_ratio_capped"] = 0.0
    valid_ratio = denominator.gt(0)
    df.loc[valid_ratio, "fail_credit_ratio_capped"] = (
        fail_credits.loc[valid_ratio] / denominator.loc[valid_ratio]
    )
    return df


def add_degree_course_key(df):
    """Build the degree-course key for later diagnostics or encoding."""
    # Difficulty and requirement context are degree-specific, so retain a joined
    # key for lookups and downstream audits.
    if "course_id" in df.columns:
        course_key = _normalize_key_series(df["course_id"])
    else:
        course_key = pd.Series(MISSING_KEY_SENTINEL, index=df.index, dtype="string")
    degree_key = _normalize_key_series(df["degree_id"])
    df["degree_course_key"] = degree_key + "__" + course_key
    return df


def add_remaining_rowwise_features(df, diagnostics=None):
    """Build all remaining row-wise feature columns."""
    # Row-wise features do not require timeline aggregation and can be layered
    # after semester history is merged back to course rows.
    df = add_part_split_features(df)
    df = add_start_level_features(df, diagnostics=diagnostics)
    df = add_requirement_features(df)
    df = add_fail_ratio_feature(df)
    df = add_degree_course_key(df)
    return df


def report_suspicious_zero_fallback_rows(df, diagnostics):
    """Report zero-fallback rows for diagnostic review.

    Produces two separate reports:
    - suspicious: a returning student (is_first_active_semester == 0) who got
      zero_fallback — they should have prior history; worth investigating.
    - advanced_standing: a first-semester student admitted above level 1 who got
      zero_fallback — normal transfer/cold-start behaviour, NOT an alarm.
    """
    report_columns = [
        "university_id",
        "student_id",
        "degree_id",
        "part_id",
        "course_id",
        "prev_gpa_points",
        "last_valid_gpa_before_current_semester",
        "start_agpa_points",
        "prev_gpa_points_clean",
        "prev_gpa_fill_source",
        "is_first_active_semester",
        "is_first_row_in_timeline",
        "no_previous_progress",
        "start_level_ord",
    ]
    report_columns = [c for c in report_columns if c in df.columns]

    # Zero fallback is acceptable for true cold-start students, but suspicious
    # for returning students who should have prior GPA history.
    zero_fallback = df["prev_gpa_fill_source"].eq("zero_fallback")

    # Real alarm: returning student who should have history but does not.
    suspicious_mask = zero_fallback & df["is_first_active_semester"].eq(0)
    suspicious_report = df.loc[suspicious_mask, report_columns].copy()
    diagnostics["suspicious_zero_fallback_count"] = int(suspicious_mask.sum())
    diagnostics["suspicious_zero_fallback_rows"] = suspicious_report
    print(
        "Suspicious zero_fallback rows (returning student, no history):",
        diagnostics["suspicious_zero_fallback_count"],
    )
    if not suspicious_report.empty:
        safe_display(suspicious_report.head(25))

    # Informational: transfer/advanced-standing cold-start, not an alarm.
    start_level_ord = _numeric_series(df, "start_level_ord", default=0).fillna(0)
    advanced_mask = (
        zero_fallback
        & df["is_first_active_semester"].eq(1)
        & start_level_ord.ge(2)
    )
    advanced_report = df.loc[advanced_mask, report_columns].copy()
    diagnostics["advanced_standing_zero_fallback_count"] = int(advanced_mask.sum())
    diagnostics["advanced_standing_zero_fallback_rows"] = advanced_report
    print(
        "Advanced-standing cold-start zero_fallback rows (not suspicious):",
        diagnostics["advanced_standing_zero_fallback_count"],
    )
    if not advanced_report.empty:
        safe_display(advanced_report.head(25))

    return suspicious_report


def check_semester_stability(df_model_audit, diagnostics):
    """Warn if semester-level repaired columns vary within a semester group.

    Columns that are derived at semester granularity should be repeated (constant)
    across all course rows in the same (university_id, student_id, degree_id, part_id)
    group. Conflicts indicate a merge or aggregation issue upstream.
    This is a WARNING only — it never raises.
    """
    # Stable columns should be identical across all course rows for a semester;
    # variance here usually means aggregation or merge grain drifted.
    STABLE_COLUMNS = [
        "prev_gpa_points_clean",
        "prev_gpa_fill_source",
        "last_valid_gpa_before_current_semester",
        "is_interruption_semester",
        "prev_semester_was_interruption",
        "prior_interruption_count",
        "consecutive_interruption_count",
    ]
    present = [c for c in STABLE_COLUMNS if c in df_model_audit.columns]

    if not present:
        diagnostics["semester_stability_conflicts"] = {}
        print("Semester stability check: no columns to check.")
        return

    conflict_counts = {}
    example_frames = []

    # Scan each stable column independently so diagnostics point directly to the
    # feature whose semester-level contract was violated.
    for col in present:
        n_unique = (
            df_model_audit.groupby(SEMESTER_KEY, dropna=False)[col]
            .nunique(dropna=False)
        )
        n_conflicts = int((n_unique > 1).sum())
        conflict_counts[col] = n_conflicts
        if n_conflicts > 0 and len(example_frames) < 25:
            bad = (
                n_unique[n_unique > 1]
                .reset_index()
                .assign(column=col)
                .head(25 - len(example_frames))
            )
            example_frames.append(bad)

    diagnostics["semester_stability_conflicts"] = conflict_counts
    total = sum(conflict_counts.values())
    print("Semester stability conflicts (per column):")
    safe_display(pd.Series(conflict_counts, name="conflict_count").to_frame())

    if total > 0:
        print("WARNING: semester-level columns have conflicting values within semester groups.")
        if example_frames:
            safe_display(pd.concat(example_frames, ignore_index=True).head(25))
    else:
        print("Semester stability check: all columns stable within semester groups.")


def _last_valid_gpa_ratio(frame):
    # Small helper for merge-health diagnostics across intermediate frames.
    if len(frame) == 0 or "last_valid_gpa_before_current_semester" not in frame.columns:
        return 0.0
    return float(frame["last_valid_gpa_before_current_semester"].notna().mean())


def _build_audit_frame(df, diagnostics):
    """Build course rows with semester history before policy exclusions."""

    df_model_audit = df.copy(deep=True)
    df_model_audit = ensure_university_id(df_model_audit, diagnostics)
    df_model_audit = normalize_timeline_keys(df_model_audit)
    df_model_audit = add_policy_and_fail_flags(df_model_audit, diagnostics)
    semester_df = build_semester_history(df_model_audit, diagnostics)
    df_model_audit = merge_semester_history(
        df_model_audit,
        semester_df,
        diagnostics,
    )
    return df_model_audit, semester_df


def _add_row_features(df_model_audit, diagnostics, structural_zero_as_nan):
    """Repair prior GPA values and add course-level feature columns."""

    df_model_audit = repair_previous_gpa_chain(
        df_model_audit,
        diagnostics,
        structural_zero_as_nan=structural_zero_as_nan,
    )
    df_model_audit = add_remaining_rowwise_features(
        df_model_audit,
        diagnostics=diagnostics,
    )
    report_suspicious_zero_fallback_rows(df_model_audit, diagnostics)
    check_semester_stability(df_model_audit, diagnostics)
    return df_model_audit


def _split_policy_rows(df_model_audit):
    """Return model rows and over-policy rows without changing their order."""

    excluded = _boolean_series(
        df_model_audit,
        "exclude_over_policy_semester",
        default=False,
    )
    return (
        df_model_audit.loc[~excluded].copy(),
        df_model_audit.loc[excluded].copy(),
    )


def run_feature_engineering_job(df, structural_zero_as_nan=STRUCTURAL_ZERO_AS_NAN):
    """Run mark-prediction feature engineering on an in-memory dataframe."""
    # Fail fast on the public API boundary; every downstream step assumes pandas
    # indexing and vectorized operations.
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    # Track original shape and columns so the returned diagnostics can explain
    # what the job added, repaired, or excluded.
    diagnostics = {"warnings": []}
    original_columns = set(df.columns)
    original_row_count = int(len(df))

    print("Original df rows:", original_row_count)

    # The audit frame intentionally keeps rows that may later be excluded from
    # primary training. Timeline features are built at semester grain before
    # they are merged back to course rows.
    df_model_audit, semester_df = _build_audit_frame(df, diagnostics)

    # Compare GPA-history availability before and after merge to catch silent
    # key mismatches early.
    semester_ratio = _last_valid_gpa_ratio(semester_df)
    audit_ratio = _last_valid_gpa_ratio(df_model_audit)
    diagnostics["last_valid_gpa_non_null_ratios"] = {
        "semester_frame": semester_ratio,
        "df_model_audit": audit_ratio,
    }
    print("Last valid GPA non-null ratio in semester frame:", semester_ratio)
    print("Last valid GPA non-null ratio after merge in df_model_audit:", audit_ratio)

    df_model_audit = _add_row_features(
        df_model_audit,
        diagnostics,
        structural_zero_as_nan,
    )

    # Split the audit frame into the normal training set and the policy-excluded
    # set while keeping both available to callers.
    df_primary, df_excluded_over_policy = _split_policy_rows(df_model_audit)

    primary_ratio = _last_valid_gpa_ratio(df_primary)
    diagnostics["last_valid_gpa_non_null_ratios"]["df_primary"] = primary_ratio
    print("Last valid GPA non-null ratio in df_primary:", primary_ratio)

    # Guard against the highest-impact failure mode: semester history existed
    # but disappeared during merge or filtering.
    if semester_ratio > 0 and primary_ratio == 0:
        raise ValueError(
            "last_valid_gpa_before_current_semester merge failed: "
            "semester frame has values but df_primary has none."
        )

    # Publish compact row-count diagnostics for notebooks, tests, and future
    # monitoring hooks.
    row_counts = {
        "original_df": original_row_count,
        "df_model_audit": int(len(df_model_audit)),
        "df_primary": int(len(df_primary)),
        "df_excluded_over_policy": int(len(df_excluded_over_policy)),
    }
    diagnostics["row_counts"] = row_counts
    print("Row counts:")
    safe_display(pd.Series(row_counts, name="row_count").to_frame())

    # Report newly created or repaired feature columns so downstream model code
    # can quickly audit schema drift.
    added_columns = [column for column in df_model_audit.columns if column not in original_columns]
    repaired_columns = [
        column
        for column in FEATURE_COLUMNS
        if column in original_columns and column in df_model_audit.columns
    ]
    added_or_repaired_columns = list(dict.fromkeys(added_columns + repaired_columns))
    diagnostics["added_or_repaired_columns"] = added_or_repaired_columns

    print("Final added or repaired columns:")
    safe_display(pd.Series(added_or_repaired_columns, name="column").to_frame())

    return {
        "df_model_audit": df_model_audit,
        "df_primary": df_primary,
        "df_excluded_over_policy": df_excluded_over_policy,
        "diagnostics": diagnostics,
    }


__all__ = [
    "FAIL_CREDITS_CAP",
    "HIGH_COURSE_CREDITS_VALUE",
    "LEAKAGE_COLUMNS",
    "MAX_ALLOWED_SEMESTER_COURSES",
    "MAX_ALLOWED_SEMESTER_CREDITS",
    "STRUCTURAL_ZERO_AS_NAN",
    "add_valid_gpa_history_features",
    "assert_no_leakage_columns",
    "run_feature_engineering_job",
]
