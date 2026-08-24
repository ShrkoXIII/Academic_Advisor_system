"""Helpers for building the two historical tables used by KNN.

The public functions in this module keep validation and dataframe mechanics out
of the command-line builder.  Every prior-attempt value is shifted within the
student/degree/course timeline, so the current course outcome never describes
its own history.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEMESTER_KEY = ["university_id", "student_id", "degree_id", "part_id"]
COURSE_TIMELINE_KEY = ["university_id", "student_id", "degree_id", "course_id"]

TRAIN_COLUMNS = [
    "student_course_id",
    "student_status_id",
    *SEMESTER_KEY,
    "course_id",
    "grade_id",
    "final_mark",
    "course_credits",
    "attempt_number",
    "prev_gpa_points",
    "gpa_points",
    "start_agpa_points",
    "start_total_in_courses",
    "start_total_in_credits",
    "semester_reg_credits",
    "semester_reg_courses",
    "semester_pass_credits",
    "total_fail_credits",
    "reg_total_semesters",
    "start_level_ord",
    "is_first_active_semester",
    "model_prev_gpa",
    "last_valid_gpa_before_current_semester",
    "fail_credit_ratio_capped",
    "prior_interruption_count",
    "consecutive_interruption_count",
    "prev_semester_was_interruption",
    "part_semester",
    "diploma_gpa",
    "diploma_type_bucket",
]

STATUS_COLUMNS = [
    "student_status_id",
    "end_agpa_points",
    "end_total_in_courses",
    "end_total_in_credits",
]

LEVEL_STATUS_COLUMNS = [
    "start_level_name_short",
    "end_level_name_short",
]

GRADE_COLUMNS = ["grade_id", "finish_status", "grade_show"]


def require_columns(frame: pd.DataFrame, columns: list[str], *, name: str) -> None:
    """Raise one readable error when an input is missing required columns."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def normalize_identifier(series: pd.Series) -> pd.Series:
    """Normalize numeric/string database identifiers to trimmed strings."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
    )


def assert_unique(frame: pd.DataFrame, key: list[str], *, name: str) -> None:
    """Ensure a produced table has exactly one row at its documented grain."""
    duplicated = frame.duplicated(key, keep=False)
    if duplicated.any():
        sample = frame.loc[duplicated, key].head(5).to_dict("records")
        raise ValueError(f"{name} is not unique on {key}; sample={sample}")


def ensure_new_output_files(output_paths: list[Path]) -> None:
    """Refuse to overwrite an already-built history version."""
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "History output already exists; choose a new --output-dir: "
            + ", ".join(existing)
        )


def attach_official_references(
    train: pd.DataFrame,
    student_status: pd.DataFrame,
    grades: pd.DataFrame,
) -> pd.DataFrame:
    """Attach official end-of-semester AGPA and pass/fail grade meanings."""
    require_columns(train, TRAIN_COLUMNS, name="train")
    require_columns(student_status, STATUS_COLUMNS, name="student_status")
    require_columns(grades, GRADE_COLUMNS, name="grades")

    result = train.copy()
    for column in [
        "student_course_id",
        "student_status_id",
        *SEMESTER_KEY,
        "course_id",
        "grade_id",
    ]:
        result[column] = normalize_identifier(result[column])

    has_any_level_column = any(
        column in student_status.columns for column in LEVEL_STATUS_COLUMNS
    )
    has_all_level_columns = all(
        column in student_status.columns for column in LEVEL_STATUS_COLUMNS
    )
    if has_any_level_column and not has_all_level_columns:
        require_columns(
            student_status,
            LEVEL_STATUS_COLUMNS,
            name="student_status level fields",
        )
    status_columns = STATUS_COLUMNS + (
        LEVEL_STATUS_COLUMNS if has_all_level_columns else []
    )
    status_ref = student_status[status_columns].copy()
    status_ref["student_status_id"] = normalize_identifier(
        status_ref["student_status_id"]
    )
    assert_unique(status_ref, ["student_status_id"], name="student_status")

    grade_ref = grades[GRADE_COLUMNS].copy()
    grade_ref["grade_id"] = normalize_identifier(grade_ref["grade_id"])
    grade_ref["finish_status"] = (
        grade_ref["finish_status"].astype("string").str.strip().str.upper()
    )
    grade_ref["grade_show"] = (
        grade_ref["grade_show"].astype("string").str.strip().str.upper()
    )
    assert_unique(grade_ref, ["grade_id"], name="grades")

    result = result.merge(
        status_ref,
        on="student_status_id",
        how="left",
        validate="many_to_one",
    )
    result = result.merge(
        grade_ref,
        on="grade_id",
        how="left",
        validate="many_to_one",
    )

    if result["finish_status"].isna().any():
        raise ValueError("Some TRAIN grade_id values are absent from the grade reference")
    if result["end_agpa_points"].isna().any():
        raise ValueError(
            "Some TRAIN student_status_id values have no official end_agpa_points"
        )
    if has_all_level_columns:
        for column in LEVEL_STATUS_COLUMNS:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype(
                "Int64"
            )
            if result[column].isna().any():
                raise ValueError(
                    f"Some TRAIN student_status_id values have no official {column}"
                )
        derived_start_level = pd.to_numeric(
            result["start_level_ord"], errors="coerce"
        ).astype("Int64")
        mismatch = result["start_level_name_short"].ne(derived_start_level)
        if mismatch.any():
            raise ValueError(
                "Official start_level_name_short does not match TRAIN start_level_ord "
                f"for {int(mismatch.sum())} rows"
            )

    result["is_passed"] = result["finish_status"].eq("P").astype("int8")
    result["is_failed"] = (
        result["finish_status"].isin(["F", "FE"])
        | result["grade_show"].eq("F")
    ).astype("int8")
    result["outcome_status"] = pd.Series(
        np.select(
            [result["is_passed"].eq(1), result["is_failed"].eq(1)],
            ["passed", "failed"],
            default="other",
        ),
        index=result.index,
        dtype="string",
    )
    return result


def _sort_course_timelines(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["__part_sort"] = pd.to_numeric(result["part_id"], errors="coerce")
    result["__attempt_sort"] = pd.to_numeric(
        result["attempt_number"], errors="coerce"
    )
    return result.sort_values(
        COURSE_TIMELINE_KEY
        + ["__part_sort", "part_id", "__attempt_sort", "student_course_id"],
        kind="stable",
    ).reset_index(drop=True)


def build_student_semester_courses(enriched: pd.DataFrame) -> pd.DataFrame:
    """Build one row per completed TRAIN course with past-only attempt history."""
    required = [
        "student_course_id",
        *SEMESTER_KEY,
        "course_id",
        "course_credits",
        "attempt_number",
        "grade_id",
        "grade_show",
        "final_mark",
        "finish_status",
        "outcome_status",
        "is_passed",
        "is_failed",
    ]
    require_columns(enriched, required, name="enriched train")

    result = _sort_course_timelines(enriched)
    grouped = result.groupby(COURSE_TIMELINE_KEY, dropna=False, sort=False)

    result["course_attempts_prior"] = grouped.cumcount().astype("Int64")
    result["__attempt_max_including"] = grouped["attempt_number"].cummax()
    result["course_max_attempt_number_prior"] = (
        result.groupby(COURSE_TIMELINE_KEY, dropna=False, sort=False)[
            "__attempt_max_including"
        ]
        .shift(1)
        .astype("Int64")
    )

    result["course_last_mark_prior"] = grouped["final_mark"].shift(1)
    result["__best_mark_including"] = grouped["final_mark"].cummax()
    result["course_best_mark_prior"] = result.groupby(
        COURSE_TIMELINE_KEY, dropna=False, sort=False
    )["__best_mark_including"].shift(1)

    result["__mark_value"] = pd.to_numeric(
        result["final_mark"], errors="coerce"
    ).fillna(0.0)
    result["__mark_seen"] = result["final_mark"].notna().astype("int64")
    prior_mark_sum = (
        result.groupby(COURSE_TIMELINE_KEY, dropna=False, sort=False)[
            "__mark_value"
        ].cumsum()
        - result["__mark_value"]
    )
    prior_mark_count = (
        result.groupby(COURSE_TIMELINE_KEY, dropna=False, sort=False)[
            "__mark_seen"
        ].cumsum()
        - result["__mark_seen"]
    )
    result["course_mean_mark_prior"] = prior_mark_sum.div(
        prior_mark_count.where(prior_mark_count.gt(0))
    )

    result["course_failures_prior"] = (
        result.groupby(COURSE_TIMELINE_KEY, dropna=False, sort=False)[
            "is_failed"
        ].cumsum()
        - result["is_failed"]
    ).astype("Int64")
    result["course_last_status_prior"] = grouped["outcome_status"].shift(1)
    result["course_last_attempt_part"] = grouped["part_id"].shift(1)
    result["is_retake"] = result["course_attempts_prior"].gt(0).astype("int8")

    columns = [
        "student_course_id",
        *SEMESTER_KEY,
        "course_id",
        "course_credits",
        "attempt_number",
        "course_attempts_prior",
        "course_max_attempt_number_prior",
        "course_last_mark_prior",
        "course_best_mark_prior",
        "course_mean_mark_prior",
        "course_failures_prior",
        "course_last_status_prior",
        "course_last_attempt_part",
        "is_retake",
        "grade_id",
        "grade_show",
        "final_mark",
        "finish_status",
        "outcome_status",
        "is_passed",
        "is_failed",
    ]
    output = result[columns].copy()
    assert_unique(output, ["student_course_id"], name="student_semester_courses")
    return output


def _nullable_positive_flag(values: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=values.index, dtype="Int8")
    available = values.notna()
    result.loc[available] = values.loc[available].gt(0).astype("int8")
    return result


def build_student_semester_outcomes(enriched: pd.DataFrame) -> pd.DataFrame:
    """Build one pre-state plus one official outcome row per student semester."""
    has_official_levels = all(
        column in enriched.columns for column in LEVEL_STATUS_COLUMNS
    )
    snapshot_columns = [
        "student_status_id",
        "start_agpa_points",
        "prev_gpa_points",
        "last_valid_gpa_before_current_semester",
        "model_prev_gpa",
        "start_level_ord",
        "start_total_in_courses",
        "start_total_in_credits",
        "reg_total_semesters",
        "total_fail_credits",
        "fail_credit_ratio_capped",
        "prior_interruption_count",
        "consecutive_interruption_count",
        "prev_semester_was_interruption",
        "is_first_active_semester",
        "part_semester",
        "diploma_gpa",
        "diploma_type_bucket",
        "semester_reg_courses",
        "semester_reg_credits",
        "semester_pass_credits",
        "gpa_points",
        "end_agpa_points",
        "end_total_in_courses",
        "end_total_in_credits",
    ]
    if has_official_levels:
        snapshot_columns.extend(LEVEL_STATUS_COLUMNS)
    required = [
        "student_course_id",
        *SEMESTER_KEY,
        "course_id",
        "course_credits",
        "final_mark",
        "is_passed",
        "is_failed",
        *snapshot_columns,
    ]
    require_columns(enriched, required, name="enriched train")

    grouped = enriched.groupby(SEMESTER_KEY, dropna=False, sort=False)
    snapshots = grouped[snapshot_columns].first().reset_index()
    observed = grouped.agg(
        observed_course_count=("student_course_id", "count"),
        observed_course_credit_sum=("course_credits", "sum"),
        semester_average_mark=("final_mark", "mean"),
        passed_course_count=("is_passed", "sum"),
        failed_course_count=("is_failed", "sum"),
    ).reset_index()

    result = snapshots.merge(
        observed,
        on=SEMESTER_KEY,
        how="inner",
        validate="one_to_one",
    )
    result = result.rename(
        columns={
            "start_agpa_points": "cumulative_gpa_before",
            "prev_gpa_points": "previous_term_gpa",
            "last_valid_gpa_before_current_semester": "previous_valid_term_gpa",
            "model_prev_gpa": "model_previous_term_gpa",
            "start_level_ord": "academic_level",
            "start_total_in_courses": "completed_courses_before",
            "start_total_in_credits": "completed_credits_before",
            "total_fail_credits": "failed_credits_before",
            "semester_reg_courses": "registered_course_count",
            "semester_reg_credits": "registered_credit_count",
            "semester_pass_credits": "passed_credit_count",
            "gpa_points": "term_gpa",
            "end_agpa_points": "cumulative_gpa_after",
            "end_total_in_courses": "completed_courses_after",
            "end_total_in_credits": "completed_credits_after",
        }
    )
    result["prior_semester_count"] = (
        pd.to_numeric(result["reg_total_semesters"], errors="coerce") - 1
    ).clip(lower=0)
    result["term_gpa_delta"] = (
        result["term_gpa"] - result["previous_valid_term_gpa"]
    )
    result["cumulative_gpa_delta"] = (
        result["cumulative_gpa_after"] - result["cumulative_gpa_before"]
    )
    result["term_gpa_improved"] = _nullable_positive_flag(
        result["term_gpa_delta"]
    )
    result["cumulative_gpa_improved"] = _nullable_positive_flag(
        result["cumulative_gpa_delta"]
    )
    if has_official_levels:
        result["academic_level_before"] = pd.to_numeric(
            result["start_level_name_short"], errors="coerce"
        ).astype("Int64")
        result["academic_level_after"] = pd.to_numeric(
            result["end_level_name_short"], errors="coerce"
        ).astype("Int64")
        result["academic_level_delta"] = (
            result["academic_level_after"] - result["academic_level_before"]
        ).astype("Int64")
        result["academic_level_advanced"] = (
            result["academic_level_delta"].gt(0).astype("int8")
        )
    result["any_course_failed"] = result["failed_course_count"].gt(0).astype("int8")
    result["all_courses_passed"] = (
        result["observed_course_count"].gt(0)
        & result["passed_course_count"].eq(result["observed_course_count"])
    ).astype("int8")

    level_output_columns = (
        [
            "academic_level_before",
            "academic_level_after",
            "academic_level_delta",
            "academic_level_advanced",
        ]
        if has_official_levels
        else ["academic_level"]
    )
    columns = [
        *SEMESTER_KEY,
        "student_status_id",
        "cumulative_gpa_before",
        "previous_term_gpa",
        "previous_valid_term_gpa",
        "model_previous_term_gpa",
        *level_output_columns,
        "completed_courses_before",
        "completed_credits_before",
        "prior_semester_count",
        "failed_credits_before",
        "fail_credit_ratio_capped",
        "prior_interruption_count",
        "consecutive_interruption_count",
        "prev_semester_was_interruption",
        "is_first_active_semester",
        "part_semester",
        "diploma_gpa",
        "diploma_type_bucket",
        "registered_course_count",
        "registered_credit_count",
        "observed_course_count",
        "observed_course_credit_sum",
        "passed_credit_count",
        "semester_average_mark",
        "term_gpa",
        "cumulative_gpa_after",
        "completed_courses_after",
        "completed_credits_after",
        "term_gpa_delta",
        "cumulative_gpa_delta",
        "term_gpa_improved",
        "cumulative_gpa_improved",
        "passed_course_count",
        "failed_course_count",
        "any_course_failed",
        "all_courses_passed",
    ]
    output = result[columns].sort_values(
        SEMESTER_KEY, kind="stable"
    ).reset_index(drop=True)
    assert_unique(output, SEMESTER_KEY, name="student_semester_outcomes")
    return output


__all__ = [
    "GRADE_COLUMNS",
    "LEVEL_STATUS_COLUMNS",
    "SEMESTER_KEY",
    "STATUS_COLUMNS",
    "TRAIN_COLUMNS",
    "assert_unique",
    "attach_official_references",
    "build_student_semester_courses",
    "build_student_semester_outcomes",
    "ensure_new_output_files",
    "normalize_identifier",
    "require_columns",
]
