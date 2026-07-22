"""Audit the leak-safe GPA trend feature on the true pre-split timeline.

The audit reruns feature engineering from the selected modeling population so
over-policy semesters remain available to past-only GPA history. It writes a
compact JSON/Markdown report and never mutates pipeline datasets.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import (  # noqa: E402
    MAX_ALLOWED_SEMESTER_COURSES,
    MAX_ALLOWED_SEMESTER_CREDITS,
    SEMESTER_AGGREGATION_COLUMNS,
    _derive_university_id,
    add_valid_gpa_history_features,
    add_part_split_features,
    ensure_university_id,
    normalize_timeline_keys,
)
from src.paths import (  # noqa: E402
    GPA_TREND_REPORTS_DIR,
    MODEL_DATA_DIR,
    SELECTED_MODEL_POPULATION_PATH,
    assert_data_root,
    model_split_path,
)


SEMESTER_KEY = ["university_id", "student_id", "degree_id", "part_id"]
STUDENT_DEGREE_KEY = ["university_id", "student_id", "degree_id"]
SPLITS = ("train", "valid", "test")


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if np.isnan(value) else value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_(no rows)_"

    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value).replace("|", "\\|")

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def _sort_semester_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = list(
        dict.fromkeys(
            SEMESTER_KEY
            + [
                "gpa_points",
                "is_interruption_semester",
                "exclude_over_policy_semester",
                "last_valid_gpa_before_current_semester",
                "gpa_trend_delta",
                "gpa_trend_missing",
                "attempt_number",
                "part_year",
                "part_semester",
            ]
        )
    )
    columns = [column for column in columns if column in frame.columns]
    semester = frame[columns].copy()
    attempt_max = None
    if "attempt_number" in semester.columns:
        attempt_max = (
            semester.groupby(SEMESTER_KEY, dropna=False, sort=False)["attempt_number"]
            .max()
            .reset_index()
        )
        semester = semester.drop(columns=["attempt_number"])
    semester = semester.drop_duplicates(SEMESTER_KEY, keep="first")
    if attempt_max is not None:
        semester = semester.merge(
            attempt_max,
            on=SEMESTER_KEY,
            how="left",
            validate="one_to_one",
        )
    semester["__part_num"] = pd.to_numeric(semester["part_id"], errors="coerce")
    semester["__part_text"] = semester["part_id"].astype("string")
    semester = semester.sort_values(
        STUDENT_DEGREE_KEY + ["__part_num", "__part_text"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    return semester.drop(columns=["__part_num", "__part_text"])


def _split_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    year = pd.to_numeric(frame["part_year"], errors="coerce")
    semester = pd.to_numeric(frame["part_semester"], errors="coerce")
    masks = {
        "train": year.between(2005, 2021, inclusive="both"),
        "valid": year.between(2022, 2023, inclusive="both"),
        "test": year.eq(2024) | (year.eq(2025) & semester.eq(1)),
    }
    return {name: frame.loc[mask].copy() for name, mask in masks.items()}


def _coverage_table(frame: pd.DataFrame, *, semester_grain: bool) -> pd.DataFrame:
    split_frames = _split_frames(frame)
    rows = []
    for name in SPLITS:
        split = split_frames[name]
        if semester_grain:
            split = split.drop_duplicates(SEMESTER_KEY, keep="first")
        has_t2 = split["gpa_trend_delta"].notna()
        has_t1 = split["last_valid_gpa_before_current_semester"].notna()
        rows.append(
            {
                "split": name,
                "n_rows": int(len(split)),
                "delta_computable": int(has_t2.sum()),
                "delta_computable_pct": float(has_t2.mean() * 100) if len(split) else np.nan,
                "t1_only": int((~has_t2 & has_t1).sum()),
                "neither": int((~has_t1).sum()),
                "missing_indicator": int(split["gpa_trend_missing"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _strictly_prior_last_valid(semester: pd.DataFrame) -> pd.Series:
    gpa = pd.to_numeric(semester["gpa_points"], errors="coerce")
    valid = gpa.gt(0) & semester["is_interruption_semester"].eq(0)
    valid_gpa = gpa.where(valid)
    return valid_gpa.groupby(
        [semester[column] for column in STUDENT_DEGREE_KEY],
        sort=False,
        dropna=False,
    ).transform(lambda values: values.ffill().shift(1))


def _rejected_shift2_diagnostics(full_semester: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Identify silent wrong-zero deltas from the rejected row-shift method."""
    semester = _sort_semester_frame(full_semester)
    gpa = pd.to_numeric(semester["gpa_points"], errors="coerce")
    valid = gpa.gt(0) & semester["is_interruption_semester"].eq(0)
    valid_gpa = gpa.where(valid)
    group_keys = [semester[column] for column in STUDENT_DEGREE_KEY]
    filled = valid_gpa.groupby(group_keys, sort=False, dropna=False).transform(
        lambda values: values.ffill()
    )
    wrong_t2 = filled.groupby(group_keys, sort=False, dropna=False).shift(2)
    wrong_delta = semester["last_valid_gpa_before_current_semester"] - wrong_t2
    correct_delta = semester["gpa_trend_delta"]
    comparable = correct_delta.notna() & wrong_delta.notna()
    mismatch = comparable & correct_delta.ne(wrong_delta)
    wrong_zero = mismatch & wrong_delta.eq(0)
    primary = ~semester["exclude_over_policy_semester"].astype(bool)
    wrong_keys = semester.loc[primary & wrong_zero, SEMESTER_KEY].copy()
    diagnostics = {
        "primary_semester_mismatch_count": int((primary & mismatch).sum()),
        "primary_semester_wrong_exact_zero_count": int((primary & wrong_zero).sum()),
    }
    return wrong_keys, diagnostics


def _equal_with_nan(left: pd.Series, right: pd.Series) -> pd.Series:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    return (left_num.isna() & right_num.isna()) | left_num.eq(right_num)


def _reconstruction_gap(
    audit: pd.DataFrame,
    primary: pd.DataFrame,
    *,
    course_split_dir: str | Path | None = None,
) -> dict[str, Any]:
    full_semester = _sort_semester_frame(audit)
    primary_semester = _sort_semester_frame(primary)

    # Reconstructing from primary-only history intentionally omits over-policy
    # semesters. The stored value was computed earlier on the full audit frame.
    primary_reconstruction = _strictly_prior_last_valid(primary_semester)
    stored = primary_semester["last_valid_gpa_before_current_semester"]
    mismatch = ~_equal_with_nan(stored, primary_reconstruction)

    full_gpa = pd.to_numeric(full_semester["gpa_points"], errors="coerce")
    full_valid = full_gpa.gt(0) & full_semester["is_interruption_semester"].eq(0)
    valid_source_was_excluded = full_semester["exclude_over_policy_semester"].where(full_valid)
    last_source_was_excluded = valid_source_was_excluded.groupby(
        [full_semester[column] for column in STUDENT_DEGREE_KEY],
        sort=False,
        dropna=False,
    ).transform(lambda values: values.ffill().shift(1))
    full_source = full_semester[SEMESTER_KEY].copy()
    full_source["last_valid_source_was_over_policy"] = (
        last_source_was_excluded.fillna(False).astype(bool)
    )
    primary_semester = primary_semester.merge(
        full_source,
        on=SEMESTER_KEY,
        how="left",
        validate="one_to_one",
    )
    attributed = mismatch & primary_semester["last_valid_source_was_over_policy"].fillna(False)

    mismatch_keys = primary_semester.loc[mismatch, SEMESTER_KEY]
    course_mismatch_by_split: dict[str, int] = {}
    if course_split_dir is None:
        course_mismatch_count = int(
            primary[SEMESTER_KEY]
            .merge(
                mismatch_keys.assign(__mismatch=1),
                on=SEMESTER_KEY,
                how="left",
                validate="many_to_one",
            )["__mismatch"]
            .fillna(0)
            .astype(bool)
            .sum()
        )
    else:
        split_dir = Path(course_split_dir)
        for split in SPLITS:
            keys = pd.read_parquet(
                model_split_path(split, "final", split_dir), columns=SEMESTER_KEY
            )
            split_count = int(
                keys.merge(
                    mismatch_keys.assign(__mismatch=1),
                    on=SEMESTER_KEY,
                    how="left",
                    validate="many_to_one",
                )["__mismatch"]
                .fillna(0)
                .astype(bool)
                .sum()
            )
            course_mismatch_by_split[split] = split_count
        course_mismatch_count = int(sum(course_mismatch_by_split.values()))

    return {
        "primary_semester_rows": int(len(primary_semester)),
        "semester_mismatch_count": int(mismatch.sum()),
        "semester_mismatch_pct": float(mismatch.mean() * 100),
        "course_row_mismatch_count": course_mismatch_count,
        "course_row_mismatch_by_split": course_mismatch_by_split,
        "mismatches_attributed_to_last_valid_over_policy_semester": int(attributed.sum()),
        "all_mismatches_attributed_to_over_policy_history": bool(
            int(attributed.sum()) == int(mismatch.sum())
        ),
    }


def _spot_check_table(audit: pd.DataFrame, n_students: int = 5) -> pd.DataFrame:
    semester = _sort_semester_frame(audit)
    group_meta = semester.groupby(STUDENT_DEGREE_KEY, dropna=False, sort=False).agg(
        n_semesters=("part_id", "size"),
        has_interruption=("is_interruption_semester", "max"),
        max_attempt=("attempt_number", "max") if "attempt_number" in semester.columns else ("part_id", "size"),
    )
    eligible = group_meta.loc[group_meta["n_semesters"].ge(3)]
    selected: list[tuple[Any, ...]] = []

    def add_first(mask: pd.Series) -> None:
        for key in eligible.loc[mask].index:
            key_tuple = key if isinstance(key, tuple) else (key,)
            if key_tuple not in selected:
                selected.append(key_tuple)
                return

    add_first(eligible["has_interruption"].eq(1))
    if "attempt_number" in semester.columns:
        add_first(pd.to_numeric(eligible["max_attempt"], errors="coerce").gt(1))
    for key in eligible.index:
        key_tuple = key if isinstance(key, tuple) else (key,)
        if key_tuple not in selected:
            selected.append(key_tuple)
        if len(selected) >= n_students:
            break

    if len(selected) < n_students:
        raise AssertionError(f"Only {len(selected)} enrollment timelines have at least 3 semesters")

    rows: list[dict[str, Any]] = []
    for spot_number, key in enumerate(selected[:n_students], start=1):
        mask = pd.Series(True, index=semester.index)
        for column, value in zip(STUDENT_DEGREE_KEY, key):
            mask &= semester[column].eq(value)
        timeline = semester.loc[mask]
        valid_history: list[float] = []
        for row in timeline.itertuples(index=False):
            oracle_t1 = valid_history[-1] if valid_history else np.nan
            oracle_t2 = valid_history[-2] if len(valid_history) >= 2 else np.nan
            oracle_delta = oracle_t1 - oracle_t2 if len(valid_history) >= 2 else np.nan
            pipeline_t1 = row.last_valid_gpa_before_current_semester
            pipeline_delta = row.gpa_trend_delta
            if not _equal_with_nan(pd.Series([pipeline_t1]), pd.Series([oracle_t1])).iloc[0]:
                raise AssertionError(f"Spot check {spot_number}: t-1 differs from oracle")
            if not _equal_with_nan(pd.Series([pipeline_delta]), pd.Series([oracle_delta])).iloc[0]:
                raise AssertionError(f"Spot check {spot_number}: delta differs from oracle")
            valid = bool(row.gpa_points > 0 and row.is_interruption_semester == 0)
            rows.append(
                {
                    "spot": spot_number,
                    "university_id": row.university_id,
                    "student_id": row.student_id,
                    "degree_id": row.degree_id,
                    "part_id": row.part_id,
                    "gpa_points": row.gpa_points,
                    "valid_gpa": int(valid),
                    "interruption": int(row.is_interruption_semester),
                    "attempt_number": getattr(row, "attempt_number", np.nan),
                    "pipeline_t1": pipeline_t1,
                    "oracle_t2": oracle_t2,
                    "pipeline_delta": pipeline_delta,
                    "oracle_delta": oracle_delta,
                }
            )
            if valid:
                valid_history.append(float(row.gpa_points))

    return pd.DataFrame(rows)


def build_semester_trend_from_source(
    input_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], int]:
    """Build the trend on a narrow pre-split frame to keep memory bounded."""
    input_path = Path(input_path)
    parquet_file = pq.ParquetFile(input_path)
    available = set(parquet_file.schema.names)
    required = set(SEMESTER_KEY).difference({"university_id"}) | {
        "gpa_points",
        "semester_reg_credits",
        "semester_reg_courses",
        "semester_pass_credits",
    }
    missing = sorted(required.difference(available))
    if missing:
        raise KeyError(f"Pre-split source is missing GPA trend columns: {missing}")
    desired = (
        SEMESTER_KEY
        + SEMESTER_AGGREGATION_COLUMNS
        + ["attempt_number"]
    )
    columns = [column for column in dict.fromkeys(desired) if column in available]
    source = pd.read_parquet(input_path, columns=columns)
    original_rows = int(len(source))
    diagnostics: dict[str, Any] = {"warnings": []}
    if "university_id" in source.columns:
        source = ensure_university_id(source, diagnostics)
    else:
        # The canonical selected-population parquet predates the explicit
        # university column. Derive it vectorially without the much wider
        # source-table suffix-conflict diagnostic used by the full job.
        source["university_id"] = _derive_university_id(source)
        diagnostics["university_id_source"] = "derived_from_id_suffix"
    source = normalize_timeline_keys(source)
    semester_columns = [
        column for column in SEMESTER_AGGREGATION_COLUMNS if column in source.columns
    ]
    numeric_source = source[SEMESTER_KEY].copy()
    for column in semester_columns:
        numeric_source[column] = pd.to_numeric(source[column], errors="coerce")
    if "attempt_number" in source.columns:
        numeric_source["attempt_number"] = pd.to_numeric(
            source["attempt_number"], errors="coerce"
        )

    # One grouped aggregation replaces the wide pipeline's per-column conflict
    # scan. It preserves the same max aggregation and fails if any semester-level
    # source varies within a semester.
    aggregations: dict[str, tuple[str, str]] = {}
    for column in semester_columns:
        aggregations[column] = (column, "max")
        aggregations[f"__{column}_nunique"] = (column, "nunique")
    if "attempt_number" in numeric_source.columns:
        aggregations["attempt_number"] = ("attempt_number", "max")
    aggregations["semester_row_count"] = ("gpa_points", "size")
    semester = (
        numeric_source.groupby(SEMESTER_KEY, dropna=False, sort=False)
        .agg(**aggregations)
        .reset_index()
    )
    conflict_columns = [f"__{column}_nunique" for column in semester_columns]
    conflicts = semester[conflict_columns].gt(1)
    if conflicts.any().any():
        counts = {
            column.removeprefix("__").removesuffix("_nunique"): int(
                conflicts[column].sum()
            )
            for column in conflict_columns
            if conflicts[column].any()
        }
        raise AssertionError(f"Semester-level source conflicts detected: {counts}")
    semester = semester.drop(columns=conflict_columns)
    expected_semester_columns = set(SEMESTER_KEY + semester_columns)
    missing_after_aggregation = sorted(expected_semester_columns.difference(semester.columns))
    if missing_after_aggregation:
        raise AssertionError(
            "Semester aggregation dropped required columns "
            f"{missing_after_aggregation}; available={semester.columns.tolist()}"
        )
    semester["__gpa_trend_part_num"] = pd.to_numeric(
        semester["part_id"], errors="coerce"
    )
    semester["__gpa_trend_part_text"] = semester["part_id"].astype("string")
    semester = semester.sort_values(
        STUDENT_DEGREE_KEY
        + ["__gpa_trend_part_num", "__gpa_trend_part_text"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    semester = semester.drop(
        columns=["__gpa_trend_part_num", "__gpa_trend_part_text"]
    )
    gpa_points = pd.to_numeric(semester["gpa_points"], errors="coerce")
    reg_credits = pd.to_numeric(semester["semester_reg_credits"], errors="coerce")
    pass_credits = pd.to_numeric(semester["semester_pass_credits"], errors="coerce")
    semester["is_interruption_semester"] = (
        reg_credits.gt(0) & pass_credits.eq(0) & gpa_points.eq(0).fillna(False)
    ).astype(int)
    semester = add_valid_gpa_history_features(semester, gpa_points)
    reg_credits = pd.to_numeric(semester["semester_reg_credits"], errors="coerce")
    reg_courses = pd.to_numeric(semester["semester_reg_courses"], errors="coerce")
    semester["exclude_over_policy_semester"] = (
        reg_credits.gt(MAX_ALLOWED_SEMESTER_CREDITS)
        | reg_courses.gt(MAX_ALLOWED_SEMESTER_COURSES)
    )
    semester["gpa_trend_missing"] = semester["gpa_trend_delta"].isna().astype(int)
    semester = add_part_split_features(semester)
    primary = semester.loc[~semester["exclude_over_policy_semester"]].copy()
    return semester, primary, diagnostics, original_rows


def map_trend_to_course_keys(
    course_keys: pd.DataFrame,
    primary_semester: pd.DataFrame,
) -> pd.DataFrame:
    """Map semester trend columns to course rows with a many-to-one guard."""
    mapping = primary_semester[
        SEMESTER_KEY
        + [
            "last_valid_gpa_before_current_semester",
            "gpa_trend_delta",
            "gpa_trend_missing",
        ]
    ]
    if mapping.duplicated(SEMESTER_KEY).any():
        raise AssertionError("Primary GPA trend mapping is not unique at semester grain")
    original_index = course_keys.index
    mapped = course_keys[SEMESTER_KEY].merge(
        mapping,
        on=SEMESTER_KEY,
        how="left",
        validate="many_to_one",
        indicator=True,
        sort=False,
    )
    if len(mapped) != len(course_keys):
        raise AssertionError("GPA trend merge changed the course-row count")
    unmatched = int(mapped["_merge"].ne("both").sum())
    if unmatched:
        raise AssertionError(f"GPA trend mapping left {unmatched} course rows unmatched")
    features = mapped[
        [
            "last_valid_gpa_before_current_semester",
            "gpa_trend_delta",
            "gpa_trend_missing",
        ]
    ].copy()
    features.index = original_index
    return features


def _course_coverage_from_split_files(
    primary_semester: pd.DataFrame,
    split_dir: str | Path,
    wrong_shift_keys: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    split_dir = Path(split_dir)
    rows = []
    train_delta = pd.Series(dtype="float64")
    for name in SPLITS:
        path = model_split_path(name, "final", split_dir)
        keys = pd.read_parquet(path, columns=SEMESTER_KEY)
        features = map_trend_to_course_keys(keys, primary_semester)
        has_t2 = features["gpa_trend_delta"].notna()
        has_t1 = features["last_valid_gpa_before_current_semester"].notna()
        rows.append(
            {
                "split": name,
                "n_rows": int(len(features)),
                "delta_computable": int(has_t2.sum()),
                "delta_computable_pct": float(has_t2.mean() * 100),
                "t1_only": int((~has_t2 & has_t1).sum()),
                "neither": int((~has_t1).sum()),
                "missing_indicator": int(features["gpa_trend_missing"].sum()),
                "wrong_zero_under_shift2": int(
                    keys.merge(
                        wrong_shift_keys.assign(__wrong_shift=1),
                        on=SEMESTER_KEY,
                        how="left",
                        validate="many_to_one",
                    )["__wrong_shift"]
                    .fillna(0)
                    .sum()
                )
                if wrong_shift_keys is not None
                else 0,
            }
        )
        if name == "train":
            train_delta = features["gpa_trend_delta"].dropna().astype("float64")
    return pd.DataFrame(rows), train_delta


def create_semester_audit_report(
    full_semester: pd.DataFrame,
    primary_semester: pd.DataFrame,
    output_dir: str | Path,
    *,
    source_path: str | Path,
    source_course_rows: int,
    split_dir: str | Path = MODEL_DATA_DIR,
) -> dict[str, Any]:
    """Persist the complete audit from memory-bounded semester calculations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    wrong_shift_keys, wrong_shift_diagnostics = _rejected_shift2_diagnostics(
        full_semester
    )
    course_coverage, train_delta = _course_coverage_from_split_files(
        primary_semester,
        split_dir,
        wrong_shift_keys=wrong_shift_keys,
    )
    semester_coverage = _coverage_table(primary_semester, semester_grain=True)
    reconstructed_t1 = _strictly_prior_last_valid(full_semester)
    t1_consistent = _equal_with_nan(
        full_semester["last_valid_gpa_before_current_semester"], reconstructed_t1
    )
    if not t1_consistent.all():
        raise AssertionError("Produced t-1 does not match the existing last-valid GPA feature")
    if not full_semester["gpa_trend_missing"].equals(
        full_semester["gpa_trend_delta"].isna().astype(int)
    ):
        raise AssertionError("gpa_trend_missing does not exactly match delta nullness")

    quantiles = train_delta.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    distribution = {
        "n": int(len(train_delta)),
        "min": float(train_delta.min()),
        "max": float(train_delta.max()),
        "mean": float(train_delta.mean()),
        "q05": float(quantiles.loc[0.05]),
        "q25": float(quantiles.loc[0.25]),
        "q50": float(quantiles.loc[0.50]),
        "q75": float(quantiles.loc[0.75]),
        "q95": float(quantiles.loc[0.95]),
        "exact_zero_count": int(train_delta.eq(0).sum()),
        "exact_zero_pct": float(train_delta.eq(0).mean() * 100),
    }
    if distribution["min"] < -4 or distribution["max"] > 4:
        raise AssertionError("GPA trend delta lies outside the plausible [-4, 4] range")

    spot_checks = _spot_check_table(full_semester)
    gap = _reconstruction_gap(
        full_semester,
        primary_semester,
        course_split_dir=split_dir,
    )
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_path": str(source_path),
        "strictly_before_mechanism": "grouped ffill followed by shift(1)",
        "row_counts": {
            "source_course_rows": source_course_rows,
            "full_semesters": int(len(full_semester)),
            "primary_semesters": int(len(primary_semester)),
            "excluded_over_policy_semesters": int(
                full_semester["exclude_over_policy_semester"].sum()
            ),
        },
        "course_coverage": course_coverage.to_dict(orient="records"),
        "semester_coverage": semester_coverage.to_dict(orient="records"),
        "train_delta_distribution": distribution,
        "rejected_shift2": wrong_shift_diagnostics,
        "reconstruction_gap": gap,
        "t1_consistency": {
            "semester_rows_checked": int(len(full_semester)),
            "exact_matches_including_nulls": int(t1_consistent.sum()),
        },
        "persisted_candidate_dtypes": {
            "gpa_trend_delta": str(primary_semester["gpa_trend_delta"].dtype),
            "gpa_trend_missing": str(primary_semester["gpa_trend_missing"].dtype),
        },
        "spot_checks": spot_checks.to_dict(orient="records"),
    }
    (output_dir / "gpa_trend_audit.json").write_text(
        json.dumps(
            _json_safe(report),
            indent=2,
            ensure_ascii=False,
            default=_json_value,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    markdown = f"""# GPA trend feature audit

Created: {report['created_at']}

The feature was computed on the true pre-split, unique-semester timeline. Both
prior GPAs are strictly before the current semester via grouped
`ffill().shift(1)` operations.

## Course-row coverage

{_markdown_table(course_coverage)}

## Enrollment-semester coverage

{_markdown_table(semester_coverage)}

## Train delta distribution

{_markdown_table(pd.DataFrame([distribution]))}

## Reconstruction-gap attribution

{_markdown_table(pd.DataFrame([gap]))}

## Five manual/oracle spot checks

{_markdown_table(spot_checks)}
"""
    (output_dir / "GPA_TREND_AUDIT.md").write_text(markdown, encoding="utf-8")
    return report


def run_audit(
    input_path: str | Path = SELECTED_MODEL_POPULATION_PATH,
    output_dir: str | Path | None = None,
    split_dir: str | Path = MODEL_DATA_DIR,
) -> tuple[dict[str, Any], Path]:
    input_path = Path(input_path)
    assert_data_root(input_path)
    if output_dir is None:
        run_id = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S__gpa_trend")
        output_dir = GPA_TREND_REPORTS_DIR / run_id
    output_dir = Path(output_dir)
    full_semester, primary_semester, _, source_rows = build_semester_trend_from_source(
        input_path
    )
    report = create_semester_audit_report(
        full_semester,
        primary_semester,
        output_dir,
        source_path=input_path,
        source_course_rows=source_rows,
        split_dir=split_dir,
    )
    return report, output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=SELECTED_MODEL_POPULATION_PATH,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split-dir", type=Path, default=MODEL_DATA_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report, output_dir = run_audit(args.input, args.output_dir, args.split_dir)
    print(f"GPA trend audit passed: {output_dir}")
    print(json.dumps(report["reconstruction_gap"], indent=2))


if __name__ == "__main__":
    main()
