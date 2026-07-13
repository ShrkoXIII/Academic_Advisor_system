"""Read-only A2/A3 diagnostics for missed failures in the reference M1 model.

The script loads an already-trained LightGBM model and the persisted feature
contract.  It never trains or saves a model and never changes source parquet
files or reference-run artifacts.  Its only writes are four new files inside a
new, uniquely named diagnostics directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import model_training as training


REFERENCE_RUN_NAME = "2026-07-12_1513__remove-dead-const"
TARGET = "final_mark"
THRESHOLD = 0.5
MISSING_GROUP = "<MISSING>"
SEGMENT_COLUMNS = (
    "part_year",
    "degree_id",
    "requirement_type_id",
    "attempt_number",
    "is_first_active_semester",
    "no_previous_progress",
    "difficulty_fallback_level",
    "course_difficulty_missing",
)
OVERLAP_SAMPLE_COLUMNS = (
    "student_id",
    "degree_id",
    "part_id",
    "is_first_active_semester",
    "no_previous_progress",
    "prev_gpa_points_clean",
    "last_valid_gpa_before_current_semester",
)
MARK_BANDS = (
    ("final_mark < 30", lambda values: values.lt(30)),
    ("30 <= final_mark < 40", lambda values: values.ge(30) & values.lt(40)),
    ("40 <= final_mark < 45", lambda values: values.ge(40) & values.lt(45)),
    ("45 <= final_mark < 50", lambda values: values.ge(45) & values.lt(50)),
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    run_dir = root / "models" / "runs" / REFERENCE_RUN_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=run_dir)    
    parser.add_argument(
        "--valid",
        type=Path,
        default=root / "data" / "model_data" / "df_valid_final.parquet",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=root / "data" / "model_data" / "df_train_final.parquet",
        help="Read only to rebuild the train-derived A1 course-coverage key set.",
    )
    parser.add_argument(
        "--feature-engineering-source",
        type=Path,
        default=root / "src" / "feature_engineering.py",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "models" / "diagnostics",
    )
    parser.add_argument("--mismatch-sample-size", type=int, default=25)
    return parser.parse_args()


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required file(s) missing:\n- " + "\n- ".join(missing))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        contract = json.load(handle)
    required = {
        "features",
        "categorical_features",
        "categorical_levels",
        "unknown_category_code",
    }
    absent = sorted(required - set(contract))
    if absent:
        raise ValueError(f"Feature contract is missing keys: {absent}")
    features = contract["features"]
    if len(features) != contract.get("n_features", len(features)):
        raise ValueError("Feature contract feature count is inconsistent.")
    return contract


def prepare_features(
    df: pd.DataFrame,
    contract: dict[str, Any],
    model_features: list[str],
) -> pd.DataFrame:
    """Apply the persisted feature contract without fitting or learning anything."""
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    target = pd.to_numeric(df[TARGET], errors="coerce")
    if target.isna().any() or target.lt(0).any() or target.gt(100).any():
        raise ValueError(f"{TARGET} must be non-null and within [0, 100].")

    work = df.copy()
    if "requirement_size_bucket_ord" in contract["features"]:
        if "requirement_size_bucket" not in work.columns:
            raise ValueError("Missing source column: requirement_size_bucket")
        work["requirement_size_bucket_ord"] = (
            work["requirement_size_bucket"]
            .astype("string")
            .map(training.REQUIREMENT_BUCKET_ORD)
            .fillna(0)
            .astype(int)
        )

    missing = [column for column in contract["features"] if column not in work.columns]
    if missing:
        raise ValueError(f"Missing contract feature column(s): {missing}")

    unknown = int(contract["unknown_category_code"])
    for column in contract["categorical_features"]:
        if column not in contract["features"]:
            raise ValueError(f"Categorical feature outside contract features: {column}")
        if column not in contract["categorical_levels"]:
            raise ValueError(f"Missing persisted categorical levels for: {column}")
        allowed = [int(value) for value in contract["categorical_levels"][column]]
        values = pd.to_numeric(work[column], errors="coerce")
        mapped = values.where(values.isin(allowed), other=unknown).fillna(unknown).astype(int)
        work[column] = pd.Categorical(
            mapped,
            categories=sorted(set(allowed + [unknown])),
        )

    features = contract["features"]
    x = work[features].copy()
    numeric = [
        column for column in features if column not in contract["categorical_features"]
    ]
    x[numeric] = x[numeric].astype("float64")
    if list(x.columns) != model_features:
        raise ValueError("Prepared feature names/order do not match the loaded model.")
    return x


def rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def percentage(value: float | None) -> float | None:
    return float(value * 100) if value is not None else None


def mean_or_none(values: pd.Series) -> float | None:
    if values.empty:
        return None
    result = float(values.mean())
    return result if np.isfinite(result) else None


def scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    return value


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return scalar(value)


def group_key(value: Any) -> Any:
    converted = scalar(value)
    return MISSING_GROUP if converted is None else converted


def mark_band_analysis(missed: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(missed)
    marks = pd.to_numeric(missed[TARGET], errors="coerce")
    for label, selector in MARK_BANDS:
        subset = missed.loc[selector(marks)]
        probabilities = subset["predicted_pass_probability"]
        share = rate(len(subset), total)
        rows.append(
            {
                "final_mark_band": label,
                "row_count": int(len(subset)),
                "share_of_missed_failures": share,
                "share_of_missed_failures_pct": percentage(share),
                "mean_predicted_pass_probability": mean_or_none(probabilities),
                "median_predicted_pass_probability": (
                    float(probabilities.median()) if not probabilities.empty else None
                ),
                "min_predicted_pass_probability": (
                    float(probabilities.min()) if not probabilities.empty else None
                ),
                "max_predicted_pass_probability": (
                    float(probabilities.max()) if not probabilities.empty else None
                ),
            }
        )
    if sum(row["row_count"] for row in rows) != total:
        raise AssertionError("Final-mark bands do not exhaust all missed failures.")
    return rows


def finish_status_analysis(missed: pd.DataFrame) -> dict[str, Any]:
    if "finish_status" not in missed.columns:
        return {
            "available": False,
            "reason": "finish_status is absent from df_valid_final.parquet; target definition is unchanged.",
            "groups": [],
        }

    keys = missed["finish_status"].astype("object").map(group_key)
    rows: list[dict[str, Any]] = []
    for key in pd.unique(keys):
        subset = missed.loc[keys.eq(key)]
        share = rate(len(subset), len(missed))
        rows.append(
            {
                "finish_status": key,
                "row_count": int(len(subset)),
                "share_of_missed_failures": share,
                "share_of_missed_failures_pct": percentage(share),
                "mean_final_mark": mean_or_none(subset[TARGET]),
                "mean_predicted_pass_probability": mean_or_none(
                    subset["predicted_pass_probability"]
                ),
            }
        )
    rows.sort(key=lambda row: (-row["row_count"], str(row["finish_status"])))
    return {"available": True, "reason": None, "groups": rows}


def grouped_failure_analysis(df: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    failures = df.loc[df["actual_fail"]].copy()
    keys = failures[column].astype("object").map(group_key)
    total_missed = int(failures["missed_fail"].sum())
    rows: list[dict[str, Any]] = []
    for key in pd.unique(keys):
        subset = failures.loc[keys.eq(key)]
        missed = subset.loc[subset["missed_fail"]]
        missed_count = len(missed)
        missed_rate = rate(missed_count, len(subset))
        share = rate(missed_count, total_missed)
        rows.append(
            {
                "group_value": key,
                "actual_failures": int(len(subset)),
                "missed_failures": int(missed_count),
                "missed_failure_rate": missed_rate,
                "missed_failure_rate_pct": percentage(missed_rate),
                "share_of_all_missed_failures": share,
                "share_of_all_missed_failures_pct": percentage(share),
                "mean_predicted_pass_probability_actual_failures": mean_or_none(
                    subset["predicted_pass_probability"]
                ),
                "mean_predicted_pass_probability_missed_failures": mean_or_none(
                    missed["predicted_pass_probability"]
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            -row["missed_failures"],
            -row["actual_failures"],
            str(row["group_value"]),
        )
    )
    return rows


def comparison_metrics(df: pd.DataFrame, label: str, mask: pd.Series) -> dict[str, Any]:
    subset = df.loc[mask]
    failures = subset.loc[subset["actual_fail"]]
    missed = failures.loc[failures["missed_fail"]]
    failure_rate = rate(len(failures), len(subset))
    missed_rate = rate(len(missed), len(failures))
    return {
        "group": label,
        "total_rows": int(len(subset)),
        "actual_failures": int(len(failures)),
        "failure_rate": failure_rate,
        "failure_rate_pct": percentage(failure_rate),
        "missed_failures": int(len(missed)),
        "missed_failure_rate": missed_rate,
        "missed_failure_rate_pct": percentage(missed_rate),
        "mean_predicted_pass_probability_actual_failures": mean_or_none(
            failures["predicted_pass_probability"]
        ),
        "mean_predicted_pass_probability_missed_failures": mean_or_none(
            missed["predicted_pass_probability"]
        ),
    }


def attempt_analysis(df: pd.DataFrame) -> dict[str, Any]:
    attempts = pd.to_numeric(df["attempt_number"], errors="coerce")
    first = attempts.eq(1)
    repeat = attempts.gt(1)
    covered = first | repeat
    return {
        "groups": [
            comparison_metrics(df, "attempt_number == 1", first),
            comparison_metrics(df, "attempt_number > 1", repeat),
        ],
        "rows_outside_requested_groups": int((~covered).sum()),
    }


def load_a1_audit(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        audit = json.load(handle)
    required = {"coverage_definition", "train_cid_stats_course_id_count", "subset_metrics"}
    absent = sorted(required - set(audit))
    if absent:
        raise ValueError(f"A1 audit is missing keys: {absent}")
    return audit


def a1_expected_valid_covered_rows(audit: dict[str, Any]) -> int | None:
    for row in audit["subset_metrics"]:
        if row.get("subset") == "valid-covered":
            value = row.get("row_count", row.get("n_rows"))
            return int(value) if value is not None else None
    return None


def apply_a1_coverage(
    df: pd.DataFrame,
    train_path: Path,
    audit: dict[str, Any],
) -> tuple[pd.Series, dict[str, Any]]:
    if "course_id" not in df.columns:
        raise ValueError("Validation data are missing course_id required by A1 coverage.")
    train = pd.read_parquet(train_path, columns=["degree_course_key"])
    train_course_ids = (
        train["degree_course_key"].astype("string").str.rsplit("__", n=1).str[-1]
    )
    covered_ids = set(train_course_ids.dropna().unique().tolist())
    expected_key_count = int(audit["train_cid_stats_course_id_count"])
    if len(covered_ids) != expected_key_count:
        raise ValueError(
            "Rebuilt A1 course-key count does not match the reference audit: "
            f"{len(covered_ids)} != {expected_key_count}."
        )

    covered = df["course_id"].astype("string").isin(covered_ids)
    expected_rows = a1_expected_valid_covered_rows(audit)
    if expected_rows is not None and int(covered.sum()) != expected_rows:
        raise ValueError(
            "Rebuilt validation coverage does not match the reference A1 audit: "
            f"{int(covered.sum())} != {expected_rows}."
        )
    metadata = {
        "definition": audit["coverage_definition"],
        "train_rows_read_for_key_reconstruction": int(len(train)),
        "train_cid_stats_course_id_count": int(len(covered_ids)),
        "validation_covered_rows": int(covered.sum()),
        "validation_uncovered_rows": int((~covered).sum()),
        "validated_against_reference_a1": True,
    }
    return covered, metadata


def coverage_analysis(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for label in ("covered", "uncovered"):
        mask = df["coverage_status"].eq(label)
        rows.append(comparison_metrics(df, label, mask))
    return rows


def find_source_line(lines: list[str], needle: str) -> dict[str, Any] | None:
    for number, text in enumerate(lines, start=1):
        if needle in text:
            return {"line": number, "text": text.strip()}
    return None


def inspect_segment_definition(source_path: Path) -> dict[str, Any]:
    lines = source_path.read_text(encoding="utf-8").splitlines()
    no_progress = find_source_line(
        lines,
        'semester_df["no_previous_progress"] = no_previous_progress.astype(int)',
    )
    first_active = find_source_line(
        lines,
        'semester_df["is_first_active_semester"] = no_previous_progress.astype(int)',
    )
    if no_progress is None or first_active is None:
        raise ValueError("Could not locate both A3 assignments in feature_engineering.py.")
    no_rhs = no_progress["text"].split("=", 1)[1].strip()
    first_rhs = first_active["text"].split("=", 1)[1].strip()
    return {
        "source_path": str(source_path.resolve()),
        "no_previous_progress_assignment": no_progress,
        "is_first_active_semester_assignment": first_active,
        "same_right_hand_side_expression": no_rhs == first_rhs,
        "definition_explanation": (
            "Both columns are assigned from the same no_previous_progress boolean series; "
            "the current preprocessing therefore defines them as row-wise aliases."
            if no_rhs == first_rhs
            else "The two current assignments use different source expressions."
        ),
    }


def overlap_analysis(
    df: pd.DataFrame,
    source_path: Path,
    sample_size: int,
) -> dict[str, Any]:
    if sample_size < 0:
        raise ValueError("mismatch-sample-size must be non-negative.")
    required = set(OVERLAP_SAMPLE_COLUMNS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing A3 comparison/sample column(s): {missing}")

    first = pd.to_numeric(df["is_first_active_semester"], errors="coerce")
    no_progress = pd.to_numeric(df["no_previous_progress"], errors="coerce")
    for name, values in (
        ("is_first_active_semester", first),
        ("no_previous_progress", no_progress),
    ):
        if values.isna().any() or not values.isin([0, 1]).all():
            raise ValueError(f"{name} must contain only non-null binary 0/1 values.")

    both_one = first.eq(1) & no_progress.eq(1)
    first_only = first.eq(1) & no_progress.eq(0)
    progress_only = first.eq(0) & no_progress.eq(1)
    both_zero = first.eq(0) & no_progress.eq(0)
    matches = first.eq(no_progress)
    mismatch = ~matches
    source_definition = inspect_segment_definition(source_path)
    exact = bool(matches.all())
    if exact:
        conclusion = (
            "The two segments are exactly identical on validation. This agrees with the "
            "current preprocessing, which assigns both from no_previous_progress.astype(int)."
        )
    elif source_definition["same_right_hand_side_expression"]:
        conclusion = (
            "The validation columns differ even though current preprocessing assigns the same "
            "expression to both. The mismatch would indicate that the parquet was produced by a "
            "different preprocessing revision or changed after generation."
        )
    else:
        conclusion = "The differing current source expressions explain the row-level mismatch."

    samples = (
        df.loc[mismatch, list(OVERLAP_SAMPLE_COLUMNS)]
        .head(sample_size)
        .to_dict(orient="records")
    )
    match_rate = rate(int(matches.sum()), len(df))
    return {
        "diagnostic": "A3 segment overlap",
        "rows_compared": int(len(df)),
        "combination_counts": {
            "both_1": int(both_one.sum()),
            "is_first_active_semester_1_no_previous_progress_0": int(first_only.sum()),
            "is_first_active_semester_0_no_previous_progress_1": int(progress_only.sum()),
            "both_0": int(both_zero.sum()),
        },
        "matching_rows": int(matches.sum()),
        "mismatching_rows": int(mismatch.sum()),
        "overall_match_rate": match_rate,
        "overall_match_rate_pct": percentage(match_rate),
        "exactly_identical": exact,
        "mismatch_sample_columns": list(OVERLAP_SAMPLE_COLUMNS),
        "mismatch_samples": samples,
        "preprocessing_definition": source_definition,
        "conclusion": conclusion,
    }


def top_group(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return rows[0] if rows else None


def fmt_int(value: int) -> str:
    return f"{value:,}"


def fmt_pct(value: float | None, digits: int = 2) -> str:
    return "غير متاح" if value is None else f"{value:.{digits}f}%"


def fmt_prob(value: float | None) -> str:
    return "غير متاح" if value is None else f"{value:.4f}"


def report_text(summary: dict[str, Any], overlap: dict[str, Any]) -> str:
    overall = summary["overall"]
    mark_top = max(
        summary["final_mark_bands"],
        key=lambda row: row["row_count"],
    )
    segment_tops = summary["top_problem_groups_by_missed_count"]
    attempt_rows = {row["group"]: row for row in summary["attempt_comparison"]["groups"]}
    first = attempt_rows["attempt_number == 1"]
    repeat = attempt_rows["attempt_number > 1"]
    coverage_rows = {
        row["group"]: row for row in summary["course_difficulty_coverage"]["groups"]
    }
    covered = coverage_rows["covered"]
    uncovered = coverage_rows["uncovered"]

    attempt_difference = None
    if first["missed_failure_rate_pct"] is not None and repeat["missed_failure_rate_pct"] is not None:
        attempt_difference = repeat["missed_failure_rate_pct"] - first["missed_failure_rate_pct"]
    coverage_difference = None
    if covered["missed_failure_rate_pct"] is not None and uncovered["missed_failure_rate_pct"] is not None:
        coverage_difference = (
            uncovered["missed_failure_rate_pct"] - covered["missed_failure_rate_pct"]
        )

    higher_attempt = repeat if (attempt_difference or 0) >= 0 else first
    higher_coverage = uncovered if (coverage_difference or 0) >= 0 else covered
    finish_note = (
        "تم تحليله."
        if summary["finish_status"]["available"]
        else "غير موجود في ملف التحقق، لذلك لم يُحلل ولم يتغير تعريف الهدف."
    )
    overlap_result = (
        "متطابقتان تماماً"
        if overlap["exactly_identical"]
        else f"غير متطابقتين ({fmt_int(overlap['mismatching_rows'])} صفاً مختلفاً)"
    )

    lines = [
        "# تقرير A2 وA3 — تحليل حالات الرسوب غير المكتشفة",
        "",
        "تحليل قراءة فقط للنسخة المرجعية، مبني على بيانات التحقق وحدها وبحد قرار ثابت 0.5. "
        "لم تُقرأ بيانات الاختبار ولم يُدرّب أو يُعدّل أي مودل.",
        "",
        "## الخلاصة",
        "",
        f"- صفوف التحقق المحللة: {fmt_int(overall['validation_rows'])}.",
        f"- حالات الرسوب الفعلية: {fmt_int(overall['actual_failures'])}.",
        f"- حالات الرسوب غير المكتشفة: {fmt_int(overall['missed_failures'])} "
        f"({fmt_pct(overall['missed_failure_rate_pct'])} من الراسبين).",
        f"- أكبر شريحة علامات: `{mark_top['final_mark_band']}` بعدد "
        f"{fmt_int(mark_top['row_count'])} ({fmt_pct(mark_top['share_of_missed_failures_pct'])}).",
        f"- أعلى تركّز عددي حسب السنة: part_year=`{segment_tops['part_year']['group_value']}` "
        f"بعدد {fmt_int(segment_tops['part_year']['missed_failures'])}؛ حسب البرنامج: "
        f"degree_id=`{segment_tops['degree_id']['group_value']}` بعدد "
        f"{fmt_int(segment_tops['degree_id']['missed_failures'])}؛ وحسب نوع المقرر: "
        f"requirement_type_id=`{segment_tops['requirement_type_id']['group_value']}` بعدد "
        f"{fmt_int(segment_tops['requirement_type_id']['missed_failures'])}.",
        f"- finish_status: {finish_note}",
        "",
        "## المقارنات المطلوبة",
        "",
        f"- المحاولة الأولى: نسبة الرسوب {fmt_pct(first['failure_rate_pct'])}، ونسبة الرسوب غير "
        f"المكتشف {fmt_pct(first['missed_failure_rate_pct'])}، ومتوسط احتمال النجاح للراسبين "
        f"{fmt_prob(first['mean_predicted_pass_probability_actual_failures'])}.",
        f"- الإعادة: نسبة الرسوب {fmt_pct(repeat['failure_rate_pct'])}، ونسبة الرسوب غير المكتشف "
        f"{fmt_pct(repeat['missed_failure_rate_pct'])}، ومتوسط احتمال النجاح للراسبين "
        f"{fmt_prob(repeat['mean_predicted_pass_probability_actual_failures'])}. الفرق في عدم الاكتشاف "
        f"(الإعادة ناقص الأولى) = {attempt_difference:+.2f} نقطة مئوية."
        if attempt_difference is not None
        else "- تعذّر حساب فرق المحاولة بسبب مقام صفري.",
        f"- covered: {fmt_int(covered['actual_failures'])} راسباً، "
        f"{fmt_int(covered['missed_failures'])} غير مكتشف، بنسبة "
        f"{fmt_pct(covered['missed_failure_rate_pct'])}.",
        f"- uncovered: {fmt_int(uncovered['actual_failures'])} راسباً، "
        f"{fmt_int(uncovered['missed_failures'])} غير مكتشف، بنسبة "
        f"{fmt_pct(uncovered['missed_failure_rate_pct'])}. الفرق (uncovered ناقص covered) = "
        f"{coverage_difference:+.2f} نقطة مئوية."
        if coverage_difference is not None
        else "- تعذّر حساب فرق التغطية بسبب مقام صفري.",
        f"- تطابق الشريحتين: {overlap_result}؛ نسبة التطابق "
        f"{fmt_pct(overlap['overall_match_rate_pct'])}.",
        "",
        "## أهم ثلاث ملاحظات",
        "",
        f"1. شريحة العلامات الأكبر هي `{mark_top['final_mark_band']}` وتمثل "
        f"{fmt_pct(mark_top['share_of_missed_failures_pct'])} من جميع حالات الرسوب غير المكتشفة.",
        f"2. المجموعة الأعلى في عدم الاكتشاف بين الأولى والإعادة هي `{higher_attempt['group']}` "
        f"بنسبة {fmt_pct(higher_attempt['missed_failure_rate_pct'])}.",
        f"3. المجموعة الأعلى في عدم الاكتشاف حسب تغطية الصعوبة هي `{higher_coverage['group']}` "
        f"بنسبة {fmt_pct(higher_coverage['missed_failure_rate_pct'])}؛ وقد طابق إعادة بناء التغطية "
        "تدقيق A1 المرجعي.",
        "",
        "## الخطوة التالية المقترحة فقط",
        "",
        "إجراء تحليل تفسيري قراءة فقط (مثل SHAP) يقارن حالات الرسوب غير المكتشفة بحالات الرسوب "
        "المكتشفة داخل بيانات التحقق، لتحديد الإشارات التي ترفع احتمال النجاح خطأً، دون تدريب أو "
        "تغيير الحد في هذه المرحلة.",
    ]
    return "\n".join(lines) + "\n"


def unique_output_dir(root: Path, timestamp: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base = root / f"{timestamp}__missed-fail-analysis"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = root / f"{base.name}__{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    model_path = run_dir / "m1_pass_model.lgbm"
    contract_path = run_dir / "feature_contract.json"
    a1_path = run_dir / "a1_coverage_audit.json"
    model_training_path = Path(training.__file__).resolve()
    required_paths = [
        model_path,
        contract_path,
        a1_path,
        args.valid,
        args.train,
        args.feature_engineering_source,
        model_training_path,
    ]
    require_files(required_paths)

    protected_hashes_before = {str(path.resolve()): sha256(path) for path in required_paths}
    contract = load_contract(contract_path)
    audit = load_a1_audit(a1_path)
    model = lgb.Booster(model_file=str(model_path))
    model_features = model.feature_name()
    if contract["features"] != model_features:
        raise ValueError("feature_contract.json does not match the model feature names/order.")

    valid = pd.read_parquet(args.valid)
    required_analysis_columns = set(SEGMENT_COLUMNS) | {
        TARGET,
        "course_id",
        *OVERLAP_SAMPLE_COLUMNS,
    }
    missing_analysis = sorted(required_analysis_columns - set(valid.columns))
    if missing_analysis:
        raise ValueError(f"Validation data are missing analysis columns: {missing_analysis}")

    x_valid = prepare_features(valid, contract, model_features)
    probabilities = np.asarray(model.predict(x_valid), dtype=float)
    if (
        len(probabilities) != len(valid)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("Validation predictions are not finite probabilities in [0, 1].")

    analysis = valid.copy()
    analysis.insert(0, "validation_row_number", np.arange(1, len(analysis) + 1))
    analysis["predicted_pass_probability"] = probabilities
    analysis["actual_fail"] = analysis[TARGET].lt(50)
    analysis["predicted_pass"] = analysis["predicted_pass_probability"].ge(THRESHOLD)
    analysis["missed_fail"] = analysis["actual_fail"] & analysis["predicted_pass"]

    covered, coverage_metadata = apply_a1_coverage(analysis, args.train, audit)
    analysis["coverage_status"] = np.where(covered, "covered", "uncovered")
    missed = analysis.loc[analysis["missed_fail"]].copy()
    marks = pd.to_numeric(missed[TARGET], errors="coerce")
    conditions = [selector(marks) for _, selector in MARK_BANDS]
    labels = [label for label, _ in MARK_BANDS]
    missed["final_mark_band"] = np.select(conditions, labels, default="OUT_OF_RANGE")

    actual_failure_count = int(analysis["actual_fail"].sum())
    missed_count = int(analysis["missed_fail"].sum())
    missed_rate = rate(missed_count, actual_failure_count)
    segment_breakdowns = {
        column: grouped_failure_analysis(analysis, column) for column in SEGMENT_COLUMNS
    }
    overlap = overlap_analysis(
        analysis,
        args.feature_engineering_source,
        args.mismatch_sample_size,
    )
    top_problem_groups = {
        column: top_group(segment_breakdowns[column])
        for column in ("part_year", "degree_id", "requirement_type_id")
    }

    generated_at = datetime.now().astimezone()
    timestamp = generated_at.strftime("%Y-%m-%d_%H%M%S")
    summary = {
        "diagnostic": "A2 missed-failure analysis",
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "analysis_scope": {
            "primary_split": "validation",
            "validation_used_for_all_findings": True,
            "test_data_read": False,
            "test_data_used_for_decisions": False,
            "training_performed": False,
            "model_or_data_mutation_performed": False,
        },
        "provenance": {
            "reference_run_directory": str(run_dir),
            "m1_model_path": str(model_path.resolve()),
            "feature_contract_path": str(contract_path.resolve()),
            "a1_coverage_audit_path": str(a1_path.resolve()),
            "validation_path": str(args.valid.resolve()),
            "coverage_key_source_train_path": str(args.train.resolve()),
            "model_training_source_path": str(model_training_path),
            "feature_engineering_source_path": str(args.feature_engineering_source.resolve()),
        },
        "definitions": {
            "actual_fail": "final_mark < 50",
            "predicted_pass": "predicted_pass_probability >= 0.5",
            "missed_fail": "actual_fail AND predicted_pass",
            "decision_threshold": THRESHOLD,
        },
        "overall": {
            "validation_rows": int(len(analysis)),
            "actual_failures": actual_failure_count,
            "predicted_pass_rows": int(analysis["predicted_pass"].sum()),
            "missed_failures": missed_count,
            "missed_failure_rate": missed_rate,
            "missed_failure_rate_pct": percentage(missed_rate),
        },
        "final_mark_bands": mark_band_analysis(missed),
        "finish_status": finish_status_analysis(missed),
        "segment_breakdowns": segment_breakdowns,
        "attempt_comparison": attempt_analysis(analysis),
        "course_difficulty_coverage": {
            **coverage_metadata,
            "groups": coverage_analysis(analysis),
        },
        "top_problem_groups_by_missed_count": top_problem_groups,
    }

    protected_hashes_after = {str(path.resolve()): sha256(path) for path in required_paths}
    if protected_hashes_before != protected_hashes_after:
        raise RuntimeError("A protected input/source file changed during the diagnostic run.")
    summary["read_only_verification"] = {
        "protected_files_unchanged_during_run": True,
        "sha256": protected_hashes_after,
    }

    output_dir = unique_output_dir(args.output_root, timestamp)
    missed_output = missed[
        [
            "validation_row_number",
            "predicted_pass_probability",
            "actual_fail",
            "predicted_pass",
            "missed_fail",
            "final_mark_band",
            "coverage_status",
        ]
        + [
            column
            for column in valid.columns
            if column
            not in {
                "validation_row_number",
                "predicted_pass_probability",
                "actual_fail",
                "predicted_pass",
                "missed_fail",
                "final_mark_band",
                "coverage_status",
            }
        ]
    ]
    missed_output.to_csv(output_dir / "missed_failures.csv", index=False)
    (output_dir / "missed_failures_summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "segment_overlap.json").write_text(
        json.dumps(json_ready(overlap), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(
        report_text(summary, overlap),
        encoding="utf-8",
    )

    created = sorted(output_dir.iterdir(), key=lambda path: path.name)
    if {path.name for path in created} != {
        "missed_failures.csv",
        "missed_failures_summary.json",
        "segment_overlap.json",
        "REPORT.md",
    }:
        raise AssertionError("The output directory does not contain exactly the four expected files.")

    print(f"Reference run: {run_dir}")
    print(f"Validation rows analyzed: {len(analysis):,}")
    print(f"Actual failures: {actual_failure_count:,}")
    print(f"Missed failures at threshold {THRESHOLD:.1f}: {missed_count:,}")
    print(f"A3 exact segment match: {overlap['exactly_identical']}")
    print("Test data were not read. No training was performed.")
    print("Protected existing files passed before/after SHA-256 checks.")
    print("Created files:")
    for path in created:
        print(f"- {path.resolve()}")


if __name__ == "__main__":
    main()
