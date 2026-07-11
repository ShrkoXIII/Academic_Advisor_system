"""Audit ID-like columns in raw academic data tables.

This utility is intentionally read-only with respect to source data. It scans
raw parquet files under ``src.paths.RAW_DIR`` by default, profiles conservative
ID-like column candidates, and writes audit reports under:

    DATA_DIR / "audit" / "id_dtype_audit"

The output is for manual canonical-ID planning only. It does not cast,
normalize, rename, or overwrite any source column.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXCLUDED_PATH_PARTS = {"archive", "backup", "backups", "temp", "tmp"}

EXACT_ID_NAMES = {
    "id",
    "part_id",
}

SEMESTER_PART_NAMES = {
    "part_id",
    "start_part_id",
    "finish_part_id",
}

ENTITY_JOIN_NAMES = {
    "student_id",
    "course_id",
    "degree_id",
    "degree_course_id",
    "student_course_id",
    "student_status_id",
}

CATEGORICAL_CODE_NAMES = {
    "requirement_type_id",
    "diploma_type_id",
    "level_id",
}

CONFIDENCE_HIGH = "HIGH"
CONFIDENCE_MEDIUM = "MEDIUM"
CONFIDENCE_LOW = "LOW"

CATEGORY_ENTITY_JOIN_ID = "ENTITY_JOIN_ID"
CATEGORY_SEMESTER_PART_CODE = "SEMESTER_PART_CODE"
CATEGORY_CATEGORICAL_CODE = "CATEGORICAL_CODE"
CATEGORY_COMPOSITE_KEY = "COMPOSITE_KEY"
CATEGORY_UNKNOWN = "UNKNOWN_REVIEW_REQUIRED"

INTEGER_RE = re.compile(r"^[+-]?\d+$")
DOTTED_RE = re.compile(r"^[+-]?\d+\.\d+$")
NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
LEADING_ZERO_RE = re.compile(r"^[+-]?0\d+$")


@dataclass
class FileCandidate:
    file: str
    file_type: str
    column_name: str
    canonical_column_name: str
    candidate_kind: str
    candidate_rule: str
    observed_dtype: str
    row_count: int | None = None
    sample_size_requested: int = 0
    sample_rows_read: int = 0
    non_null_sample_count: int = 0
    null_sample_count: int = 0
    distinct_sample_count: int = 0
    integer_like_count: int = 0
    numeric_like_count: int = 0
    dotted_suffix_count: int = 0
    leading_zero_count: int = 0
    text_like_count: int = 0
    min_string_length: int | None = None
    max_string_length: int | None = None
    pattern_signature: str = "NO_SAMPLE"
    error: str | None = None


@dataclass
class ColumnGroup:
    canonical_column_name: str
    observed_column_names: list[str]
    candidate_kinds: list[str]
    files: list[str]
    file_count: int
    observed_dtypes: list[str]
    dtype_drift: bool
    pattern_signatures: list[str]
    value_pattern_drift: bool
    has_dotted_suffix_values: bool
    dotted_suffix_file_count: int
    total_non_null_sample_count: int
    total_dotted_suffix_count: int
    suggested_semantic_category: str
    suggestion_confidence: str
    notes: list[str] = field(default_factory=list)


def _load_project_paths() -> Any:
    try:
        from src import paths as project_paths  # type: ignore

        return project_paths
    except Exception as exc:  # pragma: no cover - depends on caller environment
        return exc


def _path_from_attr(module: Any, attr: str) -> Path | None:
    value = getattr(module, attr, None)
    if value is None:
        return None
    return Path(value).expanduser()


def resolve_data_and_raw_dirs(data_root_override: str | None) -> tuple[Path, Path, str]:
    """Resolve DATA_DIR and RAW_DIR without hardcoding project data paths."""

    project_paths = _load_project_paths()
    import_note = "src.paths imported"

    if isinstance(project_paths, Exception):
        env_data_root = os.environ.get("ACADEMIC_ADVISOR_DATA_DIR")
        if not env_data_root:
            raise RuntimeError(
                "Unable to import src.paths and ACADEMIC_ADVISOR_DATA_DIR is not set. "
                f"Original import error: {project_paths}"
            ) from project_paths
        data_dir = Path(data_root_override or env_data_root).expanduser().resolve()
        raw_dir = data_dir / "raw"
        return data_dir, raw_dir, (
            "src.paths import failed; resolved DATA_DIR from "
            "ACADEMIC_ADVISOR_DATA_DIR and RAW_DIR as DATA_DIR/raw"
        )

    default_data_dir = _path_from_attr(project_paths, "DATA_DIR")
    default_raw_dir = _path_from_attr(project_paths, "RAW_DIR")
    if default_data_dir is None or default_raw_dir is None:
        raise RuntimeError("src.paths must define DATA_DIR and RAW_DIR")

    if data_root_override:
        data_dir = Path(data_root_override).expanduser().resolve()
        try:
            raw_relative = default_raw_dir.resolve().relative_to(default_data_dir.resolve())
        except ValueError:
            raw_relative = Path("raw")
            import_note += "; RAW_DIR is not below DATA_DIR, using override/raw"
        raw_dir = (data_dir / raw_relative).resolve()
    else:
        data_dir = default_data_dir.resolve()
        raw_dir = default_raw_dir.resolve()

    return data_dir, raw_dir, import_note


def is_excluded_path(path: Path) -> bool:
    return any(part.lower() in EXCLUDED_PATH_PARTS for part in path.parts)


def iter_table_files(raw_dir: Path, include_csv: bool) -> list[Path]:
    suffixes = [".parquet"]
    if include_csv:
        suffixes.append(".csv")

    files: list[Path] = []
    for suffix in suffixes:
        files.extend(
            path
            for path in raw_dir.rglob(f"*{suffix}")
            if path.is_file() and not is_excluded_path(path.relative_to(raw_dir))
        )
    return sorted(files, key=lambda path: (path.suffix.lower() != ".parquet", str(path).lower()))


def canonicalize_column_name(column_name: str) -> str:
    return column_name.strip().lower()


def classify_candidate(column_name: str) -> tuple[bool, str, str]:
    canonical = canonicalize_column_name(column_name)
    if canonical.endswith("_key"):
        return True, "KEY_LIKE", "suffix__key"
    if canonical in EXACT_ID_NAMES:
        return True, "ID_LIKE", "exact_name"
    if canonical.endswith("_id"):
        return True, "ID_LIKE", "suffix__id"
    return False, "", ""


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def normalize_scalar_for_pattern(value: Any) -> str:
    """Convert a scalar to a comparable audit string without changing source data."""

    if _is_missing(value):
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    text = str(value).strip()
    if text.endswith(".0") and INTEGER_RE.match(text[:-2]):
        return text[:-2]
    return text


def build_pattern_signature(
    non_null_count: int,
    integer_like_count: int,
    numeric_like_count: int,
    dotted_suffix_count: int,
    text_like_count: int,
    leading_zero_count: int,
) -> str:
    if non_null_count == 0:
        return "EMPTY_SAMPLE"

    parts: list[str] = []
    if dotted_suffix_count:
        parts.append("DOTTED_SUFFIX")
    if integer_like_count:
        parts.append("INTEGER_LIKE")
    if numeric_like_count > integer_like_count:
        parts.append("NUMERIC_NON_INTEGER")
    if text_like_count:
        parts.append("TEXT_LIKE")
    if leading_zero_count:
        parts.append("LEADING_ZERO")
    return "+".join(parts) if parts else "UNCLASSIFIED_VALUES"


def profile_values(values: Sequence[Any], sample_size: int, base: FileCandidate) -> FileCandidate:
    strings: list[str] = []
    nulls = 0

    for value in values:
        if _is_missing(value):
            nulls += 1
            continue
        text = normalize_scalar_for_pattern(value)
        if text == "":
            nulls += 1
            continue
        strings.append(text)

    integer_like_count = sum(1 for text in strings if INTEGER_RE.match(text))
    numeric_like_count = sum(1 for text in strings if NUMERIC_RE.match(text))
    dotted_suffix_count = sum(1 for text in strings if DOTTED_RE.match(text))
    leading_zero_count = sum(1 for text in strings if LEADING_ZERO_RE.match(text))
    text_like_count = len(strings) - numeric_like_count
    lengths = [len(text) for text in strings]

    base.sample_size_requested = sample_size
    base.sample_rows_read = len(values)
    base.non_null_sample_count = len(strings)
    base.null_sample_count = nulls
    base.distinct_sample_count = len(set(strings))
    base.integer_like_count = integer_like_count
    base.numeric_like_count = numeric_like_count
    base.dotted_suffix_count = dotted_suffix_count
    base.leading_zero_count = leading_zero_count
    base.text_like_count = text_like_count
    base.min_string_length = min(lengths) if lengths else None
    base.max_string_length = max(lengths) if lengths else None
    base.pattern_signature = build_pattern_signature(
        non_null_count=len(strings),
        integer_like_count=integer_like_count,
        numeric_like_count=numeric_like_count,
        dotted_suffix_count=dotted_suffix_count,
        text_like_count=text_like_count,
        leading_zero_count=leading_zero_count,
    )
    return base


def read_parquet_schema(path: Path) -> tuple[list[tuple[str, str]], int | None]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        parquet_file = pq.ParquetFile(path)
        schema = parquet_file.schema_arrow
        fields = [(field.name, str(field.type)) for field in schema]
        row_count = parquet_file.metadata.num_rows if parquet_file.metadata is not None else None
        return fields, row_count
    except ImportError:
        try:
            import pandas as pd  # type: ignore

            empty = pd.read_parquet(path).head(0)
            return [(str(column), str(dtype)) for column, dtype in empty.dtypes.items()], None
        except Exception:
            raise


def read_parquet_sample_columns(
    path: Path,
    columns: Sequence[str],
    sample_size: int,
) -> Mapping[str, list[Any]]:
    if not columns:
        return {}

    try:
        import pyarrow.parquet as pq  # type: ignore

        parquet_file = pq.ParquetFile(path)
        target_rows = max(sample_size, 0)
        samples = {column: [] for column in columns}
        if target_rows == 0:
            return samples

        batch_size = min(max(target_rows, 1024), 65536)
        rows_read = 0
        for batch in parquet_file.iter_batches(columns=list(columns), batch_size=batch_size):
            batch_dict = batch.to_pydict()
            remaining = target_rows - rows_read
            for column in columns:
                samples[column].extend(batch_dict.get(column, [])[:remaining])
            rows_read += min(batch.num_rows, remaining)
            if rows_read >= target_rows:
                break
        return samples
    except ImportError:
        import pandas as pd  # type: ignore

        frame = pd.read_parquet(path, columns=list(columns))
        return {column: frame[column].head(sample_size).tolist() for column in columns}


def read_csv_schema(path: Path) -> list[tuple[str, str]]:
    import pandas as pd  # type: ignore

    empty = pd.read_csv(path, nrows=0)
    return [(str(column), "csv:string_sample") for column in empty.columns]


def read_csv_sample_columns(
    path: Path,
    columns: Sequence[str],
    sample_size: int,
) -> Mapping[str, list[Any]]:
    if not columns or sample_size == 0:
        return {column: [] for column in columns}

    import pandas as pd  # type: ignore

    frame = pd.read_csv(path, usecols=list(columns), nrows=sample_size, dtype="string")
    return {column: frame[column].tolist() for column in columns}


def profile_file(path: Path, raw_dir: Path, sample_size: int) -> tuple[list[FileCandidate], str | None]:
    relative_file = path.relative_to(raw_dir).as_posix()
    file_type = path.suffix.lower().lstrip(".")

    try:
        if file_type == "parquet":
            schema_fields, row_count = read_parquet_schema(path)
        elif file_type == "csv":
            schema_fields = read_csv_schema(path)
            row_count = None
        else:
            return [], f"Unsupported file type: {path.suffix}"
    except Exception as exc:
        return [], f"Schema read failed: {type(exc).__name__}: {exc}"

    candidates: list[FileCandidate] = []
    candidate_columns: list[str] = []
    dtype_by_column: dict[str, str] = {}
    candidate_info: dict[str, tuple[str, str, str]] = {}

    for column_name, dtype in schema_fields:
        is_candidate, candidate_kind, candidate_rule = classify_candidate(column_name)
        if not is_candidate:
            continue
        canonical = canonicalize_column_name(column_name)
        candidate_columns.append(column_name)
        dtype_by_column[column_name] = dtype
        candidate_info[column_name] = (canonical, candidate_kind, candidate_rule)

    if not candidate_columns:
        return [], None

    try:
        if file_type == "parquet":
            samples = read_parquet_sample_columns(path, candidate_columns, sample_size)
        else:
            samples = read_csv_sample_columns(path, candidate_columns, sample_size)
    except Exception as exc:
        for column_name in candidate_columns:
            canonical, candidate_kind, candidate_rule = candidate_info[column_name]
            candidates.append(
                FileCandidate(
                    file=relative_file,
                    file_type=file_type,
                    column_name=column_name,
                    canonical_column_name=canonical,
                    candidate_kind=candidate_kind,
                    candidate_rule=candidate_rule,
                    observed_dtype=dtype_by_column[column_name],
                    row_count=row_count,
                    sample_size_requested=sample_size,
                    error=f"Column sample read failed: {type(exc).__name__}: {exc}",
                )
            )
        return candidates, None

    for column_name in candidate_columns:
        canonical, candidate_kind, candidate_rule = candidate_info[column_name]
        base = FileCandidate(
            file=relative_file,
            file_type=file_type,
            column_name=column_name,
            canonical_column_name=canonical,
            candidate_kind=candidate_kind,
            candidate_rule=candidate_rule,
            observed_dtype=dtype_by_column[column_name],
            row_count=row_count,
        )
        candidates.append(profile_values(samples.get(column_name, []), sample_size, base))

    return candidates, None


def suggest_category(canonical_name: str, profiles: Sequence[FileCandidate]) -> tuple[str, str, list[str]]:
    notes: list[str] = ["Suggestion only; no data was cast, normalized, or rewritten."]
    has_dotted = any(profile.dotted_suffix_count > 0 for profile in profiles)
    has_key = any(profile.candidate_kind == "KEY_LIKE" for profile in profiles)
    has_text = any(profile.text_like_count > 0 for profile in profiles)

    if has_key:
        notes.append("Column name ends with _key and is kept separate from _id candidates.")
        if has_text:
            notes.append("Sample contains non-numeric text-like values.")
        return CATEGORY_COMPOSITE_KEY, CONFIDENCE_MEDIUM, notes

    if canonical_name in SEMESTER_PART_NAMES:
        if has_dotted:
            notes.append("Name suggests semester/part code, but dotted suffix values need review.")
            return CATEGORY_SEMESTER_PART_CODE, CONFIDENCE_MEDIUM, notes
        notes.append("Name matches configured semester/part code candidates.")
        return CATEGORY_SEMESTER_PART_CODE, CONFIDENCE_HIGH, notes

    if canonical_name in ENTITY_JOIN_NAMES:
        notes.append("Name matches configured entity/join ID candidates.")
        if has_dotted:
            notes.append("Dotted suffix evidence supports treating this as a lossless canonical ID candidate.")
        return CATEGORY_ENTITY_JOIN_ID, CONFIDENCE_HIGH, notes

    if canonical_name in CATEGORICAL_CODE_NAMES:
        if has_dotted:
            notes.append("Name suggests categorical/code ID, but dotted suffix values need review.")
            return CATEGORY_CATEGORICAL_CODE, CONFIDENCE_MEDIUM, notes
        notes.append("Name matches configured categorical/code ID candidates.")
        return CATEGORY_CATEGORICAL_CODE, CONFIDENCE_HIGH, notes

    if has_dotted:
        notes.append("Dotted suffix values were observed; suffix-preserving review is required.")
        return CATEGORY_ENTITY_JOIN_ID, CONFIDENCE_MEDIUM, notes

    if canonical_name.endswith("_id") or canonical_name in EXACT_ID_NAMES:
        notes.append("Detected by conservative ID-name rule only.")
        return CATEGORY_UNKNOWN, CONFIDENCE_LOW, notes

    notes.append("No configured heuristic matched this candidate.")
    return CATEGORY_UNKNOWN, CONFIDENCE_LOW, notes


def build_column_groups(profiles: Sequence[FileCandidate]) -> list[ColumnGroup]:
    grouped: dict[str, list[FileCandidate]] = defaultdict(list)
    for profile in profiles:
        grouped[profile.canonical_column_name].append(profile)

    groups: list[ColumnGroup] = []
    for canonical_name, column_profiles in sorted(grouped.items()):
        observed_column_names = sorted({profile.column_name for profile in column_profiles})
        candidate_kinds = sorted({profile.candidate_kind for profile in column_profiles})
        files = sorted({profile.file for profile in column_profiles})
        observed_dtypes = sorted({profile.observed_dtype for profile in column_profiles})
        pattern_signatures = sorted(
            {profile.pattern_signature for profile in column_profiles if not profile.error}
        )
        dtype_drift = len(observed_dtypes) > 1
        value_pattern_drift = len(pattern_signatures) > 1
        has_dotted_suffix_values = any(profile.dotted_suffix_count > 0 for profile in column_profiles)
        dotted_suffix_file_count = sum(1 for profile in column_profiles if profile.dotted_suffix_count > 0)
        total_non_null_sample_count = sum(profile.non_null_sample_count for profile in column_profiles)
        total_dotted_suffix_count = sum(profile.dotted_suffix_count for profile in column_profiles)
        category, confidence, notes = suggest_category(canonical_name, column_profiles)

        if dtype_drift:
            notes.append("Observed dtype drift across files.")
        if value_pattern_drift:
            notes.append("Observed value-pattern drift across files.")
        errors = [profile.error for profile in column_profiles if profile.error]
        if errors:
            notes.append(f"{len(errors)} file/column profile error(s) recorded in JSON details.")

        groups.append(
            ColumnGroup(
                canonical_column_name=canonical_name,
                observed_column_names=observed_column_names,
                candidate_kinds=candidate_kinds,
                files=files,
                file_count=len(files),
                observed_dtypes=observed_dtypes,
                dtype_drift=dtype_drift,
                pattern_signatures=pattern_signatures,
                value_pattern_drift=value_pattern_drift,
                has_dotted_suffix_values=has_dotted_suffix_values,
                dotted_suffix_file_count=dotted_suffix_file_count,
                total_non_null_sample_count=total_non_null_sample_count,
                total_dotted_suffix_count=total_dotted_suffix_count,
                suggested_semantic_category=category,
                suggestion_confidence=confidence,
                notes=notes,
            )
        )
    return groups


def csv_join(values: Iterable[Any]) -> str:
    return " | ".join(str(value) for value in values)


def write_csv_report(path: Path, groups: Sequence[ColumnGroup]) -> None:
    fieldnames = [
        "canonical_column_name",
        "observed_column_names",
        "candidate_kinds",
        "files",
        "file_count",
        "observed_dtypes",
        "dtype_drift",
        "pattern_signatures",
        "value_pattern_drift",
        "has_dotted_suffix_values",
        "dotted_suffix_file_count",
        "total_non_null_sample_count",
        "total_dotted_suffix_count",
        "suggested_semantic_category",
        "suggestion_confidence",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            row = asdict(group)
            for key in (
                "observed_column_names",
                "candidate_kinds",
                "files",
                "observed_dtypes",
                "pattern_signatures",
                "notes",
            ):
                row[key] = csv_join(row[key])
            writer.writerow(row)


def build_summary(
    *,
    data_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    include_csv: bool,
    sample_size: int,
    import_note: str,
    scanned_files: Sequence[Path],
    profiles: Sequence[FileCandidate],
    groups: Sequence[ColumnGroup],
    file_errors: Mapping[str, str],
) -> dict[str, Any]:
    repeated = [group for group in groups if group.file_count > 1]
    dtype_drift = [group for group in groups if group.dtype_drift]
    pattern_drift = [group for group in groups if group.value_pattern_drift]
    dotted = [group for group in groups if group.has_dotted_suffix_values]
    unknown = [group for group in groups if group.suggested_semantic_category == CATEGORY_UNKNOWN]
    key_like = [group for group in groups if "KEY_LIKE" in group.candidate_kinds]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "source_scope": "RAW_DIR only",
        "include_csv": include_csv,
        "sample_size": sample_size,
        "path_resolution": import_note,
        "files_discovered": len(scanned_files),
        "files_with_profile_errors": len(file_errors),
        "candidate_column_file_occurrences": len(profiles),
        "canonical_column_count": len(groups),
        "repeated_column_name_count": len(repeated),
        "dtype_drift_count": len(dtype_drift),
        "value_pattern_drift_count": len(pattern_drift),
        "dotted_suffix_column_count": len(dotted),
        "unknown_review_required_count": len(unknown),
        "key_like_column_count": len(key_like),
        "category_counts": dict(Counter(group.suggested_semantic_category for group in groups)),
    }


def write_json_report(
    path: Path,
    summary: Mapping[str, Any],
    groups: Sequence[ColumnGroup],
    profiles: Sequence[FileCandidate],
    file_errors: Mapping[str, str],
) -> None:
    payload = {
        "summary": summary,
        "groups": [asdict(group) for group in groups],
        "file_column_profiles": [asdict(profile) for profile in profiles],
        "file_errors": dict(file_errors),
        "schema": {
            "groups": list(asdict(groups[0]).keys()) if groups else list(ColumnGroup.__dataclass_fields__),
            "file_column_profiles": list(asdict(profiles[0]).keys())
            if profiles
            else list(FileCandidate.__dataclass_fields__),
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def markdown_table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        escaped = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return lines


def write_markdown_summary(
    path: Path,
    summary: Mapping[str, Any],
    groups: Sequence[ColumnGroup],
    file_errors: Mapping[str, str],
) -> None:
    repeated = [group for group in groups if group.file_count > 1]
    dtype_drift = [group for group in groups if group.dtype_drift]
    pattern_drift = [group for group in groups if group.value_pattern_drift]
    dotted = [group for group in groups if group.has_dotted_suffix_values]
    unknown = [group for group in groups if group.suggested_semantic_category == CATEGORY_UNKNOWN]
    key_like = [group for group in groups if "KEY_LIKE" in group.candidate_kinds]

    lines: list[str] = [
        "# ID Column Audit Summary",
        "",
        "This is a discovery report for manual canonical ID planning. "
        "Suggestions are heuristic only; the audit does not cast, normalize, rename, "
        "or rewrite source data.",
        "",
        "## Scope",
        "",
        f"- Data root: `{summary['data_dir']}`",
        f"- Raw directory scanned: `{summary['raw_dir']}`",
        "- Source scope: `RAW_DIR` only",
        f"- CSV included: `{summary['include_csv']}`",
        f"- Sample size per candidate column: `{summary['sample_size']}`",
        "",
        "## Counts",
        "",
    ]

    count_rows = [
        ("Files discovered", summary["files_discovered"]),
        ("Candidate column file occurrences", summary["candidate_column_file_occurrences"]),
        ("Canonical column names", summary["canonical_column_count"]),
        ("Repeated column names", summary["repeated_column_name_count"]),
        ("Dtype drift cases", summary["dtype_drift_count"]),
        ("Value-pattern drift cases", summary["value_pattern_drift_count"]),
        ("Dotted-suffix cases", summary["dotted_suffix_column_count"]),
        ("Unknown/review-required cases", summary["unknown_review_required_count"]),
        ("Key-like columns", summary["key_like_column_count"]),
        ("Files with profile errors", summary["files_with_profile_errors"]),
    ]
    lines.extend(markdown_table(count_rows, ["Metric", "Value"]))
    lines.extend(["", "## Repeated Column Names", ""])
    lines.extend(
        markdown_table(
            [
                (
                    group.canonical_column_name,
                    group.file_count,
                    csv_join(group.observed_dtypes),
                    group.suggested_semantic_category,
                    group.suggestion_confidence,
                )
                for group in repeated
            ],
            ["Column", "Files", "Dtypes", "Suggested Category", "Confidence"],
        )
        if repeated
        else ["No repeated candidate column names found."]
    )

    lines.extend(["", "## Dtype Drift", ""])
    lines.extend(
        markdown_table(
            [
                (
                    group.canonical_column_name,
                    group.file_count,
                    csv_join(group.observed_dtypes),
                    group.suggested_semantic_category,
                )
                for group in dtype_drift
            ],
            ["Column", "Files", "Observed Dtypes", "Suggested Category"],
        )
        if dtype_drift
        else ["No dtype drift found among candidate columns."]
    )

    lines.extend(["", "## Value-Pattern Drift", ""])
    lines.extend(
        markdown_table(
            [
                (
                    group.canonical_column_name,
                    group.file_count,
                    csv_join(group.pattern_signatures),
                    group.suggested_semantic_category,
                )
                for group in pattern_drift
            ],
            ["Column", "Files", "Pattern Signatures", "Suggested Category"],
        )
        if pattern_drift
        else ["No value-pattern drift found among candidate columns."]
    )

    lines.extend(["", "## Dotted-Suffix Evidence", ""])
    lines.extend(
        markdown_table(
            [
                (
                    group.canonical_column_name,
                    group.dotted_suffix_file_count,
                    group.total_dotted_suffix_count,
                    group.suggested_semantic_category,
                    group.suggestion_confidence,
                )
                for group in dotted
            ],
            ["Column", "Files With Dotted Values", "Dotted Sample Count", "Suggested Category", "Confidence"],
        )
        if dotted
        else ["No dotted-suffix values found in sampled candidate columns."]
    )

    lines.extend(["", "## Key-Like Columns", ""])
    lines.extend(
        markdown_table(
            [
                (
                    group.canonical_column_name,
                    group.file_count,
                    csv_join(group.pattern_signatures),
                    group.suggested_semantic_category,
                    group.suggestion_confidence,
                )
                for group in key_like
            ],
            ["Column", "Files", "Pattern Signatures", "Suggested Category", "Confidence"],
        )
        if key_like
        else ["No `_key` columns found."]
    )

    lines.extend(["", "## Unknown / Review Required", ""])
    lines.extend(
        markdown_table(
            [
                (
                    group.canonical_column_name,
                    group.file_count,
                    csv_join(group.observed_dtypes),
                    csv_join(group.pattern_signatures),
                )
                for group in unknown
            ],
            ["Column", "Files", "Observed Dtypes", "Pattern Signatures"],
        )
        if unknown
        else ["No unknown/review-required candidate columns found."]
    )

    if file_errors:
        lines.extend(["", "## File Errors", ""])
        lines.extend(
            markdown_table(
                [(file, error) for file, error in sorted(file_errors.items())],
                ["File", "Error"],
            )
        )

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_reports(
    output_dir: Path,
    summary: Mapping[str, Any],
    groups: Sequence[ColumnGroup],
    profiles: Sequence[FileCandidate],
    file_errors: Mapping[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_report(output_dir / "id_columns_audit.csv", groups)
    write_json_report(output_dir / "id_columns_audit.json", summary, groups, profiles, file_errors)
    write_markdown_summary(output_dir / "id_columns_summary.md", summary, groups, file_errors)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("sample size must be zero or greater")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover and profile conservative ID-like columns in raw academic tables. "
            "This is an audit only and does not modify source data."
        )
    )
    parser.add_argument(
        "--data-root",
        help=(
            "Optional DATA_DIR override. RAW_DIR is resolved using the same relative raw "
            "location as src.paths.RAW_DIR when possible."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=positive_int,
        default=10_000,
        help="Maximum rows to sample per candidate column for value-pattern profiling.",
    )
    parser.add_argument(
        "--include-csv",
        action="store_true",
        help="Also scan CSV files under RAW_DIR. Parquet files are always scanned first.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir, raw_dir, import_note = resolve_data_and_raw_dirs(args.data_root)
    output_dir = data_dir / "audit" / "id_dtype_audit"

    if not raw_dir.exists():
        raise FileNotFoundError(f"RAW_DIR does not exist: {raw_dir}")
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"RAW_DIR is not a directory: {raw_dir}")

    table_files = iter_table_files(raw_dir, include_csv=args.include_csv)
    profiles: list[FileCandidate] = []
    file_errors: dict[str, str] = {}

    for table_file in table_files:
        relative_file = table_file.relative_to(raw_dir).as_posix()
        file_profiles, file_error = profile_file(table_file, raw_dir, args.sample_size)
        profiles.extend(file_profiles)
        if file_error:
            file_errors[relative_file] = file_error

    groups = build_column_groups(profiles)
    summary = build_summary(
        data_dir=data_dir,
        raw_dir=raw_dir,
        output_dir=output_dir,
        include_csv=args.include_csv,
        sample_size=args.sample_size,
        import_note=import_note,
        scanned_files=table_files,
        profiles=profiles,
        groups=groups,
        file_errors=file_errors,
    )
    write_reports(output_dir, summary, groups, profiles, file_errors)

    print(f"Reports written to: {output_dir}")
    print(f"Files scanned: {summary['files_discovered']}")
    print(f"ID-like column occurrences: {summary['candidate_column_file_occurrences']}")
    print(f"Canonical ID-like columns: {summary['canonical_column_count']}")
    print(f"Repeated column names: {summary['repeated_column_name_count']}")
    print(f"Dtype drift cases: {summary['dtype_drift_count']}")
    print(f"Dotted-suffix cases: {summary['dotted_suffix_column_count']}")
    print(f"Unknown/review-required cases: {summary['unknown_review_required_count']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
