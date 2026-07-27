"""Build the registration-roster concurrent feature generation.

This job corrects only the peer-membership source used by the accepted
leave-one-out aggregation.  It reads the live difficulty/final splits as
immutable templates, reconstructs complete registration-time rosters from CRG,
enriches roster courses with leakage-safe difficulty, and writes a new
versioned generation for human review.

The job never trains a model, promotes a version, modifies a live split,
creates a Git commit, or pushes a branch.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from functools import reduce
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.concurrent_group_features import (  # noqa: E402
    AUDIT_CONCURRENT_FEATURES,
    CONCURRENT_FEATURE_COLUMNS,
    MODEL_CONCURRENT_FEATURES,
    STABLE_OCCURRENCE_KEY,
    compute_concurrent_group_features,
)
from src.course_difficulty import (  # noqa: E402
    AUDIT_OUTPUT_COLUMNS,
    DIFFICULTY_OUTPUT_COLUMNS,
    STAT_OUTPUT_COLUMNS,
    apply_difficulty_state,
    build_temporal_query_difficulty,
    load_difficulty_state,
)
from src.feature_engineering import SEMESTER_KEY  # noqa: E402
from src.model_training import CONCURRENT_44_FEATURES  # noqa: E402
from src.paths import (  # noqa: E402
    MODEL_DATA_DIR,
    MODEL_DATA_VERSIONS_DIR,
    PREPROCESSED_DIR,
    RAW_DIR,
    assert_data_root,
    model_split_path,
)
from src.registration_roster import (  # noqa: E402
    KNOWN_OUTCOME_COLUMNS,
    PEER_MEMBERSHIP_KEY,
    ROW_AUDIT_ONLY_COLUMNS,
    assert_unique_peer_membership,
    build_registration_roster,
    model_facing_roster,
    roster_row_audit,
)


SPLITS = ("train", "valid", "test")
FIVE_ID_COLUMNS = SEMESTER_KEY + ["course_id"]
ALIGNMENT_COLUMNS = [STABLE_OCCURRENCE_KEY] + FIVE_ID_COLUMNS

CURRENT_VERSION_FILE = MODEL_DATA_DIR / "CURRENT_VERSION.txt"
RAW_CRG_PATH = RAW_DIR / "v_crg_student_course_raw.parquet"
ACD_METADATA_PATH = (
    PREPROCESSED_DIR
    / "V_ACD_DEGREE_COURSE"
    / "clean_v_acd_degree_course.parquet"
)
PRIOR_BUILD_ID = "2026-07-23_160509__concurrent_group_feature"
PRIOR_BUILD_DIR = MODEL_DATA_VERSIONS_DIR / PRIOR_BUILD_ID

SUPERSESSION_STATEMENT = (
    "Concurrent LOO implementation is mathematically verified, but peer membership\n"
    "was derived from post-filter target rows. This version is superseded for model\n"
    "training by the registration-time peer-roster implementation."
)

EXPECTED_LEGACY_MODEL_POSITION = 35  # zero-based; pinned by the concurrent_44 contract.
MATERIAL_MEAN_SHIFT_THRESHOLD = 0.02
MATERIALLY_UNCHANGED_GAP_RATIO = 0.80
CHANGE_ATOL = 1e-12
CHANGE_RTOL = 1e-12

RAW_CRG_COLUMNS = [
    "student_course_id",
    "student_id",
    "course_id",
    "part_id",
    "degree_id",
    "faculty_id",
    "course_credits",
    "active",
    "register_status",
    "finish_status",
]

TARGET_BUILD_COLUMNS = list(
    dict.fromkeys(
        [
            *ALIGNMENT_COLUMNS,
            "faculty_id",
            "course_credits",
            "requirement_type_id",
            "degree_requirement_credits_count",
            "degree_course_key",
            "semester_reg_courses",
            "final_mark",
            "attempt_number",
            *DIFFICULTY_OUTPUT_COLUMNS,
        ]
    )
)

PRIOR_COMPARISON_COLUMNS = list(
    dict.fromkeys(
        [
            *ALIGNMENT_COLUMNS,
            "concurrent_peer_difficulty_mean",
            "concurrent_peer_difficulty_max",
            "concurrent_peer_difficulty_missing",
            "concurrent_peer_observed_count",
        ]
    )
)

APPROXIMATE_PRIOR_VALUES = {
    "peer_difficulty_mean": {
        "train": 0.186,
        "valid": 0.134,
        "test": 0.135,
    },
    "maximum_peer_count": {
        "train": 14,
        "valid": 7,
        "test": 7,
    },
}

_BASE_REUSE_KEYS = ["_part", "_degree", "_course"]
_CONTEXT_REUSE_KEYS = ["_faculty", "_requirement", "_rounded_credits"]
_FULL_REUSE_KEYS = _BASE_REUSE_KEYS + _CONTEXT_REUSE_KEYS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_record(path: Path, *, published_path: Path | None = None) -> dict[str, Any]:
    return {
        "path": str(published_path or path),
        "sha256": _sha256(path),
        "bytes": int(path.stat().st_size),
    }


def _json_default(value: Any) -> Any:
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def _records(frame: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    return frame.head(limit).to_dict(orient="records")


def _pct(count: int, total: int) -> float:
    return round(float(count / total * 100.0), 6) if total else 0.0


def _read_current_version(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Current-version marker not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            return value
    raise ValueError(f"Current-version marker has no build id: {path}")


def _null_safe_equal(left: pd.Series, right: pd.Series) -> np.ndarray:
    return (
        left.eq(right) | (left.isna() & right.isna())
    ).fillna(False).to_numpy(dtype=bool)


def _assert_positional_alignment(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    if len(left) != len(right):
        raise AssertionError(
            f"{label}: row counts differ ({len(left):,} vs {len(right):,})"
        )
    for column in columns:
        if column not in left.columns or column not in right.columns:
            raise KeyError(f"{label}: alignment column missing: {column}")
        equal = _null_safe_equal(
            left[column].reset_index(drop=True),
            right[column].reset_index(drop=True),
        )
        if not equal.all():
            first = int(np.flatnonzero(~equal)[0])
            raise AssertionError(
                f"{label}: positional mismatch at row {first}, column "
                f"{column!r}: {left[column].iloc[first]!r} != "
                f"{right[column].iloc[first]!r}"
            )


def _assert_template_alignment(diff_path: Path, final_path: Path, split: str) -> int:
    """Assert full positional identity of the five established ID columns."""

    diff_ids = pd.read_parquet(diff_path, columns=FIVE_ID_COLUMNS)
    final_ids = pd.read_parquet(final_path, columns=FIVE_ID_COLUMNS)
    _assert_positional_alignment(
        diff_ids,
        final_ids,
        FIVE_ID_COLUMNS,
        f"{split}: live difficulty/final",
    )
    return int(len(diff_ids))


def _assert_contract() -> dict[str, Any]:
    # Re-based 2026-07-27 for the concurrent_43 contract task: this gate used
    # to read the deprecated MODEL_FEATURES / EXPECTED_FEATURE_COUNT globals
    # from src.model_training, which are aliases for "whichever list a
    # maintainer last pointed them at". Now that concurrent_43 exists
    # (concurrent_44 minus this exact legacy indicator) and is a real,
    # separately selectable contract, this gate is re-pinned to the explicit,
    # named CONCURRENT_44_FEATURES list so it keeps protecting concurrent_44's
    # column position specifically, and does not silently start failing (or
    # silently stop checking anything) if those deprecated globals are ever
    # repointed. The dataset layout itself is unaffected: the builder still
    # writes all 8 concurrent columns (3 model + 5 audit) regardless of which
    # named model contract is selected for training.
    legacy = "concurrent_peer_difficulty_missing"
    semantic_audits = {
        "concurrent_peer_set_empty",
        "concurrent_peer_difficulty_values_missing",
    }
    gates = {
        "expected_feature_count_is_44": len(CONCURRENT_44_FEATURES) == 44,
        "model_feature_count_is_44": len(CONCURRENT_44_FEATURES) == 44,
        "legacy_indicator_in_model_features": legacy in CONCURRENT_44_FEATURES,
        "legacy_indicator_position_preserved": (
            legacy in CONCURRENT_44_FEATURES
            and CONCURRENT_44_FEATURES.index(legacy) == EXPECTED_LEGACY_MODEL_POSITION
        ),
        "semantic_indicators_are_audit_only": semantic_audits.isdisjoint(
            CONCURRENT_44_FEATURES
        ),
        "model_concurrent_contract_unchanged": MODEL_CONCURRENT_FEATURES
        == [
            "concurrent_peer_difficulty_mean",
            "concurrent_peer_difficulty_max",
            legacy,
        ],
        "all_concurrent_columns_are_unique": len(CONCURRENT_FEATURE_COLUMNS)
        == len(set(CONCURRENT_FEATURE_COLUMNS)),
    }
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise AssertionError(f"44-feature contract gates failed: {failed}")
    return {
        "gates": gates,
        "expected_feature_count": len(CONCURRENT_44_FEATURES),
        "model_feature_count": len(CONCURRENT_44_FEATURES),
        "legacy_indicator": legacy,
        "legacy_indicator_zero_based_position": CONCURRENT_44_FEATURES.index(legacy),
        "model_concurrent_features": MODEL_CONCURRENT_FEATURES,
        "audit_concurrent_features": AUDIT_CONCURRENT_FEATURES,
    }


def _read_target(path: Path) -> pd.DataFrame:
    schema = set(pq.ParquetFile(path).schema_arrow.names)
    missing = [column for column in TARGET_BUILD_COLUMNS if column not in schema]
    if missing:
        raise ValueError(f"{path} is missing builder columns: {missing}")
    return pd.read_parquet(path, columns=TARGET_BUILD_COLUMNS)


def _read_raw_crg_for_parts(path: Path, part_values: pd.Series) -> pd.DataFrame:
    """Read only raw columns and semester partitions needed by one split."""

    numeric_parts = sorted(
        pd.to_numeric(part_values, errors="raise").dropna().unique().tolist()
    )
    filters = [("part_id", "in", [float(value) for value in numeric_parts])]
    try:
        raw = pd.read_parquet(path, columns=RAW_CRG_COLUMNS, filters=filters)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, TypeError, ValueError):
        # Some parquet engines cannot push an ``in`` predicate through a
        # non-partitioned float column. The fallback still reads only the ten
        # registration/audit columns, then filters before the roster builder
        # performs any normalization or copying.
        raw = pd.read_parquet(path, columns=RAW_CRG_COLUMNS)
        raw_part = pd.to_numeric(raw["part_id"], errors="coerce")
        raw = raw.loc[raw_part.isin(numeric_parts)].copy()
    return raw


def _stream_append_columns(
    source_path: Path,
    output_path: Path,
    features: pd.DataFrame,
    *,
    batch_size: int = 25_000,
) -> None:
    """Append all concurrent columns without changing any template column.

    The append is Arrow-native on purpose. A pandas round-trip
    (``to_pandas`` -> ``from_pandas``) re-derives the schema from pandas
    dtypes, which both risks silent dtype drift and re-positions the
    template's real ``__index_level_0__`` column. Carrying the source table
    through untouched and appending the concurrent arrays after it guarantees
    the output is exactly ``source columns (original order) + concurrent
    columns``, with the pandas index metadata preserved verbatim.
    """

    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_path}")
    parquet_file = pq.ParquetFile(source_path)
    source_schema = parquet_file.schema_arrow
    for column in CONCURRENT_FEATURE_COLUMNS:
        if column in source_schema.names:
            raise AssertionError(
                f"Template already contains concurrent column {column}: "
                f"{source_path}"
            )

    writer: pq.ParquetWriter | None = None
    offset = 0
    try:
        for record_batch in parquet_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([record_batch], schema=source_schema)
            end = offset + table.num_rows
            feature_batch = features.iloc[offset:end]
            if len(feature_batch) != table.num_rows:
                raise AssertionError(
                    f"Streaming row mismatch for {source_path.name}: "
                    f"source batch={table.num_rows}, features={len(feature_batch)}"
                )
            for column in CONCURRENT_FEATURE_COLUMNS:
                table = table.append_column(
                    column, pa.array(feature_batch[column].to_numpy())
                )
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="snappy",
                )
            writer.write_table(table)
            offset = end
    finally:
        if writer is not None:
            writer.close()

    if offset != len(features):
        raise AssertionError(
            f"Streaming row mismatch: source={offset:,}, features={len(features):,}"
        )


def _write_parquet_refuse(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {path}")
    frame.to_parquet(path, index=False)


def _verify_streamed_output(
    source_path: Path,
    output_path: Path,
    features: pd.DataFrame,
    split: str,
    generation: str,
) -> dict[str, Any]:
    """Read back every source column and all appended concurrent columns.

    The schema contract is exact: the output column names, in order, must be
    the source names followed by ``CONCURRENT_FEATURE_COLUMNS``. Extra columns,
    reordered columns, and any source dtype change are all rejected — a dtype
    difference is never cast away and accepted.
    """

    source_file = pq.ParquetFile(source_path)
    output_file = pq.ParquetFile(output_path)
    if source_file.metadata.num_rows != output_file.metadata.num_rows:
        raise AssertionError(f"{split}/{generation}: output row count changed")

    source_schema = source_file.schema_arrow
    output_schema = output_file.schema_arrow
    source_names = list(source_schema.names)
    output_names = list(output_schema.names)
    expected_names = source_names + list(CONCURRENT_FEATURE_COLUMNS)
    if output_names != expected_names:
        lost = [c for c in source_names if c not in output_names]
        unexpected = [c for c in output_names if c not in expected_names]
        raise AssertionError(
            f"{split}/{generation}: output schema is not "
            f"source + concurrent columns. Lost: {lost}; unexpected: "
            f"{unexpected}; expected order {expected_names[:3]}..."
            f"{expected_names[-3:]}, got {output_names[:3]}..."
            f"{output_names[-3:]}"
        )

    for column in source_names:
        before_type = source_schema.field(column).type
        after_type = output_schema.field(column).type
        if before_type != after_type:
            raise AssertionError(
                f"{split}/{generation}: dtype change for template column "
                f"{column}: {before_type} -> {after_type}"
            )
        before = pq.read_table(source_path, columns=[column]).column(0).combine_chunks()
        after = pq.read_table(output_path, columns=[column]).column(0).combine_chunks()
        if not before.equals(after):
            raise AssertionError(
                f"{split}/{generation}: template column changed: {column}"
            )

    readback = pd.read_parquet(output_path, columns=CONCURRENT_FEATURE_COLUMNS)
    if len(readback) != len(features):
        raise AssertionError(f"{split}/{generation}: feature row count changed")
    for column in CONCURRENT_FEATURE_COLUMNS:
        expected = features[column].reset_index(drop=True)
        actual = readback[column].reset_index(drop=True)
        if pd.api.types.is_float_dtype(expected.dtype):
            equal = np.isclose(
                pd.to_numeric(expected, errors="coerce"),
                pd.to_numeric(actual, errors="coerce"),
                rtol=CHANGE_RTOL,
                atol=CHANGE_ATOL,
                equal_nan=True,
            )
        else:
            equal = _null_safe_equal(expected, actual)
        if not bool(np.asarray(equal).all()):
            raise AssertionError(
                f"{split}/{generation}: appended feature changed on readback: "
                f"{column}"
            )

    return {
        "rows": int(source_file.metadata.num_rows),
        "source_columns_preserved": int(len(source_names)),
        "concurrent_columns_appended": list(CONCURRENT_FEATURE_COLUMNS),
        "schema_is_exactly_source_plus_concurrent": True,
        "source_dtypes_identical": True,
        "pandas_index_columns_preserved": (
            source_names.count("__index_level_0__")
            == output_names.count("__index_level_0__")
        ),
        "output_column_count": int(len(output_names)),
        "status": "pass",
    }


def _select_generated_target_difficulty(
    roster: pd.DataFrame,
    difficulty: pd.DataFrame,
    target: pd.DataFrame,
) -> pd.DataFrame:
    mask = roster["is_target_occurrence"].to_numpy(dtype=bool)
    positions = (
        roster.loc[mask, "target_row_position"].astype("int64").to_numpy()
    )
    if len(positions) != len(target):
        raise AssertionError(
            "Roster target-occurrence count does not equal target row count"
        )
    if not np.array_equal(np.sort(positions), np.arange(len(target))):
        raise AssertionError(
            "Roster target_row_position is not a one-to-one cover of targets"
        )
    selected = difficulty.iloc[np.flatnonzero(mask)].copy()
    selected["_target_row_position"] = positions
    selected = (
        selected.sort_values("_target_row_position", kind="stable")
        .drop(columns="_target_row_position")
        .reset_index(drop=True)
    )
    selected.index = target.index
    return selected.loc[:, DIFFICULTY_OUTPUT_COLUMNS]


def _assert_target_difficulty_matches_template(
    regenerated: pd.DataFrame,
    target: pd.DataFrame,
    split: str,
    gate_name: str,
) -> dict[str, Any]:
    expected = target.loc[:, DIFFICULTY_OUTPUT_COLUMNS]
    if len(regenerated) != len(expected):
        raise AssertionError(f"{split}: difficulty gate row count differs")

    max_abs_difference: dict[str, float | None] = {}
    for column in DIFFICULTY_OUTPUT_COLUMNS:
        left = regenerated[column].reset_index(drop=True)
        right = expected[column].reset_index(drop=True)
        if column in STAT_OUTPUT_COLUMNS:
            left_num = pd.to_numeric(left, errors="coerce").to_numpy(
                dtype="float64"
            )
            right_num = pd.to_numeric(right, errors="coerce").to_numpy(
                dtype="float64"
            )
            equal = np.isclose(
                left_num,
                right_num,
                rtol=CHANGE_RTOL,
                atol=CHANGE_ATOL,
                equal_nan=True,
            )
            finite = np.isfinite(left_num) & np.isfinite(right_num)
            max_abs_difference[column] = (
                float(np.max(np.abs(left_num[finite] - right_num[finite])))
                if finite.any()
                else None
            )
        else:
            equal = _null_safe_equal(left, right)
            max_abs_difference[column] = None
        if not bool(np.asarray(equal).all()):
            first = int(np.flatnonzero(~np.asarray(equal))[0])
            raise AssertionError(
                f"{split}: {gate_name} target difficulty differs from the live "
                f"template at row {first}, column {column}: "
                f"{left.iloc[first]!r} != {right.iloc[first]!r}"
            )

    return {
        "status": "pass",
        "rows_compared": int(len(expected)),
        "columns_compared": list(DIFFICULTY_OUTPUT_COLUMNS),
        "max_abs_difference": max_abs_difference,
    }


def _canonical_reuse_keys(frame: pd.DataFrame) -> pd.DataFrame:
    keys = pd.DataFrame(index=frame.index)
    keys["_part"] = (
        pd.to_numeric(frame["part_id"], errors="raise")
        .round()
        .astype("Int64")
        .astype("string")
        .fillna("<MISSING>")
    )
    keys["_degree"] = (
        frame["degree_id"].astype("string").str.strip().fillna("<MISSING>")
    )
    keys["_course"] = (
        frame["course_id"].astype("string").str.strip().fillna("<MISSING>")
    )
    keys["_faculty"] = (
        frame["faculty_id"].astype("string").str.strip().fillna("<MISSING>")
    )
    keys["_requirement"] = (
        frame["requirement_type_id"]
        .astype("string")
        .str.strip()
        .fillna("<MISSING>")
    )
    rounded = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    credits = pd.to_numeric(frame["course_credits"], errors="coerce")
    present = credits.notna()
    rounded.loc[present] = credits.loc[present].round().astype("int64")
    keys["_rounded_credits"] = rounded.astype("string").fillna("<MISSING>")
    return keys


def _reuse_same_semester_target_difficulty(
    roster: pd.DataFrame,
    generated: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reuse target difficulty according to the binding temporal priority.

    A degree-course-semester lookup is used whenever completed target rows have
    one consistent difficulty output for that base key, even if those targets
    have multiple registration contexts. Only base keys with conflicting target
    outputs are narrowed to faculty + requirement type + rounded credits. A
    conflicting output even at that full context is a hard error. Rows with no
    safe match retain the independently regenerated temporal value.
    """

    target_keys = _canonical_reuse_keys(target).reset_index(drop=True)
    target_values = target[DIFFICULTY_OUTPUT_COLUMNS].reset_index(drop=True)
    target_records = pd.concat([target_keys, target_values], axis=1)
    roster_keys = _canonical_reuse_keys(roster).reset_index(drop=True)
    roster_keys["_roster_position"] = np.arange(len(roster), dtype="int64")

    context_variants = (
        target_records[_BASE_REUSE_KEYS + _CONTEXT_REUSE_KEYS]
        .drop_duplicates()
        .groupby(_BASE_REUSE_KEYS, dropna=False, sort=False)
        .size()
        .rename("_context_variants")
        .reset_index()
    )
    output_variants = (
        target_records[_BASE_REUSE_KEYS + DIFFICULTY_OUTPUT_COLUMNS]
        .drop_duplicates()
        .groupby(_BASE_REUSE_KEYS, dropna=False, sort=False)
        .size()
        .rename("_output_variants")
        .reset_index()
    )
    base_summary = context_variants.merge(
        output_variants,
        on=_BASE_REUSE_KEYS,
        how="outer",
        validate="one_to_one",
    )
    safe_base_keys = base_summary.loc[
        base_summary["_output_variants"].eq(1),
        _BASE_REUSE_KEYS,
    ]
    conflict_base_keys = base_summary.loc[
        base_summary["_output_variants"].gt(1),
        _BASE_REUSE_KEYS,
    ]

    reuse_names = {
        column: f"_reuse_{column}" for column in DIFFICULTY_OUTPUT_COLUMNS
    }
    safe_lookup = (
        target_records.merge(
            safe_base_keys,
            on=_BASE_REUSE_KEYS,
            how="inner",
            validate="many_to_one",
        )
        .drop_duplicates(_BASE_REUSE_KEYS)
        .rename(columns=reuse_names)
    )
    safe_matches = roster_keys.merge(
        safe_lookup[
            _BASE_REUSE_KEYS
            + list(reuse_names.values())
        ],
        on=_BASE_REUSE_KEYS,
        how="inner",
        validate="many_to_one",
        sort=False,
    )

    conflicting_records = target_records.merge(
        conflict_base_keys,
        on=_BASE_REUSE_KEYS,
        how="inner",
        validate="many_to_one",
    )
    full_output_variants = (
        conflicting_records[_FULL_REUSE_KEYS + DIFFICULTY_OUTPUT_COLUMNS]
        .drop_duplicates()
        .groupby(_FULL_REUSE_KEYS, dropna=False, sort=False)
        .size()
    )
    conflicting_full = full_output_variants[full_output_variants > 1]
    if not conflicting_full.empty:
        raise AssertionError(
            "Same-semester target difficulty conflicts within the full "
            "registration context; refusing arbitrary reuse. Examples: "
            f"{conflicting_full.head(5).index.tolist()}"
        )

    full_lookup = (
        conflicting_records.drop_duplicates(
            _FULL_REUSE_KEYS + DIFFICULTY_OUTPUT_COLUMNS
        )
        .drop_duplicates(_FULL_REUSE_KEYS)
        .rename(columns=reuse_names)
    )
    full_matches = roster_keys.merge(
        full_lookup[_FULL_REUSE_KEYS + list(reuse_names.values())],
        on=_FULL_REUSE_KEYS,
        how="inner",
        validate="many_to_one",
        sort=False,
    )

    result = generated.loc[:, DIFFICULTY_OUTPUT_COLUMNS].copy()
    strategy = np.full(len(roster), "temporal_regeneration", dtype=object)
    for matches, label in (
        (safe_matches, "base_key_consistent_target_value"),
        (full_matches, "full_registration_context"),
    ):
        positions = matches["_roster_position"].to_numpy(dtype="int64")
        for column in DIFFICULTY_OUTPUT_COLUMNS:
            result.iloc[
                positions, result.columns.get_loc(column)
            ] = matches[f"_reuse_{column}"].to_numpy()
        strategy[positions] = label

    target_mask = roster["is_target_occurrence"].to_numpy(dtype=bool)
    roster_only = ~target_mask
    reused = strategy != "temporal_regeneration"

    # Registration-context spread must be measured over the ROSTER rows that
    # actually consumed each base key. Counting it over target rows alone
    # under-reports: a roster-only row that reused an anchor under a DIFFERENT
    # faculty / requirement / rounded-credits context is invisible there, which
    # is why train previously reported 0 despite such rows existing.
    target_contexts = target_records[_FULL_REUSE_KEYS].drop_duplicates()
    target_contexts["_context_seen_in_target"] = True
    roster_context = roster_keys.merge(
        target_contexts,
        on=_FULL_REUSE_KEYS,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    context_seen_in_target = (
        roster_context["_context_seen_in_target"].fillna(False).to_numpy(dtype=bool)
    )
    reused_across_differing_context = reused & ~context_seen_in_target

    reused_roster_keys = roster_keys.loc[reused]
    roster_context_variants = (
        reused_roster_keys[_FULL_REUSE_KEYS]
        .drop_duplicates()
        .groupby(_BASE_REUSE_KEYS, dropna=False, sort=False)
        .size()
    )

    diagnostics = {
        "base_key": "part_id + degree_id + course_id",
        "conflict_context": (
            "faculty_id + requirement_type_id + rounded_course_credits"
        ),
        "base_keys_total": int(len(base_summary)),
        "base_keys_safe": int(len(safe_base_keys)),
        "base_keys_requiring_full_context": int(len(conflict_base_keys)),
        "base_keys_with_multiple_registration_contexts": int(
            roster_context_variants.gt(1).sum()
        ),
        "base_keys_with_multiple_registration_contexts_scope": (
            "distinct full contexts among the ROSTER rows that reused each "
            "base key"
        ),
        "base_keys_with_multiple_target_registration_contexts": int(
            base_summary["_context_variants"].gt(1).sum()
        ),
        "roster_rows_reused_across_a_differing_context": int(
            reused_across_differing_context.sum()
        ),
        "roster_only_rows_reused_across_a_differing_context": int(
            (roster_only & reused_across_differing_context).sum()
        ),
        "differing_context_reuse_note": (
            "Reuse matches on the base key by design; these rows took an "
            "anchor whose full registration context differs from their own. "
            "Bounded residual risk, tracked in Decisions_Log.md; tightening "
            "the merge is a separate one-change experiment."
        ),
        "rows_reused_by_base_key": int(
            (strategy == "base_key_consistent_target_value").sum()
        ),
        "rows_reused_by_full_context": int(
            (strategy == "full_registration_context").sum()
        ),
        "rows_kept_from_temporal_regeneration": int(
            (strategy == "temporal_regeneration").sum()
        ),
        "roster_only_rows_reused_by_base_key": int(
            (
                roster_only
                & (strategy == "base_key_consistent_target_value")
            ).sum()
        ),
        "roster_only_rows_reused_by_full_context": int(
            (roster_only & (strategy == "full_registration_context")).sum()
        ),
        "roster_only_rows_without_equivalent_same_semester_target": int(
            (roster_only & (strategy == "temporal_regeneration")).sum()
        ),
        "arbitrary_degree_course_semester_reuse_allowed": False,
    }
    return result, diagnostics


def _counts(series: pd.Series) -> dict[str, dict[str, float | int]]:
    total = len(series)
    return {
        str(key): {
            "count": int(value),
            "pct": round(float(value / total * 100.0), 6) if total else 0.0,
        }
        for key, value in series.value_counts(dropna=False).sort_index().items()
    }


def _numeric_distribution(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce")
    description = values.describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    )
    result: dict[str, Any] = {
        key: (None if pd.isna(value) else float(value))
        for key, value in description.items()
    }
    result["null_count"] = int(values.isna().sum())
    result["null_pct"] = _pct(int(values.isna().sum()), len(values))
    return result


def _change_summary(old: pd.Series, new: pd.Series) -> dict[str, Any]:
    old_values = pd.to_numeric(old, errors="coerce").to_numpy(dtype="float64")
    new_values = pd.to_numeric(new, errors="coerce").to_numpy(dtype="float64")
    same = np.isclose(
        old_values,
        new_values,
        rtol=CHANGE_RTOL,
        atol=CHANGE_ATOL,
        equal_nan=True,
    )
    changed = ~same
    finite_both = np.isfinite(old_values) & np.isfinite(new_values)
    changed_finite = changed & finite_both
    return {
        "changed_count": int(changed.sum()),
        "changed_pct": _pct(int(changed.sum()), len(changed)),
        "unchanged_count": int(same.sum()),
        "old_null_new_value_count": int(
            (np.isnan(old_values) & np.isfinite(new_values)).sum()
        ),
        "old_value_new_null_count": int(
            (np.isfinite(old_values) & np.isnan(new_values)).sum()
        ),
        "mean_absolute_change_when_both_finite": (
            float(np.mean(np.abs(old_values[changed_finite] - new_values[changed_finite])))
            if changed_finite.any()
            else 0.0
        ),
    }


def _temporal_coverage(
    roster: pd.DataFrame,
    difficulty: pd.DataFrame,
    split: str,
) -> dict[str, Any]:
    roster_only = ~roster["is_target_occurrence"].to_numpy(dtype=bool)

    def subset(mask: np.ndarray) -> dict[str, Any]:
        part = difficulty.iloc[np.flatnonzero(mask)]
        usable = part["course_pass_rate_historical"].notna()
        return {
            "rows": int(len(part)),
            "usable_pass_rate_count": int(usable.sum()),
            "usable_pass_rate_pct": _pct(int(usable.sum()), len(part)),
            "stat_null_counts": {
                column: int(part[column].isna().sum())
                for column in STAT_OUTPUT_COLUMNS
            },
            "fallback_levels": _counts(part["difficulty_fallback_level"]),
            "course_difficulty_missing": _counts(
                part["course_difficulty_missing"]
            ),
        }

    all_rows = np.ones(len(roster), dtype=bool)
    problems: list[str] = []
    roster_only_nulls = int(
        difficulty.loc[
            roster_only, "course_pass_rate_historical"
        ].isna().sum()
    )
    if split in {"valid", "test"} and roster_only_nulls:
        problems.append(
            f"{roster_only_nulls} roster-only rows lack a pass-rate despite "
            "the frozen full-train state"
        )
    if split == "train" and roster_only_nulls:
        problems.append(
            "Train nulls are confined by the temporal mechanism; verify they "
            "are expected no-history/fallback cases in the report."
        )
    return {
        "policy": (
            "strictly prior target-history semesters only"
            if split == "train"
            else "frozen state fitted on complete promoted training window"
        ),
        "all_roster_rows": subset(all_rows),
        "roster_only_rows": subset(roster_only),
        "metadata_missing_counts_roster_only": {
            column: int(roster.loc[roster_only, column].isna().sum())
            for column in (
                "faculty_id",
                "requirement_type_id",
                "course_credits",
                "degree_course_key",
            )
        },
        "problems": problems,
    }


def _impact_diagnostics(
    target: pd.DataFrame,
    corrected: pd.DataFrame,
    prior: pd.DataFrame,
) -> dict[str, Any]:
    old_peer_count = pd.to_numeric(
        prior["concurrent_peer_observed_count"], errors="raise"
    ).to_numpy(dtype="int64")
    new_peer_count = pd.to_numeric(
        corrected["concurrent_peer_observed_count"], errors="raise"
    ).to_numpy(dtype="int64")
    if (new_peer_count < old_peer_count).any():
        first = int(np.flatnonzero(new_peer_count < old_peer_count)[0])
        raise AssertionError(
            "Corrected registration roster unexpectedly removed a post-filter "
            f"target peer at target row {first}"
        )
    membership_changed = new_peer_count != old_peer_count

    old_singleton = (
        pd.to_numeric(
            prior["concurrent_peer_difficulty_missing"], errors="raise"
        )
        .to_numpy(dtype="int64")
        .astype(bool)
    )
    new_singleton = (
        corrected["concurrent_peer_difficulty_missing"]
        .to_numpy(dtype="int64")
        .astype(bool)
    )
    if not np.array_equal(
        new_singleton,
        corrected["concurrent_peer_set_empty"].to_numpy(dtype="int64").astype(bool),
    ):
        raise AssertionError(
            "Legacy difficulty-missing indicator differs from corrected empty "
            "registered peer-set definition"
        )

    return {
        "target_rows": int(len(target)),
        "registered_peer_membership_changed": {
            "count": int(membership_changed.sum()),
            "pct": _pct(int(membership_changed.sum()), len(target)),
            "definition": (
                "Corrected roster is a proven superset of target occurrences; "
                "membership changes exactly when corrected peer count exceeds "
                "the post-filter peer count."
            ),
        },
        "peer_mean_changed": _change_summary(
            prior["concurrent_peer_difficulty_mean"],
            corrected["concurrent_peer_difficulty_mean"],
        ),
        "peer_maximum_changed": _change_summary(
            prior["concurrent_peer_difficulty_max"],
            corrected["concurrent_peer_difficulty_max"],
        ),
        "singletons": {
            "old_post_filter": {
                "count": int(old_singleton.sum()),
                "pct": _pct(int(old_singleton.sum()), len(target)),
            },
            "corrected_registration_roster": {
                "count": int(new_singleton.sum()),
                "pct": _pct(int(new_singleton.sum()), len(target)),
            },
            "removed_singletons": int(old_singleton.sum() - new_singleton.sum()),
        },
        "semantic_indicators": {
            "concurrent_peer_set_empty": {
                "count": int(
                    corrected["concurrent_peer_set_empty"].sum()
                ),
                "pct": _pct(
                    int(corrected["concurrent_peer_set_empty"].sum()),
                    len(target),
                ),
            },
            "concurrent_peer_difficulty_values_missing": {
                "count": int(
                    corrected[
                        "concurrent_peer_difficulty_values_missing"
                    ].sum()
                ),
                "pct": _pct(
                    int(
                        corrected[
                            "concurrent_peer_difficulty_values_missing"
                        ].sum()
                    ),
                    len(target),
                ),
                "empty_peer_set_value": 0,
            },
            "legacy_indicator_equals_peer_set_empty": True,
        },
        "peer_difficulty_mean_distribution": {
            "old_post_filter": _numeric_distribution(
                prior["concurrent_peer_difficulty_mean"]
            ),
            "corrected_registration_roster": _numeric_distribution(
                corrected["concurrent_peer_difficulty_mean"]
            ),
        },
        "maximum_peer_count": {
            "old_post_filter": int(old_peer_count.max(initial=0)),
            "corrected_registration_roster": int(
                new_peer_count.max(initial=0)
            ),
        },
    }


def _overall_impact(split_reports: dict[str, Any]) -> dict[str, Any]:
    total_rows = sum(
        split_reports[split]["impact"]["target_rows"] for split in SPLITS
    )

    def summed(path: tuple[str, ...]) -> int:
        return sum(
            int(
                reduce(
                    lambda value, key: value[key],
                    path,
                    split_reports[split]["impact"],
                )
            )
            for split in SPLITS
        )

    membership = summed(("registered_peer_membership_changed", "count"))
    mean_changed = summed(("peer_mean_changed", "changed_count"))
    maximum_changed = summed(("peer_maximum_changed", "changed_count"))
    old_singletons = summed(("singletons", "old_post_filter", "count"))
    corrected_singletons = summed(
        ("singletons", "corrected_registration_roster", "count")
    )
    values_missing = summed(
        (
            "semantic_indicators",
            "concurrent_peer_difficulty_values_missing",
            "count",
        )
    )
    comparable_groups = sum(
        split_reports[split]["roster"]["semester_count_match"][
            "comparable_group_count"
        ]
        for split in SPLITS
    )
    matched_groups = sum(
        split_reports[split]["roster"]["semester_count_match"][
            "matched_group_count"
        ]
        for split in SPLITS
    )
    return {
        "target_rows": total_rows,
        "registered_peer_membership_changed": {
            "count": membership,
            "pct": _pct(membership, total_rows),
        },
        "peer_mean_changed": {
            "count": mean_changed,
            "pct": _pct(mean_changed, total_rows),
        },
        "peer_maximum_changed": {
            "count": maximum_changed,
            "pct": _pct(maximum_changed, total_rows),
        },
        "old_singletons": {
            "count": old_singletons,
            "pct": _pct(old_singletons, total_rows),
        },
        "corrected_singletons": {
            "count": corrected_singletons,
            "pct": _pct(corrected_singletons, total_rows),
        },
        "concurrent_peer_difficulty_values_missing": {
            "count": values_missing,
            "pct": _pct(values_missing, total_rows),
            "empty_peer_set_value": 0,
        },
        "semester_registration_count_match": {
            "matched_group_count": matched_groups,
            "comparable_group_count": comparable_groups,
            "match_rate": (
                float(matched_groups / comparable_groups)
                if comparable_groups
                else None
            ),
        },
    }


def _distribution_shift(split_reports: dict[str, Any]) -> dict[str, Any]:
    prior_means = {
        split: split_reports[split]["impact"][
            "peer_difficulty_mean_distribution"
        ]["old_post_filter"]["mean"]
        for split in SPLITS
    }
    corrected_means = {
        split: split_reports[split]["impact"][
            "peer_difficulty_mean_distribution"
        ]["corrected_registration_roster"]["mean"]
        for split in SPLITS
    }

    comparisons: dict[str, Any] = {}
    for split in ("valid", "test"):
        prior_gap = float(prior_means["train"] - prior_means[split])
        corrected_gap = float(corrected_means["train"] - corrected_means[split])
        absolute_removed = abs(prior_gap) - abs(corrected_gap)
        retained_gap_ratio = (
            abs(corrected_gap) / abs(prior_gap) if prior_gap else None
        )
        comparisons[split] = {
            "prior_train_minus_split": prior_gap,
            "corrected_train_minus_split": corrected_gap,
            "absolute_gap_removed": absolute_removed,
            "absolute_gap_removed_pct": (
                float(absolute_removed / abs(prior_gap) * 100.0)
                if prior_gap
                else None
            ),
            "corrected_to_prior_absolute_gap_ratio": retained_gap_ratio,
            "materially_unchanged": (
                retained_gap_ratio is not None
                and retained_gap_ratio >= MATERIALLY_UNCHANGED_GAP_RATIO
            ),
            "material_shift_remains": (
                abs(corrected_gap) >= MATERIAL_MEAN_SHIFT_THRESHOLD
            ),
        }

    material_remains = any(
        values["material_shift_remains"] for values in comparisons.values()
    )
    materially_unchanged = any(
        values["materially_unchanged"] for values in comparisons.values()
    )
    return {
        "material_threshold_absolute_mean_gap": MATERIAL_MEAN_SHIFT_THRESHOLD,
        "materially_unchanged_gap_ratio_threshold": (
            MATERIALLY_UNCHANGED_GAP_RATIO
        ),
        "prior_means": prior_means,
        "corrected_means": corrected_means,
        "comparisons": comparisons,
        "material_train_vs_validation_or_test_shift_remains": material_remains,
        "earlier_shift_materially_unchanged": materially_unchanged,
        "interpretation": (
            "A remaining shift is diagnostic, not an automatic build failure."
        ),
        "recommendation": (
            "The earlier distribution shift remains materially unchanged after "
            "correcting peer membership, so withdrawal filtering was not its "
            "primary cause. Investigate temporal cohort changes, curriculum "
            "changes, course-offering composition, difficulty-state construction, "
            "support/fallback composition, registration-limit enforcement, and "
            "degree/program mix separately."
            if materially_unchanged
            else (
                "A material shift remains, but the registration-roster correction "
                "removed a non-trivial share of the earlier gap. Treat the residual "
                "as diagnostic and separately investigate temporal cohort changes, "
                "curriculum changes, course-offering composition, difficulty-state "
                "construction, support/fallback composition, registration-limit "
                "enforcement, and degree/program mix."
                if material_remains
                else "Registration-roster correction removed the material mean gap "
                "under the stated threshold; retain cohort/composition monitoring."
            )
        ),
    }


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Registration-time peer-roster correction - Phase 7 review",
        "",
        f"- Build: `{report['build_id']}`",
        "- Status: **awaiting explicit human approval**",
        "- Live model-data files: **untouched**",
        "- Model training, TEST-based selection, promotion, commits, and push: "
        "**not performed**",
        "",
        "## Precise supersession reason",
        "",
        "```text",
        SUPERSESSION_STATEMENT,
        "```",
        "",
        "## Impact summary",
        "",
    ]
    overall = report["overall_impact"]
    overall_match = overall["semester_registration_count_match"]
    lines.extend(
        [
            f"- Overall membership changed: "
            f"{overall['registered_peer_membership_changed']['count']:,}/"
            f"{overall['target_rows']:,} "
            f"({overall['registered_peer_membership_changed']['pct']:.3f}%).",
            f"- Overall peer mean changed: "
            f"{overall['peer_mean_changed']['count']:,}/"
            f"{overall['target_rows']:,} "
            f"({overall['peer_mean_changed']['pct']:.3f}%).",
            f"- Overall peer maximum changed: "
            f"{overall['peer_maximum_changed']['count']:,}/"
            f"{overall['target_rows']:,} "
            f"({overall['peer_maximum_changed']['pct']:.3f}%).",
            f"- Overall singletons: "
            f"{overall['old_singletons']['count']:,} "
            f"({overall['old_singletons']['pct']:.3f}%) old; "
            f"{overall['corrected_singletons']['count']:,} "
            f"({overall['corrected_singletons']['pct']:.3f}%) corrected.",
            f"- Overall registration-count match: "
            f"{overall_match['matched_group_count']:,}/"
            f"{overall_match['comparable_group_count']:,} "
            f"({overall_match['match_rate'] * 100:.3f}%).",
            f"- Registered peers existed but none had usable difficulty for "
            f"{overall['concurrent_peer_difficulty_values_missing']['count']:,} "
            f"target rows "
            f"({overall['concurrent_peer_difficulty_values_missing']['pct']:.3f}%). "
            "This indicator is pinned to 0 when the peer set is empty.",
            "",
        ]
    )
    lines.extend(
        [
        "| Split | Membership changed | Mean changed | Maximum changed | "
        "Old singletons | Corrected singletons | Count match |",
        "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in SPLITS:
        values = report["splits"][split]
        impact = values["impact"]
        match = values["roster"]["semester_count_match"]
        lines.append(
            f"| {split} | "
            f"{impact['registered_peer_membership_changed']['count']:,} "
            f"({impact['registered_peer_membership_changed']['pct']:.3f}%) | "
            f"{impact['peer_mean_changed']['changed_count']:,} "
            f"({impact['peer_mean_changed']['changed_pct']:.3f}%) | "
            f"{impact['peer_maximum_changed']['changed_count']:,} "
            f"({impact['peer_maximum_changed']['changed_pct']:.3f}%) | "
            f"{impact['singletons']['old_post_filter']['count']:,} "
            f"({impact['singletons']['old_post_filter']['pct']:.3f}%) | "
            f"{impact['singletons']['corrected_registration_roster']['count']:,} "
            f"({impact['singletons']['corrected_registration_roster']['pct']:.3f}%) | "
            f"{(match['match_rate'] or 0) * 100:.3f}% |"
        )

    lines.extend(
        [
            "",
            "## Peer difficulty distribution",
            "",
            "| Split | Prior mean | Corrected mean | Prior max peers | "
            "Corrected max peers | Approximate prior reference |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split in SPLITS:
        impact = report["splits"][split]["impact"]
        old_dist = impact["peer_difficulty_mean_distribution"]["old_post_filter"]
        new_dist = impact["peer_difficulty_mean_distribution"][
            "corrected_registration_roster"
        ]
        lines.append(
            f"| {split} | {old_dist['mean']:.6f} | {new_dist['mean']:.6f} | "
            f"{impact['maximum_peer_count']['old_post_filter']} | "
            f"{impact['maximum_peer_count']['corrected_registration_roster']} | "
            f"{APPROXIMATE_PRIOR_VALUES['peer_difficulty_mean'][split]:.3f}, "
            f"max {APPROXIMATE_PRIOR_VALUES['maximum_peer_count'][split]} |"
        )

    shift = report["distribution_shift"]
    lines.extend(
        [
            "",
            "## Train-versus-validation/test shift",
            "",
        ]
    )
    for split, values in shift["comparisons"].items():
        lines.append(
            f"- {split}: prior absolute gap "
            f"`{abs(values['prior_train_minus_split']):.6f}`, corrected "
            f"`{abs(values['corrected_train_minus_split']):.6f}`, removed "
            f"`{values['absolute_gap_removed']:.6f}` "
            f"({values['absolute_gap_removed_pct']:.2f}% of the prior gap)."
        )
    lines.extend(
        [
            f"- Material shift remains: "
            f"**{shift['material_train_vs_validation_or_test_shift_remains']}** "
            f"(absolute mean-gap threshold "
            f"`{shift['material_threshold_absolute_mean_gap']}`).",
            f"- {shift['recommendation']}",
            "",
            "## Roster-count residuals",
            "",
        ]
    )
    for split in SPLITS:
        roster = report["splits"][split]["roster"]
        match = roster["semester_count_match"]
        lines.append(
            f"### {split}: {match['matched_group_count']:,}/"
            f"{match['comparable_group_count']:,} comparable groups match "
            f"({(match['match_rate'] or 0) * 100:.3f}%)"
        )
        lines.append("")
        for reason, count in match["mismatch_reason_counts"].items():
            lines.append(f"- `{reason}`: {count:,}")
        examples = report["splits"][split]["residual_mismatch_examples"]
        if examples:
            lines.append("")
            lines.append("Residual examples:")
            for example in examples:
                identity = ", ".join(
                    f"{column}={example.get(column)!r}"
                    for column in (
                        "university_id",
                        "student_id",
                        "degree_id",
                        "part_id",
                    )
                )
                lines.append(
                    "- "
                    f"{identity}; roster="
                    f"{example.get('registration_roster_occurrence_count')!r}, "
                    f"recorded={example.get('semester_reg_courses')!r}, "
                    f"delta={example.get('registration_count_delta')!r}, "
                    f"reason="
                    f"`{example.get('registration_count_mismatch_reason')}`"
                )
        lines.append("")

    lines.extend(
        [
            "## Temporal difficulty",
            "",
            "- Train roster queries use only completed target history with "
            "`history part_id < query part_id`; query/withdrawal rows never "
            "update the accumulator.",
            "- Validation and test roster queries use the promoted full-train "
            "frozen state and never fit on validation/test.",
            "- Same-semester degree-course values are reused whenever completed "
            "targets have one consistent output; conflicting base-key outputs "
            "narrow to faculty, requirement type, and rounded credits.",
            "- Every regenerated target difficulty row matched its live template "
            "before any output was published.",
            "",
        ]
    )
    for split in SPLITS:
        coverage = report["splits"][split]["temporal_difficulty"]
        subset = coverage["roster_only_rows"]
        lines.append(
            f"- {split}: roster-only usable pass-rate "
            f"{subset['usable_pass_rate_count']:,}/{subset['rows']:,} "
            f"({subset['usable_pass_rate_pct']:.3f}%); problems: "
            f"{coverage['problems'] or 'none'}."
        )

    lines.extend(
        [
            "",
            "## Versioned outputs",
            "",
        ]
    )
    for name, values in report["outputs"].items():
        lines.append(
            f"- `{name}`: `{values['path']}` - SHA-256 "
            f"`{values['sha256']}`"
        )
    lines.extend(
        [
            "",
            "A separate `SHA256SUMS.txt` also hashes the JSON and Markdown "
            "reports. No live file was changed.",
        ]
    )
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> Path:
    template_dir = Path(args.template_dir)
    output_root = Path(args.output_root)
    raw_crg_path = Path(args.raw_crg)
    acd_metadata_path = Path(args.acd_metadata)
    current_version_file = Path(args.current_version_file)
    comparison_dir = Path(args.comparison_dir)

    promoted_build_id = _read_current_version(current_version_file)
    difficulty_state_dir = (
        Path(args.difficulty_state_dir)
        if args.difficulty_state_dir
        else output_root / promoted_build_id / "difficulty_state"
    )

    template_paths = {
        f"{split}_{generation}": model_split_path(
            split, generation, template_dir
        )
        for split in SPLITS
        for generation in ("difficulty", "final")
    }
    prior_paths = {
        split: model_split_path(split, "concurrent", comparison_dir)
        for split in SPLITS
    }
    assert_data_root(
        *template_paths.values(),
        *prior_paths.values(),
        raw_crg_path,
        acd_metadata_path,
        current_version_file,
        difficulty_state_dir / "manifest.json",
    )

    build_id = args.build_id or datetime.now().astimezone().strftime(
        "%Y-%m-%d_%H%M%S__registration_roster_concurrent"
    )
    output_dir = output_root / build_id
    staging = output_root / f".{build_id}.incomplete"
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite versioned dataset: {output_dir}"
        )
    if staging.exists():
        raise FileExistsError(
            f"Refusing to overwrite incomplete build: {staging}"
        )

    feature_contract = _assert_contract()
    for split in SPLITS:
        _assert_template_alignment(
            template_paths[f"{split}_difficulty"],
            template_paths[f"{split}_final"],
            split,
        )

    live_paths = {
        **template_paths,
        "current_version_marker": current_version_file,
    }
    live_before = {
        name: _path_record(path) for name, path in live_paths.items()
    }

    state = load_difficulty_state(difficulty_state_dir)
    clean_acd = pd.read_parquet(acd_metadata_path)

    output_root.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True)
    split_reports: dict[str, Any] = {}
    readback_checks: dict[str, Any] = {}
    staged_outputs: dict[str, Path] = {}

    try:
        for split in SPLITS:
            diff_path = template_paths[f"{split}_difficulty"]
            final_path = template_paths[f"{split}_final"]
            target = _read_target(diff_path)
            raw_crg = _read_raw_crg_for_parts(raw_crg_path, target["part_id"])

            roster_result = build_registration_roster(
                raw_crg,
                target,
                clean_acd,
            )
            roster = roster_result.roster
            leaked_outcomes = sorted(KNOWN_OUTCOME_COLUMNS.intersection(roster.columns))
            if leaked_outcomes:
                raise AssertionError(
                    f"{split}: outcome columns entered registration roster: "
                    f"{leaked_outcomes}"
                )

            if split == "train":
                generated_difficulty = build_temporal_query_difficulty(
                    target,
                    roster,
                    state.config,
                    include_source=False,
                )
            else:
                generated_difficulty = apply_difficulty_state(
                    roster,
                    state,
                    include_source=False,
                )

            generated_target = _select_generated_target_difficulty(
                roster,
                generated_difficulty,
                target,
            )
            pre_reuse_gate = _assert_target_difficulty_matches_template(
                generated_target,
                target,
                split,
                "regenerated",
            )

            roster_difficulty, reuse_diagnostics = (
                _reuse_same_semester_target_difficulty(
                    roster,
                    generated_difficulty,
                    target,
                )
            )
            reused_target = _select_generated_target_difficulty(
                roster,
                roster_difficulty,
                target,
            )
            post_reuse_gate = _assert_target_difficulty_matches_template(
                reused_target,
                target,
                split,
                "post-reuse",
            )

            enriched_roster = pd.concat(
                [
                    roster.reset_index(drop=True),
                    roster_difficulty.reset_index(drop=True),
                ],
                axis=1,
            )
            concurrent = compute_concurrent_group_features(
                target,
                enriched_roster,
            )
            if not concurrent.index.equals(target.index) or len(concurrent) != len(
                target
            ):
                raise AssertionError(
                    f"{split}: two-input concurrent features changed target order"
                )
            if not (
                concurrent["concurrent_peer_difficulty_missing"]
                .eq(concurrent["concurrent_peer_set_empty"])
                .all()
            ):
                raise AssertionError(
                    f"{split}: legacy indicator is not the empty peer-set flag"
                )
            if (
                concurrent["concurrent_peer_difficulty_missing"].dtype
                != np.dtype("int64")
            ):
                raise AssertionError(
                    f"{split}: legacy indicator dtype changed: "
                    f"{concurrent['concurrent_peer_difficulty_missing'].dtype}"
                )

            prior = pd.read_parquet(
                prior_paths[split],
                columns=PRIOR_COMPARISON_COLUMNS,
            )
            _assert_positional_alignment(
                target,
                prior,
                ALIGNMENT_COLUMNS,
                f"{split}: live template/prior comparison build",
            )
            impact = _impact_diagnostics(target, concurrent, prior)

            # The model-facing roster carries no current-semester outcome and
            # no target proxy; row-level diagnostics move to a separate audit
            # artifact that no feature or training code reads.
            published_roster = model_facing_roster(enriched_roster)
            assert_unique_peer_membership(published_roster)
            roster_audit = roster_row_audit(enriched_roster)

            concurrent_out = model_split_path(split, "concurrent", staging)
            final_out = model_split_path(split, "final", staging)
            roster_out = staging / f"registration_roster_{split}.parquet"
            roster_audit_out = (
                staging / f"registration_roster_row_audit_{split}.parquet"
            )
            count_audit_out = (
                staging / f"registration_roster_count_comparison_{split}.parquet"
            )
            mismatch_out = (
                staging / f"registration_roster_mismatch_examples_{split}.parquet"
            )
            excluded_register_out = (
                staging / f"registration_roster_excluded_status_{split}.parquet"
            )

            _stream_append_columns(diff_path, concurrent_out, concurrent)
            _stream_append_columns(final_path, final_out, concurrent)
            _write_parquet_refuse(published_roster, roster_out)
            _write_parquet_refuse(roster_audit, roster_audit_out)
            _write_parquet_refuse(
                roster_result.semester_count_comparison,
                count_audit_out,
            )
            _write_parquet_refuse(roster_result.mismatch_examples, mismatch_out)
            _write_parquet_refuse(
                roster_result.excluded_registration_status_summary,
                excluded_register_out,
            )

            readback_checks[f"{split}_concurrent"] = _verify_streamed_output(
                diff_path,
                concurrent_out,
                concurrent,
                split,
                "concurrent",
            )
            readback_checks[f"{split}_final"] = _verify_streamed_output(
                final_path,
                final_out,
                concurrent,
                split,
                "final",
            )
            output_schema = pq.ParquetFile(final_out).schema_arrow
            legacy_arrow_type = str(
                output_schema.field("concurrent_peer_difficulty_missing").type
            )
            if legacy_arrow_type != "int64":
                raise AssertionError(
                    f"{split}: persisted legacy indicator dtype is "
                    f"{legacy_arrow_type}, expected int64"
                )

            for label, path in (
                (f"df_{split}_concurrent", concurrent_out),
                (f"df_{split}_final", final_out),
                (f"registration_roster_{split}", roster_out),
                (f"registration_roster_row_audit_{split}", roster_audit_out),
                (f"registration_count_comparison_{split}", count_audit_out),
                (f"registration_mismatch_examples_{split}", mismatch_out),
                (f"registration_excluded_status_{split}", excluded_register_out),
            ):
                staged_outputs[label] = path

            split_reports[split] = {
                "target_rows": int(len(target)),
                "roster_rows": int(len(roster)),
                "model_facing_roster": {
                    "peer_membership_key": list(PEER_MEMBERSHIP_KEY),
                    "rows": int(len(published_roster)),
                    "columns": int(len(published_roster.columns)),
                    "duplicate_membership_rows": int(
                        published_roster.duplicated(
                            PEER_MEMBERSHIP_KEY, keep=False
                        ).sum()
                    ),
                    "row_audit_columns_removed": list(ROW_AUDIT_ONLY_COLUMNS),
                    "row_audit_artifact_rows": int(len(roster_audit)),
                    "outcome_or_target_proxy_columns": [],
                    "gates": {
                        "unique_on_peer_membership_key": True,
                        "no_outcome_or_target_proxy_column": True,
                    },
                },
                "roster": roster_result.diagnostics,
                "residual_mismatch_examples": _records(
                    roster_result.mismatch_examples,
                    limit=30,
                ),
                "same_semester_target_reuse": reuse_diagnostics,
                "target_difficulty_gate": {
                    "before_reuse": pre_reuse_gate,
                    "after_reuse": post_reuse_gate,
                },
                "temporal_difficulty": _temporal_coverage(
                    roster,
                    roster_difficulty,
                    split,
                ),
                "impact": impact,
                "legacy_indicator_persisted_dtype": legacy_arrow_type,
            }

            del (
                target,
                raw_crg,
                roster_result,
                roster,
                generated_difficulty,
                generated_target,
                roster_difficulty,
                reused_target,
                enriched_roster,
                published_roster,
                roster_audit,
                concurrent,
                prior,
            )
            gc.collect()

        distribution_shift = _distribution_shift(split_reports)
        overall_impact = _overall_impact(split_reports)

        live_after = {
            name: _path_record(path) for name, path in live_paths.items()
        }
        live_unchanged = all(
            live_before[name]["sha256"] == live_after[name]["sha256"]
            and live_before[name]["bytes"] == live_after[name]["bytes"]
            for name in live_paths
        )
        if not live_unchanged:
            changed = [
                name
                for name in live_paths
                if live_before[name] != live_after[name]
            ]
            raise AssertionError(f"Live model-data files changed: {changed}")

        output_records = {
            label: _path_record(
                path,
                published_path=output_dir / path.relative_to(staging),
            )
            for label, path in staged_outputs.items()
        }
        temporal_problems = {
            split: split_reports[split]["temporal_difficulty"]["problems"]
            for split in SPLITS
            if split_reports[split]["temporal_difficulty"]["problems"]
        }

        report: dict[str, Any] = {
            "artifact": "registration_time_peer_roster_dataset",
            "build_id": build_id,
            "created_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "status": "awaiting_explicit_human_review",
            "supersession_statement": SUPERSESSION_STATEMENT,
            "scope": {
                "accepted_math_reused": True,
                "corrected_contract": (
                    "peer membership comes from complete active R/E "
                    "registration-time CRG roster"
                ),
                "model_training_performed": False,
                "test_used_for_experiment_selection": False,
                "version_promoted": False,
                "live_dataset_modified": False,
                "git_commit_created": False,
                "git_push_performed": False,
            },
            "inputs": {
                "live_template_dir": str(template_dir),
                "raw_crg": _path_record(raw_crg_path),
                "clean_acd_metadata": _path_record(acd_metadata_path),
                "promoted_build_id": promoted_build_id,
                "difficulty_state": {
                    "path": str(difficulty_state_dir),
                    "manifest_sha256": _sha256(
                        difficulty_state_dir / "manifest.json"
                    ),
                    "config": {
                        "min_support": state.config.min_support,
                        "shrinkage_k": state.config.shrinkage_k,
                        "target_threshold": state.config.target_threshold,
                        "version": state.config.version,
                    },
                },
                "prior_comparison_build": {
                    "build_id": PRIOR_BUILD_ID,
                    "path": str(comparison_dir),
                    "role": (
                        "mathematically verified post-filter peer-source "
                        "engineering comparison"
                    ),
                },
                "approximate_prior_reference_values": APPROXIMATE_PRIOR_VALUES,
            },
            "feature_contract": feature_contract,
            "overall_impact": overall_impact,
            "temporal_contract": {
                "train_roster": (
                    "query semester X sees target-history rows with part_id < X; "
                    "query rows never update state"
                ),
                "valid_test_roster": (
                    "promoted frozen full-train difficulty state only"
                ),
                "same_semester_target_reuse": (
                    "base degree+course+part whenever completed targets have one "
                    "consistent difficulty output; conflicting base outputs narrow "
                    "to full faculty+requirement+rounded-credits context; no "
                    "arbitrary mapping"
                ),
                "future_semesters_contribute_to_roster_peer_features": False,
            },
            "splits": split_reports,
            "distribution_shift": distribution_shift,
            "temporal_difficulty_coverage_or_fallback_problems": temporal_problems,
            "readback_checks": readback_checks,
            "outputs": output_records,
            "live_integrity": {
                "before": live_before,
                "after": live_after,
                "unchanged": live_unchanged,
                "confirmation": (
                    "All live difficulty/final templates and CURRENT_VERSION "
                    "marker retained identical SHA-256 hashes and byte sizes."
                ),
            },
            "review_gate": {
                "required_next_action": (
                    "Stop and obtain explicit human approval before any commits "
                    "or push."
                ),
                "model_training_allowed_in_this_build": False,
                "promotion_allowed_in_this_build": False,
            },
        }

        json_path = staging / "phase7_registration_roster_report.json"
        markdown_path = staging / "PHASE7_REGISTRATION_ROSTER_REPORT.md"
        json_path.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        markdown_path.write_text(
            _report_markdown(report),
            encoding="utf-8",
        )

        hash_targets = {
            **staged_outputs,
            "phase7_json_report": json_path,
            "phase7_markdown_report": markdown_path,
        }
        checksum_path = staging / "SHA256SUMS.txt"
        checksum_lines = [
            f"{_sha256(path)}  {path.relative_to(staging).as_posix()}"
            for _, path in sorted(hash_targets.items())
        ]
        checksum_path.write_text(
            "\n".join(checksum_lines) + "\n",
            encoding="utf-8",
        )

        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=MODEL_DATA_DIR,
        help="Read-only live difficulty/final templates.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=MODEL_DATA_VERSIONS_DIR,
        help="Version root; a new immutable build directory is created.",
    )
    parser.add_argument("--build-id")
    parser.add_argument(
        "--raw-crg",
        type=Path,
        default=RAW_CRG_PATH,
        help="Raw V_CRG_STUDENT_COURSE registration source.",
    )
    parser.add_argument(
        "--acd-metadata",
        type=Path,
        default=ACD_METADATA_PATH,
        help="Clean/normalizable V_ACD_DEGREE_COURSE metadata.",
    )
    parser.add_argument(
        "--current-version-file",
        type=Path,
        default=CURRENT_VERSION_FILE,
        help="Read-only promoted-version marker.",
    )
    parser.add_argument(
        "--difficulty-state-dir",
        type=Path,
        help=(
            "Override promoted difficulty_state. Default resolves from "
            "CURRENT_VERSION.txt under --output-root."
        ),
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=PRIOR_BUILD_DIR,
        help=(
            "Accepted post-filter comparison build "
            "2026-07-23_160509__concurrent_group_feature."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    output_dir = build(parse_args(argv))
    print(f"Corrected versioned dataset built: {output_dir}")
    print(
        "Phase 7 reports generated. STOP: human review and explicit approval "
        "are required before commits or push."
    )


if __name__ == "__main__":
    main()
