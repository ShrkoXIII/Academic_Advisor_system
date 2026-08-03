"""Phase 1 temporal split for ``2026-08_temporal_rebuild_v1``.

Builds TRAIN / VALID / provisional TEST candidates from the safest pre-feature
source and writes every Phase 1 report. Adds no feature, changes no identifier,
and writes only under ``MODEL_DATA_VERSIONS_DIR / 2026-08_temporal_rebuild_v1``.

Boundaries (Decisions_Log.md 2026-08-03 Amendment 2, Correction 1 — authoritative):

    TRAIN = every eligible row chronologically through 20233
    VALID = the whole of academic year 2024 (20241 + 20242 + 20243)
    TEST  = 20251 only, provisional; 20252 excluded as incomplete

Why ``model_split_path`` is not used
------------------------------------
``src/paths.py`` validates ``generation`` against exactly base/difficulty/
concurrent/final and always yields ``df_{split}_{generation}.parquet``. A Phase 1
candidate is neither a live generation nor safe to carry a live basename.
Widening that validated vocabulary would be a production change made for a
candidate's convenience, so candidates are written directly under the versioned
phase directories instead and ``model_split_path`` is left untouched.
"""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import FINAL_DIR, MODEL_DATA_VERSIONS_DIR, RAW_DIR  # noqa: E402

REBUILD_VERSION = "2026-08_temporal_rebuild_v1"
FROZEN_VERSION = "2026-07-26_batched_fixes__registration_roster_concurrent"

VERSION_ROOT = MODEL_DATA_VERSIONS_DIR / REBUILD_VERSION
SPLIT_DIR = VERSION_ROOT / "01_split"
PREFLIGHT_DIR = VERSION_ROOT / "00_preflight"
BASELINE_MANIFEST = PREFLIGHT_DIR / "current_artifacts_baseline_manifest.csv"

SOURCE_PATH = FINAL_DIR / "without_outliers.parquet"
RAW_CRG_PATH = RAW_DIR / "v_crg_student_course_raw.parquet"
FROZEN_DIR = MODEL_DATA_VERSIONS_DIR / FROZEN_VERSION

SOURCE_ROW_KEY = "student_course_id"
CHRONOLOGY_COLUMN = "part_id"

TRAIN_MAX_PART = "20233"
VALID_PARTS = ("20241", "20242", "20243")
TEST_PARTS = ("20251",)

# Candidate basenames deliberately differ from every live basename.
TRAIN_OUT = SPLIT_DIR / "train_base_candidate.parquet"
VALID_OUT = SPLIT_DIR / "valid_base_candidate.parquet"
TEST_OUT = SPLIT_DIR / "test_provisional_base_candidate.parquet"

SPLIT_ASSIGNMENT_COLUMN = "split_assignment"
PIPELINE_VERSION_COLUMN = "pipeline_version"
EXCLUSION_REASON_COLUMN = "exclusion_reason"
PROVISIONAL_COLUMN = "test_provisional_20251_only"

# Columns present in the source that are engineered but split-INDEPENDENT: each
# is computed from a student's own chronological timeline, never from a
# train/valid boundary or a TRAIN-fitted statistic. They are carried through
# unchanged. Split-DEPENDENT features (course difficulty, concurrent peer) are
# absent from this source by construction and are Phase 3's job.
CARRIED_ENGINEERED_COLUMNS = (
    "university_id", "is_high_credit_course", "over_policy_semester_credits",
    "over_policy_semester_courses", "exclude_over_policy_semester",
    "is_extreme_fail_history", "total_fail_credits_capped",
    "is_interruption_semester", "prev_semester_was_interruption",
    "prior_interruption_count", "consecutive_interruption_count",
    "no_previous_progress", "is_first_active_semester",
    "is_first_row_in_timeline", "last_valid_gpa_before_current_semester",
    "prev_gpa_points_missing", "prev_gpa_points_zero",
    "prev_gpa_invalid_zero_case", "prev_gpa_points_clean",
    "prev_gpa_fill_source", "prev_gpa_replaced_due_to_invalid_zero",
    "prev_gpa_actual_zero_performance", "model_prev_gpa", "part_year",
    "part_semester", "start_year", "start_semester", "start_level_missing",
    "start_level_ord", "requirement_type_missing",
    "degree_requirement_credits_count_missing", "course_share_of_requirement",
    "requirement_size_bucket", "fail_credit_ratio_capped", "degree_course_key",
)


def stop(message: str) -> None:
    raise SystemExit(f"STOP GATE: {message}")


def frame_fingerprint(df: pd.DataFrame) -> str:
    """SHA-256 over serialised sorted content.

    Determinism is defined on content, not on Parquet bytes: Parquet writers
    embed non-deterministic metadata, so byte inequality is not a failure.
    """
    ordered = df.sort_values(SOURCE_ROW_KEY, kind="mergesort").reset_index(drop=True)
    ordered = ordered[sorted(ordered.columns)]
    return hashlib.sha256(
        ordered.to_csv(index=False).encode("utf-8")
    ).hexdigest()


def assign_split(part_id: pd.Series) -> pd.Series:
    """Deterministic, total assignment from the chronology column alone."""
    assignment = pd.Series("excluded", index=part_id.index, dtype="object")
    assignment[part_id <= TRAIN_MAX_PART] = "train"
    assignment[part_id.isin(VALID_PARTS)] = "valid"
    assignment[part_id.isin(TEST_PARTS)] = "test"
    return assignment


def exclusion_reason(part_id: pd.Series, assignment: pd.Series) -> pd.Series:
    reason = pd.Series("", index=part_id.index, dtype="object")
    excluded = assignment == "excluded"
    reason[excluded & (part_id == "20252")] = "20252_PARTIAL_FOUND_EXCLUDED"
    # `year_2024_semester_3_outside_declared_valid_enumeration` is void per the
    # 2026-08-03 Amendment 2, Correction 1: 20243 belongs in VALID.
    still_blank = excluded & (reason == "")
    reason[still_blank] = "unassigned_part_id_outside_declared_boundaries"
    return reason


def chronology_report(part_id: pd.Series) -> list[dict[str, object]]:
    rows = []
    counts = part_id.value_counts().sort_index()
    assignment = assign_split(pd.Series(counts.index, index=counts.index))
    for value, count in counts.items():
        rows.append(
            {
                "part_id": value,
                "row_count": int(count),
                "length": len(value),
                "all_digits": value.isdigit(),
                "year_prefix": value[:4],
                "semester_suffix": value[4:],
                "lexicographic_rank": int(
                    (counts.index < value).sum()  # type: ignore[operator]
                ),
                "split_assignment": assignment[value],
            }
        )
    return rows


def coverage_frame(frames: dict[str, pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    records = []
    for split, df in frames.items():
        for value, count in df.groupby(keys, dropna=False).size().items():
            values = value if isinstance(value, tuple) else (value,)
            record = {"split": split}
            record.update(dict(zip(keys, [str(v) for v in values])))
            record["row_count"] = int(count)
            records.append(record)
    return pd.DataFrame(records)


def describe(split: str, df: pd.DataFrame) -> dict[str, object]:
    marks = pd.to_numeric(df["final_mark"], errors="coerce")
    group_sizes = df.groupby(["student_id", CHRONOLOGY_COLUMN], dropna=False).size()
    return {
        "split": split,
        "row_count": int(len(df)),
        "student_count": int(df["student_id"].nunique()),
        "unique_degree_id": int(df["degree_id"].nunique()),
        "unique_course_id": int(df["course_id"].nunique()),
        "unique_degree_course_pairs": int(
            df.groupby(["degree_id", "course_id"], dropna=False).ngroups
        ),
        "min_part_id": str(df[CHRONOLOGY_COLUMN].min()),
        "max_part_id": str(df[CHRONOLOGY_COLUMN].max()),
        "duplicate_source_row_count": int(len(df) - df[SOURCE_ROW_KEY].nunique()),
        "null_student_course_id": int(df[SOURCE_ROW_KEY].isna().sum()),
        "null_student_id": int(df["student_id"].isna().sum()),
        "null_course_id": int(df["course_id"].isna().sum()),
        "null_degree_id": int(df["degree_id"].isna().sum()),
        "null_part_id": int(df[CHRONOLOGY_COLUMN].isna().sum()),
        "null_final_mark": int(marks.isna().sum()),
        "raw_pass_rate_mark_ge_50": (
            round(float((marks >= 50).mean()), 6) if len(df) else None
        ),
        "semester_group_size_mean": round(float(group_sizes.mean()), 4),
        "semester_group_size_median": float(group_sizes.median()),
        "semester_group_size_max": int(group_sizes.max()),
        # student_status_id is high-cardinality (≈130k distinct values in TRAIN),
        # i.e. an identifier rather than a status enum. Emitting the full
        # distribution would bloat this file to megabytes and inform nobody, so
        # the cardinality plus the top 20 are recorded instead.
        "status_distinct_count": int(df["student_status_id"].nunique(dropna=False)),
        "status_null_count": int(df["student_status_id"].isna().sum()),
        "status_distribution_top20": {
            str(k): int(v)
            for k, v in df["student_status_id"]
            .value_counts(dropna=False)
            .head(20)
            .items()
        },
        "content_sha256": frame_fingerprint(df),
    }


def main() -> int:
    if not BASELINE_MANIFEST.is_file():
        stop(
            f"baseline manifest missing at {BASELINE_MANIFEST}. "
            "Run scripts/rebuild_2026_08_preflight.py first; the baseline must "
            "exist before any candidate dataset is written."
        )
    for path in (TRAIN_OUT, VALID_OUT, TEST_OUT):
        if path.exists():
            stop(f"candidate output already exists and would be overwritten: {path}")
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    started = datetime.now().astimezone()
    source = pd.read_parquet(SOURCE_PATH).reset_index(drop=True)

    # --- chronology validation -------------------------------------------
    part = source[CHRONOLOGY_COLUMN].astype("string")
    if part.isna().any():
        stop(f"{CHRONOLOGY_COLUMN} contains nulls; chronology cannot be proven")
    part = part.astype(str)
    lengths = sorted(part.str.len().unique())
    if lengths != [5]:
        stop(f"{CHRONOLOGY_COLUMN} values are not all length 5: {lengths}")
    if not part.str.fullmatch(r"\d{5}").all():
        stop(f"{CHRONOLOGY_COLUMN} contains a non-numeric value")
    source[CHRONOLOGY_COLUMN] = part

    if source[SOURCE_ROW_KEY].isna().any():
        stop(f"{SOURCE_ROW_KEY} contains nulls; source rows are not identifiable")
    duplicate_keys = source[SOURCE_ROW_KEY][source[SOURCE_ROW_KEY].duplicated()]
    if len(duplicate_keys):
        stop(f"{SOURCE_ROW_KEY} is not unique: {len(duplicate_keys)} duplicates")

    # --- assignment -------------------------------------------------------
    source[SPLIT_ASSIGNMENT_COLUMN] = assign_split(source[CHRONOLOGY_COLUMN])
    source[EXCLUSION_REASON_COLUMN] = exclusion_reason(
        source[CHRONOLOGY_COLUMN], source[SPLIT_ASSIGNMENT_COLUMN]
    )
    source[PIPELINE_VERSION_COLUMN] = REBUILD_VERSION

    counts = Counter(source[SPLIT_ASSIGNMENT_COLUMN])
    if sum(counts.values()) != len(source):
        stop("assignment is not total")

    frames = {
        "train": source[source[SPLIT_ASSIGNMENT_COLUMN] == "train"].copy(),
        "valid": source[source[SPLIT_ASSIGNMENT_COLUMN] == "valid"].copy(),
        "test": source[source[SPLIT_ASSIGNMENT_COLUMN] == "test"].copy(),
    }
    excluded = source[source[SPLIT_ASSIGNMENT_COLUMN] == "excluded"].copy()

    # --- invariants -------------------------------------------------------
    if (frames["train"][CHRONOLOGY_COLUMN] > TRAIN_MAX_PART).any():
        stop("TRAIN contains a row after 20233")
    if not frames["valid"][CHRONOLOGY_COLUMN].isin(VALID_PARTS).all():
        stop("VALID contains a row outside 20241/20242")
    if not frames["test"][CHRONOLOGY_COLUMN].isin(TEST_PARTS).all():
        stop("TEST contains a row outside 20251")

    key_sets = {name: set(df[SOURCE_ROW_KEY]) for name, df in frames.items()}
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = key_sets[a] & key_sets[b]
        if overlap:
            stop(f"{len(overlap)} source rows appear in both {a} and {b}")

    reconciled = sum(len(df) for df in frames.values()) + len(excluded)
    if reconciled != len(source):
        stop(f"row reconciliation failed: {reconciled} != {len(source)}")
    if (excluded[EXCLUSION_REASON_COLUMN] == "").any():
        stop("an excluded row carries no explicit reason")

    found_20252 = int((source[CHRONOLOGY_COLUMN] == "20252").sum())
    provisional_20251_only = True  # 20252 confirmed partial; never included.
    frames["test"][PROVISIONAL_COLUMN] = provisional_20251_only

    # --- write candidates -------------------------------------------------
    fingerprints = {}
    for name, path in (("train", TRAIN_OUT), ("valid", VALID_OUT), ("test", TEST_OUT)):
        df = frames[name].sort_values(SOURCE_ROW_KEY, kind="mergesort").reset_index(
            drop=True
        )
        frames[name] = df
        fingerprints[name] = frame_fingerprint(df)
        df.to_parquet(path, index=False)

    # --- reports ----------------------------------------------------------
    summaries = [describe(name, frames[name]) for name in ("train", "valid", "test")]

    pd.DataFrame(chronology_report(source[CHRONOLOGY_COLUMN])).to_csv(
        SPLIT_DIR / "part_id_chronology_report.csv", index=False
    )
    pd.DataFrame(
        [{k: v for k, v in s.items() if k != "status_distribution_top20"} for s in summaries]
    ).to_csv(SPLIT_DIR / "split_row_counts.csv", index=False)

    coverage_frame(frames, ["degree_id"]).to_csv(
        SPLIT_DIR / "split_degree_coverage.csv", index=False
    )
    coverage_frame(frames, ["course_id"]).to_csv(
        SPLIT_DIR / "split_course_coverage.csv", index=False
    )
    coverage_frame(frames, ["degree_id", "course_id"]).to_csv(
        SPLIT_DIR / "split_degree_course_coverage.csv", index=False
    )

    excl_summary = (
        excluded.groupby([EXCLUSION_REASON_COLUMN, CHRONOLOGY_COLUMN])
        .size()
        .reset_index(name="row_count")
        .sort_values([EXCLUSION_REASON_COLUMN, CHRONOLOGY_COLUMN])
    )
    excl_summary.to_csv(SPLIT_DIR / "excluded_rows_by_reason.csv", index=False)

    dup_rows = []
    for name, df in frames.items():
        dups = df[df[SOURCE_ROW_KEY].duplicated(keep=False)]
        for key, count in dups[SOURCE_ROW_KEY].value_counts().items():
            dup_rows.append(
                {"split": name, SOURCE_ROW_KEY: key, "occurrences": int(count)}
            )
    pd.DataFrame(
        dup_rows or [{"split": "", SOURCE_ROW_KEY: "", "occurrences": 0}]
    ).to_csv(SPLIT_DIR / "duplicate_source_row_report.csv", index=False)

    summary = {
        "rebuild_version": REBUILD_VERSION,
        "generated_at": started.isoformat(timespec="seconds"),
        "source_dataset": str(SOURCE_PATH),
        "source_row_count": int(len(source)),
        "source_row_key": SOURCE_ROW_KEY,
        "chronology_column": CHRONOLOGY_COLUMN,
        "boundaries": {
            "train": f"{CHRONOLOGY_COLUMN} <= {TRAIN_MAX_PART}",
            "valid": list(VALID_PARTS),
            "test": list(TEST_PARTS),
        },
        PROVISIONAL_COLUMN: provisional_20251_only,
        "rows_20252_found_in_source": found_20252,
        "rows_20252_disposition": "20252_PARTIAL_FOUND_EXCLUDED",
        "splits": {s["split"]: s for s in summaries},
        "excluded_row_count": int(len(excluded)),
        "excluded_by_reason": {
            str(k): int(v)
            for k, v in excluded[EXCLUSION_REASON_COLUMN].value_counts().items()
        },
        "content_fingerprints_sha256": fingerprints,
        "outputs": {
            "train": str(TRAIN_OUT),
            "valid": str(VALID_OUT),
            "test": str(TEST_OUT),
        },
    }
    (SPLIT_DIR / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps({k: summary[k] for k in (
        "source_row_count", "excluded_row_count", PROVISIONAL_COLUMN,
        "rows_20252_found_in_source", "excluded_by_reason",
    )}, indent=2))
    for s in summaries:
        print(f"  {s['split']:>5}: rows={s['row_count']:>8,} "
              f"students={s['student_count']:>7,} "
              f"parts={s['min_part_id']}..{s['max_part_id']} "
              f"pass_rate={s['raw_pass_rate_mark_ge_50']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
