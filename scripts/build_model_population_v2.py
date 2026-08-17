"""Build the v2 model population: remove barred-absence rows and zero marks.

Reads ``data/final/without_outliers.parquet`` and writes
``data/final/model_population_v2.parquet``. The input is opened read-only and
never rewritten.

Two removals, in this order and for two different reasons:

* **Filter A — barred absence.** Selected by ``grade_id`` ALONE, never by the
  mark value. ``686.111``/``987.111``/``96.111`` all carry the label
  ``محروم بالغياب`` (finish_status ``FA``, declared range 0-0): the student was
  barred for absence, so no exam was sat and the stored ``0`` is an
  administrative placeholder rather than a score. Selecting these by code and
  then ASSERTING every removed row is ``final_mark == 0`` keeps the code the
  criterion while still proving the placeholder claim; if the assertion ever
  fails the premise is wrong and this script refuses to write.
* **Filter B — remaining zero marks.** Every row still carrying
  ``final_mark == 0`` after A, whatever its grade code. Its grade breakdown is
  printed so the composition is visible rather than assumed.

Grade codes ``684.111``, ``94.111``, ``985.111`` and ``995.111`` are NOT removed
by code. Their non-zero rows are carried through untouched; only their
zero-mark rows leave, via Filter B. Their wider handling is an open decision and
this script deliberately does not pre-empt it.

Nothing is written unless the removal totals land within
``TOTAL_TOLERANCE_ROWS`` of the pre-registered expectation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import FINAL_DIR  # noqa: E402

SOURCE_PATH = FINAL_DIR / "without_outliers.parquet"
OUTPUT_PATH = FINAL_DIR / "model_population_v2.parquet"
REPORT_PATH = PROJECT_ROOT / "models" / "diagnostics" / "model_population_v2_report.json"

TARGET = "final_mark"
GRADE_COLUMN = "grade_id"
CHRONOLOGY_COLUMN = "part_id"

# Barred-for-absence codes across the three grade-scheme versions. Selected by
# code, never by mark.
BARRED_CODES = ["686.111", "987.111", "96.111"]

# Deliberately untouched by code; open decision.
UNRESOLVED_CODES = ["684.111", "94.111", "985.111", "995.111"]

# Pre-registered expectations. A build that misses these is not the build that
# was reviewed, so it does not get written.
EXPECTED_SOURCE_ROWS = 749_523
EXPECTED_FILTER_A = 1_117
EXPECTED_FILTER_B = 2_733
EXPECTED_TOTAL_REMOVED = 3_850
EXPECTED_REMAINING = 745_673
TOTAL_TOLERANCE_ROWS = 10

# v1 semester boundaries, restated here only to pre-register the split sizes the
# rebuild will produce. This script performs no split.
V1_TRAIN_MAX_PART = "20233"
V1_VALID_PARTS = ("20241", "20242", "20243")
V1_TEST_PARTS = ("20251",)

FAIL_STATUSES = ("F", "FA", "FE")
GRADE_LOOKUP_PATH = (
    PROJECT_ROOT / "data" / "preprocessed" / "V_ACS_GRADE" / "clean_v_acs_grade.parquet"
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_epoch": float(stat.st_mtime),
        "mtime_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(timespec="microseconds"),
    }


def breakdown(frame: pd.DataFrame) -> list[dict[str, object]]:
    total = len(frame)
    return [
        {"grade_id": str(code), "count": int(count),
         "pct": round(float(count / total * 100), 4) if total else 0.0}
        for code, count in frame[GRADE_COLUMN].value_counts().items()
    ]


def main() -> int:
    if OUTPUT_PATH.exists():
        raise SystemExit(f"STOP: refusing to overwrite existing output: {OUTPUT_PATH}")

    source_before = file_state(SOURCE_PATH)
    df = pd.read_parquet(SOURCE_PATH)
    df[GRADE_COLUMN] = df[GRADE_COLUMN].astype(str)
    marks = pd.to_numeric(df[TARGET], errors="raise")
    source_rows = len(df)

    print(f"Source              : {SOURCE_PATH}")
    print(f"Source rows         : {source_rows:,} (expected {EXPECTED_SOURCE_ROWS:,})")
    if source_rows != EXPECTED_SOURCE_ROWS:
        print(
            f"  NOTE: source row count differs from the pre-registered "
            f"{EXPECTED_SOURCE_ROWS:,} by {source_rows - EXPECTED_SOURCE_ROWS:+,}"
        )

    # --- Filter A: barred absence, by grade code alone ---------------------
    mask_a = df[GRADE_COLUMN].isin(BARRED_CODES)
    removed_a = df[mask_a]
    n_a = int(mask_a.sum())

    # The premise of Filter A is that these codes never carry a real score.
    # Prove it rather than assume it; a violation invalidates the filter.
    offenders = removed_a[marks[mask_a] != 0]
    if len(offenders):
        raise AssertionError(
            f"Filter A premise violated: {len(offenders)} barred-absence row(s) "
            f"carry a non-zero {TARGET}. Codes: "
            f"{sorted(offenders[GRADE_COLUMN].unique().tolist())}; "
            f"marks {sorted(pd.to_numeric(offenders[TARGET]).unique().tolist())[:10]}. "
            "Refusing to write."
        )

    print()
    print(f"Filter A (barred absence, by grade_id only): {n_a:,} removed "
          f"(expected {EXPECTED_FILTER_A:,})")
    for row in breakdown(removed_a):
        print(f"  {row['grade_id']:>9} : {row['count']:>6,}")
    print(f"  assertion: every removed row has {TARGET} == 0  -> PASS")

    kept_after_a = df[~mask_a]
    marks_after_a = marks[~mask_a]

    # --- Filter B: remaining zero marks ------------------------------------
    mask_b = marks_after_a == 0
    removed_b = kept_after_a[mask_b]
    n_b = int(mask_b.sum())

    print()
    print(f"Filter B (remaining {TARGET} == 0): {n_b:,} removed "
          f"(expected {EXPECTED_FILTER_B:,})")
    b_rows = breakdown(removed_b)
    for row in b_rows:
        note = "  <- open-decision code" if row["grade_id"] in UNRESOLVED_CODES else ""
        print(f"  {row['grade_id']:>9} : {row['count']:>6,} ({row['pct']:>7.4f}%){note}")

    result = kept_after_a[~mask_b].copy()
    total_removed = n_a + n_b
    remaining = len(result)

    print()
    print(f"Total removed       : {total_removed:,} (expected {EXPECTED_TOTAL_REMOVED:,})")
    print(f"Remaining           : {remaining:,} (expected {EXPECTED_REMAINING:,})")

    deviation = abs(total_removed - EXPECTED_TOTAL_REMOVED)
    if deviation > TOTAL_TOLERANCE_ROWS:
        print()
        print(f"DISCREPANCY: total removed deviates by {deviation:,} rows, "
              f"tolerance is {TOTAL_TOLERANCE_ROWS}.")
        print(f"  filter A: {n_a:,} vs expected {EXPECTED_FILTER_A:,} "
              f"({n_a - EXPECTED_FILTER_A:+,})")
        print(f"  filter B: {n_b:,} vs expected {EXPECTED_FILTER_B:,} "
              f"({n_b - EXPECTED_FILTER_B:+,})")
        print("STOP: output file was NOT written.")
        return 1

    if remaining != EXPECTED_REMAINING:
        print(f"  NOTE: remaining differs from expectation by "
              f"{remaining - EXPECTED_REMAINING:+,} (within removal tolerance)")

    # --- observations only, nothing is acted on ----------------------------
    lookup = pd.read_parquet(GRADE_LOOKUP_PATH)
    lookup[GRADE_COLUMN] = lookup[GRADE_COLUMN].astype(str)
    status = lookup.set_index(GRADE_COLUMN)["finish_status"].astype("string")
    result_status = result[GRADE_COLUMN].map(status)
    result_marks = pd.to_numeric(result[TARGET], errors="raise")

    contradiction = result[(result_marks >= 50) & result_status.isin(FAIL_STATUSES)]
    print()
    print(f"OBSERVATION — remaining rows with {TARGET} >= 50 and finish_status in "
          f"{FAIL_STATUSES}: {len(contradiction):,}")
    contradiction_rows = breakdown(contradiction)
    for row in contradiction_rows:
        print(f"  {row['grade_id']:>9} : {row['count']:>6,} "
              f"(finish_status {status.get(row['grade_id'])})")
    print("  Not acted on: these codes are an open decision.")

    part = result[CHRONOLOGY_COLUMN].astype(str)
    buckets = {
        "train_le_20233": int((part <= V1_TRAIN_MAX_PART).sum()),
        "valid_20241_20243": int(part.isin(V1_VALID_PARTS).sum()),
        "test_20251": int(part.isin(V1_TEST_PARTS).sum()),
    }
    buckets["other_excluded_by_v1_boundaries"] = int(
        remaining - sum(buckets.values())
    )
    print()
    print("OBSERVATION — v1 semester boundaries applied to the OUTPUT "
          "(this script performs no split):")
    for name, value in buckets.items():
        print(f"  {name:<34}: {value:>8,}")

    # --- write --------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)

    output_sha = sha256_of(OUTPUT_PATH)
    print()
    print(f"Written             : {OUTPUT_PATH}")
    print(f"  rows              : {remaining:,}")
    print(f"  columns           : {len(result.columns)}")
    print(f"  sha256            : {output_sha}")

    source_after = file_state(SOURCE_PATH)
    unchanged = (
        source_before["size_bytes"] == source_after["size_bytes"]
        and source_before["mtime_epoch"] == source_after["mtime_epoch"]
    )
    print()
    print(f"Source unchanged    : {unchanged}")
    print(f"  size  before/after: {source_before['size_bytes']:,} / "
          f"{source_after['size_bytes']:,}")
    print(f"  mtime before/after: {source_before['mtime_utc']} / "
          f"{source_after['mtime_utc']}")
    if not unchanged:
        raise AssertionError("Source parquet changed during the build.")

    report = {
        "artifact": "model_population_v2",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source_before,
        "source_after": source_after,
        "source_unchanged": unchanged,
        "source_rows": source_rows,
        "filter_a": {
            "reason": "barred absence, selected by grade_id only",
            "codes": BARRED_CODES,
            "removed": n_a,
            "expected": EXPECTED_FILTER_A,
            "breakdown": breakdown(removed_a),
            "all_removed_rows_had_zero_mark": True,
        },
        "filter_b": {
            "reason": f"remaining {TARGET} == 0, any grade code",
            "removed": n_b,
            "expected": EXPECTED_FILTER_B,
            "breakdown": b_rows,
        },
        "untouched_open_decision_codes": UNRESOLVED_CODES,
        "total_removed": total_removed,
        "remaining": remaining,
        "output": {
            "path": str(OUTPUT_PATH),
            "rows": remaining,
            "columns": int(len(result.columns)),
            "sha256": output_sha,
        },
        "observation_fail_status_but_pass_mark": {
            "total": int(len(contradiction)),
            "by_grade_id": contradiction_rows,
            "acted_on": False,
        },
        "observation_v1_boundary_counts": buckets,
        "split_performed": False,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Report              : {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
