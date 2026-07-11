"""SUPERSEDED - DO NOT RUN (governance decision D1, 2026-07-07).

The diploma join moved upstream to note_books/merge/02_merge_diploma.ipynb,
which writes the distinct artifact MERGE_DIR/merged_with_diploma.parquet.
This script's historical in-place extension of the feature-engineered artifact
(now features/feature_engineered_primary.parquet) must never run again: the
feature-engineering stage (02_feature_engineering.ipynb) is its sole writer (contract 2,
docs/governance_contracts.md). Kept in place for history per plan Phase 4 -
git preserves it; the guard below makes it inert.
"""

raise RuntimeError(
    "merge_diploma.py is SUPERSEDED (D1): the diploma merge now happens upstream in "
    "02_merge_diploma, and after_fet_eng.parquet has a single writer (feature engineering). "
    "This script must not be executed. See docs/data_governance_plan.md and "
    "docs/governance_contracts.md (contract 2)."
)

# Historical header (pre-D1), kept for context:
# IMPORTANT: This is a STANDALONE merge script.
# It does NOT modify feature_engineering.py or model_training.py.
#
# DEPENDENCY WARNING: If feature_engineering.py is ever rerun and regenerates
# after_fet_eng.parquet from scratch, this script MUST be rerun again BEFORE
# split_diagnostics.ipynb is rerun, or diploma_gpa/diploma_type_id will be
# missing from all downstream model splits (df_train, df_valid, df_test).

import os
import re
from src.feature_engineering import _normalize_key_series
from src.paths import PREPROCESSED_DIR, AUDIT_DIR
import pandas as pd


# ── Configuration ──────────────────────────────────────────────────────────────
# TODO: Fill in the path to your diploma source file (parquet, CSV, or Excel).
DIPLOMA_SOURCE_PATH = PREPROCESSED_DIR / "V_add_academic_info" / "v_add_adcademic_info_cleaned.parquet"

# Must match PARQUET_PATH in split_diagnostics.ipynb — this is the file we extend.
AFTER_FET_ENG_PATH = AUDIT_DIR / "after_fet_eng.parquet"

# Set to True ONLY when you intentionally want to overwrite existing diploma columns
# (e.g. re-running after fixing the diploma source). Prevents silent double-merges.
OVERWRITE_EXISTING_DIPLOMA_COLUMNS = False

# ──────────────────────────────────────────────────────────────────────────────


# ── Step 1: Load diploma source ────────────────────────────────────────────────
if DIPLOMA_SOURCE_PATH is None:
    raise ValueError(
        "DIPLOMA_SOURCE_PATH is not set.\n"
        "Assign the path to your diploma source file at the top of this script.\n"
        "Expected columns (case-insensitive): student_id, diploma_gpa, diploma_type_id"
    )

_ext = os.path.splitext(DIPLOMA_SOURCE_PATH)[-1].lower()
if _ext == ".parquet":
    diploma_df = pd.read_parquet(DIPLOMA_SOURCE_PATH)
elif _ext in (".csv", ".txt"):
    diploma_df = pd.read_csv(DIPLOMA_SOURCE_PATH)
elif _ext in (".xlsx", ".xls"):
    diploma_df = pd.read_excel(DIPLOMA_SOURCE_PATH)
else:
    raise ValueError(
        f"Unsupported file extension '{_ext}' for DIPLOMA_SOURCE_PATH. "
        "Supported: .parquet, .csv, .txt, .xlsx, .xls"
    )

print(f"Step 1 — Loaded diploma source: {DIPLOMA_SOURCE_PATH}")
print(f"  Shape     : {diploma_df.shape}")
print(f"  Columns   : {list(diploma_df.columns)}")
print(f"  dtypes:\n{diploma_df.dtypes.to_string()}")


# ── Step 2: Normalize column names ─────────────────────────────────────────────
diploma_df.columns = diploma_df.columns.str.lower()
print(f"\nStep 2 — Normalized column names: {list(diploma_df.columns)}")


# ── Step 3: Keep only student_id, diploma_gpa, diploma_type_id ─────────────────
# Drop any text certificate-name column or other extras if present.
_required_cols = ['student_id', 'diploma_gpa', 'diploma_type_id']
_missing_required = [c for c in _required_cols if c not in diploma_df.columns]
if _missing_required:
    raise AssertionError(
        f"Required columns missing from diploma source (after lowercasing): {_missing_required}\n"
        f"Available columns: {list(diploma_df.columns)}"
    )

_dropped = [c for c in diploma_df.columns if c not in _required_cols]
if _dropped:
    print(f"\nStep 3 — Dropping non-required columns: {_dropped}")
else:
    print(f"\nStep 3 — No extra columns to drop.")

diploma_df = diploma_df[_required_cols].copy()
print(f"  Kept columns : {_required_cols}")
print(f"  Row count    : {len(diploma_df):,}")
print(f"  Null counts  :\n{diploma_df.isnull().sum().to_string()}")
_source_nulls = diploma_df[_required_cols].isna().sum()
assert _source_nulls.sum() == 0, (
    "Diploma source has nulls after keeping required columns:\n"
    f"{_source_nulls.to_string()}"
)

# ── Step 4: Assert unique by student_id ─────────────────────────────────────────
_dup_mask = diploma_df['student_id'].duplicated(keep=False)
_dup_count = int(_dup_mask.sum())
assert _dup_count == 0, (
    f"Diploma source has {_dup_count} rows that share a duplicate student_id "
    f"({diploma_df['student_id'].duplicated().sum():,} extra rows beyond the first occurrence). "
    "Resolve duplicates in the source before merging."
)
print(f"\nStep 4 — student_id uniqueness: PASSED ({len(diploma_df):,} unique student_ids)")


# ── Step 5: Suffix-consistency check on diploma_type_id (BEFORE stripping) ─────
# Uses the same regex pattern as _derive_university_id in feature_engineering.py.
# r"\.([^.]+)$" extracts the dotted suffix that identifies the university
# (e.g. 15.111 -> suffix "111", 16.111 -> suffix "111").
# If more than one distinct suffix exists, the IDs span multiple universities
# and the bucketing logic must be revisited before proceeding.
_id_as_str = diploma_df['diploma_type_id'].astype('string')
_suffix_series = _id_as_str.str.extract(r'\.([^.]+)$', expand=False)
_unique_suffixes = set(_suffix_series.dropna().unique())

print(f"\nStep 5 — Suffix-consistency check on diploma_type_id:")
print(f"  Unique suffixes found: {_unique_suffixes}")

if len(_unique_suffixes) > 1:
    raise AssertionError(
        f"diploma_type_id spans more than one university suffix: {_unique_suffixes}.\n"
        "Each suffix identifies a different university. Rows from multiple universities\n"
        "must be kept separate for bucketing to be meaningful.\n"
        "The merge/bucketing logic must be revisited before proceeding."
    )

if len(_unique_suffixes) == 0:
    print(
        "  WARNING: No dotted suffix found in diploma_type_id values.\n"
        "  Values may already be clean integers — no stripping will be performed."
    )


# ── Step 6: Strip suffix and cast diploma_type_id to clean int ─────────────────
# ONLY diploma_type_id is touched. student_id is NEVER modified.
diploma_df['student_id'] = _normalize_key_series(diploma_df['student_id'])
if len(_unique_suffixes) == 1:
    _stripped_suffix = next(iter(_unique_suffixes))
    print(f"\nStep 6 — Suffix '.{_stripped_suffix}' is uniform — stripping and casting to int.")        
    diploma_df['diploma_type_id'] = (
        _id_as_str
        .str.split('.')
        .str[0]
        .astype("Int64")
    )
    print(f"  Suffix stripped : .{_stripped_suffix}")
    print(f"  dtype after cast: {diploma_df['diploma_type_id'].dtype}")
    print(f"  Sample values   : {diploma_df['diploma_type_id'].head(5).tolist()}")
else:
    print("\nStep 6 — No suffix to strip. Casting diploma_type_id to int directly.")
    diploma_df['diploma_type_id'] = (
        pd.to_numeric(diploma_df['diploma_type_id'], errors='coerce')
        .round()
        .astype('Int64')
    )

print(f"\n  diploma_type_id value_counts in source (after suffix strip):")
print(diploma_df['diploma_type_id'].value_counts(dropna=False).to_string())

print("\nStep 6b — student_id normalized to match project convention:")
print(f"  dtype (diploma source)      : {diploma_df['student_id'].dtype}")
print(f"  Sample (diploma source)     : {diploma_df['student_id'].head(3).tolist()}")

# ── Step 7: Read after_fet_eng.parquet ──────────────────────────────────────────
print(f"\nStep 7 — Loading after_fet_eng.parquet:")
print(f"  Path: {AFTER_FET_ENG_PATH}")
df = pd.read_parquet(AFTER_FET_ENG_PATH)
print(f"  Shape: {df.shape}")

# دلوقتي df موجود فعلاً، نقدر نقارن
print("\nStep 7b — Compare student_id format against after_fet_eng.parquet:")
print(f"  dtype (after_fet_eng.parquet): {df['student_id'].dtype}")
print(f"  Sample (after_fet_eng.parquet): {df['student_id'].head(3).tolist()}")


# ── Step 8: Guard against duplicate columns from a previous run ──────────────────
_existing_diploma_cols = [c for c in ['diploma_gpa', 'diploma_type_id'] if c in df.columns]
if _existing_diploma_cols:
    if not OVERWRITE_EXISTING_DIPLOMA_COLUMNS:
        raise AssertionError(
            f"Columns {_existing_diploma_cols} already exist in after_fet_eng.parquet.\n"
            "This means the script has already been run. To overwrite, set:\n"
            "    OVERWRITE_EXISTING_DIPLOMA_COLUMNS = True\n"
            "at the top of this script and rerun. Do not set this flag accidentally."
        )
    print(
        f"\nStep 8 — OVERWRITE_EXISTING_DIPLOMA_COLUMNS=True: "
        f"dropping {_existing_diploma_cols} before re-merge."
    )
    df = df.drop(columns=_existing_diploma_cols)
else:
    print(f"\nStep 8 — No existing diploma columns in after_fet_eng.parquet: OK")


# ── Step 9: Print shape BEFORE merge ─────────────────────────────────────────────
_n_rows_before = len(df)
_n_cols_before = df.shape[1]
print(f"\nStep 9 — BEFORE merge:")
print(f"  Row count    : {_n_rows_before:,}")
print(f"  Column count : {_n_cols_before}")


# ── Step 10: Left-join diploma columns onto the feature-engineered frame ──────────
# validate='many_to_one' raises if diploma_df has duplicate student_ids (belt-and-suspenders
# in addition to the Step 4 assertion above).
df_merged = pd.merge(
    df,
    diploma_df[['student_id', 'diploma_gpa', 'diploma_type_id']],
    on='student_id',
    how='left',
    validate='many_to_one',
)


# ── Step 11: Assert row count is IDENTICAL after merge ──────────────────────────
_n_rows_after = len(df_merged)
assert _n_rows_after == _n_rows_before, (
    f"Row count changed after merge — this must NEVER happen with a left join!\n"
    f"  Before : {_n_rows_before:,}\n"
    f"  After  : {_n_rows_after:,}\n"
    f"  Delta  : {_n_rows_after - _n_rows_before:+,}\n"
    "Investigate: possible key dtype mismatch between student_id in df and diploma_df."
)
print(f"\nStep 11 — Row count AFTER merge: {_n_rows_after:,}  (UNCHANGED — OK)")


# ── Step 12: Match-coverage diagnostics ──────────────────────────────────────────
_matched_rows       = int(df_merged['diploma_type_id'].notna().sum())
_unmatched_rows     = int(df_merged['diploma_type_id'].isna().sum())
_unmatched_pct      = _unmatched_rows / _n_rows_after * 100
_unmatched_students = int(
    df_merged.loc[df_merged['diploma_type_id'].isna(), 'student_id'].nunique()
)

print(f"\nStep 12 — Match-coverage diagnostics:")
print(f"  Matched rows   (diploma_type_id not null)     : {_matched_rows:,}")
print(f"  !! Unmatched rows (diploma_type_id null)      : {_unmatched_rows:,}  ({_unmatched_pct:.2f}%)")
print(f"  !! Unique student_ids with no diploma match   : {_unmatched_students:,}")
if _unmatched_rows > 0:
    print(
        "\n  NOTE: Unmatched rows occur when a student in the course data has no\n"
        "  record in the diploma source. This is expected for some student populations\n"
        "  (e.g. non-graduated students). Review the counts above and confirm the\n"
        "  unmatched percentage is within your expected range before saving."
    )


# ── Step 13: Null counts — reported separately, visibly ─────────────────────────
_null_gpa  = int(df_merged['diploma_gpa'].isna().sum())
_null_type = int(df_merged['diploma_type_id'].isna().sum())

print(f"\nStep 13 — Null counts after merge (review these carefully):")
print(f"  diploma_gpa        nulls : {_null_gpa:,}")
print(f"  diploma_type_id    nulls : {_null_type:,}")
if _unmatched_rows > 0:
    print(
        "  (Nulls above are expected for unmatched students — "
        "see Step 12 for details. Not an automatic error.)"
    )


# ── Step 14: diploma_type_id value_counts — first look ─────────────────────────
print(f"\nStep 14 — diploma_type_id value_counts after merge (whole dataset, first look only):")
print(f"  (dropna=False so nulls are visible)")
print(df_merged['diploma_type_id'].value_counts(dropna=False).to_string())


# ── Step 15: Overwrite after_fet_eng.parquet in place ────────────────────────────
df_merged.to_parquet(AFTER_FET_ENG_PATH, index=True)
_size_mb = os.path.getsize(AFTER_FET_ENG_PATH) / 1_048_576

print(f"\nStep 15 — Saved to: {AFTER_FET_ENG_PATH}")
print(f"  Final shape  : {df_merged.shape}")
print(f"  New columns  : diploma_gpa, diploma_type_id")
print(f"  File size    : {_size_mb:.1f} MB")
print(f"\nDone. Re-run split_diagnostics.ipynb next to propagate diploma columns to the splits.")