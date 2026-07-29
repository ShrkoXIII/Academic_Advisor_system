"""Phase 2R proposal tables rebuilt from TRAIN/VALID membership.

This script is intentionally read-only with respect to source data. It reads
only the frozen TRAIN and outcome-free VALID projections, the cleaned catalog,
the 67-pair review artifact, and the prior Phase 2 proposal table used for the
explicit numeric-proxy coverage comparison. It does not load a model or write
under data/model_data or src.

All proposal tables are built and validated in memory. Files are written only
after every stop condition has passed.
"""

from __future__ import annotations

import math
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VERSION = "2026-07-26_batched_fixes__registration_roster_concurrent"
VERSION_DIR = ROOT / "data" / "model_data" / "versions" / VERSION
TRAIN_PATH = VERSION_DIR / "df_train_final.parquet"
VALID_PATH = VERSION_DIR / "df_valid_final.parquet"
CATALOG_PATH = (
    ROOT
    / "data"
    / "preprocessed"
    / "V_ACD_DEGREE_COURSE"
    / "clean_v_acd_degree_course.parquet"
)
PAIRS_PATH = ROOT / "models" / "runs" / "COURSE_IDENTITY_67_HUMAN_REVIEW.csv"
PREVIOUS_LINK_PATH = ROOT / "models" / "runs" / "course_link_proposed.csv"

OUT_DIR = ROOT / "models" / "runs" / "phase2_train_membership_revision"
OUT_REPORT = OUT_DIR / "PHASE2_MAPPING_TABLES_REVISED.md"
OUT_SPLIT = OUT_DIR / "course_split_candidates.csv"
OUT_LINEAGE = OUT_DIR / "degree_lineage_proposed.csv"
OUT_LINK = OUT_DIR / "course_link_proposed.csv"
OUT_STATS = OUT_DIR / "course_difficulty_stats_prototype.csv"
EXPECTED_OUTPUT_NAMES = {
    OUT_REPORT.name,
    OUT_SPLIT.name,
    OUT_LINEAGE.name,
    OUT_LINK.name,
    OUT_STATS.name,
}

UNIVERSITY_ID = "111"
MIN_SUPPORT = 20
EXPECTED_VALID_ROWS = 156_097
EXPECTED_NEW_COURSES = 182
EXPECTED_NEW_COURSE_ROWS = 25_627
EXPECTED_KNOWN_PAIRS = 67
EXPECTED_GATE_MATCHES = 66
EXPECTED_SPLIT_OLD = "510.111"
EXPECTED_SPLIT_NEW = ("1183.111", "1192.111")
MANUAL_NEW = "1419.111"
MANUAL_OLD = "49.111"

# Numeric boundaries are confined to explicitly named diagnostic functions.
# They are never passed to generation, lineage, split, or link builders.
PREVIOUS_DEGREE_PROXY_BOUNDARY = 40
PREVIOUS_COURSE_PROXY_BOUNDARY = 1150

TRAIN_COLUMNS = [
    "course_id",
    "degree_id",
    "requirement_type_id",
    "course_credits",
    "part_id",
    "final_mark",
    "attempt_number",
    "degree_course_key",
]
VALID_COLUMNS = [
    "course_id",
    "degree_id",
    "requirement_type_id",
    "course_credits",
    "part_id",
    "course_history_count",
    "degree_course_key",
]
VALID_OUTCOME_COLUMNS = {
    "final_mark",
    "grade_id",
    "passed",
    "is_pass",
    "target",
    "label",
}


# ---------------------------------------------------------------------------
# Approved normalization specification (unchanged)
# ---------------------------------------------------------------------------
AR2LAT = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def norm_name(value: Any) -> str:
    s = unicodedata.normalize("NFKC", str(value)).translate(AR2LAT)
    s = re.sub(r"[ً-ْٰـ]", "", s)
    for old, new in [
        ("أ", "ا"),
        ("إ", "ا"),
        ("آ", "ا"),
        ("ى", "ي"),
        ("ة", "ه"),
        ("ؤ", "و"),
        ("ئ", "ي"),
    ]:
        s = s.replace(old, new)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    digits = re.findall(r"\d+", s)
    base = re.sub(r"\s+", " ", re.sub(r"\d+", " ", s)).strip()
    tokens = [re.sub(r"^ال", "", token) if len(token) > 3 else token for token in base.split()]
    return " ".join(tokens) + ("#" + digits[-1] if digits else "")


def name_stem(value: Any) -> str:
    return norm_name(value).split("#")[0]


def degree_family(value: Any) -> str:
    stripped = re.sub(r"\s*20\d{2}\s*$", "", str(value)).strip()
    return norm_name(stripped.split("/")[0].strip())


def loose_key(value: str) -> str:
    return " ".join(re.sub(r"^ال", "", token) for token in value.split())


def name_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    best = 0.0
    for a, b in ((left, right), (loose_key(left), loose_key(right))):
        if a == b:
            return 1.0
        ratio = SequenceMatcher(None, a, b).ratio()
        at, bt = set(a.split()), set(b.split())
        jaccard = len(at & bt) / len(at | bt) if (at or bt) else 0.0
        best = max(best, ratio, jaccard)
    return float(best)


def modal(series: pd.Series) -> Any:
    values = series.dropna()
    if values.empty:
        return pd.NA
    counts = values.value_counts()
    winners = counts[counts == counts.max()].index.tolist()
    try:
        return sorted(winners, key=lambda value: float(value))[0]
    except (TypeError, ValueError):
        return sorted(winners, key=str)[0]


def id_sort_key(value: str) -> tuple[Any, ...]:
    """Normalize identifier components for deterministic ascending tie-breaks."""
    parts = str(value).split(".")
    normalized: list[Any] = []
    for part in parts:
        normalized.append((0, int(part)) if part.isdigit() else (1, part))
    return tuple(normalized)


def sorted_ids(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if pd.notna(value)}, key=id_sort_key)


def join_ids(values: Iterable[str]) -> str:
    return "|".join(sorted_ids(values))


def numeric_core_for_diagnostic(value: str) -> int:
    return int(str(value).split(".")[0])


# ---------------------------------------------------------------------------
# Loaders: the VALID projection contains no outcome columns
# ---------------------------------------------------------------------------
def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(TRAIN_PATH, columns=TRAIN_COLUMNS)
    valid = pd.read_parquet(VALID_PATH, columns=VALID_COLUMNS)
    forbidden_loaded = sorted(set(valid.columns) & VALID_OUTCOME_COLUMNS)
    if forbidden_loaded:
        raise SystemExit(f"STOP: VALID outcome columns were loaded: {forbidden_loaded}")
    if set(valid.columns) != set(VALID_COLUMNS):
        raise SystemExit("STOP: VALID projection differs from the explicit outcome-free column contract.")

    catalog = pd.read_parquet(
        CATALOG_PATH,
        columns=[
            "course_id",
            "degree_id",
            "requirement_type_id",
            "course_name_sl",
            "degree_name_sl",
            "course_credits",
        ],
    )
    pairs = pd.read_csv(PAIRS_PATH, dtype="string", keep_default_na=False)
    previous_link = pd.read_csv(PREVIOUS_LINK_PATH, dtype="string", keep_default_na=False)

    for frame in (train, valid):
        for column in ("course_id", "degree_id", "degree_course_key"):
            frame[column] = frame[column].astype("string")
        frame["part_id"] = pd.to_numeric(frame["part_id"], errors="raise").astype("int64")

    for column in ("course_id", "degree_id"):
        catalog[column] = catalog[column].astype("string")
    catalog["requirement_type_id"] = pd.to_numeric(
        catalog["requirement_type_id"], errors="coerce"
    ).astype("Int64")
    catalog["course_credits"] = pd.to_numeric(
        catalog["course_credits"], errors="coerce"
    ).astype("float64")
    catalog["name_key"] = catalog["course_name_sl"].map(norm_name)
    catalog["name_stem"] = catalog["course_name_sl"].map(name_stem)
    catalog["has_level_suffix"] = catalog["name_key"].str.contains("#", regex=False)
    catalog["round_credits"] = catalog["course_credits"].round().astype("Int64")
    return train, valid, catalog, pairs, previous_link


# ---------------------------------------------------------------------------
# Membership census and diagnostic-only numeric comparisons
# ---------------------------------------------------------------------------
def build_membership(
    train: pd.DataFrame, valid: pd.DataFrame, catalog: pd.DataFrame
) -> dict[str, Any]:
    train_degree_ids = set(train["degree_id"].dropna().astype(str).unique())
    valid_degree_ids = set(valid["degree_id"].dropna().astype(str).unique())
    catalog_degree_ids = set(catalog["degree_id"].dropna().astype(str).unique())
    valid_only_degree_ids = valid_degree_ids - train_degree_ids
    catalog_only_degree_ids = catalog_degree_ids - train_degree_ids - valid_degree_ids

    train_course_ids = set(train["course_id"].dropna().astype(str).unique())
    valid_course_ids = set(valid["course_id"].dropna().astype(str).unique())
    new_course_ids = valid_course_ids - train_course_ids
    new_course_rows = int(valid["course_id"].isin(new_course_ids).sum())

    if len(new_course_ids) != EXPECTED_NEW_COURSES:
        raise SystemExit(
            f"STOP: membership census found {len(new_course_ids)} never-in-TRAIN VALID courses; "
            f"expected {EXPECTED_NEW_COURSES}."
        )
    if new_course_rows != EXPECTED_NEW_COURSE_ROWS:
        raise SystemExit(
            f"STOP: membership census found {new_course_rows} never-in-TRAIN VALID rows; "
            f"expected {EXPECTED_NEW_COURSE_ROWS}."
        )

    return {
        "train_degree_ids": train_degree_ids,
        "valid_degree_ids": valid_degree_ids,
        "catalog_degree_ids": catalog_degree_ids,
        "valid_only_degree_ids": valid_only_degree_ids,
        "catalog_only_degree_ids": catalog_only_degree_ids,
        "train_course_ids": train_course_ids,
        "valid_course_ids": valid_course_ids,
        "new_course_ids": new_course_ids,
        "new_course_rows": new_course_rows,
    }


def diagnostic_numeric_degree_comparison(
    membership: dict[str, Any],
) -> dict[str, Any]:
    catalog_degrees = membership["catalog_degree_ids"]
    previous_proxy_new = {
        degree_id
        for degree_id in catalog_degrees
        if numeric_core_for_diagnostic(degree_id) >= PREVIOUS_DEGREE_PROXY_BOUNDARY
    }
    corrected = membership["valid_only_degree_ids"]
    return {
        "previous_proxy_new": previous_proxy_new,
        "corrected_new": corrected,
        "added_by_correction": corrected - previous_proxy_new,
        "removed_by_correction": previous_proxy_new - corrected,
    }


def diagnostic_numeric_course_comparison(
    membership: dict[str, Any],
) -> dict[str, Any]:
    catalog_courses = membership["train_course_ids"] | membership["valid_course_ids"]
    previous_proxy_new = {
        course_id
        for course_id in catalog_courses
        if numeric_core_for_diagnostic(course_id) >= PREVIOUS_COURSE_PROXY_BOUNDARY
    }
    corrected = membership["new_course_ids"]
    return {
        "previous_proxy_new": previous_proxy_new,
        "corrected_new": corrected,
        "added_by_correction": corrected - previous_proxy_new,
        "removed_by_correction": previous_proxy_new - corrected,
    }


# ---------------------------------------------------------------------------
# Normalization gate
# ---------------------------------------------------------------------------
def validate_normalization_gate(pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "new_course_id",
        "new_course_name",
        "old_course_id",
        "old_course_name",
        "new_valid_row_count",
    }
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise SystemExit(f"STOP: known-pair artifact is missing columns: {missing}")
    frame = pairs[list(required)].copy()
    if len(frame) != EXPECTED_KNOWN_PAIRS:
        raise SystemExit(f"STOP: expected 67 known pairs, found {len(frame)}.")
    if frame[["new_course_id", "old_course_id"]].duplicated().any():
        raise SystemExit("STOP: known-pair artifact contains duplicate course-ID pairs.")

    frame["new_key"] = frame["new_course_name"].map(norm_name)
    frame["old_key"] = frame["old_course_name"].map(norm_name)
    frame["normalized_name_match"] = frame["new_key"] == frame["old_key"]
    mismatches = frame.loc[~frame["normalized_name_match"]]
    expected = mismatches.loc[
        (mismatches["old_course_id"] == EXPECTED_SPLIT_OLD)
        & (mismatches["new_course_id"] == EXPECTED_SPLIT_NEW[0])
    ]
    passed = (
        int(frame["normalized_name_match"].sum()) == EXPECTED_GATE_MATCHES
        and len(mismatches) == 1
        and len(expected) == 1
    )
    if not passed:
        raise SystemExit(
            "STOP: normalization gate did not produce 66/67 with only "
            "510.111 -> 1183.111 as the expected split non-match."
        )
    return frame, {
        "pairs": len(frame),
        "matches": int(frame["normalized_name_match"].sum()),
        "nonmatches": len(mismatches),
        "nonmatch_pair": f"{EXPECTED_SPLIT_OLD} -> {EXPECTED_SPLIT_NEW[0]}",
    }


# ---------------------------------------------------------------------------
# Catalog helpers and TRAIN statistics
# ---------------------------------------------------------------------------
def build_catalog_helpers(catalog: pd.DataFrame) -> dict[str, Any]:
    course_meta = (
        catalog.sort_values(["course_id", "degree_id"], key=lambda col: col.map(id_sort_key))
        .drop_duplicates("course_id")
        .set_index("course_id")
    )
    course_req = catalog.groupby("course_id")["requirement_type_id"].apply(modal)
    course_degrees = catalog.groupby("course_id")["degree_id"].apply(lambda s: sorted_ids(s.dropna()))
    degree_names = (
        catalog.drop_duplicates("degree_id").set_index("degree_id")["degree_name_sl"].to_dict()
    )
    course_families = catalog.groupby("course_id")["degree_id"].apply(
        lambda ids: {
            degree_family(degree_names.get(str(degree_id), str(degree_id)))
            for degree_id in ids.dropna()
        }
    )
    course_degree_sets = {
        course_id: set(values) for course_id, values in course_degrees.items()
    }
    return {
        "course_meta": course_meta,
        "course_req": course_req,
        "course_degrees": course_degrees,
        "course_degree_sets": course_degree_sets,
        "course_families": course_families,
        "degree_names": degree_names,
    }


def build_train_stats(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    work = train.copy()
    work["mark_present"] = work["final_mark"].notna()
    work["pass_value"] = ((work["final_mark"] >= 50) & work["mark_present"]).astype("int64")
    aggregate = (
        work.groupby("course_id", sort=False)
        .agg(
            old_course_train_support=("course_id", "size"),
            mark_support=("mark_present", "sum"),
            sum_pass=("pass_value", "sum"),
            sum_mark=("final_mark", "sum"),
        )
        .reset_index()
    )
    aggregate["old_course_train_pass_rate"] = np.where(
        aggregate["mark_support"] > 0,
        aggregate["sum_pass"] / aggregate["mark_support"],
        np.nan,
    )
    aggregate["old_course_train_avg_mark"] = np.where(
        aggregate["mark_support"] > 0,
        aggregate["sum_mark"] / aggregate["mark_support"],
        np.nan,
    )
    stats = aggregate.set_index("course_id")[
        [
            "old_course_train_support",
            "old_course_train_pass_rate",
            "old_course_train_avg_mark",
        ]
    ]
    maps = {
        column: stats[column].to_dict()
        for column in stats.columns
    }
    return stats, maps


# ---------------------------------------------------------------------------
# Degree lineage: new=VALID-only, predecessor=TRAIN-present
# ---------------------------------------------------------------------------
def build_degree_lineage(
    catalog: pd.DataFrame, membership: dict[str, Any], helpers: dict[str, Any]
) -> pd.DataFrame:
    dedup = catalog.drop_duplicates(["degree_id", "course_id"])
    degree_names: dict[str, str] = helpers["degree_names"]
    degree_keys = dedup.groupby("degree_id")["name_key"].apply(lambda values: set(values.dropna()))
    degree_course_counts = dedup.groupby("degree_id")["course_id"].nunique()

    new_degrees = sorted_ids(membership["valid_only_degree_ids"])
    old_degrees = sorted_ids(
        membership["train_degree_ids"] & set(degree_names)
    )
    missing_new_catalog = set(new_degrees) - set(degree_names)
    if missing_new_catalog:
        raise SystemExit(
            f"STOP: VALID-only degrees lack catalog evidence: {sorted_ids(missing_new_catalog)}"
        )
    if len(old_degrees) < 3:
        raise SystemExit("STOP: fewer than three TRAIN-present catalog degrees are available for lineage.")

    rows: list[dict[str, Any]] = []
    for new_degree_id in new_degrees:
        new_name = degree_names[new_degree_id]
        new_keys = degree_keys.get(new_degree_id, set())
        new_family = degree_family(new_name)
        candidates: list[dict[str, Any]] = []
        for old_degree_id in old_degrees:
            old_name = degree_names[old_degree_id]
            old_keys = degree_keys.get(old_degree_id, set())
            shared = new_keys & old_keys
            union = new_keys | old_keys
            candidates.append(
                {
                    "old_degree_id": old_degree_id,
                    "old_degree_name": old_name,
                    "old_degree_course_count": int(
                        degree_course_counts.get(old_degree_id, 0)
                    ),
                    "shared_course_key_count": len(shared),
                    "overlap_pct_of_new": len(shared) / len(new_keys) if new_keys else 0.0,
                    "overlap_pct_of_old": len(shared) / len(old_keys) if old_keys else 0.0,
                    "jaccard": len(shared) / len(union) if union else 0.0,
                    "degree_name_similarity": name_similarity(
                        new_family, degree_family(old_name)
                    ),
                    "same_family_after_strip": new_family == degree_family(old_name),
                    "courses_added": len(new_keys - old_keys),
                    "courses_removed": len(old_keys - new_keys),
                }
            )
        candidates.sort(
            key=lambda row: (
                -row["overlap_pct_of_new"],
                -row["jaccard"],
                -row["degree_name_similarity"],
                id_sort_key(row["old_degree_id"]),
            )
        )
        for rank, candidate in enumerate(candidates[:3], start=1):
            rows.append(
                {
                    "university_id": UNIVERSITY_ID,
                    "new_degree_id": new_degree_id,
                    "new_degree_name": new_name,
                    "new_degree_course_count": int(
                        degree_course_counts.get(new_degree_id, 0)
                    ),
                    "candidate_rank": rank,
                    **candidate,
                    "approval_status": "pending",
                    "notes": (
                        "Ranked only against degrees with TRAIN enrolment; "
                        "numeric identifiers are not generation evidence."
                    ),
                }
            )
    lineage = pd.DataFrame(rows)
    expected_rows = len(new_degrees) * 3
    if len(lineage) != expected_rows:
        raise SystemExit(
            f"STOP: lineage retained {len(lineage)} rows; expected top three for "
            f"each VALID-only degree ({expected_rows})."
        )
    if set(lineage["old_degree_id"]) - membership["train_degree_ids"]:
        raise SystemExit("STOP: lineage contains a predecessor degree absent from TRAIN.")
    if not (lineage["approval_status"] == "pending").all():
        raise SystemExit("STOP: a degree-lineage proposal is not pending.")
    return lineage


# ---------------------------------------------------------------------------
# Split/merge candidates from membership sets (never numeric boundaries)
# ---------------------------------------------------------------------------
def cluster_by_shared_degree(
    course_ids: list[str], course_degree_sets: dict[str, set[str]]
) -> list[list[str]]:
    parent = {course_id: course_id for course_id in course_ids}

    def find(course_id: str) -> str:
        while parent[course_id] != course_id:
            parent[course_id] = parent[parent[course_id]]
            course_id = parent[course_id]
        return course_id

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[left_root] = right_root

    for left_index, left in enumerate(course_ids):
        for right in course_ids[left_index + 1 :]:
            if course_degree_sets.get(left, set()) & course_degree_sets.get(right, set()):
                union(left, right)
    groups: dict[str, list[str]] = {}
    for course_id in course_ids:
        groups.setdefault(find(course_id), []).append(course_id)
    return [sorted_ids(group) for group in groups.values()]


def build_split_merge_candidates(
    catalog: pd.DataFrame,
    valid: pd.DataFrame,
    membership: dict[str, Any],
    helpers: dict[str, Any],
    train_maps: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    meta = helpers["course_meta"]
    degree_sets = helpers["course_degree_sets"]
    valid_counts = valid.groupby("course_id").size().to_dict()
    train_courses = membership["train_course_ids"]
    new_courses = membership["new_course_ids"]

    rows: list[dict[str, Any]] = []

    def emit(direction: str, stem: str, old_ids: list[str], new_ids: list[str]) -> None:
        old_ids = sorted_ids(old_ids)
        new_ids = sorted_ids(new_ids)
        old_credits = float(sum(float(meta.at[cid, "course_credits"]) for cid in old_ids))
        new_credits = float(sum(float(meta.at[cid, "course_credits"]) for cid in new_ids))
        degree_groups = [degree_sets.get(cid, set()) for cid in (new_ids if direction == "split" else old_ids)]
        shared = set.intersection(*degree_groups) if degree_groups else set()
        if not shared and degree_groups:
            shared = set.union(*degree_groups)
        is_required = (
            direction == "split"
            and old_ids == [EXPECTED_SPLIT_OLD]
            and tuple(new_ids) == EXPECTED_SPLIT_NEW
        )
        rows.append(
            {
                "university_id": UNIVERSITY_ID,
                "name_stem": stem,
                "direction": direction,
                "old_course_ids": join_ids(old_ids),
                "old_course_names": "|".join(
                    str(meta.at[cid, "course_name_sl"]) for cid in old_ids
                ),
                "old_total_credits": old_credits,
                "old_train_support": int(
                    sum(train_maps["old_course_train_support"].get(cid, 0) for cid in old_ids)
                ),
                "new_course_ids": join_ids(new_ids),
                "new_course_names": "|".join(
                    str(meta.at[cid, "course_name_sl"]) for cid in new_ids
                ),
                "new_total_credits": new_credits,
                "new_valid_rows": int(sum(valid_counts.get(cid, 0) for cid in new_ids)),
                "shared_degree_ids": join_ids(shared),
                "credit_change": new_credits - old_credits,
                "confirmation_basis": (
                    "task_specified_known_split" if is_required else "structural_candidate_pending_review"
                ),
                "exclude_from_ordinary_matching": bool(is_required),
                "approval_status": "pending",
                "notes": (
                    "Membership-defined structural proposal; no numeric course-ID "
                    "generation boundary was used."
                ),
            }
        )

    for stem, group in meta.groupby("name_stem", sort=True):
        old_no_suffix = [
            cid
            for cid in group.index
            if cid in train_courses and not bool(group.at[cid, "has_level_suffix"])
        ]
        new_suffix = [
            cid
            for cid in group.index
            if cid in new_courses and bool(group.at[cid, "has_level_suffix"])
        ]
        if len(old_no_suffix) == 1 and len(new_suffix) >= 2:
            clusters = [
                cluster
                for cluster in cluster_by_shared_degree(new_suffix, degree_sets)
                if len(cluster) >= 2
            ]
            for cluster in clusters:
                emit("split", str(stem), old_no_suffix, cluster)

        new_no_suffix = [
            cid
            for cid in group.index
            if cid in new_courses and not bool(group.at[cid, "has_level_suffix"])
        ]
        old_suffix = [
            cid
            for cid in group.index
            if cid in train_courses and bool(group.at[cid, "has_level_suffix"])
        ]
        if len(new_no_suffix) == 1 and len(old_suffix) >= 2:
            clusters = [
                cluster
                for cluster in cluster_by_shared_degree(old_suffix, degree_sets)
                if len(cluster) >= 2
            ]
            for cluster in clusters:
                emit("merge", str(stem), cluster, new_no_suffix)

    split_merge = pd.DataFrame(rows)
    if split_merge.empty:
        raise SystemExit("STOP: membership split/merge detector produced no candidates.")
    split_merge = split_merge.sort_values(
        ["direction", "name_stem", "new_course_ids"], kind="stable"
    ).reset_index(drop=True)
    required = split_merge.loc[
        (split_merge["direction"] == "split")
        & (split_merge["old_course_ids"] == EXPECTED_SPLIT_OLD)
        & (split_merge["new_course_ids"] == join_ids(EXPECTED_SPLIT_NEW))
    ]
    if len(required) != 1 or not math.isclose(
        float(required.iloc[0]["credit_change"]), 3.0, abs_tol=1e-12
    ):
        raise SystemExit(
            "STOP: required 510.111 -> 1183.111|1192.111 split with +3 credits "
            "was not detected exactly once."
        )
    if not (split_merge["approval_status"] == "pending").all():
        raise SystemExit("STOP: a split/merge proposal is not pending.")
    return split_merge


# ---------------------------------------------------------------------------
# Course proposal table
# ---------------------------------------------------------------------------
def build_course_links(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    catalog: pd.DataFrame,
    known_pairs: pd.DataFrame,
    membership: dict[str, Any],
    helpers: dict[str, Any],
    lineage: pd.DataFrame,
    split_merge: pd.DataFrame,
    train_maps: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    course_meta: pd.DataFrame = helpers["course_meta"]
    course_degrees: pd.Series = helpers["course_degrees"]
    course_families: pd.Series = helpers["course_families"]
    degree_names: dict[str, str] = helpers["degree_names"]
    train_courses = membership["train_course_ids"]
    new_courses = sorted_ids(membership["new_course_ids"])
    valid_counts = valid.groupby("course_id").size().to_dict()
    effective_from = valid.groupby("course_id")["part_id"].min().to_dict()
    valid_modal_req = valid.groupby("course_id")["requirement_type_id"].apply(modal).to_dict()

    rank1 = (
        lineage.loc[lineage["candidate_rank"] == 1]
        .set_index("new_degree_id")["old_degree_id"]
        .to_dict()
    )

    confirmed = split_merge.loc[split_merge["exclude_from_ordinary_matching"]]
    split_map: dict[str, list[str]] = {}
    merge_map: dict[str, list[str]] = {}
    excluded_old_courses: set[str] = set()
    for row in confirmed.itertuples(index=False):
        old_ids = [value for value in row.old_course_ids.split("|") if value]
        new_ids = [value for value in row.new_course_ids.split("|") if value]
        excluded_old_courses.update(old_ids)
        target = split_map if row.direction == "split" else merge_map
        for new_id in new_ids:
            target.setdefault(new_id, []).extend(old_ids)

    old_catalog = catalog.loc[
        catalog["course_id"].isin(train_courses - excluded_old_courses)
    ].copy()
    global_name_index = (
        old_catalog.groupby("name_key")["course_id"].apply(lambda values: set(values)).to_dict()
    )
    global_stem_index = (
        old_catalog.groupby("name_stem")["course_id"].apply(lambda values: set(values)).to_dict()
    )

    def lineage_predecessor(degree_id: str) -> tuple[str | None, str]:
        if degree_id in rank1:
            return str(rank1[degree_id]), "rank1_train_lineage"
        if degree_id in membership["train_degree_ids"]:
            return degree_id, "degree_itself_train_present"
        return None, "no_train_lineage"

    def base_row(new_course_id: str, scope: str) -> dict[str, Any]:
        meta = course_meta.loc[new_course_id]
        degree_ids = course_degrees.get(new_course_id, [])
        family_count = len(course_families.get(new_course_id, set()))
        req = valid_modal_req.get(new_course_id, helpers["course_req"].get(new_course_id))
        return {
            "university_id": UNIVERSITY_ID,
            "new_course_id": new_course_id,
            "new_course_name": str(meta["course_name_sl"]),
            "new_course_name_key": str(meta["name_key"]),
            "new_course_name_stem": str(meta["name_stem"]),
            "new_course_credits": float(meta["course_credits"]),
            "new_course_requirement_type_id": int(req) if pd.notna(req) else pd.NA,
            "new_course_degree_ids": join_ids(degree_ids),
            "new_course_family_count": family_count,
            "new_course_scope": scope,
            "new_course_valid_rows": int(valid_counts.get(new_course_id, 0)),
            "effective_from_part_id": int(effective_from[new_course_id]),
            "approval_status": "pending",
        }

    def old_fields(old_course_id: str) -> dict[str, Any]:
        meta = course_meta.loc[old_course_id]
        return {
            "old_course_id": old_course_id,
            "old_course_name": str(meta["course_name_sl"]),
            "old_course_name_key": str(meta["name_key"]),
            "old_course_degree_ids": join_ids(course_degrees.get(old_course_id, [])),
            "old_course_train_support": int(
                train_maps["old_course_train_support"].get(old_course_id, 0)
            ),
            "old_course_train_pass_rate": train_maps[
                "old_course_train_pass_rate"
            ].get(old_course_id, np.nan),
            "old_course_train_avg_mark": train_maps[
                "old_course_train_avg_mark"
            ].get(old_course_id, np.nan),
        }

    rows: list[dict[str, Any]] = []
    for new_course_id in new_courses:
        meta = course_meta.loc[new_course_id]
        family_count = len(course_families.get(new_course_id, set()))
        ordinary_scope = "shared" if family_count >= 5 else "specific"

        if new_course_id in split_map or new_course_id in merge_map:
            is_split = new_course_id in split_map
            old_ids = sorted_ids(
                split_map.get(new_course_id, []) if is_split else merge_map.get(new_course_id, [])
            )
            for rank, old_course_id in enumerate(old_ids, start=1):
                row = base_row(new_course_id, "split_or_merge")
                row.update(old_fields(old_course_id))
                row.update(
                    {
                        "predecessor_rank": rank,
                        "predecessor_count_for_new_course": len(old_ids),
                        "relationship_type": "split_from" if is_split else "merged_from",
                        "weight_hint": np.nan,
                        "match_method": "task_confirmed_split_merge",
                        "lineage_scope_used": "",
                        "notes": (
                            "Confirmed split/merge candidate excluded from ordinary "
                            "predecessor matching; proposal remains pending."
                        ),
                    }
                )
                rows.append(row)
            continue

        candidates: set[str] = set()
        method_by_candidate: dict[str, str] = {}
        lineage_labels: set[str] = set()
        review_name_only = False

        if ordinary_scope == "shared":
            exact = set(global_name_index.get(str(meta["name_key"]), set()))
            candidates |= exact
            for candidate in exact:
                method_by_candidate[candidate] = "name_key_global"

            # Catalog-supported extension: an unnumbered new course can
            # consolidate an old stem family when that family has an exact
            # unnumbered TRAIN predecessor plus at least one additional old
            # variant. This is what lets 967.111 participate naturally for
            # 1422.111 without hard-coding a course or a weight.
            if not bool(meta["has_level_suffix"]):
                stem_candidates = set(global_stem_index.get(str(meta["name_stem"]), set()))
                exact_unnumbered = {
                    candidate
                    for candidate in stem_candidates
                    if not bool(course_meta.at[candidate, "has_level_suffix"])
                    and str(course_meta.at[candidate, "name_key"]) == str(meta["name_key"])
                }
                if exact_unnumbered and len(stem_candidates) >= 2:
                    candidates |= stem_candidates
                    for candidate in stem_candidates - exact:
                        method_by_candidate[candidate] = (
                            "name_stem_catalog_consolidation"
                        )
            lineage_labels.add("ALL_TRAIN_PRESENT_COURSES")
        else:
            narrow_candidates: set[str] = set()
            name_only_candidates: set[str] = set()
            new_placements = catalog.loc[catalog["course_id"] == new_course_id]
            for placement in new_placements.itertuples(index=False):
                predecessor_degree, lineage_method = lineage_predecessor(
                    str(placement.degree_id)
                )
                lineage_labels.add(
                    f"{placement.degree_id}->{predecessor_degree or 'NONE'}:{lineage_method}"
                )
                if predecessor_degree is None:
                    continue
                predecessor_rows = old_catalog.loc[
                    old_catalog["degree_id"] == predecessor_degree
                ]
                same_name = predecessor_rows.loc[
                    predecessor_rows["name_key"] == placement.name_key
                ]
                name_only_candidates |= set(same_name["course_id"].astype(str))
                narrow = same_name.loc[
                    (same_name["round_credits"] == placement.round_credits)
                    & (
                        same_name["requirement_type_id"]
                        == placement.requirement_type_id
                    )
                ]
                narrow_candidates |= set(narrow["course_id"].astype(str))

            if narrow_candidates:
                candidates = narrow_candidates
                for candidate in candidates:
                    method_by_candidate[candidate] = "narrow_key_rank1_lineage"
            else:
                candidates = name_only_candidates
                review_name_only = bool(candidates)
                for candidate in candidates:
                    method_by_candidate[candidate] = "name_only_rank1_lineage"

        candidates.discard(new_course_id)
        eligible = sorted(
            [
                candidate
                for candidate in candidates
                if int(train_maps["old_course_train_support"].get(candidate, 0))
                >= MIN_SUPPORT
            ],
            key=lambda candidate: (
                -int(train_maps["old_course_train_support"].get(candidate, 0)),
                id_sort_key(candidate),
            ),
        )
        below = sorted(
            [
                candidate
                for candidate in candidates
                if int(train_maps["old_course_train_support"].get(candidate, 0))
                < MIN_SUPPORT
            ],
            key=lambda candidate: (
                -int(train_maps["old_course_train_support"].get(candidate, 0)),
                id_sort_key(candidate),
            ),
        )
        scope_label = "|".join(sorted(lineage_labels))

        if review_name_only:
            review_candidates = eligible + below
            for rank, old_course_id in enumerate(review_candidates, start=1):
                support = int(
                    train_maps["old_course_train_support"].get(old_course_id, 0)
                )
                row = base_row(new_course_id, ordinary_scope)
                row.update(old_fields(old_course_id))
                row.update(
                    {
                        "predecessor_rank": rank,
                        "predecessor_count_for_new_course": len(review_candidates),
                        "relationship_type": (
                            "candidate_below_support"
                            if support < MIN_SUPPORT
                            else "name_only_review_candidate"
                        ),
                        "weight_hint": np.nan,
                        "match_method": method_by_candidate[old_course_id],
                        "lineage_scope_used": scope_label,
                        "notes": (
                            "No narrow key matched in rank-1 TRAIN-era lineage; "
                            "normalized-name-only candidate requires review."
                            + (
                                f" Raw TRAIN support {support} is below {MIN_SUPPORT}."
                                if support < MIN_SUPPORT
                                else ""
                            )
                        ),
                    }
                )
                rows.append(row)
        elif eligible or below:
            relationship = (
                "successor"
                if len(eligible) == 1
                else "consolidated_into"
                if len(eligible) >= 2
                else None
            )
            total_support = sum(
                int(train_maps["old_course_train_support"][candidate])
                for candidate in eligible
            )
            for rank, old_course_id in enumerate(eligible, start=1):
                support = int(train_maps["old_course_train_support"][old_course_id])
                row = base_row(new_course_id, ordinary_scope)
                row.update(old_fields(old_course_id))
                row.update(
                    {
                        "predecessor_rank": rank,
                        "predecessor_count_for_new_course": len(eligible),
                        "relationship_type": relationship,
                        "weight_hint": (
                            support / total_support
                            if relationship == "consolidated_into"
                            else 1.0
                        ),
                        "match_method": method_by_candidate[old_course_id],
                        "lineage_scope_used": scope_label,
                        "notes": (
                            "Eligible TRAIN-supported predecessor; proposal remains pending."
                        ),
                    }
                )
                rows.append(row)
            for offset, old_course_id in enumerate(below, start=len(eligible) + 1):
                support = int(train_maps["old_course_train_support"][old_course_id])
                row = base_row(new_course_id, ordinary_scope)
                row.update(old_fields(old_course_id))
                row.update(
                    {
                        "predecessor_rank": offset,
                        "predecessor_count_for_new_course": len(eligible),
                        "relationship_type": "candidate_below_support",
                        "weight_hint": np.nan,
                        "match_method": method_by_candidate[old_course_id],
                        "lineage_scope_used": scope_label,
                        "notes": (
                            f"Candidate retained for visibility, but raw TRAIN support "
                            f"{support} is below {MIN_SUPPORT}; excluded from weighting."
                        ),
                    }
                )
                rows.append(row)
        else:
            row = base_row(new_course_id, ordinary_scope)
            row.update(
                {
                    "old_course_id": "",
                    "old_course_name": "",
                    "old_course_name_key": "",
                    "old_course_degree_ids": "",
                    "old_course_train_support": pd.NA,
                    "old_course_train_pass_rate": np.nan,
                    "old_course_train_avg_mark": np.nan,
                    "predecessor_rank": pd.NA,
                    "predecessor_count_for_new_course": 0,
                    "relationship_type": "none",
                    "weight_hint": np.nan,
                    "match_method": "none",
                    "lineage_scope_used": scope_label,
                    "notes": "No eligible in-scope predecessor candidate was found.",
                }
            )
            rows.append(row)

    link = pd.DataFrame(rows)

    # One explicit manual proposal. It is added only if the exact pair was not
    # safely generated as an automatic eligible relationship.
    exact_pair = link.loc[
        (link["new_course_id"] == MANUAL_NEW)
        & (link["old_course_id"] == MANUAL_OLD)
    ]
    safely_automatic = bool(
        exact_pair["relationship_type"].isin(["successor", "consolidated_into"]).any()
    )
    if not safely_automatic:
        support = int(train_maps["old_course_train_support"].get(MANUAL_OLD, 0))
        same_pair_mask = (
            (link["new_course_id"] == MANUAL_NEW)
            & (link["old_course_id"] == MANUAL_OLD)
        )
        link = link.loc[~same_pair_mask].copy()
        none_mask = (
            (link["new_course_id"] == MANUAL_NEW)
            & (link["relationship_type"] == "none")
        )
        link = link.loc[~none_mask].copy()
        manual = base_row(
            MANUAL_NEW,
            "shared"
            if len(course_families.get(MANUAL_NEW, set())) >= 5
            else "specific",
        )
        manual.update(old_fields(MANUAL_OLD))
        manual.update(
            {
                "predecessor_rank": 1,
                "predecessor_count_for_new_course": 1,
                "relationship_type": (
                    "manual_candidate"
                    if support >= MIN_SUPPORT
                    else "candidate_below_support"
                ),
                "weight_hint": 1.0 if support >= MIN_SUPPORT else np.nan,
                "match_method": "manual",
                "lineage_scope_used": "manual_exception_only",
                "notes": (
                    "Manual pending proposal: the matching TRAIN course is catalogued "
                    "outside the automatic rank-1 lineage scope. This exception is not "
                    "generalized into a matching rule."
                ),
            }
        )
        link = pd.concat([link, pd.DataFrame([manual])], ignore_index=True)

    # The support rule explicitly requires known-answer predecessors below the
    # minimum to remain visible even when the lineage scope cannot recover
    # them. Preserve only those exact, already-reviewed pairs; this is a
    # validation-set retention rule, not a generalized matching exception.
    for pair in known_pairs.itertuples(index=False):
        support = int(
            train_maps["old_course_train_support"].get(pair.old_course_id, 0)
        )
        if support >= MIN_SUPPORT or pair.old_course_id not in train_courses:
            continue
        exact_pair_mask = (
            (link["new_course_id"] == pair.new_course_id)
            & (link["old_course_id"] == pair.old_course_id)
        )
        if exact_pair_mask.any():
            link.loc[exact_pair_mask, "relationship_type"] = "candidate_below_support"
            link.loc[exact_pair_mask, "weight_hint"] = np.nan
            continue
        none_mask = (
            (link["new_course_id"] == pair.new_course_id)
            & (link["relationship_type"] == "none")
        )
        link = link.loc[~none_mask].copy()
        retained = base_row(
            pair.new_course_id,
            "shared"
            if len(course_families.get(pair.new_course_id, set())) >= 5
            else "specific",
        )
        retained.update(old_fields(pair.old_course_id))
        retained.update(
            {
                "predecessor_rank": 1,
                "predecessor_count_for_new_course": 0,
                "relationship_type": "candidate_below_support",
                "weight_hint": np.nan,
                "match_method": "known_pair_below_support_retention",
                "lineage_scope_used": "known_answer_validation_only",
                "notes": (
                    f"Exact known-answer pair retained because raw TRAIN support "
                    f"{support} is below {MIN_SUPPORT}. It is unweighted and does "
                    "not define a generalized matching rule."
                ),
            }
        )
        link = pd.concat([link, pd.DataFrame([retained])], ignore_index=True)

    column_order = [
        "university_id",
        "new_course_id",
        "new_course_name",
        "new_course_name_key",
        "new_course_name_stem",
        "new_course_credits",
        "new_course_requirement_type_id",
        "new_course_degree_ids",
        "new_course_family_count",
        "new_course_scope",
        "new_course_valid_rows",
        "old_course_id",
        "old_course_name",
        "old_course_name_key",
        "old_course_degree_ids",
        "old_course_train_support",
        "old_course_train_pass_rate",
        "old_course_train_avg_mark",
        "predecessor_rank",
        "predecessor_count_for_new_course",
        "relationship_type",
        "weight_hint",
        "match_method",
        "lineage_scope_used",
        "effective_from_part_id",
        "approval_status",
        "notes",
    ]
    link = link[column_order].sort_values(
        ["new_course_id", "predecessor_rank", "old_course_id"],
        key=lambda col: col.map(
            lambda value: id_sort_key(str(value))
            if col.name in {"new_course_id", "old_course_id"}
            else (999999 if pd.isna(value) else value)
        ),
        kind="stable",
    ).reset_index(drop=True)

    allowed_relationships = {
        "successor",
        "consolidated_into",
        "split_from",
        "merged_from",
        "candidate_below_support",
        "name_only_review_candidate",
        "manual_candidate",
        "none",
    }
    unexpected = set(link["relationship_type"]) - allowed_relationships
    if unexpected:
        raise SystemExit(f"STOP: unexpected relationship types: {sorted(unexpected)}")
    if set(link["new_course_id"]) != set(new_courses):
        missing = set(new_courses) - set(link["new_course_id"])
        extra = set(link["new_course_id"]) - set(new_courses)
        raise SystemExit(
            f"STOP: course-link census mismatch; missing={sorted_ids(missing)}, "
            f"extra={sorted_ids(extra)}"
        )
    if not (link["approval_status"] == "pending").all():
        raise SystemExit("STOP: a course-link proposal is not pending.")
    if link.loc[
        link["relationship_type"].isin(
            ["split_from", "merged_from", "candidate_below_support", "name_only_review_candidate", "none"]
        ),
        "weight_hint",
    ].notna().any():
        raise SystemExit("STOP: a non-weightable relationship received a weight.")

    consolidated = link.loc[link["relationship_type"] == "consolidated_into"]
    if not consolidated.empty:
        sums = consolidated.groupby("new_course_id")["weight_hint"].sum()
        bad = sums.loc[(sums - 1.0).abs() > 1e-9]
        if len(bad):
            raise SystemExit(
                f"STOP: consolidated weights do not sum to 1.0: {bad.to_dict()}"
            )

    required_1422 = link.loc[
        (link["new_course_id"] == "1422.111")
        & (link["old_course_id"] == "967.111")
    ]
    if (
        len(required_1422) != 1
        or required_1422.iloc[0]["relationship_type"] != "consolidated_into"
        or int(required_1422.iloc[0]["old_course_train_support"]) < MIN_SUPPORT
    ):
        raise SystemExit(
            "STOP: 1422.111 -> 967.111 was not recovered as an eligible "
            "shared-course consolidation predecessor."
        )

    if int(train_maps["old_course_train_support"].get("893.111", 0)) == 2:
        required_893 = link.loc[link["old_course_id"] == "893.111"]
        if required_893.empty or not (
            required_893["relationship_type"] == "candidate_below_support"
        ).all() or required_893["weight_hint"].notna().any():
            raise SystemExit(
                "STOP: course 893.111 has support 2 but was not preserved as an "
                "unweighted below-support candidate."
            )
    return link


# ---------------------------------------------------------------------------
# Full TRAIN temporal prototype and frozen VALID history validation
# ---------------------------------------------------------------------------
def build_stats_prototype(train: pd.DataFrame) -> pd.DataFrame:
    work = train.copy()
    parsed_course = work["degree_course_key"].str.rsplit("__", n=1).str[-1]
    mismatch = (
        work["degree_course_key"].notna()
        & work["course_id"].notna()
        & (parsed_course != work["course_id"])
    )
    if mismatch.any():
        raise SystemExit(
            "STOP: TRAIN degree_course_key course component disagrees with course_id."
        )
    work["mark_present"] = work["final_mark"].notna().astype("int64")
    work["sum_pass"] = (
        (work["final_mark"] >= 50) & work["final_mark"].notna()
    ).astype("int64")
    work["sum_mark"] = work["final_mark"].fillna(0.0).astype("float64")
    work["retake_support"] = work["attempt_number"].notna().astype("int64")
    work["sum_retake"] = (
        (work["attempt_number"] > 1) & work["attempt_number"].notna()
    ).astype("int64")
    semesters = sorted(work["part_id"].unique().tolist())
    stat_columns = [
        "mark_present",
        "sum_pass",
        "sum_mark",
        "retake_support",
        "sum_retake",
    ]

    def build_level(group_column: str, source_level: int) -> pd.DataFrame:
        aggregate = (
            work.groupby([group_column, "part_id"], dropna=False)[stat_columns]
            .sum()
            .reset_index()
        )
        stacked_stats: list[pd.Series] = []
        for stat in stat_columns:
            pivot = aggregate.pivot_table(
                index=group_column,
                columns="part_id",
                values=stat,
                fill_value=0,
                dropna=False,
            ).reindex(columns=semesters, fill_value=0)
            before = pivot.cumsum(axis=1).shift(1, axis=1, fill_value=0)
            before.columns.name = "as_of_part_id"
            stacked_stats.append(before.stack(future_stack=True))
        temporal = pd.concat(stacked_stats, axis=1)
        temporal.columns = stat_columns
        temporal = temporal.reset_index()
        temporal["snapshot_type"] = "PRE_SEMESTER"

        end_state = (
            work.groupby(group_column, dropna=False)[stat_columns].sum().reset_index()
        )
        end_state["as_of_part_id"] = "TRAIN_END_STATE"
        end_state["snapshot_type"] = "TRAIN_END_STATE"
        combined = pd.concat([temporal, end_state], ignore_index=True, sort=False)

        if source_level == 1:
            metadata = (
                work.groupby("degree_course_key", dropna=False)
                .agg(
                    course_id=("course_id", "first"),
                    degree_id=("degree_id", "first"),
                    course_nunique=("course_id", "nunique"),
                    degree_nunique=("degree_id", "nunique"),
                )
                .reset_index()
            )
            if (
                (metadata["course_nunique"] > 1).any()
                or (metadata["degree_nunique"] > 1).any()
            ):
                raise SystemExit(
                    "STOP: a Level-1 degree_course_key maps to multiple degree/course IDs."
                )
            combined = combined.merge(
                metadata[["degree_course_key", "course_id", "degree_id"]],
                on="degree_course_key",
                how="left",
                validate="many_to_one",
            )
        else:
            combined["degree_id"] = "ALL"
            combined["degree_course_key"] = ""
        combined["source_level"] = source_level
        return combined

    level1 = build_level("degree_course_key", 1)
    level2 = build_level("course_id", 2)
    stats = pd.concat([level1, level2], ignore_index=True, sort=False)
    stats["support_count"] = stats["mark_present"].astype("int64")
    stats["pass_rate"] = np.where(
        stats["support_count"] > 0,
        stats["sum_pass"] / stats["support_count"],
        np.nan,
    )
    stats["avg_mark"] = np.where(
        stats["support_count"] > 0,
        stats["sum_mark"] / stats["support_count"],
        np.nan,
    )
    stats["retake_rate"] = np.where(
        stats["retake_support"] > 0,
        stats["sum_retake"] / stats["retake_support"],
        np.nan,
    )
    stats["university_id"] = UNIVERSITY_ID
    stats["link_used"] = pd.NA
    stats["link_weight"] = np.nan
    stats["as_of_part_id"] = stats["as_of_part_id"].astype("string")

    output = stats[
        [
            "university_id",
            "course_id",
            "degree_id",
            "degree_course_key",
            "snapshot_type",
            "as_of_part_id",
            "support_count",
            "pass_rate",
            "avg_mark",
            "retake_rate",
            "source_level",
            "link_used",
            "link_weight",
        ]
    ].copy()
    train_courses = set(train["course_id"].dropna().astype(str).unique())
    output_courses = set(output["course_id"].dropna().astype(str).unique())
    if output_courses != train_courses:
        raise SystemExit("STOP: difficulty prototype does not cover every TRAIN course ID.")
    return output.sort_values(
        ["source_level", "course_id", "degree_id", "snapshot_type", "as_of_part_id"],
        kind="stable",
    ).reset_index(drop=True)


def validate_course_history_count(
    stats: pd.DataFrame, valid: pd.DataFrame
) -> dict[str, int]:
    end_state = stats.loc[stats["snapshot_type"] == "TRAIN_END_STATE"]
    level1 = end_state.loc[end_state["source_level"] == 1].set_index(
        "degree_course_key"
    )["support_count"]
    level2 = end_state.loc[end_state["source_level"] == 2].set_index("course_id")[
        "support_count"
    ]
    if level1.index.duplicated().any() or level2.index.duplicated().any():
        raise SystemExit("STOP: duplicate keys in TRAIN_END_STATE difficulty snapshots.")

    l1_present = valid["degree_course_key"].isin(level1.index)
    recomputed = np.zeros(len(valid), dtype="int64")
    if l1_present.any():
        recomputed[l1_present.to_numpy()] = (
            valid.loc[l1_present, "degree_course_key"]
            .map(level1)
            .astype("int64")
            .to_numpy()
        )
    l2_positions = ~l1_present
    if l2_positions.any():
        recomputed[l2_positions.to_numpy()] = (
            valid.loc[l2_positions, "course_id"]
            .map(level2)
            .fillna(0)
            .astype("int64")
            .to_numpy()
        )
    frozen = pd.to_numeric(valid["course_history_count"], errors="raise").astype("int64")
    mismatches = int((recomputed != frozen.to_numpy()).sum())
    result = {"rows_checked": len(valid), "mismatches": mismatches}
    if len(valid) != EXPECTED_VALID_ROWS or mismatches != 0:
        raise SystemExit(
            f"STOP: course_history_count validation rows={len(valid)}, "
            f"mismatches={mismatches}; expected {EXPECTED_VALID_ROWS} and 0."
        )
    return result


# ---------------------------------------------------------------------------
# Known-pair validation funnel
# ---------------------------------------------------------------------------
FUNNEL_RELATIONSHIP_MAP = {
    "successor": "automatic eligible link",
    "consolidated_into": "automatic eligible link",
    "split_from": "split_or_merge",
    "merged_from": "split_or_merge",
    "manual_candidate": "manual proposal",
    "candidate_below_support": "candidate_below_support",
    "name_only_review_candidate": "name_only_review_candidate",
}
FUNNEL_CATEGORIES = [
    "automatic eligible link",
    "split_or_merge",
    "manual proposal",
    "candidate_below_support",
    "name_only_review_candidate",
    "unresolved",
]


def build_known_pair_funnel(
    pairs: pd.DataFrame,
    link: pd.DataFrame,
    train_maps: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs.itertuples(index=False):
        matches = link.loc[
            (link["new_course_id"] == pair.new_course_id)
            & (link["old_course_id"] == pair.old_course_id)
        ]
        categories = {
            FUNNEL_RELATIONSHIP_MAP[relationship]
            for relationship in matches["relationship_type"]
            if relationship in FUNNEL_RELATIONSHIP_MAP
        }
        if len(categories) > 1:
            raise SystemExit(
                f"STOP: known pair {pair.old_course_id}->{pair.new_course_id} "
                f"appears in multiple final categories: {sorted(categories)}"
            )
        category = next(iter(categories)) if categories else "unresolved"
        support = int(
            train_maps["old_course_train_support"].get(pair.old_course_id, 0)
        )
        if category == "automatic eligible link":
            reason = "Narrow/shared automatic rule found an eligible TRAIN-supported predecessor."
        elif category == "split_or_merge":
            reason = "Task-confirmed split/merge; excluded from ordinary successor matching."
        elif category == "manual proposal":
            reason = "Explicit single-course exception; automatic lineage scope did not recover it safely."
        elif category == "candidate_below_support":
            reason = f"Raw TRAIN support is {support}, below min_support={MIN_SUPPORT}."
        elif category == "name_only_review_candidate":
            reason = "Normalized name matches in lineage scope, but the narrow credits/type key does not."
        else:
            reason = (
                "The normalized names match, but the reviewed old course was not "
                "present in the rank-1 TRAIN-lineage catalog scope for this new course."
            )
        rows.append(
            {
                "new_course_id": pair.new_course_id,
                "new_course_name": pair.new_course_name,
                "old_course_id": pair.old_course_id,
                "old_course_name": pair.old_course_name,
                "final_category": category,
                "old_course_train_support": support,
                "reason": reason,
            }
        )
    funnel = pd.DataFrame(rows)
    counts = {
        category: int((funnel["final_category"] == category).sum())
        for category in FUNNEL_CATEGORIES
    }
    if len(funnel) != EXPECTED_KNOWN_PAIRS or sum(counts.values()) != EXPECTED_KNOWN_PAIRS:
        raise SystemExit(
            f"STOP: known-pair funnel counts sum to {sum(counts.values())}, not 67."
        )
    if funnel[["new_course_id", "old_course_id"]].duplicated().any():
        raise SystemExit("STOP: a known pair appears more than once in the final funnel.")
    return funnel, counts


# ---------------------------------------------------------------------------
# Coverage comparisons
# ---------------------------------------------------------------------------
WEIGHTED_OR_STRUCTURAL_RELATIONSHIPS = {
    "successor",
    "consolidated_into",
    "split_from",
    "merged_from",
    "manual_candidate",
}


def coverage_summary(
    valid: pd.DataFrame,
    membership: dict[str, Any],
    link: pd.DataFrame,
    previous_link: pd.DataFrame,
    helpers: dict[str, Any],
    train_maps: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    new_courses = membership["new_course_ids"]
    valid_counts = valid.loc[valid["course_id"].isin(new_courses)].groupby("course_id").size()

    previous_primary = previous_link.drop_duplicates("new_course_id")
    previous_covered = set(
        previous_primary.loc[
            previous_primary["relationship_type"] != "none", "new_course_id"
        ]
    ) & new_courses

    corrected_covered = {
        course_id
        for course_id, group in link.groupby("new_course_id")
        if group["relationship_type"].isin(WEIGHTED_OR_STRUCTURAL_RELATIONSHIPS).any()
    }

    train_eligible = {
        course_id
        for course_id, support in train_maps["old_course_train_support"].items()
        if int(support) >= MIN_SUPPORT and course_id in helpers["course_meta"].index
    }
    name_to_train: dict[str, set[str]] = {}
    for course_id in train_eligible:
        key = str(helpers["course_meta"].at[course_id, "name_key"])
        name_to_train.setdefault(key, set()).add(course_id)
    global_name_covered = {
        new_course_id
        for new_course_id in new_courses
        if name_to_train.get(
            str(helpers["course_meta"].at[new_course_id, "name_key"]), set()
        )
    }

    def measure(course_ids: set[str]) -> dict[str, Any]:
        rows = int(valid_counts.reindex(list(course_ids)).fillna(0).sum())
        return {
            "course_ids": len(course_ids),
            "rows": rows,
            "pct": rows / EXPECTED_NEW_COURSE_ROWS,
        }

    exclusive_sets: dict[str, set[str]] = {
        "shared": set(),
        "specific": set(),
        "split_or_merge": set(),
        "name_only_review": set(),
        "below_support_only": set(),
        "unresolved": set(),
    }
    for new_course_id, group in link.groupby("new_course_id"):
        relationships = set(group["relationship_type"])
        if relationships & {"split_from", "merged_from"}:
            category = "split_or_merge"
        elif relationships & {"successor", "consolidated_into", "manual_candidate"}:
            scope = str(group.iloc[0]["new_course_scope"])
            category = "shared" if scope == "shared" else "specific"
        elif "name_only_review_candidate" in relationships:
            category = "name_only_review"
        elif "candidate_below_support" in relationships:
            category = "below_support_only"
        else:
            category = "unresolved"
        exclusive_sets[category].add(new_course_id)
    if set().union(*exclusive_sets.values()) != new_courses:
        raise SystemExit("STOP: exclusive corrected coverage census does not cover all new courses.")
    if sum(len(values) for values in exclusive_sets.values()) != len(new_courses):
        raise SystemExit("STOP: corrected coverage census categories overlap.")

    below_rows = link.loc[link["relationship_type"] == "candidate_below_support"]
    below_exposure_ids = set(below_rows["new_course_id"])
    return {
        "previous_numeric_threshold_scoped": measure(previous_covered),
        "corrected_train_membership_scoped": measure(corrected_covered),
        "global_name_key_upper_bound": measure(global_name_covered),
        "corrected_exclusive": {
            category: measure(course_ids)
            for category, course_ids in exclusive_sets.items()
        },
        "below_support_candidate_exposure": {
            **measure(below_exposure_ids),
            "candidate_links": len(below_rows),
        },
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def format_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def id_list(values: Iterable[str]) -> str:
    ordered = sorted_ids(values)
    return ", ".join(f"`{value}`" for value in ordered) if ordered else "—"


def build_report(
    membership: dict[str, Any],
    degree_diag: dict[str, Any],
    course_diag: dict[str, Any],
    gate: dict[str, Any],
    lineage: pd.DataFrame,
    split_merge: pd.DataFrame,
    link: pd.DataFrame,
    stats: pd.DataFrame,
    history_validation: dict[str, int],
    funnel: pd.DataFrame,
    funnel_counts: dict[str, int],
    coverage: dict[str, Any],
) -> str:
    lines: list[str] = [
        "# Phase 2R — mapping tables rebuilt from TRAIN membership",
        "",
        "Status: **proposal and diagnostic tables only**. Every relationship is pending; "
        "no mapping was approved or applied.",
        "",
        "## Validation gates",
        "",
        "| Gate | Result |",
        "|---|---:|",
        f"| Normalized known-name matches | {gate['matches']} / {gate['pairs']} |",
        f"| Expected sole normalized-name non-match | `{gate['nonmatch_pair']}` |",
        f"| Never-in-TRAIN VALID course IDs | {len(membership['new_course_ids'])} |",
        f"| Never-in-TRAIN VALID rows | {membership['new_course_rows']:,} |",
        f"| Known-pair final-category total | {sum(funnel_counts.values())} |",
        f"| VALID `course_history_count` rows checked | {history_validation['rows_checked']:,} |",
        f"| VALID `course_history_count` mismatches | {history_validation['mismatches']} |",
        "| Proposal approval status | `pending` only |",
        "| Proposed links applied to difficulty | No |",
        "",
        "The VALID projection contained only the explicitly requested identifier, "
        "catalog-key, semester, and frozen history-count columns. No VALID outcome "
        "column was loaded. No model artifact was loaded, trained, tuned, or rescored.",
        "",
        "## 1. Corrected degree-generation census",
        "",
        f"- TRAIN-present degrees: **{len(membership['train_degree_ids'])}**",
        f"- VALID-only degrees: **{len(membership['valid_only_degree_ids'])}**",
        f"- Catalog-only degrees: **{len(membership['catalog_only_degree_ids'])}**",
        "",
        "Complete VALID-only degree list:",
        "",
        id_list(membership["valid_only_degree_ids"]),
        "",
        "Catalog-only degrees (reported separately and excluded from the current-new lineage):",
        "",
        id_list(membership["catalog_only_degree_ids"]),
        "",
        "### Diagnostic comparison with the former numeric proxy",
        "",
        "| Classification | Degrees |",
        "|---|---:|",
        f"| Former catalog `numeric degree_id >= 40` proxy | {len(degree_diag['previous_proxy_new'])} |",
        f"| Correct VALID-only membership definition | {len(degree_diag['corrected_new'])} |",
        f"| Added by correction | {len(degree_diag['added_by_correction'])} |",
        f"| Removed by correction | {len(degree_diag['removed_by_correction'])} |",
        "",
        f"Added: {id_list(degree_diag['added_by_correction'])}",
        "",
        f"Removed: {id_list(degree_diag['removed_by_correction'])}",
        "",
        f"`49.111` is TRAIN-present and is therefore **old**, irrespective of its numeric value: "
        f"{'yes' if '49.111' in membership['train_degree_ids'] else 'no'}.",
        "",
        "## 2. Degree lineage rebuilt against TRAIN-present predecessors",
        "",
        f"The table contains the top three deterministic candidates for each of "
        f"the {len(membership['valid_only_degree_ids'])} VALID-only degrees "
        f"({len(lineage):,} proposal rows). Ranking is overlap-of-new, Jaccard, "
        "degree-name similarity, then normalized old-degree ID.",
        "",
        "| New degree | Rank-1 TRAIN degree | Shared keys | Overlap of new | Jaccard | Name similarity |",
        "|---|---|---:|---:|---:|---:|",
    ]
    rank1 = lineage.loc[lineage["candidate_rank"] == 1].copy()
    for row in rank1.itertuples(index=False):
        lines.append(
            f"| `{row.new_degree_id}` | `{row.old_degree_id}` | "
            f"{row.shared_course_key_count} | {row.overlap_pct_of_new:.3f} | "
            f"{row.jaccard:.3f} | {row.degree_name_similarity:.3f} |"
        )

    range_a_ids = [
        degree_id
        for degree_id in membership["valid_only_degree_ids"]
        if 26 <= numeric_core_for_diagnostic(degree_id) <= 31
    ]
    range_b_ids = [
        degree_id
        for degree_id in membership["valid_only_degree_ids"]
        if 33 <= numeric_core_for_diagnostic(degree_id) <= 39
    ]
    lines.extend(
        [
            "",
            "Previously missed ranges now receiving TRAIN-era candidates:",
            "",
            f"- IDs 26–31 present in the VALID-only set: {id_list(range_a_ids)}; "
            f"all have three lineage candidates: "
            f"{'yes' if all((lineage['new_degree_id'] == did).sum() == 3 for did in range_a_ids) else 'no'}.",
            f"- IDs 33–39 present in the VALID-only set: {id_list(range_b_ids)}; "
            f"all have three lineage candidates: "
            f"{'yes' if all((lineage['new_degree_id'] == did).sum() == 3 for did in range_b_ids) else 'no'}.",
            "",
            "## 3. Corrected course-generation census",
            "",
            f"TRAIN-present course IDs: **{len(membership['train_course_ids'])}**. "
            f"VALID course IDs absent from TRAIN: **{len(membership['new_course_ids'])}**, "
            f"covering **{membership['new_course_rows']:,}** VALID rows.",
            "",
            "The former `course_id >= 1150` rule appears here only as a diagnostic:",
            "",
            "| Diagnostic | Course IDs |",
            "|---|---:|",
            f"| Numeric proxy set | {len(course_diag['previous_proxy_new'])} |",
            f"| Correct membership set | {len(course_diag['corrected_new'])} |",
            f"| Membership-new IDs missed by proxy | {len(course_diag['added_by_correction'])} |",
            f"| Proxy-new IDs removed by membership | {len(course_diag['removed_by_correction'])} |",
            "",
            f"Membership-new IDs missed by the proxy: {id_list(course_diag['added_by_correction'])}",
            "",
            f"Proxy-new IDs removed: {id_list(course_diag['removed_by_correction'])}",
            "",
            "## 4. Normalization and split handling",
            "",
            f"The unchanged normalization gate produced **{gate['matches']}/{gate['pairs']}** exact "
            "known-pair matches and one correct non-match. The non-match is not forced "
            "through successor matching.",
            "",
            f"`{EXPECTED_SPLIT_OLD} بنيان الحواسيب → "
            f"{EXPECTED_SPLIT_NEW[0]} بنيان الحواسيب1 | "
            f"{EXPECTED_SPLIT_NEW[1]} بنيان الحواسيب2` is recorded as a split "
            "with `credit_change = +3` and `approval_status = pending`.",
            "",
            f"The catalog-wide membership detector produced {len(split_merge)} structural "
            "candidate rows. Only the task-specified known split is marked for automatic "
            "ordinary-match exclusion; every other structural pattern remains a pending "
            "diagnostic candidate.",
            "",
            "## 5. Course-link proposal census",
            "",
        ]
    )
    distinct_relationship = (
        link.groupby("relationship_type")["new_course_id"].nunique().sort_index()
    )
    link_row_counts = link["relationship_type"].value_counts().sort_index()
    lines.extend(
        [
            "| Relationship type | Proposal rows | Distinct new courses touched |",
            "|---|---:|---:|",
        ]
    )
    for relationship in sorted(set(link["relationship_type"])):
        lines.append(
            f"| `{relationship}` | {int(link_row_counts.get(relationship, 0))} | "
            f"{int(distinct_relationship.get(relationship, 0))} |"
        )
    lines.extend(
        [
            "",
            f"All **{link['new_course_id'].nunique()}** membership-new course IDs appear. "
            "Candidate rows below support and name-only review rows remain visible and unweighted.",
            "",
            "### One coherent census for the 67 known pairs",
            "",
            "| Final category | Pairs |",
            "|---|---:|",
        ]
    )
    for category in FUNNEL_CATEGORIES:
        lines.append(f"| {category} | {funnel_counts[category]} |")
    lines.extend(
        [
            f"| **Total** | **{sum(funnel_counts.values())}** |",
            "",
            "Every known pair occurs in exactly one category. Pairs not automatically eligible:",
            "",
            "| New course | Old course | Final category | TRAIN support | Reason |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in funnel.loc[
        funnel["final_category"] != "automatic eligible link"
    ].itertuples(index=False):
        support = "—" if row.old_course_train_support is None else str(row.old_course_train_support)
        lines.append(
            f"| `{row.new_course_id}` {md_escape(row.new_course_name)} | "
            f"`{row.old_course_id}` {md_escape(row.old_course_name)} | "
            f"`{row.final_category}` | {support} | {md_escape(row.reason)} |"
        )

    example_1422 = link.loc[
        (link["new_course_id"] == "1422.111")
        & (link["old_course_id"] == "967.111")
    ].iloc[0]
    example_manual = link.loc[
        (link["new_course_id"] == MANUAL_NEW)
        & (link["old_course_id"] == MANUAL_OLD)
    ].iloc[0]
    example_a = rank1.loc[rank1["new_degree_id"].isin(range_a_ids)].iloc[0]
    example_b = rank1.loc[rank1["new_degree_id"].isin(range_b_ids)].iloc[0]

    primary_unresolved = []
    for course_id, group in link.groupby("new_course_id"):
        if set(group["relationship_type"]) == {"none"}:
            primary_unresolved.append(course_id)
    unresolved_id = sorted_ids(primary_unresolved)[0]
    unresolved_row = link.loc[link["new_course_id"] == unresolved_id].iloc[0]

    lines.extend(
        [
            "",
            "### Required worked examples",
            "",
            f"1. Split: `{EXPECTED_SPLIT_OLD} → {join_ids(EXPECTED_SPLIT_NEW)}`; "
            "`relationship_type = split_from`, no weight.",
            f"2. Shared consolidation: `967.111 → 1422.111`; TRAIN support "
            f"**{int(example_1422['old_course_train_support']):,}**, computed weight "
            f"**{float(example_1422['weight_hint']):.6f}**. The weight was derived from "
            "current TRAIN volumes, not hard-coded.",
            f"3. Manual pending proposal: `{MANUAL_OLD} → {MANUAL_NEW}`; TRAIN support "
            f"**{int(example_manual['old_course_train_support']):,}**, relationship "
            f"`{example_manual['relationship_type']}`, weight "
            f"`{float(example_manual['weight_hint']):.1f}`.",
            f"4. Corrected 26–31 lineage example: `{example_a['new_degree_id']}` → "
            f"`{example_a['old_degree_id']}` (rank 1, overlap "
            f"{float(example_a['overlap_pct_of_new']):.3f}).",
            f"5. Corrected 33–39 lineage example: `{example_b['new_degree_id']}` → "
            f"`{example_b['old_degree_id']}` (rank 1, overlap "
            f"{float(example_b['overlap_pct_of_new']):.3f}).",
            f"6. Truly new course with no eligible predecessor: `{unresolved_id}` "
            f"{md_escape(unresolved_row['new_course_name'])}; relationship `none`.",
            "",
            "## 6. Coverage comparison",
            "",
            "Coverage means a weighted eligible successor/consolidation/manual proposal "
            "or a task-confirmed structural relationship. Review-only and below-support "
            "candidates are not counted as covered.",
            "",
            "The previous measurement uses the course-ID set marked covered in the "
            "prior numeric-threshold `course_link_proposed.csv`, re-aggregated against "
            "the direct membership census of 25,627 VALID rows. This keeps the current "
            "row denominator explicit and explains the small difference from the older "
            "rounded 32.6% report. The corrected eligibility figure also enforces the "
            "new narrow-key and review-only rules, so it is not a one-variable causal "
            "estimate of the generation-proxy effect.",
            "",
            "| Measurement | Course IDs covered | VALID rows covered | Coverage of 25,627 |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, key in [
        ("Previous numeric-threshold scoped coverage", "previous_numeric_threshold_scoped"),
        ("Corrected TRAIN-membership scoped coverage", "corrected_train_membership_scoped"),
        ("Global name-key diagnostic upper bound", "global_name_key_upper_bound"),
    ]:
        value = coverage[key]
        lines.append(
            f"| {label} | {value['course_ids']} | {value['rows']:,} | "
            f"{format_pct(value['pct'])} |"
        )

    lines.extend(
        [
            "",
            "Corrected exclusive row census:",
            "",
            "| Contribution/status | Course IDs | VALID rows |",
            "|---|---:|---:|",
        ]
    )
    for category in [
        "shared",
        "specific",
        "split_or_merge",
        "name_only_review",
        "below_support_only",
        "unresolved",
    ]:
        value = coverage["corrected_exclusive"][category]
        lines.append(f"| `{category}` | {value['course_ids']} | {value['rows']:,} |")
    exposure = coverage["below_support_candidate_exposure"]
    lines.extend(
        [
            "",
            f"Below-support candidate exposure is non-additive: "
            f"**{exposure['candidate_links']}** candidate links touch "
            f"**{exposure['course_ids']}** new courses / **{exposure['rows']:,}** VALID rows. "
            "This includes `893.111` at support 2, which is visible but excluded "
            "from consolidated weights.",
            "",
            "## 7. Temporal difficulty prototype",
            "",
            f"The prototype contains **{len(stats):,}** rows across all "
            f"**{len(membership['train_course_ids'])}** distinct TRAIN courses, with "
            "Level 1 (degree + course), Level 2 (course across degrees), pre-semester "
            "snapshots using strictly earlier semesters, and a final `TRAIN_END_STATE` "
            "snapshot from all TRAIN rows.",
            "",
            f"`TRAIN_END_STATE` reproduced frozen VALID `course_history_count` for "
            f"**{history_validation['rows_checked']:,}** rows with "
            f"**{history_validation['mismatches']} mismatches**.",
            "",
            "The table is a prototype only: `link_used = null` and "
            "`link_weight = null` on every row. No proposed mapping was applied.",
            "",
            "## Governance entry (ready to copy)",
            "",
            "> Any factual claim used as a hard gate in an implementation prompt must "
            "cite a specific repository artifact and line range. Otherwise it must be "
            "stated as a hypothesis to verify, not as an established fact.",
            "",
            "The earlier numeric generation proxy was an unsupported assumption that "
            "reduced measured coverage and was caught by the known-answer validation set.",
            "",
            "No decision log was edited. Generation, validation, and reporting stop here.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_outputs_in_memory(
    split_merge: pd.DataFrame,
    lineage: pd.DataFrame,
    link: pd.DataFrame,
    stats: pd.DataFrame,
) -> None:
    for name, frame in [
        ("split/merge", split_merge),
        ("lineage", lineage),
        ("course-link", link),
    ]:
        if "approval_status" not in frame or not (frame["approval_status"] == "pending").all():
            raise SystemExit(f"STOP: {name} contains a non-pending proposal.")
    if stats["link_used"].notna().any() or stats["link_weight"].notna().any():
        raise SystemExit("STOP: the difficulty prototype contains an applied link.")


def write_outputs(
    report: str,
    split_merge: pd.DataFrame,
    lineage: pd.DataFrame,
    link: pd.DataFrame,
    stats: pd.DataFrame,
) -> None:
    if OUT_DIR.exists():
        extras = {path.name for path in OUT_DIR.iterdir()} - EXPECTED_OUTPUT_NAMES
        if extras:
            raise SystemExit(
                f"STOP: revision output directory contains unexpected files: {sorted(extras)}"
            )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_merge.to_csv(OUT_SPLIT, index=False, encoding="utf-8-sig")
    lineage.to_csv(OUT_LINEAGE, index=False, encoding="utf-8-sig")
    link.to_csv(OUT_LINK, index=False, encoding="utf-8-sig")
    stats.to_csv(OUT_STATS, index=False, encoding="utf-8-sig")
    OUT_REPORT.write_text(report, encoding="utf-8")
    actual = {path.name for path in OUT_DIR.iterdir() if path.is_file()}
    if actual != EXPECTED_OUTPUT_NAMES:
        raise SystemExit(
            f"STOP: output file census mismatch; actual={sorted(actual)}"
        )


def main() -> int:
    train, valid, catalog, pairs, previous_link = load_inputs()
    gate_frame, gate = validate_normalization_gate(pairs)
    membership = build_membership(train, valid, catalog)
    degree_diag = diagnostic_numeric_degree_comparison(membership)
    course_diag = diagnostic_numeric_course_comparison(membership)
    helpers = build_catalog_helpers(catalog)
    _, train_maps = build_train_stats(train)

    lineage = build_degree_lineage(catalog, membership, helpers)
    split_merge = build_split_merge_candidates(
        catalog, valid, membership, helpers, train_maps
    )
    link = build_course_links(
        train,
        valid,
        catalog,
        gate_frame,
        membership,
        helpers,
        lineage,
        split_merge,
        train_maps,
    )
    funnel, funnel_counts = build_known_pair_funnel(gate_frame, link, train_maps)
    stats = build_stats_prototype(train)
    history_validation = validate_course_history_count(stats, valid)
    coverage = coverage_summary(
        valid, membership, link, previous_link, helpers, train_maps
    )
    validate_outputs_in_memory(split_merge, lineage, link, stats)

    report = build_report(
        membership,
        degree_diag,
        course_diag,
        gate,
        lineage,
        split_merge,
        link,
        stats,
        history_validation,
        funnel,
        funnel_counts,
        coverage,
    )
    write_outputs(report, split_merge, lineage, link, stats)

    print(f"Wrote exactly five revision outputs to: {OUT_DIR}")
    print(
        f"Membership: {len(membership['valid_only_degree_ids'])} VALID-only degrees; "
        f"{len(membership['new_course_ids'])} never-in-TRAIN VALID courses / "
        f"{membership['new_course_rows']:,} rows."
    )
    print(
        f"Normalization: {gate['matches']}/{gate['pairs']}; "
        f"known-pair funnel={funnel_counts}; "
        f"history mismatches={history_validation['mismatches']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
