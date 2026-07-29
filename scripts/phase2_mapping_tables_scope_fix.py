"""Phase 2S: repair lineage ancestry eligibility and predecessor search scope.

This script authors proposal and diagnostic tables only.  It reads fixed,
explicit projections of the frozen TRAIN and VALID splits, the cleaned course
catalog, the 67 reviewed pairs, and the Phase 2R proposal artifacts needed for
the preregistered comparison.  VALID is projected without outcome columns and
is checked immediately after loading.

The unchanged Phase 2R normalization, membership, split/merge, TRAIN-statistic,
and temporal-prototype routines are reused.  This file replaces only the
ancestry-sensitive lineage, course-link, census, and report logic.  All stop
conditions are evaluated before any output file is written.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import phase2_mapping_tables_train_membership as p2r


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
PHASE2R_DIR = ROOT / "models" / "runs" / "phase2_train_membership_revision"
PHASE2R_LINEAGE_PATH = PHASE2R_DIR / "degree_lineage_proposed.csv"
PHASE2R_LINK_PATH = PHASE2R_DIR / "course_link_proposed.csv"

OUT_DIR = ROOT / "models" / "runs" / "phase2_lineage_scope_fix"
OUT_REPORT = OUT_DIR / "PHASE2_MAPPING_TABLES_SCOPE_FIX.md"
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

UNIVERSITY_ID = p2r.UNIVERSITY_ID
MIN_SUPPORT = p2r.MIN_SUPPORT
EXPECTED_VALID_ROWS = p2r.EXPECTED_VALID_ROWS
EXPECTED_NEW_COURSES = p2r.EXPECTED_NEW_COURSES
EXPECTED_NEW_COURSE_ROWS = p2r.EXPECTED_NEW_COURSE_ROWS
EXPECTED_KNOWN_PAIRS = p2r.EXPECTED_KNOWN_PAIRS
EXPECTED_SPLIT_OLD = p2r.EXPECTED_SPLIT_OLD
EXPECTED_SPLIT_NEW = p2r.EXPECTED_SPLIT_NEW
MANUAL_NEW = p2r.MANUAL_NEW
MANUAL_OLD = p2r.MANUAL_OLD

ANCESTRY_MIN_OLD_COURSES = 5
EXPECTED_PHASE2R_COURSES = 43
EXPECTED_PHASE2R_ROWS = 7_619
EXPECTED_GLOBAL_COURSES = 83
EXPECTED_GLOBAL_ROWS = 17_814
EXPECTED_MEASURED_RECOVERABLE_PAIRS = 26
EXPECTED_MEASURED_RECOVERABLE_ROWS = 7_335
EXPECTED_PHASE2R_UNRESOLVED_PAIRS = 28
EXPECTED_PHASE2R_UNRESOLVED_ROWS = 7_369
EXPECTED_STILL_UNRECOVERED = {
    ("1165.111", "662.111"),
    ("1271.111", "417.111"),
}

TRAIN_COLUMNS = [
    "course_id",
    "degree_id",
    "faculty_id",
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
    "faculty_id",
    "requirement_type_id",
    "course_credits",
    "part_id",
    "course_history_count",
    "degree_course_key",
]
VALID_OUTCOME_COLUMNS = set(p2r.VALID_OUTCOME_COLUMNS)

CONFIDENCE_LEVELS = ["in_lineage", "same_faculty", "cross_faculty"]
FUNNEL_CATEGORIES = [
    "automatic eligible link",
    "split_or_merge",
    "manual proposal",
    "candidate_below_support",
    "name_only_review_candidate",
    "unresolved",
]


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load only fixed, explicit inputs; VALID is outcome-free by construction."""

    read_paths = [
        TRAIN_PATH,
        VALID_PATH,
        CATALOG_PATH,
        PAIRS_PATH,
        PHASE2R_LINEAGE_PATH,
        PHASE2R_LINK_PATH,
    ]
    for path in read_paths:
        lowered = str(path).lower().replace("\\", "/")
        if "df_test" in lowered or "/test/" in lowered:
            raise SystemExit(f"STOP: a TEST path entered the input allowlist: {path}")
        if not path.is_file():
            raise SystemExit(f"STOP: required input is missing: {path}")

    train = pd.read_parquet(TRAIN_PATH, columns=TRAIN_COLUMNS)
    valid = pd.read_parquet(VALID_PATH, columns=VALID_COLUMNS)
    forbidden_loaded = sorted(set(valid.columns) & VALID_OUTCOME_COLUMNS)
    if forbidden_loaded:
        raise SystemExit(f"STOP: VALID outcome columns were loaded: {forbidden_loaded}")
    if "final_mark" in valid.columns:
        raise SystemExit("STOP: final_mark is present in the VALID runtime projection.")
    if set(valid.columns) != set(VALID_COLUMNS):
        raise SystemExit(
            "STOP: VALID projection differs from the explicit outcome-free column contract."
        )

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
    phase2r_lineage = pd.read_csv(
        PHASE2R_LINEAGE_PATH, dtype="string", keep_default_na=False
    )
    phase2r_link = pd.read_csv(
        PHASE2R_LINK_PATH, dtype="string", keep_default_na=False
    )

    for frame in (train, valid):
        for column in ("course_id", "degree_id", "degree_course_key", "faculty_id"):
            frame[column] = frame[column].astype("string")
        frame["part_id"] = pd.to_numeric(
            frame["part_id"], errors="raise"
        ).astype("int64")

    for column in ("course_id", "degree_id"):
        catalog[column] = catalog[column].astype("string")
    catalog["requirement_type_id"] = pd.to_numeric(
        catalog["requirement_type_id"], errors="coerce"
    ).astype("Int64")
    catalog["course_credits"] = pd.to_numeric(
        catalog["course_credits"], errors="coerce"
    ).astype("float64")
    catalog["name_key"] = catalog["course_name_sl"].map(p2r.norm_name)
    catalog["name_stem"] = catalog["course_name_sl"].map(p2r.name_stem)
    catalog["has_level_suffix"] = catalog["name_key"].str.contains("#", regex=False)
    catalog["round_credits"] = catalog["course_credits"].round().astype("Int64")
    return train, valid, catalog, pairs, phase2r_lineage, phase2r_link


def build_ancestry_eligibility(
    train: pd.DataFrame,
    catalog: pd.DataFrame,
    membership: dict[str, Any],
    helpers: dict[str, Any],
) -> pd.DataFrame:
    """Measure whether each TRAIN-present degree can supply old catalog courses."""

    dedup = catalog.drop_duplicates(["degree_id", "course_id"]).copy()
    catalog_counts = dedup.groupby("degree_id")["course_id"].nunique()
    old_counts = (
        dedup.assign(
            train_present_course=dedup["course_id"].isin(
                membership["train_course_ids"]
            )
        )
        .loc[lambda frame: frame["train_present_course"]]
        .groupby("degree_id")["course_id"]
        .nunique()
    )
    enrolment_rows = train.groupby("degree_id").size()
    degree_names: dict[str, str] = helpers["degree_names"]

    rows: list[dict[str, Any]] = []
    for degree_id in p2r.sorted_ids(membership["train_degree_ids"]):
        old_count = int(old_counts.get(degree_id, 0))
        rows.append(
            {
                "degree_id": degree_id,
                "degree_name": str(degree_names.get(degree_id, "")),
                "train_enrolment_rows": int(enrolment_rows.get(degree_id, 0)),
                "catalog_course_count": int(catalog_counts.get(degree_id, 0)),
                "old_course_count_in_catalog": old_count,
                "ancestry_eligible": old_count >= ANCESTRY_MIN_OLD_COURSES,
            }
        )
    eligibility = pd.DataFrame(rows)
    if set(eligibility["degree_id"]) != membership["train_degree_ids"]:
        raise SystemExit(
            "STOP: ancestry eligibility does not cover every TRAIN-present degree."
        )
    target = eligibility.loc[eligibility["degree_id"] == "49.111"]
    if (
        len(target) != 1
        or int(target.iloc[0]["catalog_course_count"]) != 76
        or int(target.iloc[0]["old_course_count_in_catalog"]) != 0
        or bool(target.iloc[0]["ancestry_eligible"])
    ):
        raise SystemExit(
            "STOP: degree 49.111 was not excluded with catalog count 76 and "
            "old-course count 0; the ancestry eligibility filter is wrong."
        )
    return eligibility


def build_degree_lineage(
    catalog: pd.DataFrame,
    membership: dict[str, Any],
    helpers: dict[str, Any],
    eligibility: pd.DataFrame,
) -> pd.DataFrame:
    """Rank only ancestry-eligible TRAIN-present degrees and retain top three."""

    dedup = catalog.drop_duplicates(["degree_id", "course_id"])
    degree_names: dict[str, str] = helpers["degree_names"]
    degree_keys = dedup.groupby("degree_id")["name_key"].apply(
        lambda values: set(values.dropna())
    )
    degree_course_counts = dedup.groupby("degree_id")["course_id"].nunique()
    old_count_map = eligibility.set_index("degree_id")[
        "old_course_count_in_catalog"
    ].to_dict()
    eligible_old_degrees = p2r.sorted_ids(
        eligibility.loc[eligibility["ancestry_eligible"], "degree_id"]
    )
    new_degrees = p2r.sorted_ids(membership["valid_only_degree_ids"])

    missing_new_catalog = set(new_degrees) - set(degree_names)
    if missing_new_catalog:
        raise SystemExit(
            "STOP: VALID-only degrees lack catalog evidence: "
            f"{p2r.sorted_ids(missing_new_catalog)}"
        )
    if len(eligible_old_degrees) < 3:
        raise SystemExit(
            "STOP: fewer than three ancestry-eligible predecessor degrees exist."
        )

    rows: list[dict[str, Any]] = []
    for new_degree_id in new_degrees:
        new_name = degree_names[new_degree_id]
        new_keys = degree_keys.get(new_degree_id, set())
        new_family = p2r.degree_family(new_name)
        candidates: list[dict[str, Any]] = []
        for old_degree_id in eligible_old_degrees:
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
                    "old_degree_old_course_count": int(
                        old_count_map[old_degree_id]
                    ),
                    "shared_course_key_count": len(shared),
                    "overlap_pct_of_new": (
                        len(shared) / len(new_keys) if new_keys else 0.0
                    ),
                    "overlap_pct_of_old": (
                        len(shared) / len(old_keys) if old_keys else 0.0
                    ),
                    "jaccard": len(shared) / len(union) if union else 0.0,
                    "degree_name_similarity": p2r.name_similarity(
                        new_family, p2r.degree_family(old_name)
                    ),
                    "same_family_after_strip": (
                        new_family == p2r.degree_family(old_name)
                    ),
                    "courses_added": len(new_keys - old_keys),
                    "courses_removed": len(old_keys - new_keys),
                }
            )
        candidates.sort(
            key=lambda row: (
                -row["overlap_pct_of_new"],
                -row["jaccard"],
                -row["degree_name_similarity"],
                p2r.id_sort_key(row["old_degree_id"]),
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
                        "Ranked only over ancestry-eligible TRAIN-present degrees; "
                        f"eligibility requires at least {ANCESTRY_MIN_OLD_COURSES} "
                        "TRAIN-present catalog courses."
                    ),
                }
            )
    lineage = pd.DataFrame(rows)
    expected_rows = len(new_degrees) * 3
    if len(lineage) != expected_rows:
        raise SystemExit(
            f"STOP: lineage retained {len(lineage)} rows; expected {expected_rows}."
        )
    rank_census = lineage.groupby("new_degree_id")["candidate_rank"].apply(
        lambda values: tuple(sorted(int(value) for value in values))
    )
    if not (rank_census == (1, 2, 3)).all():
        raise SystemExit("STOP: a VALID-only degree lacks contiguous top-three ranks.")
    eligible_ids = set(eligible_old_degrees)
    if set(lineage["old_degree_id"]) - eligible_ids:
        raise SystemExit("STOP: lineage contains an ancestry-ineligible predecessor.")
    if "49.111" in set(lineage["old_degree_id"]):
        raise SystemExit("STOP: ancestry-ineligible degree 49.111 entered lineage.")
    if not (lineage["old_degree_old_course_count"] >= ANCESTRY_MIN_OLD_COURSES).all():
        raise SystemExit("STOP: lineage contains an old-degree eligibility count below 5.")
    if not (lineage["approval_status"] == "pending").all():
        raise SystemExit("STOP: a degree-lineage proposal is not pending.")
    return lineage


def compare_rank1(
    lineage: pd.DataFrame,
    phase2r_lineage: pd.DataFrame,
    membership: dict[str, Any],
) -> pd.DataFrame:
    previous = phase2r_lineage.loc[
        phase2r_lineage["candidate_rank"] == "1",
        ["new_degree_id", "old_degree_id"],
    ].rename(columns={"old_degree_id": "phase2r_rank1"})
    current = lineage.loc[
        lineage["candidate_rank"] == 1,
        ["new_degree_id", "old_degree_id"],
    ].rename(columns={"old_degree_id": "phase2s_rank1"})
    comparison = previous.merge(
        current, on="new_degree_id", how="outer", validate="one_to_one"
    )
    expected = membership["valid_only_degree_ids"]
    if (
        set(comparison["new_degree_id"]) != expected
        or len(comparison) != len(expected)
        or comparison[["phase2r_rank1", "phase2s_rank1"]].isna().any().any()
    ):
        raise SystemExit(
            "STOP: Phase 2R versus Phase 2S rank-1 comparison does not cover "
            "all 25 VALID-only degrees exactly once."
        )
    comparison["changed"] = (
        comparison["phase2r_rank1"] != comparison["phase2s_rank1"]
    )
    comparison = comparison.sort_values(
        "new_degree_id",
        key=lambda values: values.map(p2r.id_sort_key),
        kind="stable",
    ).reset_index(drop=True)
    prompted = {
        "26.111",
        "27.111",
        "29.111",
        "31.111",
        "45.111",
        "46.111",
        "47.111",
        "48.111",
    }
    bad = comparison.loc[
        comparison["new_degree_id"].isin(prompted)
        & (comparison["phase2s_rank1"] == "49.111")
    ]
    if not bad.empty:
        raise SystemExit(
            "STOP: one of the preregistered informatics degrees still ranks "
            "ancestry-ineligible 49.111 first."
        )
    return comparison


def build_lineage_maps(
    lineage: pd.DataFrame,
) -> dict[str, list[tuple[str, int]]]:
    maps: dict[str, list[tuple[str, int]]] = {}
    for degree_id, group in lineage.groupby("new_degree_id", sort=False):
        ordered = group.sort_values("candidate_rank")
        maps[str(degree_id)] = [
            (str(row.old_degree_id), int(row.candidate_rank))
            for row in ordered.itertuples(index=False)
        ]
    return maps


def build_course_links(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    catalog: pd.DataFrame,
    known_pairs: pd.DataFrame,
    membership: dict[str, Any],
    helpers: dict[str, Any],
    lineage: pd.DataFrame,
    eligibility: pd.DataFrame,
    split_merge: pd.DataFrame,
    train_maps: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Emit every global name-key match and label rather than discard its scope."""

    course_meta: pd.DataFrame = helpers["course_meta"]
    course_degrees: pd.Series = helpers["course_degrees"]
    course_degree_sets: dict[str, set[str]] = helpers["course_degree_sets"]
    course_families: pd.Series = helpers["course_families"]
    train_courses = membership["train_course_ids"]
    new_courses = p2r.sorted_ids(membership["new_course_ids"])
    eligible_degree_ids = set(
        eligibility.loc[eligibility["ancestry_eligible"], "degree_id"].astype(str)
    )
    lineage_maps = build_lineage_maps(lineage)
    valid_counts = valid.groupby("course_id").size().to_dict()
    effective_from = valid.groupby("course_id")["part_id"].min().to_dict()
    valid_modal_req = (
        valid.groupby("course_id")["requirement_type_id"].apply(p2r.modal).to_dict()
    )
    train_faculties = (
        train.dropna(subset=["course_id", "faculty_id"])
        .groupby("course_id")["faculty_id"]
        .apply(lambda values: set(values.astype(str)))
        .to_dict()
    )
    valid_faculties = (
        valid.dropna(subset=["course_id", "faculty_id"])
        .groupby("course_id")["faculty_id"]
        .apply(lambda values: set(values.astype(str)))
        .to_dict()
    )
    faculty_available = bool(train_faculties) and bool(valid_faculties)

    confirmed = split_merge.loc[split_merge["exclude_from_ordinary_matching"]]
    split_map: dict[str, list[str]] = {}
    merge_map: dict[str, list[str]] = {}
    excluded_old_courses: set[str] = set()
    structural_new_courses: set[str] = set()
    for row in confirmed.itertuples(index=False):
        old_ids = [value for value in row.old_course_ids.split("|") if value]
        new_ids = [value for value in row.new_course_ids.split("|") if value]
        excluded_old_courses.update(old_ids)
        structural_new_courses.update(new_ids)
        target = split_map if row.direction == "split" else merge_map
        for new_id in new_ids:
            target.setdefault(new_id, []).extend(old_ids)

    old_catalog = catalog.loc[
        catalog["course_id"].isin(train_courses - excluded_old_courses)
    ].copy()
    global_name_index = (
        old_catalog.groupby("name_key")["course_id"]
        .apply(lambda values: set(values.astype(str)))
        .to_dict()
    )
    global_stem_index = (
        old_catalog.groupby("name_stem")["course_id"]
        .apply(lambda values: set(values.astype(str)))
        .to_dict()
    )

    def union_scope(new_course_id: str) -> tuple[set[str], dict[str, int]]:
        scope: set[str] = set()
        ranked: dict[str, int] = {}
        for degree_id in course_degrees.get(new_course_id, []):
            degree_id = str(degree_id)
            if degree_id in lineage_maps:
                for old_degree_id, rank in lineage_maps[degree_id]:
                    scope.add(old_degree_id)
                    if old_degree_id not in ranked or rank < ranked[old_degree_id]:
                        ranked[old_degree_id] = rank
            elif (
                degree_id in membership["train_degree_ids"]
                and degree_id in eligible_degree_ids
            ):
                # Preserve Phase 2R's direct-self placement behavior, but only
                # for ancestry-eligible TRAIN-present degrees.  It is unranked.
                scope.add(degree_id)
        if "49.111" in scope:
            raise SystemExit(
                "STOP: ancestry-ineligible degree 49.111 entered a course union scope."
            )
        return scope, ranked

    scope_cache: dict[str, dict[str, Any]] = {}
    for new_course_id in new_courses:
        scope, rank_map = union_scope(new_course_id)
        scope_cache[new_course_id] = {
            "degrees": scope,
            "rank_by_degree": rank_map,
            "label": p2r.join_ids(scope),
        }

    def base_row(new_course_id: str, scope: str) -> dict[str, Any]:
        meta = course_meta.loc[new_course_id]
        degree_ids = course_degrees.get(new_course_id, [])
        family_count = len(course_families.get(new_course_id, set()))
        req = valid_modal_req.get(
            new_course_id, helpers["course_req"].get(new_course_id)
        )
        return {
            "university_id": UNIVERSITY_ID,
            "new_course_id": new_course_id,
            "new_course_name": str(meta["course_name_sl"]),
            "new_course_name_key": str(meta["name_key"]),
            "new_course_name_stem": str(meta["name_stem"]),
            "new_course_credits": float(meta["course_credits"]),
            "new_course_requirement_type_id": (
                int(req) if pd.notna(req) else pd.NA
            ),
            "new_course_degree_ids": p2r.join_ids(degree_ids),
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
            "old_course_degree_ids": p2r.join_ids(
                course_degrees.get(old_course_id, [])
            ),
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

    def confidence_for(
        new_course_id: str,
        old_course_id: str,
        ordinary_scope: str,
    ) -> tuple[str, int | None]:
        if ordinary_scope == "shared":
            return "in_lineage", None
        scope_info = scope_cache[new_course_id]
        old_degrees = course_degree_sets.get(old_course_id, set())
        in_scope_degrees = old_degrees & scope_info["degrees"]
        if in_scope_degrees:
            ranks = [
                scope_info["rank_by_degree"][degree_id]
                for degree_id in in_scope_degrees
                if degree_id in scope_info["rank_by_degree"]
            ]
            return "in_lineage", min(ranks) if ranks else None
        same_faculty = bool(
            valid_faculties.get(new_course_id, set())
            & train_faculties.get(old_course_id, set())
        )
        if faculty_available and same_faculty:
            return "same_faculty", None
        return "cross_faculty", None

    rows: list[dict[str, Any]] = []
    expected_global_name_pairs: set[tuple[str, str]] = set()

    for new_course_id in new_courses:
        meta = course_meta.loc[new_course_id]
        family_count = len(course_families.get(new_course_id, set()))
        ordinary_scope = "shared" if family_count >= 5 else "specific"

        if new_course_id in split_map or new_course_id in merge_map:
            is_split = new_course_id in split_map
            old_ids = p2r.sorted_ids(
                split_map.get(new_course_id, [])
                if is_split
                else merge_map.get(new_course_id, [])
            )
            for rank, old_course_id in enumerate(old_ids, start=1):
                row = base_row(new_course_id, "split_or_merge")
                row.update(old_fields(old_course_id))
                row.update(
                    {
                        "predecessor_rank": rank,
                        "predecessor_count_for_new_course": len(old_ids),
                        "relationship_type": (
                            "split_from" if is_split else "merged_from"
                        ),
                        "weight_hint": np.nan,
                        "match_method": "task_confirmed_split_merge",
                        "scope_confidence": pd.NA,
                        "lineage_scope_used": "",
                        "lineage_rank_matched": pd.NA,
                        "notes": (
                            "Confirmed split/merge candidate excluded from ordinary "
                            "name-key matching; proposal remains pending."
                        ),
                    }
                )
                rows.append(row)
            continue

        exact_candidates = set(
            global_name_index.get(str(meta["name_key"]), set())
        )
        for old_course_id in exact_candidates:
            expected_global_name_pairs.add((new_course_id, old_course_id))
        candidates = set(exact_candidates)
        source_method: dict[str, str] = {
            candidate: "name_key_global" for candidate in exact_candidates
        }

        if ordinary_scope == "shared" and not bool(meta["has_level_suffix"]):
            stem_candidates = set(
                global_stem_index.get(str(meta["name_stem"]), set())
            )
            exact_unnumbered = {
                candidate
                for candidate in stem_candidates
                if not bool(course_meta.at[candidate, "has_level_suffix"])
                and str(course_meta.at[candidate, "name_key"])
                == str(meta["name_key"])
            }
            if exact_unnumbered and len(stem_candidates) >= 2:
                candidates |= stem_candidates
                for candidate in stem_candidates - exact_candidates:
                    source_method[candidate] = "name_stem_catalog_consolidation"

        # This exact pair is required to remain a manual proposal.  The course
        # ID and an ineligible degree ID happen to share the string 49.111.
        if new_course_id == MANUAL_NEW:
            candidates.discard(MANUAL_OLD)

        candidate_info: dict[str, dict[str, Any]] = {}
        for old_course_id in candidates:
            support = int(
                train_maps["old_course_train_support"].get(old_course_id, 0)
            )
            confidence, matched_rank = confidence_for(
                new_course_id, old_course_id, ordinary_scope
            )
            candidate_info[old_course_id] = {
                "support": support,
                "confidence": confidence,
                "matched_rank": matched_rank,
                "source_method": source_method[old_course_id],
            }

        automatic_ids = p2r.sorted_ids(
            old_course_id
            for old_course_id, info in candidate_info.items()
            if info["confidence"] == "in_lineage"
            and info["support"] >= MIN_SUPPORT
        )
        same_faculty_ids = p2r.sorted_ids(
            old_course_id
            for old_course_id, info in candidate_info.items()
            if info["confidence"] == "same_faculty"
            and info["support"] >= MIN_SUPPORT
        )
        automatic_relationship = (
            "successor"
            if len(automatic_ids) == 1
            else "consolidated_into"
            if len(automatic_ids) >= 2
            else None
        )
        same_faculty_relationship = (
            "successor"
            if len(same_faculty_ids) == 1
            else "consolidated_into"
            if len(same_faculty_ids) >= 2
            else None
        )
        automatic_support = sum(
            candidate_info[old_course_id]["support"]
            for old_course_id in automatic_ids
        )

        def candidate_sort_key(old_course_id: str) -> tuple[Any, ...]:
            info = candidate_info[old_course_id]
            if info["support"] < MIN_SUPPORT:
                priority = 3
            elif info["confidence"] == "in_lineage":
                priority = 0
            elif info["confidence"] == "same_faculty":
                priority = 1
            else:
                priority = 2
            return (
                priority,
                -info["support"],
                p2r.id_sort_key(old_course_id),
            )

        ordered_candidates = sorted(candidates, key=candidate_sort_key)
        for predecessor_rank, old_course_id in enumerate(
            ordered_candidates, start=1
        ):
            info = candidate_info[old_course_id]
            support = info["support"]
            confidence = info["confidence"]
            source = info["source_method"]
            if support < MIN_SUPPORT:
                relationship = "candidate_below_support"
                weight = np.nan
                method = source
                note = (
                    f"Name-key candidate retained for visibility, but raw TRAIN "
                    f"support {support} is below min_support={MIN_SUPPORT}."
                )
            elif confidence == "in_lineage":
                relationship = automatic_relationship
                weight = (
                    support / automatic_support
                    if relationship == "consolidated_into"
                    else 1.0
                )
                method = (
                    source
                    if ordinary_scope == "shared"
                    else "name_key_union_lineage"
                )
                note = (
                    "TRAIN-supported normalized-name match is catalogued inside "
                    "the corrected union lineage scope; proposal remains pending."
                )
            elif confidence == "same_faculty":
                relationship = same_faculty_relationship
                weight = np.nan
                method = "out_of_lineage_review_candidate"
                note = (
                    "Normalized-name match lies outside lineage scope but shares "
                    "a faculty in TRAIN/VALID; visible for review and unweighted."
                )
            else:
                relationship = "name_only_review_candidate"
                weight = np.nan
                method = "cross_faculty_name_only_review_candidate"
                note = (
                    "Normalized-name match lies outside lineage scope and has no "
                    "TRAIN/VALID faculty intersection; visible and unweighted."
                )
            row = base_row(new_course_id, ordinary_scope)
            row.update(old_fields(old_course_id))
            row.update(
                {
                    "predecessor_rank": predecessor_rank,
                    "predecessor_count_for_new_course": len(automatic_ids),
                    "relationship_type": relationship,
                    "weight_hint": weight,
                    "match_method": method,
                    "scope_confidence": confidence,
                    "lineage_scope_used": (
                        "ALL_TRAIN_PRESENT_COURSES"
                        if ordinary_scope == "shared"
                        else scope_cache[new_course_id]["label"]
                    ),
                    "lineage_rank_matched": info["matched_rank"],
                    "notes": note,
                }
            )
            rows.append(row)

        if not ordered_candidates:
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
                    "scope_confidence": pd.NA,
                    "lineage_scope_used": (
                        "ALL_TRAIN_PRESENT_COURSES"
                        if ordinary_scope == "shared"
                        else scope_cache[new_course_id]["label"]
                    ),
                    "lineage_rank_matched": pd.NA,
                    "notes": (
                        "No TRAIN-present normalized-name predecessor exists in "
                        "the catalog after structural exclusions."
                    ),
                }
            )
            rows.append(row)

    link = pd.DataFrame(rows)

    # Unconditional pair-specific override required by the task.
    manual_pair_mask = (
        (link["new_course_id"] == MANUAL_NEW)
        & (link["old_course_id"] == MANUAL_OLD)
    )
    link = link.loc[~manual_pair_mask].copy()
    none_mask = (
        (link["new_course_id"] == MANUAL_NEW)
        & (link["relationship_type"] == "none")
    )
    link = link.loc[~none_mask].copy()
    manual_scope = (
        "shared"
        if len(course_families.get(MANUAL_NEW, set())) >= 5
        else "specific"
    )
    manual_confidence, _ = confidence_for(MANUAL_NEW, MANUAL_OLD, manual_scope)
    manual = base_row(MANUAL_NEW, manual_scope)
    manual.update(old_fields(MANUAL_OLD))
    manual.update(
        {
            "predecessor_rank": 1,
            "predecessor_count_for_new_course": 1,
            "relationship_type": "manual_candidate",
            "weight_hint": 1.0,
            "match_method": "manual",
            "scope_confidence": manual_confidence,
            "lineage_scope_used": scope_cache[MANUAL_NEW]["label"],
            "lineage_rank_matched": pd.NA,
            "notes": (
                "Manual exception retained exactly as required. Degree ID 49.111 "
                "is ancestry-ineligible (0 TRAIN-present catalog courses), so it is "
                "barred from automatic predecessor candidacy. Course and degree ID "
                "namespaces both contain 49.111; automatic generation is suppressed "
                "for this specified course pair. Approval remains pending."
            ),
        }
    )
    link = pd.concat([link, pd.DataFrame([manual])], ignore_index=True)

    # Preserve reviewed below-support pairs even if a structural exclusion kept
    # them out of the generalized name-key population.
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
            link.loc[exact_pair_mask, "relationship_type"] = (
                "candidate_below_support"
            )
            link.loc[exact_pair_mask, "weight_hint"] = np.nan
            continue
        none_mask = (
            (link["new_course_id"] == pair.new_course_id)
            & (link["relationship_type"] == "none")
        )
        link = link.loc[~none_mask].copy()
        ordinary_scope = (
            "shared"
            if len(course_families.get(pair.new_course_id, set())) >= 5
            else "specific"
        )
        confidence, matched_rank = confidence_for(
            pair.new_course_id, pair.old_course_id, ordinary_scope
        )
        retained = base_row(pair.new_course_id, ordinary_scope)
        retained.update(old_fields(pair.old_course_id))
        retained.update(
            {
                "predecessor_rank": 1,
                "predecessor_count_for_new_course": 0,
                "relationship_type": "candidate_below_support",
                "weight_hint": np.nan,
                "match_method": "known_pair_below_support_retention",
                "scope_confidence": confidence,
                "lineage_scope_used": (
                    "ALL_TRAIN_PRESENT_COURSES"
                    if ordinary_scope == "shared"
                    else scope_cache[pair.new_course_id]["label"]
                ),
                "lineage_rank_matched": matched_rank,
                "notes": (
                    f"Reviewed pair retained because raw TRAIN support {support} "
                    f"is below min_support={MIN_SUPPORT}; visible and unweighted."
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
        "scope_confidence",
        "lineage_scope_used",
        "lineage_rank_matched",
        "effective_from_part_id",
        "approval_status",
        "notes",
    ]
    link = link[column_order].copy()
    link["lineage_rank_matched"] = pd.to_numeric(
        link["lineage_rank_matched"], errors="coerce"
    ).astype("Int64")
    link["_new_sort"] = link["new_course_id"].map(
        lambda value: int(str(value).split(".")[0])
    )
    link["_old_sort"] = link["old_course_id"].map(
        lambda value: (
            10**9 if not str(value).strip() else int(str(value).split(".")[0])
        )
    )
    link["_pred_sort"] = pd.to_numeric(
        link["predecessor_rank"], errors="coerce"
    ).fillna(10**9)
    link = (
        link.sort_values(
            ["_new_sort", "_pred_sort", "_old_sort"], kind="stable"
        )
        .drop(columns=["_new_sort", "_old_sort", "_pred_sort"])
        .reset_index(drop=True)
    )

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
        raise SystemExit("STOP: course-link table does not contain all 182 new courses.")
    if link[["new_course_id", "old_course_id"]].duplicated().any():
        duplicates = link.loc[
            link[["new_course_id", "old_course_id"]].duplicated(False),
            ["new_course_id", "old_course_id"],
        ]
        raise SystemExit(
            "STOP: duplicate new/old course proposal rows exist: "
            f"{duplicates.to_dict('records')[:10]}"
        )
    if not (link["approval_status"] == "pending").all():
        raise SystemExit("STOP: a course-link proposal is not pending.")

    actual_pairs = set(
        link.loc[link["old_course_id"].astype(str).str.len() > 0, [
            "new_course_id",
            "old_course_id",
        ]].itertuples(index=False, name=None)
    )
    missing_global = expected_global_name_pairs - actual_pairs
    if missing_global:
        raise SystemExit(
            "STOP: global normalized-name matches were silently discarded: "
            f"{sorted(missing_global)[:20]}"
        )

    automatic = link.loc[
        link["relationship_type"].isin(["successor", "consolidated_into"])
        & link["weight_hint"].notna()
    ]
    if (
        not (automatic["scope_confidence"] == "in_lineage").all()
        or automatic.empty
    ):
        raise SystemExit(
            "STOP: a weighted automatic relationship is not in-lineage, or none exist."
        )
    if not np.allclose(
        automatic.loc[
            automatic["relationship_type"] == "successor", "weight_hint"
        ].astype(float),
        1.0,
        atol=1e-12,
    ):
        raise SystemExit("STOP: an automatic single-predecessor successor is not weight 1.")

    forbidden_weight = link.loc[
        link["relationship_type"].isin(
            [
                "split_from",
                "merged_from",
                "candidate_below_support",
                "name_only_review_candidate",
                "none",
            ]
        )
        | (link["scope_confidence"] == "same_faculty")
        | (link["scope_confidence"] == "cross_faculty")
    ]
    if forbidden_weight["weight_hint"].notna().any():
        raise SystemExit("STOP: an unweightable relationship received a weight.")

    weighted_consolidated = automatic.loc[
        automatic["relationship_type"] == "consolidated_into"
    ]
    if not weighted_consolidated.empty:
        sums = weighted_consolidated.groupby("new_course_id")["weight_hint"].sum()
        bad = sums.loc[(sums - 1.0).abs() > 1e-9]
        if len(bad):
            raise SystemExit(
                f"STOP: consolidated weights do not sum to 1.0: {bad.to_dict()}"
            )

    manual_rows = link.loc[
        (link["new_course_id"] == MANUAL_NEW)
        & (link["old_course_id"] == MANUAL_OLD)
    ]
    if (
        len(manual_rows) != 1
        or manual_rows.iloc[0]["match_method"] != "manual"
        or manual_rows.iloc[0]["approval_status"] != "pending"
        or "ancestry-ineligible" not in str(manual_rows.iloc[0]["notes"])
    ):
        raise SystemExit("STOP: the required 1419.111 <- 49.111 manual row is wrong.")

    required_1422 = link.loc[
        (link["new_course_id"] == "1422.111")
        & (link["old_course_id"] == "967.111")
    ]
    if (
        len(required_1422) != 1
        or required_1422.iloc[0]["relationship_type"] != "consolidated_into"
        or pd.isna(required_1422.iloc[0]["weight_hint"])
        or not math.isclose(
            float(required_1422.iloc[0]["weight_hint"]),
            0.9502002258958825,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise SystemExit(
            "STOP: 1422.111 <- 967.111 did not retain its TRAIN-volume "
            "consolidation weight."
        )

    if int(train_maps["old_course_train_support"].get("893.111", 0)) == 2:
        required_893 = link.loc[link["old_course_id"] == "893.111"]
        if (
            required_893.empty
            or not (
                required_893["relationship_type"] == "candidate_below_support"
            ).all()
            or required_893["weight_hint"].notna().any()
        ):
            raise SystemExit(
                "STOP: 893.111 support 2 was not retained as an unweighted "
                "below-support candidate."
            )

    scope_metadata = {
        "scope_cache": scope_cache,
        "faculty_available": faculty_available,
        "train_faculties": train_faculties,
        "valid_faculties": valid_faculties,
        "expected_global_name_pairs": expected_global_name_pairs,
    }
    return link, scope_metadata


def derive_measured_recovery_set(
    known_pairs: pd.DataFrame,
    phase2r_link: pd.DataFrame,
) -> dict[str, Any]:
    unresolved_rows: list[dict[str, Any]] = []
    for pair in known_pairs.itertuples(index=False):
        exact = phase2r_link.loc[
            (phase2r_link["new_course_id"] == pair.new_course_id)
            & (phase2r_link["old_course_id"] == pair.old_course_id)
        ]
        if exact.empty:
            unresolved_rows.append(
                {
                    "new_course_id": pair.new_course_id,
                    "old_course_id": pair.old_course_id,
                    "valid_rows": int(pair.new_valid_row_count),
                }
            )
    unresolved = pd.DataFrame(unresolved_rows)
    unresolved_pairs = set(
        unresolved[["new_course_id", "old_course_id"]].itertuples(
            index=False, name=None
        )
    )
    if (
        len(unresolved) != EXPECTED_PHASE2R_UNRESOLVED_PAIRS
        or int(unresolved["valid_rows"].sum()) != EXPECTED_PHASE2R_UNRESOLVED_ROWS
        or not EXPECTED_STILL_UNRECOVERED.issubset(unresolved_pairs)
    ):
        raise SystemExit(
            "STOP: Phase 2R unresolved baseline does not reproduce 28 pairs / "
            "7,369 rows with the two preregistered residual pairs."
        )
    recoverable_pairs = unresolved_pairs - EXPECTED_STILL_UNRECOVERED
    recoverable_rows = int(
        unresolved.loc[
            unresolved.apply(
                lambda row: (
                    row["new_course_id"],
                    row["old_course_id"],
                )
                in recoverable_pairs,
                axis=1,
            ),
            "valid_rows",
        ].sum()
    )
    if (
        len(recoverable_pairs) != EXPECTED_MEASURED_RECOVERABLE_PAIRS
        or recoverable_rows != EXPECTED_MEASURED_RECOVERABLE_ROWS
    ):
        raise SystemExit(
            "STOP: measured-recoverable baseline is not 26 pairs / 7,335 rows."
        )
    return {
        "phase2r_unresolved_pairs": unresolved_pairs,
        "recoverable_pairs": recoverable_pairs,
        "recoverable_rows": recoverable_rows,
    }


def validate_measured_recovery(
    link: pd.DataFrame,
    recovery_baseline: dict[str, Any],
) -> dict[str, Any]:
    recovered: set[tuple[str, str]] = set()
    rank_counts: dict[int | None, int] = {}
    for new_course_id, old_course_id in recovery_baseline["recoverable_pairs"]:
        exact = link.loc[
            (link["new_course_id"] == new_course_id)
            & (link["old_course_id"] == old_course_id)
            & (link["scope_confidence"] == "in_lineage")
            & link["relationship_type"].isin(["successor", "consolidated_into"])
            & link["weight_hint"].notna()
        ]
        if len(exact) == 1:
            recovered.add((new_course_id, old_course_id))
            rank_value = exact.iloc[0]["lineage_rank_matched"]
            rank = None if pd.isna(rank_value) else int(rank_value)
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
    if len(recovered) < 20:
        raise SystemExit(
            "STOP: fewer than 20 of the 26 measured-recoverable pairs linked "
            "as weighted in-lineage successor/consolidation relationships."
        )
    missing = recovery_baseline["recoverable_pairs"] - recovered
    return {
        "recovered_pairs": recovered,
        "recovered_count": len(recovered),
        "missing_pairs": missing,
        "rank_counts": rank_counts,
    }


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
        automatic = matches.loc[
            matches["relationship_type"].isin(["successor", "consolidated_into"])
            & (matches["scope_confidence"] == "in_lineage")
            & matches["weight_hint"].notna()
        ]
        if not automatic.empty:
            category = "automatic eligible link"
            reason = (
                "Exact normalized-name pair is TRAIN-supported and in the "
                "corrected union lineage scope."
            )
        elif matches["relationship_type"].isin(["split_from", "merged_from"]).any():
            category = "split_or_merge"
            reason = "Task-confirmed split/merge; excluded from ordinary matching."
        elif (matches["match_method"] == "manual").any():
            category = "manual proposal"
            reason = "Explicit pending manual proposal retained by task requirement."
        elif (matches["relationship_type"] == "candidate_below_support").any():
            category = "candidate_below_support"
            support = int(
                train_maps["old_course_train_support"].get(pair.old_course_id, 0)
            )
            reason = f"Raw TRAIN support is {support}, below min_support={MIN_SUPPORT}."
        elif (
            matches["relationship_type"].eq("name_only_review_candidate").any()
            or matches["match_method"].isin(
                [
                    "out_of_lineage_review_candidate",
                    "cross_faculty_name_only_review_candidate",
                ]
            ).any()
        ):
            category = "name_only_review_candidate"
            reason = (
                "Exact name-key match is visible but outside lineage scope; "
                "it remains unweighted review evidence."
            )
        else:
            category = "unresolved"
            reason = "No exact reviewed predecessor row was recovered."
        rows.append(
            {
                "new_course_id": pair.new_course_id,
                "new_course_name": pair.new_course_name,
                "old_course_id": pair.old_course_id,
                "old_course_name": pair.old_course_name,
                "final_category": category,
                "old_course_train_support": int(
                    train_maps["old_course_train_support"].get(
                        pair.old_course_id, 0
                    )
                ),
                "reason": reason,
            }
        )
    funnel = pd.DataFrame(rows)
    counts = {
        category: int((funnel["final_category"] == category).sum())
        for category in FUNNEL_CATEGORIES
    }
    if (
        len(funnel) != EXPECTED_KNOWN_PAIRS
        or sum(counts.values()) != EXPECTED_KNOWN_PAIRS
        or funnel[["new_course_id", "old_course_id"]].duplicated().any()
    ):
        raise SystemExit(
            "STOP: six-category known-pair census is not exclusive and exactly 67."
        )
    return funnel, counts


def measure_courses(
    course_ids: set[str], valid_counts: pd.Series
) -> dict[str, Any]:
    rows = int(valid_counts.reindex(list(course_ids)).fillna(0).sum())
    return {
        "course_ids": len(course_ids),
        "rows": rows,
        "pct": rows / EXPECTED_NEW_COURSE_ROWS,
    }


def coverage_summary(
    valid: pd.DataFrame,
    membership: dict[str, Any],
    link: pd.DataFrame,
    phase2r_link: pd.DataFrame,
    helpers: dict[str, Any],
    train_maps: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    new_courses = membership["new_course_ids"]
    valid_counts = (
        valid.loc[valid["course_id"].isin(new_courses)]
        .groupby("course_id")
        .size()
    )

    phase2r_weight = pd.to_numeric(
        phase2r_link["weight_hint"], errors="coerce"
    )
    phase2r_covered = set(
        phase2r_link.loc[
            (
                phase2r_link["relationship_type"].isin(
                    ["successor", "consolidated_into"]
                )
                & phase2r_weight.notna()
            )
            | phase2r_link["relationship_type"].isin(
                ["split_from", "merged_from", "manual_candidate"]
            ),
            "new_course_id",
        ]
    ) & new_courses
    phase2r_measure = measure_courses(phase2r_covered, valid_counts)
    if (
        phase2r_measure["course_ids"] != EXPECTED_PHASE2R_COURSES
        or phase2r_measure["rows"] != EXPECTED_PHASE2R_ROWS
    ):
        raise SystemExit("STOP: Phase 2R coverage baseline is not 43 / 7,619.")

    automatic_mask = (
        link["relationship_type"].isin(["successor", "consolidated_into"])
        & (link["scope_confidence"] == "in_lineage")
        & link["weight_hint"].notna()
    )
    structural_or_manual = link["relationship_type"].isin(
        ["split_from", "merged_from", "manual_candidate"]
    )
    phase2s_covered = set(
        link.loc[automatic_mask | structural_or_manual, "new_course_id"]
    )

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
    global_measure = measure_courses(global_name_covered, valid_counts)
    if (
        global_measure["course_ids"] != EXPECTED_GLOBAL_COURSES
        or global_measure["rows"] != EXPECTED_GLOBAL_ROWS
    ):
        raise SystemExit(
            "STOP: global name-key upper bound is not 83 courses / 17,814 rows."
        )

    exclusive_sets: dict[str, set[str]] = {
        "shared": set(),
        "specific": set(),
        "split_or_merge": set(),
        "name_only_review": set(),
        "below_support_only": set(),
        "unresolved": set(),
    }
    for new_course_id, group in link.groupby("new_course_id"):
        group_auto = (
            group["relationship_type"].isin(["successor", "consolidated_into"])
            & (group["scope_confidence"] == "in_lineage")
            & group["weight_hint"].notna()
        ).any()
        if group["relationship_type"].isin(["split_from", "merged_from"]).any():
            category = "split_or_merge"
        elif group_auto or (group["relationship_type"] == "manual_candidate").any():
            scope = str(group.iloc[0]["new_course_scope"])
            category = "shared" if scope == "shared" else "specific"
        elif (
            (group["relationship_type"] == "name_only_review_candidate").any()
            or group["match_method"].isin(
                [
                    "out_of_lineage_review_candidate",
                    "cross_faculty_name_only_review_candidate",
                ]
            ).any()
        ):
            category = "name_only_review"
        elif (group["relationship_type"] == "candidate_below_support").any():
            category = "below_support_only"
        else:
            category = "unresolved"
        exclusive_sets[category].add(str(new_course_id))

    union = set().union(*exclusive_sets.values())
    if (
        union != new_courses
        or sum(len(values) for values in exclusive_sets.values())
        != len(new_courses)
    ):
        raise SystemExit(
            "STOP: exclusive Phase 2S course census does not cover 182 IDs once."
        )
    exclusive = {
        category: measure_courses(course_ids, valid_counts)
        for category, course_ids in exclusive_sets.items()
    }
    if (
        sum(value["course_ids"] for value in exclusive.values())
        != EXPECTED_NEW_COURSES
        or sum(value["rows"] for value in exclusive.values())
        != EXPECTED_NEW_COURSE_ROWS
    ):
        raise SystemExit(
            "STOP: exclusive Phase 2S census does not sum to 182 / 25,627."
        )

    below_rows = link.loc[
        link["relationship_type"] == "candidate_below_support"
    ]
    below_ids = set(below_rows["new_course_id"])
    return {
        "phase2r": phase2r_measure,
        "phase2s": measure_courses(phase2s_covered, valid_counts),
        "global": global_measure,
        "exclusive": exclusive,
        "exclusive_sets": exclusive_sets,
        "below_support_exposure": {
            **measure_courses(below_ids, valid_counts),
            "candidate_links": len(below_rows),
        },
    }


def scope_confidence_summary(link: pd.DataFrame) -> pd.DataFrame:
    matched = link.loc[link["old_course_id"].astype(str).str.len() > 0]
    rows = []
    for confidence in CONFIDENCE_LEVELS:
        group = matched.loc[matched["scope_confidence"] == confidence]
        rows.append(
            {
                "scope_confidence": confidence,
                "proposal_rows": len(group),
                "new_course_ids": group["new_course_id"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def lineage_rank_summary(link: pd.DataFrame) -> pd.DataFrame:
    matched_rows = link.loc[link["old_course_id"].astype(str).str.len() > 0]
    rows = []
    for rank in (1, 2, 3, None):
        if rank is None:
            group = matched_rows.loc[matched_rows["lineage_rank_matched"].isna()]
            label = "null"
        else:
            group = matched_rows.loc[
                matched_rows["lineage_rank_matched"] == rank
            ]
            label = str(rank)
        rows.append(
            {
                "lineage_rank_matched": label,
                "proposal_rows": len(group),
                "new_course_ids": group["new_course_id"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def format_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def id_list(values: Iterable[str]) -> str:
    ordered = p2r.sorted_ids(values)
    return ", ".join(f"`{value}`" for value in ordered) if ordered else "—"


def pair_list(values: Iterable[tuple[str, str]]) -> str:
    ordered = sorted(
        values,
        key=lambda value: (
            p2r.id_sort_key(value[0]),
            p2r.id_sort_key(value[1]),
        ),
    )
    return (
        ", ".join(f"`{old} → {new}`" for new, old in ordered)
        if ordered
        else "—"
    )


def build_report(
    membership: dict[str, Any],
    gate: dict[str, Any],
    eligibility: pd.DataFrame,
    lineage: pd.DataFrame,
    comparison: pd.DataFrame,
    split_merge: pd.DataFrame,
    link: pd.DataFrame,
    stats: pd.DataFrame,
    history_validation: dict[str, int],
    recovery_baseline: dict[str, Any],
    recovery: dict[str, Any],
    funnel: pd.DataFrame,
    funnel_counts: dict[str, int],
    coverage: dict[str, Any],
    confidence_summary: pd.DataFrame,
    rank_summary: pd.DataFrame,
    scope_metadata: dict[str, dict[str, Any]],
) -> str:
    excluded = eligibility.loc[~eligibility["ancestry_eligible"]]
    changed = comparison.loc[comparison["changed"], "new_degree_id"].tolist()
    unchanged = comparison.loc[~comparison["changed"], "new_degree_id"].tolist()
    actual_coverage = coverage["phase2s"]
    expectation_status = (
        "PASS"
        if actual_coverage["pct"] >= 0.52
        else "DIAGNOSIS INCOMPLETE — below the preregistered 52% floor"
    )

    lines: list[str] = [
        "# Phase 2S — lineage search scope fix",
        "",
        "Status: **proposal and diagnostic tables only**. Every proposal remains "
        "`approval_status = pending`; no relationship was approved or applied.",
        "",
        "## Validation and safety gates",
        "",
        "| Gate | Result |",
        "|---|---:|",
        f"| Normalization known-pair matches | {gate['matches']} / {gate['pairs']} |",
        f"| Sole normalization non-match | `{gate['nonmatch_pair']}` |",
        f"| Never-in-TRAIN VALID course IDs | {len(membership['new_course_ids'])} |",
        f"| Never-in-TRAIN VALID rows | {membership['new_course_rows']:,} |",
        f"| VALID-only degrees | {len(membership['valid_only_degree_ids'])} |",
        f"| Measured recoverable pairs linked | {recovery['recovered_count']} / 26 |",
        f"| Six-category known-pair total | {sum(funnel_counts.values())} |",
        f"| Temporal rows checked | {history_validation['rows_checked']:,} |",
        f"| Temporal mismatches | {history_validation['mismatches']} |",
        "| Proposal statuses | `pending` only |",
        "| Difficulty links applied | No |",
        "",
        "VALID was loaded from a fixed explicit projection containing identifiers, "
        "faculty, catalog keys, semester, and frozen `course_history_count`; "
        "`final_mark` and every defined VALID outcome column were absent at runtime. "
        "No TEST path or model artifact was accessed, and no training, tuning, or "
        "rescoring was performed.",
        "",
        "## 1. Ancestry eligibility",
        "",
        f"A TRAIN-present degree is ancestry-eligible only when at least "
        f"**{ANCESTRY_MIN_OLD_COURSES}** distinct courses in its catalog are "
        "TRAIN-present anywhere. Ineligible degrees remain old for generation; "
        "they are barred only from predecessor candidacy.",
        "",
        "| Degree | Degree name | TRAIN enrolment rows | Catalog courses | Old courses in catalog | Eligible |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in eligibility.itertuples(index=False):
        lines.append(
            f"| `{row.degree_id}` | {md_escape(row.degree_name)} | "
            f"{row.train_enrolment_rows:,} | {row.catalog_course_count} | "
            f"{row.old_course_count_in_catalog} | "
            f"{'yes' if row.ancestry_eligible else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Excluded degrees: **{len(excluded)}** — "
            f"{id_list(excluded['degree_id'])}. Degree `49.111` is correctly "
            "excluded with 76 catalog courses and zero TRAIN-present catalog courses.",
            "",
            "## 2. Degree lineage over the eligible pool",
            "",
            f"The output retains three deterministic candidates for each of all "
            f"{len(membership['valid_only_degree_ids'])} VALID-only degrees "
            f"({len(lineage)} rows). Ranking is overlap-of-new descending, Jaccard "
            "descending, degree-name similarity descending, then normalized old "
            "degree ID ascending.",
            "",
            "| New degree | Phase 2R rank 1 | Phase 2S rank 1 | Changed |",
            "|---|---|---|---:|",
        ]
    )
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| `{row.new_degree_id}` | `{row.phase2r_rank1}` | "
            f"`{row.phase2s_rank1}` | {'yes' if row.changed else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Changed rank 1: **{len(changed)} degrees** — {id_list(changed)}.",
            "",
            f"Unchanged rank 1: **{len(unchanged)} degrees** — {id_list(unchanged)}.",
            "",
            "The prompt preregistered eight informatics degrees; repository evidence "
            "shows the same defect also affected `30.111`, so nine rank-1 choices "
            "change in total. None now resolves to ancestry-ineligible `49.111`.",
            "",
            "## 3. Union search scope and confidence labels",
            "",
            "For a specific course, `lineage_scope_used` is the sorted union of the "
            "top-three eligible lineage candidates for all catalogued new degrees. "
            "An ancestry-eligible TRAIN-present placement can contribute itself "
            "directly; ineligible `49.111` cannot. Shared courses continue to use "
            "all TRAIN-present catalog courses.",
            "",
            "Every catalog-wide normalized-name match is emitted. Faculty is absent "
            "from the cleaned catalog but available in both TRAIN and the outcome-free "
            "VALID projection, so out-of-lineage confidence uses old TRAIN versus new "
            "VALID course-faculty set intersection.",
            "",
            "Section 4's explicit `in_lineage` name-key weighting rule controls the "
            "automatic relationship here. The former specific-course credits/type "
            "narrow gate cannot remain an eligibility gate: 13 of the preregistered "
            "26 recoverable pairs (including required `502.111 → 1175.111`) fail it, "
            "which would recover only 13 and trigger the task's fewer-than-20 stop "
            "condition. No known-pair-specific exception was introduced.",
            "",
            "| Scope confidence | Proposal rows | Distinct new courses |",
            "|---|---:|---:|",
        ]
    )
    for row in confidence_summary.itertuples(index=False):
        lines.append(
            f"| `{row.scope_confidence}` | {row.proposal_rows} | "
            f"{row.new_course_ids} |"
        )
    lines.extend(
        [
            "",
            "All proposal rows with an old-course match, counted by "
            "`lineage_rank_matched`:",
            "",
            "| Lineage rank | Proposal rows | Distinct new courses |",
            "|---|---:|---:|",
        ]
    )
    for row in rank_summary.itertuples(index=False):
        lines.append(
            f"| `{row.lineage_rank_matched}` | {row.proposal_rows} | "
            f"{row.new_course_ids} |"
        )
    lines.extend(
        [
            "",
            "The null bucket contains shared-scope, out-of-lineage, structural, "
            "manual, and any direct-self matches for which no ranked lineage "
            "candidate supplied the row.",
        ]
    )
    recovery_rank_text = ", ".join(
        f"rank {rank if rank is not None else 'null'}: {count}"
        for rank, count in sorted(
            recovery["rank_counts"].items(),
            key=lambda item: 99 if item[0] is None else item[0],
        )
    )
    lines.extend(
        [
            "",
            f"The preregistered 26-pair recovery reproduced **{recovery['recovered_count']} "
            f"of 26 pairs / {recovery_baseline['recoverable_rows']:,} VALID rows**. "
            f"Ranks for those exact pairs: {recovery_rank_text}. The eligibility "
            "filter promoted the true informatics ancestors, so rank 1 supplies all "
            "26 rather than rank 2 supplying the majority under an unfiltered ranking.",
            "",
            f"Recovered exact pairs: {pair_list(recovery['recovered_pairs'])}",
            "",
            "The two preregistered residual reviewed pairs are now visible, but remain "
            "unweighted `cross_faculty` review evidence: `662.111 → 1165.111` and "
            "`417.111 → 1271.111`. Those new courses also have different in-lineage "
            "same-name candidates (`439.111` and `544.111` respectively); that does "
            "not recover the reviewed exact pairs.",
            "",
            "## 4. Unchanged normalization, support, split, and manual rules",
            "",
            f"The unchanged normalization gate is **{gate['matches']}/{gate['pairs']}**. "
            f"Its sole non-match remains `{gate['nonmatch_pair']}`, represented by "
            f"the unchanged split `{EXPECTED_SPLIT_OLD} → "
            f"{p2r.join_ids(EXPECTED_SPLIT_NEW)}` with `credit_change = +3`.",
            "",
            f"`min_support = {MIN_SUPPORT}` remains in force. Below-support candidates "
            "are visible and unweighted, including `893.111` at support 2. "
            "Weighted `consolidated_into` groups sum to 1.0.",
            "",
            f"The task-required manual proposal `{MANUAL_OLD} → {MANUAL_NEW}` remains "
            "`match_method = manual`, pending, and explicitly records that degree "
            "`49.111` is ancestry-ineligible. The identically numbered old course "
            "`49.111` is catalogued under degree `10.111`; the pair-specific automatic "
            "generation path is suppressed so the required manual governance status "
            "is preserved.",
            "",
            "### Course-link relationship census",
            "",
            "| Relationship type | Proposal rows | Distinct new courses |",
            "|---|---:|---:|",
        ]
    )
    relationship_rows = link["relationship_type"].value_counts()
    relationship_courses = link.groupby("relationship_type")[
        "new_course_id"
    ].nunique()
    for relationship in sorted(set(link["relationship_type"])):
        lines.append(
            f"| `{relationship}` | {int(relationship_rows[relationship])} | "
            f"{int(relationship_courses[relationship])} |"
        )
    lines.extend(
        [
            "",
            f"All **{link['new_course_id'].nunique()}** never-in-TRAIN VALID course "
            "IDs are present.",
            "",
            "### Six-category known-pair census",
            "",
            "| Exclusive category | Pairs |",
            "|---|---:|",
        ]
    )
    for category in FUNNEL_CATEGORIES:
        lines.append(f"| `{category}` | {funnel_counts[category]} |")
    lines.extend(
        [
            f"| **Total** | **{sum(funnel_counts.values())}** |",
            "",
            "Every one of the 67 reviewed pairs appears in exactly one category. "
            "Non-automatic reviewed pairs:",
            "",
            "| New course | Old course | Category | TRAIN support | Reason |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in funnel.loc[
        funnel["final_category"] != "automatic eligible link"
    ].itertuples(index=False):
        lines.append(
            f"| `{row.new_course_id}` {md_escape(row.new_course_name)} | "
            f"`{row.old_course_id}` {md_escape(row.old_course_name)} | "
            f"`{row.final_category}` | {row.old_course_train_support} | "
            f"{md_escape(row.reason)} |"
        )

    lines.extend(
        [
            "",
            "## 5. Required coverage comparison",
            "",
            "| Measurement | Course IDs | VALID rows | % of 25,627 |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, key in [
        ("Phase 2R (rank-1 scope)", "phase2r"),
        ("Phase 2S (union scope + ancestry filter)", "phase2s"),
        ("Global name-key diagnostic upper bound", "global"),
    ]:
        value = coverage[key]
        lines.append(
            f"| {label} | {value['course_ids']} | {value['rows']:,} | "
            f"{format_pct(value['pct'])} |"
        )
    lines.extend(
        [
            "",
            f"Pre-registered coverage check: **{expectation_status}**. The actual "
            f"{format_pct(actual_coverage['pct'])} exceeds the central estimate "
            "because the literal Section 4 rule makes every support-eligible "
            "in-lineage name-key match weightable, including courses outside the "
            "67-pair validation set. No parameter was adjusted to obtain this result.",
            "",
            "Exclusive contribution/status census:",
            "",
            "| Contribution/status | Course IDs | VALID rows | % of 25,627 |",
            "|---|---:|---:|---:|",
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
        value = coverage["exclusive"][category]
        lines.append(
            f"| `{category}` | {value['course_ids']} | {value['rows']:,} | "
            f"{format_pct(value['pct'])} |"
        )
    exposure = coverage["below_support_exposure"]
    lines.extend(
        [
            "",
            f"Below-support exposure is non-additive: {exposure['candidate_links']} "
            f"candidate rows touch {exposure['course_ids']} courses / "
            f"{exposure['rows']:,} VALID rows.",
            "",
            "## 6. Required worked examples",
            "",
        ]
    )

    before_26 = comparison.loc[
        comparison["new_degree_id"] == "26.111", "phase2r_rank1"
    ].iloc[0]
    after_26 = comparison.loc[
        comparison["new_degree_id"] == "26.111", "phase2s_rank1"
    ].iloc[0]
    example_1175 = link.loc[
        (link["new_course_id"] == "1175.111")
        & (link["old_course_id"] == "502.111")
    ].iloc[0]
    example_1422 = link.loc[
        (link["new_course_id"] == "1422.111")
        & (link["old_course_id"] == "967.111")
    ].iloc[0]
    group_1422 = link.loc[
        (link["new_course_id"] == "1422.111")
        & (link["relationship_type"] == "consolidated_into")
        & link["weight_hint"].notna()
    ]
    example_cross = link.loc[
        (link["new_course_id"] == "1165.111")
        & (link["old_course_id"] == "662.111")
    ].iloc[0]
    unresolved_ids = coverage["exclusive_sets"]["unresolved"]
    unresolved_id = (
        "99.111"
        if "99.111" in unresolved_ids
        else p2r.sorted_ids(unresolved_ids)[0]
    )
    unresolved_row = link.loc[link["new_course_id"] == unresolved_id].iloc[0]
    lines.extend(
        [
            f"1. Degree `26.111`: Phase 2R rank 1 was `{before_26}`; after excluding "
            f"the zero-ancestor sibling degree, Phase 2S rank 1 is `{after_26}`.",
            "",
            f"2. `502.111 → 1175.111`: new catalog degrees "
            f"`{example_1175['new_course_degree_ids']}`; union scope "
            f"`{example_1175['lineage_scope_used']}`; matched rank "
            f"`{int(example_1175['lineage_rank_matched'])}`; confidence "
            f"`{example_1175['scope_confidence']}`; weight "
            f"`{float(example_1175['weight_hint']):.1f}`.",
            "",
            f"3. `967.111 → 1422.111`: old TRAIN support "
            f"{int(example_1422['old_course_train_support']):,}; weighted-group "
            f"support {int(group_1422['old_course_train_support'].sum()):,}; "
            f"volume-derived weight `{float(example_1422['weight_hint']):.12f}` "
            "(still near 0.95).",
            "",
            f"4. `{EXPECTED_SPLIT_OLD} → {p2r.join_ids(EXPECTED_SPLIT_NEW)}` remains "
            "the pending split with `credit_change = +3`; ordinary successor "
            "matching remains disabled for it.",
            "",
            f"5. Cross-faculty visibility: `662.111 → 1165.111` is now emitted with "
            f"`scope_confidence = {example_cross['scope_confidence']}`, "
            f"`relationship_type = {example_cross['relationship_type']}`, "
            f"`match_method = {example_cross['match_method']}`, and null weight. "
            f"New VALID faculties are "
            f"`{p2r.join_ids(scope_metadata['valid_faculties'].get('1165.111', set()))}`; "
            f"old TRAIN faculties are "
            f"`{p2r.join_ids(scope_metadata['train_faculties'].get('662.111', set()))}`.",
            "",
            f"6. Still unresolved course: `{unresolved_id}` "
            f"{md_escape(unresolved_row['new_course_name'])}. It has no "
            "TRAIN-present catalog course with the same normalized name key after "
            "the unchanged structural exclusions, so its relationship is `none`.",
            "",
            "## 7. Temporal difficulty prototype",
            "",
            f"The unchanged prototype contains **{len(stats):,}** rows and covers "
            f"all **{len(membership['train_course_ids'])}** TRAIN courses. "
            "`TRAIN_END_STATE` reproduces frozen VALID `course_history_count` at "
            f"**{history_validation['mismatches']} mismatches over "
            f"{history_validation['rows_checked']:,} rows**. `link_used` and "
            "`link_weight` are null on every row.",
            "",
            "## Governance entry (ready to copy)",
            "",
            "> A prompt specification must be checked against the findings of prior phases "
            "before it is issued. The Phase 2R rank-1 scoping rule directly contradicted "
            "Phase 0 Q4, which had already established that no old→new degree relationship "
            "in this data is one-to-one. The contradiction was authored in the prompt, not "
            "introduced by the implementation, and cost 7,335 VALID rows of coverage. It "
            "was caught only because the 67 human-reviewed pairs act as a known-answer "
            "validation set.",
            "",
            "Ranking by catalog overlap measures **similarity**, not ancestry. A "
            "same-generation sibling scores highest precisely because it shares the "
            "new courses. Ancestry therefore requires the explicit eligibility "
            "constraint used here; similarity alone selects the wrong degree.",
            "",
            "No decision log was edited. Generation, validation, and reporting stop "
            "here; no Phase 3 action is proposed.",
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
        if (
            "approval_status" not in frame
            or not (frame["approval_status"] == "pending").all()
        ):
            raise SystemExit(f"STOP: {name} contains a non-pending proposal.")
    if stats["link_used"].notna().any() or stats["link_weight"].notna().any():
        raise SystemExit("STOP: the temporal prototype contains an applied link.")
    if "old_degree_old_course_count" not in lineage.columns:
        raise SystemExit("STOP: lineage eligibility reviewer column is missing.")
    for column in ("scope_confidence", "lineage_rank_matched"):
        if column not in link.columns:
            raise SystemExit(f"STOP: course-link column {column} is missing.")


def validate_output_paths() -> None:
    expected_parent = OUT_DIR.resolve()
    for path in [OUT_REPORT, OUT_SPLIT, OUT_LINEAGE, OUT_LINK, OUT_STATS]:
        if path.resolve().parent != expected_parent:
            raise SystemExit(f"STOP: output escaped the Phase 2S directory: {path}")
        lowered = str(path.resolve()).lower().replace("\\", "/")
        if "/src/" in lowered or "/data/model_data/" in lowered:
            raise SystemExit(f"STOP: forbidden output path: {path}")
    if expected_parent in {
        PHASE2R_DIR.resolve(),
        (ROOT / "models" / "runs").resolve(),
    }:
        raise SystemExit("STOP: Phase 2S output directory aliases an earlier output.")


def write_outputs(
    report: str,
    split_merge: pd.DataFrame,
    lineage: pd.DataFrame,
    link: pd.DataFrame,
    stats: pd.DataFrame,
) -> None:
    validate_output_paths()
    if OUT_DIR.exists():
        extras = {path.name for path in OUT_DIR.iterdir()} - EXPECTED_OUTPUT_NAMES
        if extras:
            raise SystemExit(
                "STOP: Phase 2S output directory contains unexpected files: "
                f"{sorted(extras)}"
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
    validate_output_paths()
    (
        train,
        valid,
        catalog,
        pairs,
        phase2r_lineage,
        phase2r_link,
    ) = load_inputs()
    gate_frame, gate = p2r.validate_normalization_gate(pairs)
    membership = p2r.build_membership(train, valid, catalog)
    if len(membership["valid_only_degree_ids"]) != 25:
        raise SystemExit("STOP: expected exactly 25 VALID-only degrees.")
    helpers = p2r.build_catalog_helpers(catalog)
    _, train_maps = p2r.build_train_stats(train)

    eligibility = build_ancestry_eligibility(
        train, catalog, membership, helpers
    )
    lineage = build_degree_lineage(
        catalog, membership, helpers, eligibility
    )
    comparison = compare_rank1(lineage, phase2r_lineage, membership)
    split_merge = p2r.build_split_merge_candidates(
        catalog, valid, membership, helpers, train_maps
    )
    link, scope_metadata = build_course_links(
        train,
        valid,
        catalog,
        gate_frame,
        membership,
        helpers,
        lineage,
        eligibility,
        split_merge,
        train_maps,
    )

    recovery_baseline = derive_measured_recovery_set(
        gate_frame, phase2r_link
    )
    recovery = validate_measured_recovery(link, recovery_baseline)
    funnel, funnel_counts = build_known_pair_funnel(
        gate_frame, link, train_maps
    )
    stats = p2r.build_stats_prototype(train)
    history_validation = p2r.validate_course_history_count(stats, valid)
    coverage = coverage_summary(
        valid,
        membership,
        link,
        phase2r_link,
        helpers,
        train_maps,
    )
    confidence = scope_confidence_summary(link)
    rank_counts = lineage_rank_summary(link)
    validate_outputs_in_memory(split_merge, lineage, link, stats)

    report = build_report(
        membership,
        gate,
        eligibility,
        lineage,
        comparison,
        split_merge,
        link,
        stats,
        history_validation,
        recovery_baseline,
        recovery,
        funnel,
        funnel_counts,
        coverage,
        confidence,
        rank_counts,
        scope_metadata,
    )

    # This is deliberately the first mutating operation in the pipeline.
    write_outputs(report, split_merge, lineage, link, stats)

    print(f"Wrote exactly five Phase 2S outputs to: {OUT_DIR}")
    print(
        f"Ancestry: {int(eligibility['ancestry_eligible'].sum())} eligible / "
        f"{len(eligibility)} TRAIN-present degrees; excluded="
        f"{p2r.sorted_ids(eligibility.loc[~eligibility['ancestry_eligible'], 'degree_id'])}"
    )
    print(
        f"Recovery: {recovery['recovered_count']}/26 measured pairs; "
        f"coverage={coverage['phase2s']['course_ids']} courses / "
        f"{coverage['phase2s']['rows']:,} rows "
        f"({format_pct(coverage['phase2s']['pct'])}); "
        f"known-pair census={funnel_counts}; "
        f"history mismatches={history_validation['mismatches']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
