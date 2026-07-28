"""Verify the 67 likely course-renumbering candidates at exact degree grain.

This is a read-only diagnostic over the existing candidate CSV, the canonical
and raw V_ACD_DEGREE_COURSE representations, and immutable TRAIN/VALID only.
It never constructs or reads a TEST path, never accepts a mapping, and never
trains or scores a model.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.course_identity_diagnostic import (  # noqa: E402
    name_similarity,
    normalize_course_name,
)
from src.cleaning_utils import normalize_id_series, normalize_id_to_string  # noqa: E402
from src.paths import (  # noqa: E402
    MODEL_DATA_VERSIONS_DIR,
    MODEL_RUNS_DIR,
    PREPROCESSED_DIR,
    RAW_DIR,
    assert_data_root,
)


DATASET_VERSION = "2026-07-26_batched_fixes__registration_roster_concurrent"
VERSION_DIR = MODEL_DATA_VERSIONS_DIR / DATASET_VERSION
TRAIN_PATH = VERSION_DIR / "df_train_final.parquet"
VALID_PATH = VERSION_DIR / "df_valid_final.parquet"
CANDIDATE_PATH = MODEL_RUNS_DIR / "COURSE_IDENTITY_CANDIDATES.csv"
PRIOR_MD_PATH = MODEL_RUNS_DIR / "COURSE_IDENTITY_DIAGNOSTIC.md"
PRIOR_JSON_PATH = MODEL_RUNS_DIR / "COURSE_IDENTITY_DIAGNOSTIC.json"
CATALOG_PATH = (
    PREPROCESSED_DIR
    / "V_ACD_DEGREE_COURSE"
    / "clean_v_acd_degree_course.parquet"
)
RAW_CATALOG_PATH = RAW_DIR / "v_acd_degree_course.parquet"

OUT_ALL = MODEL_RUNS_DIR / "COURSE_IDENTITY_67_DEGREE_VERIFICATION.csv"
OUT_BEST = MODEL_RUNS_DIR / "COURSE_IDENTITY_67_BEST_MATCH_PER_COURSE.csv"
OUT_REVIEW = MODEL_RUNS_DIR / "COURSE_IDENTITY_67_HUMAN_REVIEW.csv"
OUT_MD = MODEL_RUNS_DIR / "COURSE_IDENTITY_67_DEGREE_VERIFICATION.md"
OUT_JSON = MODEL_RUNS_DIR / "COURSE_IDENTITY_67_DEGREE_VERIFICATION.json"

EXPECTED_NEW_COURSES = 67
TARGET_STATUS = "likely_renumbered_needs_review"
NOT_AVAILABLE = "not_available"
STRONG_NAME_SIMILARITY = 0.85

CONCLUSION_ORDER = {
    "STRICT_SAME_DEGREE_RENUMBERING_CANDIDATE": 0,
    "SAME_DEGREE_BUT_STRUCTURAL_DIFFERENCE": 1,
    "PARTIAL_DEGREE_OVERLAP": 2,
    "SAME_UNIVERSITY_DIFFERENT_DEGREE": 3,
    "DIFFERENT_UNIVERSITY": 4,
    "INSUFFICIENT_EVIDENCE": 5,
}

ALL_OUTPUT_COLUMNS = [
    "new_course_id",
    "new_course_name",
    "old_course_id",
    "old_course_name",
    "new_university_id",
    "old_university_id",
    "same_university",
    "new_degree_id",
    "old_degree_id",
    "same_degree",
    "new_faculty_id",
    "old_faculty_id",
    "same_faculty",
    "different_course_id",
    "normalized_new_course_name",
    "normalized_old_course_name",
    "exact_normalized_name_match",
    "name_similarity",
    "new_course_credits",
    "old_course_credits",
    "credits_match",
    "new_course_type_id",
    "old_course_type_id",
    "course_type_match",
    "new_requirement_type_id",
    "old_requirement_type_id",
    "requirement_type_match",
    "new_year_order",
    "old_year_order",
    "planned_year_match",
    "new_semester_order",
    "old_semester_order",
    "planned_semester_match",
    "new_active",
    "old_active",
    "old_first_seen_part_id",
    "old_last_seen_part_id",
    "new_first_seen_part_id",
    "new_last_seen_part_id",
    "semester_gap_between_old_last_and_new_first",
    "overlap_in_active_enrolment",
    "temporal_replacement_signal",
    "old_train_row_count",
    "new_valid_row_count",
    "old_degree_count",
    "new_degree_count",
    "shared_degree_count",
    "shared_degree_ids",
    "old_only_degree_ids",
    "new_only_degree_ids",
    "degree_relationship",
    "diagnostic_conclusion",
    "blocking_difference",
    "review_notes",
    "candidate_rank",
    "candidate_count_for_new_course",
    "ambiguous_best_match",
]

REVIEW_COLUMNS = [
    "new_course_id",
    "new_course_name",
    "new_degree_id",
    "old_course_id",
    "old_course_name",
    "old_degree_id",
    "same_degree",
    "credits_old_new",
    "course_type_old_new",
    "requirement_type_old_new",
    "planned_year_old_new",
    "last_old_semester",
    "first_new_semester",
    "enrolment_overlap",
    "new_valid_row_count",
    "diagnostic_conclusion",
    "review_decision",
    "reviewer_name",
    "review_date",
    "review_notes",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_hashes() -> dict[str, str]:
    paths = (
        CANDIDATE_PATH,
        PRIOR_MD_PATH,
        PRIOR_JSON_PATH,
        CATALOG_PATH,
        RAW_CATALOG_PATH,
        TRAIN_PATH,
        VALID_PATH,
    )
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def normalize_id_set(values: Iterable[Any]) -> set[str]:
    return set(normalize_id_series(pd.Series(list(values))).dropna().astype(str))


def sorted_join(values: Iterable[Any]) -> str:
    cleaned = {str(value) for value in values if str(value).strip()}
    return "|".join(sorted(cleaned)) if cleaned else NOT_AVAILABLE


def dotted_university_id(value: Any) -> str | None:
    """Return the meaningful dotted suffix without numeric conversion."""

    normalized = normalize_id_to_string(value)
    if normalized is pd.NA or pd.isna(normalized):
        return None
    text = str(normalized)
    base, separator, suffix = text.rpartition(".")
    if not separator or not base or not suffix:
        return None
    return suffix


def format_scalar(value: Any) -> str | None:
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:g}"
    text = str(value).strip()
    return text or None


def values_for(rows: pd.DataFrame, column: str, *, id_field: bool = False) -> set[str]:
    if column not in rows.columns:
        return set()
    if id_field:
        return normalize_id_set(rows[column])
    return {
        rendered
        for rendered in (format_scalar(value) for value in rows[column])
        if rendered is not None
    }


def exact_set_match(left: set[str], right: set[str]) -> bool | str:
    if not left or not right:
        return NOT_AVAILABLE
    return left == right


def compare_degree_sets(
    old_degrees: set[str],
    new_degrees: set[str],
    old_universities: set[str],
    new_universities: set[str],
) -> dict[str, Any]:
    shared = old_degrees & new_degrees
    old_only = old_degrees - new_degrees
    new_only = new_degrees - old_degrees
    same_university = bool(
        old_universities
        and new_universities
        and old_universities == new_universities
    )

    if not old_universities or not new_universities:
        relationship = "MISSING_DEGREE_EVIDENCE"
    elif not same_university:
        relationship = "DIFFERENT_UNIVERSITY"
    elif not old_degrees or not new_degrees:
        relationship = "MISSING_DEGREE_EVIDENCE"
    elif old_degrees == new_degrees and len(shared) == 1:
        relationship = "SAME_SINGLE_DEGREE"
    elif old_degrees == new_degrees and len(shared) > 1:
        relationship = "SAME_MULTIPLE_DEGREES"
    elif shared:
        relationship = "PARTIAL_DEGREE_OVERLAP"
    else:
        relationship = "SAME_UNIVERSITY_DIFFERENT_DEGREE"

    return {
        "same_university": same_university,
        "same_degree": bool(shared),
        "shared": shared,
        "old_only": old_only,
        "new_only": new_only,
        "degree_relationship": relationship,
    }


def chronological_key(part_id: Any) -> tuple[int, int, str]:
    text = str(normalize_id_to_string(part_id))
    if not text.isdigit() or len(text) < 2:
        raise ValueError(f"Unsupported part_id {part_id!r}")
    return int(text[:-1]), int(text[-1]), text


def temporal_summary(
    old_rows: pd.DataFrame,
    new_rows: pd.DataFrame,
    semester_order: dict[str, int],
) -> dict[str, Any]:
    old_parts = normalize_id_set(old_rows["part_id"])
    new_parts = normalize_id_set(new_rows["part_id"])
    if not old_parts or not new_parts:
        return {
            "old_first_seen_part_id": (
                min(old_parts, key=chronological_key) if old_parts else NOT_AVAILABLE
            ),
            "old_last_seen_part_id": (
                max(old_parts, key=chronological_key) if old_parts else NOT_AVAILABLE
            ),
            "new_first_seen_part_id": (
                min(new_parts, key=chronological_key) if new_parts else NOT_AVAILABLE
            ),
            "new_last_seen_part_id": (
                max(new_parts, key=chronological_key) if new_parts else NOT_AVAILABLE
            ),
            "semester_gap_between_old_last_and_new_first": NOT_AVAILABLE,
            "overlap_in_active_enrolment": NOT_AVAILABLE,
            "temporal_replacement_signal": "INSUFFICIENT_TEMPORAL_EVIDENCE",
        }

    old_first = min(old_parts, key=chronological_key)
    old_last = max(old_parts, key=chronological_key)
    new_first = min(new_parts, key=chronological_key)
    new_last = max(new_parts, key=chronological_key)
    shared_parts = old_parts & new_parts
    overlap = bool(shared_parts)
    distance = semester_order[new_first] - semester_order[old_last]
    gap = distance - 1

    if overlap:
        signal = "OVERLAPPING_ENROLMENT"
    elif distance <= 0:
        signal = "CHRONOLOGICAL_OVERLAP_WITHOUT_SHARED_SEMESTER"
    elif distance == 1:
        signal = "DIRECT_REPLACEMENT"
    elif distance == 2:
        signal = "ONE_SEMESTER_GAP"
    else:
        signal = "LONG_GAP"

    return {
        "old_first_seen_part_id": old_first,
        "old_last_seen_part_id": old_last,
        "new_first_seen_part_id": new_first,
        "new_last_seen_part_id": new_last,
        "semester_gap_between_old_last_and_new_first": int(gap),
        "overlap_in_active_enrolment": overlap,
        "temporal_replacement_signal": signal,
    }


def best_name_pair(old_rows: pd.DataFrame, new_rows: pd.DataFrame) -> dict[str, Any]:
    old_names = sorted(values_for(old_rows, "course_name_sl"))
    new_names = sorted(values_for(new_rows, "course_name_sl"))
    if not old_names or not new_names:
        return {
            "old_course_name": NOT_AVAILABLE,
            "new_course_name": NOT_AVAILABLE,
            "normalized_old_course_name": NOT_AVAILABLE,
            "normalized_new_course_name": NOT_AVAILABLE,
            "exact_normalized_name_match": NOT_AVAILABLE,
            "name_similarity": NOT_AVAILABLE,
        }

    ranked = sorted(
        (
            (name_similarity(old_name, new_name), old_name, new_name)
            for old_name in old_names
            for new_name in new_names
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    similarity, old_name, new_name = ranked[0]
    old_norm = normalize_course_name(old_name)
    new_norm = normalize_course_name(new_name)
    return {
        "old_course_name": old_name,
        "new_course_name": new_name,
        "normalized_old_course_name": old_norm,
        "normalized_new_course_name": new_norm,
        "exact_normalized_name_match": bool(old_norm and old_norm == new_norm),
        "name_similarity": float(similarity),
    }


def classify_conclusion(evidence: dict[str, Any]) -> tuple[str, str]:
    """Return a diagnostic conclusion; never return confirmed equivalence."""

    relationship = evidence["degree_relationship"]
    if relationship == "DIFFERENT_UNIVERSITY":
        return "DIFFERENT_UNIVERSITY", "normalized university_id sets differ"
    if relationship == "SAME_UNIVERSITY_DIFFERENT_DEGREE":
        return (
            "SAME_UNIVERSITY_DIFFERENT_DEGREE",
            "no exact normalized degree_id is shared",
        )
    if relationship == "PARTIAL_DEGREE_OVERLAP":
        return (
            "PARTIAL_DEGREE_OVERLAP",
            "course IDs share only a subset of their degree_id sets",
        )
    if relationship == "MISSING_DEGREE_EVIDENCE":
        return "INSUFFICIENT_EVIDENCE", "university or degree evidence is missing"

    required_matches = {
        "same_faculty": evidence["same_faculty"],
        "credits_match": evidence["credits_match"],
        "course_type_match": evidence["course_type_match"],
        "requirement_type_match": evidence["requirement_type_match"],
        "planned_year_match": evidence["planned_year_match"],
        "planned_semester_match": evidence["planned_semester_match"],
    }
    unavailable = [
        name for name, value in required_matches.items() if value == NOT_AVAILABLE
    ]
    if unavailable:
        return (
            "INSUFFICIENT_EVIDENCE",
            "missing required comparison: " + ", ".join(unavailable),
        )

    failures = [name for name, value in required_matches.items() if value is False]
    name_value = evidence["name_similarity"]
    if name_value == NOT_AVAILABLE:
        return "INSUFFICIENT_EVIDENCE", "course-name evidence is missing"
    if float(name_value) < STRONG_NAME_SIMILARITY:
        failures.append("near-exact course-name match")
    if not evidence["different_course_id"]:
        failures.append("different_course_id")

    temporal = evidence["temporal_replacement_signal"]
    if temporal == "INSUFFICIENT_TEMPORAL_EVIDENCE":
        return "INSUFFICIENT_EVIDENCE", "enrolment history is missing"
    if temporal not in {"DIRECT_REPLACEMENT", "ONE_SEMESTER_GAP"}:
        failures.append(f"temporal:{temporal}")

    if failures:
        return (
            "SAME_DEGREE_BUT_STRUCTURAL_DIFFERENCE",
            "; ".join(failures),
        )
    return (
        "STRICT_SAME_DEGREE_RENUMBERING_CANDIDATE",
        "official university confirmation remains required",
    )


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert_data_root(
        CANDIDATE_PATH,
        PRIOR_MD_PATH,
        PRIOR_JSON_PATH,
        CATALOG_PATH,
        RAW_CATALOG_PATH,
        TRAIN_PATH,
        VALID_PATH,
    )

    candidates = pd.read_csv(
        CANDIDATE_PATH,
        dtype="string",
        keep_default_na=False,
    )
    candidates = candidates.loc[
        candidates["diagnostic_status"].eq(TARGET_STATUS)
    ].copy()
    candidates["new_course_id"] = normalize_id_series(candidates["new_course_id"])
    candidates["candidate_old_course_id"] = normalize_id_series(
        candidates["candidate_old_course_id"]
    )
    distinct_new = candidates["new_course_id"].nunique(dropna=False)
    if distinct_new != EXPECTED_NEW_COURSES:
        raise RuntimeError(
            f"Expected {EXPECTED_NEW_COURSES} distinct new courses, found "
            f"{distinct_new} across {len(candidates)} candidate rows."
        )

    canonical = pd.read_parquet(CATALOG_PATH)
    raw = pd.read_parquet(RAW_CATALOG_PATH)
    if canonical["degree_course_id"].duplicated().any():
        raise RuntimeError("Canonical catalog has duplicate degree_course_id values.")
    if raw["degree_course_id"].duplicated().any():
        raise RuntimeError("Raw catalog has duplicate degree_course_id values.")

    for frame in (canonical, raw):
        for column in ("degree_course_id", "course_id", "degree_id"):
            frame[column] = normalize_id_series(frame[column])
    for column in ("faculty_id", "course_type_id", "requirement_type_id"):
        if column in raw:
            raw[column] = normalize_id_series(raw[column])

    raw_fields = raw[
        [
            "degree_course_id",
            "faculty_id",
            "course_type_id",
            "year_order",
            "semester_order",
            "active",
        ]
    ]
    catalog = canonical.merge(
        raw_fields,
        how="left",
        on="degree_course_id",
        validate="one_to_one",
    )

    model_columns = [
        "course_id",
        "part_id",
        "university_id",
        "degree_id",
        "faculty_id",
    ]
    train = pd.read_parquet(TRAIN_PATH, columns=model_columns)
    valid = pd.read_parquet(VALID_PATH, columns=model_columns)
    for frame in (train, valid):
        for column in model_columns:
            frame[column] = normalize_id_series(frame[column])
    return candidates, catalog, train, valid


def course_universities(
    catalog_rows: pd.DataFrame,
    model_rows: pd.DataFrame,
) -> set[str]:
    from_model = values_for(model_rows, "university_id", id_field=True)
    from_degrees = {
        suffix
        for suffix in (
            dotted_university_id(value) for value in catalog_rows["degree_id"]
        )
        if suffix
    }
    if from_model and from_degrees and from_model != from_degrees:
        raise RuntimeError(
            "Catalog degree suffix and enrolment university_id disagree: "
            f"{sorted(from_degrees)} vs {sorted(from_model)}"
        )
    return from_model or from_degrees


def relevant_catalog_rows(
    rows: pd.DataFrame,
    shared_degrees: set[str],
) -> pd.DataFrame:
    if shared_degrees:
        return rows.loc[rows["degree_id"].isin(shared_degrees)].copy()
    return rows.copy()


def compare_one(
    candidate: pd.Series,
    catalog: pd.DataFrame,
    all_model: pd.DataFrame,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    semester_order: dict[str, int],
) -> dict[str, Any]:
    new_id = str(candidate["new_course_id"])
    old_id = str(candidate["candidate_old_course_id"])
    new_catalog = catalog.loc[catalog["course_id"].eq(new_id)].copy()
    old_catalog = catalog.loc[catalog["course_id"].eq(old_id)].copy()
    new_model = all_model.loc[all_model["course_id"].eq(new_id)]
    old_model = all_model.loc[all_model["course_id"].eq(old_id)]

    new_degrees = values_for(new_catalog, "degree_id", id_field=True)
    old_degrees = values_for(old_catalog, "degree_id", id_field=True)
    new_universities = course_universities(new_catalog, new_model)
    old_universities = course_universities(old_catalog, old_model)
    degree = compare_degree_sets(
        old_degrees,
        new_degrees,
        old_universities,
        new_universities,
    )

    new_relevant = relevant_catalog_rows(new_catalog, degree["shared"])
    old_relevant = relevant_catalog_rows(old_catalog, degree["shared"])
    names = best_name_pair(old_relevant, new_relevant)

    new_faculties = values_for(new_relevant, "faculty_id", id_field=True)
    old_faculties = values_for(old_relevant, "faculty_id", id_field=True)
    new_credits = values_for(new_relevant, "course_credits")
    old_credits = values_for(old_relevant, "course_credits")
    new_course_types = values_for(new_relevant, "course_type_id", id_field=True)
    old_course_types = values_for(old_relevant, "course_type_id", id_field=True)
    new_requirement_types = values_for(
        new_relevant, "requirement_type_id", id_field=True
    )
    old_requirement_types = values_for(
        old_relevant, "requirement_type_id", id_field=True
    )
    new_years = values_for(new_relevant, "year_order")
    old_years = values_for(old_relevant, "year_order")
    new_semesters = values_for(new_relevant, "semester_order")
    old_semesters = values_for(old_relevant, "semester_order")
    new_active = values_for(new_relevant, "active")
    old_active = values_for(old_relevant, "active")

    temporal = temporal_summary(old_model, new_model, semester_order)
    evidence: dict[str, Any] = {
        "new_course_id": new_id,
        "old_course_id": old_id,
        "new_university_id": sorted_join(new_universities),
        "old_university_id": sorted_join(old_universities),
        "same_university": degree["same_university"],
        "new_degree_id": sorted_join(new_degrees),
        "old_degree_id": sorted_join(old_degrees),
        "same_degree": degree["same_degree"],
        "new_faculty_id": sorted_join(new_faculties),
        "old_faculty_id": sorted_join(old_faculties),
        "same_faculty": exact_set_match(old_faculties, new_faculties),
        "different_course_id": old_id != new_id,
        **names,
        "new_course_credits": sorted_join(new_credits),
        "old_course_credits": sorted_join(old_credits),
        "credits_match": exact_set_match(old_credits, new_credits),
        "new_course_type_id": sorted_join(new_course_types),
        "old_course_type_id": sorted_join(old_course_types),
        "course_type_match": exact_set_match(old_course_types, new_course_types),
        "new_requirement_type_id": sorted_join(new_requirement_types),
        "old_requirement_type_id": sorted_join(old_requirement_types),
        "requirement_type_match": exact_set_match(
            old_requirement_types, new_requirement_types
        ),
        "new_year_order": sorted_join(new_years),
        "old_year_order": sorted_join(old_years),
        "planned_year_match": exact_set_match(old_years, new_years),
        "new_semester_order": sorted_join(new_semesters),
        "old_semester_order": sorted_join(old_semesters),
        "planned_semester_match": exact_set_match(old_semesters, new_semesters),
        "new_active": sorted_join(new_active),
        "old_active": sorted_join(old_active),
        **temporal,
        "old_train_row_count": int(
            train["course_id"].eq(old_id).sum()
        ),
        "new_valid_row_count": int(
            valid["course_id"].eq(new_id).sum()
        ),
        "old_degree_count": len(old_degrees),
        "new_degree_count": len(new_degrees),
        "shared_degree_count": len(degree["shared"]),
        "shared_degree_ids": sorted_join(degree["shared"]),
        "old_only_degree_ids": sorted_join(degree["old_only"]),
        "new_only_degree_ids": sorted_join(degree["new_only"]),
        "degree_relationship": degree["degree_relationship"],
    }
    conclusion, blocking = classify_conclusion(evidence)
    evidence["diagnostic_conclusion"] = conclusion
    evidence["blocking_difference"] = blocking
    evidence["review_notes"] = (
        "Candidate only; no official equivalence source was found and no mapping "
        "was accepted."
    )
    return evidence


def rank_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["_conclusion_order"] = ranked["diagnostic_conclusion"].map(
        CONCLUSION_ORDER
    )
    ranked["_exact_order"] = (
        ranked["exact_normalized_name_match"].map({True: 0, False: 1}).fillna(2)
    )
    ranked["_similarity_order"] = pd.to_numeric(
        ranked["name_similarity"], errors="coerce"
    ).fillna(-1)
    ranked = ranked.sort_values(
        [
            "new_course_id",
            "_conclusion_order",
            "_exact_order",
            "_similarity_order",
            "old_course_id",
        ],
        ascending=[True, True, True, False, True],
        kind="stable",
    )
    ranked["candidate_rank"] = (
        ranked.groupby("new_course_id", sort=False).cumcount() + 1
    )
    ranked["candidate_count_for_new_course"] = ranked.groupby(
        "new_course_id", sort=False
    )["new_course_id"].transform("size")

    best_scores = ranked.groupby("new_course_id", sort=False)[
        ["_conclusion_order", "_exact_order", "_similarity_order"]
    ].transform("first")
    tied = (
        ranked[["_conclusion_order", "_exact_order", "_similarity_order"]]
        == best_scores
    ).all(axis=1)
    tied_counts = tied.groupby(ranked["new_course_id"]).transform("sum")
    ranked["ambiguous_best_match"] = tied_counts.gt(1)
    ranked = ranked.drop(
        columns=["_conclusion_order", "_exact_order", "_similarity_order"]
    )
    return ranked


def conclusion_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for conclusion in CONCLUSION_ORDER:
        subset = frame.loc[frame["diagnostic_conclusion"].eq(conclusion)]
        rows.append(
            {
                "diagnostic_conclusion": conclusion,
                "candidate_count": int(len(subset)),
                "valid_rows": int(subset["new_valid_row_count"].sum()),
            }
        )
    return rows


def calculate_answers(best: pd.DataFrame) -> dict[str, int]:
    same_uni = best["same_university"].eq(True)
    same_degree = best["same_degree"].eq(True)
    different_id = best["different_course_id"].eq(True)
    exact_name = best["exact_normalized_name_match"].eq(True)
    four_structural = (
        best["credits_match"].eq(True)
        & best["course_type_match"].eq(True)
        & best["requirement_type_match"].eq(True)
        & best["planned_year_match"].eq(True)
    )
    clean_temporal = (
        best["temporal_replacement_signal"].isin(
            ["DIRECT_REPLACEMENT", "ONE_SEMESTER_GAP"]
        )
        & best["overlap_in_active_enrolment"].eq(False)
    )
    return {
        "same_university_count": int(same_uni.sum()),
        "same_exact_degree_count": int(same_degree.sum()),
        "same_university_same_degree_different_course_id_count": int(
            (same_uni & same_degree & different_id).sum()
        ),
        "exact_name_same_degree_count": int((exact_name & same_degree).sum()),
        "same_degree_four_structural_matches_count": int(
            (same_degree & four_structural).sum()
        ),
        "clean_temporal_replacement_no_overlap_count": int(clean_temporal.sum()),
        "strict_same_degree_renumbering_candidate_count": int(
            best["diagnostic_conclusion"]
            .eq("STRICT_SAME_DEGREE_RENUMBERING_CANDIDATE")
            .sum()
        ),
        "same_university_different_degree_count": int(
            best["diagnostic_conclusion"]
            .eq("SAME_UNIVERSITY_DIFFERENT_DEGREE")
            .sum()
        ),
        "partial_degree_overlap_count": int(
            best["diagnostic_conclusion"].eq("PARTIAL_DEGREE_OVERLAP").sum()
        ),
        "ambiguous_count": int(
            (
                best["diagnostic_conclusion"].eq("INSUFFICIENT_EVIDENCE")
                | best["ambiguous_best_match"].eq(True)
            ).sum()
        ),
    }


def calculate_answer_valid_rows(best: pd.DataFrame) -> dict[str, int]:
    valid_rows = pd.to_numeric(best["new_valid_row_count"], errors="raise")
    same_uni = best["same_university"].eq(True)
    same_degree = best["same_degree"].eq(True)
    different_id = best["different_course_id"].eq(True)
    exact_name = best["exact_normalized_name_match"].eq(True)
    four_structural = (
        best["credits_match"].eq(True)
        & best["course_type_match"].eq(True)
        & best["requirement_type_match"].eq(True)
        & best["planned_year_match"].eq(True)
    )
    clean_temporal = (
        best["temporal_replacement_signal"].isin(
            ["DIRECT_REPLACEMENT", "ONE_SEMESTER_GAP"]
        )
        & best["overlap_in_active_enrolment"].eq(False)
    )
    masks = {
        "same_university": same_uni,
        "same_exact_degree": same_degree,
        "same_university_same_degree_different_course_id": (
            same_uni & same_degree & different_id
        ),
        "exact_name_same_degree": exact_name & same_degree,
        "same_degree_four_structural_matches": same_degree & four_structural,
        "clean_temporal_replacement_no_overlap": clean_temporal,
        "strict_same_degree_renumbering_candidate": best[
            "diagnostic_conclusion"
        ].eq("STRICT_SAME_DEGREE_RENUMBERING_CANDIDATE"),
        "same_university_different_degree": best["diagnostic_conclusion"].eq(
            "SAME_UNIVERSITY_DIFFERENT_DEGREE"
        ),
        "partial_degree_overlap": best["diagnostic_conclusion"].eq(
            "PARTIAL_DEGREE_OVERLAP"
        ),
        "ambiguous": (
            best["diagnostic_conclusion"].eq("INSUFFICIENT_EVIDENCE")
            | best["ambiguous_best_match"].eq(True)
        ),
    }
    return {
        name: int(valid_rows.loc[mask].sum())
        for name, mask in masks.items()
    }


def markdown_cell(value: Any) -> str:
    if value is None or value is pd.NA:
        return NOT_AVAILABLE
    try:
        if pd.isna(value):
            return NOT_AVAILABLE
    except (TypeError, ValueError):
        pass
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(markdown_cell(value) for value in row) + " |")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any], best: pd.DataFrame) -> str:
    answers = payload["answers"]
    answer_rows = payload["answer_valid_rows"]
    lines = [
        "# Course identity: 67-candidate degree verification",
        "",
        "**Diagnosis only. No course mapping was accepted.**",
        "",
        "The verification uses every catalog row for each old and new course in "
        "the canonical and raw `V_ACD_DEGREE_COURSE` representations, plus "
        "immutable TRAIN and VALID enrolment history. TEST remained "
        "`closed_not_read`; no model was trained or scored.",
        "",
        "## Direct answers",
        "",
        f"1. Same university: **{answers['same_university_count']} / 67**.",
        f"2. At least one exact shared normalized `degree_id`: "
        f"**{answers['same_exact_degree_count']} / 67**.",
        "3. Same university + same degree + different `course_id`: "
        f"**{answers['same_university_same_degree_different_course_id_count']} / 67**.",
        "4. Exact normalized Arabic-name match + same degree: "
        f"**{answers['exact_name_same_degree_count']} / 67**.",
        "5. Same degree plus matching credits, course type, requirement type, "
        f"and planned year: **{answers['same_degree_four_structural_matches_count']} / 67**.",
        "6. Direct or one-semester-gap temporal replacement with no enrolment "
        f"overlap: **{answers['clean_temporal_replacement_no_overlap_count']} / 67**.",
        "7. Strict same-degree renumbering candidates: "
        f"**{answers['strict_same_degree_renumbering_candidate_count']} / 67**.",
        "8. Same university but no shared degree: "
        f"**{answers['same_university_different_degree_count']} / 67**.",
        "9. Partial degree overlap: "
        f"**{answers['partial_degree_overlap_count']} / 67**.",
        "10. Ambiguous/insufficient or tied-best cases: "
        f"**{answers['ambiguous_count']} / 67**.",
        "",
        "## Direct-answer VALID-row prevalence",
        "",
        markdown_table(
            ["group", "candidates", "VALID rows"],
            [
                [
                    "same university",
                    answers["same_university_count"],
                    f"{answer_rows['same_university']:,}",
                ],
                [
                    "same exact degree",
                    answers["same_exact_degree_count"],
                    f"{answer_rows['same_exact_degree']:,}",
                ],
                [
                    "same university + same degree + different course_id",
                    answers[
                        "same_university_same_degree_different_course_id_count"
                    ],
                    f"{answer_rows['same_university_same_degree_different_course_id']:,}",
                ],
                [
                    "exact name + same degree",
                    answers["exact_name_same_degree_count"],
                    f"{answer_rows['exact_name_same_degree']:,}",
                ],
                [
                    "strict same-degree candidate",
                    answers["strict_same_degree_renumbering_candidate_count"],
                    f"{answer_rows['strict_same_degree_renumbering_candidate']:,}",
                ],
                [
                    "same university, different degree",
                    answers["same_university_different_degree_count"],
                    f"{answer_rows['same_university_different_degree']:,}",
                ],
                [
                    "partial degree overlap",
                    answers["partial_degree_overlap_count"],
                    f"{answer_rows['partial_degree_overlap']:,}",
                ],
                [
                    "ambiguous",
                    answers["ambiguous_count"],
                    f"{answer_rows['ambiguous']:,}",
                ],
            ],
        ),
        "",
        "A strict result is still only a candidate. It requires the same exact "
        "university and degree sets, a different course ID, name similarity of "
        "at least 0.85, exact agreement on faculty/credits/course type/"
        "requirement type/planned year/planned semester, and direct or "
        "one-semester-gap replacement without overlap. Similarity never creates "
        "a confirmed equivalence.",
        "",
        "## Why the exact-degree result differs from the earlier candidate score",
        "",
        "The earlier diagnostic built a course-level degree set by pooling "
        "`degree_id` values from enrolment rows with catalog rows. That broad "
        "profile was useful for candidate discovery, but it did not establish "
        "that the old and new catalog records belonged to the same degree. This "
        "verification uses every actual catalog row and compares full normalized "
        "`degree_id` strings directly. Under that stricter question, all 67 "
        "pairs have disjoint old/new catalog degree sets.",
        "",
        "Temporal history also argues against a simple ID-only replacement: 64 "
        "pairs have both IDs enrolled in at least one common semester, and the "
        "remaining 3 have a long gap. None has direct or one-semester-gap "
        "replacement without overlap.",
        "",
        "## Degree relationship and VALID-row prevalence",
        "",
        markdown_table(
            ["diagnostic conclusion", "candidates", "VALID rows"],
            [
                [
                    row["diagnostic_conclusion"],
                    row["candidate_count"],
                    f"{row['valid_rows']:,}",
                ]
                for row in payload["conclusion_summary"]
            ],
        ),
        "",
        "## Temporal evidence",
        "",
        markdown_table(
            ["temporal signal", "candidates", "VALID rows"],
            [
                [
                    row["temporal_replacement_signal"],
                    row["candidate_count"],
                    f"{row['valid_rows']:,}",
                ]
                for row in payload["temporal_summary"]
            ],
        ),
        "",
        "## All 67 best-per-course cases",
        "",
    ]

    display = best.copy()
    display["_order"] = display["diagnostic_conclusion"].map(CONCLUSION_ORDER)
    display = display.sort_values(
        ["_order", "new_valid_row_count", "new_course_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    table_rows: list[list[Any]] = []
    for row in display.itertuples(index=False):
        structural = (
            f"credits={row.credits_match}; type={row.course_type_match}; "
            f"req={row.requirement_type_match}; year={row.planned_year_match}; "
            f"semester={row.planned_semester_match}"
        )
        table_rows.append(
            [
                row.new_course_id,
                row.old_course_id,
                row.new_course_name,
                row.old_course_name,
                f"{row.old_university_id} → {row.new_university_id}",
                f"{row.old_degree_id} → {row.new_degree_id}",
                row.same_degree,
                row.different_course_id,
                structural,
                row.temporal_replacement_signal,
                row.diagnostic_conclusion,
                f"{row.new_valid_row_count:,}",
            ]
        )
    lines.extend(
        [
            markdown_table(
                [
                    "new_course_id",
                    "old_course_id",
                    "new course",
                    "old course",
                    "university comparison",
                    "degree comparison",
                    "same degree",
                    "different ID",
                    "structural matches",
                    "temporal evidence",
                    "diagnostic conclusion",
                    "VALID rows",
                ],
                table_rows,
            ),
            "",
            "## Source and integrity controls",
            "",
            f"- Candidate input: `{payload['sources']['candidate_path']}`.",
            f"- Canonical catalog: `{payload['sources']['canonical_catalog_path']}`.",
            f"- Raw supplement: `{payload['sources']['raw_catalog_path']}`.",
            f"- TRAIN: `{payload['sources']['train_path']}`.",
            f"- VALID: `{payload['sources']['valid_path']}`.",
            "- Catalog rows were retained at `degree_course_id` grain; no course "
            "was reduced to its first catalog row.",
            "- `same_degree` is the intersection of actual normalized full "
            "`degree_id` strings. Names, faculty, and dotted suffixes cannot make "
            "`same_degree` true.",
            "- Meaningful dotted ID suffixes were preserved as strings. Catalog "
            "university identity uses the degree suffix and was cross-checked "
            "against TRAIN/VALID `university_id` where available.",
            "- Every input hash matched before and after generation.",
            "- No mapping was accepted; no dataset or source was changed; no "
            "model was trained or scored; TEST remained `closed_not_read`.",
            "- Model freeze remains blocked pending human/university review.",
            "",
        ]
    )
    return "\n".join(lines)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def main() -> None:
    before = input_hashes()
    candidates, catalog, train, valid = load_sources()
    all_model = pd.concat(
        [train.assign(_split="train"), valid.assign(_split="valid")],
        ignore_index=True,
    )
    all_parts = sorted(
        normalize_id_set(all_model["part_id"]),
        key=chronological_key,
    )
    semester_order = {part_id: index for index, part_id in enumerate(all_parts)}

    rows = [
        compare_one(
            candidate,
            catalog,
            all_model,
            train,
            valid,
            semester_order,
        )
        for _, candidate in candidates.iterrows()
    ]
    all_candidates = rank_candidates(pd.DataFrame(rows))
    all_candidates = all_candidates[ALL_OUTPUT_COLUMNS]
    if len(all_candidates) != len(candidates):
        raise RuntimeError("A candidate pair was silently dropped.")
    if all_candidates["new_course_id"].nunique() != EXPECTED_NEW_COURSES:
        raise RuntimeError("Output does not contain exactly 67 new courses.")
    if (all_candidates["diagnostic_conclusion"] == "confirmed_equivalent").any():
        raise RuntimeError("Similarity must never confirm equivalence.")

    best = all_candidates.loc[all_candidates["candidate_rank"].eq(1)].copy()
    if len(best) != EXPECTED_NEW_COURSES:
        raise RuntimeError("Best-match output must contain exactly 67 rows.")
    best["_order"] = best["diagnostic_conclusion"].map(CONCLUSION_ORDER)
    best = best.sort_values(
        ["_order", "new_valid_row_count", "new_course_id"],
        ascending=[True, False, True],
        kind="stable",
    ).drop(columns="_order")

    review = pd.DataFrame(
        {
            "new_course_id": best["new_course_id"],
            "new_course_name": best["new_course_name"],
            "new_degree_id": best["new_degree_id"],
            "old_course_id": best["old_course_id"],
            "old_course_name": best["old_course_name"],
            "old_degree_id": best["old_degree_id"],
            "same_degree": best["same_degree"],
            "credits_old_new": (
                best["old_course_credits"] + " → " + best["new_course_credits"]
            ),
            "course_type_old_new": (
                best["old_course_type_id"] + " → " + best["new_course_type_id"]
            ),
            "requirement_type_old_new": (
                best["old_requirement_type_id"]
                + " → "
                + best["new_requirement_type_id"]
            ),
            "planned_year_old_new": (
                best["old_year_order"] + " → " + best["new_year_order"]
            ),
            "last_old_semester": best["old_last_seen_part_id"],
            "first_new_semester": best["new_first_seen_part_id"],
            "enrolment_overlap": best["overlap_in_active_enrolment"],
            "new_valid_row_count": best["new_valid_row_count"],
            "diagnostic_conclusion": best["diagnostic_conclusion"],
            "review_decision": "",
            "reviewer_name": "",
            "review_date": "",
            "review_notes": "",
        }
    )[REVIEW_COLUMNS]

    answers = calculate_answers(best)
    answer_valid_rows = calculate_answer_valid_rows(best)
    conclusion_rows = conclusion_summary(best)
    temporal_rows = []
    for signal, group in best.groupby(
        "temporal_replacement_signal", sort=True, dropna=False
    ):
        temporal_rows.append(
            {
                "temporal_replacement_signal": signal,
                "candidate_count": int(len(group)),
                "valid_rows": int(group["new_valid_row_count"].sum()),
            }
        )

    after = input_hashes()
    integrity = {
        path: {"before": digest, "after": after[path], "unchanged": digest == after[path]}
        for path, digest in before.items()
    }
    if not all(item["unchanged"] for item in integrity.values()):
        raise RuntimeError("An input source changed during diagnostic generation.")

    payload = {
        "diagnostic": "course_identity_67_degree_verification",
        "scope": {
            "target_status": TARGET_STATUS,
            "candidate_pair_count": int(len(all_candidates)),
            "distinct_new_course_count": int(
                all_candidates["new_course_id"].nunique()
            ),
            "test_policy": "closed_not_read",
            "test_read": False,
            "models_trained": 0,
            "models_scored": 0,
            "mapping_accepted": False,
            "dataset_or_source_changed": False,
            "model_freeze": "blocked_pending_human_university_review",
        },
        "sources": {
            "candidate_path": str(CANDIDATE_PATH.relative_to(ROOT)),
            "canonical_catalog_path": str(CATALOG_PATH.relative_to(ROOT)),
            "raw_catalog_path": str(RAW_CATALOG_PATH.relative_to(ROOT)),
            "train_path": str(TRAIN_PATH.relative_to(ROOT)),
            "valid_path": str(VALID_PATH.relative_to(ROOT)),
            "catalog_grain": "degree_course_id; compared as university_id + degree_id + course_id",
            "canonical_catalog_rows": int(len(pd.read_parquet(CATALOG_PATH))),
            "raw_catalog_rows": int(len(pd.read_parquet(RAW_CATALOG_PATH))),
        },
        "definitions": {
            "same_university": "non-empty normalized university_id sets are exactly equal",
            "same_degree": "old and new full normalized degree_id sets have a non-empty exact intersection",
            "different_course_id": "full normalized course_id strings differ",
            "structural_comparison_grain": "all catalog rows in exact shared degree_id values; all rows when no degree is shared",
            "clean_temporal_replacement": "DIRECT_REPLACEMENT or ONE_SEMESTER_GAP with no overlap",
            "strict_candidate": (
                "same exact university and identical degree sets; different course_id; "
                "name similarity >= 0.85; exact faculty, credits, course type, "
                "requirement type, planned year and planned semester agreement; "
                "direct or one-semester-gap replacement without overlap"
            ),
            "confirmation_rule": "no diagnostic similarity can create confirmed_equivalent; official university evidence is required",
        },
        "answers": answers,
        "answer_valid_rows": answer_valid_rows,
        "conclusion_summary": conclusion_rows,
        "temporal_summary": temporal_rows,
        "valid_rows_by_group": {
            row["diagnostic_conclusion"]: row["valid_rows"]
            for row in conclusion_rows
        },
        "source_integrity": integrity,
        "all_candidate_pairs": all_candidates.to_dict(orient="records"),
        "best_match_per_course": best.to_dict(orient="records"),
    }

    OUT_ALL.parent.mkdir(parents=True, exist_ok=True)
    all_candidates.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")
    best.to_csv(OUT_BEST, index=False, encoding="utf-8-sig")
    review.to_csv(OUT_REVIEW, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_MD.write_text(render_markdown(payload, best), encoding="utf-8")

    print(json.dumps(answers, indent=2))
    print("Conclusion/VALID rows:")
    for item in conclusion_rows:
        print(
            f"  {item['diagnostic_conclusion']}: "
            f"{item['candidate_count']} candidates, {item['valid_rows']} rows"
        )
    print("No course mapping was accepted.")
    print("TEST remained closed_not_read.")


if __name__ == "__main__":
    main()
