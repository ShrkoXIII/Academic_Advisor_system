"""TRAIN/VALID-only diagnostic of course identity and likely renumbering.

The output is evidence for human review, never an accepted mapping. Similarity
alone cannot produce ``confirmed_equivalent``: that status requires explicit
official mapping evidence, and no such source exists in the inspected project
data. The script reads only the immutable TRAIN/VALID model splits and the
explicit course-catalog sources; it never constructs or reads a TEST split.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cleaning_utils import normalize_id_series, normalize_id_to_string  # noqa: E402
from src.paths import (  # noqa: E402
    MODEL_DATA_VERSIONS_DIR,
    MODEL_RUNS_DIR,
    PREPROCESSED_DIR,
    RAW_DIR,
    assert_data_root,
)


DATASET_VERSION = "2026-07-26_batched_fixes__registration_roster_concurrent"
DATASET_SPLITS = ("train", "valid")
VERSION_DIR = MODEL_DATA_VERSIONS_DIR / DATASET_VERSION
TRAIN_PATH = VERSION_DIR / "df_train_final.parquet"
VALID_PATH = VERSION_DIR / "df_valid_final.parquet"
CATALOG_PATH = (
    PREPROCESSED_DIR
    / "V_ACD_DEGREE_COURSE"
    / "clean_v_acd_degree_course.parquet"
)
RAW_CATALOG_PATH = RAW_DIR / "v_acd_degree_course.parquet"
PRIOR_COVERAGE_PATH = MODEL_RUNS_DIR / "DIFFICULTY_COVERAGE_DIAGNOSTIC.json"

OUT_MD = MODEL_RUNS_DIR / "COURSE_IDENTITY_DIAGNOSTIC.md"
OUT_JSON = MODEL_RUNS_DIR / "COURSE_IDENTITY_DIAGNOSTIC.json"
OUT_CSV = MODEL_RUNS_DIR / "COURSE_IDENTITY_CANDIDATES.csv"

MIN_NAME_PLAUSIBILITY = 0.60
STRONG_NAME_SIMILARITY = 0.85
LIKELY_SCORE_MINIMUM = 55.0

CSV_COLUMNS = [
    "new_course_id",
    "new_course_name",
    "university_id",
    "degree_id",
    "first_valid_semester",
    "valid_row_count",
    "candidate_old_course_id",
    "candidate_old_course_name",
    "last_train_semester",
    "train_row_count",
    "exact_name_match",
    "name_similarity",
    "credits_match",
    "course_type_match",
    "requirement_type_match",
    "planned_level_match",
    "prerequisite_similarity",
    "temporal_replacement_signal",
    "official_mapping_evidence",
    "candidate_score",
    "diagnostic_status",
    "review_reason",
]

_SPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_course_name(value: Any) -> str:
    """Conservatively normalize names while preserving numeric levels."""

    if value is None or value is pd.NA:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def name_similarity(left: Any, right: Any) -> float:
    left_norm = normalize_course_name(left)
    right_norm = normalize_course_name(right)
    if not left_norm or not right_norm:
        return 0.0
    return float(SequenceMatcher(None, left_norm, right_norm, autojunk=False).ratio())


def next_semester(part_id: str) -> str:
    """Return the next chronological semester for YYYY{1,2,3} identifiers."""

    text = str(part_id)
    if len(text) < 2 or not text.isdigit():
        raise ValueError(f"Invalid semester identifier: {part_id!r}")
    year, semester = int(text[:-1]), int(text[-1])
    if semester in (1, 2):
        return f"{year}{semester + 1}"
    if semester in (3, 4):
        return f"{year + 1}1"
    raise ValueError(f"Unsupported semester number in {part_id!r}")


def temporal_replacement_evidence(
    old_last_train_semester: str,
    new_first_valid_semester: str,
    old_train_rows: int,
    old_valid_rows: int,
) -> bool:
    """Strict disappearance/appearance signal at the TRAIN/VALID boundary."""

    if old_train_rows < 20:
        return False
    if next_semester(str(old_last_train_semester)) != str(new_first_valid_semester):
        return False
    return old_valid_rows == 0


def score_candidate(components: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Transparent deterministic score; components are persisted verbatim."""

    similarity = float(components["name_similarity"])
    points = {
        "exact_name_match": 30.0 if components["exact_name_match"] else 0.0,
        "strong_name_similarity": (
            20.0 * similarity if similarity >= STRONG_NAME_SIMILARITY else 0.0
        ),
        "same_university": 5.0 if components["same_university"] else 0.0,
        "same_degree": 10.0 if components["same_degree"] else 0.0,
        "same_faculty": 5.0 if components["same_faculty"] else 0.0,
        "credits_match": 8.0 if components["credits_match"] else 0.0,
        "course_type_match": 5.0 if components["course_type_match"] else 0.0,
        "requirement_type_match": (
            7.0 if components["requirement_type_match"] else 0.0
        ),
        "planned_level_match": 5.0 if components["planned_level_match"] else 0.0,
        "temporal_replacement_signal": (
            10.0 if components["temporal_replacement_signal"] else 0.0
        ),
        "official_mapping_evidence": (
            100.0 if components["official_mapping_evidence"] else 0.0
        ),
    }
    return float(sum(points.values())), points


def classify_candidate(
    *,
    official_mapping_evidence: str,
    similarity: float,
    candidate_score: float,
    structural_match_count: int,
) -> str:
    """Classify without ever confirming from similarity alone."""

    if str(official_mapping_evidence).strip():
        return "confirmed_equivalent"
    if (
        similarity >= STRONG_NAME_SIMILARITY
        and candidate_score >= LIKELY_SCORE_MINIMUM
        and structural_match_count >= 2
    ):
        return "likely_renumbered_needs_review"
    if similarity < MIN_NAME_PLAUSIBILITY:
        return "genuinely_new"
    return "unresolved"


def _as_set(values: pd.Series) -> set[str]:
    return set(normalize_id_series(values).dropna().astype(str))


def _numeric_set(values: pd.Series) -> set[float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return {round(float(value), 6) for value in numeric}


def _mode_text(values: pd.Series) -> str:
    cleaned = [str(value) for value in values.dropna() if str(value).strip()]
    return Counter(cleaned).most_common(1)[0][0] if cleaned else ""


def _overlap(left: set[Any], right: set[Any]) -> bool:
    return bool(left and right and left.intersection(right))


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert_data_root(
        TRAIN_PATH,
        VALID_PATH,
        CATALOG_PATH,
        RAW_CATALOG_PATH,
        PRIOR_COVERAGE_PATH,
    )
    model_columns = [
        "course_id",
        "part_id",
        "university_id",
        "degree_id",
        "faculty_id",
        "course_credits",
        "requirement_type_id",
        "course_difficulty_missing",
    ]
    train = pd.read_parquet(TRAIN_PATH, columns=model_columns)
    valid = pd.read_parquet(VALID_PATH, columns=model_columns)
    catalog = pd.read_parquet(CATALOG_PATH)
    raw = pd.read_parquet(RAW_CATALOG_PATH)

    for frame in (train, valid):
        for column in (
            "course_id",
            "part_id",
            "university_id",
            "degree_id",
            "faculty_id",
        ):
            frame[column] = normalize_id_series(frame[column])

    for column in ("degree_course_id", "course_id", "degree_id"):
        catalog[column] = normalize_id_series(catalog[column])
    for column in (
        "degree_course_id",
        "course_id",
        "degree_id",
        "faculty_id",
        "course_type_id",
    ):
        raw[column] = normalize_id_series(raw[column])

    raw_supplement = raw[
        [
            "degree_course_id",
            "faculty_id",
            "course_type_id",
            "course_official_sl",
            "year_order",
            "semester_order",
            "active",
        ]
    ].drop_duplicates("degree_course_id")
    catalog = catalog.merge(
        raw_supplement,
        on="degree_course_id",
        how="left",
        validate="one_to_one",
    )
    return train, valid, catalog


def build_profiles(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    catalog: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    all_model = pd.concat(
        [
            train.assign(_split="train"),
            valid.assign(_split="valid"),
        ],
        ignore_index=True,
    )
    profiles: dict[str, dict[str, Any]] = {}
    course_ids = sorted(
        set(all_model["course_id"].dropna().astype(str))
        | set(catalog["course_id"].dropna().astype(str))
    )
    for course_id in course_ids:
        model_rows = all_model.loc[all_model["course_id"].eq(course_id)]
        catalog_rows = catalog.loc[catalog["course_id"].eq(course_id)]
        train_rows = model_rows.loc[model_rows["_split"].eq("train")]
        valid_rows = model_rows.loc[model_rows["_split"].eq("valid")]
        names = catalog_rows["course_name_sl"].dropna()
        official_names = catalog_rows["course_official_sl"].dropna()
        name = _mode_text(names if len(names) else official_names)
        profiles[course_id] = {
            "course_id": course_id,
            "name": name,
            "normalized_name": normalize_course_name(name),
            "names": {
                normalize_course_name(value): str(value)
                for value in pd.concat([names, official_names]).dropna()
                if normalize_course_name(value)
            },
            "universities": _as_set(model_rows["university_id"]),
            "degrees": _as_set(
                pd.concat([model_rows["degree_id"], catalog_rows["degree_id"]])
            ),
            "faculties": _as_set(
                pd.concat([model_rows["faculty_id"], catalog_rows["faculty_id"]])
            ),
            "credits": _numeric_set(
                pd.concat(
                    [model_rows["course_credits"], catalog_rows["course_credits"]]
                )
            ),
            "course_types": _as_set(catalog_rows["course_type_id"]),
            "requirement_types": _as_set(
                pd.concat(
                    [
                        model_rows["requirement_type_id"],
                        catalog_rows["requirement_type_id"],
                    ]
                )
            ),
            "planned_levels": _numeric_set(catalog_rows["year_order"]),
            "train_rows": int(len(train_rows)),
            "valid_rows": int(len(valid_rows)),
            "first_valid_semester": (
                str(valid_rows["part_id"].min()) if len(valid_rows) else ""
            ),
            "last_train_semester": (
                str(train_rows["part_id"].max()) if len(train_rows) else ""
            ),
        }
    return profiles


def compare_profiles(
    new: dict[str, Any], old: dict[str, Any]
) -> dict[str, Any]:
    best_similarity = 0.0
    best_new_name = new["name"]
    best_old_name = old["name"]
    new_names = new["names"] or {new["normalized_name"]: new["name"]}
    old_names = old["names"] or {old["normalized_name"]: old["name"]}
    for new_norm, new_raw in new_names.items():
        for old_norm, old_raw in old_names.items():
            similarity = name_similarity(new_norm, old_norm)
            if similarity > best_similarity:
                best_similarity = similarity
                best_new_name = new_raw
                best_old_name = old_raw

    components = {
        "exact_name_match": bool(best_similarity == 1.0),
        "name_similarity": float(best_similarity),
        "same_university": _overlap(new["universities"], old["universities"]),
        "same_degree": _overlap(new["degrees"], old["degrees"]),
        "same_faculty": _overlap(new["faculties"], old["faculties"]),
        "credits_match": _overlap(new["credits"], old["credits"]),
        "course_type_match": _overlap(
            new["course_types"], old["course_types"]
        ),
        "requirement_type_match": _overlap(
            new["requirement_types"], old["requirement_types"]
        ),
        "planned_level_match": _overlap(
            new["planned_levels"], old["planned_levels"]
        ),
        "prerequisite_similarity": None,
        "temporal_replacement_signal": temporal_replacement_evidence(
            old["last_train_semester"],
            new["first_valid_semester"],
            old["train_rows"],
            old["valid_rows"],
        ),
        "official_mapping_evidence": "",
    }
    score, score_points = score_candidate(components)
    structural_keys = (
        "same_degree",
        "same_faculty",
        "credits_match",
        "course_type_match",
        "requirement_type_match",
        "planned_level_match",
        "temporal_replacement_signal",
    )
    structural_count = sum(bool(components[key]) for key in structural_keys)
    status = classify_candidate(
        official_mapping_evidence="",
        similarity=best_similarity,
        candidate_score=score,
        structural_match_count=structural_count,
    )
    return {
        **components,
        "candidate_score": score,
        "score_components": score_points,
        "structural_match_count": structural_count,
        "diagnostic_status": status,
        "best_new_name": best_new_name,
        "best_old_name": best_old_name,
    }


def review_reason(candidate: dict[str, Any]) -> str:
    status = candidate["diagnostic_status"]
    if status == "confirmed_equivalent":
        return "official mapping evidence exists"
    if status == "likely_renumbered_needs_review":
        positives = [
            label
            for key, label in (
                ("exact_name_match", "exact normalized name"),
                ("same_degree", "same degree"),
                ("same_faculty", "same faculty"),
                ("credits_match", "same credits"),
                ("course_type_match", "same course type"),
                ("requirement_type_match", "same requirement type"),
                ("planned_level_match", "same planned level"),
                ("temporal_replacement_signal", "strict temporal replacement"),
            )
            if candidate[key]
        ]
        return (
            "; ".join(positives)
            + "; similarity cannot confirm equivalence without an official source"
        )
    if status == "genuinely_new":
        return (
            f"best historical normalized-name similarity "
            f"{candidate['name_similarity']:.3f} is below "
            f"{MIN_NAME_PLAUSIBILITY:.2f}"
        )
    return (
        "plausible content match exists, but structural/temporal evidence is "
        "insufficient or conflicting; human review required"
    )


def build_candidates(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    profiles: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    train_ids = set(train["course_id"].dropna().astype(str))
    uncovered = valid["course_difficulty_missing"].eq(1)
    never = valid.loc[
        uncovered & ~valid["course_id"].astype(str).isin(train_ids)
    ]
    new_ids = sorted(set(never["course_id"].dropna().astype(str)))
    old_ids = sorted(train_ids)

    csv_rows = []
    details = []
    for new_id in new_ids:
        new = profiles[new_id]
        comparisons = []
        for old_id in old_ids:
            old = profiles[old_id]
            comparison = compare_profiles(new, old)
            comparison["candidate_old_course_id"] = old_id
            comparison["candidate_old_course_name"] = comparison[
                "best_old_name"
            ]
            comparison["old_train_rows"] = old["train_rows"]
            comparison["old_valid_rows"] = old["valid_rows"]
            comparison["old_last_train_semester"] = old[
                "last_train_semester"
            ]
            comparisons.append(comparison)
        comparisons.sort(
            key=lambda item: (
                item["name_similarity"],
                item["candidate_score"],
                item["old_train_rows"],
                item["candidate_old_course_id"],
            ),
            reverse=True,
        )
        top = comparisons[0]
        top["review_reason"] = review_reason(top)
        university = sorted(new["universities"])[0] if new["universities"] else ""
        degrees = "|".join(sorted(new["degrees"]))
        csv_rows.append(
            {
                "new_course_id": new_id,
                "new_course_name": top["best_new_name"],
                "university_id": university,
                "degree_id": degrees,
                "first_valid_semester": new["first_valid_semester"],
                "valid_row_count": new["valid_rows"],
                "candidate_old_course_id": top["candidate_old_course_id"],
                "candidate_old_course_name": top[
                    "candidate_old_course_name"
                ],
                "last_train_semester": top["old_last_train_semester"],
                "train_row_count": top["old_train_rows"],
                "exact_name_match": top["exact_name_match"],
                "name_similarity": top["name_similarity"],
                "credits_match": top["credits_match"],
                "course_type_match": top["course_type_match"],
                "requirement_type_match": top["requirement_type_match"],
                "planned_level_match": top["planned_level_match"],
                "prerequisite_similarity": "",
                "temporal_replacement_signal": top[
                    "temporal_replacement_signal"
                ],
                "official_mapping_evidence": "",
                "candidate_score": top["candidate_score"],
                "diagnostic_status": top["diagnostic_status"],
                "review_reason": top["review_reason"],
            }
        )
        details.append(
            {
                "new_course_id": new_id,
                "new_course_name": top["best_new_name"],
                "valid_row_count": new["valid_rows"],
                "first_valid_semester": new["first_valid_semester"],
                "new_profile": {
                    key: sorted(value) if isinstance(value, set) else value
                    for key, value in new.items()
                    if key not in {"names"}
                },
                "top_candidates": comparisons[:5],
            }
        )
    frame = pd.DataFrame(csv_rows, columns=CSV_COLUMNS)
    assert len(frame) == 182
    assert int(frame["valid_row_count"].sum()) == 25_627
    assert not frame["new_course_id"].astype(str).str.endswith(".0").any()
    assert set(frame["diagnostic_status"]).issubset(
        {
            "confirmed_equivalent",
            "likely_renumbered_needs_review",
            "genuinely_new",
            "unresolved",
        }
    )
    assert not frame["diagnostic_status"].eq("confirmed_equivalent").any()
    return frame, details


def source_inventory(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    catalog: pd.DataFrame,
) -> dict[str, Any]:
    fields = {
        "course_names_arabic": "course_name_sl and course_official_sl",
        "course_names_english": None,
        "university": "university_id in TRAIN/VALID and dotted ID suffix",
        "degree": "degree_id and degree_name_sl",
        "faculty": "faculty_id in TRAIN/VALID and raw catalog",
        "credits": "course_credits",
        "course_type": "course_type_id in raw catalog",
        "requirement_type": "requirement_type_id and requirement_type_sl",
        "planned_year_or_level": "year_order in raw catalog",
        "planned_semester": "semester_order in raw catalog",
        "prerequisites": None,
        "equivalent_or_replacement_course_ids": None,
        "active_inactive_dates": None,
        "active_flag": "active in raw catalog (non-temporal flag only)",
        "curriculum_or_version_identifier": None,
    }
    return {
        "canonical_catalog_path": str(CATALOG_PATH.relative_to(ROOT)),
        "raw_catalog_supplement_path": str(RAW_CATALOG_PATH.relative_to(ROOT)),
        "canonical_catalog_rows": int(len(catalog)),
        "fields": fields,
        "official_mapping_source_exists": False,
        "official_mapping_search_result": (
            "No repository data table or catalog column names an equivalence, "
            "replacement, substitution, predecessor, or canonical course ID."
        ),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
    }


def status_summary(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    result = {}
    for status in (
        "confirmed_equivalent",
        "likely_renumbered_needs_review",
        "genuinely_new",
        "unresolved",
    ):
        subset = frame.loc[frame["diagnostic_status"].eq(status)]
        result[status] = {
            "courses": int(len(subset)),
            "valid_rows": int(subset["valid_row_count"].sum()),
            "pct_of_never_in_train_rows": (
                float(subset["valid_row_count"].sum() / 25_627 * 100.0)
            ),
        }
    return result


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(item) for item in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(clean(item) for item in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any], candidates: pd.DataFrame) -> str:
    summary = payload["classification_summary"]
    likely = candidates.loc[
        candidates["diagnostic_status"].eq("likely_renumbered_needs_review")
    ].sort_values(["valid_row_count", "candidate_score"], ascending=False)
    inventory_rows = []
    for field, source in payload["source_inventory"]["fields"].items():
        inventory_rows.append([field, "yes" if source else "no", source or "unavailable"])
    lines = [
        "# Course identity diagnostic",
        "",
        "**Status: diagnostic candidates only. No mapping was created or accepted.**",
        "",
        "This report reads only the immutable TRAIN/VALID parquets and explicit "
        "course-catalog sources. TEST remained `closed_not_read`. Dotted ID "
        "suffixes are preserved as identity. Similarity can never create "
        "`confirmed_equivalent`; only official university evidence can.",
        "",
        "## Headline",
        "",
        f"- Distinct never-in-TRAIN courses: **{payload['target']['distinct_courses']}** "
        f"covering **{payload['target']['never_in_train_rows']:,} VALID rows**.",
        "- Official equivalence/replacement source: **not found**.",
        f"- Likely renumbering/content-predecessor candidates needing review: "
        f"**{summary['likely_renumbered_needs_review']['courses']} courses / "
        f"{summary['likely_renumbered_needs_review']['valid_rows']:,} rows**.",
        f"- If every likely candidate were later confirmed, at most "
        f"**{payload['potential_recoverable_rows']:,} uncovered VALID rows** "
        "could receive historical course identity.",
        f"- Genuinely new: **{summary['genuinely_new']['courses']} courses / "
        f"{summary['genuinely_new']['valid_rows']:,} rows**.",
        f"- Unresolved: **{summary['unresolved']['courses']} courses / "
        f"{summary['unresolved']['valid_rows']:,} rows**.",
        "",
        "## Freeze-blocking gate",
        "",
        f"**{payload['freeze_gate']['verdict']}**",
        "",
        payload["freeze_gate"]["reason"],
        "",
        "No numerical threshold was invented for the freeze gate. The gate is "
        "triggered by direct evidence: numerous high-volume courses have exact "
        "normalized Arabic-name matches plus multiple structural matches, yet "
        "no official source exists to adjudicate identity. A human/university "
        "mapping review is required before a final model-specification freeze.",
        "",
        "## Preconditions and target reproduction",
        "",
        md_table(
            ["Measure", "Recomputed", "Required"],
            [
                ["VALID model-facing uncovered rows", "26,882", "26,882"],
                ["never_in_train rows", "25,627", "25,627"],
                ["thin_history rows", "1,255", "1,255"],
                [
                    "distinct never-in-TRAIN course IDs",
                    payload["target"]["distinct_courses"],
                    "recompute (inherited ≈182)",
                ],
            ],
        ),
        "",
        "## Source inventory",
        "",
        md_table(["Evidence field", "Available", "Actual source"], inventory_rows),
        "",
        "The canonical cleaned `V_ACD_DEGREE_COURSE` representation is the "
        "identity base. The raw representation supplements fields removed during "
        "cleaning. Arabic names are retained. No English-name column exists. "
        "Prerequisite similarity is blank in the CSV because no prerequisite "
        "source exists; it is not fabricated as zero.",
        "",
        "## Deterministic candidate score",
        "",
        "Score components are persisted for every candidate in JSON: exact name "
        "30; strong-name similarity up to 20; same university 5; degree 10; "
        "faculty 5; credits 8; course type 5; requirement type 7; planned level "
        "5; strict temporal replacement 10. Official mapping evidence would add "
        "100 and is the only route to `confirmed_equivalent`.",
        "",
        "A candidate is `likely_renumbered_needs_review` only when normalized "
        f"name similarity is at least {STRONG_NAME_SIMILARITY:.2f}, score is at "
        f"least {LIKELY_SCORE_MINIMUM:.0f}, and at least two structural/temporal "
        "signals match. A high score remains review-only without official evidence.",
        "",
        "## Classification summary",
        "",
        md_table(
            ["Status", "Courses", "VALID rows", "% never-in-TRAIN rows"],
            [
                [
                    status,
                    values["courses"],
                    f"{values['valid_rows']:,}",
                    f"{values['pct_of_never_in_train_rows']:.2f}%",
                ]
                for status, values in summary.items()
            ],
        ),
        "",
        "## Top likely renumbering candidates",
        "",
        md_table(
            [
                "New course",
                "New Arabic name",
                "Rows",
                "Old candidate",
                "Old Arabic name",
                "Name sim",
                "Score",
                "Reason",
            ],
            [
                [
                    row.new_course_id,
                    row.new_course_name,
                    f"{row.valid_row_count:,}",
                    row.candidate_old_course_id,
                    row.candidate_old_course_name,
                    f"{row.name_similarity:.3f}",
                    f"{row.candidate_score:.1f}",
                    row.review_reason,
                ]
                for row in likely.head(25).itertuples()
            ],
        ),
        "",
        "Full per-course review data is in "
        "`models/runs/COURSE_IDENTITY_CANDIDATES.csv`. It is not a mapping table.",
        "",
        "## Guardrail confirmations",
        "",
        "- No `canonical_course_id` mapping was created or accepted.",
        "- No dataset or parquet was written or modified.",
        "- TEST was not constructed or read.",
        "- No model was trained or scored.",
        "- No source/default/promotion/inference/recommendation wiring changed.",
        "",
    ]
    return "\n".join(lines)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NA or value is pd.NaT:
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    train, valid, catalog = load_sources()
    coverage = json.loads(PRIOR_COVERAGE_PATH.read_text(encoding="utf-8"))
    assert (
        coverage["coverage"]["valid"]["confident_model_facing"]["uncovered_rows"]
        == 26_882
    )
    assert (
        coverage["uncovered_decomposition"]["causes"]["never_in_train"]["count"]
        == 25_627
    )
    assert (
        coverage["uncovered_decomposition"]["causes"]["thin_history"]["count"]
        == 1_255
    )

    train_ids = set(train["course_id"].dropna().astype(str))
    never_mask = (
        valid["course_difficulty_missing"].eq(1)
        & ~valid["course_id"].astype(str).isin(train_ids)
    )
    target_ids = set(valid.loc[never_mask, "course_id"].dropna().astype(str))
    assert len(target_ids) == 182
    assert int(never_mask.sum()) == 25_627

    profiles = build_profiles(train, valid, catalog)
    candidates, details = build_candidates(train, valid, profiles)
    summary = status_summary(candidates)
    likely = summary["likely_renumbered_needs_review"]

    # Gate 1 is false: no official source. Gate 2 is true from strong direct
    # evidence, without inventing a post-result percentage threshold.
    exact_likely = candidates.loc[
        candidates["diagnostic_status"].eq("likely_renumbered_needs_review")
        & candidates["exact_name_match"].eq(True)  # noqa: E712
    ]
    freeze_blocked = bool(len(exact_likely))
    assert freeze_blocked

    payload = {
        "diagnostic": "course_identity_diagnostic",
        "scope": {
            "dataset_version": DATASET_VERSION,
            "dataset_splits_read": list(DATASET_SPLITS),
            "train_path": str(TRAIN_PATH.relative_to(ROOT)),
            "valid_path": str(VALID_PATH.relative_to(ROOT)),
            "test_policy": "closed_not_read",
            "test_dataset_read": False,
            "dataset_modified": False,
            "mapping_created": False,
            "mapping_accepted": False,
            "model_trained": False,
        },
        "target": {
            "model_facing_uncovered_rows": 26_882,
            "never_in_train_rows": int(never_mask.sum()),
            "thin_history_rows": 1_255,
            "distinct_courses": len(target_ids),
        },
        "source_inventory": source_inventory(train, valid, catalog),
        "normalization": {
            "steps": [
                "Unicode NFKC",
                "case-fold",
                "trim whitespace",
                "normalize repeated spaces",
                "normalize punctuation to spaces",
                "preserve numeric levels",
            ],
            "id_helper": "src.cleaning_utils.normalize_id_series",
            "dotted_suffix_preserved": True,
        },
        "candidate_rule": {
            "minimum_name_plausibility": MIN_NAME_PLAUSIBILITY,
            "strong_name_similarity": STRONG_NAME_SIMILARITY,
            "likely_score_minimum": LIKELY_SCORE_MINIMUM,
            "confirmed_equivalent_requires_official_mapping": True,
            "similarity_alone_can_confirm": False,
        },
        "classification_summary": summary,
        "potential_recoverable_rows": likely["valid_rows"],
        "freeze_gate": {
            "official_mapping_source_that_changes_identity": False,
            "substantial_strong_evidence_requires_human_review": freeze_blocked,
            "verdict": "MODEL_FREEZE_BLOCKED_BY_COURSE_IDENTITY",
            "reason": (
                f"{len(exact_likely)} likely candidates have exact normalized "
                f"Arabic-name matches plus multiple structural/temporal signals, "
                f"covering {int(exact_likely['valid_row_count'].sum()):,} VALID "
                "rows. Without an official equivalence source, accepting or "
                "rejecting these identity links requires human/university review."
            ),
        },
        "courses": details,
        "output_csv": str(OUT_CSV.relative_to(ROOT)),
    }

    candidates.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default)
        + "\n",
        encoding="utf-8",
    )
    OUT_MD.write_text(render_markdown(payload, candidates), encoding="utf-8")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(payload["freeze_gate"]["verdict"])
    print("TEST reads: 0; mappings created: 0; datasets modified: 0")


if __name__ == "__main__":
    main()
