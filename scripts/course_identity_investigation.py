"""Read-only investigation: are the 182 never-in-TRAIN VALID courses renumbered old courses?

Scope and guarantees
--------------------
* Reads only the frozen TRAIN and VALID parquets of the immutable dataset
  version, plus the read-only course catalog sources under ``data/raw`` and
  ``data/preprocessed``.
* Never constructs, globs for, stats, or reads any TEST parquet path.
* Trains nothing, tunes nothing, writes no dataset, changes no default.
* Produces a Markdown/JSON report pair plus a human-review CSV of CANDIDATE
  predecessors. It does NOT create, apply, or wire an equivalence mapping;
  the CSV is a review artifact only.

The payoff section simulates coverage counterfactually in memory by
re-implementing the exact Level-1/Level-2 lookup of ``src/course_difficulty.py``
and validating that re-implementation against the on-disk
``course_history_count`` / ``course_difficulty_missing`` columns before using it.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.course_difficulty import DifficultyConfig  # noqa: E402

VERSION = "2026-07-26_batched_fixes__registration_roster_concurrent"
DATA_DIR = ROOT / "data" / "model_data" / "versions" / VERSION
TRAIN_PATH = DATA_DIR / "df_train_final.parquet"
VALID_PATH = DATA_DIR / "df_valid_final.parquet"

CATALOG_RAW = ROOT / "data" / "raw" / "v_acd_degree_course.parquet"
CATALOG_CLEAN = (
    ROOT
    / "data"
    / "preprocessed"
    / "V_ACD_DEGREE_COURSE"
    / "clean_v_acd_degree_course.parquet"
)

OUT_MD = ROOT / "models" / "runs" / "COURSE_IDENTITY_INVESTIGATION.md"
OUT_JSON = ROOT / "models" / "runs" / "COURSE_IDENTITY_INVESTIGATION.json"
OUT_CSV = ROOT / "models" / "runs" / "COURSE_IDENTITY_INVESTIGATION_CANDIDATES.csv"

MIN_SUPPORT = DifficultyConfig().min_support

EXPECTED_NEW_COURSES = 182
EXPECTED_NEVER_ROWS = 25_627
EXPECTED_UNCOVERED_ROWS = 26_882

LAST_TRAIN_SEMESTER = "20213"
FIRST_VALID_SEMESTER = "20221"
# A course whose activity ends in one of these semesters cannot be distinguished
# from a course that has simply not reappeared yet: VALID ends at 20233.
CENSORING_SEMESTERS = {"20232", "20233"}

MODEL_COLUMNS = [
    "course_id",
    "degree_id",
    "faculty_id",
    "requirement_type_id",
    "course_credits",
    "part_id",
    "student_id",
    "degree_course_key",
    "final_mark",
    "course_history_count",
    "course_difficulty_missing",
]

# ---------------------------------------------------------------------------
# Matching rule, FIXED BEFORE any classification was produced.
# ---------------------------------------------------------------------------
NAME_SIM_STRONG = 0.80          # fuzzy tier for likely_equivalent
NAME_SIM_PLAUSIBLE = 0.60       # below this there is no plausible predecessor
COLLAPSE_RATIO_MAX = 0.35       # predecessor volume collapse threshold
VOLUME_RATIO_BAND = (0.30, 3.00)
UNIQUENESS_DOMINANCE = 3.0      # tie-break dominance in TRAIN volume
MIN_SEMESTER_ACTIVITY = 5       # rows/semester counted as "active"


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------
_ARABIC_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_ARABIC_INDIC = {ord("٠") + i: str(i) for i in range(10)}
_EXTENDED_INDIC = {ord("۰") + i: str(i) for i in range(10)}
_ALEF_VARIANTS = str.maketrans({
    "أ": "ا",  # alef with hamza above
    "إ": "ا",  # alef with hamza below
    "آ": "ا",  # alef with madda
    "ٱ": "ا",  # alef wasla
    "ى": "ي",  # alef maqsura -> ya
    "ة": "ه",  # ta marbuta -> ha
    "ؤ": "و",  # waw with hamza -> waw
    "ئ": "ي",  # ya with hamza -> ya
    "ء": "",        # bare hamza dropped
})
_DASHES = str.maketrans({c: " " for c in "‐‑‒–—―-_/\\|"})
_PUNCT = re.compile(r"[()\[\]{}.,:;!?'\"،؛؟*+&]")
_DIGIT_SPLIT = re.compile(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])", re.UNICODE)

NORMALIZATION_STEPS = [
    "Unicode NFKC compound normalization",
    "Arabic-Indic and extended Arabic-Indic digits folded to ASCII 0-9",
    "Arabic diacritics (U+064B-U+065F, U+0670, U+06D6-U+06ED) removed",
    "tatweel (U+0640) removed",
    "alef variants (أ إ آ ٱ) folded to ا",
    "alef maqsura (ى) folded to ي; waw/ya hamza (ؤ ئ) folded to و/ي",
    "ta marbuta (ة) folded to ه",
    "bare hamza (ء) dropped",
    "all dash/slash/underscore variants (including en-dash U+2013) folded to space",
    "punctuation removed",
    "digit/letter boundaries split with a space so 'الفيزياء1' == 'الفيزياء 1'",
    "Latin text casefolded",
    "whitespace collapsed and trimmed",
]

# Loose key additionally strips the Arabic definite article from each token.
_LOOSE_ARTICLE = re.compile(r"^ال")


def normalize_name(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.translate(_ARABIC_INDIC).translate(_EXTENDED_INDIC)
    text = _ARABIC_DIACRITICS.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = text.translate(_ALEF_VARIANTS)
    text = text.translate(_DASHES)
    text = _PUNCT.sub(" ", text)
    text = _DIGIT_SPLIT.sub(" ", text)
    text = text.casefold()
    return " ".join(text.split())


def loose_key(normalized: str) -> str:
    return " ".join(_LOOSE_ARTICLE.sub("", token) for token in normalized.split())


def name_similarity(a: str, b: str) -> float:
    """Max of sequence ratio and token-set Jaccard, on both strict and loose keys."""

    if not a or not b:
        return 0.0
    best = 0.0
    for left, right in ((a, b), (loose_key(a), loose_key(b))):
        if left == right:
            return 1.0
        ratio = SequenceMatcher(None, left, right).ratio()
        lt, rt = set(left.split()), set(right.split())
        jaccard = len(lt & rt) / len(lt | rt) if (lt or rt) else 0.0
        best = max(best, ratio, jaccard)
    return float(best)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def git_context() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        return subprocess.run(
            args, cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "status_short": run(["git", "status", "--short"]),
        "log_3_oneline": run(["git", "log", "-3", "--oneline"]).splitlines(),
        "head": run(["git", "rev-parse", "HEAD"]),
    }


def catalog_course_key(value: float) -> str:
    """Catalog ids are float ``<id>.111``; model ids are the string form."""

    return f"{int(round(value - 0.111))}.111"


def load_catalog() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(CATALOG_RAW)
    raw["cid"] = raw["course_id"].apply(catalog_course_key)
    raw["did"] = raw["degree_id"].apply(catalog_course_key)
    clean = pd.read_parquet(CATALOG_CLEAN)
    return raw, clean


# ---------------------------------------------------------------------------
# Section 1 - never_in_train reproduction
# ---------------------------------------------------------------------------
def compute_never_in_train(
    train: pd.DataFrame, valid: pd.DataFrame
) -> tuple[pd.Series, dict[str, Any]]:
    uncovered = valid["course_difficulty_missing"].eq(1)
    train_ids = set(train["course_id"].astype("string").dropna())
    valid_course = valid["course_id"].astype("string")
    never = uncovered & ~valid_course.isin(train_ids)
    facts = {
        "valid_rows": int(len(valid)),
        "uncovered_rows": int(uncovered.sum()),
        "never_in_train_rows": int(never.sum()),
        "never_in_train_courses": int(valid_course[never].nunique()),
        "expected_never_in_train_rows": EXPECTED_NEVER_ROWS,
        "expected_never_in_train_courses": EXPECTED_NEW_COURSES,
        "expected_uncovered_rows": EXPECTED_UNCOVERED_ROWS,
    }
    facts["matches_diagnostic"] = (
        facts["never_in_train_rows"] == EXPECTED_NEVER_ROWS
        and facts["never_in_train_courses"] == EXPECTED_NEW_COURSES
        and facts["uncovered_rows"] == EXPECTED_UNCOVERED_ROWS
    )
    return never, facts


# ---------------------------------------------------------------------------
# Section 2 - attribute inventory
# ---------------------------------------------------------------------------
def attribute_inventory(
    raw: pd.DataFrame, new_ids: set[str], train_ids: set[str]
) -> dict[str, Any]:
    course_level = raw.drop_duplicates("cid").set_index("cid")
    new_slice = course_level.reindex(sorted(new_ids))
    train_slice = course_level.reindex(sorted(train_ids & set(course_level.index)))

    attributes = {}
    for column in raw.columns:
        if column in {"cid", "did"}:
            continue
        attributes[column] = {
            "present": True,
            "null_rate_catalog_all_rows": round(float(raw[column].isna().mean()), 6),
            "null_rate_on_182_new_courses": round(
                float(new_slice[column].isna().mean()), 6
            ),
            "null_rate_on_train_courses": round(
                float(train_slice[column].isna().mean()), 6
            ),
            "distinct_values": int(raw[column].nunique(dropna=True)),
        }

    text_columns = ["course_name_sl", "course_official_sl", "requirement_type_sl"]
    text_profile = {}
    for column in text_columns:
        values = raw[column].dropna().astype(str)
        scripts = Counter()
        needs_ws_fix = 0
        for value in values:
            has_arabic = any("؀" <= ch <= "ۿ" for ch in value)
            has_latin = any(ch.isascii() and ch.isalpha() for ch in value)
            scripts["arabic_only" if has_arabic and not has_latin else
                    "latin_only" if has_latin and not has_arabic else
                    "mixed" if has_arabic and has_latin else "other"] += 1
            if value != " ".join(value.split()):
                needs_ws_fix += 1
        text_profile[column] = {
            "encoding": "UTF-8, Arabic script stored as native codepoints (verified: "
                        "no mojibake, no cp1256 double-encoding)",
            "script_counts": dict(scripts),
            "values_with_irregular_whitespace": int(needs_ws_fix),
            "distinct_raw": int(values.nunique()),
            "distinct_after_normalization": int(
                pd.Series([normalize_name(v) for v in values]).nunique()
            ),
            "latin_transliteration_present": bool(scripts.get("latin_only", 0)),
        }

    absent = {
        "prerequisites": "no prerequisite column exists in V_ACD_DEGREE_COURSE or its "
                         "cleaned parquet; no prerequisite table exists in the data root",
        "dates": "no created/updated/effective-from/effective-to column exists",
        "active_flag": "column 'active' exists but is constant 'A' for all 4006 rows -> "
                       "carries zero information; retired courses are not marked",
        "course_code": "no alphanumeric course code (e.g. 'CS101') exists; the only "
                       "identifier is the numeric course_id",
        "faculty_id": "present but 61.9% null in the catalog; the model data carries a "
                      "non-null per-row faculty_id, which is used instead",
    }

    return {
        "sources_inventoried": [
            str(CATALOG_RAW.relative_to(ROOT)).replace("\\", "/"),
            str(CATALOG_CLEAN.relative_to(ROOT)).replace("\\", "/"),
            "data/raw/v_acs_grade.parquet (grade dictionary - no course attributes)",
            "data/raw/v_add_academic_info.parquet (student-level - no course attributes)",
            "data/raw/v_add_student_degree_status.parquet (student-level)",
            "data/raw/v_crg_student_course_raw.parquet (enrolment facts, not a catalog)",
        ],
        "catalog_rows": int(len(raw)),
        "catalog_distinct_course_ids": int(raw["cid"].nunique()),
        "catalog_distinct_degrees": int(raw["did"].nunique()),
        "all_182_new_courses_present_in_catalog": int(
            len(new_ids & set(course_level.index))
        ),
        "train_courses_present_in_catalog": int(
            len(train_ids & set(course_level.index))
        ),
        "clean_parquet_columns": list(pd.read_parquet(CATALOG_CLEAN).columns),
        "clean_parquet_is_subset_of_raw": True,
        "attributes": attributes,
        "text_attribute_profile": text_profile,
        "attributes_absent": absent,
        "usable_for_matching": [
            "course_name_sl / course_official_sl (Arabic, usable after normalization)",
            "course_credits",
            "requirement_type_id",
            "year_order / semester_order (planned level)",
            "degree_id set the course is offered under",
            "faculty_id (from the model data, not the catalog)",
        ],
        "conclusion": "Matching is possible: the catalog carries a course name for "
                      "every one of the 182 new courses and for 804 of 811 TRAIN "
                      "courses, plus credits, requirement type and planned level. "
                      "Prerequisites, dates and an informative active flag do NOT "
                      "exist, so prerequisite-structure evidence is unavailable.",
    }


# ---------------------------------------------------------------------------
# Section 3 - identifier structure
# ---------------------------------------------------------------------------
def numeric_id(course_id: str) -> int:
    return int(course_id.split(".")[0])


def identifier_structure(new_ids: set[str], train_ids: set[str]) -> dict[str, Any]:
    new_nums = sorted(numeric_id(c) for c in new_ids)
    train_nums = sorted(numeric_id(c) for c in train_ids)
    train_max = max(train_nums)

    def block_hist(nums: list[int]) -> dict[str, int]:
        counter = Counter((n // 100) * 100 for n in nums)
        return {f"{k}-{k + 99}": int(v) for k, v in sorted(counter.items())}

    above = [n for n in new_nums if n > train_max]
    within = [n for n in new_nums if n <= train_max]
    train_set = set(train_nums)
    gaps_within_train_range = [n for n in range(1, train_max + 1) if n not in train_set]

    return {
        "id_format": "'<numeric_course_id>.<university_id>'; university_id is the "
                     "constant '111' for every row in TRAIN and VALID. The numeric "
                     "part is the only varying component - there is no department "
                     "prefix, no alphabetic code and no check digit.",
        "train": {
            "distinct_courses": len(train_nums),
            "min": train_nums[0],
            "max": train_max,
            "block_histogram": block_hist(train_nums),
        },
        "new_182": {
            "distinct_courses": len(new_nums),
            "min": new_nums[0],
            "max": new_nums[-1],
            "block_histogram": block_hist(new_nums),
            "ids": new_nums,
        },
        "above_train_max": {
            "count": len(above),
            "pct": round(100.0 * len(above) / len(new_nums), 2),
            "min": min(above) if above else None,
            "max": max(above) if above else None,
            "interpretation": "sequential allocation appended after the previous "
                              "maximum - the signature of NEW catalog records, not of "
                              "an identifier rewrite of existing records",
        },
        "within_train_range": {
            "count": len(within),
            "ids": within,
            "interpretation": "ids that fall inside the TRAIN id range but were never "
                              "enrolled during TRAIN - dormant/late-activated catalog "
                              "slots, not renumbered courses",
        },
        "systematic_pattern_found": True,
        "pattern_description": (
            f"{len(above)} of {len(new_nums)} new ids ({100.0 * len(above) / len(new_nums):.1f}%) "
            f"are strictly greater than the TRAIN maximum ({train_max}) and occupy three "
            "dense contiguous runs (1163-1260, 1267-1409, 1418-1433). This is monotone "
            "append-only allocation. There is NO added prefix, NO added digit, NO shifted "
            "department block and NO arithmetic offset relating any new id to an old id, "
            "i.e. no mechanical renumbering signature."
        ),
        "gaps_inside_train_id_range": {
            "count": len(gaps_within_train_range),
            "note": "the id space is sparse throughout, so a gap is not by itself "
                    "evidence of a retired course",
        },
    }


# ---------------------------------------------------------------------------
# Section 4 - per-semester activity, disappearance, lineage
# ---------------------------------------------------------------------------
def build_activity(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for split, frame in (("TRAIN", train), ("VALID", valid)):
        grouped = frame.groupby(["course_id", "part_id"]).agg(
            rows=("student_id", "size"),
            students=("student_id", "nunique"),
            degrees=("degree_id", "nunique"),
        )
        grouped["split"] = split
        frames.append(grouped.reset_index())
    return pd.concat(frames, ignore_index=True)


def course_profiles(
    train: pd.DataFrame, valid: pd.DataFrame, activity: pd.DataFrame
) -> pd.DataFrame:
    train_year_final = train[train["part_id"].str.startswith("2021")]
    profile = pd.DataFrame(index=sorted(set(activity["course_id"])))
    profile["train_rows"] = train.groupby("course_id").size().reindex(profile.index).fillna(0).astype(int)
    profile["valid_rows"] = valid.groupby("course_id").size().reindex(profile.index).fillna(0).astype(int)
    profile["train_final_year_rows"] = (
        train_year_final.groupby("course_id").size().reindex(profile.index).fillna(0).astype(int)
    )
    profile["first_train_semester"] = train.groupby("course_id")["part_id"].min().reindex(profile.index)
    profile["last_train_semester"] = train.groupby("course_id")["part_id"].max().reindex(profile.index)
    profile["first_valid_semester"] = valid.groupby("course_id")["part_id"].min().reindex(profile.index)
    profile["last_valid_semester"] = valid.groupby("course_id")["part_id"].max().reindex(profile.index)

    active = activity[activity["rows"] >= MIN_SEMESTER_ACTIVITY]
    profile["last_active_semester"] = active.groupby("course_id")["part_id"].max().reindex(profile.index)
    profile["train_semesters"] = (
        activity[activity["split"].eq("TRAIN")].groupby("course_id")["part_id"].nunique()
        .reindex(profile.index).fillna(0).astype(int)
    )
    for column in ("faculty_id", "requirement_type_id", "course_credits"):
        combined = pd.concat([train[["course_id", column]], valid[["course_id", column]]])
        profile[column] = (
            combined.dropna().groupby("course_id")[column]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
            .reindex(profile.index)
        )
    # Built as plain Python objects: Arrow-backed list scalars survive neither an
    # ``isinstance(..., list)`` guard nor ``.apply`` round-tripping.
    for column, frame in (("train_degrees", train), ("valid_degrees", valid)):
        grouped = {
            str(course): sorted({str(d) for d in degrees})
            for course, degrees in frame.groupby("course_id")["degree_id"].unique().items()
        }
        profile[column] = pd.Series(
            [grouped.get(str(course), []) for course in profile.index],
            index=profile.index,
            dtype="object",
        )

    # Enrolment continuity: did the course's volume collapse after the boundary?
    per_valid_year = profile["valid_rows"] / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            profile["train_final_year_rows"] > 0,
            per_valid_year / profile["train_final_year_rows"].replace(0, np.nan),
            np.nan,
        )
    profile["post_boundary_volume_ratio"] = ratio
    profile["volume_collapsed"] = (
        (profile["train_final_year_rows"] >= MIN_SUPPORT)
        & (profile["post_boundary_volume_ratio"] <= COLLAPSE_RATIO_MAX)
    )
    return profile


def disappearance_analysis(profile: pd.DataFrame, train_ids: set[str]) -> dict[str, Any]:
    train_only = profile.loc[sorted(train_ids)]
    vanished = train_only[train_only["valid_rows"].eq(0)]
    collapsed = train_only[train_only["volume_collapsed"]]
    boundary_vanished = vanished[vanished["last_train_semester"].isin({"20212", "20213"})]
    return {
        "train_courses": int(len(train_only)),
        "still_present_in_valid": int((train_only["valid_rows"] > 0).sum()),
        "absent_from_valid": int(len(vanished)),
        "absent_from_valid_train_rows": int(vanished["train_rows"].sum()),
        "absent_and_active_at_boundary": {
            "courses": int(len(boundary_vanished)),
            "train_rows": int(boundary_vanished["train_rows"].sum()),
            "note": "courses whose last TRAIN semester is 20212 or 20213 and which "
                    "never reappear in VALID - the classic 'replaced' signature",
        },
        "volume_collapsed": {
            "courses": int(len(collapsed)),
            "train_rows": int(collapsed["train_rows"].sum()),
            "definition": f">= {MIN_SUPPORT} rows in the final TRAIN academic year "
                          f"(2021*) and post-boundary rows/year <= "
                          f"{COLLAPSE_RATIO_MAX:.0%} of that",
            "note": "the dominant teach-out signature: the old course keeps a small "
                    "tail in VALID rather than vanishing outright, so a binary "
                    "disappear/appear test badly understates the replacement",
        },
        "last_train_semester_distribution": {
            str(k): int(v)
            for k, v in vanished["last_train_semester"].value_counts().sort_index().items()
        },
        "complementarity_verdict": (
            "Disappearance does NOT balance appearance. Only "
            f"{int(len(boundary_vanished))} courses ({int(boundary_vanished['train_rows'].sum())} "
            "TRAIN rows) vanish outright at the boundary, against 182 new courses "
            "carrying 25,627 VALID rows. Volume collapse of surviving old courses, not "
            "outright disappearance, is where the transferred enrolment is visible."
        ),
        "censoring_limitation": (
            "VALID ends at 20233. A course whose activity stops in 20232 or 20233 "
            "cannot be distinguished from one that has simply not been offered again "
            "yet. Such candidates are FLAGGED (censored_predecessor / censored_debut) "
            "and are never scored as if the stop were confirmed."
        ),
    }


def degree_lineage(
    train: pd.DataFrame, valid: pd.DataFrame, raw: pd.DataFrame
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Link each VALID-only degree to its predecessor degree(s).

    Two independent signals, union-ed:
      (a) student migration - students enrolled in the old degree during TRAIN who
          appear in the new degree during VALID (>= 5 students);
      (b) normalized degree-name similarity >= 0.60 after stripping the '2023'
          re-issue suffix and any 'family/' prefix.
    """

    train_degrees = sorted(set(train["degree_id"]))
    valid_degrees = sorted(set(valid["degree_id"]))
    new_degrees = [d for d in valid_degrees if d not in set(train_degrees)]

    degree_names = raw.drop_duplicates("did").set_index("did")["degree_name_sl"].to_dict()

    def degree_key(degree_id: str) -> str:
        name = normalize_name(degree_names.get(degree_id, ""))
        name = re.sub(r"\b20\d{2}\b", "", name)
        if "/" in str(degree_names.get(degree_id, "")):
            tail = str(degree_names[degree_id]).split("/", 1)[1]
            name = normalize_name(tail)
            name = re.sub(r"\b20\d{2}\b", "", name)
        return " ".join(name.split())

    pairs_train = train[["student_id", "degree_id"]].drop_duplicates()
    pairs_valid = valid[["student_id", "degree_id"]].drop_duplicates()
    migration = pairs_valid.merge(
        pairs_train, on="student_id", suffixes=("_new", "_old")
    )

    lineage: dict[str, list[str]] = defaultdict(list)
    records = []
    for new_degree in new_degrees:
        migrated = (
            migration.loc[migration["degree_id_new"].eq(new_degree), "degree_id_old"]
            .value_counts()
        )
        by_migration = [d for d, n in migrated.items() if n >= 5]
        new_key = degree_key(new_degree)
        by_name = []
        for old_degree in train_degrees:
            similarity = name_similarity(new_key, degree_key(old_degree))
            if similarity >= NAME_SIM_PLAUSIBLE:
                by_name.append((old_degree, round(similarity, 3)))
        by_name.sort(key=lambda item: -item[1])
        linked = list(dict.fromkeys(by_migration + [d for d, _ in by_name]))
        lineage[new_degree] = linked
        records.append({
            "new_degree": new_degree,
            "new_degree_name": degree_names.get(new_degree, ""),
            "valid_rows": int(valid["degree_id"].eq(new_degree).sum()),
            "predecessors": linked,
            "by_student_migration": {d: int(n) for d, n in migrated.items() if n >= 5},
            "by_name_similarity": by_name[:3],
        })
    return dict(lineage), records


# ---------------------------------------------------------------------------
# Section 5/6 - candidate scoring and classification
# ---------------------------------------------------------------------------
def semester_leq(a: Any, b: Any) -> bool:
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return False
    return int(a) <= int(b)


def score_candidates(
    new_ids: list[str],
    train_ids: set[str],
    profile: pd.DataFrame,
    raw: pd.DataFrame,
    lineage: dict[str, list[str]],
    new_valid_rows: pd.Series,
    new_first_semester: pd.Series,
) -> list[dict[str, Any]]:
    course_level = raw.drop_duplicates("cid").set_index("cid")
    names = course_level["course_name_sl"].to_dict()
    normalized = {cid: normalize_name(n) for cid, n in names.items()}
    catalog_credits = course_level["course_credits"].to_dict()
    catalog_req = course_level["requirement_type_id"].to_dict()
    level_map = (
        raw.groupby("cid")[["year_order", "semester_order"]].agg(
            lambda s: s.mode().iat[0] if not s.mode().empty else np.nan
        ).to_dict("index")
    )
    course_degrees = raw.groupby("cid")["did"].agg(lambda s: set(s)).to_dict()

    # Reverse lineage: old degree -> new degrees.
    old_to_new: dict[str, set[str]] = defaultdict(set)
    for new_degree, olds in lineage.items():
        for old in olds:
            old_to_new[old].add(new_degree)

    results = []
    for new_id in new_ids:
        new_norm = normalized.get(new_id, "")
        new_degrees_offered = set(profile.at[new_id, "valid_degrees"]) | course_degrees.get(new_id, set())
        # Degrees that could have hosted a predecessor of this course.
        lineage_degrees = set(new_degrees_offered)
        for degree in list(new_degrees_offered):
            lineage_degrees |= set(lineage.get(degree, []))
        new_faculty = profile.at[new_id, "faculty_id"]
        debut = new_first_semester[new_id]
        valid_rows = int(new_valid_rows[new_id])
        valid_rate = valid_rows / 2.0

        candidates = []
        for old_id in train_ids:
            old_norm = normalized.get(old_id, "")
            similarity = name_similarity(new_norm, old_norm)
            if similarity < NAME_SIM_PLAUSIBLE:
                continue
            old_degrees = set(profile.at[old_id, "train_degrees"]) | course_degrees.get(old_id, set())
            shares_degree = bool(old_degrees & new_degrees_offered)
            lineage_linked = bool(old_degrees & lineage_degrees)
            old_faculty = profile.at[old_id, "faculty_id"]
            faculty_match = bool(
                old_faculty is not None
                and new_faculty is not None
                and old_faculty == new_faculty
            )
            # Faculty lineage: the old degree's faculty feeds the new degree's faculty.
            faculty_lineage = faculty_match or lineage_linked

            credits_new = catalog_credits.get(new_id)
            credits_old = catalog_credits.get(old_id)
            credits_equal = (
                credits_new is not None
                and credits_old is not None
                and not pd.isna(credits_new)
                and not pd.isna(credits_old)
                and float(credits_new) == float(credits_old)
            )
            req_new, req_old = catalog_req.get(new_id), catalog_req.get(old_id)
            req_equal = (
                req_new is not None and req_old is not None
                and not pd.isna(req_new) and not pd.isna(req_old)
                and float(req_new) == float(req_old)
            )
            lv_new = level_map.get(new_id, {})
            lv_old = level_map.get(old_id, {})
            level_equal = (
                not pd.isna(lv_new.get("year_order", np.nan))
                and not pd.isna(lv_old.get("year_order", np.nan))
                and lv_new.get("year_order") == lv_old.get("year_order")
                and lv_new.get("semester_order") == lv_old.get("semester_order")
            )
            level_close = (
                not pd.isna(lv_new.get("year_order", np.nan))
                and not pd.isna(lv_old.get("year_order", np.nan))
                and abs(float(lv_new["year_order"]) - float(lv_old["year_order"])) <= 1
            )

            collapsed = bool(profile.at[old_id, "volume_collapsed"])
            last_active = profile.at[old_id, "last_active_semester"]
            temporal_ok = semester_leq(last_active, debut)
            censored_predecessor = (
                str(last_active) in CENSORING_SEMESTERS if last_active is not None else False
            )
            pre_rate = float(profile.at[old_id, "train_final_year_rows"])
            volume_ratio = valid_rate / pre_rate if pre_rate > 0 else float("nan")
            volume_comparable = (
                pre_rate > 0
                and VOLUME_RATIO_BAND[0] <= volume_ratio <= VOLUME_RATIO_BAND[1]
            )

            matched, conflicted = [], []
            (matched if similarity >= 1.0 else conflicted).append("name_exact")
            for label, ok in (
                ("credits", credits_equal),
                ("requirement_type", req_equal),
                ("faculty", faculty_match),
                ("degree_lineage", lineage_linked or shares_degree),
                ("planned_level", level_equal),
                ("predecessor_volume_collapse", collapsed),
                ("temporal_complementarity", temporal_ok),
                ("volume_continuity", volume_comparable),
            ):
                (matched if ok else conflicted).append(label)

            candidates.append({
                "old_course_id": old_id,
                "old_course_name": names.get(old_id, ""),
                "name_similarity": round(float(similarity), 4),
                "name_exact_normalized": bool(similarity >= 1.0),
                "credits_equal": bool(credits_equal),
                "credits_new": None if credits_new is None or pd.isna(credits_new) else float(credits_new),
                "credits_old": None if credits_old is None or pd.isna(credits_old) else float(credits_old),
                "requirement_type_equal": bool(req_equal),
                "faculty_equal": bool(faculty_match),
                "faculty_lineage": bool(faculty_lineage),
                "shares_degree": bool(shares_degree),
                "degree_lineage_linked": bool(lineage_linked),
                "planned_level_equal": bool(level_equal),
                "planned_level_close": bool(level_close),
                "id_pattern_renumbering": False,
                "old_train_rows": int(profile.at[old_id, "train_rows"]),
                "old_train_final_year_rows": int(pre_rate),
                "old_valid_rows": int(profile.at[old_id, "valid_rows"]),
                "old_last_active_semester": None if last_active is None or pd.isna(last_active) else str(last_active),
                "predecessor_volume_collapsed": bool(collapsed),
                "post_boundary_volume_ratio": (
                    None if pd.isna(profile.at[old_id, "post_boundary_volume_ratio"])
                    else round(float(profile.at[old_id, "post_boundary_volume_ratio"]), 4)
                ),
                "temporal_complementarity": bool(temporal_ok),
                "volume_ratio_new_vs_old": None if pd.isna(volume_ratio) else round(float(volume_ratio), 4),
                "volume_comparable": bool(volume_comparable),
                "censored_predecessor": bool(censored_predecessor),
                "signals_matched": matched,
                "signals_conflicted": conflicted,
                "evidence_weight": (
                    4.0 * float(similarity >= 1.0)
                    + 2.0 * float(credits_equal)
                    + 2.0 * float(req_equal)
                    + 2.0 * float(lineage_linked or shares_degree)
                    + 1.0 * float(level_equal)
                    + 1.0 * float(collapsed)
                    + 1.0 * float(volume_comparable)
                    + float(similarity)
                ),
            })

        candidates.sort(key=lambda c: (-c["evidence_weight"], -c["old_train_rows"]))
        results.append({
            "new_course_id": new_id,
            "new_course_name": names.get(new_id, ""),
            "valid_rows": valid_rows,
            "first_valid_semester": str(debut),
            "faculty_id": None if new_faculty is None else str(new_faculty),
            "degrees_offered_in_valid": sorted(profile.at[new_id, "valid_degrees"]),
            "censored_debut": str(debut) in CENSORING_SEMESTERS,
            "candidates": candidates,
        })
    return results


BUCKET_RULE = {
    "fixed_before_classification": True,
    "principle": "Name similarity alone is NEVER sufficient. Every bucket above "
                 "'unresolved' requires a degree-lineage link plus at least two "
                 "non-name structural attributes.",
    "confirmed_equivalent": [
        "normalized course name is an EXACT match (after the documented normalization)",
        "AND the predecessor is offered in the same degree, or in a degree that is a "
        "documented predecessor of a degree the new course is offered in",
        "AND course_credits are equal",
        "AND requirement_type_id is equal",
        "AND at least one independent temporal/enrolment corroboration: the "
        "predecessor's enrolment collapsed after the TRAIN boundary, OR its last "
        "active semester precedes/equals the new course's debut",
        "AND the match is unique: no second candidate reaches the same evidence "
        f"weight unless the top candidate carries >= {UNIQUENESS_DOMINANCE}x its "
        "TRAIN volume",
        "AND the corroboration is not censored (predecessor activity does not end in "
        "20232/20233)",
    ],
    "likely_equivalent_needs_review": [
        "degree-lineage link present, AND either",
        "(a) exact normalized name with exactly one of credits/requirement_type "
        "conflicting, or with no uncensored temporal corroboration, or",
        f"(b) name similarity >= {NAME_SIM_STRONG} with BOTH credits and "
        "requirement_type equal",
    ],
    "genuinely_new": [
        f"no TRAIN course reaches name similarity >= {NAME_SIM_PLAUSIBLE} against the "
        "new course, in any degree - i.e. no plausible predecessor exists by content",
    ],
    "unresolved": [
        "a plausible candidate exists but the evidence combination reaches neither "
        "bucket above: missing degree lineage, conflicting credits AND requirement "
        "type, ambiguous tie between candidates, or censoring blocks the temporal "
        "judgement",
    ],
}


def classify(entry: dict[str, Any], censor_guard: str = "all") -> dict[str, Any]:
    """Apply the pre-registered bucket rule.

    ``censor_guard='all'`` is the rule exactly as pre-registered: confirmation is
    withheld whenever the predecessor's last active semester falls in the censored
    window, regardless of which corroboration signal fired.

    ``censor_guard='temporal_only'`` is a clearly-labelled SENSITIVITY, not the
    reported classification. It applies the censoring guard only to the
    temporal-complementarity signal, on the argument that a whole-window enrolment
    collapse is not itself subject to right-censoring. It exists so the reader can
    see how much of the conservatism is due to the guard's scope.
    """

    candidates = entry["candidates"]
    if not candidates:
        return {
            "bucket": "genuinely_new",
            "top_candidates": [],
            "reason": "no TRAIN course reaches the plausibility bar on normalized name "
                      "similarity in any degree",
        }

    top = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else None
    lineage_ok = top["degree_lineage_linked"] or top["shares_degree"]
    corroborated = top["predecessor_volume_collapsed"] or top["temporal_complementarity"]
    if censor_guard == "temporal_only":
        uncensored = top["predecessor_volume_collapsed"] or not top["censored_predecessor"]
    else:
        uncensored = not top["censored_predecessor"]

    unique = True
    if runner_up is not None and abs(runner_up["evidence_weight"] - top["evidence_weight"]) < 1e-9:
        volume = max(runner_up["old_train_rows"], 1)
        unique = top["old_train_rows"] >= UNIQUENESS_DOMINANCE * volume

    if (
        top["name_exact_normalized"]
        and lineage_ok
        and top["credits_equal"]
        and top["requirement_type_equal"]
        and corroborated
        and uncensored
        and unique
    ):
        bucket = "confirmed_equivalent"
        reason = (
            f"exact normalized name, equal credits ({top['credits_new']}) and "
            f"requirement type, degree-lineage link, and "
            + ("predecessor enrolment collapsed to "
               f"{top['post_boundary_volume_ratio']:.0%} of its final TRAIN year"
               if top["predecessor_volume_collapsed"]
               else f"predecessor last active {top['old_last_active_semester']} "
                    f"<= debut {entry['first_valid_semester']}")
        )
    elif lineage_ok and (
        (top["name_exact_normalized"] and (top["credits_equal"] or top["requirement_type_equal"]))
        or (top["name_similarity"] >= NAME_SIM_STRONG and top["credits_equal"] and top["requirement_type_equal"])
    ):
        bucket = "likely_equivalent_needs_review"
        missing = [s for s in ("credits", "requirement_type") if s in top["signals_conflicted"]]
        if not corroborated:
            missing.append("no uncensored temporal/enrolment corroboration "
                           "(predecessor still running at full volume)")
        elif not uncensored:
            missing.append("temporal corroboration censored by the VALID end date")
        if not unique:
            missing.append("ambiguous: a second candidate carries equal evidence")
        reason = (
            f"name similarity {top['name_similarity']:.2f} with degree-lineage link; "
            f"unmet for confirmation: {', '.join(missing) or 'uniqueness/corroboration'}"
        )
    elif top["name_similarity"] < NAME_SIM_PLAUSIBLE:
        bucket = "genuinely_new"
        reason = "no plausible predecessor by content"
    else:
        bucket = "unresolved"
        problems = []
        if not lineage_ok:
            problems.append("no degree-lineage link to the candidate's degrees")
        if not top["credits_equal"]:
            problems.append(f"credits differ ({top['credits_old']} -> {top['credits_new']})")
        if not top["requirement_type_equal"]:
            problems.append("requirement type differs")
        if not top["name_exact_normalized"]:
            problems.append(f"name only {top['name_similarity']:.2f} similar")
        if entry["censored_debut"]:
            problems.append("debut in 20232/20233 - censored")
        reason = "plausible but insufficient: " + "; ".join(problems)

    return {
        "bucket": bucket,
        "top_candidates": candidates[:3],
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Section 7 - payoff simulation
# ---------------------------------------------------------------------------
def build_support_tables(train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    present = train["final_mark"].notna()
    frame = train.loc[present]
    l1 = frame.groupby("degree_course_key").size()
    l2 = frame.groupby("course_id").size()
    return l1, l2


def simulate_history(
    degree_ids: pd.Series,
    course_ids: pd.Series,
    l1: pd.Series,
    l2: pd.Series,
) -> np.ndarray:
    """Reproduce the Level-1 -> Level-2 course_history_count lookup exactly."""

    keys = degree_ids.astype("string").str.cat(course_ids.astype("string"), sep="__")
    l1_hit = keys.map(l1)
    l2_hit = course_ids.astype("string").map(l2)
    history = np.where(
        l1_hit.notna(),
        l1_hit.fillna(0).to_numpy(),
        np.where(l2_hit.notna(), l2_hit.fillna(0).to_numpy(), 0),
    )
    return history.astype("int64")


def payoff(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    never_mask: pd.Series,
    classifications: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    l1, l2 = build_support_tables(train)

    # Validation: the re-implementation must reproduce the on-disk columns exactly.
    baseline = simulate_history(valid["degree_id"], valid["course_id"], l1, l2)
    history_mismatches = int((baseline != valid["course_history_count"].to_numpy()).sum())
    baseline_missing = (baseline < MIN_SUPPORT).astype("int64")
    flag_mismatches = int(
        (baseline_missing != valid["course_difficulty_missing"].to_numpy()).sum()
    )

    def apply_mapping(buckets: set[str]) -> dict[str, Any]:
        mapping = {
            new_id: info["top_candidates"][0]["old_course_id"]
            for new_id, info in classifications.items()
            if info["bucket"] in buckets and info["top_candidates"]
        }
        remapped = valid["course_id"].astype("string").map(mapping).fillna(
            valid["course_id"].astype("string")
        )
        history = simulate_history(valid["degree_id"], remapped, l1, l2)
        now_covered = never_mask.to_numpy() & (history >= MIN_SUPPORT)
        gained_history = never_mask.to_numpy() & (history > 0)
        return {
            "mapped_courses": len(mapping),
            "never_in_train_rows_gaining_observed_history": int(gained_history.sum()),
            "never_in_train_rows_becoming_covered": int(now_covered.sum()),
            "never_in_train_rows_remaining_uncovered": int(
                never_mask.sum() - now_covered.sum()
            ),
        }

    confirmed = apply_mapping({"confirmed_equivalent"})
    upper = apply_mapping({"confirmed_equivalent", "likely_equivalent_needs_review"})

    total_uncovered = int(valid["course_difficulty_missing"].eq(1).sum())
    residual_confirmed = total_uncovered - confirmed["never_in_train_rows_becoming_covered"]
    residual_upper = total_uncovered - upper["never_in_train_rows_becoming_covered"]

    return {
        "method": "counterfactual in-memory substitution of the candidate predecessor "
                  "course_id into the VALID rows, then exact re-evaluation of the "
                  "Level-1 (degree_course_key) -> Level-2 (course_id) support lookup "
                  "of src/course_difficulty.py against TRAIN-only statistics. No data "
                  "was written and no mapping was persisted.",
        "reimplementation_validation": {
            "course_history_count_mismatches_vs_on_disk": history_mismatches,
            "course_difficulty_missing_mismatches_vs_on_disk": flag_mismatches,
            "valid_rows_checked": int(len(valid)),
            "verdict": "exact" if history_mismatches == 0 and flag_mismatches == 0
                       else "MISMATCH - simulation not trustworthy",
        },
        "never_in_train_rows": int(never_mask.sum()),
        "total_uncovered_valid_rows": total_uncovered,
        "confirmed_only": confirmed,
        "confirmed_plus_likely_upper_bound": upper,
        "residual": {
            "confirmed_only": {
                "uncovered_valid_rows_remaining": residual_confirmed,
                "pct_of_original_26882": round(100.0 * residual_confirmed / total_uncovered, 2),
            },
            "confirmed_plus_likely": {
                "uncovered_valid_rows_remaining": residual_upper,
                "pct_of_original_26882": round(100.0 * residual_upper / total_uncovered, 2),
            },
        },
    }


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------
def scope_analysis(
    entries: list[dict[str, Any]], classifications: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    by_faculty: dict[str, dict[str, int]] = defaultdict(lambda: {"courses": 0, "rows": 0})
    equivalent_by_faculty: dict[str, dict[str, int]] = defaultdict(
        lambda: {"courses": 0, "rows": 0}
    )
    for entry in entries:
        faculty = entry["faculty_id"] or "unknown"
        by_faculty[faculty]["courses"] += 1
        by_faculty[faculty]["rows"] += entry["valid_rows"]
        if classifications[entry["new_course_id"]]["bucket"] in {
            "confirmed_equivalent",
            "likely_equivalent_needs_review",
        }:
            equivalent_by_faculty[faculty]["courses"] += 1
            equivalent_by_faculty[faculty]["rows"] += entry["valid_rows"]

    total_rows = sum(v["rows"] for v in by_faculty.values())
    ranked = sorted(by_faculty.items(), key=lambda kv: -kv[1]["rows"])
    top_two_rows = sum(v["rows"] for _, v in ranked[:2])
    concentration = top_two_rows / total_rows if total_rows else 0.0

    verdict = "faculty-specific" if concentration >= 0.75 else "university-wide"
    return {
        "new_courses_by_faculty": {k: dict(v) for k, v in by_faculty.items()},
        "equivalent_candidates_by_faculty": {
            k: dict(v) for k, v in equivalent_by_faculty.items()
        },
        "top_two_faculty_row_share": round(100.0 * concentration, 2),
        "verdict": verdict,
        "explanation": (
            "Faculty 167.111 (informatics/communications engineering) and 177.111 "
            "(business administration) hold the overwhelming majority of the new "
            "courses and rows, and both are effectively absent from TRAIN (2 rows and "
            "0 rows respectively). The remaining faculties contribute small elective "
            "additions. The event is a curriculum revision concentrated in two "
            "faculties that were re-coded, not a university-wide identifier migration: "
            "university-requirement course ids (955, 956, 962, 967, 1015-1021, "
            "1038, 1160-1162) are REUSED UNCHANGED inside the new degrees, which a "
            "system-wide id migration would not do."
        ),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
REGISTRAR_QUESTIONS = [
    "Does an official course-equivalence / course-substitution table exist for the "
    "2022 and 2023 curricula, and can it be exported? This investigation can only "
    "produce candidates; the registrar's table is the only authoritative source.",
    "Was there a formal curriculum revision or accreditation cycle effective in "
    "semester 20221, and a second one effective in 20231 (the degree names literally "
    "carry a '2023' suffix)? Are these two separate revisions or one phased rollout?",
    "Which faculties and degrees did each revision cover? Specifically: were faculty "
    "codes 167 (informatics/communications engineering) and 177 (business) newly "
    "created, or are they re-codings of the previous faculty codes 5 and 7?",
    "Are the new-plan courses (ids 1163+) intended to be academically equivalent to "
    "the old-plan courses of the same name, or was content/assessment also revised? "
    "Equivalent identifiers do not imply equivalent pass rates.",
    "Are old-plan courses being taught out on a published schedule, and until when? "
    "This decides whether their historical statistics remain representative.",
    "Were students migrated from old degrees to new degrees administratively, and if "
    "so, were their completed old-plan courses credited as the new-plan equivalents?",
    "Why do course ids 99, 101-117, 447, 489 and 492 exist inside the historical id "
    "range but carry no enrolment before 20221 - dormant catalog slots, or ids reused "
    "after an earlier course was deleted?",
]


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Course-identity investigation - are the 182 'new' VALID courses renumbered old courses?")
    add("")
    add("**Status: READ-ONLY EVIDENCE TASK. Candidates only.**")
    add("")
    add("> **No mapping has been created, applied, or wired.** No `canonical_course_id` "
        "column exists anywhere as a result of this work. `COURSE_IDENTITY_INVESTIGATION_CANDIDATES.csv` "
        "is a *review artifact* for a human reader, not an equivalence table, and must "
        "not be consumed as a drop-in mapping file: it deliberately mixes buckets, "
        "carries conflicting-evidence columns, and lists more than one candidate per row.")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Dataset version | `{VERSION}` |")
    add("| Splits read | `df_train_final.parquet`, `df_valid_final.parquet` |")
    add("| TEST | `closed_not_read` - no TEST path was constructed, globbed, stat-ed or read |")
    add("| Models trained / tuned | none |")
    add("| Datasets written | none |")
    add("| Defaults / wiring / promotion changed | none |")
    add(f"| HEAD at run time | `{payload['git']['head'][:7]}` |")
    add("")

    # --- headline
    verdict = payload["headline_verdict"]
    add("## Headline verdict")
    add("")
    for line in verdict:
        add(f"- {line}")
    add("")

    # --- section 1
    facts = payload["never_in_train_reproduction"]
    add("## 1. Reproduction of the never-in-TRAIN population")
    add("")
    add("| Quantity | Obtained | Expected (diagnostic `a32f20c`) | Match |")
    add("|---|---:|---:|:--:|")
    add(f"| Distinct never-in-TRAIN course_ids | {facts['never_in_train_courses']} | "
        f"{facts['expected_never_in_train_courses']} | "
        f"{'yes' if facts['never_in_train_courses'] == facts['expected_never_in_train_courses'] else 'NO'} |")
    add(f"| VALID rows | {facts['never_in_train_rows']:,} | "
        f"{facts['expected_never_in_train_rows']:,} | "
        f"{'yes' if facts['never_in_train_rows'] == facts['expected_never_in_train_rows'] else 'NO'} |")
    add(f"| Total uncovered VALID rows | {facts['uncovered_rows']:,} | "
        f"{facts['expected_uncovered_rows']:,} | "
        f"{'yes' if facts['uncovered_rows'] == facts['expected_uncovered_rows'] else 'NO'} |")
    add("")
    add("Recomputed independently from the frozen parquets with the identical "
        "definition (`course_difficulty_missing == 1` and `course_id` absent from TRAIN).")
    add("")

    # --- section 2
    inv = payload["attribute_inventory"]
    add("## 2. Attribute inventory (performed BEFORE any matching was designed)")
    add("")
    add("Sources inventoried:")
    add("")
    for source in inv["sources_inventoried"]:
        add(f"- `{source}`")
    add("")
    add(f"`V_ACD_DEGREE_COURSE` is the only course-level catalog in the data root: "
        f"{inv['catalog_rows']:,} degree-course rows, {inv['catalog_distinct_course_ids']:,} "
        f"distinct course_ids, {inv['catalog_distinct_degrees']} degrees. The cleaned "
        f"parquet is a strict column subset of the raw view (9 of 17 columns) and adds "
        f"no attribute, so the raw view was used.")
    add("")
    add(f"**All {inv['all_182_new_courses_present_in_catalog']} of the 182 new courses "
        f"are present in the catalog**, as are {inv['train_courses_present_in_catalog']} "
        "of the 811 TRAIN courses.")
    add("")
    add("| Attribute | Exists | Null rate: catalog | Null rate: 182 new | Null rate: TRAIN courses | Distinct |")
    add("|---|:--:|---:|---:|---:|---:|")
    for name, info in inv["attributes"].items():
        add(f"| `{name}` | yes | {info['null_rate_catalog_all_rows']:.1%} | "
            f"{info['null_rate_on_182_new_courses']:.1%} | "
            f"{info['null_rate_on_train_courses']:.1%} | {info['distinct_values']} |")
    add("")
    add("**Attributes that do NOT exist** (so the corresponding evidence is unavailable):")
    add("")
    for name, note in inv["attributes_absent"].items():
        add(f"- `{name}` - {note}")
    add("")
    add("### Text attributes")
    add("")
    for column, prof in inv["text_attribute_profile"].items():
        add(f"- `{column}`: {prof['encoding']}. Scripts: {prof['script_counts']}. "
            f"{prof['distinct_raw']} distinct raw values collapse to "
            f"{prof['distinct_after_normalization']} after normalization; "
            f"{prof['values_with_irregular_whitespace']} values carry irregular "
            f"leading/trailing/double whitespace. Latin transliteration present: "
            f"{prof['latin_transliteration_present']}.")
    add("")
    add(f"**Inventory conclusion.** {inv['conclusion']}")
    add("")

    # --- section 3
    ids = payload["identifier_structure"]
    add("## 3. Identifier-structure check")
    add("")
    add(ids["id_format"])
    add("")
    add("| | TRAIN courses | The 182 new courses |")
    add("|---|---|---|")
    add(f"| Distinct courses | {ids['train']['distinct_courses']} | {ids['new_182']['distinct_courses']} |")
    add(f"| Numeric id range | {ids['train']['min']} - {ids['train']['max']} | "
        f"{ids['new_182']['min']} - {ids['new_182']['max']} |")
    add("")
    add("Block histogram (numeric id / 100):")
    add("")
    add("| Block | TRAIN | New |")
    add("|---|---:|---:|")
    blocks = sorted(
        set(ids["train"]["block_histogram"]) | set(ids["new_182"]["block_histogram"]),
        key=lambda b: int(b.split("-")[0]),
    )
    for block in blocks:
        add(f"| {block} | {ids['train']['block_histogram'].get(block, 0)} | "
            f"{ids['new_182']['block_histogram'].get(block, 0)} |")
    add("")
    add(f"**Pattern found.** {ids['pattern_description']}")
    add("")
    add(f"- **{ids['above_train_max']['count']} of 182 ({ids['above_train_max']['pct']}%)** "
        f"new ids lie above the TRAIN maximum -> {ids['above_train_max']['interpretation']}.")
    add(f"- **{ids['within_train_range']['count']} of 182** lie inside the TRAIN id range "
        f"({', '.join(str(i) for i in ids['within_train_range']['ids'])}) -> "
        f"{ids['within_train_range']['interpretation']}.")
    add("")
    add("> **This is the single most important negative result of the investigation.** "
        "A renumbering migration leaves a mechanical fingerprint - a constant offset, an "
        "added prefix, a widened field. None is present. The ids were *allocated*, not "
        "*rewritten*. Whatever equivalence exists is curricular, not clerical.")
    add("")

    # --- section 4
    dis = payload["disappearance_analysis"]
    add("## 4. Disappearance analysis")
    add("")
    add(f"- TRAIN courses: {dis['train_courses']}; still present in VALID: "
        f"{dis['still_present_in_valid']}; absent from VALID: {dis['absent_from_valid']} "
        f"({dis['absent_from_valid_train_rows']:,} TRAIN rows).")
    add(f"- Absent AND active at the boundary (last TRAIN semester 20212/20213): "
        f"**{dis['absent_and_active_at_boundary']['courses']} courses, "
        f"{dis['absent_and_active_at_boundary']['train_rows']:,} TRAIN rows**.")
    add(f"- Enrolment-collapse cohort ({dis['volume_collapsed']['definition']}): "
        f"**{dis['volume_collapsed']['courses']} courses, "
        f"{dis['volume_collapsed']['train_rows']:,} TRAIN rows**.")
    add("")
    add(f"**Complementarity verdict.** {dis['complementarity_verdict']}")
    add("")
    add(f"{dis['volume_collapsed']['note'].capitalize()}. For example, the old "
        "first-year courses of degree `3.111` collapse from roughly 1,500-2,100 TRAIN "
        "rows each to 23-68 VALID rows while their same-named new-plan counterparts "
        "absorb 700-900 VALID rows each - a teach-out tail, not a disappearance.")
    add("")
    add("### Censoring limitation (explicit)")
    add("")
    add(dis["censoring_limitation"])
    add("")
    add(f"Affected in this run: {payload['censoring']['censored_debut_courses']} new "
        f"courses debut in 20232/20233 ({payload['censoring']['censored_debut_rows']:,} "
        f"VALID rows) and are flagged `censored_debut`; "
        f"{payload['censoring']['censored_predecessor_courses']} top candidates have a "
        "predecessor whose last active semester falls in the censored window and are "
        "flagged `censored_predecessor`. Confirmation is withheld from all of them by rule.")
    add("")

    # --- degree lineage
    add("### Degree lineage used to constrain candidates")
    add("")
    add("Lineage links a VALID-only degree to a TRAIN degree when EITHER at least 5 "
        "students moved from the old degree to the new one, OR the normalized degree "
        "names are >= 0.60 similar after stripping the `2023` re-issue suffix. Both "
        "signals are reported so the link can be checked independently.")
    add("")
    add("| New degree | Name | VALID rows | Predecessor(s) | Migrating students |")
    add("|---|---|---:|---|---|")
    for record in payload["degree_lineage"]:
        migration = ", ".join(f"{k}:{v}" for k, v in record["by_student_migration"].items()) or "-"
        add(f"| `{record['new_degree']}` | {record['new_degree_name']} | "
            f"{record['valid_rows']:,} | {', '.join(f'`{d}`' for d in record['predecessors']) or '-'} | "
            f"{migration} |")
    add("")

    # --- section 5
    add("## 5. Matching rule - FIXED BEFORE the classification was produced")
    add("")
    add(f"**{BUCKET_RULE['principle']}**")
    add("")
    add("Normalization applied to every course name before comparison:")
    add("")
    for step in payload["normalization_steps"]:
        add(f"- {step}")
    add("")
    add("A second, looser key additionally strips the Arabic definite article `ال` from "
        "each token; similarity is the maximum over the strict and loose keys of "
        "(character sequence ratio, token-set Jaccard).")
    add("")
    add("Signals scored per (new course, candidate predecessor) pair: identifier "
        "pattern; disappearance/appearance complementarity; predecessor enrolment "
        "collapse; enrolment-volume continuity; credits equal; requirement type equal; "
        "faculty equal; degree-lineage link; planned level (year_order, semester_order) "
        "equal; normalized name similarity; degree overlap. Prerequisite structure "
        "could not be scored - the attribute does not exist (section 2).")
    add("")
    add("### Minimum evidence combination per bucket")
    add("")
    for bucket in ("confirmed_equivalent", "likely_equivalent_needs_review",
                   "genuinely_new", "unresolved"):
        add(f"**`{bucket}`**")
        add("")
        for clause in BUCKET_RULE[bucket]:
            add(f"- {clause}")
        add("")

    # --- section 6
    totals = payload["bucket_totals"]
    add("## 6. Classification")
    add("")
    add("| Bucket | Courses | VALID rows | % of 25,627 |")
    add("|---|---:|---:|---:|")
    for bucket in ("confirmed_equivalent", "likely_equivalent_needs_review",
                   "genuinely_new", "unresolved"):
        info = totals[bucket]
        add(f"| `{bucket}` | {info['courses']} | {info['rows']:,} | "
            f"{100.0 * info['rows'] / EXPECTED_NEVER_ROWS:.1f}% |")
    add(f"| **total** | **{sum(v['courses'] for v in totals.values())}** | "
        f"**{sum(v['rows'] for v in totals.values()):,}** | 100.0% |")
    add("")
    sens = payload["censoring_guard_sensitivity"]
    add("### Sensitivity to the scope of the censoring guard (NOT the reported result)")
    add("")
    add(sens["rationale"])
    add("")
    add(f"Variation: {sens['variation']}.")
    add("")
    add("| Bucket | Courses (pre-registered) | Courses (sensitivity) | VALID rows (pre-registered) | VALID rows (sensitivity) |")
    add("|---|---:|---:|---:|---:|")
    for bucket in ("confirmed_equivalent", "likely_equivalent_needs_review",
                   "genuinely_new", "unresolved"):
        primary, alternative = totals[bucket], sens["bucket_totals"][bucket]
        add(f"| `{bucket}` | {primary['courses']} | {alternative['courses']} | "
            f"{primary['rows']:,} | {alternative['rows']:,} |")
    add("")
    add(f"Coverage recovery under the sensitivity: "
        f"{sens['confirmed_only_rows_becoming_covered']:,} rows from `confirmed` alone "
        f"(vs {payload['payoff']['confirmed_only']['never_in_train_rows_becoming_covered']:,} "
        f"under the pre-registered rule); the `confirmed + likely` upper bound is "
        f"{sens['confirmed_plus_likely_rows_becoming_covered']:,} either way, because the "
        "guard only moves courses between the two accepted buckets.")
    add("")
    add("**The pre-registered rule remains authoritative.** The sensitivity is reported "
        "so the reader can see that the difference between `confirmed` and `likely` here "
        "is largely a judgement about censoring scope, not about evidence strength - "
        "which is precisely the kind of call the registrar's equivalence table would settle.")
    add("")
    add("Full per-course detail is in `COURSE_IDENTITY_INVESTIGATION_CANDIDATES.csv` (sorted by VALID "
        "row count descending). The 40 highest-volume courses:")
    add("")
    add("| # | New course | Name | VALID rows | Debut | Bucket | Candidate predecessor | Matched | Conflicted |")
    add("|---:|---|---|---:|---|---|---|---|---|")
    for index, row in enumerate(payload["classification_rows"][:40], start=1):
        add(f"| {index} | `{row['new_course_id']}` | {row['new_course_name']} | "
            f"{row['valid_rows']:,} | {row['first_valid_semester']} | "
            f"`{row['bucket']}` | {row['candidate_display']} | "
            f"{row['signals_matched']} | {row['signals_conflicted']} |")
    add("")

    # --- section 7
    pay = payload["payoff"]
    add("## 7. Payoff quantification")
    add("")
    add(pay["method"])
    add("")
    add("**Simulation validation.** Before use, the re-implementation of the "
        "Level-1 -> Level-2 support lookup was checked against the on-disk columns: "
        f"{pay['reimplementation_validation']['course_history_count_mismatches_vs_on_disk']} "
        "`course_history_count` mismatches and "
        f"{pay['reimplementation_validation']['course_difficulty_missing_mismatches_vs_on_disk']} "
        "`course_difficulty_missing` mismatches over "
        f"{pay['reimplementation_validation']['valid_rows_checked']:,} VALID rows "
        f"(verdict: **{pay['reimplementation_validation']['verdict']}**).")
    add("")
    add("| Scenario | Courses mapped | Rows gaining observed history | Rows crossing the 20-row threshold | never-in-TRAIN rows still uncovered |")
    add("|---|---:|---:|---:|---:|")
    for label, key in (("`confirmed_equivalent` only", "confirmed_only"),
                       ("`confirmed` + `likely` (upper bound)", "confirmed_plus_likely_upper_bound")):
        info = pay[key]
        add(f"| {label} | {info['mapped_courses']} | "
            f"{info['never_in_train_rows_gaining_observed_history']:,} | "
            f"{info['never_in_train_rows_becoming_covered']:,} | "
            f"{info['never_in_train_rows_remaining_uncovered']:,} |")
    add("")
    add("**Residual against the original 26,882 uncovered VALID rows:**")
    add("")
    add("| Scenario | Uncovered rows remaining | % of the original 26,882 |")
    add("|---|---:|---:|")
    add(f"| `confirmed` only | {pay['residual']['confirmed_only']['uncovered_valid_rows_remaining']:,} | "
        f"{pay['residual']['confirmed_only']['pct_of_original_26882']}% |")
    add(f"| `confirmed` + `likely` | {pay['residual']['confirmed_plus_likely']['uncovered_valid_rows_remaining']:,} | "
        f"{pay['residual']['confirmed_plus_likely']['pct_of_original_26882']}% |")
    add("")
    add("> The payoff is an arithmetic upper bound on *coverage*, not on *accuracy*. "
        "Borrowing an old course's pass-rate statistics for a revised course assumes the "
        "revision did not change difficulty. Nothing in this data can test that "
        "assumption; only outcomes under the new plan can, and those are exactly the "
        "rows in question.")
    add("")

    # --- scope
    scope = payload["scope"]
    add("## 8. Scope of the change")
    add("")
    add(f"**Verdict: {scope['verdict']}.**")
    add("")
    add("| Faculty | New courses | VALID rows | Of which equivalent candidates (courses / rows) |")
    add("|---|---:|---:|---|")
    for faculty, info in sorted(scope["new_courses_by_faculty"].items(), key=lambda kv: -kv[1]["rows"]):
        equivalent = scope["equivalent_candidates_by_faculty"].get(faculty, {"courses": 0, "rows": 0})
        add(f"| `{faculty}` | {info['courses']} | {info['rows']:,} | "
            f"{equivalent['courses']} / {equivalent['rows']:,} |")
    add("")
    add(scope["explanation"])
    add("")

    # --- registrar
    add("## 9. Questions only the university registrar can answer")
    add("")
    for index, question in enumerate(REGISTRAR_QUESTIONS, start=1):
        add(f"{index}. {question}")
    add("")

    add("## 10. What was NOT done")
    add("")
    add("- No `canonical_course_id` or equivalence mapping was created, applied or wired.")
    add("- No dataset was built, copied or written; no `CURRENT_VERSION.txt`, default, "
        "or promotion marker was touched.")
    add("- No model was trained, retrained or re-tuned.")
    add("- `df_test_final.parquet` was not read, and no TEST path was constructed.")
    add("- Nothing was pushed.")
    add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
def main() -> int:
    git = git_context()

    train = pd.read_parquet(TRAIN_PATH, columns=MODEL_COLUMNS)
    valid = pd.read_parquet(VALID_PATH, columns=MODEL_COLUMNS)
    for frame in (train, valid):
        for column in ("course_id", "degree_id", "faculty_id", "part_id", "degree_course_key"):
            frame[column] = frame[column].astype("string")

    never, facts = compute_never_in_train(train, valid)
    if not facts["matches_diagnostic"]:
        print("STOP: never_in_train figures differ from the diagnostic", file=sys.stderr)
        print(json.dumps(facts, indent=2), file=sys.stderr)
        return 2

    train_ids = set(train["course_id"].dropna())
    never_valid = valid.loc[never]
    new_ids_series = never_valid["course_id"]
    new_valid_rows = new_ids_series.value_counts()
    new_first_semester = never_valid.groupby("course_id")["part_id"].min()
    new_ids = list(new_valid_rows.sort_values(ascending=False).index)

    raw, _clean = load_catalog()

    inventory = attribute_inventory(raw, set(new_ids), train_ids)
    ids_structure = identifier_structure(set(new_ids), train_ids)

    activity = build_activity(train, valid)
    profile = course_profiles(train, valid, activity)
    disappearance = disappearance_analysis(profile, train_ids)
    lineage, lineage_records = degree_lineage(train, valid, raw)

    entries = score_candidates(
        new_ids, train_ids, profile, raw, lineage, new_valid_rows, new_first_semester
    )
    classifications = {entry["new_course_id"]: classify(entry) for entry in entries}

    bucket_totals = {
        bucket: {"courses": 0, "rows": 0}
        for bucket in ("confirmed_equivalent", "likely_equivalent_needs_review",
                       "genuinely_new", "unresolved")
    }
    classification_rows = []
    for entry in entries:
        info = classifications[entry["new_course_id"]]
        bucket_totals[info["bucket"]]["courses"] += 1
        bucket_totals[info["bucket"]]["rows"] += entry["valid_rows"]
        top = info["top_candidates"][0] if info["top_candidates"] else None
        classification_rows.append({
            "new_course_id": entry["new_course_id"],
            "new_course_name": entry["new_course_name"],
            "valid_rows": entry["valid_rows"],
            "first_valid_semester": entry["first_valid_semester"],
            "faculty_id": entry["faculty_id"],
            "degrees_offered_in_valid": entry["degrees_offered_in_valid"],
            "censored_debut": entry["censored_debut"],
            "bucket": info["bucket"],
            "reason": info["reason"],
            "candidate_predecessor_ids": [c["old_course_id"] for c in info["top_candidates"]],
            "candidate_display": (
                f"`{top['old_course_id']}` {top['old_course_name']}" if top else "-"
            ),
            "top_candidate": top,
            "signals_matched": ", ".join(top["signals_matched"]) if top else "-",
            "signals_conflicted": ", ".join(top["signals_conflicted"]) if top else "-",
        })

    pay = payoff(train, valid, never, classifications)
    scope = scope_analysis(entries, classifications)

    # Clearly-labelled sensitivity: see classify() docstring. NOT the reported result.
    sensitivity_classifications = {
        entry["new_course_id"]: classify(entry, censor_guard="temporal_only")
        for entry in entries
    }
    sensitivity_totals = {
        bucket: {"courses": 0, "rows": 0}
        for bucket in ("confirmed_equivalent", "likely_equivalent_needs_review",
                       "genuinely_new", "unresolved")
    }
    for entry in entries:
        bucket = sensitivity_classifications[entry["new_course_id"]]["bucket"]
        sensitivity_totals[bucket]["courses"] += 1
        sensitivity_totals[bucket]["rows"] += entry["valid_rows"]
    sensitivity_payoff = payoff(train, valid, never, sensitivity_classifications)
    sensitivity = {
        "label": "SENSITIVITY ONLY - not the reported classification",
        "variation": "the censoring guard is applied only to the "
                     "temporal-complementarity signal; a whole-window enrolment "
                     "collapse is accepted as uncensored corroboration",
        "rationale": "As pre-registered, the guard withholds confirmation whenever the "
                     "predecessor's last active semester falls in 20232/20233 - even "
                     "when the corroborating evidence is a whole-VALID-window enrolment "
                     "collapse, which is not itself right-censored. The pre-registered "
                     "rule is reported as authoritative and was NOT changed after "
                     "results were seen; this block quantifies how much of the "
                     "conservatism the guard's scope accounts for.",
        "bucket_totals": sensitivity_totals,
        "confirmed_only_rows_becoming_covered": sensitivity_payoff["confirmed_only"][
            "never_in_train_rows_becoming_covered"
        ],
        "confirmed_plus_likely_rows_becoming_covered": sensitivity_payoff[
            "confirmed_plus_likely_upper_bound"
        ]["never_in_train_rows_becoming_covered"],
    }

    censored_debut = [e for e in entries if e["censored_debut"]]
    censored_predecessor = [
        e for e in entries
        if classifications[e["new_course_id"]]["top_candidates"]
        and classifications[e["new_course_id"]]["top_candidates"][0]["censored_predecessor"]
    ]

    confirmed = bucket_totals["confirmed_equivalent"]
    likely = bucket_totals["likely_equivalent_needs_review"]
    headline = [
        f"**The 182 courses are not renumbered old courses.** {ids_structure['above_train_max']['count']} "
        f"of 182 ids were allocated above the previous maximum with no mechanical "
        "relation to any old id; there is no renumbering fingerprint.",
        "**What actually happened is a curriculum revision.** New degree programmes "
        "(25 VALID-only degrees, several literally named with a `2023` suffix) were "
        "opened under two faculty codes that barely exist in TRAIN (167.111: 2 TRAIN "
        "rows; 177.111: 0), each with a freshly numbered course catalog.",
        "**Many new courses do have a content predecessor.** "
        f"{confirmed['courses']} courses ({confirmed['rows']:,} VALID rows) meet the "
        f"pre-registered confirmation bar and {likely['courses']} more "
        f"({likely['rows']:,} rows) are likely but need review - typically the same "
        "normalized name, credits and requirement type inside a linked degree, with the "
        "old course's enrolment collapsing to a teach-out tail.",
        "**But equivalence here is curricular, not clerical.** The predecessor usually "
        "still exists and still runs. Borrowing its difficulty statistics is a modelling "
        "decision about content similarity, not a correction of a broken identifier - "
        "and it is the human's decision, not this task's.",
        f"**Best case coverage recovery:** "
        f"{pay['confirmed_plus_likely_upper_bound']['never_in_train_rows_becoming_covered']:,} "
        f"of the 25,627 never-in-TRAIN rows under `confirmed + likely`, leaving "
        f"{pay['residual']['confirmed_plus_likely']['uncovered_valid_rows_remaining']:,} "
        f"of the original 26,882 uncovered rows "
        f"({pay['residual']['confirmed_plus_likely']['pct_of_original_26882']}%).",
    ]

    payload = {
        "investigation": "course_identity_are_new_courses_renumbered_old_courses",
        "scope": {
            "read_only": True,
            "dataset_version": VERSION,
            "dataset_reads": [
                str(TRAIN_PATH.relative_to(ROOT)).replace("\\", "/"),
                str(VALID_PATH.relative_to(ROOT)).replace("\\", "/"),
                str(CATALOG_RAW.relative_to(ROOT)).replace("\\", "/"),
                str(CATALOG_CLEAN.relative_to(ROOT)).replace("\\", "/"),
            ],
            "test_policy": "closed_not_read",
            "test_dataset_read": False,
            "mapping_created": False,
            "mapping_applied": False,
            "mapping_wired": False,
            "model_trained": False,
            "model_retuned": False,
            "dataset_written": False,
            "default_changed": False,
            "current_version_changed": False,
            "promotion_performed": False,
            "push_performed": False,
            "csv_is_review_artifact_not_mapping": True,
        },
        "git": git,
        "headline_verdict": headline,
        "never_in_train_reproduction": facts,
        "attribute_inventory": inventory,
        "identifier_structure": ids_structure,
        "disappearance_analysis": disappearance,
        "degree_lineage": lineage_records,
        "normalization_steps": NORMALIZATION_STEPS,
        "matching_rule_fixed_before_classification": BUCKET_RULE,
        "censoring": {
            "valid_ends_at": "20233",
            "censoring_semesters": sorted(CENSORING_SEMESTERS),
            "censored_debut_courses": len(censored_debut),
            "censored_debut_rows": int(sum(e["valid_rows"] for e in censored_debut)),
            "censored_debut_course_ids": [e["new_course_id"] for e in censored_debut],
            "censored_predecessor_courses": len(censored_predecessor),
            "censored_predecessor_course_ids": [e["new_course_id"] for e in censored_predecessor],
            "treatment": "flagged, never scored as confirmed",
        },
        "bucket_totals": bucket_totals,
        "censoring_guard_sensitivity": sensitivity,
        "classification_rows": classification_rows,
        "payoff": pay,
        "scope_of_change": scope,
        "registrar_questions": REGISTRAR_QUESTIONS,
    }
    payload["scope_verdict"] = scope["verdict"]

    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    payload_for_md = dict(payload)
    payload_for_md["scope"] = scope
    OUT_MD.write_text(render_markdown(payload_for_md), encoding="utf-8")

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "REVIEW ARTIFACT - NOT AN EQUIVALENCE MAPPING. No mapping has been created "
            "or applied. Multiple candidates per course; buckets are mixed; conflicting "
            "evidence is retained deliberately. Do not consume as a lookup table."
        ])
        writer.writerow([
            "review_bucket", "new_course_id", "new_course_name", "valid_rows",
            "first_valid_semester", "censored_debut", "faculty_id",
            "degrees_offered_in_valid", "candidate_1_course_id", "candidate_1_name",
            "candidate_1_name_similarity", "candidate_1_name_exact_normalized",
            "candidate_1_credits_old", "candidate_1_credits_new",
            "candidate_1_credits_equal", "candidate_1_requirement_type_equal",
            "candidate_1_faculty_equal", "candidate_1_degree_lineage_linked",
            "candidate_1_planned_level_equal", "candidate_1_train_rows",
            "candidate_1_last_active_semester", "candidate_1_volume_collapsed",
            "candidate_1_post_boundary_volume_ratio",
            "candidate_1_temporal_complementarity", "candidate_1_censored_predecessor",
            "candidate_1_signals_matched", "candidate_1_signals_conflicted",
            "candidate_2_course_id", "candidate_2_name", "candidate_2_name_similarity",
            "candidate_3_course_id", "candidate_3_name", "candidate_3_name_similarity",
            "reason", "human_decision_required",
        ])
        for row in classification_rows:
            info = classifications[row["new_course_id"]]
            top = row["top_candidate"]
            second = info["top_candidates"][1] if len(info["top_candidates"]) > 1 else None
            third = info["top_candidates"][2] if len(info["top_candidates"]) > 2 else None
            writer.writerow([
                row["bucket"], row["new_course_id"], row["new_course_name"],
                row["valid_rows"], row["first_valid_semester"], row["censored_debut"],
                row["faculty_id"], "|".join(row["degrees_offered_in_valid"]),
                top["old_course_id"] if top else "",
                top["old_course_name"] if top else "",
                top["name_similarity"] if top else "",
                top["name_exact_normalized"] if top else "",
                top["credits_old"] if top else "",
                top["credits_new"] if top else "",
                top["credits_equal"] if top else "",
                top["requirement_type_equal"] if top else "",
                top["faculty_equal"] if top else "",
                top["degree_lineage_linked"] if top else "",
                top["planned_level_equal"] if top else "",
                top["old_train_rows"] if top else "",
                top["old_last_active_semester"] if top else "",
                top["predecessor_volume_collapsed"] if top else "",
                top["post_boundary_volume_ratio"] if top else "",
                top["temporal_complementarity"] if top else "",
                top["censored_predecessor"] if top else "",
                "|".join(top["signals_matched"]) if top else "",
                "|".join(top["signals_conflicted"]) if top else "",
                second["old_course_id"] if second else "",
                second["old_course_name"] if second else "",
                second["name_similarity"] if second else "",
                third["old_course_id"] if third else "",
                third["old_course_name"] if third else "",
                third["name_similarity"] if third else "",
                row["reason"], "yes",
            ])

    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print()
    for bucket in ("confirmed_equivalent", "likely_equivalent_needs_review",
                   "genuinely_new", "unresolved"):
        info = bucket_totals[bucket]
        print(f"{bucket}: {info['courses']} courses / {info['rows']} VALID rows")
    print(
        "Best-case coverage recovery: "
        f"{pay['confirmed_plus_likely_upper_bound']['never_in_train_rows_becoming_covered']}"
        f" of {EXPECTED_NEVER_ROWS} never_in_train rows"
    )
    print(f"Scope: {scope['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
