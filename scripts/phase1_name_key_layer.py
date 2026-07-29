"""Phase 1 read-only measurement of the name-key layer for course identity.

Answers Q1-Q7 of the Phase 1 name-key task from repository artifacts only.

Guarantees
----------
* Reads only: the frozen TRAIN/VALID parquets of dataset version
  ``2026-07-26_batched_fixes__registration_roster_concurrent``, the read-only
  cleaned catalog (``clean_v_acd_degree_course.parquet``), and
  ``COURSE_IDENTITY_67_HUMAN_REVIEW.csv``.
* Never constructs, globs, stats or reads any TEST path. ``df_test_final`` is
  never referenced.
* ``final_mark`` and every other VALID outcome column are never loaded from
  VALID. Asserted at runtime immediately after the VALID read.
* No model is loaded, trained, fit, or scored. ``src.course_difficulty`` is
  NOT imported -- every statistic here (support counts, raw pass rates, raw
  average marks) is recomputed directly from TRAIN rows as simple counts and
  means, independent of the shrinkage/fallback machinery that produced the
  on-disk difficulty columns. This mirrors, but does not reuse, the existing
  Level-2 definition (course_id-only grouping).
* Writes exactly one CSV (``models/runs/phase1_name_key_per_course.csv``);
  the markdown report is written separately by hand from this script's JSON
  payload (printed to stdout) plus the CSV.

Interpretive choices made where the spec is under-determined (documented here
and again in the report):

1. Name source. Neither df_train_final.parquet nor df_valid_final.parquet
   contains a course_name_sl column (verified against the on-disk schema).
   The "TRAIN row name" fallback described in the task therefore never
   fires -- every name is catalog-sourced or absent. Courses absent from the
   catalog get no name and cannot enter the name-key layer; they are
   reported separately, not silently dropped.
2. Credits. clean_v_acd_degree_course.parquet has exactly one
   course_credits value per course_id (verified: 0 of 1,503 catalog courses
   have >1 distinct value), and every TRAIN course_id's row-level
   course_credits matches its catalog value exactly (0 mismatches over 811
   TRAIN courses). Credits is therefore unambiguous; the catalog value is
   used everywhere.
3. requirement_type_id. This is NOT course-invariant: 35 of 1,503 catalog
   courses and 58 of 811 TRAIN courses carry more than one requirement_type_id
   across the degrees they belong to (a course can be elective in one degree
   and required in another). The spec's per-course narrow key needs a single
   requirement_type_id per course_id, so the MODAL requirement_type_id
   (most frequent among that course's rows in the relevant split; ties break
   on the smaller requirement_type_id) is used as the canonical value. This
   is a documented interpretation, not a change to the key formula.
4. "Level 2, TRAIN-only" pass rate / avg mark (Q2 point 4, Q4). Computed as a
   raw, unsmoothed per-course_id mean over TRAIN rows with a non-null
   final_mark -- NOT the shrinkage-smoothed table src.course_difficulty
   would produce. This keeps the whole measurement independent of the
   difficulty module, consistent with the instruction (for train_support_l2)
   not to load the difficulty state.
5. Degree family (Q5). family_of(degree_name_sl) = strip a trailing "2023"
   token (with or without a preceding space), then keep only the text before
   the first "/" (dropping the specialisation suffix), then strip whitespace.
   Verified against the catalog: every "family/specialisation" degree name
   uses "/" as the separator and every year-suffixed name uses the literal
   token "2023" as the final token.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

VERSION = "2026-07-26_batched_fixes__registration_roster_concurrent"
VERSION_DIR = ROOT / "data" / "model_data" / "versions" / VERSION
TRAIN_PATH = VERSION_DIR / "df_train_final.parquet"
VALID_PATH = VERSION_DIR / "df_valid_final.parquet"

CATALOG_PATH = ROOT / "data" / "preprocessed" / "V_ACD_DEGREE_COURSE" / "clean_v_acd_degree_course.parquet"
PAIRS_PATH = ROOT / "models" / "runs" / "COURSE_IDENTITY_67_HUMAN_REVIEW.csv"

OUT_CSV = ROOT / "models" / "runs" / "phase1_name_key_per_course.csv"

MIN_SUPPORT = 20
SECONDARY_SUPPORT = 50

TRAIN_COLUMNS = [
    "course_id",
    "course_credits",
    "requirement_type_id",
    "part_id",
    "final_mark",
    "degree_id",
    "faculty_id",
]

VALID_COLUMNS = [
    "course_id",
    "course_credits",
    "requirement_type_id",
    "difficulty_fallback_level",
    "course_pass_rate_historical",
    "course_avg_mark_historical",
    "course_difficulty_missing",
    "course_history_count",
    "degree_id",
]


# ---------------------------------------------------------------------------
# Normalization spec (verbatim from the task; do not modify)
# ---------------------------------------------------------------------------
AR2LAT = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).translate(AR2LAT)
    s = re.sub(r"[ً-ْٰـ]", "", s)  # harakat + tatweel
    for a, b in [
        ("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"),
        ("ة", "ه"), ("ؤ", "و"), ("ئ", "ي"),
    ]:
        s = s.replace(a, b)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    digits = re.findall(r"\d+", s)
    base = re.sub(r"\s+", " ", re.sub(r"\d+", " ", s)).strip()
    tokens = [re.sub(r"^ال", "", t) if len(t) > 3 else t for t in base.split()]
    base = " ".join(tokens)
    return base + ("#" + digits[-1] if digits else "")


YEAR_RE = re.compile(r"\s*2023\s*$")


def family_of(name: str) -> str:
    text = str(name)
    text = YEAR_RE.sub("", text)
    text = text.split("/", 1)[0]
    return text.strip()


def modal(series: pd.Series) -> Any:
    counts = series.value_counts()
    top = counts.max()
    candidates = sorted(counts[counts == top].index.tolist())
    return candidates[0]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_catalog() -> pd.DataFrame:
    cat = pd.read_parquet(CATALOG_PATH)
    for col in ("course_id", "degree_id"):
        cat[col] = cat[col].astype("string")
    return cat


def catalog_course_names(catalog: pd.DataFrame) -> tuple[dict[str, str], dict[str, float]]:
    dedup = catalog.drop_duplicates("course_id")
    names = dict(zip(dedup["course_id"], dedup["course_name_sl"]))
    credits = dict(zip(dedup["course_id"], dedup["course_credits"].astype("float64")))
    return names, credits


def catalog_degree_families(catalog: pd.DataFrame) -> dict[str, str]:
    dedup = catalog.drop_duplicates("degree_id")
    return {did: family_of(name) for did, name in zip(dedup["degree_id"], dedup["degree_name_sl"])}


def load_train() -> pd.DataFrame:
    train = pd.read_parquet(TRAIN_PATH, columns=TRAIN_COLUMNS)
    for col in ("course_id", "degree_id", "faculty_id"):
        train[col] = train[col].astype("string")
    return train


def load_valid() -> pd.DataFrame:
    valid = pd.read_parquet(VALID_PATH, columns=VALID_COLUMNS)
    if "final_mark" in valid.columns:
        raise SystemExit("STOP: a VALID outcome column was loaded.")
    for col in ("course_id", "degree_id"):
        valid[col] = valid[col].astype("string")
    return valid


# ---------------------------------------------------------------------------
# Q1 - normalization validation
# ---------------------------------------------------------------------------
def answer_q1(pairs: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for record in pairs.itertuples(index=False):
        old_key = norm_name(record.old_course_name)
        new_key = norm_name(record.new_course_name)
        rows.append(
            {
                "new_course_id": record.new_course_id,
                "old_course_id": record.old_course_id,
                "new_course_name": record.new_course_name,
                "old_course_name": record.old_course_name,
                "new_key": new_key,
                "old_key": old_key,
                "match": new_key == old_key,
                "new_valid_row_count": int(record.new_valid_row_count),
            }
        )
    frame = pd.DataFrame(rows)
    matched = int(frame["match"].sum())
    mismatches = frame.loc[~frame["match"]]

    known_1 = mismatches.loc[
        mismatches["old_course_name"].str.contains("بنيان الحواسيب", na=False)
    ]
    known_2 = mismatches.loc[
        mismatches["old_course_name"].str.contains("علم اجتماع", na=False)
        | mismatches["new_course_name"].str.contains("علم اجتماع", na=False)
        | mismatches["old_course_name"].str.contains("علم الاجتماع", na=False)
        | mismatches["new_course_name"].str.contains("علم الاجتماع", na=False)
    ]

    return {
        "pairs_total": int(len(frame)),
        "matched_count": matched,
        "mismatch_count": int(len(mismatches)),
        "mismatches": mismatches.to_dict(orient="records"),
        "known_failure_1_bunyan_alhawasib_present": bool(len(known_1)),
        "known_failure_1_rows": known_1.to_dict(orient="records"),
        "known_failure_2_ilm_ijtimaa_present_in_mismatches": bool(len(known_2)),
        "known_failure_2_rows": known_2.to_dict(orient="records"),
    }, frame


# ---------------------------------------------------------------------------
# Q2 - TRAIN name-key index
# ---------------------------------------------------------------------------
def build_train_index(
    train: pd.DataFrame,
    catalog_names: dict[str, str],
    catalog_credits: dict[str, float],
) -> dict[str, Any]:
    train = train.copy()
    train["mark_present"] = train["final_mark"].notna()
    train["pass_value"] = (train["final_mark"] >= 50) & train["mark_present"]

    course_ids_train = sorted(train["course_id"].dropna().unique().tolist())
    catalog_backed = [cid for cid in course_ids_train if cid in catalog_names]
    not_catalog_backed = [cid for cid in course_ids_train if cid not in catalog_names]

    agg = (
        train.groupby("course_id", sort=False)
        .agg(
            support_l2=("mark_present", "sum"),
            sum_pass=("pass_value", "sum"),
            sum_mark=("final_mark", "sum"),
        )
        .reset_index()
    )
    agg["pass_rate_l2_raw"] = np.where(
        agg["support_l2"] > 0, agg["sum_pass"] / agg["support_l2"], np.nan
    )
    agg["avg_mark_l2_raw"] = np.where(
        agg["support_l2"] > 0, agg["sum_mark"] / agg["support_l2"], np.nan
    )

    modal_req = (
        train.groupby("course_id")["requirement_type_id"]
        .apply(modal)
        .rename("modal_req_type")
        .reset_index()
    )
    agg = agg.merge(modal_req, on="course_id", how="left")

    agg = agg.loc[agg["course_id"].isin(catalog_backed)].copy()
    agg["course_name_sl"] = agg["course_id"].map(catalog_names)
    agg["course_credits"] = agg["course_id"].map(catalog_credits)
    agg["norm_name"] = agg["course_name_sl"].map(norm_name)
    agg["round_credits"] = agg["course_credits"].round().astype("int64")
    agg["name_key_narrow"] = list(
        zip(agg["norm_name"], agg["round_credits"], agg["modal_req_type"].astype("int64"))
    )
    agg["name_key_wide"] = agg["norm_name"]

    narrow_index: dict[Any, list[tuple[str, int]]] = {}
    for key, group in agg.groupby("name_key_narrow"):
        entries = sorted(
            zip(group["course_id"], group["support_l2"]), key=lambda t: (-t[1], t[0])
        )
        narrow_index[key] = entries

    wide_index: dict[Any, list[tuple[str, int]]] = {}
    for key, group in agg.groupby("name_key_wide"):
        entries = sorted(
            zip(group["course_id"], group["support_l2"]), key=lambda t: (-t[1], t[0])
        )
        wide_index[key] = entries

    narrow_sizes = pd.Series({k: len(v) for k, v in narrow_index.items()})
    wide_sizes = pd.Series({k: len(v) for k, v in wide_index.items()})

    def spread_stats(index: dict[Any, list[tuple[str, int]]], sizes: pd.Series) -> dict[str, Any]:
        multi_keys = sizes[sizes > 1].index.tolist()
        pass_rate_map = dict(zip(agg["course_id"], agg["pass_rate_l2_raw"]))
        diffs: list[float] = []
        for key in multi_keys:
            course_ids = [cid for cid, _ in index[key]]
            rates = [pass_rate_map[c] for c in course_ids if pd.notna(pass_rate_map.get(c))]
            for i in range(len(rates)):
                for j in range(i + 1, len(rates)):
                    diffs.append(abs(rates[i] - rates[j]))
        arr = np.array(diffs, dtype="float64")
        if arr.size == 0:
            return {"pairs": 0, "mean_abs_diff": None, "p50": None, "p90": None}
        return {
            "pairs": int(arr.size),
            "mean_abs_diff": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
        }

    return {
        "summary": {
            "train_distinct_courses": len(course_ids_train),
            "catalog_backed_courses": len(catalog_backed),
            "name_from_row_courses": 0,
            "not_catalog_backed_courses": len(not_catalog_backed),
            "not_catalog_backed_course_ids": not_catalog_backed,
            "distinct_narrow_keys": int(len(narrow_index)),
            "distinct_wide_keys": int(len(wide_index)),
            "narrow_keys_multi_course": int((narrow_sizes > 1).sum()),
            "narrow_key_size_distribution": {
                str(k): int(v) for k, v in narrow_sizes.value_counts().sort_index().items()
            },
            "wide_keys_multi_course": int((wide_sizes > 1).sum()),
            "wide_key_size_distribution": {
                str(k): int(v) for k, v in wide_sizes.value_counts().sort_index().items()
            },
            "narrow_multi_course_pass_rate_spread": spread_stats(narrow_index, narrow_sizes),
            "wide_multi_course_pass_rate_spread": spread_stats(wide_index, wide_sizes),
        },
        "agg": agg,
        "narrow_index": narrow_index,
        "wide_index": wide_index,
    }


# ---------------------------------------------------------------------------
# Q5 - service course risk (computed over both narrow and wide TRAIN index groups)
# ---------------------------------------------------------------------------
def risk_level(spread: float) -> str:
    if spread > 0.15:
        return "HIGH_RISK"
    if spread >= 0.05:
        return "MEDIUM_RISK"
    return "OK"


def compute_service_risk(
    index: dict[Any, list[tuple[str, int]]],
    course_families: dict[str, set[str]],
    pass_rate_map: dict[str, float],
) -> dict[Any, dict[str, Any]]:
    risk: dict[Any, dict[str, Any]] = {}
    for key, entries in index.items():
        families: set[str] = set()
        for cid, _ in entries:
            families |= course_families.get(cid, set())
        if len(families) >= 5:
            rates = [pass_rate_map[cid] for cid, _ in entries if pd.notna(pass_rate_map.get(cid))]
            spread = float(max(rates) - min(rates)) if len(rates) >= 2 else 0.0
            risk[key] = {
                "n_families": len(families),
                "families": sorted(families),
                "spread": spread,
                "risk": risk_level(spread),
                "n_courses": len(entries),
            }
    return risk


# ---------------------------------------------------------------------------
# Q3/Q4/Q5 combined - VALID never-in-TRAIN matching
# ---------------------------------------------------------------------------
def match_course(
    narrow_key,
    wide_key,
    own_families: set[str],
    narrow_index: dict[Any, list[tuple[str, int]]],
    wide_index: dict[Any, list[tuple[str, int]]],
    narrow_risk: dict[Any, dict[str, Any]],
    wide_risk: dict[Any, dict[str, Any]],
    course_families: dict[str, set[str]],
    threshold: int,
    exclude_service: bool,
) -> tuple[str, str | None]:
    """Return (match_type, best_course_id_or_None)."""

    def qualifying(entries: list[tuple[str, int]]) -> list[tuple[str, int]]:
        return [(cid, sup) for cid, sup in entries if sup >= threshold]

    def same_family_only(entries: list[tuple[str, int]]) -> list[tuple[str, int]]:
        return [(cid, sup) for cid, sup in entries if course_families.get(cid, set()) & own_families]

    if narrow_key in narrow_index:
        entries = qualifying(narrow_index[narrow_key])
        if entries:
            if exclude_service and narrow_risk.get(narrow_key, {}).get("risk") in (
                "HIGH_RISK", "MEDIUM_RISK",
            ):
                entries = same_family_only(entries)
            if entries:
                return "narrow", entries[0][0]

    if wide_key in wide_index:
        entries = qualifying(wide_index[wide_key])
        if entries:
            if exclude_service and wide_risk.get(wide_key, {}).get("risk") in (
                "HIGH_RISK", "MEDIUM_RISK",
            ):
                entries = same_family_only(entries)
            if entries:
                return "wide", entries[0][0]

    return "none", None


def build_valid_query(
    valid: pd.DataFrame,
    catalog_names: dict[str, str],
    catalog_credits: dict[str, float],
) -> pd.DataFrame:
    never = valid.loc[
        (valid["course_difficulty_missing"] == 1) & (valid["course_history_count"] == 0)
    ].copy()

    modal_req = (
        never.groupby("course_id")["requirement_type_id"].apply(modal).rename("modal_req_type")
    )
    own_family_sets = never.groupby("course_id")["degree_id"].apply(lambda s: set(s.tolist()))

    per_course = (
        never.groupby("course_id")
        .agg(
            valid_row_count=("course_id", "size"),
            current_pass_rate=("course_pass_rate_historical", "mean"),
            current_avg_mark=("course_avg_mark_historical", "mean"),
        )
        .reset_index()
    )
    per_course = per_course.merge(modal_req, on="course_id", how="left")
    per_course["degree_ids"] = per_course["course_id"].map(own_family_sets)
    per_course["course_name_sl"] = per_course["course_id"].map(catalog_names)
    per_course["course_credits"] = per_course["course_id"].map(catalog_credits)
    per_course["has_catalog_name"] = per_course["course_name_sl"].notna()
    per_course["norm_name"] = per_course["course_name_sl"].map(
        lambda x: norm_name(x) if pd.notna(x) else None
    )
    per_course["round_credits"] = per_course["course_credits"].round()
    return per_course, never


def main() -> int:
    pairs_raw = pd.read_csv(PAIRS_PATH, dtype="string", keep_default_na=False)
    pairs = pairs_raw[
        ["new_course_id", "new_course_name", "old_course_id", "old_course_name", "new_valid_row_count"]
    ].copy()

    catalog = load_catalog()
    catalog_names, catalog_credits = catalog_course_names(catalog)
    degree_families = catalog_degree_families(catalog)

    q1, q1_frame = answer_q1(pairs)

    train = load_train()
    valid = load_valid()

    train_bundle = build_train_index(train, catalog_names, catalog_credits)
    agg = train_bundle["agg"]
    narrow_index = train_bundle["narrow_index"]
    wide_index = train_bundle["wide_index"]

    course_families: dict[str, set[str]] = (
        train.groupby("course_id")["degree_id"]
        .apply(lambda s: {degree_families[d] for d in s if d in degree_families})
        .to_dict()
    )
    pass_rate_map = dict(zip(agg["course_id"], agg["pass_rate_l2_raw"]))
    avg_mark_map = dict(zip(agg["course_id"], agg["avg_mark_l2_raw"]))
    support_map = dict(zip(agg["course_id"], agg["support_l2"]))
    course_name_map = dict(zip(agg["course_id"], agg["course_name_sl"]))

    narrow_risk = compute_service_risk(narrow_index, course_families, pass_rate_map)
    wide_risk = compute_service_risk(wide_index, course_families, pass_rate_map)

    per_course, never_frame = build_valid_query(valid, catalog_names, catalog_credits)

    total_never_rows = int(len(never_frame))
    total_never_courses = int(per_course.shape[0])
    no_name_courses = per_course.loc[~per_course["has_catalog_name"]]

    def run_matching(threshold: int, exclude_service: bool) -> pd.DataFrame:
        results = []
        for record in per_course.itertuples(index=False):
            if not record.has_catalog_name:
                results.append(("none", None))
                continue
            narrow_key = (record.norm_name, int(record.round_credits), int(record.modal_req_type))
            wide_key = record.norm_name
            match_type, best_cid = match_course(
                narrow_key,
                wide_key,
                set(record.degree_ids),
                narrow_index,
                wide_index,
                narrow_risk,
                wide_risk,
                course_families,
                threshold,
                exclude_service,
            )
            results.append((match_type, best_cid))
        out = per_course.copy()
        out["match_type"], out["best_match_train_course_id"] = zip(*results)
        return out

    matched_20 = run_matching(MIN_SUPPORT, exclude_service=False)
    matched_50 = run_matching(SECONDARY_SUPPORT, exclude_service=False)
    matched_20_excl = run_matching(MIN_SUPPORT, exclude_service=True)

    def funnel(matched: pd.DataFrame) -> dict[str, Any]:
        out = {}
        for mt in ("narrow", "wide", "none"):
            sub = matched.loc[matched["match_type"] == mt]
            out[mt] = {
                "courses": int(len(sub)),
                "rows": int(sub["valid_row_count"].sum()),
            }
        return out

    def support_distribution(matched: pd.DataFrame) -> dict[str, Any]:
        matched_rows = matched.loc[matched["match_type"] != "none"].copy()
        if matched_rows.empty:
            return {}
        matched_rows["best_support"] = matched_rows["best_match_train_course_id"].map(support_map)
        weighted = np.repeat(
            matched_rows["best_support"].to_numpy(dtype="float64"),
            matched_rows["valid_row_count"].to_numpy(dtype="int64"),
        )
        return {
            "min": float(np.min(weighted)),
            "p25": float(np.percentile(weighted, 25)),
            "median": float(np.median(weighted)),
            "p75": float(np.percentile(weighted, 75)),
            "max": float(np.max(weighted)),
        }

    q3 = {
        "never_in_train_rows_total": total_never_rows,
        "never_in_train_courses_total": total_never_courses,
        "courses_without_catalog_name": int(len(no_name_courses)),
        "rows_without_catalog_name": int(no_name_courses["valid_row_count"].sum()) if len(no_name_courses) else 0,
        "threshold_20": {
            "funnel": funnel(matched_20),
            "support_distribution_matched": support_distribution(matched_20),
        },
        "threshold_50_secondary": {
            "funnel": funnel(matched_50),
            "support_distribution_matched": support_distribution(matched_50),
        },
        "threshold_20_excluding_service": {
            "funnel": funnel(matched_20_excl),
        },
    }

    def q4_stats(matched: pd.DataFrame, subset_match_types: tuple[str, ...]) -> dict[str, Any]:
        sub = matched.loc[matched["match_type"].isin(subset_match_types)].copy()
        if sub.empty:
            return {"rows": 0}
        sub["name_key_pass_rate"] = sub["best_match_train_course_id"].map(pass_rate_map)
        sub["name_key_avg_mark"] = sub["best_match_train_course_id"].map(avg_mark_map)
        sub = sub.dropna(subset=["name_key_pass_rate", "current_pass_rate"])
        if sub.empty:
            return {"rows": 0}

        def row_weighted(values: pd.Series, weights: pd.Series) -> np.ndarray:
            return np.repeat(values.to_numpy(dtype="float64"), weights.to_numpy(dtype="int64"))

        abs_pass = row_weighted(
            (sub["current_pass_rate"] - sub["name_key_pass_rate"]).abs(), sub["valid_row_count"]
        )
        signed_pass = row_weighted(
            sub["name_key_pass_rate"] - sub["current_pass_rate"], sub["valid_row_count"]
        )
        abs_mark = row_weighted(
            (sub["current_avg_mark"] - sub["name_key_avg_mark"]).abs(), sub["valid_row_count"]
        )
        signed_mark = row_weighted(
            sub["name_key_avg_mark"] - sub["current_avg_mark"], sub["valid_row_count"]
        )

        return {
            "rows": int(abs_pass.size),
            "abs_diff_pass_rate": {
                "mean": float(np.mean(abs_pass)),
                "median": float(np.median(abs_pass)),
                "p75": float(np.percentile(abs_pass, 75)),
                "p90": float(np.percentile(abs_pass, 90)),
                "max": float(np.max(abs_pass)),
            },
            "signed_diff_pass_rate_mean": float(np.mean(signed_pass)),
            "abs_diff_avg_mark": {
                "mean": float(np.mean(abs_mark)),
                "median": float(np.median(abs_mark)),
                "p75": float(np.percentile(abs_mark, 75)),
                "p90": float(np.percentile(abs_mark, 90)),
                "max": float(np.max(abs_mark)),
            },
            "signed_diff_avg_mark_mean": float(np.mean(signed_mark)),
            "rows_moving_gt_010_pass_rate": int((abs_pass > 0.10).sum()),
            "rows_moving_gt_020_pass_rate": int((abs_pass > 0.20).sum()),
        }

    q4 = {
        "all_matched": q4_stats(matched_20, ("narrow", "wide")),
        "narrow_only": q4_stats(matched_20, ("narrow",)),
        "wide_only": q4_stats(matched_20, ("wide",)),
        "excluding_service_all_matched": q4_stats(matched_20_excl, ("narrow", "wide")),
    }

    def risk_report(risk: dict[Any, dict[str, Any]], index: dict[Any, list[tuple[str, int]]], matched: pd.DataFrame, mt: str) -> list[dict[str, Any]]:
        rows_covered_by_key: dict[Any, int] = {}
        for record in matched.loc[matched["match_type"] == mt].itertuples(index=False):
            key = (
                (record.norm_name, int(record.round_credits), int(record.modal_req_type))
                if mt == "narrow"
                else record.norm_name
            )
            rows_covered_by_key[key] = rows_covered_by_key.get(key, 0) + int(record.valid_row_count)
        out = []
        for key, info in risk.items():
            if info["risk"] == "OK":
                continue
            out.append(
                {
                    "key": key if mt == "wide" else list(key),
                    "n_families": info["n_families"],
                    "spread": info["spread"],
                    "risk": info["risk"],
                    "n_courses": info["n_courses"],
                    "valid_rows_covered": rows_covered_by_key.get(key, 0),
                }
            )
        return sorted(out, key=lambda r: -r["valid_rows_covered"])

    q5 = {
        "narrow_risk_keys": risk_report(narrow_risk, narrow_index, matched_20, "narrow"),
        "wide_risk_keys": risk_report(wide_risk, wide_index, matched_20, "wide"),
        "narrow_risk_key_count": len(narrow_risk),
        "wide_risk_key_count": len(wide_risk),
        "high_risk_rows_as_pct_of_covered": None,
    }
    covered_rows_20 = funnel(matched_20)["narrow"]["rows"] + funnel(matched_20)["wide"]["rows"]
    high_risk_rows = int(
        sum(r["valid_rows_covered"] for r in q5["narrow_risk_keys"] if r["risk"] == "HIGH_RISK")
        + sum(r["valid_rows_covered"] for r in q5["wide_risk_keys"] if r["risk"] == "HIGH_RISK")
    )
    q5["high_risk_rows"] = high_risk_rows
    q5["covered_rows_20_all"] = covered_rows_20
    q5["high_risk_rows_as_pct_of_covered"] = (
        float(high_risk_rows / covered_rows_20) if covered_rows_20 else None
    )

    # ------------------------------------------------------------------
    # Q6 - the two unmatched pairs
    # ------------------------------------------------------------------
    def q6_pair(old_name_substr: str) -> dict[str, Any]:
        rows = q1_frame.loc[
            q1_frame["old_course_name"].str.contains(old_name_substr, na=False)
            | q1_frame["new_course_name"].str.contains(old_name_substr, na=False)
        ]
        out = []
        for r in rows.to_dict(orient="records"):
            old_id = r["old_course_id"]
            new_id = r["new_course_id"]
            old_support = int(support_map.get(old_id, 0))
            old_pass = pass_rate_map.get(old_id)
            new_valid_rows = valid.loc[valid["course_id"] == new_id]
            new_pass_mean = (
                float(new_valid_rows["course_pass_rate_historical"].mean())
                if len(new_valid_rows)
                else None
            )
            out.append(
                {
                    **r,
                    "old_train_support_l2": old_support,
                    "old_train_pass_rate_raw": old_pass,
                    "new_valid_row_count_actual": int(len(new_valid_rows)),
                    "new_valid_current_pass_rate_mean": new_pass_mean,
                    "pass_rate_diff_old_minus_new": (
                        float(old_pass - new_pass_mean)
                        if old_pass is not None and new_pass_mean is not None
                        else None
                    ),
                }
            )
        return {"rows": out}

    q6 = {
        "bunyan_alhawasib": q6_pair("بنيان الحواسيب"),
        "ilm_ijtimaa": q6_pair("اجتماع"),
    }

    # ------------------------------------------------------------------
    # Q7 - full-catalog structural coverage
    # ------------------------------------------------------------------
    cat_dedup = catalog.drop_duplicates("course_id").copy()
    cat_dedup["numeric_id"] = cat_dedup["course_id"].str.split(".").str[0].astype("int64")
    modal_req_cat = catalog.groupby("course_id")["requirement_type_id"].apply(modal)
    cat_dedup = cat_dedup.merge(
        modal_req_cat.rename("modal_req_type"), left_on="course_id", right_index=True
    )
    cat_dedup["norm_name"] = cat_dedup["course_name_sl"].map(norm_name)
    cat_dedup["round_credits"] = cat_dedup["course_credits"].round().astype("int64")
    cat_dedup["name_key_narrow"] = list(
        zip(cat_dedup["norm_name"], cat_dedup["round_credits"], cat_dedup["modal_req_type"].astype("int64"))
    )
    cat_dedup["name_key_wide"] = cat_dedup["norm_name"]

    old_side = cat_dedup.loc[cat_dedup["numeric_id"] < 1150]
    recent_side = cat_dedup.loc[cat_dedup["numeric_id"] >= 1150]

    old_narrow_keys = set(old_side["name_key_narrow"])
    old_wide_keys = set(old_side["name_key_wide"])

    recent_narrow_match = recent_side.loc[recent_side["name_key_narrow"].isin(old_narrow_keys)]
    recent_wide_only = recent_side.loc[
        ~recent_side["name_key_narrow"].isin(old_narrow_keys)
        & recent_side["name_key_wide"].isin(old_wide_keys)
    ]
    recent_no_match = recent_side.loc[
        ~recent_side["name_key_narrow"].isin(old_narrow_keys)
        & ~recent_side["name_key_wide"].isin(old_wide_keys)
    ]

    n_recent = int(len(recent_side))
    q7 = {
        "total_recent_catalog_courses": n_recent,
        "total_old_catalog_courses": int(len(old_side)),
        "narrow_match": {
            "count": int(len(recent_narrow_match)),
            "pct": float(len(recent_narrow_match) / n_recent) if n_recent else None,
        },
        "wide_match_only": {
            "count": int(len(recent_wide_only)),
            "pct": float(len(recent_wide_only) / n_recent) if n_recent else None,
        },
        "no_match": {
            "count": int(len(recent_no_match)),
            "pct": float(len(recent_no_match) / n_recent) if n_recent else None,
        },
    }

    # ------------------------------------------------------------------
    # Go/no-go
    # ------------------------------------------------------------------
    narrow_excl_rows = q3["threshold_20_excluding_service"]["funnel"]["narrow"]["rows"]
    pct_covered_narrow_excl = narrow_excl_rows / total_never_rows if total_never_rows else 0.0

    row_weighted_abs_diff = q4["excluding_service_all_matched"].get("abs_diff_pass_rate", {}).get("mean")

    hr_pct = q5["high_risk_rows_as_pct_of_covered"]

    def go_no_go(value, go_cmp, nogo_cmp):
        if value is None:
            return "UNVERIFIED"
        if go_cmp(value):
            return "GO"
        if nogo_cmp(value):
            return "NO_GO"
        return "BORDERLINE"

    signal_1 = go_no_go(pct_covered_narrow_excl, lambda v: v >= 0.40, lambda v: v < 0.20)
    signal_2 = go_no_go(row_weighted_abs_diff, lambda v: v >= 0.04, lambda v: v < 0.02)
    signal_3 = go_no_go(hr_pct, lambda v: v < 0.30, lambda v: v >= 0.50)

    verdict = "PROCEED_TO_PHASE2"
    if "NO_GO" in (signal_1, signal_2, signal_3):
        verdict = "STOP"
    elif "BORDERLINE" in (signal_1, signal_2, signal_3) or "UNVERIFIED" in (signal_1, signal_2, signal_3):
        verdict = "BORDERLINE"

    go_no_go_block = {
        "signal_1_never_in_train_covered_narrow_excl_service_pct": pct_covered_narrow_excl,
        "signal_1_verdict": signal_1,
        "signal_2_row_weighted_mean_abs_diff_pass_rate": row_weighted_abs_diff,
        "signal_2_verdict": signal_2,
        "signal_3_high_risk_pct_of_covered": hr_pct,
        "signal_3_verdict": signal_3,
        "overall_verdict": verdict,
    }

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    matched_20 = matched_20.set_index("course_id")
    matched_20_excl = matched_20_excl.set_index("course_id")

    csv_rows = []
    for cid, row in per_course.set_index("course_id").iterrows():
        m = matched_20.loc[cid]
        best_cid = m["best_match_train_course_id"]
        narrow_key = (
            (row["norm_name"], int(row["round_credits"]), int(row["modal_req_type"]))
            if row["has_catalog_name"]
            else None
        )
        wide_key = row["norm_name"] if row["has_catalog_name"] else None
        service_flag = "no"
        service_risk = "NA"
        if narrow_key in narrow_risk:
            service_flag = "yes"
            service_risk = narrow_risk[narrow_key]["risk"]
        elif wide_key in wide_risk:
            service_flag = "yes"
            service_risk = wide_risk[wide_key]["risk"]

        name_key_pass = pass_rate_map.get(best_cid) if best_cid else None
        name_key_mark = avg_mark_map.get(best_cid) if best_cid else None
        current_pass = row["current_pass_rate"]
        current_mark = row["current_avg_mark"]

        csv_rows.append(
            {
                "new_course_id": cid,
                "new_course_name": row["course_name_sl"] if row["has_catalog_name"] else "UNVERIFIED - no catalog name",
                "name_key_narrow": str(narrow_key) if narrow_key else "",
                "name_key_wide": wide_key if wide_key else "",
                "match_type": m["match_type"],
                "best_match_train_course_id": best_cid if best_cid else "",
                "best_match_train_course_name": course_name_map.get(best_cid, "") if best_cid else "",
                "best_match_train_support_l2": int(support_map.get(best_cid, 0)) if best_cid else "",
                "current_pass_rate": current_pass,
                "name_key_pass_rate": name_key_pass,
                "abs_diff_pass_rate": (
                    abs(current_pass - name_key_pass)
                    if pd.notna(current_pass) and name_key_pass is not None and pd.notna(name_key_pass)
                    else ""
                ),
                "signed_diff_pass_rate": (
                    (name_key_pass - current_pass)
                    if pd.notna(current_pass) and name_key_pass is not None and pd.notna(name_key_pass)
                    else ""
                ),
                "current_avg_mark": current_mark,
                "name_key_avg_mark": name_key_mark,
                "abs_diff_avg_mark": (
                    abs(current_mark - name_key_mark)
                    if pd.notna(current_mark) and name_key_mark is not None and pd.notna(name_key_mark)
                    else ""
                ),
                "service_course_flag": service_flag,
                "service_course_risk": service_risk,
                "valid_row_count": int(row["valid_row_count"]),
            }
        )

    csv_frame = pd.DataFrame(csv_rows).sort_values("valid_row_count", ascending=False, kind="stable")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    csv_frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    payload = {
        "q1": q1,
        "q2": train_bundle["summary"],
        "q3": q3,
        "q4": q4,
        "q5": q5,
        "q6": q6,
        "q7": q7,
        "go_no_go": go_no_go_block,
    }
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    print("TEST reads: 0; models loaded/trained/rescored: 0; datasets written: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
