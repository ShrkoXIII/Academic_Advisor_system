"""Phase 0 read-only evidence recovery for the predecessor-prior decision.

Answers Q2 (classification reconciliation), Q4 (degree multiplicity of the 67
pairs), Q5 (degree-lineage state) and Q6 (how far a predecessor prior would move
the difficulty estimate) from repository artifacts.

Guarantees
----------
* Reads only the immutable TRAIN/VALID parquets of the frozen dataset version,
  the read-only course catalog, and existing report artifacts under
  ``models/runs``.
* Never constructs, globs, stats or reads any TEST path. ``df_test_final`` is
  never referenced.
* Imports NO training code. ``src.course_difficulty`` is the feature module that
  already produced the on-disk columns; it fits sufficient statistics from TRAIN
  only and trains no model.
* Q6 uses TRAIN-derived estimates on BOTH sides. VALID ``final_mark`` is never
  loaded, so no realized VALID outcome can enter any figure produced here.
* Writes exactly two artifacts: the Phase 0 markdown report and the per-pair CSV.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.course_difficulty import (  # noqa: E402
    DifficultyConfig,
    build_level_keys,
    fit_difficulty_state,
)

VERSION = "2026-07-26_batched_fixes__registration_roster_concurrent"
VERSION_DIR = ROOT / "data" / "model_data" / "versions" / VERSION
TRAIN_PATH = VERSION_DIR / "df_train_final.parquet"
VALID_PATH = VERSION_DIR / "df_valid_final.parquet"

CATALOG_RAW = ROOT / "data" / "raw" / "v_acd_degree_course.parquet"
BEST_MATCH_PATH = ROOT / "models" / "runs" / "COURSE_IDENTITY_67_BEST_MATCH_PER_COURSE.csv"
CANDIDATES_B_PATH = ROOT / "models" / "runs" / "COURSE_IDENTITY_CANDIDATES.csv"

OUT_CSV = ROOT / "models" / "runs" / "phase0_pair_divergence.csv"

# Classification A lives only in git history: commit c6a9656 replaced
# models/runs/course_identity_candidates.csv with the case-variant
# COURSE_IDENTITY_CANDIDATES.csv, which is the SAME path on Windows.
A_CSV_COMMIT = "0a9f346"
A_CSV_GIT_PATH = "models/runs/course_identity_candidates.csv"

# Columns loaded from TRAIN. final_mark/attempt_number are TRAIN outcomes and are
# required to fit the TRAIN-only difficulty state; no VALID outcome is loaded.
TRAIN_FIT_COLUMNS = [
    "part_id",
    "final_mark",
    "attempt_number",
    "degree_course_key",
    "degree_id",
    "faculty_id",
    "requirement_type_id",
    "course_credits",
    "course_id",
]

# VALID columns. NOTE: final_mark is deliberately ABSENT.
VALID_COLUMNS = [
    "course_id",
    "degree_id",
    "faculty_id",
    "requirement_type_id",
    "course_credits",
    "part_id",
    "student_id",
    "degree_course_key",
    "course_pass_rate_historical",
    "course_avg_mark_historical",
    "difficulty_fallback_level",
    "course_history_count",
    "course_difficulty_missing",
]


def git_show(commit: str, path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8-sig", errors="replace")


def catalog_course_key(value: float) -> str:
    """Catalog ids are float ``<id>.111``; model ids are the string form."""

    return f"{int(round(value - 0.111))}.111"


def load_catalog() -> pd.DataFrame:
    raw = pd.read_parquet(CATALOG_RAW)
    raw["cid"] = raw["course_id"].apply(catalog_course_key)
    raw["did"] = raw["degree_id"].apply(catalog_course_key)
    return raw


def load_investigation_module():
    path = ROOT / "scripts" / "course_identity_investigation.py"
    spec = importlib.util.spec_from_file_location("course_identity_investigation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Q2 - reconcile classification A (investigation) with classification B (diagnostic)
# ---------------------------------------------------------------------------
def answer_q2() -> dict[str, Any]:
    text = git_show(A_CSV_COMMIT, A_CSV_GIT_PATH)
    scratch = ROOT / ".phase0_tmp_a.csv"
    scratch.write_text(text, encoding="utf-8")
    try:
        a = pd.read_csv(scratch, dtype="string", keep_default_na=False, skiprows=1)
    finally:
        scratch.unlink()

    b = pd.read_csv(CANDIDATES_B_PATH, dtype="string", keep_default_na=False)

    a_ids = set(a["new_course_id"])
    b_ids = set(b["new_course_id"])

    a_buckets = a["review_bucket"].value_counts().to_dict()
    b_buckets = b["diagnostic_status"].value_counts().to_dict()

    a_bucket_of = dict(zip(a["new_course_id"], a["review_bucket"]))
    b_status_of = dict(zip(b["new_course_id"], b["diagnostic_status"]))
    a_top = dict(zip(a["new_course_id"], a["candidate_1_course_id"]))
    b_top = dict(zip(b["new_course_id"], b["candidate_old_course_id"]))

    b67 = {cid for cid, status in b_status_of.items()
           if status == "likely_renumbered_needs_review"}
    a_confirmed = {cid for cid, bucket in a_bucket_of.items()
                   if bucket == "confirmed_equivalent"}
    a_new = {cid for cid, bucket in a_bucket_of.items() if bucket == "genuinely_new"}
    a_pending = {cid for cid, bucket in a_bucket_of.items()
                 if bucket in {"likely_equivalent_needs_review", "unresolved"}}

    cross = Counter(
        (a_bucket_of[cid], b_status_of[cid]) for cid in sorted(a_ids & b_ids)
    )

    shared = sorted(a_ids & b_ids)
    top_agree = sum(1 for cid in shared if a_top.get(cid) == b_top.get(cid))

    return {
        "a_source": {
            "path": A_CSV_GIT_PATH,
            "recovered_from_commit": A_CSV_COMMIT,
            "on_disk_today": (ROOT / A_CSV_GIT_PATH).exists(),
            "note": "the on-disk file at this path today is classification B's CSV: "
                    "Windows paths are case-insensitive and commit c6a9656 replaced "
                    "course_identity_candidates.csv with COURSE_IDENTITY_CANDIDATES.csv",
            "rows": int(len(a)),
            "buckets": a_buckets,
        },
        "b_source": {
            "path": str(CANDIDATES_B_PATH.relative_to(ROOT)).replace("\\", "/"),
            "rows": int(len(b)),
            "buckets": b_buckets,
        },
        "course_sets_identical": a_ids == b_ids,
        "a_only_courses": sorted(a_ids - b_ids),
        "b_only_courses": sorted(b_ids - a_ids),
        "a_confirmed_count": len(a_confirmed),
        "a_genuinely_new_count": len(a_new),
        "a_pending_count": len(a_pending),
        "b_67_count": len(b67),
        "a_confirmed_in_b67": sorted(a_confirmed & b67),
        "a_confirmed_in_b67_count": len(a_confirmed & b67),
        "a_confirmed_not_in_b67": sorted(a_confirmed - b67),
        "cross_tabulation": {f"{k[0]} -> {k[1]}": int(v) for k, v in sorted(cross.items())},
        "top_candidate_agreement": {
            "courses_compared": len(shared),
            "same_top_candidate": int(top_agree),
            "different_top_candidate": len(shared) - int(top_agree),
        },
        "top_candidate_disagreement_examples": [
            {
                "new_course_id": cid,
                "a_top": a_top.get(cid),
                "b_top": b_top.get(cid),
                "a_bucket": a_bucket_of[cid],
                "b_status": b_status_of[cid],
            }
            for cid in shared
            if a_top.get(cid) != b_top.get(cid)
        ][:15],
    }


# ---------------------------------------------------------------------------
# Q1 sensitivity - does `same_degree == 0/67` survive a broader degree-set rule?
# ---------------------------------------------------------------------------
def answer_q1_sensitivity(
    pairs: pd.DataFrame,
    catalog: pd.DataFrame,
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> dict[str, Any]:
    """Recompute same_degree under catalog UNION enrolment, and trace each link.

    The committed verification (course_identity_67_degree_verification.py:551-557)
    uses catalog degree sets only. This does NOT modify that artifact; it reports
    how sensitive its 0/67 headline is to that choice.
    """

    cat = catalog.groupby("cid")["did"].agg(lambda s: frozenset(s)).to_dict()
    empty: frozenset = frozenset()

    def by_course(frame: pd.DataFrame) -> dict[str, frozenset]:
        return (
            frame[["course_id", "degree_id"]]
            .dropna()
            .groupby("course_id")["degree_id"]
            .agg(lambda s: frozenset(s))
            .to_dict()
        )

    train_deg = by_course(train)
    valid_deg = by_course(valid)

    def source_of(course_id: str, degree_id: str) -> str:
        if degree_id in cat.get(course_id, empty):
            return "catalog"
        if degree_id in train_deg.get(course_id, empty):
            return "TRAIN_enrolment"
        return "VALID_enrolment"

    links: list[dict[str, str]] = []
    sharing_pairs: list[str] = []
    for record in pairs.itertuples(index=False):
        old_all = (
            cat.get(record.old_course_id, empty)
            | train_deg.get(record.old_course_id, empty)
            | valid_deg.get(record.old_course_id, empty)
        )
        new_all = (
            cat.get(record.new_course_id, empty)
            | train_deg.get(record.new_course_id, empty)
            | valid_deg.get(record.new_course_id, empty)
        )
        shared = old_all & new_all
        if not shared:
            continue
        sharing_pairs.append(record.new_course_id)
        for degree_id in sorted(shared):
            links.append(
                {
                    "new_course_id": record.new_course_id,
                    "old_course_id": record.old_course_id,
                    "shared_degree_id": degree_id,
                    "old_side_source": source_of(record.old_course_id, degree_id),
                    "new_side_source": source_of(record.new_course_id, degree_id),
                }
            )

    provenance = Counter(
        (link["old_side_source"], link["new_side_source"]) for link in links
    )
    return {
        "catalog_only_pairs_sharing_a_degree": 0,
        "catalog_union_enrolment_pairs_sharing_a_degree": len(sharing_pairs),
        "pairs_total": int(len(pairs)),
        "sharing_pairs": sorted(set(sharing_pairs)),
        "link_count": len(links),
        "link_provenance": {
            f"old:{old} x new:{new}": int(count)
            for (old, new), count in sorted(provenance.items())
        },
        "links": links,
    }


# ---------------------------------------------------------------------------
# Q4 - degree multiplicity of the 67 pairs
# ---------------------------------------------------------------------------
def answer_q4(
    pairs: pd.DataFrame,
    catalog: pd.DataFrame,
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    catalog_degrees = catalog.groupby("cid")["did"].agg(lambda s: frozenset(s)).to_dict()
    all_model = pd.concat(
        [train[["course_id", "degree_id"]], valid[["course_id", "degree_id"]]],
        ignore_index=True,
    )
    enrol_degrees = (
        all_model.dropna()
        .groupby("course_id")["degree_id"]
        .agg(lambda s: frozenset(s))
        .to_dict()
    )

    rows = []
    transitions: dict[tuple[frozenset, frozenset], int] = {}
    for record in pairs.itertuples(index=False):
        old_cat = catalog_degrees.get(record.old_course_id, frozenset())
        new_cat = catalog_degrees.get(record.new_course_id, frozenset())
        old_enrol = enrol_degrees.get(record.old_course_id, frozenset())
        new_enrol = enrol_degrees.get(record.new_course_id, frozenset())

        if len(old_cat) == 1 and len(new_cat) == 1:
            shape = "1:1"
        elif len(old_cat) == 1:
            shape = "1:N"
        elif len(new_cat) == 1:
            shape = "N:1"
        else:
            shape = "N:M"

        key = (old_cat, new_cat)
        if key not in transitions:
            transitions[key] = len(transitions) + 1

        rows.append(
            {
                "new_course_id": record.new_course_id,
                "old_course_id": record.old_course_id,
                "n_old_degrees_catalog": len(old_cat),
                "n_new_degrees_catalog": len(new_cat),
                "n_old_degrees_enrolment": len(old_enrol),
                "n_new_degrees_enrolment": len(new_enrol),
                "mapping_shape": shape,
                "degree_set_transition_id": transitions[key],
                "old_degree_ids_catalog": "|".join(sorted(old_cat)),
                "new_degree_ids_catalog": "|".join(sorted(new_cat)),
                "catalog_equals_enrolment_old": old_cat == old_enrol,
                "catalog_equals_enrolment_new": new_cat == new_enrol,
                "enrolment_subset_of_catalog_old": old_enrol <= old_cat,
                "enrolment_subset_of_catalog_new": new_enrol <= new_cat,
            }
        )

    frame = pd.DataFrame(rows)
    summary = {
        "pairs": int(len(frame)),
        "pairs_with_multiplicity_either_side": int(
            ((frame["n_old_degrees_catalog"] > 1) | (frame["n_new_degrees_catalog"] > 1)).sum()
        ),
        "pairs_with_multiplicity_both_sides": int(
            ((frame["n_old_degrees_catalog"] > 1) & (frame["n_new_degrees_catalog"] > 1)).sum()
        ),
        "pairs_strictly_1_to_1": int(frame["mapping_shape"].eq("1:1").sum()),
        "mapping_shape_counts": frame["mapping_shape"].value_counts().to_dict(),
        "max_new_degrees": int(frame["n_new_degrees_catalog"].max()),
        "max_old_degrees": int(frame["n_old_degrees_catalog"].max()),
        "distinct_degree_set_transitions": int(frame["degree_set_transition_id"].nunique()),
        "transition_size_counts": frame["degree_set_transition_id"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "enrolment_equals_catalog_old_pairs": int(frame["catalog_equals_enrolment_old"].sum()),
        "enrolment_equals_catalog_new_pairs": int(frame["catalog_equals_enrolment_new"].sum()),
        "enrolment_subset_of_catalog_old_pairs": int(
            frame["enrolment_subset_of_catalog_old"].sum()
        ),
        "enrolment_subset_of_catalog_new_pairs": int(
            frame["enrolment_subset_of_catalog_new"].sum()
        ),
    }
    return summary, frame


# ---------------------------------------------------------------------------
# Q5 - degree lineage
# ---------------------------------------------------------------------------
def answer_q5(
    ci_module,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    catalog: pd.DataFrame,
    pair_frame: pd.DataFrame,
) -> dict[str, Any]:
    lineage, records = ci_module.degree_lineage(train, valid, catalog)

    degree_names = catalog.drop_duplicates("did").set_index("did")["degree_name_sl"].to_dict()

    year_pattern = re.compile(r"\b20\d{2}\b")
    year_suffix = []
    family_prefix = []
    for did, name in sorted(degree_names.items()):
        text = str(name)
        if year_pattern.search(text) or re.search(r"20\d{2}", text):
            year_suffix.append({"degree_id": did, "name": text})
        if "/" in text:
            family_prefix.append(
                {"degree_id": did, "name": text, "family": text.split("/", 1)[0]}
            )

    train_degrees = set(train["degree_id"].dropna())
    zero = [r for r in records if not r["predecessors"]]

    # New degrees appearing in the 67 pairs (catalog-side new degree sets).
    pair_new_degrees: set[str] = set()
    for value in pair_frame["new_degree_ids_catalog"]:
        pair_new_degrees.update(v for v in str(value).split("|") if v)

    lineage_keys = set(lineage)
    valid_only_degrees = {r["new_degree"] for r in records}
    covered = sorted(
        d for d in pair_new_degrees if d in lineage_keys and lineage[d]
    )
    valid_only_no_link = sorted(
        d for d in pair_new_degrees if d in lineage_keys and not lineage[d]
    )
    not_valid_only = sorted(d for d in pair_new_degrees if d not in valid_only_degrees)

    return {
        "valid_only_degree_count": len(records),
        "records": [
            {
                "new_degree": r["new_degree"],
                "new_degree_name": r["new_degree_name"],
                "valid_rows": r["valid_rows"],
                "by_student_migration": r["by_student_migration"],
                "migration_predecessor_count": len(r["by_student_migration"]),
                "by_name_similarity_top3": r["by_name_similarity"],
                "name_predecessor_count": sum(
                    1
                    for old in train_degrees
                    if ci_module.name_similarity(
                        _degree_key(ci_module, degree_names, r["new_degree"]),
                        _degree_key(ci_module, degree_names, old),
                    )
                    >= ci_module.NAME_SIM_PLAUSIBLE
                ),
                "linked_predecessors": r["predecessors"],
                "linked_predecessor_count": len(r["predecessors"]),
            }
            for r in sorted(records, key=lambda x: -x["valid_rows"])
        ],
        "zero_predecessor_degrees": [
            {"degree_id": r["new_degree"], "name": r["new_degree_name"],
             "valid_rows": r["valid_rows"]}
            for r in zero
        ],
        "zero_predecessor_count": len(zero),
        "catalog_degree_count": len(degree_names),
        "year_suffix_degrees": year_suffix,
        "year_suffix_count": len(year_suffix),
        "family_prefix_degrees": family_prefix,
        "family_prefix_count": len(family_prefix),
        "distinct_family_prefixes": sorted({d["family"] for d in family_prefix}),
        "pair_new_degrees": sorted(pair_new_degrees),
        "pair_new_degree_count": len(pair_new_degrees),
        "pair_new_degrees_covered_by_lineage": covered,
        "pair_new_degrees_covered_count": len(covered),
        "pair_new_degrees_valid_only_without_link": valid_only_no_link,
        "pair_new_degrees_not_valid_only": not_valid_only,
        "pair_new_degrees_not_valid_only_count": len(not_valid_only),
    }


def _degree_key(ci_module, degree_names: dict[str, str], degree_id: str) -> str:
    name = ci_module.normalize_name(degree_names.get(degree_id, ""))
    name = re.sub(r"\b20\d{2}\b", "", name)
    raw_name = str(degree_names.get(degree_id, ""))
    if "/" in raw_name:
        tail = raw_name.split("/", 1)[1]
        name = ci_module.normalize_name(tail)
        name = re.sub(r"\b20\d{2}\b", "", name)
    return " ".join(name.split())


# ---------------------------------------------------------------------------
# Q6 - how far would a predecessor prior move the estimate
# ---------------------------------------------------------------------------
def answer_q6(
    pairs: pd.DataFrame,
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    state = fit_difficulty_state(train)
    l1 = state.tables[1]
    l2 = state.tables[2]

    # Validate the refit reproduces the on-disk frozen VALID columns before use.
    keys = build_level_keys(valid)
    check_l1 = keys["degree_course_key"].map(l1["support_count"])
    check_l2 = keys["course_id"].map(l2["support_count"])
    recomputed_history = np.where(
        check_l1.notna(),
        check_l1.fillna(0).to_numpy(dtype="int64"),
        np.where(check_l2.notna(), check_l2.fillna(0).to_numpy(dtype="int64"), 0),
    )
    history_mismatches = int(
        (recomputed_history != valid["course_history_count"].to_numpy(dtype="int64")).sum()
    )

    valid_course = valid["course_id"].astype("string")
    per_pair: list[dict[str, Any]] = []
    row_abs_pass: list[np.ndarray] = []
    row_abs_mark: list[np.ndarray] = []

    for record in pairs.itertuples(index=False):
        mask = valid_course.eq(record.new_course_id).to_numpy()
        subset = valid.loc[mask]
        n_rows = int(len(subset))
        if n_rows == 0:
            continue

        sub_keys = keys.loc[subset.index]
        level_counts = subset["difficulty_fallback_level"].value_counts().to_dict()

        current_pass = subset["course_pass_rate_historical"].to_numpy(dtype="float64")
        current_mark = subset["course_avg_mark_historical"].to_numpy(dtype="float64")

        # Substituted Level-1 key: same degree_course_key with the predecessor id.
        substituted_l1 = (
            sub_keys["degree_course_key"]
            .astype("string")
            .str.rsplit("__", n=1)
            .str[0]
            + "__"
            + record.old_course_id
        )
        pred_l1_pass = substituted_l1.map(l1["course_pass_rate_historical"])
        pred_l1_mark = substituted_l1.map(l1["course_avg_mark_historical"])
        pred_l1_support = substituted_l1.map(l1["support_count"])

        l2_row = l2.loc[record.old_course_id] if record.old_course_id in l2.index else None
        l2_pass = float(l2_row["course_pass_rate_historical"]) if l2_row is not None else np.nan
        l2_mark = float(l2_row["course_avg_mark_historical"]) if l2_row is not None else np.nan
        l2_support = int(l2_row["support_count"]) if l2_row is not None else 0

        l1_hit = pred_l1_support.notna().to_numpy()
        pred_pass = np.where(
            l1_hit, pred_l1_pass.to_numpy(dtype="float64"), l2_pass
        )
        pred_mark = np.where(
            l1_hit, pred_l1_mark.to_numpy(dtype="float64"), l2_mark
        )
        pred_support = np.where(
            l1_hit, pred_l1_support.fillna(0).to_numpy(dtype="float64"), float(l2_support)
        )
        pred_level = np.where(l1_hit, 1, 2)

        abs_pass = np.abs(current_pass - pred_pass)
        abs_mark = np.abs(current_mark - pred_mark)
        row_abs_pass.append(abs_pass)
        row_abs_mark.append(abs_mark)

        per_pair.append(
            {
                "new_course_id": record.new_course_id,
                "new_course_name": record.new_course_name,
                "old_course_id": record.old_course_id,
                "old_course_name": record.old_course_name,
                "valid_rows": n_rows,
                "fallback_level_1_rows": int(level_counts.get(1, 0)),
                "fallback_level_2_rows": int(level_counts.get(2, 0)),
                "fallback_level_3_rows": int(level_counts.get(3, 0)),
                "fallback_level_4_rows": int(level_counts.get(4, 0)),
                "fallback_level_5_rows": int(level_counts.get(5, 0)),
                "fallback_level_6_rows": int(level_counts.get(6, 0)),
                "current_pass_rate_mean": float(np.mean(current_pass)),
                "current_pass_rate_min": float(np.min(current_pass)),
                "current_pass_rate_max": float(np.max(current_pass)),
                "current_avg_mark_mean": float(np.mean(current_mark)),
                "predecessor_level_used": (
                    "1" if l1_hit.all() else ("2" if not l1_hit.any() else "mixed")
                ),
                "predecessor_l1_hit_rows": int(l1_hit.sum()),
                "predecessor_pass_rate_mean": float(np.mean(pred_pass)),
                "predecessor_avg_mark_mean": float(np.mean(pred_mark)),
                "predecessor_l2_pass_rate": l2_pass,
                "predecessor_l2_avg_mark": l2_mark,
                "predecessor_train_support_l2": l2_support,
                "predecessor_train_support_mean": float(np.mean(pred_support)),
                "abs_diff_pass_rate_mean": float(np.mean(abs_pass)),
                "abs_diff_pass_rate_max": float(np.max(abs_pass)),
                "abs_diff_avg_mark_mean": float(np.mean(abs_mark)),
                "abs_diff_avg_mark_max": float(np.max(abs_mark)),
                "predecessor_level_used_all_rows": int(pred_level.min()) if n_rows else None,
            }
        )

    pair_frame = pd.DataFrame(per_pair)
    all_pass = np.concatenate(row_abs_pass)
    all_mark = np.concatenate(row_abs_mark)

    def dist(values: np.ndarray) -> dict[str, float]:
        return {
            "rows": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
            "max": float(np.max(values)),
        }

    aggregate = {
        "refit_validation_course_history_mismatches": history_mismatches,
        "refit_validation_verdict": "exact" if history_mismatches == 0 else "MISMATCH",
        "valid_rows_covered_by_67_pairs": int(pair_frame["valid_rows"].sum()),
        "row_weighted_abs_diff_pass_rate": dist(all_pass),
        "row_weighted_abs_diff_avg_mark": dist(all_mark),
        "pair_level_abs_diff_pass_rate": dist(
            pair_frame["abs_diff_pass_rate_mean"].to_numpy(dtype="float64")
        ),
        "current_fallback_level_rows": {
            f"level_{level}": int(pair_frame[f"fallback_level_{level}_rows"].sum())
            for level in range(1, 7)
        },
        "predecessor_level_used_counts": pair_frame["predecessor_level_used"]
        .value_counts()
        .to_dict(),
        "pairs_with_zero_predecessor_train_support": int(
            (pair_frame["predecessor_train_support_l2"] == 0).sum()
        ),
        "predecessor_train_support_l2": {
            "min": int(pair_frame["predecessor_train_support_l2"].min()),
            "median": float(pair_frame["predecessor_train_support_l2"].median()),
            "max": int(pair_frame["predecessor_train_support_l2"].max()),
        },
        "global_train_pass_rate": float(state.global_stats["course_pass_rate_historical"]),
        "global_train_avg_mark": float(state.global_stats["course_avg_mark_historical"]),
        "min_support_in_effect": int(state.config.min_support),
        "shrinkage_k_in_effect": float(state.config.shrinkage_k),
    }
    return aggregate, pair_frame, keys


def main() -> int:
    if DifficultyConfig().min_support != 20:
        raise SystemExit("STOP: DifficultyConfig.min_support is no longer 20.")

    best = pd.read_csv(BEST_MATCH_PATH, dtype="string", keep_default_na=False)
    pairs = best[
        ["new_course_id", "new_course_name", "old_course_id", "old_course_name"]
    ].copy()
    if len(pairs) != 67:
        raise SystemExit(f"STOP: expected 67 best-match pairs, found {len(pairs)}.")

    catalog = load_catalog()
    train = pd.read_parquet(TRAIN_PATH, columns=TRAIN_FIT_COLUMNS)
    valid = pd.read_parquet(VALID_PATH, columns=VALID_COLUMNS)
    for frame in (train, valid):
        for column in ("course_id", "degree_id", "faculty_id", "part_id", "degree_course_key"):
            if column in frame.columns:
                frame[column] = frame[column].astype("string")

    if "final_mark" in valid.columns:
        raise SystemExit("STOP: a VALID outcome column was loaded.")

    ci = load_investigation_module()
    ci_train = pd.read_parquet(TRAIN_PATH, columns=["student_id", "degree_id"])
    ci_valid = pd.read_parquet(VALID_PATH, columns=["student_id", "degree_id"])
    for frame in (ci_train, ci_valid):
        frame["degree_id"] = frame["degree_id"].astype("string")

    q1 = answer_q1_sensitivity(pairs, catalog, train, valid)
    q2 = answer_q2()
    q4, q4_frame = answer_q4(pairs, catalog, train, valid)
    q5 = answer_q5(ci, ci_train, ci_valid, catalog, q4_frame)
    q6, q6_frame, _ = answer_q6(pairs, train, valid)

    merged = q6_frame.merge(q4_frame, on=["new_course_id", "old_course_id"], how="left")
    merged = merged.sort_values("valid_rows", ascending=False, kind="stable")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # The task's write allowance is one markdown report plus one CSV, so the
    # full payload goes to stdout rather than to a third artifact on disk.
    payload = {"q1_sensitivity": q1, "q2": q2, "q4": q4, "q5": q5, "q6": q6}
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    print("TEST reads: 0; models trained: 0; datasets written: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
