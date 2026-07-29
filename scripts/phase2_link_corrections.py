"""Phase 2T: three narrow corrections to the frozen Phase 2S link table.

This is intentionally a patch, not a rebuild.  It reads the Phase 2S proposal
CSVs, the reviewed-pair artifact, and the cleaned catalog needed for audit
fields.  It does not read TRAIN, VALID, TEST, or any model artifact.

The Phase 2S split, lineage, and temporal-prototype files are carried forward
as bytes.  The only relationship/provenance change is discovered generically:
an in-lineage, support-eligible row still carrying manual provenance is passed
through the same ordinary relationship rule as every other in-lineage row.
All validation completes before the five Phase 2T outputs are written.
"""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

import phase2_mapping_tables_scope_fix as phase2s


ROOT = Path(__file__).resolve().parents[1]
PHASE2S_DIR = ROOT / "models" / "runs" / "phase2_lineage_scope_fix"
PHASE2S_REPORT = PHASE2S_DIR / "PHASE2_MAPPING_TABLES_SCOPE_FIX.md"
PHASE2S_LINK = PHASE2S_DIR / "course_link_proposed.csv"
PHASE2S_SPLIT = PHASE2S_DIR / "course_split_candidates.csv"
PHASE2S_LINEAGE = PHASE2S_DIR / "degree_lineage_proposed.csv"
PHASE2S_STATS = PHASE2S_DIR / "course_difficulty_stats_prototype.csv"

CATALOG_PATH = phase2s.CATALOG_PATH
PAIRS_PATH = phase2s.PAIRS_PATH

OUT_DIR = ROOT / "models" / "runs" / "phase2_link_corrections"
OUT_REPORT = OUT_DIR / "PHASE2_MAPPING_TABLES_CORRECTED.md"
OUT_LINK = OUT_DIR / "course_link_proposed.csv"
OUT_SPLIT = OUT_DIR / "course_split_candidates.csv"
OUT_LINEAGE = OUT_DIR / "degree_lineage_proposed.csv"
OUT_STATS = OUT_DIR / "course_difficulty_stats_prototype.csv"

CARRY_FORWARD = {
    "course_split_candidates.csv": (PHASE2S_SPLIT, OUT_SPLIT),
    "degree_lineage_proposed.csv": (PHASE2S_LINEAGE, OUT_LINEAGE),
    "course_difficulty_stats_prototype.csv": (PHASE2S_STATS, OUT_STATS),
}
EXPECTED_OUTPUT_NAMES = {
    OUT_REPORT.name,
    OUT_LINK.name,
    *CARRY_FORWARD.keys(),
}

MIN_SUPPORT = phase2s.MIN_SUPPORT
KNOWN_PAIR_CATEGORIES = [
    "automatic eligible link",
    "split_or_merge",
    "manual proposal",
    "candidate_below_support",
    "name_only_review_candidate",
    "unresolved",
]

EXPECTED_PHASE2S_COVERAGE_COURSES = 82
EXPECTED_PHASE2S_COVERAGE_ROWS = 17_036
EXPECTED_GLOBAL_UPPER_COURSES = 83
EXPECTED_GLOBAL_UPPER_ROWS = 17_814
EXPECTED_NEW_COURSES = 182
EXPECTED_NEW_ROWS = 25_627
EXPECTED_CHANGED_CREDIT_COURSES = 20
EXPECTED_CHANGED_CREDIT_ROWS = 8_302
EXPECTED_CHANGED_DIRECTIONS = {
    "2 → 3": 13,
    "4 → 3": 6,
    "3 → 2": 1,
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_inputs() -> dict[str, Any]:
    """Read Phase 2S artifacts and catalog evidence; no model split is accessed."""

    required = [
        PHASE2S_REPORT,
        PHASE2S_LINK,
        PHASE2S_SPLIT,
        PHASE2S_LINEAGE,
        PHASE2S_STATS,
        CATALOG_PATH,
        PAIRS_PATH,
    ]
    for path in required:
        lowered = str(path).lower().replace("\\", "/")
        if "df_test" in lowered or "/test/" in lowered:
            raise SystemExit(f"STOP: a TEST path entered the input allowlist: {path}")
        if not path.is_file():
            raise SystemExit(f"STOP: required Phase 2T input is missing: {path}")

    baseline_link = pd.read_csv(
        PHASE2S_LINK, dtype="string", keep_default_na=False
    )
    lineage = pd.read_csv(
        PHASE2S_LINEAGE, dtype="string", keep_default_na=False
    )
    split = pd.read_csv(
        PHASE2S_SPLIT, dtype="string", keep_default_na=False
    )
    pairs = pd.read_csv(PAIRS_PATH, dtype="string", keep_default_na=False)
    catalog = pd.read_parquet(
        CATALOG_PATH,
        columns=[
            "course_id",
            "degree_id",
            "course_credits",
            "requirement_type_id",
        ],
    )
    for column in ("course_id", "degree_id"):
        catalog[column] = catalog[column].astype("string")
    catalog["course_credits"] = pd.to_numeric(
        catalog["course_credits"], errors="coerce"
    )
    catalog["requirement_type_id"] = pd.to_numeric(
        catalog["requirement_type_id"], errors="coerce"
    ).astype("Int64")

    carry_bytes = {
        name: source.read_bytes()
        for name, (source, _) in CARRY_FORWARD.items()
    }
    return {
        "baseline_link": baseline_link,
        "lineage": lineage,
        "split": split,
        "pairs": pairs,
        "catalog": catalog,
        "carry_bytes": carry_bytes,
    }


def numeric_weight(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(parsed) else float(parsed)


def relation_counter(frame: pd.DataFrame) -> Counter[tuple[Any, ...]]:
    tuples = []
    for row in frame.itertuples(index=False):
        tuples.append(
            (
                str(row.new_course_id),
                str(row.old_course_id),
                str(row.relationship_type),
                numeric_weight(row.weight_hint),
            )
        )
    return Counter(tuples)


def split_ids(value: Any) -> list[str]:
    return [
        token
        for token in str(value).split("|")
        if token and token.lower() != "<na>"
    ]


def ranked_match_for_row(
    row: pd.Series, lineage: pd.DataFrame
) -> int | None:
    new_degrees = set(split_ids(row["new_course_degree_ids"]))
    old_degrees = set(split_ids(row["old_course_degree_ids"]))
    candidates = lineage.loc[
        lineage["new_degree_id"].isin(new_degrees)
        & lineage["old_degree_id"].isin(old_degrees),
        "candidate_rank",
    ]
    ranks = pd.to_numeric(candidates, errors="coerce").dropna()
    return int(ranks.min()) if not ranks.empty else None


def apply_ordinary_in_lineage_rules(
    baseline: pd.DataFrame, lineage: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove stale manual provenance without naming a course pair."""

    corrected = baseline.copy()
    support = pd.to_numeric(
        corrected["old_course_train_support"], errors="coerce"
    )
    ordinary_mask = (
        corrected["old_course_id"].ne("")
        & corrected["scope_confidence"].eq("in_lineage")
        & support.ge(MIN_SUPPORT)
        & corrected["new_course_scope"].isin(["specific", "shared"])
        & ~corrected["relationship_type"].isin(
            [
                "split_from",
                "merged_from",
                "candidate_below_support",
                "name_only_review_candidate",
                "none",
            ]
        )
    )
    legacy_mask = ordinary_mask & (
        corrected["match_method"].eq("manual")
        | corrected["relationship_type"].eq("manual_candidate")
    )
    legacy_indices = corrected.index[legacy_mask].tolist()
    if len(legacy_indices) != 1:
        raise SystemExit(
            "STOP: expected exactly one in-lineage ordinary candidate with stale "
            f"manual provenance; found {len(legacy_indices)}."
        )

    for _, group in corrected.loc[ordinary_mask].groupby(
        "new_course_id", sort=False
    ):
        expected_relationship = (
            "successor" if len(group) == 1 else "consolidated_into"
        )
        group_support = pd.to_numeric(
            group["old_course_train_support"], errors="raise"
        ).astype(float)
        total_support = float(group_support.sum())
        expected_weights = (
            pd.Series(1.0, index=group.index)
            if expected_relationship == "successor"
            else group_support / total_support
        )
        for index in group.index:
            expected_weight = float(expected_weights.loc[index])
            current_weight = numeric_weight(corrected.at[index, "weight_hint"])
            if current_weight is None or not np.isclose(
                current_weight, expected_weight, atol=1e-12, rtol=0.0
            ):
                raise SystemExit(
                    "STOP: an existing in-lineage ordinary weight disagrees "
                    "with the unchanged TRAIN-volume rule."
                )
            if index not in legacy_indices:
                if corrected.at[index, "relationship_type"] != expected_relationship:
                    raise SystemExit(
                        "STOP: a non-manual Phase 2S ordinary relationship "
                        "would change during the Phase 2T patch."
                    )
                continue

            corrected.at[index, "relationship_type"] = expected_relationship
            corrected.at[index, "predecessor_count_for_new_course"] = str(
                len(group)
            )
            corrected.at[index, "match_method"] = (
                "name_key_union_lineage"
                if corrected.at[index, "new_course_scope"] == "specific"
                else "name_key_global"
            )
            rank = ranked_match_for_row(corrected.loc[index], lineage)
            corrected.at[index, "lineage_rank_matched"] = (
                "" if rank is None else str(rank)
            )
            corrected.at[index, "notes"] = (
                "TRAIN-supported normalized-name match is catalogued inside "
                "the lineage scope and was generated by the ordinary automatic "
                "rule; proposal remains pending."
            )

    before_counter = relation_counter(baseline)
    after_counter = relation_counter(corrected)
    removed = list((before_counter - after_counter).elements())
    added = list((after_counter - before_counter).elements())
    if len(removed) != 1 or len(added) != 1:
        raise SystemExit(
            "STOP: relationship/weight tuple diff is not exactly one removed "
            "and one added tuple."
        )
    removed_tuple, added_tuple = removed[0], added[0]
    if (
        removed_tuple[:2] != added_tuple[:2]
        or removed_tuple[3] != added_tuple[3]
        or removed_tuple[2] != "manual_candidate"
        or added_tuple[2] not in {"successor", "consolidated_into"}
    ):
        raise SystemExit(
            "STOP: the sole relationship/weight tuple change is not the generic "
            "manual-to-ordinary provenance correction."
        )

    changed_index = legacy_indices[0]
    before_row = baseline.loc[changed_index].copy()
    after_row = corrected.loc[changed_index].copy()
    companion_rows = corrected.loc[
        corrected["new_course_id"].eq(str(after_row["new_course_id"]))
        & corrected["old_course_id"].ne(str(after_row["old_course_id"]))
        & corrected["scope_confidence"].eq("cross_faculty")
        & corrected["relationship_type"].eq("name_only_review_candidate")
        & corrected["weight_hint"].eq("")
    ].copy()
    if companion_rows.empty:
        raise SystemExit(
            "STOP: the corrected course lost its cross-faculty, unweighted "
            "name-only review candidate."
        )

    return corrected, {
        "before_row": before_row,
        "after_row": after_row,
        "companion_rows": companion_rows,
        "removed_tuple": removed_tuple,
        "added_tuple": added_tuple,
        "changed_index": changed_index,
    }


def tri_state_changed(
    left: pd.Series, right: pd.Series
) -> pd.Series:
    known = left.notna() & right.notna()
    equal = pd.Series(False, index=left.index)
    equal.loc[known] = np.isclose(
        pd.to_numeric(left.loc[known], errors="coerce").astype(float),
        pd.to_numeric(right.loc[known], errors="coerce").astype(float),
        atol=1e-12,
        rtol=0.0,
    )
    result = pd.Series("unknown", index=left.index, dtype="string")
    result.loc[known & equal] = "false"
    result.loc[known & ~equal] = "true"
    return result


def add_credit_audit_fields(
    corrected: pd.DataFrame, catalog: pd.DataFrame
) -> pd.DataFrame:
    """Add audit-only fields after all relationship logic has completed."""

    output = corrected.copy()
    meta = (
        catalog.sort_values(
            ["course_id", "degree_id"],
            key=lambda values: values.map(phase2s.p2r.id_sort_key),
        )
        .drop_duplicates("course_id")
        .set_index("course_id")
    )
    old_credit_map = meta["course_credits"].to_dict()
    old_requirement_map = (
        catalog.groupby("course_id")["requirement_type_id"]
        .apply(phase2s.p2r.modal)
        .to_dict()
    )

    old_credits = pd.to_numeric(
        output["old_course_id"].map(old_credit_map), errors="coerce"
    )
    new_credits = pd.to_numeric(
        output["new_course_credits"], errors="coerce"
    )
    credit_change = new_credits - old_credits
    credit_changed = tri_state_changed(old_credits, new_credits)

    old_requirement = pd.to_numeric(
        output["old_course_id"].map(old_requirement_map), errors="coerce"
    )
    new_requirement = pd.to_numeric(
        output["new_course_requirement_type_id"], errors="coerce"
    )
    requirement_changed = tri_state_changed(
        old_requirement, new_requirement
    )

    insert_at = output.columns.get_loc("old_course_name_key") + 1
    output.insert(insert_at, "old_course_credits", old_credits)
    output.insert(insert_at + 1, "credit_change", credit_change)
    output.insert(insert_at + 2, "credit_changed", credit_changed)
    output.insert(
        insert_at + 3,
        "requirement_type_changed",
        requirement_changed,
    )
    required = {
        "old_course_credits",
        "new_course_credits",
        "credit_change",
        "credit_changed",
        "requirement_type_changed",
    }
    if not required.issubset(output.columns):
        raise SystemExit("STOP: one or more credit audit fields are missing.")
    return output


def credit_direction_summary(link: pd.DataFrame) -> dict[str, Any]:
    weight = pd.to_numeric(link["weight_hint"], errors="coerce")
    weighted = link.loc[
        link["relationship_type"].isin(["successor", "consolidated_into"])
        & weight.notna()
    ].copy()
    weighted["old_credit_numeric"] = pd.to_numeric(
        weighted["old_course_credits"], errors="coerce"
    )
    weighted["new_credit_numeric"] = pd.to_numeric(
        weighted["new_course_credits"], errors="coerce"
    )

    def credit_text(value: Any) -> str:
        if pd.isna(value):
            return "unknown"
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:g}"

    weighted["direction"] = weighted.apply(
        lambda row: (
            f"{credit_text(row.old_credit_numeric)} → "
            f"{credit_text(row.new_credit_numeric)}"
        ),
        axis=1,
    )
    row_map = (
        link.drop_duplicates("new_course_id")
        .set_index("new_course_id")["new_course_valid_rows"]
        .map(lambda value: int(float(value)))
    )
    rows = []
    for direction, group in weighted.groupby("direction", sort=True):
        course_ids = set(group["new_course_id"].astype(str))
        rows.append(
            {
                "direction": direction,
                "weighted_link_rows": len(group),
                "new_course_ids": len(course_ids),
                "valid_rows": int(row_map.reindex(list(course_ids)).sum()),
                "credit_changed": (
                    "unknown"
                    if "unknown" in direction
                    else "false"
                    if direction.split(" → ")[0]
                    == direction.split(" → ")[1]
                    else "true"
                ),
            }
        )
    direction_table = pd.DataFrame(rows)
    changed_ids = set(
        weighted.loc[
            weighted["credit_changed"].eq("true"), "new_course_id"
        ].astype(str)
    )
    changed_rows = int(row_map.reindex(list(changed_ids)).sum())
    actual_changed = (
        direction_table.loc[direction_table["credit_changed"] == "true"]
        .set_index("direction")["new_course_ids"]
        .to_dict()
    )
    if (
        len(changed_ids) != EXPECTED_CHANGED_CREDIT_COURSES
        or changed_rows != EXPECTED_CHANGED_CREDIT_ROWS
        or actual_changed != EXPECTED_CHANGED_DIRECTIONS
    ):
        raise SystemExit(
            "STOP: weighted credit-change audit does not reproduce 20 courses / "
            "8,302 rows with the preregistered direction counts."
        )
    weighted_course_ids = set(weighted["new_course_id"].astype(str))
    return {
        "table": direction_table,
        "changed_course_ids": changed_ids,
        "changed_rows": changed_rows,
        "weighted_course_ids": weighted_course_ids,
        "weighted_rows": int(
            row_map.reindex(list(weighted_course_ids)).sum()
        ),
        "weighted_link_rows": len(weighted),
    }


def classify_known_pair(matches: pd.DataFrame) -> str:
    weight = pd.to_numeric(matches["weight_hint"], errors="coerce")
    automatic = (
        matches["relationship_type"].isin(["successor", "consolidated_into"])
        & matches["scope_confidence"].eq("in_lineage")
        & weight.notna()
    )
    if automatic.any():
        return "automatic eligible link"
    if matches["relationship_type"].isin(["split_from", "merged_from"]).any():
        return "split_or_merge"
    if (
        matches["relationship_type"].eq("manual_candidate").any()
        or matches["match_method"].eq("manual").any()
    ):
        return "manual proposal"
    if matches["relationship_type"].eq("candidate_below_support").any():
        return "candidate_below_support"
    if (
        matches["relationship_type"].eq("name_only_review_candidate").any()
        or matches["match_method"].isin(
            [
                "out_of_lineage_review_candidate",
                "cross_faculty_name_only_review_candidate",
            ]
        ).any()
    ):
        return "name_only_review_candidate"
    return "unresolved"


def known_pair_census(
    pairs: pd.DataFrame, link: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows = []
    for pair in pairs.itertuples(index=False):
        matches = link.loc[
            link["new_course_id"].eq(pair.new_course_id)
            & link["old_course_id"].eq(pair.old_course_id)
        ]
        rows.append(
            {
                "new_course_id": pair.new_course_id,
                "old_course_id": pair.old_course_id,
                "category": classify_known_pair(matches),
            }
        )
    frame = pd.DataFrame(rows)
    counts = {
        category: int(frame["category"].eq(category).sum())
        for category in KNOWN_PAIR_CATEGORIES
    }
    if (
        len(frame) != phase2s.EXPECTED_KNOWN_PAIRS
        or sum(counts.values()) != phase2s.EXPECTED_KNOWN_PAIRS
        or frame[["new_course_id", "old_course_id"]].duplicated().any()
    ):
        raise SystemExit(
            "STOP: known-pair census is not exclusive and exactly 67."
        )
    return frame, counts


def rank_distribution(link: pd.DataFrame) -> pd.DataFrame:
    matched = link.loc[link["old_course_id"].ne("")].copy()
    ranks = pd.to_numeric(matched["lineage_rank_matched"], errors="coerce")
    rows = []
    for rank in (1, 2, 3, None):
        group = matched.loc[ranks.isna()] if rank is None else matched.loc[ranks.eq(rank)]
        rows.append(
            {
                "lineage_rank_matched": "null" if rank is None else str(rank),
                "proposal_rows": len(group),
                "new_course_ids": group["new_course_id"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def coverage_measure(
    course_ids: set[str], row_map: pd.Series
) -> dict[str, Any]:
    rows = int(row_map.reindex(list(course_ids)).sum())
    return {
        "course_ids": len(course_ids),
        "rows": rows,
        "pct": rows / EXPECTED_NEW_ROWS,
    }


def union_and_rank1_coverage(link: pd.DataFrame) -> dict[str, Any]:
    weight = pd.to_numeric(link["weight_hint"], errors="coerce")
    automatic = (
        link["relationship_type"].isin(["successor", "consolidated_into"])
        & link["scope_confidence"].eq("in_lineage")
        & weight.notna()
    )
    structural = link["relationship_type"].isin(
        ["split_from", "merged_from"]
    )
    manual = link["relationship_type"].eq("manual_candidate")
    union_ids = set(
        link.loc[automatic | structural | manual, "new_course_id"].astype(str)
    )

    ranks = pd.to_numeric(link["lineage_rank_matched"], errors="coerce")
    rank1_automatic = automatic & (
        link["new_course_scope"].eq("shared")
        | ranks.eq(1)
        | (link["new_course_scope"].eq("specific") & ranks.isna())
    )
    rank1_ids = set(
        link.loc[
            rank1_automatic | structural | manual, "new_course_id"
        ].astype(str)
    )
    row_map = (
        link.drop_duplicates("new_course_id")
        .set_index("new_course_id")["new_course_valid_rows"]
        .map(lambda value: int(float(value)))
    )
    union_measure = coverage_measure(union_ids, row_map)
    rank1_measure = coverage_measure(rank1_ids, row_map)
    if (
        union_measure["course_ids"] != EXPECTED_PHASE2S_COVERAGE_COURSES
        or union_measure["rows"] != EXPECTED_PHASE2S_COVERAGE_ROWS
    ):
        raise SystemExit(
            "STOP: Phase 2T union-scope coverage differs from the frozen "
            "Phase 2S 82 / 17,036 result."
        )
    return {
        "union": union_measure,
        "rank1": rank1_measure,
        "course_difference": (
            union_measure["course_ids"] - rank1_measure["course_ids"]
        ),
        "row_difference": union_measure["rows"] - rank1_measure["rows"],
    }


def conditional_literal_scan(
    script_path: Path,
) -> dict[str, Any]:
    """AST-backed grep for course-like literals inside conditional expressions."""

    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script_path))
    literal_pattern = re.compile(r"\b\d+\.111\b")
    tests: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp, ast.While, ast.Assert)):
            tests.append(node.test)
        elif isinstance(node, ast.comprehension):
            tests.extend(node.ifs)
        elif isinstance(node, ast.match_case) and node.guard is not None:
            tests.append(node.guard)

    ast_hits = []
    for test in tests:
        for value in ast.walk(test):
            if (
                isinstance(value, ast.Constant)
                and isinstance(value.value, str)
                and literal_pattern.search(value.value)
            ):
                ast_hits.append(
                    {
                        "line": getattr(value, "lineno", None),
                        "literal": value.value,
                    }
                )

    line_pattern = re.compile(
        r"^\s*(?:if|elif|while|assert)\b.*[\"']\d+\.111"
    )
    line_hits = [
        {"line": number, "text": line.strip()}
        for number, line in enumerate(source.splitlines(), start=1)
        if line_pattern.search(line)
    ]
    if ast_hits or line_hits:
        raise SystemExit(
            "STOP: course-ID literal remains in a Phase 2T conditional: "
            f"ast={ast_hits}, line_scan={line_hits}"
        )
    return {
        "ast_hits": ast_hits,
        "line_hits": line_hits,
        "tests_scanned": len(tests),
        "pattern": line_pattern.pattern,
    }


def format_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def md_escape(value: Any) -> str:
    if value is None or pd.isna(value) or str(value) == "":
        return "null"
    return str(value).replace("|", "\\|").replace("\n", " ")


def tuple_text(value: tuple[Any, ...]) -> str:
    new_id, old_id, relationship, weight = value
    return (
        f"(`{new_id}`, `{old_id}`, `{relationship}`, "
        f"`{'null' if weight is None else weight}`)"
    )


def build_report(
    baseline: pd.DataFrame,
    corrected: pd.DataFrame,
    change: dict[str, Any],
    gate: dict[str, Any],
    before_census: dict[str, int],
    after_census: dict[str, int],
    credit: dict[str, Any],
    before_ranks: pd.DataFrame,
    after_ranks: pd.DataFrame,
    coverage: dict[str, Any],
    scan: dict[str, Any],
    carry_bytes: dict[str, bytes],
) -> str:
    before_row: pd.Series = change["before_row"]
    after_row = corrected.loc[change["changed_index"]]
    companion_rows: pd.DataFrame = change["companion_rows"]
    changed_new_id = str(after_row["new_course_id"])
    changed_old_id = str(after_row["old_course_id"])
    audit_columns = {
        "old_course_credits",
        "credit_change",
        "credit_changed",
        "requirement_type_changed",
    }
    requirement_counts = (
        corrected["requirement_type_changed"].value_counts(dropna=False).to_dict()
    )

    lines: list[str] = [
        "# Phase 2T — link table corrections",
        "",
        "Status: **proposal tables only**. Every proposal remains pending; no "
        "mapping was approved or applied.",
        "",
        "## What changed from Phase 2S",
        "",
        f"- Correction 1: the sole stale in-lineage manual row became an ordinary "
        f"`successor`; known-pair automatic links changed "
        f"{before_census['automatic eligible link']} → "
        f"{after_census['automatic eligible link']} and manual proposals changed "
        f"{before_census['manual proposal']} → {after_census['manual proposal']}.",
        f"- Correction 2: five credit/requirement audit fields are now present "
        "(`new_course_credits` was already present; four columns were added). "
        f"Credit changes touch {len(credit['changed_course_ids'])} weighted courses / "
        f"{credit['changed_rows']:,} VALID rows; relationship and weight changes "
        "outside Correction 1 are zero.",
        f"- Correction 3: rank-1 matched rows changed "
        f"{int(before_ranks.loc[before_ranks['lineage_rank_matched'] == '1', 'proposal_rows'].iloc[0])} "
        f"→ {int(after_ranks.loc[after_ranks['lineage_rank_matched'] == '1', 'proposal_rows'].iloc[0])} "
        "because Correction 1 gained its ordinary rank; ranks 2 and 3 remain zero, "
        f"and rank-1-only coverage differs from union coverage by "
        f"{coverage['course_difference']} courses / {coverage['row_difference']:,} rows.",
        "",
        f"## Correction 1 — {changed_new_id}",
        "",
        f"The ordinary rule now emits `{changed_old_id} → {changed_new_id}` without "
        "any course-pair conditional. Full row before and after:",
        "",
        "| Field | Phase 2S before | Phase 2T after |",
        "|---|---|---|",
    ]
    for column in corrected.columns:
        before_value = (
            "not present"
            if column in audit_columns
            else md_escape(before_row.get(column, ""))
        )
        after_value = md_escape(after_row.get(column, ""))
        lines.append(
            f"| `{column}` | {before_value} | {after_value} |"
        )

    lines.extend(
        [
            "",
            "Other same-name candidates for this new course remain visible:",
            "",
            "| Old course | Name | TRAIN support | Relationship | Confidence | Weight |",
            "|---|---|---:|---|---|---:|",
        ]
    )
    for row in companion_rows.itertuples(index=False):
        lines.append(
            f"| `{row.old_course_id}` | {md_escape(row.old_course_name)} | "
            f"{int(float(row.old_course_train_support))} | "
            f"`{row.relationship_type}` | `{row.scope_confidence}` | null |"
        )

    lines.extend(
        [
            "",
            "Grep result:",
            "",
            f"- AST conditional tests scanned: **{scan['tests_scanned']}**.",
            "- Course-ID literals found in conditionals: **0**.",
            f"- Grep-style line matches for `{md_escape(scan['pattern'])}`: **0**.",
            "- Pair-specific conditional branches remaining in the Phase 2T script: **0**.",
            "",
            "Known-pair census delta:",
            "",
            "| Category | Phase 2S | Phase 2T | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for category in KNOWN_PAIR_CATEGORIES:
        before = before_census[category]
        after = after_census[category]
        lines.append(
            f"| `{category}` | {before} | {after} | {after - before:+d} |"
        )
    lines.extend(
        [
            f"| **Total** | **{sum(before_census.values())}** | "
            f"**{sum(after_census.values())}** | **0** |",
            "",
            "## Correction 2 — credit-change audit",
            "",
            "The audit fields are descriptive only. They are populated after all "
            "relationship, scope, ranking, support, and weighting logic completes.",
            "",
            "| Credit direction | Credit changed | Weighted link rows | Distinct new courses | VALID rows |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in credit["table"].itertuples(index=False):
        lines.append(
            f"| `{row.direction}` | `{row.credit_changed}` | "
            f"{row.weighted_link_rows} | {row.new_course_ids} | "
            f"{row.valid_rows:,} |"
        )
    lines.extend(
        [
            "",
            "Direction rows are non-additive for consolidations: one new course can "
            "have weighted predecessors with both unchanged and changed credits. "
            f"Across distinct courses, exactly **{len(credit['changed_course_ids'])}** "
            f"weighted courses / **{credit['changed_rows']:,}** VALID rows have at "
            "least one changed-credit predecessor.",
            "",
            f"Requirement-type audit row counts: `false` = "
            f"{int(requirement_counts.get('false', 0))}, `true` = "
            f"{int(requirement_counts.get('true', 0))}, `unknown` = "
            f"{int(requirement_counts.get('unknown', 0))}.",
            "",
            "Relationship/weight tuple assertion:",
            "",
            f"- Removed: {tuple_text(change['removed_tuple'])}",
            f"- Added: {tuple_text(change['added_tuple'])}",
            "- Every other `(new_course_id, old_course_id, relationship_type, "
            "weight_hint)` tuple is identical to Phase 2S.",
            "- The audit fields changed no relationship, rank, filter, or weight.",
            "",
            "## Correction 3 — rank-2/3 counterfactual",
            "",
            "| Rank | Phase 2S rows | Phase 2S courses | Phase 2T rows | Phase 2T courses |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    merged_ranks = before_ranks.merge(
        after_ranks,
        on="lineage_rank_matched",
        suffixes=("_phase2s", "_phase2t"),
        validate="one_to_one",
    )
    for row in merged_ranks.itertuples(index=False):
        lines.append(
            f"| `{row.lineage_rank_matched}` | {row.proposal_rows_phase2s} | "
            f"{row.new_course_ids_phase2s} | {row.proposal_rows_phase2t} | "
            f"{row.new_course_ids_phase2t} |"
        )
    lines.extend(
        [
            "",
            "| Scope used for diagnostic | Covered courses | VALID rows | Coverage |",
            "|---|---:|---:|---:|",
            f"| Emitted union scope | {coverage['union']['course_ids']} | "
            f"{coverage['union']['rows']:,} | {format_pct(coverage['union']['pct'])} |",
            f"| Counterfactual rank-1 only | {coverage['rank1']['course_ids']} | "
            f"{coverage['rank1']['rows']:,} | {format_pct(coverage['rank1']['pct'])} |",
            f"| **Difference** | **{coverage['course_difference']}** | "
            f"**{coverage['row_difference']:,}** | **0.0 pp** |",
            "",
            "The emitted table still uses union scope. The counterfactual confirms "
            "that ranks 2 and 3 contribute zero matched rows and zero coverage. "
            "The ancestry eligibility filter—not the union breadth—is the mechanism "
            "that recovered all 26 preregistered pairs.",
            "",
            "## Unchanged carry-forward",
            "",
            "Coverage:",
            "",
            "| Measurement | Course IDs | VALID rows | % of 25,627 | Source |",
            "|---|---:|---:|---:|---|",
            f"| Phase 2S union scope | {EXPECTED_PHASE2S_COVERAGE_COURSES} | "
            f"{EXPECTED_PHASE2S_COVERAGE_ROWS:,} | "
            f"{format_pct(EXPECTED_PHASE2S_COVERAGE_ROWS / EXPECTED_NEW_ROWS)} | "
            "Phase 2S report |",
            f"| Phase 2T corrected link | {coverage['union']['course_ids']} | "
            f"{coverage['union']['rows']:,} | {format_pct(coverage['union']['pct'])} | "
            "Phase 2T tuple census |",
            f"| Global name-key upper bound | {EXPECTED_GLOBAL_UPPER_COURSES} | "
            f"{EXPECTED_GLOBAL_UPPER_ROWS:,} | "
            f"{format_pct(EXPECTED_GLOBAL_UPPER_ROWS / EXPECTED_NEW_ROWS)} | "
            "Phase 2S report |",
            "",
            "Phase 2T six-category known-pair census:",
            "",
            "| Category | Pairs |",
            "|---|---:|",
        ]
    )
    for category in KNOWN_PAIR_CATEGORIES:
        lines.append(f"| `{category}` | {after_census[category]} |")
    lines.extend(
        [
            f"| **Total** | **{sum(after_census.values())}** |",
            "",
            "Validation gates carried forward or rechecked:",
            "",
            "| Gate | Result | Source |",
            "|---|---:|---|",
            f"| Normalization | {gate['matches']} / {gate['pairs']} | "
            "67-pair artifact + unchanged normalizer |",
            f"| Sole normalization non-match | `{gate['nonmatch_pair']}` | "
            "67-pair artifact + unchanged normalizer |",
            "| Ancestry-ineligible degree count | 1 (`49.111`, old catalog courses 0) | "
            "byte-identical Phase 2S lineage/report |",
            "| Measured recoverable pairs | 26 / 26 (7,335 rows) | "
            "Phase 2S result + one-tuple diff proof |",
            "| Never-in-TRAIN census | 182 courses / 25,627 rows | "
            "Phase 2S result + unchanged link course census |",
            "| Temporal validation | 0 / 156,097 mismatches | "
            "byte-identical Phase 2S prototype |",
            "| Proposal statuses | `pending` only | Phase 2T in-memory validation |",
            "| Consolidated weight sums | 1.0 | Phase 2T in-memory validation |",
            "| VALID/TEST/model access | None | Phase 2T input allowlist |",
            "",
            "Byte-identical carry-forward files:",
            "",
            "| File | Phase 2S SHA-256 | Phase 2T SHA-256 | Identical |",
            "|---|---|---|---:|",
        ]
    )
    for name in CARRY_FORWARD:
        digest = sha256_bytes(carry_bytes[name])
        lines.append(
            f"| `{name}` | `{digest}` | `{digest}` | yes |"
        )
    lines.extend(
        [
            "",
            "## Ready for human review",
            "",
            "- In `course_link_proposed.csv`, reviewers must assess every pending "
            "relationship using `scope_confidence`, the five credit/requirement audit "
            "fields, support, and weight. The corrected course above has both an "
            "in-lineage automatic predecessor and a cross-faculty unweighted candidate "
            "that should be considered together.",
            "- In `degree_lineage_proposed.csv`, reviewers must fill the approval "
            "decision for the unchanged pending lineage candidates.",
            "- In `course_split_candidates.csv`, reviewers must fill the approval "
            "decision for the unchanged pending structural candidates.",
            "- `course_difficulty_stats_prototype.csv` requires no approval entry and "
            "contains no applied link; it remains diagnostic only.",
            "",
            "## Governance entry (ready to copy)",
            "",
            "> Identifier namespaces overlap in this data: `49.111` is simultaneously an "
            "ancestry-ineligible degree and a TRAIN-present course with support 48. A "
            "Phase 2S prompt instruction conflated the two and forced a `manual` provenance "
            "label onto a link the ordinary rules discover automatically. Any identifier "
            "written into a specification must carry its namespace — course, degree, or "
            "faculty — explicitly.",
            "",
            "The union search scope contributed zero matched rows. The ancestry "
            "eligibility filter alone recovered all 26 pairs. Scope breadth was a "
            "remedy for a symptom; the eligibility constraint addressed the cause.",
            "",
            "No decision log was edited. No mapping was applied, and no Phase 3 action "
            "is proposed.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_in_memory(
    baseline: pd.DataFrame,
    corrected_core: pd.DataFrame,
    corrected_output: pd.DataFrame,
    lineage: pd.DataFrame,
    split: pd.DataFrame,
    change: dict[str, Any],
) -> None:
    for name, frame in [
        ("course link", corrected_output),
        ("lineage", lineage),
        ("split/merge", split),
    ]:
        if (
            "approval_status" not in frame.columns
            or not frame["approval_status"].eq("pending").all()
        ):
            raise SystemExit(f"STOP: {name} contains a non-pending proposal.")
    if corrected_output["new_course_id"].nunique() != EXPECTED_NEW_COURSES:
        raise SystemExit("STOP: corrected link table does not retain all 182 courses.")

    core_counter = relation_counter(corrected_core)
    output_counter = relation_counter(corrected_output)
    if core_counter != output_counter:
        raise SystemExit(
            "STOP: adding audit fields changed a relationship or weight tuple."
        )

    baseline_counter = relation_counter(baseline)
    removed = list((baseline_counter - output_counter).elements())
    added = list((output_counter - baseline_counter).elements())
    if removed != [change["removed_tuple"]] or added != [change["added_tuple"]]:
        raise SystemExit(
            "STOP: final tuple diff changed after credit audit fields were added."
        )

    weight = pd.to_numeric(corrected_output["weight_hint"], errors="coerce")
    consolidated = corrected_output.loc[
        corrected_output["relationship_type"].eq("consolidated_into")
        & weight.notna()
    ].copy()
    consolidated["_weight"] = pd.to_numeric(
        consolidated["weight_hint"], errors="raise"
    )
    sums = consolidated.groupby("new_course_id")["_weight"].sum()
    bad = sums.loc[(sums - 1.0).abs() > 1e-9]
    if not bad.empty:
        raise SystemExit(
            f"STOP: consolidated weights do not sum to 1.0: {bad.to_dict()}"
        )


def validate_output_paths() -> None:
    parent = OUT_DIR.resolve()
    for path in [OUT_REPORT, OUT_LINK, OUT_SPLIT, OUT_LINEAGE, OUT_STATS]:
        resolved = path.resolve()
        if resolved.parent != parent:
            raise SystemExit(f"STOP: output escaped Phase 2T directory: {path}")
        lowered = str(resolved).lower().replace("\\", "/")
        if "/src/" in lowered or "/data/model_data/" in lowered:
            raise SystemExit(f"STOP: forbidden output path: {path}")
    if parent == PHASE2S_DIR.resolve():
        raise SystemExit("STOP: Phase 2T output aliases the Phase 2S directory.")


def write_outputs(
    report: str,
    corrected: pd.DataFrame,
    carry_bytes: dict[str, bytes],
) -> dict[str, str]:
    """First mutation in the pipeline: write exactly five prevalidated payloads."""

    validate_output_paths()
    if OUT_DIR.exists():
        extras = {path.name for path in OUT_DIR.iterdir()} - EXPECTED_OUTPUT_NAMES
        if extras:
            raise SystemExit(
                "STOP: Phase 2T output directory contains unexpected files: "
                f"{sorted(extras)}"
            )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    link_payload = corrected.to_csv(index=False, lineterminator="\n").encode(
        "utf-8-sig"
    )
    report_payload = report.encode("utf-8")
    payloads = {
        OUT_REPORT: report_payload,
        OUT_LINK: link_payload,
        OUT_SPLIT: carry_bytes["course_split_candidates.csv"],
        OUT_LINEAGE: carry_bytes["degree_lineage_proposed.csv"],
        OUT_STATS: carry_bytes["course_difficulty_stats_prototype.csv"],
    }
    for path, payload in payloads.items():
        path.write_bytes(payload)

    actual = {path.name for path in OUT_DIR.iterdir() if path.is_file()}
    if actual != EXPECTED_OUTPUT_NAMES:
        raise SystemExit(
            f"STOP: output file census mismatch; actual={sorted(actual)}"
        )
    hashes = {
        path.name: sha256_bytes(path.read_bytes()) for path in payloads
    }
    for name, (_, destination) in CARRY_FORWARD.items():
        if destination.read_bytes() != carry_bytes[name]:
            raise SystemExit(
                f"STOP: carry-forward file is not byte-identical: {name}"
            )
    return hashes


def main() -> int:
    validate_output_paths()
    inputs = read_inputs()
    baseline: pd.DataFrame = inputs["baseline_link"]
    lineage: pd.DataFrame = inputs["lineage"]
    split: pd.DataFrame = inputs["split"]
    pairs: pd.DataFrame = inputs["pairs"]
    catalog: pd.DataFrame = inputs["catalog"]
    carry_bytes: dict[str, bytes] = inputs["carry_bytes"]

    _, gate = phase2s.p2r.validate_normalization_gate(pairs)
    corrected_core, change = apply_ordinary_in_lineage_rules(
        baseline, lineage
    )
    corrected_output = add_credit_audit_fields(corrected_core, catalog)

    _, before_census = known_pair_census(pairs, baseline)
    _, after_census = known_pair_census(pairs, corrected_output)
    if sum(after_census.values()) != phase2s.EXPECTED_KNOWN_PAIRS:
        raise SystemExit(
            "STOP: removing stale manual provenance changed the known-pair "
            "census total away from 67."
        )

    credit = credit_direction_summary(corrected_output)
    before_ranks = rank_distribution(baseline)
    after_ranks = rank_distribution(corrected_output)
    coverage = union_and_rank1_coverage(corrected_output)
    scan = conditional_literal_scan(Path(__file__).resolve())

    validate_in_memory(
        baseline,
        corrected_core,
        corrected_output,
        lineage,
        split,
        change,
    )
    report = build_report(
        baseline,
        corrected_output,
        change,
        gate,
        before_census,
        after_census,
        credit,
        before_ranks,
        after_ranks,
        coverage,
        scan,
        carry_bytes,
    )

    hashes = write_outputs(report, corrected_output, carry_bytes)
    print(f"Wrote exactly five Phase 2T outputs to: {OUT_DIR}")
    print(
        f"Known-pair census: {after_census}; coverage="
        f"{coverage['union']['course_ids']} courses / "
        f"{coverage['union']['rows']:,} rows; rank-1 counterfactual delta="
        f"{coverage['course_difference']} courses / "
        f"{coverage['row_difference']:,} rows."
    )
    print(
        f"Credit changes: {len(credit['changed_course_ids'])} weighted courses / "
        f"{credit['changed_rows']:,} rows; conditional course-ID literals=0; "
        f"output hashes={hashes}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
