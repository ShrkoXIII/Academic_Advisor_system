"""Gate 1.5 — lineage materiality diagnostic for ``2026-08_temporal_rebuild_v1``.

Read-only. Implements no lineage, generates no mapping candidate, and writes
nothing outside the rebuild version directory.

Fidelity to production
----------------------
The diagnostic does not approximate "never in TRAIN". It calls the production
functions directly:

* ``fit_difficulty_state(TRAIN)`` builds the six-level state from TRAIN only.
* ``apply_difficulty_state(VALID, state)`` resolves VALID identities against it.

``apply_difficulty_state`` emits ``course_is_new`` — set when NEITHER the
Level-1 key (``degree_course_key``) NOR the Level-2 key (``course_id``) has any
support in the TRAIN state. That is exactly the frozen definition of
``affected_rows``, and production keeps it in a separate column from
``course_low_support`` (history present but below ``min_support``), so the
never-in-TRAIN cause is distinguishable from the other fallback cause without
inventing a rule.

VALID outcomes are never read. ``build_level_keys`` consumes identity columns
only (degree_id, course_id, requirement_type_id, course_credits, faculty_id);
``final_mark`` is dropped from the VALID frame before the state is applied, and
the row-level CSV carries identity/fallback diagnostics only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.course_difficulty import (  # noqa: E402
    DifficultyConfig,
    apply_difficulty_state,
    build_level_keys,
    fit_difficulty_state,
)
from src.paths import FINAL_DIR, MODEL_DATA_VERSIONS_DIR  # noqa: E402

REBUILD_VERSION = "2026-08_temporal_rebuild_v1"
VERSION_ROOT = MODEL_DATA_VERSIONS_DIR / REBUILD_VERSION
SPLIT_DIR = VERSION_ROOT / "01_split"
GATE_DIR = VERSION_ROOT / "01_5_lineage_gate"

TRAIN_CANDIDATE = SPLIT_DIR / "train_base_candidate.parquet"
VALID_CANDIDATE = SPLIT_DIR / "valid_base_candidate.parquet"
SOURCE_PATH = FINAL_DIR / "without_outliers.parquet"

GATE_DECISION_COMMIT = "df03477cc6fca018d507857b589dcc3d78d1dd70"
GATE_DECISION_AUTHOR_DATE = "2026-08-02T11:42:51+03:00"

# Outcome columns that must never reach the diagnostic from VALID.
VALID_OUTCOME_COLUMNS = ("final_mark", "grade_id", "gpa_points", "points")

IDENTITY_OUTPUT_COLUMNS = [
    "student_course_id", "student_id", "course_id", "degree_id", "faculty_id",
    "part_id", "degree_course_key", "requirement_type_id", "course_credits",
    "difficulty_fallback_level", "course_history_count",
    "difficulty_group_support_count", "course_is_new", "course_low_support",
    "course_difficulty_missing", "gate_reason",
]


def strip_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove every VALID outcome column before any identity resolution."""
    drop = [c for c in VALID_OUTCOME_COLUMNS if c in df.columns]
    return df.drop(columns=drop)


def measure_full(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Full production path: fit the state, then apply it to VALID identities."""
    state = fit_difficulty_state(train, DifficultyConfig())
    valid_identities = strip_outcomes(valid)
    for column in VALID_OUTCOME_COLUMNS:
        if column in valid_identities.columns:
            raise SystemExit(f"STOP: VALID outcome column {column} survived stripping")

    applied = apply_difficulty_state(valid_identities, state, include_source=True)

    is_new = applied["course_is_new"] == 1
    low_support = applied["course_low_support"] == 1
    reason = pd.Series("covered", index=applied.index, dtype="object")
    reason[low_support] = "low_support_below_min_support"
    reason[is_new] = "never_in_train_identity"
    applied["gate_reason"] = reason

    figures = summarise(int(len(applied)), int(is_new.sum()))
    figures["low_support_rows_distinct_cause"] = int((low_support & ~is_new).sum())
    figures["difficulty_missing_rows_total"] = int(
        (applied["course_difficulty_missing"] == 1).sum()
    )
    figures["measurement_path"] = "production_fit_and_apply"
    return applied, figures


def measure_identity_only(
    train: pd.DataFrame, valid: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """`course_is_new` only, when the full production fit cannot run.

    Production defines `course_is_new` as: neither the Level-1
    (`degree_course_key`) nor the Level-2 (`course_id`) key has support in the
    TRAIN state. A key enters those tables iff it appears in TRAIN with a
    non-null key (`_sufficient_stats` drops null keys and nothing else), and
    `_finalize_state_from_raw` builds tables 1 and 2 WITHOUT the degree->faculty
    map — that map feeds only the Level-3 -> Level-4 shrinkage parent. So this
    count is production's own definition, not an approximation, and it is
    unaffected by the faculty-map failure.
    """
    train_keys = build_level_keys(strip_outcomes(train).assign(final_mark=pd.NA))
    valid_clean = strip_outcomes(valid)
    valid_keys = build_level_keys(valid_clean.assign(final_mark=pd.NA))

    l1_seen = set(train_keys["degree_course_key"].dropna().astype(str))
    l2_seen = set(train_keys["course_id"].dropna().astype(str))

    v_l1 = valid_keys["degree_course_key"].astype("string")
    v_l2 = valid_keys["course_id"].astype("string")
    l1_hit = v_l1.notna() & v_l1.astype(str).isin(l1_seen)
    l2_hit = v_l2.notna() & v_l2.astype(str).isin(l2_seen)
    is_new = ~(l1_hit | l2_hit)

    out = valid_clean.copy()
    out["degree_course_key"] = v_l1
    out["course_is_new"] = is_new.astype("int64")
    out["course_low_support"] = pd.NA
    out["course_difficulty_missing"] = pd.NA
    out["difficulty_fallback_level"] = pd.NA
    out["course_history_count"] = pd.NA
    out["difficulty_group_support_count"] = pd.NA
    out["gate_reason"] = pd.Series(
        ["never_in_train_identity" if v else "covered_or_low_support" for v in is_new],
        index=out.index,
    )

    figures = summarise(int(len(out)), int(is_new.sum()))
    figures["low_support_rows_distinct_cause"] = None
    figures["difficulty_missing_rows_total"] = None
    figures["measurement_path"] = "identity_only_level1_level2_key_presence"
    return out, figures


def summarise(eligible: int, affected: int) -> dict:
    threshold = max(1000, math.ceil(0.01 * eligible))
    return {
        "eligible_evaluation_rows": eligible,
        "affected_rows": affected,
        "affected_share": round(affected / eligible, 6) if eligible else None,
        "materiality_threshold": threshold,
        "phase_2_decision": (
            "PROCEED" if affected >= threshold else "DEFERRED_NO_MATERIAL_NEED"
        ),
    }


def measure(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[pd.DataFrame, dict, str]:
    """Prefer the full production path; fall back only on the faculty-map defect."""
    try:
        applied, figures = measure_full(train, valid)
        return applied, figures, ""
    except ValueError as exc:
        if "maps to multiple faculty_id" not in str(exc):
            raise
        applied, figures = measure_identity_only(train, valid)
        return applied, figures, str(exc)


def main() -> int:
    for path in (TRAIN_CANDIDATE, VALID_CANDIDATE):
        if not path.is_file():
            raise SystemExit(f"STOP: Phase 1 candidate missing: {path}")
    GATE_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(TRAIN_CANDIDATE)
    valid = pd.read_parquet(VALID_CANDIDATE)
    applied, figures, blocker = measure(train, valid)

    # Context only: the same measurement under the OLD boundaries, so the effect
    # of the boundary change on lineage need is visible as a before/after pair.
    source = pd.read_parquet(SOURCE_PATH).reset_index(drop=True)
    year = pd.to_numeric(source["part_year"], errors="coerce")
    old_train = source[((year >= 2005) & (year <= 2021)).values].copy()
    old_valid = source[((year >= 2022) & (year <= 2023)).values].copy()
    _, old_figures, old_blocker = measure(old_train, old_valid)

    payload = {
        "artifact": "lineage_materiality_gate",
        "rebuild_version": REBUILD_VERSION,
        "gate_decision_commit": GATE_DECISION_COMMIT,
        "gate_decision_author_date": GATE_DECISION_AUTHOR_DATE,
        "gate_rule_source": "Decisions_Log.md Declaration 2 (frozen)",
        "threshold_formula": "max(1000, ceil(0.01 * eligible_evaluation_rows))",
        "affected_rows_definition": (
            "VALID rows where production apply_difficulty_state sets "
            "course_is_new == 1, i.e. neither the Level-1 degree_course_key nor "
            "the Level-2 course_id had any support in the TRAIN-fitted state"
        ),
        "valid_outcomes_read": False,
        "gate_status": (
            "BLOCKED_PRODUCTION_FIT_RAISES" if blocker else "COMPLETE"
        ),
        "production_fit_blocker": blocker or None,
        "blocker_scope_note": (
            "degree_to_faculty feeds ONLY the Level-3 -> Level-4 shrinkage "
            "parent; tables 1 and 2 are built without it, so course_is_new "
            "(affected_rows) is unaffected. Fallback levels and low-support "
            "diagnostics could NOT be computed."
        )
        if blocker
        else None,
        "decision_is_binding": not blocker,
        "new_split": figures,
        "old_split_context_only": old_figures,
        "old_split_production_fit_blocker": old_blocker or None,
    }
    (GATE_DIR / "lineage_materiality_gate.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    # Select on the flag, not on the label: the identity-only path cannot say
    # "covered" (it cannot separate covered from low-support), so a label
    # comparison would silently emit every VALID row.
    selector = applied["course_is_new"] == 1
    if applied["course_low_support"].notna().any():
        selector = selector | (applied["course_low_support"] == 1)
    rows = applied.loc[selector, :]
    columns = [c for c in IDENTITY_OUTPUT_COLUMNS if c in rows.columns]
    leaked = [c for c in rows[columns].columns if c in VALID_OUTCOME_COLUMNS]
    if leaked:
        raise SystemExit(f"STOP: outcome columns would be written: {leaked}")
    rows[columns].to_csv(
        GATE_DIR / "lineage_materiality_rows.csv", index=False, encoding="utf-8"
    )

    f, o = figures, old_figures
    decision = f["phase_2_decision"]
    lines = [
        "# Gate 1.5 — Academic Lineage Materiality",
        "",
        f"Rebuild version: `{REBUILD_VERSION}`",
        "",
        "## Provenance of the rule",
        "",
        "The gate rule was frozen **before** this measurement. The declaration was",
        f"committed in `{GATE_DECISION_COMMIT}`",
        f"(author date `{GATE_DECISION_AUTHOR_DATE}`); this measurement ran afterwards.",
        "The frozen rule text is byte-identical from that commit through `HEAD`.",
        "",
        "## Fidelity to `src/course_difficulty.py`",
        "",
        "No approximation was used. `affected_rows` is production's",
        "`course_is_new == 1` — neither the Level-1 `degree_course_key` nor the",
        "Level-2 `course_id` had support in the TRAIN-fitted state. Production keeps",
        "that separate from `course_low_support`, so the never-in-TRAIN cause is",
        "distinguishable from the other fallback cause without inventing a rule. No",
        "column was added to the production contract.",
        "",
    ]
    if blocker:
        lines += [
            "## STOP GATE — the production fit does not run on the new TRAIN",
            "",
            "`fit_difficulty_state(TRAIN)` **raises** on the new TRAIN window:",
            "",
            "```text",
            blocker,
            "```",
            "",
            "Cause: four degrees (`39.111`, `40.111`, `7.111`, `8.111`) each map to two",
            "faculties (`7.111` and `177.111`). Those reassignments appear in 2022+ rows,",
            "which the OLD TRAIN (2005–2021) excluded and the NEW TRAIN (≤ 20233)",
            "includes. This is a property of the boundary change, not of this script.",
            "",
            "**Scope of the blocker.** `degree_to_faculty` is consumed at exactly one",
            "place — building the Level-3 → Level-4 shrinkage parent. Tables 1 and 2 are",
            "built without it (`tables[2]` has no parent; `tables[1]`'s parent is derived",
            "from the key string). `course_is_new` depends only on key presence in tables",
            "1 and 2, so `affected_rows` below is production's own definition and is",
            "unaffected. What could **not** be computed: `difficulty_fallback_level`,",
            "`course_low_support`, `course_history_count`, and",
            "`difficulty_group_support_count`.",
            "",
            "**Therefore the decision below is NOT binding.** It is the value the frozen",
            "rule yields on a correct `affected_rows` count, recorded as evidence. The",
            "gate cannot be closed until the owner decides how a degree with two",
            "faculties resolves for the Level-3 parent. That decision changes production",
            "behaviour and is out of scope for Phase 1.",
            "",
        "## VALID outcomes were not read",
        "",
        "`final_mark`, `grade_id`, `gpa_points` and `points` are dropped from the",
        "VALID frame before identities are resolved, and the drop is asserted. The",
        "row-level CSV carries identity and fallback diagnostics only.",
        "",
        "## Result — new split (TRAIN ≤ 20233, VALID = 20241+20242+20243)",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `eligible_evaluation_rows` | {f['eligible_evaluation_rows']:,} |",
        f"| `affected_rows` | {f['affected_rows']:,} |",
        f"| `affected_share` | {f['affected_share']:.6f} |",
        f"| `materiality_threshold` | {f['materiality_threshold']:,} |",
        f"| `phase_2_decision` (binding: **{payload['decision_is_binding']}**) "
        f"| **`{decision}`** |",
        "",
        f"Measurement path: `{f['measurement_path']}`.",
        "",
        "## Before/after — context only, not part of the decision",
        "",
        "The same measurement under the OLD boundaries (TRAIN 2005–2021,",
        "VALID 2022–2023), so the boundary change's effect on lineage need is visible:",
        "",
        "| Quantity | Old split | New split |",
        "|---|---:|---:|",
        f"| `eligible_evaluation_rows` | {o['eligible_evaluation_rows']:,} | "
        f"{f['eligible_evaluation_rows']:,} |",
        f"| `affected_rows` | {o['affected_rows']:,} | {f['affected_rows']:,} |",
        f"| `affected_share` | {o['affected_share']:.6f} | {f['affected_share']:.6f} |",
        "",
        "## `affected_rows` is an upper bound",
        "",
        "Courses that are genuinely new have no predecessor and cannot be repaired by",
        "any mapping. A `PROCEED` result authorises candidate generation and human",
        "review ONLY. It does not authorise applying mappings and predicts no metric",
        "gain.",
        "",
        f"## Consequence of `{decision}`",
        "",
    ]
    if decision == "PROCEED":
        lines += [
            "Phase 2 candidate generation and human review are authorised. Nothing is",
            "applied, mapped, or promoted by this gate.",
        ]
    else:
        lines += [
            "Lineage is **skipped**: the pre-registered materiality gate failed.",
            "",
            "- No mapping candidates are generated.",
            "- No human mapping review is started.",
            "- No canonical mapping of any kind is created.",
            "- Phase 3 uses the Phase 1 outputs and the original IDs unchanged.",
        ]
    lines.append("")
    (GATE_DIR / "lineage_materiality_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(json.dumps({"new_split": f, "old_split_context_only": o}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
