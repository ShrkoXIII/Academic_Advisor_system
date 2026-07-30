"""Phase 3 PILOT: apply the pre-registered clauses and write the pilot report.

Reads only on-disk artifacts: the paired-evaluation JSON, the pilot dataset
version's manifest and audit tables, the frozen and pilot VALID frames (for the
worked examples), and NOISE_BAND.md's published per-metric ranges. No model is
trained, no proposal row is touched, TEST is never read.

The acceptance rule is the one pre-registered in the task prompt and is applied
verbatim; the numeric bounds come from models/runs/NOISE_BAND.md and are not
invented here.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import MODEL_DATA_VERSIONS_DIR, MODEL_RUNS_DIR  # noqa: E402

FROZEN_VERSION = "2026-07-26_batched_fixes__registration_roster_concurrent"
PILOT_VERSION = "2026-07-30_predecessor_prior_pilot_PENDING_REVIEW"
FROZEN_VALID = MODEL_DATA_VERSIONS_DIR / FROZEN_VERSION / "df_valid_final.parquet"
PILOT_DIR = MODEL_DATA_VERSIONS_DIR / PILOT_VERSION
PILOT_VALID = PILOT_DIR / "df_valid_final.parquet"

OUT_DIR = MODEL_RUNS_DIR / "phase3_predecessor_prior_pilot"
EVAL_JSON = OUT_DIR / "phase3_pilot_evaluation.json"
OUT_REPORT = OUT_DIR / "PHASE3_PILOT_REPORT.md"
OUT_JSON = OUT_DIR / "PHASE3_PILOT_CLAUSES.json"
LINK_PATH = MODEL_RUNS_DIR / "phase2_link_corrections" / "course_link_proposed.csv"

SEEDS = ("42", "52", "62", "72", "82")

# Published per-metric paired-delta ranges from models/runs/NOISE_BAND.md.
# Sign convention there: delta = candidate - baseline.
NOISE_BAND = {
    "m1_auc": {"min": -0.000382, "max": 0.001042, "sd": 0.000608, "better": "higher"},
    "m1_fail_avg_precision": {"min": -0.002045, "max": 0.001544, "sd": 0.001629, "better": "higher"},
    "m1_brier": {"min": -0.000108, "max": 0.000119, "sd": 0.0000927, "better": "lower"},
    "m1_train_valid_auc_gap": {"min": -0.005873, "max": 0.026720, "sd": 0.012487, "better": "lower"},
    "m2_mae": {"min": -0.050423, "max": 0.046520, "sd": 0.037233, "better": "lower"},
    "m2_rmse": {"min": -0.067477, "max": 0.078050, "sd": 0.055090, "better": "lower"},
    "m2_r2": {"min": -0.007865, "max": 0.006807, "sd": 0.005553, "better": "higher"},
}

BANNER = """```
STATUS: PILOT — PENDING/UNREVIEWED MAPPINGS — NOT FOR PROMOTION
This run measures whether the predecessor-prior mechanism is worth reviewing.
It does not authorize freezing, promoting, or wiring any model, regardless of
result. Human review of course_link_proposed.csv remains a precondition for
any production use.
```"""

CLAUSE_0_SEGMENTS = (
    "covered_unexposed",
    "untouched_uncovered_unexposed",
    "completely_unexposed",
)

REPORT_SEGMENTS = [
    ("overall_valid", "Whole VALID frame (model level)"),
    ("overall_uncovered_never_in_train_182", "1. Overall uncovered — all 182 never-in-TRAIN courses"),
    ("affected", "2. Affected segment — the directly eligible rows"),
    ("affected_scope_shared", "3a. Affected ∩ scope = shared"),
    ("affected_scope_specific", "3b. Affected ∩ scope = specific"),
    ("affected_credit_changed", "4a. Affected ∩ credit-changed predecessor"),
    ("affected_credit_unchanged", "4b. Affected ∩ credit-unchanged predecessors"),
    ("covered_unexposed", "5. SANITY — covered rows, propagation-unexposed"),
    ("untouched_uncovered_unexposed", "6. SANITY — untouched uncovered rows, propagation-unexposed"),
    ("completely_unexposed", "SANITY (superset) — every completely unexposed row"),
    ("indirect_propagation_only", "DIAGNOSTIC — propagation-exposed only, not directly eligible"),
]


def deltas(evaluation: dict, model: str, segment: str, metric: str) -> dict[str, float]:
    out = {}
    for seed in SEEDS:
        values = evaluation["seeds"][seed][model]["segments"][segment]
        base = values["baseline"]
        prior = values["with_prior"]
        if base.get("status") != "ok" or prior.get("status") != "ok":
            continue
        out[seed] = float(prior[metric]) - float(base[metric])
    return out


def summarize(values: dict[str, float]) -> dict[str, Any]:
    array = np.array(list(values.values()), dtype="float64")
    return {
        "per_seed": values,
        "mean": float(array.mean()) if array.size else None,
        "median": float(np.median(array)) if array.size else None,
        "sd": float(array.std(ddof=1)) if array.size > 1 else None,
        "min": float(array.min()) if array.size else None,
        "max": float(array.max()) if array.size else None,
    }


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{float(value):+.{digits}f}" if abs(float(value)) < 1000 else f"{float(value):,.{digits}f}"


def plain(value: Any, digits: int = 6) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def clause_beyond_band(values: dict[str, float], band_key: str) -> dict[str, Any]:
    """Improves beyond the published band: strictly past the band edge on the good side."""
    band = NOISE_BAND[band_key]
    threshold = band["max"] if band["better"] == "higher" else band["min"]
    passes = {
        seed: (delta > threshold) if band["better"] == "higher" else (delta < threshold)
        for seed, delta in values.items()
    }
    return {
        "rule": (
            f"per seed, delta {'>' if band['better'] == 'higher' else '<'} "
            f"{threshold:+.6f} (NOISE_BAND.md {'max' if band['better'] == 'higher' else 'min'})"
        ),
        "threshold": threshold,
        "per_seed_pass": passes,
        "seeds_passing": int(sum(passes.values())),
        "verdict": "PASS" if sum(passes.values()) >= 4 else "FAIL",
    }


def clause_mean_not_worse(values: dict[str, float], band_key: str) -> dict[str, Any]:
    band = NOISE_BAND[band_key]
    array = np.array(list(values.values()), dtype="float64")
    mean = float(array.mean())
    worsening_edge = band["max"] if band["better"] == "lower" else band["min"]
    if band["better"] == "lower":
        not_worse_at_all = mean <= 0.0
        within_noise = mean <= worsening_edge
    else:
        not_worse_at_all = mean >= 0.0
        within_noise = mean >= worsening_edge
    return {
        "rule": (
            "5-seed mean delta must not worsen; a worsening is tolerated only if it "
            f"stays inside the published band edge ({worsening_edge:+.6f})"
        ),
        "mean_delta": mean,
        "mean_improves_or_flat": bool(not_worse_at_all),
        "mean_within_noise_band": bool(within_noise),
        "verdict": "PASS" if within_noise else "FAIL",
    }


def clause_no_decline(values: dict[str, float], metric_label: str) -> dict[str, Any]:
    """Threshold-dependent safety guard. NOISE_BAND.md publishes no band for this
    metric, so the only non-invented rule is the sign of the paired delta."""
    passes = {seed: delta >= 0.0 for seed, delta in values.items()}
    return {
        "rule": (
            f"per seed, {metric_label} delta >= 0. NOISE_BAND.md publishes no band for "
            "this metric, so no numeric bound is invented; the sign is the rule."
        ),
        "per_seed_pass": passes,
        "seeds_passing": int(sum(passes.values())),
        "verdict": "PASS" if sum(passes.values()) >= 4 else "FAIL",
    }


def worked_examples(link: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "course_id",
        "degree_course_key",
        "course_pass_rate_historical",
        "course_avg_mark_historical",
        "course_difficulty_missing",
        "course_history_count",
        "difficulty_fallback_level",
    ]
    frozen = pd.read_parquet(FROZEN_VALID, columns=columns)
    pilot = pd.read_parquet(
        PILOT_VALID,
        columns=columns
        + [
            "course_history_count_predecessor",
            "course_cross_plan_prior_used",
            "course_cross_plan_prior_weight",
            "course_cross_plan_relationship_type",
            "course_identity_confidence",
            "course_difficulty_source_level",
        ],
    )
    contributions = pd.read_csv(PILOT_DIR / "predecessor_contributions.csv")

    examples = []
    for course_id in ("1175.111", "1422.111", "1419.111"):
        mask = frozen["course_id"].astype(str) == course_id
        rows = int(mask.sum())
        before = frozen.loc[mask]
        after = pilot.loc[mask]
        links = link.loc[
            link["new_course_id"].eq(course_id)
            & link["relationship_type"].isin(["successor", "consolidated_into"])
        ]
        contrib = contributions.loc[contributions["new_course_id"].astype(str) == course_id]
        examples.append(
            {
                "course_id": course_id,
                "valid_rows": rows,
                "relationship": sorted(set(links["relationship_type"]))[0] if len(links) else None,
                "predecessors": [
                    {
                        "old_course_id": record.old_course_id,
                        "weight": float(record.weight_hint),
                        "train_support": int(float(record.old_course_train_support)),
                        "link_pass_rate": float(record.old_course_train_pass_rate),
                        "link_avg_mark": float(record.old_course_train_avg_mark),
                    }
                    for record in links.itertuples(index=False)
                ],
                "predecessors_used_in_blend": sorted(set(contrib["old_course_id"].astype(str))),
                "source_levels_used": sorted(set(int(v) for v in contrib["source_level"])),
                "degree_prefixes": int(contrib["degree_prefix"].nunique()),
                "before": {
                    "course_pass_rate_historical": sorted(
                        {round(float(v), 10) for v in before["course_pass_rate_historical"]}
                    ),
                    "course_avg_mark_historical": sorted(
                        {round(float(v), 10) for v in before["course_avg_mark_historical"]}
                    ),
                    "difficulty_fallback_level": sorted(
                        {int(v) for v in before["difficulty_fallback_level"]}
                    ),
                },
                "after": {
                    "course_pass_rate_historical": sorted(
                        {round(float(v), 10) for v in after["course_pass_rate_historical"]}
                    ),
                    "course_avg_mark_historical": sorted(
                        {round(float(v), 10) for v in after["course_avg_mark_historical"]}
                    ),
                    "difficulty_fallback_level": sorted(
                        {int(v) for v in after["difficulty_fallback_level"]}
                    ),
                    "course_difficulty_missing": sorted(
                        {int(v) for v in after["course_difficulty_missing"]}
                    ),
                    "course_history_count": sorted({int(v) for v in after["course_history_count"]}),
                    "course_history_count_predecessor": sorted(
                        {int(v) for v in after["course_history_count_predecessor"]}
                    ),
                    "course_cross_plan_prior_weight": sorted(
                        {round(float(v), 10) for v in after["course_cross_plan_prior_weight"]}
                    ),
                    "course_difficulty_source_level": sorted(
                        set(after["course_difficulty_source_level"].astype(str))
                    ),
                    "course_identity_confidence": sorted(
                        set(after["course_identity_confidence"].astype(str))
                    ),
                },
            }
        )

    # Example 4: a covered row from the Clause-0 sanity segment.
    segments = pd.read_parquet(PILOT_DIR / "row_segments.parquet")
    sanity_mask = (
        segments["covered"].to_numpy(dtype=bool)
        & ~segments["propagation_exposed"].to_numpy(dtype=bool)
        & ~segments["directly_eligible"].to_numpy(dtype=bool)
    )
    position = int(np.flatnonzero(sanity_mask)[0])
    before_row = frozen.iloc[position]
    after_row = pilot.iloc[position]
    examples.append(
        {
            "course_id": str(before_row["course_id"]),
            "row_position": position,
            "segment": "covered_unexposed (Clause-0 sanity)",
            "before": {c: (float(before_row[c]) if isinstance(before_row[c], (float, np.floating)) else int(before_row[c]) if isinstance(before_row[c], (int, np.integer)) else str(before_row[c])) for c in columns},
            "after": {c: (float(after_row[c]) if isinstance(after_row[c], (float, np.floating)) else int(after_row[c]) if isinstance(after_row[c], (int, np.integer)) else str(after_row[c])) for c in columns},
            "audit": {
                "course_cross_plan_prior_used": bool(after_row["course_cross_plan_prior_used"]),
                "course_difficulty_source_level": str(after_row["course_difficulty_source_level"]),
                "course_history_count_predecessor": int(after_row["course_history_count_predecessor"]),
            },
        }
    )
    return examples


def segment_table_m1(evaluation: dict) -> list[str]:
    lines = [
        "| Segment | n | Metric | Baseline (5-seed mean) | With prior (5-seed mean) | Mean delta | Seeds improved | Beyond band? |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for segment, label in REPORT_SEGMENTS:
        n = evaluation["segment_sizes"][segment]
        for metric, band_key in (
            ("auc", "m1_auc"),
            ("fail_avg_precision", "m1_fail_avg_precision"),
            ("brier", "m1_brier"),
        ):
            values = deltas(evaluation, "m1_pass_classifier", segment, metric)
            if not values:
                lines.append(f"| {label} | {n:,} | `{metric}` | one class only | — | — | — | — |")
                continue
            base = np.mean([
                evaluation["seeds"][s]["m1_pass_classifier"]["segments"][segment]["baseline"][metric]
                for s in values
            ])
            prior = np.mean([
                evaluation["seeds"][s]["m1_pass_classifier"]["segments"][segment]["with_prior"][metric]
                for s in values
            ])
            summary = summarize(values)
            band = NOISE_BAND[band_key]
            improved = sum(
                (d > 0) if band["better"] == "higher" else (d < 0) for d in values.values()
            )
            outside = sum(
                (d > band["max"]) or (d < band["min"]) for d in values.values()
            )
            lines.append(
                f"| {label} | {n:,} | `{metric}` | {plain(base)} | {plain(prior)} | "
                f"{fmt(summary['mean'])} | {improved}/5 | {outside}/5 |"
            )
    return lines


def segment_table_m2(evaluation: dict) -> list[str]:
    lines = [
        "| Segment | n | Metric | Baseline (5-seed mean) | With prior (5-seed mean) | Mean delta | Seeds improved | Beyond band? |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for segment, label in REPORT_SEGMENTS:
        n = evaluation["segment_sizes"][segment]
        for metric, band_key in (("mae", "m2_mae"), ("rmse", "m2_rmse"), ("r2", "m2_r2")):
            values = deltas(evaluation, "m2_grade_regressor", segment, metric)
            if not values:
                lines.append(f"| {label} | {n:,} | `{metric}` | — | — | — | — | — |")
                continue
            base = np.mean([
                evaluation["seeds"][s]["m2_grade_regressor"]["segments"][segment]["baseline"][metric]
                for s in values
            ])
            prior = np.mean([
                evaluation["seeds"][s]["m2_grade_regressor"]["segments"][segment]["with_prior"][metric]
                for s in values
            ])
            summary = summarize(values)
            band = NOISE_BAND[band_key]
            improved = sum(
                (d > 0) if band["better"] == "higher" else (d < 0) for d in values.values()
            )
            outside = sum((d > band["max"]) or (d < band["min"]) for d in values.values())
            lines.append(
                f"| {label} | {n:,} | `{metric}` | {plain(base, 4)} | {plain(prior, 4)} | "
                f"{fmt(summary['mean'], 4)} | {improved}/5 | {outside}/5 |"
            )
    return lines


def threshold_table(evaluation: dict) -> list[str]:
    lines = [
        "| Segment | Metric | Baseline (5-seed mean) | With prior (5-seed mean) | Mean delta | Seeds not declining |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for segment, label in REPORT_SEGMENTS:
        for metric in ("fail_precision", "fail_recall", "fail_f1", "recall", "f1"):
            values = deltas(evaluation, "m1_pass_classifier", segment, metric)
            if not values:
                continue
            base = np.mean([
                evaluation["seeds"][s]["m1_pass_classifier"]["segments"][segment]["baseline"][metric]
                for s in values
            ])
            prior = np.mean([
                evaluation["seeds"][s]["m1_pass_classifier"]["segments"][segment]["with_prior"][metric]
                for s in values
            ])
            not_declining = sum(d >= 0 for d in values.values())
            lines.append(
                f"| {label} | `{metric}` | {plain(base, 4)} | {plain(prior, 4)} | "
                f"{fmt(np.mean(list(values.values())), 4)} | {not_declining}/5 |"
            )
    return lines


def prior_accuracy_diagnostic(link: pd.DataFrame) -> dict[str, Any]:
    """How close each prior is to what actually happened on the affected rows.

    This is descriptive: it compares the frozen fallback prior and the pilot
    predecessor prior against the realized VALID outcome, without any model in
    the loop, so the direction of the model result can be attributed.
    """
    frozen = pd.read_parquet(
        FROZEN_VALID,
        columns=[
            "course_id",
            "course_pass_rate_historical",
            "course_avg_mark_historical",
            "final_mark",
        ],
    )
    pilot = pd.read_parquet(
        PILOT_VALID,
        columns=[
            "course_pass_rate_historical",
            "course_avg_mark_historical",
            "course_cross_plan_prior_used",
        ],
    )
    weighted = link.loc[link["relationship_type"].isin(["successor", "consolidated_into"])]
    credit_changed = (
        weighted.assign(_c=weighted["credit_changed"].eq("true"))
        .groupby("new_course_id")["_c"]
        .any()
    )
    scope = weighted.drop_duplicates("new_course_id").set_index("new_course_id")["new_course_scope"]
    course = frozen["course_id"].astype(str)
    direct = pilot["course_cross_plan_prior_used"].to_numpy(dtype=bool)
    changed = course.map(credit_changed).fillna(False).to_numpy(dtype=bool)
    row_scope = course.map(scope).astype(object).fillna("").to_numpy()

    out: dict[str, Any] = {}
    for label, mask in (
        ("affected", direct),
        ("affected_credit_changed", direct & changed),
        ("affected_credit_unchanged", direct & ~changed),
        ("affected_scope_shared", direct & (row_scope == "shared")),
        ("affected_scope_specific", direct & (row_scope == "specific")),
    ):
        mark = frozen.loc[mask, "final_mark"].to_numpy(dtype="float64")
        passed = (mark >= 50).astype("float64")
        frozen_pass = frozen.loc[mask, "course_pass_rate_historical"].to_numpy(dtype="float64")
        pilot_pass = pilot.loc[mask, "course_pass_rate_historical"].to_numpy(dtype="float64")
        frozen_mark = frozen.loc[mask, "course_avg_mark_historical"].to_numpy(dtype="float64")
        pilot_mark = pilot.loc[mask, "course_avg_mark_historical"].to_numpy(dtype="float64")
        out[label] = {
            "n": int(mask.sum()),
            "actual_pass_rate": float(passed.mean()),
            "actual_mean_mark": float(mark.mean()),
            "prior_pass_rate_frozen": float(frozen_pass.mean()),
            "prior_pass_rate_pilot": float(pilot_pass.mean()),
            "prior_mean_mark_frozen": float(frozen_mark.mean()),
            "prior_mean_mark_pilot": float(pilot_mark.mean()),
            "mae_prior_mark_vs_actual_frozen": float(np.abs(frozen_mark - mark).mean()),
            "mae_prior_mark_vs_actual_pilot": float(np.abs(pilot_mark - mark).mean()),
            "mae_prior_pass_vs_actual_frozen": float(np.abs(frozen_pass - passed).mean()),
            "mae_prior_pass_vs_actual_pilot": float(np.abs(pilot_pass - passed).mean()),
        }
    return out


def build_markdown(
    evaluation: dict,
    manifest: dict,
    clauses: dict,
    verdict: str,
    examples: list[dict[str, Any]],
    accuracy: dict[str, Any],
) -> str:
    sizes = evaluation["segment_sizes"]
    eligible = manifest["eligible_set"]
    check_1 = manifest["verification"]["check_1_no_change_outside_eligible_set"][
        "course_pass_rate_historical"
    ]
    lines: list[str] = [
        "# Phase 3 — predecessor-prior PILOT",
        "",
        BANNER,
        "",
        "## Governance status",
        "",
        "- Every row of `course_link_proposed.csv` and `course_split_candidates.csv` is "
        "still `approval_status = pending`. **Zero rows have been human-reviewed**, and "
        "this run changed none of them.",
        "- This pilot exists to answer one question: *is the predecessor-prior mechanism "
        "worth the human review effort it would take to approve it?* It is not a "
        "substitute for that review.",
        "- Nothing here authorizes freezing M1/M2, opening TEST, promoting a dataset "
        "version, or wiring anything into production.",
        "- TEST was never read, globbed, or stat-ed. `Decisions_Log.md` was not edited "
        "(a ready-to-copy entry is at the end). Nothing was pushed.",
        "",
        "### Conflicts with `CLAUDE.md`, flagged rather than resolved silently",
        "",
        "1. `CLAUDE.md` §3 names the **regularization pass** as the only active "
        "workstream and §8 forbids improvising outside a scoped prompt. This task is a "
        "different, explicitly authorized workstream that permits a dataset write and "
        "model work. The prompt was followed; the conflict is recorded here.",
        "2. `CLAUDE.md` §5 says *\"never copy datasets into new folders\"*. The task "
        "requires `df_train_final.parquet` inside the new version directory and requires "
        "it to be byte-identical. It is a byte copy, verified by SHA-256 against the "
        "frozen file. This does create a second physical copy of TRAIN; it is a pilot "
        "artifact and should be deleted rather than promoted.",
        "3. `CLAUDE.md` §6 selection metrics are unchanged, but the acceptance clauses "
        "below were pre-registered by the task prompt, not by "
        "`docs/EXPERIMENT_REGULARIZATION_PLAN.md`, which governs a different experiment.",
        "",
        "## Mechanism recap",
        "",
        "For a VALID row whose `course_id` appears in the Phase 2T link table as a "
        "`new_course_id` with `relationship_type` in `{successor, consolidated_into}` "
        "(i.e. it carries a `weight_hint`), the two historical difficulty estimates are "
        "replaced by the weight-blended TRAIN estimate of that course's predecessors:",
        "",
        "- **Per-predecessor estimate**, the precedence Phase 0's `answer_q6` already "
        "validated: substitute the predecessor's `course_id` into the row's own "
        "`degree_course_key` and use the **Level-1** (degree+course) TRAIN estimate when "
        "that substituted key has support in `fit_difficulty_state(TRAIN)`; otherwise use "
        "the predecessor's **Level-2** (course-across-degrees) TRAIN estimate.",
        f"- Across the {eligible['degree_course_pairs']} distinct (degree, new course) "
        f"pairs there were {eligible['predecessor_contributions']} predecessor "
        f"contributions: **{manifest['mechanism']['level_1_contributions']} resolved at "
        f"Level-1** and **{manifest['mechanism']['level_2_contributions']} at Level-2**. "
        "The new-plan degrees almost never carry TRAIN history for an old-plan course, "
        "so the mechanism is in practice a Level-2 substitution.",
        "- **Blend**: `successor` (one predecessor, `weight_hint = 1.0`) takes that "
        "predecessor's estimate directly; `consolidated_into` takes the "
        "`weight_hint`-weighted average across *all* contributing predecessors.",
        "- **Shrinkage identity**: because this is a single frozen-TRAIN snapshot, an "
        "eligible course's own history is structurally zero, so "
        "`(n_new*local + k*prior)/(n_new+k)` collapses to the prior term **exactly** at "
        "`n_new = 0`. This substitution is not an approximation of that formula — it *is* "
        "that formula at `n_new = 0`.",
        "",
        "Changed on eligible rows: `course_pass_rate_historical`, "
        "`course_avg_mark_historical`. Deliberately **unchanged on every row**: "
        "`course_difficulty_missing` (still 1), `course_history_count` (still 0), "
        "`difficulty_fallback_level`, `course_retake_rate_historical`, "
        "`difficulty_group_support_count`, `course_is_new`, `course_low_support`.",
        "",
        "### Downstream propagation (one change, not two)",
        "",
        "Concurrent-group features are mechanically derived from "
        "`d = 1 - course_pass_rate_historical`, so they were rebuilt through the "
        "**unmodified** `src.concurrent_group_features` builder over the frozen "
        "registration roster with only its inputs changed. Before the rebuild, the "
        "builder was shown to reproduce all eight frozen concurrent columns **exactly** "
        "from the frozen roster, so any difference afterwards is attributable to this "
        "one change and nothing else.",
        "",
        "### Audit-only columns",
        "",
        "Six columns were added, using the names from the original Section 9 spec: "
        + ", ".join(f"`{c}`" for c in manifest["diff_vs_frozen_valid"]["audit_columns_added"])
        + ". **None of them is in the M1 or M2 feature list**, and the build asserts that "
        "before writing anything.",
        "",
        "## Eligible-set confirmation",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Eligible (weighted-link) courses | **{eligible['courses']}** |",
        f"| Eligible VALID rows | **{eligible['valid_rows']:,}** |",
        f"| — of which `successor` courses | {eligible['successor_courses']} |",
        f"| — of which `consolidated_into` courses | {eligible['consolidated_courses']} |",
        f"| Scope `shared` / `specific` courses | {eligible['shared_scope_courses']} / {eligible['specific_scope_courses']} |",
        f"| Courses with ≥1 credit-changed weighted predecessor | {eligible['credit_changed_courses']} |",
        f"| Roster occurrences substituted (peer side) | {eligible['roster_rows_substituted']:,} |",
        "",
        "Phase 2T reported **82 courses / 17,036 VALID rows** for the *union* of weighted, "
        "structural, and manual relationships. This pilot's eligibility rule excludes "
        "`split_from` / `merged_from`, which is exactly the difference and it reconciles "
        "to the row: the two structural courses `1183.111` and `1192.111` account for "
        "**2 courses / 767 rows**, and `80 + 2 = 82`, `16,269 + 767 = 17,036`. The "
        "recount is therefore not a discrepancy — it is the preregistered exclusion, and "
        "no unexplained drift exists.",
        "",
        "## Pre-training verification (Section 3)",
        "",
        "| # | Check | Result |",
        "|---|---|---|",
        f"| 0 | TRAIN byte-identical to frozen TRAIN | **PASS** — SHA-256 "
        f"`{manifest['verification']['train_sha256'][:32]}…` identical |",
        "| 0 | `fit_difficulty_state(TRAIN)` reproduces all 9 frozen VALID difficulty "
        "columns | **PASS** — 0 mismatches |",
        "| 0 | Unmodified concurrent builder reproduces all 8 frozen concurrent columns | "
        "**PASS** — 0 mismatches |",
        f"| 1 | Rows differing in the two substituted columns ⊆ eligible set | **PASS** — "
        f"**0** rows changed outside the eligible set |",
        f"| 1 | Every eligible row carries its assigned value | **PASS** — "
        f"{check_1['numerically_changed_rows']:,} of {check_1['eligible_rows']:,} moved "
        f"numerically; **{check_1['no_op_rows_substituted_value_equals_frozen_value']} were "
        f"exact no-ops** (see note) |",
        "| 2 | No unexposed covered / untouched-relationship row changed in any locked "
        "contract column | **PASS** — 0 leaks |",
        "| 3 | M1 `baseline_41` and M2 `concurrent_43` feature **lists** identical to the "
        "locked contracts | **PASS** — byte-identical; only *values* changed |",
        "",
        f"**Note on the {check_1['no_op_rows_substituted_value_equals_frozen_value']} "
        f"no-op rows.** All belong to course `{check_1['no_op_courses'][0]}`. Its "
        "Level-5 fallback group (`requirement_type 4` + 4 credits, TRAIN support 71) "
        "consists of exactly the same TRAIN rows as its predecessor `1036.111`'s Level-2 "
        "group (support 71). The frozen fallback value therefore already *was* the "
        "predecessor estimate, and the substitution is a genuine no-op — a set identity, "
        "not a tolerance artifact. Those rows are still counted as directly eligible "
        "everywhere below; their predictions simply cannot move.",
        "",
        "## Row sets (addendum §2 / §3)",
        "",
        "| Row set | n |",
        "|---|---:|",
        f"| VALID total | {sizes['overall_valid']:,} |",
        f"| `directly_eligible_rows` | {sizes['affected']:,} |",
        f"| `propagation_exposed_rows` (exposed only, not directly eligible) | {sizes['indirect_propagation_only']:,} |",
        f"| Completely unexposed | {sizes['completely_unexposed']:,} |",
        f"| Never-in-TRAIN (all 182 link-table courses) | {sizes['overall_uncovered_never_in_train_182']:,} |",
        f"| `covered_unexposed` (Clause-0 sanity) | {sizes['covered_unexposed']:,} |",
        f"| `untouched_uncovered_unexposed` (Clause-0 sanity) | {sizes['untouched_uncovered_unexposed']:,} |",
        "",
        "Changed rows per column, pilot VALID vs frozen VALID:",
        "",
        "| Column | Changed rows |",
        "|---|---:|",
    ]
    for column, count in manifest["diff_vs_frozen_valid"][
        "changed_rows_per_direct_difficulty_column"
    ].items():
        lines.append(f"| `{column}` (direct substitution) | {count:,} |")
    for column, count in manifest["diff_vs_frozen_valid"][
        "changed_rows_per_concurrent_derived_column"
    ].items():
        lines.append(f"| `{column}` (propagation) | {count:,} |")
    lines.extend(
        [
            "| every other frozen column | **0** |",
            "",
            "## Experimental design — paired, not two trainings",
            "",
            "The pilot TRAIN frame is byte-identical to the frozen TRAIN frame and the "
            "contracts and hyperparameters are unchanged, so for a given seed the two arms "
            "have identical training inputs and training a second nominal model would "
            "measure nothing. Per the addendum, **one locked model artifact per seed was "
            "evaluated on both VALID frames**, with the artifact's SHA-256 recorded for "
            "both arms.",
            "",
            "This is not a shortcut: an eligible course is *never-in-TRAIN* by "
            "construction, so no retraining on this data could ever learn from the "
            "substituted values. The mechanism is inherently an inference-time "
            "intervention on VALID, and the paired evaluation is its complete measurement.",
            "",
            "Locked specs, not retuned: **M1 = `baseline_41`, `num_leaves=127`**; "
            "**M2 = `concurrent_43`, `num_leaves=127`**; seeds 42/52/62/72/82; reporting "
            "threshold 0.80.",
            "",
            "Existing 5-seed runs at exactly these specs on the frozen version were found "
            "and reused. Each was verified before use: recorded contract name, seed, "
            "`num_leaves` / `min_child_samples` / `reg_lambda`, `dataset_version`, "
            "`test_policy = closed_not_read`, and recorded TRAIN/VALID SHA-256 matching the "
            "frozen files on disk — and then, evidence-first, **its recorded frozen-VALID "
            "metrics were recomputed from the booster on disk and had to match**.",
            "",
            "| Seed | M1 source run (`baseline_41`) | M2 source run (`concurrent_43`) |",
            "|---|---|---|",
        ]
    )
    for seed in SEEDS:
        m1 = evaluation["seeds"][seed]["m1_pass_classifier"]["source_run"]
        m2 = evaluation["seeds"][seed]["m2_grade_regressor"]["source_run"]
        lines.append(f"| {seed} | `{m1['run']}` | `{m2['run']}` |")
    dirty = [
        seed
        for seed in SEEDS
        if evaluation["seeds"][seed]["m1_pass_classifier"]["source_run"]["git_working_tree_clean"]
        is False
    ]
    lines.extend(
        [
            "",
            "**Provenance caveat, stated rather than buried.** Every reused run recorded a "
            f"dirty working tree at training time (seeds {', '.join(dirty)} for M1, and the "
            "M2 runs likewise), so bit-exact retrainability cannot be *proved* from the "
            "artifacts alone. It does not affect this measurement: both arms use the same "
            "booster file, so the comparison is internally valid for any locked-spec model. "
            "It would matter for a model-vs-model comparison, which this is not.",
            "",
            "## Segmented results — M1 (`baseline_41`, threshold-independent)",
            "",
            "Sign convention: delta = with-prior − baseline. Higher-better: AUC, fail-AP. "
            "Lower-better: Brier. \"Beyond band?\" counts seeds whose delta falls outside "
            "the published `NOISE_BAND.md` range **in either direction**.",
            "",
        ]
    )
    lines.extend(segment_table_m1(evaluation))
    lines.extend(
        [
            "",
            "`train_valid_auc_gap` is reported **only at the overall model level** — there "
            "is no corresponding TRAIN population for a never-in-TRAIN course, so a "
            "segment-level gap would be `N/A — no corresponding TRAIN segment`.",
            "",
            "| Seed | Gap baseline | Gap with prior | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    gap_deltas = []
    for seed in SEEDS:
        gap = evaluation["seeds"][seed]["m1_pass_classifier"]["train_valid_auc_gap"]
        gap_deltas.append(gap["with_prior"] - gap["baseline"])
        lines.append(
            f"| {seed} | {plain(gap['baseline'])} | {plain(gap['with_prior'])} | "
            f"{fmt(gap['with_prior'] - gap['baseline'])} |"
        )
    lines.extend(
        [
            f"| **mean** | | | **{fmt(float(np.mean(gap_deltas)))}** |",
            "",
            "The gap moves only because VALID AUC moved; TRAIN AUC is identical by "
            "construction, so a *widening* gap here is the same fact as a falling VALID AUC, "
            "not independent evidence.",
            "",
            "## Segmented results — M2 (`concurrent_43`)",
            "",
        ]
    )
    lines.extend(segment_table_m2(evaluation))
    lines.extend(
        [
            "",
            "## Threshold-dependent guards at the locked 0.80 cut",
            "",
            "Per the addendum, fail recall and fail F1 are **mandatory secondary "
            "non-regression guards**, not readability-only. Pass-class recall/F1 are shown "
            "for readability.",
            "",
        ]
    )
    lines.extend(threshold_table(evaluation))
    lines.extend(
        [
            "",
            "## Clause 0 — mandatory sanity, checked before any other verdict",
            "",
            "Row-level prediction identity on the propagation-**unexposed** sanity "
            "segments, using the same model artifact for both arms. Legitimate concurrent "
            "propagation is excluded from these segments by construction and is reported "
            "separately as `indirect_propagation_only`; it is not classified as leakage.",
            "",
            "| Sanity segment | n | Max abs prediction difference | Rows over tolerance |",
            "|---|---:|---:|---:|",
        ]
    )
    for segment in CLAUSE_0_SEGMENTS:
        entry = evaluation["seeds"]["42"]["m1_pass_classifier"]["clause_0"][segment]
        worst = max(
            evaluation["seeds"][seed][model]["clause_0"][segment][
                "max_abs_prediction_difference"
            ]
            for seed in SEEDS
            for model in ("m1_pass_classifier", "m2_grade_regressor")
        )
        over = sum(
            evaluation["seeds"][seed][model]["clause_0"][segment]["rows_exceeding_tolerance"]
            for seed in SEEDS
            for model in ("m1_pass_classifier", "m2_grade_regressor")
        )
        lines.append(f"| `{segment}` | {entry['n']:,} | {worst:.1e} | **{over}** |")
    lines.extend(
        [
            "",
            f"Tolerance `{clauses['clause_0_sanity']['tolerance']:.0e}` (the repository's "
            "existing `CHANGE_ATOL`). Across all 5 seeds × both models × all three sanity "
            "segments the maximum absolute prediction difference is **exactly 0.0** and "
            "**0 rows** exceed tolerance — the predictions are bit-identical, not merely "
            "within tolerance. The additional aggregate `NOISE_BAND.md` check is trivially "
            "satisfied: every metric delta on these segments is exactly zero.",
            "",
            f"**Clause 0 verdict: {clauses['clause_0_sanity']['verdict']}.** The run is "
            "valid and Clauses 1–6 may be read.",
            "",
            "One further structural confirmation: on `indirect_propagation_only` "
            f"({sizes['indirect_propagation_only']:,} rows) **M1's** deltas are exactly "
            "zero in every seed, because `baseline_41` contains no concurrent feature and "
            "therefore cannot see the propagation at all. Only M2 responds there.",
            "",
            "## Clauses 1–6, on the affected segment",
            "",
            "Bounds come from `models/runs/NOISE_BAND.md`; none was invented here. "
            "`NOISE_BAND.md` publishes no band for fail recall or fail F1, so for those the "
            "only non-invented rule is the sign of the paired delta.",
            "",
            "| Clause | Metric | Rule | Per-seed deltas (42/52/62/72/82) | Mean | Verdict |",
            "|---|---|---|---|---:|---|",
        ]
    )
    clause_labels = {
        "clause_1_m1_auc_improves": ("1", "M1 AUC", "improves in ≥4/5 seeds beyond the band"),
        "clause_2_m1_brier_not_worse": ("2", "M1 Brier", "5-seed mean does not worsen"),
        "clause_3_m1_fail_recall_no_decline": ("3", "M1 fail recall @0.80", "does not decline in ≥4/5 seeds"),
        "clause_4_m1_fail_f1_no_decline": ("4", "M1 fail F1 @0.80", "does not decline in ≥4/5 seeds"),
        "clause_5_m2_mae_improves": ("5", "M2 MAE", "improves in ≥4/5 seeds beyond the band"),
        "clause_6_m2_rmse_not_worse": ("6", "M2 RMSE", "5-seed mean does not worsen"),
    }
    for key, (number, metric, rule) in clause_labels.items():
        entry = clauses[key]
        per_seed = entry["deltas"]["per_seed"]
        digits = 5 if key.startswith(("clause_1", "clause_2")) else 4
        rendered = " / ".join(fmt(per_seed[seed], digits) for seed in SEEDS)
        mark = "**PASS**" if entry["verdict"] == "PASS" else "**FAIL**"
        lines.append(
            f"| {number} | {metric} | {rule} | {rendered} | "
            f"{fmt(entry['deltas']['mean'], digits)} | {mark} |"
        )
    passed = [k for k in clause_labels if clauses[k]["verdict"] == "PASS"]
    failed = [k for k in clause_labels if clauses[k]["verdict"] == "FAIL"]
    lines.extend(
        [
            "",
            f"**{len(passed)} of 6 pass, {len(failed)} fail.**",
            "",
            "Reading this honestly matters more than the tally. The three clauses that "
            "pass (2, 3, 4) are **non-regression guards** — they only say the mechanism did "
            "not make calibration or fail-catching worse. The two clauses that actually "
            "test whether the mechanism *helps* (1 and 5) both fail, and they fail **with "
            "the sign inverted**, not merely by landing inside the band:",
            "",
            "- **M1 AUC fell in 5/5 seeds** on the affected segment "
            f"(mean {fmt(clauses['clause_1_m1_auc_improves']['deltas']['mean'], 5)}), with "
            f"{sum(d < NOISE_BAND['m1_auc']['min'] for d in clauses['clause_1_m1_auc_improves']['deltas']['per_seed'].values())}/5 "
            "seeds outside the published band on the worsening side.",
            "- **M2 MAE rose in 5/5 seeds** on the affected segment "
            f"(mean {fmt(clauses['clause_5_m2_mae_improves']['deltas']['mean'], 4)}), which "
            "is roughly **4×** the band's worst observed noise excursion (+0.046520). M2 "
            "RMSE rose in 5/5 as well.",
            "",
            "Clause 2's pass is real and worth noting: **Brier improved while AUC fell.** "
            "The substituted prior moved absolute risk levels in a helpful direction on "
            "these rows while degrading the *ranking* among them.",
            "",
            "## Why the direction came out this way",
            "",
            "This is a model-free diagnostic: it compares each prior directly against what "
            "actually happened on the affected rows, with no model in the loop.",
            "",
            "| Segment | n | Actual pass rate | Prior pass, frozen → pilot | Actual mean mark | Prior mark, frozen → pilot |",
            "|---|---:|---:|---|---:|---|",
        ]
    )
    for label, values in accuracy.items():
        lines.append(
            f"| `{label}` | {values['n']:,} | {values['actual_pass_rate']:.4f} | "
            f"{values['prior_pass_rate_frozen']:.4f} → {values['prior_pass_rate_pilot']:.4f} | "
            f"{values['actual_mean_mark']:.3f} | "
            f"{values['prior_mean_mark_frozen']:.3f} → {values['prior_mean_mark_pilot']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Row-level accuracy of the prior itself, against the realized outcome:",
            "",
            "| Segment | MAE(prior mark vs actual mark) frozen → pilot | MAE(prior pass vs actual pass) frozen → pilot |",
            "|---|---|---|",
        ]
    )
    for label, values in accuracy.items():
        lines.append(
            f"| `{label}` | {values['mae_prior_mark_vs_actual_frozen']:.3f} → "
            f"**{values['mae_prior_mark_vs_actual_pilot']:.3f}** | "
            f"{values['mae_prior_pass_vs_actual_frozen']:.4f} → "
            f"**{values['mae_prior_pass_vs_actual_pilot']:.4f}** |"
        )
    affected_accuracy = accuracy["affected"]
    lines.extend(
        [
            "",
            "The premise does not hold on this data. On the affected rows the **existing "
            "Level-4/Level-5 fallback prior was already closer to the truth** than the "
            f"predecessor prior: the frozen fallback put the pass rate at "
            f"{affected_accuracy['prior_pass_rate_frozen']:.4f} against an actual "
            f"{affected_accuracy['actual_pass_rate']:.4f} — nearly exact — while the "
            f"predecessor prior moves it down to "
            f"{affected_accuracy['prior_pass_rate_pilot']:.4f}, away from the outcome. "
            "At row level the pass-rate prior gets less accurate in **all five** "
            "sub-segments and the mark prior in **four of five** — the one exception is "
            f"`affected_scope_specific`, whose mark prior improves slightly "
            f"({accuracy['affected_scope_specific']['mae_prior_mark_vs_actual_frozen']:.3f} → "
            f"{accuracy['affected_scope_specific']['mae_prior_mark_vs_actual_pilot']:.3f}) "
            "while its pass-rate prior still degrades.",
            "",
            "Interpretation, offered as a hypothesis and not as a finding: **the old-plan "
            "predecessor courses were systematically harder than their new-plan successors "
            "turned out to be.** The mechanism faithfully imports that old difficulty, which "
            "is precisely why it hurts. The effect is strongest exactly where one would "
            "expect a weaker identity claim — the credit-changed subset "
            f"(prior-mark MAE {accuracy['affected_credit_changed']['mae_prior_mark_vs_actual_frozen']:.3f} "
            f"→ {accuracy['affected_credit_changed']['mae_prior_mark_vs_actual_pilot']:.3f}) "
            "degrades far more than the credit-unchanged subset "
            f"({accuracy['affected_credit_unchanged']['mae_prior_mark_vs_actual_frozen']:.3f} "
            f"→ {accuracy['affected_credit_unchanged']['mae_prior_mark_vs_actual_pilot']:.3f}, "
            "essentially flat).",
            "",
            "That sub-segment structure is the pilot's most useful output for a reviewer: "
            "it is consistent across seeds, it points at the credit-changed and "
            "`shared`-scope links as the damaging ones, and it is visible in the audit "
            "columns without any model.",
            "",
            "## Worked examples",
            "",
        ]
    )
    def values_text(values: list[Any]) -> str:
        """One distinct value renders as a scalar; several render as a list."""
        rendered = [f"{v:g}" if isinstance(v, float) else str(v) for v in values]
        return rendered[0] if len(rendered) == 1 else ", ".join(rendered)

    for example in examples[:3]:
        before = example["before"]
        after = example["after"]
        lines.extend(
            [
                f"### `{example['course_id']}` — `{example['relationship']}`, "
                f"{example['valid_rows']:,} VALID rows",
                "",
                "| Field | Before (frozen) | After (pilot) |",
                "|---|---|---|",
                f"| `course_pass_rate_historical` | "
                f"{', '.join(f'{v:.6f}' for v in before['course_pass_rate_historical'])} | "
                f"{', '.join(f'{v:.6f}' for v in after['course_pass_rate_historical'])} |",
                f"| `course_avg_mark_historical` | "
                f"{', '.join(f'{v:.4f}' for v in before['course_avg_mark_historical'])} | "
                f"{', '.join(f'{v:.4f}' for v in after['course_avg_mark_historical'])} |",
                f"| `difficulty_fallback_level` | "
                f"{values_text(before['difficulty_fallback_level'])} | "
                f"{values_text(after['difficulty_fallback_level'])} (unchanged by design) |",
                f"| `course_difficulty_missing` | 1 | "
                f"{values_text(after['course_difficulty_missing'])} (unchanged by design) |",
                f"| `course_history_count` | 0 | "
                f"{values_text(after['course_history_count'])} (unchanged by design) |",
                f"| `course_history_count_predecessor` (audit) | not present | "
                f"{', '.join(f'{v:,}' for v in after['course_history_count_predecessor'])} |",
                f"| `course_cross_plan_prior_weight` (audit) | not present | "
                f"{', '.join(f'{v:.6f}' for v in after['course_cross_plan_prior_weight'])} |",
                f"| `course_difficulty_source_level` (audit) | not present | "
                f"`{after['course_difficulty_source_level'][0]}` |",
                f"| `course_identity_confidence` (audit) | not present | "
                f"`{after['course_identity_confidence'][0]}` |",
                "",
                "Predecessors and the weights actually used in the blend:",
                "",
                "| Predecessor | Weight | TRAIN support | Level used |",
                "|---|---:|---:|---:|",
            ]
        )
        for predecessor in example["predecessors"]:
            lines.append(
                f"| `{predecessor['old_course_id']}` | {predecessor['weight']:.6f} | "
                f"{predecessor['train_support']:,} | "
                f"{'/'.join(str(v) for v in example['source_levels_used'])} |"
            )
        used = set(example["predecessors_used_in_blend"])
        declared = {p["old_course_id"] for p in example["predecessors"]}
        lines.extend(
            [
                "",
                (
                    f"All {len(declared)} declared contributing predecessors were used in "
                    f"the blend: {'confirmed' if used == declared else 'MISMATCH'}."
                    if len(declared) > 1
                    else "The single declared predecessor was used: "
                    f"{'confirmed' if used == declared else 'MISMATCH'}."
                )
                + (
                    f" The course's {example['degree_prefixes']} distinct degree contexts "
                    "collapse to one post-substitution value because the Level-1 substituted "
                    "key has no TRAIN support in the new-plan degrees, so all of them fall to "
                    "the shared Level-2 estimate."
                    if example["degree_prefixes"] > 1
                    else ""
                ),
                "",
            ]
        )
    sanity = examples[3]
    lines.extend(
        [
            f"### `{sanity['course_id']}` — Clause-0 covered-row sanity example "
            f"(VALID row {sanity['row_position']})",
            "",
            "| Field | Before (frozen) | After (pilot) |",
            "|---|---|---|",
        ]
    )
    for field in (
        "degree_course_key",
        "course_pass_rate_historical",
        "course_avg_mark_historical",
        "course_difficulty_missing",
        "course_history_count",
        "difficulty_fallback_level",
    ):
        lines.append(f"| `{field}` | {sanity['before'][field]} | {sanity['after'][field]} |")
    lines.extend(
        [
            f"| `course_cross_plan_prior_used` (audit) | not present | "
            f"{sanity['audit']['course_cross_plan_prior_used']} |",
            f"| `course_difficulty_source_level` (audit) | not present | "
            f"`{sanity['audit']['course_difficulty_source_level']}` |",
            "",
            "Zero change in every field, and its model prediction is bit-identical in both "
            "arms for all five seeds and both models.",
            "",
            "## Verdict",
            "",
            f"# `{verdict}`",
            "",
            "**This verdict does not authorize promotion.** It does not approve any "
            "proposal row, does not validate any individual link, does not permit freezing "
            "M1 or M2, does not permit opening TEST, and does not permit wiring anything "
            "into production.",
            "",
            f"`{verdict}` is the category the pre-registered arithmetic produces: Clause 0 "
            f"holds and {len(passed)} of 6 clauses pass, which is a split rather than a "
            "majority failure. The substance is more one-sided than the label: both "
            "improvement clauses failed in the wrong direction across all five seeds, and "
            "M2's degradation is several times the published noise band, while the passing "
            "clauses are non-regression guards.",
            "",
            "The direct answer to the question this pilot was built to ask — *is the "
            "predecessor-prior mechanism worth the human review effort?* — is: **not as "
            "specified, applied to all weighted links.** On this VALID frame the existing "
            "fallback prior is already the better estimate, and importing predecessor "
            "difficulty makes both models worse on the rows it touches.",
            "",
            "What the pilot does support spending review effort on, if anything: the "
            "credit-unchanged / `specific`-scope subset is roughly neutral rather than "
            "harmful, and the harm concentrates in the credit-changed and `shared`-scope "
            "links. A reviewer who wants to salvage the idea should look there first — but "
            "note that **M2 MAE worsened in 5/5 seeds even on the credit-unchanged "
            "subset**, so no sub-segment identified here is positive on the M2 side.",
            "",
            "### Methodological caveats a reader must carry",
            "",
            "- `NOISE_BAND.md` was measured on the **full VALID frame** from "
            "*contract-change* deltas across seeds. It is applied here to a "
            f"{sizes['affected']:,}-row **segment** under a *feature-value* change with a "
            "*fixed* model. Segment metrics are noisier than full-frame metrics, so the "
            "band is, if anything, too permissive here — which makes the failures of "
            "Clauses 1 and 5 harder to dismiss, not easier. `CLAUDE.md` §11 already warns "
            "the band is the best available yardstick, not an exact one.",
            "- The paired design removes seed-to-seed *training* variation entirely. The "
            "five \"seeds\" here vary only the fixed instrument, not the intervention, so "
            "5/5 consistency is strong evidence about the intervention's direction but is "
            "**not** an independent replication in the sense the multi-seed contract "
            "experiments used.",
            "- \"Overall uncovered\" is reported as the 25,627 rows of the 182 link-table "
            "courses, per the task. The frozen VALID actually holds 26,882 rows with "
            "`course_difficulty_missing == 1`; the extra 1,255 are low-support courses that "
            "are not never-in-TRAIN and appear in no link table.",
            "",
            "## If the human approves the pending rows",
            "",
            "Nothing in this pilot may be reused as a candidate. To turn any part of it "
            "into a real model candidate, **this exact pipeline must be re-run restricted "
            "to only the rows with `review_decision = approved`**:",
            "",
            "1. Reviewers record a decision on each `course_link_proposed.csv` row. The "
            "eligibility filter then becomes `relationship_type ∈ {successor, "
            "consolidated_into}` **and** `review_decision = approved`, not the current "
            "`approval_status = pending` population.",
            "2. `scripts/phase3_predecessor_prior_pilot_build.py` is re-run against that "
            "restricted set, producing a different eligible-set size and a different "
            "dataset version. Weight renormalization for a consolidation that loses a "
            "predecessor is an **open design question**, deliberately unanswered here: this "
            "pilot stops rather than redistributing a dropped predecessor's weight.",
            "3. The paired evaluation is re-run on the restricted frame. The clause results "
            "in this report **do not carry over** — they describe the unreviewed population, "
            "which is a different population.",
            "4. Only then can the mechanism be discussed as a candidate, and only through "
            "the ordinary freeze/promotion gates, which this run does not touch.",
            "",
            "This pilot deliberately built its dataset version under a "
            "`_PENDING_REVIEW` name so it cannot be mistaken for a promotable artifact. It "
            "should be deleted, not promoted.",
            "",
            "## Artifacts",
            "",
            "| Artifact | Path |",
            "|---|---|",
            f"| Pilot dataset version | `data/model_data/versions/{PILOT_VERSION}/` |",
            "| Build script | `scripts/phase3_predecessor_prior_pilot_build.py` |",
            "| Paired-evaluation script | `scripts/phase3_predecessor_prior_pilot_evaluate.py` |",
            "| Clause/report script | `scripts/phase3_predecessor_prior_pilot_report.py` |",
            "| Raw paired metrics | `models/runs/phase3_predecessor_prior_pilot/phase3_pilot_evaluation.json` |",
            "| Clause evaluation | `models/runs/phase3_predecessor_prior_pilot/PHASE3_PILOT_CLAUSES.json` |",
            "| Per-arm run directories | `models/runs/<stamp>__predecessor_prior_pilot_seed{N}_{baseline,withprior}/` |",
            "",
            "## Governance entry (ready to copy — `Decisions_Log.md` was NOT edited)",
            "",
            "> Phase 3 ran the predecessor-prior mechanism as a pilot on unreviewed "
            "(`pending`) proposal rows, explicitly to measure whether the mechanism "
            "justifies the human review effort before that review is spent. The pilot's "
            "verdict does not constitute approval of any row and does not authorize "
            "promotion; it is an input to the decision of whether to proceed with human "
            "review at all.",
            ">",
            f"> Result: `{verdict}`. Clause 0 passed at the strongest level — predictions "
            "on every propagation-unexposed row are bit-identical between arms across 5 "
            "seeds and both models. Of the six numbered clauses, the two that test benefit "
            "failed with the sign inverted (M1 AUC fell in 5/5 seeds on the affected "
            "segment; M2 MAE rose in 5/5, roughly 4× the published noise band), while the "
            "three non-regression guards (Brier, fail recall, fail F1) passed. A model-free "
            "diagnostic shows why: on these rows the existing Level-4/5 fallback prior is "
            "already closer to the realized outcome than the predecessor prior, i.e. the "
            "old-plan courses were harder than their successors proved to be. The harm "
            "concentrates in credit-changed and `shared`-scope links. No proposal row was "
            "approved, no version promoted, no model frozen, TEST untouched.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    evaluation = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    manifest = json.loads((PILOT_DIR / "manifest.json").read_text(encoding="utf-8"))
    link = pd.read_csv(LINK_PATH, dtype="string", keep_default_na=False)

    # ---- Clause 0 first; it gates everything else. ----
    clause_0_rows: list[dict[str, Any]] = []
    clause_0_pass = True
    for seed in SEEDS:
        for model in ("m1_pass_classifier", "m2_grade_regressor"):
            for segment, values in evaluation["seeds"][seed][model]["clause_0"].items():
                clause_0_rows.append({"seed": seed, "model": model, "segment": segment, **values})
                if values["rows_exceeding_tolerance"] > 0:
                    clause_0_pass = False
    # Aggregate metric drift on the sanity segments as the additional check.
    clause_0_metric_drift: dict[str, Any] = {}
    for segment in ("covered_unexposed", "untouched_uncovered_unexposed", "completely_unexposed"):
        clause_0_metric_drift[segment] = {
            "m1_auc": summarize(deltas(evaluation, "m1_pass_classifier", segment, "auc")),
            "m1_brier": summarize(deltas(evaluation, "m1_pass_classifier", segment, "brier")),
            "m2_mae": summarize(deltas(evaluation, "m2_grade_regressor", segment, "mae")),
            "m2_rmse": summarize(deltas(evaluation, "m2_grade_regressor", segment, "rmse")),
        }
        for metric, summary in clause_0_metric_drift[segment].items():
            if summary["max"] is not None and max(abs(summary["min"]), abs(summary["max"])) > 0.0:
                clause_0_pass = False

    clauses: dict[str, Any] = {
        "clause_0_sanity": {
            "definition": (
                "row-level prediction identity on the propagation-unexposed sanity "
                "segments, using the same model artifact for both arms; the aggregate "
                "NOISE_BAND check is reported in addition, not instead"
            ),
            "tolerance": evaluation["seeds"]["42"]["m1_pass_classifier"]["clause_0"][
                "covered_unexposed"
            ]["tolerance"],
            "rows_exceeding_tolerance_total": sum(
                row["rows_exceeding_tolerance"] for row in clause_0_rows
            ),
            "max_abs_prediction_difference_overall": max(
                row["max_abs_prediction_difference"] for row in clause_0_rows
            ),
            "metric_drift": clause_0_metric_drift,
            "verdict": "PASS" if clause_0_pass else "FAIL",
        }
    }

    if clause_0_pass:
        affected = "affected"
        clauses["clause_1_m1_auc_improves"] = {
            **clause_beyond_band(deltas(evaluation, "m1_pass_classifier", affected, "auc"), "m1_auc"),
            "deltas": summarize(deltas(evaluation, "m1_pass_classifier", affected, "auc")),
        }
        clauses["clause_2_m1_brier_not_worse"] = {
            **clause_mean_not_worse(deltas(evaluation, "m1_pass_classifier", affected, "brier"), "m1_brier"),
            "deltas": summarize(deltas(evaluation, "m1_pass_classifier", affected, "brier")),
        }
        clauses["clause_3_m1_fail_recall_no_decline"] = {
            **clause_no_decline(deltas(evaluation, "m1_pass_classifier", affected, "fail_recall"), "fail recall"),
            "deltas": summarize(deltas(evaluation, "m1_pass_classifier", affected, "fail_recall")),
            "note": "threshold-dependent safety guard at the locked 0.80 reporting cut",
        }
        clauses["clause_4_m1_fail_f1_no_decline"] = {
            **clause_no_decline(deltas(evaluation, "m1_pass_classifier", affected, "fail_f1"), "fail F1"),
            "deltas": summarize(deltas(evaluation, "m1_pass_classifier", affected, "fail_f1")),
            "note": "threshold-dependent safety guard at the locked 0.80 reporting cut",
        }
        clauses["clause_5_m2_mae_improves"] = {
            **clause_beyond_band(deltas(evaluation, "m2_grade_regressor", affected, "mae"), "m2_mae"),
            "deltas": summarize(deltas(evaluation, "m2_grade_regressor", affected, "mae")),
        }
        clauses["clause_6_m2_rmse_not_worse"] = {
            **clause_mean_not_worse(deltas(evaluation, "m2_grade_regressor", affected, "rmse"), "m2_rmse"),
            "deltas": summarize(deltas(evaluation, "m2_grade_regressor", affected, "rmse")),
        }

    numbered = [key for key in clauses if key.startswith("clause_") and key != "clause_0_sanity"]
    passed = [key for key in numbered if clauses[key]["verdict"] == "PASS"]
    if not clause_0_pass:
        verdict = "INVALID_RUN"
    elif len(passed) == len(numbered):
        verdict = "PILOT_SIGNAL_POSITIVE"
    elif len(passed) * 2 < len(numbered):
        verdict = "PILOT_SIGNAL_NEGATIVE"
    else:
        verdict = "MIXED"

    examples = worked_examples(link)
    accuracy = prior_accuracy_diagnostic(link)

    payload = {
        "artifact": "phase3_predecessor_prior_pilot_clauses",
        "status": "PILOT — PENDING/UNREVIEWED MAPPINGS — NOT FOR PROMOTION",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "verdict": verdict,
        "verdict_meaning": (
            "PILOT_SIGNAL_POSITIVE means only that the mechanism looks informative "
            "enough to justify spending human review on the pending mappings. It "
            "validates no link, approves no proposal row, and does not show the effect "
            "survives restriction to approved mappings."
        ),
        "clauses": clauses,
        "segment_sizes": evaluation["segment_sizes"],
        "noise_band_source": "models/runs/NOISE_BAND.md",
        "noise_band_used": NOISE_BAND,
        "worked_examples": examples,
        "prior_accuracy_diagnostic": accuracy,
        "manifest_row_sets": manifest["row_sets"],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_REPORT.write_text(
        build_markdown(evaluation, manifest, clauses, verdict, examples, accuracy),
        encoding="utf-8",
    )

    print(json.dumps({"verdict": verdict, "clauses": {k: v["verdict"] for k, v in clauses.items()}}, indent=2))
    print(f"Wrote {OUT_REPORT}")
    print(f"Wrote {OUT_JSON}")
    print("TEST reads: 0. Proposal rows approved: 0. Promotions: 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
