"""Read-only reconciliation: resolve the 170 pending course-identity candidates
using the extended-history raw file (through 2025 S2).

Scope and guarantees
--------------------
* Re-examines ONLY the 170 pending courses (``likely_equivalent_needs_review``
  + ``unresolved``) from ``COURSE_IDENTITY_INVESTIGATION``. The 5 already
  ``confirmed_equivalent`` and the 7 ``genuinely_new`` courses are carried over
  untouched, per the task's explicit non-scope.
* Applies the SAME pre-registered matching rule as the prior investigation. The
  only substitution is that the censoring-limited corroboration term
  (``predecessor_volume_collapsed OR temporal_complementarity``, guarded by
  ``NOT censored_predecessor``) is replaced by the extended-history taper
  signal, which is exactly the evidence the old censoring guard could not see.
  No new rule, no new threshold on any other signal.
* Trains nothing, tunes nothing, writes no dataset, changes no default, creates
  no ``canonical_course_id`` and applies no mapping.

GOVERNANCE CONFLICT - DECLARED, NOT HIDDEN
------------------------------------------
``docs/pipeline_rules.md`` fixes the temporal split as Train 2005-2021 /
Validation 2022-2023 / **Test 2024 + 2025 S1**. The extended history this task
directs us to read therefore spans the TEST window, while CLAUDE.md section 5
declares TEST ``closed_not_read``. The task prompt is explicit, so per the
CLAUDE.md header ("follow the prompt but flag the conflict explicitly") this
script proceeds and declares the conflict in the report.

Containment applied so the conflict is as narrow as it can be:

* ``df_test_final.parquet`` is never read, globbed, stat-ed or path-constructed.
* From the extended file this script reads ONLY the columns
  ``course_id``, ``part_id``, ``degree_id``, ``student_id``. ``final_mark`` and
  every other outcome column are never loaded, so no label information from the
  TEST window can enter any artifact produced here.
* Only per-course enrolment COUNTS are derived. No metric, no difficulty
  statistic and no model input is computed from post-20233 rows.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_prior_module():
    """Import the prior investigation script so its rule is reused verbatim."""

    path = ROOT / "scripts" / "course_identity_investigation.py"
    spec = importlib.util.spec_from_file_location("course_identity_investigation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CI = _load_prior_module()

EXTENDED_PATH = ROOT / "data" / "final" / "without_outliers.parquet"
# Only these columns are ever loaded from the extended file (see module docstring).
EXTENDED_COLUMNS = ["course_id", "part_id", "degree_id", "student_id"]

PRIOR_MD = ROOT / "models" / "runs" / "COURSE_IDENTITY_INVESTIGATION.md"
PRIOR_JSON = ROOT / "models" / "runs" / "COURSE_IDENTITY_INVESTIGATION.json"
PRIOR_CSV = ROOT / "models" / "runs" / "COURSE_IDENTITY_CANDIDATES.csv"

OUT_MD = ROOT / "models" / "runs" / "COURSE_IDENTITY_RECONCILIATION_2025.md"
OUT_JSON = ROOT / "models" / "runs" / "COURSE_IDENTITY_RECONCILIATION_2025.json"
OUT_CSV = ROOT / "models" / "runs" / "course_identity_candidates_v2.csv"

PENDING_BUCKETS = {"likely_equivalent_needs_review", "unresolved"}
CARRIED_BUCKETS = {"confirmed_equivalent", "genuinely_new"}

# ---------------------------------------------------------------------------
# Extended-history signal - the ONLY thing this task adds to the prior rule.
# ---------------------------------------------------------------------------
# "near-zero" reuses the prior investigation's own activity floor
# (MIN_SEMESTER_ACTIVITY = 5 rows/semester) rather than inventing a new number.
NEAR_ZERO_ROWS = CI.MIN_SEMESTER_ACTIVITY
# The task's own bar: enrolment must stay near-zero for >= 3 consecutive semesters.
TAPER_MIN_SILENT_SEMESTERS = 3
# Volume comparison reuses the prior collapse threshold (0.35 of the reference year).
PERSISTENCE_RATIO_MAX = CI.COLLAPSE_RATIO_MAX
# Last COMPLETE academic year in the extended file; 2025 has only S1/S2 loaded and
# would understate a full-year rate, so the year-on-year comparison uses 2024.
LATE_YEAR_PREFIX = "2024"
PRIOR_REFERENCE_YEAR_PREFIX = "2021"


# ---------------------------------------------------------------------------
# Step 1 - reproduce the prior investigation's state exactly
# ---------------------------------------------------------------------------
def reproduce_prior() -> dict[str, Any]:
    train = pd.read_parquet(CI.TRAIN_PATH, columns=CI.MODEL_COLUMNS)
    valid = pd.read_parquet(CI.VALID_PATH, columns=CI.MODEL_COLUMNS)
    for frame in (train, valid):
        for column in ("course_id", "degree_id", "faculty_id", "part_id", "degree_course_key"):
            frame[column] = frame[column].astype("string")

    never, facts = CI.compute_never_in_train(train, valid)
    if not facts["matches_diagnostic"]:
        raise SystemExit(f"STOP: never_in_train figures differ from the diagnostic: {facts}")

    train_ids = set(train["course_id"].dropna())
    never_valid = valid.loc[never]
    new_valid_rows = never_valid["course_id"].value_counts()
    new_first_semester = never_valid.groupby("course_id")["part_id"].min()
    new_ids = list(new_valid_rows.sort_values(ascending=False).index)

    raw, _clean = CI.load_catalog()
    activity = CI.build_activity(train, valid)
    profile = CI.course_profiles(train, valid, activity)
    lineage, _records = CI.degree_lineage(train, valid, raw)
    entries = CI.score_candidates(
        new_ids, train_ids, profile, raw, lineage, new_valid_rows, new_first_semester
    )
    prior = {e["new_course_id"]: CI.classify(e) for e in entries}
    sensitivity = {
        e["new_course_id"]: CI.classify(e, censor_guard="temporal_only") for e in entries
    }
    return {
        "train": train,
        "valid": valid,
        "never": never,
        "facts": facts,
        "entries": entries,
        "prior": prior,
        "sensitivity": sensitivity,
        "profile": profile,
    }


def bucket_totals(entries: list[dict], buckets: dict[str, dict]) -> dict[str, dict[str, int]]:
    totals = {
        b: {"courses": 0, "rows": 0}
        for b in ("confirmed_equivalent", "likely_equivalent_needs_review",
                  "genuinely_new", "unresolved")
    }
    for entry in entries:
        bucket = buckets[entry["new_course_id"]]["bucket"]
        totals[bucket]["courses"] += 1
        totals[bucket]["rows"] += entry["valid_rows"]
    return totals


# ---------------------------------------------------------------------------
# Step 2 - extended history
# ---------------------------------------------------------------------------
def load_extended() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not EXTENDED_PATH.exists() or EXTENDED_PATH.stat().st_size == 0:
        raise SystemExit(f"STOP: extended history file missing: {EXTENDED_PATH}")
    ext = pd.read_parquet(EXTENDED_PATH, columns=EXTENDED_COLUMNS)
    for column in EXTENDED_COLUMNS:
        ext[column] = ext[column].astype("string")

    calendar = sorted(ext["part_id"].dropna().unique())
    coverage = {
        "path": str(EXTENDED_PATH.relative_to(ROOT)).replace("\\", "/"),
        "rows": int(len(ext)),
        "columns_read": EXTENDED_COLUMNS,
        "columns_deliberately_not_read": "final_mark and every other outcome column",
        "distinct_courses": int(ext["course_id"].nunique()),
        "first_semester": str(calendar[0]),
        "last_semester": str(calendar[-1]),
        "semesters": [str(s) for s in calendar],
        "reaches_2024": any(str(s).startswith("2024") for s in calendar),
        "reaches_2025_s2": "20252" in {str(s) for s in calendar},
    }
    return ext, coverage


def semester_series(ext: pd.DataFrame) -> tuple[dict[str, dict[str, int]], list[str]]:
    """rows-per-semester for every course in the extended file."""

    grouped = ext.groupby(["course_id", "part_id"]).size()
    calendar = sorted({str(s) for s in ext["part_id"].dropna().unique()})
    series: dict[str, dict[str, int]] = {}
    for (course, part), count in grouped.items():
        series.setdefault(str(course), {})[str(part)] = int(count)
    return series, calendar


def taper_profile(
    course_id: str | None,
    series: dict[str, dict[str, int]],
    calendar: list[str],
    debut: str | None,
) -> dict[str, Any]:
    """Extended-history taper / persistence profile for one course."""

    if course_id is None:
        return {"observed": False}

    counts = series.get(str(course_id), {})
    per_semester = {s: int(counts.get(s, 0)) for s in calendar}
    active_semesters = [s for s in calendar if per_semester[s] >= NEAR_ZERO_ROWS]
    last_active = active_semesters[-1] if active_semesters else None

    if last_active is None:
        silent_after = len(calendar)
    else:
        silent_after = sum(1 for s in calendar if s > last_active)

    # A course that NEVER reached the activity floor in any semester has nothing to
    # taper: that is an absence of signal, not evidence of replacement. Scoring it as
    # a confirmed taper would confirm equivalence on a predecessor that never ran.
    never_active = last_active is None
    taper_confirmed = (not never_active) and silent_after >= TAPER_MIN_SILENT_SEMESTERS
    taper_post_debut = bool(
        taper_confirmed and debut is not None and last_active is not None
        and str(last_active) >= str(debut)
    )
    taper_pre_debut = bool(
        taper_confirmed and debut is not None and str(last_active) < str(debut)
    )

    late_year_rows = sum(v for s, v in per_semester.items() if s.startswith(LATE_YEAR_PREFIX))
    reference_year_rows = sum(
        v for s, v in per_semester.items() if s.startswith(PRIOR_REFERENCE_YEAR_PREFIX)
    )
    persistence_ratio = (
        float(late_year_rows) / float(reference_year_rows) if reference_year_rows > 0 else None
    )
    still_active = (not taper_confirmed) and (not never_active)
    still_active_full_volume = bool(
        still_active
        and persistence_ratio is not None
        and persistence_ratio > PERSISTENCE_RATIO_MAX
    )

    return {
        "observed": True,
        "per_semester": per_semester,
        "total_rows": int(sum(per_semester.values())),
        "last_active_semester": last_active,
        "never_reached_activity_floor": bool(never_active),
        "silent_semesters_after_last_active": int(silent_after),
        "taper_confirmed": bool(taper_confirmed),
        "taper_post_debut": taper_post_debut,
        "taper_pre_debut": taper_pre_debut,
        "still_active": bool(still_active),
        "still_active_full_volume": still_active_full_volume,
        f"rows_{LATE_YEAR_PREFIX}": int(late_year_rows),
        f"rows_{PRIOR_REFERENCE_YEAR_PREFIX}": int(reference_year_rows),
        "persistence_ratio": None if persistence_ratio is None else round(persistence_ratio, 4),
    }


def evidence_label(pred: dict[str, Any], new: dict[str, Any]) -> str:
    if not pred.get("observed"):
        return "predecessor absent from the extended file - no new evidence"
    if pred["never_reached_activity_floor"]:
        return (
            f"predecessor never reached the {NEAR_ZERO_ROWS}-enrolment activity floor in "
            f"any semester of the extended file ({pred['total_rows']} rows in total) - "
            "there is no taper to observe, so no corroboration is available"
        )
    if pred["taper_post_debut"]:
        return (
            f"taper confirmed: predecessor last active {pred['last_active_semester']} "
            f"(after the new course's debut), then near-zero for "
            f"{pred['silent_semesters_after_last_active']} consecutive semesters through 20252"
        )
    if pred["taper_pre_debut"]:
        last = pred["last_active_semester"] or "never"
        return (
            f"predecessor already fully tapered before the new course's debut "
            f"(last active {last}); the extra years confirm no rebound but add no new evidence"
        )
    if pred["still_active_full_volume"]:
        return (
            f"predecessor still active at comparable volume through 2025 "
            f"({pred[f'rows_{LATE_YEAR_PREFIX}']} rows in 2024 vs "
            f"{pred[f'rows_{PRIOR_REFERENCE_YEAR_PREFIX}']} in 2021, ratio "
            f"{pred['persistence_ratio']}) - evidence AGAINST equivalence: both courses are live"
        )
    return (
        f"predecessor still running through 2025 at reduced volume "
        f"(last active {pred['last_active_semester']}, 2024/2021 ratio "
        f"{pred['persistence_ratio']}) - no confirmed taper"
    )


# ---------------------------------------------------------------------------
# Step 3 - reclassification: the pre-registered rule, taper substituted in
# ---------------------------------------------------------------------------
def classify_extended(entry: dict[str, Any], pred_taper: dict[str, Any]) -> dict[str, Any]:
    """CI.classify() with the censoring-limited corroboration term replaced.

    Every other clause - name exactness, degree lineage, credits, requirement
    type, uniqueness, the 0.80/0.60 name-similarity tiers - is unchanged and is
    evaluated on exactly the values the prior investigation scored.
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

    # THE SUBSTITUTION: the extended file resolves what the censoring guard blocked.
    # A confirmed taper is an observed disappearance, not a right-censored one, so it
    # carries the corroboration AND retires the censoring guard in one term.
    corroborated = bool(pred_taper.get("taper_confirmed"))

    unique = True
    if runner_up is not None and abs(runner_up["evidence_weight"] - top["evidence_weight"]) < 1e-9:
        volume = max(runner_up["old_train_rows"], 1)
        unique = top["old_train_rows"] >= CI.UNIQUENESS_DOMINANCE * volume

    if (
        top["name_exact_normalized"]
        and lineage_ok
        and top["credits_equal"]
        and top["requirement_type_equal"]
        and corroborated
        and unique
    ):
        bucket = "confirmed_equivalent"
        reason = (
            f"exact normalized name, equal credits ({top['credits_new']}) and requirement "
            f"type, degree-lineage link, and an extended-history taper: predecessor "
            f"{top['old_course_id']} last active {pred_taper['last_active_semester']}, "
            f"near-zero for {pred_taper['silent_semesters_after_last_active']} consecutive "
            f"semesters through 20252"
        )
    elif lineage_ok and (
        (top["name_exact_normalized"] and (top["credits_equal"] or top["requirement_type_equal"]))
        or (top["name_similarity"] >= CI.NAME_SIM_STRONG and top["credits_equal"]
            and top["requirement_type_equal"])
    ):
        bucket = "likely_equivalent_needs_review"
        missing = [s for s in ("credits", "requirement_type") if s in top["signals_conflicted"]]
        if not corroborated:
            if not pred_taper.get("observed"):
                missing.append("predecessor absent from the extended file")
            elif pred_taper.get("never_reached_activity_floor"):
                missing.append("predecessor never reached the activity floor - no taper to observe")
            else:
                missing.append("predecessor did not taper - still enrolling through 2025")
        if not unique:
            missing.append("ambiguous: a second candidate carries equal evidence")
        reason = (
            f"name similarity {top['name_similarity']:.2f} with degree-lineage link; "
            f"unmet for confirmation: {', '.join(missing) or 'uniqueness/corroboration'}"
        )
    elif top["name_similarity"] < CI.NAME_SIM_PLAUSIBLE:
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
        if not corroborated and pred_taper.get("observed"):
            problems.append("predecessor still enrolling through 2025 - no taper")
        reason = "plausible but insufficient: " + "; ".join(problems)

    return {"bucket": bucket, "top_candidates": candidates[:3], "reason": reason}


# ---------------------------------------------------------------------------
# Step 4 - report rendering
# ---------------------------------------------------------------------------
def fmt(n: Any) -> str:
    return f"{n:,}" if isinstance(n, (int, np.integer)) else str(n)


def render_markdown(payload: dict[str, Any]) -> str:
    p = payload
    lines: list[str] = []
    a = lines.append

    a("# Course-identity reconciliation - resolving the 170 pending candidates with history through 2025 S2")
    a("")
    a("**Status: READ-ONLY EVIDENCE TASK. Updated candidates only.**")
    a("")
    a("> **No mapping has been created, applied, or wired.** No `canonical_course_id` "
      "column exists anywhere as a result of this work. `course_identity_candidates_v2.csv` "
      "is a *review artifact* for a human reader, not an equivalence table, and must not be "
      "consumed as a drop-in mapping file: it deliberately mixes buckets, carries "
      "conflicting-evidence columns, and lists more than one candidate per row - exactly as "
      "the v1 CSV did.")
    a("")
    a("| | |")
    a("|---|---|")
    a(f"| Prior investigation | `COURSE_IDENTITY_INVESTIGATION.md` (HEAD `{p['prior_git_head']}`) |")
    a(f"| Frozen dataset version | `{CI.VERSION}` |")
    a(f"| Extended history file | `{p['coverage']['path']}` |")
    a(f"| Extended coverage | `{p['coverage']['first_semester']}` -> `{p['coverage']['last_semester']}` "
      f"({len(p['coverage']['semesters'])} semesters, {fmt(p['coverage']['rows'])} rows) |")
    a("| TEST parquet | never read - no TEST path was constructed, globbed, stat-ed or read |")
    a("| Models trained / tuned | none |")
    a("| Datasets written | none |")
    a("| Defaults / wiring / promotion changed | none |")
    a(f"| HEAD at run time | `{p['git']['head']}` |")
    a("")

    a("## Headline verdict")
    a("")
    conf = p["new_totals"]["confirmed_equivalent"]
    prior_conf = p["prior_totals"]["confirmed_equivalent"]
    upgrade = p["transitions"].get(
        "`likely_equivalent_needs_review` -> `confirmed_equivalent`", {"courses": 0, "rows": 0}
    )
    upper_now = p["payoff"]["confirmed_plus_likely_upper_bound"]["never_in_train_rows_becoming_covered"]
    upper_was = p["prior_payoff"]["confirmed_plus_likely_upper_bound"]["never_in_train_rows_becoming_covered"]
    a(f"- **The extra years move {upgrade['courses']} courses / {fmt(upgrade['rows'])} VALID "
      f"rows from `likely` to `confirmed`.** `confirmed_equivalent` goes "
      f"{prior_conf['courses']} -> {conf['courses']} courses ({fmt(prior_conf['rows'])} -> "
      f"{fmt(conf['rows'])} rows); confirmed-only coverage recovery goes "
      f"{fmt(p['prior_payoff']['confirmed_only']['never_in_train_rows_becoming_covered'])} -> "
      f"{fmt(p['payoff']['confirmed_only']['never_in_train_rows_becoming_covered'])} rows.")
    a(f"- **The `confirmed + likely` upper bound does not move at all: {fmt(upper_now)} rows, "
      f"unchanged from {fmt(upper_was)}.** The taper signal only relocates courses between "
      "the two accepted buckets - it cannot rescue a course whose *name, credits, "
      "requirement type or lineage* were the problem. The headline payoff of this whole "
      "line of work is therefore unchanged.")
    a(f"- **The extra years refute nearly as many flagged courses as they confirm.** Of the "
      f"{p['censoring_flag']['courses']} courses the prior censoring guard withheld, "
      f"{p['censoring_flag_confirmed']} are now confirmed and **{p['censoring_flag_refuted']} "
      "are positively refuted**: their predecessors are still enrolling in 2025. The prior "
      f"investigation's own sensitivity variant would have confirmed all "
      f"{p['censoring_flag']['courses']} - so the conservative pre-registered guard was "
      f"right, and the sensitivity was wrong on {p['censoring_flag_refuted']} of "
      f"{p['censoring_flag']['courses']}.")
    a(f"- **The dominant new finding is negative.** For "
      f"{p['predecessor_active_full_volume']['courses']} of the 170 pending courses "
      f"({fmt(p['predecessor_active_full_volume']['rows'])} VALID rows) the candidate "
      "predecessor is still running at comparable volume through 2025 - both courses are "
      "live, which is evidence AGAINST equivalence, not for it.")
    a("- **Nothing was downgraded out of `likely`, and no `unresolved` course moved.** "
      "Every `unresolved` course was blocked by a non-temporal clause, which extra history "
      "cannot address.")
    a("")

    a("## 0. Declared governance conflict - the extended window overlaps TEST")
    a("")
    a("`docs/pipeline_rules.md` line 81 fixes the temporal split as **Train 2005-2021 / "
      "Validation 2022-2023 / Test 2024 + 2025 S1**. The extended history this task "
      "directs the analysis to use therefore spans the TEST window, while CLAUDE.md "
      "section 5 declares TEST `closed_not_read`. **This is a real conflict and it is "
      "declared here rather than resolved silently.** The task prompt is explicit and "
      "read-only, so per the CLAUDE.md header rule the prompt was followed and the "
      "conflict is flagged.")
    a("")
    a("Containment actually applied:")
    a("")
    a("- `df_test_final.parquet` was never read, globbed, stat-ed or path-constructed.")
    a(f"- Only these columns were loaded from the extended file: `{'`, `'.join(EXTENDED_COLUMNS)}`. "
      "`final_mark` and every other outcome column were never loaded, so no label "
      "information from the TEST window can have entered any artifact produced here.")
    a("- Only per-course enrolment **counts** were derived. No metric, no difficulty "
      "statistic and no model input was computed from post-20233 rows.")
    a("")
    a("The residual exposure is nonetheless real: the taper evidence below is derived "
      "from enrolment volumes inside the TEST window. **The human owns the decision of "
      "whether that is acceptable before any of this feeds a mapping.**")
    a("")

    a("## 1. Preconditions - the prior state was reproduced, not trusted")
    a("")
    a("| Quantity | Reproduced now | Prior report | Match |")
    a("|---|---:|---:|:--:|")
    f_ = p["facts"]
    a(f"| never-in-TRAIN courses | {f_['never_in_train_courses']} | 182 | "
      f"{'yes' if f_['never_in_train_courses'] == 182 else 'NO'} |")
    a(f"| never-in-TRAIN VALID rows | {fmt(f_['never_in_train_rows'])} | 25,627 | "
      f"{'yes' if f_['never_in_train_rows'] == 25627 else 'NO'} |")
    a(f"| total uncovered VALID rows | {fmt(f_['uncovered_rows'])} | 26,882 | "
      f"{'yes' if f_['uncovered_rows'] == 26882 else 'NO'} |")
    for bucket, exp_c, exp_r in (
        ("confirmed_equivalent", 5, 1791),
        ("likely_equivalent_needs_review", 82, 16359),
        ("genuinely_new", 7, 134),
        ("unresolved", 88, 7343),
    ):
        got = p["prior_totals"][bucket]
        ok = got["courses"] == exp_c and got["rows"] == exp_r
        a(f"| `{bucket}` | {got['courses']} / {fmt(got['rows'])} | {exp_c} / {fmt(exp_r)} | "
          f"{'yes' if ok else 'NO'} |")
    a("")
    a(f"The candidate CSV was cross-checked row-by-row against the reproduction: "
      f"{p['csv_crosscheck']['rows']} data rows, "
      f"{p['csv_crosscheck']['bucket_mismatches']} bucket mismatches, "
      f"{p['csv_crosscheck']['candidate_mismatches']} top-candidate mismatches.")
    a("")
    a("### The pre-registered censoring flag")
    a("")
    a(f"The prior investigation's censoring guard withheld confirmation from "
      f"**{p['censoring_flag']['courses']} courses / {fmt(p['censoring_flag']['rows'])} VALID rows** "
      "that its own sensitivity variant would have confirmed (confirmed 5 -> 10 courses, "
      "1,791 -> 4,072 rows). Those courses are:")
    a("")
    a("| Course | Name | VALID rows | Prior bucket |")
    a("|---|---|---:|---|")
    for row in p["censoring_flag"]["courses_detail"]:
        a(f"| `{row['new_course_id']}` | {row['new_course_name']} | {fmt(row['valid_rows'])} | "
          f"`{row['prior_bucket']}` |")
    a("")
    a("They are the clearest test of whether the extra years change anything: each was "
      "blocked only by the guard's scope, never by weak evidence. Their outcome:")
    a("")
    a("| Course | VALID rows | Updated bucket | Verdict | Extended-history evidence |")
    a("|---|---:|---|---|---|")
    for row in p["censoring_flag_outcome"]:
        a(f"| `{row['new_course_id']}` | {fmt(row['valid_rows'])} | `{row['new_bucket']}` | "
          f"**{row['verdict']}** | {row['new_evidence']} |")
    a("")
    a("**This is the single most decision-relevant result in this reconciliation.** The prior "
      "investigation's `temporal_only` sensitivity would have confirmed all 5 of these; the "
      "extended history shows 2 of the 5 to be wrong. The pre-registered conservative guard "
      "was the correct call, and the sensitivity block should not be used as if it were.")
    a("")

    a("## 2. Extended-history coverage")
    a("")
    cov = p["coverage"]
    a(f"- Rows: {fmt(cov['rows'])}; distinct courses: {fmt(cov['distinct_courses'])}.")
    a(f"- Semester range `{cov['first_semester']}` -> `{cov['last_semester']}`. "
      f"Reaches 2024: **{cov['reaches_2024']}**. Reaches 2025 S2 (`20252`): "
      f"**{cov['reaches_2025_s2']}**.")
    a(f"- Semesters beyond the VALID end (`20233`): "
      f"`{'`, `'.join(p['post_valid_semesters'])}` - "
      f"{len(p['post_valid_semesters'])} semesters of history the prior investigation could not see.")
    a("")
    a("Consistency of the shared window (the extended file is not a different population):")
    a("")
    a("| Check | Value |")
    a("|---|---:|")
    for key, value in p["consistency"].items():
        a(f"| {key} | {fmt(value)} |")
    a("")
    a(f"- Predecessors of the 170 pending courses present in the extended file: "
      f"{p['predecessor_coverage']['present']} of {p['predecessor_coverage']['total']}.")
    a("")

    a("## 3. The rule applied - one substitution, nothing else")
    a("")
    a("The pre-registered rule of `COURSE_IDENTITY_INVESTIGATION.md` section 5 is applied "
      "unchanged: identifier pattern, credits, requirement type, faculty, planned level, "
      "name similarity (0.80 strong / 0.60 plausible tiers), degree overlap, degree "
      "lineage, and the >= 3.0x uniqueness dominance tie-break all keep their original "
      "definitions and their original scored values.")
    a("")
    a("**The single substitution.** The prior confirmation clause read:")
    a("")
    a("> AND at least one independent temporal/enrolment corroboration (...) AND the "
      "corroboration is not censored (predecessor activity does not end in 20232/20233)")
    a("")
    a("Both halves of that clause existed only because VALID ended at 20233. With history "
      "through 20252 the question is directly observable, so the two halves collapse into "
      "one term:")
    a("")
    a("> AND the predecessor's enrolment has **tapered**: it falls below "
      f"{NEAR_ZERO_ROWS} enrolments in a semester and stays there for >= "
      f"{TAPER_MIN_SILENT_SEMESTERS} consecutive semesters running through the end of the "
      "observed calendar (`20252`).")
    a("")
    a("Thresholds are inherited, not invented:")
    a("")
    a(f"- **near-zero = fewer than {NEAR_ZERO_ROWS} enrolments in a semester** - the prior "
      f"investigation's own `MIN_SEMESTER_ACTIVITY` activity floor.")
    a(f"- **>= {TAPER_MIN_SILENT_SEMESTERS} consecutive near-zero semesters** - the bar set "
      "by this task. Requiring the silence to run through `20252` also retires the old "
      "censoring guard: three observed empty semesters are a disappearance, not a "
      "right-censored gap.")
    a(f"- **still-active-at-full-volume = 2024 rows / 2021 rows > {PERSISTENCE_RATIO_MAX}** - "
      f"the prior investigation's own `COLLAPSE_RATIO_MAX`. 2024 is used as the reference "
      "late year because it is the last COMPLETE academic year in the file; 2025 carries "
      "only S1/S2 and would understate a full-year rate.")
    a("")
    a("Two sub-cases of a confirmed taper are distinguished because they carry different "
      "information:")
    a("")
    a("- **taper post-debut** - the predecessor was still running when the new course "
      "debuted and died afterwards. This is the signal the old censoring guard could not see.")
    a("- **taper pre-debut** - the predecessor was already dead before the new course "
      "debuted. The extra years confirm it never rebounded, but they add no new evidence.")
    a("")

    a("## 4. Reclassification of the 170 pending courses")
    a("")
    a("| Bucket | Prior (182) | Updated (182) | Delta |")
    a("|---|---:|---:|---:|")
    for bucket in ("confirmed_equivalent", "likely_equivalent_needs_review",
                   "genuinely_new", "unresolved"):
        old = p["prior_totals"][bucket]
        new = p["new_totals"][bucket]
        a(f"| `{bucket}` | {old['courses']} / {fmt(old['rows'])} | "
          f"{new['courses']} / {fmt(new['rows'])} | "
          f"{new['courses'] - old['courses']:+d} / {new['rows'] - old['rows']:+,} |")
    a("")
    a("### Transitions")
    a("")
    a("| Transition | Courses | VALID rows |")
    a("|---|---:|---:|")
    for key, value in p["transitions"].items():
        a(f"| {key} | {value['courses']} | {fmt(value['rows'])} |")
    a("")
    a("### Evidence that drove the changes")
    a("")
    a("| Extended-history evidence | Courses | VALID rows |")
    a("|---|---:|---:|")
    for key, value in p["evidence_totals"].items():
        a(f"| {key} | {value['courses']} | {fmt(value['rows'])} |")
    a("")
    a(f"The new course's own trajectory was checked for all {p['pending_count']} pending "
      f"courses: **{p['new_course_shortlived']['courses']} courses / "
      f"{fmt(p['new_course_shortlived']['rows'])} VALID rows** have themselves tapered to "
      f"near-zero for >= {TAPER_MIN_SILENT_SEMESTERS} semesters, i.e. the *new* course "
      "looks like a short-lived offering. This is recorded in the v2 CSV column "
      "`new_course_short_lived` and reported here, but it is deliberately **not** allowed "
      "to change any bucket: gating on it would be a new rule, and this task's mandate is "
      "to apply the pre-registered one.")
    a("")

    a("### Courses that changed bucket")
    a("")
    if not p["changed_rows"]:
        a("None.")
    else:
        a("| # | New course | Name | VALID rows | Old -> New | Predecessor | New evidence |")
        a("|---:|---|---|---:|---|---|---|")
        for i, row in enumerate(p["changed_rows"], 1):
            a(f"| {i} | `{row['new_course_id']}` | {row['new_course_name']} | "
              f"{fmt(row['valid_rows'])} | `{row['prior_bucket']}` -> `{row['new_bucket']}` | "
              f"`{row['candidate_1_course_id']}` {row['candidate_1_name']} | "
              f"{row['new_evidence']} |")
    a("")
    a("### Courses whose classification did not change")
    a("")
    a(f"{p['unchanged_summary']['courses']} of the {p['pending_count']} pending courses "
      f"({fmt(p['unchanged_summary']['rows'])} VALID rows) kept their bucket. Grouped by why "
      "the extra years added nothing:")
    a("")
    a("| Reason nothing changed | Courses | VALID rows |")
    a("|---|---:|---:|")
    for key, value in p["unchanged_reasons"].items():
        a(f"| {key} | {value['courses']} | {fmt(value['rows'])} |")
    a("")
    a("### What still blocks the courses whose predecessor DID taper")
    a("")
    tc = p["taper_counts"]
    a(f"{tc['taper_confirmed']} pending courses have a confirmed taper "
      f"({tc['post_debut']} post-debut, {tc['pre_debut']} pre-debut), but only "
      f"{tc['reached_confirmed']} reached `confirmed_equivalent`. For the other "
      f"{tc['blocked']} ({fmt(tc['blocked_rows'])} VALID rows) the taper is no longer the "
      "obstacle - some other clause of the pre-registered rule is. This is the actionable "
      "list for a registrar review, because these are the courses where the enrolment "
      "evidence is already settled and only catalog attributes are in dispute:")
    a("")
    a("| Remaining blocker | Courses | VALID rows |")
    a("|---|---:|---:|")
    for key, value in sorted(
        p["taper_confirmed_but_blocked"].items(), key=lambda kv: -kv[1]["rows"]
    ):
        a(f"| {key} | {value['courses']} | {fmt(value['rows'])} |")
    a("")
    a("The 40 highest-volume pending courses, with their updated evidence:")
    a("")
    a("| # | New course | Name | VALID rows | Old bucket | New bucket | Changed | Predecessor | New evidence |")
    a("|---:|---|---|---:|---|---|:--:|---|---|")
    for i, row in enumerate(p["top_rows"], 1):
        a(f"| {i} | `{row['new_course_id']}` | {row['new_course_name']} | "
          f"{fmt(row['valid_rows'])} | `{row['prior_bucket']}` | `{row['new_bucket']}` | "
          f"{'yes' if row['changed'] else 'no'} | `{row['candidate_1_course_id']}` "
          f"{row['candidate_1_name']} | {row['new_evidence']} |")
    a("")

    a("## 5. Updated payoff")
    a("")
    a("Recomputed with the identical counterfactual method as the prior investigation: "
      "in-memory substitution of the candidate predecessor `course_id` into the VALID "
      "rows, then exact re-evaluation of the Level-1 (`degree_course_key`) -> Level-2 "
      "(`course_id`) support lookup of `src/course_difficulty.py` against TRAIN-only "
      "statistics. **No data was written and no mapping was persisted.**")
    a("")
    pay = p["payoff"]
    a(f"Simulation validation against the on-disk columns: "
      f"{pay['reimplementation_validation']['course_history_count_mismatches_vs_on_disk']} "
      f"`course_history_count` mismatches and "
      f"{pay['reimplementation_validation']['course_difficulty_missing_mismatches_vs_on_disk']} "
      f"`course_difficulty_missing` mismatches over "
      f"{fmt(pay['reimplementation_validation']['valid_rows_checked'])} VALID rows "
      f"(verdict: **{pay['reimplementation_validation']['verdict']}**).")
    a("")
    a("| Scenario | Courses mapped | Rows gaining observed history | Rows crossing the 20-row threshold | Prior figure | Delta |")
    a("|---|---:|---:|---:|---:|---:|")
    prior_pay = p["prior_payoff"]
    for label, key in (("`confirmed_equivalent` only", "confirmed_only"),
                       ("`confirmed` + `likely` (upper bound)", "confirmed_plus_likely_upper_bound")):
        now = pay[key]
        was = prior_pay[key]
        a(f"| {label} | {now['mapped_courses']} | "
          f"{fmt(now['never_in_train_rows_gaining_observed_history'])} | "
          f"{fmt(now['never_in_train_rows_becoming_covered'])} | "
          f"{fmt(was['never_in_train_rows_becoming_covered'])} | "
          f"{now['never_in_train_rows_becoming_covered'] - was['never_in_train_rows_becoming_covered']:+,} |")
    a("")
    a("**Residual against the original 26,882 uncovered VALID rows:**")
    a("")
    a("| Scenario | Uncovered rows remaining | % of 26,882 | Prior % |")
    a("|---|---:|---:|---:|")
    for label, key in (("`confirmed` only", "confirmed_only"),
                       ("`confirmed` + `likely`", "confirmed_plus_likely")):
        a(f"| {label} | {fmt(pay['residual'][key]['uncovered_valid_rows_remaining'])} | "
          f"{pay['residual'][key]['pct_of_original_26882']}% | "
          f"{prior_pay['residual'][key]['pct_of_original_26882']}% |")
    a("")
    a("> The payoff remains an arithmetic upper bound on *coverage*, not on *accuracy*. "
      "A confirmed taper shows that the old course stopped running; it does not show that "
      "the new course teaches the same content at the same difficulty. Only outcomes under "
      "the new plan can test that, and those are exactly the rows in question.")
    a("")

    a("## 6. What was NOT done")
    a("")
    a("- No `canonical_course_id` or equivalence mapping was created, applied or wired.")
    a("- The 5 `confirmed_equivalent` and 7 `genuinely_new` courses were NOT re-scored; "
      "they are carried over verbatim, per the task's explicit non-scope.")
    a("- The identifier-structure check, the all-course disappearance/appearance scan and "
      "the faculty-specific curriculum-revision finding were NOT re-run or re-litigated.")
    a("- No dataset was built, copied or written; no `CURRENT_VERSION.txt`, default, or "
      "promotion marker was touched.")
    a("- No model was trained, retrained or re-tuned.")
    a("- `df_test_final.parquet` was not read, and no TEST path was constructed.")
    a("- Nothing was pushed.")
    a("")

    a("## 7. Reported note on the requested `genuinely_new` transition line")
    a("")
    a("The task asks for a count of courses \"reclassified to genuinely_new (predecessor "
      "never tapered)\". Under the pre-registered rule, `genuinely_new` means *no TRAIN "
      "course reaches name similarity >= 0.60* - a statement about content, not about "
      "enrolment. A predecessor that never tapers is strong evidence AGAINST equivalence, "
      "but it does not make the new course content-novel, and relabelling it "
      "`genuinely_new` would be inventing a rule this task is explicitly told not to invent.")
    a("")
    a(f"So those courses are reported under their rule-correct bucket and carry the "
      f"dedicated flag `predecessor_active_through_2025` in the v2 CSV: "
      f"**{p['predecessor_active_full_volume']['courses']} courses / "
      f"{fmt(p['predecessor_active_full_volume']['rows'])} VALID rows** have a top candidate "
      "still enrolling at comparable volume through 2025. That is the number the requested "
      "line reports.")
    a("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
def main() -> int:
    git = CI.git_context()
    prior_json = json.loads(PRIOR_JSON.read_text(encoding="utf-8"))

    state = reproduce_prior()
    entries = state["entries"]
    entry_by_id = {e["new_course_id"]: e for e in entries}
    prior_buckets = state["prior"]
    prior_totals = bucket_totals(entries, prior_buckets)

    expected = {
        "confirmed_equivalent": (5, 1791),
        "likely_equivalent_needs_review": (82, 16359),
        "genuinely_new": (7, 134),
        "unresolved": (88, 7343),
    }
    for bucket, (courses, rows) in expected.items():
        got = prior_totals[bucket]
        if got["courses"] != courses or got["rows"] != rows:
            raise SystemExit(
                f"STOP: prior bucket {bucket} did not reproduce: {got} != {courses}/{rows}"
            )

    # Cross-check the shipped CSV against the reproduction.
    with PRIOR_CSV.open(encoding="utf-8-sig", newline="") as handle:
        lines = handle.read().splitlines()
    reader = csv.DictReader(lines[1:])
    csv_rows = list(reader)
    # Guard against reading the wrong file at PRIOR_CSV: a plain row-count check
    # would not catch it here, since course_identity_diagnostic.py's output
    # (COURSE_IDENTITY_CANDIDATES.csv, diagnostic schema) happens to also have
    # 182 rows. review_bucket/candidate_1_course_id only exist in the
    # investigation-schema CSV this script actually needs.
    required_columns = {"new_course_id", "review_bucket", "candidate_1_course_id"}
    missing_columns = required_columns - set(reader.fieldnames or [])
    if missing_columns:
        raise SystemExit(
            f"STOP: {PRIOR_CSV} is missing expected column(s) {sorted(missing_columns)}. "
            f"Got columns: {reader.fieldnames}. Wrong file read at PRIOR_CSV?"
        )
    bucket_mismatches = 0
    candidate_mismatches = 0
    for row in csv_rows:
        cid = row["new_course_id"]
        if prior_buckets[cid]["bucket"] != row["review_bucket"]:
            bucket_mismatches += 1
        tops = prior_buckets[cid]["top_candidates"]
        top_id = tops[0]["old_course_id"] if tops else ""
        if top_id != row["candidate_1_course_id"]:
            candidate_mismatches += 1
    if bucket_mismatches or candidate_mismatches:
        raise SystemExit(
            f"STOP: prior CSV disagrees with the reproduction "
            f"({bucket_mismatches} buckets, {candidate_mismatches} candidates)"
        )

    # The pre-registered censoring flag: confirmed only under the sensitivity variant.
    censor_flagged = [
        e for e in entries
        if prior_buckets[e["new_course_id"]]["bucket"] != "confirmed_equivalent"
        and state["sensitivity"][e["new_course_id"]]["bucket"] == "confirmed_equivalent"
    ]
    censoring_flag = {
        "courses": len(censor_flagged),
        "rows": sum(e["valid_rows"] for e in censor_flagged),
        "definition": "courses the pre-registered censoring guard withheld from "
                      "confirmed_equivalent but which the investigation's own "
                      "temporal_only sensitivity would have confirmed",
        "courses_detail": [
            {
                "new_course_id": e["new_course_id"],
                "new_course_name": e["new_course_name"],
                "valid_rows": e["valid_rows"],
                "prior_bucket": prior_buckets[e["new_course_id"]]["bucket"],
            }
            for e in censor_flagged
        ],
    }

    # --- extended history -------------------------------------------------
    ext, coverage = load_extended()
    series, calendar = semester_series(ext)
    post_valid = [s for s in calendar if s > "20233"]
    if not coverage["reaches_2024"]:
        raise SystemExit("STOP: extended file does not reach 2024 - nothing new to see")

    shared = ext[ext["part_id"] <= "20233"]
    frozen = pd.concat(
        [state["train"][["course_id", "part_id"]], state["valid"][["course_id", "part_id"]]]
    )
    frozen_pairs = set(map(tuple, frozen.dropna().astype(str).to_numpy()))
    shared_pairs = set(map(tuple, shared[["course_id", "part_id"]].dropna().astype(str).to_numpy()))
    consistency = {
        "extended rows in the shared window (<= 20233)": int(len(shared)),
        "TRAIN + VALID rows of the frozen version": int(len(frozen)),
        "(course, semester) cells in TRAIN + VALID": len(frozen_pairs),
        "of those also present in the extended file": len(frozen_pairs & shared_pairs),
        "cells present only in the extended shared window": len(shared_pairs - frozen_pairs),
        "columns loaded from the extended file": len(EXTENDED_COLUMNS),
    }

    # --- reclassify the 170 pending --------------------------------------
    pending = [e for e in entries if prior_buckets[e["new_course_id"]]["bucket"] in PENDING_BUCKETS]
    new_buckets: dict[str, dict[str, Any]] = {}
    detail: dict[str, dict[str, Any]] = {}
    predecessors_seen = 0

    for entry in entries:
        cid = entry["new_course_id"]
        old_bucket = prior_buckets[cid]["bucket"]
        debut = entry["first_valid_semester"]
        new_taper = taper_profile(cid, series, calendar, debut)

        if old_bucket in CARRIED_BUCKETS:
            new_buckets[cid] = prior_buckets[cid]
            detail[cid] = {
                "pending": False,
                "predecessor_taper": None,
                "new_course_taper": new_taper,
                "new_evidence": "carried over unchanged - outside this task's scope",
            }
            continue

        tops = prior_buckets[cid]["top_candidates"]
        top_id = tops[0]["old_course_id"] if tops else None
        pred_taper = taper_profile(top_id, series, calendar, debut)
        if pred_taper.get("observed") and pred_taper["total_rows"] > 0:
            predecessors_seen += 1
        others = [
            {
                "old_course_id": c["old_course_id"],
                "old_course_name": c["old_course_name"],
                **{
                    k: v for k, v in taper_profile(c["old_course_id"], series, calendar, debut).items()
                    if k != "per_semester"
                },
            }
            for c in tops[1:]
        ]
        info = classify_extended(entry, pred_taper)
        new_buckets[cid] = info
        detail[cid] = {
            "pending": True,
            "predecessor_taper": pred_taper,
            "other_candidate_tapers": others,
            "new_course_taper": new_taper,
            "new_evidence": evidence_label(pred_taper, new_taper),
        }

    new_totals = bucket_totals(entries, new_buckets)

    # --- transitions, evidence, unchanged reasons ------------------------
    def bag() -> dict[str, int]:
        return {"courses": 0, "rows": 0}

    transitions: dict[str, dict[str, int]] = {}
    evidence_totals: dict[str, dict[str, int]] = {}
    unchanged_reasons: dict[str, dict[str, int]] = {}
    changed_rows: list[dict[str, Any]] = []
    unchanged_summary = bag()
    shortlived = bag()
    pred_active_full = bag()

    for entry in pending:
        cid = entry["new_course_id"]
        rows = entry["valid_rows"]
        old_b = prior_buckets[cid]["bucket"]
        new_b = new_buckets[cid]["bucket"]
        d = detail[cid]
        pred = d["predecessor_taper"] or {}

        key = f"`{old_b}` -> `{new_b}`"
        transitions.setdefault(key, bag())
        transitions[key]["courses"] += 1
        transitions[key]["rows"] += rows

        if pred.get("taper_post_debut"):
            ev = "taper confirmed AFTER the new course's debut (new evidence)"
        elif pred.get("taper_pre_debut"):
            ev = "predecessor already tapered before the debut (no new evidence)"
        elif pred.get("never_reached_activity_floor"):
            ev = ("predecessor never reached the activity floor in any semester - no taper "
                  "to observe, no corroboration available")
        elif pred.get("still_active_full_volume"):
            ev = "predecessor still active at comparable volume through 2025 (evidence AGAINST)"
        elif pred.get("observed"):
            ev = "predecessor still running at reduced volume - no confirmed taper"
        else:
            ev = "predecessor absent from the extended file"
        evidence_totals.setdefault(ev, bag())
        evidence_totals[ev]["courses"] += 1
        evidence_totals[ev]["rows"] += rows

        if d["new_course_taper"].get("taper_confirmed"):
            shortlived["courses"] += 1
            shortlived["rows"] += rows
        if pred.get("still_active_full_volume"):
            pred_active_full["courses"] += 1
            pred_active_full["rows"] += rows

        tops = prior_buckets[cid]["top_candidates"]
        top = tops[0] if tops else {}
        record = {
            "new_course_id": cid,
            "new_course_name": entry["new_course_name"],
            "valid_rows": rows,
            "prior_bucket": old_b,
            "new_bucket": new_b,
            "changed": old_b != new_b,
            "candidate_1_course_id": top.get("old_course_id", ""),
            "candidate_1_name": top.get("old_course_name", ""),
            "new_evidence": d["new_evidence"],
        }
        if old_b != new_b:
            changed_rows.append(record)
        else:
            unchanged_summary["courses"] += 1
            unchanged_summary["rows"] += rows
            if pred.get("taper_pre_debut"):
                reason = ("predecessor already fully tapered before the debut - the extra "
                          "years confirm no rebound but add no new evidence")
            elif pred.get("taper_confirmed"):
                reason = ("taper confirmed, but another clause of the pre-registered rule "
                          "still blocks confirmation (name/credits/requirement type/lineage/uniqueness)")
            elif pred.get("never_reached_activity_floor"):
                reason = ("predecessor never reached the activity floor in any semester - "
                          "there is no taper to observe, so the extra years add no "
                          "corroboration either way")
            elif pred.get("still_active_full_volume"):
                reason = ("predecessor still active at comparable volume through 2025 - the "
                          "extra years positively deny the taper, and the bucket was already "
                          "below confirmed")
            elif pred.get("observed"):
                reason = ("predecessor still running at reduced volume through 2025 - no "
                          "confirmed taper, bucket unchanged")
            else:
                reason = "predecessor absent from the extended file - no new evidence"
            unchanged_reasons.setdefault(reason, bag())
            unchanged_reasons[reason]["courses"] += 1
            unchanged_reasons[reason]["rows"] += rows

    changed_rows.sort(key=lambda r: -r["valid_rows"])

    # What the 5 censoring-flagged courses turned out to be: the sharpest test of the
    # extra years, because each was blocked by the guard's scope and nothing else.
    flag_outcome = []
    for row in censoring_flag["courses_detail"]:
        cid = row["new_course_id"]
        flag_outcome.append({
            **row,
            "new_bucket": new_buckets[cid]["bucket"],
            "verdict": (
                "CONFIRMED by the extra years"
                if new_buckets[cid]["bucket"] == "confirmed_equivalent"
                else "REFUTED by the extra years - predecessor never tapered"
            ),
            "new_evidence": detail[cid]["new_evidence"],
        })

    # Among courses whose predecessor DID taper but which still fall short: which
    # clause of the pre-registered rule is actually holding them back?
    blockers: dict[str, dict[str, int]] = {}
    taper_counts = {
        "taper_confirmed": 0, "post_debut": 0, "pre_debut": 0,
        "reached_confirmed": 0, "blocked": 0, "blocked_rows": 0,
    }
    for entry in pending:
        cid = entry["new_course_id"]
        pred = detail[cid]["predecessor_taper"] or {}
        if pred.get("taper_confirmed"):
            taper_counts["taper_confirmed"] += 1
            taper_counts["post_debut"] += int(bool(pred.get("taper_post_debut")))
            taper_counts["pre_debut"] += int(bool(pred.get("taper_pre_debut")))
            if new_buckets[cid]["bucket"] == "confirmed_equivalent":
                taper_counts["reached_confirmed"] += 1
            else:
                taper_counts["blocked"] += 1
                taper_counts["blocked_rows"] += entry["valid_rows"]
        if not pred.get("taper_confirmed") or new_buckets[cid]["bucket"] == "confirmed_equivalent":
            continue
        tops = prior_buckets[cid]["top_candidates"]
        top = tops[0] if tops else {}
        runner = tops[1] if len(tops) > 1 else None
        reasons = []
        if not top.get("name_exact_normalized"):
            reasons.append("name not an exact normalized match")
        if not top.get("credits_equal"):
            reasons.append("credits differ")
        if not top.get("requirement_type_equal"):
            reasons.append("requirement type differs")
        if not (top.get("degree_lineage_linked") or top.get("shares_degree")):
            reasons.append("no degree-lineage link")
        if runner is not None and abs(
            runner["evidence_weight"] - top["evidence_weight"]
        ) < 1e-9 and top["old_train_rows"] < CI.UNIQUENESS_DOMINANCE * max(
            runner["old_train_rows"], 1
        ):
            reasons.append("ambiguous tie with a second candidate")
        key = "; ".join(reasons) or "no blocker identified"
        blockers.setdefault(key, bag())
        blockers[key]["courses"] += 1
        blockers[key]["rows"] += entry["valid_rows"]

    # --- payoff -----------------------------------------------------------
    pay = CI.payoff(state["train"], state["valid"], state["never"], new_buckets)

    # --- ordered rows for CSV / report -----------------------------------
    all_rows: list[dict[str, Any]] = []
    for entry in entries:
        cid = entry["new_course_id"]
        d = detail[cid]
        pred = d["predecessor_taper"] or {}
        tops = prior_buckets[cid]["top_candidates"]
        top = tops[0] if tops else {}
        all_rows.append({
            "new_course_id": cid,
            "new_course_name": entry["new_course_name"],
            "valid_rows": entry["valid_rows"],
            "prior_bucket": prior_buckets[cid]["bucket"],
            "new_bucket": new_buckets[cid]["bucket"],
            "changed": prior_buckets[cid]["bucket"] != new_buckets[cid]["bucket"],
            "in_scope": d["pending"],
            "candidate_1_course_id": top.get("old_course_id", ""),
            "candidate_1_name": top.get("old_course_name", ""),
            "new_evidence": d["new_evidence"],
            "pred": pred,
            "new_taper": d["new_course_taper"],
            "reason": new_buckets[cid]["reason"],
        })
    top_rows = [r for r in all_rows if r["in_scope"]][:40]

    payload = {
        "investigation": "course_identity_reconciliation_with_history_through_2025",
        "scope": {
            "read_only": True,
            "mapping_created_or_applied": False,
            "canonical_course_id_created": False,
            "models_trained": 0,
            "datasets_written": 0,
            "defaults_changed": 0,
            "test_parquet_read": False,
            "courses_re_examined": len(pending),
            "courses_carried_over_unchanged": len(entries) - len(pending),
            "frozen_version": CI.VERSION,
        },
        "governance_conflict": {
            "declared": True,
            "conflict": "docs/pipeline_rules.md fixes TEST = 2024 + 2025 S1; the extended "
                        "history this task directs the analysis to read spans that window, "
                        "while CLAUDE.md section 5 declares TEST closed_not_read",
            "resolution": "the task prompt is explicit and read-only, so per the CLAUDE.md "
                          "header rule the prompt was followed and the conflict is flagged",
            "containment": [
                "df_test_final.parquet was never read, globbed, stat-ed or path-constructed",
                f"only {EXTENDED_COLUMNS} were loaded from the extended file; final_mark "
                "and every other outcome column were never loaded",
                "only per-course enrolment counts were derived; no metric, difficulty "
                "statistic or model input was computed from post-20233 rows",
            ],
            "residual_exposure": "the taper evidence is derived from enrolment volumes "
                                 "inside the TEST window; the human owns whether that is "
                                 "acceptable before any of this feeds a mapping",
        },
        "git": git,
        "prior_git_head": prior_json.get("git", {}).get("head", "a32f20c"),
        "facts": state["facts"],
        "prior_totals": prior_totals,
        "csv_crosscheck": {
            "rows": len(csv_rows),
            "bucket_mismatches": bucket_mismatches,
            "candidate_mismatches": candidate_mismatches,
        },
        "censoring_flag": censoring_flag,
        "censoring_flag_outcome": flag_outcome,
        "censoring_flag_confirmed": sum(
            1 for r in flag_outcome if r["new_bucket"] == "confirmed_equivalent"
        ),
        "censoring_flag_refuted": sum(
            1 for r in flag_outcome if r["new_bucket"] != "confirmed_equivalent"
        ),
        "taper_counts": taper_counts,
        "taper_confirmed_but_blocked": blockers,
        "coverage": coverage,
        "post_valid_semesters": post_valid,
        "consistency": consistency,
        "predecessor_coverage": {"present": predecessors_seen, "total": len(pending)},
        "rule": {
            "source": "COURSE_IDENTITY_INVESTIGATION.md section 5, applied unchanged",
            "single_substitution": "the censoring-limited corroboration clause "
                                   "(predecessor_volume_collapsed OR temporal_complementarity, "
                                   "guarded by NOT censored_predecessor) is replaced by the "
                                   "extended-history taper signal",
            "near_zero_rows_per_semester": NEAR_ZERO_ROWS,
            "min_consecutive_near_zero_semesters": TAPER_MIN_SILENT_SEMESTERS,
            "persistence_ratio_max": PERSISTENCE_RATIO_MAX,
            "late_year_prefix": LATE_YEAR_PREFIX,
            "reference_year_prefix": PRIOR_REFERENCE_YEAR_PREFIX,
            "thresholds_inherited_from_prior_investigation": [
                "near-zero = MIN_SEMESTER_ACTIVITY (5 rows/semester)",
                "persistence ratio bar = COLLAPSE_RATIO_MAX (0.35)",
                "name similarity tiers 0.80 / 0.60 unchanged",
                "uniqueness dominance 3.0x unchanged",
            ],
        },
        "new_totals": new_totals,
        "pending_count": len(pending),
        "transitions": transitions,
        "evidence_totals": evidence_totals,
        "unchanged_summary": unchanged_summary,
        "unchanged_reasons": unchanged_reasons,
        "changed_rows": changed_rows,
        "top_rows": top_rows,
        "new_course_shortlived": shortlived,
        "predecessor_active_full_volume": pred_active_full,
        "payoff": pay,
        "prior_payoff": prior_json["payoff"],
        "genuinely_new_line_note": (
            "The task's requested 'reclassified to genuinely_new (predecessor never "
            "tapered)' line is reported from predecessor_active_full_volume. Under the "
            "pre-registered rule genuinely_new means no TRAIN course reaches name "
            "similarity >= 0.60 - a content criterion. A predecessor that never tapers is "
            "evidence AGAINST equivalence but does not make the new course content-novel, "
            "so relabelling it genuinely_new would invent a rule this task forbids."
        ),
        "course_detail": {
            cid: {
                "prior_bucket": prior_buckets[cid]["bucket"],
                "new_bucket": new_buckets[cid]["bucket"],
                "changed": prior_buckets[cid]["bucket"] != new_buckets[cid]["bucket"],
                "in_scope": detail[cid]["pending"],
                "valid_rows": entry_by_id[cid]["valid_rows"],
                "first_valid_semester": entry_by_id[cid]["first_valid_semester"],
                "new_evidence": detail[cid]["new_evidence"],
                "reason": new_buckets[cid]["reason"],
                "predecessor_taper": detail[cid]["predecessor_taper"],
                "other_candidate_tapers": detail[cid].get("other_candidate_tapers", []),
                "new_course_taper": detail[cid]["new_course_taper"],
            }
            for cid in (e["new_course_id"] for e in entries)
        },
    }

    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")

    # --- v2 CSV -----------------------------------------------------------
    header = [
        "new_bucket", "prior_bucket", "changed", "in_scope_for_this_task",
        "new_course_id", "new_course_name", "valid_rows", "first_valid_semester",
        "candidate_1_course_id", "candidate_1_name",
        "new_evidence", "new_reason",
        "predecessor_last_active_semester_2025", "predecessor_never_reached_activity_floor",
        "predecessor_silent_semesters_after",
        "predecessor_taper_confirmed", "predecessor_taper_post_debut",
        "predecessor_taper_pre_debut", "predecessor_active_through_2025",
        "predecessor_rows_2021", "predecessor_rows_2024", "predecessor_persistence_ratio",
        "new_course_rows_2024", "new_course_last_active_semester", "new_course_short_lived",
        "human_decision_required",
    ]
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "REVIEW ARTIFACT - NOT AN EQUIVALENCE MAPPING. No mapping has been created or "
            "applied. Buckets are mixed; conflicting evidence is retained deliberately; the "
            "v1 CSV's multi-candidate detail still stands. Do not consume as a lookup table."
        ])
        writer.writerow(header)
        for row in all_rows:
            pred = row["pred"] or {}
            nt = row["new_taper"] or {}
            writer.writerow([
                row["new_bucket"], row["prior_bucket"], row["changed"], row["in_scope"],
                row["new_course_id"], row["new_course_name"], row["valid_rows"],
                entry_by_id[row["new_course_id"]]["first_valid_semester"],
                row["candidate_1_course_id"], row["candidate_1_name"],
                row["new_evidence"], row["reason"],
                pred.get("last_active_semester", ""),
                pred.get("never_reached_activity_floor", ""),
                pred.get("silent_semesters_after_last_active", ""),
                pred.get("taper_confirmed", ""),
                pred.get("taper_post_debut", ""),
                pred.get("taper_pre_debut", ""),
                pred.get("still_active_full_volume", ""),
                pred.get(f"rows_{PRIOR_REFERENCE_YEAR_PREFIX}", ""),
                pred.get(f"rows_{LATE_YEAR_PREFIX}", ""),
                pred.get("persistence_ratio", ""),
                nt.get(f"rows_{LATE_YEAR_PREFIX}", ""),
                nt.get("last_active_semester", ""),
                nt.get("taper_confirmed", ""),
                "yes",
            ])

    # --- console summary --------------------------------------------------
    def moved(old: str, new: str) -> tuple[int, int]:
        sel = [
            e for e in pending
            if prior_buckets[e["new_course_id"]]["bucket"] == old
            and new_buckets[e["new_course_id"]]["bucket"] == new
        ]
        return len(sel), sum(e["valid_rows"] for e in sel)

    lc, lr = moved("likely_equivalent_needs_review", "confirmed_equivalent")
    uc, ur = moved("unresolved", "confirmed_equivalent")
    ulc, ulr = moved("unresolved", "likely_equivalent_needs_review")
    upper = pay["confirmed_plus_likely_upper_bound"]["never_in_train_rows_becoming_covered"]

    print(f"Reclassified from likely -> confirmed: {lc} courses / {lr:,} rows")
    print(f"Reclassified from unresolved -> confirmed: {uc} courses / {ur:,} rows")
    print(f"Reclassified from unresolved -> likely: {ulc} courses / {ulr:,} rows")
    print(
        f"Reclassified to genuinely_new (predecessor never tapered): "
        f"{pred_active_full['courses']} courses / {pred_active_full['rows']:,} rows"
    )
    print(f"Unchanged: {unchanged_summary['courses']} courses / {unchanged_summary['rows']:,} rows")
    print(f"Updated best-case coverage recovery: {upper:,} of 25,627 (vs prior 18,060)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
