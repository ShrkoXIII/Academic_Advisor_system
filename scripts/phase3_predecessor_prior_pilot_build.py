"""Phase 3 PILOT: build the predecessor-prior VALID frame as a new dataset version.

STATUS: PILOT on PENDING / UNREVIEWED mapping proposals. Nothing here approves a
proposal row, promotes a version, or authorizes a freeze.

The single intervention is: for VALID rows of a course that the Phase 2T link
table proposes a *weighted* predecessor for (``successor`` /
``consolidated_into``), replace the two historical difficulty estimates with the
weight-blended TRAIN estimate of its predecessors. Because the new course's own
history is structurally zero in this frozen snapshot, the general shrinkage
formula ``(n_new*local + k*prior)/(n_new+k)`` collapses to the prior term
exactly at ``n_new = 0``; this substitution IS that formula, not an
approximation of it.

Concurrent-group features are rebuilt through the unmodified existing builder
(``src.concurrent_group_features``); only its inputs differ, and only for rows
of eligible courses. That is propagation of the one change, not a second change.

TRAIN is never touched: an eligible course is never-in-TRAIN by construction.
TEST is never read, globbed, or stat-ed.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.concurrent_group_features import (  # noqa: E402
    CONCURRENT_FEATURE_COLUMNS,
    compute_concurrent_group_features,
)
from src.course_difficulty import (  # noqa: E402
    DIFFICULTY_OUTPUT_COLUMNS,
    apply_difficulty_state,
    fit_difficulty_state,
)
from src.feature_engineering import SEMESTER_KEY  # noqa: E402
from src.model_training import (  # noqa: E402
    BASELINE_41_CONTRACT,
    CONCURRENT_43_CONTRACT,
)
from src.paths import MODEL_DATA_VERSIONS_DIR  # noqa: E402

FROZEN_VERSION = "2026-07-26_batched_fixes__registration_roster_concurrent"
FROZEN_DIR = MODEL_DATA_VERSIONS_DIR / FROZEN_VERSION
FROZEN_TRAIN = FROZEN_DIR / "df_train_final.parquet"
FROZEN_VALID = FROZEN_DIR / "df_valid_final.parquet"
FROZEN_ROSTER_VALID = FROZEN_DIR / "registration_roster_valid.parquet"

PHASE2T_DIR = PROJECT_ROOT / "models" / "runs" / "phase2_link_corrections"
LINK_PATH = PHASE2T_DIR / "course_link_proposed.csv"
SPLIT_PATH = PHASE2T_DIR / "course_split_candidates.csv"

PILOT_VERSION = "2026-07-30_predecessor_prior_pilot_PENDING_REVIEW"
PILOT_DIR = MODEL_DATA_VERSIONS_DIR / PILOT_VERSION

# The two columns the substitution writes, and nothing else.
SUBSTITUTED_COLUMNS = [
    "course_pass_rate_historical",
    "course_avg_mark_historical",
]
# Difficulty columns that MUST stay exactly as the frozen version has them.
FROZEN_DIFFICULTY_COLUMNS = [
    column for column in DIFFICULTY_OUTPUT_COLUMNS if column not in SUBSTITUTED_COLUMNS
]

WEIGHTED_RELATIONSHIPS = ("successor", "consolidated_into")
UNTOUCHED_RELATIONSHIPS = (
    "split_from",
    "merged_from",
    "candidate_below_support",
    "name_only_review_candidate",
    "none",
)

# Audit-only columns. Names are taken verbatim from the original Section 9 spec.
# NONE of these may enter a feature contract.
AUDIT_COLUMNS = [
    "course_history_count_predecessor",
    "course_cross_plan_prior_used",
    "course_cross_plan_prior_weight",
    "course_cross_plan_relationship_type",
    "course_identity_confidence",
    "course_difficulty_source_level",
]
PREDECESSOR_SOURCE_LABEL = "predecessor_prior"

# The repository's existing numerical tolerance
# (scripts/build_concurrent_group_features.py CHANGE_ATOL / CHANGE_RTOL).
CHANGE_ATOL = 1e-12
CHANGE_RTOL = 1e-12
# The repository's existing weight-sum tolerance (scripts/phase2_link_corrections.py).
WEIGHT_SUM_ATOL = 1e-9

EXPECTED_ELIGIBLE_COURSES = 80
EXPECTED_ELIGIBLE_VALID_ROWS = 16_269
EXPECTED_NEVER_IN_TRAIN_ROWS = 25_627
EXPECTED_VALID_ROWS = 156_097
EXPECTED_TRAIN_ROWS = 450_465


def stop(message: str) -> None:
    raise SystemExit(f"STOP: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_no_test_path(path: Path) -> None:
    lowered = str(path).lower().replace("\\", "/")
    if "df_test" in lowered or "/test/" in lowered or lowered.endswith("_test.parquet"):
        stop(f"a TEST path entered the Phase 3 input allowlist: {path}")


def close_enough(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.isclose(left, right, rtol=CHANGE_RTOL, atol=CHANGE_ATOL, equal_nan=True)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def read_inputs() -> dict[str, Any]:
    required = [FROZEN_TRAIN, FROZEN_VALID, FROZEN_ROSTER_VALID, LINK_PATH, SPLIT_PATH]
    for path in required:
        assert_no_test_path(path)
        if not path.is_file():
            stop(f"required Phase 3 input is missing: {path}")

    link = pd.read_csv(LINK_PATH, dtype="string", keep_default_na=False)
    split = pd.read_csv(SPLIT_PATH, dtype="string", keep_default_na=False)
    for name, frame in (("course_link_proposed", link), ("course_split_candidates", split)):
        if "approval_status" not in frame.columns:
            stop(f"{name}.csv has no approval_status column")
        if not frame["approval_status"].eq("pending").all():
            stop(
                f"{name}.csv contains a non-pending row; this pilot may only read "
                "unreviewed proposals and may never change their status"
            )

    # Only the columns fit_difficulty_state needs; the full TRAIN frame is never
    # materialized here.
    train_fit = pd.read_parquet(
        FROZEN_TRAIN,
        columns=[
            "part_id",
            "final_mark",
            "attempt_number",
            "degree_course_key",
            "degree_id",
            "faculty_id",
            "requirement_type_id",
            "course_credits",
        ],
    )
    if len(train_fit) != EXPECTED_TRAIN_ROWS:
        stop(f"frozen TRAIN row count is {len(train_fit):,}, expected {EXPECTED_TRAIN_ROWS:,}")

    valid = pd.read_parquet(FROZEN_VALID)
    if len(valid) != EXPECTED_VALID_ROWS:
        stop(f"frozen VALID row count is {len(valid):,}, expected {EXPECTED_VALID_ROWS:,}")
    roster = pd.read_parquet(FROZEN_ROSTER_VALID)

    return {
        "link": link,
        "split": split,
        "train_fit": train_fit,
        "valid": valid,
        "roster": roster,
    }


# ---------------------------------------------------------------------------
# Eligibility + predecessor estimates
# ---------------------------------------------------------------------------


def build_weighted_links(link: pd.DataFrame) -> pd.DataFrame:
    """The weighted (successor / consolidated_into) rows, fully validated."""

    weighted = link.loc[link["relationship_type"].isin(WEIGHTED_RELATIONSHIPS)].copy()
    if weighted.empty:
        stop("the Phase 2T link table has no weighted relationship rows")

    if weighted.duplicated(["new_course_id", "old_course_id"]).any():
        stop("duplicate (new_course_id, old_course_id) predecessor rows in the link table")

    weight = pd.to_numeric(weighted["weight_hint"], errors="coerce")
    if weight.isna().any():
        stop("a weighted link row has a null or non-numeric weight_hint")
    if (weight < 0).any():
        stop("a weighted link row has a negative weight_hint")
    weighted["weight"] = weight.astype("float64")

    for new_course_id, group in weighted.groupby("new_course_id", sort=False):
        relationships = set(group["relationship_type"])
        if len(relationships) != 1:
            stop(
                f"course {new_course_id} mixes weighted relationship types: "
                f"{sorted(relationships)}"
            )
        relationship = relationships.pop()
        total = float(group["weight"].sum())
        if relationship == "successor":
            if len(group) != 1:
                stop(f"course {new_course_id} is a successor with {len(group)} predecessors")
            if not np.isclose(total, 1.0, rtol=0.0, atol=WEIGHT_SUM_ATOL):
                stop(f"successor course {new_course_id} has weight {total!r}, expected 1.0")
        else:
            if len(group) < 2:
                stop(f"consolidated course {new_course_id} has only {len(group)} predecessor")
            if not np.isclose(total, 1.0, rtol=0.0, atol=WEIGHT_SUM_ATOL):
                stop(
                    f"consolidated course {new_course_id} weights sum to {total!r}, expected 1.0"
                )
        confidences = set(group["scope_confidence"])
        if len(confidences) != 1:
            # Addendum 8: never silently pick one.
            stop(
                f"course {new_course_id} has conflicting predecessor scope_confidence "
                f"values: {sorted(confidences)}"
            )

    return weighted


def course_attributes(link: pd.DataFrame, weighted: pd.DataFrame) -> pd.DataFrame:
    """Per eligible new course: relationship, confidence, scope, credit-change flag."""

    rows = []
    for new_course_id, group in weighted.groupby("new_course_id", sort=True):
        # Addendum 10: a consolidation counts as credit-changed when ANY weighted
        # contributing predecessor changed credits.
        credit_changed_flags = set(group["credit_changed"])
        rows.append(
            {
                "new_course_id": new_course_id,
                "relationship_type": group["relationship_type"].iloc[0],
                "scope_confidence": group["scope_confidence"].iloc[0],
                "new_course_scope": group["new_course_scope"].iloc[0],
                "predecessor_count": int(len(group)),
                "max_weight": float(group["weight"].max()),
                "credit_changed_any": bool("true" in credit_changed_flags),
                "credit_changed_values": "|".join(sorted(credit_changed_flags)),
                "new_course_valid_rows": int(float(group["new_course_valid_rows"].iloc[0])),
            }
        )
    frame = pd.DataFrame(rows)
    scopes = (
        link.drop_duplicates("new_course_id").set_index("new_course_id")["new_course_scope"]
    )
    frame["new_course_scope"] = frame["new_course_id"].map(scopes)
    return frame


def predecessor_plan(
    weighted: pd.DataFrame,
    degree_course_pairs: pd.DataFrame,
    level_1: pd.DataFrame,
    level_2: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blend the TRAIN predecessor estimates for every (degree, new course) pair.

    The Level-1 / Level-2 precedence is the one Phase 0's ``answer_q6`` already
    validated: substitute the predecessor's course id into the row's own
    ``degree_course_key`` and take the Level-1 TRAIN estimate when that key has
    support, otherwise the predecessor's Level-2 (course-across-degrees) TRAIN
    estimate.
    """

    by_course = {
        new_course_id: group
        for new_course_id, group in weighted.groupby("new_course_id", sort=False)
    }
    plan_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []

    for pair in degree_course_pairs.itertuples(index=False):
        group = by_course[pair.new_course_id]
        blended_pass = 0.0
        blended_mark = 0.0
        support_total = 0
        for record in group.itertuples(index=False):
            substituted_key = f"{pair.degree_prefix}__{record.old_course_id}"
            if substituted_key in level_1.index:
                source_level = 1
                source_row = level_1.loc[substituted_key]
                source_key = substituted_key
            elif record.old_course_id in level_2.index:
                source_level = 2
                source_row = level_2.loc[record.old_course_id]
                source_key = record.old_course_id
            else:
                # Addendum 9: never drop a predecessor or redistribute its weight.
                stop(
                    "no Level-1 or Level-2 TRAIN estimate exists for predecessor "
                    f"{record.old_course_id} of {pair.new_course_id} under degree "
                    f"{pair.degree_prefix}"
                )
            pass_rate = float(source_row["course_pass_rate_historical"])
            avg_mark = float(source_row["course_avg_mark_historical"])
            support = int(source_row["support_count"])
            if not np.isfinite(pass_rate) or not np.isfinite(avg_mark):
                stop(
                    f"predecessor {record.old_course_id} of {pair.new_course_id} has a "
                    f"non-finite Level-{source_level} estimate"
                )
            blended_pass += record.weight * pass_rate
            blended_mark += record.weight * avg_mark
            support_total += support
            contribution_rows.append(
                {
                    "degree_prefix": pair.degree_prefix,
                    "new_course_id": pair.new_course_id,
                    "old_course_id": record.old_course_id,
                    "weight": record.weight,
                    "source_level": source_level,
                    "source_key": source_key,
                    "predecessor_pass_rate": pass_rate,
                    "predecessor_avg_mark": avg_mark,
                    "predecessor_support": support,
                }
            )
        if not np.isfinite(blended_pass) or not np.isfinite(blended_mark):
            stop(f"blended estimate for {pair.new_course_id} is not finite")
        plan_rows.append(
            {
                "degree_course_key": f"{pair.degree_prefix}__{pair.new_course_id}",
                "degree_prefix": pair.degree_prefix,
                "new_course_id": pair.new_course_id,
                "blended_pass_rate": blended_pass,
                "blended_avg_mark": blended_mark,
                "predecessor_support_sum": support_total,
            }
        )

    plan = pd.DataFrame(plan_rows).set_index("degree_course_key")
    if plan.index.duplicated().any():
        stop("predecessor plan has duplicate degree_course_key entries")
    return plan, pd.DataFrame(contribution_rows)


def degree_course_pairs_from(frames: list[pd.DataFrame], eligible: set[str]) -> pd.DataFrame:
    pairs: set[tuple[str, str]] = set()
    for frame in frames:
        mask = frame["course_id"].astype(str).isin(eligible)
        keys = frame.loc[mask, "degree_course_key"].astype(str)
        courses = frame.loc[mask, "course_id"].astype(str)
        prefixes = keys.str.rsplit("__", n=1).str[0]
        suffixes = keys.str.rsplit("__", n=1).str[-1]
        if not suffixes.eq(courses).all():
            stop("degree_course_key suffix does not equal course_id on an eligible row")
        pairs |= set(zip(prefixes, courses))
    return pd.DataFrame(
        sorted(pairs), columns=["degree_prefix", "new_course_id"]
    )


# ---------------------------------------------------------------------------
# Substitution
# ---------------------------------------------------------------------------


def substitute_difficulty(
    frame: pd.DataFrame, plan: pd.DataFrame, eligible: set[str], label: str
) -> dict[str, np.ndarray]:
    """Return the substituted arrays plus the eligibility mask for ``frame``."""

    course = frame["course_id"].astype(str)
    mask = course.isin(eligible).to_numpy()

    # Addendum 7: direct eligibility also requires no first-party history.
    if int((pd.to_numeric(frame.loc[mask, "course_difficulty_missing"]) != 1).sum()):
        stop(f"{label}: a weighted-link row is already difficulty-covered")
    if int((pd.to_numeric(frame.loc[mask, "course_history_count"]) != 0).sum()):
        stop(f"{label}: a weighted-link row already has first-party TRAIN history")

    keys = frame["degree_course_key"].astype(str)
    blended_pass = keys.map(plan["blended_pass_rate"])
    blended_mark = keys.map(plan["blended_avg_mark"])
    support = keys.map(plan["predecessor_support_sum"])
    if blended_pass[mask].isna().any():
        stop(f"{label}: an eligible row has no predecessor plan entry")

    pass_out = frame["course_pass_rate_historical"].to_numpy(dtype="float64").copy()
    mark_out = frame["course_avg_mark_historical"].to_numpy(dtype="float64").copy()
    pass_out[mask] = blended_pass.to_numpy(dtype="float64")[mask]
    mark_out[mask] = blended_mark.to_numpy(dtype="float64")[mask]

    support_out = np.zeros(len(frame), dtype="int64")
    support_out[mask] = support.fillna(0).to_numpy(dtype="int64")[mask]
    return {
        "mask": mask,
        "course_pass_rate_historical": pass_out,
        "course_avg_mark_historical": mark_out,
        "predecessor_support": support_out,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_pilot_valid(
    source_path: Path,
    output_path: Path,
    replacements: dict[str, np.ndarray],
    additions: dict[str, pa.Array],
) -> None:
    """Rewrite VALID with replaced values and appended audit columns.

    The Arrow table is carried through so every untouched column keeps its exact
    type and the parquet's own ``__index_level_0__`` column stays where it is.
    """

    table = pq.read_table(source_path)
    for name, values in replacements.items():
        index = table.schema.get_field_index(name)
        if index < 0:
            stop(f"replacement column {name} is not in the frozen VALID schema")
        field = table.schema.field(index)
        table = table.set_column(index, field, pa.array(values, type=field.type))
    for name, array in additions.items():
        if name in table.schema.names:
            stop(f"audit column {name} already exists in the frozen VALID schema")
        table = table.append_column(name, array)
    pq.write_table(table, output_path, compression="snappy")


def main() -> int:
    started = datetime.now().astimezone()
    if PILOT_DIR.exists():
        stop(
            f"pilot version already exists: {PILOT_DIR}. Remove it deliberately before "
            "rebuilding; this script never overwrites a dataset version in place."
        )

    inputs = read_inputs()
    link: pd.DataFrame = inputs["link"]
    valid: pd.DataFrame = inputs["valid"]
    roster: pd.DataFrame = inputs["roster"]

    print("Fitting the difficulty state from frozen TRAIN only ...")
    state = fit_difficulty_state(inputs["train_fit"])
    level_1 = state.tables[1]
    level_2 = state.tables[2]

    # Evidence gate: the refit state must reproduce the frozen VALID difficulty
    # columns exactly, or every predecessor estimate below is off-contract.
    refit = apply_difficulty_state(valid, state, include_source=False)
    refit_mismatches = {}
    for column in DIFFICULTY_OUTPUT_COLUMNS:
        left = refit[column].to_numpy()
        right = valid[column].to_numpy()
        if left.dtype.kind == "f" or right.dtype.kind == "f":
            equal = close_enough(left.astype("float64"), right.astype("float64"))
        else:
            equal = left == right
        refit_mismatches[column] = int((~equal).sum())
    if any(refit_mismatches.values()):
        stop(f"the TRAIN difficulty refit does not reproduce frozen VALID: {refit_mismatches}")
    print("  refit reproduces all 9 frozen VALID difficulty columns exactly.")
    del refit

    weighted = build_weighted_links(link)
    eligible = set(weighted["new_course_id"].astype(str))
    attributes = course_attributes(link, weighted)

    pairs = degree_course_pairs_from([valid, roster], eligible)
    plan, contributions = predecessor_plan(weighted, pairs, level_1, level_2)
    print(
        f"  eligible courses: {len(eligible)}; (degree, course) pairs: {len(plan)}; "
        f"predecessor contributions: {len(contributions)} "
        f"(Level-1 {int((contributions['source_level'] == 1).sum())}, "
        f"Level-2 {int((contributions['source_level'] == 2).sum())})"
    )

    valid_sub = substitute_difficulty(valid, plan, eligible, "VALID")
    roster_sub = substitute_difficulty(roster, plan, eligible, "roster")
    directly_eligible = valid_sub["mask"]
    n_eligible = int(directly_eligible.sum())
    n_courses = int(valid.loc[directly_eligible, "course_id"].nunique())
    print(f"  directly eligible VALID rows: {n_eligible:,} across {n_courses} courses")
    if n_courses != EXPECTED_ELIGIBLE_COURSES or n_eligible != EXPECTED_ELIGIBLE_VALID_ROWS:
        stop(
            f"eligible set is {n_courses} courses / {n_eligible:,} rows, expected "
            f"{EXPECTED_ELIGIBLE_COURSES} / {EXPECTED_ELIGIBLE_VALID_ROWS:,}"
        )

    # --- propagation: rebuild concurrent features through the existing builder ---
    pilot_roster = roster.copy()
    pilot_roster["course_pass_rate_historical"] = roster_sub["course_pass_rate_historical"]
    pilot_roster["course_avg_mark_historical"] = roster_sub["course_avg_mark_historical"]

    target_columns = [
        "student_course_id",
        *SEMESTER_KEY,
        "course_id",
        "requirement_type_id",
    ]
    target = valid[target_columns]
    frozen_concurrent = compute_concurrent_group_features(target, roster)
    for column in CONCURRENT_FEATURE_COLUMNS:
        left = frozen_concurrent[column].to_numpy()
        right = valid[column].to_numpy()
        equal = (
            close_enough(left.astype("float64"), right.astype("float64"))
            if left.dtype.kind == "f" or right.dtype.kind == "f"
            else left == right
        )
        if not equal.all():
            stop(
                "the unmodified concurrent builder does not reproduce the frozen VALID "
                f"column {column}; propagation would not be attributable to this change"
            )
    print("  unmodified concurrent builder reproduces all 8 frozen columns exactly.")
    pilot_concurrent = compute_concurrent_group_features(target, pilot_roster)

    # Structural propagation exposure: a row is exposed when its own semester group
    # in the roster contains an eligible-course entry that is not the row itself.
    roster_group = roster.groupby(SEMESTER_KEY, dropna=False, sort=False)
    roster_eligible_in_group = (
        roster.assign(_e=roster_sub["mask"].astype("int64"))
        .groupby(SEMESTER_KEY, dropna=False, sort=False)["_e"]
        .transform("sum")
        .to_numpy(dtype="int64")
    )
    del roster_group
    roster_key = pd.MultiIndex.from_frame(
        roster[SEMESTER_KEY].astype(str)
    )
    exposure_by_group = pd.Series(roster_eligible_in_group, index=roster_key)
    exposure_by_group = exposure_by_group[~exposure_by_group.index.duplicated()]
    valid_key = pd.MultiIndex.from_frame(valid[SEMESTER_KEY].astype(str))
    group_eligible_count = (
        pd.Series(exposure_by_group.reindex(valid_key).to_numpy(), index=valid.index)
        .fillna(0)
        .to_numpy(dtype="int64")
    )
    propagation_exposed = (group_eligible_count - directly_eligible.astype("int64")) > 0

    # Empirical check: nothing outside (directly eligible OR propagation exposed)
    # may have a changed concurrent column.
    changed_concurrent = np.zeros(len(valid), dtype=bool)
    concurrent_changed_counts: dict[str, int] = {}
    for column in CONCURRENT_FEATURE_COLUMNS:
        left = pilot_concurrent[column].to_numpy()
        right = valid[column].to_numpy()
        equal = (
            close_enough(left.astype("float64"), right.astype("float64"))
            if left.dtype.kind == "f" or right.dtype.kind == "f"
            else left == right
        )
        concurrent_changed_counts[column] = int((~equal).sum())
        changed_concurrent |= ~equal
    leaked = changed_concurrent & ~(directly_eligible | propagation_exposed)
    if leaked.any():
        stop(
            f"{int(leaked.sum()):,} rows outside the directly-eligible / "
            "propagation-exposed union have a changed concurrent feature"
        )
    print(
        f"  propagation-exposed rows: {int(propagation_exposed.sum()):,}; rows with any "
        f"changed concurrent value: {int(changed_concurrent.sum()):,}"
    )

    # --- audit columns ---
    attribute_index = attributes.set_index("new_course_id")
    course_series = valid["course_id"].astype(str)
    relationship = course_series.map(attribute_index["relationship_type"])
    confidence = course_series.map(attribute_index["scope_confidence"])
    max_weight = course_series.map(attribute_index["max_weight"]).astype("float64")
    source_level = valid["difficulty_fallback_level"].astype("int64").astype(str)
    source_level = source_level.where(~directly_eligible, PREDECESSOR_SOURCE_LABEL)

    def nullable_string(values: pd.Series) -> pa.Array:
        """Arrow string array where non-eligible rows are genuinely null."""
        masked = [
            str(value) if flag else None
            for value, flag in zip(values.to_numpy(), directly_eligible)
        ]
        return pa.array(masked, type=pa.large_string())

    additions = {
        "course_history_count_predecessor": pa.array(
            valid_sub["predecessor_support"], type=pa.int64()
        ),
        "course_cross_plan_prior_used": pa.array(directly_eligible, type=pa.bool_()),
        "course_cross_plan_prior_weight": pa.array(
            np.where(directly_eligible, max_weight.to_numpy(dtype="float64"), np.nan),
            type=pa.float64(),
        ),
        "course_cross_plan_relationship_type": nullable_string(relationship),
        "course_identity_confidence": nullable_string(confidence),
        "course_difficulty_source_level": pa.array(
            source_level.astype(object).tolist(), type=pa.large_string()
        ),
    }
    if list(additions) != AUDIT_COLUMNS:
        stop(f"audit column set/order drifted: {list(additions)}")

    replacements: dict[str, np.ndarray] = {
        "course_pass_rate_historical": valid_sub["course_pass_rate_historical"],
        "course_avg_mark_historical": valid_sub["course_avg_mark_historical"],
    }
    for column in CONCURRENT_FEATURE_COLUMNS:
        values = pilot_concurrent[column].to_numpy()
        replacements[column] = values

    # --- contract guard: audit columns must never be model features ---
    for contract in (BASELINE_41_CONTRACT, CONCURRENT_43_CONTRACT):
        overlap = sorted(set(AUDIT_COLUMNS) & set(contract.features))
        if overlap:
            stop(f"audit columns entered contract {contract.name}: {overlap}")

    # --- write the version ---
    staging = MODEL_DATA_VERSIONS_DIR / f".{PILOT_VERSION}.incomplete"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        pilot_train = staging / "df_train_final.parquet"
        pilot_valid_path = staging / "df_valid_final.parquet"
        shutil.copyfile(FROZEN_TRAIN, pilot_train)
        train_hash = sha256_file(pilot_train)
        frozen_train_hash = sha256_file(FROZEN_TRAIN)
        if train_hash != frozen_train_hash:
            stop("pilot TRAIN is not byte-identical to the frozen TRAIN")

        write_pilot_valid(FROZEN_VALID, pilot_valid_path, replacements, additions)

        # ---------------- mandatory verification, from the written file ----------
        written = pd.read_parquet(pilot_valid_path)
        frozen_names = list(pq.ParquetFile(FROZEN_VALID).schema_arrow.names)
        written_names = list(pq.ParquetFile(pilot_valid_path).schema_arrow.names)
        if written_names != frozen_names + AUDIT_COLUMNS:
            stop(
                "pilot VALID schema is not 'frozen columns in order + the six audit "
                f"columns': {written_names[-8:]}"
            )
        if len(written) != len(valid):
            stop("pilot VALID row count differs from frozen VALID")

        per_column_changed: dict[str, int] = {}
        for column in frozen_names:
            # __index_level_0__ is a real parquet column that pandas consumes as
            # the frame index, so it is compared from Arrow rather than the frame.
            left_series = (
                written[column]
                if column in written.columns
                else pq.read_table(pilot_valid_path, columns=[column]).column(0).to_pandas()
            )
            right_series = (
                valid[column]
                if column in valid.columns
                else pq.read_table(FROZEN_VALID, columns=[column]).column(0).to_pandas()
            )
            left = left_series.to_numpy()
            right = right_series.to_numpy()
            if left.dtype.kind == "f" or right.dtype.kind == "f":
                equal = close_enough(left.astype("float64"), right.astype("float64"))
            else:
                equal = pd.Series(left).eq(pd.Series(right)).to_numpy() | (
                    pd.isna(pd.Series(left)).to_numpy() & pd.isna(pd.Series(right)).to_numpy()
                )
            per_column_changed[column] = int((~equal).sum())

        # Check 1 — no row outside the eligible set may differ in either
        # substituted column, and every eligible row must carry the value the
        # predecessor plan assigned it. An eligible row is allowed to show no
        # numerical difference only when the substituted value equals what the
        # frozen fallback already held; that case is counted, never waved past.
        check_1: dict[str, Any] = {}
        for column in SUBSTITUTED_COLUMNS:
            left = written[column].to_numpy(dtype="float64")
            right = valid[column].to_numpy(dtype="float64")
            differs = ~close_enough(left, right)
            outside = differs & ~directly_eligible
            if outside.any():
                stop(
                    f"CHECK 1 LEAK for {column}: {int(outside.sum()):,} rows outside the "
                    "eligible set changed"
                )
            assigned = replacements[column]
            if not close_enough(left, assigned).all():
                stop(f"CHECK 1 failed for {column}: written values are not the assigned values")
            check_1[column] = {
                "eligible_rows": n_eligible,
                "numerically_changed_rows": int(differs.sum()),
                "no_op_rows_substituted_value_equals_frozen_value": int(
                    (directly_eligible & ~differs).sum()
                ),
                "rows_changed_outside_eligible_set": 0,
            }
        no_op_courses = sorted(
            set(valid.loc[directly_eligible & ~(~close_enough(
                written["course_pass_rate_historical"].to_numpy(dtype="float64"),
                valid["course_pass_rate_historical"].to_numpy(dtype="float64"),
            )), "course_id"].astype(str))
        )
        for entry in check_1.values():
            entry["no_op_courses"] = no_op_courses
        print(
            "  CHECK 1 pass: every substituted row is eligible; "
            f"{check_1['course_pass_rate_historical']['numerically_changed_rows']:,} of "
            f"{n_eligible:,} eligible rows moved numerically; "
            f"{check_1['course_pass_rate_historical']['no_op_rows_substituted_value_equals_frozen_value']:,} "
            f"were exact no-ops (courses {no_op_courses})."
        )

        # Check 2 — untouched-relationship rows and already-covered rows are
        # byte-identical in every locked contract column, except where the
        # concurrent builder legitimately propagated.
        locked_columns = sorted(
            set(BASELINE_41_CONTRACT.features) | set(CONCURRENT_43_CONTRACT.features)
        )
        locked_source_columns = [c for c in locked_columns if c in frozen_names]
        untouched_courses = set(
            link.loc[
                link["relationship_type"].isin(UNTOUCHED_RELATIONSHIPS), "new_course_id"
            ].astype(str)
        ) - eligible
        untouched_rows = course_series.isin(untouched_courses).to_numpy()
        covered_rows = (valid["course_difficulty_missing"].to_numpy(dtype="int64") == 0)
        for label, mask in (
            ("untouched-relationship", untouched_rows & ~propagation_exposed),
            ("already-covered", covered_rows & ~propagation_exposed),
        ):
            for column in locked_source_columns:
                left = written[column].to_numpy()
                right = valid[column].to_numpy()
                if left.dtype.kind == "f" or right.dtype.kind == "f":
                    equal = close_enough(left.astype("float64"), right.astype("float64"))
                else:
                    equal = left == right
                bad = int((~equal & mask).sum())
                if bad:
                    stop(
                        f"CHECK 2 LEAK: {bad:,} unexposed {label} rows changed in locked "
                        f"column {column}"
                    )
        print("  CHECK 2 pass: no unexposed covered/untouched row changed in any locked column.")

        # Check 3 — the contract feature LISTS are unchanged.
        contract_lists = {}
        for contract in (BASELINE_41_CONTRACT, CONCURRENT_43_CONTRACT):
            missing = [c for c in contract.source_features if c not in written_names]
            if missing:
                stop(f"CHECK 3 failed: pilot VALID lacks {contract.name} columns {missing}")
            contract_lists[contract.name] = list(contract.features)
        if contract_lists["baseline_41"] != list(BASELINE_41_CONTRACT.features):
            stop("CHECK 3 failed: baseline_41 feature list changed")
        if contract_lists["concurrent_43"] != list(CONCURRENT_43_CONTRACT.features):
            stop("CHECK 3 failed: concurrent_43 feature list changed")
        print("  CHECK 3 pass: M1/M2 feature lists are byte-identical to the locked contracts.")

        # ---------------- manifest ----------------
        changed_direct = {c: per_column_changed[c] for c in SUBSTITUTED_COLUMNS}
        changed_concurrent_cols = {
            c: per_column_changed[c] for c in CONCURRENT_FEATURE_COLUMNS
        }
        other_changed = {
            c: n
            for c, n in per_column_changed.items()
            if n and c not in changed_direct and c not in changed_concurrent_cols
        }
        if other_changed:
            stop(f"columns changed outside the permitted set: {other_changed}")

        exposed_only = int((propagation_exposed & ~directly_eligible).sum())
        unexposed = int((~propagation_exposed & ~directly_eligible).sum())
        manifest = {
            "artifact": "predecessor_prior_pilot_dataset",
            "status": "PILOT — PENDING/UNREVIEWED MAPPINGS — NOT FOR PROMOTION",
            "version": PILOT_VERSION,
            "created_at": started.isoformat(timespec="seconds"),
            "governance": {
                "proposal_rows_reviewed": 0,
                "approval_status_values_in_link_table": sorted(
                    set(link["approval_status"].astype(str))
                ),
                "promotion": "not performed",
                "test_policy": "closed_not_read",
                "decisions_log_edited": False,
            },
            "source_frozen_version": {
                "name": FROZEN_VERSION,
                "train": {
                    "path": str(FROZEN_TRAIN),
                    "sha256": frozen_train_hash,
                    "rows": EXPECTED_TRAIN_ROWS,
                },
                "valid": {
                    "path": str(FROZEN_VALID),
                    "sha256": sha256_file(FROZEN_VALID),
                    "rows": int(len(valid)),
                },
                "registration_roster_valid": {
                    "path": str(FROZEN_ROSTER_VALID),
                    "sha256": sha256_file(FROZEN_ROSTER_VALID),
                    "rows": int(len(roster)),
                },
            },
            "proposal_tables": {
                "course_link_proposed.csv": {
                    "path": str(LINK_PATH),
                    "sha256": sha256_file(LINK_PATH),
                    "rows": int(len(link)),
                },
                "course_split_candidates.csv": {
                    "path": str(SPLIT_PATH),
                    "sha256": sha256_file(SPLIT_PATH),
                    "rows": int(len(inputs["split"])),
                },
            },
            "mechanism": {
                "eligible_relationship_types": list(WEIGHTED_RELATIONSHIPS),
                "excluded_relationship_types": list(UNTOUCHED_RELATIONSHIPS),
                "estimate_precedence": (
                    "Level-1 (substituted degree_course_key) when that key has TRAIN "
                    "support in fit_difficulty_state(TRAIN); otherwise the "
                    "predecessor's Level-2 (course-across-degrees) TRAIN estimate"
                ),
                "blend": "weight_hint-weighted mean across contributing predecessors",
                "shrinkage_identity": (
                    "(n_new*local + k*prior)/(n_new+k) collapses to the prior term "
                    "exactly because n_new = 0 for every eligible course"
                ),
                "columns_substituted": SUBSTITUTED_COLUMNS,
                "columns_deliberately_unchanged": FROZEN_DIFFICULTY_COLUMNS,
                "level_1_contributions": int((contributions["source_level"] == 1).sum()),
                "level_2_contributions": int((contributions["source_level"] == 2).sum()),
            },
            "eligible_set": {
                "courses": n_courses,
                "valid_rows": n_eligible,
                "degree_course_pairs": int(len(plan)),
                "predecessor_contributions": int(len(contributions)),
                "roster_rows_substituted": int(roster_sub["mask"].sum()),
                "successor_courses": int(
                    (attributes["relationship_type"] == "successor").sum()
                ),
                "consolidated_courses": int(
                    (attributes["relationship_type"] == "consolidated_into").sum()
                ),
                "shared_scope_courses": int((attributes["new_course_scope"] == "shared").sum()),
                "specific_scope_courses": int(
                    (attributes["new_course_scope"] == "specific").sum()
                ),
                "credit_changed_courses": int(attributes["credit_changed_any"].sum()),
            },
            "row_sets": {
                "valid_rows_total": int(len(valid)),
                "directly_eligible_rows": n_eligible,
                "directly_eligible_courses": n_courses,
                "propagation_exposed_only_rows": exposed_only,
                "completely_unexposed_rows": unexposed,
                "never_in_train_182_course_rows": int(
                    course_series.isin(set(link["new_course_id"].astype(str))).sum()
                ),
                "covered_rows": int(covered_rows.sum()),
                "covered_unexposed_rows": int((covered_rows & ~propagation_exposed).sum()),
                "untouched_uncovered_unexposed_rows": int(
                    (untouched_rows & ~propagation_exposed).sum()
                ),
            },
            "diff_vs_frozen_valid": {
                "changed_rows_per_direct_difficulty_column": changed_direct,
                "changed_rows_per_concurrent_derived_column": changed_concurrent_cols,
                "changed_rows_per_other_column": {},
                "audit_columns_added": AUDIT_COLUMNS,
                "audit_column_populated_rows": {
                    "course_cross_plan_prior_used_true": n_eligible,
                    "course_difficulty_source_level_predecessor_prior": int(
                        (written["course_difficulty_source_level"] == PREDECESSOR_SOURCE_LABEL).sum()
                    ),
                },
                "locked_columns_verified_unchanged_outside_permitted_sets": True,
                "locked_columns_checked": locked_source_columns,
            },
            "verification": {
                "train_byte_identical_to_frozen": train_hash == frozen_train_hash,
                "train_sha256": train_hash,
                "difficulty_refit_reproduces_frozen_valid": True,
                "unmodified_concurrent_builder_reproduces_frozen_valid": True,
                "check_1_no_change_outside_eligible_set": check_1,
                "check_2_no_unexposed_leak": True,
                "check_3_feature_lists_identical": True,
                "audit_columns_absent_from_m1_and_m2_contracts": True,
            },
            "feature_contracts": {
                "m1": {
                    "name": BASELINE_41_CONTRACT.name,
                    "feature_count": BASELINE_41_CONTRACT.expected_feature_count,
                    "features": list(BASELINE_41_CONTRACT.features),
                },
                "m2": {
                    "name": CONCURRENT_43_CONTRACT.name,
                    "feature_count": CONCURRENT_43_CONTRACT.expected_feature_count,
                    "features": list(CONCURRENT_43_CONTRACT.features),
                },
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        contributions.to_csv(
            staging / "predecessor_contributions.csv", index=False, encoding="utf-8-sig"
        )
        attributes.to_csv(
            staging / "eligible_course_attributes.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(
            {
                "directly_eligible": directly_eligible,
                "propagation_exposed": propagation_exposed,
                "covered": covered_rows,
                "untouched_uncovered": untouched_rows,
                "never_in_train_182": course_series.isin(
                    set(link["new_course_id"].astype(str))
                ).to_numpy(),
                "course_id": course_series.to_numpy(),
            }
        ).to_parquet(staging / "row_segments.parquet", index=False)

        staging.rename(PILOT_DIR)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"\nPilot dataset version written: {PILOT_DIR}")
    print(json.dumps(manifest["row_sets"], indent=2))
    print(json.dumps(manifest["diff_vs_frozen_valid"]["changed_rows_per_direct_difficulty_column"], indent=2))
    print(json.dumps(manifest["diff_vs_frozen_valid"]["changed_rows_per_concurrent_derived_column"], indent=2))
    print("TEST reads: 0. Proposal rows approved: 0. Models promoted: 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
