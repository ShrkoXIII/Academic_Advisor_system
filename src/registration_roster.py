"""Build the registration-time course roster used by concurrent features.

The model target contains only completed, post-filter course occurrences.  It
therefore cannot define the membership of a registration-time peer set: a
withdrawn (or otherwise unfinished) registration is absent from the target but
was still part of the student's registered semester load.

This module reconstructs that membership from the raw CRG student-course table.
Only the two registration-time eligibility rules are applied:

* ``active == "A"``
* ``register_status in {"R", "E"}``

``finish_status`` is deliberately *not* a filter.  It is retained only as the
audit column ``finish_status_audit``; marks, grades, points, and other outcome
columns are never copied into the roster.

Every occurrence is identified by its normalized ``student_course_id`` and is
matched to targets using that ID plus the exact semester/course keys.

Peer membership is course-grained: the emitted roster is unique on
``SEMESTER_KEY + course_id``.  Repeated registrations of one course inside one
semester are collapsed to a single row, so a course contributes one peer no
matter how many occurrences the source carries.  ``student_course_id`` survives
only as the representative occurrence identifier used for exact target
matching.  Collapsing is refused when the duplicate occurrences disagree on the
registration metadata that downstream features read.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterator

import numpy as np
import pandas as pd

from src.cleaning_utils import normalize_id_series
from src.feature_engineering import SEMESTER_KEY
from src.validation import find_missing_columns


VALID_ACTIVE_STATUS = "A"
VALID_REGISTER_STATUSES = frozenset({"R", "E"})

ENTITY_ID_COLUMNS = (
    "student_course_id",
    "student_id",
    "course_id",
    "degree_id",
    "faculty_id",
    "degree_course_id",
)

# A target occurrence is allowed to replace raw/derived metadata only for these
# trusted fields.  The first four are the binding override contract; the final
# one is the target's already-clean requirement-size metadata when available.
TARGET_METADATA_OVERRIDE_COLUMNS = (
    "faculty_id",
    "course_credits",
    "requirement_type_id",
    "degree_course_key",
    "degree_requirement_credits_count",
)

OCCURRENCE_MATCH_KEY = SEMESTER_KEY + ["student_course_id", "course_id"]

# The grain of peer membership.  The emitted roster is unique on these columns.
PEER_MEMBERSHIP_KEY = SEMESTER_KEY + ["course_id"]

# Duplicate occurrences of one course may only be collapsed when they agree on
# every one of these.  A disagreement makes the survivor an arbitrary pick.
COLLAPSE_CONSISTENCY_COLUMNS = (
    "faculty_id",
    "course_credits",
    "requirement_type_id",
    "degree_requirement_credits_count",
    "degree_course_key",
)

# Row-level diagnostics that must NOT reach the model-facing roster artifact.
# finish_status_audit is a current-semester outcome; is_target_occurrence and
# target_row_position encode "this row survived the completed-attempt filter",
# which is an outcome proxy.  They stay available in memory for the builder and
# are published only in the separate row-audit artifact.
ROW_AUDIT_ONLY_COLUMNS = (
    "finish_status_audit",
    "is_target_occurrence",
    "target_row_position",
)

# Columns the row-audit artifact carries, in order.
REGISTRATION_ROSTER_ROW_AUDIT_ARTIFACT_COLUMNS = [
    *PEER_MEMBERSHIP_KEY,
    "student_course_id",
    *ROW_AUDIT_ONLY_COLUMNS,
]

# Any model-facing roster column whose name matches this is rejected unless it
# is explicitly allow-listed below.
MODEL_FACING_FORBIDDEN_PATTERN = re.compile(
    r"finish|mark|grade|points|gpa|outcome|pass|fail|withdraw|is_target|target_row",
    re.IGNORECASE,
)

# Legitimate prior-semester difficulty statistics.  These are historical by
# construction and are exact-name allow-listed, never pattern-matched.
MODEL_FACING_NAME_ALLOWLIST = frozenset(
    {
        "course_pass_rate_historical",
        "course_avg_mark_historical",
        "course_retake_rate_historical",
    }
)

# Downstream feature jobs should select from this allow-list.  In particular,
# finish_status_audit is intentionally absent.
REGISTRATION_ROSTER_FEATURE_INPUT_COLUMNS = [
    *SEMESTER_KEY,
    "student_course_id",
    "course_id",
    "faculty_id",
    "course_credits",
    "requirement_type_id",
    "degree_requirement_credits_count",
    "degree_course_key",
]

REGISTRATION_ROSTER_AUDIT_COLUMNS = [
    "active_audit",
    "register_status_audit",
    "finish_status_audit",
    "is_target_occurrence",
    "target_row_position",
    "faculty_id_source",
    "course_credits_source",
    "requirement_metadata_source",
    "degree_course_key_source",
    "acd_resolved_degree_id",
    "registration_roster_occurrence_count",
    "registration_roster_unique_course_count",
    "semester_reg_courses",
]

# Raw CRG columns in this set are intentionally never propagated.  The roster
# is assembled from an explicit allow-list rather than by dropping these names,
# so newly introduced source outcomes also cannot slip through accidentally.
KNOWN_OUTCOME_COLUMNS = frozenset(
    {
        "final_mark",
        "points",
        "grade_id",
        "course_outcome_status",
        "gpa_points",
        "semester_pass_credits",
    }
)


@dataclass
class RegistrationRosterResult:
    """Registration roster plus compact summaries and diagnostic tables.

    ``diagnostics`` contains JSON-friendly scalar/dict summaries.  Potentially
    larger tables live under ``audits`` so build reports can persist them
    separately without embedding them in JSON.

    The object supports ``roster, diagnostics = result`` for simple callers;
    richer callers can use ``result.audits``.
    """

    roster: pd.DataFrame
    diagnostics: dict[str, Any]
    audits: dict[str, pd.DataFrame]

    def __iter__(self) -> Iterator[Any]:
        yield self.roster
        yield self.diagnostics

    @property
    def semester_count_comparison(self) -> pd.DataFrame:
        return self.audits["semester_count_comparison"]

    @property
    def mismatch_examples(self) -> pd.DataFrame:
        return self.audits["semester_count_mismatch_examples"]

    @property
    def excluded_registration_status_summary(self) -> pd.DataFrame:
        return self.audits["excluded_register_status_by_group"]


def _normalize_column_names(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    # Every normalized column is replaced below, so a shallow structural copy
    # protects the caller without duplicating the large raw roster blocks.
    result = frame.copy(deep=False)
    normalized = (
        pd.Index(result.columns)
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"[^0-9a-z]+", "_", regex=True)
        .str.strip("_")
    )
    duplicated = normalized[normalized.duplicated()].tolist()
    if duplicated:
        raise ValueError(
            f"{source_name}: column-name normalization created duplicates: {duplicated}"
        )
    result.columns = normalized
    return result


def _require_columns(
    frame: pd.DataFrame, columns: list[str] | tuple[str, ...], source_name: str
) -> None:
    missing = find_missing_columns(frame, columns)
    if missing:
        raise KeyError(f"{source_name} is missing required columns: {missing}")


def _normalize_code_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.upper()
    null_like = normalized.fillna("").str.lower().isin(
        {"", "nan", "none", "null", "<na>"}
    )
    return normalized.mask(null_like, pd.NA)


def _normalize_part_id(series: pd.Series, source_name: str) -> pd.Series:
    normalized = normalize_id_series(series)
    numeric = pd.to_numeric(normalized, errors="coerce")
    source_has_value = normalized.notna()
    invalid = source_has_value & numeric.isna()
    fractional = numeric.notna() & ~np.isclose(numeric % 1, 0, atol=1e-9)
    if invalid.any() or fractional.any():
        bad = normalized.loc[invalid | fractional].drop_duplicates().head(10).tolist()
        raise ValueError(
            f"{source_name}.part_id contains invalid semester identifiers: {bad}"
        )
    return numeric.round().astype("Int64")


def _normalize_entity_ids(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    result = frame.copy(deep=False)
    for column in ENTITY_ID_COLUMNS:
        if column in result.columns:
            result[column] = normalize_id_series(result[column])
    if "part_id" in result.columns:
        result["part_id"] = _normalize_part_id(result["part_id"], source_name)
    return result


def _ensure_university_id(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Derive university_id from dotted IDs and reject conflicting suffixes."""

    result = frame.copy(deep=False)
    derived = pd.Series(pd.NA, index=result.index, dtype="string")
    for column in ENTITY_ID_COLUMNS:
        if column not in result.columns:
            continue
        # Entity IDs were normalized by _normalize_entity_ids immediately
        # before this helper. Avoid normalizing and materializing them again.
        suffix = result[column].astype("string").str.extract(
            r"\.([^.]+)$", expand=False
        )
        conflict = derived.notna() & suffix.notna() & derived.ne(suffix)
        if conflict.any():
            examples = (
                result.loc[
                    conflict,
                    [c for c in ENTITY_ID_COLUMNS if c in result.columns],
                ]
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(
                f"{source_name}: ambiguous university suffixes across dotted IDs; "
                f"{int(conflict.sum())} conflicting rows. Examples: {examples}"
            )
        derived = derived.fillna(suffix)

    if "university_id" in result.columns:
        explicit = normalize_id_series(result["university_id"])
        explicit_conflict = explicit.notna() & derived.notna() & explicit.ne(derived)
        if explicit_conflict.any():
            examples = (
                pd.DataFrame(
                    {
                        "university_id": explicit,
                        "derived_university_id": derived,
                    }
                )
                .loc[explicit_conflict]
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(
                f"{source_name}: explicit university_id conflicts with dotted ID "
                f"suffixes in {int(explicit_conflict.sum())} rows. Examples: {examples}"
            )
        result["university_id"] = explicit.fillna(derived).astype("string")
    else:
        result["university_id"] = derived.astype("string")

    return result


def _normalize_source_frame(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    result = _normalize_column_names(frame, source_name)
    result = _normalize_entity_ids(result, source_name)
    result = _ensure_university_id(result, source_name)
    return result


def _to_nullable_integer(series: pd.Series, column: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    source_text = series.astype("string").str.strip()
    source_has_value = series.notna() & source_text.ne("")
    invalid = source_has_value & numeric.isna()
    fractional = numeric.notna() & ~np.isclose(numeric % 1, 0, atol=1e-9)
    if invalid.any() or fractional.any():
        examples = series.loc[invalid | fractional].drop_duplicates().head(10).tolist()
        raise ValueError(f"{column} contains invalid integer values: {examples}")
    return numeric.round().astype("Int64")


def _to_nullable_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Float64")


def _value_counts_dict(series: pd.Series) -> dict[str, int]:
    values = series.astype("string").fillna("<MISSING>")
    counts = values.value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def _prepare_target(target_frame: pd.DataFrame) -> pd.DataFrame:
    target = _normalize_source_frame(target_frame, "target_frame")
    _require_columns(
        target,
        [*SEMESTER_KEY, "student_course_id", "course_id"],
        "target_frame",
    )

    null_identity = target[OCCURRENCE_MATCH_KEY].isna().any(axis=1)
    if null_identity.any():
        examples = target.loc[null_identity, OCCURRENCE_MATCH_KEY].head(5)
        raise ValueError(
            "target_frame contains null occurrence/semester identifiers. "
            f"Examples: {examples.to_dict(orient='records')}"
        )

    duplicated_id = target["student_course_id"].duplicated(keep=False)
    if duplicated_id.any():
        examples = (
            target.loc[duplicated_id, OCCURRENCE_MATCH_KEY]
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(
            "target_frame student_course_id must identify exactly one occurrence; "
            f"duplicates found. Examples: {examples}"
        )

    target["_target_row_position"] = pd.Series(
        np.arange(len(target)), index=target.index, dtype="Int64"
    )

    if "faculty_id" in target.columns:
        target["faculty_id"] = normalize_id_series(target["faculty_id"])
    if "course_credits" in target.columns:
        target["course_credits"] = _to_nullable_float(target["course_credits"])
    if "requirement_type_id" in target.columns:
        target["requirement_type_id"] = _to_nullable_integer(
            target["requirement_type_id"], "target_frame.requirement_type_id"
        )
    if "degree_requirement_credits_count" in target.columns:
        target["degree_requirement_credits_count"] = _to_nullable_float(
            target["degree_requirement_credits_count"]
        )
    if "degree_course_key" in target.columns:
        target["degree_course_key"] = (
            target["degree_course_key"].astype("string").str.strip()
        )

    return target


def _prepare_acd(clean_acd: pd.DataFrame) -> pd.DataFrame:
    acd = _normalize_source_frame(clean_acd, "clean_acd")
    _require_columns(
        acd,
        ["degree_id", "course_id", "requirement_type_id"],
        "clean_acd",
    )

    null_keys = acd[["degree_id", "course_id"]].isna().any(axis=1)
    if null_keys.any():
        examples = (
            acd.loc[null_keys, ["degree_id", "course_id"]]
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(f"clean_acd contains null degree-course keys: {examples}")

    duplicate_key = acd.duplicated(["degree_id", "course_id"], keep=False)
    if duplicate_key.any():
        examples = (
            acd.loc[duplicate_key, ["degree_id", "course_id"]]
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(
            "clean_acd must be unique on degree_id + course_id, matching the "
            f"current ACD merge contract. Examples: {examples}"
        )

    acd["requirement_type_id"] = _to_nullable_integer(
        acd["requirement_type_id"], "clean_acd.requirement_type_id"
    )
    if "course_credits" in acd.columns:
        acd["course_credits"] = _to_nullable_float(acd["course_credits"])
    if "credits_count" in acd.columns:
        acd["credits_count"] = _to_nullable_float(acd["credits_count"])
    return acd


def _acd_metadata_columns(acd: pd.DataFrame) -> list[str]:
    candidates = [
        "degree_course_id",
        "requirement_type_id",
        "requirement_type_sl",
        "course_credits",
        "credits_count",
        "course_name_sl",
        "degree_name_sl",
    ]
    return [column for column in candidates if column in acd.columns]


def _enrich_roster_only_from_acd(
    roster: pd.DataFrame, clean_acd: pd.DataFrame
) -> pd.DataFrame:
    """Apply the existing exact-pair/unique-course-degree ACD resolution."""

    result = roster.copy()
    acd = _prepare_acd(clean_acd)
    metadata_columns = _acd_metadata_columns(acd)

    exact_lookup = acd[
        ["degree_id", "course_id", *metadata_columns]
    ].rename(
        columns={
            "degree_id": "_acd_exact_degree_id",
            **{column: f"_acd_exact_{column}" for column in metadata_columns},
        }
    )
    result = result.merge(
        exact_lookup,
        left_on=["degree_id", "course_id"],
        right_on=["_acd_exact_degree_id", "course_id"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    exact_match = result["_acd_exact_degree_id"].notna()

    course_degree_counts = (
        acd.groupby("course_id", dropna=False, sort=False)["degree_id"]
        .nunique(dropna=True)
        .rename("_acd_course_degree_count")
    )
    result = result.merge(
        course_degree_counts,
        left_on="course_id",
        right_index=True,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    result["_acd_course_degree_count"] = (
        result["_acd_course_degree_count"].fillna(0).astype("Int64")
    )

    unique_courses = (
        acd.loc[
            acd["course_id"].isin(
                course_degree_counts[course_degree_counts.eq(1)].index
            ),
            ["degree_id", "course_id", *metadata_columns],
        ]
        .rename(
            columns={
                "degree_id": "_acd_fallback_degree_id",
                **{
                    column: f"_acd_fallback_{column}"
                    for column in metadata_columns
                },
            }
        )
        .copy()
    )
    result = result.merge(
        unique_courses,
        on="course_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    roster_only = ~result["is_target_occurrence"]
    fallback_match = (
        roster_only
        & ~exact_match
        & result["_acd_course_degree_count"].eq(1)
        & result["_acd_fallback_degree_id"].notna()
    )
    exact_roster_only = roster_only & exact_match
    ambiguous = roster_only & ~exact_match & result["_acd_course_degree_count"].gt(1)
    absent = roster_only & result["_acd_course_degree_count"].eq(0)

    result["requirement_metadata_source"] = pd.Series(
        "target_occurrence", index=result.index, dtype="string"
    )
    result.loc[
        exact_roster_only, "requirement_metadata_source"
    ] = "acd_exact_degree_course_match"
    result.loc[
        fallback_match, "requirement_metadata_source"
    ] = "acd_unique_course_degree_fallback"
    result.loc[
        ambiguous, "requirement_metadata_source"
    ] = "acd_ambiguous_multiple_degrees"
    result.loc[absent, "requirement_metadata_source"] = "acd_course_absent"

    result["acd_resolved_degree_id"] = pd.Series(
        pd.NA, index=result.index, dtype="string"
    )
    result.loc[exact_roster_only, "acd_resolved_degree_id"] = result.loc[
        exact_roster_only, "_acd_exact_degree_id"
    ]
    result.loc[fallback_match, "acd_resolved_degree_id"] = result.loc[
        fallback_match, "_acd_fallback_degree_id"
    ]

    # Build roster-only metadata from the exact row first, then the proven-safe
    # unique-course fallback.  Ambiguous/absent rows intentionally stay missing.
    for column in metadata_columns:
        exact_column = f"_acd_exact_{column}"
        fallback_column = f"_acd_fallback_{column}"
        output_column = {
            "credits_count": "degree_requirement_credits_count",
            "course_credits": "acd_course_credits",
            "course_name_sl": "acd_course_name_sl",
            "degree_name_sl": "acd_degree_name_sl",
        }.get(column, column)
        if output_column not in result.columns:
            result[output_column] = pd.NA
        result.loc[exact_roster_only, output_column] = result.loc[
            exact_roster_only, exact_column
        ].to_numpy()
        result.loc[fallback_match, output_column] = result.loc[
            fallback_match, fallback_column
        ].to_numpy()

    helper_columns = [
        column
        for column in result.columns
        if column.startswith("_acd_exact_")
        or column.startswith("_acd_fallback_")
        or column == "_acd_course_degree_count"
    ]
    result = result.drop(columns=helper_columns)
    return result


def _collapse_to_peer_membership(
    roster: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reduce occurrence rows to one row per ``SEMESTER_KEY + course_id``.

    The target occurrence is always kept as its course's representative, so
    exact target matching by ``student_course_id`` keeps working and the
    target's authoritative metadata survives.  Original row order is preserved.
    """

    n = len(roster)
    if n == 0:
        return roster, {
            "occurrence_rows": 0,
            "peer_rows": 0,
            "collapsed_duplicate_occurrence_rows": 0,
            "courses_with_multiple_occurrences": 0,
        }

    group_id = (
        roster.groupby(PEER_MEMBERSHIP_KEY, dropna=False, sort=False)
        .ngroup()
        .to_numpy(dtype="int64")
    )
    duplicated_row = pd.Index(group_id).duplicated(keep=False)
    duplicate_rows = int(duplicated_row.sum())
    is_target = roster["is_target_occurrence"].to_numpy(dtype=bool)
    group_has_target = (
        pd.Series(is_target).groupby(group_id, sort=False).transform("any")
    ).to_numpy(dtype=bool)

    if duplicate_rows:
        # A course-grained roster cannot represent two completed target
        # attempts at one course in one semester: they would collapse into a
        # single peer entry and the target->peer mapping would stop being 1:1.
        targets_per_group = (
            pd.Series(is_target).groupby(group_id, sort=False).sum()
        )
        ambiguous = targets_per_group.index[targets_per_group.to_numpy() > 1]
        if len(ambiguous):
            examples = (
                roster.loc[np.isin(group_id, ambiguous[:2]), OCCURRENCE_MATCH_KEY]
                .head(6)
                .to_dict(orient="records")
            )
            raise ValueError(
                "Two target occurrences share one semester course; peer "
                f"membership is unique on {PEER_MEMBERSHIP_KEY} and cannot "
                f"represent both. Examples: {examples}"
            )

        # Precedence is already defined when the course has a target
        # occurrence: the target's metadata is authoritative and the target row
        # is always the survivor, so a difference against a roster-only
        # occurrence is a resolved override, not an arbitrary pick. Only
        # all-roster-only duplicates have genuinely equal-authority candidates.
        undetermined = duplicated_row & ~group_has_target

        checked = [c for c in COLLAPSE_CONSISTENCY_COLUMNS if c in roster.columns]
        conflict_frame = roster.loc[undetermined, checked]
        conflict_groups = group_id[undetermined]
        for column in checked if undetermined.any() else []:
            # nunique(dropna=False) treats missing as its own value, so this
            # also catches a value-versus-missing disagreement.
            distinct = conflict_frame.groupby(conflict_groups, sort=False)[
                column
            ].nunique(dropna=False)
            bad_groups = distinct.index[distinct.to_numpy() > 1]
            if len(bad_groups):
                examples = (
                    roster.loc[
                        np.isin(group_id, bad_groups[:2]),
                        OCCURRENCE_MATCH_KEY + [column],
                    ]
                    .head(6)
                    .to_dict(orient="records")
                )
                raise ValueError(
                    "Duplicate occurrences of one semester course disagree on "
                    f"{column!r}; refusing to pick a survivor arbitrarily. "
                    f"Conflicting courses: {int(len(bad_groups))}. "
                    f"Examples: {examples}"
                )

    # Target rows win the representative slot; ties fall back to source order.
    order = np.lexsort((np.arange(n, dtype="int64"), np.where(is_target, 0, 1)))
    first_of_group = ~pd.Index(group_id[order]).duplicated(keep="first")
    keep_positions = np.sort(order[first_of_group])

    collapsed = roster.iloc[keep_positions]
    diagnostics = {
        "occurrence_rows": n,
        "peer_rows": int(len(collapsed)),
        "collapsed_duplicate_occurrence_rows": int(n - len(collapsed)),
        "courses_with_multiple_occurrences": (
            int(pd.Index(group_id[duplicated_row]).nunique())
            if duplicate_rows
            else 0
        ),
        "courses_collapsed_by_target_precedence": (
            int(pd.Index(group_id[duplicated_row & group_has_target]).nunique())
            if duplicate_rows
            else 0
        ),
    }
    return collapsed, diagnostics


def model_facing_roster(roster: pd.DataFrame) -> pd.DataFrame:
    """Drop row-level audit columns and assert no outcome column survives."""

    result = roster.drop(
        columns=[c for c in ROW_AUDIT_ONLY_COLUMNS if c in roster.columns]
    )
    assert_model_facing_roster_columns(result)
    return result


def assert_model_facing_roster_columns(roster: pd.DataFrame) -> list[str]:
    """Raise when a model-facing roster carries an outcome/target-proxy column."""

    offenders = [
        column
        for column in roster.columns
        if column not in MODEL_FACING_NAME_ALLOWLIST
        and MODEL_FACING_FORBIDDEN_PATTERN.search(str(column))
    ]
    if offenders:
        raise AssertionError(
            "Model-facing registration roster carries outcome or target-proxy "
            f"columns: {offenders}"
        )
    return offenders


def assert_unique_peer_membership(roster: pd.DataFrame) -> None:
    """Raise when the persisted roster is not unique on the membership key."""

    duplicated = roster.duplicated(PEER_MEMBERSHIP_KEY, keep=False)
    count = int(duplicated.sum())
    if count:
        examples = (
            roster.loc[duplicated, PEER_MEMBERSHIP_KEY]
            .head(10)
            .to_dict(orient="records")
        )
        raise AssertionError(
            f"Registration roster is not unique on {PEER_MEMBERSHIP_KEY}: "
            f"{count} duplicated rows. Examples: {examples}"
        )


def roster_row_audit(roster: pd.DataFrame) -> pd.DataFrame:
    """Return the separate row-level audit table (never read by features)."""

    columns = [
        column
        for column in REGISTRATION_ROSTER_ROW_AUDIT_ARTIFACT_COLUMNS
        if column in roster.columns
    ]
    return roster.loc[:, columns].reset_index(drop=True)


def _target_semester_counts(target: pd.DataFrame) -> pd.DataFrame:
    target_counts = (
        target.groupby(SEMESTER_KEY, dropna=False, sort=False)
        .size()
        .rename("target_occurrence_count")
        .reset_index()
    )
    if "semester_reg_courses" not in target.columns:
        target_counts["semester_reg_courses"] = pd.Series(
            pd.NA, index=target_counts.index, dtype="Float64"
        )
        target_counts["semester_reg_courses_value_count"] = 0
        return target_counts

    work = target[SEMESTER_KEY].copy()
    work["_semester_reg_courses"] = pd.to_numeric(
        target["semester_reg_courses"], errors="coerce"
    )
    expected = (
        work.groupby(SEMESTER_KEY, dropna=False, sort=False)[
            "_semester_reg_courses"
        ]
        .agg(
            semester_reg_courses=lambda values: (
                values.dropna().iloc[0] if not values.dropna().empty else np.nan
            ),
            semester_reg_courses_value_count=lambda values: int(
                values.dropna().nunique()
            ),
        )
        .reset_index()
    )
    expected["semester_reg_courses"] = expected["semester_reg_courses"].astype(
        "Float64"
    )
    expected["semester_reg_courses_value_count"] = expected[
        "semester_reg_courses_value_count"
    ].astype("Int64")
    return target_counts.merge(
        expected, on=SEMESTER_KEY, how="left", validate="one_to_one"
    )


def _group_status_summary(
    frame: pd.DataFrame, status_column: str, count_column: str
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[*SEMESTER_KEY, status_column, count_column]
        )
    work = frame[SEMESTER_KEY + [status_column]].copy()
    work[status_column] = work[status_column].astype("string").fillna("<MISSING>")
    return (
        work.groupby(
            SEMESTER_KEY + [status_column],
            dropna=False,
            sort=False,
        )
        .size()
        .rename(count_column)
        .reset_index()
    )


def _collapse_status_breakdown(
    grouped_status: pd.DataFrame, status_column: str, count_column: str
) -> pd.DataFrame:
    output_column = f"{status_column}_breakdown"
    if grouped_status.empty:
        return pd.DataFrame(columns=[*SEMESTER_KEY, output_column])

    work = grouped_status.copy()
    work["_label"] = (
        work[status_column].astype("string")
        + ":"
        + work[count_column].astype("Int64").astype("string")
    )
    return (
        work.groupby(SEMESTER_KEY, dropna=False, sort=False)["_label"]
        .agg(lambda values: "; ".join(values.astype(str)))
        .rename(output_column)
        .reset_index()
    )


def _build_count_comparison(
    target: pd.DataFrame,
    roster: pd.DataFrame,
    excluded_inactive: pd.DataFrame,
    excluded_register: pd.DataFrame,
    excluded_register_by_group: pd.DataFrame,
) -> pd.DataFrame:
    comparison = _target_semester_counts(target)
    roster_counts = (
        roster.groupby(SEMESTER_KEY, dropna=False, sort=False)
        .agg(
            registration_roster_occurrence_count=("student_course_id", "size"),
            registration_roster_unique_course_count=("course_id", "nunique"),
        )
        .reset_index()
    )
    comparison = comparison.merge(
        roster_counts, on=SEMESTER_KEY, how="left", validate="one_to_one"
    )

    for source, column in (
        (excluded_inactive, "excluded_inactive_count"),
        (excluded_register, "excluded_register_status_count"),
    ):
        counts = (
            source.groupby(SEMESTER_KEY, dropna=False, sort=False)
            .size()
            .rename(column)
            .reset_index()
            if not source.empty
            else pd.DataFrame(columns=[*SEMESTER_KEY, column])
        )
        comparison = comparison.merge(
            counts, on=SEMESTER_KEY, how="left", validate="one_to_one"
        )
        comparison[column] = comparison[column].fillna(0).astype("Int64")

    breakdown = _collapse_status_breakdown(
        excluded_register_by_group,
        "register_status",
        "excluded_register_status_count",
    )
    comparison = comparison.merge(
        breakdown, on=SEMESTER_KEY, how="left", validate="one_to_one"
    )
    comparison["register_status_breakdown"] = comparison[
        "register_status_breakdown"
    ].astype("string")

    for column in (
        "target_occurrence_count",
        "registration_roster_occurrence_count",
        "registration_roster_unique_course_count",
    ):
        comparison[column] = comparison[column].fillna(0).astype("Int64")

    expected = comparison["semester_reg_courses"]
    actual = comparison["registration_roster_occurrence_count"].astype("Float64")
    conflict = comparison["semester_reg_courses_value_count"].gt(1)
    comparable = expected.notna() & ~conflict
    comparison["registration_count_delta"] = (expected - actual).astype("Float64")
    comparison["registration_count_matches"] = pd.Series(
        pd.NA, index=comparison.index, dtype="boolean"
    )
    comparison.loc[comparable, "registration_count_matches"] = expected.loc[
        comparable
    ].eq(actual.loc[comparable]).to_numpy()

    reason = pd.Series("match", index=comparison.index, dtype="string")
    reason.loc[expected.isna()] = "semester_reg_courses_missing"
    reason.loc[conflict] = "target_semester_reg_courses_conflict"
    shortfall = comparable & expected.gt(actual)
    excess = comparable & expected.lt(actual)
    has_inactive = comparison["excluded_inactive_count"].gt(0)
    has_bad_register = comparison["excluded_register_status_count"].gt(0)
    reason.loc[shortfall & has_bad_register & ~has_inactive] = (
        "roster_shortfall_with_excluded_register_status_rows"
    )
    reason.loc[shortfall & has_inactive & ~has_bad_register] = (
        "roster_shortfall_with_excluded_inactive_rows"
    )
    reason.loc[shortfall & has_inactive & has_bad_register] = (
        "roster_shortfall_with_excluded_inactive_and_register_status_rows"
    )
    reason.loc[shortfall & ~has_inactive & ~has_bad_register] = (
        "unexplained_roster_shortfall"
    )
    reason.loc[excess] = "roster_exceeds_recorded_semester_reg_courses"
    comparison["registration_count_mismatch_reason"] = reason
    return comparison


def build_registration_roster(
    raw_crg: pd.DataFrame,
    target_frame: pd.DataFrame,
    clean_acd: pd.DataFrame,
) -> RegistrationRosterResult:
    """Return the complete active, registered roster for target semesters.

    Parameters
    ----------
    raw_crg:
        Raw ``V_CRG_STUDENT_COURSE`` rows.  Column names may use source casing;
        they are normalized on a copy.
    target_frame:
        Completed target occurrences.  Every row must match exactly one
        eligible raw occurrence on ``OCCURRENCE_MATCH_KEY``.
    clean_acd:
        Clean ``V_ACD_DEGREE_COURSE`` metadata used only for roster-only
        occurrences.  Resolution reproduces the current exact-pair followed by
        unique-course-degree fallback.
    """

    target = _prepare_target(target_frame)
    raw = _normalize_source_frame(raw_crg, "raw_crg")
    _require_columns(
        raw,
        [
            "student_course_id",
            "student_id",
            "course_id",
            "part_id",
            "degree_id",
            "faculty_id",
            "course_credits",
            "active",
            "register_status",
        ],
        "raw_crg",
    )
    raw["active"] = _normalize_code_series(raw["active"])
    raw["register_status"] = _normalize_code_series(raw["register_status"])
    if "finish_status" in raw.columns:
        raw["finish_status"] = _normalize_code_series(raw["finish_status"])
    else:
        raw["finish_status"] = pd.Series(pd.NA, index=raw.index, dtype="string")
    raw["course_credits"] = _to_nullable_float(raw["course_credits"])
    raw["_raw_row_position"] = np.arange(len(raw), dtype="int64")

    target_groups = target[SEMESTER_KEY].drop_duplicates()
    in_scope = raw.merge(
        target_groups,
        on=SEMESTER_KEY,
        how="inner",
        validate="many_to_one",
        sort=False,
    )

    active_mask = in_scope["active"].eq(VALID_ACTIVE_STATUS).fillna(False)
    valid_register_mask = (
        in_scope["register_status"].isin(VALID_REGISTER_STATUSES).fillna(False)
    )
    excluded_inactive = in_scope.loc[~active_mask].copy()
    excluded_register = in_scope.loc[active_mask & ~valid_register_mask].copy()
    eligible = in_scope.loc[active_mask & valid_register_mask].copy()

    null_occurrence = eligible[
        ["student_course_id", "student_id", "degree_id", "part_id", "course_id"]
    ].isna().any(axis=1)
    if null_occurrence.any():
        examples = (
            eligible.loc[
                null_occurrence,
                ["student_course_id", *SEMESTER_KEY, "course_id"],
            ]
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Eligible raw roster rows require stable occurrence and semester "
            f"identifiers. Examples: {examples}"
        )

    duplicate_occurrence_id = eligible["student_course_id"].duplicated(keep=False)
    if duplicate_occurrence_id.any():
        examples = (
            eligible.loc[duplicate_occurrence_id, OCCURRENCE_MATCH_KEY]
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(
            "raw_crg student_course_id is ambiguous within target semesters; "
            f"each ID must identify one occurrence. Examples: {examples}"
        )

    coverage = target[OCCURRENCE_MATCH_KEY + ["_target_row_position"]].merge(
        eligible[OCCURRENCE_MATCH_KEY],
        on=OCCURRENCE_MATCH_KEY,
        how="left",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    unmatched = coverage["_merge"].ne("both")
    if unmatched.any():
        missing_rows = coverage.loc[
            unmatched, OCCURRENCE_MATCH_KEY + ["_target_row_position"]
        ]
        missing_ids = set(missing_rows["student_course_id"].dropna().astype(str))
        alternate_raw = raw.loc[
            raw["student_course_id"].astype("string").isin(missing_ids),
            OCCURRENCE_MATCH_KEY + ["active", "register_status"],
        ].head(10)
        raise ValueError(
            "Every target must match exactly one active R/E raw occurrence with "
            "exact semester keys and course_id. "
            f"Unmatched targets: {missing_rows.head(10).to_dict(orient='records')}; "
            f"raw rows sharing those student_course_id values: "
            f"{alternate_raw.to_dict(orient='records')}"
        )

    raw_roster_columns = [
        *SEMESTER_KEY,
        "student_course_id",
        "course_id",
        "faculty_id",
        "course_credits",
        "active",
        "register_status",
        "finish_status",
        "_raw_row_position",
    ]
    roster = eligible[raw_roster_columns].copy()

    target_merge_columns = OCCURRENCE_MATCH_KEY + ["_target_row_position"]
    for column in TARGET_METADATA_OVERRIDE_COLUMNS:
        if column in target.columns and column not in target_merge_columns:
            target_merge_columns.append(column)
    if "semester_reg_courses" in target.columns:
        target_merge_columns.append("semester_reg_courses")
    target_occurrences = target[target_merge_columns].rename(
        columns={
            column: f"_target_{column}"
            for column in target_merge_columns
            if column not in OCCURRENCE_MATCH_KEY
        }
    )
    roster = roster.merge(
        target_occurrences,
        on=OCCURRENCE_MATCH_KEY,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    roster["is_target_occurrence"] = roster["_target__target_row_position"].notna()
    roster["target_row_position"] = roster[
        "_target__target_row_position"
    ].astype("Int64")

    # Apply the binding target-over-raw metadata contract.
    roster["faculty_id_source"] = pd.Series(
        np.where(roster["faculty_id"].notna(), "raw_crg", "missing"),
        index=roster.index,
        dtype="string",
    )
    roster["course_credits_source"] = pd.Series(
        np.where(roster["course_credits"].notna(), "raw_crg", "missing"),
        index=roster.index,
        dtype="string",
    )
    for column in ("faculty_id", "course_credits"):
        target_column = f"_target_{column}"
        if target_column in roster.columns:
            target_rows = roster["is_target_occurrence"]
            roster.loc[target_rows, column] = roster.loc[
                target_rows, target_column
            ].to_numpy()
            roster.loc[target_rows, f"{column}_source"] = "target_occurrence"

    roster = _enrich_roster_only_from_acd(roster, clean_acd)

    # Target requirement metadata is authoritative.  When a target field is
    # absent altogether, the ACD result remains available as a compatibility
    # fallback; when present, even an explicit missing value overrides it.
    for column in ("requirement_type_id", "degree_requirement_credits_count"):
        target_column = f"_target_{column}"
        if target_column in roster.columns:
            target_rows = roster["is_target_occurrence"]
            if column not in roster.columns:
                roster[column] = pd.NA
            roster.loc[target_rows, column] = roster.loc[
                target_rows, target_column
            ].to_numpy()
            roster.loc[
                target_rows, "requirement_metadata_source"
            ] = "target_occurrence"

    if "degree_requirement_credits_count" not in roster.columns:
        roster["degree_requirement_credits_count"] = pd.NA

    derived_key = (
        roster["degree_id"].astype("string")
        + "__"
        + roster["course_id"].astype("string")
    )
    roster["degree_course_key"] = derived_key
    roster["degree_course_key_source"] = "derived_from_normalized_ids"
    if "_target_degree_course_key" in roster.columns:
        target_rows = roster["is_target_occurrence"]
        roster.loc[target_rows, "degree_course_key"] = roster.loc[
            target_rows, "_target_degree_course_key"
        ].to_numpy()
        roster.loc[
            target_rows, "degree_course_key_source"
        ] = "target_occurrence"

    # The target's semester_reg_courses is a group-level comparison input, not a
    # source membership filter.
    target_semester = _target_semester_counts(target)[
        SEMESTER_KEY
        + [
            "semester_reg_courses",
            "semester_reg_courses_value_count",
        ]
    ]
    target_semester = target_semester.loc[
        target_semester["semester_reg_courses_value_count"].le(1)
    ]
    roster = roster.drop(
        columns=[
            column
            for column in ["_target_semester_reg_courses"]
            if column in roster.columns
        ]
    ).merge(
        target_semester[SEMESTER_KEY + ["semester_reg_courses"]],
        on=SEMESTER_KEY,
        how="left",
        validate="many_to_one",
        sort=False,
    )

    # Collapse to the peer-membership grain BEFORE any group-level count, so
    # every published count describes the roster that features actually read.
    roster, collapse_diagnostics = _collapse_to_peer_membership(roster)
    roster = roster.reset_index(drop=True)

    group = roster.groupby(SEMESTER_KEY, dropna=False, sort=False)
    roster["registration_roster_occurrence_count"] = (
        group["student_course_id"].transform("size").astype("Int64")
    )
    roster["registration_roster_unique_course_count"] = (
        group["course_id"].transform("nunique").astype("Int64")
    )

    excluded_register_by_group = _group_status_summary(
        excluded_register,
        "register_status",
        "excluded_register_status_count",
    )
    excluded_inactive_by_group = _group_status_summary(
        excluded_inactive,
        "active",
        "excluded_inactive_count",
    )
    count_comparison = _build_count_comparison(
        target,
        roster,
        excluded_inactive,
        excluded_register,
        excluded_register_by_group,
    )
    mismatches = count_comparison.loc[
        count_comparison["registration_count_matches"].ne(True).fillna(True)
    ]
    mismatch_examples = (
        mismatches.groupby(
            "registration_count_mismatch_reason",
            dropna=False,
            sort=False,
            group_keys=False,
        )
        .head(10)
        .reset_index(drop=True)
    )

    # Rename registration/outcome context explicitly as audit-only and remove
    # every helper/target merge column.  This explicit final projection keeps
    # raw outcome columns out even when the source schema grows.
    roster = roster.rename(
        columns={
            "active": "active_audit",
            "register_status": "register_status_audit",
            "finish_status": "finish_status_audit",
        }
    )
    helper_columns = [
        column
        for column in roster.columns
        if column.startswith("_target_") or column == "_raw_row_position"
    ]
    roster = roster.drop(columns=helper_columns)

    ordered_columns = [
        *SEMESTER_KEY,
        "student_course_id",
        "course_id",
        "faculty_id",
        "course_credits",
        "requirement_type_id",
        "degree_requirement_credits_count",
        "degree_course_key",
        "is_target_occurrence",
        "target_row_position",
        "active_audit",
        "register_status_audit",
        "finish_status_audit",
        "faculty_id_source",
        "course_credits_source",
        "requirement_metadata_source",
        "degree_course_key_source",
        "acd_resolved_degree_id",
        "registration_roster_occurrence_count",
        "registration_roster_unique_course_count",
        "semester_reg_courses",
    ]
    optional_acd_columns = [
        "degree_course_id",
        "requirement_type_sl",
        "acd_course_credits",
        "acd_course_name_sl",
        "acd_degree_name_sl",
    ]
    ordered_columns += [
        column for column in optional_acd_columns if column in roster.columns
    ]
    remaining_safe_columns = [
        column
        for column in roster.columns
        if column not in ordered_columns and column not in KNOWN_OUTCOME_COLUMNS
    ]
    roster = roster[ordered_columns + remaining_safe_columns]

    # Re-pin canonical dtypes after merges/overrides.
    roster["requirement_type_id"] = _to_nullable_integer(
        roster["requirement_type_id"], "roster.requirement_type_id"
    )
    roster["course_credits"] = _to_nullable_float(roster["course_credits"])
    roster["degree_requirement_credits_count"] = _to_nullable_float(
        roster["degree_requirement_credits_count"]
    )
    roster["is_target_occurrence"] = roster["is_target_occurrence"].astype(bool)
    roster = roster.reset_index(drop=True)

    comparable = count_comparison["registration_count_matches"].notna()
    matched = count_comparison.loc[
        comparable, "registration_count_matches"
    ].fillna(False)
    match_rate = float(matched.mean()) if len(matched) else None

    diagnostics: dict[str, Any] = {
        "row_counts": {
            "raw_crg": int(len(raw)),
            "raw_rows_in_target_semester_groups": int(len(in_scope)),
            "excluded_inactive": int(len(excluded_inactive)),
            "excluded_register_status": int(len(excluded_register)),
            "eligible_registration_roster": int(len(roster)),
            "target": int(len(target)),
            "target_occurrences_matched": int(
                roster["is_target_occurrence"].sum()
            ),
            "roster_only": int((~roster["is_target_occurrence"]).sum()),
        },
        "target_semester_group_count": int(len(target_groups)),
        "filters": {
            "active": VALID_ACTIVE_STATUS,
            "register_status": sorted(VALID_REGISTER_STATUSES),
            "finish_status_filter_applied": False,
        },
        "excluded_register_status_counts": _value_counts_dict(
            excluded_register["register_status"]
        ),
        "excluded_inactive_status_counts": _value_counts_dict(
            excluded_inactive["active"]
        ),
        "roster_only_finish_status_counts": _value_counts_dict(
            roster.loc[~roster["is_target_occurrence"], "finish_status_audit"]
        ),
        "requirement_metadata_source_counts": _value_counts_dict(
            roster["requirement_metadata_source"]
        ),
        "faculty_id_source_counts": _value_counts_dict(
            roster["faculty_id_source"]
        ),
        "course_credits_source_counts": _value_counts_dict(
            roster["course_credits_source"]
        ),
        "semester_count_match": {
            "comparable_group_count": int(comparable.sum()),
            "matched_group_count": int(matched.sum()),
            "match_rate": match_rate,
            "mismatch_reason_counts": _value_counts_dict(
                count_comparison["registration_count_mismatch_reason"]
            ),
        },
        "peer_membership_grain": PEER_MEMBERSHIP_KEY,
        "peer_membership_collapse": collapse_diagnostics,
        "duplicate_course_occurrence_rows": int(
            roster.duplicated(PEER_MEMBERSHIP_KEY, keep=False).sum()
        ),
        "outcome_columns_present_in_roster": sorted(
            KNOWN_OUTCOME_COLUMNS.intersection(roster.columns)
        ),
    }

    audits = {
        "semester_count_comparison": count_comparison.reset_index(drop=True),
        "semester_count_mismatch_examples": mismatch_examples.reset_index(
            drop=True
        ),
        "excluded_register_status_by_group": (
            excluded_register_by_group.reset_index(drop=True)
        ),
        "excluded_inactive_by_group": excluded_inactive_by_group.reset_index(
            drop=True
        ),
    }
    return RegistrationRosterResult(
        roster=roster,
        diagnostics=diagnostics,
        audits=audits,
    )


# A short alias reads naturally in orchestration code while keeping the explicit
# public function name available to tests and notebooks.
build_roster = build_registration_roster


__all__ = [
    "COLLAPSE_CONSISTENCY_COLUMNS",
    "KNOWN_OUTCOME_COLUMNS",
    "MODEL_FACING_FORBIDDEN_PATTERN",
    "MODEL_FACING_NAME_ALLOWLIST",
    "OCCURRENCE_MATCH_KEY",
    "PEER_MEMBERSHIP_KEY",
    "REGISTRATION_ROSTER_AUDIT_COLUMNS",
    "REGISTRATION_ROSTER_FEATURE_INPUT_COLUMNS",
    "REGISTRATION_ROSTER_ROW_AUDIT_ARTIFACT_COLUMNS",
    "ROW_AUDIT_ONLY_COLUMNS",
    "RegistrationRosterResult",
    "TARGET_METADATA_OVERRIDE_COLUMNS",
    "VALID_ACTIVE_STATUS",
    "VALID_REGISTER_STATUSES",
    "assert_model_facing_roster_columns",
    "assert_unique_peer_membership",
    "build_registration_roster",
    "build_roster",
    "model_facing_roster",
    "roster_row_audit",
]
