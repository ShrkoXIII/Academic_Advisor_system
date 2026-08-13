"""Concurrent Course Group feature family (Experiment 1).

For each row (a student's target course C in a given semester), "peers" are the
other registered rows sharing the same semester group
``[university_id, student_id, degree_id, part_id]`` (identical to
``feature_engineering.SEMESTER_KEY``). This module derives how difficult a
course's same-semester peers are, using vectorized groupby leave-one-out (LOO)
aggregation — no per-row Python loops.

Training/build callers pass target rows plus the complete enriched registration
roster. One-input mode remains available for inference plans, where the
supplied targets are themselves the complete planned roster.

Peer membership grain
---------------------
Peer membership is **course-grained**: it is unique on
``SEMESTER_KEY + course_id``. A course that a student registered more than once
in the same semester contributes exactly ONE peer, not one per occurrence.
Duplicate occurrences are collapsed before any aggregation; if they disagree on
the difficulty or registration metadata that feeds this family, the collapse is
ambiguous and raises rather than silently picking a row.

``student_course_id`` is NOT the membership grain. It is retained only as the
occurrence identifier used to match a target row to its peer entry, which keeps
the "every target matches exactly one roster occurrence" guarantee intact.

Binding NaN-robust LOO semantics
--------------------------------
Let ``d = 1 - course_pass_rate_historical`` (peer difficulty). A **valid peer**
is a peer row whose ``course_pass_rate_historical`` is non-NaN. For the target
row let ``own_valid`` = (its own pass_rate is non-NaN).

* ``concurrent_peer_difficulty_mean`` = sum(d over valid peers) / count(valid
  peers):
    - numerator   = group_sum_valid - (own_d if own_valid else 0)
    - denominator = group_valid_count - 1 if own_valid else group_valid_count
    - The denominator is NEVER ``size - 1``. If the denominator is 0 -> NaN.
* ``concurrent_peer_difficulty_max`` = max(d over valid peers). Where the target
  row holds the *unique* group maximum of d, the second-highest value is
  returned; ties for the maximum keep the group maximum (another holder remains
  a peer).
* Peers exist (``size - 1 > 0``) but zero valid peer values -> mean = max = NaN
  while ``concurrent_peer_difficulty_missing`` stays 0.
* ``concurrent_peer_difficulty_missing`` = 1 iff ``(size - 1) == 0`` (empty peer
  set). Only then. When 1, mean and max are NaN.
* ``concurrent_peer_set_empty`` is the audit-only explicit name for that same
  condition. The legacy model feature equals it exactly.
* ``concurrent_peer_difficulty_values_missing`` = 1 iff peers exist but none
  has a usable difficulty value. It is 0 when the peer set is empty.
* Weak-fallback peers (``course_difficulty_missing == 1``) with a non-NaN
  pass_rate ARE valid peers and ARE included — no exclusion or down-weighting.

Audit columns (NOT model features) use OBSERVED-peer denominators (``size - 1``),
not valid-peer counts:

* ``concurrent_peer_observed_count`` = ``size - 1``.
* ``concurrent_peer_weak_ratio`` = share of peers with
  ``course_difficulty_missing == 1`` (0 when no peers).
* ``concurrent_peer_same_req_type_ratio`` = share of peers whose
  ``requirement_type_id`` equals the target's. NaN target req_type -> NaN; a peer
  with a missing req_type counts as a non-match; 0 when no peers.

Leakage
-------
Only ``course_pass_rate_historical``, ``course_difficulty_missing``,
``requirement_type_id`` (all pre-existing, train-only-derived, per-degree
columns) and the grouping keys are read. Current-semester outcomes
(``final_mark``, ``gpa_points``, ``semester_pass_credits``, …) are never touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.cleaning_utils import normalize_id_series
from src.feature_engineering import SEMESTER_KEY
<<<<<<< HEAD
from src.validation import assert_shape_preserved, require_columns
=======
from src.validation import find_missing_columns, shape_changed
>>>>>>> 770c7a147a9b3b0664121b3c8c165bf6cbfc57a9

# The three columns that enter MODEL_FEATURES.
MODEL_CONCURRENT_FEATURES = [
    "concurrent_peer_difficulty_mean",
    "concurrent_peer_difficulty_max",
    "concurrent_peer_difficulty_missing",
]

# Audit-only columns: persisted to the parquet but never added to MODEL_FEATURES.
AUDIT_CONCURRENT_FEATURES = [
    "concurrent_peer_set_empty",
    "concurrent_peer_difficulty_values_missing",
    "concurrent_peer_observed_count",
    "concurrent_peer_weak_ratio",
    "concurrent_peer_same_req_type_ratio",
]

CONCURRENT_FEATURE_COLUMNS = MODEL_CONCURRENT_FEATURES + AUDIT_CONCURRENT_FEATURES

# Columns the aggregation reads (besides the grouping keys). Kept explicit so the
# builder can gate on their presence before computing anything.
CONCURRENT_INPUT_COLUMNS = [
    "course_pass_rate_historical",
    "course_difficulty_missing",
    "requirement_type_id",
]

# Peer membership is unique on these columns: one registered course contributes
# one peer, however many occurrences of it the source carries.
PEER_MEMBERSHIP_KEY = SEMESTER_KEY + ["course_id"]

REQUIRED_INPUT_COLUMNS = PEER_MEMBERSHIP_KEY + CONCURRENT_INPUT_COLUMNS
STABLE_OCCURRENCE_KEY = "student_course_id"
TWO_INPUT_TARGET_REQUIRED_COLUMNS = [STABLE_OCCURRENCE_KEY] + PEER_MEMBERSHIP_KEY
TWO_INPUT_ROSTER_REQUIRED_COLUMNS = [STABLE_OCCURRENCE_KEY] + REQUIRED_INPUT_COLUMNS

# Columns that must agree across duplicate occurrences of one course before the
# occurrences may be collapsed into a single peer. The first three drive the
# aggregation directly; the last two are the registration metadata the roster
# resolves per occurrence, so a disagreement means the collapse would be an
# arbitrary pick rather than a deduplication.
PEER_COLLAPSE_CONSISTENCY_COLUMNS = CONCURRENT_INPUT_COLUMNS + [
    "course_credits",
    "faculty_id",
]


def _collapse_to_peer_membership(
    df: pd.DataFrame, source: str
) -> tuple[pd.DataFrame, np.ndarray]:
    """Collapse occurrences to one peer per ``SEMESTER_KEY + course_id``.

    Returns the course-grained peer frame and, for every input row, the
    positional index of the peer it belongs to. Rows are kept in first-seen
    order, so a frame that is already course-grained is returned unchanged.

    Raises when duplicate occurrences of one course disagree on any column in
    ``PEER_COLLAPSE_CONSISTENCY_COLUMNS``: collapsing those would be an
    arbitrary pick, not a deduplication.
    """
<<<<<<< HEAD
    require_columns(
        df,
        REQUIRED_INPUT_COLUMNS,
        label=f"concurrent group features ({source})",
        error=KeyError,
    )
=======
    missing_cols = find_missing_columns(df, REQUIRED_INPUT_COLUMNS)
    if missing_cols:
        raise KeyError(
            f"compute_concurrent_group_features {label} missing required "
            f"columns: {missing_cols}"
        )
>>>>>>> 770c7a147a9b3b0664121b3c8c165bf6cbfc57a9

    n = len(df)
    if n == 0:
        return df, np.empty(0, dtype="int64")

    grouped = df.groupby(PEER_MEMBERSHIP_KEY, dropna=False, sort=False)
    group_id = grouped.ngroup().to_numpy(dtype="int64")

    duplicated_row = pd.Index(group_id).duplicated(keep=False)
    if duplicated_row.any():
        checked = [
            c for c in PEER_COLLAPSE_CONSISTENCY_COLUMNS if c in df.columns
        ]
        conflict_frame = df.loc[duplicated_row, checked]
        conflict_groups = group_id[duplicated_row]
        for column in checked:
            # nunique(dropna=False) counts NaN as its own value, so this also
            # catches a value-versus-missing disagreement.
            distinct = conflict_frame.groupby(conflict_groups, sort=False)[
                column
            ].nunique(dropna=False)
            bad_groups = distinct.index[distinct.to_numpy() > 1]
            if len(bad_groups):
                example_rows = df.loc[np.isin(group_id, bad_groups[:2])]
                examples = (
                    example_rows.loc[:, PEER_MEMBERSHIP_KEY + [column]]
                    .head(6)
                    .to_dict(orient="records")
                )
                raise ValueError(
                    f"{source}: duplicate occurrences of one course disagree on "
                    f"{column!r}; peer membership is unique on "
                    f"{PEER_MEMBERSHIP_KEY} and cannot collapse them. "
                    f"Conflicting groups: {int(len(bad_groups))}. "
                    f"Examples: {examples}"
                )

    keep_positions = np.flatnonzero(~pd.Index(group_id).duplicated(keep="first"))
    if len(keep_positions) == n:
        # Already course-grained: identity mapping, no copy.
        return df, np.arange(n, dtype="int64")

    peer_index_by_group = np.empty(int(group_id.max()) + 1, dtype="int64")
    peer_index_by_group[group_id[keep_positions]] = np.arange(
        len(keep_positions), dtype="int64"
    )
    return df.iloc[keep_positions], peer_index_by_group[group_id]


def _compute_roster_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the accepted LOO aggregation for every peer-membership row.

    ``df`` must already be course-grained (see
    :func:`_collapse_to_peer_membership`). See the module docstring for the
    exact NaN-robust leave-one-out contract.
    """
<<<<<<< HEAD
    require_columns(
        df,
        REQUIRED_INPUT_COLUMNS,
        label="concurrent group features (roster aggregation)",
        error=KeyError,
    )
=======
    missing_cols = find_missing_columns(df, REQUIRED_INPUT_COLUMNS)
    if missing_cols:
        raise KeyError(
            f"compute_concurrent_group_features missing required columns: {missing_cols}"
        )
>>>>>>> 770c7a147a9b3b0664121b3c8c165bf6cbfc57a9

    n = len(df)

    pass_rate = pd.to_numeric(df["course_pass_rate_historical"], errors="coerce")
    d = 1.0 - pass_rate
    own_valid = pass_rate.notna().to_numpy()
    d_np = d.to_numpy(dtype="float64")  # NaN where the row itself is not valid

    # Single grouped work-frame; add helper columns, then transform per column.
    work = df.loc[:, SEMESTER_KEY].copy()
    work["_d"] = d_np
    work["_own_valid"] = own_valid.astype("int64")

    grouped = work.groupby(SEMESTER_KEY, dropna=False, sort=False)
    # 'size' counts all rows (incl. NaN); 'sum'/'max' skip NaN by default.
    size = grouped["_own_valid"].transform("size").to_numpy().astype("int64")
    group_valid_count = grouped["_own_valid"].transform("sum").to_numpy().astype("float64")
    group_sum_valid = grouped["_d"].transform("sum").to_numpy().astype("float64")
    group_max = grouped["_d"].transform("max").to_numpy().astype("float64")

    observed_count = size - 1
    missing = observed_count == 0
    has_peers = observed_count > 0
    observed_f = observed_count.astype("float64")

    # ----- mean: LOO over VALID peers (denominator is never size-1) -----
    own_d_contrib = np.where(own_valid, d_np, 0.0)
    numerator = group_sum_valid - own_d_contrib
    denom = np.where(own_valid, group_valid_count - 1.0, group_valid_count)
    mean = np.full(n, np.nan, dtype="float64")
    positive_denom = denom > 0
    values_missing = has_peers & ~positive_denom
    mean[positive_denom] = numerator[positive_denom] / denom[positive_denom]

    # ----- max: LOO over VALID peers, with unique-max -> second-highest -----
    at_max = own_valid & (d_np == group_max)
    below = np.where(own_valid & (d_np < group_max), d_np, np.nan)
    work["_at_max"] = at_max.astype("int64")
    work["_below"] = below
    grouped2 = work.groupby(SEMESTER_KEY, dropna=False, sort=False)
    count_at_max = grouped2["_at_max"].transform("sum").to_numpy()
    second_max = grouped2["_below"].transform("max").to_numpy().astype("float64")
    unique_max_holder = at_max & (count_at_max == 1)
    peer_max = np.where(unique_max_holder, second_max, group_max).astype("float64")

    # No peer set at all (singleton group) -> both value columns NaN.
    mean[missing] = np.nan
    peer_max[missing] = np.nan

    # ----- audit: weak-fallback share among OBSERVED peers -----
    weak = (pd.to_numeric(df["course_difficulty_missing"], errors="coerce") == 1)
    weak_f = weak.astype("float64").to_numpy()
    work["_weak"] = weak.astype("int64").to_numpy()
    grouped3 = work.groupby(SEMESTER_KEY, dropna=False, sort=False)
    group_weak_sum = grouped3["_weak"].transform("sum").to_numpy().astype("float64")
    peer_weak_sum = group_weak_sum - weak_f
    weak_ratio = np.zeros(n, dtype="float64")
    weak_ratio[has_peers] = peer_weak_sum[has_peers] / observed_f[has_peers]

    # ----- audit: same-requirement-type share among OBSERVED peers -----
    rt = df["requirement_type_id"]
    rt_valid = rt.notna().to_numpy()
    work_rt = df.loc[:, SEMESTER_KEY].copy()
    work_rt["_rt"] = rt.to_numpy()
    # Grouping by the semester keys + the exact req_type isolates same-type rows;
    # NaN-req_type rows form their own subgroup and are masked out below.
    grouped_rt = work_rt.groupby(SEMESTER_KEY + ["_rt"], dropna=False, sort=False)
    same_type_incl_self = grouped_rt["_rt"].transform("size").to_numpy().astype("float64")
    matching_peers = same_type_incl_self - 1.0
    same_ratio = np.full(n, np.nan, dtype="float64")  # NaN target req_type -> NaN
    rt_valid_peers = rt_valid & has_peers
    same_ratio[rt_valid_peers] = matching_peers[rt_valid_peers] / observed_f[rt_valid_peers]
    same_ratio[rt_valid & ~has_peers] = 0.0  # valid req_type but no peers -> 0

    return pd.DataFrame(
        {
            "concurrent_peer_difficulty_mean": mean,
            "concurrent_peer_difficulty_max": peer_max,
            "concurrent_peer_difficulty_missing": missing.astype("int64"),
            "concurrent_peer_set_empty": missing.astype("int64"),
            "concurrent_peer_difficulty_values_missing": values_missing.astype(
                "int64"
            ),
            "concurrent_peer_observed_count": observed_count.astype("int64"),
            "concurrent_peer_weak_ratio": weak_ratio,
            "concurrent_peer_same_req_type_ratio": same_ratio,
        },
        index=df.index,
    )


def _validate_two_input_contract(
    target_df: pd.DataFrame,
    roster_df: pd.DataFrame,
) -> np.ndarray:
    """Return each target's unique positional occurrence in ``roster_df``."""
<<<<<<< HEAD
    require_columns(
        target_df,
        TWO_INPUT_TARGET_REQUIRED_COLUMNS,
        label="concurrent group features (target rows)",
        error=KeyError,
    )
    require_columns(
        roster_df,
        TWO_INPUT_ROSTER_REQUIRED_COLUMNS,
        label="concurrent group features (roster)",
        error=KeyError,
    )
=======
    target_missing = find_missing_columns(
        target_df, TWO_INPUT_TARGET_REQUIRED_COLUMNS
    )
    if target_missing:
        raise KeyError(
            "compute_concurrent_group_features target rows missing required "
            f"columns: {target_missing}"
        )
    roster_missing = find_missing_columns(
        roster_df, TWO_INPUT_ROSTER_REQUIRED_COLUMNS
    )
    if roster_missing:
        raise KeyError(
            "compute_concurrent_group_features roster missing required "
            f"columns: {roster_missing}"
        )
>>>>>>> 770c7a147a9b3b0664121b3c8c165bf6cbfc57a9

    target_ids = target_df[STABLE_OCCURRENCE_KEY]
    roster_ids = roster_df[STABLE_OCCURRENCE_KEY]
    if target_ids.isna().any():
        raise ValueError(
            f"Target {STABLE_OCCURRENCE_KEY} must be non-null in two-input mode"
        )
    if roster_ids.isna().any():
        raise ValueError(
            f"Roster {STABLE_OCCURRENCE_KEY} must be non-null in two-input mode"
        )

    duplicate_target = target_ids.duplicated(keep=False)
    if duplicate_target.any():
        examples = target_ids.loc[duplicate_target].astype(str).unique()[:5].tolist()
        raise ValueError(
            f"Target {STABLE_OCCURRENCE_KEY} values must be unique; "
            f"duplicate examples: {examples}"
        )
    duplicate_roster = roster_ids.duplicated(keep=False)
    if duplicate_roster.any():
        examples = roster_ids.loc[duplicate_roster].astype(str).unique()[:5].tolist()
        raise ValueError(
            f"Roster {STABLE_OCCURRENCE_KEY} values must identify exactly one "
            f"occurrence; duplicate examples: {examples}"
        )

    roster_position_by_id = pd.Series(
        np.arange(len(roster_df), dtype="int64"),
        index=pd.Index(roster_ids.to_numpy(), name=STABLE_OCCURRENCE_KEY),
    )
    unmatched = ~target_ids.isin(roster_position_by_id.index)
    if unmatched.any():
        examples = target_ids.loc[unmatched].astype(str).unique()[:5].tolist()
        raise ValueError(
            "Every target must have exactly one roster occurrence; unmatched "
            f"{STABLE_OCCURRENCE_KEY} examples: {examples}"
        )
    target_positions = roster_position_by_id.loc[target_ids.to_numpy()].to_numpy(
        dtype="int64"
    )

    matched_roster = roster_df.iloc[target_positions]
    for column in PEER_MEMBERSHIP_KEY:
        target_raw = target_df[column].reset_index(drop=True)
        roster_raw = matched_roster[column].reset_index(drop=True)
        if column == "part_id":
            target_values = pd.to_numeric(target_raw, errors="coerce").astype(
                "Int64"
            )
            roster_values = pd.to_numeric(roster_raw, errors="coerce").astype(
                "Int64"
            )
        else:
            target_values = normalize_id_series(target_raw)
            roster_values = normalize_id_series(roster_raw)
        equal = target_values.eq(roster_values) | (
            target_values.isna() & roster_values.isna()
        )
        mismatch = ~equal.fillna(False).to_numpy(dtype=bool)
        if mismatch.any():
            first = int(np.flatnonzero(mismatch)[0])
            occurrence_id = target_ids.iloc[first]
            raise ValueError(
                f"Target/roster SEMESTER_KEY mismatch for "
                f"{STABLE_OCCURRENCE_KEY}={occurrence_id!r}: {column} "
                f"{target_values.iloc[first]!r} != {roster_values.iloc[first]!r}"
            )

    return target_positions


def _select_for_targets(
    features: pd.DataFrame,
    peer_positions: np.ndarray,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """Project peer-level features back onto target rows, order preserved."""
    out = features.iloc[peer_positions].copy()
    out.index = target_df.index
<<<<<<< HEAD
    assert_shape_preserved(
        target_df, out, label="concurrent feature selection", check_index=True,
        error=AssertionError,
    )
=======
    if shape_changed(target_df, out, check_index=True):
        raise AssertionError(
            "Concurrent feature selection changed target row count, order, or index"
        )
>>>>>>> 770c7a147a9b3b0664121b3c8c165bf6cbfc57a9
    return out


def compute_concurrent_group_features(
    target_df: pd.DataFrame,
    roster_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return concurrent columns in target row order and with the target index.

    With ``roster_df`` omitted, ``target_df`` is treated as the complete roster
    (the inference-plan contract). In two-input mode, aggregation is performed
    over the complete enriched roster, then the peer entry for each target is
    selected via its ``student_course_id``.

    Both modes collapse peer membership to one row per
    ``SEMESTER_KEY + course_id`` first, so the two modes agree exactly whenever
    the roster equals the target rows.
    """
    if roster_df is None:
        peers, row_to_peer = _collapse_to_peer_membership(target_df, "target rows")
        return _select_for_targets(
            _compute_roster_features(peers), row_to_peer, target_df
        )

    peers, row_to_peer = _collapse_to_peer_membership(roster_df, "roster")
    target_roster_positions = _validate_two_input_contract(target_df, roster_df)
    return _select_for_targets(
        _compute_roster_features(peers),
        row_to_peer[target_roster_positions],
        target_df,
    )


def add_concurrent_group_features(
    target_df: pd.DataFrame,
    roster_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Append concurrent columns to target rows, preserving order and index."""
    already = [c for c in CONCURRENT_FEATURE_COLUMNS if c in target_df.columns]
    if already:
        raise ValueError(f"Concurrent columns already present: {already}")

    features = compute_concurrent_group_features(target_df, roster_df)
    out = pd.concat([target_df, features], axis=1)
<<<<<<< HEAD
    assert_shape_preserved(
        target_df, out, label="concurrent enrichment", check_index=True,
        error=AssertionError,
    )
=======
    if shape_changed(target_df, out, check_index=True):
        raise AssertionError("Concurrent enrichment changed row count, order, or index")
>>>>>>> 770c7a147a9b3b0664121b3c8c165bf6cbfc57a9
    return out
