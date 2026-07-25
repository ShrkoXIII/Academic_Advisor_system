"""Concurrent Course Group feature family (Experiment 1).

For each row (a student's target course C in a given semester), "peers" are the
other rows sharing the same semester group
``[university_id, student_id, degree_id, part_id]`` (identical to
``feature_engineering.SEMESTER_KEY``). This module derives how difficult a
course's same-semester peers are, using vectorized groupby leave-one-out (LOO)
aggregation — no per-row Python loops.

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

from src.feature_engineering import SEMESTER_KEY

# The three columns that enter MODEL_FEATURES.
MODEL_CONCURRENT_FEATURES = [
    "concurrent_peer_difficulty_mean",
    "concurrent_peer_difficulty_max",
    "concurrent_peer_difficulty_missing",
]

# Audit-only columns: persisted to the parquet but never added to MODEL_FEATURES.
AUDIT_CONCURRENT_FEATURES = [
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

REQUIRED_INPUT_COLUMNS = SEMESTER_KEY + CONCURRENT_INPUT_COLUMNS


def compute_concurrent_group_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the six concurrent columns, indexed like ``df``.

    See the module docstring for the exact NaN-robust leave-one-out contract.
    """
    missing_cols = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"compute_concurrent_group_features missing required columns: {missing_cols}"
        )

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
            "concurrent_peer_observed_count": observed_count.astype("int64"),
            "concurrent_peer_weak_ratio": weak_ratio,
            "concurrent_peer_same_req_type_ratio": same_ratio,
        },
        index=df.index,
    )


def add_concurrent_group_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with the six concurrent columns appended, order preserved."""
    already = [c for c in CONCURRENT_FEATURE_COLUMNS if c in df.columns]
    if already:
        raise ValueError(f"Concurrent columns already present: {already}")

    features = compute_concurrent_group_features(df)
    out = pd.concat([df, features], axis=1)
    if len(out) != len(df) or not out.index.equals(df.index):
        raise AssertionError("Concurrent enrichment changed row count, order, or index")
    return out
