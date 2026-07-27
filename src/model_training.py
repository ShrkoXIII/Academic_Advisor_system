"""Train LightGBM pass/fail (M1) and final-mark (M2) models — V1 baseline.

Model naming — LOCKED (do not reverse):
    M1 = pass/fail CLASSIFIER   target = (final_mark >= 50).astype(int)
    M2 = final_mark REGRESSOR    target = final_mark   (raw 0-100, no transform)

Artifact mapping:
    persistent --run-name runs -> MODELS_DIR/runs/<timestamp>__<case>/
    no --run-name quick runs  -> MODELS_DIR/quick/latest/

V1 scope: LightGBM only. No sample weights. No scale_pos_weight. No XGBoost.

Usage
-----
python -m src.model_training

All arguments default to the canonical final-generation splits returned by
``src.paths.model_split_path`` (written by 03_diploma_type_bucketing) and
MODELS_DIR. Pass --train/--valid/--test/--out only to override. From a
notebook, call main([]) so Jupyter's own argv is not parsed.

This module never builds the parquet splits. Every invocation trains both
models from scratch; loading saved weights for prediction is handled by the
inference/analysis code instead.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Mapping, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.feature_engineering import assert_no_leakage_columns
from src import experiment_tracking
from src.paths import MODELS_DIR, model_split_path

# ---------------------------------------------------------------------------
# Feature contracts — explicit named registries
# ---------------------------------------------------------------------------
# These are explicit allowlists. They are NOT derived from fe.FEATURE_COLUMNS,
# because that list contains diagnostic / string / leakage columns and is not a
# model allowlist.
#
# Three contracts exist so the concurrent-feature effect can be measured in
# isolation:
#
#   baseline_41   the established list including GPA trend, WITHOUT the three
#                 concurrent model features
#   concurrent_44 baseline_41 plus exactly those three
#   concurrent_43 concurrent_44 minus concurrent_peer_difficulty_missing, the
#                 one of the three found effectively dead (Decisions_Log.md:
#                 used by any model in only 2 of 5 seeds, under 0.001% of
#                 total gain, zero splits everywhere else)
#
# ORDER NOTE (deliberate, see the module tests): the three concurrent features
# sit at zero-based positions 33/34/35 of the accepted 44-feature contract, not
# at the end. That position is pinned by the dataset builder
# (scripts/build_concurrent_group_features.py EXPECTED_LEGACY_MODEL_POSITION=35)
# and by tests. baseline_41 is therefore the 44-list with those three removed,
# preserving every other feature's relative order. The relationship is a
# set-difference identity, not a list concatenation. concurrent_43 preserves
# the order of the remaining 43 features exactly as they sit in concurrent_44.

CONCURRENT_MODEL_FEATURES: List[str] = [
    "concurrent_peer_difficulty_mean",
    "concurrent_peer_difficulty_max",
    "concurrent_peer_difficulty_missing",
]

CONCURRENT_44_FEATURES: List[str] = [
    # --- Student GPA history ---
    "prev_gpa_points_clean",                  # zero + flags design (NOT model_prev_gpa)
    "start_agpa_points",                      # cumulative AGPA at semester start
    "last_valid_gpa_before_current_semester", # last real GPA, skips interruptions
    "gpa_trend_delta",                        # last valid GPA minus second-last valid GPA
    "gpa_trend_missing",                      # trend undefined until two prior valid GPAs
    # --- Student workload / fail history ---
    "start_total_in_credits",
    "start_total_in_courses",
    "total_fail_credits_capped",
    "fail_credit_ratio_capped",
    "is_extreme_fail_history",
    "reg_total_semesters",
    # --- Academic standing ---
    "start_level_ord",
    "start_semester",                         # 1/2/3 term started (NOT start_year)
    # --- Timeline / interruptions (PAST-ONLY; is_interruption_semester dropped) ---
    "prior_interruption_count",
    "consecutive_interruption_count",
    "prev_semester_was_interruption",
    "no_previous_progress",
    "is_first_active_semester",
    "is_first_row_in_timeline",
    # --- Previous-GPA data-quality flags ---
    "prev_gpa_points_missing",
    "prev_gpa_points_zero",
    "prev_gpa_invalid_zero_case",
    # --- Current semester context ---
    "semester_reg_credits",
    "semester_reg_courses",
    "part_semester",                          # 1/2/3 term (NOT part_year)
    # --- Course properties ---
    "course_credits",
    "attempt_number",
    "is_high_credit_course",
    # --- Course difficulty (empirical, train-only) ---
    "course_pass_rate_historical",
    "course_avg_mark_historical",
    "course_retake_rate_historical",
    "course_history_count",                   # support count -> reliability of difficulty stats
    "course_difficulty_missing",
    # --- Concurrent course group (same-semester peer difficulty, LOO) ---
    "concurrent_peer_difficulty_mean",        # mean (1 - pass_rate) over valid peers
    "concurrent_peer_difficulty_max",         # hardest valid peer
    "concurrent_peer_difficulty_missing",     # 1 when the peer set is empty (singleton)
    # --- Degree / requirement context ---
    "requirement_type_id",                    # CATEGORICAL (handled specially)
    "requirement_type_missing",
    "degree_requirement_credits_count",
    "degree_requirement_credits_count_missing",
    "course_share_of_requirement",
    "requirement_size_bucket_ord",            # ordinal; raw string never enters X
    # --- Pre-admission diploma signals ---
    "diploma_gpa",
    "diploma_type_bucket",                    # CATEGORICAL; raw diploma_type_id stays audit-only
]

# baseline_41 preserves the order of every non-concurrent feature exactly.
BASELINE_41_FEATURES: List[str] = [
    feature
    for feature in CONCURRENT_44_FEATURES
    if feature not in set(CONCURRENT_MODEL_FEATURES)
]

# DEPRECATED alias kept so existing importers keep working unchanged
# (src/inference.py, scripts/build_*.py, notebooks). New code should resolve a
# named contract via resolve_feature_contract() instead of reading this global.
MODEL_FEATURES: List[str] = CONCURRENT_44_FEATURES

# --- binding contract invariants (checked at import) ---
assert len(BASELINE_41_FEATURES) == 41, len(BASELINE_41_FEATURES)
assert len(CONCURRENT_44_FEATURES) == 44, len(CONCURRENT_44_FEATURES)
assert len(set(BASELINE_41_FEATURES)) == 41, "duplicate in BASELINE_41_FEATURES"
assert len(set(CONCURRENT_44_FEATURES)) == 44, "duplicate in CONCURRENT_44_FEATURES"
assert set(CONCURRENT_44_FEATURES) - set(BASELINE_41_FEATURES) == set(
    CONCURRENT_MODEL_FEATURES
), "concurrent_44 minus baseline_41 must be exactly the three concurrent features"
assert set(BASELINE_41_FEATURES) - set(CONCURRENT_44_FEATURES) == set(), (
    "baseline_41 must be a subset of concurrent_44"
)
# Order-preserving form of "44 = 41 + the three": dropping the concurrent
# features from the 44-list reproduces the 41-list exactly, in order.
assert [
    f for f in CONCURRENT_44_FEATURES if f not in set(CONCURRENT_MODEL_FEATURES)
] == BASELINE_41_FEATURES
for _gpa_feature in ("gpa_trend_delta", "gpa_trend_missing"):
    assert _gpa_feature in BASELINE_41_FEATURES, _gpa_feature
    assert _gpa_feature in CONCURRENT_44_FEATURES, _gpa_feature

# concurrent_43 = concurrent_44 minus the dead legacy indicator, order preserved.
CONCURRENT_43_FEATURES: List[str] = [
    feature
    for feature in CONCURRENT_44_FEATURES
    if feature != "concurrent_peer_difficulty_missing"
]

assert len(CONCURRENT_43_FEATURES) == 43, len(CONCURRENT_43_FEATURES)
assert len(set(CONCURRENT_43_FEATURES)) == 43, "duplicate in CONCURRENT_43_FEATURES"
assert "concurrent_peer_difficulty_missing" not in CONCURRENT_43_FEATURES
assert set(CONCURRENT_44_FEATURES) - set(CONCURRENT_43_FEATURES) == {
    "concurrent_peer_difficulty_missing"
}, "concurrent_44 minus concurrent_43 must be exactly the dead legacy indicator"
assert set(CONCURRENT_43_FEATURES) - set(CONCURRENT_44_FEATURES) == set(), (
    "concurrent_43 must be a subset of concurrent_44"
)
for _gpa_feature in ("gpa_trend_delta", "gpa_trend_missing"):
    assert _gpa_feature in CONCURRENT_43_FEATURES, _gpa_feature
for _remaining_concurrent_feature in (
    "concurrent_peer_difficulty_mean",
    "concurrent_peer_difficulty_max",
):
    assert _remaining_concurrent_feature in CONCURRENT_43_FEATURES, _remaining_concurrent_feature

# Columns deliberately EXCLUDED (kept here as a guard list, asserted absent from X).
DROPPED_FEATURES: List[str] = [
    "is_interruption_semester",          # LEAKAGE: current-semester pass_credits/gpa_points
    "model_prev_gpa",                    # depends on structural_zero_as_nan; use clean + flags
    "prev_gpa_actual_zero_performance",  # dead stub, hardcoded 0
    "start_level_missing",               # audit-only; constant in current train population
    "difficulty_fallback_level",         # audit-only; train is intentionally all Level 1
    "part_year",                         # absolute calendar year -> drift
    "start_year",                        # absolute calendar year -> drift
    "difficulty_group_support_count",    # audit only
]

CATEGORICAL_FEATURES: List[str] = ["requirement_type_id", "diploma_type_bucket"]
UNKNOWN_CATEGORY = -1   # NaN + categories unseen in train map here

REQUIREMENT_BUCKET_ORD = {
    "none_or_unknown": 0,
    "small": 1,
    "medium": 2,
    "large": 3,
    "very_large": 4,
}

# Patterns that indicate a leftover composite/lookup key from difficulty building.
_LEFTOVER_KEY_PATTERNS = ["l3_key", "l4_key", "_key", "tmp", "temp", "idx"]

TARGET_GRADE = "final_mark"       # M2 regressor target
EXPECTED_FEATURE_COUNT = 44       # DEPRECATED alias: concurrent_44's count
DERIVED_FEATURE_SOURCES = {
    "requirement_size_bucket_ord": "requirement_size_bucket",
}
SEGMENT_ONLY_COLUMNS = ["difficulty_fallback_level"]

# LOCKED reporting threshold. Precision / recall / F1 / confusion matrix are all
# reported at this probability cut. It is NOT a training or early-stopping
# parameter, and it is NOT optimized per run — it is fixed so runs stay
# comparable. Runs made in the earlier 0.5 / 0.85 era are therefore NOT
# P/R-comparable with runs made at this threshold.
REPORTING_THRESHOLD = 0.80

TARGET_M1_DEFINITION = "(final_mark >= 50).astype(int)"
TARGET_M2_DEFINITION = "final_mark"


@dataclass(frozen=True)
class FeatureContract:
    """One immutable, explicitly named model-input contract."""

    name: str
    version: str
    features: Tuple[str, ...]
    categorical_features: Tuple[str, ...]
    derived_feature_sources: Mapping[str, str]
    reporting_threshold: float
    requires_concurrent_plan_context: bool
    description: str

    @property
    def expected_feature_count(self) -> int:
        return len(self.features)

    @property
    def source_features(self) -> List[str]:
        """Contract features that are read from the parquet as-is."""
        return [c for c in self.features if c not in self.derived_feature_sources]

    @property
    def numeric_features(self) -> List[str]:
        return [c for c in self.features if c not in self.categorical_features]

    @property
    def training_data_columns(self) -> List[str]:
        """Only the columns this contract needs: features, derivations, target, segments.

        The final parquet files carry many audit/string columns that training
        never consumes; loading them added hundreds of MiB per process. The
        baseline_41 contract simply does not list the concurrent columns, so
        they stay on disk.
        """
        return list(dict.fromkeys(
            self.source_features
            + list(self.derived_feature_sources.values())
            + [TARGET_GRADE]
            + SEGMENT_ONLY_COLUMNS
        ))


BASELINE_41_CONTRACT = FeatureContract(
    name="baseline_41",
    version="v1",
    features=tuple(BASELINE_41_FEATURES),
    categorical_features=tuple(CATEGORICAL_FEATURES),
    derived_feature_sources=DERIVED_FEATURE_SOURCES,
    reporting_threshold=REPORTING_THRESHOLD,
    requires_concurrent_plan_context=False,
    description=(
        "Established feature list including GPA trend, excluding the three "
        "concurrent peer-difficulty features. Controlled-experiment baseline."
    ),
)

CONCURRENT_44_CONTRACT = FeatureContract(
    name="concurrent_44",
    version="v1",
    features=tuple(CONCURRENT_44_FEATURES),
    categorical_features=tuple(CATEGORICAL_FEATURES),
    derived_feature_sources=DERIVED_FEATURE_SOURCES,
    reporting_threshold=REPORTING_THRESHOLD,
    requires_concurrent_plan_context=True,
    description=(
        "baseline_41 plus concurrent_peer_difficulty_{mean,max,missing}. "
        "Valid at serve time only once a full plan exists and score_plan has "
        "recomputed the same-semester peer context."
    ),
)

CONCURRENT_43_CONTRACT = FeatureContract(
    name="concurrent_43",
    version="v1",
    features=tuple(CONCURRENT_43_FEATURES),
    categorical_features=tuple(CATEGORICAL_FEATURES),
    derived_feature_sources=DERIVED_FEATURE_SOURCES,
    reporting_threshold=REPORTING_THRESHOLD,
    requires_concurrent_plan_context=True,
    description=(
        "concurrent_44 minus concurrent_peer_difficulty_missing (dead: used by "
        "any model in only 2 of 5 seeds, under 0.001% of total gain, zero "
        "splits everywhere else; see Decisions_Log.md). Still requires plan "
        "context at serve time because concurrent_peer_difficulty_mean/max "
        "remain in the contract."
    ),
)

# Known production limitation, recorded with every run so it cannot be lost.
# NOT wired into recommendation by this experiment.
SERVING_LIMITATION_NOTE = (
    "score() pre-plan candidate ranking must use baseline_41; concurrent_44 is "
    "semantically valid only after a complete plan is formed and score_plan "
    "recomputes the same-semester concurrent context."
)

FEATURE_CONTRACTS: Dict[str, FeatureContract] = {
    BASELINE_41_CONTRACT.name: BASELINE_41_CONTRACT,
    CONCURRENT_44_CONTRACT.name: CONCURRENT_44_CONTRACT,
    CONCURRENT_43_CONTRACT.name: CONCURRENT_43_CONTRACT,
}

DEFAULT_FEATURE_CONTRACT = CONCURRENT_44_CONTRACT.name


def resolve_feature_contract(
    contract: "str | FeatureContract | None" = None,
) -> FeatureContract:
    """Resolve a contract by name. Never infers one from available columns."""
    if isinstance(contract, FeatureContract):
        return contract
    if contract is None:
        return FEATURE_CONTRACTS[DEFAULT_FEATURE_CONTRACT]
    try:
        return FEATURE_CONTRACTS[contract]
    except KeyError:
        raise ValueError(
            f"Unknown feature contract {contract!r}. "
            f"Choose one of: {sorted(FEATURE_CONTRACTS)}"
        ) from None


# DEPRECATED alias: the concurrent_44 column set. Kept for existing importers.
TRAINING_DATA_COLUMNS = CONCURRENT_44_CONTRACT.training_data_columns


# ---------------------------------------------------------------------------
# Categorical level handling (learn from train ONLY, apply to all splits)
# ---------------------------------------------------------------------------

def learn_categorical_levels(
    df_train: pd.DataFrame,
    contract: "str | FeatureContract | None" = None,
) -> Dict[str, List[int]]:
    """Learn the allowed category set for each categorical feature from TRAIN only."""
    resolved = resolve_feature_contract(contract)
    levels: Dict[str, List[int]] = {}
    for col in resolved.categorical_features:
        vals = pd.to_numeric(df_train[col], errors="coerce").dropna().astype(int).unique()
        levels[col] = sorted(int(v) for v in vals)
        print(f"  [cat] {col}: learned {len(levels[col])} levels from train -> {levels[col]}")
    return levels


def _apply_categorical_levels(df: pd.DataFrame, levels: Dict[str, List[int]]) -> pd.DataFrame:
    """Map each categorical column: NaN + unseen-in-train -> UNKNOWN_CATEGORY (-1),
    then store as an explicit pandas Categorical so -1 is a real bucket, not 'missing'."""
    for col, allowed in levels.items():
        s = pd.to_numeric(df[col], errors="coerce")
        mapped = s.where(s.isin(allowed), other=UNKNOWN_CATEGORY).fillna(UNKNOWN_CATEGORY).astype(int)
        categories = sorted(set(allowed + [UNKNOWN_CATEGORY]))
        df[col] = pd.Categorical(mapped, categories=categories)
    return df


# ---------------------------------------------------------------------------
# Target checks
# ---------------------------------------------------------------------------

def _check_target_column(df: pd.DataFrame) -> None:
    """Fail loudly if final_mark is missing, has nulls, or is out of [0, 100]."""
    if TARGET_GRADE not in df.columns:
        raise ValueError(f"Target column '{TARGET_GRADE}' is missing from the dataframe.")
    null_count = int(df[TARGET_GRADE].isna().sum())
    if null_count > 0:
        raise ValueError(f"Target '{TARGET_GRADE}' has {null_count:,} null value(s).")
    out_of_range = int(((df[TARGET_GRADE] < 0) | (df[TARGET_GRADE] > 100)).sum())
    if out_of_range > 0:
        raise ValueError(
            f"Target '{TARGET_GRADE}' has {out_of_range:,} value(s) outside [0, 100] "
            f"(min={df[TARGET_GRADE].min()}, max={df[TARGET_GRADE].max()})."
        )


# ---------------------------------------------------------------------------
# X / y construction
# ---------------------------------------------------------------------------

def prepare_X_y(
    df: pd.DataFrame,
    target: str,
    categorical_levels: Dict[str, List[int]],
    contract: "str | FeatureContract | None" = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) ready for LightGBM, with exactly the contract's columns in order.

    target='pass'  -> M1 classifier label (final_mark >= 50) as int
    target='grade' -> M2 regressor label  final_mark (float)

    categorical_levels MUST be learned from df_train and passed in for every
    split, so valid/test never define their own category set.

    ``contract`` selects the named feature contract. It defaults to
    concurrent_44 for backward compatibility with existing importers; the
    contract is never inferred from which columns happen to be present.
    """
    resolved = resolve_feature_contract(contract)
    if categorical_levels is None:
        raise ValueError("categorical_levels is required (learn from df_train first).")

    _check_target_column(df)

    # --- requirement_size_bucket -> ordinal, with a LOUD warning on unmapped values ---
    raw_bucket = df["requirement_size_bucket"].astype("string")
    seen = set(raw_bucket.dropna().unique())
    unmapped = sorted(seen - set(REQUIREMENT_BUCKET_ORD.keys()))
    if unmapped:
        counts = raw_bucket[raw_bucket.isin(unmapped)].value_counts().to_dict()
        print(
            "WARNING: requirement_size_bucket has values NOT in REQUIREMENT_BUCKET_ORD; "
            f"they will map to 0. Unmapped values + counts: {counts}"
        )
    requirement_size_bucket_ord = (
        raw_bucket.map(REQUIREMENT_BUCKET_ORD).fillna(0).astype(int)
    )

    # --- presence check: a contract fails loudly on ANY missing column ---
    source_features = resolved.source_features
    missing_cols = [c for c in source_features if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing expected feature columns for contract "
            f"{resolved.name!r}: {missing_cols}"
        )

    # Copy only the model matrix, not the full source dataframe.
    X = df[source_features].copy()
    X.insert(
        resolved.features.index("requirement_size_bucket_ord"),
        "requirement_size_bucket_ord",
        requirement_size_bucket_ord,
    )

    # --- categorical handling (NaN + unseen -> -1, explicit category) ---
    X = _apply_categorical_levels(X, categorical_levels)

    # --- dtype casting: numeric -> float64; categorical stays 'category' ---
    # Cast one column at a time to avoid a second full numeric-frame temporary.
    for col in resolved.numeric_features:
        X[col] = X[col].astype("float64")

    # --- leakage gate + dropped-feature guard ---
    assert_no_leakage_columns(X)
    leaked_back = [c for c in DROPPED_FEATURES if c in X.columns]
    if leaked_back:
        raise ValueError(f"Dropped features reappeared in X: {leaked_back}")

    # --- target ---
    if target == "grade":
        y = df[TARGET_GRADE].astype(float)
    elif target == "pass":
        y = (df[TARGET_GRADE] >= 50).astype(int)
    else:
        raise ValueError(f"Unknown target '{target}'. Use 'grade' or 'pass'.")

    return X, y


# ---------------------------------------------------------------------------
# Pre-training diagnostics (print loudly BEFORE any training)
# ---------------------------------------------------------------------------

def run_pre_training_diagnostics(
    df_train: pd.DataFrame,
    categorical_levels: Dict[str, List[int]],
    contract: "str | FeatureContract | None" = None,
) -> Dict[str, str]:
    """Print the feature contract checks and return {feature: dtype} after casting."""
    resolved = resolve_feature_contract(contract)
    features = list(resolved.features)
    print("\n=== PRE-TRAINING DIAGNOSTICS ===")

    # 1) final feature list + count
    print(
        f"\n[1] contract={resolved.name!r} feature count = {len(features)} "
        f"(expected {resolved.expected_feature_count})"
    )
    for i, c in enumerate(features, 1):
        marker = "  <-- concurrent" if c in CONCURRENT_MODEL_FEATURES else ""
        print(f"    {i:>2}. {c}{marker}")
    assert len(features) == resolved.expected_feature_count, (
        f"Feature count {len(features)} != {resolved.expected_feature_count}"
    )
    assert len(set(features)) == len(features), (
        f"Duplicate feature in contract {resolved.name!r}"
    )

    # 2) missing features (fail loudly) — uses prepare_X_y so derived cols count
    X_train, _ = prepare_X_y(df_train, "pass", categorical_levels, resolved)

    # 3) dropped-feature guard
    still_present = [c for c in DROPPED_FEATURES if c in X_train.columns]
    print(f"\n[2] Dropped features present in X (must be empty): {still_present}")
    assert not still_present, f"Forbidden features in X: {still_present}"

    # 4) loaded non-model columns (target/derivation/segment inputs; informational)
    used = set(features) | {TARGET_GRADE}
    unused = sorted(c for c in df_train.columns if c not in used)
    print(f"\n[3] Loaded columns NOT used directly by the model ({len(unused)}):")
    print("    " + ", ".join(unused))

    # 5) leftover-key audit (flag only)
    flagged = sorted(
        c for c in df_train.columns
        if any(p in c.lower() for p in _LEFTOVER_KEY_PATTERNS)
    )
    print(f"\n[4] Leftover-key audit (flag only, not auto-dropped): {flagged}")

    # 6) train constant-column report
    nun = X_train.nunique(dropna=False)
    constants = nun[nun <= 1]
    print(f"\n[5] Train constant columns (nunique <= 1): {list(constants.index)}")
    if "is_high_credit_course" in constants.index:
        print("    NOTE: is_high_credit_course flagged constant — grad exam should be in train; investigate.")

    # dtypes for the feature contract
    dtypes = {c: str(X_train[c].dtype) for c in features}
    print("\n[6] Feature dtypes after casting:")
    for c in features:
        print(f"    {c}: {dtypes[c]}")

    del X_train
    gc.collect()
    print("\n=== END DIAGNOSTICS ===\n")
    return dtypes


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

_SHARED_PARAMS = {
    "learning_rate": 0.05, ## default 1 
    "num_leaves": 127, #depth 
    "min_child_samples": 50, ## 
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    # Bound the histogram cache and avoid LightGBM's extra row-wise Dataset
    # allocation. This trades a little speed for a predictable RAM ceiling.
    "histogram_pool_size": 256,
    "force_col_wise": True,
    "num_threads": 4,
    "verbose": -1,
    "seed": 42,
}

# Objective/metric are the only LightGBM settings that legitimately differ
# between M1 and M2. Everything else comes from _SHARED_PARAMS, so a run's two
# models are identical apart from these two keys plus per-run overrides.
M1_OBJECTIVE, M1_METRIC = "binary", "auc"
M2_OBJECTIVE, M2_METRIC = "regression_l1", "mae"

# Regularization levers exposed as explicit CLI flags. Kept as a named tuple of
# keys (not a free-form params string) so every run's deviation from
# _SHARED_PARAMS is enumerable and auditable from the artifacts alone.
TUNABLE_PARAM_NAMES = ("num_leaves", "min_child_samples", "reg_lambda")


def effective_lgbm_params(
    objective: str,
    metric: str,
    overrides: "Dict[str, Any] | None" = None,
) -> Dict[str, Any]:
    """The COMPLETE LightGBM parameter dict one model is trained with.

    Single source of truth: the training functions build their parameters with
    this, and the CLI records its output in metrics.json. A recorded dict and
    the dict handed to ``lgb.train`` therefore cannot drift apart.
    """
    params = {"objective": objective, "metric": metric, **_SHARED_PARAMS}
    if overrides:
        params.update(overrides)
    return params


def _assert_booster_matches_params(
    booster: lgb.Booster, expected: Dict[str, Any], label: str
) -> None:
    """Verify the trained model actually received the parameters we recorded.

    LightGBM keeps extra keys of its own (``num_iterations``,
    ``early_stopping_round``); only the ones we set are checked.
    """
    actual = booster.params or {}
    mismatched = {
        key: (value, actual.get(key))
        for key, value in expected.items()
        if key in actual and actual[key] != value
    }
    if mismatched:
        raise AssertionError(
            f"{label}: recorded LightGBM parameters differ from the trained "
            f"booster's own (expected, actual): {mismatched}"
        )


NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 50
EFFECTIVE_SEED_PARAM_NAMES = (
    "seed",
    "data_random_seed",
    "feature_fraction_seed",
    "bagging_seed",
    "drop_seed",
)


def _effective_seed_settings(model_path: "str | Path") -> Dict[str, int]:
    """Read LightGBM's resolved stochastic seeds from a serialized model."""
    values: Dict[str, int] = {}
    with Path(model_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("[") or ": " not in line:
                continue
            name, raw_value = line[1:-2].split(": ", 1)
            if name in EFFECTIVE_SEED_PARAM_NAMES:
                values[name] = int(raw_value)
                if len(values) == len(EFFECTIVE_SEED_PARAM_NAMES):
                    break
    missing = [name for name in EFFECTIVE_SEED_PARAM_NAMES if name not in values]
    if missing:
        raise AssertionError(f"LightGBM did not expose effective seed parameters: {missing}")
    return {name: values[name] for name in EFFECTIVE_SEED_PARAM_NAMES}


def _make_datasets(X_tr, y_tr, X_va, y_va, categorical_features=None):
    categorical = list(categorical_features or CATEGORICAL_FEATURES)
    dtrain = lgb.Dataset(
        X_tr, label=y_tr, free_raw_data=True, categorical_feature=categorical
    )
    dvalid = lgb.Dataset(
        X_va, label=y_va, reference=dtrain, free_raw_data=True,
        categorical_feature=categorical,
    )
    # Construct now so LightGBM can release its references to the pandas frames
    # before boosting begins. The native binned datasets retain everything needed.
    dtrain.construct()
    dvalid.construct()
    return dtrain, dvalid


def train_pass_model(  # M1 classifier
    df_train, df_valid, categorical_levels,
    params=None, num_boost_round=NUM_BOOST_ROUND,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    contract: "str | FeatureContract | None" = None,
) -> Tuple[lgb.Booster, dict]:
    """M1 — LightGBM binary classifier for pass/fail. NO scale_pos_weight in V1."""
    resolved = resolve_feature_contract(contract)
    X_tr, y_tr = prepare_X_y(df_train, "pass", categorical_levels, resolved)
    X_va, y_va = prepare_X_y(df_valid, "pass", categorical_levels, resolved)

    # Class balance is printed for information only — NOT used to weight the model.
    pass_rate = float(y_tr.mean())
    print(f"M1 class balance — train pass rate: {pass_rate:.4f} "
          f"(pass={int(y_tr.sum()):,}  fail={int(len(y_tr) - y_tr.sum()):,})")

    default_params = effective_lgbm_params(M1_OBJECTIVE, M1_METRIC, params)

    dtrain, dvalid = _make_datasets(
        X_tr, y_tr, X_va, y_va, resolved.categorical_features
    )
    del X_tr, y_tr, X_va, y_va
    gc.collect()
    evals_result: dict = {}
    # Only validation may select the stopping iteration. Test is never passed
    # to LightGBM during training, and under --evaluate-test it is not even read.
    callbacks = [
        lgb.early_stopping(early_stopping_rounds, verbose=True),
        lgb.log_evaluation(100),
        lgb.record_evaluation(evals_result),
    ]
    booster = lgb.train(
        default_params, dtrain, num_boost_round=num_boost_round,
        valid_sets=[dvalid, dtrain],
        valid_names=["valid", "train"],
        callbacks=callbacks,
    )
    return booster, evals_result


def train_grade_model(  # M2 regressor
    df_train, df_valid, categorical_levels,
    params=None, num_boost_round=NUM_BOOST_ROUND,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    contract: "str | FeatureContract | None" = None,
) -> Tuple[lgb.Booster, dict]:
    """M2 — LightGBM regressor for final_mark (0-100), MAE loss."""
    resolved = resolve_feature_contract(contract)
    X_tr, y_tr = prepare_X_y(df_train, "grade", categorical_levels, resolved)
    X_va, y_va = prepare_X_y(df_valid, "grade", categorical_levels, resolved)

    default_params = effective_lgbm_params(M2_OBJECTIVE, M2_METRIC, params)

    dtrain, dvalid = _make_datasets(
        X_tr, y_tr, X_va, y_va, resolved.categorical_features
    )
    del X_tr, y_tr, X_va, y_va
    gc.collect()
    evals_result: dict = {}
    # Only validation may select the stopping iteration. Test is never passed
    # to LightGBM during training, and under --evaluate-test it is not even read.
    callbacks = [
        lgb.early_stopping(early_stopping_rounds, verbose=True),
        lgb.log_evaluation(100),
        lgb.record_evaluation(evals_result),
    ]
    booster = lgb.train(
        default_params, dtrain, num_boost_round=num_boost_round,
        valid_sets=[dvalid, dtrain],
        valid_names=["valid", "train"],
        callbacks=callbacks,
    )
    return booster, evals_result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_pass(
    model, df, categorical_levels, label="",
    contract: "str | FeatureContract | None" = None,
    threshold: float = REPORTING_THRESHOLD,
) -> dict:  # M1
    """M1 metrics, including fail-class AP used for experiment selection.

    ``threshold`` is the LOCKED reporting cut (see REPORTING_THRESHOLD). It is
    never tuned per run; both experiment arms must pass the same value.
    """
    resolved = resolve_feature_contract(contract)
    X, y = prepare_X_y(df, "pass", categorical_levels, resolved)
    y_prob = model.predict(X)
    y_bin = (y_prob >= threshold).astype(int)

    # labels=[0,1] -> row 0 = actual fail, row 1 = actual pass
    cm = confusion_matrix(y, y_bin, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    metrics = {
        # The threshold every P/R/F1/confusion value below was computed at.
        # Persisted so runs at different thresholds are never silently compared.
        "reporting_threshold": float(threshold),
        # Keep selection metrics unrounded in artifacts. Formatting is applied
        # only when printing/reporting.
        "auc": float(roc_auc_score(y, y_prob)),
        "avg_precision": float(average_precision_score(y, y_prob)),
        "fail_avg_precision": float(average_precision_score(1 - y, 1 - y_prob)),
        "accuracy": round(float(accuracy_score(y, y_bin)), 4),
        # Pass-class (pos_label=1, default)
        "precision": round(float(precision_score(y, y_bin, zero_division=0)), 4),
        "recall": round(float(recall_score(y, y_bin, zero_division=0)), 4),
        "f1": round(float(f1_score(y, y_bin, zero_division=0)), 4),
        # Fail-class (pos_label=0)
        "fail_precision": round(float(precision_score(y, y_bin, pos_label=0, zero_division=0)), 4),
        "fail_recall": round(float(recall_score(y, y_bin, pos_label=0, zero_division=0)), 4),
        "fail_f1": round(float(f1_score(y, y_bin, pos_label=0, zero_division=0)), 4),
        "brier": float(brier_score_loss(y, y_prob)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    if label:
        print(f"[{label}] M1 pass  AUC={metrics['auc']:.4f}  AP={metrics['avg_precision']:.4f}  "
              f"Acc={metrics['accuracy']:.4f}  P={metrics['precision']:.4f}  "
              f"R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  Brier={metrics['brier']:.4f}")
        print(f"[{label}] M1 fail  AP={metrics['fail_avg_precision']:.4f}  "
              f"P={metrics['fail_precision']:.4f}  "
              f"R={metrics['fail_recall']:.4f}  F1={metrics['fail_f1']:.4f}  "
              f"(threshold={threshold:g})")
        print(f"[{label}] M1 confusion matrix (rows=actual, cols=predicted):")
        print(f"              Pred-FAIL  Pred-PASS")
        print(f"  Act-FAIL    {tn:>9,}  {fp:>9,}")
        print(f"  Act-PASS    {fn:>9,}  {tp:>9,}")
    return metrics


def evaluate_grade(
    model, df, categorical_levels, label="",
    contract: "str | FeatureContract | None" = None,
) -> dict:  # M2
    """M2 metrics: MAE, RMSE, R²."""
    X, y = prepare_X_y(df, "grade", categorical_levels, resolve_feature_contract(contract))
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    metrics = {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}
    if label:
        print(f"[{label}] M2 grade  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
    return metrics


def _segment_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """The EXISTING segment definitions. Do not add segments here."""
    return {
        "first_semester": df["is_first_active_semester"] == 1,
        "retake_attempt": df["attempt_number"] > 1,
        "low_difficulty_support": df["difficulty_fallback_level"] >= 3,
        "cold_start_gpa": df["no_previous_progress"] == 1,
    }


def collect_segment_auc(
    model, df, categorical_levels,
    contract: "str | FeatureContract | None" = None,
) -> Dict[str, dict]:
    """Collect the existing M1 segment AUCs without changing their calculation."""
    X_full, y_full = prepare_X_y(
        df, "pass", categorical_levels, resolve_feature_contract(contract)
    )
    y_prob_full = model.predict(X_full)
    segments = _segment_masks(df)
    results = {}
    for seg_name, mask in segments.items():
        sub = mask.values
        n = int(sub.sum())
        if n < 100:
            results[seg_name] = {"n": n, "auc": None, "status": "skipped_lt_100"}
            continue
        y_seg = y_full[sub]
        if y_seg.nunique() < 2:
            results[seg_name] = {"n": n, "auc": None, "status": "skipped_one_class"}
            continue
        auc = roc_auc_score(y_seg, y_prob_full[sub])
        results[seg_name] = {
            "n": n,
            "auc": round(float(auc), 4),
            "auc_unrounded": float(auc),
            "positive_rate": float(y_seg.mean()),
            "status": "ok",
        }
    return results


def stratified_eval_pass(
    model, df, categorical_levels,
    contract: "str | FeatureContract | None" = None,
) -> Dict[str, dict]:
    """Print the existing M1 AUC segment breakdown and return it for persistence."""
    results = collect_segment_auc(model, df, categorical_levels, contract)
    for seg_name, values in results.items():
        if values["status"] == "skipped_one_class":
            print(f"  [{seg_name}]  n={values['n']:,}  AUC=SKIPPED (one class only)")
        elif values["status"] != "skipped_lt_100":
            print(f"  [{seg_name}]  n={values['n']:,}  AUC={values['auc']:.4f}")
    return results


_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def _threshold_grid(reporting_threshold: float) -> List[float]:
    """The review grid, always including the locked reporting threshold."""
    return sorted({*_THRESHOLDS, float(reporting_threshold)})


def print_threshold_table(
    model, df_valid, df_test, categorical_levels,
    contract: "str | FeatureContract | None" = None,
    threshold: float = REPORTING_THRESHOLD,
) -> None:
    """Print fail-class P/R/F1 at each threshold on valid (and test when supplied).

    Selection/decision is on valid only. ``df_test`` may be None, in which case
    TEST is not read at all. No threshold is auto-selected — the grid is for
    human review and always contains the locked reporting threshold.
    """
    resolved = resolve_feature_contract(contract)
    grid = _threshold_grid(threshold)
    print("\n=== M1 THRESHOLD TABLE (fail-class P/R/F1) ===")
    print("  Thresholds:", grid)
    print(f"  Locked reporting threshold: {threshold:g} (marked *)")
    if df_test is None:
        print("  TEST is CLOSED for this run and was not read.")
    else:
        print("  Decision on VALID only — TEST shown for consistency comparison.")

    splits = [("VALID", df_valid)]
    if df_test is not None:
        splits.append(("TEST ", df_test))
    for split_label, df in splits:
        X, y = prepare_X_y(df, "pass", categorical_levels, resolved)
        y_prob = model.predict(X)
        print(f"\n  [{split_label}]  thr    fail_P    fail_R   fail_F1")
        print(  "  " + "-" * 46)
        for thr in grid:
            y_bin = (y_prob >= thr).astype(int)
            fp_val = precision_score(y, y_bin, pos_label=0, zero_division=0)
            fr_val = recall_score(y, y_bin, pos_label=0, zero_division=0)
            ff_val = f1_score(y, y_bin, pos_label=0, zero_division=0)
            star = " *" if thr == float(threshold) else ""
            print(
                f"  [{split_label}]  {thr:g}   {fp_val:.4f}    {fr_val:.4f}   "
                f"{ff_val:.4f}{star}"
            )

    print("\n  (No threshold auto-selected; * marks the locked reporting threshold.)")
    print("=== END THRESHOLD TABLE ===\n")


def stratified_threshold_table(
    model, df_test, categorical_levels,
    contract: "str | FeatureContract | None" = None,
) -> None:
    """Fail-class P/R/F1 threshold table broken down by segments (test set, descriptive only).

    Uses the same segments as stratified_eval_pass and the same _THRESHOLDS list.
    Does NOT change the model's prediction threshold or apply per-segment thresholding.
    """
    print("\n=== M1 SEGMENTED THRESHOLD TABLE (test set, fail-class P/R/F1, descriptive only) ===")
    print("  Thresholds:", _THRESHOLDS)

    X_full, y_full = prepare_X_y(
        df_test, "pass", categorical_levels, resolve_feature_contract(contract)
    )
    y_prob_full = model.predict(X_full)

    segments = _segment_masks(df_test)

    for seg_name, mask in segments.items():
        sub = mask.values
        n = int(sub.sum())
        if n < 100:
            print(f"\n  [{seg_name}]  n={n:,}  SKIPPED (< 100 samples)")
            continue
        y_seg = y_full[sub]
        if y_seg.nunique() < 2:
            print(f"\n  [{seg_name}]  n={n:,}  SKIPPED (one class only)")
            continue
        y_prob_seg = y_prob_full[sub]

        print(f"\n  [{seg_name}]  n={n:,}")
        print(f"    thr    fail_P    fail_R   fail_F1")
        print(f"    " + "-" * 36)
        for thr in _THRESHOLDS:
            y_bin = (y_prob_seg >= thr).astype(int)
            fp_val = precision_score(y_seg, y_bin, pos_label=0, zero_division=0)
            fr_val = recall_score(y_seg, y_bin, pos_label=0, zero_division=0)
            ff_val = f1_score(y_seg, y_bin, pos_label=0, zero_division=0)
            print(f"    {thr:g}   {fp_val:.4f}    {fr_val:.4f}   {ff_val:.4f}")

    print("\n=== END SEGMENTED THRESHOLD TABLE ===\n")


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

def _save_feature_importance(model, path: Path) -> None:
    imp = pd.DataFrame({
        "feature": model.feature_name(),
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    imp.to_csv(path, index=False)
    print(f"  saved {path}")


def _git_state() -> Dict[str, Any]:
    """Best-effort Git commit + working-tree cleanliness for provenance."""
    def _run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True, text=True, timeout=15, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = _run("rev-parse", "HEAD")
    status = _run("status", "--porcelain")
    return {
        "commit": commit,
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "working_tree_clean": (status == "") if status is not None else None,
        "dirty_paths": (
            [line[3:] for line in status.splitlines()] if status else []
        ),
    }


def _file_record(path: "str | Path | None") -> Optional[Dict[str, Any]]:
    """Path + size + sha256 so a run can be tied to the exact input bytes."""
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "exists": False}
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {
        "path": str(path),
        "exists": True,
        "bytes": int(path.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def _dataset_version(train_path: "str | Path") -> Optional[str]:
    """Infer the versioned build id from the train path, when it is under versions/."""
    parts = Path(train_path).resolve().parts
    if "versions" in parts:
        index = parts.index("versions")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def build_run_contract(
    contract: FeatureContract,
    dtypes: Dict[str, str],
    levels: Dict[str, List[int]],
    *,
    lgbm_params: Dict[str, Any],
    train_path: "str | Path",
    valid_path: "str | Path",
    test_policy: str,
    created_at: str,
    effective_seed_settings: Optional[Dict[str, int]] = None,
    training_control: Optional[Dict[str, Any]] = None,
    diploma_gpa_handling: Optional[Dict[str, Any]] = None,
    data_rows: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Assemble the complete, self-describing contract persisted with every run."""
    return {
        "contract_name": contract.name,
        "contract_version": contract.version,
        "contract_description": contract.description,
        "feature_count": contract.expected_feature_count,
        "ordered_features": list(contract.features),
        "categorical_features": list(contract.categorical_features),
        "categorical_levels_learned_from": "train_only",
        "categorical_levels": levels,
        "unknown_category_code": UNKNOWN_CATEGORY,
        "dtypes_after_model_preparation": dtypes,
        "derived_feature_sources": dict(contract.derived_feature_sources),
        "dropped_feature_guard": list(DROPPED_FEATURES),
        "target_m1_classifier": TARGET_M1_DEFINITION,
        "target_m2_regressor": TARGET_M2_DEFINITION,
        "reporting_threshold": contract.reporting_threshold,
        "reporting_threshold_policy": (
            "locked constant; never optimized per run; applies to precision, "
            "recall, F1 and the confusion matrix only — not to training or "
            "early stopping"
        ),
        "random_seed": lgbm_params.get("seed"),
        "effective_seed_settings": effective_seed_settings,
        "lightgbm_params": lgbm_params,
        "training_control": training_control,
        "diploma_gpa_handling": diploma_gpa_handling,
        "data_rows": data_rows,
        "requires_concurrent_plan_context": contract.requires_concurrent_plan_context,
        "serving_limitation": SERVING_LIMITATION_NOTE,
        "test_policy": test_policy,
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "dataset_version": _dataset_version(train_path),
        "dataset_inputs": {
            "train": _file_record(train_path),
            "valid": _file_record(valid_path),
        },
        "created_at": created_at,
        "git": _git_state(),
    }


def _save_feature_contract(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  saved {path}")


def _save_training_curves(
    path: Path,
    m1_evals_result: dict,
    m2_evals_result: dict,
) -> None:
    """Write per-iteration metric history for M1 and M2 across train/valid/test sets."""
    # Dynamically resolve the metric key (LightGBM may use internal aliases, e.g. "l1" for mae).
    def _metric_key(evals: dict) -> str:
        return next(iter(next(iter(evals.values()))))

    m1_metric = _metric_key(m1_evals_result)
    m2_metric = _metric_key(m2_evals_result)

    curves = {
        "m1_pass_classifier": {
            "metric": m1_metric,
            "train": m1_evals_result["train"][m1_metric],
            "valid": m1_evals_result["valid"][m1_metric],
        },
        "m2_grade_regressor": {
            "metric": m2_metric,
            "train": m2_evals_result["train"][m2_metric],
            "valid": m2_evals_result["valid"][m2_metric],
        },
    }
    path.write_text(json.dumps(curves, indent=2))
    print(f"\n=== TRAINING CURVES SAVED ===")
    print(f"  path    : {path}")
    print(f"  M1 metric : {m1_metric}  rounds: {len(curves['m1_pass_classifier']['valid'])}")
    print(f"  M2 metric : {m2_metric}  rounds: {len(curves['m2_grade_regressor']['valid'])}")
    print(f"=== END TRAINING CURVES ===\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _read_existing_split(
    path: str | Path,
    contract: "str | FeatureContract | None" = None,
) -> pd.DataFrame:
    """Read an existing model-ready split without invoking any data builder.

    Only the selected contract's columns are read, so a baseline_41 run leaves
    every concurrent column on disk.
    """
    resolved = resolve_feature_contract(contract)
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model-ready split not found: {path}. "
            "Training never rebuilds data; run the data pipeline explicitly first."
        )
    return pd.read_parquet(path, columns=resolved.training_data_columns)


def main(argv: List[str] | None = None) -> None:
    # Defaults point at the final-generation splits (bucketing output), so the
    # common case needs no arguments. argv=None reads sys.argv (CLI); pass an
    # explicit list (e.g. []) when calling from a notebook so Jupyter's own
    # kernel arguments are not parsed.
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(model_split_path("train", "final")))
    parser.add_argument("--valid", default=str(model_split_path("valid", "final")))
    parser.add_argument("--test", default=str(model_split_path("test", "final")))
    parser.add_argument("--out", default=str(MODELS_DIR))
    parser.add_argument("--run-name", help="Persistent readable experiment name (creates a timestamped run).")
    parser.add_argument("--note", default="", help="One-line description used in the run report and leaderboard.")
    parser.add_argument("--compare-to", help="Persistent baseline run ID used for validation-metric deltas.")
    parser.add_argument(
        "--feature-contract",
        choices=sorted(FEATURE_CONTRACTS),
        help=(
            "Named model-input contract. REQUIRED for persistent runs "
            "(--run-name); quick runs fall back to "
            f"{DEFAULT_FEATURE_CONTRACT!r}."
        ),
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help=(
            "Read and score df_test. OFF by default: during contract "
            "comparison TEST stays closed and is never read."
        ),
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="LightGBM worker threads (default: 4; raise for speed if RAM/CPU allow).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_SHARED_PARAMS["seed"],
        help=(
            "Master LightGBM random seed. LightGBM deterministically derives "
            "data, feature-fraction, bagging, and drop seeds from this value."
        ),
    )
    # Regularization levers. Explicit typed flags, never a free-form params
    # string, so every deviation from _SHARED_PARAMS stays auditable. Defaults
    # are read from _SHARED_PARAMS, so omitting them reproduces current
    # behaviour exactly. _SHARED_PARAMS is shared by M1 and M2: each flag
    # changes BOTH models.
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=_SHARED_PARAMS["num_leaves"],
        help=(
            "LightGBM num_leaves for M1 and M2 "
            f"(default: {_SHARED_PARAMS['num_leaves']}). Lower = stronger "
            "regularization."
        ),
    )
    parser.add_argument(
        "--min-child-samples",
        type=int,
        default=_SHARED_PARAMS["min_child_samples"],
        help=(
            "LightGBM min_child_samples for M1 and M2 "
            f"(default: {_SHARED_PARAMS['min_child_samples']}). Higher = "
            "stronger regularization."
        ),
    )
    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=_SHARED_PARAMS["reg_lambda"],
        help=(
            "LightGBM reg_lambda (L2) for M1 and M2 "
            f"(default: {_SHARED_PARAMS['reg_lambda']}). Higher = stronger "
            "regularization."
        ),
    )
    args = parser.parse_args(argv)
    if args.num_threads < 1:
        parser.error("--num-threads must be at least 1")
    if args.num_leaves < 2:
        parser.error("--num-leaves must be at least 2")
    if args.min_child_samples < 1:
        parser.error("--min-child-samples must be at least 1")
    if args.reg_lambda < 0:
        parser.error("--reg-lambda must not be negative")
    if args.run_name and not args.feature_contract:
        parser.error(
            "--feature-contract is required for persistent runs (--run-name). "
            f"Choose one of: {sorted(FEATURE_CONTRACTS)}. A persistent run is "
            "never silently defaulted to a contract."
        )

    contract = resolve_feature_contract(
        args.feature_contract or DEFAULT_FEATURE_CONTRACT
    )
    reporting_threshold = contract.reporting_threshold
    test_policy = (
        "evaluated_descriptive_only" if args.evaluate_test else "closed_not_read"
    )
    print(f"Feature contract : {contract.name} ({contract.expected_feature_count} features)")
    print(f"Reporting threshold: {reporting_threshold:g} (locked, not optimized)")
    print(f"TEST policy      : {test_policy}")

    run_context = experiment_tracking.resolve_output(
        Path(args.out), args.run_name, args.note, args.compare_to
    )
    baseline_metrics = experiment_tracking.load_baseline_metrics(run_context)
    out_dir = run_context.output_dir
    print(f"Artifacts will be written to: {out_dir}")

    print("Loading data …")
    print("  Existing splits only; this module never invokes a data builder.")
    df_train = _read_existing_split(args.train, contract)
    df_valid = _read_existing_split(args.valid, contract)
    df_test = _read_existing_split(args.test, contract) if args.evaluate_test else None
    print(
        f"  train {df_train.shape}  valid {df_valid.shape}  "
        f"test {df_test.shape if df_test is not None else 'CLOSED (not read)'}"
    )
    print(
        f"  loaded {len(contract.training_data_columns)} required columns for "
        f"contract {contract.name!r}; every other column stayed on disk"
    )

    # TEMPORARY: remove after rebuilding final splits from 02_merge_diploma.
    # Fit once on train only; valid/test must reuse the same frozen value.
    diploma_gpa_median = float(df_train["diploma_gpa"].median())
    diploma_gpa_nulls: Dict[str, Dict[str, int]] = {}
    for split_name, df_split in (
        ("train", df_train),
        ("valid", df_valid),
        *((("test", df_test),) if df_test is not None else ()),
    ):
        missing_before = int(df_split["diploma_gpa"].isna().sum())
        df_split["diploma_gpa"] = df_split["diploma_gpa"].fillna(diploma_gpa_median)
        missing_after = int(df_split["diploma_gpa"].isna().sum())
        diploma_gpa_nulls[split_name] = {
            "missing_before": missing_before,
            "missing_after": missing_after,
        }
        assert missing_after == 0, f"{split_name}: diploma_gpa still contains missing values"
        print(
            f"  [temporary diploma_gpa fill] {split_name}: "
            f"{missing_before:,} -> {missing_after:,} nulls; train median={diploma_gpa_median}"
        )

    # Learn categorical levels from TRAIN only, then run loud diagnostics.
    print("\nLearning categorical levels (train only) …")
    categorical_levels = learn_categorical_levels(df_train, contract)
    dtypes = run_pre_training_diagnostics(df_train, categorical_levels, contract)
    X_flags, y_flags = prepare_X_y(df_train, "pass", categorical_levels, contract)
    constant_features = list(X_flags.nunique(dropna=False)[lambda s: s <= 1].index)
    del X_flags, y_flags
    gc.collect()
    important_flags = []
    if constant_features:
        important_flags.append("Train constant features: " + ", ".join(constant_features))
    if "diploma_gpa" in contract.features:
        important_flags.append(
            "diploma_gpa has no dedicated missing-value indicator; source-level median fill remains unchanged and unmatched diploma records may be null."
        )

    # Leakage gate on every split that is open BEFORE training.
    gate_splits = [("train", df_train), ("valid", df_valid)]
    if df_test is not None:
        gate_splits.append(("test", df_test))
    for name, d in gate_splits:
        X_chk, y_chk = prepare_X_y(d, "pass", categorical_levels, contract)
        assert_no_leakage_columns(X_chk)
        assert list(X_chk.columns) == list(contract.features), (
            f"{name}: X columns do not match contract {contract.name!r}"
        )
        print(f"Leakage gate passed on {name} ({X_chk.shape[1]} features).")
        del X_chk, y_chk
    gc.collect()

    # --- M1: pass/fail classifier ---
    print("\n=== Training M1 (pass/fail classifier) ===")
    training_params = {
        "num_threads": args.num_threads,
        "seed": args.seed,
        "num_leaves": args.num_leaves,
        "min_child_samples": args.min_child_samples,
        "reg_lambda": args.reg_lambda,
    }
    tuned_off_default = {
        name: {"default": _SHARED_PARAMS[name], "this_run": training_params[name]}
        for name in TUNABLE_PARAM_NAMES
        if training_params[name] != _SHARED_PARAMS[name]
    }
    print(
        "LightGBM levers off default: "
        + (json.dumps(tuned_off_default) if tuned_off_default else "none (defaults)")
    )
    pass_model, m1_evals = train_pass_model(
        df_train, df_valid, categorical_levels, params=training_params,
        num_boost_round=NUM_BOOST_ROUND,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        contract=contract,
    )
    eval_kwargs = {"contract": contract, "threshold": reporting_threshold}
    m1_train = evaluate_pass(pass_model, df_train, categorical_levels, "train", **eval_kwargs)
    m1_valid = evaluate_pass(pass_model, df_valid, categorical_levels, "valid", **eval_kwargs)
    m1_valid["train_valid_auc_gap"] = float(m1_train["auc"] - m1_valid["auc"])
    m1_valid["best_iteration"] = int(pass_model.best_iteration or 0)
    m1_valid_segments = collect_segment_auc(
        pass_model, df_valid, categorical_levels, contract
    )
    print("M1 stratified breakdown (VALID):")
    stratified_eval_pass(pass_model, df_valid, categorical_levels, contract)
    m1_test = None
    if df_test is not None:
        m1_test = evaluate_pass(
            pass_model, df_test, categorical_levels, "test (descriptive)", **eval_kwargs
        )
        stratified_threshold_table(pass_model, df_test, categorical_levels, contract)
    print_threshold_table(
        pass_model, df_valid, df_test, categorical_levels,
        contract=contract, threshold=reporting_threshold,
    )
    pass_model.save_model(str(out_dir / "m1_pass_model.lgbm"))
    _save_feature_importance(pass_model, out_dir / "m1_feature_importance.csv")

    # --- M2: final_mark regressor ---
    print("\n=== Training M2 (final_mark regressor) ===")
    grade_model, m2_evals = train_grade_model(
        df_train, df_valid, categorical_levels, params=training_params,
        num_boost_round=NUM_BOOST_ROUND,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        contract=contract,
    )
    m2_train = evaluate_grade(grade_model, df_train, categorical_levels, "train", contract)
    m2_valid = evaluate_grade(grade_model, df_valid, categorical_levels, "valid", contract)
    m2_valid["best_iteration"] = int(grade_model.best_iteration or 0)
    m2_test = (
        evaluate_grade(
            grade_model, df_test, categorical_levels, "test (descriptive)", contract
        )
        if df_test is not None
        else None
    )
    grade_model.save_model(str(out_dir / "m2_grade_model.lgbm"))
    _save_feature_importance(grade_model, out_dir / "m2_feature_importance.csv")

    # --- Artifacts ---
    m1_effective_seeds = _effective_seed_settings(out_dir / "m1_pass_model.lgbm")
    m2_effective_seeds = _effective_seed_settings(out_dir / "m2_grade_model.lgbm")
    assert m1_effective_seeds == m2_effective_seeds, (
        "M1 and M2 received different effective LightGBM seed settings"
    )
    # The COMPLETE effective parameter dict per model, verified against each
    # trained booster's own record so a future comparison can check equality
    # programmatically instead of trusting a report's prose.
    m1_effective_params = effective_lgbm_params(M1_OBJECTIVE, M1_METRIC, training_params)
    m2_effective_params = effective_lgbm_params(M2_OBJECTIVE, M2_METRIC, training_params)
    _assert_booster_matches_params(pass_model, m1_effective_params, "M1")
    _assert_booster_matches_params(grade_model, m2_effective_params, "M2")
    lightgbm_params_by_model = {
        "m1_pass_classifier": m1_effective_params,
        "m2_grade_regressor": m2_effective_params,
        # Objective/metric are the only intended M1/M2 difference; anything
        # else differing between the two dicts is a defect.
        "differing_keys": sorted(
            key
            for key in set(m1_effective_params) | set(m2_effective_params)
            if m1_effective_params.get(key) != m2_effective_params.get(key)
        ),
        "tuned_off_default": tuned_off_default,
    }
    assert lightgbm_params_by_model["differing_keys"] == ["metric", "objective"], (
        "M1 and M2 differ on more than objective/metric: "
        f"{lightgbm_params_by_model['differing_keys']}"
    )
    training_control = {
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "early_stopping_selection_split": "valid_only",
    }
    diploma_gpa_handling = {
        "method": "train_median_fill",
        "learned_from": "train_only",
        "fill_value": diploma_gpa_median,
        "null_counts": diploma_gpa_nulls,
    }
    data_rows = {
        "train": int(len(df_train)),
        "valid": int(len(df_valid)),
    }
    run_contract = build_run_contract(
        contract,
        dtypes,
        categorical_levels,
        lgbm_params={**_SHARED_PARAMS, **training_params},
        train_path=args.train,
        valid_path=args.valid,
        test_policy=test_policy,
        created_at=run_context.created_at.isoformat(timespec="seconds"),
        effective_seed_settings=m1_effective_seeds,
        training_control=training_control,
        diploma_gpa_handling=diploma_gpa_handling,
        data_rows=data_rows,
    )
    _save_feature_contract(out_dir / "feature_contract.json", run_contract)
    _save_training_curves(out_dir / "training_curves.json", m1_evals, m2_evals)
    results = {
        "m1_pass_classifier": {"train": m1_train, "valid": m1_valid, "test": m1_test},
        "m2_grade_regressor": {"train": m2_train, "valid": m2_valid, "test": m2_test},
    }
    selection_summary = {
        "selection_split": "valid",
        "feature_contract": contract.name,
        "feature_count": contract.expected_feature_count,
        "reporting_threshold": reporting_threshold,
        "test_policy": test_policy,
        "directions": {
            "fail_avg_precision": "higher_is_better",
            "auc": "higher_is_better",
            "brier": "lower_is_better",
            "train_valid_auc_gap": "lower_is_better",
        },
        "valid": {
            "fail_avg_precision": m1_valid["fail_avg_precision"],
            "auc": m1_valid["auc"],
            "brier": m1_valid["brier"],
            "train_valid_auc_gap": m1_valid["train_valid_auc_gap"],
        },
        "test_descriptive": (
            {
                "fail_avg_precision": m1_test["fail_avg_precision"],
                "auc": m1_test["auc"],
                "brier": m1_test["brier"],
            }
            if m1_test is not None
            else "closed_not_read"
        ),
    }
    (out_dir / "selection_summary.json").write_text(
        json.dumps(selection_summary, indent=2), encoding="utf-8"
    )
    segment_metrics = {"valid": m1_valid_segments}
    # Recorded inside metrics.json so a comparison can never silently mix runs
    # made at different thresholds or under different contracts.
    run_settings = {
        "feature_contract": contract.name,
        "feature_count": contract.expected_feature_count,
        "reporting_threshold": reporting_threshold,
        "test_policy": test_policy,
        "random_seed": args.seed,
        "effective_seed_settings": m1_effective_seeds,
        "num_threads": args.num_threads,
        "lightgbm_params": lightgbm_params_by_model,
        "training_control": training_control,
        "diploma_gpa_handling": diploma_gpa_handling,
        "data_rows": data_rows,
        "train_path": str(args.train),
        "valid_path": str(args.valid),
        "dataset_version": _dataset_version(args.train),
    }
    if run_context.persistent:
        experiment_tracking.finalize_persistent_run(
            run_context,
            results,
            segment_metrics,
            contract.expected_feature_count,
            important_flags,
            baseline_metrics,
            run_settings=run_settings,
        )
    else:
        (out_dir / "metrics.json").write_text(
            json.dumps(
                {**results, "segments": segment_metrics, "run_settings": run_settings},
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"\nMetrics saved to {out_dir / 'metrics.json'}")
    print(json.dumps({**results, "segments": segment_metrics}, indent=2))


if __name__ == "__main__":
    main()
