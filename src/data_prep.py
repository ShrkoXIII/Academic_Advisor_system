"""Split reading, categorical levels, X/y construction and pre-training diagnostics."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.feature_contracts import (
    CONCURRENT_MODEL_FEATURES,
    DROPPED_FEATURES,
    REQUIREMENT_BUCKET_ORD,
    TARGET_GRADE,
    UNKNOWN_CATEGORY,
    FeatureContract,
    _LEFTOVER_KEY_PATTERNS,
    resolve_feature_contract,
)
from src.feature_engineering import assert_no_leakage_columns


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
