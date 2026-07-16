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

All arguments default to the canonical final-generation splits
(MODEL_DATA_DIR/df_{train,valid,test}_final.parquet, written by
03_diploma_type_bucketing) and MODELS_DIR. Pass --train/--valid/--test/--out
only to override. From a notebook, call main([]) so Jupyter's own argv is
not parsed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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
from src.paths import MODEL_DATA_DIR, MODELS_DIR

# ---------------------------------------------------------------------------
# Feature contract — V1 LOCKED ALLOWLIST (39 features)
# ---------------------------------------------------------------------------
# This is an explicit allowlist. It is NOT derived from fe.FEATURE_COLUMNS,
# because that list contains diagnostic / string / leakage columns and is not a
# model allowlist.

MODEL_FEATURES: List[str] = [
    # --- Student GPA history ---
    "prev_gpa_points_clean",                  # zero + flags design (NOT model_prev_gpa)
    "start_agpa_points",                      # cumulative AGPA at semester start
    "last_valid_gpa_before_current_semester", # last real GPA, skips interruptions
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
EXPECTED_FEATURE_COUNT = 39


# ---------------------------------------------------------------------------
# Categorical level handling (learn from train ONLY, apply to all splits)
# ---------------------------------------------------------------------------

def learn_categorical_levels(df_train: pd.DataFrame) -> Dict[str, List[int]]:
    """Learn the allowed category set for each categorical feature from TRAIN only."""
    levels: Dict[str, List[int]] = {}
    for col in CATEGORICAL_FEATURES:
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
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) ready for LightGBM.

    target='pass'  -> M1 classifier label (final_mark >= 50) as int
    target='grade' -> M2 regressor label  final_mark (float)

    categorical_levels MUST be learned from df_train and passed in for every
    split, so valid/test never define their own category set.
    """
    if categorical_levels is None:
        raise ValueError("categorical_levels is required (learn from df_train first).")

    df = df.copy()
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
    df["requirement_size_bucket_ord"] = (
        raw_bucket.map(REQUIREMENT_BUCKET_ORD).fillna(0).astype(int)
    )

    # --- categorical handling (NaN + unseen -> -1, explicit category) ---
    df = _apply_categorical_levels(df, categorical_levels)

    # --- presence check ---
    missing_cols = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected feature columns: {missing_cols}")

    X = df[MODEL_FEATURES].copy()

    # --- dtype casting: numeric -> float64; categorical stays 'category' ---
    numeric_feats = [c for c in MODEL_FEATURES if c not in CATEGORICAL_FEATURES]
    X[numeric_feats] = X[numeric_feats].astype("float64")

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
) -> Dict[str, str]:
    """Print the feature contract checks and return {feature: dtype} after casting."""
    print("\n=== PRE-TRAINING DIAGNOSTICS ===")

    # 1) final feature list + count
    print(f"\n[1] MODEL_FEATURES count = {len(MODEL_FEATURES)} (expected {EXPECTED_FEATURE_COUNT})")
    for i, c in enumerate(MODEL_FEATURES, 1):
        print(f"    {i:>2}. {c}")
    assert len(MODEL_FEATURES) == EXPECTED_FEATURE_COUNT, (
        f"Feature count {len(MODEL_FEATURES)} != {EXPECTED_FEATURE_COUNT}"
    )
    assert len(set(MODEL_FEATURES)) == len(MODEL_FEATURES), "Duplicate feature in MODEL_FEATURES"

    # 2) missing features (fail loudly) — uses prepare_X_y so derived cols count
    X_train, _ = prepare_X_y(df_train, "pass", categorical_levels)

    # 3) dropped-feature guard
    still_present = [c for c in DROPPED_FEATURES if c in X_train.columns]
    print(f"\n[2] Dropped features present in X (must be empty): {still_present}")
    assert not still_present, f"Forbidden features in X: {still_present}"

    # 4) unused parquet columns (informational)
    used = set(MODEL_FEATURES) | {TARGET_GRADE}
    unused = sorted(c for c in df_train.columns if c not in used)
    print(f"\n[3] Parquet columns NOT used by the model ({len(unused)}):")
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
    dtypes = {c: str(X_train[c].dtype) for c in MODEL_FEATURES}
    print("\n[6] Feature dtypes after casting:")
    for c in MODEL_FEATURES:
        print(f"    {c}: {dtypes[c]}")

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
    "n_jobs": -1,
    "verbose": -1,
    "seed": 42,
}


def _make_datasets(X_tr, y_tr, X_va, y_va):
    dtrain = lgb.Dataset(
        X_tr, label=y_tr, free_raw_data=False, categorical_feature=CATEGORICAL_FEATURES
    )
    dvalid = lgb.Dataset(
        X_va, label=y_va, reference=dtrain, free_raw_data=False,
        categorical_feature=CATEGORICAL_FEATURES,
    )
    return dtrain, dvalid


def train_pass_model(  # M1 classifier
    df_train, df_valid, categorical_levels,
    params=None, num_boost_round=2000, early_stopping_rounds=50,
) -> Tuple[lgb.Booster, dict]:
    """M1 — LightGBM binary classifier for pass/fail. NO scale_pos_weight in V1."""
    X_tr, y_tr = prepare_X_y(df_train, "pass", categorical_levels)
    X_va, y_va = prepare_X_y(df_valid, "pass", categorical_levels)

    # Class balance is printed for information only — NOT used to weight the model.
    pass_rate = float(y_tr.mean())
    print(f"M1 class balance — train pass rate: {pass_rate:.4f} "
          f"(pass={int(y_tr.sum()):,}  fail={int(len(y_tr) - y_tr.sum()):,})")

    default_params = {"objective": "binary", "metric": "auc", **_SHARED_PARAMS}
    if params:
        default_params.update(params)

    dtrain, dvalid = _make_datasets(X_tr, y_tr, X_va, y_va)
    evals_result: dict = {}
    # Only validation may select the stopping iteration. Test is evaluated
    # after fitting and is never passed to LightGBM during training.
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
    params=None, num_boost_round=2000, early_stopping_rounds=50,
) -> Tuple[lgb.Booster, dict]:
    """M2 — LightGBM regressor for final_mark (0-100), MAE loss."""
    X_tr, y_tr = prepare_X_y(df_train, "grade", categorical_levels)
    X_va, y_va = prepare_X_y(df_valid, "grade", categorical_levels)

    default_params = {"objective": "regression_l1", "metric": "mae", **_SHARED_PARAMS}
    if params:
        default_params.update(params)

    dtrain, dvalid = _make_datasets(X_tr, y_tr, X_va, y_va)
    evals_result: dict = {}
    # Only validation may select the stopping iteration. Test is evaluated
    # after fitting and is never passed to LightGBM during training.
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

def evaluate_pass(model, df, categorical_levels, label="") -> dict:  # M1
    """M1 metrics, including fail-class AP used for experiment selection."""
    X, y = prepare_X_y(df, "pass", categorical_levels)
    y_prob = model.predict(X)
    y_bin = (y_prob >= 0.5).astype(int)

    # labels=[0,1] -> row 0 = actual fail, row 1 = actual pass
    cm = confusion_matrix(y, y_bin, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    metrics = {
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
              f"R={metrics['fail_recall']:.4f}  F1={metrics['fail_f1']:.4f}")
        print(f"[{label}] M1 confusion matrix (rows=actual, cols=predicted):")
        print(f"              Pred-FAIL  Pred-PASS")
        print(f"  Act-FAIL    {tn:>9,}  {fp:>9,}")
        print(f"  Act-PASS    {fn:>9,}  {tp:>9,}")
    return metrics


def evaluate_grade(model, df, categorical_levels, label="") -> dict:  # M2
    """M2 metrics: MAE, RMSE, R²."""
    X, y = prepare_X_y(df, "grade", categorical_levels)
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


def collect_segment_auc(model, df, categorical_levels) -> Dict[str, dict]:
    """Collect the existing M1 segment AUCs without changing their calculation."""
    X_full, y_full = prepare_X_y(df, "pass", categorical_levels)
    y_prob_full = model.predict(X_full)
    segments = {
        "first_semester": df["is_first_active_semester"] == 1,
        "retake_attempt": df["attempt_number"] > 1,
        "low_difficulty_support": df["difficulty_fallback_level"] >= 3,
        "cold_start_gpa": df["no_previous_progress"] == 1,
    }
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
        results[seg_name] = {"n": n, "auc": round(float(auc), 4), "status": "ok"}
    return results


def stratified_eval_pass(model, df, categorical_levels) -> Dict[str, dict]:
    """Print the existing M1 AUC segment breakdown and return it for persistence."""
    results = collect_segment_auc(model, df, categorical_levels)
    for seg_name, values in results.items():
        if values["status"] == "skipped_one_class":
            print(f"  [{seg_name}]  n={values['n']:,}  AUC=SKIPPED (one class only)")
        elif values["status"] != "skipped_lt_100":
            print(f"  [{seg_name}]  n={values['n']:,}  AUC={values['auc']:.4f}")
    return results


_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def print_threshold_table(model, df_valid, df_test, categorical_levels) -> None:
    """Print fail-class P/R/F1 at each threshold on valid and test.

    Selection/decision is on valid only; test is shown for consistency comparison.
    No threshold is auto-selected — the table is for human review.
    """
    print("\n=== M1 THRESHOLD TABLE (fail-class P/R/F1) ===")
    print("  Thresholds:", _THRESHOLDS)
    print("  Decision on VALID only — TEST shown for consistency comparison.")

    for split_label, df in (("VALID", df_valid), ("TEST ", df_test)):
        X, y = prepare_X_y(df, "pass", categorical_levels)
        y_prob = model.predict(X)
        print(f"\n  [{split_label}]  thr    fail_P    fail_R   fail_F1")
        print(  "  " + "-" * 46)
        for thr in _THRESHOLDS:
            y_bin = (y_prob >= thr).astype(int)
            fp_val = precision_score(y, y_bin, pos_label=0, zero_division=0)
            fr_val = recall_score(y, y_bin, pos_label=0, zero_division=0)
            ff_val = f1_score(y, y_bin, pos_label=0, zero_division=0)
            print(f"  [{split_label}]  {thr:.1f}   {fp_val:.4f}    {fr_val:.4f}   {ff_val:.4f}")

    print("\n  (No threshold auto-selected — choose after reviewing the table above.)")
    print("=== END THRESHOLD TABLE ===\n")


def stratified_threshold_table(model, df_test, categorical_levels) -> None:
    """Fail-class P/R/F1 threshold table broken down by segments (test set, descriptive only).

    Uses the same segments as stratified_eval_pass and the same _THRESHOLDS list.
    Does NOT change the model's prediction threshold or apply per-segment thresholding.
    """
    print("\n=== M1 SEGMENTED THRESHOLD TABLE (test set, fail-class P/R/F1, descriptive only) ===")
    print("  Thresholds:", _THRESHOLDS)

    X_full, y_full = prepare_X_y(df_test, "pass", categorical_levels)
    y_prob_full = model.predict(X_full)

    segments = {
        "first_semester": df_test["is_first_active_semester"] == 1,
        "retake_attempt": df_test["attempt_number"] > 1,
        "low_difficulty_support": df_test["difficulty_fallback_level"] >= 3,
        "cold_start_gpa": df_test["no_previous_progress"] == 1,
    }

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
            print(f"    {thr:.1f}   {fp_val:.4f}    {fr_val:.4f}   {ff_val:.4f}")

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


def _save_feature_contract(path: Path, dtypes: Dict[str, str], levels: Dict[str, List[int]]) -> None:
    contract = {
        "version": "v1",
        "n_features": len(MODEL_FEATURES),
        "features": MODEL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_levels": levels,
        "unknown_category_code": UNKNOWN_CATEGORY,
        "dropped_features": DROPPED_FEATURES,
        "dtypes": dtypes,
        "target_m1_classifier": "(final_mark >= 50).astype(int)",
        "target_m2_regressor": "final_mark",
    }
    path.write_text(json.dumps(contract, indent=2))
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

def main(argv: List[str] | None = None) -> None:
    # Defaults point at the final-generation splits (bucketing output), so the
    # common case needs no arguments. argv=None reads sys.argv (CLI); pass an
    # explicit list (e.g. []) when calling from a notebook so Jupyter's own
    # kernel arguments are not parsed.
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(MODEL_DATA_DIR / "df_train_final.parquet"))
    parser.add_argument("--valid", default=str(MODEL_DATA_DIR / "df_valid_final.parquet"))
    parser.add_argument("--test", default=str(MODEL_DATA_DIR / "df_test_final.parquet"))
    parser.add_argument("--out", default=str(MODELS_DIR))
    parser.add_argument("--run-name", help="Persistent readable experiment name (creates a timestamped run).")
    parser.add_argument("--note", default="", help="One-line description used in the run report and leaderboard.")
    parser.add_argument("--compare-to", help="Persistent baseline run ID used for validation-metric deltas.")
    args = parser.parse_args(argv)

    run_context = experiment_tracking.resolve_output(
        Path(args.out), args.run_name, args.note, args.compare_to
    )
    baseline_metrics = experiment_tracking.load_baseline_metrics(run_context)
    out_dir = run_context.output_dir
    print(f"Artifacts will be written to: {out_dir}")

    print("Loading data …")
    df_train = pd.read_parquet(args.train)
    df_valid = pd.read_parquet(args.valid)
    df_test = pd.read_parquet(args.test)
    print(f"  train {df_train.shape}  valid {df_valid.shape}  test {df_test.shape}")

    # TEMPORARY: remove after rebuilding final splits from 02_merge_diploma.
    # Fit once on train only; valid/test must reuse the same frozen value.
    diploma_gpa_median = df_train["diploma_gpa"].median()
    for split_name, df_split in (
        ("train", df_train),
        ("valid", df_valid),
        ("test", df_test),
    ):
        missing_before = int(df_split["diploma_gpa"].isna().sum())
        df_split["diploma_gpa"] = df_split["diploma_gpa"].fillna(diploma_gpa_median)
        missing_after = int(df_split["diploma_gpa"].isna().sum())
        assert missing_after == 0, f"{split_name}: diploma_gpa still contains missing values"
        print(
            f"  [temporary diploma_gpa fill] {split_name}: "
            f"{missing_before:,} -> {missing_after:,} nulls; train median={diploma_gpa_median}"
        )

    # Learn categorical levels from TRAIN only, then run loud diagnostics.
    print("\nLearning categorical levels (train only) …")
    categorical_levels = learn_categorical_levels(df_train)
    dtypes = run_pre_training_diagnostics(df_train, categorical_levels)
    X_flags, _ = prepare_X_y(df_train, "pass", categorical_levels)
    constant_features = list(X_flags.nunique(dropna=False)[lambda s: s <= 1].index)
    important_flags = []
    if constant_features:
        important_flags.append("Train constant features: " + ", ".join(constant_features))
    if "diploma_gpa" in MODEL_FEATURES:
        important_flags.append(
            "diploma_gpa has no dedicated missing-value indicator; source-level median fill remains unchanged and unmatched diploma records may be null."
        )

    # Leakage gate on all three splits BEFORE training.
    for name, d in (("train", df_train), ("valid", df_valid), ("test", df_test)):
        X_chk, _ = prepare_X_y(d, "pass", categorical_levels)
        assert_no_leakage_columns(X_chk)
        print(f"Leakage gate passed on {name} ({X_chk.shape[1]} features).")

    # --- M1: pass/fail classifier ---
    print("\n=== Training M1 (pass/fail classifier) ===")
    pass_model, m1_evals = train_pass_model(df_train, df_valid, categorical_levels)
    m1_train = evaluate_pass(pass_model, df_train, categorical_levels, "train")
    m1_valid = evaluate_pass(pass_model, df_valid, categorical_levels, "valid")
    m1_valid["train_valid_auc_gap"] = float(m1_train["auc"] - m1_valid["auc"])
    m1_test = evaluate_pass(pass_model, df_test, categorical_levels, "test (descriptive)")
    m1_valid_segments = collect_segment_auc(pass_model, df_valid, categorical_levels)
    print("M1 stratified breakdown (test set, descriptive only):")
    stratified_eval_pass(pass_model, df_test, categorical_levels)
    print_threshold_table(pass_model, df_valid, df_test, categorical_levels)
    stratified_threshold_table(pass_model, df_test, categorical_levels)
    pass_model.save_model(str(out_dir / "m1_pass_model.lgbm"))
    _save_feature_importance(pass_model, out_dir / "m1_feature_importance.csv")

    # --- M2: final_mark regressor ---
    print("\n=== Training M2 (final_mark regressor) ===")
    grade_model, m2_evals = train_grade_model(df_train, df_valid, categorical_levels)
    m2_train = evaluate_grade(grade_model, df_train, categorical_levels, "train")
    m2_valid = evaluate_grade(grade_model, df_valid, categorical_levels, "valid")
    m2_test = evaluate_grade(grade_model, df_test, categorical_levels, "test (descriptive)")
    grade_model.save_model(str(out_dir / "m2_grade_model.lgbm"))
    _save_feature_importance(grade_model, out_dir / "m2_feature_importance.csv")

    # --- Artifacts ---
    _save_feature_contract(out_dir / "feature_contract.json", dtypes, categorical_levels)
    _save_training_curves(out_dir / "training_curves.json", m1_evals, m2_evals)
    results = {
        "m1_pass_classifier": {"train": m1_train, "valid": m1_valid, "test": m1_test},
        "m2_grade_regressor": {"train": m2_train, "valid": m2_valid, "test": m2_test},
    }
    selection_summary = {
        "selection_split": "valid",
        "test_policy": "descriptive_only; never supplied to fitting or early stopping",
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
        "test_descriptive": {
            "fail_avg_precision": m1_test["fail_avg_precision"],
            "auc": m1_test["auc"],
            "brier": m1_test["brier"],
        },
    }
    (out_dir / "selection_summary.json").write_text(
        json.dumps(selection_summary, indent=2), encoding="utf-8"
    )
    segment_metrics = {"valid": m1_valid_segments}
    if run_context.persistent:
        experiment_tracking.finalize_persistent_run(
            run_context,
            results,
            segment_metrics,
            len(MODEL_FEATURES),
            important_flags,
            baseline_metrics,
        )
    else:
        (out_dir / "metrics.json").write_text(
            json.dumps({**results, "segments": segment_metrics}, indent=2), encoding="utf-8"
        )
    print(f"\nMetrics saved to {out_dir / 'metrics.json'}")
    print(json.dumps({**results, "segments": segment_metrics}, indent=2))


if __name__ == "__main__":
    main()
