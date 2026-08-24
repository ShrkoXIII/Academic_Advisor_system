"""Evaluate the fitted sklearn KNN as a standalone temporal VALID predictor.

The persisted classifiers/regressors are fitted from TRAIN only. This evaluator
reconstructs official semester outcomes for VALID and calls native sklearn
``predict``, ``predict_proba``, and regression ``predict`` through the advisor's
public batch API. It does not tune or read TEST.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.knn_advisor_v2 import KNNAdvisorV2Level  # noqa: E402
from src.knn_history_helpers import (  # noqa: E402
    GRADE_COLUMNS,
    LEVEL_STATUS_COLUMNS,
    STATUS_COLUMNS,
    TRAIN_COLUMNS,
    attach_official_references,
    build_student_semester_outcomes,
)
from src.paths import RAW_DIR, assert_data_root  # noqa: E402


DEFAULT_VALID = (
    PROJECT_ROOT
    / "data"
    / "model_data"
    / "versions"
    / "2026-08_temporal_rebuild_v2"
    / "05_dataset"
    / "valid_dataset_candidate.parquet"
)
DEFAULT_STATUS = RAW_DIR / "v_add_student_degree_status_v2.parquet"
DEFAULT_GRADES = RAW_DIR / "v_acs_grade.parquet"
DEFAULT_KNN = (
    PROJECT_ROOT
    / "data"
    / "artifacts"
    / "knn"
    / "2026-08-24_sklearn_v3"
    / "knn_v2_level_sklearn_k20.pkl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "knn"
    / "2026-08-24_knn_sklearn_v3_valid_k20"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-path", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--status-path", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--grades-path", type=Path, default=DEFAULT_GRADES)
    parser.add_argument("--knn-path", type=Path, default=DEFAULT_KNN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_float_metric(function, *args) -> float | None:
    try:
        value = float(function(*args))
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def _classification_metrics(frame: pd.DataFrame) -> dict:
    observed = frame["actual_any_course_failed"].astype(int)
    probability = frame["pred_any_course_failed"].astype(float)
    predicted = frame["predicted_any_course_failed"].astype(int)
    tn, fp, fn, tp = confusion_matrix(observed, predicted, labels=[0, 1]).ravel()
    return {
        "decision_rule": "native_KNeighborsClassifier.predict",
        "positive_class": "any_course_failed",
        "positive_rate": float(observed.mean()),
        "roc_auc": _safe_float_metric(roc_auc_score, observed, probability),
        "average_precision": _safe_float_metric(
            average_precision_score, observed, probability
        ),
        "brier": float(brier_score_loss(observed, probability)),
        "accuracy": float(accuracy_score(observed, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(observed, predicted)),
        "precision": float(precision_score(observed, predicted, zero_division=0)),
        "recall": float(recall_score(observed, predicted, zero_division=0)),
        "f1": float(f1_score(observed, predicted, zero_division=0)),
        "always_no_failure_accuracy": float(1.0 - observed.mean()),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def _regression_metrics(
    frame: pd.DataFrame,
    *,
    actual_column: str,
    prediction_column: str,
) -> dict:
    usable = frame[[actual_column, prediction_column]].dropna()
    observed = usable[actual_column].astype(float)
    predicted = usable[prediction_column].astype(float)
    return {
        "n": int(len(usable)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "r2": _safe_float_metric(r2_score, observed, predicted),
        "bias_pred_minus_observed": float((predicted - observed).mean()),
    }


def _metrics_for_slice(frame: pd.DataFrame, requested_k: int) -> dict:
    covered = frame.loc[frame["support"].gt(0)].copy()
    distances = covered["median_neighbour_gpa_distance"].dropna()
    result = {
        "query_rows": int(len(frame)),
        "covered_rows": int(len(covered)),
        "coverage": float(len(covered) / len(frame)) if len(frame) else 0.0,
        "full_k_rows": int(frame["support"].ge(requested_k).sum()),
        "full_k_coverage": float(frame["support"].ge(requested_k).mean())
        if len(frame)
        else 0.0,
        "mean_support": float(covered["support"].mean()) if len(covered) else 0.0,
        "median_neighbour_gpa_distance": float(distances.median())
        if len(distances)
        else None,
    }
    if covered.empty:
        result["failure_classification"] = None
        result["term_gpa_regression"] = None
        result["semester_average_mark_regression"] = None
        return result

    result["failure_classification"] = _classification_metrics(covered)
    result["term_gpa_regression"] = _regression_metrics(
        covered,
        actual_column="actual_term_gpa",
        prediction_column="pred_term_gpa",
    )
    result["semester_average_mark_regression"] = _regression_metrics(
        covered,
        actual_column="actual_semester_average_mark",
        prediction_column="pred_semester_average_mark",
    )
    return result


def _build_valid_outcomes(
    valid_path: Path,
    status_path: Path,
    grades_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = pd.read_parquet(valid_path, columns=TRAIN_COLUMNS)
    status = pd.read_parquet(
        status_path, columns=STATUS_COLUMNS + LEVEL_STATUS_COLUMNS
    )
    grades = pd.read_parquet(grades_path, columns=GRADE_COLUMNS)
    enriched = attach_official_references(valid, status, grades)
    outcomes = build_student_semester_outcomes(enriched)

    cumulative_gpa = pd.to_numeric(
        outcomes["cumulative_gpa_before"], errors="coerce"
    )
    diploma_gpa = pd.to_numeric(outcomes["diploma_gpa"], errors="coerce")
    returning = outcomes["is_first_active_semester"].eq(0) & cumulative_gpa.gt(0)
    cold_start = outcomes["is_first_active_semester"].eq(1) & diploma_gpa.notna()
    eligible = outcomes.loc[returning | cold_start].copy()
    eligible["evaluation_route"] = np.where(
        eligible["is_first_active_semester"].eq(1), "cold_start", "returning"
    )
    eligible["query_gpa"] = np.where(
        eligible["evaluation_route"].eq("cold_start"),
        pd.to_numeric(eligible["diploma_gpa"], errors="coerce"),
        pd.to_numeric(eligible["cumulative_gpa_before"], errors="coerce"),
    )
    return outcomes, eligible


def _evaluate_queries(
    advisor: KNNAdvisorV2Level,
    queries: pd.DataFrame,
) -> pd.DataFrame:
    """Call fitted sklearn estimators in degree-level batches."""
    model_queries = pd.DataFrame(
        {
            "degree_id": queries["degree_id"],
            "academic_level": queries["academic_level_before"],
            "gpa": queries["query_gpa"],
            "cold_start": queries["evaluation_route"].eq("cold_start"),
        },
        index=queries.index,
    )
    model_predictions = advisor.predict_frame(model_queries)
    predictions = pd.DataFrame(
        {
            "university_id": queries["university_id"],
            "student_id": queries["student_id"],
            "degree_id": queries["degree_id"],
            "part_id": queries["part_id"],
            "evaluation_route": queries["evaluation_route"],
            "academic_level_before": queries["academic_level_before"].astype(int),
            "query_gpa": queries["query_gpa"].astype(float),
            "support": model_predictions["support"].astype(int),
            "knn_route": model_predictions["knn_route"],
            "actual_any_course_failed": queries["any_course_failed"].astype(int),
            "predicted_any_course_failed": model_predictions[
                "predicted_any_course_failed"
            ],
            "pred_any_course_failed": model_predictions["failure_probability"],
            "actual_term_gpa": queries["term_gpa"].astype(float),
            "pred_term_gpa": model_predictions["predicted_term_gpa"],
            "actual_semester_average_mark": queries[
                "semester_average_mark"
            ].astype(float),
            "pred_semester_average_mark": model_predictions[
                "predicted_semester_average_mark"
            ],
        },
        index=queries.index,
    )
    predictions["median_neighbour_gpa_distance"] = np.nan
    return predictions.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    assert_data_root(
        args.valid_path,
        args.status_path,
        args.grades_path,
        args.knn_path,
    )
    if args.output_dir.exists():
        raise FileExistsError(f"Evaluation output already exists: {args.output_dir}")

    all_outcomes, queries = _build_valid_outcomes(
        args.valid_path, args.status_path, args.grades_path
    )
    print(
        f"VALID semester rows: {len(all_outcomes):,}; "
        f"KNN-queryable rows: {len(queries):,}",
        flush=True,
    )
    advisor = KNNAdvisorV2Level.load(args.knn_path)
    predictions = _evaluate_queries(advisor, queries)

    fitted_k = int(advisor.metadata["n_neighbors"])
    metrics = {
        "all": _metrics_for_slice(predictions, fitted_k),
        "returning": _metrics_for_slice(
            predictions.loc[predictions["evaluation_route"].eq("returning")],
            fitted_k,
        ),
        "cold_start": _metrics_for_slice(
            predictions.loc[predictions["evaluation_route"].eq("cold_start")],
            fitted_k,
        ),
    }

    report = {
        "evaluation": "knn_v2_level_sklearn_native_predict_temporal_valid",
        "selection_split": "valid_only",
        "test_read": False,
        "leakage_policy": (
            "KNN estimators are fitted on TRAIN-only snapshots; VALID outcomes are "
            "used only as evaluation targets. Native sklearn predict is reported, "
            "so a student's earlier TRAIN snapshot may be a neighbour in temporal VALID."
        ),
        "prediction_api": [
            "KNeighborsClassifier.predict",
            "KNeighborsClassifier.predict_proba",
            "KNeighborsRegressor.predict",
        ],
        "knn_metadata": advisor.metadata,
        "data": {
            "valid_semester_rows": int(len(all_outcomes)),
            "queryable_semester_rows": int(len(queries)),
            "excluded_unqueryable_state_rows": int(len(all_outcomes) - len(queries)),
            "returning_query_rows": int(
                queries["evaluation_route"].eq("returning").sum()
            ),
            "cold_start_query_rows": int(
                queries["evaluation_route"].eq("cold_start").sum()
            ),
        },
        "inputs": {
            "valid_path": str(args.valid_path),
            "valid_sha256": _sha256(args.valid_path),
            "status_path": str(args.status_path),
            "status_sha256": _sha256(args.status_path),
            "grades_path": str(args.grades_path),
            "grades_sha256": _sha256(args.grades_path),
            "knn_path": str(args.knn_path),
            "knn_sha256": _sha256(args.knn_path),
        },
        "fitted_k": fitted_k,
        "metrics": metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(args.output_dir / "predictions.parquet", index=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Report: {args.output_dir / 'metrics.json'}")
    print(f"Predictions: {args.output_dir / 'predictions.parquet'}")


if __name__ == "__main__":
    main()
