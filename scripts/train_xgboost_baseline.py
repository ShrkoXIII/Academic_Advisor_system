"""Train an auditable XGBoost M1/M2 baseline on existing temporal splits.

The script never rebuilds datasets. TRAIN fits both models, VALID is the only
early-stopping/selection split, and TEST stays closed unless --evaluate-test is
explicitly supplied. Artifacts are isolated under models/xgboost_runs/.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

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
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep import learn_categorical_levels, prepare_X_y  # noqa: E402
from src.feature_contracts import REPORTING_THRESHOLD, resolve_feature_contract  # noqa: E402


DATASET_DIR = PROJECT_ROOT / "data/model_data/versions/2026-08_temporal_rebuild_v2/05_dataset"
DEFAULT_TRAIN = DATASET_DIR / "train_dataset_candidate.parquet"
DEFAULT_VALID = DATASET_DIR / "valid_dataset_candidate.parquet"
DEFAULT_TEST = DATASET_DIR / "test_provisional_dataset_candidate.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "models/xgboost_runs"
DEFAULT_LIGHTGBM_RUN = PROJECT_ROOT / "models/runs/2026-08-18_1647__rebuild-v2-baseline41-s82"
AUXILIARY_COLUMNS = ("degree_id", "part_id", "start_part_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="xgb-baseline41")
    parser.add_argument("--feature-contract", default="baseline_41")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--seed", type=int, default=82)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--num-boost-round", type=int, default=2_000)
    parser.add_argument("--early-stopping-rounds", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-child-weight", type=float, default=20.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=0.1)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--max-bin", type=int, default=255)
    args = parser.parse_args()
    if args.num_threads < 1 or args.num_boost_round < 1 or args.early_stopping_rounds < 1:
        parser.error("thread/round counts must be positive")
    if args.max_depth < 1 or args.min_child_weight < 0 or args.max_bin < 2:
        parser.error("invalid tree complexity parameter")
    for name in ("subsample", "colsample_bytree"):
        value = getattr(args, name)
        if not 0 < value <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in (0,1]")
    return args


def _slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    if not value:
        raise ValueError("--run-name must contain a letter or number")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_split(path: Path, contract) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    columns = list(dict.fromkeys(contract.training_data_columns + list(AUXILIARY_COLUMNS)))
    return pd.read_parquet(path, columns=columns)


def _true_start_mask(frame: pd.DataFrame) -> np.ndarray:
    part = pd.to_numeric(frame["part_id"], errors="coerce")
    start = pd.to_numeric(frame["start_part_id"], errors="coerce")
    return start.eq(part).fillna(False).to_numpy(dtype=bool)


def _best_predict(model: xgb.Booster, matrix: xgb.DMatrix) -> np.ndarray:
    end = int(model.best_iteration) + 1 if model.best_iteration is not None else 0
    return model.predict(matrix, iteration_range=(0, end)) if end else model.predict(matrix)


def _m1_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    predicted = (probability >= REPORTING_THRESHOLD).astype(int)
    tn, fp, fn, tp = (
        int(value) for value in confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    )
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "reporting_threshold": float(REPORTING_THRESHOLD),
        "auc": float(roc_auc_score(y, probability)),
        "avg_precision": float(average_precision_score(y, probability)),
        "fail_avg_precision": float(average_precision_score(1 - y, 1 - probability)),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "fail_precision": float(precision_score(y, predicted, pos_label=0, zero_division=0)),
        "fail_recall": float(recall_score(y, predicted, pos_label=0, zero_division=0)),
        "fail_f1": float(f1_score(y, predicted, pos_label=0, zero_division=0)),
        "brier": float(brier_score_loss(y, probability)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _m2_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = prediction - y
    total = float(np.square(y - y.mean()).sum())
    residual = float(np.square(error).sum())
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(1.0 - residual / total) if total else None,
        "bias_pred_minus_observed": float(error.mean()),
    }


def _split_metrics(
    frame: pd.DataFrame,
    y_mark: np.ndarray,
    probability: np.ndarray,
    mark_prediction: np.ndarray,
) -> dict[str, Any]:
    cold = _true_start_mask(frame)
    returning = ~cold
    result: dict[str, Any] = {}
    for name, mask in (
        ("all", np.ones(len(frame), dtype=bool)),
        ("true_cold_start", cold),
        ("returning", returning),
    ):
        y_segment = y_mark[mask]
        result[name] = {
            "m1": _m1_metrics((y_segment >= 50).astype(int), probability[mask]),
            "m2": _m2_metrics(y_segment, mark_prediction[mask]),
        }
    return result


def _importance(model: xgb.Booster, features: list[str]) -> pd.DataFrame:
    gain = model.get_score(importance_type="gain")
    weight = model.get_score(importance_type="weight")
    return pd.DataFrame(
        {
            "feature": features,
            "gain": [float(gain.get(feature, 0.0)) for feature in features],
            "splits": [int(weight.get(feature, 0)) for feature in features],
        }
    ).sort_values(["gain", "splits"], ascending=False)


def main() -> None:
    args = parse_args()
    contract = resolve_feature_contract(args.feature_contract)
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S")
    run_dir = args.out / f"{timestamp}__{_slug(args.run_name)}-s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)

    print(f"XGBoost {xgb.__version__}; contract={contract.name}; seed={args.seed}")
    print(f"TEST policy: {'evaluated_descriptive_only' if args.evaluate_test else 'closed_not_read'}")
    print(f"Artifacts: {run_dir}")
    train = _load_split(args.train, contract)
    valid = _load_split(args.valid, contract)
    test = _load_split(args.test, contract) if args.evaluate_test else None
    print(
        f"Rows: train={len(train):,}; valid={len(valid):,}; "
        f"test={len(test):,}" if test is not None else
        f"Rows: train={len(train):,}; valid={len(valid):,}; test=CLOSED"
    )

    diploma_fill = float(pd.to_numeric(train["diploma_gpa"], errors="coerce").median())
    for frame in (train, valid, *((test,) if test is not None else ())):
        frame["diploma_gpa"] = pd.to_numeric(frame["diploma_gpa"], errors="coerce").fillna(diploma_fill)

    categorical_levels = learn_categorical_levels(train, contract)
    X_train, y_train_series = prepare_X_y(train, "grade", categorical_levels, contract)
    X_valid, y_valid_series = prepare_X_y(valid, "grade", categorical_levels, contract)
    X_test = y_test_series = None
    if test is not None:
        X_test, y_test_series = prepare_X_y(test, "grade", categorical_levels, contract)
    if list(X_train.columns) != list(contract.features):
        raise AssertionError("feature order differs from contract")

    y_train = y_train_series.to_numpy(dtype=float)
    y_valid = y_valid_series.to_numpy(dtype=float)
    y_test = y_test_series.to_numpy(dtype=float) if y_test_series is not None else None
    dtrain = xgb.DMatrix(
        X_train,
        label=(y_train >= 50).astype(int),
        enable_categorical=True,
    )
    dvalid = xgb.DMatrix(
        X_valid,
        label=(y_valid >= 50).astype(int),
        enable_categorical=True,
    )
    dtest = (
        xgb.DMatrix(X_test, label=(y_test >= 50).astype(int), enable_categorical=True)
        if X_test is not None and y_test is not None
        else None
    )
    del X_train, X_valid, X_test, y_train_series, y_valid_series, y_test_series

    shared_params = {
        "tree_method": "hist",
        "device": "cpu",
        "eta": args.learning_rate,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "max_bin": args.max_bin,
        "max_cat_to_onehot": 4,
        "seed": args.seed,
        "nthread": args.num_threads,
    }

    print("\n=== Training XGBoost M1 ===")
    m1_history: dict[str, Any] = {}
    m1_params = {
        **shared_params,
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
    }
    m1 = xgb.train(
        m1_params,
        dtrain,
        num_boost_round=args.num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=args.early_stopping_rounds,
        evals_result=m1_history,
        verbose_eval=50,
    )
    m1_train_probability = _best_predict(m1, dtrain)
    m1_valid_probability = _best_predict(m1, dvalid)
    m1_test_probability = _best_predict(m1, dtest) if dtest is not None else None
    m1.save_model(run_dir / "m1_pass_model.ubj")
    _importance(m1, list(contract.features)).to_csv(run_dir / "m1_feature_importance.csv", index=False)

    print("\n=== Training XGBoost M2 ===")
    dtrain.set_label(y_train)
    dvalid.set_label(y_valid)
    if dtest is not None and y_test is not None:
        dtest.set_label(y_test)
    m2_history: dict[str, Any] = {}
    m2_params = {
        **shared_params,
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
    }
    m2 = xgb.train(
        m2_params,
        dtrain,
        num_boost_round=args.num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=args.early_stopping_rounds,
        evals_result=m2_history,
        verbose_eval=50,
    )
    m2_train_prediction = _best_predict(m2, dtrain)
    m2_valid_prediction = _best_predict(m2, dvalid)
    m2_test_prediction = _best_predict(m2, dtest) if dtest is not None else None
    m2.save_model(run_dir / "m2_grade_model.ubj")
    _importance(m2, list(contract.features)).to_csv(run_dir / "m2_feature_importance.csv", index=False)

    metrics = {
        "model_family": "xgboost",
        "xgboost_version": xgb.__version__,
        "selection_split": "valid_only",
        "test_policy": "evaluated_descriptive_only" if test is not None else "closed_not_read",
        "feature_contract": contract.name,
        "feature_count": contract.expected_feature_count,
        "m1_best_iteration": int(m1.best_iteration),
        "m2_best_iteration": int(m2.best_iteration),
        "train": _split_metrics(train, y_train, m1_train_probability, m2_train_prediction),
        "valid": _split_metrics(valid, y_valid, m1_valid_probability, m2_valid_prediction),
        "test": (
            _split_metrics(test, y_test, m1_test_probability, m2_test_prediction)
            if test is not None and y_test is not None
            and m1_test_probability is not None and m2_test_prediction is not None
            else None
        ),
    }
    config = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "dataset_inputs": {
            "train": {"path": str(args.train), "rows": len(train), "sha256": _sha256(args.train)},
            "valid": {"path": str(args.valid), "rows": len(valid), "sha256": _sha256(args.valid)},
            "test": (
                {"path": str(args.test), "rows": len(test), "sha256": _sha256(args.test)}
                if test is not None
                else "closed_not_read"
            ),
        },
        "ordered_features": list(contract.features),
        "categorical_levels": categorical_levels,
        "diploma_gpa_fill": {"method": "train_median", "value": diploma_fill},
        "m1_params": m1_params,
        "m2_params": m2_params,
        "num_boost_round": args.num_boost_round,
        "early_stopping_rounds": args.early_stopping_rounds,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (run_dir / "training_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (run_dir / "training_curves.json").write_text(
        json.dumps({"m1": m1_history, "m2": m2_history}, indent=2) + "\n",
        encoding="utf-8",
    )

    valid_all = metrics["valid"]["all"]
    valid_cold = metrics["valid"]["true_cold_start"]
    print("\n=== VALID RESULT ===")
    print(
        f"ALL: M1 AUC={valid_all['m1']['auc']:.6f}; "
        f"fail AP={valid_all['m1']['fail_avg_precision']:.6f}; "
        f"M2 MAE={valid_all['m2']['mae']:.6f}; R2={valid_all['m2']['r2']:.6f}"
    )
    print(
        f"TRUE COLD: n={valid_cold['m1']['n']:,}; "
        f"M1 AUC={valid_cold['m1']['auc']:.6f}; "
        f"M2 MAE={valid_cold['m2']['mae']:.6f}"
    )
    if metrics["test"] is not None:
        test_all = metrics["test"]["all"]
        test_cold = metrics["test"]["true_cold_start"]
        print("\n=== TEST DESCRIPTIVE RESULT ===")
        print(
            f"ALL: M1 AUC={test_all['m1']['auc']:.6f}; "
            f"fail AP={test_all['m1']['fail_avg_precision']:.6f}; "
            f"M2 MAE={test_all['m2']['mae']:.6f}; R2={test_all['m2']['r2']:.6f}"
        )
        print(
            f"TRUE COLD: n={test_cold['m1']['n']:,}; "
            f"M1 AUC={test_cold['m1']['auc']:.6f}; "
            f"M2 MAE={test_cold['m2']['mae']:.6f}"
        )
    print(f"\nSaved: {run_dir}")


if __name__ == "__main__":
    main()
