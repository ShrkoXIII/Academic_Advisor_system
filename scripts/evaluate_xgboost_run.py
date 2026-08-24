"""Evaluate a saved XGBoost run on VALID and descriptively on TEST."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for location in (PROJECT_ROOT, SCRIPT_DIR):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from src.data_prep import prepare_X_y  # noqa: E402
from src.feature_contracts import resolve_feature_contract  # noqa: E402
from train_xgboost_baseline import (  # noqa: E402
    AUXILIARY_COLUMNS,
    DEFAULT_TEST,
    _best_predict,
    _split_metrics,
)


DEFAULT_RUN = (
    PROJECT_ROOT
    / "models/xgboost_runs/2026-08-22_094650__xgb-baseline41-valid-s82"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    return parser.parse_args()


def _read(path: Path, contract) -> pd.DataFrame:
    columns = list(dict.fromkeys(contract.training_data_columns + list(AUXILIARY_COLUMNS)))
    return pd.read_parquet(path, columns=columns)


def main() -> None:
    args = parse_args()
    config = json.loads((args.run_dir / "training_config.json").read_text(encoding="utf-8"))
    saved = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8"))
    contract = resolve_feature_contract(saved["feature_contract"])
    levels = {
        key: [int(value) for value in values]
        for key, values in config["categorical_levels"].items()
    }
    fill = float(config["diploma_gpa_fill"]["value"])
    valid_path = Path(config["dataset_inputs"]["valid"]["path"])
    valid = _read(valid_path, contract)
    test = _read(args.test, contract)
    for frame in (valid, test):
        frame["diploma_gpa"] = pd.to_numeric(
            frame["diploma_gpa"], errors="coerce"
        ).fillna(fill)

    X_valid, y_valid = prepare_X_y(valid, "grade", levels, contract)
    X_test, y_test = prepare_X_y(test, "grade", levels, contract)
    y_valid_array = y_valid.to_numpy(dtype=float)
    y_test_array = y_test.to_numpy(dtype=float)
    dvalid = xgb.DMatrix(X_valid, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    m1 = xgb.Booster(model_file=str(args.run_dir / "m1_pass_model.ubj"))
    m2 = xgb.Booster(model_file=str(args.run_dir / "m2_grade_model.ubj"))
    valid_metrics = _split_metrics(
        valid,
        y_valid_array,
        _best_predict(m1, dvalid),
        _best_predict(m2, dvalid),
    )
    if abs(valid_metrics["all"]["m1"]["auc"] - saved["valid"]["all"]["m1"]["auc"]) > 1e-12:
        raise AssertionError("saved M1 VALID reproduction failed")
    if abs(valid_metrics["all"]["m2"]["mae"] - saved["valid"]["all"]["m2"]["mae"]) > 1e-12:
        raise AssertionError("saved M2 VALID reproduction failed")

    test_metrics = _split_metrics(
        test,
        y_test_array,
        _best_predict(m1, dtest),
        _best_predict(m2, dtest),
    )
    report = {
        "status": "complete",
        "model_training": False,
        "selection_split": "valid_already_fixed",
        "test_policy": "descriptive_only",
        "valid_reproduced": valid_metrics,
        "test": test_metrics,
    }
    output = args.run_dir / "test_metrics_descriptive.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for split, values in (("VALID", valid_metrics), ("TEST", test_metrics)):
        overall = values["all"]
        cold = values["true_cold_start"]
        returning = values["returning"]
        print(f"{split} ALL: M1 AUC={overall['m1']['auc']:.6f}; "
              f"fail AP={overall['m1']['fail_avg_precision']:.6f}; "
              f"M2 MAE={overall['m2']['mae']:.6f}; R2={overall['m2']['r2']:.6f}")
        print(f"{split} COLD: n={cold['m1']['n']:,}; M1 AUC={cold['m1']['auc']:.6f}; "
              f"M2 MAE={cold['m2']['mae']:.6f}; bias={cold['m2']['bias_pred_minus_observed']:.6f}")
        print(f"{split} RETURNING: n={returning['m1']['n']:,}; "
              f"M1 AUC={returning['m1']['auc']:.6f}; M2 MAE={returning['m2']['mae']:.6f}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
