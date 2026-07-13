"""Diagnostic-only sweep of M1 failure-warning thresholds.

This script never trains, saves, or alters a model.  It uses the persisted
feature contract beside ``m1_pass_model.lgbm`` so the input schema is exactly
the schema the loaded model expects.  Test data are reported only after all
validation-only diagnostic selections have been made.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, fbeta_score, f1_score, precision_score, recall_score

from src import model_training as training


THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
SEGMENTS = {
    "first_active_semester": lambda df: df["is_first_active_semester"] == 1,
    "retake_course": lambda df: df["attempt_number"] > 1,
    "no_previous_valid_progress": lambda df: df["no_previous_progress"] == 1,
    "low_difficulty_support": lambda df: df["difficulty_fallback_level"] >= 3,
}
TARGET = "final_mark"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=root / "models" / "m1_pass_model.lgbm")
    parser.add_argument("--contract", type=Path, default=root / "models" / "feature_contract.json")
    parser.add_argument("--valid", type=Path, default=root / "data" / "model_data" / "df_valid_final.parquet")
    parser.add_argument("--test", type=Path, default=root / "data" / "model_data" / "df_test_final.parquet")
    parser.add_argument("--output-root", type=Path, default=root / "models" / "diagnostics")
    return parser.parse_args()


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required file(s) missing:\n- " + "\n- ".join(missing))


def load_contract(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        contract = json.load(handle)
    required = {"features", "categorical_features", "categorical_levels", "unknown_category_code"}
    absent = sorted(required - set(contract))
    if absent:
        raise ValueError(f"Feature contract is missing keys: {absent}")
    if len(contract["features"]) != contract.get("n_features", len(contract["features"])):
        raise ValueError("Feature contract feature count is inconsistent.")
    return contract


def prepare_features(df: pd.DataFrame, contract: dict[str, Any], model_features: list[str]) -> pd.DataFrame:
    """Reproduce the persisted contract's feature and categorical treatment."""
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    if df[TARGET].isna().any() or ((df[TARGET] < 0) | (df[TARGET] > 100)).any():
        raise ValueError(f"{TARGET} must be non-null and within [0, 100].")

    work = df.copy()
    # This ordinal derivation is the same one used in src/model_training.py.
    if "requirement_size_bucket_ord" in contract["features"]:
        if "requirement_size_bucket" not in work.columns:
            raise ValueError("Missing source column: requirement_size_bucket")
        work["requirement_size_bucket_ord"] = (
            work["requirement_size_bucket"].astype("string")
            .map(training.REQUIREMENT_BUCKET_ORD)
            .fillna(0)
            .astype(int)
        )

    missing = [column for column in contract["features"] if column not in work.columns]
    if missing:
        raise ValueError(f"Missing contract feature column(s): {missing}")

    unknown = int(contract["unknown_category_code"])
    for column in contract["categorical_features"]:
        if column not in contract["features"]:
            raise ValueError(f"Categorical feature outside contract features: {column}")
        allowed = [int(value) for value in contract["categorical_levels"][column]]
        values = pd.to_numeric(work[column], errors="coerce")
        mapped = values.where(values.isin(allowed), other=unknown).fillna(unknown).astype(int)
        work[column] = pd.Categorical(mapped, categories=sorted(set(allowed + [unknown])))

    x = work[contract["features"]].copy()
    numeric = [column for column in contract["features"] if column not in contract["categorical_features"]]
    x[numeric] = x[numeric].astype("float64")
    if list(x.columns) != model_features:
        raise ValueError("Prepared feature names/order do not match the loaded model.")
    return x


def metrics_for_threshold(y_failure: np.ndarray, warnings: np.ndarray, threshold: float) -> dict[str, Any]:
    # Rows = actual [failure, pass], columns = warning [yes, no].
    cm = confusion_matrix(y_failure, warnings, labels=[True, False])
    detected, missed = (int(cm[0, 0]), int(cm[0, 1]))
    false_warnings, correctly_unwarned = (int(cm[1, 0]), int(cm[1, 1]))
    return {
        "threshold_pass_probability": threshold,
        "failure_precision": precision_score(y_failure, warnings, pos_label=True, zero_division=0),
        "failure_recall": recall_score(y_failure, warnings, pos_label=True, zero_division=0),
        "failure_f1": f1_score(y_failure, warnings, pos_label=True, zero_division=0),
        "failure_f2": fbeta_score(y_failure, warnings, beta=2, pos_label=True, zero_division=0),
        "detected_failures": detected,
        "undetected_failures": missed,
        "false_warnings_on_passes": false_warnings,
        "correctly_unwarned_passes": correctly_unwarned,
        "warning_rate": float(warnings.mean()),
        "n_rows": int(len(y_failure)),
        "n_actual_failures": int(y_failure.sum()),
        "n_actual_passes": int((~y_failure).sum()),
        "confusion_matrix": f"[[{detected}, {missed}], [{false_warnings}, {correctly_unwarned}]]",
    }


def sweep(df: pd.DataFrame, probabilities: np.ndarray, split: str, segment: str | None = None) -> pd.DataFrame:
    y_failure = (df[TARGET].to_numpy() < 50)
    rows = []
    for threshold in THRESHOLDS:
        row = metrics_for_threshold(y_failure, probabilities < threshold, threshold)
        row["split"] = split
        if segment is not None:
            row["segment"] = segment
        rows.append(row)
    return pd.DataFrame(rows)


def segment_sweep(df: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for name, selector in SEGMENTS.items():
        mask = selector(df).fillna(False).to_numpy(dtype=bool)
        n_rows = int(mask.sum())
        if n_rows < 100:
            rows.append(pd.DataFrame({
                "split": "validation", "segment": name, "threshold_pass_probability": THRESHOLDS,
                "n_rows": n_rows, "status": "insufficient_lt_100",
            }))
            continue
        result = sweep(df.loc[mask], probabilities[mask], "validation", name)
        result["status"] = "ok" if result["n_actual_failures"].iloc[0] else "no_actual_failures"
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def choose(validation: pd.DataFrame) -> dict[str, pd.Series | None]:
    def top(frame: pd.DataFrame, order: list[str]) -> pd.Series | None:
        if frame.empty:
            return None
        return frame.sort_values(order, ascending=[False] * len(order), kind="stable").iloc[0]

    return {
        "highest_f2": top(validation, ["failure_f2", "failure_recall", "failure_precision"]),
        "best_recall_precision_ge_020": top(validation[validation["failure_precision"] >= 0.20], ["failure_recall", "failure_precision"]),
        "best_recall_precision_ge_030": top(validation[validation["failure_precision"] >= 0.30], ["failure_recall", "failure_precision"]),
        "best_recall_warning_rate_le_025": top(validation[validation["warning_rate"] <= 0.25], ["failure_recall", "failure_precision"]),
    }


def format_result(label: str, row: pd.Series | None) -> str:
    if row is None:
        return f"- {label}: لا توجد عتبة ضمن الشرط."
    return (
        f"- {label}: حد {row['threshold_pass_probability']:.2f}؛ "
        f"الدقة={row['failure_precision']:.3f}، الاسترجاع={row['failure_recall']:.3f}، "
        f"F2={row['failure_f2']:.3f}، التحذيرات={row['warning_rate']:.1%}."
    )


def write_report(path: Path, model_path: Path, timestamp: str, selections: dict[str, pd.Series | None], test: pd.DataFrame) -> None:
    lines = [
        "# تقرير تشخيص حدود اكتشاف الفشل",
        "",
        f"- المودل المستخدم: `{model_path.name}`",
        f"- تاريخ التشخيص: {timestamp}",
        "",
        "## أفضل النتائج التشخيصية من التحقق",
        "",
        format_result("أعلى F2", selections["highest_f2"]),
        format_result("أعلى استرجاع مع دقة ≥ 0.20", selections["best_recall_precision_ge_020"]),
        format_result("أعلى استرجاع مع دقة ≥ 0.30", selections["best_recall_precision_ge_030"]),
        format_result("أعلى استرجاع مع تحذيرات ≤ 25%", selections["best_recall_warning_rate_le_025"]),
        "",
        "## مقارنة مختصرة مع الاختبار",
        "",
    ]
    for label, row in selections.items():
        if row is None:
            continue
        test_row = test.loc[test["threshold_pass_probability"] == row["threshold_pass_probability"]].iloc[0]
        lines.append(
            f"- {label} عند {row['threshold_pass_probability']:.2f}: "
            f"التحقق F2={row['failure_f2']:.3f}/استرجاع={row['failure_recall']:.3f}؛ "
            f"الاختبار F2={test_row['failure_f2']:.3f}/استرجاع={test_row['failure_recall']:.3f}."
        )
    lines += [
        "",
        "أهم مقايضة: رفع حد احتمال النجاح يوسّع التحذير، فيرفع عادةً اكتشاف الراسبين "
        "ويزيد معه تحذير الناجحين خطأً.",
        "",
        "> **تحذير:** هذه النتائج تشخيصية فقط، ولا تمثل اعتماداً نهائياً لحد القرار.  ",
        "> يجب إعادة الاختيار بعد تثبيت النسخة النهائية من المودل.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def unique_output_dir(root: Path, timestamp: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base = root / f"{timestamp}__failure-threshold-sweep"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = root / f"{base.name}__{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def main() -> None:
    args = parse_args()
    require_files([args.model, args.contract, args.valid, args.test])
    contract = load_contract(args.contract)
    model = lgb.Booster(model_file=str(args.model))
    model_features = model.feature_name()
    if contract["features"] != model_features:
        raise ValueError("feature_contract.json does not exactly match the model feature names/order.")

    valid = pd.read_parquet(args.valid)
    x_valid = prepare_features(valid, contract, model_features)
    valid_prob = model.predict(x_valid)
    if not np.isfinite(valid_prob).all() or (valid_prob < 0).any() or (valid_prob > 1).any():
        raise ValueError("validation predictions are not finite probabilities in [0, 1].")

    validation_results = sweep(valid, valid_prob, "validation")
    selections = choose(validation_results)  # Deliberately before any test results are examined.

    test = pd.read_parquet(args.test)
    x_test = prepare_features(test, contract, model_features)
    test_prob = model.predict(x_test)
    if not np.isfinite(test_prob).all() or (test_prob < 0).any() or (test_prob > 1).any():
        raise ValueError("test predictions are not finite probabilities in [0, 1].")
    test_results = sweep(test, test_prob, "test")
    segment_results = segment_sweep(valid, valid_prob)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    output_dir = unique_output_dir(args.output_root, timestamp)
    validation_results.to_csv(output_dir / "validation_thresholds.csv", index=False)
    test_results.to_csv(output_dir / "test_thresholds.csv", index=False)
    segment_results.to_csv(output_dir / "segment_thresholds.csv", index=False)
    write_report(output_dir / "REPORT.md", args.model, timestamp, selections, test_results)

    print(f"Validation rows used: {len(valid):,}")
    print(f"Test rows used: {len(test):,}")
    print("Predictions verified as probabilities in [0, 1].")
    print("Diagnostic selections were made from validation only; test was not used for selection.")
    print("No model training was performed and no existing file was modified.")
    print("Created files:")
    for file_path in sorted(output_dir.iterdir()):
        print(f"- {file_path}")


if __name__ == "__main__":
    main()
