"""Read-only verification for the development V0 safety freeze.

The default check enforces immutable data/model/difficulty fingerprints,
contract consistency, and golden VALID predictions.  ``--strict-code`` also
checks the recorded code snapshot; it is optional because later product work
is expected to change inference, recommendation, and KNN source files.

This verifier never opens the provisional TEST parquet and never writes files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd
import pyarrow.parquet as pq

from src.data_prep import prepare_X_y
from src.feature_contracts import resolve_feature_contract


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "work_plan/state/current_state_manifest.json"
GOLDEN_PATH = PROJECT_ROOT / "tests/fixtures/golden_predictions_v0.json"


class FreezeVerificationError(AssertionError):
    """Raised when the development freeze no longer reproduces."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_vector_sha256(row: pd.Series) -> str:
    payload = "|".join(
        "<NA>" if pd.isna(value) else str(value)
        for value in row.astype(object).tolist()
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_artifact_fingerprints(*, strict_code: bool = False) -> dict[str, int]:
    manifest = _read_json(MANIFEST_PATH)
    checked = 0
    recorded_only = 0
    errors: list[str] = []

    for item in manifest["artifacts"]:
        required = item["verification"] == "required"
        if not required and not strict_code:
            recorded_only += 1
            continue

        path = PROJECT_ROOT / item["path"]
        if not path.is_file():
            errors.append(f"missing: {item['path']}")
            continue
        actual_bytes = path.stat().st_size
        if actual_bytes != item["bytes"]:
            errors.append(
                f"size mismatch: {item['path']} expected={item['bytes']} "
                f"actual={actual_bytes}"
            )
            continue
        actual_hash = _sha256(path)
        if actual_hash != item["sha256"]:
            errors.append(
                f"sha256 mismatch: {item['path']} expected={item['sha256']} "
                f"actual={actual_hash}"
            )
            continue
        checked += 1

    if errors:
        raise FreezeVerificationError("\n".join(errors))
    return {"checked": checked, "recorded_only_skipped": recorded_only}


def verify_contract_and_rows() -> dict[str, Any]:
    manifest = _read_json(MANIFEST_PATH)
    run_dir = PROJECT_ROOT / manifest["reference_run"]
    stored = _read_json(run_dir / "feature_contract.json")
    contract = resolve_feature_contract(manifest["reference_contract"])

    if stored["contract_name"] != contract.name:
        raise FreezeVerificationError(
            f"contract name mismatch: stored={stored['contract_name']} code={contract.name}"
        )
    if stored["ordered_features"] != list(contract.features):
        raise FreezeVerificationError("stored ordered features differ from baseline_41")
    if stored["feature_count"] != manifest["reference_contract_feature_count"]:
        raise FreezeVerificationError("feature count differs from freeze manifest")

    train_path = PROJECT_ROOT / stored["train_path"]
    valid_path = PROJECT_ROOT / stored["valid_path"]
    train_rows = pq.ParquetFile(train_path).metadata.num_rows
    valid_rows = pq.ParquetFile(valid_path).metadata.num_rows
    expected = manifest["test_policy"]
    if train_rows != expected["train_rows"] or valid_rows != expected["valid_rows"]:
        raise FreezeVerificationError(
            f"row-count mismatch: train={train_rows} valid={valid_rows}"
        )

    m1 = lgb.Booster(model_file=str(run_dir / manifest["reference_models"]["m1"]))
    m2 = lgb.Booster(model_file=str(run_dir / manifest["reference_models"]["m2"]))
    expected_features = manifest["reference_contract_feature_count"]
    if m1.num_feature() != expected_features or m2.num_feature() != expected_features:
        raise FreezeVerificationError(
            f"model widths differ: m1={m1.num_feature()} m2={m2.num_feature()}"
        )

    return {
        "contract": contract.name,
        "feature_count": expected_features,
        "train_rows": train_rows,
        "valid_rows": valid_rows,
    }


def _validate_segment(case: dict[str, Any], row: pd.Series) -> None:
    segment = case["segment"]
    same_start = str(row["start_part_id"]) == str(row["part_id"])
    if segment == "cold_start_start_part_equals_part" and not same_start:
        raise FreezeVerificationError("cold-start fixture no longer matches its definition")
    if segment == "returning" and same_start:
        raise FreezeVerificationError("returning fixture now matches the cold-start definition")
    if segment == "difficulty_fallback" and int(row["course_difficulty_missing"]) != 1:
        raise FreezeVerificationError("difficulty fixture no longer has missing difficulty")
    if segment == "retake" and int(row["attempt_number"]) <= 1:
        raise FreezeVerificationError("retake fixture no longer has attempt_number > 1")


def verify_golden_predictions() -> dict[str, Any]:
    manifest = _read_json(MANIFEST_PATH)
    golden = _read_json(GOLDEN_PATH)
    if golden["freeze_id"] != manifest["freeze_id"]:
        raise FreezeVerificationError("golden fixture belongs to a different freeze")
    if "test" in golden["dataset_path"].lower():
        raise FreezeVerificationError("golden fixture must never point to TEST")

    contract = resolve_feature_contract(golden["feature_contract"])
    run_dir = PROJECT_ROOT / golden["run_path"]
    stored = _read_json(run_dir / "feature_contract.json")
    categorical_levels = {
        name: [int(value) for value in values]
        for name, values in stored["categorical_levels"].items()
    }

    auxiliary = [
        "part_id",
        "start_part_id",
        "attempt_number",
        "difficulty_fallback_level",
        "course_difficulty_missing",
        "final_mark",
    ]
    columns = list(dict.fromkeys(contract.training_data_columns + auxiliary))
    valid = pd.read_parquet(PROJECT_ROOT / golden["dataset_path"], columns=columns)
    positions = [case["source_row_index"] for case in golden["cases"]]
    if len(positions) != len(set(positions)):
        raise FreezeVerificationError("golden source row positions are not unique")
    if not positions or min(positions) < 0 or max(positions) >= len(valid):
        raise FreezeVerificationError("golden source row position is outside VALID")

    selected = valid.iloc[positions].copy().reset_index(drop=True)
    X, _ = prepare_X_y(selected, "grade", categorical_levels, contract)
    m1 = lgb.Booster(model_file=str(run_dir / "m1_pass_model.lgbm"))
    m2 = lgb.Booster(model_file=str(run_dir / "m2_grade_model.lgbm"))
    pass_probabilities = m1.predict(X)
    predicted_marks = m2.predict(X)
    tolerance = float(golden["absolute_tolerance"])

    for index, case in enumerate(golden["cases"]):
        row = selected.iloc[index]
        _validate_segment(case, row)
        scalar_checks = {
            "actual_final_mark": int(row["final_mark"]),
            "attempt_number": int(row["attempt_number"]),
            "difficulty_fallback_level": int(row["difficulty_fallback_level"]),
            "course_difficulty_missing": int(row["course_difficulty_missing"]),
        }
        for name, actual in scalar_checks.items():
            if actual != case[name]:
                raise FreezeVerificationError(
                    f"case {index} {name} mismatch: expected={case[name]} actual={actual}"
                )

        feature_hash = _feature_vector_sha256(X.iloc[index])
        if feature_hash != case["feature_vector_sha256"]:
            raise FreezeVerificationError(
                f"case {index} feature-vector hash mismatch: "
                f"expected={case['feature_vector_sha256']} actual={feature_hash}"
            )
        if abs(float(pass_probabilities[index]) - case["m1_pass_probability"]) > tolerance:
            raise FreezeVerificationError(f"case {index} M1 prediction changed")
        if abs(float(predicted_marks[index]) - case["m2_predicted_mark"]) > tolerance:
            raise FreezeVerificationError(f"case {index} M2 prediction changed")

    return {
        "case_count": len(golden["cases"]),
        "absolute_tolerance": tolerance,
        "segments": sorted({case["segment"] for case in golden["cases"]}),
    }


def verify_freeze(*, strict_code: bool = False) -> dict[str, Any]:
    return {
        "freeze_id": _read_json(MANIFEST_PATH)["freeze_id"],
        "artifacts": verify_artifact_fingerprints(strict_code=strict_code),
        "contract_and_rows": verify_contract_and_rows(),
        "golden_predictions": verify_golden_predictions(),
        "strict_code": strict_code,
        "test_read": False,
        "status": "PASS",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-code",
        action="store_true",
        help="also enforce recorded source-code hashes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(verify_freeze(strict_code=args.strict_code), indent=2))


if __name__ == "__main__":
    main()
