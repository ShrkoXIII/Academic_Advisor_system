"""Parity check: an R2 run must differ from its control by num_leaves ALONE.

Used two ways: called by the training runner after every run so a broken pair
stops immediately, and imported by the confirmation report generator so the
same checks appear in the report. One implementation, one definition of
"parity", no chance of the runner and the report disagreeing.

Exit code 0 = every check passed; 1 = at least one failed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.model_training import NUM_BOOST_ROUND, _effective_seed_settings

ROOT = Path(__file__).resolve().parents[1]

# LightGBM canonicalises alias names when it serializes a model.
LEVER_MODEL_FILE_ALIAS = "num_leaves"
R2_NUM_LEAVES = 31.0
CONTROL_NUM_LEAVES = 127.0


def model_file_params(model_path: Path) -> dict:
    """Every parameter LightGBM itself serialized into a model file."""
    values: dict[str, str] = {}
    in_block = False
    for line in model_path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "parameters:":
            in_block = True
            continue
        if in_block:
            if not line.strip():
                break
            if line.startswith("[") and ": " in line:
                name, raw = line[1:-1].split(": ", 1)
                values[name] = raw
    if not values:
        raise AssertionError(f"No parameter block found in {model_path}")
    return values


def check(run_dir: Path, control_dir: Path, arm: str, seed: int) -> dict:
    """Every way in which an R2 run must match its same-seed control."""
    checks: dict[str, dict] = {}
    run_contract = json.loads((run_dir / "feature_contract.json").read_text(encoding="utf-8"))
    ctl_contract = json.loads((control_dir / "feature_contract.json").read_text(encoding="utf-8"))
    run_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    ctl_metrics = json.loads((control_dir / "metrics.json").read_text(encoding="utf-8"))

    def record(name: str, ok: bool, run_value, control_value, note: str = "") -> None:
        checks[name] = {
            "match": bool(ok),
            "run": run_value,
            "control": control_value,
            **({"note": note} if note else {}),
        }

    # --- contract identity -------------------------------------------------
    record("feature_contract",
           run_contract["contract_name"] == ctl_contract["contract_name"] == arm,
           run_contract["contract_name"], ctl_contract["contract_name"])
    record("ordered_features",
           run_contract["ordered_features"] == ctl_contract["ordered_features"],
           run_contract["feature_count"], ctl_contract["feature_count"])
    same_levels = run_contract["categorical_levels"] == ctl_contract["categorical_levels"]
    record("categorical_levels", same_levels,
           "identical" if same_levels else "DIFFERS", "identical")
    record("reporting_threshold",
           run_contract["reporting_threshold"] == ctl_contract["reporting_threshold"] == 0.80,
           run_contract["reporting_threshold"], ctl_contract["reporting_threshold"])
    record("test_policy",
           run_contract["test_policy"] == ctl_contract["test_policy"] == "closed_not_read",
           run_contract["test_policy"], ctl_contract["test_policy"])

    # --- dataset identity --------------------------------------------------
    run_inputs = run_contract["dataset_inputs"]
    ctl_inputs = ctl_contract["dataset_inputs"]
    for split in ("train", "valid"):
        record(f"{split}_sha256",
               run_inputs[split]["sha256"] == ctl_inputs[split]["sha256"],
               run_inputs[split]["sha256"][:16] + "…",
               ctl_inputs[split]["sha256"][:16] + "…")
    record("dataset_version",
           run_contract["dataset_version"] == ctl_contract["dataset_version"],
           run_contract["dataset_version"], ctl_contract["dataset_version"])

    # Row counts: the seed-42 baseline_41 control predates data_rows. Identical
    # file hashes already prove identical rows, so fall back to that rather
    # than silently skipping the check.
    ctl_rows = ctl_contract.get("data_rows")
    if ctl_rows is None:
        record("data_rows",
               run_inputs["train"]["sha256"] == ctl_inputs["train"]["sha256"]
               and run_inputs["valid"]["sha256"] == ctl_inputs["valid"]["sha256"],
               run_contract.get("data_rows"), "not recorded (predates the field)",
               "control predates data_rows; identical TRAIN/VALID SHA-256 proves "
               "identical row counts")
    else:
        record("data_rows", run_contract.get("data_rows") == ctl_rows,
               run_contract.get("data_rows"), ctl_rows)

    # --- seeds, read from the serialized models ----------------------------
    run_seeds = _effective_seed_settings(run_dir / "m1_pass_model.lgbm")
    ctl_seeds = _effective_seed_settings(control_dir / "m1_pass_model.lgbm")
    record("effective_seed_settings",
           run_seeds == ctl_seeds and run_seeds["seed"] == seed, run_seeds, ctl_seeds)
    record("m1_m2_same_seeds",
           run_seeds == _effective_seed_settings(run_dir / "m2_grade_model.lgbm"),
           "identical", "identical")

    # --- THE core check: only num_leaves differs, for BOTH models ----------
    for model_file in ("m1_pass_model.lgbm", "m2_grade_model.lgbm"):
        run_params = model_file_params(run_dir / model_file)
        ctl_params = model_file_params(control_dir / model_file)
        differing = sorted(
            key for key in set(run_params) | set(ctl_params)
            if run_params.get(key) != ctl_params.get(key)
        )
        record(f"{model_file}_only_num_leaves_differs",
               differing == [LEVER_MODEL_FILE_ALIAS],
               differing or "none", f"expected exactly ['{LEVER_MODEL_FILE_ALIAS}']")
        record(f"{model_file}_num_leaves_values",
               float(run_params[LEVER_MODEL_FILE_ALIAS]) == R2_NUM_LEAVES
               and float(ctl_params[LEVER_MODEL_FILE_ALIAS]) == CONTROL_NUM_LEAVES,
               float(run_params[LEVER_MODEL_FILE_ALIAS]),
               float(ctl_params[LEVER_MODEL_FILE_ALIAS]))
        record(f"{model_file}_num_iterations",
               int(run_params["num_iterations"]) == int(ctl_params["num_iterations"])
               == NUM_BOOST_ROUND,
               int(run_params["num_iterations"]), int(ctl_params["num_iterations"]))
        record(f"{model_file}_num_threads",
               int(run_params["num_threads"]) == int(ctl_params["num_threads"]) == 4,
               int(run_params["num_threads"]), int(ctl_params["num_threads"]))

    # --- early stopping and diploma fill -----------------------------------
    run_control_block = run_contract.get("training_control") or {}
    ctl_control_block = ctl_contract.get("training_control") or {}
    if ctl_control_block:
        record("training_control", run_control_block == ctl_control_block,
               run_control_block, ctl_control_block)
    else:
        record("training_control",
               run_control_block.get("early_stopping_rounds") == 50
               and run_control_block.get("num_boost_round") == NUM_BOOST_ROUND
               and run_control_block.get("early_stopping_selection_split") == "valid_only",
               run_control_block, "not recorded (predates the field)",
               "control predates training_control; its serialized num_iterations "
               "is verified above")

    run_diploma = run_contract.get("diploma_gpa_handling") or {}
    ctl_diploma = ctl_contract.get("diploma_gpa_handling") or {}
    if ctl_diploma:
        record("diploma_gpa_fill", run_diploma == ctl_diploma,
               run_diploma.get("fill_value"), ctl_diploma.get("fill_value"))
    else:
        record("diploma_gpa_fill",
               run_diploma.get("method") == "train_median_fill"
               and run_diploma.get("learned_from") == "train_only",
               run_diploma.get("fill_value"), "not recorded (predates the field)",
               "control predates diploma_gpa_handling; the fill is the TRAIN "
               "median of an identically hashed TRAIN file, so it is "
               "deterministic and equal")

    # --- TEST really stayed closed -----------------------------------------
    record("test_never_read",
           run_metrics["m1_pass_classifier"]["test"] is None
           and run_metrics["m2_grade_regressor"]["test"] is None
           and ctl_metrics["m1_pass_classifier"]["test"] is None
           and ctl_metrics["m2_grade_regressor"]["test"] is None,
           "null", "null")

    failed = sorted(name for name, entry in checks.items() if not entry["match"])
    return {
        "run_path": str(run_dir.relative_to(ROOT)).replace("\\", "/"),
        "control_path": str(control_dir.relative_to(ROOT)).replace("\\", "/"),
        "arm": arm,
        "seed": seed,
        "checks": checks,
        "check_count": len(checks),
        "failed_checks": failed,
        "all_passed": not failed,
    }


def main() -> int:
    if len(sys.argv) != 5:
        print("usage: python -m scripts.r2_parity <run_dir> <control_dir> <arm> <seed>")
        return 2
    run_dir, control_dir, arm, seed = sys.argv[1:]
    result = check(ROOT / run_dir, ROOT / control_dir, arm, int(seed))
    status = "PASS" if result["all_passed"] else "FAIL"
    print(f"  parity {status}: {result['check_count']} checks, "
          f"failed={result['failed_checks'] or 'none'}")
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
