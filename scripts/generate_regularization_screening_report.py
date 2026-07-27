"""Generate the seed-42 regularization screening report (four single-lever configs).

Screening design and the LOCKED acceptance rule live in
docs/EXPERIMENT_REGULARIZATION_PLAN.md, committed before any of these runs
existed. This script only applies that rule; it never invents a threshold.

Every metric here is recomputed by re-scoring each run's saved LightGBM models
against TRAIN/VALID, rather than trusting the possibly-rounded values already
stored in metrics.json (CLAUDE.md sec 7: verification is evidence-first). The
`level_1_difficulty` segment is not stored in metrics.json at all and only
exists because it is recomputed here.

This analysis reads TRAIN and VALID only. It never constructs or reads a TEST
path, and it modifies no model or dataset artifact. Paired delta is defined as
CONFIGURATION minus SAME-CONTRACT CONTROL, both at seed 42.
"""

from __future__ import annotations

import gc
import hashlib
import json
import subprocess
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from src.model_training import (
    NUM_BOOST_ROUND,
    REPORTING_THRESHOLD,
    _effective_seed_settings,
    prepare_X_y,
    resolve_feature_contract,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "models" / "runs"
VERSION = "2026-07-26_batched_fixes__registration_roster_concurrent"
TRAIN = ROOT / f"data/model_data/versions/{VERSION}/df_train_final.parquet"
VALID = ROOT / f"data/model_data/versions/{VERSION}/df_valid_final.parquet"
OUT_JSON = RUNS / "REGULARIZATION_SCREENING_SEED42_REPORT.json"
OUT_MD = RUNS / "REGULARIZATION_SCREENING_SEED42_REPORT.md"

SEED = 42

CONTROLS = {
    "baseline_41": "2026-07-26_1551__baseline-41-gpa-trend-control",
    "concurrent_43": "2026-07-27_1327__seed42-concurrent-43-drop-dead-missing-flag",
}

# Configuration -> the ONE LightGBM parameter it moves, and its control value.
# Model-file alias is what LightGBM writes into the serialized .lgbm.
CONFIGS = {
    "R1": {
        "label": "num_leaves 127 -> 63",
        "param": "num_leaves",
        "model_file_alias": "num_leaves",
        "control_value": 127.0,
        "value": 63.0,
        "slug": "reg-r1-leaves63",
    },
    "R2": {
        "label": "num_leaves 127 -> 31",
        "param": "num_leaves",
        "model_file_alias": "num_leaves",
        "control_value": 127.0,
        "value": 31.0,
        "slug": "reg-r2-leaves31",
    },
    "R3": {
        "label": "min_child_samples 50 -> 200",
        "param": "min_child_samples",
        "model_file_alias": "min_data_in_leaf",
        "control_value": 50.0,
        "value": 200.0,
        "slug": "reg-r3-minchild200",
    },
    "R4": {
        "label": "reg_lambda 1.0 -> 10.0",
        "param": "reg_lambda",
        "model_file_alias": "lambda_l2",
        "control_value": 1.0,
        "value": 10.0,
        "slug": "reg-r4-lambda10",
    },
}

ARMS = ("baseline_41", "concurrent_43")

# Acceptance yardstick, verbatim from models/runs/NOISE_BAND.md.
# LIMITATION, repeated in the report: this band was measured from
# CONTRACT-change deltas across seeds, not from HYPERPARAMETER-change deltas.
# It is the best available yardstick, not an exact one. Not to be treated as
# precise.
NOISE_BAND = {
    "m1_valid_auc": {"min": -0.000382, "max": 0.001042},
    "m1_valid_fail_ap": {"min": -0.002045, "max": 0.001544},
    "m1_valid_brier": {"min": -0.000108, "max": 0.000119},
    "m1_train_valid_auc_gap": {"min": -0.005873, "max": 0.026720},
    "m2_valid_mae": {"min": -0.050423, "max": 0.046520},
    "m2_valid_rmse": {"min": -0.067477, "max": 0.078050},
    "m2_valid_r2": {"min": -0.007865, "max": 0.006807},
    "cold_start_auc": {"min": -0.011618, "max": 0.008190},
    "low_difficulty_support_auc": {"min": -0.006657, "max": 0.008522},
    "level_1_auc": {"min": -0.000538, "max": 0.001140},
}

LOWER_IS_BETTER = {
    "m1_valid_brier",
    "m1_train_valid_auc_gap",
    "m2_valid_mae",
    "m2_valid_rmse",
}

# B6 clause membership.
PRIMARY_METRIC = "m1_train_valid_auc_gap"
GUARDRAIL_M1 = ("m1_valid_auc", "m1_valid_fail_ap", "m1_valid_brier")
GUARDRAIL_M2 = ("m2_valid_mae", "m2_valid_rmse", "m2_valid_r2")
SEGMENT_METRICS = ("cold_start_auc", "low_difficulty_support_auc", "level_1_auc")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return result.stdout.rstrip()


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


def find_run_dir(slug: str, arm: str) -> Path:
    """Locate the single persistent run folder for a configuration and arm."""
    pattern = f"*__{slug}-{arm.replace('_', '-')}"
    matches = sorted(RUNS.glob(pattern))
    if len(matches) != 1:
        raise AssertionError(
            f"Expected exactly one run folder matching {pattern!r}, found {matches}"
        )
    return matches[0]


def exact_m1(y: pd.Series, probability: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "fail_average_precision": float(average_precision_score(1 - y, 1 - probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "reporting_threshold": REPORTING_THRESHOLD,
    }


def exact_m2(y: pd.Series, prediction: np.ndarray) -> dict:
    residual = y.to_numpy(dtype=float) - prediction
    ss_res = float(np.square(residual).sum())
    centered = y.to_numpy(dtype=float) - float(y.mean())
    ss_tot = float(np.square(centered).sum())
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(1 - ss_res / ss_tot),
    }


def segment_masks(df: pd.DataFrame) -> dict:
    """The five EXISTING segments. Definitions unchanged; none added."""
    return {
        "first_semester": (df["is_first_active_semester"] == 1).to_numpy(),
        "cold_start_gpa": (df["no_previous_progress"] == 1).to_numpy(),
        "retake_attempt": (df["attempt_number"] > 1).to_numpy(),
        "low_difficulty_support": (df["difficulty_fallback_level"] >= 3).to_numpy(),
        "level_1_difficulty": (df["difficulty_fallback_level"] == 1).to_numpy(),
    }


def segment_metrics(y: pd.Series, probability: np.ndarray, masks: dict) -> dict:
    values = {}
    y_array = y.to_numpy()
    for name, mask in masks.items():
        segment_y = y_array[mask]
        values[name] = {
            "n": int(mask.sum()),
            "positive_rate": float(segment_y.mean()),
            "auc": float(roc_auc_score(segment_y, probability[mask])),
        }
    return values


def judge(metric_key: str, delta: float) -> str:
    band = NOISE_BAND[metric_key]
    if band["min"] <= delta <= band["max"]:
        return "inside_band"
    if metric_key in LOWER_IS_BETTER:
        return "outside_band_beneficial" if delta < band["min"] else "outside_band_harmful"
    return "outside_band_beneficial" if delta > band["max"] else "outside_band_harmful"


# ---------------------------------------------------------------------------
# Verification: everything must match the control except the single lever
# ---------------------------------------------------------------------------

def verify_against_control(run_dir: Path, control_dir: Path, config: dict, arm: str) -> dict:
    """Confirm a screening run differs from its control by the ONE lever only."""
    checks: dict[str, dict] = {}
    run_contract = load_json(run_dir / "feature_contract.json")
    ctl_contract = load_json(control_dir / "feature_contract.json")
    run_metrics = load_json(run_dir / "metrics.json")
    ctl_metrics = load_json(control_dir / "metrics.json")

    def record(name: str, ok: bool, run_value, control_value, note: str = "") -> None:
        checks[name] = {
            "match": bool(ok),
            "run": run_value,
            "control": control_value,
            **({"note": note} if note else {}),
        }

    # --- contract identity -------------------------------------------------
    record("feature_contract", run_contract["contract_name"] == ctl_contract["contract_name"]
           == arm, run_contract["contract_name"], ctl_contract["contract_name"])
    record("ordered_features", run_contract["ordered_features"] == ctl_contract["ordered_features"],
           run_contract["feature_count"], ctl_contract["feature_count"])
    record("categorical_levels", run_contract["categorical_levels"] == ctl_contract["categorical_levels"],
           "identical" if run_contract["categorical_levels"] == ctl_contract["categorical_levels"]
           else run_contract["categorical_levels"], "see run")
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
           run_contract["dataset_version"] == ctl_contract["dataset_version"] == VERSION,
           run_contract["dataset_version"], ctl_contract["dataset_version"])

    # Row counts: the older seed-42 baseline control predates data_rows in its
    # contract JSON. Identical file hashes already prove identical rows, so the
    # check falls back to that rather than being silently skipped.
    run_rows = run_contract.get("data_rows")
    ctl_rows = ctl_contract.get("data_rows")
    if ctl_rows is None:
        record("data_rows", run_inputs["train"]["sha256"] == ctl_inputs["train"]["sha256"]
               and run_inputs["valid"]["sha256"] == ctl_inputs["valid"]["sha256"],
               run_rows, "not recorded (predates the field)",
               "control predates data_rows; identical train/valid SHA-256 proves "
               "identical row counts")
    else:
        record("data_rows", run_rows == ctl_rows, run_rows, ctl_rows)

    # --- seeds -------------------------------------------------------------
    # Read from the serialized models, which every run has, including the ones
    # that predate effective_seed_settings in feature_contract.json.
    run_seeds = _effective_seed_settings(run_dir / "m1_pass_model.lgbm")
    ctl_seeds = _effective_seed_settings(control_dir / "m1_pass_model.lgbm")
    record("effective_seed_settings", run_seeds == ctl_seeds and run_seeds["seed"] == SEED,
           run_seeds, ctl_seeds)
    run_m2_seeds = _effective_seed_settings(run_dir / "m2_grade_model.lgbm")
    record("m1_m2_same_seeds", run_seeds == run_m2_seeds, run_m2_seeds, run_seeds)

    # --- the LightGBM parameter blocks, straight from the models ------------
    lever_alias = config["model_file_alias"]
    for model_file in ("m1_pass_model.lgbm", "m2_grade_model.lgbm"):
        run_params = model_file_params(run_dir / model_file)
        ctl_params = model_file_params(control_dir / model_file)
        differing = sorted(
            key for key in set(run_params) | set(ctl_params)
            if run_params.get(key) != ctl_params.get(key)
        )
        record(f"{model_file}_only_lever_differs", differing == [lever_alias],
               differing or "none", f"expected exactly ['{lever_alias}']")
        record(f"{model_file}_lever_value",
               float(run_params[lever_alias]) == config["value"]
               and float(ctl_params[lever_alias]) == config["control_value"],
               float(run_params[lever_alias]), float(ctl_params[lever_alias]))
        # Mechanics that must hold regardless of the lever.
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
               "is verified above and its REPORT.md records the same 50-round "
               "VALID-only early stopping")

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

    failures = sorted(name for name, check in checks.items() if not check["match"])
    return {"checks": checks, "failed_checks": failures, "all_passed": not failures}


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

def collect() -> dict:
    collected: dict[str, dict] = {}
    verification: dict[str, dict] = {}

    for arm in ARMS:
        contract = resolve_feature_contract(arm)
        train = pd.read_parquet(TRAIN, columns=contract.training_data_columns)
        valid = pd.read_parquet(VALID, columns=contract.training_data_columns)
        # Train-only median fill, exactly as training did it.
        median = float(train["diploma_gpa"].median())
        train["diploma_gpa"] = train["diploma_gpa"].fillna(median)
        valid["diploma_gpa"] = valid["diploma_gpa"].fillna(median)
        levels = load_json(RUNS / CONTROLS[arm] / "feature_contract.json")["categorical_levels"]

        # X is identical for both targets; only y differs.
        x_train, y_train_pass = prepare_X_y(train, "pass", levels, contract)
        x_valid, y_valid_pass = prepare_X_y(valid, "pass", levels, contract)
        y_train_grade = train["final_mark"].astype(float)
        y_valid_grade = valid["final_mark"].astype(float)

        masks = segment_masks(valid)
        assert np.array_equal(masks["first_semester"], masks["cold_start_gpa"]), (
            "first_semester and cold_start_gpa populations stopped being identical"
        )
        del train, valid
        gc.collect()

        targets = [("control", RUNS / CONTROLS[arm], None)]
        for name, config in CONFIGS.items():
            targets.append((name, find_run_dir(config["slug"], arm), config))

        for label, run_dir, config in targets:
            stored = load_json(run_dir / "metrics.json")
            m1 = lgb.Booster(model_file=str(run_dir / "m1_pass_model.lgbm"))
            probability_train = m1.predict(x_train)
            probability_valid = m1.predict(x_valid)
            m1_train = exact_m1(y_train_pass, probability_train)
            m1_valid = exact_m1(y_valid_pass, probability_valid)
            m1_valid["train_valid_auc_gap"] = m1_train["roc_auc"] - m1_valid["roc_auc"]
            m1_valid["best_iteration"] = int(
                stored["m1_pass_classifier"]["valid"]["best_iteration"]
            )
            segments = segment_metrics(y_valid_pass, probability_valid, masks)
            del m1, probability_train, probability_valid
            gc.collect()

            m2 = lgb.Booster(model_file=str(run_dir / "m2_grade_model.lgbm"))
            m2_train = exact_m2(y_train_grade, m2.predict(x_train))
            m2_valid = exact_m2(y_valid_grade, m2.predict(x_valid))
            m2_valid["best_iteration"] = int(
                stored["m2_grade_regressor"]["valid"]["best_iteration"]
            )
            del m2
            gc.collect()

            collected[f"{label}_{arm}"] = {
                "configuration": label,
                "arm": arm,
                "run_path": str(run_dir.relative_to(ROOT)).replace("\\", "/"),
                "lever": None if config is None else config["label"],
                "m1": {"train": m1_train, "valid": m1_valid},
                "m2": {"train": m2_train, "valid": m2_valid},
                "valid_segments": segments,
                "hit_round_cap": {
                    "m1": m1_valid["best_iteration"] >= NUM_BOOST_ROUND,
                    "m2": m2_valid["best_iteration"] >= NUM_BOOST_ROUND,
                },
            }
            if config is not None:
                verification[f"{label}_{arm}"] = verify_against_control(
                    run_dir, RUNS / CONTROLS[arm], config, arm
                )
            print(f"  scored {label:<8} {arm:<14} {run_dir.name}")

        del x_train, x_valid, y_train_pass, y_valid_pass, y_train_grade, y_valid_grade
        gc.collect()

    return {"runs": collected, "verification": verification}


def metric_values(entry: dict) -> dict:
    """The ten band-comparable metrics for one run."""
    return {
        "m1_valid_auc": entry["m1"]["valid"]["roc_auc"],
        "m1_valid_fail_ap": entry["m1"]["valid"]["fail_average_precision"],
        "m1_valid_brier": entry["m1"]["valid"]["brier_score"],
        "m1_train_valid_auc_gap": entry["m1"]["valid"]["train_valid_auc_gap"],
        "m2_valid_mae": entry["m2"]["valid"]["mae"],
        "m2_valid_rmse": entry["m2"]["valid"]["rmse"],
        "m2_valid_r2": entry["m2"]["valid"]["r2"],
        "cold_start_auc": entry["valid_segments"]["first_semester"]["auc"],
        "low_difficulty_support_auc": entry["valid_segments"]["low_difficulty_support"]["auc"],
        "level_1_auc": entry["valid_segments"]["level_1_difficulty"]["auc"],
    }


def evaluate(collected: dict) -> dict:
    """Apply the pre-registered B6 rule. No threshold is invented here."""
    results: dict[str, dict] = {}
    for name, config in CONFIGS.items():
        arms_result: dict[str, dict] = {}
        for arm in ARMS:
            control = metric_values(collected["runs"][f"control_{arm}"])
            candidate = metric_values(collected["runs"][f"{name}_{arm}"])
            deltas = {
                key: candidate[key] - control[key] for key in NOISE_BAND
            }
            judgments = {key: judge(key, value) for key, value in deltas.items()}
            arms_result[arm] = {
                "control": control,
                "candidate": candidate,
                "deltas": deltas,
                "judgments": judgments,
            }

        # --- the three pre-registered clauses ------------------------------
        primary_ok = all(
            arms_result[arm]["judgments"][PRIMARY_METRIC] == "outside_band_beneficial"
            for arm in ARMS
        )
        guardrail_m1_breaches = [
            f"{arm}:{key}"
            for arm in ARMS for key in GUARDRAIL_M1
            if arms_result[arm]["judgments"][key] == "outside_band_harmful"
        ]
        guardrail_m2_breaches = [
            f"{arm}:{key}"
            for arm in ARMS for key in GUARDRAIL_M2
            if arms_result[arm]["judgments"][key] == "outside_band_harmful"
        ]
        clauses = {
            "clause_1_primary_gap_outside_band_beneficial_in_both_arms": {
                "satisfied": primary_ok,
                "per_arm": {
                    arm: arms_result[arm]["judgments"][PRIMARY_METRIC] for arm in ARMS
                },
            },
            "clause_2_guardrail_m1": {
                "satisfied": not guardrail_m1_breaches,
                "breaches": guardrail_m1_breaches,
            },
            "clause_3_guardrail_m2": {
                "satisfied": not guardrail_m2_breaches,
                "breaches": guardrail_m2_breaches,
            },
        }
        failing = [key for key, value in clauses.items() if not value["satisfied"]]
        verdict = "PASS" if not failing else "FAIL"

        # Did M1 and M2 move together? M1 read on VALID AUC, M2 on VALID MAE.
        same_direction = {}
        for arm in ARMS:
            m1_improved = arms_result[arm]["deltas"]["m1_valid_auc"] > 0
            m2_improved = arms_result[arm]["deltas"]["m2_valid_mae"] < 0
            same_direction[arm] = {
                "m1_valid_auc_improved": m1_improved,
                "m2_valid_mae_improved": m2_improved,
                "same_direction": m1_improved == m2_improved,
            }

        results[name] = {
            "label": config["label"],
            "lever": config["param"],
            "arms": arms_result,
            "clauses": clauses,
            "failing_clauses": failing,
            "verdict": verdict,
            "m1_m2_direction": same_direction,
        }
    return results


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

BAND_LIMITATION = (
    "**Stated limitation.** `NOISE_BAND.md` was measured from CONTRACT-change "
    "deltas across five seeds, not from HYPERPARAMETER-change deltas. It is the "
    "best available yardstick for this pass, not an exact one. Do not treat it "
    "as precise."
)

CLAUSE_NAMES = {
    "clause_1_primary_gap_outside_band_beneficial_in_both_arms":
        "clause 1 (PRIMARY: M1 train-valid AUC gap outside band, beneficial, in BOTH arms)",
    "clause_2_guardrail_m1": "clause 2 (GUARDRAIL M1)",
    "clause_3_guardrail_m2": "clause 3 (GUARDRAIL M2)",
}


def render(collected: dict, evaluation: dict, provenance: dict) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Regularization screening — seed 42, four single-lever configurations")
    add("")
    add("Screening only: seed 42, four configurations, both arms = 8 runs. Seeds "
        "52/62/72/82 were NOT run; confirmation is a separate approved task.")
    add("")
    add("Acceptance rule pre-registered in "
        "[`docs/EXPERIMENT_REGULARIZATION_PLAN.md`](../../docs/EXPERIMENT_REGULARIZATION_PLAN.md), "
        "committed before any of these runs existed. No threshold was invented "
        "after seeing results.")
    add("")
    add("Paired delta = **configuration minus same-contract control**, both at seed 42.")
    add("")
    add(BAND_LIMITATION)
    add("")

    # --- verdicts ----------------------------------------------------------
    add("## Verdicts")
    add("")
    for name in CONFIGS:
        result = evaluation[name]
        if result["verdict"] == "PASS":
            add(f"- **{name} ({result['label']}): PASS** — all three "
                "pre-registered clauses satisfied in both arms.")
        else:
            named = "; ".join(CLAUSE_NAMES[c] for c in result["failing_clauses"])
            add(f"- **{name} ({result['label']}): FAIL** — failing: {named}.")
    add("")
    passing = [n for n in CONFIGS if evaluation[n]["verdict"] == "PASS"]
    if passing:
        add(f"Candidate(s) for five-seed confirmation: {', '.join(passing)}.")
    else:
        add("**No configuration passes screening.** Per the pre-registered rule "
            "this is a legitimate result: the current parameters stand and M1 "
            "freezes as-is. No candidate goes forward to five-seed confirmation.")
    add("")

    # --- how to read the verdicts -----------------------------------------
    add("## How to read these verdicts")
    add("")
    add("This is ONE seed. Nothing here is confirmed; screening selects what is "
        "worth spending five seeds on, and nothing more. The caveats below are "
        "reported because they are true, not to reopen the locked rule — no "
        "verdict above was adjusted after the numbers were seen.")
    add("")

    # Out-of-band harmful movement on metrics that are NOT clauses of B6.
    non_clause_harmful = []
    for name in CONFIGS:
        for arm in ARMS:
            for key in SEGMENT_METRICS:
                if evaluation[name]["arms"][arm]["judgments"][key] == "outside_band_harmful":
                    non_clause_harmful.append(
                        (name, arm, key, evaluation[name]["arms"][arm]["deltas"][key])
                    )
    if non_clause_harmful:
        add("**Out-of-band harmful movement on metrics the rule does not score.** "
            "The pre-registered clauses cover the M1 gap, the three M1 VALID "
            "guardrails and the three M2 VALID guardrails. Segment AUCs are "
            "reported but are explicitly NOT clauses, so the following did not "
            "and must not change any verdict — they are flagged for the "
            "confirmation task to watch:")
        add("")
        for name, arm, key, delta in non_clause_harmful:
            verdict = evaluation[name]["verdict"]
            add(f"- {name} · {arm} · `{key}` delta {delta:+.6f} "
                f"(band {NOISE_BAND[key]['min']:+.6f} … {NOISE_BAND[key]['max']:+.6f}) "
                f"— configuration verdict remains **{verdict}**.")
        add("")

    # For every passing configuration, decompose the gap movement.
    for name in passing:
        add(f"**{name}: what actually moved.** A gap shrinks either because "
            "VALID improved or because TRAIN came down. Guardrail 2 exists to "
            "reject the case where the gap closed by a VALID collapse; the "
            "decomposition per arm:")
        add("")
        add("| Arm | TRAIN AUC delta | VALID AUC delta | Gap delta | Mechanism |")
        add("|---|---:|---:|---:|---|")
        for arm in ARMS:
            entry = collected["runs"][f"{name}_{arm}"]
            ctl = collected["runs"][f"control_{arm}"]
            train_delta = entry["m1"]["train"]["roc_auc"] - ctl["m1"]["train"]["roc_auc"]
            valid_delta = evaluation[name]["arms"][arm]["deltas"]["m1_valid_auc"]
            gap_delta = evaluation[name]["arms"][arm]["deltas"]["m1_train_valid_auc_gap"]
            if valid_delta > 0 and train_delta < 0:
                mechanism = "TRAIN down **and** VALID up — genuine generalization gain"
            elif valid_delta > 0:
                mechanism = "VALID up — generalization gain"
            elif abs(train_delta) > abs(valid_delta) * 4:
                mechanism = ("mostly TRAIN coming down; VALID roughly held "
                             "(no collapse, so guardrail 2 is not breached)")
            else:
                mechanism = "TRAIN and VALID moved comparably"
            add(f"| {arm} | {train_delta:+.6f} | {valid_delta:+.6f} | "
                f"{gap_delta:+.6f} | {mechanism} |")
        add("")
        # How close did the tightest M1 guardrail come to its harmful edge?
        worst = None
        for arm in ARMS:
            for key in GUARDRAIL_M1:
                delta = evaluation[name]["arms"][arm]["deltas"][key]
                band = NOISE_BAND[key]
                # Distance from the harmful edge: the max for lower-is-better
                # metrics, the min for higher-is-better ones.
                margin = (
                    band["max"] - delta if key in LOWER_IS_BETTER
                    else delta - band["min"]
                )
                if worst is None or margin < worst[2]:
                    worst = (arm, key, margin, delta)
        if worst is not None:
            add(f"Tightest M1 guardrail margin: `{worst[1]}` in {worst[0]}, delta "
                f"{worst[3]:+.6f}, only {worst[2]:.6f} inside the harmful edge. "
                "Inside the band is inside the band — but this is close enough "
                "that a second seed could land the other side of it.")
            add("")
        # M2 direction under a passing configuration.
        m2_worse = [
            arm for arm in ARMS
            if evaluation[name]["arms"][arm]["deltas"]["m2_valid_mae"] > 0
        ]
        if m2_worse:
            add(f"M2 VALID MAE worsened in: {', '.join(m2_worse)} (inside the band, "
                "so guardrail 3 is not breached). Because `_SHARED_PARAMS` is "
                "shared, this configuration cannot help M1 without also moving "
                "M2. Per B3 that is a finding to report, not a licence to split "
                "the parameters per model — that architectural decision is not "
                "made here.")
            add("")

    # --- controls ----------------------------------------------------------
    add("## Controls (not retrained)")
    add("")
    for arm in ARMS:
        add(f"- `{arm}` seed 42: `models/runs/{CONTROLS[arm]}`")
    add("")
    add("## The eight screening runs")
    add("")
    add("| Config | Lever | Arm | Run path |")
    add("|---|---|---|---|")
    for name, config in CONFIGS.items():
        for arm in ARMS:
            entry = collected["runs"][f"{name}_{arm}"]
            add(f"| {name} | {config['label']} | {arm} | `{entry['run_path']}` |")
    add("")

    # --- verification ------------------------------------------------------
    add("## Control-parity verification")
    add("")
    add("Each run checked against its same-contract control: contract identity "
        "and ordered features, categorical levels, reporting threshold, test "
        "policy, dataset version and TRAIN/VALID SHA-256, row counts, effective "
        "seeds (read from the serialized models), M1/M2 seed equality, the full "
        "serialized LightGBM parameter block for both models, boost-round cap, "
        "threads, early stopping, and diploma-GPA fill.")
    add("")
    add("| Config | Arm | Checks | Result | Failed |")
    add("|---|---|---:|:---:|---|")
    for name in CONFIGS:
        for arm in ARMS:
            v = collected["verification"][f"{name}_{arm}"]
            status = "PASS" if v["all_passed"] else "**FAIL**"
            failed = ", ".join(v["failed_checks"]) or "none"
            add(f"| {name} | {arm} | {len(v['checks'])} | {status} | {failed} |")
    add("")
    add("The only serialized LightGBM parameter that differs between any run and "
        "its control is that run's single lever — verified independently for "
        "M1 and M2 in every one of the eight runs.")
    add("")
    add("Two checks are satisfied by inference rather than by direct JSON "
        "equality, because the seed-42 `baseline_41` control predates the "
        "`data_rows`, `training_control` and `diploma_gpa_handling` fields "
        "(the provenance caveat already recorded in `Decisions_Log.md`). Row "
        "counts follow from identical TRAIN/VALID SHA-256; the diploma fill is "
        "the TRAIN median of an identically hashed TRAIN file and is therefore "
        "deterministic and equal. Both are labelled with a note in the JSON.")
    add("")

    # --- best iteration ----------------------------------------------------
    add("## Best iteration (round cap = 2000)")
    add("")
    add("| Config | Arm | M1 best_iter | M1 vs control | M2 best_iter | M2 vs control | Hit cap |")
    add("|---|---|---:|---:|---:|---:|:---:|")
    capped = []
    for name in list(CONFIGS) + ["control"]:
        for arm in ARMS:
            entry = collected["runs"][f"{name}_{arm}"]
            ctl = collected["runs"][f"control_{arm}"]
            m1_it = entry["m1"]["valid"]["best_iteration"]
            m2_it = entry["m2"]["valid"]["best_iteration"]
            m1_d = "—" if name == "control" else f"{m1_it - ctl['m1']['valid']['best_iteration']:+d}"
            m2_d = "—" if name == "control" else f"{m2_it - ctl['m2']['valid']['best_iteration']:+d}"
            hit = entry["hit_round_cap"]["m1"] or entry["hit_round_cap"]["m2"]
            if hit:
                capped.append(f"{name}/{arm}")
            add(f"| {name} | {arm} | {m1_it} | {m1_d} | {m2_it} | {m2_d} | "
                f"{'**YES**' if hit else 'no'} |")
    add("")
    if capped:
        add(f"**FLAG — round cap reached:** {', '.join(capped)}. Early stopping "
            "never fired there, so that run is truncated, not converged, and its "
            "comparison must be read as such.")
    else:
        add("No run reached the 2000-round cap: early stopping fired in all "
            "eight screening runs and both controls, so every comparison is "
            "between converged models.")
    add("")

    # --- per-configuration detail -----------------------------------------
    for name, config in CONFIGS.items():
        result = evaluation[name]
        add(f"## {name} — {config['label']}")
        add("")
        add(f"Verdict: **{result['verdict']}**"
            + ("" if result["verdict"] == "PASS"
               else " — failing " + "; ".join(
                   CLAUSE_NAMES[c] for c in result["failing_clauses"])))
        add("")
        for arm in ARMS:
            block = result["arms"][arm]
            entry = collected["runs"][f"{name}_{arm}"]
            ctl = collected["runs"][f"control_{arm}"]
            add(f"### {name} · {arm}")
            add("")
            add("M1 TRAIN AUC is shown beside the gap so a gap that shrank only "
                "because TRAIN collapsed is visible.")
            add("")
            add("| Metric | Control | Config | Delta | Band min | Band max | Judgment | B6 role |")
            add("|---|---:|---:|---:|---:|---:|:---|:---|")
            add(f"| M1 TRAIN AUC | {ctl['m1']['train']['roc_auc']:.6f} | "
                f"{entry['m1']['train']['roc_auc']:.6f} | "
                f"{entry['m1']['train']['roc_auc'] - ctl['m1']['train']['roc_auc']:+.6f} "
                "| — | — | context | reported, not a clause |")
            roles = {
                PRIMARY_METRIC: "**PRIMARY (clause 1)**",
                **{k: "guardrail (clause 2)" for k in GUARDRAIL_M1},
                **{k: "guardrail (clause 3)" for k in GUARDRAIL_M2},
                **{k: "reported, not a clause" for k in SEGMENT_METRICS},
            }
            order = [
                "m1_train_valid_auc_gap", "m1_valid_auc", "m1_valid_fail_ap",
                "m1_valid_brier", "m2_valid_mae", "m2_valid_rmse", "m2_valid_r2",
                "cold_start_auc", "low_difficulty_support_auc", "level_1_auc",
            ]
            for key in order:
                band = NOISE_BAND[key]
                add(f"| {key} | {block['control'][key]:.6f} | "
                    f"{block['candidate'][key]:.6f} | {block['deltas'][key]:+.6f} | "
                    f"{band['min']:+.6f} | {band['max']:+.6f} | "
                    f"{block['judgments'][key]} | {roles[key]} |")
            add("")
            add(f"M2 TRAIN MAE {ctl['m2']['train']['mae']:.6f} -> "
                f"{entry['m2']['train']['mae']:.6f} "
                f"({entry['m2']['train']['mae'] - ctl['m2']['train']['mae']:+.6f}); "
                f"TRAIN RMSE {ctl['m2']['train']['rmse']:.6f} -> "
                f"{entry['m2']['train']['rmse']:.6f}; "
                f"TRAIN R2 {ctl['m2']['train']['r2']:.6f} -> "
                f"{entry['m2']['train']['r2']:.6f}.")
            add("")
            direction = result["m1_m2_direction"][arm]
            add("M1/M2 direction: M1 VALID AUC "
                f"{'improved' if direction['m1_valid_auc_improved'] else 'worsened'}, "
                "M2 VALID MAE "
                f"{'improved' if direction['m2_valid_mae_improved'] else 'worsened'} — "
                f"**{'same direction' if direction['same_direction'] else 'OPPOSED'}**.")
            add("")

        # segments
        add(f"### {name} — segment AUCs (VALID)")
        add("")
        add("`first_semester` and `cold_start_gpa` are the SAME population "
            "(n=14,732, open defect) — ONE piece of evidence, not two.")
        add("")
        add("| Arm | Segment | n | Control AUC | Config AUC | Delta |")
        add("|---|---|---:|---:|---:|---:|")
        for arm in ARMS:
            entry = collected["runs"][f"{name}_{arm}"]
            ctl = collected["runs"][f"control_{arm}"]
            for segment in ("first_semester", "cold_start_gpa", "retake_attempt",
                            "low_difficulty_support", "level_1_difficulty"):
                c = ctl["valid_segments"][segment]
                k = entry["valid_segments"][segment]
                add(f"| {arm} | {segment} | {k['n']} | {c['auc']:.6f} | "
                    f"{k['auc']:.6f} | {k['auc'] - c['auc']:+.6f} |")
        add("")

    # --- direction summary -------------------------------------------------
    add("## Did M1 and M2 move together?")
    add("")
    add("| Config | Arm | M1 VALID AUC | M2 VALID MAE | Same direction |")
    add("|---|---|:---|:---|:---:|")
    for name in CONFIGS:
        for arm in ARMS:
            d = evaluation[name]["m1_m2_direction"][arm]
            add(f"| {name} | {arm} | "
                f"{'improved' if d['m1_valid_auc_improved'] else 'worsened'} | "
                f"{'improved' if d['m2_valid_mae_improved'] else 'worsened'} | "
                f"{'yes' if d['same_direction'] else '**NO**'} |")
    add("")
    add("Every configuration moves `_SHARED_PARAMS`, so M1 and M2 always change "
        "together. Where they move in opposite directions, that is the "
        "finding B3 asks to report: per-model parameters would be a new "
        "architectural divergence and that decision is not made here.")
    add("")

    # --- integrity ---------------------------------------------------------
    add("## Integrity confirmations")
    add("")
    add(f"- TEST is `closed_not_read` in all eight runs; every M1/M2 `test` "
        "metric field is null; each run passed a NONEXISTENT `--test` path "
        f"(`{provenance['test_path']}`, exists={provenance['test_path_exists']}), "
        "so completing at all proves TEST was never opened. `--evaluate-test` "
        "was never passed.")
    add(f"- TRAIN SHA-256 `{provenance['train_sha256'][:16]}…`, VALID SHA-256 "
        f"`{provenance['valid_sha256'][:16]}…`, identical across all ten runs "
        "(eight screening + two controls).")
    add(f"- Dataset version `{VERSION}`; TRAIN {provenance['train_rows']:,} rows, "
        f"VALID {provenance['valid_rows']:,} rows. No dataset was copied or moved.")
    add("- Both controls were reused unchanged; nothing was retrained.")
    add("- Every metric above was recomputed by re-scoring the saved models "
        "against TRAIN/VALID. Only `best_iteration` is read from each run's "
        "`metrics.json`. `level_1_difficulty` is not stored in `metrics.json` "
        "and exists only because it is recomputed here.")
    add("- No `CURRENT_VERSION.txt`, promotion marker, live model artifact, "
        "default parameter, inference wiring, or recommendation wiring was "
        "changed.")
    add("")
    add(f"Generated at commit `{provenance['git_commit']}` "
        f"(working tree clean: {provenance['git_clean']}).")
    return "\n".join(lines) + "\n"


def main() -> None:
    print("Collecting and re-scoring runs …")
    collected = collect()
    evaluation = evaluate(collected)

    test_path = ROOT / f"data/model_data/versions/{VERSION}/df_test_CLOSED_DO_NOT_READ.parquet"
    provenance = {
        "train_sha256": sha256(TRAIN),
        "valid_sha256": sha256(VALID),
        "train_rows": int(pd.read_parquet(TRAIN, columns=["final_mark"]).shape[0]),
        "valid_rows": int(pd.read_parquet(VALID, columns=["final_mark"]).shape[0]),
        "test_path": str(test_path.relative_to(ROOT)).replace("\\", "/"),
        "test_path_exists": test_path.exists(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_clean": git("status", "--porcelain") == "",
        "dataset_version": VERSION,
        "seed": SEED,
    }
    assert not provenance["test_path_exists"], "The --test path must not exist."

    payload = {
        "experiment": "regularization_screening_seed42",
        "pre_registered_plan": "docs/EXPERIMENT_REGULARIZATION_PLAN.md",
        "noise_band_source": "models/runs/NOISE_BAND.md",
        "noise_band_limitation": (
            "Band measured from contract-change deltas across seeds, not from "
            "hyperparameter-change deltas. Best available yardstick, not exact."
        ),
        "noise_band": NOISE_BAND,
        "controls": CONTROLS,
        "configurations": CONFIGS,
        "provenance": provenance,
        "runs": collected["runs"],
        "verification": collected["verification"],
        "evaluation": evaluation,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(collected, evaluation, provenance), encoding="utf-8")
    print(f"\nWrote {OUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")
    for name in CONFIGS:
        result = evaluation[name]
        failing = ", ".join(result["failing_clauses"]) or "none"
        print(f"  {name} ({result['label']}): {result['verdict']}  failing={failing}")


if __name__ == "__main__":
    main()
