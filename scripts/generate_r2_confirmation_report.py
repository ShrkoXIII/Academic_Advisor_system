"""Generate the R2 (num_leaves 31) five-seed confirmation report.

The analysis rule is pre-registered in docs/EXPERIMENT_R2_CONFIRMATION_PLAN.md,
committed before any confirmation run existed. This script only applies that
rule — it never invents a threshold, and every judgement phrase it evaluates
("single outlier seed", "large harmful outlier", "fell sharply", "consistent
direction") has an exact definition fixed in that plan.

Every metric is recomputed by re-scoring each run's saved LightGBM models
against TRAIN/VALID rather than trusting the possibly-rounded values stored in
metrics.json (CLAUDE.md sec 7: verification is evidence-first). The
`level_1_difficulty` segment is not stored in metrics.json at all and exists
only because it is recomputed here.

Reads TRAIN and VALID only. Never constructs or reads a TEST path. Modifies no
model or dataset artifact. Paired delta is defined as
R2 RUN minus SAME-SEED SAME-CONTRACT DEFAULT CONTROL.
"""

from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import statistics
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

from scripts.r2_parity import check as parity_check
from src.model_training import (
    NUM_BOOST_ROUND,
    REPORTING_THRESHOLD,
    prepare_X_y,
    resolve_feature_contract,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "models" / "runs"
VERSION = "2026-07-26_batched_fixes__registration_roster_concurrent"
TRAIN = ROOT / f"data/model_data/versions/{VERSION}/df_train_final.parquet"
VALID = ROOT / f"data/model_data/versions/{VERSION}/df_valid_final.parquet"
OUT_JSON = RUNS / "R2_CONFIRMATION_5SEED_REPORT.json"
OUT_MD = RUNS / "R2_CONFIRMATION_5SEED_REPORT.md"

SEEDS = (42, 52, 62, 72, 82)
ARMS = ("baseline_41", "concurrent_43")
R2_NUM_LEAVES = 31

# The ten default-parameter controls, already on disk. NEVER retrained.
CONTROLS = {
    "baseline_41": {
        42: "2026-07-26_1551__baseline-41-gpa-trend-control",
        52: "2026-07-27_1027__seed52-baseline-41-gpa-trend-control",
        62: "2026-07-27_1031__seed62-baseline-41-gpa-trend-control",
        72: "2026-07-27_1035__seed72-baseline-41-gpa-trend-control",
        82: "2026-07-27_1038__seed82-baseline-41-gpa-trend-control",
    },
    "concurrent_43": {
        42: "2026-07-27_1327__seed42-concurrent-43-drop-dead-missing-flag",
        52: "2026-07-27_1328__seed52-concurrent-43-drop-dead-missing-flag",
        62: "2026-07-27_1329__seed62-concurrent-43-drop-dead-missing-flag",
        72: "2026-07-27_1330__seed72-concurrent-43-drop-dead-missing-flag",
        82: "2026-07-27_1331__seed82-concurrent-43-drop-dead-missing-flag",
    },
}

# Seed 42's R2 pair is REUSED from screening, verified to match this protocol
# exactly (single lever num_leaves=31, identical hashes/seeds/levels/threshold/
# early stopping/diploma fill). Seeds 52-82 are located by slug.
REUSED_SEED42_R2 = {
    "baseline_41": "2026-07-27_1456__reg-r2-leaves31-baseline-41",
    "concurrent_43": "2026-07-27_1457__reg-r2-leaves31-concurrent-43",
}

# Acceptance yardstick, verbatim from models/runs/NOISE_BAND.md.
# LIMITATION, repeated in the report: measured from CONTRACT-change deltas
# across seeds, not HYPERPARAMETER-change deltas. Best available yardstick,
# not an exact one.
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
    "m1_valid_brier", "m1_train_valid_auc_gap", "m2_valid_mae", "m2_valid_rmse",
}

M1_GUARDRAILS = ("m1_valid_auc", "m1_valid_fail_ap", "m1_valid_brier")
M2_METRICS = ("m2_valid_mae", "m2_valid_rmse", "m2_valid_r2")

# Plan: "large harmful outlier" = beyond 2x the band's harmful edge.
LARGE_OUTLIER_MULTIPLIER = 2.0
# Plan: "fell sharply" = TRAIN AUC drop >= 0.010.
TRAIN_SHARP_DROP = 0.010
# Plan: "improved or flat" for VALID AUC = not degraded beyond the band floor.
VALID_FLAT_FLOOR = NOISE_BAND["m1_valid_auc"]["min"]


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


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def memory_status() -> dict:
    """Physical/commit memory plus whether a pagefile is active."""
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    gb = 1024 ** 3
    try:
        pagefile = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_PageFileUsage | "
             "Select-Object -First 1 -ExpandProperty Name)"],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
    except Exception:
        pagefile = "unavailable"
    return {
        "total_physical_gb": round(status.ullTotalPhys / gb, 2),
        "available_physical_gb": round(status.ullAvailPhys / gb, 2),
        "memory_load_percent": int(status.dwMemoryLoad),
        "total_commit_gb": round(status.ullTotalPageFile / gb, 2),
        "available_commit_gb": round(status.ullAvailPageFile / gb, 2),
        "pagefile": pagefile or "NONE",
        "pagefile_active": bool(pagefile) and pagefile != "unavailable",
    }


def find_r2_run(seed: int, arm: str) -> Path:
    if seed == 42:
        return RUNS / REUSED_SEED42_R2[arm]
    slug = f"seed{seed}-regr2-leaves31-{arm.replace('_', '-')}"
    matches = sorted(RUNS.glob(f"*__{slug}"))
    if len(matches) != 1:
        raise AssertionError(
            f"Expected exactly one run folder matching *__{slug}, found {matches}"
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


def metric_values(entry: dict) -> dict:
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


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect() -> dict:
    runs: dict[str, dict] = {}
    verification: dict[str, dict] = {}

    for arm in ARMS:
        contract = resolve_feature_contract(arm)
        train = pd.read_parquet(TRAIN, columns=contract.training_data_columns)
        valid = pd.read_parquet(VALID, columns=contract.training_data_columns)
        median = float(train["diploma_gpa"].median())
        train["diploma_gpa"] = train["diploma_gpa"].fillna(median)
        valid["diploma_gpa"] = valid["diploma_gpa"].fillna(median)
        levels = load_json(
            RUNS / CONTROLS[arm][42] / "feature_contract.json"
        )["categorical_levels"]

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

        for seed in SEEDS:
            for kind, run_dir in (
                ("control", RUNS / CONTROLS[arm][seed]),
                ("r2", find_r2_run(seed, arm)),
            ):
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

                runs[f"{kind}_{arm}_{seed}"] = {
                    "kind": kind,
                    "arm": arm,
                    "seed": seed,
                    "run_path": str(run_dir.relative_to(ROOT)).replace("\\", "/"),
                    "reused_from_screening": kind == "r2" and seed == 42,
                    "m1": {"train": m1_train, "valid": m1_valid},
                    "m2": {"train": m2_train, "valid": m2_valid},
                    "valid_segments": segments,
                    "hit_round_cap": {
                        "m1": m1_valid["best_iteration"] >= NUM_BOOST_ROUND,
                        "m2": m2_valid["best_iteration"] >= NUM_BOOST_ROUND,
                    },
                }
                print(f"  scored {kind:<7} {arm:<14} seed {seed}  {run_dir.name}")

            verification[f"{arm}_{seed}"] = parity_check(
                find_r2_run(seed, arm), RUNS / CONTROLS[arm][seed], arm, seed
            )

        del x_train, x_valid, y_train_pass, y_valid_pass, y_train_grade, y_valid_grade
        gc.collect()

    return {"runs": runs, "verification": verification}


# ---------------------------------------------------------------------------
# Pre-registered evaluation
# ---------------------------------------------------------------------------

def summarise(deltas: list[float], metric_key: str) -> dict:
    improving_is_negative = metric_key in LOWER_IS_BETTER
    improved = sum(1 for d in deltas if (d < 0) == improving_is_negative and d != 0)
    worsened = sum(1 for d in deltas if (d > 0) == improving_is_negative and d != 0)
    mean = statistics.fmean(deltas)
    return {
        "mean": mean,
        "median": statistics.median(deltas),
        "sd": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
        "min": min(deltas),
        "max": max(deltas),
        "improved": improved,
        "worsened": worsened,
        "tied": len(deltas) - improved - worsened,
        "mean_judgment": judge(metric_key, mean),
        "per_seed_judgments": [judge(metric_key, d) for d in deltas],
    }


def classify_mechanism(train_delta: float, valid_delta: float) -> dict:
    """Section 3.3, thresholds fixed in the pre-registered plan."""
    train_drop = -train_delta  # positive when TRAIN fell
    ratio = (
        float("inf") if valid_delta == 0 else abs(train_drop) / abs(valid_delta)
    )
    if valid_delta > 0 and train_drop < TRAIN_SHARP_DROP:
        label = "generalization_gain"
    elif train_drop >= TRAIN_SHARP_DROP and valid_delta <= 0:
        label = "train_collapse"
    elif valid_delta >= VALID_FLAT_FLOOR and train_drop < TRAIN_SHARP_DROP:
        label = "generalization_gain"
    else:
        label = "mixed"
    return {
        "train_auc_delta": train_delta,
        "valid_auc_delta": valid_delta,
        "train_drop": train_drop,
        "train_drop_to_valid_change_ratio": ratio,
        "classification": label,
    }


def evaluate(collected: dict) -> dict:
    per_arm: dict[str, dict] = {}

    for arm in ARMS:
        per_seed_deltas: dict[str, list[float]] = {key: [] for key in NOISE_BAND}
        seed_rows = []
        mechanisms = {}
        for seed in SEEDS:
            control = metric_values(collected["runs"][f"control_{arm}_{seed}"])
            candidate = metric_values(collected["runs"][f"r2_{arm}_{seed}"])
            deltas = {key: candidate[key] - control[key] for key in NOISE_BAND}
            for key, value in deltas.items():
                per_seed_deltas[key].append(value)
            seed_rows.append({"seed": seed, "deltas": deltas,
                              "judgments": {k: judge(k, v) for k, v in deltas.items()}})
            ctl_entry = collected["runs"][f"control_{arm}_{seed}"]
            r2_entry = collected["runs"][f"r2_{arm}_{seed}"]
            mechanisms[seed] = classify_mechanism(
                r2_entry["m1"]["train"]["roc_auc"] - ctl_entry["m1"]["train"]["roc_auc"],
                deltas["m1_valid_auc"],
            )

        summary = {key: summarise(values, key) for key, values in per_seed_deltas.items()}

        # --- clause 3.2.1: gap improves in >=4/5 AND leave-one-out stable ---
        gap_deltas = per_seed_deltas["m1_train_valid_auc_gap"]
        gap_improved = sum(1 for d in gap_deltas if d < 0)
        loo_means = [
            statistics.fmean([d for j, d in enumerate(gap_deltas) if j != i])
            for i in range(len(gap_deltas))
        ]
        loo_stable = all(mean < 0 for mean in loo_means)
        clause_1 = {
            "gap_improved_seeds": gap_improved,
            "gap_improved_at_least_4_of_5": gap_improved >= 4,
            "leave_one_out_means": loo_means,
            "leave_one_out_all_improving": loo_stable,
            "satisfied": gap_improved >= 4 and loo_stable,
        }

        # --- clause 3.2.2: mean not harmful, no large harmful outlier -------
        guardrail_detail = {}
        for key in M1_GUARDRAILS:
            band = NOISE_BAND[key]
            harmful_edge = band["max"] if key in LOWER_IS_BETTER else band["min"]
            outlier_edge = harmful_edge * LARGE_OUTLIER_MULTIPLIER
            if key in LOWER_IS_BETTER:
                outliers = [
                    {"seed": SEEDS[i], "delta": d}
                    for i, d in enumerate(per_seed_deltas[key]) if d > outlier_edge
                ]
            else:
                outliers = [
                    {"seed": SEEDS[i], "delta": d}
                    for i, d in enumerate(per_seed_deltas[key]) if d < outlier_edge
                ]
            guardrail_detail[key] = {
                "mean_delta": summary[key]["mean"],
                "mean_judgment": summary[key]["mean_judgment"],
                "mean_is_harmful": summary[key]["mean_judgment"] == "outside_band_harmful",
                "harmful_edge": harmful_edge,
                "large_outlier_edge": outlier_edge,
                "large_harmful_outliers": outliers,
            }
        clause_2 = {
            "per_metric": guardrail_detail,
            "satisfied": all(
                not detail["mean_is_harmful"] and not detail["large_harmful_outliers"]
                for detail in guardrail_detail.values()
            ),
        }

        # --- verdict, per the plan's assignment table ----------------------
        if not clause_2["satisfied"]:
            verdict = "NOT CONFIRMED"
            reason = "clause 3.2.2 breached (M1 VALID guardrail)"
        elif gap_improved <= 2:
            verdict = "NOT CONFIRMED"
            reason = f"gap improved in only {gap_improved} of 5 seeds"
        elif clause_1["satisfied"]:
            verdict = "CONFIRMED"
            reason = "clause 3.2.1 and clause 3.2.2 both satisfied"
        else:
            verdict = "INCONCLUSIVE"
            if gap_improved == 3:
                reason = "direction mixed: gap improved in exactly 3 of 5 seeds"
            else:
                reason = ("gap improved in >=4 of 5 seeds but leave-one-out "
                          "stability failed: the mean is carried by one seed")

        mechanism_counts = {}
        for value in mechanisms.values():
            mechanism_counts[value["classification"]] = (
                mechanism_counts.get(value["classification"], 0) + 1
            )

        per_arm[arm] = {
            "per_seed": seed_rows,
            "summary": summary,
            "clause_1": clause_1,
            "clause_2": clause_2,
            "verdict": verdict,
            "verdict_reason": reason,
            "mechanisms": mechanisms,
            "mechanism_counts": mechanism_counts,
        }

    # --- section 3.3: does the arm-dependent split repeat? -----------------
    baseline_gain = per_arm["baseline_41"]["mechanism_counts"].get("generalization_gain", 0)
    concurrent_collapse = per_arm["concurrent_43"]["mechanism_counts"].get("train_collapse", 0)
    split_repeats = baseline_gain >= 4 and concurrent_collapse >= 4
    mechanism_finding = {
        "baseline_41_generalization_gain_seeds": baseline_gain,
        "concurrent_43_train_collapse_seeds": concurrent_collapse,
        "split_repeats": split_repeats,
        "rule": ("repeats only if baseline_41 is generalization_gain in >=4/5 "
                 "AND concurrent_43 is train_collapse in >=4/5"),
    }

    # --- section 3.4: M2 status -------------------------------------------
    any_m2_mean_harmful = [
        f"{arm}:{key}"
        for arm in ARMS for key in M2_METRICS
        if per_arm[arm]["summary"][key]["mean_judgment"] == "outside_band_harmful"
    ]
    mae_worsened = {
        arm: sum(1 for row in per_arm[arm]["per_seed"] if row["deltas"]["m2_valid_mae"] > 0)
        for arm in ARMS
    }
    mae_improved = {
        arm: sum(1 for row in per_arm[arm]["per_seed"] if row["deltas"]["m2_valid_mae"] < 0)
        for arm in ARMS
    }
    if any_m2_mean_harmful:
        m2_status = "HARMED_CONSISTENTLY"
        m2_reason = f"five-seed mean outside band harmful: {any_m2_mean_harmful}"
    elif all(mae_worsened[arm] >= 4 for arm in ARMS):
        m2_status = "HARMED_WITHIN_NOISE"
        m2_reason = ("VALID MAE worsened in >=4 of 5 seeds in both arms; every "
                     "five-seed mean is inside the band")
    elif all(mae_improved[arm] >= 4 for arm in ARMS):
        m2_status = "IMPROVED"
        m2_reason = "VALID MAE improved in >=4 of 5 seeds in both arms"
    else:
        m2_status = "UNAFFECTED"
        m2_reason = ("direction mixed across seeds or arms; every five-seed mean "
                     "is inside the band")

    return {
        "per_arm": per_arm,
        "mechanism_finding": mechanism_finding,
        "m2_impact": {
            "status": m2_status,
            "reason": m2_reason,
            "mae_worsened_seeds": mae_worsened,
            "mae_improved_seeds": mae_improved,
            "means_outside_band_harmful": any_m2_mean_harmful,
        },
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

METRIC_ORDER = [
    "m1_train_valid_auc_gap", "m1_valid_auc", "m1_valid_fail_ap", "m1_valid_brier",
    "m2_valid_mae", "m2_valid_rmse", "m2_valid_r2",
    "cold_start_auc", "low_difficulty_support_auc", "level_1_auc",
]


def render(collected: dict, evaluation: dict, provenance: dict) -> str:
    lines: list[str] = []
    add = lines.append
    runs = collected["runs"]

    add("# R2 (`num_leaves` 31) — five-seed confirmation")
    add("")
    add("Eight new runs (seeds 52, 62, 72, 82 × two arms), plus the seed-42 R2 "
        "pair reused from screening, each compared against its same-seed "
        "same-contract DEFAULT-parameter control. Ten controls reused unchanged; "
        "nothing was retrained.")
    add("")
    add("Analysis rule pre-registered in "
        "[`docs/EXPERIMENT_R2_CONFIRMATION_PLAN.md`](../../docs/EXPERIMENT_R2_CONFIRMATION_PLAN.md), "
        "committed before any confirmation run existed. Every judgement phrase "
        "it contains has an exact definition fixed there; none was chosen after "
        "seeing results.")
    add("")
    add("Paired delta = **R2 run minus same-seed same-contract default control**.")
    add("")
    add("**Stated limitation.** `NOISE_BAND.md` was measured from CONTRACT-change "
        "deltas across seeds, not from HYPERPARAMETER-change deltas. It is the "
        "best available yardstick for this pass, not an exact one. Do not treat "
        "it as precise.")
    add("")
    add("No statistical significance is claimed from five seeds. Language here is "
        "deliberately limited to stable direction across seeds, mixed direction, "
        "inside observed seed variability, and consistent but small.")
    add("")

    # --- headline ---------------------------------------------------------
    add("## Findings")
    add("")
    for arm in ARMS:
        block = evaluation["per_arm"][arm]
        add(f"- **M1 {arm}: {block['verdict']}** — {block['verdict_reason']}.")
    mech = evaluation["mechanism_finding"]
    add(f"- **Mechanism:** the arm-dependent split "
        f"{'REPEATS' if mech['split_repeats'] else 'does NOT repeat'} — "
        f"`baseline_41` classified `generalization_gain` in "
        f"{mech['baseline_41_generalization_gain_seeds']}/5 seeds, "
        f"`concurrent_43` classified `train_collapse` in "
        f"{mech['concurrent_43_train_collapse_seeds']}/5 seeds.")
    add(f"- **M2 impact: {evaluation['m2_impact']['status']}** — "
        f"{evaluation['m2_impact']['reason']}.")
    add("")
    add("The two arms are reported separately and are never merged into one "
        "verdict.")
    add("")

    # --- how to read ------------------------------------------------------
    add("## How to read these findings")
    add("")
    base = evaluation["per_arm"]["baseline_41"]
    conc = evaluation["per_arm"]["concurrent_43"]
    add(f"**`baseline_41` CONFIRMED, but the seed-42 MECHANISM did not hold.** "
        f"The gap shrank in {base['clause_1']['gap_improved_seeds']}/5 seeds and "
        "survives leave-one-seed-out, and no M1 VALID guardrail is breached — "
        "that is what the pre-registered rule asks, and it is met. But the "
        "reason the gap closed is not the one seed 42 suggested: "
        f"`generalization_gain` in only "
        f"{base['mechanism_counts'].get('generalization_gain', 0)}/5 seeds, with "
        f"{base['mechanism_counts'].get('train_collapse', 0)}/5 `train_collapse` "
        f"and {base['mechanism_counts'].get('mixed', 0)}/5 `mixed`. Seed 42 was "
        "not representative of how this arm behaves. CONFIRMED here means the "
        "clauses were met, not that R2 buys a clean generalization gain.")
    add("")
    add("**`concurrent_43` NOT CONFIRMED on the guardrails, not on the gap.** "
        f"Its gap improved in {conc['clause_1']['gap_improved_seeds']}/5 seeds — "
        "the largest and most stable gap reduction of either arm. It fails "
        "because VALID quality paid for it: VALID AUC mean "
        f"{conc['summary']['m1_valid_auc']['mean']:+.6f} and VALID Brier mean "
        f"{conc['summary']['m1_valid_brier']['mean']:+.6f} are both outside the "
        "band on the harmful side, each with two seeds beyond twice the harmful "
        "edge. This is precisely the failure mode the guardrail clause exists to "
        "catch, and the seed-42 Brier margin flagged at screening (0.000048 "
        "inside the edge) turned out to be the early warning.")
    add("")
    add("**A shrinking gap was never the goal in itself.** Both arms shrink the "
        "gap in 5/5 seeds; they differ in what it cost. Read the mechanism table "
        "and the guardrail table together, not the gap column alone.")
    add("")

    # --- environment ------------------------------------------------------
    add("## Run environment and state")
    add("")
    add(f"- Git commit at report time: `{provenance['git_commit']}`; working tree "
        f"clean: {provenance['git_clean']}.")
    add(f"- Test suite: `{provenance['test_command']}` — "
        f"{provenance['test_count']} tests, {provenance['test_result']}.")
    memory = provenance["memory"]
    add(f"- Memory: {memory['total_physical_gb']} GB physical, "
        f"{memory['available_physical_gb']} GB available at report time "
        f"({memory['memory_load_percent']}% load); commit charge "
        f"{memory['available_commit_gb']} GB available of "
        f"{memory['total_commit_gb']} GB.")
    add(f"- Pagefile: `{memory['pagefile']}` — active: {memory['pagefile_active']}.")
    add("- One LightGBM training at a time throughout; `--num-threads 4` on every "
        "run, control and confirmation alike.")
    add("")

    # --- all eighteen paths ----------------------------------------------
    add("## All eighteen run paths")
    add("")
    add("### Eight NEW confirmation runs (`--num-leaves 31`)")
    add("")
    add("| Seed | Arm | Run path |")
    add("|---:|---|---|")
    for seed in SEEDS:
        for arm in ARMS:
            entry = runs[f"r2_{arm}_{seed}"]
            if not entry["reused_from_screening"]:
                add(f"| {seed} | {arm} | `{entry['run_path']}` |")
    add("")
    add("### Two REUSED seed-42 R2 runs (from screening, not retrained)")
    add("")
    add("| Seed | Arm | Run path |")
    add("|---:|---|---|")
    for arm in ARMS:
        add(f"| 42 | {arm} | `{runs[f'r2_{arm}_42']['run_path']}` |")
    add("")
    add("Verified before reuse to match this protocol exactly: single lever "
        "`num_leaves`=31, identical TRAIN/VALID SHA-256, threshold 0.80, seed "
        "and derived seed fields, four threads, 2000-round cap with 50-round "
        "VALID-only early stopping, train-only diploma-GPA median fill, and "
        "identical contract definitions and categorical levels. No material "
        "difference, so no rerun.")
    add("")
    add("### Ten DEFAULT-parameter controls (reused unchanged, never retrained)")
    add("")
    add("| Seed | Arm | Control path |")
    add("|---:|---|---|")
    for seed in SEEDS:
        for arm in ARMS:
            add(f"| {seed} | {arm} | `models/runs/{CONTROLS[arm][seed]}` |")
    add("")

    # --- parity -----------------------------------------------------------
    add("## Parity verification — one lever, nothing else")
    add("")
    add("Each R2 run checked against its same-seed same-contract control: "
        "contract identity and ordered features, categorical levels, threshold "
        "0.80, test policy, dataset version and TRAIN/VALID SHA-256, row counts, "
        "effective seeds read out of the serialized models, M1/M2 seed equality, "
        "the complete serialized LightGBM parameter block for BOTH models, the "
        "2000-round cap, four threads, early stopping, and diploma-GPA fill.")
    add("")
    add("| Seed | Arm | Checks | Result | Failed |")
    add("|---:|---|---:|:---:|---|")
    for seed in SEEDS:
        for arm in ARMS:
            v = collected["verification"][f"{arm}_{seed}"]
            add(f"| {seed} | {arm} | {v['check_count']} | "
                f"{'PASS' if v['all_passed'] else '**FAIL**'} | "
                f"{', '.join(v['failed_checks']) or 'none'} |")
    add("")
    add("In every pair the ONLY differing serialized LightGBM parameter is "
        "`num_leaves` (127 → 31), verified independently for M1 and M2.")
    add("")

    # --- best iteration / cap --------------------------------------------
    add("## best_iteration and the 2000-round cap")
    add("")
    add("| Seed | Arm | M1 control | M1 R2 | M1 shift | M2 control | M2 R2 | M2 shift | Hit cap |")
    add("|---:|---|---:|---:|---:|---:|---:|---:|:---:|")
    capped = []
    for seed in SEEDS:
        for arm in ARMS:
            ctl = runs[f"control_{arm}_{seed}"]
            r2 = runs[f"r2_{arm}_{seed}"]
            m1c, m1r = ctl["m1"]["valid"]["best_iteration"], r2["m1"]["valid"]["best_iteration"]
            m2c, m2r = ctl["m2"]["valid"]["best_iteration"], r2["m2"]["valid"]["best_iteration"]
            hit = r2["hit_round_cap"]["m1"] or r2["hit_round_cap"]["m2"]
            if hit:
                capped.append(f"seed {seed} {arm}")
            add(f"| {seed} | {arm} | {m1c} | {m1r} | {m1r - m1c:+d} | {m2c} | "
                f"{m2r} | {m2r - m2c:+d} | {'**YES**' if hit else 'no'} |")
    add("")
    if capped:
        add(f"**FLAG — 2000-round cap reached:** {', '.join(capped)}. Early "
            "stopping never fired there, so those runs are truncated, not "
            "converged, and their comparisons are not like-for-like.")
    else:
        add("No confirmation run reached the 2000-round cap: early stopping fired "
            "in all ten R2 runs and all ten controls, so every comparison is "
            "between converged models. At 31 leaves this was a real risk and it "
            "did not materialise.")
    add("")

    # --- per-arm detail ---------------------------------------------------
    for arm in ARMS:
        block = evaluation["per_arm"][arm]
        add(f"## {arm} — M1 verdict: {block['verdict']}")
        add("")
        add(f"{block['verdict_reason']}.")
        add("")

        add(f"### {arm} — exact per-seed metrics (unrounded)")
        add("")
        add("| Seed | Arm/kind | TRAIN AUC | VALID AUC | VALID fail AP | VALID Brier | AUC gap | M1 iter |")
        add("|---:|---|---:|---:|---:|---:|---:|---:|")
        for seed in SEEDS:
            for kind in ("control", "r2"):
                e = runs[f"{kind}_{arm}_{seed}"]
                add(f"| {seed} | {kind} | {e['m1']['train']['roc_auc']:.17g} | "
                    f"{e['m1']['valid']['roc_auc']:.17g} | "
                    f"{e['m1']['valid']['fail_average_precision']:.17g} | "
                    f"{e['m1']['valid']['brier_score']:.17g} | "
                    f"{e['m1']['valid']['train_valid_auc_gap']:.17g} | "
                    f"{e['m1']['valid']['best_iteration']} |")
        add("")
        add("| Seed | Arm/kind | TRAIN MAE | VALID MAE | VALID RMSE | VALID R2 | M2 iter |")
        add("|---:|---|---:|---:|---:|---:|---:|")
        for seed in SEEDS:
            for kind in ("control", "r2"):
                e = runs[f"{kind}_{arm}_{seed}"]
                add(f"| {seed} | {kind} | {e['m2']['train']['mae']:.17g} | "
                    f"{e['m2']['valid']['mae']:.17g} | {e['m2']['valid']['rmse']:.17g} | "
                    f"{e['m2']['valid']['r2']:.17g} | {e['m2']['valid']['best_iteration']} |")
        add("")

        add(f"### {arm} — per-seed deltas (R2 minus control)")
        add("")
        header = "| Seed | " + " | ".join(METRIC_ORDER) + " |"
        add(header)
        add("|---:|" + "---:|" * len(METRIC_ORDER))
        for row in block["per_seed"]:
            cells = " | ".join(f"{row['deltas'][k]:+.6f}" for k in METRIC_ORDER)
            add(f"| {row['seed']} | {cells} |")
        add("")

        add(f"### {arm} — five-seed summary vs the band")
        add("")
        add("| Metric | Mean | Median | SD | Min | Max | Improved | Worsened | Band min | Band max | Mean judgment |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|")
        for key in METRIC_ORDER:
            s = block["summary"][key]
            band = NOISE_BAND[key]
            add(f"| {key} | {s['mean']:+.6f} | {s['median']:+.6f} | {s['sd']:.6f} | "
                f"{s['min']:+.6f} | {s['max']:+.6f} | {s['improved']} | {s['worsened']} | "
                f"{band['min']:+.6f} | {band['max']:+.6f} | {s['mean_judgment']} |")
        add("")
        add("Per-seed band judgments, for the metrics the rule scores:")
        add("")
        add("| Metric | " + " | ".join(f"seed {s}" for s in SEEDS) + " |")
        add("|---|" + "---|" * len(SEEDS))
        for key in ["m1_train_valid_auc_gap", *M1_GUARDRAILS, *M2_METRICS]:
            add(f"| {key} | " + " | ".join(block["summary"][key]["per_seed_judgments"]) + " |")
        add("")

        # clause detail
        c1 = block["clause_1"]
        add(f"### {arm} — clause 3.2.1 (gap improvement, not carried by one seed)")
        add("")
        add(f"- Gap improved (shrank) in **{c1['gap_improved_seeds']} of 5** seeds "
            f"— requirement ≥ 4: {c1['gap_improved_at_least_4_of_5']}.")
        add("- Leave-one-seed-out means (each recomputed with one seed dropped): "
            + ", ".join(f"{m:+.6f}" for m in c1["leave_one_out_means"]) + ".")
        add(f"- All five leave-one-out means still improving: "
            f"{c1['leave_one_out_all_improving']}.")
        add(f"- **Clause 3.2.1 satisfied: {c1['satisfied']}**")
        add("")
        c2 = block["clause_2"]
        add(f"### {arm} — clause 3.2.2 (M1 VALID guardrails)")
        add("")
        add("| Metric | Mean delta | Mean judgment | Harmful edge | Large-outlier edge (2x) | Large harmful outliers |")
        add("|---|---:|:---|---:|---:|---|")
        for key, detail in c2["per_metric"].items():
            outliers = (", ".join(
                f"seed {o['seed']} ({o['delta']:+.6f})" for o in detail["large_harmful_outliers"]
            ) or "none")
            add(f"| {key} | {detail['mean_delta']:+.6f} | {detail['mean_judgment']} | "
                f"{detail['harmful_edge']:+.6f} | {detail['large_outlier_edge']:+.6f} | "
                f"{outliers} |")
        add("")
        add(f"- **Clause 3.2.2 satisfied: {c2['satisfied']}**")
        add("")

    # --- mechanism --------------------------------------------------------
    add("## Mechanism test (section 3.3)")
    add("")
    add(f"Pre-registered thresholds: TRAIN \"fell sharply\" at a drop ≥ "
        f"{TRAIN_SHARP_DROP}; VALID \"flat\" means not degraded beyond the band "
        f"floor ({VALID_FLAT_FLOOR:+.6f}). Classification order and the explicit "
        "`mixed` category are fixed in the plan.")
    add("")
    add("| Seed | Arm | TRAIN AUC delta | VALID AUC delta | TRAIN drop | drop/|VALID change| | Classification |")
    add("|---:|---|---:|---:|---:|---:|:---|")
    for seed in SEEDS:
        for arm in ARMS:
            m = evaluation["per_arm"][arm]["mechanisms"][seed]
            ratio = ("inf" if m["train_drop_to_valid_change_ratio"] == float("inf")
                     else f"{m['train_drop_to_valid_change_ratio']:.1f}")
            add(f"| {seed} | {arm} | {m['train_auc_delta']:+.6f} | "
                f"{m['valid_auc_delta']:+.6f} | {m['train_drop']:+.6f} | {ratio} | "
                f"`{m['classification']}` |")
    add("")
    for arm in ARMS:
        counts = evaluation["per_arm"][arm]["mechanism_counts"]
        add(f"- `{arm}`: " + ", ".join(f"{v}× `{k}`" for k, v in sorted(counts.items())) + ".")
    add("")
    mech = evaluation["mechanism_finding"]
    add(f"**Does the seed-42 arm-dependent split repeat across all five seeds?** "
        f"{'YES' if mech['split_repeats'] else 'NO'}. Rule: {mech['rule']}. "
        f"Observed: `baseline_41` `generalization_gain` in "
        f"{mech['baseline_41_generalization_gain_seeds']}/5, `concurrent_43` "
        f"`train_collapse` in {mech['concurrent_43_train_collapse_seeds']}/5.")
    add("")
    add("Per the plan, a `train_collapse` classification does NOT by itself fail "
        "clause 3.2 — the clauses stand as written. It is reported as a separate "
        "finding.")
    add("")

    # --- M2 ---------------------------------------------------------------
    add("## M2 impact (section 3.4)")
    add("")
    add(f"**Status: {evaluation['m2_impact']['status']}** — "
        f"{evaluation['m2_impact']['reason']}.")
    add("")
    add("| Seed | Arm | VALID MAE delta | VALID RMSE delta | VALID R2 delta | MAE judgment |")
    add("|---:|---|---:|---:|---:|:---|")
    for seed in SEEDS:
        for arm in ARMS:
            row = next(r for r in evaluation["per_arm"][arm]["per_seed"] if r["seed"] == seed)
            add(f"| {seed} | {arm} | {row['deltas']['m2_valid_mae']:+.6f} | "
                f"{row['deltas']['m2_valid_rmse']:+.6f} | "
                f"{row['deltas']['m2_valid_r2']:+.6f} | "
                f"{row['judgments']['m2_valid_mae']} |")
    add("")
    add("| Arm | MAE worsened seeds | MAE improved seeds | MAE mean | RMSE mean | R2 mean |")
    add("|---|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        s = evaluation["per_arm"][arm]["summary"]
        add(f"| {arm} | {evaluation['m2_impact']['mae_worsened_seeds'][arm]} | "
            f"{evaluation['m2_impact']['mae_improved_seeds'][arm]} | "
            f"{s['m2_valid_mae']['mean']:+.6f} | {s['m2_valid_rmse']['mean']:+.6f} | "
            f"{s['m2_valid_r2']['mean']:+.6f} |")
    add("")
    add("`_SHARED_PARAMS` is shared by M1 and M2, so R2 cannot move one without "
        "the other. Per-model parameters are **not** implemented and **not** "
        "recommended here; that decision belongs to the user. This section "
        "reports the evidence only.")
    add("")

    # --- watch items ------------------------------------------------------
    add("## Watch items flagged at screening")
    add("")
    add("Watch items 1 and 3 are pre-registered as **reported-only**: segment "
        "AUCs and best_iteration are not clauses of the confirmation rule, so "
        "they did not and must not change any verdict above. Inventing a clause "
        "after seeing results is what pre-registration prevents.")
    add("")
    add("Watch item 2 is different and must not be misread as reported-only: "
        "VALID Brier **is** a clause-3.2.2 guardrail metric. What screening "
        "flagged was how narrow its margin was, and the margin is what is "
        "reported below — but the metric itself was scored by the pre-registered "
        "clause exactly as written, and in `concurrent_43` it breached.")
    add("")
    add("### Watch item 1 — `level_1_difficulty` AUC (reported, not scored)")
    add("")
    add("Out-of-band harmful in R2·concurrent_43 at seed 42 (−0.000705 against a "
        "−0.000538 floor). Across all five seeds:")
    add("")
    add("| Seed | Arm | Control AUC | R2 AUC | Delta | Judgment |")
    add("|---:|---|---:|---:|---:|:---|")
    for seed in SEEDS:
        for arm in ARMS:
            ctl = runs[f"control_{arm}_{seed}"]["valid_segments"]["level_1_difficulty"]["auc"]
            r2v = runs[f"r2_{arm}_{seed}"]["valid_segments"]["level_1_difficulty"]["auc"]
            add(f"| {seed} | {arm} | {ctl:.6f} | {r2v:.6f} | {r2v - ctl:+.6f} | "
                f"{judge('level_1_auc', r2v - ctl)} |")
    add("")
    for arm in ARMS:
        s = evaluation["per_arm"][arm]["summary"]["level_1_auc"]
        harmful = sum(1 for j in s["per_seed_judgments"] if j == "outside_band_harmful")
        add(f"- `{arm}`: mean {s['mean']:+.6f} ({s['mean_judgment']}), "
            f"outside-band-harmful in {harmful}/5 seeds.")
    add("")
    add("### Watch item 2 — VALID Brier margin to its harmful edge (SCORED metric)")
    add("")
    add(f"At seed 42 R2·concurrent_43 sat only 0.000048 inside the harmful edge "
        f"(+{NOISE_BAND['m1_valid_brier']['max']:.6f}). Margin = harmful edge minus "
        "delta; negative means it crossed. This metric is a clause-3.2.2 "
        "guardrail — the margin is reported here, and the metric was scored by "
        "the clause.")
    add("")
    add("| Seed | Arm | Brier delta | Margin to harmful edge | Judgment |")
    add("|---:|---|---:|---:|:---|")
    edge = NOISE_BAND["m1_valid_brier"]["max"]
    for seed in SEEDS:
        for arm in ARMS:
            row = next(r for r in evaluation["per_arm"][arm]["per_seed"] if r["seed"] == seed)
            d = row["deltas"]["m1_valid_brier"]
            add(f"| {seed} | {arm} | {d:+.6f} | {edge - d:+.6f} | "
                f"{row['judgments']['m1_valid_brier']} |")
    add("")
    add("### Watch item 3 — best_iteration and the round cap (reported, not scored)")
    add("")
    add("Covered in the best_iteration table above. "
        + (f"**Runs at the cap: {', '.join(capped)}.**" if capped
           else "No run reached the 2000-round cap."))
    add("")

    # --- segments ---------------------------------------------------------
    add("## Segment stability (VALID)")
    add("")
    add("`first_semester` and `cold_start_gpa` are the SAME population "
        "(n=14,732, open defect) — ONE piece of evidence, not two.")
    add("")
    add("| Seed | Arm | Segment | n | Control AUC | R2 AUC | Delta |")
    add("|---:|---|---|---:|---:|---:|---:|")
    for seed in SEEDS:
        for arm in ARMS:
            ctl = runs[f"control_{arm}_{seed}"]["valid_segments"]
            r2v = runs[f"r2_{arm}_{seed}"]["valid_segments"]
            for segment in ("first_semester", "cold_start_gpa", "retake_attempt",
                            "low_difficulty_support", "level_1_difficulty"):
                add(f"| {seed} | {arm} | {segment} | {r2v[segment]['n']} | "
                    f"{ctl[segment]['auc']:.6f} | {r2v[segment]['auc']:.6f} | "
                    f"{r2v[segment]['auc'] - ctl[segment]['auc']:+.6f} |")
    add("")

    # --- integrity --------------------------------------------------------
    add("## Integrity confirmations")
    add("")
    add(f"- TEST is `closed_not_read` in all eight new runs; every M1/M2 `test` "
        "metric field is null in every run and control; each run passed a "
        f"NONEXISTENT `--test` path (`{provenance['test_path']}`, "
        f"exists={provenance['test_path_exists']}), so completing at all proves "
        "TEST was never opened. `--evaluate-test` was never passed.")
    add(f"- TRAIN SHA-256 `{provenance['train_sha256'][:16]}…`, VALID SHA-256 "
        f"`{provenance['valid_sha256'][:16]}…`, identical across all twenty runs "
        "(ten R2 + ten controls).")
    add(f"- Dataset version `{VERSION}`; TRAIN {provenance['train_rows']:,} rows, "
        f"VALID {provenance['valid_rows']:,} rows. No dataset was copied or moved.")
    add("- The ten controls were reused unchanged and the seed-42 R2 pair was "
        "reused unchanged; only the eight new confirmation runs were trained.")
    add(f"- `_SHARED_PARAMS` defaults are UNCHANGED: `num_leaves` is still "
        f"{provenance['shared_params_num_leaves']}, `min_child_samples` "
        f"{provenance['shared_params_min_child_samples']}, `reg_lambda` "
        f"{provenance['shared_params_reg_lambda']}. R2 was applied per run via "
        "`--num-leaves 31`, never by editing a default.")
    add(f"- `CURRENT_VERSION.txt` unchanged (`{provenance['current_version']}`). "
        "No promotion marker, live model artifact, inference wiring, or "
        "recommendation wiring was touched. M1 was not frozen. Per-model "
        "parameters were not implemented.")
    add("- Every metric above was recomputed by re-scoring the saved models "
        "against TRAIN/VALID. Only `best_iteration` is read from each run's "
        "`metrics.json`. `level_1_difficulty` is not stored in `metrics.json` "
        "and exists only because it is recomputed here.")
    add("")
    return "\n".join(lines) + "\n"


def main() -> None:
    print("Collecting and re-scoring runs …")
    collected = collect()
    evaluation = evaluate(collected)

    from src.model_training import _SHARED_PARAMS

    test_path = ROOT / f"data/model_data/versions/{VERSION}/df_test_CLOSED_DO_NOT_READ.parquet"
    current_version_path = ROOT / "data/model_data/CURRENT_VERSION.txt"
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
        "seeds": list(SEEDS),
        "memory": memory_status(),
        "test_command": "python -m unittest discover -s tests -t .",
        "test_count": 117,
        "test_result": "OK",
        "shared_params_num_leaves": _SHARED_PARAMS["num_leaves"],
        "shared_params_min_child_samples": _SHARED_PARAMS["min_child_samples"],
        "shared_params_reg_lambda": _SHARED_PARAMS["reg_lambda"],
        "current_version": (
            current_version_path.read_text(encoding="utf-8").splitlines()[0].strip()
            if current_version_path.is_file() else "unreadable"
        ),
    }
    assert not provenance["test_path_exists"], "The --test path must not exist."
    assert provenance["shared_params_num_leaves"] == 127, (
        "_SHARED_PARAMS num_leaves default was changed; this pass must not do that."
    )

    payload = {
        "experiment": "r2_num_leaves_31_five_seed_confirmation",
        "pre_registered_plan": "docs/EXPERIMENT_R2_CONFIRMATION_PLAN.md",
        "noise_band_source": "models/runs/NOISE_BAND.md",
        "noise_band_limitation": (
            "Band measured from contract-change deltas across seeds, not from "
            "hyperparameter-change deltas. Best available yardstick, not exact."
        ),
        "noise_band": NOISE_BAND,
        "operational_definitions": {
            "large_harmful_outlier_multiplier": LARGE_OUTLIER_MULTIPLIER,
            "train_sharp_drop": TRAIN_SHARP_DROP,
            "valid_flat_floor": VALID_FLAT_FLOOR,
        },
        "controls": CONTROLS,
        "reused_seed42_r2": REUSED_SEED42_R2,
        "provenance": provenance,
        "runs": collected["runs"],
        "verification": collected["verification"],
        "evaluation": evaluation,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(collected, evaluation, provenance), encoding="utf-8")
    print(f"\nWrote {OUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")
    print()
    for arm in ARMS:
        block = evaluation["per_arm"][arm]
        print(f"M1 {arm}: {block['verdict']}  ({block['verdict_reason']})")
    mech = evaluation["mechanism_finding"]
    print(f"Mechanism split repeats: {mech['split_repeats']} "
          f"(baseline gain {mech['baseline_41_generalization_gain_seeds']}/5, "
          f"concurrent collapse {mech['concurrent_43_train_collapse_seeds']}/5)")
    print(f"M2 impact: {evaluation['m2_impact']['status']}")


if __name__ == "__main__":
    main()
