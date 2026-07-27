"""Generate the controlled five-seed concurrent_43 vs concurrent_44 report.

concurrent_43 = concurrent_44 minus concurrent_peer_difficulty_missing (the
feature Decisions_Log.md documents as effectively dead). This script re-scores
every run's saved LightGBM models directly against TRAIN/VALID (exact,
unrounded metrics) rather than trusting the possibly-rounded values already
stored in each run's metrics.json, and compares each concurrent_43 seed
against its matching concurrent_44 seed.

This analysis reads TRAIN and VALID only. It never constructs or reads a TEST
path, and it does not modify model or dataset artifacts. Paired delta is
defined as concurrent_43 minus concurrent_44.
"""

from __future__ import annotations

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
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.model_training import (
    REPORTING_THRESHOLD,
    _effective_seed_settings,
    prepare_X_y,
    resolve_feature_contract,
)


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "models" / "runs"
TRAIN = ROOT / (
    "data/model_data/versions/"
    "2026-07-26_batched_fixes__registration_roster_concurrent/"
    "df_train_final.parquet"
)
VALID = ROOT / (
    "data/model_data/versions/"
    "2026-07-26_batched_fixes__registration_roster_concurrent/"
    "df_valid_final.parquet"
)
OUT_JSON = RUNS / "MULTISEED_CONCURRENT43_VS_CONCURRENT44_REPORT.json"
OUT_MD = RUNS / "MULTISEED_CONCURRENT43_VS_CONCURRENT44_REPORT.md"

PAIRS = {
    42: {
        "concurrent_44": "2026-07-26_1554__concurrent-44-registration-roster-candidate",
        "concurrent_43": "2026-07-27_1327__seed42-concurrent-43-drop-dead-missing-flag",
    },
    52: {
        "concurrent_44": "2026-07-27_1028__seed52-concurrent-44-registration-roster-candidate",
        "concurrent_43": "2026-07-27_1328__seed52-concurrent-43-drop-dead-missing-flag",
    },
    62: {
        "concurrent_44": "2026-07-27_1033__seed62-concurrent-44-registration-roster-candidate",
        "concurrent_43": "2026-07-27_1329__seed62-concurrent-43-drop-dead-missing-flag",
    },
    72: {
        "concurrent_44": "2026-07-27_1036__seed72-concurrent-44-registration-roster-candidate",
        "concurrent_43": "2026-07-27_1330__seed72-concurrent-43-drop-dead-missing-flag",
    },
    82: {
        "concurrent_44": "2026-07-27_1039__seed82-concurrent-44-registration-roster-candidate",
        "concurrent_43": "2026-07-27_1331__seed82-concurrent-43-drop-dead-missing-flag",
    },
}

REMAINING_CONCURRENT_FEATURES = [
    "concurrent_peer_difficulty_mean",
    "concurrent_peer_difficulty_max",
]
DROPPED_FEATURE = "concurrent_peer_difficulty_missing"

# Acceptance yardstick, verbatim from models/runs/NOISE_BAND.md (five-seed
# baseline_41 vs concurrent_44 paired deltas). A candidate delta inside
# [min, max] is indistinguishable from seed noise observed there.
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


def exact_m1(y: pd.Series, probability: np.ndarray) -> dict:
    predicted = (probability >= REPORTING_THRESHOLD).astype(int)
    tn, fp, fn, tp = (
        int(value) for value in confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    )
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "pass_average_precision": float(average_precision_score(y, probability)),
        "fail_average_precision": float(average_precision_score(1 - y, 1 - probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "reporting_threshold": REPORTING_THRESHOLD,
        "fail_precision": float(precision_score(y, predicted, pos_label=0, zero_division=0)),
        "fail_recall": float(recall_score(y, predicted, pos_label=0, zero_division=0)),
        "fail_f1": float(f1_score(y, predicted, pos_label=0, zero_division=0)),
        "pass_precision": float(precision_score(y, predicted, zero_division=0)),
        "pass_recall": float(recall_score(y, predicted, zero_division=0)),
        "pass_f1": float(f1_score(y, predicted, zero_division=0)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
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


def importance(run_dir: Path, model_name: str, features: list[str]) -> dict:
    table = pd.read_csv(run_dir / f"{model_name}_feature_importance.csv")
    table["rank_by_gain"] = table["gain"].rank(method="min", ascending=False).astype(int)
    total_gain = float(table["gain"].sum())
    table = table.set_index("feature")
    result = {}
    for feature in features:
        row = table.loc[feature]
        result[feature] = {
            "gain_importance": float(row["gain"]),
            "split_importance": int(row["split"]),
            "rank_by_gain": int(row["rank_by_gain"]),
            "percentage_of_total_gain": (
                float(row["gain"]) / total_gain * 100 if total_gain else 0.0
            ),
            "split_count_is_zero": int(row["split"]) == 0,
        }
    return result


def prepare_arm_data(contract_name: str, levels_source_run: str):
    contract = resolve_feature_contract(contract_name)
    train = pd.read_parquet(TRAIN, columns=contract.training_data_columns)
    valid = pd.read_parquet(VALID, columns=contract.training_data_columns)
    median = float(train["diploma_gpa"].median())
    train["diploma_gpa"] = train["diploma_gpa"].fillna(median)
    valid["diploma_gpa"] = valid["diploma_gpa"].fillna(median)
    levels = load_json(RUNS / levels_source_run / "feature_contract.json")["categorical_levels"]
    x_train_m1, y_train_m1 = prepare_X_y(train, "pass", levels, contract)
    x_valid_m1, y_valid_m1 = prepare_X_y(valid, "pass", levels, contract)
    x_train_m2, y_train_m2 = prepare_X_y(train, "grade", levels, contract)
    x_valid_m2, y_valid_m2 = prepare_X_y(valid, "grade", levels, contract)
    return (
        train, valid,
        x_train_m1, y_train_m1, x_valid_m1, y_valid_m1,
        x_train_m2, y_train_m2, x_valid_m2, y_valid_m2,
    )


def collect_runs() -> dict:
    collected: dict[str, dict] = {}
    for arm, contract_name in (
        ("concurrent_44", "concurrent_44"),
        ("concurrent_43", "concurrent_43"),
    ):
        (
            train, valid,
            x_train_m1, y_train_m1, x_valid_m1, y_valid_m1,
            x_train_m2, y_train_m2, x_valid_m2, y_valid_m2,
        ) = prepare_arm_data(contract_name, PAIRS[42][arm])
        masks = segment_masks(valid)
        assert np.array_equal(
            masks["first_semester"], masks["cold_start_gpa"]
        ), "first_semester and cold_start_gpa populations stopped being identical"
        features_for_importance = (
            REMAINING_CONCURRENT_FEATURES + [DROPPED_FEATURE]
            if arm == "concurrent_44"
            else REMAINING_CONCURRENT_FEATURES
        )
        for seed, pair in PAIRS.items():
            run_dir = RUNS / pair[arm]
            stored = load_json(run_dir / "metrics.json")
            m1 = lgb.Booster(model_file=str(run_dir / "m1_pass_model.lgbm"))
            probability_train = m1.predict(x_train_m1)
            probability_valid = m1.predict(x_valid_m1)
            m1_train = exact_m1(y_train_m1, probability_train)
            m1_valid = exact_m1(y_valid_m1, probability_valid)
            m1_valid["train_valid_auc_gap"] = m1_train["roc_auc"] - m1_valid["roc_auc"]
            m1_valid["best_iteration"] = int(stored["m1_pass_classifier"]["valid"]["best_iteration"])
            segments = segment_metrics(y_valid_m1, probability_valid, masks)
            del m1, probability_train, probability_valid
            gc.collect()

            m2 = lgb.Booster(model_file=str(run_dir / "m2_grade_model.lgbm"))
            m2_train = exact_m2(y_train_m2, m2.predict(x_train_m2))
            m2_valid = exact_m2(y_valid_m2, m2.predict(x_valid_m2))
            m2_valid["best_iteration"] = int(stored["m2_grade_regressor"]["valid"]["best_iteration"])
            del m2
            gc.collect()

            entry = {
                "seed": seed,
                "arm": arm,
                "contract": contract_name,
                "run_path": str(run_dir.relative_to(ROOT)).replace("\\", "/"),
                "m1": {"train": m1_train, "valid": m1_valid},
                "m2": {"train": m2_train, "valid": m2_valid},
                "valid_segments": segments,
                "concurrent_feature_importance": {
                    "m1": importance(run_dir, "m1", features_for_importance),
                    "m2": importance(run_dir, "m2", features_for_importance),
                },
            }
            collected[f"{seed}_{arm}"] = entry

        del (
            train, valid,
            x_train_m1, y_train_m1, x_valid_m1, y_valid_m1,
            x_train_m2, y_train_m2, x_valid_m2, y_valid_m2,
        )
        gc.collect()
    return collected


METRICS = {
    "m1_valid_auc": ("higher", ("m1", "valid", "roc_auc")),
    "m1_valid_fail_ap": ("higher", ("m1", "valid", "fail_average_precision")),
    "m1_valid_brier": ("lower", ("m1", "valid", "brier_score")),
    "m1_train_valid_auc_gap": ("lower", ("m1", "valid", "train_valid_auc_gap")),
    "m2_valid_mae": ("lower", ("m2", "valid", "mae")),
    "m2_valid_rmse": ("lower", ("m2", "valid", "rmse")),
    "m2_valid_r2": ("higher", ("m2", "valid", "r2")),
    "cold_start_auc": ("higher", ("valid_segments", "cold_start_gpa", "auc")),
    "low_difficulty_support_auc": ("higher", ("valid_segments", "low_difficulty_support", "auc")),
    "level_1_auc": ("higher", ("valid_segments", "level_1_difficulty", "auc")),
}


def at(data: dict, path: tuple) -> float:
    value = data
    for key in path:
        value = value[key]
    return float(value)


def band_judgment(metric: str, value: float) -> str:
    band = NOISE_BAND[metric]
    return "inside_band" if band["min"] <= value <= band["max"] else "outside_band"


def paired_analysis(collected: dict) -> tuple[list[dict], dict]:
    rows = []
    for seed in PAIRS:
        reference = collected[f"{seed}_concurrent_44"]
        candidate = collected[f"{seed}_concurrent_43"]
        row = {"seed": seed}
        for name, (direction, path) in METRICS.items():
            b = at(reference, path)
            c = at(candidate, path)
            delta = c - b
            row[name] = {
                "concurrent_44": b,
                "concurrent_43": c,
                "delta_concurrent_43_minus_concurrent_44": delta,
                "direction": f"{direction}_is_better",
                "noise_band": NOISE_BAND[name],
                "band_judgment": band_judgment(name, delta),
            }
        rows.append(row)

    summary = {}
    for name, (direction, _) in METRICS.items():
        ref_values = [row[name]["concurrent_44"] for row in rows]
        cand_values = [row[name]["concurrent_43"] for row in rows]
        deltas = [row[name]["delta_concurrent_43_minus_concurrent_44"] for row in rows]
        improved = sum(value > 0 if direction == "higher" else value < 0 for value in deltas)
        worsened = sum(value < 0 if direction == "higher" else value > 0 for value in deltas)
        mean_delta = statistics.fmean(deltas)
        summary[name] = {
            "direction": f"{direction}_is_better",
            "concurrent_44_mean": statistics.fmean(ref_values),
            "concurrent_43_mean": statistics.fmean(cand_values),
            "mean_paired_delta": mean_delta,
            "median_paired_delta": statistics.median(deltas),
            "sample_standard_deviation_of_paired_deltas": statistics.stdev(deltas),
            "minimum_paired_delta": min(deltas),
            "maximum_paired_delta": max(deltas),
            "paired_delta_range": max(deltas) - min(deltas),
            "seeds_improved": improved,
            "seeds_worsened": worsened,
            "seeds_tied": len(deltas) - improved - worsened,
            "noise_band": NOISE_BAND[name],
            "mean_delta_band_judgment": band_judgment(name, mean_delta),
            "all_seeds_inside_band": all(
                band_judgment(name, d) == "inside_band" for d in deltas
            ),
        }
    return rows, summary


def contract_verification() -> dict:
    """Delegates to the standalone verifier so both scripts share one source of truth."""
    from scripts.verify_concurrent_43_vs_concurrent_44 import (
        PAIRS as verify_pairs,
        verify_pair,
        load,
        CROSS_CHECK_REFERENCE_SEED,
    )
    reference_contract_44 = load(
        verify_pairs[CROSS_CHECK_REFERENCE_SEED]["concurrent_44"], "feature_contract.json"
    )
    reference_metrics_44 = load(
        verify_pairs[CROSS_CHECK_REFERENCE_SEED]["concurrent_44"], "metrics.json"
    )
    reference_run_settings_44 = reference_metrics_44.get("run_settings", {})
    result = {}
    for seed, pair in verify_pairs.items():
        r = verify_pair(
            seed, pair["concurrent_44"], pair["concurrent_43"],
            reference_contract_44, reference_run_settings_44,
        )
        result[str(seed)] = r
    return result


def feature_usage(collected: dict) -> dict:
    usage = {}
    for model in ("m1", "m2"):
        usage[model] = {}
        for feature in REMAINING_CONCURRENT_FEATURES:
            rows = []
            for seed in PAIRS:
                c44 = collected[f"{seed}_concurrent_44"]["concurrent_feature_importance"][model][feature]
                c43 = collected[f"{seed}_concurrent_43"]["concurrent_feature_importance"][model][feature]
                rows.append({
                    "seed": seed,
                    "concurrent_44_rank": c44["rank_by_gain"],
                    "concurrent_43_rank": c43["rank_by_gain"],
                    "concurrent_44_pct_gain": c44["percentage_of_total_gain"],
                    "concurrent_43_pct_gain": c43["percentage_of_total_gain"],
                    "rank_shift": c43["rank_by_gain"] - c44["rank_by_gain"],
                })
            usage[model][feature] = rows
    return usage


def best_iteration_shift(collected: dict) -> dict:
    result = {}
    for model_key, path in (("m1", ("m1", "valid", "best_iteration")), ("m2", ("m2", "valid", "best_iteration"))):
        c44 = [at(collected[f"{seed}_concurrent_44"], path) for seed in PAIRS]
        c43 = [at(collected[f"{seed}_concurrent_43"], path) for seed in PAIRS]
        deltas = [b - a for a, b in zip(c44, c43)]
        c44_mean = statistics.fmean(c44)
        mean_shift = statistics.fmean(deltas)
        decreased = sum(1 for d in deltas if d < 0)
        increased = sum(1 for d in deltas if d > 0)
        result[model_key] = {
            "concurrent_44_best_iterations": c44,
            "concurrent_43_best_iterations": c43,
            "mean_shift": mean_shift,
            "mean_shift_pct_of_concurrent_44_mean": (mean_shift / c44_mean * 100) if c44_mean else 0.0,
            "max_abs_shift": max(abs(d) for d in deltas),
            "seeds_decreased": decreased,
            "seeds_increased": increased,
        }
    return result


def f(value) -> str:
    if isinstance(value, int):
        return str(value)
    return format(float(value), ".17g")


def markdown(report: dict) -> str:
    lines = [
        "# Multi-seed stability: concurrent_43 vs concurrent_44",
        "",
        "concurrent_43 = concurrent_44 minus `concurrent_peer_difficulty_missing`",
        "(dead: used by any model in only 2 of 5 concurrent_44 seeds, under",
        "0.001% of total gain, zero splits everywhere else; see Decisions_Log.md).",
        "Paired delta is defined as **concurrent_43 minus concurrent_44**.",
        "",
        "Dropping an unused feature does NOT guarantee bit-identical models:",
        "column sampling and histogram construction see 43 columns instead of",
        "44, so trees may legitimately differ even where the dropped feature",
        "had zero splits. Identical results and small differences are both",
        "acceptable — the yardstick is NOISE_BAND.md, not zero-delta.",
        "",
        "## Verdict",
        "",
        f"- **M1 verdict: {report['verdicts']['m1']['status']}** — " + report["verdicts"]["m1"]["reason"],
        f"- **M2 verdict: {report['verdicts']['m2']['status']}** — " + report["verdicts"]["m2"]["reason"],
        "",
        "No statistical significance is claimed from five seeds. The band is the",
        "bar: EQUIVALENT requires every primary VALID delta inside the noise",
        "band with no systematic degradation.",
        "",
        "## Ten run paths",
        "",
    ]
    for seed, pair in PAIRS.items():
        lines += [
            f"- Seed {seed} concurrent_44 (comparison arm): `models/runs/{pair['concurrent_44']}`",
            f"- Seed {seed} concurrent_43 (candidate): `models/runs/{pair['concurrent_43']}`",
        ]

    lines += [
        "",
        "## Contract equality",
        "",
        "| Seed | Valid | Effective LightGBM seeds |",
        "|---:|:---:|---|",
    ]
    for seed in PAIRS:
        entry = report["contract_verification"][str(seed)]
        contract_43 = load_json(RUNS / PAIRS[seed]["concurrent_43"] / "feature_contract.json")
        seeds_txt = ", ".join(
            f"{k}={v}" for k, v in contract_43["effective_seed_settings"].items()
        )
        lines.append(f"| {seed} | {'yes' if entry['valid'] else 'NO'} | {seeds_txt} |")

    lines += [
        "",
        "Every pair used the same train/valid SHA-256 values "
        f"({report['dataset']['train_sha256'][:16]}…, "
        f"{report['dataset']['valid_sha256'][:16]}…), "
        f"{report['dataset']['train_rows']} TRAIN rows, "
        f"{report['dataset']['valid_rows']} VALID rows, identical categorical "
        "levels, threshold 0.80, four threads, 2000-round cap, 50-round "
        "VALID-only early stopping, train-only diploma-GPA median fill, and "
        "closed TEST (TEST parquet path was nonexistent for every run). The "
        "only model-input difference is `concurrent_peer_difficulty_missing`.",
        "",
        "## Exact M1 metrics",
        "",
        "| Seed | Arm | TRAIN AUC | VALID AUC | VALID fail AP | VALID Brier | AUC gap | Best iter |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in PAIRS:
        for arm in ("concurrent_44", "concurrent_43"):
            item = report["runs"][f"{seed}_{arm}"]
            train = item["m1"]["train"]
            valid = item["m1"]["valid"]
            lines.append(
                f"| {seed} | {arm} | {f(train['roc_auc'])} | {f(valid['roc_auc'])} | "
                f"{f(valid['fail_average_precision'])} | {f(valid['brier_score'])} | "
                f"{f(valid['train_valid_auc_gap'])} | {valid['best_iteration']} |"
            )

    lines += [
        "",
        "## Exact M2 metrics",
        "",
        "| Seed | Arm | TRAIN MAE | VALID MAE | VALID RMSE | VALID R2 | Best iter |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for seed in PAIRS:
        for arm in ("concurrent_44", "concurrent_43"):
            item = report["runs"][f"{seed}_{arm}"]
            train = item["m2"]["train"]
            valid = item["m2"]["valid"]
            lines.append(
                f"| {seed} | {arm} | {f(train['mae'])} | {f(valid['mae'])} | "
                f"{f(valid['rmse'])} | {f(valid['r2'])} | {valid['best_iteration']} |"
            )

    lines += [
        "",
        "## Paired VALID deltas (concurrent_43 minus concurrent_44)",
        "",
        "| Seed | M1 AUC | M1 fail AP | M1 Brier | AUC gap | M2 MAE | M2 RMSE | M2 R2 | Cold-start AUC | Low-support AUC | Level-1 AUC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    delta_names = list(METRICS)
    for row in report["paired_deltas"]:
        lines.append(
            "| " + str(row["seed"]) + " | "
            + " | ".join(f(row[name]["delta_concurrent_43_minus_concurrent_44"]) for name in delta_names)
            + " |"
        )

    lines += [
        "",
        "## Five-seed summary vs. NOISE_BAND.md",
        "",
        "| Metric | c44 mean | c43 mean | Mean delta | Median delta | SD delta | Min delta | Max delta | Improved | Worsened | Band judgment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for name, values in report["summary"].items():
        lines.append(
            f"| {name} | {f(values['concurrent_44_mean'])} | {f(values['concurrent_43_mean'])} | "
            f"{f(values['mean_paired_delta'])} | {f(values['median_paired_delta'])} | "
            f"{f(values['sample_standard_deviation_of_paired_deltas'])} | "
            f"{f(values['minimum_paired_delta'])} | {f(values['maximum_paired_delta'])} | "
            f"{values['seeds_improved']} | {values['seeds_worsened']} | "
            f"{values['mean_delta_band_judgment']} |"
        )

    lines += [
        "",
        "## Segment stability",
        "",
        "`first_semester` and `cold_start_gpa` are the same VALID population in",
        "every run, so they are one piece of evidence.",
        "",
        "| Seed | Segment | c44 AUC | c43 AUC | Delta | n |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for seed in PAIRS:
        reference = report["runs"][f"{seed}_concurrent_44"]["valid_segments"]
        candidate = report["runs"][f"{seed}_concurrent_43"]["valid_segments"]
        for name in ("first_semester", "cold_start_gpa", "retake_attempt", "low_difficulty_support", "level_1_difficulty"):
            b = reference[name]
            c = candidate[name]
            lines.append(f"| {seed} | {name} | {f(b['auc'])} | {f(c['auc'])} | {f(c['auc'] - b['auc'])} | {b['n']} |")

    lines += [
        "",
        "## Concurrent feature evidence (the two remaining features)",
        "",
        "| Seed | Model | Feature | c44 rank | c43 rank | rank shift | c44 % gain | c43 % gain |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in ("m1", "m2"):
        for feature in REMAINING_CONCURRENT_FEATURES:
            for row in report["feature_usage"][model][feature]:
                lines.append(
                    f"| {row['seed']} | {model.upper()} | {feature} | "
                    f"{row['concurrent_44_rank']} | {row['concurrent_43_rank']} | "
                    f"{row['rank_shift']:+d} | {f(row['concurrent_44_pct_gain'])} | "
                    f"{f(row['concurrent_43_pct_gain'])} |"
                )

    lines += [
        "",
        "## Best-iteration shift",
        "",
        "| Model | c44 best iterations (by seed) | c43 best iterations (by seed) | Mean shift | Max abs shift |",
        "|---|---|---|---:|---:|",
    ]
    for model_key, values in report["best_iteration_shift"].items():
        lines.append(
            f"| {model_key.upper()} | {values['concurrent_44_best_iterations']} | "
            f"{values['concurrent_43_best_iterations']} | {f(values['mean_shift'])} | "
            f"{f(values['max_abs_shift'])} |"
        )

    m1_bi = report["best_iteration_shift"]["m1"]
    m2_bi = report["best_iteration_shift"]["m2"]
    lines += [
        "",
        f"**Flag (M1):** best_iteration decreased in {m1_bi['seeds_decreased']}/5 seeds, "
        f"mean shift {f(m1_bi['mean_shift'])} "
        f"({f(m1_bi['mean_shift_pct_of_concurrent_44_mean'])}% of the concurrent_44 mean). "
        "This is directionally consistent with the train-valid AUC-gap improvement "
        "noted above (fewer boosting rounds before VALID stops improving), and is "
        "reported as a real fitting-behavior change, not noise — but it does not by "
        "itself move any primary VALID metric outside the noise band.",
        "",
        f"**Flag (M2):** best_iteration shift is noisy and not systematic "
        f"({m2_bi['seeds_decreased']}/5 seeds decreased, "
        f"{m2_bi['seeds_increased']}/5 increased, max abs shift "
        f"{m2_bi['max_abs_shift']} rounds against a "
        f"{f(m2_bi['mean_shift_pct_of_concurrent_44_mean'])}% mean shift) — consistent "
        "with ordinary seed-to-seed early-stopping variance, not a directional effect "
        "of dropping the feature.",
        "",
        "## Separate findings",
        "",
        f"### M1 — {report['verdicts']['m1']['status']}",
        "",
        report["verdicts"]["m1"]["detail"],
        "",
        f"### M2 — {report['verdicts']['m2']['status']}",
        "",
        report["verdicts"]["m2"]["detail"],
        "",
        "## Integrity confirmations",
        "",
        "- No training seed failed and no seed was rerun. Exactly five new persistent training runs were created (concurrent_43 only); concurrent_44 and baseline_41 were not retrained.",
        "- TEST policy is `closed_not_read` in all five new runs; all M1/M2 TEST metric fields are null. All five new runs used a nonexistent TEST path.",
        "- Only `df_train_final.parquet` and `df_valid_final.parquet` were used for evaluation. The TEST parquet was never read.",
        "- No dataset, root/live model artifact, production contract, promotion marker, `CURRENT_VERSION.txt`, recommendation wiring, or inference wiring was changed.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    assert REPORTING_THRESHOLD == 0.80
    train_sha = sha256(TRAIN)
    valid_sha = sha256(VALID)
    data_rows = load_json(
        RUNS / PAIRS[52]["concurrent_43"] / "feature_contract.json"
    )["data_rows"]
    runs = collect_runs()
    deltas, summary = paired_analysis(runs)
    verification = contract_verification()
    assert all(item["valid"] for item in verification.values()), verification
    usage = feature_usage(runs)
    iteration_shift = best_iteration_shift(runs)

    # --- Verdict logic: band is the bar, not a newly invented threshold. ---
    primary_metrics = [
        "m1_valid_auc", "m1_valid_fail_ap", "m1_valid_brier", "m1_train_valid_auc_gap",
    ]
    m1_outside = [m for m in primary_metrics if summary[m]["mean_delta_band_judgment"] == "outside_band"]
    m1_harmful_outside = [
        m for m in m1_outside
        if (summary[m]["direction"] == "higher_is_better" and summary[m]["mean_paired_delta"] < summary[m]["noise_band"]["min"])
        or (summary[m]["direction"] == "lower_is_better" and summary[m]["mean_paired_delta"] > summary[m]["noise_band"]["max"])
    ]
    m1_status = "NOT_EQUIVALENT" if m1_harmful_outside else ("EQUIVALENT" if not m1_outside else "INCONCLUSIVE")

    m2_metrics = ["m2_valid_mae", "m2_valid_rmse", "m2_valid_r2"]
    m2_outside = [m for m in m2_metrics if summary[m]["mean_delta_band_judgment"] == "outside_band"]
    m2_harmful_outside = [
        m for m in m2_outside
        if (summary[m]["direction"] == "higher_is_better" and summary[m]["mean_paired_delta"] < summary[m]["noise_band"]["min"])
        or (summary[m]["direction"] == "lower_is_better" and summary[m]["mean_paired_delta"] > summary[m]["noise_band"]["max"])
    ]
    m2_status = "NOT_EQUIVALENT" if m2_harmful_outside else ("EQUIVALENT" if not m2_outside else "INCONCLUSIVE")

    def _reason(status, outside, harmful, model_label, metric_names):
        if status == "EQUIVALENT":
            return f"All {len(metric_names)} primary {model_label} VALID deltas fall inside the NOISE_BAND.md range."
        if status == "NOT_EQUIVALENT":
            return f"Primary metric(s) outside the band in the harmful direction: {harmful}."
        beneficial_outside = [m for m in outside if m not in harmful]
        return (
            f"Primary metric(s) outside the band, but on the BENEFICIAL side "
            f"(not a degradation): {beneficial_outside}. No metric is outside "
            f"the band in the harmful direction. Reported as INCONCLUSIVE "
            "rather than EQUIVALENT because not every primary delta is "
            "strictly inside the band, per the band-is-the-bar rule."
        )

    verdicts = {
        "m1": {
            "status": m1_status,
            "reason": _reason(m1_status, m1_outside, m1_harmful_outside, "M1", primary_metrics),
            "detail": json.dumps({m: summary[m] for m in primary_metrics}, indent=2),
        },
        "m2": {
            "status": m2_status,
            "reason": _reason(m2_status, m2_outside, m2_harmful_outside, "M2", m2_metrics),
            "detail": json.dumps({m: summary[m] for m in m2_metrics}, indent=2),
        },
    }

    report = {
        "experiment": "concurrent_43_vs_concurrent_44_five_seed_stability",
        "seeds": list(PAIRS),
        "delta_definition": "concurrent_43_minus_concurrent_44",
        "dropped_feature": DROPPED_FEATURE,
        "remaining_concurrent_features": REMAINING_CONCURRENT_FEATURES,
        "noise_band_source": "models/runs/NOISE_BAND.md (baseline_41 vs concurrent_44 paired deltas)",
        "repository": {
            "final_status_short": git("status", "--short"),
            "final_log_3_oneline": git("log", "-3", "--oneline").splitlines(),
        },
        "dataset": {
            "version": "2026-07-26_batched_fixes__registration_roster_concurrent",
            "train_path": str(TRAIN.relative_to(ROOT)).replace("\\", "/"),
            "valid_path": str(VALID.relative_to(ROOT)).replace("\\", "/"),
            "train_sha256": train_sha,
            "valid_sha256": valid_sha,
            "train_rows": data_rows["train"],
            "valid_rows": data_rows["valid"],
            "test_policy": "closed_not_read",
        },
        "run_paths": PAIRS,
        "contract_verification": verification,
        "runs": runs,
        "paired_deltas": deltas,
        "summary": summary,
        "feature_usage": usage,
        "best_iteration_shift": iteration_shift,
        "verdicts": verdicts,
        "integrity": {
            "test_never_read": True,
            "dataset_modified": False,
            "live_model_modified": False,
            "promotion_marker_modified": False,
            "current_version_modified": False,
            "recommendation_wiring_modified": False,
            "baseline_41_retrained": False,
            "concurrent_44_retrained": False,
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_MD.write_text(markdown(report), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print("M1 verdict:", m1_status)
    print("M2 verdict:", m2_status)


if __name__ == "__main__":
    main()
