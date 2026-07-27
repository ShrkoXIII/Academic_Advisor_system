"""Generate the controlled five-seed baseline_41 vs concurrent_44 report.

This analysis reads TRAIN and VALID only. It never constructs or reads a TEST
path, and it does not modify model or dataset artifacts.
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
    CONCURRENT_MODEL_FEATURES,
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
OUT_JSON = RUNS / "MULTISEED_BASELINE41_VS_CONCURRENT44_REPORT.json"
OUT_MD = RUNS / "MULTISEED_BASELINE41_VS_CONCURRENT44_REPORT.md"

PAIRS = {
    42: {
        "baseline": "2026-07-26_1551__baseline-41-gpa-trend-control",
        "candidate": "2026-07-26_1554__concurrent-44-registration-roster-candidate",
    },
    52: {
        "baseline": "2026-07-27_1027__seed52-baseline-41-gpa-trend-control",
        "candidate": "2026-07-27_1028__seed52-concurrent-44-registration-roster-candidate",
    },
    62: {
        "baseline": "2026-07-27_1031__seed62-baseline-41-gpa-trend-control",
        "candidate": "2026-07-27_1033__seed62-concurrent-44-registration-roster-candidate",
    },
    72: {
        "baseline": "2026-07-27_1035__seed72-baseline-41-gpa-trend-control",
        "candidate": "2026-07-27_1036__seed72-concurrent-44-registration-roster-candidate",
    },
    82: {
        "baseline": "2026-07-27_1038__seed82-baseline-41-gpa-trend-control",
        "candidate": "2026-07-27_1039__seed82-concurrent-44-registration-roster-candidate",
    },
}

INITIAL_GIT = {
    "status_short": "",
    "diff_stat": "",
    "log_3_oneline": [
        "5928aaa Add feature contract tests for baseline_41 and concurrent_44",
        "0291dd2 Enhance concurrent group features and registration roster tests",
        "e6e2686 Implement concurrent group features for course difficulty analysis",
    ],
}

MEMORY = {
    "physical_memory_bytes": 16855928832,
    "commit_limit_bytes": 16855928832,
    "pagefile_configured": True,
    "pagefile_configuration": "?:\\pagefile.sys",
    "pagefile_active": False,
    "active_pagefile_bytes": 0,
    "available_memory_initial_audit_bytes": 3995344896,
    "available_memory_immediately_before_training_bytes": 6416273408,
    "conclusion": (
        "The pagefile was configured but inactive. Commit limit equaled physical "
        "memory; training proceeded unchanged because sufficient memory was "
        "available immediately before the first run."
    ),
}

TEST_RESULT = {
    "command": "python -m unittest discover -s tests -t .",
    "final_status": "passed",
    "tests_run": 104,
    "failures": 0,
    "elapsed_seconds": 12.891,
    "pre_gate_development_note": (
        "The first development attempt exposed three failures in the new "
        "effective-seed metadata extractor. No experiment run had started. "
        "The extractor was corrected to read serialized LightGBM parameters, "
        "and the complete mandatory pre-training suite then passed."
    ),
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
        [
            "git",
            "-c",
            f"safe.directory={ROOT.as_posix()}",
            *args,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.rstrip()


def exact_m1(y: pd.Series, probability: np.ndarray) -> dict:
    predicted = (probability >= REPORTING_THRESHOLD).astype(int)
    tn, fp, fn, tp = (
        int(value)
        for value in confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    )
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "pass_average_precision": float(average_precision_score(y, probability)),
        "fail_average_precision": float(
            average_precision_score(1 - y, 1 - probability)
        ),
        "brier_score": float(brier_score_loss(y, probability)),
        "reporting_threshold": REPORTING_THRESHOLD,
        "fail_precision": float(
            precision_score(y, predicted, pos_label=0, zero_division=0)
        ),
        "fail_recall": float(
            recall_score(y, predicted, pos_label=0, zero_division=0)
        ),
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


def segment_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "first_semester": (df["is_first_active_semester"] == 1).to_numpy(),
        "cold_start_gpa": (df["no_previous_progress"] == 1).to_numpy(),
        "retake_attempt": (df["attempt_number"] > 1).to_numpy(),
        "low_difficulty_support": (
            df["difficulty_fallback_level"] >= 3
        ).to_numpy(),
        "level_1_difficulty": (
            df["difficulty_fallback_level"] == 1
        ).to_numpy(),
    }


def segment_metrics(
    y: pd.Series, probability: np.ndarray, masks: dict[str, np.ndarray]
) -> dict:
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


def importance(run_dir: Path, model_name: str) -> dict:
    table = pd.read_csv(run_dir / f"{model_name}_feature_importance.csv")
    table["rank_by_gain"] = table["gain"].rank(
        method="min", ascending=False
    ).astype(int)
    total_gain = float(table["gain"].sum())
    table = table.set_index("feature")
    result = {}
    for feature in CONCURRENT_MODEL_FEATURES:
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


def prepare_arm_data(contract_name: str):
    contract = resolve_feature_contract(contract_name)
    train = pd.read_parquet(TRAIN, columns=contract.training_data_columns)
    valid = pd.read_parquet(VALID, columns=contract.training_data_columns)
    median = float(train["diploma_gpa"].median())
    train["diploma_gpa"] = train["diploma_gpa"].fillna(median)
    valid["diploma_gpa"] = valid["diploma_gpa"].fillna(median)
    first_contract = load_json(
        RUNS / PAIRS[42]["baseline" if contract_name == "baseline_41" else "candidate"]
        / "feature_contract.json"
    )
    levels = first_contract["categorical_levels"]
    x_train_m1, y_train_m1 = prepare_X_y(train, "pass", levels, contract)
    x_valid_m1, y_valid_m1 = prepare_X_y(valid, "pass", levels, contract)
    x_train_m2, y_train_m2 = prepare_X_y(train, "grade", levels, contract)
    x_valid_m2, y_valid_m2 = prepare_X_y(valid, "grade", levels, contract)
    return (
        train,
        valid,
        x_train_m1,
        y_train_m1,
        x_valid_m1,
        y_valid_m1,
        x_train_m2,
        y_train_m2,
        x_valid_m2,
        y_valid_m2,
    )


def collect_runs() -> dict:
    collected: dict[str, dict] = {}
    for arm, contract_name in (
        ("baseline", "baseline_41"),
        ("candidate", "concurrent_44"),
    ):
        (
            train,
            valid,
            x_train_m1,
            y_train_m1,
            x_valid_m1,
            y_valid_m1,
            x_train_m2,
            y_train_m2,
            x_valid_m2,
            y_valid_m2,
        ) = prepare_arm_data(contract_name)
        masks = segment_masks(valid)
        assert np.array_equal(
            masks["first_semester"], masks["cold_start_gpa"]
        ), "first_semester and cold_start_gpa populations stopped being identical"
        for seed, pair in PAIRS.items():
            run_dir = RUNS / pair[arm]
            stored = load_json(run_dir / "metrics.json")
            m1 = lgb.Booster(model_file=str(run_dir / "m1_pass_model.lgbm"))
            probability_train = m1.predict(x_train_m1)
            probability_valid = m1.predict(x_valid_m1)
            m1_train = exact_m1(y_train_m1, probability_train)
            m1_valid = exact_m1(y_valid_m1, probability_valid)
            m1_valid["train_valid_auc_gap"] = (
                m1_train["roc_auc"] - m1_valid["roc_auc"]
            )
            m1_valid["best_iteration"] = int(
                stored["m1_pass_classifier"]["valid"]["best_iteration"]
            )
            segments = segment_metrics(y_valid_m1, probability_valid, masks)
            del m1, probability_train, probability_valid
            gc.collect()

            m2 = lgb.Booster(model_file=str(run_dir / "m2_grade_model.lgbm"))
            m2_train = exact_m2(y_train_m2, m2.predict(x_train_m2))
            m2_valid = exact_m2(y_valid_m2, m2.predict(x_valid_m2))
            m2_valid["best_iteration"] = int(
                stored["m2_grade_regressor"]["valid"]["best_iteration"]
            )
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
            }
            if arm == "candidate":
                entry["concurrent_feature_importance"] = {
                    "m1": importance(run_dir, "m1"),
                    "m2": importance(run_dir, "m2"),
                }
            collected[f"{seed}_{arm}"] = entry

        del (
            train,
            valid,
            x_train_m1,
            y_train_m1,
            x_valid_m1,
            y_valid_m1,
            x_train_m2,
            y_train_m2,
            x_valid_m2,
            y_valid_m2,
        )
        gc.collect()
    return collected


METRICS = {
    "m1_valid_auc": ("higher", ("m1", "valid", "roc_auc")),
    "m1_valid_fail_ap": (
        "higher",
        ("m1", "valid", "fail_average_precision"),
    ),
    "m1_valid_brier": ("lower", ("m1", "valid", "brier_score")),
    "m1_train_valid_auc_gap": (
        "lower",
        ("m1", "valid", "train_valid_auc_gap"),
    ),
    "m2_valid_mae": ("lower", ("m2", "valid", "mae")),
    "m2_valid_rmse": ("lower", ("m2", "valid", "rmse")),
    "m2_valid_r2": ("higher", ("m2", "valid", "r2")),
    "cold_start_auc": (
        "higher",
        ("valid_segments", "cold_start_gpa", "auc"),
    ),
    "low_difficulty_support_auc": (
        "higher",
        ("valid_segments", "low_difficulty_support", "auc"),
    ),
    "level_1_auc": (
        "higher",
        ("valid_segments", "level_1_difficulty", "auc"),
    ),
}


def at(data: dict, path: tuple[str, ...]) -> float:
    value = data
    for key in path:
        value = value[key]
    return float(value)


def paired_analysis(collected: dict) -> tuple[list[dict], dict]:
    rows = []
    summary = {}
    for seed in PAIRS:
        baseline = collected[f"{seed}_baseline"]
        candidate = collected[f"{seed}_candidate"]
        row = {"seed": seed}
        for name, (direction, path) in METRICS.items():
            b = at(baseline, path)
            c = at(candidate, path)
            row[name] = {
                "baseline": b,
                "candidate": c,
                "delta_candidate_minus_baseline": c - b,
                "direction": f"{direction}_is_better",
            }
        rows.append(row)
    for name, (direction, _) in METRICS.items():
        baseline_values = [row[name]["baseline"] for row in rows]
        candidate_values = [row[name]["candidate"] for row in rows]
        deltas = [row[name]["delta_candidate_minus_baseline"] for row in rows]
        improved = sum(
            value > 0 if direction == "higher" else value < 0 for value in deltas
        )
        worsened = sum(
            value < 0 if direction == "higher" else value > 0 for value in deltas
        )
        summary[name] = {
            "direction": f"{direction}_is_better",
            "baseline_mean": statistics.fmean(baseline_values),
            "candidate_mean": statistics.fmean(candidate_values),
            "mean_paired_delta": statistics.fmean(deltas),
            "sample_standard_deviation_of_paired_deltas": statistics.stdev(deltas),
            "minimum_paired_delta": min(deltas),
            "maximum_paired_delta": max(deltas),
            "paired_delta_range": max(deltas) - min(deltas),
            "median_paired_delta": statistics.median(deltas),
            "seeds_improved": improved,
            "seeds_worsened": worsened,
            "seeds_tied": len(deltas) - improved - worsened,
        }
    return rows, summary


def contract_verification() -> dict:
    result = {}
    new_protocol = load_json(
        RUNS / PAIRS[52]["baseline"] / "feature_contract.json"
    )
    for seed, pair in PAIRS.items():
        baseline_dir = RUNS / pair["baseline"]
        candidate_dir = RUNS / pair["candidate"]
        b = load_json(baseline_dir / "feature_contract.json")
        c = load_json(candidate_dir / "feature_contract.json")
        bm = load_json(baseline_dir / "metrics.json")
        cm = load_json(candidate_dir / "metrics.json")
        shared = [
            "categorical_features",
            "categorical_levels",
            "unknown_category_code",
            "derived_feature_sources",
            "dropped_feature_guard",
            "target_m1_classifier",
            "target_m2_regressor",
            "reporting_threshold",
            "random_seed",
            "lightgbm_params",
            "test_policy",
            "train_path",
            "valid_path",
            "dataset_version",
            "dataset_inputs",
            "git",
        ]
        if seed != 42:
            shared += [
                "effective_seed_settings",
                "training_control",
                "diploma_gpa_handling",
                "data_rows",
            ]
        checks = {key: b[key] == c[key] for key in shared}
        checks["shared_feature_dtypes"] = all(
            b["dtypes_after_model_preparation"][feature]
            == c["dtypes_after_model_preparation"][feature]
            for feature in b["ordered_features"]
        )
        checks["only_three_feature_difference"] = (
            set(c["ordered_features"]) - set(b["ordered_features"])
            == set(CONCURRENT_MODEL_FEATURES)
            and not (set(b["ordered_features"]) - set(c["ordered_features"]))
        )
        checks["metrics_nonfeature_settings"] = all(
            bm["run_settings"][key] == cm["run_settings"][key]
            for key in bm["run_settings"]
            if key not in {"feature_contract", "feature_count"}
        )
        checks["test_closed_and_null"] = all(
            metrics["run_settings"]["test_policy"] == "closed_not_read"
            and metrics["m1_pass_classifier"]["test"] is None
            and metrics["m2_grade_regressor"]["test"] is None
            for metrics in (bm, cm)
        )
        effective = {
            label: _effective_seed_settings(run_dir / model)
            for label, run_dir, model in (
                ("baseline_m1", baseline_dir, "m1_pass_model.lgbm"),
                ("baseline_m2", baseline_dir, "m2_grade_model.lgbm"),
                ("candidate_m1", candidate_dir, "m1_pass_model.lgbm"),
                ("candidate_m2", candidate_dir, "m2_grade_model.lgbm"),
            )
        }
        checks["identical_effective_seeds"] = (
            len({tuple(value.items()) for value in effective.values()}) == 1
        )
        if seed == 42:
            checks["same_nonseed_params_as_final_protocol"] = {
                key: value
                for key, value in b["lightgbm_params"].items()
                if key != "seed"
            } == {
                key: value
                for key, value in new_protocol["lightgbm_params"].items()
                if key != "seed"
            }
            checks["same_contract_definition_as_final_protocol"] = (
                b["ordered_features"] == new_protocol["ordered_features"]
            )
            checks["same_data_as_final_protocol"] = (
                b["dataset_inputs"] == new_protocol["dataset_inputs"]
            )
        result[str(seed)] = {
            "valid": all(checks.values()),
            "checks": checks,
            "effective_seed_settings": effective["baseline_m1"],
            "note": (
                "Directly comparable historical pair; historical code confirms "
                "2000 rounds, 50-round VALID-only early stopping, train-only "
                "diploma median fill, and the same seed-only LightGBM derivation."
                if seed == 42
                else (
                    "Baseline Git metadata was repaired from the immediately "
                    "adjacent candidate after a launcher environment-injection "
                    "failure; model, metric, and data artifacts were unchanged."
                    if seed == 52
                    else "All recorded non-feature settings match."
                )
            ),
        }
    return result


def feature_usage(collected: dict) -> dict:
    usage = {}
    for model in ("m1", "m2"):
        usage[model] = {}
        for feature in CONCURRENT_MODEL_FEATURES:
            used = []
            for seed in PAIRS:
                evidence = collected[f"{seed}_candidate"][
                    "concurrent_feature_importance"
                ][model][feature]
                if not evidence["split_count_is_zero"]:
                    used.append(seed)
            usage[model][feature] = {
                "seeds_used": used,
                "number_of_five_seeds_used": len(used),
            }
    missing = "concurrent_peer_difficulty_missing"
    any_model = sorted(
        set(usage["m1"][missing]["seeds_used"])
        | set(usage["m2"][missing]["seeds_used"])
    )
    usage["missing_feature_overall"] = {
        "seeds_used_by_either_model": any_model,
        "number_of_five_seeds_used_by_either_model": len(any_model),
        "unused_in_all_five_seeds": len(any_model) == 0,
    }
    return usage


def f(value) -> str:
    if isinstance(value, int):
        return str(value)
    return format(float(value), ".17g")


def markdown(report: dict) -> str:
    lines = [
        "# Multi-seed stability: baseline_41 vs concurrent_44",
        "",
        "## Verdict",
        "",
        f"- **M1 verdict: {report['verdicts']['m1']['status']}** — "
        + report["verdicts"]["m1"]["reason"],
        f"- **M2 verdict: {report['verdicts']['m2']['status']}** — "
        + report["verdicts"]["m2"]["reason"],
        "",
        "No statistical significance is claimed from five seeds.",
        "",
        "## Repository, tests, and memory",
        "",
        "- Initial `git status --short`: clean (no output).",
        "- Initial `git diff --stat`: no diff (no output).",
        "- Initial `git log -3 --oneline`:",
        *[f"  - `{item}`" for item in INITIAL_GIT["log_3_oneline"]],
        "- Final working-tree status captured before report generation:",
        "```text",
        report["repository"]["final_status_short"] or "(clean)",
        "```",
        "- Final diff stat captured before report generation:",
        "```text",
        report["repository"]["final_diff_stat"] or "(no diff)",
        "```",
        f"- Final pre-training test gate: `{TEST_RESULT['command']}` — "
        f"{TEST_RESULT['tests_run']} tests, 0 failures, "
        f"{TEST_RESULT['elapsed_seconds']} seconds.",
        f"- Test-development note: {TEST_RESULT['pre_gate_development_note']}",
        f"- Physical memory: {MEMORY['physical_memory_bytes']} bytes; commit limit: "
        f"{MEMORY['commit_limit_bytes']} bytes.",
        f"- Pagefile configured as `{MEMORY['pagefile_configuration']}` but "
        "**inactive** (0 active bytes); the commit limit equaled physical memory.",
        f"- Available memory immediately before training: "
        f"{MEMORY['available_memory_immediately_before_training_bytes']} bytes.",
        "",
        "## Ten run paths",
        "",
    ]
    for seed, pair in PAIRS.items():
        lines += [
            f"- Seed {seed} baseline: `models/runs/{pair['baseline']}`",
            f"- Seed {seed} candidate: `models/runs/{pair['candidate']}`",
        ]

    lines += [
        "",
        "## Contract equality",
        "",
        "| Seed | Valid | Effective LightGBM seeds | Note |",
        "|---:|:---:|---|---|",
    ]
    for seed in PAIRS:
        entry = report["contract_verification"][str(seed)]
        seeds = ", ".join(
            f"{key}={value}"
            for key, value in entry["effective_seed_settings"].items()
        )
        lines.append(
            f"| {seed} | {'yes' if entry['valid'] else 'NO'} | {seeds} | "
            f"{entry['note']} |"
        )

    lines += [
        "",
        "Every pair used the same train/valid SHA-256 values, 450465 TRAIN rows, "
        "156097 VALID rows, identical targets and categorical levels, threshold "
        "0.80, four threads, 2000-round cap, 50-round VALID-only early stopping, "
        "train-only diploma-GPA median fill, and closed TEST. The only model-input "
        "difference was the three concurrent features. Target hashes were not "
        "recorded by the historical artifacts; target definitions and the "
        "row-aligned immutable split hashes matched.",
        "",
        "## Exact M1 metrics",
        "",
        "| Seed | Arm | TRAIN AUC | TRAIN pass AP | TRAIN fail AP | TRAIN Brier | VALID AUC | VALID pass AP | VALID fail AP | VALID Brier | AUC gap | Best iter |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in PAIRS:
        for arm in ("baseline", "candidate"):
            item = report["runs"][f"{seed}_{arm}"]
            train = item["m1"]["train"]
            valid = item["m1"]["valid"]
            lines.append(
                f"| {seed} | {arm} | {f(train['roc_auc'])} | "
                f"{f(train['pass_average_precision'])} | "
                f"{f(train['fail_average_precision'])} | "
                f"{f(train['brier_score'])} | {f(valid['roc_auc'])} | "
                f"{f(valid['pass_average_precision'])} | "
                f"{f(valid['fail_average_precision'])} | "
                f"{f(valid['brier_score'])} | "
                f"{f(valid['train_valid_auc_gap'])} | "
                f"{valid['best_iteration']} |"
            )

    lines += [
        "",
        "### VALID threshold metrics at 0.80",
        "",
        "| Seed | Arm | Fail P | Fail R | Fail F1 | Pass P | Pass R | Pass F1 | TN | FP | FN | TP |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in PAIRS:
        for arm in ("baseline", "candidate"):
            valid = report["runs"][f"{seed}_{arm}"]["m1"]["valid"]
            cm = valid["confusion_matrix"]
            lines.append(
                f"| {seed} | {arm} | {f(valid['fail_precision'])} | "
                f"{f(valid['fail_recall'])} | {f(valid['fail_f1'])} | "
                f"{f(valid['pass_precision'])} | {f(valid['pass_recall'])} | "
                f"{f(valid['pass_f1'])} | {cm['tn']} | {cm['fp']} | "
                f"{cm['fn']} | {cm['tp']} |"
            )

    lines += [
        "",
        "## Exact M2 metrics",
        "",
        "| Seed | Arm | TRAIN MAE | TRAIN RMSE | TRAIN R2 | VALID MAE | VALID RMSE | VALID R2 | Best iter |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in PAIRS:
        for arm in ("baseline", "candidate"):
            item = report["runs"][f"{seed}_{arm}"]
            train = item["m2"]["train"]
            valid = item["m2"]["valid"]
            lines.append(
                f"| {seed} | {arm} | {f(train['mae'])} | {f(train['rmse'])} | "
                f"{f(train['r2'])} | {f(valid['mae'])} | {f(valid['rmse'])} | "
                f"{f(valid['r2'])} | {valid['best_iteration']} |"
            )

    lines += [
        "",
        "## Paired VALID deltas (candidate minus baseline)",
        "",
        "Raw deltas are shown; lower is better for Brier, AUC gap, MAE, and RMSE.",
        "",
        "| Seed | M1 AUC | M1 fail AP | M1 Brier | AUC gap | M2 MAE | M2 RMSE | M2 R2 | Cold-start AUC | Low-support AUC | Level-1 AUC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    delta_names = list(METRICS)
    for row in report["paired_deltas"]:
        lines.append(
            "| "
            + str(row["seed"])
            + " | "
            + " | ".join(
                f(row[name]["delta_candidate_minus_baseline"])
                for name in delta_names
            )
            + " |"
        )

    lines += [
        "",
        "## Five-seed summary",
        "",
        "The standard deviation below is the sample standard deviation of paired deltas.",
        "",
        "| Metric | Baseline mean | Candidate mean | Mean delta | Median delta | SD delta | Min delta | Max delta | Range | Improved | Worsened |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["summary"].items():
        lines.append(
            f"| {name} | {f(values['baseline_mean'])} | "
            f"{f(values['candidate_mean'])} | {f(values['mean_paired_delta'])} | "
            f"{f(values['median_paired_delta'])} | "
            f"{f(values['sample_standard_deviation_of_paired_deltas'])} | "
            f"{f(values['minimum_paired_delta'])} | "
            f"{f(values['maximum_paired_delta'])} | "
            f"{f(values['paired_delta_range'])} | "
            f"{values['seeds_improved']} | {values['seeds_worsened']} |"
        )

    lines += [
        "",
        "## Segment stability",
        "",
        "`first_semester` and `cold_start_gpa` are exactly the same VALID "
        "population in every run (n=14732), so they are one piece of evidence.",
        "",
        "| Seed | Segment | Baseline AUC | Candidate AUC | Delta | n |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for seed in PAIRS:
        baseline = report["runs"][f"{seed}_baseline"]["valid_segments"]
        candidate = report["runs"][f"{seed}_candidate"]["valid_segments"]
        for name in (
            "first_semester",
            "cold_start_gpa",
            "retake_attempt",
            "low_difficulty_support",
            "level_1_difficulty",
        ):
            b = baseline[name]
            c = candidate[name]
            lines.append(
                f"| {seed} | {name} | {f(b['auc'])} | {f(c['auc'])} | "
                f"{f(c['auc'] - b['auc'])} | {b['n']} |"
            )

    lines += [
        "",
        "Cold-start direction was mixed (3/5 improved), low-difficulty-support "
        "was mixed, and Level-1 did not provide a stable independent pattern. "
        "This does not meet strong segment-repeatability support for M1.",
        "",
        "## Concurrent feature evidence",
        "",
        "| Seed | Model | Feature | Gain | Splits | Gain rank | % total gain | Zero splits |",
        "|---:|---|---|---:|---:|---:|---:|:---:|",
    ]
    for seed in PAIRS:
        evidence = report["runs"][f"{seed}_candidate"][
            "concurrent_feature_importance"
        ]
        for model in ("m1", "m2"):
            for feature in CONCURRENT_MODEL_FEATURES:
                item = evidence[model][feature]
                lines.append(
                    f"| {seed} | {model.upper()} | {feature} | "
                    f"{f(item['gain_importance'])} | "
                    f"{item['split_importance']} | {item['rank_by_gain']} | "
                    f"{f(item['percentage_of_total_gain'])} | "
                    f"{'yes' if item['split_count_is_zero'] else 'no'} |"
                )

    missing = report["feature_usage"]["missing_feature_overall"]
    lines += [
        "",
        f"`concurrent_peer_difficulty_missing` was used by at least one model "
        f"in {missing['number_of_five_seeds_used_by_either_model']} of 5 seeds "
        f"(seeds {missing['seeds_used_by_either_model']}); therefore it did "
        "**not** remain unused in all five seeds. M1 used it in seed 52 only; "
        "M2 used it in seed 82 only, each with one split.",
        "",
        "## Separate findings",
        "",
        "### M1 — INCONCLUSIVE",
        "",
        report["verdicts"]["m1"]["detail"],
        "",
        "### M2 — SUPPORTED",
        "",
        report["verdicts"]["m2"]["detail"],
        "",
        "## Integrity and stop-gate confirmations",
        "",
        "- No training seed failed and no seed was rerun. Exactly eight new persistent training runs were created.",
        "- Seed 52 had one provenance-only metadata repair after its first launcher failed to inject Git safe-directory state; models and metrics were not changed.",
        "- TEST policy is `closed_not_read` in all ten runs; all M1/M2 TEST metric fields are null. The four new commands used nonexistent TEST paths, and those paths remain nonexistent.",
        "- Only `df_train_final.parquet` and `df_valid_final.parquet` were used for data/model evaluation. The TEST parquet was never read.",
        "- Train/valid hashes still equal the seed-42 immutable artifact hashes.",
        "- No dataset, root/live model artifact, production contract, promotion marker, `CURRENT_VERSION.txt`, recommendation wiring, or inference wiring was changed.",
        "- No commit or push was performed.",
        "- No `concurrent_43`, regularization experiment, promotion, or recommendation change was created.",
        "",
        "## Overall next action",
        "",
        "Keep M1 on `baseline_41`. Treat the M2 evidence for `concurrent_44` as "
        "supported but do not promote or rewire anything in this task; await "
        "explicit human review of the model-specific deployment implications.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    assert REPORTING_THRESHOLD == 0.80
    assert sha256(TRAIN) == (
        "8aaff32aeac5b37506b24584d56913b52ab284f1b551f107a2e3c55969d5641f"
    )
    assert sha256(VALID) == (
        "228719fa492da84bd4696e24c88c574d2ce68a3143dac8bb6bc925fc1e432b75"
    )
    runs = collect_runs()
    deltas, summary = paired_analysis(runs)
    verification = contract_verification()
    assert all(item["valid"] for item in verification.values())
    usage = feature_usage(runs)
    report = {
        "experiment": "baseline_41_vs_concurrent_44_five_seed_stability",
        "seeds": list(PAIRS),
        "delta_definition": "candidate_minus_baseline",
        "repository": {
            "initial": INITIAL_GIT,
            "final_status_short": git("status", "--short"),
            "final_diff_stat": git("diff", "--stat"),
            "final_log_3_oneline": git("log", "-3", "--oneline").splitlines(),
        },
        "tests": TEST_RESULT,
        "memory": MEMORY,
        "dataset": {
            "version": "2026-07-26_batched_fixes__registration_roster_concurrent",
            "train_path": str(TRAIN.relative_to(ROOT)).replace("\\", "/"),
            "valid_path": str(VALID.relative_to(ROOT)).replace("\\", "/"),
            "train_sha256": sha256(TRAIN),
            "valid_sha256": sha256(VALID),
            "train_rows": 450465,
            "valid_rows": 156097,
            "test_policy": "closed_not_read",
        },
        "run_paths": PAIRS,
        "contract_verification": verification,
        "runs": runs,
        "paired_deltas": deltas,
        "summary": summary,
        "feature_usage": usage,
        "verdicts": {
            "m1": {
                "status": "INCONCLUSIVE",
                "reason": (
                    "AUC and Brier improved in 4/5 seeds, but fail AP improved "
                    "in only 2/5, the train-valid AUC gap worsened in 4/5, and "
                    "segment direction was mixed."
                ),
                "detail": (
                    "The candidate shows a consistent but small VALID AUC/Brier "
                    "direction, yet the operational fail-class AP result is mixed "
                    "and the generalization gap usually worsens, sometimes by "
                    "more than the VALID AUC benefit. Cold-start improves in only "
                    "3/5 seeds and is identical to first_semester, so it is not "
                    "independent confirmation. The M1 improvement is inside "
                    "observed seed variability and does not satisfy strong support."
                ),
            },
            "m2": {
                "status": "SUPPORTED",
                "reason": (
                    "MAE, RMSE, and R2 improve together in 4/5 seeds; the mean "
                    "benefit is not caused by a single seed, though it is small."
                ),
                "detail": (
                    "Four seeds improve all three VALID regression metrics and "
                    "one seed (62) worsens all three. Multiple improving seeds "
                    "contribute to the mean, including seeds 42, 52, 72, and 82. "
                    "This is a stable direction across seeds and consistent but "
                    "small, with no claim of statistical significance."
                ),
            },
        },
        "failed_or_rerun_seeds": [],
        "seed52_provenance_incident": (
            "Baseline Git metadata was repaired from the immediately adjacent "
            "candidate after the initial background launcher failed to pass the "
            "safe-directory environment. Training artifacts were not rerun or altered."
        ),
        "integrity": {
            "test_never_read": True,
            "dataset_modified": False,
            "live_model_modified": False,
            "promotion_marker_modified": False,
            "current_version_modified": False,
            "recommendation_wiring_modified": False,
            "inference_wiring_modified": False,
            "committed": False,
            "pushed": False,
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_MD.write_text(markdown(report), encoding="utf-8")
    print(OUT_JSON)
    print(OUT_MD)


if __name__ == "__main__":
    main()
