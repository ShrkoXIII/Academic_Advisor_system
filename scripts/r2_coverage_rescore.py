"""Read-only five-seed M1 R2 rescore by difficulty-coverage segment.

All parity checks complete before any model is loaded for prediction. Existing
``baseline_41`` M1 binaries are reused; no model is trained or tuned. Only the
immutable TRAIN course IDs and VALID model rows are read. TEST remains closed.
"""

from __future__ import annotations

import gc
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.r2_parity import check as parity_check  # noqa: E402
from src.model_training import prepare_X_y, resolve_feature_contract  # noqa: E402
from src.paths import (  # noqa: E402
    MODEL_DATA_VERSIONS_DIR,
    MODEL_RUNS_DIR,
    PROJECT_ROOT,
    assert_data_root,
)


DATASET_VERSION = "2026-07-26_batched_fixes__registration_roster_concurrent"
VERSION_DIR = MODEL_DATA_VERSIONS_DIR / DATASET_VERSION
TRAIN_PATH = VERSION_DIR / "df_train_final.parquet"
VALID_PATH = VERSION_DIR / "df_valid_final.parquet"
CONFIRMATION_PATH = MODEL_RUNS_DIR / "R2_CONFIRMATION_5SEED_REPORT.json"
PLAN_PATH = PROJECT_ROOT / "docs" / "EXPERIMENT_R2_COVERAGE_DECISION_PLAN.md"
OUT_MD = MODEL_RUNS_DIR / "R2_COVERED_UNCOVERED_FIVE_SEED_REPORT.md"
OUT_JSON = MODEL_RUNS_DIR / "R2_COVERED_UNCOVERED_FIVE_SEED_REPORT.json"

SEEDS = (42, 52, 62, 72, 82)
ARMS = ("control", "r2")
SEGMENTS = (
    "complete_valid",
    "covered",
    "uncovered",
    "never_in_train",
    "thin_history",
)
THRESHOLD = 0.80
MIN_SUPPORT = 20
SCALAR_METRICS = (
    "fail_rate",
    "roc_auc",
    "fail_average_precision",
    "brier",
    "fail_precision",
    "fail_recall",
    "fail_f1",
)
BENEFICIAL_POSITIVE = {
    "roc_auc",
    "fail_average_precision",
    "fail_precision",
    "fail_recall",
    "fail_f1",
}
BENEFICIAL_NEGATIVE = {"brier"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def locate_pairs(
    confirmation: dict[str, Any],
) -> dict[int, dict[str, Path]]:
    pairs: dict[int, dict[str, Path]] = {}
    runs = confirmation["runs"]
    for seed in SEEDS:
        control_key = f"control_baseline_41_{seed}"
        r2_key = f"r2_baseline_41_{seed}"
        if control_key not in runs or r2_key not in runs:
            raise FileNotFoundError(
                f"Five-seed report lacks required pair for seed {seed}"
            )
        pairs[seed] = {
            "control": ROOT / runs[control_key]["run_path"],
            "r2": ROOT / runs[r2_key]["run_path"],
        }
    return pairs


def verify_all_parity(
    confirmation: dict[str, Any],
    pairs: dict[int, dict[str, Path]],
) -> dict[str, Any]:
    """Complete every provenance gate before callers may load predictions."""

    required_global = [
        TRAIN_PATH,
        VALID_PATH,
        CONFIRMATION_PATH,
        PLAN_PATH,
    ]
    assert_data_root(*required_global)

    provenance = confirmation["provenance"]
    actual_hashes = {
        "train_sha256": sha256(TRAIN_PATH),
        "valid_sha256": sha256(VALID_PATH),
    }
    hash_match = {
        key: actual_hashes[key] == provenance[key] for key in actual_hashes
    }
    if not all(hash_match.values()):
        raise AssertionError(
            f"Immutable dataset hash mismatch: {hash_match}"
        )

    results = {}
    for seed, paths in pairs.items():
        required = []
        for arm in ARMS:
            required.extend(
                [
                    paths[arm] / "m1_pass_model.lgbm",
                    paths[arm] / "m2_grade_model.lgbm",
                    paths[arm] / "metrics.json",
                    paths[arm] / "feature_contract.json",
                ]
            )
        missing = [relative(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Frozen artifacts missing for seed {seed}: {missing}"
            )
        result = parity_check(
            paths["r2"], paths["control"], "baseline_41", seed
        )
        results[str(seed)] = result
        if not result["all_passed"]:
            raise AssertionError(
                f"Parity failed for seed {seed}: {result['failed_checks']}"
            )

    return {
        "actual_dataset_hashes": actual_hashes,
        "expected_dataset_hashes": {
            key: provenance[key] for key in actual_hashes
        },
        "dataset_hashes_match": hash_match,
        "pairs": results,
        "all_pairs_passed": True,
        "prediction_loading_started_only_after_this_gate": True,
    }


def segment_masks(
    train_course_ids: set[str], valid: pd.DataFrame
) -> dict[str, np.ndarray]:
    covered = valid["course_difficulty_missing"].eq(0)
    uncovered = valid["course_difficulty_missing"].eq(1)
    course_in_train = valid["course_id"].astype("string").isin(
        train_course_ids
    )
    never = uncovered & ~course_in_train
    thin = (
        uncovered
        & course_in_train
        & valid["course_history_count"].lt(MIN_SUPPORT)
    )
    other = uncovered & ~never & ~thin
    if int(other.sum()) != 0:
        raise AssertionError(
            f"Unexpected uncovered cause rows: {int(other.sum())}"
        )
    masks = {
        "complete_valid": np.ones(len(valid), dtype=bool),
        "covered": covered.to_numpy(),
        "uncovered": uncovered.to_numpy(),
        "never_in_train": never.to_numpy(),
        "thin_history": thin.to_numpy(),
    }
    expected = {
        "complete_valid": 156_097,
        "covered": 129_215,
        "uncovered": 26_882,
        "never_in_train": 25_627,
        "thin_history": 1_255,
    }
    actual = {name: int(mask.sum()) for name, mask in masks.items()}
    if actual != expected:
        raise AssertionError(
            f"Coverage segment counts changed: {actual} != {expected}"
        )
    return masks


def metrics(
    y_pass: np.ndarray, probability: np.ndarray
) -> dict[str, Any]:
    predicted = (probability >= THRESHOLD).astype(int)
    cm = confusion_matrix(y_pass, predicted, labels=[0, 1])
    tn, fp, fn, tp = (
        int(cm[0, 0]),
        int(cm[0, 1]),
        int(cm[1, 0]),
        int(cm[1, 1]),
    )
    return {
        "n": int(len(y_pass)),
        "fail_rate": float((y_pass == 0).mean()),
        "roc_auc": float(roc_auc_score(y_pass, probability)),
        "fail_average_precision": float(
            average_precision_score(1 - y_pass, 1 - probability)
        ),
        "brier": float(brier_score_loss(y_pass, probability)),
        "threshold": THRESHOLD,
        "fail_precision": float(
            precision_score(
                y_pass, predicted, pos_label=0, zero_division=0
            )
        ),
        "fail_recall": float(
            recall_score(y_pass, predicted, pos_label=0, zero_division=0)
        ),
        "fail_f1": float(
            f1_score(y_pass, predicted, pos_label=0, zero_division=0)
        ),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def score_existing_models(
    pairs: dict[int, dict[str, Path]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load predictions only after verify_all_parity has returned."""

    contract = resolve_feature_contract("baseline_41")
    required_columns = list(
        dict.fromkeys(
            [
                *contract.training_data_columns,
                "course_id",
                "course_history_count",
                "course_difficulty_missing",
            ]
        )
    )
    train_course = pd.read_parquet(TRAIN_PATH, columns=["course_id"])
    valid = pd.read_parquet(VALID_PATH, columns=required_columns)
    train_course_ids = set(
        train_course["course_id"].astype("string").dropna().astype(str)
    )
    masks = segment_masks(train_course_ids, valid)

    seed42_contract = load_json(
        pairs[42]["control"] / "feature_contract.json"
    )
    categorical_levels = seed42_contract["categorical_levels"]
    x_valid, y_valid = prepare_X_y(
        valid, "pass", categorical_levels, contract
    )
    y_array = y_valid.to_numpy(dtype=int)

    scored: dict[str, Any] = {}
    for seed in SEEDS:
        scored[str(seed)] = {}
        for arm in ARMS:
            model_path = pairs[seed][arm] / "m1_pass_model.lgbm"
            model = lgb.Booster(model_file=str(model_path))
            probability = model.predict(x_valid)
            arm_metrics = {
                segment: metrics(
                    y_array[mask],
                    probability[mask],
                )
                for segment, mask in masks.items()
            }

            # Whole-VALID re-score must reproduce the saved run artifact.
            saved = load_json(pairs[seed][arm] / "metrics.json")[
                "m1_pass_classifier"
            ]["valid"]
            overall = arm_metrics["complete_valid"]
            reproduction = {
                "auc_absolute_error": abs(
                    overall["roc_auc"] - saved["auc"]
                ),
                "fail_ap_absolute_error": abs(
                    overall["fail_average_precision"]
                    - saved["fail_avg_precision"]
                ),
                "brier_absolute_error": abs(
                    overall["brier"] - saved["brier"]
                ),
            }
            if max(reproduction.values()) >= 1e-12:
                raise AssertionError(
                    f"Saved metric reproduction failed seed={seed} arm={arm}: "
                    f"{reproduction}"
                )
            scored[str(seed)][arm] = {
                "run_path": relative(pairs[seed][arm]),
                "model_path": relative(model_path),
                "segments": arm_metrics,
                "saved_metric_reproduction": reproduction,
            }
            del model, probability
            gc.collect()

    segment_base_rates = {
        segment: {
            "n": int(mask.sum()),
            "fail_rate": float((y_array[mask] == 0).mean()),
        }
        for segment, mask in masks.items()
    }
    return scored, segment_base_rates


def direction(metric: str, delta: float) -> str:
    if delta == 0:
        return "zero"
    if metric in BENEFICIAL_NEGATIVE:
        return "beneficial" if delta < 0 else "harmful"
    if metric in BENEFICIAL_POSITIVE:
        return "beneficial" if delta > 0 else "harmful"
    return "not_directional"


def compute_deltas(
    scored: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    per_seed: dict[str, Any] = {}
    values: dict[str, dict[str, list[float]]] = {
        segment: {metric: [] for metric in SCALAR_METRICS}
        for segment in SEGMENTS
    }
    for seed in SEEDS:
        seed_key = str(seed)
        per_seed[seed_key] = {}
        control = scored[seed_key]["control"]["segments"]
        challenger = scored[seed_key]["r2"]["segments"]
        for segment in SEGMENTS:
            scalar = {
                metric: float(
                    challenger[segment][metric] - control[segment][metric]
                )
                for metric in SCALAR_METRICS
            }
            confusion_delta = {
                key: int(
                    challenger[segment]["confusion_matrix"][key]
                    - control[segment]["confusion_matrix"][key]
                )
                for key in ("tn", "fp", "fn", "tp")
            }
            per_seed[seed_key][segment] = {
                "r2_minus_control": scalar,
                "directions": {
                    metric: direction(metric, value)
                    for metric, value in scalar.items()
                },
                "confusion_matrix_count_delta": confusion_delta,
            }
            for metric, value in scalar.items():
                values[segment][metric].append(value)

    summaries: dict[str, Any] = {}
    for segment in SEGMENTS:
        summaries[segment] = {}
        for metric in SCALAR_METRICS:
            metric_values = values[segment][metric]
            directions = [direction(metric, value) for value in metric_values]
            summaries[segment][metric] = {
                "mean": float(statistics.fmean(metric_values)),
                "median": float(statistics.median(metric_values)),
                "sd_population": float(statistics.pstdev(metric_values)),
                "min": float(min(metric_values)),
                "max": float(max(metric_values)),
                "beneficial_count": directions.count("beneficial"),
                "harmful_count": directions.count("harmful"),
                "zero_count": directions.count("zero"),
            }
    return per_seed, summaries


def harmful_band_breach(
    metric: str, value: float, bands: dict[str, dict[str, float]]
) -> bool:
    key_map = {
        "roc_auc": "m1_valid_auc",
        "fail_average_precision": "m1_valid_fail_ap",
        "brier": "m1_valid_brier",
    }
    band = bands[key_map[metric]]
    if metric == "brier":
        return value > band["max"]
    return value < band["min"]


def evaluate_rule(
    per_seed: dict[str, Any],
    summaries: dict[str, Any],
    bands: dict[str, dict[str, float]],
) -> dict[str, Any]:
    uncovered = summaries["uncovered"]
    covered = summaries["covered"]

    clause1_checks = {
        "uncovered_auc_beneficial_at_least_4_of_5": (
            uncovered["roc_auc"]["beneficial_count"] >= 4
        ),
        "uncovered_brier_beneficial_at_least_4_of_5": (
            uncovered["brier"]["beneficial_count"] >= 4
        ),
        "uncovered_mean_auc_beneficial": uncovered["roc_auc"]["mean"] > 0,
        "uncovered_mean_brier_beneficial": uncovered["brier"]["mean"] < 0,
    }
    clause1 = {
        "checks": clause1_checks,
        "satisfied": all(clause1_checks.values()),
    }
    clause2 = {
        "uncovered_fail_ap_harmful_seeds": uncovered[
            "fail_average_precision"
        ]["harmful_count"],
        "maximum_allowed": 2,
    }
    clause2["satisfied"] = (
        clause2["uncovered_fail_ap_harmful_seeds"] <= 2
    )
    clause3_checks = {
        "covered_auc_harmful_no_more_than_2": (
            covered["roc_auc"]["harmful_count"] <= 2
        ),
        "covered_brier_harmful_no_more_than_2": (
            covered["brier"]["harmful_count"] <= 2
        ),
    }
    clause3 = {
        "checks": clause3_checks,
        "satisfied": all(clause3_checks.values()),
    }

    guardrail_metrics = ("roc_auc", "fail_average_precision", "brier")
    clause4_detail = {}
    clause4_pass = True
    for metric in guardrail_metrics:
        values = [
            per_seed[str(seed)]["complete_valid"]["r2_minus_control"][
                metric
            ]
            for seed in SEEDS
        ]
        breaches = [
            seed
            for seed, value in zip(SEEDS, values)
            if harmful_band_breach(metric, value, bands)
        ]
        mean_value = statistics.fmean(values)
        mean_breach = harmful_band_breach(metric, mean_value, bands)
        clause4_detail[metric] = {
            "per_seed_deltas": dict(zip(map(str, SEEDS), values)),
            "harmful_breach_seeds": breaches,
            "mean_delta": mean_value,
            "mean_harmful_breach": mean_breach,
        }
        clause4_pass &= not breaches and not mean_breach
    clause4 = {"metrics": clause4_detail, "satisfied": bool(clause4_pass)}

    non42 = (52, 62, 72, 82)
    non42_values = {
        segment: {
            metric: [
                per_seed[str(seed)][segment]["r2_minus_control"][metric]
                for seed in non42
            ]
            for metric in SCALAR_METRICS
        }
        for segment in ("uncovered", "covered", "complete_valid")
    }
    non42_directions = {
        segment: {
            metric: [
                direction(metric, value)
                for value in non42_values[segment][metric]
            ]
            for metric in SCALAR_METRICS
        }
        for segment in non42_values
    }
    non42_guardrail_mean_ok = all(
        not harmful_band_breach(
            metric,
            statistics.fmean(non42_values["complete_valid"][metric]),
            bands,
        )
        for metric in guardrail_metrics
    )
    clause5_checks = {
        "uncovered_auc_beneficial_at_least_3_of_4": (
            non42_directions["uncovered"]["roc_auc"].count("beneficial")
            >= 3
        ),
        "uncovered_brier_beneficial_at_least_3_of_4": (
            non42_directions["uncovered"]["brier"].count("beneficial") >= 3
        ),
        "uncovered_fail_ap_harmful_no_more_than_2_of_4": (
            non42_directions["uncovered"][
                "fail_average_precision"
            ].count("harmful")
            <= 2
        ),
        "covered_auc_harmful_no_more_than_2_of_4": (
            non42_directions["covered"]["roc_auc"].count("harmful") <= 2
        ),
        "covered_brier_harmful_no_more_than_2_of_4": (
            non42_directions["covered"]["brier"].count("harmful") <= 2
        ),
        "uncovered_mean_auc_beneficial_without_seed42": (
            statistics.fmean(non42_values["uncovered"]["roc_auc"]) > 0
        ),
        "uncovered_mean_brier_beneficial_without_seed42": (
            statistics.fmean(non42_values["uncovered"]["brier"]) < 0
        ),
        "complete_valid_mean_guardrails_unbreached_without_seed42": (
            non42_guardrail_mean_ok
        ),
    }
    clause5 = {
        "seeds": list(non42),
        "checks": clause5_checks,
        "satisfied": all(clause5_checks.values()),
    }

    clauses = {
        "1_uncovered_auc_and_brier": clause1,
        "2_uncovered_fail_ap": clause2,
        "3_covered_no_systematic_harm": clause3,
        "4_complete_valid_guardrails": clause4,
        "5_not_seed42_dependent": clause5,
    }
    all_pass = all(clause["satisfied"] for clause in clauses.values())
    return {
        "clauses": clauses,
        "all_clauses_satisfied": all_pass,
        "decision": (
            "ADOPT_R2_FOR_M1" if all_pass else "KEEP_DEFAULT_127_FOR_M1"
        ),
        "incumbent_wins_if_not_fully_met": True,
    }


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def f6(value: float, signed: bool = False) -> str:
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def render_markdown(payload: dict[str, Any]) -> str:
    scored = payload["scored_models"]
    deltas = payload["per_seed_deltas"]
    summaries = payload["five_seed_summaries"]
    decision = payload["locked_rule_evaluation"]
    lines = [
        "# R2 covered/uncovered five-seed M1 decision",
        "",
        f"**Final decision: `{decision['decision']}`.**",
        "",
        "This is a read-only rescore of existing frozen `baseline_41` M1 "
        "binaries. No model was trained, tuned, promoted, or wired. TEST remained "
        "`closed_not_read`.",
        "",
        "This read-only result does not change M2.",
        "M2 remains `concurrent_43` with `num_leaves=127`.",
        "",
        "## 1. Pre-registration",
        "",
        "Locked plan: `docs/EXPERIMENT_R2_COVERAGE_DECISION_PLAN.md`, committed "
        f"at `{payload['pre_registered_plan_commit']}` before predictions were "
        "loaded.",
        "",
        "Paired delta is always **R2 minus control**. Positive is beneficial for "
        "AUC/AP/precision/recall/F1; negative is beneficial for Brier.",
        "",
        "## 2. Exact frozen pairs",
        "",
        md_table(
            ["Seed", "Control (127)", "R2 (31)"],
            [
                [
                    seed,
                    scored[str(seed)]["control"]["run_path"],
                    scored[str(seed)]["r2"]["run_path"],
                ]
                for seed in SEEDS
            ],
        ),
        "",
        "## 3. Parity gate",
        "",
        "All five pairs passed the shared `scripts/r2_parity.py` implementation "
        "before any prediction was loaded.",
        "",
        md_table(
            ["Seed", "Checks", "Failed", "Result"],
            [
                [
                    seed,
                    payload["parity"]["pairs"][str(seed)]["check_count"],
                    ", ".join(
                        payload["parity"]["pairs"][str(seed)][
                            "failed_checks"
                        ]
                    )
                    or "none",
                    "PASS",
                ]
                for seed in SEEDS
            ],
        ),
        "",
        "Checks cover immutable TRAIN/VALID hashes, `baseline_41`, exact feature "
        "ordering, root/derived seeds, categorical levels, diploma fill, locked "
        "threshold, early stopping, TEST policy, serialized parameters, and the "
        "fact that only `num_leaves` differs (`127 → 31`). Every binary exists. "
        "Every complete-VALID re-score exactly reproduced its saved metrics.",
        "",
        "## 4. Segment population",
        "",
        md_table(
            ["Segment", "n", "Fail rate"],
            [
                [
                    segment,
                    f"{values['n']:,}",
                    f"{values['fail_rate'] * 100:.2f}%",
                ]
                for segment, values in payload["segment_base_rates"].items()
            ],
        ),
        "",
        "Covered is `course_difficulty_missing == 0`; uncovered is `== 1`. "
        "`never_in_train` and `thin_history` remain separate causes. Base rates "
        "are identical between paired arms because every model scores the same rows.",
        "",
    ]

    for segment in SEGMENTS:
        lines.extend(
            [
                f"## 5.{SEGMENTS.index(segment) + 1} Absolute metrics — `{segment}`",
                "",
                md_table(
                    [
                        "Seed",
                        "Arm",
                        "n",
                        "Fail rate",
                        "AUC",
                        "Fail AP",
                        "Brier",
                        "Fail P",
                        "Fail R",
                        "Fail F1",
                        "CM TN/FP/FN/TP",
                    ],
                    [
                        [
                            seed,
                            arm,
                            f"{entry['n']:,}",
                            f"{entry['fail_rate'] * 100:.2f}%",
                            f6(entry["roc_auc"]),
                            f6(entry["fail_average_precision"]),
                            f6(entry["brier"]),
                            f6(entry["fail_precision"]),
                            f6(entry["fail_recall"]),
                            f6(entry["fail_f1"]),
                            "/".join(
                                str(entry["confusion_matrix"][key])
                                for key in ("tn", "fp", "fn", "tp")
                            ),
                        ]
                        for seed in SEEDS
                        for arm in ARMS
                        for entry in [
                            scored[str(seed)][arm]["segments"][segment]
                        ]
                    ],
                ),
                "",
                "Paired deltas:",
                "",
                md_table(
                    [
                        "Seed",
                        "Δ AUC",
                        "Δ Fail AP",
                        "Δ Brier",
                        "Δ Fail P",
                        "Δ Fail R",
                        "Δ Fail F1",
                    ],
                    [
                        [
                            seed,
                            f6(
                                deltas[str(seed)][segment][
                                    "r2_minus_control"
                                ]["roc_auc"],
                                True,
                            ),
                            f6(
                                deltas[str(seed)][segment][
                                    "r2_minus_control"
                                ]["fail_average_precision"],
                                True,
                            ),
                            f6(
                                deltas[str(seed)][segment][
                                    "r2_minus_control"
                                ]["brier"],
                                True,
                            ),
                            f6(
                                deltas[str(seed)][segment][
                                    "r2_minus_control"
                                ]["fail_precision"],
                                True,
                            ),
                            f6(
                                deltas[str(seed)][segment][
                                    "r2_minus_control"
                                ]["fail_recall"],
                                True,
                            ),
                            f6(
                                deltas[str(seed)][segment][
                                    "r2_minus_control"
                                ]["fail_f1"],
                                True,
                            ),
                        ]
                        for seed in SEEDS
                    ],
                ),
                "",
            ]
        )

    lines.extend(["## 6. Five-seed delta summaries", ""])
    summary_rows = []
    for segment in SEGMENTS:
        for metric in SCALAR_METRICS:
            entry = summaries[segment][metric]
            summary_rows.append(
                [
                    segment,
                    metric,
                    f6(entry["mean"], True),
                    f6(entry["median"], True),
                    f6(entry["sd_population"]),
                    f6(entry["min"], True),
                    f6(entry["max"], True),
                    entry["beneficial_count"],
                    entry["harmful_count"],
                    entry["zero_count"],
                ]
            )
    lines.extend(
        [
            md_table(
                [
                    "Segment",
                    "Metric",
                    "Mean",
                    "Median",
                    "SD",
                    "Min",
                    "Max",
                    "Beneficial",
                    "Harmful",
                    "Zero",
                ],
                summary_rows,
            ),
            "",
            "## 7. Locked rule, clause by clause",
            "",
        ]
    )
    for name, clause in decision["clauses"].items():
        lines.append(
            f"- `{name}`: **{'PASS' if clause['satisfied'] else 'FAIL'}**"
        )
        if "checks" in clause:
            for check_name, passed in clause["checks"].items():
                lines.append(
                    f"  - `{check_name}`: {'pass' if passed else 'fail'}"
                )
        elif name.startswith("2_"):
            lines.append(
                f"  - harmful seeds: "
                f"{clause['uncovered_fail_ap_harmful_seeds']} "
                f"(maximum {clause['maximum_allowed']})"
            )
        elif name.startswith("4_"):
            for metric, values in clause["metrics"].items():
                lines.append(
                    f"  - `{metric}`: harmful breach seeds "
                    f"{values['harmful_breach_seeds'] or 'none'}; mean "
                    f"{values['mean_delta']:+.6f}; mean breach "
                    f"{values['mean_harmful_breach']}"
                )
    lines.extend(
        [
            "",
            f"All clauses satisfied: **{decision['all_clauses_satisfied']}**.",
            "",
            f"Therefore: **`{decision['decision']}`**.",
            "",
            "The incumbent wins whenever the rule is not fully met. No result "
            "is promoted or wired by this report.",
            "",
            "## 8. Scope confirmations",
            "",
            "- Existing frozen M1 binaries only; no retraining or retuning.",
            "- `concurrent_43` and `concurrent_44` were not scored for M1.",
            "- M2 was not rescored and its parameter decision was not reopened.",
            "- TEST was never read; policy remained `closed_not_read`.",
            "- No parquet, default, model path, promotion marker, inference, "
            "recommendation, API, eligibility, or plan-generation wiring changed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    confirmation = load_json(CONFIRMATION_PATH)
    pairs = locate_pairs(confirmation)

    # Mandatory sequencing: no LightGBM Booster is loaded above this line.
    parity = verify_all_parity(confirmation, pairs)
    print("Parity gate passed for all five pairs; loading frozen predictions.")

    scored, base_rates = score_existing_models(pairs)
    per_seed, summaries = compute_deltas(scored)
    bands = {
        key: confirmation["noise_band"][key]
        for key in (
            "m1_valid_auc",
            "m1_valid_fail_ap",
            "m1_valid_brier",
        )
    }
    decision = evaluate_rule(per_seed, summaries, bands)

    payload = {
        "experiment": "M1 baseline_41 R2 coverage decision",
        "pre_registered_plan": relative(PLAN_PATH),
        "pre_registered_plan_commit": "6fa053e",
        "scope": {
            "dataset_version": DATASET_VERSION,
            "train_path": relative(TRAIN_PATH),
            "valid_path": relative(VALID_PATH),
            "test_policy": "closed_not_read",
            "test_dataset_read": False,
            "models_trained": 0,
            "models_retuned": 0,
            "m1_contract": "baseline_41",
            "control_num_leaves": 127,
            "r2_num_leaves": 31,
            "m2_changed": False,
        },
        "noise_band": bands,
        "parity": parity,
        "segment_definitions": {
            "covered": "course_difficulty_missing == 0",
            "uncovered": "course_difficulty_missing == 1",
            "never_in_train": (
                "uncovered and course_id absent from TRAIN"
            ),
            "thin_history": (
                "uncovered and course_id present in TRAIN and "
                "course_history_count < 20"
            ),
        },
        "segment_base_rates": base_rates,
        "scored_models": scored,
        "per_seed_deltas": per_seed,
        "five_seed_summaries": summaries,
        "locked_rule_evaluation": decision,
        "m2_statement": (
            "This read-only result does not change M2. "
            "M2 remains concurrent_43 with num_leaves=127."
        ),
    }
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(decision["decision"])
    print("TEST reads: 0; models trained: 0; M2 rescored: 0")


if __name__ == "__main__":
    main()
