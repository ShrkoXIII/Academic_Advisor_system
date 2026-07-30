"""Phase 3 PILOT: paired evaluation of one locked model on two VALID frames.

STATUS: PILOT on PENDING / UNREVIEWED mapping proposals. Nothing here approves a
proposal row, promotes a model, or authorizes a freeze. TEST is never read.

Design (per the task addendum, section 1): the pilot TRAIN frame is byte-identical
to the frozen TRAIN frame and the contracts and hyperparameters are unchanged, so
for a given seed the baseline and with-prior training inputs are identical. Training
a second nominal model would measure nothing. Instead, for each seed ONE locked
model artifact is evaluated on the frozen VALID frame and on the pilot VALID frame,
and the identity of that artifact across both evaluations is recorded by hash.

Locked specs, not retuned:
    M1 = baseline_41,   num_leaves=127   (reused from the 5-seed baseline runs)
    M2 = concurrent_43, num_leaves=127   (reused from the 5-seed concurrent_43 runs)

Every reused artifact is verified by recomputing its recorded frozen-VALID metrics
from the booster on disk before it is used as the paired instrument.
"""

from __future__ import annotations

from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_training import (  # noqa: E402
    BASELINE_41_CONTRACT,
    CONCURRENT_43_CONTRACT,
    REPORTING_THRESHOLD,
    learn_categorical_levels,
    prepare_X_y,
)
from src.paths import MODEL_DATA_VERSIONS_DIR, MODEL_RUNS_DIR  # noqa: E402

FROZEN_VERSION = "2026-07-26_batched_fixes__registration_roster_concurrent"
PILOT_VERSION = "2026-07-30_predecessor_prior_pilot_PENDING_REVIEW"
FROZEN_DIR = MODEL_DATA_VERSIONS_DIR / FROZEN_VERSION
PILOT_DIR = MODEL_DATA_VERSIONS_DIR / PILOT_VERSION

FROZEN_TRAIN = FROZEN_DIR / "df_train_final.parquet"
FROZEN_VALID = FROZEN_DIR / "df_valid_final.parquet"
PILOT_TRAIN = PILOT_DIR / "df_train_final.parquet"
PILOT_VALID = PILOT_DIR / "df_valid_final.parquet"
ROW_SEGMENTS = PILOT_DIR / "row_segments.parquet"
LINK_PATH = PROJECT_ROOT / "models" / "runs" / "phase2_link_corrections" / "course_link_proposed.csv"

OUT_DIR = MODEL_RUNS_DIR / "phase3_predecessor_prior_pilot"
SEEDS = (42, 52, 62, 72, 82)

# Locked-spec source runs on the frozen dataset version. M1 comes from the
# baseline_41 arm, M2 from the concurrent_43 arm, per the locked decisions.
M1_SOURCE_RUNS = {
    42: "2026-07-26_1551__baseline-41-gpa-trend-control",
    52: "2026-07-27_1027__seed52-baseline-41-gpa-trend-control",
    62: "2026-07-27_1031__seed62-baseline-41-gpa-trend-control",
    72: "2026-07-27_1035__seed72-baseline-41-gpa-trend-control",
    82: "2026-07-27_1038__seed82-baseline-41-gpa-trend-control",
}
M2_SOURCE_RUNS = {
    42: "2026-07-27_1327__seed42-concurrent-43-drop-dead-missing-flag",
    52: "2026-07-27_1328__seed52-concurrent-43-drop-dead-missing-flag",
    62: "2026-07-27_1329__seed62-concurrent-43-drop-dead-missing-flag",
    72: "2026-07-27_1330__seed72-concurrent-43-drop-dead-missing-flag",
    82: "2026-07-27_1331__seed82-concurrent-43-drop-dead-missing-flag",
}
LOCKED_PARAMS = {"num_leaves": 127, "min_child_samples": 50, "reg_lambda": 1.0}

# The repository's existing numerical tolerance.
PREDICTION_ATOL = 1e-12
# Recomputed-vs-recorded metric agreement; metrics.json stores several values
# rounded to 4 decimals, so equality is asserted at that resolution.
METRIC_ATOL = 5e-5

WEIGHTED_RELATIONSHIPS = ("successor", "consolidated_into")


def stop(message: str) -> None:
    raise SystemExit(f"STOP: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_no_test(path: Path) -> None:
    lowered = str(path).lower().replace("\\", "/")
    if "df_test" in lowered or lowered.endswith("_test.parquet"):
        stop(f"a TEST path entered the Phase 3 evaluation allowlist: {path}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def m1_metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    n = int(len(y))
    if n == 0:
        return {"n": 0, "status": "empty"}
    positives = int(y.sum())
    if positives in (0, n):
        return {"n": n, "status": "one_class_only", "positive_rate": positives / n}
    binary = (prob >= REPORTING_THRESHOLD).astype(int)
    return {
        "n": n,
        "status": "ok",
        "positive_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, prob)),
        "fail_avg_precision": float(average_precision_score(1 - y, 1 - prob)),
        "brier": float(brier_score_loss(y, prob)),
        "fail_precision": float(precision_score(y, binary, pos_label=0, zero_division=0)),
        "fail_recall": float(recall_score(y, binary, pos_label=0, zero_division=0)),
        "fail_f1": float(f1_score(y, binary, pos_label=0, zero_division=0)),
        "precision": float(precision_score(y, binary, zero_division=0)),
        "recall": float(recall_score(y, binary, zero_division=0)),
        "f1": float(f1_score(y, binary, zero_division=0)),
    }


def m2_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    n = int(len(y))
    if n == 0:
        return {"n": 0, "status": "empty"}
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float(((y - pred) ** 2).sum())
    return {
        "n": n,
        "status": "ok",
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
    }


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------


def build_segments(link: pd.DataFrame, segments: pd.DataFrame) -> dict[str, np.ndarray]:
    weighted = link.loc[link["relationship_type"].isin(WEIGHTED_RELATIONSHIPS)]
    scope = (
        weighted.drop_duplicates("new_course_id")
        .set_index("new_course_id")["new_course_scope"]
    )
    credit_changed = (
        weighted.assign(_c=weighted["credit_changed"].eq("true"))
        .groupby("new_course_id")["_c"]
        .any()
    )

    course = segments["course_id"].astype(str)
    direct = segments["directly_eligible"].to_numpy(dtype=bool)
    exposed = segments["propagation_exposed"].to_numpy(dtype=bool)
    covered = segments["covered"].to_numpy(dtype=bool)
    untouched = segments["untouched_uncovered"].to_numpy(dtype=bool)
    never_in_train = segments["never_in_train_182"].to_numpy(dtype=bool)

    row_scope = course.map(scope).astype(object).fillna("").to_numpy()
    row_credit_changed = course.map(credit_changed).fillna(False).to_numpy(dtype=bool)
    if int((direct & (row_scope == "")).sum()):
        stop("a directly eligible row has no new_course_scope in the link table")

    return {
        # 1. headline, diluted
        "overall_valid": np.ones(len(segments), dtype=bool),
        "overall_uncovered_never_in_train_182": never_in_train,
        # 2. the real signal
        "affected": direct,
        # 3. scope split within the affected segment
        "affected_scope_shared": direct & (row_scope == "shared"),
        "affected_scope_specific": direct & (row_scope == "specific"),
        # 4. credit-change split within the affected segment
        "affected_credit_changed": direct & row_credit_changed,
        "affected_credit_unchanged": direct & ~row_credit_changed,
        # 5/6. Clause-0 sanity segments, EXPOSED ROWS EXCLUDED
        "covered_unexposed": covered & ~exposed & ~direct,
        "untouched_uncovered_unexposed": untouched & ~exposed & ~direct,
        "completely_unexposed": ~exposed & ~direct,
        # diagnostic
        "indirect_propagation_only": exposed & ~direct,
    }


CLAUSE_0_SEGMENTS = (
    "covered_unexposed",
    "untouched_uncovered_unexposed",
    "completely_unexposed",
)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def verify_source_run(run_dir: Path, seed: int, contract_name: str, model_file: str) -> dict[str, Any]:
    contract_path = run_dir / "feature_contract.json"
    metrics_path = run_dir / "metrics.json"
    model_path = run_dir / model_file
    for path in (contract_path, metrics_path, model_path):
        if not path.is_file():
            stop(f"reusable run {run_dir.name} is missing {path.name}")
    recorded = json.loads(contract_path.read_text(encoding="utf-8"))
    params = recorded["lightgbm_params"]
    if recorded["contract_name"] != contract_name:
        stop(f"{run_dir.name}: contract is {recorded['contract_name']}, expected {contract_name}")
    if int(params["seed"]) != seed:
        stop(f"{run_dir.name}: seed is {params['seed']}, expected {seed}")
    for key, value in LOCKED_PARAMS.items():
        if params[key] != value:
            stop(f"{run_dir.name}: {key} is {params[key]!r}, expected {value!r} (locked spec)")
    if recorded["dataset_version"] != FROZEN_VERSION:
        stop(f"{run_dir.name}: dataset_version is {recorded['dataset_version']}")
    if recorded["test_policy"] != "closed_not_read":
        stop(f"{run_dir.name}: test_policy is {recorded['test_policy']}")
    frozen_train_hash = sha256_file(FROZEN_TRAIN)
    frozen_valid_hash = sha256_file(FROZEN_VALID)
    if recorded["dataset_inputs"]["train"]["sha256"] != frozen_train_hash:
        stop(f"{run_dir.name}: recorded TRAIN hash does not match the frozen TRAIN on disk")
    if recorded["dataset_inputs"]["valid"]["sha256"] != frozen_valid_hash:
        stop(f"{run_dir.name}: recorded VALID hash does not match the frozen VALID on disk")
    return {
        "run": run_dir.name,
        "model_file": model_file,
        "model_sha256": sha256_file(model_path),
        "contract": recorded["contract_name"],
        "seed": seed,
        "lightgbm_params": {k: params[k] for k in sorted(LOCKED_PARAMS)},
        # The earliest run predates this provenance field; when it is absent the
        # fill is still verified indirectly, because a different fill could not
        # reproduce the run's own recorded frozen-VALID metrics.
        "diploma_gpa_fill_value": (recorded.get("diploma_gpa_handling") or {}).get("fill_value"),
        "git_working_tree_clean": recorded["git"]["working_tree_clean"],
        "git_commit": recorded["git"]["commit"],
        "recorded_train_sha256_matches_disk": True,
        "recorded_valid_sha256_matches_disk": True,
    }


def main() -> int:
    started = datetime.now().astimezone()
    stamp = started.strftime("%Y-%m-%d_%H%M")
    for path in (FROZEN_TRAIN, FROZEN_VALID, PILOT_TRAIN, PILOT_VALID):
        assert_no_test(path)
        if not path.is_file():
            stop(f"required frame is missing: {path}")

    if sha256_file(PILOT_TRAIN) != sha256_file(FROZEN_TRAIN):
        stop("pilot TRAIN is not byte-identical to frozen TRAIN")

    link = pd.read_csv(LINK_PATH, dtype="string", keep_default_na=False)
    segments_frame = pd.read_parquet(ROW_SEGMENTS)
    segments = build_segments(link, segments_frame)
    segment_sizes = {name: int(mask.sum()) for name, mask in segments.items()}
    print("Segment sizes:", json.dumps(segment_sizes, indent=1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "artifact": "phase3_predecessor_prior_pilot_evaluation",
        "status": "PILOT — PENDING/UNREVIEWED MAPPINGS — NOT FOR PROMOTION",
        "created_at": started.isoformat(timespec="seconds"),
        "design": (
            "paired evaluation: one locked model artifact per seed, evaluated on the "
            "frozen VALID frame and on the pilot VALID frame; TRAIN identical by hash"
        ),
        "frozen_version": FROZEN_VERSION,
        "pilot_version": PILOT_VERSION,
        "reporting_threshold": REPORTING_THRESHOLD,
        "test_policy": "closed_not_read",
        "segment_sizes": segment_sizes,
        "seeds": {},
    }

    for model_key, contract, source_runs, model_file, target in (
        ("m1_pass_classifier", BASELINE_41_CONTRACT, M1_SOURCE_RUNS, "m1_pass_model.lgbm", "pass"),
        ("m2_grade_regressor", CONCURRENT_43_CONTRACT, M2_SOURCE_RUNS, "m2_grade_model.lgbm", "grade"),
    ):
        print(f"\n=== {model_key} — contract {contract.name} ===")
        train = pd.read_parquet(FROZEN_TRAIN, columns=contract.training_data_columns)
        frozen_valid = pd.read_parquet(FROZEN_VALID, columns=contract.training_data_columns)
        pilot_valid = pd.read_parquet(PILOT_VALID, columns=contract.training_data_columns)

        # Same train-median diploma_gpa fill the CLI applies, learned on TRAIN only.
        fill_value = float(train["diploma_gpa"].median())
        for frame in (train, frozen_valid, pilot_valid):
            frame["diploma_gpa"] = frame["diploma_gpa"].fillna(fill_value)

        levels = learn_categorical_levels(train, contract)
        X_train, y_train = prepare_X_y(train, target, levels, contract)
        X_frozen, y_frozen = prepare_X_y(frozen_valid, target, levels, contract)
        X_pilot, y_pilot = prepare_X_y(pilot_valid, target, levels, contract)
        if not np.array_equal(y_frozen.to_numpy(), y_pilot.to_numpy()):
            stop(f"{model_key}: the pilot VALID target differs from the frozen VALID target")
        del train, frozen_valid, pilot_valid
        gc.collect()

        y_arr = y_frozen.to_numpy()
        for seed in SEEDS:
            run_dir = MODEL_RUNS_DIR / source_runs[seed]
            provenance = verify_source_run(run_dir, seed, contract.name, model_file)
            if provenance["diploma_gpa_fill_value"] is not None and not np.isclose(
                float(provenance["diploma_gpa_fill_value"]), fill_value
            ):
                stop(
                    f"{run_dir.name}: recorded diploma_gpa fill {provenance['diploma_gpa_fill_value']} "
                    f"differs from the recomputed TRAIN median {fill_value}"
                )
            booster = lgb.Booster(model_file=str(run_dir / model_file))

            pred_frozen = booster.predict(X_frozen)
            pred_pilot = booster.predict(X_pilot)

            # Verify the reused artifact reproduces its own recorded frozen-VALID
            # metrics before it is trusted as the paired instrument.
            recorded = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            reproduction: dict[str, Any] = {}
            if model_key == "m1_pass_classifier":
                recomputed = m1_metrics(y_arr, pred_frozen)
                train_auc = float(roc_auc_score(y_train.to_numpy(), booster.predict(X_train)))
                for name, recorded_value in (
                    ("auc", recorded["m1_pass_classifier"]["valid"]["auc"]),
                    ("brier", recorded["m1_pass_classifier"]["valid"]["brier"]),
                    ("fail_avg_precision", recorded["m1_pass_classifier"]["valid"]["fail_avg_precision"]),
                    ("fail_recall", recorded["m1_pass_classifier"]["valid"]["fail_recall"]),
                    ("fail_f1", recorded["m1_pass_classifier"]["valid"]["fail_f1"]),
                ):
                    delta = abs(recomputed[name] - float(recorded_value))
                    reproduction[name] = {"recomputed": recomputed[name], "recorded": float(recorded_value), "abs_delta": delta}
                    if delta > METRIC_ATOL:
                        stop(
                            f"{run_dir.name}: recomputed {name} {recomputed[name]!r} does not "
                            f"reproduce the recorded {recorded_value!r}"
                        )
                recorded_train_auc = float(recorded["m1_pass_classifier"]["train"]["auc"])
                if abs(train_auc - recorded_train_auc) > METRIC_ATOL:
                    stop(f"{run_dir.name}: recomputed TRAIN AUC does not reproduce the recorded value")
            else:
                recomputed = m2_metrics(y_arr, pred_frozen)
                train_auc = None
                for name in ("mae", "rmse", "r2"):
                    recorded_value = float(recorded["m2_grade_regressor"]["valid"][name])
                    delta = abs(recomputed[name] - recorded_value)
                    reproduction[name] = {"recomputed": recomputed[name], "recorded": recorded_value, "abs_delta": delta}
                    if delta > METRIC_ATOL:
                        stop(
                            f"{run_dir.name}: recomputed {name} {recomputed[name]!r} does not "
                            f"reproduce the recorded {recorded_value!r}"
                        )

            metric_fn = m1_metrics if model_key == "m1_pass_classifier" else m2_metrics
            per_segment = {}
            for name, mask in segments.items():
                per_segment[name] = {
                    "baseline": metric_fn(y_arr[mask], pred_frozen[mask]),
                    "with_prior": metric_fn(y_arr[mask], pred_pilot[mask]),
                }

            # Clause 0 — row-level prediction identity on the unexposed segments.
            difference = np.abs(pred_pilot - pred_frozen)
            clause_0 = {}
            for name in CLAUSE_0_SEGMENTS:
                mask = segments[name]
                sub = difference[mask]
                clause_0[name] = {
                    "n": int(mask.sum()),
                    "max_abs_prediction_difference": float(sub.max()) if sub.size else 0.0,
                    "rows_exceeding_tolerance": int((sub > PREDICTION_ATOL).sum()),
                    "tolerance": PREDICTION_ATOL,
                }

            seed_key = str(seed)
            results["seeds"].setdefault(seed_key, {})
            results["seeds"][seed_key][model_key] = {
                "source_run": provenance,
                "recorded_metric_reproduction": reproduction,
                "train_auc_frozen": train_auc,
                "rows_with_changed_prediction": int((difference > PREDICTION_ATOL).sum()),
                "max_abs_prediction_difference_overall": float(difference.max()),
                "segments": per_segment,
                "clause_0": clause_0,
            }
            if model_key == "m1_pass_classifier":
                results["seeds"][seed_key][model_key]["train_valid_auc_gap"] = {
                    "baseline": train_auc - per_segment["overall_valid"]["baseline"]["auc"],
                    "with_prior": train_auc - per_segment["overall_valid"]["with_prior"]["auc"],
                }

            # Non-overwriting paired run directories, one per arm.
            for arm, predictions, valid_path in (
                ("baseline", pred_frozen, FROZEN_VALID),
                ("withprior", pred_pilot, PILOT_VALID),
            ):
                arm_dir = MODEL_RUNS_DIR / f"{stamp}__predecessor_prior_pilot_seed{seed}_{arm}"
                arm_dir.mkdir(parents=True, exist_ok=True)
                np.save(arm_dir / f"{model_key}_valid_predictions.npy", predictions)
                payload = {
                    "status": "PILOT — PENDING/UNREVIEWED MAPPINGS — NOT FOR PROMOTION",
                    "arm": arm,
                    "seed": seed,
                    "model": model_key,
                    "model_artifact_reused_from": provenance["run"],
                    "model_sha256": provenance["model_sha256"],
                    "model_trained_on": str(FROZEN_TRAIN),
                    "evaluated_on": str(valid_path),
                    "evaluated_on_sha256": sha256_file(valid_path),
                    "feature_contract": contract.name,
                    "feature_count": contract.expected_feature_count,
                    "lightgbm_params": provenance["lightgbm_params"],
                    "reporting_threshold": REPORTING_THRESHOLD,
                    "test_policy": "closed_not_read",
                    "segments": {
                        name: values[
                            "baseline" if arm == "baseline" else "with_prior"
                        ]
                        for name, values in per_segment.items()
                    },
                }
                if model_key == "m1_pass_classifier":
                    payload["train_auc_frozen"] = train_auc
                    payload["train_valid_auc_gap_overall"] = (
                        train_auc - payload["segments"]["overall_valid"]["auc"]
                    )
                existing = arm_dir / "metrics.json"
                merged = json.loads(existing.read_text(encoding="utf-8")) if existing.is_file() else {}
                merged[model_key] = payload
                existing.write_text(json.dumps(merged, indent=2), encoding="utf-8")

            print(
                f"  seed {seed}: reused {provenance['run']} "
                f"({provenance['model_sha256'][:12]}); rows with changed prediction "
                f"{int((difference > PREDICTION_ATOL).sum()):,}"
            )
            del booster, pred_frozen, pred_pilot, difference
            gc.collect()

        del X_train, y_train, X_frozen, y_frozen, X_pilot, y_pilot
        gc.collect()

    (OUT_DIR / "phase3_pilot_evaluation.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {OUT_DIR / 'phase3_pilot_evaluation.json'}")
    print("TEST reads: 0. Models trained: 0 (locked artifacts reused). Promotions: 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
