"""One-off read-only diploma_gpa diagnostic for baseline_41 v2.

Reads TRAIN metadata/features, VALID, and five existing M2 model artifacts.
Never opens TEST, trains a model, or writes into the project repository.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data_prep import prepare_X_y
from src.feature_contracts import resolve_feature_contract


ROOT = Path(r"D:\AI\Real projects\Academic_Advisor")
DATASET = ROOT / "data/model_data/versions/2026-08_temporal_rebuild_v2/05_dataset"
TRAIN_PATH = DATASET / "train_dataset_candidate.parquet"
VALID_PATH = DATASET / "valid_dataset_candidate.parquet"
RUN_NAMES = [
    "2026-08-18_1638__rebuild-v2-baseline41-s42",
    "2026-08-18_1640__rebuild-v2-baseline41-s52",
    "2026-08-18_1643__rebuild-v2-baseline41-s62",
    "2026-08-18_1645__rebuild-v2-baseline41-s72",
    "2026-08-18_1647__rebuild-v2-baseline41-s82",
]
N_PERM = 30
RNG_SEED = 20260820


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def pct_interval(values: pd.Series) -> tuple[float, float]:
    return float(values.quantile(0.025)), float(values.quantile(0.975))


def main() -> None:
    print("=" * 88)
    print("DIPLOMA_GPA READ-ONLY DIAGNOSTIC — baseline_41, temporal rebuild v2")
    print("=" * 88)

    main_run = ROOT / "models/runs" / RUN_NAMES[0]
    stored_contract = json.loads((main_run / "feature_contract.json").read_text(encoding="utf-8"))
    contract = resolve_feature_contract("baseline_41")
    assert list(contract.features) == stored_contract["ordered_features"]
    expected_hashes = stored_contract["dataset_inputs"]
    got_train_hash = sha256(TRAIN_PATH)
    got_valid_hash = sha256(VALID_PATH)
    assert got_train_hash == expected_hashes["train"]["sha256"]
    assert got_valid_hash == expected_hashes["valid"]["sha256"]
    print("Dataset hashes: TRAIN OK, VALID OK (exact files recorded by seed-42 run)")

    aux = [
        "university_id", "student_id", "degree_id", "part_id",
        "is_first_row_in_timeline", "is_first_active_semester",
    ]
    valid_cols = sorted(set(contract.training_data_columns) | set(aux))
    train_cols = sorted(set(contract.training_data_columns))
    train = pd.read_parquet(TRAIN_PATH, columns=train_cols)
    valid = pd.read_parquet(VALID_PATH, columns=valid_cols)
    assert len(train) == 603_068 and len(valid) == 75_155
    expected_mark = {
        "train": (66.9820, 68.0, 16.4697),
        "valid": (73.0396, 75.0, 15.7681),
    }
    for name, frame in (("train", train), ("valid", valid)):
        got = (
            float(frame.final_mark.mean()),
            float(frame.final_mark.median()),
            float(frame.final_mark.std(ddof=1)),
        )
        assert all(round(a, 2) == round(b, 2) for a, b in zip(got, expected_mark[name]))
    fill_value = float(train["diploma_gpa"].median())
    assert round(fill_value, 2) == 85.48
    train["diploma_gpa"] = train["diploma_gpa"].fillna(fill_value)
    valid["diploma_gpa"] = valid["diploma_gpa"].fillna(fill_value)
    cat_levels = {
        key: [int(v) for v in values]
        for key, values in stored_contract["categorical_levels"].items()
    }
    X_valid, y_valid_s = prepare_X_y(valid, "grade", cat_levels, contract)
    y_valid = y_valid_s.to_numpy(dtype=float)
    cold_mask = valid["is_first_row_in_timeline"].eq(1).to_numpy()
    X_cold = X_valid.loc[cold_mask].reset_index(drop=True)
    y_cold = y_valid[cold_mask]
    meta = valid.loc[cold_mask, aux].reset_index(drop=True)
    gpa = X_cold["diploma_gpa"].to_numpy(copy=True)
    print(
        f"Rows: TRAIN={len(train):,}, VALID={len(valid):,}, "
        f"cold-start exact flag={len(X_cold):,} "
        f"({100 * len(X_cold) / len(valid):.2f}% of VALID)"
    )
    student_cols = ["university_id", "student_id", "degree_id"]
    student_index = pd.MultiIndex.from_frame(meta[student_cols])
    student_codes, student_uniques = pd.factorize(student_index, sort=False)
    student_gpa = pd.Series(gpa).groupby(student_codes).first().to_numpy()
    max_student_gpa_nunique = int(pd.Series(gpa).groupby(student_codes).nunique().max())
    assert max_student_gpa_nunique == 1
    print(
        f"Cold-start entities: {len(student_uniques):,} student-degree timelines; "
        f"rows/entity mean={len(X_cold) / len(student_uniques):.2f}; diploma_gpa constant within each"
    )

    models: dict[int, lgb.Booster] = {}
    baseline_predictions: dict[int, np.ndarray] = {}
    baseline_rows = []
    for run_name in RUN_NAMES:
        run_dir = ROOT / "models/runs" / run_name
        run_contract = json.loads((run_dir / "feature_contract.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        seed = int(run_contract["random_seed"])
        assert run_contract["ordered_features"] == stored_contract["ordered_features"]
        assert run_contract["dataset_inputs"]["valid"]["sha256"] == got_valid_hash
        model = lgb.Booster(model_file=str(run_dir / "m2_grade_model.lgbm"))
        pred = model.predict(X_valid)
        pred_cold = pred[cold_mask]
        full_mae = mae(y_valid, pred)
        cold_mae = mae(y_cold, pred_cold)
        recorded = float(metrics["m2_grade_regressor"]["valid"]["mae"])
        assert abs(full_mae - recorded) < 0.00006, (seed, full_mae, recorded)
        models[seed] = model
        baseline_predictions[seed] = pred_cold
        baseline_rows.append(
            {"seed": seed, "full_valid_mae": full_mae, "cold_mae": cold_mae,
             "iterations": model.current_iteration()}
        )
    baseline = pd.DataFrame(baseline_rows).sort_values("seed")
    print("\nBASELINE REPRODUCTION")
    print(baseline.to_string(index=False, formatters={
        "full_valid_mae": "{:.6f}".format,
        "cold_mae": "{:.6f}".format,
    }))
    ensemble_base = np.mean(np.vstack(list(baseline_predictions.values())), axis=0)
    print(f"five-seed ensemble cold MAE={mae(y_cold, ensemble_base):.6f}")

    rng = np.random.default_rng(RNG_SEED)
    perm_rows: list[dict] = []
    degree_rows: list[dict] = []
    degrees = meta["degree_id"].to_numpy()
    unique_degrees = pd.unique(degrees)
    degree_indices = {degree: np.flatnonzero(degrees == degree) for degree in unique_degrees}
    student_degree = meta.groupby(student_codes, sort=False)["degree_id"].first().to_numpy()

    for repeat in range(N_PERM):
        assignments: dict[str, np.ndarray] = {}
        assignments["row_global"] = gpa[rng.permutation(len(gpa))]
        assignments["student_global"] = student_gpa[rng.permutation(len(student_gpa))][student_codes]
        within_degree = gpa.copy()
        for idx in degree_indices.values():
            within_degree[idx] = within_degree[idx][rng.permutation(len(idx))]
        assignments["row_within_degree"] = within_degree
        student_within = student_gpa.copy()
        for degree in pd.unique(student_degree):
            idx = np.flatnonzero(student_degree == degree)
            student_within[idx] = student_within[idx][rng.permutation(len(idx))]
        assignments["student_within_degree"] = student_within[student_codes]

        for method, shuffled in assignments.items():
            X_perm = X_cold.copy()
            X_perm["diploma_gpa"] = shuffled
            for seed, model in models.items():
                perm_pred = model.predict(X_perm)
                base_pred = baseline_predictions[seed]
                base_abs = np.abs(y_cold - base_pred)
                perm_abs = np.abs(y_cold - perm_pred)
                perm_rows.append({
                    "repeat": repeat, "method": method, "seed": seed,
                    "baseline_mae": float(base_abs.mean()),
                    "permuted_mae": float(perm_abs.mean()),
                    "delta_mae": float((perm_abs - base_abs).mean()),
                })
                if method in {"student_global", "student_within_degree"}:
                    for degree, idx in degree_indices.items():
                        degree_rows.append({
                            "repeat": repeat, "seed": seed, "method": method,
                            "degree_id": str(degree),
                            "n": len(idx),
                            "delta_mae": float((perm_abs[idx] - base_abs[idx]).mean()),
                        })

    perm = pd.DataFrame(perm_rows)
    print("\nPERMUTATION IMPORTANCE — 30 repeats × 5 existing models")
    print("Positive delta means shuffling diploma_gpa made MAE worse.")
    method_summary = perm.groupby("method")["delta_mae"].agg(["mean", "std", "min", "max"])
    intervals = perm.groupby("method")["delta_mae"].apply(pct_interval)
    method_summary["p2.5"] = [intervals.loc[i][0] for i in method_summary.index]
    method_summary["p97.5"] = [intervals.loc[i][1] for i in method_summary.index]
    print(method_summary[["mean", "std", "p2.5", "p97.5", "min", "max"]].to_string(float_format=lambda x: f"{x:.6f}"))
    print("\nMean delta MAE by model seed and permutation method")
    print(perm.pivot_table(index="seed", columns="method", values="delta_mae", aggfunc="mean").to_string(float_format=lambda x: f"{x:.6f}"))

    degree_perm = pd.DataFrame(degree_rows)
    degree_summary = degree_perm.groupby(["method", "degree_id"]).agg(
        n=("n", "first"), mean_delta=("delta_mae", "mean"),
        std_delta=("delta_mae", "std"), min_delta=("delta_mae", "min"),
        max_delta=("delta_mae", "max"),
    )
    degree_summary = degree_summary[degree_summary.n >= 50].sort_values(
        ["method", "mean_delta"], ascending=[True, False]
    )
    print("\nSTUDENT-CLUSTER PERMUTATION BY DEGREE (global and within-degree; degrees with >=50 cold rows)")
    print(degree_summary.to_string(float_format=lambda x: f"{x:.6f}"))

    # Distribution and calibration on unmodified VALID cold-start rows.
    train_gpa = train["diploma_gpa"].to_numpy(dtype=float)
    percentiles = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99], dtype=float)
    grid = np.percentile(train_gpa, percentiles)
    print("\nDIPLOMA_GPA SUPPORT")
    support = pd.DataFrame({
        "percentile": percentiles,
        "train": grid,
        "valid_cold": np.percentile(gpa, percentiles),
    })
    print(support.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    calibration = pd.DataFrame({
        "gpa": gpa, "y": y_cold, "pred": ensemble_base,
        "student_code": student_codes,
    })
    calibration["bin"] = pd.qcut(calibration.gpa, q=10, duplicates="drop")
    cal = calibration.groupby("bin", observed=True).agg(
        n=("y", "size"), students=("student_code", "nunique"),
        gpa_mean=("gpa", "mean"), observed_mean=("y", "mean"),
        predicted_mean=("pred", "mean"),
    )
    cal["bias_pred_minus_obs"] = cal.predicted_mean - cal.observed_mean
    cal["mae"] = calibration.assign(abs_err=np.abs(calibration.y - calibration.pred)).groupby(
        "bin", observed=True
    ).abs_err.mean()
    print("\nOBSERVED CALIBRATION BY COLD-START DIPLOMA_GPA DECILE (five-seed ensemble)")
    print(cal.to_string(float_format=lambda x: f"{x:.4f}"))
    row_corr_obs = calibration[["gpa", "y"]].corr(method="spearman").iloc[0, 1]
    row_corr_pred = calibration[["gpa", "pred"]].corr(method="spearman").iloc[0, 1]
    student_cal = calibration.groupby("student_code").agg(
        gpa=("gpa", "first"), y=("y", "mean"), pred=("pred", "mean")
    )
    student_corr_obs = student_cal[["gpa", "y"]].corr(method="spearman").iloc[0, 1]
    student_corr_pred = student_cal[["gpa", "pred"]].corr(method="spearman").iloc[0, 1]
    print(
        "Spearman diploma_gpa correlation: "
        f"row-level observed={row_corr_obs:.4f}, predicted={row_corr_pred:.4f}; "
        f"student-mean observed={student_corr_obs:.4f}, predicted={student_corr_pred:.4f}"
    )

    # ICE/PDP: stable sample of rows; use the five-model average prediction.
    sample_n = min(2_000, len(X_cold))
    sample_idx = np.random.default_rng(RNG_SEED + 1).choice(len(X_cold), sample_n, replace=False)
    X_sample = X_cold.iloc[sample_idx].copy()
    ice_by_grid = []
    for value in grid:
        X_grid = X_sample.copy()
        X_grid["diploma_gpa"] = float(value)
        pred_grid = np.mean(
            np.vstack([model.predict(X_grid) for model in models.values()]), axis=0
        )
        ice_by_grid.append(pred_grid)
    ice = np.vstack(ice_by_grid)
    median_idx = int(np.flatnonzero(percentiles == 50)[0])
    centered = ice - ice[median_idx]
    pdp_rows = []
    for i, (pct, value) in enumerate(zip(percentiles, grid)):
        pdp_rows.append({
            "train_percentile": pct, "diploma_gpa": value,
            "mean_prediction": float(ice[i].mean()),
            "mean_change_vs_p50": float(centered[i].mean()),
            "median_change_vs_p50": float(np.median(centered[i])),
            "ice_p10_change": float(np.percentile(centered[i], 10)),
            "ice_p90_change": float(np.percentile(centered[i], 90)),
        })
    pdp = pd.DataFrame(pdp_rows)
    print("\nPDP / CENTERED ICE — 2,000 cold rows, five-model ensemble")
    print(pdp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    pdp_range = float(pdp.mean_prediction.max() - pdp.mean_prediction.min())
    low_high = float(pdp.iloc[-1].mean_prediction - pdp.iloc[0].mean_prediction)
    individual_ranges = ice.max(axis=0) - ice.min(axis=0)
    print(
        f"PDP range across p1..p99={pdp_range:.4f} marks; "
        f"p99-minus-p1={low_high:.4f}; "
        f"individual ICE range median={np.median(individual_ranges):.4f}, "
        f"p90={np.percentile(individual_ranges, 90):.4f}"
    )

    print("\nEND — TEST not opened; no model trained; no repository file modified.")


if __name__ == "__main__":
    main()
