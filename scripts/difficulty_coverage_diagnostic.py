"""Read-only TRAIN/VALID diagnostic for course-difficulty coverage decay.

This script:

* reads only the frozen TRAIN and VALID parquets named below;
* loads existing LightGBM model binaries and re-scores VALID;
* never trains or tunes a model;
* writes only the requested Markdown/JSON diagnostic pair.

It deliberately does not construct, glob for, stat, or read any TEST parquet
path. The split-extension calculation is a counterfactual in memory; it does
not write a dataset or change an existing split.
"""

from __future__ import annotations

import json
import math
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
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.course_difficulty import DifficultyConfig  # noqa: E402
from src.feature_engineering import SEMESTER_KEY  # noqa: E402
from src.model_training import prepare_X_y, resolve_feature_contract  # noqa: E402


VERSION = "2026-07-26_batched_fixes__registration_roster_concurrent"
DATA_DIR = ROOT / "data" / "model_data" / "versions" / VERSION
TRAIN_PATH = DATA_DIR / "df_train_final.parquet"
VALID_PATH = DATA_DIR / "df_valid_final.parquet"

M1_RUN = (
    ROOT
    / "models"
    / "runs"
    / "2026-07-26_1551__baseline-41-gpa-trend-control"
)
M2_RUN = (
    ROOT
    / "models"
    / "runs"
    / "2026-07-27_1327__seed42-concurrent-43-drop-dead-missing-flag"
)
M1_MODEL_PATH = M1_RUN / "m1_pass_model.lgbm"
M2_MODEL_PATH = M2_RUN / "m2_grade_model.lgbm"
M1_CONTRACT_PATH = M1_RUN / "feature_contract.json"
M2_CONTRACT_PATH = M2_RUN / "feature_contract.json"
M1_METRICS_PATH = M1_RUN / "metrics.json"
M2_METRICS_PATH = M2_RUN / "metrics.json"

OUT_JSON = ROOT / "models" / "runs" / "DIFFICULTY_COVERAGE_DIAGNOSTIC.json"
OUT_MD = ROOT / "models" / "runs" / "DIFFICULTY_COVERAGE_DIAGNOSTIC.md"

MIN_SUPPORT = DifficultyConfig().min_support
REPORTING_THRESHOLD = 0.80
INHERITED_LEVEL1_COVERAGE_PCT = {
    "train": 93.6,
    "valid": 76.2,
    "test_recorded_context_only": 44.7,
}
INHERITED_PEER_DIFFICULTY_MEAN = {"train": 0.186, "valid": 0.134}

INITIAL_GIT_STATUS_SHORT = ""
INITIAL_GIT_LOG = [
    "653e7f1 R2 five-seed confirmation: CONFIRMED for baseline_41, NOT CONFIRMED for concurrent_43",
    "235a1db Pre-register the R2 five-seed confirmation analysis rule",
    "0914e8f Regularization screening, seed 42: R2 (num_leaves 31) is the only candidate",
]
FULL_SUITE = {
    "command": ".venv\\Scripts\\python.exe -m unittest discover -s tests -t .",
    "tests_run": 117,
    "status": "OK",
    "elapsed_seconds": 36.433,
    "note": (
        "Existing unit tests exercise the training CLI on synthetic fixtures "
        "and write toy models under the OS temporary directory. They did not "
        "read the project TEST split, change a project dataset, or retrain "
        "either frozen run used by this diagnostic."
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if math.isnan(value) else value
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NA or value is pd.NaT:
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def pct(count: int, total: int) -> float:
    return float(count / total * 100.0) if total else 0.0


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def m1_metrics(y_pass: np.ndarray, pass_probability: np.ndarray) -> dict[str, Any]:
    predicted_pass = (pass_probability >= REPORTING_THRESHOLD).astype(int)
    cm = confusion_matrix(y_pass, predicted_pass, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    return {
        "n": int(len(y_pass)),
        "fail_rate": float((y_pass == 0).mean()),
        "roc_auc": float(roc_auc_score(y_pass, pass_probability)),
        "fail_average_precision": float(
            average_precision_score(1 - y_pass, 1 - pass_probability)
        ),
        "brier": float(brier_score_loss(y_pass, pass_probability)),
        "threshold": REPORTING_THRESHOLD,
        "fail_precision": float(
            precision_score(
                y_pass, predicted_pass, pos_label=0, zero_division=0
            )
        ),
        "fail_recall": float(
            recall_score(y_pass, predicted_pass, pos_label=0, zero_division=0)
        ),
        "fail_f1": float(
            f1_score(y_pass, predicted_pass, pos_label=0, zero_division=0)
        ),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def m2_metrics(mark: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    residual = mark - prediction
    ss_res = float(np.square(residual).sum())
    centered = mark - float(mark.mean())
    ss_tot = float(np.square(centered).sum())
    return {
        "n": int(len(mark)),
        "mean_final_mark": float(mark.mean()),
        "mae": float(mean_absolute_error(mark, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(mark, prediction))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def scalar_gap(
    covered: dict[str, Any],
    uncovered: dict[str, Any],
    keys: list[str],
) -> dict[str, float]:
    """Return uncovered minus covered for explicitly comparable scalars."""

    return {key: float(uncovered[key] - covered[key]) for key in keys}


def verify_preconditions() -> tuple[dict[str, Any], dict[str, Any]]:
    required = [
        TRAIN_PATH,
        VALID_PATH,
        M1_MODEL_PATH,
        M2_MODEL_PATH,
        M1_CONTRACT_PATH,
        M2_CONTRACT_PATH,
        M1_METRICS_PATH,
        M2_METRICS_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required frozen artifacts are missing: {missing}")

    m1_metrics_saved = load_json(M1_METRICS_PATH)
    m2_metrics_saved = load_json(M2_METRICS_PATH)
    m1_settings = m1_metrics_saved["run_settings"]
    m2_settings = m2_metrics_saved["run_settings"]
    assert m1_settings["random_seed"] == 42
    assert m2_settings["random_seed"] == 42
    assert m1_settings["feature_contract"] == "baseline_41"
    assert m2_settings["feature_contract"] == "concurrent_43"
    assert m1_settings["dataset_version"] == VERSION
    assert m2_settings["dataset_version"] == VERSION
    assert m1_settings["test_policy"] == "closed_not_read"
    assert m2_settings["test_policy"] == "closed_not_read"
    return m1_metrics_saved, m2_metrics_saved


def coverage_summary(df: pd.DataFrame) -> dict[str, Any]:
    level1 = df["difficulty_fallback_level"].eq(1)
    confident = df["course_difficulty_missing"].eq(0)
    return {
        "rows": int(len(df)),
        "level1": {
            "definition": "difficulty_fallback_level == 1",
            "covered_rows": int(level1.sum()),
            "coverage_pct": pct(int(level1.sum()), len(df)),
        },
        "confident_model_facing": {
            "definition": "course_difficulty_missing == 0",
            "covered_rows": int(confident.sum()),
            "uncovered_rows": int((~confident).sum()),
            "coverage_pct": pct(int(confident.sum()), len(df)),
        },
    }


def decompose_uncovered(
    train: pd.DataFrame, valid: pd.DataFrame
) -> tuple[dict[str, Any], pd.Series]:
    uncovered = valid["course_difficulty_missing"].eq(1)
    train_course_ids = set(train["course_id"].astype("string").dropna())
    valid_course = valid["course_id"].astype("string")

    never = uncovered & ~valid_course.isin(train_course_ids)
    thin = (
        uncovered
        & valid_course.isin(train_course_ids)
        & valid["course_history_count"].lt(MIN_SUPPORT)
    )
    other = uncovered & ~never & ~thin

    cause = pd.Series("covered", index=valid.index, dtype="string")
    cause.loc[never] = "never_in_train"
    cause.loc[thin] = "thin_history"
    cause.loc[other] = "other"

    n_uncovered = int(uncovered.sum())
    rows = {}
    for name, mask, description in (
        (
            "never_in_train",
            never,
            "course_id does not appear anywhere in TRAIN",
        ),
        (
            "thin_history",
            thin,
            f"course_id appears in TRAIN but course_history_count < {MIN_SUPPORT}",
        ),
        (
            "other",
            other,
            "uncovered row not explained by course absence or support below threshold",
        ),
    ):
        count = int(mask.sum())
        rows[name] = {
            "count": count,
            "pct_of_uncovered": pct(count, n_uncovered),
            "pct_of_valid": pct(count, len(valid)),
            "description": description,
        }

    assert sum(entry["count"] for entry in rows.values()) == n_uncovered
    assert rows["other"]["count"] == 0
    return {
        "uncovered_rows": n_uncovered,
        "coverage_definition": "course_difficulty_missing == 0",
        "causes": rows,
        "never_in_train_ceiling_pct_of_uncovered": rows["never_in_train"][
            "pct_of_uncovered"
        ],
    }, cause


def first_appearance_distribution(
    valid: pd.DataFrame, cause: pd.Series
) -> list[dict[str, Any]]:
    never_rows = valid.loc[cause.eq("never_in_train"), ["course_id", "part_id"]].copy()
    never_rows["course_id"] = never_rows["course_id"].astype("string")
    never_rows["part_id"] = never_rows["part_id"].astype("string")
    first_by_course = never_rows.groupby("course_id", sort=False)["part_id"].min()
    assigned_first = never_rows["course_id"].map(first_by_course)

    records = []
    for semester in sorted(first_by_course.unique(), key=int):
        course_count = int(first_by_course.eq(semester).sum())
        all_rows_for_courses = int(assigned_first.eq(semester).sum())
        debut_rows = int(
            (
                assigned_first.eq(semester)
                & never_rows["part_id"].eq(semester)
            ).sum()
        )
        records.append(
            {
                "first_semester": str(semester),
                "distinct_courses": course_count,
                "pct_of_never_in_train_courses": pct(
                    course_count, len(first_by_course)
                ),
                "all_valid_rows_for_those_courses": all_rows_for_courses,
                "rows_in_first_semester": debut_rows,
            }
        )
    return records


def coverage_over_time(train: pd.DataFrame, valid: pd.DataFrame) -> list[dict[str, Any]]:
    frames = []
    for split, frame in (("TRAIN", train), ("VALID", valid)):
        block = frame[
            [
                "part_id",
                "difficulty_fallback_level",
                "course_difficulty_missing",
            ]
        ].copy()
        block["split"] = split
        frames.append(block)
    combined = pd.concat(frames, ignore_index=True)
    combined["part_id"] = combined["part_id"].astype("string")
    combined["covered"] = combined["course_difficulty_missing"].eq(0)
    combined["level1"] = combined["difficulty_fallback_level"].eq(1)

    grouped = (
        combined.groupby(["part_id", "split"], sort=False)
        .agg(
            rows=("covered", "size"),
            covered_rows=("covered", "sum"),
            level1_rows=("level1", "sum"),
        )
        .reset_index()
    )
    grouped["sort_key"] = grouped["part_id"].astype(int)
    grouped = grouped.sort_values("sort_key").drop(columns="sort_key")
    grouped["coverage_pct"] = (
        grouped["covered_rows"] / grouped["rows"] * 100.0
    )
    grouped["level1_coverage_pct"] = (
        grouped["level1_rows"] / grouped["rows"] * 100.0
    )
    return grouped.to_dict(orient="records")


def set_level_prevalence(valid: pd.DataFrame) -> dict[str, Any]:
    work = valid[[*SEMESTER_KEY, "course_difficulty_missing"]].copy()
    work["uncovered"] = work["course_difficulty_missing"].eq(1).astype(int)
    grouped = (
        work.groupby(SEMESTER_KEY, dropna=False, sort=False)
        .agg(
            course_count=("uncovered", "size"),
            uncovered_course_count=("uncovered", "sum"),
        )
        .reset_index()
    )
    affected = grouped["uncovered_course_count"].gt(0)
    majority = (
        grouped["uncovered_course_count"] * 2 > grouped["course_count"]
    )
    distribution = []
    counts = grouped["uncovered_course_count"].value_counts().sort_index()
    for uncovered_count, group_count in counts.items():
        distribution.append(
            {
                "uncovered_course_count": int(uncovered_count),
                "student_semesters": int(group_count),
                "pct_of_all_student_semesters": pct(
                    int(group_count), len(grouped)
                ),
                "pct_of_affected_student_semesters": (
                    pct(int(group_count), int(affected.sum()))
                    if int(uncovered_count) > 0
                    else 0.0
                ),
            }
        )
    return {
        "grouping_columns": SEMESTER_KEY,
        "student_semesters": int(len(grouped)),
        "with_at_least_one_uncovered": int(affected.sum()),
        "share_with_at_least_one_uncovered_pct": pct(
            int(affected.sum()), len(grouped)
        ),
        "distribution": distribution,
        "affected_with_majority_uncovered": int((affected & majority).sum()),
        "share_of_affected_with_majority_uncovered_pct": pct(
            int((affected & majority).sum()), int(affected.sum())
        ),
    }


def extension_counterfactual(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    current_cause: pd.Series,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Estimate shifted cutoffs from the exact Level-1/Level-2 support rules."""

    valid_semesters = sorted(valid["part_id"].astype("string").unique(), key=int)
    current_uncovered = valid["course_difficulty_missing"].eq(1)
    original_uncovered_n = int(current_uncovered.sum())

    def project_model_facing_coverage(
        history: pd.DataFrame, query: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        """Return covered and selected history count using source lines 541-590.

        Coverage does not depend on Level 3-6 values. The source selects Level 1
        (degree_course_key) first, otherwise Level 2 (course_id), sets
        course_is_new only when neither exists, and flags support in (0, 20) as
        low. This direct sufficient-statistic projection avoids rebuilding
        unrelated Level-3/4 parent maps in the counterfactual.
        """

        observed = history.loc[history["final_mark"].notna()]
        l1_counts = observed.groupby(
            observed["degree_course_key"].astype("string"), sort=False
        ).size()
        l2_counts = observed.groupby(
            observed["course_id"].astype("string"), sort=False
        ).size()
        query_l1 = query["degree_course_key"].astype("string").map(l1_counts)
        query_l2 = query["course_id"].astype("string").map(l2_counts)
        selected = query_l1.where(query_l1.notna(), query_l2).fillna(0).astype(int)
        is_new = query_l1.isna() & query_l2.isna()
        low_support = selected.gt(0) & selected.lt(MIN_SUPPORT)
        covered = ~(is_new | low_support)
        return covered, selected

    # Verify that the exact Level-1/Level-2 rule on unchanged TRAIN reproduces
    # every persisted VALID model-facing flag before counterfactual use.
    rebuilt_covered, rebuilt_history = project_model_facing_coverage(train, valid)
    flag_mismatch = int(
        (
            rebuilt_covered.to_numpy()
            != valid["course_difficulty_missing"].eq(0).to_numpy()
        ).sum()
    )
    history_count_mismatch = int(
        (
            rebuilt_history.to_numpy()
            != valid["course_history_count"].to_numpy()
        ).sum()
    )
    assert flag_mismatch == 0
    assert history_count_mismatch == 0

    rows = []
    for semester_count in (1, 2, 3):
        admitted = valid_semesters[:semester_count]
        admitted_mask = valid["part_id"].astype("string").isin(admitted)
        remaining_mask = ~admitted_mask
        extended_history = pd.concat(
            [train, valid.loc[admitted_mask]], ignore_index=True
        )
        remaining = valid.loc[remaining_mask]
        projected_covered, _ = project_model_facing_coverage(
            extended_history, remaining
        )

        current_uncovered_remaining = current_uncovered.loc[remaining.index]
        newly_covered = current_uncovered_remaining & projected_covered
        still_uncovered = current_uncovered_remaining & ~projected_covered
        absorbed = int((current_uncovered & admitted_mask).sum())
        remaining_causes = current_cause.loc[remaining.index]
        newly_by_cause = {
            cause_name: int(
                (newly_covered & remaining_causes.eq(cause_name)).sum()
            )
            for cause_name in ("never_in_train", "thin_history", "other")
        }

        rows.append(
            {
                "semesters_moved": semester_count,
                "admitted_semesters": admitted,
                "current_uncovered_rows_absorbed_into_train": absorbed,
                "current_uncovered_rows_remaining_in_valid": int(
                    current_uncovered_remaining.sum()
                ),
                "newly_covered_rows_among_remaining": int(newly_covered.sum()),
                "newly_covered_by_current_cause": newly_by_cause,
                "newly_covered_pct_of_original_uncovered": pct(
                    int(newly_covered.sum()), original_uncovered_n
                ),
                "newly_covered_pct_of_remaining_uncovered": pct(
                    int(newly_covered.sum()),
                    int(current_uncovered_remaining.sum()),
                ),
                "still_uncovered_rows_among_remaining": int(
                    still_uncovered.sum()
                ),
                "remaining_valid_rows": int(remaining_mask.sum()),
            }
        )

    verification = {
        "baseline_flag_mismatches": flag_mismatch,
        "baseline_course_history_count_mismatches": history_count_mismatch,
        "method": (
            "For k=1..3, admit the first k VALID semesters to an in-memory "
            "TRAIN history, recompute the exact Level-1 degree-course and "
            "Level-2 course support counts that determine "
            "course_difficulty_missing, and apply them only to later VALID "
            "rows. Admitted rows leave VALID and are not counted as newly "
            "covered. The baseline projection exactly reproduces the persisted "
            "VALID flag and course_history_count."
        ),
        "never_in_train_rows": int(current_cause.eq("never_in_train").sum()),
        "never_in_train_pct_of_current_uncovered": pct(
            int(current_cause.eq("never_in_train").sum()),
            original_uncovered_n,
        ),
    }
    return rows, verification


def score_frozen_models(
    valid: pd.DataFrame,
    covered: np.ndarray,
    m1_saved_metrics: dict[str, Any],
    m2_saved_metrics: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    m1_contract_artifact = load_json(M1_CONTRACT_PATH)
    m2_contract_artifact = load_json(M2_CONTRACT_PATH)

    m1_contract = resolve_feature_contract("baseline_41")
    m2_contract = resolve_feature_contract("concurrent_43")
    m1_levels = m1_contract_artifact["categorical_levels"]
    m2_levels = m2_contract_artifact["categorical_levels"]

    x_m1, y_m1 = prepare_X_y(valid, "pass", m1_levels, m1_contract)
    x_m2, y_m2 = prepare_X_y(valid, "grade", m2_levels, m2_contract)
    m1_model = lgb.Booster(model_file=str(M1_MODEL_PATH))
    m2_model = lgb.Booster(model_file=str(M2_MODEL_PATH))
    m1_probability = m1_model.predict(x_m1)
    m2_prediction = m2_model.predict(x_m2)

    y_pass = y_m1.to_numpy(dtype=int)
    y_mark = y_m2.to_numpy(dtype=float)
    groups = {"covered": covered, "uncovered": ~covered}

    m1_groups = {
        name: m1_metrics(y_pass[mask], m1_probability[mask])
        for name, mask in groups.items()
    }
    m2_groups = {
        name: m2_metrics(y_mark[mask], m2_prediction[mask])
        for name, mask in groups.items()
    }
    m1_groups["gap_uncovered_minus_covered"] = scalar_gap(
        m1_groups["covered"],
        m1_groups["uncovered"],
        [
            "fail_rate",
            "roc_auc",
            "fail_average_precision",
            "brier",
            "fail_precision",
            "fail_recall",
            "fail_f1",
        ],
    )
    m1_groups["confusion_matrix_count_gap_uncovered_minus_covered"] = {
        key: int(
            m1_groups["uncovered"]["confusion_matrix"][key]
            - m1_groups["covered"]["confusion_matrix"][key]
        )
        for key in ("tn", "fp", "fn", "tp")
    }
    m2_groups["gap_uncovered_minus_covered"] = scalar_gap(
        m2_groups["covered"],
        m2_groups["uncovered"],
        ["mean_final_mark", "mae", "rmse", "r2"],
    )

    # Re-score the complete VALID split and compare to each frozen run's saved
    # summary. This proves the loaded binaries and contracts reproduce the
    # recorded runs without retraining.
    m1_overall = m1_metrics(y_pass, m1_probability)
    m2_overall = m2_metrics(y_mark, m2_prediction)
    saved_m1 = m1_saved_metrics["m1_pass_classifier"]["valid"]
    saved_m2 = m2_saved_metrics["m2_grade_regressor"]["valid"]
    reproduction = {
        "m1": {
            "auc_absolute_error": abs(m1_overall["roc_auc"] - saved_m1["auc"]),
            "fail_ap_absolute_error": abs(
                m1_overall["fail_average_precision"]
                - saved_m1["fail_avg_precision"]
            ),
            "brier_absolute_error": abs(
                m1_overall["brier"] - saved_m1["brier"]
            ),
        },
        "m2": {
            "mae_absolute_error_vs_saved_4dp": abs(
                round(m2_overall["mae"], 4) - saved_m2["mae"]
            ),
            "rmse_absolute_error_vs_saved_4dp": abs(
                round(m2_overall["rmse"], 4) - saved_m2["rmse"]
            ),
            "r2_absolute_error_vs_saved_4dp": abs(
                round(m2_overall["r2"], 4) - saved_m2["r2"]
            ),
        },
    }
    assert max(reproduction["m1"].values()) < 1e-12
    assert max(reproduction["m2"].values()) < 1e-12
    return {"groups": m1_groups, "overall": m1_overall}, {
        "groups": m2_groups,
        "overall": m2_overall,
        "reproduction": reproduction,
    }


def diagnose_identical_segments(valid: pd.DataFrame) -> dict[str, Any]:
    first_mask = valid["is_first_active_semester"].eq(1)
    cold_mask = valid["no_previous_progress"].eq(1)
    return {
        "mask_definitions": {
            "first_semester": 'df["is_first_active_semester"] == 1',
            "cold_start_gpa": 'df["no_previous_progress"] == 1',
            "source": "src/model_training.py:850 and src/model_training.py:853",
        },
        "independent_counts": {
            "first_semester": int(first_mask.sum()),
            "cold_start_gpa": int(cold_mask.sum()),
        },
        "mask_mismatch_rows": int((first_mask != cold_mask).sum()),
        "columns_mismatch_rows": int(
            (
                valid["is_first_active_semester"].to_numpy()
                != valid["no_previous_progress"].to_numpy()
            ).sum()
        ),
        "upstream_assignments": [
            'semester_df["no_previous_progress"] = no_previous_progress.astype(int)',
            'semester_df["is_first_active_semester"] = no_previous_progress.astype(int)',
        ],
        "upstream_source": "src/feature_engineering.py:402-403",
        "diagnosis": (
            "The masks name different columns, but the current preprocessing "
            "assigns both columns from the identical no_previous_progress "
            "boolean series. Their equality is structural in this prepared "
            "data, not an accidental equality of two independently computed "
            "populations."
        ),
    }


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    output = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(output)


def f6(value: float) -> str:
    return f"{value:.6f}"


def p2(value: float) -> str:
    return f"{value:.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    decomp = report["uncovered_decomposition"]
    m1 = report["frozen_valid_scoring"]["m1"]["groups"]
    m2 = report["frozen_valid_scoring"]["m2"]["groups"]
    set_stats = report["set_level_prevalence"]
    segments = report["part_b_identical_segments"]

    lines = [
        "# Difficulty coverage diagnostic",
        "",
        "**Decision status: diagnosis only. No remedy is selected or implemented.**",
        "",
        "This report uses only the frozen TRAIN and VALID parquets and two existing "
        "seed-42 LightGBM binaries. TEST remained `closed_not_read`; the recorded "
        "44.7% TEST figure is cited only as inherited context. The diagnostic "
        "trained and tuned no model; neither frozen run was retrained, promoted, "
        "or rewired, and no dataset/default/source artifact was changed.",
        "",
        "## 1. Preconditions and frozen artifacts",
        "",
        f"- Initial `git status --short`: **clean** (empty output).",
        f"- Initial `git log -3 --oneline`:",
        "",
        "```text",
        *INITIAL_GIT_LOG,
        "```",
        "",
        f"- M1 run: `{report['runs']['m1']['absolute_path']}`",
        f"- M1 binary: `{report['runs']['m1']['model_absolute_path']}`",
        f"- M2 run: `{report['runs']['m2']['absolute_path']}`",
        f"- M2 binary: `{report['runs']['m2']['model_absolute_path']}`",
        "- Both binaries existed before analysis. Their saved metadata records "
        "seed 42, the requested feature contract, this dataset version, and "
        "`test_policy = closed_not_read`.",
        f"- Full suite: `{FULL_SUITE['command']}` — "
        f"{FULL_SUITE['tests_run']} tests, {FULL_SUITE['status']} in "
        f"{FULL_SUITE['elapsed_seconds']:.3f}s.",
        "",
        "## 2. Definitions pinned before interpretation",
        "",
        "### Exact difficulty computation",
        "",
        "`src/course_difficulty.py` is the current implementation. It fits all "
        "statistics from TRAIN history only. TRAIN is processed semester by "
        "semester, so semester `t` sees strictly earlier TRAIN semesters "
        "(`build_temporal_train`, lines 604-646). VALID receives a frozen state "
        "fit on complete TRAIN (`fit_difficulty_state`, lines 410-429).",
        "",
        "The raw pass statistic is:",
        "",
        "```python",
        'work["pass_value"] = (',
        '    (frame["final_mark"] >= 50) & frame["final_mark"].notna()',
        ').astype("int64")',
        'support_count=("mark_present", "sum")',
        'sum_pass=("pass_value", "sum")',
        "```",
        "",
        "Each level is empirical-Bayes smoothed toward its parent/global value "
        "with `k = 20`:",
        "",
        "```python",
        'table[output] = (local_sum + k * parent_value) / (n + k)',
        "```",
        "",
        "The fallback hierarchy is Level 1 degree+course, Level 2 course across "
        "degrees, Level 3 degree+requirement type+rounded credits, Level 4 "
        "faculty+requirement type+rounded credits, Level 5 requirement "
        "type+rounded credits, then Level 6 global TRAIN history. The concurrent "
        "difficulty scalar is exactly `d = 1.0 - course_pass_rate_historical` "
        "(`src/concurrent_group_features.py:29,208-209`).",
        "",
        "### What coverage means",
        "",
        "Two related definitions must not be conflated:",
        "",
        "1. **Inherited Level-1 coverage** is "
        "`difficulty_fallback_level == 1`. It means the exact degree-course key "
        "was found. It has no minimum-support requirement.",
        "2. **Model-facing confident coverage**, used for all covered/uncovered "
        "splits below, is `course_difficulty_missing == 0`. Current source sets "
        "the missing flag when `course_is_new == 1` or "
        "`course_low_support == 1`; low support means "
        f"`0 < course_history_count < {MIN_SUPPORT}`. Therefore covered means a "
        "Level-1 or Level-2 known course with at least 20 historical rows. The "
        "statistics are TRAIN-only.",
        "",
        "Exact current code (`src/course_difficulty.py:578-590`):",
        "",
        "```python",
        "course_is_new = (~supports[1].notna() & ~supports[2].notna())",
        "course_low_support = (",
        "    (course_history > 0) & (course_history < state.config.min_support)",
        ")",
        'feature_values["course_difficulty_missing"] = (',
        "    (course_is_new == 1) | (course_low_support == 1)",
        ")",
        "```",
        "",
        "### Fallback/imputation and its indicator",
        "",
        "There is not one universal imputed course value. If an exact course "
        "degree pairing (Level 1) is unavailable, the code first tries the same "
        "course across degrees (Level 2). If the course has no history at either "
        "level, it uses the first available Level 3-5 group estimate, each "
        "smoothed toward its parent/global TRAIN statistic. Only when no group "
        "exists does Level 6 use the global TRAIN values. On "
        f"this TRAIN those are pass rate {f6(report['definitions']['global_train_values']['course_pass_rate_historical'])} "
        f"(difficulty {f6(report['definitions']['global_train_values']['course_difficulty'])}), "
        f"mean mark {f6(report['definitions']['global_train_values']['course_avg_mark_historical'])}, "
        f"and retake rate {f6(report['definitions']['global_train_values']['course_retake_rate_historical'])}.",
        "",
        "Both frozen feature contracts contain the model feature "
        "`course_difficulty_missing`, so the models are told that a value is "
        "weak/imputed rather than confidently observed. `course_history_count` "
        "is also a model feature. `difficulty_fallback_level`, `course_is_new`, "
        "and `course_low_support` are audit-only and do not enter either model.",
        "",
        "### What 0.186 and 0.134 are",
        "",
        "- **0.186:** recorded approximate TRAIN mean of "
        "`peer_difficulty_mean` in "
        "`scripts/build_concurrent_group_features.py:141-145`.",
        "- **0.134:** recorded approximate VALID mean of "
        "`peer_difficulty_mean` in the same constant block.",
        "",
        "They are split-level approximate means of the concurrent peer-difficulty "
        "feature, not accuracy metrics. Their difference is -0.052 (VALID minus "
        "TRAIN); the inherited note calls this a shift, but this diagnostic does "
        "not treat the note's causal wording as proof.",
        "",
        "### Recomputed coverage",
        "",
        md_table(
            [
                "Split",
                "Rows",
                "Level-1 rows",
                "Recomputed Level-1",
                "Inherited Level-1",
                "Difference (pp)",
                "Model-facing confident",
            ],
            [
                [
                    split.upper(),
                    f"{coverage[split]['rows']:,}",
                    f"{coverage[split]['level1']['covered_rows']:,}",
                    p2(coverage[split]["level1"]["coverage_pct"]),
                    p2(coverage[split]["inherited_level1_coverage_pct"]),
                    f"{coverage[split]['level1_discrepancy_pp']:+.2f}",
                    p2(
                        coverage[split]["confident_model_facing"][
                            "coverage_pct"
                        ]
                    ),
                ]
                for split in ("train", "valid")
            ],
        ),
        "",
        "The inherited Level-1 figures do **not** reproduce on the specified "
        "parquets: TRAIN is +0.77 percentage points and VALID is +1.22 points "
        "higher. The recorded 44.7% TEST value was not recomputed because TEST "
        "was never read.",
        "",
        "## 3. VALID uncovered-row decomposition",
        "",
        f"Under the model-facing definition, VALID has {decomp['uncovered_rows']:,} "
        "uncovered rows.",
        "",
        md_table(
            ["Cause", "Rows", "% uncovered", "% VALID", "Meaning"],
            [
                [
                    name,
                    f"{entry['count']:,}",
                    p2(entry["pct_of_uncovered"]),
                    p2(entry["pct_of_valid"]),
                    entry["description"],
                ]
                for name, entry in decomp["causes"].items()
            ],
        ),
        "",
        "`thin_history` is potentially recoverable from more historical "
        "observations. `never_in_train` is the current ceiling for what no mere "
        "re-cut of the existing TRAIN history can fix.",
        "",
        "### First appearance of never-in-TRAIN courses",
        "",
        md_table(
            [
                "First semester",
                "Distinct courses",
                "% never-in-TRAIN courses",
                "All VALID rows for those courses",
                "Rows in debut semester",
            ],
            [
                [
                    row["first_semester"],
                    row["distinct_courses"],
                    p2(row["pct_of_never_in_train_courses"]),
                    f"{row['all_valid_rows_for_those_courses']:,}",
                    f"{row['rows_in_first_semester']:,}",
                ]
                for row in report["never_in_train_first_appearance"]
            ],
        ),
        "",
        "## 4. Coverage over time",
        "",
        md_table(
            [
                "Semester",
                "Split",
                "Rows",
                "Confident coverage",
                "Level-1 coverage",
            ],
            [
                [
                    row["part_id"],
                    row["split"],
                    f"{row['rows']:,}",
                    p2(row["coverage_pct"]),
                    p2(row["level1_coverage_pct"]),
                ]
                for row in report["coverage_over_time"]
            ],
        ),
        "",
        f"Early TRAIN coverage ramps up from a history-free first semester; by "
        f"the last six TRAIN semesters it is consistently above 98%. The "
        f"TRAIN/VALID boundary is between "
        f"{report['time_interpretation']['last_train_semester']} and "
        f"{report['time_interpretation']['first_valid_semester']}, where "
        f"coverage changes by "
        f"{report['time_interpretation']['boundary_coverage_change_pp']:+.2f} "
        f"percentage points. "
        f"{report['time_interpretation']['assessment']}",
        "",
        "## 5. Accuracy on covered versus uncovered VALID rows",
        "",
        "These predictions were produced by loading the existing frozen binaries "
        "listed in section 1. No model was retrained, re-tuned, or threshold-tuned. "
        "The complete-VALID re-score exactly reproduces the saved run metrics "
        "(M1 unrounded; M2 at the saved four-decimal precision). Gaps below are "
        "**uncovered minus covered**.",
        "",
        "### M1 — frozen seed-42 `baseline_41` control",
        "",
        md_table(
            [
                "Group",
                "n",
                "Fail rate",
                "ROC AUC",
                "Fail AP",
                "Brier",
                "Fail P @.80",
                "Fail R @.80",
                "Fail F1 @.80",
                "CM (TN,FP,FN,TP)",
            ],
            [
                [
                    name,
                    f"{m1[name]['n']:,}",
                    p2(m1[name]["fail_rate"] * 100),
                    f6(m1[name]["roc_auc"]),
                    f6(m1[name]["fail_average_precision"]),
                    f6(m1[name]["brier"]),
                    f6(m1[name]["fail_precision"]),
                    f6(m1[name]["fail_recall"]),
                    f6(m1[name]["fail_f1"]),
                    ",".join(
                        str(m1[name]["confusion_matrix"][key])
                        for key in ("tn", "fp", "fn", "tp")
                    ),
                ]
                for name in ("covered", "uncovered")
            ],
        ),
        "",
        md_table(
            ["Gap metric", "Uncovered - covered"],
            [
                [key, f"{value:+.6f}"]
                for key, value in m1["gap_uncovered_minus_covered"].items()
            ]
            + [
                [
                    f"confusion_count_{key}",
                    f"{value:+d}",
                ]
                for key, value in m1[
                    "confusion_matrix_count_gap_uncovered_minus_covered"
                ].items()
            ],
        ),
        "",
        "### M2 — frozen seed-42 `concurrent_43` run",
        "",
        md_table(
            ["Group", "n", "Mean final mark", "MAE", "RMSE", "R2"],
            [
                [
                    name,
                    f"{m2[name]['n']:,}",
                    f6(m2[name]["mean_final_mark"]),
                    f6(m2[name]["mae"]),
                    f6(m2[name]["rmse"]),
                    f6(m2[name]["r2"]),
                ]
                for name in ("covered", "uncovered")
            ],
        ),
        "",
        md_table(
            ["Gap metric", "Uncovered - covered"],
            [
                [key, f"{value:+.6f}"]
                for key, value in m2["gap_uncovered_minus_covered"].items()
            ],
        ),
        "",
        "The base-rate columns are part of the evidence: uncovered rows are a "
        "different population, so group metric gaps must not automatically be "
        "attributed solely to imputation.",
        "",
        "## 6. Student-semester prevalence",
        "",
        f"The exact plan grain is `{', '.join(set_stats['grouping_columns'])}` "
        "(`src/feature_engineering.py:28-35`). "
        f"VALID contains {set_stats['student_semesters']:,} student-semesters; "
        f"{set_stats['with_at_least_one_uncovered']:,} "
        f"({p2(set_stats['share_with_at_least_one_uncovered_pct'])}) contain at "
        "least one uncovered course.",
        "",
        md_table(
            [
                "Uncovered courses",
                "Student-semesters",
                "% all",
                "% affected",
            ],
            [
                [
                    row["uncovered_course_count"],
                    f"{row['student_semesters']:,}",
                    p2(row["pct_of_all_student_semesters"]),
                    (
                        p2(row["pct_of_affected_student_semesters"])
                        if row["uncovered_course_count"] > 0
                        else "—"
                    ),
                ]
                for row in set_stats["distribution"]
            ],
        ),
        "",
        f"Among affected student-semesters, "
        f"{set_stats['affected_with_majority_uncovered']:,} "
        f"({p2(set_stats['share_of_affected_with_majority_uncovered_pct'])}) have "
        "a majority of courses uncovered.",
        "",
        f"**If the system recommended a plan today, "
        f"{p2(set_stats['share_with_at_least_one_uncovered_pct'])} of cases "
        "would contain at least one course carrying an imputed/weak difficulty.**",
        "",
        "## 7. Counterfactual cutoff movement",
        "",
        "This is an in-memory estimate only; no split was changed. For each "
        "cutoff, the admitted prefix leaves VALID. Only later rows that change "
        "from currently uncovered to covered are counted as “newly covered.”",
        "",
        md_table(
            [
                "Move",
                "Semesters admitted",
                "Uncovered rows absorbed",
                "Uncovered remaining",
                "Newly covered",
                "Newly covered: never / thin",
                "% original uncovered",
                "% remaining uncovered",
                "Remaining VALID rows",
            ],
            [
                [
                    f"{row['semesters_moved']} semester(s)",
                    ", ".join(row["admitted_semesters"]),
                    f"{row['current_uncovered_rows_absorbed_into_train']:,}",
                    f"{row['current_uncovered_rows_remaining_in_valid']:,}",
                    f"{row['newly_covered_rows_among_remaining']:,}",
                    (
                        f"{row['newly_covered_by_current_cause']['never_in_train']:,}"
                        " / "
                        f"{row['newly_covered_by_current_cause']['thin_history']:,}"
                    ),
                    p2(row["newly_covered_pct_of_original_uncovered"]),
                    p2(row["newly_covered_pct_of_remaining_uncovered"]),
                    f"{row['remaining_valid_rows']:,}",
                ]
                for row in report["cutoff_extension_counterfactual"]
            ],
        ),
        "",
        f"Structural ceiling requested by the task: "
        f"{decomp['causes']['never_in_train']['count']:,} current "
        f"uncovered rows ({p2(decomp['never_in_train_ceiling_pct_of_uncovered'])} "
        "of all uncovered rows) belong to courses absent from current TRAIN. "
        "No re-cut of the existing pre-VALID history can give those rows an "
        "observed course prior. A forward cutoff can consume an earlier VALID "
        "appearance into TRAIN and thereby cover later repetitions; the table "
        "counts that later-row effect but does not call the absorbed rows "
        "covered.",
        "",
        "Cost statement (not a trade-off evaluation): moving the TRAIN cutoff "
        "forward makes every existing run, dataset hash, and "
        "`models/runs/NOISE_BAND.md` non-comparable, and VALID shrinks.",
        "",
        "## 8. PART B — identical-segments defect (diagnosis only)",
        "",
        "The two masks in `src/model_training.py` are:",
        "",
        "```python",
        f'"first_semester": {segments["mask_definitions"]["first_semester"]},',
        f'"cold_start_gpa": {segments["mask_definitions"]["cold_start_gpa"]},',
        "```",
        "",
        "Computed independently on VALID:",
        "",
        md_table(
            ["Mask", "Rows"],
            [
                [
                    "first_semester",
                    f"{segments['independent_counts']['first_semester']:,}",
                ],
                [
                    "cold_start_gpa",
                    f"{segments['independent_counts']['cold_start_gpa']:,}",
                ],
            ],
        ),
        "",
        f"Mask mismatch rows: {segments['mask_mismatch_rows']:,}. "
        f"{segments['diagnosis']} The upstream assignments are quoted in "
        f"`{segments['upstream_source']}`:",
        "",
        "```python",
        *segments["upstream_assignments"],
        "```",
        "",
        "No proposal or fix is made in this section.",
        "",
        "## 9. Candidate remedies — options only",
        "",
        "- Extend historical TRAIN backward: can help thin-history courses "
        "without consuming current VALID semesters, but requires older reliable "
        "records and a new dataset/model comparability baseline.",
        "- Move the cutoff forward: the quantified effects are in section 7; "
        "VALID shrinks and all existing run/hash/noise-band comparisons break.",
        "- Keep the current fallback hierarchy and missing indicator: preserves "
        "comparability, but accepts the prevalence and group accuracy observed "
        "above.",
        "- Add non-outcome course/catalog priors for genuinely new offerings: "
        "can address `never_in_train`, but introduces new data contracts and "
        "requires separate validation.",
        "",
        "These are unranked options. This report recommends none.",
        "",
        "## 10. Scope confirmations",
        "",
        "- TEST dataset: **never read**. Policy remained `closed_not_read`.",
        "- Models: existing frozen binaries only; **no retraining or retuning**.",
        "- Test-suite nuance: existing unit tests train synthetic toy models in "
        "the OS temporary directory. They did not retrain either frozen run, "
        "read the project TEST split, or write a project model/dataset artifact.",
        "- Data: no dataset file was written or modified.",
        "- Source/defaults/wiring: no file under `src/`, no default, "
        "`CURRENT_VERSION.txt`, promotion marker, or inference/recommendation "
        "wiring was changed.",
        "- Push: not performed.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    m1_saved, m2_saved = verify_preconditions()

    # Full prepared frames are required to reproduce the frozen contracts.
    # These are the only dataset reads in this script.
    train = pd.read_parquet(TRAIN_PATH)
    valid = pd.read_parquet(VALID_PATH)

    # Pin current missingness semantics against the persisted columns.
    rebuilt_missing = (
        train["course_is_new"].eq(1) | train["course_low_support"].eq(1)
    ).astype(int)
    assert int(
        (rebuilt_missing.to_numpy() != train["course_difficulty_missing"].to_numpy()).sum()
    ) == 0
    rebuilt_missing = (
        valid["course_is_new"].eq(1) | valid["course_low_support"].eq(1)
    ).astype(int)
    assert int(
        (rebuilt_missing.to_numpy() != valid["course_difficulty_missing"].to_numpy()).sum()
    ) == 0

    coverage = {
        "train": coverage_summary(train),
        "valid": coverage_summary(valid),
    }
    for split in ("train", "valid"):
        inherited = INHERITED_LEVEL1_COVERAGE_PCT[split]
        coverage[split]["inherited_level1_coverage_pct"] = inherited
        coverage[split]["level1_discrepancy_pp"] = (
            coverage[split]["level1"]["coverage_pct"] - inherited
        )

    decomposition, causes = decompose_uncovered(train, valid)
    first_appearance = first_appearance_distribution(valid, causes)
    time_table = coverage_over_time(train, valid)
    set_stats = set_level_prevalence(valid)

    covered_mask = valid["course_difficulty_missing"].eq(0).to_numpy()
    m1_scoring, m2_scoring = score_frozen_models(
        valid, covered_mask, m1_saved, m2_saved
    )
    reproduction = m2_scoring.pop("reproduction")

    extension, extension_verification = extension_counterfactual(
        train, valid, causes
    )
    segment_diagnostic = diagnose_identical_segments(valid)

    train_pass_rate = float(train["final_mark"].ge(50).mean())
    train_avg_mark = float(train["final_mark"].mean())
    train_retake_rate = float(train["attempt_number"].gt(1).mean())

    train_semesters = sorted(train["part_id"].astype("string").unique(), key=int)
    valid_semesters = sorted(valid["part_id"].astype("string").unique(), key=int)
    valid_time = [
        row for row in time_table if row["split"] == "VALID"
    ]
    train_time = [
        row for row in time_table if row["split"] == "TRAIN"
    ]
    valid_coverage = [row["coverage_pct"] for row in valid_time]
    boundary_change = (
        valid_time[0]["coverage_pct"] - train_time[-1]["coverage_pct"]
    )
    largest_valid_drop = min(
        (
            valid_coverage[index] - valid_coverage[index - 1]
            for index in range(1, len(valid_coverage))
        ),
        default=0.0,
    )
    if largest_valid_drop <= -10.0:
        time_assessment = (
            "The decline is not gradual: after the boundary drop it contains "
            "another cliff between VALID semesters 20223 and 20231 "
            f"({largest_valid_drop:.2f} percentage points)."
        )
    else:
        time_assessment = (
            "The series is gradual rather than a single cliff "
            f"(largest adjacent VALID drop {largest_valid_drop:.2f} percentage "
            "points)."
        )

    report: dict[str, Any] = {
        "diagnostic": "difficulty_coverage_decay",
        "scope": {
            "dataset_version": VERSION,
            "dataset_reads": [rel(TRAIN_PATH), rel(VALID_PATH)],
            "test_policy": "closed_not_read",
            "test_dataset_read": False,
            "recorded_test_level1_coverage_pct_context_only": 44.7,
            "model_retrained": False,
            "model_retuned": False,
            "dataset_changed": False,
            "source_changed": False,
            "default_changed": False,
            "current_version_changed": False,
            "wiring_changed": False,
            "promotion_performed": False,
            "push_performed": False,
        },
        "git_preconditions": {
            "status_short": INITIAL_GIT_STATUS_SHORT,
            "log_3_oneline": INITIAL_GIT_LOG,
        },
        "full_test_suite": FULL_SUITE,
        "runs": {
            "m1": {
                "role": "seed-42 baseline_41 default control; M1 classifier reused",
                "path": rel(M1_RUN),
                "absolute_path": str(M1_RUN.resolve()),
                "model_path": rel(M1_MODEL_PATH),
                "model_absolute_path": str(M1_MODEL_PATH.resolve()),
                "feature_contract": "baseline_41",
                "seed": 42,
            },
            "m2": {
                "role": "seed-42 concurrent_43 run; M2 regressor reused",
                "path": rel(M2_RUN),
                "absolute_path": str(M2_RUN.resolve()),
                "model_path": rel(M2_MODEL_PATH),
                "model_absolute_path": str(M2_MODEL_PATH.resolve()),
                "feature_contract": "concurrent_43",
                "seed": 42,
            },
        },
        "definitions": {
            "difficulty_source": "src/course_difficulty.py",
            "peer_difficulty_source": "src/concurrent_group_features.py",
            "course_difficulty_scalar": "1.0 - course_pass_rate_historical",
            "min_support": MIN_SUPPORT,
            "level1_coverage": "difficulty_fallback_level == 1",
            "model_facing_confident_coverage": "course_difficulty_missing == 0",
            "model_facing_uncovered": "course_difficulty_missing == 1",
            "statistics_fit": "TRAIN history only",
            "fallback_hierarchy": {
                "1": "degree_course_key",
                "2": "course_id across degrees",
                "3": "degree_id + requirement_type_id + rounded course_credits",
                "4": "faculty_id + requirement_type_id + rounded course_credits",
                "5": "requirement_type_id + rounded course_credits",
                "6": "global prior TRAIN history",
            },
            "global_train_values": {
                "course_pass_rate_historical": train_pass_rate,
                "course_difficulty": 1.0 - train_pass_rate,
                "course_avg_mark_historical": train_avg_mark,
                "course_retake_rate_historical": train_retake_rate,
            },
            "imputation_indicator_in_both_contracts": "course_difficulty_missing",
            "additional_model_support_feature": "course_history_count",
            "audit_only_not_model_features": [
                "difficulty_fallback_level",
                "course_is_new",
                "course_low_support",
                "difficulty_group_support_count",
            ],
            "inherited_peer_difficulty_means": {
                "train_0.186": (
                    "approximate prior TRAIN mean of peer_difficulty_mean"
                ),
                "valid_0.134": (
                    "approximate prior VALID mean of peer_difficulty_mean"
                ),
                "valid_minus_train": -0.052,
                "source": "scripts/build_concurrent_group_features.py:141-145",
            },
        },
        "coverage": coverage,
        "uncovered_decomposition": decomposition,
        "never_in_train_first_appearance": first_appearance,
        "coverage_over_time": time_table,
        "time_interpretation": {
            "last_train_semester": train_semesters[-1],
            "first_valid_semester": valid_semesters[0],
            "boundary_coverage_change_pp": boundary_change,
            "largest_adjacent_valid_coverage_change_pp": largest_valid_drop,
            "assessment": time_assessment,
        },
        "frozen_valid_scoring": {
            "coverage_definition": "course_difficulty_missing == 0",
            "gap_definition": "uncovered minus covered",
            "threshold": REPORTING_THRESHOLD,
            "m1": {
                "run": rel(M1_RUN),
                "model": rel(M1_MODEL_PATH),
                **m1_scoring,
            },
            "m2": {
                "run": rel(M2_RUN),
                "model": rel(M2_MODEL_PATH),
                **m2_scoring,
            },
            "saved_metric_reproduction": reproduction,
            "frozen_models_only": True,
            "retrained": False,
            "retuned": False,
        },
        "set_level_prevalence": set_stats,
        "cutoff_extension_counterfactual": extension,
        "cutoff_extension_verification": extension_verification,
        "cutoff_cost_statement": (
            "Every existing run, dataset hash, and models/runs/NOISE_BAND.md "
            "becomes non-comparable, and VALID shrinks."
        ),
        "part_b_identical_segments": segment_diagnostic,
        "candidate_remedies": {
            "status": "unranked_options_only_no_recommendation",
            "options": [
                {
                    "option": "extend TRAIN backward",
                    "tradeoff": (
                        "may recover thin history without consuming VALID, but "
                        "requires reliable older data and resets comparability"
                    ),
                },
                {
                    "option": "move cutoff forward",
                    "tradeoff": (
                        "quantified here; shrinks VALID and resets all existing "
                        "run/hash/noise-band comparability"
                    ),
                },
                {
                    "option": "retain fallback and missing indicator",
                    "tradeoff": (
                        "preserves comparability but accepts observed prevalence "
                        "and group performance"
                    ),
                },
                {
                    "option": "add non-outcome catalog priors",
                    "tradeoff": (
                        "could address never-in-train courses but adds data "
                        "contracts and requires separate validation"
                    ),
                },
            ],
        },
    }

    OUT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=json_value) + "\n",
        encoding="utf-8",
    )
    OUT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(
        "TEST dataset reads: 0; model retraining: 0; dataset writes: 0; "
        "source/default/wiring changes: 0"
    )


if __name__ == "__main__":
    main()
