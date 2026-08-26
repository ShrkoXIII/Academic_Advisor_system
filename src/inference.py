"""Inference pipeline: build student snapshot and score candidate courses.

Given a student's historical data and a list of candidate courses for a target
semester, this module:
  1. Builds the student's snapshot (all semester-level features) from history
  2. For each candidate course, assembles the full feature row
  3. Scores each row with the trained grade and pass models

Usage
-----
from src.inference import StudentScorer
from src.paths import MODELS_DIR, ARTIFACTS_DIR

scorer = StudentScorer.load(
    grade_model_path=str(MODELS_DIR / 'grade_model.lgbm'),
    pass_model_path=str(MODELS_DIR / 'pass_model.lgbm'),
    difficulty_lookup_path=str(ARTIFACTS_DIR / 'course_difficulty_state'),
)

# df_history: all historical rows for one student (all semesters before target)
# candidate_course_ids: list of course_id strings to score for the coming semester
scored = scorer.score(
    df_history=df_history,
    candidate_course_ids=['CS101.111', 'MATH201.111', 'ENG301.111'],
    target_part_id='20261',          # e.g. 2026 term 1 (upcoming semester)
    degree_id='BSCS.111',
)
# Returns DataFrame with pred_mark and pass_prob columns for each candidate
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import lightgbm as lgb
import numpy as np
import pandas as pd

import contextlib
import io
import os

from src.feature_engineering import (
    MAX_ALLOWED_SEMESTER_CREDITS,
    MAX_ALLOWED_SEMESTER_COURSES,
    SEMESTER_KEY,
    run_feature_engineering_job,
)
from src.course_difficulty import (
    DifficultyState,
    apply_difficulty_state,
    load_difficulty_state,
)
from src.concurrent_group_features import (
    MODEL_CONCURRENT_FEATURES,
    compute_concurrent_group_features,
)
from src.model_training import (
    REQUIREMENT_BUCKET_ORD,
    SERVING_LIMITATION_NOTE,
    FeatureContract,
    resolve_feature_contract,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _part_id_to_year_sem(part_id: str) -> tuple[int, int]:
    """Parse 'YYYYS' part_id → (year, semester)."""
    # Keep term parsing local so scoring can accept compact part_id strings
    # while models receive numeric year/semester features.
    s = str(part_id).strip()
    return int(s[:4]), int(s[4:])


def _run_silent(df: pd.DataFrame) -> dict:
    """Run feature engineering job while suppressing all stdout output."""
    # The feature pipeline is diagnostic-heavy in notebooks; inference needs the
    # same features without flooding API or batch logs.
    with contextlib.redirect_stdout(io.StringIO()):
        return run_feature_engineering_job(df)


def _extract_student_snapshot(df_history: pd.DataFrame) -> pd.Series:
    """Extract the last-known semester snapshot from a student's history.

    Runs the full feature engineering job on the historical rows, then picks
    the row corresponding to the most recent semester as the snapshot.
    """
    # Reuse the training feature pipeline so inference snapshots are built with
    # the same leakage-safe history logic as model training.
    result = _run_silent(df_history)
    df_fe = result["df_model_audit"]

    if df_fe.empty:
        return pd.Series(dtype=object)

    # Sort by part_id numerically to get the most recent completed semester.
    df_fe = df_fe.copy()
    df_fe["_part_sort"] = pd.to_numeric(df_fe["part_id"], errors="coerce")
    last_part = df_fe["_part_sort"].max()
    last_rows = df_fe[df_fe["_part_sort"] == last_part]

    # Take the first row from the last semester because semester-level features
    # are expected to be identical across course rows within that semester.
    return last_rows.iloc[0]


def _attempt_number(df_history: pd.DataFrame, course_id: str) -> int:
    """Return 1 + number of prior attempts at course_id in student history."""
    # Attempt count is available at registration time and helps the model treat
    # retakes differently from first attempts.
    if "course_id" not in df_history.columns:
        return 1
    prior = (df_history["course_id"].astype(str) == str(course_id)).sum()
    return int(prior) + 1


def _estimate_semester_context(
    df_history: pd.DataFrame,
    expected_semester_credits: float | None,
    expected_semester_courses: int | None,
) -> tuple[float, int]:
    """Return the target semester workload supplied or estimated from history."""
    if expected_semester_credits is None:
        if "semester_reg_credits" in df_history.columns:
            expected_semester_credits = float(
                df_history["semester_reg_credits"].tail(3).mean()
            )
        else:
            expected_semester_credits = 15.0

    if expected_semester_courses is None:
        if "semester_reg_courses" in df_history.columns:
            expected_semester_courses = int(
                round(df_history["semester_reg_courses"].tail(3).mean())
            )
        else:
            expected_semester_courses = 5

    return expected_semester_credits, expected_semester_courses


def _index_candidate_metadata(
    candidate_metadata: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Return candidate metadata indexed by string course ID, when supplied."""
    if candidate_metadata is None:
        return None

    metadata_by_course = candidate_metadata.copy()
    if "course_id" in metadata_by_course.columns:
        metadata_by_course = metadata_by_course.set_index("course_id")
    metadata_by_course.index = metadata_by_course.index.astype(str)
    return metadata_by_course


def _candidate_metadata_row(
    metadata_by_course: pd.DataFrame | None, course_id: str
) -> pd.Series:
    """Return the first metadata row for a course, or an empty row."""
    if metadata_by_course is None or str(course_id) not in metadata_by_course.index:
        return pd.Series(dtype=object)

    candidate_info = metadata_by_course.loc[str(course_id)]
    if isinstance(candidate_info, pd.DataFrame):
        return candidate_info.iloc[0]
    return candidate_info


# ---------------------------------------------------------------------------
# Main scorer class
# ---------------------------------------------------------------------------

class StudentScorer:
    """Load models once, score candidate courses for any student snapshot."""

    # Known production limitation, surfaced on the class so callers can read it.
    SERVING_LIMITATION = SERVING_LIMITATION_NOTE

    def __init__(
        self,
        grade_model: lgb.Booster,
        pass_model: lgb.Booster,
        difficulty_lookup: pd.DataFrame | DifficultyState,
        feature_contract: "str | FeatureContract | None" = None,
    ) -> None:
        # Load expensive model and lookup artifacts once per scorer instance so
        # batch recommendation runs can score many students efficiently.
        self.grade_model = grade_model
        self.pass_model = pass_model
        # Never assume 44 features: the matrix width follows the contract the
        # model was trained under. Defaults to concurrent_44 for compatibility
        # with existing callers and bundles.
        self.feature_contract = resolve_feature_contract(feature_contract)
        # B2 persists all six lookup levels. Keep the old one-table artifact
        # readable for backward compatibility with pre-B2 model bundles.
        self._difficulty_state = (
            difficulty_lookup if isinstance(difficulty_lookup, DifficultyState) else None
        )
        self._diff = (
            difficulty_lookup.set_index("degree_course_key")
            if isinstance(difficulty_lookup, pd.DataFrame)
            else None
        )

    @classmethod
    def load(
        cls,
        grade_model_path: str,
        pass_model_path: str,
        difficulty_lookup_path: str,
    ) -> "StudentScorer":
        # Keep artifact loading behind a factory so callers do not need to know
        # LightGBM and parquet details.
        grade_model = lgb.Booster(model_file=grade_model_path)
        pass_model = lgb.Booster(model_file=pass_model_path)
        lookup_path = Path(difficulty_lookup_path)
        if lookup_path.is_dir() or lookup_path.name == "manifest.json":
            state_dir = lookup_path.parent if lookup_path.name == "manifest.json" else lookup_path
            diff = load_difficulty_state(state_dir)
        else:
            diff = pd.read_parquet(lookup_path)
        return cls(grade_model, pass_model, diff)

    def _get_difficulty_row(
        self,
        degree_id: str,
        course_id: str,
        *,
        faculty_id: str | None = None,
        requirement_type_id: int | float | None = None,
        course_credits: float | None = None,
    ) -> pd.Series:
        """Apply the same frozen six-level lookup used by the batch B2 build."""
        if self._difficulty_state is not None:
            if faculty_id is None:
                faculty_id = self._difficulty_state.degree_to_faculty.get(str(degree_id))
            candidate = pd.DataFrame(
                {
                    "degree_course_key": [f"{degree_id}__{course_id}"],
                    "degree_id": [degree_id],
                    "faculty_id": [faculty_id],
                    "requirement_type_id": [requirement_type_id],
                    "course_credits": [course_credits],
                }
            )
            return apply_difficulty_state(candidate, self._difficulty_state).iloc[0]

        if self._diff is None:
            raise RuntimeError("No course-difficulty state or legacy lookup is loaded")

        # Legacy pre-B2 lookup path.
        key = f"{degree_id}__{course_id}"
        if key in self._diff.index:
            return self._diff.loc[key]

        global_rows = self._diff[self._diff["difficulty_fallback_level"] == 6]
        if not global_rows.empty:
            row = global_rows.iloc[0].copy()
            row["course_difficulty_missing"] = 1
            row["difficulty_fallback_level"] = 6
            return row
        return pd.Series(
            {
                "course_credits": np.nan,
                "requirement_type_id": np.nan,
                "degree_requirement_credits_count": np.nan,
                "course_share_of_requirement": 0.0,
                "requirement_size_bucket": "none_or_unknown",
                "requirement_type_missing": 1,
                "degree_requirement_credits_count_missing": 1,
                "is_high_credit_course": 0,
                "course_pass_rate_historical": np.nan,
                "course_avg_mark_historical": np.nan,
                "course_retake_rate_historical": np.nan,
                "course_history_count": 0,
                "difficulty_group_support_count": 0,
                "difficulty_fallback_level": 6,
                "course_is_new": 1,
                "course_low_support": 0,
                "course_difficulty_missing": 1,
            }
        )

    def extract_snapshot(self, df_history: pd.DataFrame) -> pd.Series:
        """Extract the student snapshot from history (runs feature engineering once).

        Call this ONCE and pass the result to `score()` via the `snapshot`
        parameter to avoid re-running the pipeline for each candidate plan.
        """
        return _extract_student_snapshot(df_history)

    def score(
        self,
        df_history: pd.DataFrame,
        candidate_course_ids: List[str],
        target_part_id: str,
        degree_id: str,
        expected_semester_credits: float | None = None,
        expected_semester_courses: int | None = None,
        snapshot: pd.Series | None = None,
        candidate_metadata: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Score each candidate course for the upcoming target semester.

        Parameters
        ----------
        df_history : historical course rows for this student (all semesters
            strictly BEFORE target_part_id)
        candidate_course_ids : list of course_id strings to score
        target_part_id : the upcoming semester id, e.g. '20261'
        degree_id : student's current degree (used for difficulty lookup)
        expected_semester_credits : optional override for semester credit load
            estimate (used as context feature). Defaults to mean of prior semesters.
        snapshot : pre-computed student snapshot from extract_snapshot(). If
            provided, skips re-running the feature engineering pipeline.
        expected_semester_courses : optional override for course count estimate.
        candidate_metadata : optional course metadata indexed by ``course_id``
            (or containing a ``course_id`` column). It supplies faculty,
            requirement type, and credits needed by B2 Levels 3-5.

        Returns
        -------
        DataFrame indexed by course_id with columns:
            pred_mark, pass_prob, attempt_number,
            course_credits, course_pass_rate_historical,
            course_avg_mark_historical, course_difficulty_missing
        """
        if df_history.empty or not candidate_course_ids:
            return pd.DataFrame(columns=["pred_mark", "pass_prob"])

        # Build or reuse the student snapshot so every candidate uses the same
        # registration-time history state.
        if snapshot is None:
            snapshot = _extract_student_snapshot(df_history)

        target_year, target_sem = _part_id_to_year_sem(target_part_id)
        expected_semester_credits, expected_semester_courses = _estimate_semester_context(
            df_history,
            expected_semester_credits,
            expected_semester_courses,
        )

        # Single-course scoring has no plan context, so the concurrent peer
        # features stay on their singleton placeholder inside the frame builder.
        df_cand = self._build_candidate_frame(
            snapshot=snapshot,
            candidate_course_ids=candidate_course_ids,
            degree_id=degree_id,
            df_history=df_history,
            target_year=target_year,
            target_sem=target_sem,
            expected_semester_credits=expected_semester_credits,
            expected_semester_courses=expected_semester_courses,
            candidate_metadata=candidate_metadata,
        )
        return self._predict_from_frame(df_cand)

    def _build_candidate_frame(
        self,
        *,
        snapshot: pd.Series,
        candidate_course_ids: List[str],
        degree_id: str,
        df_history: pd.DataFrame,
        target_year: int,
        target_sem: int,
        expected_semester_credits: float,
        expected_semester_courses: int,
        candidate_metadata: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Assemble one model-ready feature row per candidate course.

        The concurrent peer features are seeded with the singleton placeholder
        (mean/max NaN, missing=1); ``score_plan`` overwrites them once the full
        plan context is known so train and serve compute them identically.
        """
        rows = []
        metadata_by_course = _index_candidate_metadata(candidate_metadata)
        for course_id in candidate_course_ids:
            candidate_info = _candidate_metadata_row(metadata_by_course, course_id)
            diff_row = self._get_difficulty_row(
                degree_id,
                course_id,
                faculty_id=candidate_info.get("faculty_id"),
                requirement_type_id=candidate_info.get("requirement_type_id"),
                course_credits=candidate_info.get("course_credits"),
            )
            attempt = _attempt_number(df_history, course_id)

            row = {
                # Student history features (from snapshot)
                "prev_gpa_points_clean": snapshot.get("prev_gpa_points_clean", np.nan),
                "start_agpa_points": snapshot.get("start_agpa_points", np.nan),
                "last_valid_gpa_before_current_semester": snapshot.get(
                    "last_valid_gpa_before_current_semester", np.nan
                ),
                "start_total_in_credits": snapshot.get("start_total_in_credits", np.nan),
                "start_total_in_courses": snapshot.get("start_total_in_courses", np.nan),
                "total_fail_credits_capped": snapshot.get("total_fail_credits_capped", np.nan),
                "fail_credit_ratio_capped": snapshot.get("fail_credit_ratio_capped", np.nan),
                "is_extreme_fail_history": snapshot.get("is_extreme_fail_history", 0),
                "reg_total_semesters": snapshot.get("reg_total_semesters", np.nan),
                "start_level_ord": snapshot.get("start_level_ord", 0),
                "start_level_missing": snapshot.get("start_level_missing", 1),
                "start_year": snapshot.get("start_year", np.nan),
                "start_semester": snapshot.get("start_semester", np.nan),
                "prior_interruption_count": snapshot.get("prior_interruption_count", 0),
                "consecutive_interruption_count": snapshot.get("consecutive_interruption_count", 0),
                "prev_semester_was_interruption": snapshot.get("prev_semester_was_interruption", 0),
                "is_interruption_semester": 0,  # Not an interruption (student is registering)
                "no_previous_progress": snapshot.get("no_previous_progress", 1),
                "is_first_active_semester": snapshot.get("is_first_active_semester", 0),
                "is_first_row_in_timeline": 0,  # Not the first row anymore
                "prev_gpa_points_missing": snapshot.get("prev_gpa_points_missing", 0),
                "prev_gpa_points_zero": snapshot.get("prev_gpa_points_zero", 0),
                "prev_gpa_invalid_zero_case": snapshot.get("prev_gpa_invalid_zero_case", 0),
                "prev_gpa_actual_zero_performance": snapshot.get("prev_gpa_actual_zero_performance", 0),
                # Current semester context
                "semester_reg_credits": expected_semester_credits,
                "semester_reg_courses": float(expected_semester_courses),
                "part_year": target_year,
                "part_semester": target_sem,
                # Course-specific features
                "course_credits": diff_row.get("course_credits", np.nan),
                "attempt_number": attempt,
                "is_high_credit_course": diff_row.get("is_high_credit_course", 0),
                # Course difficulty
                "course_pass_rate_historical": diff_row.get("course_pass_rate_historical", np.nan),
                "course_avg_mark_historical": diff_row.get("course_avg_mark_historical", np.nan),
                "course_retake_rate_historical": diff_row.get("course_retake_rate_historical", np.nan),
                "course_history_count": diff_row.get("course_history_count", 0),
                "difficulty_fallback_level": diff_row.get("difficulty_fallback_level", 6),
                "course_difficulty_missing": diff_row.get("course_difficulty_missing", 1),
                # Requirement features
                "requirement_type_id": diff_row.get("requirement_type_id", np.nan),
                "requirement_type_missing": diff_row.get("requirement_type_missing", 1),
                "degree_requirement_credits_count": diff_row.get("degree_requirement_credits_count", np.nan),
                "degree_requirement_credits_count_missing": diff_row.get(
                    "degree_requirement_credits_count_missing", 1
                ),
                "course_share_of_requirement": diff_row.get("course_share_of_requirement", 0.0),
                "requirement_size_bucket": diff_row.get("requirement_size_bucket", "none_or_unknown"),
                # Pre-admission diploma features are student-level snapshot values.
                "diploma_gpa": snapshot.get("diploma_gpa", np.nan),
                "diploma_type_bucket": snapshot.get("diploma_type_bucket", -1),
                # Metadata (not in MODEL_FEATURES, kept for output)
                "_course_id": course_id,
                "gpa_trend_delta": snapshot.get("gpa_trend_delta", np.nan),
                "gpa_trend_missing": snapshot.get("gpa_trend_missing", 1),
                # Concurrent course group — singleton placeholder until a plan
                # context is known (score_plan recomputes these across the plan).
                "concurrent_peer_difficulty_mean": np.nan,
                "concurrent_peer_difficulty_max": np.nan,
                "concurrent_peer_difficulty_missing": 1,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def _predict_from_frame(self, df_cand: pd.DataFrame) -> pd.DataFrame:
        """Encode categoricals, select MODEL_FEATURES, and score both models."""
        # Match the ordinal encoding used by training before selecting features.
        df_cand["requirement_size_bucket_ord"] = (
            df_cand["requirement_size_bucket"]
            .map(REQUIREMENT_BUCKET_ORD)
            .fillna(0)
            .astype(int)
        )

        # Build X using only the contract's ordered features so metadata cannot
        # leak into models and the width always matches the trained model.
        X = df_cand[list(self.feature_contract.features)].copy()

        # Score both outcomes; recommendation ranking consumes marks and risk.
        pred_mark = self.grade_model.predict(X)
        pass_prob = self.pass_model.predict(X)

        # Return only user-facing candidate metrics plus diagnostics useful for
        # recommendation explanations.
        result = pd.DataFrame(
            {
                "course_id": df_cand["_course_id"].values,
                "pred_mark": np.round(pred_mark, 2),
                "pass_prob": np.round(pass_prob, 4),
                "attempt_number": df_cand["attempt_number"].values,
                "course_credits": df_cand["course_credits"].values,
                "course_pass_rate_historical": df_cand["course_pass_rate_historical"].values,
                "course_avg_mark_historical": df_cand["course_avg_mark_historical"].values,
                "course_difficulty_missing": df_cand["course_difficulty_missing"].values,
            }
        ).set_index("course_id")

        return result.sort_values("pass_prob", ascending=False)

    def _plan_total_credits(self, degree_id: str, plan_course_ids: List[str]) -> float:
        """Return a plan's credits from the same lookup used for scoring."""
        fallback_course_credits = 3.0
        total_credits = 0.0
        for course_id in plan_course_ids:
            course_credits = self._get_difficulty_row(degree_id, course_id).get(
                "course_credits"
            )
            if pd.notna(course_credits):
                total_credits += float(course_credits)
            else:
                total_credits += fallback_course_credits
        return total_credits

    def score_plan(
        self,
        df_history: pd.DataFrame,
        plan_course_ids: List[str],
        target_part_id: str,
        degree_id: str,
        snapshot: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Score a specific plan where the semester context is known exactly.

        Unlike `score()`, this sets semester_reg_credits / semester_reg_courses
        to the actual totals for the given plan, and recomputes the concurrent
        peer features across the whole plan (train/serve parity). Pass `snapshot`
        to avoid re-running the feature engineering pipeline.
        """
        if df_history.empty or not plan_course_ids:
            return pd.DataFrame(columns=["pred_mark", "pass_prob"])

        if snapshot is None:
            snapshot = _extract_student_snapshot(df_history)

        total_credits = self._plan_total_credits(degree_id, plan_course_ids)

        target_year, target_sem = _part_id_to_year_sem(target_part_id)
        df_cand = self._build_candidate_frame(
            snapshot=snapshot,
            candidate_course_ids=plan_course_ids,
            degree_id=degree_id,
            df_history=df_history,
            target_year=target_year,
            target_sem=target_sem,
            expected_semester_credits=total_credits,
            expected_semester_courses=len(plan_course_ids),
            candidate_metadata=None,
        )

        # Every plan course shares one semester group, so recompute the
        # leave-one-out concurrent features across the plan with the SAME
        # function training uses, replacing the singleton placeholder. Constant
        # grouping keys collapse the plan into exactly one group.
        #
        # course_id is required because peer membership is course-grained on
        # both sides: a course listed twice in one plan contributes one peer,
        # exactly as a course registered twice in one semester does in training.
        df_group = df_cand.copy()
        df_group["course_id"] = df_cand["_course_id"].to_numpy()
        for key in SEMESTER_KEY:
            df_group[key] = "__plan__"
        computed = compute_concurrent_group_features(df_group)
        for col in MODEL_CONCURRENT_FEATURES:
            df_cand[col] = computed[col].to_numpy()

        return self._predict_from_frame(df_cand)
