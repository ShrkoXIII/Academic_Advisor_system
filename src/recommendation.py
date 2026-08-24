"""Plan generation, scoring, and ranking — the full recommendation engine.

This is the top-level module that ties together inference (grade/pass prediction)
and KNN evidence to produce ranked course-plan recommendations.

9-step pipeline (per readmi.md):
  Steps 1-3: provided externally (student snapshot, rules engine, candidate courses)
  Step 4-5: ML scoring via src.inference
  Step 6:   Plan candidate generation (this module)
  Step 7:   KNN soft evidence (this module, via src.knn_advisor_v2)
  Step 8:   Plan scoring on 4 axes (this module)
  Step 9:   Final ranked recommendations (this module)

Usage
-----
from src.recommendation import Recommender
from src.paths import MODELS_DIR, ARTIFACTS_DIR

rec = Recommender.load(
    grade_model_path=str(MODELS_DIR / 'grade_model.lgbm'),
    pass_model_path=str(MODELS_DIR / 'pass_model.lgbm'),
    difficulty_lookup_path=str(ARTIFACTS_DIR / 'course_difficulty_lookup.parquet'),
    grade_scale_path=str(
        ARTIFACTS_DIR / 'grade_scales/acs_grade_3.111.json'
    ),
    knn_index_path=str(
        ARTIFACTS_DIR / 'knn/2026-08-23_history_v2_level/'
        'knn_v2_gpa_level_nearest.pkl'
    ),
)

plans = rec.recommend(
    df_history=df_student_history,
    candidate_course_ids=['CS101.111', 'MATH201.111', ...],
    target_part_id='20261',
    degree_id='BSCS.111',
    remaining_degree_credits=60.0,   # optional — for graduation-progress axis
    n_plans=500,       # candidate plans to evaluate
    top_k=5,           # plans to return
)
"""

from __future__ import annotations

import random
from itertools import combinations
from typing import List, Optional

import numpy as np
import pandas as pd

from src.feature_engineering import (
    MAX_ALLOWED_SEMESTER_CREDITS,
    MAX_ALLOWED_SEMESTER_COURSES,
)
from src.grade_scale import (
    DEFAULT_GRADE_SCALE_PATH,
    OFFICIAL_GRADE_VERSION_ID,
    GradeScale,
)
from src.inference import StudentScorer
from src.knn_advisor_v2 import KNNAdvisorV2Level
from src.knn_history_helpers import SEMESTER_KEY as KNN_SEMESTER_KEY

# Minimum pass probability — courses below this are excluded from candidate plans
MIN_PASS_PROB_FOR_PLAN = 0.30


def _generate_candidate_plans(
    scored_candidates: pd.DataFrame,
    n_plans: int = 500,
    max_credits: float = MAX_ALLOWED_SEMESTER_CREDITS,
    max_courses: int = MAX_ALLOWED_SEMESTER_COURSES,
    seed: int = 42,
) -> List[List[str]]:
    """Generate up to n_plans valid course combinations from scored candidates.

    Pruning rules (applied before enumeration):
      1. Exclude courses where pass_prob < MIN_PASS_PROB_FOR_PLAN
      2. Plans must not exceed max_credits total
      3. Plans must not exceed max_courses courses

    Uses exhaustive enumeration for small candidate sets (< 20 courses) and
    random sampling for larger sets to keep runtime bounded.
    """
    rng = random.Random(seed)

    # Filter out very risky courses before combination generation to keep
    # obviously poor plans from dominating the search space.
    viable = scored_candidates[
        scored_candidates["pass_prob"] >= MIN_PASS_PROB_FOR_PLAN
    ].copy()

    course_ids = list(viable.index)
    credits = viable["course_credits"].fillna(3.0).to_dict()

    plans: List[List[str]] = []
    n_viable = len(course_ids)

    if n_viable == 0:
        return []

    def _is_valid_plan(courses: List[str]) -> bool:
        # Centralize policy constraints so exhaustive and sampled plans apply
        # the same credit and course-count limits.
        total_cr = sum(credits.get(c, 3.0) for c in courses)
        return total_cr <= max_credits and len(courses) <= max_courses

    if n_viable <= 18:
        # Exhaustive enumeration is acceptable for small candidate pools and
        # avoids missing high-quality combinations.
        for size in range(1, min(max_courses, n_viable) + 1):
            for combo in combinations(course_ids, size):
                if _is_valid_plan(list(combo)):
                    plans.append(list(combo))
                    if len(plans) >= n_plans * 3:
                        break
            if len(plans) >= n_plans * 3:
                break
        rng.shuffle(plans)
        plans = plans[:n_plans]
    else:
        # Random sampling bounds runtime when candidate pools make exhaustive
        # subset enumeration impractical.
        attempts = 0
        seen = set()
        while len(plans) < n_plans and attempts < n_plans * 20:
            attempts += 1
            k = rng.randint(1, max_courses)
            sample = rng.sample(course_ids, min(k, n_viable))
            key = tuple(sorted(sample))
            if key in seen:
                continue
            seen.add(key)
            if _is_valid_plan(sample):
                plans.append(sample)

    return plans


def _score_plan(
    plan_courses: List[str],
    scored_courses: pd.DataFrame,
    knn_evidence: dict,
    remaining_degree_credits: Optional[float],
    grade_scale: GradeScale,
) -> dict:
    """Compute the 4-axis plan score plus a composite blend.

    Axes
    ----
    expected_agpa       : credit-weighted mean of predicted GPA points
    risk                : 1 - weighted_avg_pass_prob (higher = more risky)
    workload_ratio      : plan credits / MAX_ALLOWED_SEMESTER_CREDITS
    graduation_progress : plan credits / remaining_degree_credits (0 if unknown)

    Composite
    ---------
    Higher is better. Weights are: AGPA=0.40, risk=0.30, knn=0.20, grad=0.10.
    """
    # Score only courses that were actually returned by the scorer; an empty
    # intersection means this plan cannot be evaluated.
    rows = scored_courses.loc[
        [c for c in plan_courses if c in scored_courses.index]
    ]
    if rows.empty:
        return {}

    credits = rows["course_credits"].fillna(3.0)
    total_credits = float(credits.sum())

    # Convert predicted marks through the persisted ACS_GRADE 3.111 authority.
    pred_gpa = grade_scale.points_for_marks(rows["pred_mark"])
    expected_agpa = float(
        (pred_gpa * credits).sum() / total_credits if total_credits > 0 else 0.0
    )

    # Risk blends average pass probability with the weakest course because a
    # single bad fit can make an otherwise strong plan unrealistic.
    pass_probs = rows["pass_prob"].values
    avg_pass_prob = float((pass_probs * credits.values).sum() / total_credits)
    min_pass_prob = float(pass_probs.min())
    risk = 1.0 - 0.6 * avg_pass_prob - 0.4 * min_pass_prob

    workload_ratio = min(total_credits / MAX_ALLOWED_SEMESTER_CREDITS, 1.0)

    if remaining_degree_credits and remaining_degree_credits > 0:
        grad_progress = min(total_credits / remaining_degree_credits, 1.0)
    else:
        grad_progress = 0.0

    # KNN evidence nudges the ML score using outcomes from similar historical
    # students without overpowering the main model predictions.
    knn_bonus = 0.0
    if "knn_avg_pass_rate" in knn_evidence and not np.isnan(knn_evidence["knn_avg_pass_rate"]):
        # Bonus if similar students tended to pass; penalty if they tended to fail
        knn_bonus = (knn_evidence["knn_avg_pass_rate"] - 0.8) * 0.5

    # Composite score turns the separate planning axes into one sortable value.
    composite = (
        0.40 * (expected_agpa / 4.0)          # normalised to [0,1]
        - 0.30 * risk                          # risk penalizes
        + 0.20 * (1.0 - workload_ratio * 0.5) # mild workload penalty
        + 0.10 * grad_progress
        + knn_bonus
    )

    return {
        "courses": plan_courses,
        "n_courses": len(plan_courses),
        "total_credits": round(total_credits, 1),
        "expected_agpa": round(expected_agpa, 3),
        "risk": round(risk, 3),
        "workload_ratio": round(workload_ratio, 3),
        "graduation_progress": round(grad_progress, 3),
        "avg_pass_prob": round(avg_pass_prob, 3),
        "min_pass_prob": round(min_pass_prob, 3),
        "knn_avg_pass_rate": round(knn_evidence.get("knn_avg_pass_rate", np.nan), 3),
        "knn_pct_any_failed": round(knn_evidence.get("knn_pct_any_failed", np.nan), 3),
        "knn_similar_plan_avg_mark": round(
            knn_evidence.get("knn_similar_plan_avg_mark", np.nan), 3
        ),
        "knn_support": int(knn_evidence.get("knn_support", 0)),
        "knn_route": knn_evidence.get("knn_route"),
        "knn_median_gpa_distance": round(
            knn_evidence.get("knn_median_gpa_distance", np.nan), 4
        ),
        "composite_score": round(composite, 4),
    }


def _latest_history_row(df_history: pd.DataFrame) -> pd.Series:
    """Return one row from the most recent completed semester."""
    if df_history.empty:
        return pd.Series(dtype=object)
    if "part_id" not in df_history.columns:
        return df_history.iloc[-1]

    part_order = pd.to_numeric(df_history["part_id"], errors="coerce")
    if part_order.notna().any():
        latest_position = int(part_order.fillna(-np.inf).to_numpy().argmax())
        return df_history.iloc[latest_position]
    return df_history.iloc[-1]


def _first_finite_positive(*values: object) -> float | None:
    """Return the first finite positive numeric value in priority order."""
    for value in values:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric) and np.isfinite(float(numeric)) and float(numeric) > 0:
            return float(numeric)
    return None


def _resolve_knn_query(
    *,
    df_history: pd.DataFrame,
    snapshot: pd.Series,
    cold_start: bool,
    knn_gpa: float | None,
    academic_level: int | None,
) -> tuple[float, int, str | None]:
    """Resolve the GPA, exact level, and self-exclusion id for KNN v2."""
    latest = _latest_history_row(df_history)

    if knn_gpa is None:
        if cold_start:
            knn_gpa = _first_finite_positive(
                snapshot.get("diploma_gpa"),
                latest.get("diploma_gpa"),
            )
        else:
            # Prefer the official state after the latest completed semester.
            # Snapshot values describe the start of that semester and are only
            # fallbacks when the serving history lacks official end-state fields.
            knn_gpa = _first_finite_positive(
                latest.get("end_agpa_points"),
                latest.get("cumulative_gpa_after"),
                snapshot.get("start_agpa_points"),
                snapshot.get("last_valid_gpa_before_current_semester"),
                snapshot.get("model_prev_gpa"),
            )
    else:
        knn_gpa = _first_finite_positive(knn_gpa)

    if knn_gpa is None:
        source = "diploma_gpa" if cold_start else "current cumulative GPA"
        raise ValueError(f"Cannot query KNN v2: no valid {source} is available")

    if academic_level is None:
        academic_level_value = _first_finite_positive(
            latest.get("end_level_name_short"),
            latest.get("academic_level_after"),
            snapshot.get("start_level_ord"),
        )
    else:
        academic_level_value = _first_finite_positive(academic_level)

    if academic_level_value is None or not float(academic_level_value).is_integer():
        raise ValueError(
            "Cannot query KNN v2: academic_level must be a positive integer"
        )
    resolved_level = int(academic_level_value)

    student_id = snapshot.get("student_id")
    if pd.isna(student_id) and "student_id" in df_history.columns:
        non_null_ids = df_history["student_id"].dropna()
        student_id = non_null_ids.iloc[-1] if len(non_null_ids) else None
    exclude_student_id = (
        None if student_id is None or pd.isna(student_id) else str(student_id)
    )
    return float(knn_gpa), resolved_level, exclude_student_id


def _base_knn_evidence(
    advisor: KNNAdvisorV2Level,
    neighbours: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Translate KNN v2 outcomes into recommendation-compatible evidence."""
    summary = advisor.summarize(neighbours)
    neighbour_courses = advisor.courses_for_neighbours(neighbours)

    pass_rate = np.nan
    if not neighbour_courses.empty and "is_passed" in neighbour_courses.columns:
        passed = pd.to_numeric(neighbour_courses["is_passed"], errors="coerce")
        if passed.notna().any():
            pass_rate = float(passed.mean())

    pct_all_passed = np.nan
    if not neighbours.empty and "all_courses_passed" in neighbours.columns:
        values = pd.to_numeric(neighbours["all_courses_passed"], errors="coerce")
        if values.notna().any():
            pct_all_passed = float(values.mean())

    def number_or_nan(value: object) -> float:
        return np.nan if value is None else float(value)

    evidence = {
        "knn_support": int(summary["support"]),
        "knn_route": summary["knn_route"],
        "knn_avg_pass_rate": pass_rate,
        "knn_avg_gpa": number_or_nan(summary["mean_term_gpa"]),
        "knn_pct_all_passed": pct_all_passed,
        "knn_pct_any_failed": number_or_nan(summary["pct_any_course_failed"]),
        "knn_mean_matched_gpa": number_or_nan(summary["mean_matched_gpa"]),
        "knn_median_gpa_distance": number_or_nan(summary["median_gpa_distance"]),
        "knn_mean_term_gpa_delta": number_or_nan(summary["mean_term_gpa_delta"]),
        "knn_mean_cumulative_gpa_delta": number_or_nan(
            summary["mean_cumulative_gpa_delta"]
        ),
    }
    return evidence, neighbour_courses


def _knn_evidence_for_plan(
    base_evidence: dict,
    neighbour_courses: pd.DataFrame,
    plan_course_ids: List[str],
) -> dict:
    """Add course-overlap evidence for one candidate plan."""
    evidence = dict(base_evidence)
    evidence["knn_similar_plan_avg_mark"] = np.nan
    if neighbour_courses.empty or not plan_course_ids:
        return evidence

    plan_set = {str(course_id) for course_id in plan_course_ids}
    similar_plan_marks: list[float] = []
    for _, semester_courses in neighbour_courses.groupby(
        KNN_SEMESTER_KEY, dropna=False, sort=False
    ):
        historical_set = set(semester_courses["course_id"].astype(str))
        overlap_ratio = len(plan_set & historical_set) / len(plan_set)
        if overlap_ratio < 0.5:
            continue
        marks = pd.to_numeric(semester_courses["final_mark"], errors="coerce").dropna()
        if len(marks):
            similar_plan_marks.append(float(marks.mean()))

    if similar_plan_marks:
        evidence["knn_similar_plan_avg_mark"] = float(np.mean(similar_plan_marks))
    return evidence


class Recommender:
    """Full recommendation engine: inference + plan generation + KNN + ranking."""

    def __init__(
        self,
        scorer: StudentScorer,
        knn_advisor: KNNAdvisorV2Level,
        grade_scale: GradeScale,
    ) -> None:
        # Compose model, KNN, and official grade-scale providers so plan ranking
        # does not depend on artifact-loading details or hand-written cutoffs.
        if grade_scale.grade_version_id != OFFICIAL_GRADE_VERSION_ID:
            raise ValueError(
                "Recommender requires ACS_GRADE grade_version_id "
                f"{OFFICIAL_GRADE_VERSION_ID}, got {grade_scale.grade_version_id!r}"
            )
        self.scorer = scorer
        self.knn = knn_advisor
        self.grade_scale = grade_scale

    @classmethod
    def load(
        cls,
        grade_model_path: str,
        pass_model_path: str,
        difficulty_lookup_path: str,
        knn_index_path: str,
        grade_scale_path: str | None = None,
    ) -> "Recommender":
        # Keep the public loader as the single entry point for trained model,
        # difficulty, and KNN artifacts.
        scorer = StudentScorer.load(grade_model_path, pass_model_path, difficulty_lookup_path)
        knn = KNNAdvisorV2Level.load(knn_index_path)
        scale_path = (
            DEFAULT_GRADE_SCALE_PATH if grade_scale_path is None else grade_scale_path
        )
        grade_scale = GradeScale.load(scale_path)
        return cls(scorer, knn, grade_scale)

    def recommend(
        self,
        df_history: pd.DataFrame,
        candidate_course_ids: List[str],
        target_part_id: str,
        degree_id: str,
        remaining_degree_credits: Optional[float] = None,
        n_plans: int = 500,
        top_k: int = 5,
        knn_gpa: Optional[float] = None,
        academic_level: Optional[int] = None,
        cold_start: bool = False,
        knn_k: int = 20,
    ) -> List[dict]:
        """Generate and rank course-plan recommendations.

        Returns
        -------
        List of up to `top_k` plan dicts, sorted by composite_score descending.
        Each dict contains:
            courses, n_courses, total_credits,
            expected_agpa, risk, workload_ratio, graduation_progress,
            avg_pass_prob, min_pass_prob,
            knn_avg_pass_rate, knn_pct_any_failed, knn_similar_plan_avg_mark,
            knn_support, knn_route, knn_median_gpa_distance,
            composite_score

        KNN query values default to the official state after the latest history
        semester. Callers may provide ``knn_gpa`` and ``academic_level``
        explicitly. For a new student set ``cold_start=True``; ``knn_gpa`` then
        means diploma GPA instead of cumulative university GPA.
        """
        # Extract the student snapshot once because every later stage needs the
        # same current academic state.
        snapshot = self.scorer.extract_snapshot(df_history)

        # Score all candidate courses before plan generation so poor individual
        # courses can be filtered out early.
        scored = self.scorer.score(
            df_history=df_history,
            candidate_course_ids=candidate_course_ids,
            target_part_id=target_part_id,
            degree_id=degree_id,
            snapshot=snapshot,
        )

        if scored.empty:
            return []

        # Generate policy-valid candidate plans from the scored course pool.
        plans = _generate_candidate_plans(scored, n_plans=n_plans)
        if not plans:
            return []

        # KNN v2 matches only inside the student's exact degree and academic
        # level, using cumulative GPA for returning students or diploma GPA for
        # cold-start students.
        resolved_gpa, resolved_level, exclude_student_id = _resolve_knn_query(
            df_history=df_history,
            snapshot=snapshot,
            cold_start=cold_start,
            knn_gpa=knn_gpa,
            academic_level=academic_level,
        )
        neighbours = self.knn.find_nearest_gpa(
            degree_id=degree_id,
            academic_level=resolved_level,
            gpa=resolved_gpa,
            cold_start=cold_start,
            k=knn_k,
            exclude_student_id=exclude_student_id,
        )
        base_knn_evidence, neighbour_courses = _base_knn_evidence(
            self.knn, neighbours
        )

        # Re-score each plan with its exact workload context before combining
        # model predictions, KNN evidence, risk, and graduation progress.
        scored_plans = []
        for plan_courses in plans:
            plan_scored = self.scorer.score_plan(
                df_history=df_history,
                plan_course_ids=plan_courses,
                target_part_id=target_part_id,
                degree_id=degree_id,
                snapshot=snapshot,
            )
            knn_evidence = _knn_evidence_for_plan(
                base_knn_evidence,
                neighbour_courses,
                plan_course_ids=plan_courses,
            )
            plan_dict = _score_plan(
                plan_courses=plan_courses,
                scored_courses=plan_scored,
                knn_evidence=knn_evidence,
                remaining_degree_credits=remaining_degree_credits,
                grade_scale=self.grade_scale,
            )
            if plan_dict:
                scored_plans.append(plan_dict)

        # Return only the strongest plans according to the composite score.
        scored_plans.sort(key=lambda p: p["composite_score"], reverse=True)
        return scored_plans[:top_k]
