"""Plan generation, scoring, and ranking — the full recommendation engine.

This is the top-level module that ties together inference (grade/pass prediction)
and KNN evidence to produce ranked course-plan recommendations.

9-step pipeline (per readmi.md):
  Steps 1-3: provided externally (student snapshot, rules engine, candidate courses)
  Step 4-5: ML scoring via src.inference
  Step 6:   Plan candidate generation (this module)
  Step 7:   KNN soft evidence (this module, via src.knn_advisor)
  Step 8:   Plan scoring on 4 axes (this module)
  Step 9:   Final ranked recommendations (this module)

Usage
-----
from src.recommendation import Recommender

rec = Recommender.load(
    grade_model_path='models/grade_model.lgbm',
    pass_model_path='models/pass_model.lgbm',
    difficulty_lookup_path='D:/AI/.../course_difficulty_lookup.parquet',
    knn_index_path='D:/AI/.../knn_index.pkl',
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
from src.inference import StudentScorer
from src.knn_advisor import KNNAdvisor

# Minimum pass probability — courses below this are excluded from candidate plans
MIN_PASS_PROB_FOR_PLAN = 0.30


def _mark_to_gpa_points(mark: float) -> float:
    """Approximate conversion from final_mark (0–100) to GPA points (0–4)."""
    # Plan scoring optimizes GPA-like utility, so model marks are converted into
    # the same coarse scale used by academic planning discussions.
    if mark >= 90:
        return 4.0
    if mark >= 80:
        return 3.5
    if mark >= 70:
        return 3.0
    if mark >= 60:
        return 2.5
    if mark >= 50:
        return 2.0
    return 0.0


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

    # Convert course-level predicted marks into a credit-weighted expected AGPA.
    pred_gpa = rows["pred_mark"].apply(_mark_to_gpa_points)
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
        "composite_score": round(composite, 4),
    }


class Recommender:
    """Full recommendation engine: inference + plan generation + KNN + ranking."""

    def __init__(
        self,
        scorer: StudentScorer,
        knn_advisor: KNNAdvisor,
    ) -> None:
        # Compose the two independent evidence providers so recommendation code
        # can rank plans without knowing artifact-loading details.
        self.scorer = scorer
        self.knn = knn_advisor

    @classmethod
    def load(
        cls,
        grade_model_path: str,
        pass_model_path: str,
        difficulty_lookup_path: str,
        knn_index_path: str,
    ) -> "Recommender":
        # Keep the public loader as the single entry point for trained model,
        # difficulty, and KNN artifacts.
        scorer = StudentScorer.load(grade_model_path, pass_model_path, difficulty_lookup_path)
        knn = KNNAdvisor.load(knn_index_path)
        return cls(scorer, knn)

    def recommend(
        self,
        df_history: pd.DataFrame,
        candidate_course_ids: List[str],
        target_part_id: str,
        degree_id: str,
        remaining_degree_credits: Optional[float] = None,
        n_plans: int = 500,
        top_k: int = 5,
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
            composite_score
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

        # Fetch similar historical semesters once and reuse them for all plan
        # evidence summaries.
        neighbours = self.knn.find_similar(snapshot, k=20)

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
            knn_evidence = self.knn.summarize_evidence(neighbours, plan_course_ids=plan_courses)
            plan_dict = _score_plan(
                plan_courses=plan_courses,
                scored_courses=plan_scored,
                knn_evidence=knn_evidence,
                remaining_degree_credits=remaining_degree_credits,
            )
            if plan_dict:
                scored_plans.append(plan_dict)

        # Return only the strongest plans according to the composite score.
        scored_plans.sort(key=lambda p: p["composite_score"], reverse=True)
        return scored_plans[:top_k]
