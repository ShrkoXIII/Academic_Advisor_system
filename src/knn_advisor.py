"""KNN similar-student advisor.

Builds an index of 96K historical student-semester snapshots from the training set.
At inference time, finds the K most similar historical cases and returns their
semester outcomes — average mark, pass rate, GPA earned, courses taken.

This provides "soft evidence": if 18 out of 20 similar students who took a
heavy load in their 4th year failed at least one course, that's a useful
risk signal even if the ML model gave a 0.75 pass probability.

Usage
-----
from src.knn_advisor import KNNAdvisor

advisor = KNNAdvisor.build(df_train)
advisor.save('D:/AI/data_clean_academic_advisor/data/artifacts/knn_index.pkl')

# At inference:
advisor = KNNAdvisor.load('D:/AI/data_clean_academic_advisor/data/artifacts/knn_index.pkl')
neighbours = advisor.find_similar(student_snapshot, k=20)
evidence = advisor.summarize_evidence(neighbours)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Features used to define "snapshot similarity".
# Must all be numeric and available at the time the student is about to register.
KNN_SNAPSHOT_FEATURES = [
    "model_prev_gpa",                       # Most recent GPA signal
    "start_agpa_points",                    # Cumulative GPA at semester start
    "last_valid_gpa_before_current_semester",
    "start_level_ord",                      # Academic year/level (1–6)
    "reg_total_semesters",                  # How many semesters registered so far
    "fail_credit_ratio_capped",             # Fail history ratio
    "total_fail_credits_capped",            # Absolute fail history
    "prior_interruption_count",             # Times interrupted before
    "part_semester",                        # Which term within the year (1/2/3)
    "start_total_in_credits",               # Credits completed before this semester
]

SEM_KEY = ["student_id", "degree_id", "part_id"]


def _build_snapshot_df(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse course-level rows to one row per student-semester.

    Takes the first row per semester group (semester-level features are
    identical across course rows within the same semester).
    """
    # KNN compares semester snapshots, not individual course rows, so collapse
    # each student-degree-semester to the stable semester-level state.
    snaps = (
        df.sort_values(SEM_KEY)
        .groupby(SEM_KEY, sort=False)
        .first()
        .reset_index()
    )

    # Keep outcome aggregates beside each snapshot so neighbours can explain
    # what happened to similar students after they registered.
    outcomes = (
        df.groupby(SEM_KEY)
        .agg(
            sem_avg_mark=("final_mark", "mean"),
            sem_pass_rate=("final_mark", lambda x: (x >= 50).mean()),
            sem_gpa_points=("gpa_points", "first"),
            sem_n_courses=("course_id", "count"),
            sem_total_credits=("course_credits", "sum"),
            sem_course_ids=("course_id", list),
        )
        .reset_index()
    )
    snaps = snaps.merge(outcomes, on=SEM_KEY, how="left")
    return snaps


class KNNAdvisor:
    """Nearest-neighbour advisor over historical student-semester snapshots."""

    def __init__(
        self,
        snapshot_df: pd.DataFrame,
        nn_model: NearestNeighbors,
        scaler: StandardScaler,
    ) -> None:
        self._snaps = snapshot_df.reset_index(drop=True)
        self._nn = nn_model
        self._scaler = scaler

    # ------------------------------------------------------------------
    # Build / persist
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        df_train: pd.DataFrame,
        n_neighbors: int = 20,
        metric: str = "euclidean",
    ) -> "KNNAdvisor":
        """Build the KNN index from a training dataframe."""
        # The index is built only from training history so inference evidence
        # does not leak validation/test outcomes.
        snap_df = _build_snapshot_df(df_train)

        # Fill NaN with column medians before scaling because nearest-neighbour
        # distance cannot represent missing values directly.
        X_raw = snap_df[KNN_SNAPSHOT_FEATURES].copy()
        medians = X_raw.median()
        X_filled = X_raw.fillna(medians).astype(float).values

        # Standardize features so credits, GPA, and counts contribute on a
        # comparable scale to the Euclidean distance.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filled)

        # Use a fitted sklearn index as the persisted query engine.
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm="ball_tree")
        nn.fit(X_scaled)

        # Store the median fill values in the scaler for later inference-time
        # snapshots that have cold-start or partial fields.
        scaler.median_fill_ = medians.to_dict()

        return cls(snap_df, nn, scaler)

    def save(self, path: str) -> None:
        """Pickle the advisor to disk."""
        # Persist the snapshots and preprocessing objects together so query-time
        # behaviour matches build-time scaling and median imputation.
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"snaps": self._snaps, "nn": self._nn, "scaler": self._scaler}, f)

    @classmethod
    def load(cls, path: str) -> "KNNAdvisor":
        # Rehydrate the full advisor state from the artifact produced by save().
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["snaps"], data["nn"], data["scaler"])

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def find_similar(
        self,
        snapshot: pd.Series | dict,
        k: int = 20,
    ) -> pd.DataFrame:
        """Return K most similar historical student-semesters.

        Parameters
        ----------
        snapshot : a Series or dict containing at least the KNN_SNAPSHOT_FEATURES.
            Missing values are filled with training medians.

        Returns
        -------
        DataFrame of K rows from the historical snapshot index, sorted by
        distance (nearest first). Includes all snapshot columns plus
        `_knn_distance`.
        """
        if isinstance(snapshot, dict):
            snapshot = pd.Series(snapshot)

        # Project the caller's snapshot into the exact feature order used when
        # fitting the KNN index.
        x = np.array(
            [
                float(snapshot.get(feat, np.nan))
                for feat in KNN_SNAPSHOT_FEATURES
            ],
            dtype=float,
        )

        # Reuse training medians so missing inference fields do not change the
        # distance scale or make sklearn reject the query.
        medians = getattr(self._scaler, "median_fill_", {})
        for i, feat in enumerate(KNN_SNAPSHOT_FEATURES):
            if np.isnan(x[i]):
                x[i] = float(medians.get(feat, 0.0))

        # Transform with the training scaler and map neighbour indices back to
        # their original historical snapshot rows.
        x_scaled = self._scaler.transform(x.reshape(1, -1))
        distances, indices = self._nn.kneighbors(x_scaled, n_neighbors=min(k, len(self._snaps)))

        result = self._snaps.iloc[indices[0]].copy()
        result["_knn_distance"] = distances[0]
        return result.sort_values("_knn_distance").reset_index(drop=True)

    def summarize_evidence(
        self,
        neighbours: pd.DataFrame,
        plan_course_ids: List[str] | None = None,
    ) -> dict:
        """Summarize KNN neighbours into a scalar evidence dict for plan scoring.

        Parameters
        ----------
        neighbours : output of find_similar()
        plan_course_ids : if given, also computes what fraction of neighbours
            took a similar set of courses and how they did

        Returns
        -------
        dict with keys:
            knn_avg_pass_rate   — mean pass rate across neighbours' semesters
            knn_avg_mark        — mean average mark across neighbours' semesters
            knn_avg_gpa         — mean GPA points earned across neighbours' semesters
            knn_pct_all_passed  — fraction of neighbours who passed ALL courses
            knn_pct_any_failed  — fraction of neighbours who failed at least 1
            knn_similar_plan_avg_mark — avg mark for neighbours who took >=50% same courses
                (NaN if plan_course_ids not given or no overlap found)
            knn_n_used          — number of neighbours used
        """
        ev: dict = {}
        n = len(neighbours)
        ev["knn_n_used"] = n

        # Return a complete evidence shape even when no neighbours are available.
        if n == 0:
            return {k: np.nan for k in [
                "knn_avg_pass_rate", "knn_avg_mark", "knn_avg_gpa",
                "knn_pct_all_passed", "knn_pct_any_failed",
                "knn_similar_plan_avg_mark", "knn_n_used",
            ]}

        # Aggregate broad semester outcomes across the neighbour set.
        ev["knn_avg_pass_rate"] = float(neighbours["sem_pass_rate"].mean())
        ev["knn_avg_mark"] = float(neighbours["sem_avg_mark"].mean())
        ev["knn_avg_gpa"] = float(
            neighbours["sem_gpa_points"].dropna().mean()
            if "sem_gpa_points" in neighbours.columns
            else np.nan
        )
        ev["knn_pct_all_passed"] = float((neighbours["sem_pass_rate"] >= 1.0).mean())
        ev["knn_pct_any_failed"] = float((neighbours["sem_pass_rate"] < 1.0).mean())

        # Add plan-specific evidence only when there is enough overlap with
        # neighbours' historical course sets to make the comparison meaningful.
        if plan_course_ids and "sem_course_ids" in neighbours.columns:
            plan_set = set(plan_course_ids)
            overlaps = []
            for _, row in neighbours.iterrows():
                hist_set = set(row["sem_course_ids"]) if isinstance(row["sem_course_ids"], list) else set()
                if len(hist_set) == 0:
                    continue
                overlap_ratio = len(plan_set & hist_set) / len(plan_set)
                if overlap_ratio >= 0.5:
                    overlaps.append(row["sem_avg_mark"])
            ev["knn_similar_plan_avg_mark"] = float(np.mean(overlaps)) if overlaps else np.nan
        else:
            ev["knn_similar_plan_avg_mark"] = np.nan

        return ev
