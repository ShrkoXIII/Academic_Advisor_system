"""GPA-nearest advisors built from the versioned history tables.

``KNNAdvisorV2`` is retained for compatibility with the original lookup
artifact.  The active level-aware ``KNNAdvisorV2Level`` uses fitted sklearn
``KNeighborsClassifier`` and ``KNeighborsRegressor`` estimators.  Returning
students are modelled within their exact degree and level using cumulative GPA;
cold-start students use diploma GPA inside the same degree and level.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from src.knn_history_helpers import SEMESTER_KEY, normalize_identifier, require_columns


RETURNING_ROUTE = "returning_cumulative_gpa"
COLD_START_ROUTE = "cold_start_diploma_gpa"

OUTCOME_COLUMNS = [
    *SEMESTER_KEY,
    "cumulative_gpa_before",
    "diploma_gpa",
    "is_first_active_semester",
    "semester_average_mark",
    "term_gpa",
    "term_gpa_delta",
    "cumulative_gpa_delta",
    "term_gpa_improved",
    "cumulative_gpa_improved",
    "any_course_failed",
    "all_courses_passed",
]

COURSE_COLUMNS = [
    "student_course_id",
    *SEMESTER_KEY,
    "course_id",
    "course_credits",
    "attempt_number",
    "final_mark",
    "finish_status",
    "is_passed",
    "is_failed",
]

LEVEL_OUTCOME_COLUMNS = [
    *OUTCOME_COLUMNS,
    "academic_level_before",
    "academic_level_after",
    "academic_level_delta",
    "academic_level_advanced",
]


@dataclass  
class _SklearnKNNGroup:
    """Fitted sklearn estimators and source-row positions for one route group."""

    positions: np.ndarray
    classifier: KNeighborsClassifier
    regressor: KNeighborsRegressor


class KNNAdvisorV2:
    """Find nearest historical student semesters by one explicit GPA signal."""

    VERSION = "knn_v2_gpa_nearest_v1"

    def __init__(
        self,
        outcomes: pd.DataFrame,
        courses: pd.DataFrame,
        route_positions: dict[str, dict[str, np.ndarray]],
    ) -> None:
        self._outcomes = outcomes.reset_index(drop=True)
        self._courses = courses.reset_index(drop=True)
        self._route_positions = route_positions

    @classmethod
    def build(
        cls,
        outcomes: pd.DataFrame,
        courses: pd.DataFrame,
    ) -> "KNNAdvisorV2":
        """Build degree-specific returning and cold-start GPA indexes."""
        require_columns(outcomes, OUTCOME_COLUMNS, name="student semester outcomes")
        require_columns(courses, COURSE_COLUMNS, name="student semester courses")

        outcome_frame = outcomes[OUTCOME_COLUMNS].copy()
        course_frame = courses[COURSE_COLUMNS].copy()
        for frame in (outcome_frame, course_frame):
            for column in SEMESTER_KEY:
                frame[column] = normalize_identifier(frame[column])
        course_frame["course_id"] = normalize_identifier(course_frame["course_id"])
        course_frame["student_course_id"] = normalize_identifier(
            course_frame["student_course_id"]
        )

        outcome_frame = outcome_frame.sort_values(
            ["degree_id", "part_id", "student_id"], kind="stable"
        ).reset_index(drop=True)
        route_positions = {
            RETURNING_ROUTE: cls._positions_by_degree(
                outcome_frame,
                outcome_frame["is_first_active_semester"].eq(0)
                & pd.to_numeric(
                    outcome_frame["cumulative_gpa_before"], errors="coerce"
                ).gt(0),
            ),
            COLD_START_ROUTE: cls._positions_by_degree(
                outcome_frame,
                outcome_frame["is_first_active_semester"].eq(1)
                & pd.to_numeric(outcome_frame["diploma_gpa"], errors="coerce").notna(),
            ),
        }
        return cls(outcome_frame, course_frame, route_positions)

    @staticmethod
    def _positions_by_degree(
        outcomes: pd.DataFrame,
        mask: pd.Series,
    ) -> dict[str, np.ndarray]:
        positions: dict[str, np.ndarray] = {}
        eligible = outcomes.loc[mask, ["degree_id"]].copy()
        eligible["__position"] = eligible.index.to_numpy()
        for degree_id, group in eligible.groupby("degree_id", sort=False):
            positions[str(degree_id)] = group["__position"].to_numpy(dtype=np.int64)
        return positions

    @property
    def metadata(self) -> dict:
        """Return compact counts for reporting and artifact inspection."""
        return {
            "version": self.VERSION,
            "semester_rows": int(len(self._outcomes)),
            "course_rows": int(len(self._courses)),
            "returning_rows": int(
                sum(
                    len(values)
                    for values in self._route_positions[RETURNING_ROUTE].values()
                )
            ),
            "cold_start_rows": int(
                sum(
                    len(values)
                    for values in self._route_positions[COLD_START_ROUTE].values()
                )
            ),
            "returning_degree_count": int(
                len(self._route_positions[RETURNING_ROUTE])
            ),
            "cold_start_degree_count": int(
                len(self._route_positions[COLD_START_ROUTE])
            ),
        }

    def find_nearest_gpa(
        self,
        *,
        degree_id: str,
        gpa: float,
        cold_start: bool = False,
        k: int = 20,
        exclude_student_id: str | None = None,
    ) -> pd.DataFrame:
        """Return K historical semesters with the closest GPA in one degree."""
        if k < 1:
            raise ValueError("k must be at least 1")
        query_gpa = float(gpa)
        if not np.isfinite(query_gpa):
            raise ValueError("gpa must be a finite number")

        normalized_degree = str(
            normalize_identifier(pd.Series([degree_id], dtype="string")).iloc[0]
        )
        route = COLD_START_ROUTE if cold_start else RETURNING_ROUTE
        gpa_column = "diploma_gpa" if cold_start else "cumulative_gpa_before"
        positions = self._route_positions[route].get(normalized_degree)
        if positions is None or len(positions) == 0:
            return self._empty_neighbours()

        candidates = self._outcomes.iloc[positions].copy()
        if exclude_student_id is not None:
            normalized_student = str(
                normalize_identifier(
                    pd.Series([exclude_student_id], dtype="string")
                ).iloc[0]
            )
            candidates = candidates.loc[
                candidates["student_id"].ne(normalized_student)
            ].copy()
        if candidates.empty:
            return self._empty_neighbours()

        candidates["matched_gpa"] = pd.to_numeric(
            candidates[gpa_column], errors="coerce"
        )
        candidates["gpa_distance"] = (candidates["matched_gpa"] - query_gpa).abs()
        candidates["knn_route"] = route
        return (
            candidates.sort_values(
                ["gpa_distance", "part_id", "student_id"], kind="stable"
            )
            .head(min(k, len(candidates)))
            .reset_index(drop=True)
        )

    def _empty_neighbours(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[*self._outcomes.columns, "matched_gpa", "gpa_distance", "knn_route"]
        )

    def courses_for_neighbours(self, neighbours: pd.DataFrame) -> pd.DataFrame:
        """Return course rows belonging to the selected neighbour semesters."""
        require_columns(neighbours, SEMESTER_KEY, name="neighbours")
        if neighbours.empty:
            return self._courses.iloc[0:0].copy()
        keys = neighbours[SEMESTER_KEY].drop_duplicates()
        return self._courses.merge(
            keys,
            on=SEMESTER_KEY,
            how="inner",
            validate="many_to_one",
        )

    def summarize(self, neighbours: pd.DataFrame) -> dict:
        """Summarize neighbour GPA distance and their subsequent outcomes."""
        if neighbours.empty:
            return {
                "support": 0,
                "knn_route": None,
                "mean_matched_gpa": None,
                "median_gpa_distance": None,
                "mean_term_gpa": None,
                "mean_term_gpa_delta": None,
                "mean_cumulative_gpa_delta": None,
                "pct_term_gpa_improved": None,
                "pct_cumulative_gpa_improved": None,
                "pct_any_course_failed": None,
            }

        def mean_or_none(column: str) -> float | None:
            values = pd.to_numeric(neighbours[column], errors="coerce").dropna()
            return float(values.mean()) if len(values) else None

        return {
            "support": int(len(neighbours)),
            "knn_route": str(neighbours["knn_route"].iloc[0]),
            "mean_matched_gpa": mean_or_none("matched_gpa"),
            "median_gpa_distance": float(neighbours["gpa_distance"].median()),
            "mean_term_gpa": mean_or_none("term_gpa"),
            "mean_term_gpa_delta": mean_or_none("term_gpa_delta"),
            "mean_cumulative_gpa_delta": mean_or_none("cumulative_gpa_delta"),
            "pct_term_gpa_improved": mean_or_none("term_gpa_improved"),
            "pct_cumulative_gpa_improved": mean_or_none(
                "cumulative_gpa_improved"
            ),
            "pct_any_course_failed": mean_or_none("any_course_failed"),
        }

    def save(self, path: str | Path) -> None:
        """Persist the trusted local history and its degree indexes together."""
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"KNN v2 artifact already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(
                {
                    "version": self.VERSION,
                    "outcomes": self._outcomes,
                    "courses": self._courses,
                    "route_positions": self._route_positions,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> "KNNAdvisorV2":
        """Load a KNN v2 artifact produced by :meth:`save`."""
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("version") != cls.VERSION:
            raise ValueError(
                f"Unsupported KNN artifact version: {payload.get('version')!r}"
            )
        return cls(
            payload["outcomes"],
            payload["courses"],
            payload["route_positions"],
        )


class KNNAdvisorV2Level(KNNAdvisorV2):
    """Fitted sklearn GPA KNN models partitioned by degree and level."""

    VERSION = "knn_v2_level_sklearn_v3"
    RETURNING_ROUTE = "returning_degree_level_cumulative_gpa"
    COLD_START_ROUTE = "cold_start_degree_level_diploma_gpa"
    DEFAULT_N_NEIGHBORS = 20
    ESTIMATOR_BACKEND = "sklearn"

    def __init__(
        self,
        outcomes: pd.DataFrame,
        courses: pd.DataFrame,
        route_positions: dict[str, dict[tuple[str, int], np.ndarray]],
        group_models: dict[str, dict[tuple[str, int], _SklearnKNNGroup]],
        n_neighbors: int,
        weights: str,
        metric: str,
    ) -> None:
        super().__init__(outcomes, courses, route_positions)
        self._group_models = group_models
        self.n_neighbors = int(n_neighbors)
        self.weights = str(weights)
        self.metric = str(metric)

    @classmethod
    def build(
        cls,
        outcomes: pd.DataFrame,
        courses: pd.DataFrame,
        *,
        n_neighbors: int = DEFAULT_N_NEIGHBORS,
        weights: str = "uniform",
        metric: str = "euclidean",
    ) -> "KNNAdvisorV2Level":
        """Fit sklearn classifiers/regressors for every route-degree-level group."""
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be either 'uniform' or 'distance'")
        require_columns(
            outcomes,
            LEVEL_OUTCOME_COLUMNS,
            name="student semester outcomes v2 level",
        )
        require_columns(courses, COURSE_COLUMNS, name="student semester courses")

        outcome_frame = outcomes[LEVEL_OUTCOME_COLUMNS].copy()
        course_frame = courses[COURSE_COLUMNS].copy()
        for frame in (outcome_frame, course_frame):
            for column in SEMESTER_KEY:
                frame[column] = normalize_identifier(frame[column])
        course_frame["course_id"] = normalize_identifier(course_frame["course_id"])
        course_frame["student_course_id"] = normalize_identifier(
            course_frame["student_course_id"]
        )
        outcome_frame["academic_level_before"] = pd.to_numeric(
            outcome_frame["academic_level_before"], errors="coerce"
        ).astype("Int64")
        if outcome_frame["academic_level_before"].isna().any():
            raise ValueError("academic_level_before contains missing values")

        outcome_frame = outcome_frame.sort_values(
            ["degree_id", "academic_level_before", "part_id", "student_id"],
            kind="stable",
        ).reset_index(drop=True)

        # A standard supervised KNN treats every row as one training sample.
        # Keep one deterministic, latest snapshot per student in each group so
        # prolific students cannot receive more voting weight than their peers.
        route_positions = {
            cls.RETURNING_ROUTE: cls._positions_by_degree_level(
                outcome_frame,
                outcome_frame["is_first_active_semester"].eq(0)
                & pd.to_numeric(
                    outcome_frame["cumulative_gpa_before"], errors="coerce"
                ).gt(0),
            ),
            cls.COLD_START_ROUTE: cls._positions_by_degree_level(
                outcome_frame,
                outcome_frame["is_first_active_semester"].eq(1)
                & pd.to_numeric(outcome_frame["diploma_gpa"], errors="coerce").notna(),
            ),
        }
        group_models = {
            cls.RETURNING_ROUTE: cls._fit_route_models(
                outcome_frame,
                route_positions[cls.RETURNING_ROUTE],
                gpa_column="cumulative_gpa_before",
                n_neighbors=n_neighbors,
                weights=weights,
                metric=metric,
            ),
            cls.COLD_START_ROUTE: cls._fit_route_models(
                outcome_frame,
                route_positions[cls.COLD_START_ROUTE],
                gpa_column="diploma_gpa",
                n_neighbors=n_neighbors,
                weights=weights,
                metric=metric,
            ),
        }
        return cls(
            outcome_frame,
            course_frame,
            route_positions,
            group_models,
            n_neighbors,
            weights,
            metric,
        )

    @staticmethod
    def _positions_by_degree_level(
        outcomes: pd.DataFrame,
        mask: pd.Series,
    ) -> dict[tuple[str, int], np.ndarray]:
        positions: dict[tuple[str, int], np.ndarray] = {}
        eligible = outcomes.loc[
            mask, ["degree_id", "academic_level_before"]
        ].copy()
        eligible["__position"] = eligible.index.to_numpy()
        for (degree_id, academic_level), group in eligible.groupby(
            ["degree_id", "academic_level_before"], sort=False
        ):
            # ``outcomes`` is sorted by part_id, so keep='last' selects the
            # latest trusted TRAIN snapshot for each student in this group.
            unique_students = (
                outcomes.iloc[group["__position"].to_numpy(dtype=np.int64)]
                .drop_duplicates("student_id", keep="last")
            )
            positions[(str(degree_id), int(academic_level))] = (
                unique_students.index.to_numpy(dtype=np.int64)
            )
        return positions

    @staticmethod
    def _fit_route_models(
        outcomes: pd.DataFrame,
        route_positions: dict[tuple[str, int], np.ndarray],
        *,
        gpa_column: str,
        n_neighbors: int,
        weights: str,
        metric: str,
    ) -> dict[tuple[str, int], _SklearnKNNGroup]:
        models: dict[tuple[str, int], _SklearnKNNGroup] = {}
        for group_key, positions in route_positions.items():
            frame = outcomes.iloc[positions]
            features = pd.to_numeric(frame[gpa_column], errors="coerce")
            targets = frame[
                ["any_course_failed", "term_gpa", "semester_average_mark"]
            ].apply(pd.to_numeric, errors="coerce")
            if features.isna().any() or targets.isna().any().any():
                raise ValueError(
                    f"KNN training group {group_key!r} contains missing features or targets"
                )

            x_train = features.to_numpy(dtype=float).reshape(-1, 1)
            classifier_target = targets["any_course_failed"].astype(int).to_numpy()
            regression_targets = targets[
                ["term_gpa", "semester_average_mark"]
            ].to_numpy(dtype=float)
            fitted_k = min(int(n_neighbors), len(frame))

            classifier = KNeighborsClassifier(
                n_neighbors=fitted_k,
                weights=weights,
                metric=metric,
                algorithm="auto",
            )
            regressor = KNeighborsRegressor(
                n_neighbors=fitted_k,
                weights=weights,
                metric=metric,
                algorithm="auto",
            )
            classifier.fit(x_train, classifier_target)
            regressor.fit(x_train, regression_targets)
            models[group_key] = _SklearnKNNGroup(
                positions=np.asarray(positions, dtype=np.int64),
                classifier=classifier,
                regressor=regressor,
            )
        return models

    @property
    def metadata(self) -> dict:
        return {
            "version": self.VERSION,
            "backend": self.ESTIMATOR_BACKEND,
            "classifier": "sklearn.neighbors.KNeighborsClassifier",
            "regressor": "sklearn.neighbors.KNeighborsRegressor",
            "fit_called": True,
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "metric": self.metric,
            "semester_rows": int(len(self._outcomes)),
            "course_rows": int(len(self._courses)),
            "returning_rows": int(
                sum(
                    len(values)
                    for values in self._route_positions[
                        self.RETURNING_ROUTE
                    ].values()
                )
            ),
            "cold_start_rows": int(
                sum(
                    len(values)
                    for values in self._route_positions[
                        self.COLD_START_ROUTE
                    ].values()
                )
            ),
            "returning_degree_level_group_count": int(
                len(self._route_positions[self.RETURNING_ROUTE])
            ),
            "cold_start_degree_level_group_count": int(
                len(self._route_positions[self.COLD_START_ROUTE])
            ),
            "unique_student_per_result": True,
            "training_snapshot_policy": "latest_per_student_degree_level_route",
            "level_fallback": "none_exact_level_only",
        }

    @staticmethod
    def _validate_query(
        *, degree_id: str, academic_level: int, gpa: float
    ) -> tuple[str, int, float]:
        query_gpa = float(gpa)
        if not np.isfinite(query_gpa):
            raise ValueError("gpa must be a finite number")
        query_level_float = float(academic_level)
        if (
            not np.isfinite(query_level_float)
            or not query_level_float.is_integer()
            or query_level_float < 1
        ):
            raise ValueError("academic_level must be a positive integer")
        normalized_degree = str(
            normalize_identifier(pd.Series([degree_id], dtype="string")).iloc[0]
        )
        return normalized_degree, int(query_level_float), query_gpa

    @staticmethod
    def _failure_probability(classifier: KNeighborsClassifier, x: np.ndarray) -> np.ndarray:
        probabilities = classifier.predict_proba(x)
        failure_columns = np.flatnonzero(classifier.classes_ == 1)
        if len(failure_columns) == 0:
            return np.zeros(len(x), dtype=float)
        return probabilities[:, int(failure_columns[0])].astype(float)

    def predict(
        self,
        *,
        degree_id: str,
        academic_level: int,
        gpa: float,
        cold_start: bool = False,
        k: int | None = None,
    ) -> dict:
        """Run native sklearn ``predict``/``predict_proba`` for one student."""
        if k is not None and int(k) != self.n_neighbors:
            raise ValueError(
                f"This artifact was fitted with n_neighbors={self.n_neighbors}; "
                f"received k={k}. Build a separate fitted artifact to change K."
            )
        normalized_degree, query_level, query_gpa = self._validate_query(
            degree_id=degree_id,
            academic_level=academic_level,
            gpa=gpa,
        )
        route = self.COLD_START_ROUTE if cold_start else self.RETURNING_ROUTE
        group = self._group_models[route].get((normalized_degree, query_level))
        if group is None:
            return {
                "covered": False,
                "support": 0,
                "knn_route": None,
                "predicted_any_course_failed": None,
                "failure_probability": None,
                "predicted_term_gpa": None,
                "predicted_semester_average_mark": None,
            }

        x_query = np.asarray([[query_gpa]], dtype=float)
        predicted_class = int(group.classifier.predict(x_query)[0])
        failure_probability = float(
            self._failure_probability(group.classifier, x_query)[0]
        )
        predicted_regression = group.regressor.predict(x_query)[0]
        return {
            "covered": True,
            "support": int(group.classifier.n_neighbors),
            "knn_route": route,
            "predicted_any_course_failed": predicted_class,
            "failure_probability": failure_probability,
            "predicted_term_gpa": float(predicted_regression[0]),
            "predicted_semester_average_mark": float(predicted_regression[1]),
        }

    def predict_frame(self, queries: pd.DataFrame) -> pd.DataFrame:
        """Vectorized native sklearn prediction for canonical KNN query columns.

        Required columns are ``degree_id``, ``academic_level``, ``gpa``, and
        ``cold_start``.  The returned frame preserves the caller's index.
        """
        require_columns(
            queries,
            ["degree_id", "academic_level", "gpa", "cold_start"],
            name="KNN prediction queries",
        )
        result = pd.DataFrame(
            {
                "covered": False,
                "support": 0,
                "knn_route": pd.Series(None, index=queries.index, dtype="object"),
                "predicted_any_course_failed": pd.Series(
                    pd.NA, index=queries.index, dtype="Int64"
                ),
                "failure_probability": np.nan,
                "predicted_term_gpa": np.nan,
                "predicted_semester_average_mark": np.nan,
            },
            index=queries.index,
        )
        prepared = queries.copy()
        prepared["degree_id"] = normalize_identifier(prepared["degree_id"])
        prepared["academic_level"] = pd.to_numeric(
            prepared["academic_level"], errors="raise"
        ).astype(int)
        prepared["gpa"] = pd.to_numeric(prepared["gpa"], errors="raise")
        if not np.isfinite(prepared["gpa"].to_numpy(dtype=float)).all():
            raise ValueError("gpa must contain only finite numbers")
        prepared["cold_start"] = prepared["cold_start"].astype(bool)

        for (cold_start, degree_id, academic_level), group_queries in prepared.groupby(
            ["cold_start", "degree_id", "academic_level"],
            sort=False,
            dropna=False,
        ):
            route = self.COLD_START_ROUTE if cold_start else self.RETURNING_ROUTE
            group = self._group_models[route].get((str(degree_id), int(academic_level)))
            if group is None:
                continue
            x_query = group_queries["gpa"].to_numpy(dtype=float).reshape(-1, 1)
            regression = group.regressor.predict(x_query)
            index = group_queries.index
            result.loc[index, "covered"] = True
            result.loc[index, "support"] = int(group.classifier.n_neighbors)
            result.loc[index, "knn_route"] = route
            result.loc[index, "predicted_any_course_failed"] = (
                group.classifier.predict(x_query).astype(int)
            )
            result.loc[index, "failure_probability"] = self._failure_probability(
                group.classifier, x_query
            )
            result.loc[index, "predicted_term_gpa"] = regression[:, 0]
            result.loc[index, "predicted_semester_average_mark"] = regression[:, 1]
        return result

    def find_nearest_gpa(
        self,
        *,
        degree_id: str,
        academic_level: int,
        gpa: float,
        cold_start: bool = False,
        k: int = 20,
        exclude_student_id: str | None = None,
    ) -> pd.DataFrame:
        """Return neighbours using sklearn's fitted ``kneighbors`` index."""
        if k < 1:
            raise ValueError("k must be at least 1")
        normalized_degree, query_level, query_gpa = self._validate_query(
            degree_id=degree_id,
            academic_level=academic_level,
            gpa=gpa,
        )
        route = self.COLD_START_ROUTE if cold_start else self.RETURNING_ROUTE
        gpa_column = "diploma_gpa" if cold_start else "cumulative_gpa_before"
        group = self._group_models[route].get((normalized_degree, query_level))
        if group is None or len(group.positions) == 0:
            return self._empty_neighbours()
        request_count = min(
            len(group.positions),
            k + (1 if exclude_student_id is not None else 0),
        )
        distances, local_indices = group.classifier.kneighbors(
            np.asarray([[query_gpa]], dtype=float),
            n_neighbors=request_count,
            return_distance=True,
        )
        selected_positions = group.positions[local_indices[0]]
        candidates = self._outcomes.iloc[selected_positions].copy()
        candidates["matched_gpa"] = pd.to_numeric(
            candidates[gpa_column], errors="coerce"
        ).to_numpy(dtype=float)
        candidates["gpa_distance"] = distances[0]
        if exclude_student_id is not None:
            normalized_student = str(
                normalize_identifier(
                    pd.Series([exclude_student_id], dtype="string")
                ).iloc[0]
            )
            candidates = candidates.loc[
                candidates["student_id"].ne(normalized_student)
            ].copy()
        if candidates.empty:
            return self._empty_neighbours()
        candidates["matched_academic_level"] = pd.to_numeric(
            candidates["academic_level_before"], errors="coerce"
        ).astype("Int64")
        candidates["level_fallback_used"] = 0
        candidates["knn_route"] = route
        return candidates.head(k).reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        """Persist fitted sklearn estimators and their trusted TRAIN rows."""
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"KNN sklearn artifact already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(
                {
                    "version": self.VERSION,
                    "outcomes": self._outcomes,
                    "courses": self._courses,
                    "route_positions": self._route_positions,
                    "group_models": self._group_models,
                    "n_neighbors": self.n_neighbors,
                    "weights": self.weights,
                    "metric": self.metric,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> "KNNAdvisorV2Level":
        """Load an artifact containing fitted sklearn KNN estimators."""
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("version") != cls.VERSION:
            raise ValueError(
                f"Unsupported sklearn KNN artifact version: {payload.get('version')!r}"
            )
        return cls(
            payload["outcomes"],
            payload["courses"],
            payload["route_positions"],
            payload["group_models"],
            payload["n_neighbors"],
            payload["weights"],
            payload["metric"],
        )


__all__ = [
    "COLD_START_ROUTE",
    "KNNAdvisorV2",
    "KNNAdvisorV2Level",
    "RETURNING_ROUTE",
]
