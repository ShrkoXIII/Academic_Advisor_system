"""GPA-nearest KNN advisor built from the versioned history tables.

Returning students are matched within their degree by cumulative GPA before
the historical semester.  Cold-start students are matched within their degree
by diploma GPA.  No model is trained; KNN here means selecting the K rows with
the smallest absolute GPA distance.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

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
    """Level-aware GPA KNN with at most one historical semester per student."""

    VERSION = "knn_v2_gpa_level_nearest_v2"
    RETURNING_ROUTE = "returning_degree_level_cumulative_gpa"
    COLD_START_ROUTE = "cold_start_degree_level_diploma_gpa"

    @classmethod
    def build(
        cls,
        outcomes: pd.DataFrame,
        courses: pd.DataFrame,
    ) -> "KNNAdvisorV2Level":
        """Build exact degree-and-level indexes for both student routes."""
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
        return cls(outcome_frame, course_frame, route_positions)

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
            positions[(str(degree_id), int(academic_level))] = group[
                "__position"
            ].to_numpy(dtype=np.int64)
        return positions

    @property
    def metadata(self) -> dict:
        return {
            "version": self.VERSION,
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
            "level_fallback": "none_exact_level_only",
        }

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
        """Return K distinct students from the exact degree and level."""
        if k < 1:
            raise ValueError("k must be at least 1")
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
        query_level = int(query_level_float)

        normalized_degree = str(
            normalize_identifier(pd.Series([degree_id], dtype="string")).iloc[0]
        )
        route = self.COLD_START_ROUTE if cold_start else self.RETURNING_ROUTE
        gpa_column = "diploma_gpa" if cold_start else "cumulative_gpa_before"
        positions = self._route_positions[route].get(
            (normalized_degree, query_level)
        )
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
        candidates["matched_academic_level"] = pd.to_numeric(
            candidates["academic_level_before"], errors="coerce"
        ).astype("Int64")
        candidates["level_fallback_used"] = 0
        candidates["knn_route"] = route
        return (
            candidates.sort_values(
                ["gpa_distance", "part_id", "student_id"], kind="stable"
            )
            .drop_duplicates("student_id", keep="first")
            .head(min(k, candidates["student_id"].nunique()))
            .reset_index(drop=True)
        )


__all__ = [
    "COLD_START_ROUTE",
    "KNNAdvisorV2",
    "KNNAdvisorV2Level",
    "RETURNING_ROUTE",
]
