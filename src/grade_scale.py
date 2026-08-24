"""Official mark-to-GPA lookup sourced from ACS_GRADE."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.knn_history_helpers import normalize_identifier
from src.paths import ARTIFACTS_DIR


GRADE_SCALE_SCHEMA_VERSION = 1
OFFICIAL_GRADE_VERSION_ID = "3.111"
DEFAULT_GRADE_SCALE_PATH = (
    ARTIFACTS_DIR / "grade_scales" / "acs_grade_3.111.json"
)

REQUIRED_ACS_GRADE_COLUMNS = [
    "grade_id",
    "grade_version_id",
    "from_percent",
    "to_percent",
    "points",
    "finish_status",
    "grade_show",
]

INTERVAL_COLUMNS = [
    "grade_id",
    "from_percent",
    "to_percent",
    "points",
    "finish_status",
    "grade_show",
]


def _validate_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    """Validate one unambiguous, contiguous integer mark scale from 0 to 100."""
    missing = [column for column in INTERVAL_COLUMNS if column not in intervals.columns]
    if missing:
        raise KeyError(f"Grade scale intervals are missing columns: {missing}")
    if intervals.empty:
        raise ValueError("Grade scale has no numeric mark intervals")

    result = intervals[INTERVAL_COLUMNS].copy()
    result["grade_id"] = normalize_identifier(result["grade_id"])
    result["finish_status"] = result["finish_status"].astype("string").str.strip()
    result["grade_show"] = result["grade_show"].astype("string").str.strip()
    for column in ["from_percent", "to_percent", "points"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["from_percent", "to_percent", "points"]].isna().any().any():
        raise ValueError("Grade scale contains non-numeric interval values")

    for column in ["from_percent", "to_percent"]:
        values = result[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{column} must contain integer percentage boundaries")
        result[column] = result[column].astype(int)

    if (result["from_percent"] < 0).any() or (result["to_percent"] > 100).any():
        raise ValueError("Grade scale boundaries must stay within 0..100")
    if (result["from_percent"] > result["to_percent"]).any():
        raise ValueError("Grade scale contains a reversed interval")
    if (~np.isfinite(result["points"].to_numpy(dtype=float))).any():
        raise ValueError("Grade scale points must be finite")
    if (result["points"] < 0).any() or (result["points"] > 4).any():
        raise ValueError("Grade scale points must stay within 0..4")

    result = result.sort_values(
        ["from_percent", "to_percent", "grade_id"], kind="stable"
    ).reset_index(drop=True)
    marks = np.arange(0, 101)
    coverage = np.zeros(len(marks), dtype=np.int16)
    for row in result.itertuples(index=False):
        coverage += (
            (marks >= int(row.from_percent)) & (marks <= int(row.to_percent))
        ).astype(np.int16)
    if not np.equal(coverage, 1).all():
        invalid = marks[coverage != 1].tolist()
        raise ValueError(
            "Grade scale must cover every integer mark 0..100 exactly once; "
            f"invalid marks: {invalid}"
        )
    return result


def build_grade_scale_intervals(
    acs_grade: pd.DataFrame,
    grade_version_id: str = OFFICIAL_GRADE_VERSION_ID,
) -> pd.DataFrame:
    """Select and validate numeric intervals for one ACS_GRADE version."""
    missing = [
        column for column in REQUIRED_ACS_GRADE_COLUMNS if column not in acs_grade.columns
    ]
    if missing:
        raise KeyError(f"ACS_GRADE is missing required columns: {missing}")

    normalized_version = str(
        normalize_identifier(pd.Series([grade_version_id], dtype="string")).iloc[0]
    )
    versions = normalize_identifier(acs_grade["grade_version_id"])
    selected = acs_grade.loc[versions.eq(normalized_version)].copy()
    if selected.empty:
        raise ValueError(
            f"ACS_GRADE has no rows for grade_version_id={normalized_version!r}"
        )

    selected["from_percent"] = pd.to_numeric(
        selected["from_percent"], errors="coerce"
    )
    selected["to_percent"] = pd.to_numeric(selected["to_percent"], errors="coerce")

    # ACS_GRADE also contains administrative outcomes (W, I, FA, FE, etc.)
    # represented as zero-width 0..0 rows. Predicted numeric marks must use only
    # the actual percentage intervals; the official F interval already owns 0..49.
    numeric_intervals = selected.loc[
        selected["to_percent"].gt(selected["from_percent"])
    ].copy()
    return _validate_intervals(numeric_intervals)


class GradeScale:
    """Validated official percentage-to-GPA scale for one grade version."""

    def __init__(self, grade_version_id: str, intervals: pd.DataFrame) -> None:
        self.grade_version_id = str(grade_version_id)
        self.intervals = _validate_intervals(intervals)
        self._lower_bounds = self.intervals["from_percent"].to_numpy(dtype=float)
        self._points = self.intervals["points"].to_numpy(dtype=float)

    @classmethod
    def from_acs_grade(
        cls,
        acs_grade: pd.DataFrame,
        grade_version_id: str = OFFICIAL_GRADE_VERSION_ID,
    ) -> "GradeScale":
        normalized_version = str(
            normalize_identifier(
                pd.Series([grade_version_id], dtype="string")
            ).iloc[0]
        )
        intervals = build_grade_scale_intervals(acs_grade, normalized_version)
        return cls(normalized_version, intervals)

    def points_for_marks(self, marks: Iterable[float] | pd.Series) -> pd.Series:
        """Convert predicted marks using the official lower interval boundaries."""
        source = marks if isinstance(marks, pd.Series) else pd.Series(marks)
        numeric = pd.to_numeric(source, errors="coerce")
        result = pd.Series(np.nan, index=source.index, dtype=float)
        valid = numeric.notna() & np.isfinite(numeric.astype(float))
        if not valid.any():
            return result

        # Model predictions may slightly exceed the physical mark range. Keep
        # recommendation serving deterministic by clipping to ACS_GRADE's 0..100.
        clipped = numeric.loc[valid].astype(float).clip(lower=0.0, upper=100.0)
        positions = np.searchsorted(
            self._lower_bounds, clipped.to_numpy(), side="right"
        ) - 1
        result.loc[valid] = self._points[positions]
        return result

    def points_for_mark(self, mark: float) -> float:
        """Convert one predicted mark to official GPA points."""
        return float(self.points_for_marks(pd.Series([mark])).iloc[0])

    def save(self, path: str | Path, *, source_path: str | Path) -> None:
        """Persist the validated scale plus source provenance as JSON."""
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"Grade-scale artifact already exists: {destination}")
        source = Path(source_path)
        payload = {
            "schema_version": GRADE_SCALE_SCHEMA_VERSION,
            "source_table": "ACS_GRADE",
            "source_path": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "grade_version_id": self.grade_version_id,
            "intervals": self.intervals.to_dict(orient="records"),
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "GradeScale":
        """Load and validate an artifact produced by :meth:`save`."""
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema_version") != GRADE_SCALE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported grade-scale schema version: "
                f"{payload.get('schema_version')!r}"
            )
        if payload.get("source_table") != "ACS_GRADE":
            raise ValueError("Grade-scale artifact source_table must be ACS_GRADE")
        if payload.get("grade_version_id") != OFFICIAL_GRADE_VERSION_ID:
            raise ValueError(
                "Recommendation requires ACS_GRADE grade_version_id "
                f"{OFFICIAL_GRADE_VERSION_ID}, got {payload.get('grade_version_id')!r}"
            )
        return cls(
            payload["grade_version_id"],
            pd.DataFrame(payload["intervals"]),
        )


__all__ = [
    "DEFAULT_GRADE_SCALE_PATH",
    "GRADE_SCALE_SCHEMA_VERSION",
    "GradeScale",
    "OFFICIAL_GRADE_VERSION_ID",
    "build_grade_scale_intervals",
]
