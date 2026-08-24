import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.grade_scale import GradeScale, build_grade_scale_intervals


def _acs_grade_fixture() -> pd.DataFrame:
    rows = [
        ("985.111", "3.111", 0, 49, 0.00, "F", "F"),
        ("984.111", "3.111", 50, 54, 1.50, "P", "D"),
        ("983.111", "3.111", 55, 59, 1.75, "P", "D+"),
        ("982.111", "3.111", 60, 64, 2.00, "P", "C-"),
        ("981.111", "3.111", 65, 69, 2.25, "P", "C"),
        ("980.111", "3.111", 70, 74, 2.50, "P", "C+"),
        ("979.111", "3.111", 75, 79, 2.75, "P", "B-"),
        ("978.111", "3.111", 80, 84, 3.00, "P", "B"),
        ("977.111", "3.111", 85, 89, 3.25, "P", "B+"),
        ("976.111", "3.111", 90, 94, 3.50, "P", "A-"),
        ("975.111", "3.111", 95, 97, 3.75, "P", "A"),
        ("974.111", "3.111", 98, 100, 4.00, "P", "A+"),
        # Administrative outcomes must not compete with the numeric F interval.
        ("986.111", "3.111", 0, 0, 0.00, "W", "W"),
        ("987.111", "3.111", 0, 0, 0.00, "FA", "Z"),
        # A different version must never enter the selected authority.
        ("other", "4.111", 0, 100, 4.00, "P", "X"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "grade_id",
            "grade_version_id",
            "from_percent",
            "to_percent",
            "points",
            "finish_status",
            "grade_show",
        ],
    )


class GradeScaleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = _acs_grade_fixture()
        self.scale = GradeScale.from_acs_grade(self.source, "3.111")

    def test_selects_only_version_3_111_numeric_intervals(self):
        intervals = build_grade_scale_intervals(self.source, "3.111")

        self.assertEqual(len(intervals), 12)
        self.assertEqual(set(intervals["finish_status"]), {"F", "P"})
        self.assertNotIn("other", set(intervals["grade_id"]))

    def test_official_boundaries_replace_the_hand_written_approximation(self):
        marks = pd.Series([49.9, 50, 54.9, 55, 75, 79.9, 80, 97.9, 98])

        actual = self.scale.points_for_marks(marks).tolist()

        self.assertEqual(
            actual,
            [0.0, 1.5, 1.5, 1.75, 2.75, 2.75, 3.0, 3.75, 4.0],
        )

    def test_predictions_are_clipped_to_the_official_mark_domain(self):
        self.assertEqual(self.scale.points_for_mark(-2), 0.0)
        self.assertEqual(self.scale.points_for_mark(104), 4.0)

    def test_artifact_round_trip_preserves_version_and_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "acs_grade.parquet"
            artifact_path = Path(tmp) / "scale.json"
            self.source.to_parquet(source_path, index=False)
            self.scale.save(artifact_path, source_path=source_path)
            loaded = GradeScale.load(artifact_path)

        self.assertEqual(loaded.grade_version_id, "3.111")
        self.assertEqual(loaded.points_for_mark(76), 2.75)


if __name__ == "__main__":
    unittest.main()
