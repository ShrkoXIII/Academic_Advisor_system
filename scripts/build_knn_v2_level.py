"""Build the exact degree-and-level GPA KNN artifact from History V2."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.knn_advisor_v2 import KNNAdvisorV2Level  # noqa: E402
from src.paths import assert_data_root  # noqa: E402


DEFAULT_HISTORY_DIR = (
    PROJECT_ROOT / "data" / "artifacts" / "knn" / "2026-08-23_history_v2_level"
)
DEFAULT_OUTPUT = DEFAULT_HISTORY_DIR / "knn_v2_gpa_level_nearest.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outcomes_path = args.history_dir / "student_semester_outcomes.parquet"
    courses_path = args.history_dir / "student_semester_courses.parquet"
    assert_data_root(outcomes_path, courses_path)

    outcomes = pd.read_parquet(outcomes_path)
    courses = pd.read_parquet(courses_path)
    advisor = KNNAdvisorV2Level.build(outcomes, courses)
    advisor.save(args.output)

    print(f"KNN v2 level artifact: {args.output}")
    for key, value in advisor.metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
