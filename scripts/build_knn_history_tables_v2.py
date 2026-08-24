"""Build level-aware TRAIN history from student-degree-status V2."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.knn_history_helpers import (  # noqa: E402
    GRADE_COLUMNS,
    LEVEL_STATUS_COLUMNS,
    STATUS_COLUMNS,
    TRAIN_COLUMNS,
    attach_official_references,
    build_student_semester_courses,
    build_student_semester_outcomes,
    ensure_new_output_files,
)
from src.paths import RAW_DIR, assert_data_root  # noqa: E402


DEFAULT_TRAIN = (
    PROJECT_ROOT
    / "data"
    / "model_data"
    / "versions"
    / "2026-08_temporal_rebuild_v2"
    / "05_dataset"
    / "train_dataset_candidate.parquet"
)
DEFAULT_STATUS = RAW_DIR / "v_add_student_degree_status_v2.parquet"
DEFAULT_GRADES = RAW_DIR / "v_acs_grade.parquet"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "artifacts" / "knn" / "2026-08-23_history_v2_level"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--status-path", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--grades-path", type=Path, default=DEFAULT_GRADES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    semester_path = args.output_dir / "student_semester_outcomes.parquet"
    courses_path = args.output_dir / "student_semester_courses.parquet"
    assert_data_root(args.train_path, args.status_path, args.grades_path)
    ensure_new_output_files([semester_path, courses_path])

    train = pd.read_parquet(args.train_path, columns=TRAIN_COLUMNS)
    student_status = pd.read_parquet(
        args.status_path, columns=STATUS_COLUMNS + LEVEL_STATUS_COLUMNS
    )
    grades = pd.read_parquet(args.grades_path, columns=GRADE_COLUMNS)

    enriched = attach_official_references(train, student_status, grades)
    semester_courses = build_student_semester_courses(enriched)
    semester_outcomes = build_student_semester_outcomes(enriched)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    semester_outcomes.to_parquet(semester_path, index=False)
    semester_courses.to_parquet(courses_path, index=False)
    print(f"student_semester_outcomes: {len(semester_outcomes):,} -> {semester_path}")
    print(f"student_semester_courses:  {len(semester_courses):,} -> {courses_path}")


if __name__ == "__main__":
    main()
