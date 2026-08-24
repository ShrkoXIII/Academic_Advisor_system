"""Build the official recommendation grade scale from ACS_GRADE version 3.111."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.grade_scale import (  # noqa: E402
    DEFAULT_GRADE_SCALE_PATH,
    OFFICIAL_GRADE_VERSION_ID,
    GradeScale,
)
from src.paths import PREPROCESSED_DIR, assert_data_root  # noqa: E402


DEFAULT_ACS_GRADE_PATH = (
    PREPROCESSED_DIR / "V_ACS_GRADE" / "clean_v_acs_grade.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acs-grade", type=Path, default=DEFAULT_ACS_GRADE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_GRADE_SCALE_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_data_root(args.acs_grade)
    acs_grade = pd.read_parquet(args.acs_grade)
    scale = GradeScale.from_acs_grade(
        acs_grade, grade_version_id=OFFICIAL_GRADE_VERSION_ID
    )
    scale.save(args.output, source_path=args.acs_grade)

    print(f"Grade-scale artifact: {args.output}")
    print(f"grade_version_id: {scale.grade_version_id}")
    print(scale.intervals.to_string(index=False))


if __name__ == "__main__":
    main()
