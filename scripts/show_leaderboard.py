"""Read-only terminal viewer for the experiment leaderboard."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.experiment_tracking import LEADERBOARD_FIELDS
from src.paths import MODELS_DIR


DESCENDING_METRICS = {
    "m1_valid_auc",
    "m1_valid_fail_precision",
    "m1_valid_fail_recall",
    "m1_valid_fail_f1",
    "m2_valid_r2",
}
ASCENDING_METRICS = {"m2_valid_mae"}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sort", required=True, choices=sorted(DESCENDING_METRICS | ASCENDING_METRICS))
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--root", default=str(MODELS_DIR))
    args = parser.parse_args(argv)
    if args.top <= 0:
        raise ValueError("--top must be positive")
    path = Path(args.root) / "runs" / "leaderboard.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Leaderboard not found: {path}")
    df = pd.read_csv(path)
    if list(df.columns) != LEADERBOARD_FIELDS:
        raise ValueError(f"Leaderboard schema mismatch: {path}")
    print(df.sort_values(args.sort, ascending=args.sort in ASCENDING_METRICS).head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
