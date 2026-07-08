from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_parent_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def save_parquet(
    df: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
    **kwargs,
) -> Path:
    output_path = ensure_parent_dir(path)

    df.to_parquet(
        output_path,
        index=index,
        **kwargs,
    )

    return output_path
