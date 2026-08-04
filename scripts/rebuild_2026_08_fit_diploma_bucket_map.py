"""Re-fit the diploma bucket map on the ``2026-08_temporal_rebuild_v1`` TRAIN.

Decisions_Log.md, 2026-08-03 Amendment 3, "New decision - the diploma bucket map
is refitted on the new TRAIN":

    The map is to be refitted on the new TRAIN (606,562 rows, through 20233) and
    persisted version-locally under
    data/model_data/versions/2026-08_temporal_rebuild_v1/. The live map is not
    modified. The fitting rule itself - top-five by TRAIN frequency, with
    rare_bucket_label and unseen_bucket_label unchanged - is not altered.

and its 2026-08-03 correction, which places the live map at
``data/artifacts/diploma_type_bucket_map.json``.

This script only fits and persists. It builds no feature, writes no split,
trains nothing, and never opens the live map for writing. TEST is read for one
purpose alone: the reserved-label collision check the rule requires across all
three splits, which reads ``diploma_type_id`` and nothing else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diploma_bucketing import (  # noqa: E402
    DIPLOMA_TYPE_COLUMN,
    assert_no_reserved_label_collision,
    bucket_map_state,
    fit_diploma_bucket_map,
    save_bucket_map,
)
from src.paths import DIPLOMA_TYPE_BUCKET_MAP_PATH, MODEL_DATA_VERSIONS_DIR  # noqa: E402
from src.rebuild_paths import (  # noqa: E402
    REBUILD_SPLITS,
    REBUILD_VERSION,
    rebuild_diploma_bucket_map_path,
    rebuild_generation_paths,
)


EXPECTED_TRAIN_ROWS = 606_562  # Amendment 3, verified against 01_split/split_summary.json
FIT_NOTE = "fitted on the rebuild train split ONLY (base generation)"


def fit(args: argparse.Namespace) -> Path:
    root = Path(args.rebuild_root)
    base_paths = rebuild_generation_paths(root, "base", must_exist=True)
    output_path = rebuild_diploma_bucket_map_path(root)

    frames = {
        split: pd.read_parquet(path, columns=[DIPLOMA_TYPE_COLUMN])
        for split, path in base_paths.items()
    }
    # The rule refuses to run if a real code collides with a reserved label, and
    # that claim is about every split, not just the one being fitted.
    assert_no_reserved_label_collision(frames)

    train = frames["train"]
    if args.expected_train_rows and len(train) != args.expected_train_rows:
        raise AssertionError(
            f"TRAIN has {len(train):,} rows; Amendment 3 records "
            f"{args.expected_train_rows:,}. Refusing to fit on an unexpected split."
        )

    bucket_map = fit_diploma_bucket_map(train)
    state = bucket_map_state(
        bucket_map,
        fit_source_path=base_paths["train"],
        fit_rows=len(train),
        train_manifest_ref=(
            Path(REBUILD_VERSION) / "01_split" / "split_summary.json"
        ).as_posix(),
        fit_note=FIT_NOTE,
        extra={
            "rebuild_version": REBUILD_VERSION,
            "supersedes": {
                "live_map": str(DIPLOMA_TYPE_BUCKET_MAP_PATH),
                "live_map_untouched": True,
                "reason": (
                    "Decisions_Log.md 2026-08-03 Amendment 3: the live map was "
                    "fitted on df_train_difficulty.parquet from the superseded "
                    "split"
                ),
            },
            "fitting_rule_changed": False,
        },
    )
    save_bucket_map(state, output_path)

    print(f"Fitted on : {base_paths['train']}")
    print(f"TRAIN rows: {len(train):,}")
    print(f"Top codes : {state['top_codes']}")
    print(f"Categories: {state['categories']}")
    print(f"Written   : {output_path}")
    print(f"Live map  : {DIPLOMA_TYPE_BUCKET_MAP_PATH} (not opened for writing)")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild-root",
        type=Path,
        default=MODEL_DATA_VERSIONS_DIR / REBUILD_VERSION,
        help="Version root the map is fitted from and persisted under.",
    )
    parser.add_argument(
        "--expected-train-rows",
        type=int,
        default=EXPECTED_TRAIN_ROWS,
        help="Row count Amendment 3 records; 0 disables the check.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    output_path = fit(parse_args(argv))
    state = json.loads(Path(output_path).read_text(encoding="utf-8"))
    print()
    print(
        "Map persisted version-locally. No split was written, no feature built, "
        "no model trained."
    )
    print(f"Splits checked for reserved-label collision: {list(REBUILD_SPLITS)}")
    print(f"Rare bucket {state['rare_bucket_label']}, unseen {state['unseen_bucket_label']} - unchanged.")


if __name__ == "__main__":
    main()
