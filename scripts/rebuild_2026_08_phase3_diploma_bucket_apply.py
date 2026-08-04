"""Apply the version-local diploma bucket map to produce the ``final`` generation.

This is the step the notebook ``03_diploma_type_bucketing.ipynb`` performed for
the live pipeline: read the ``difficulty`` generation, append the single
model-facing column ``diploma_type_bucket``, and write the ``final`` generation.
The fitting rule is not re-run here - the map was fitted once on the rebuild's
TRAIN and persisted under the version root (Decisions_Log.md 2026-08-03
Amendment 3), and this step only applies it.

``diploma_type_id`` stays untouched and audit-only, exactly one column is added,
and every other column is carried through Arrow-native so no dtype can drift.
Nothing is written outside the version root, and the live map at
``data/artifacts/diploma_type_bucket_map.json`` is never read.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diploma_bucketing import (  # noqa: E402
    DIPLOMA_BUCKET_COLUMN,
    DIPLOMA_TYPE_COLUMN,
    assert_no_reserved_label_collision,
    load_bucket_map,
)
from src.paths import MODEL_DATA_VERSIONS_DIR  # noqa: E402
from src.rebuild_paths import (  # noqa: E402
    REBUILD_SPLITS,
    REBUILD_VERSION,
    rebuild_diploma_bucket_map_path,
    rebuild_split_path,
)

TEST_PROVISIONAL_COLUMN = "test_provisional_20251_only"


def _append_bucket_column(
    source_path: Path,
    output_path: Path,
    buckets: pd.Series,
    *,
    batch_size: int = 25_000,
) -> dict[str, object]:
    """Append one column, carrying the source table through untouched.

    Arrow-native for the same reason the concurrent builder is: a pandas
    round-trip re-derives the schema from pandas dtypes and can silently move
    or retype a template column.
    """
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_path}")
    parquet_file = pq.ParquetFile(source_path)
    source_schema = parquet_file.schema_arrow
    if DIPLOMA_BUCKET_COLUMN in source_schema.names:
        raise AssertionError(
            f"Template already contains {DIPLOMA_BUCKET_COLUMN}: {source_path}"
        )

    writer: pq.ParquetWriter | None = None
    offset = 0
    output_names: list[str] = []
    try:
        for record_batch in parquet_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([record_batch], schema=source_schema)
            end = offset + table.num_rows
            batch_values = buckets.iloc[offset:end]
            if len(batch_values) != table.num_rows:
                raise AssertionError(
                    f"Streaming row mismatch for {source_path.name}: "
                    f"source batch={table.num_rows}, buckets={len(batch_values)}"
                )
            table = table.append_column(
                DIPLOMA_BUCKET_COLUMN,
                pa.array(batch_values.to_numpy(), type=pa.int64()),
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path, table.schema, compression="snappy"
                )
                output_names = list(table.schema.names)
            writer.write_table(table)
            offset = end
    finally:
        if writer is not None:
            writer.close()

    if offset != len(buckets):
        raise AssertionError(
            f"Streaming row mismatch: source={offset:,}, buckets={len(buckets):,}"
        )
    return {
        "rows": int(offset),
        "source_columns": int(len(source_schema.names)),
        "output_columns": int(len(output_names)),
        "columns_added": [DIPLOMA_BUCKET_COLUMN],
    }


def _verify_output(
    source_path: Path,
    output_path: Path,
    buckets: pd.Series,
    categories: tuple[int, ...],
) -> dict[str, object]:
    """Read the written file back and re-check it against the template."""
    source = pq.ParquetFile(source_path)
    output = pq.ParquetFile(output_path)
    if source.metadata.num_rows != output.metadata.num_rows:
        raise AssertionError(f"{output_path.name}: row count changed")
    added = [
        name for name in output.schema_arrow.names if name not in source.schema_arrow.names
    ]
    if added != [DIPLOMA_BUCKET_COLUMN]:
        raise AssertionError(f"{output_path.name}: unexpected column change {added}")
    if output.schema_arrow.names[: len(source.schema_arrow.names)] != list(
        source.schema_arrow.names
    ):
        raise AssertionError(f"{output_path.name}: template column order changed")

    written = pq.read_table(output_path, columns=[DIPLOMA_BUCKET_COLUMN]).column(0)
    written_values = written.to_numpy(zero_copy_only=False)
    if written.null_count:
        raise AssertionError(f"{output_path.name}: bucket column contains nulls")
    if not (written_values == buckets.to_numpy()).all():
        raise AssertionError(f"{output_path.name}: bucket values changed on readback")
    unknown = sorted(set(int(v) for v in written_values) - set(categories))
    if unknown:
        raise AssertionError(
            f"{output_path.name}: bucket values outside the fitted categories: {unknown}"
        )
    raw_before = pq.read_table(source_path, columns=[DIPLOMA_TYPE_COLUMN]).column(0)
    raw_after = pq.read_table(output_path, columns=[DIPLOMA_TYPE_COLUMN]).column(0)
    if raw_before.to_pylist() != raw_after.to_pylist():
        raise AssertionError(f"{output_path.name}: audit-only diploma_type_id changed")
    return {
        "rows": int(output.metadata.num_rows),
        "columns_added": added,
        "template_column_order_preserved": True,
        "bucket_value_counts": {
            str(int(value)): int(count)
            for value, count in pd.Series(written_values).value_counts().sort_index().items()
        },
    }


def apply_map(args: argparse.Namespace) -> Path:
    root = Path(args.rebuild_root)
    map_path = (
        Path(args.diploma_map_path)
        if args.diploma_map_path
        else rebuild_diploma_bucket_map_path(root, must_exist=True)
    )
    bucket_map = load_bucket_map(map_path)

    sources = {
        split: rebuild_split_path(root, split, "difficulty", must_exist=True)
        for split in REBUILD_SPLITS
    }
    outputs = {
        split: rebuild_split_path(root, split, "final") for split in REBUILD_SPLITS
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite final generation: {existing}")

    raw = {
        split: pd.read_parquet(path, columns=[DIPLOMA_TYPE_COLUMN])
        for split, path in sources.items()
    }
    # The rule refuses to run when a real code equals a reserved bucket label,
    # and that claim is about every split.
    assert_no_reserved_label_collision(
        raw, rare_bucket_label=bucket_map.rare_bucket_label
    )

    splits_report: dict[str, object] = {}
    for split in REBUILD_SPLITS:
        buckets = bucket_map.apply(raw[split][DIPLOMA_TYPE_COLUMN]).astype("int64")
        write_report = _append_bucket_column(sources[split], outputs[split], buckets)
        verify_report = _verify_output(
            sources[split], outputs[split], buckets, bucket_map.categories
        )
        provisional = TEST_PROVISIONAL_COLUMN in pq.ParquetFile(
            outputs[split]
        ).schema_arrow.names
        splits_report[split] = {
            "source": str(sources[split]),
            "output": str(outputs[split]),
            **write_report,
            "readback": verify_report,
            TEST_PROVISIONAL_COLUMN: bool(provisional),
            "null_diploma_type_id": int(raw[split][DIPLOMA_TYPE_COLUMN].isna().sum()),
        }
        print(
            f"{split:6s} rows={write_report['rows']:>7,} "
            f"cols {write_report['source_columns']} -> {write_report['output_columns']} "
            f"buckets={verify_report['bucket_value_counts']}"
        )

    report = {
        "artifact": "rebuild_diploma_bucket_apply",
        "rebuild_version": REBUILD_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "bucket_map": {
            "path": str(map_path),
            "top_codes": list(bucket_map.top_codes),
            "rare_bucket_label": bucket_map.rare_bucket_label,
            "unseen_bucket_label": bucket_map.unseen_bucket_label,
            "categories": list(bucket_map.categories),
            "fitting_rule_rerun": False,
        },
        "columns_added": [DIPLOMA_BUCKET_COLUMN],
        "audit_only_column_untouched": DIPLOMA_TYPE_COLUMN,
        "test_provisional_20251_only": True,
        "splits": splits_report,
        "model_trained": False,
        "version_promoted": False,
    }
    report_path = outputs["train"].parent / "diploma_bucket_apply_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Report: {report_path}")
    return report_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild-root",
        type=Path,
        default=MODEL_DATA_VERSIONS_DIR / REBUILD_VERSION,
        help="Version root holding 03_features and the fitted bucket map.",
    )
    parser.add_argument(
        "--diploma-map-path",
        type=Path,
        help="Override the version-local map. Never defaults to the live map.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    apply_map(parse_args(argv))
    print("final generation written. No model trained, no version promoted.")


if __name__ == "__main__":
    main()
