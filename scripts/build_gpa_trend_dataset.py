"""Build a versioned model dataset containing the isolated GPA trend feature.

The build computes the feature from the selected pre-split population, streams
it into copies of the locked split generations, and delegates course-difficulty
enrichment to the versioned B2 builder. The existing train-fitted diploma bucket
values and fitted state are preserved unchanged. Existing model-data files are
never overwritten.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_gpa_trend import (  # noqa: E402
    SEMESTER_KEY,
    build_semester_trend_from_source,
    create_semester_audit_report,
    map_trend_to_course_keys,
)
from scripts.build_b2_temporal_course_stats import (  # noqa: E402
    REFERENCE_RUN,
    build as build_b2,
)
from src.paths import (  # noqa: E402
    DIPLOMA_TYPE_BUCKET_MAP_PATH,
    MODEL_DATA_DIR,
    MODEL_DATA_VERSIONS_DIR,
    SELECTED_MODEL_POPULATION_PATH,
    assert_data_root,
    model_split_path,
)


SPLITS = ("train", "valid", "test")
TREND_COLUMNS = ["gpa_trend_delta", "gpa_trend_missing"]


def _stream_add_features(
    source_path: Path,
    output_path: Path,
    features: pd.DataFrame,
    *,
    batch_size: int = 25_000,
) -> None:
    """Append the two trend columns while preserving every source row/column."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite temporary output: {output_path}")
    parquet_file = pq.ParquetFile(source_path)
    writer: pq.ParquetWriter | None = None
    offset = 0
    try:
        for record_batch in parquet_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([record_batch], schema=parquet_file.schema_arrow)
            batch_frame = table.to_pandas()
            end = offset + len(batch_frame)
            feature_batch = features.iloc[offset:end]
            if not batch_frame.index.equals(feature_batch.index):
                raise AssertionError(
                    f"Index/order mismatch while streaming {source_path.name}: {offset}:{end}"
                )
            for column in TREND_COLUMNS:
                if column in batch_frame.columns:
                    raise AssertionError(f"Source already contains {column}: {source_path}")
                batch_frame[column] = feature_batch[column].to_numpy()
            output_table = pa.Table.from_pandas(batch_frame, preserve_index=True)
            if writer is None:
                writer = pq.ParquetWriter(output_path, output_table.schema, compression="snappy")
            writer.write_table(output_table)
            offset = end
    finally:
        if writer is not None:
            writer.close()
    if offset != len(features):
        raise AssertionError(
            f"Trend stream row mismatch: source={offset}, features={len(features)}"
        )


def build(args: argparse.Namespace) -> Path:
    input_path = Path(args.input)
    versions_root = Path(args.output_root)
    template_dir = Path(args.template_dir)
    required_templates = [
        model_split_path(split, generation, template_dir)
        for split in SPLITS
        for generation in ("base", "final")
    ]
    assert_data_root(input_path, *required_templates)
    versions_root.mkdir(parents=True, exist_ok=True)
    build_id = args.build_id or datetime.now().astimezone().strftime(
        "%Y-%m-%d_%H%M%S__gpa_trend_feature"
    )
    output_dir = versions_root / build_id
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite versioned dataset: {output_dir}")

    full_semester, primary_semester, _, source_rows = build_semester_trend_from_source(
        input_path
    )
    split_rows: dict[str, int] = {}

    # The B2 builder reads base and final templates, replaces only difficulty
    # columns, verifies lineage, and atomically publishes the new version folder.
    with TemporaryDirectory(prefix="gpa_trend_build_", dir=versions_root) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for name in SPLITS:
            base_source = model_split_path(name, "base", template_dir)
            final_source = model_split_path(name, "final", template_dir)
            keys = pd.read_parquet(
                base_source,
                columns=SEMESTER_KEY + ["last_valid_gpa_before_current_semester"],
            )
            mapped = map_trend_to_course_keys(keys, primary_semester)
            existing_t1 = pd.to_numeric(
                keys["last_valid_gpa_before_current_semester"], errors="coerce"
            )
            rebuilt_t1 = pd.to_numeric(
                mapped["last_valid_gpa_before_current_semester"], errors="coerce"
            )
            equal_t1 = (existing_t1.isna() & rebuilt_t1.isna()) | existing_t1.eq(rebuilt_t1)
            if not equal_t1.all():
                raise AssertionError(
                    f"{name}: narrow pre-split rebuild disagrees with persisted last-valid GPA"
                )
            trend = mapped[TREND_COLUMNS]
            split_rows[name] = int(len(trend))
            _stream_add_features(
                base_source,
                model_split_path(name, "base", temp_dir),
                trend,
            )
            _stream_add_features(
                final_source,
                model_split_path(name, "final", temp_dir),
                trend,
            )

        b2_args = SimpleNamespace(
            input_dir=str(temp_dir),
            output_root=str(versions_root),
            build_id=build_id,
            min_support=args.min_support,
            shrinkage_k=args.shrinkage_k,
            reference_run=args.reference_run,
        )
        published = build_b2(b2_args)

    diploma_state_source = DIPLOMA_TYPE_BUCKET_MAP_PATH
    shutil.copy2(
        diploma_state_source,
        published / DIPLOMA_TYPE_BUCKET_MAP_PATH.name,
    )
    audit_dir = published / "gpa_trend_audit"
    audit_report = create_semester_audit_report(
        full_semester,
        primary_semester,
        audit_dir,
        source_path=input_path,
        source_course_rows=source_rows,
        split_dir=published,
    )
    excluded_course_rows = int(
        full_semester.loc[
            full_semester["exclude_over_policy_semester"], "semester_row_count"
        ].sum()
    )
    primary_course_rows = source_rows - excluded_course_rows
    unassigned_rows = primary_course_rows - sum(split_rows.values())
    train_schema = pq.ParquetFile(
        model_split_path("train", "final", published)
    ).schema_arrow
    persisted_dtypes = {
        column: str(train_schema.field(column).type) for column in TREND_COLUMNS
    }
    build_report = {
        "build_id": build_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input": str(input_path),
        "output": str(published),
        "reference_run": args.reference_run,
        "template_dir": str(template_dir),
        "row_counts": split_rows,
        "unassigned_rows": unassigned_rows,
        "new_features": TREND_COLUMNS,
        "gpa_trend_dtypes": persisted_dtypes,
        "audit_candidate_dtypes": audit_report["persisted_candidate_dtypes"],
        "diploma_bucketing": {
            "policy": "existing train-fitted values and state preserved unchanged",
            "state_source": str(diploma_state_source),
        },
        "dropped_features_changed": False,
    }
    (published / "gpa_trend_build_report.json").write_text(
        json.dumps(build_report, indent=2), encoding="utf-8"
    )
    return published


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=SELECTED_MODEL_POPULATION_PATH,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=MODEL_DATA_VERSIONS_DIR,
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=MODEL_DATA_DIR,
        help="Locked base/final split generations copied into the new version.",
    )
    parser.add_argument("--build-id")
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--shrinkage-k", type=float, default=20.0)
    parser.add_argument("--reference-run", default=REFERENCE_RUN)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    output_dir = build(parse_args(argv))
    print(f"Versioned GPA trend dataset built: {output_dir}")


if __name__ == "__main__":
    main()
