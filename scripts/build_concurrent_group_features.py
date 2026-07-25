"""Build a versioned model dataset adding the Concurrent Course Group feature.

The feature is a pure within-split transform of already train-only-derived
difficulty columns (a student-semester group never spans splits), so there is no
cross-split fitting and no train-only state to persist. For each split the six
concurrent columns are computed from the locked *difficulty* generation and
streamed onto BOTH a new *concurrent* generation (difficulty + 6 columns) and a
new *final* generation (the locked final template + 6 columns). The final
template already carries the train-fitted ``diploma_type_bucket``, which is
copied through unchanged (governance contract 15 — load/copy, never refit).

Writes go ONLY into ``MODEL_DATA_VERSIONS_DIR/<build_id>/`` (governance L2/L3);
the live ``MODEL_DATA_DIR`` splits are read read-only as templates and never
modified. Promotion to a live generation is a separate later commit (L5), done
only after VALID results confirm improvement.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.concurrent_group_features import (  # noqa: E402
    AUDIT_CONCURRENT_FEATURES,
    CONCURRENT_FEATURE_COLUMNS,
    MODEL_CONCURRENT_FEATURES,
    REQUIRED_INPUT_COLUMNS,
    compute_concurrent_group_features,
)
from src.feature_engineering import SEMESTER_KEY  # noqa: E402
from src.paths import (  # noqa: E402
    MODEL_DATA_DIR,
    MODEL_DATA_VERSIONS_DIR,
    assert_data_root,
    model_split_path,
)


SPLITS = ("train", "valid", "test")
# Amendment 2: positional identity used to prove the two templates are aligned.
FIVE_ID_COLUMNS = SEMESTER_KEY + ["course_id"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_template_alignment(diff_path: Path, final_path: Path, split: str) -> int:
    """Assert FULL positional equality of the five id columns (Amendment 2).

    Row-count equality alone is not acceptable. Fails loudly at the first
    mismatching row position.
    """
    diff_ids = pd.read_parquet(diff_path, columns=FIVE_ID_COLUMNS)
    final_ids = pd.read_parquet(final_path, columns=FIVE_ID_COLUMNS)
    if len(diff_ids) != len(final_ids):
        raise AssertionError(
            f"{split}: difficulty/final row counts differ "
            f"({len(diff_ids):,} vs {len(final_ids):,})"
        )
    if not diff_ids.index.equals(final_ids.index):
        raise AssertionError(f"{split}: difficulty/final template indexes differ")
    for column in FIVE_ID_COLUMNS:
        left = diff_ids[column].to_numpy()
        right = final_ids[column].to_numpy()
        equal = (left == right) | (
            pd.isna(diff_ids[column]).to_numpy() & pd.isna(final_ids[column]).to_numpy()
        )
        if not equal.all():
            first = int(np.flatnonzero(~equal)[0])
            raise AssertionError(
                f"{split}: difficulty/final template mismatch at row {first} "
                f"column {column!r}: {left[first]!r} != {right[first]!r}"
            )
    return len(diff_ids)


def _compute_features(diff_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(diff_path, columns=REQUIRED_INPUT_COLUMNS)
    return compute_concurrent_group_features(frame)


def _stream_append_columns(
    source_path: Path,
    output_path: Path,
    features: pd.DataFrame,
    *,
    batch_size: int = 25_000,
) -> None:
    """Append the concurrent columns while preserving every source row/column."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_path}")
    parquet_file = pq.ParquetFile(source_path)
    writer: pq.ParquetWriter | None = None
    offset = 0
    try:
        for record_batch in parquet_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([record_batch], schema=parquet_file.schema_arrow)
            batch_frame = table.to_pandas()
            end = offset + len(batch_frame)
            feature_batch = features.iloc[offset:end]
            # Positional guard only: per-batch to_pandas() reconstructs its own
            # RangeIndex from 0 regardless of the batch's true offset in the
            # file (no stored index label is guaranteed here), so comparing
            # index LABELS between batch_frame and feature_batch is unverified
            # and would either pass by accident or fail spuriously. Row order
            # within each batch is preserved by iter_batches; the assignment
            # below is already positional via .to_numpy().
            if len(feature_batch) != len(batch_frame):
                raise AssertionError(
                    f"Row-count mismatch while streaming {source_path.name}: "
                    f"batch has {len(batch_frame)} rows, features slice "
                    f"[{offset}:{end}] has {len(feature_batch)} rows"
                )
            for column in CONCURRENT_FEATURE_COLUMNS:
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
            f"Concurrent stream row mismatch: source={offset}, features={len(features)}"
        )


def _assert_diploma_type_bucket_identical(final_template: Path, final_output: Path, split: str) -> None:
    """Prove the streamed final carried diploma_type_bucket through byte-for-byte."""
    before = pd.read_parquet(final_template, columns=["diploma_type_bucket"])
    after = pd.read_parquet(final_output, columns=["diploma_type_bucket"])
    left = before["diploma_type_bucket"].to_numpy()
    right = after["diploma_type_bucket"].to_numpy()
    equal = (left == right) | (
        pd.isna(before["diploma_type_bucket"]).to_numpy()
        & pd.isna(after["diploma_type_bucket"]).to_numpy()
    )
    if not bool(equal.all()):
        first = int(np.flatnonzero(~equal)[0])
        raise AssertionError(
            f"{split}: diploma_type_bucket changed at row {first}: "
            f"{left[first]!r} != {right[first]!r}"
        )


def build(args: argparse.Namespace) -> Path:
    template_dir = Path(args.template_dir)
    output_root = Path(args.output_root)

    required_templates = [
        model_split_path(split, generation, template_dir)
        for split in SPLITS
        for generation in ("difficulty", "final")
    ]
    assert_data_root(*required_templates)

    build_id = args.build_id or datetime.now().astimezone().strftime(
        "%Y-%m-%d_%H%M%S__concurrent_group_feature"
    )
    output_dir = output_root / build_id
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite versioned dataset: {output_dir}")

    # --- Cheap gates first (schema + template alignment), no split data held. ---
    for split in SPLITS:
        diff_path = model_split_path(split, "difficulty", template_dir)
        final_path = model_split_path(split, "final", template_dir)
        schema_names = list(pq.ParquetFile(diff_path).schema.names)
        missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in schema_names]
        if missing:
            raise ValueError(
                f"{split}: difficulty template {diff_path} missing required columns "
                f"{missing}; cannot build the concurrent generation"
            )
        _assert_template_alignment(diff_path, final_path, split)

    # --- Build into a staging dir, ONE split at a time to bound peak memory
    #     (the effective per-process cap is well below system RAM), then publish
    #     atomically so a failure never leaves a partial version folder. ---
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f"{build_id}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    row_counts: dict[str, int] = {}
    resolved_inputs: dict[str, dict] = {}
    try:
        for split in SPLITS:
            diff_path = model_split_path(split, "difficulty", template_dir)
            final_path = model_split_path(split, "final", template_dir)
            features = _compute_features(diff_path)
            concurrent_out = model_split_path(split, "concurrent", staging)
            final_out = model_split_path(split, "final", staging)
            _stream_append_columns(diff_path, concurrent_out, features)
            _stream_append_columns(final_path, final_out, features)
            _assert_diploma_type_bucket_identical(final_path, final_out, split)
            row_counts[split] = int(len(features))
            resolved_inputs[split] = {
                "difficulty": {"path": str(diff_path), "sha256": _sha256(diff_path)},
                "final": {"path": str(final_path), "sha256": _sha256(final_path)},
            }
            del features
            gc.collect()

        train_final_schema = pq.ParquetFile(
            model_split_path("train", "final", staging)
        ).schema_arrow
        new_dtypes = {
            column: str(train_final_schema.field(column).type)
            for column in CONCURRENT_FEATURE_COLUMNS
        }
        build_report = _build_report_dict(
            build_id, output_dir, template_dir, row_counts, new_dtypes, resolved_inputs
        )
        (staging / "concurrent_build_report.json").write_text(
            json.dumps(build_report, indent=2), encoding="utf-8"
        )
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def _build_report_dict(
    build_id, output_dir, template_dir, row_counts, new_dtypes, resolved_inputs
) -> dict:
    return {
        "build_id": build_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output": str(output_dir),
        "template_dir": str(template_dir),
        "row_counts": row_counts,
        "model_features_added": MODEL_CONCURRENT_FEATURES,
        "audit_features_added": AUDIT_CONCURRENT_FEATURES,
        "new_feature_dtypes": new_dtypes,
        "diploma_type_bucket_identical": True,
        "resolved_inputs": resolved_inputs,
        "notes": (
            "Within-split transform of difficulty columns; no cross-split "
            "fitting. Live MODEL_DATA_DIR untouched. diploma_type_bucket carried "
            "from the final template unchanged."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=MODEL_DATA_DIR,
        help="Read-only source of the locked difficulty/final split generations.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=MODEL_DATA_VERSIONS_DIR,
        help="Versions root; a new immutable <build_id> folder is created under it.",
    )
    parser.add_argument("--build-id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    output_dir = build(parse_args(argv))
    print(f"Versioned concurrent-group dataset built: {output_dir}")


if __name__ == "__main__":
    main()
