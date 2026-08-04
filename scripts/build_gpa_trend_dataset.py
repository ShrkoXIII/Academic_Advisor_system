"""Build a versioned model dataset containing the isolated GPA trend feature.

The build computes the feature from the selected pre-split population, streams
it into copies of the locked split generations, and delegates course-difficulty
enrichment to the versioned B2 builder. The existing train-fitted diploma bucket
values and fitted state are preserved unchanged. Existing model-data files are
never overwritten.

Path modes match the B2 builder: ``--template-dir`` plus generation names by
default, full explicit paths per split, or ``--rebuild-root`` to fill those in
from :mod:`src.rebuild_paths`. As in B2 the ``final`` templates are optional and
all-or-nothing; Phase 1 of the rebuild produced ``base`` only, so under
``--rebuild-root`` the trend columns are streamed into ``base`` alone and reach
the ``difficulty`` generation through B2, which copies every non-difficulty
column forward.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from typing import Any

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
    B2_ALLOWED_CONTRACTS,
    REFERENCE_RUN,
    assert_batch_alignment,
    build as build_b2,
    default_namespace as b2_namespace,
    stores_pandas_index,
)
from src.paths import (  # noqa: E402
    DIPLOMA_TYPE_BUCKET_MAP_PATH,
    MODEL_DATA_DIR,
    MODEL_DATA_VERSIONS_DIR,
    SELECTED_MODEL_POPULATION_PATH,
    assert_data_root,
    model_split_path,
)
from src.rebuild_paths import (  # noqa: E402
    rebuild_diploma_bucket_map_path,
    rebuild_split_path,
)


SPLITS = ("train", "valid", "test")
TREND_COLUMNS = ["gpa_trend_delta", "gpa_trend_missing"]


def _resolve_templates(args: argparse.Namespace) -> dict[str, dict[str, Path]]:
    """Resolve the base (and optional final) templates: explicit, rebuild, legacy."""
    rebuild_root = getattr(args, "rebuild_root", None)
    template_dir = Path(args.template_dir)

    def explicit(generation: str, split: str) -> Path | None:
        value = getattr(args, f"{split}_{generation}", None)
        return Path(value) if value else None

    templates: dict[str, dict[str, Path]] = {"base": {}}
    for split in SPLITS:
        base = explicit("base", split)
        if base is None:
            base = (
                rebuild_split_path(rebuild_root, split, "base", must_exist=True)
                if rebuild_root
                else model_split_path(split, "base", template_dir)
            )
        templates["base"][split] = Path(base)

    explicit_final = {split: explicit("final", split) for split in SPLITS}
    if any(explicit_final.values()) and not all(explicit_final.values()):
        missing = sorted(split for split, path in explicit_final.items() if not path)
        raise ValueError(
            f"final templates are all-or-nothing across splits; missing: {missing}"
        )
    if all(explicit_final.values()):
        templates["final"] = {
            split: Path(path) for split, path in explicit_final.items()
        }
    elif not rebuild_root:
        templates["final"] = {
            split: model_split_path(split, "final", template_dir) for split in SPLITS
        }
    return templates


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
    source_has_index = stores_pandas_index(parquet_file)
    writer: pq.ParquetWriter | None = None
    offset = 0
    try:
        for record_batch in parquet_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([record_batch], schema=parquet_file.schema_arrow)
            batch_frame = table.to_pandas()
            end = offset + len(batch_frame)
            feature_batch = features.iloc[offset:end]
            assert_batch_alignment(
                batch_frame,
                feature_batch,
                source_path=source_path,
                offset=offset,
                end=end,
                source_has_index=source_has_index,
            )
            for column in TREND_COLUMNS:
                if column in batch_frame.columns:
                    raise AssertionError(f"Source already contains {column}: {source_path}")
                batch_frame[column] = feature_batch[column].to_numpy()
            output_table = pa.Table.from_pandas(
                batch_frame, preserve_index=source_has_index
            )
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
    rebuild_root = getattr(args, "rebuild_root", None)
    templates = _resolve_templates(args)
    has_final = "final" in templates
    required_templates = [
        path for generation in templates for path in templates[generation].values()
    ]
    assert_data_root(input_path, *required_templates)
    versions_root.mkdir(parents=True, exist_ok=True)
    build_id = args.build_id or datetime.now().astimezone().strftime(
        "%Y-%m-%d_%H%M%S__gpa_trend_feature"
    )
    if getattr(args, "output_dir", None):
        output_dir = Path(args.output_dir)
    elif rebuild_root:
        output_dir = rebuild_split_path(rebuild_root, "train", "difficulty").parent
    else:
        output_dir = versions_root / build_id
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite versioned dataset: {output_dir}")

    full_semester, primary_semester, _, source_rows = build_semester_trend_from_source(
        input_path
    )
    split_rows: dict[str, int] = {}

    # The B2 builder reads base and final templates, replaces only difficulty
    # columns, verifies lineage, and atomically publishes the new version folder.
    # Trend-carrying copies keep their source basenames, so the temporary layout
    # is the legacy one whenever the templates are the legacy files.
    with TemporaryDirectory(prefix="gpa_trend_build_", dir=versions_root) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        staged_templates: dict[str, dict[str, Path]] = {
            generation: {} for generation in templates
        }
        for name in SPLITS:
            base_source = templates["base"][name]
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
            for generation in templates:
                source = templates[generation][name]
                staged = temp_dir / source.name
                _stream_add_features(source, staged, trend)
                staged_templates[generation][name] = staged

        b2_overrides: dict[str, Any] = {
            "input_dir": str(temp_dir),
            "output_root": str(versions_root),
            "build_id": build_id,
            "min_support": args.min_support,
            "shrinkage_k": args.shrinkage_k,
            "reference_run": args.reference_run,
            "feature_contract": args.feature_contract,
            "output_dir": output_dir,
        }
        if rebuild_root:
            b2_overrides["rebuild_root"] = rebuild_root
        for generation in staged_templates:
            for name in SPLITS:
                b2_overrides[f"{name}_{generation}"] = staged_templates[generation][name]
        published = build_b2(b2_namespace(**b2_overrides))

    # Under --rebuild-root the version-local map is the only correct one:
    # Decisions_Log.md 2026-08-03 Amendment 3 refits it on the new TRAIN and
    # leaves the live map at data/artifacts/ untouched.
    if getattr(args, "diploma_map_path", None):
        diploma_state_source = Path(args.diploma_map_path)
    elif rebuild_root:
        diploma_state_source = rebuild_diploma_bucket_map_path(
            rebuild_root, must_exist=True
        )
    else:
        diploma_state_source = DIPLOMA_TYPE_BUCKET_MAP_PATH
    if not diploma_state_source.exists():
        raise FileNotFoundError(
            f"Diploma bucket map not found: {diploma_state_source}"
        )
    shutil.copy2(
        diploma_state_source,
        published / diploma_state_source.name,
    )
    # The audit's coverage tables only need the semester keys, which every
    # generation carries; point them at whichever generation was published.
    # B2 records the paths it actually wrote, so they are read back rather than
    # recomputed from its naming rules here.
    b2_report = json.loads(
        (published / "b2_data_report.json").read_text(encoding="utf-8")
    )
    published_generation = "final" if has_final else "difficulty"
    published_outputs = {
        name: Path(path)
        for name, path in b2_report["io_plan"]["outputs"][published_generation].items()
    }
    audit_split_paths = {name: published_outputs[name] for name in SPLITS}
    audit_dir = published / "gpa_trend_audit"
    audit_report = create_semester_audit_report(
        full_semester,
        primary_semester,
        audit_dir,
        source_path=input_path,
        source_course_rows=source_rows,
        split_paths=audit_split_paths,
    )
    excluded_course_rows = int(
        full_semester.loc[
            full_semester["exclude_over_policy_semester"], "semester_row_count"
        ].sum()
    )
    primary_course_rows = source_rows - excluded_course_rows
    unassigned_rows = primary_course_rows - sum(split_rows.values())
    train_schema = pq.ParquetFile(published_outputs["train"]).schema_arrow
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
        "feature_contract": args.feature_contract,
        "templates": {
            generation: {split: str(path) for split, path in paths.items()}
            for generation, paths in templates.items()
        },
        "final_generation_built": has_final,
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
    parser.add_argument(
        "--feature-contract",
        required=True,
        choices=list(B2_ALLOWED_CONTRACTS),
        help="Forwarded to the B2 builder's named-contract gate.",
    )
    parser.add_argument(
        "--rebuild-root",
        type=Path,
        help=(
            "Version root of 2026-08_temporal_rebuild_v1. Fills in the base "
            "templates, the B2 output directory, and the version-local diploma "
            "bucket map. Final templates are never auto-resolved in this mode."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Full path of the build directory. Overrides --output-root/--build-id.",
    )
    parser.add_argument(
        "--diploma-map-path",
        type=Path,
        help=(
            "Full path of the diploma bucket map copied into the build. "
            "Defaults to the live map, or to the version-local map under "
            "--rebuild-root."
        ),
    )
    for split in SPLITS:
        parser.add_argument(
            f"--{split}-base",
            type=Path,
            help=f"Full path of the {split} base template.",
        )
    for split in SPLITS:
        parser.add_argument(
            f"--{split}-final",
            type=Path,
            help=(
                f"Full path of the {split} final template. All three or none; "
                "omit them to carry the trend columns through base alone."
            ),
        )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    output_dir = build(parse_args(argv))
    print(f"Versioned GPA trend dataset built: {output_dir}")


if __name__ == "__main__":
    main()
