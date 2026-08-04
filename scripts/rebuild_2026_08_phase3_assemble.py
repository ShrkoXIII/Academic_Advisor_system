"""Assemble the Phase 3 model-facing dataset for ``2026-08_temporal_rebuild_v1``.

One file per split carrying the union of the two named contracts:

    M1 = baseline_41
    M2 = concurrent_43

``baseline_41`` is a strict subset of ``concurrent_43``, so the union is
``concurrent_43``: 43 features, of which 42 are read from the parquet and
``requirement_size_bucket_ord`` is derived at training time from
``requirement_size_bucket``. Both are checked here rather than assumed.

``concurrent_peer_difficulty_missing`` is a dead feature, excluded by decision
(Decisions_Log.md; it was used by any model in only 2 of 5 seeds, under 0.001%
of total gain, zero splits everywhere else). The concurrent builder still emits
it, so this step drops it and asserts its absence.

Identity is the original ``degree_id`` and ``course_id``. Phase 2 is not
authorised, and every output records:

    lineage_applied = false
    lineage_reason  = NOT_AUTHORISED_BY_OWNER_DESPITE_PROCEED

per Decisions_Log.md 2026-08-03 Amendment 2, Correction 4. TEST is provisional
from ``20251`` only (Correction 2) and carries ``test_provisional_20251_only``
as both a column and file-level metadata.

Nothing here trains, evaluates, promotes, or writes outside the version root.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.concurrent_group_features import (  # noqa: E402
    CONCURRENT_FEATURE_COLUMNS,
    MODEL_CONCURRENT_FEATURES,
)
from src.course_difficulty import (  # noqa: E402
    AUDIT_OUTPUT_COLUMNS,
    STAT_OUTPUT_COLUMNS,
)
from src.diploma_bucketing import DIPLOMA_BUCKET_COLUMN  # noqa: E402
from src.model_training import (  # noqa: E402
    DERIVED_FEATURE_SOURCES,
    SEGMENT_ONLY_COLUMNS,
    TARGET_GRADE,
    resolve_feature_contract,
)
from src.paths import MODEL_DATA_VERSIONS_DIR  # noqa: E402
from src.rebuild_paths import (  # noqa: E402
    REBUILD_SPLITS,
    REBUILD_VERSION,
    rebuild_dataset_path,
    rebuild_split_path,
)

M1_CONTRACT = "baseline_41"
M2_CONTRACT = "concurrent_43"

DEAD_FEATURE = "concurrent_peer_difficulty_missing"
TEST_PROVISIONAL_COLUMN = "test_provisional_20251_only"

LINEAGE_APPLIED = False
LINEAGE_REASON = "NOT_AUTHORISED_BY_OWNER_DESPITE_PROCEED"
IDENTITY_COLUMNS = ["degree_id", "course_id"]

# The difficulty contract is imported, never restated. Amendment-scope rule:
# exactly these nine columns, no more.
DIFFICULTY_CONTRACT = tuple(STAT_OUTPUT_COLUMNS) + tuple(AUDIT_OUTPUT_COLUMNS)

TREND_FEATURES = ("gpa_trend_delta", "gpa_trend_missing")

# Which module produced each family. Every entry is a constant imported from the
# module it names, so the manifest cannot drift from the code.
BUILDER_MODULES: dict[str, tuple[str, ...]] = {
    "src/course_difficulty.py": DIFFICULTY_CONTRACT,
    "src/concurrent_group_features.py": tuple(CONCURRENT_FEATURE_COLUMNS),
    "src/diploma_bucketing.py": (DIPLOMA_BUCKET_COLUMN,),
    "scripts/audit_gpa_trend.py": TREND_FEATURES,
    "src/model_training.py": tuple(DERIVED_FEATURE_SOURCES),
    "scripts/rebuild_2026_08_phase1_split.py": (
        "split_assignment",
        "exclusion_reason",
        "pipeline_version",
        TEST_PROVISIONAL_COLUMN,
    ),
}
UPSTREAM_MODULE = "src/feature_engineering.py"


def _builder_module(feature: str) -> str:
    for module, columns in BUILDER_MODULES.items():
        if feature in columns:
            return module
    return UPSTREAM_MODULE


def _union_features() -> list[str]:
    m1 = resolve_feature_contract(M1_CONTRACT)
    m2 = resolve_feature_contract(M2_CONTRACT)
    if not set(m1.features).issubset(m2.features):
        raise AssertionError(
            f"{M1_CONTRACT} is not a subset of {M2_CONTRACT}; the union is not "
            "the larger contract and this assembly's assumption is void"
        )
    union = list(m2.features)
    if DEAD_FEATURE in union:
        raise AssertionError(
            f"{DEAD_FEATURE} is in the union contract; it is excluded by decision"
        )
    return union


def _build_manifest(columns_by_split: dict[str, list[str]]) -> list[dict[str, object]]:
    """One row per produced column, tying it to the contracts that consume it.

    Built from the union of all three splits, not from TRAIN alone: TEST carries
    ``test_provisional_20251_only``, which the other two do not, and a manifest
    that omitted it would not account for every produced column.
    """
    m1 = set(resolve_feature_contract(M1_CONTRACT).features)
    m2 = set(resolve_feature_contract(M2_CONTRACT).features)
    union = set(_union_features())

    ordered: list[str] = []
    for split in REBUILD_SPLITS:
        for name in columns_by_split[split]:
            if name not in ordered:
                ordered.append(name)
    # The derived feature is never a parquet column; list it so the manifest
    # accounts for every contract feature, not only the persisted ones.
    for name in _union_features():
        if name not in ordered:
            ordered.append(name)

    rows: list[dict[str, object]] = []
    for name in ordered:
        if name in union:
            role = (
                "derived_at_training"
                if name in DERIVED_FEATURE_SOURCES
                else "model_feature"
            )
        elif name == TEST_PROVISIONAL_COLUMN:
            role = "test_policy_flag"
        elif name in set(DERIVED_FEATURE_SOURCES.values()):
            role = "derivation_source"
        elif name == TARGET_GRADE:
            role = "target"
        elif name in SEGMENT_ONLY_COLUMNS:
            role = "segment_only"
        elif name in IDENTITY_COLUMNS:
            role = "identity"
        else:
            role = "audit_only"
        rows.append(
            {
                "feature_name": name,
                "in_m1_baseline_41": name in m1,
                "in_m2_concurrent_43": name in m2,
                "builder_module": _builder_module(name),
                "role": role,
                "present_in_splits": "|".join(
                    split
                    for split in REBUILD_SPLITS
                    if name in columns_by_split[split]
                )
                or "derived_at_training",
            }
        )
    return rows


def _assemble_split(
    source_path: Path,
    output_path: Path,
    split: str,
    union: list[str],
    *,
    batch_size: int = 25_000,
) -> dict[str, object]:
    """Drop the dead feature, stamp metadata, and stream everything else through."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output_path}")
    parquet_file = pq.ParquetFile(source_path)
    source_names = list(parquet_file.schema_arrow.names)

    if DEAD_FEATURE not in source_names:
        raise AssertionError(
            f"{source_path.name}: {DEAD_FEATURE} absent from the concurrent "
            "output; the builder contract changed and this drop is now wrong"
        )
    keep = [name for name in source_names if name != DEAD_FEATURE]

    missing = [
        feature
        for feature in union
        if feature not in keep and feature not in DERIVED_FEATURE_SOURCES
    ]
    if missing:
        raise AssertionError(f"{split}: union contract columns not built: {missing}")
    for source_column in DERIVED_FEATURE_SOURCES.values():
        if source_column not in keep:
            raise AssertionError(
                f"{split}: derivation source {source_column} missing; "
                "requirement_size_bucket_ord could not be derived at training time"
            )
    if TARGET_GRADE not in keep:
        raise AssertionError(f"{split}: target {TARGET_GRADE} missing")
    missing_difficulty = [c for c in DIFFICULTY_CONTRACT if c not in keep]
    if missing_difficulty:
        raise AssertionError(f"{split}: difficulty contract incomplete: {missing_difficulty}")

    is_test = split == "test"
    if is_test and TEST_PROVISIONAL_COLUMN not in keep:
        raise AssertionError(
            f"{split}: {TEST_PROVISIONAL_COLUMN} column missing from a provisional split"
        )

    metadata = {
        b"rebuild_version": REBUILD_VERSION.encode(),
        b"split": split.encode(),
        b"m1_contract": M1_CONTRACT.encode(),
        b"m2_contract": M2_CONTRACT.encode(),
        b"union_feature_count": str(len(union)).encode(),
        b"excluded_dead_feature": DEAD_FEATURE.encode(),
        b"lineage_applied": b"false" if not LINEAGE_APPLIED else b"true",
        b"lineage_reason": LINEAGE_REASON.encode(),
        b"identity_columns": ",".join(IDENTITY_COLUMNS).encode(),
        TEST_PROVISIONAL_COLUMN.encode(): b"true" if is_test else b"false",
    }

    writer: pq.ParquetWriter | None = None
    offset = 0
    output_schema: pa.Schema | None = None
    try:
        for record_batch in parquet_file.iter_batches(
            batch_size=batch_size, columns=keep
        ):
            table = pa.Table.from_batches([record_batch])
            if writer is None:
                existing = table.schema.metadata or {}
                output_schema = table.schema.with_metadata({**existing, **metadata})
                writer = pq.ParquetWriter(
                    output_path, output_schema, compression="snappy"
                )
            writer.write_table(table.cast(output_schema))
            offset += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    return {
        "source": str(source_path),
        "output": str(output_path),
        "rows": int(offset),
        "source_columns": len(source_names),
        "output_columns": len(keep),
        "dropped": [DEAD_FEATURE],
    }


def _verify_split(
    source_path: Path,
    output_path: Path,
    split: str,
    union: list[str],
) -> dict[str, object]:
    """Read the dataset back and re-check every claim made about it."""
    source = pq.ParquetFile(source_path)
    output = pq.ParquetFile(output_path)
    names = list(output.schema_arrow.names)

    if source.metadata.num_rows != output.metadata.num_rows:
        raise AssertionError(f"{split}: row count changed during assembly")
    if DEAD_FEATURE in names:
        raise AssertionError(f"{split}: {DEAD_FEATURE} survived into the dataset")
    dropped = [n for n in source.schema_arrow.names if n not in names]
    if dropped != [DEAD_FEATURE]:
        raise AssertionError(f"{split}: unexpected columns dropped: {dropped}")

    built_difficulty = [c for c in names if c in DIFFICULTY_CONTRACT]
    if sorted(built_difficulty) != sorted(DIFFICULTY_CONTRACT):
        raise AssertionError(
            f"{split}: difficulty columns are not exactly the imported contract: "
            f"{sorted(built_difficulty)}"
        )

    # Original identity, byte for byte against the Phase 1 split.
    identity_ok = {}
    for column in IDENTITY_COLUMNS:
        before = pq.read_table(source_path, columns=[column]).column(0).to_pylist()
        after = pq.read_table(output_path, columns=[column]).column(0).to_pylist()
        identity_ok[column] = before == after
    if not all(identity_ok.values()):
        raise AssertionError(f"{split}: identity columns changed: {identity_ok}")

    metadata = {
        key.decode(): value.decode()
        for key, value in (output.schema_arrow.metadata or {}).items()
        if not key.decode().startswith("pandas")
    }
    if metadata.get("lineage_applied") != "false":
        raise AssertionError(f"{split}: lineage_applied not recorded as false")
    if metadata.get("lineage_reason") != LINEAGE_REASON:
        raise AssertionError(f"{split}: lineage_reason not recorded")
    expected_provisional = "true" if split == "test" else "false"
    if metadata.get(TEST_PROVISIONAL_COLUMN) != expected_provisional:
        raise AssertionError(
            f"{split}: {TEST_PROVISIONAL_COLUMN} metadata is "
            f"{metadata.get(TEST_PROVISIONAL_COLUMN)!r}, expected {expected_provisional!r}"
        )

    persisted_union = [f for f in union if f in names]
    derived = [f for f in union if f in DERIVED_FEATURE_SOURCES]
    return {
        "rows": int(output.metadata.num_rows),
        "columns": len(names),
        "union_features_persisted": len(persisted_union),
        "union_features_derived_at_training": derived,
        "difficulty_columns_exactly_contract": True,
        "identity_columns_unchanged": identity_ok,
        f"{DEAD_FEATURE}_absent": True,
        "file_metadata": metadata,
        "test_provisional_column_present": TEST_PROVISIONAL_COLUMN in names,
    }


def assemble(args: argparse.Namespace) -> Path:
    root = Path(args.rebuild_root)
    union = _union_features()

    sources = {
        split: rebuild_split_path(
            root, split, "final", stage="concurrent", must_exist=True
        )
        for split in REBUILD_SPLITS
    }
    outputs = {split: rebuild_dataset_path(root, split) for split in REBUILD_SPLITS}
    existing = [str(p) for p in outputs.values() if p.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite dataset files: {existing}")
    outputs["train"].parent.mkdir(parents=True, exist_ok=True)

    splits_report: dict[str, object] = {}
    for split in REBUILD_SPLITS:
        write_report = _assemble_split(sources[split], outputs[split], split, union)
        verify_report = _verify_split(sources[split], outputs[split], split, union)
        splits_report[split] = {**write_report, "readback": verify_report}
        print(
            f"{split:6s} rows={write_report['rows']:>7,} "
            f"cols {write_report['source_columns']} -> {write_report['output_columns']} "
            f"(dropped {DEAD_FEATURE})"
        )

    columns_by_split = {
        split: list(pq.ParquetFile(outputs[split]).schema_arrow.names)
        for split in REBUILD_SPLITS
    }
    manifest = _build_manifest(columns_by_split)
    manifest_path = outputs["train"].parent / "feature_manifest.csv"
    with open(manifest_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "feature_name",
                "in_m1_baseline_41",
                "in_m2_concurrent_43",
                "builder_module",
                "role",
                "present_in_splits",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)

    listed = {row["feature_name"] for row in manifest}
    unlisted = sorted(
        {name for names in columns_by_split.values() for name in names} - listed
    )
    if unlisted:
        raise AssertionError(f"Columns absent from the manifest: {unlisted}")
    listed_union = {
        row["feature_name"] for row in manifest if row["in_m2_concurrent_43"]
    }
    if listed_union != set(union):
        raise AssertionError("Manifest contract membership disagrees with the contracts")

    report = {
        "artifact": "rebuild_phase3_dataset",
        "rebuild_version": REBUILD_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "contracts": {
            "m1": M1_CONTRACT,
            "m2": M2_CONTRACT,
            "m1_is_subset_of_m2": True,
            "union": union,
            "union_feature_count": len(union),
            "excluded_dead_feature": DEAD_FEATURE,
            "excluded_dead_feature_absent": True,
        },
        "difficulty_contract": {
            "source": "src/course_difficulty.py",
            "columns": list(DIFFICULTY_CONTRACT),
            "count": len(DIFFICULTY_CONTRACT),
            "restated_in_this_script": False,
        },
        "identity": {
            "columns": IDENTITY_COLUMNS,
            "lineage_applied": LINEAGE_APPLIED,
            "lineage_reason": LINEAGE_REASON,
        },
        "test_policy": {
            TEST_PROVISIONAL_COLUMN: True,
            "note": "TEST is built from 20251 only; 20252 excluded as incomplete",
            "test_outcomes_read": False,
        },
        "manifest": str(manifest_path),
        "manifest_rows": len(manifest),
        "splits": splits_report,
        "model_trained": False,
        "model_evaluated": False,
        "version_promoted": False,
    }
    report_path = outputs["train"].parent / "phase3_dataset_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Manifest: {manifest_path} ({len(manifest)} rows)")
    print(f"Report  : {report_path}")
    return report_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild-root",
        type=Path,
        default=MODEL_DATA_VERSIONS_DIR / REBUILD_VERSION,
        help="Version root holding 04_concurrent; 05_dataset is written under it.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    assemble(parse_args(argv))
    print(
        "Dataset assembled. No model trained, no model evaluated, "
        "no version promoted."
    )


if __name__ == "__main__":
    main()
