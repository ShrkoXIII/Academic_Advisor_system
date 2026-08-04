"""Build the versioned B2 temporal course-difficulty data generation.

This script never trains a model and never overwrites the current model-data
files.  It reads the base splits, replaces only course-difficulty columns, and
persists a frozen full-train lookup state for validation/test and inference.

Path modes
----------

* **Legacy (default).**  ``--input-dir`` and ``--output-root``/``--build-id``
  behave exactly as before: filenames come from
  :func:`src.paths.model_split_path`, and the build directory is published by
  renaming its staging directory.
* **Explicit.**  Every input and output parquet can be named in full on the
  command line.  The six output paths must share one parent directory, which is
  the build directory; that keeps the staging-then-rename publish atomic.
* **Rebuild.**  ``--rebuild-root`` fills those explicit paths in from
  :mod:`src.rebuild_paths` for ``2026-08_temporal_rebuild_v1``.

The ``final`` generation is optional.  Phase 1 of the rebuild produced ``base``
only; the ``final`` templates do not exist until the diploma bucketing step has
run.  When no ``final`` templates are supplied, this script writes the
``difficulty`` generation alone and checks ``diploma_gpa`` isolation against the
``difficulty`` output instead.  ``final`` templates are all-or-nothing across the
three splits, and under ``--rebuild-root`` they are used only when passed
explicitly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import gc
import json
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.course_difficulty import (  # noqa: E402
    DIFFICULTY_OUTPUT_COLUMNS,
    LEVEL_LABELS,
    STAT_OUTPUT_COLUMNS,
    DifficultyConfig,
    apply_difficulty_state,
    build_temporal_train,
    fit_difficulty_state,
    save_difficulty_state,
)
from src.paths import (  # noqa: E402
    MODEL_DATA_DIR,
    MODEL_DATA_VERSIONS_DIR,
    MODEL_RUNS_DIR,
    model_split_path,
)
from src.rebuild_paths import rebuild_split_path  # noqa: E402
from src.model_training import resolve_feature_contract  # noqa: E402


SPLITS = ("train", "valid", "test")
REFERENCE_RUN = "2026-07-16_1025__new-difficulty-logic"

# The contract this build's outputs are meant to feed. concurrent_44 is archived
# (CLAUDE.md section 4) and is deliberately not accepted here.
B2_ALLOWED_CONTRACTS = ("baseline_41", "concurrent_43")

# Difficulty columns B2 computes that must stay out of any model contract.
DIFFICULTY_AUDIT_ONLY_COLUMNS = (
    "course_is_new",
    "course_low_support",
    "difficulty_fallback_level",
)

GPA_TREND_FEATURES = ("gpa_trend_delta", "gpa_trend_missing")

DIFFICULTY_INPUT_COLUMNS = [
    "part_id",
    "part_year",
    "final_mark",
    "attempt_number",
    "degree_course_key",
    "degree_id",
    "faculty_id",
    "requirement_type_id",
    "course_credits",
]


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if np.isnan(value) else value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def stores_pandas_index(parquet_file: pq.ParquetFile) -> bool:
    """Whether the parquet file materializes a pandas index column.

    The live splits were written with ``preserve_index=True`` and carry
    ``__index_level_0__``; the rebuild's Phase 1 splits were not and carry no
    index at all. That difference decides how a streamed batch can be checked
    against a whole-file feature frame, because a file without a stored index
    hands every batch a fresh ``RangeIndex`` starting at zero.
    """
    return any(
        field.startswith("__index_level_") for field in parquet_file.schema_arrow.names
    )


def assert_batch_alignment(
    batch_frame: pd.DataFrame,
    feature_batch: pd.DataFrame,
    *,
    source_path: Path,
    offset: int,
    end: int,
    source_has_index: bool,
) -> None:
    """Refuse to write unless the batch and the feature slice are the same rows.

    With a stored index the two indexes must match outright, which is the
    strongest available check and the one the live splits have always used.
    Without one, the file offers no index to compare, so the check is that
    pandas handed back the untouched positional index for this batch - i.e.
    nothing was reordered - and alignment is by position, which is exact
    because both sides read the same file in file order.
    """
    if len(batch_frame) != len(feature_batch):
        raise AssertionError(
            f"Row-count mismatch while streaming {source_path.name}: "
            f"rows {offset}:{end}"
        )
    if source_has_index:
        if not batch_frame.index.equals(feature_batch.index):
            raise AssertionError(
                f"Index/order mismatch while streaming {source_path.name}: rows {offset}:{end}"
            )
        return
    if not batch_frame.index.equals(pd.RangeIndex(len(batch_frame))):
        raise AssertionError(
            f"Unindexed source {source_path.name} returned a non-positional batch "
            f"index at rows {offset}:{end}; refusing to align by position"
        )


def _stream_write_enriched(
    source_path: Path,
    output_path: Path,
    difficulty_features: pd.DataFrame,
    *,
    batch_size: int = 25_000,
) -> Dict[str, Any]:
    """Replace/append difficulty columns while streaming all other columns unchanged."""

    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite output parquet: {output_path}")
    parquet_file = pq.ParquetFile(source_path)
    # Mirror the source: fabricating an index for a file that has none would
    # write a counter that restarts on every batch.
    source_has_index = stores_pandas_index(parquet_file)
    writer: pq.ParquetWriter | None = None
    offset = 0
    output_columns: list[str] | None = None
    try:
        for record_batch in parquet_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([record_batch], schema=parquet_file.schema_arrow)
            batch_frame = table.to_pandas()
            end = offset + len(batch_frame)
            feature_batch = difficulty_features.iloc[offset:end]
            assert_batch_alignment(
                batch_frame,
                feature_batch,
                source_path=source_path,
                offset=offset,
                end=end,
                source_has_index=source_has_index,
            )
            for column in DIFFICULTY_OUTPUT_COLUMNS:
                batch_frame[column] = feature_batch[column].to_numpy()

            output_table = pa.Table.from_pandas(
                batch_frame, preserve_index=source_has_index
            )
            if writer is None:
                writer = pq.ParquetWriter(output_path, output_table.schema, compression="snappy")
                output_columns = list(batch_frame.columns)
            writer.write_table(output_table)
            offset = end
    finally:
        if writer is not None:
            writer.close()

    if offset != len(difficulty_features):
        raise AssertionError(
            f"Row count mismatch for {source_path.name}: source={offset}, features={len(difficulty_features)}"
        )
    return {
        "rows": int(offset),
        "columns": output_columns or [],
        "source": str(source_path),
        "output": str(output_path),
    }


@dataclass(frozen=True)
class IOPlan:
    """Fully resolved input/output parquet paths for one B2 build.

    ``inputs``/``outputs`` are keyed ``{generation: {split: path}}``. The
    ``final`` generation is absent from both when no ``final`` templates were
    supplied. Every output shares ``output_dir`` as its parent so the build can
    still be published by renaming a single staging directory.
    """

    mode: str
    output_dir: Path
    inputs: Dict[str, Dict[str, Path]]
    outputs: Dict[str, Dict[str, Path]]

    @property
    def has_final(self) -> bool:
        return "final" in self.outputs

    def source_for(self, split: str, generation: str) -> Path | None:
        """The template a given output generation was streamed from."""
        source_generation = "base" if generation == "difficulty" else "final"
        return self.inputs.get(source_generation, {}).get(split)

    @staticmethod
    def staged(output_path: Path, staging_dir: Path) -> Path:
        """Where an output is written before the staging directory is published."""
        return staging_dir / output_path.name

    def required_inputs(self) -> list[Path]:
        return [
            path
            for generation in self.inputs
            for path in self.inputs[generation].values()
        ]


def _plan_io(args: argparse.Namespace, output_dir: Path) -> IOPlan:
    """Resolve every input/output path: explicit flag, then rebuild root, then legacy."""

    rebuild_root = getattr(args, "rebuild_root", None)
    input_dir = Path(args.input_dir)

    def explicit(prefix: str, split: str) -> Path | None:
        value = getattr(args, f"{split}_{prefix}", None)
        return Path(value) if value else None

    inputs: Dict[str, Dict[str, Path]] = {"base": {}}
    outputs: Dict[str, Dict[str, Path]] = {"difficulty": {}}

    for split in SPLITS:
        base = explicit("base", split)
        if base is None:
            base = (
                rebuild_split_path(rebuild_root, split, "base", must_exist=True)
                if rebuild_root
                else model_split_path(split, "base", input_dir)
            )
        inputs["base"][split] = Path(base)

        difficulty_out = explicit("difficulty_out", split)
        if difficulty_out is None:
            difficulty_out = (
                rebuild_split_path(rebuild_root, split, "difficulty").name
                if rebuild_root
                else model_split_path(split, "difficulty", input_dir).name
            )
            difficulty_out = output_dir / difficulty_out
        outputs["difficulty"][split] = Path(difficulty_out)

    # The final generation is all-or-nothing. Under --rebuild-root it is used
    # only when named explicitly: Phase 1 produced base only, so silently
    # resolving a final template that does not exist would be a lie.
    explicit_final = {split: explicit("final", split) for split in SPLITS}
    if any(explicit_final.values()) and not all(explicit_final.values()):
        missing = sorted(split for split, path in explicit_final.items() if not path)
        raise ValueError(
            "final templates are all-or-nothing across splits; missing: "
            f"{missing}"
        )
    if all(explicit_final.values()):
        final_inputs = {split: Path(path) for split, path in explicit_final.items()}
    elif rebuild_root:
        final_inputs = {}
    else:
        final_inputs = {
            split: model_split_path(split, "final", input_dir) for split in SPLITS
        }

    if final_inputs:
        inputs["final"] = final_inputs
        outputs["final"] = {}
        for split in SPLITS:
            final_out = explicit("final_out", split)
            if final_out is None:
                final_out = (
                    rebuild_split_path(rebuild_root, split, "final").name
                    if rebuild_root
                    else model_split_path(split, "final", input_dir).name
                )
                final_out = output_dir / final_out
            outputs["final"][split] = Path(final_out)

    stray = sorted(
        str(path)
        for generation in outputs
        for path in outputs[generation].values()
        if path.parent != output_dir
    )
    if stray:
        raise ValueError(
            "every B2 output must sit directly in the build directory "
            f"{output_dir}; these do not: {stray}"
        )
    basenames = [
        path.name for generation in outputs for path in outputs[generation].values()
    ]
    if len(set(basenames)) != len(basenames):
        raise ValueError(f"B2 output basenames collide: {sorted(basenames)}")

    if rebuild_root:
        mode = "rebuild_root"
    elif any(
        getattr(args, f"{split}_{prefix}", None)
        for split in SPLITS
        for prefix in ("base", "final", "difficulty_out", "final_out")
    ) or getattr(args, "output_dir", None):
        mode = "explicit_paths"
    else:
        mode = "legacy_generation_names"

    return IOPlan(
        mode=mode,
        output_dir=output_dir,
        inputs=inputs,
        outputs=outputs,
    )


def _feature_contract_gate(
    contract_name: str,
    reference_features: list[str],
) -> tuple[bool, Dict[str, Any]]:
    """Verify the contract that is actually named, not a hard-coded count.

    The predecessor of this gate asserted ``EXPECTED_FEATURE_COUNT ==
    len(MODEL_FEATURES) == 41``. Those deprecated globals now alias
    ``concurrent_44``, so the clause is unsatisfiable and B2 could not run at
    all. The checks below preserve every claim the old gate actually made -
    nothing was dropped relative to the reference run, the GPA-trend pair is the
    addition, and B2's audit-only difficulty columns never enter a contract -
    while reading them off the named contract.
    """
    contract = resolve_feature_contract(contract_name)
    features = list(contract.features)
    declared_count = int(contract.name.rsplit("_", 1)[-1])
    reference = set(reference_features)

    checks = {
        "contract_is_accepted_for_b2": contract.name in B2_ALLOWED_CONTRACTS,
        "feature_count_matches_contract_name": (
            len(features) == contract.expected_feature_count == declared_count
        ),
        "no_duplicate_features": len(set(features)) == len(features),
        "reference_run_features_all_retained": not reference.difference(features),
        "gpa_trend_pair_is_present": set(GPA_TREND_FEATURES).issubset(features),
        "gpa_trend_pair_is_added_over_reference": set(GPA_TREND_FEATURES).issubset(
            set(features).difference(reference)
        ),
        "difficulty_audit_columns_stay_out_of_the_contract": not set(
            DIFFICULTY_AUDIT_ONLY_COLUMNS
        ).intersection(features),
    }
    detail = {
        "contract": contract.name,
        "contract_version": contract.version,
        "feature_count": len(features),
        "declared_count_from_name": declared_count,
        "reference_feature_count": len(reference_features),
        "added_over_reference": sorted(set(features).difference(reference)),
        "removed_versus_reference": sorted(reference.difference(features)),
        "checks": checks,
    }
    return all(checks.values()), detail


def verify_versioned_outputs(
    io_plan: "IOPlan",
    staging_dir: Path,
) -> Dict[str, Any]:
    """Independently compare every non-difficulty Arrow column after write."""

    checks: Dict[str, Any] = {}
    for split in SPLITS:
        for generation in ("difficulty", "final"):
            source_path = io_plan.source_for(split, generation)
            if source_path is None:
                continue
            output_path = io_plan.staged(io_plan.outputs[generation][split], staging_dir)
            source_file = pq.ParquetFile(source_path)
            output_file = pq.ParquetFile(output_path)
            if source_file.metadata.num_rows != output_file.metadata.num_rows:
                raise AssertionError(f"{split}/{generation}: parquet row count changed")

            source_names = source_file.schema_arrow.names
            output_names = output_file.schema_arrow.names
            compare_names = [
                name
                for name in source_names
                if name not in DIFFICULTY_OUTPUT_COLUMNS
            ]
            missing = [name for name in compare_names if name not in output_names]
            if missing:
                raise AssertionError(f"{split}/{generation}: missing source columns {missing}")

            for column in compare_names:
                source_array = pq.read_table(source_path, columns=[column]).column(0).combine_chunks()
                output_array = pq.read_table(output_path, columns=[column]).column(0).combine_chunks()
                if source_array.type != output_array.type:
                    try:
                        output_array = output_array.cast(source_array.type)
                    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
                        raise AssertionError(
                            f"{split}/{generation}: incompatible type change for {column}: "
                            f"{source_array.type} -> {output_array.type}"
                        ) from exc
                if not source_array.equals(output_array):
                    raise AssertionError(
                        f"{split}/{generation}: non-difficulty column changed: {column}"
                    )

            checks[f"{split}_{generation}"] = {
                "rows": int(source_file.metadata.num_rows),
                "source_columns_compared": int(len(compare_names)),
                "output_columns": int(len(output_names) - (1 if "__index_level_0__" in output_names else 0)),
                "status": "pass",
            }
    return checks


def _counts(series: pd.Series) -> Dict[str, Dict[str, float | int]]:
    total = len(series)
    return {
        str(key): {"count": int(value), "pct": round(float(value / total * 100), 4)}
        for key, value in series.value_counts(dropna=False).sort_index().items()
    }


def _split_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    fallback = _counts(df["difficulty_fallback_level"])
    for level, values in fallback.items():
        values["label"] = LEVEL_LABELS[int(level)]
    return {
        "rows": int(len(df)),
        "fallback_levels": fallback,
        "course_is_new": _counts(df["course_is_new"]),
        "course_low_support": _counts(df["course_low_support"]),
        "course_difficulty_missing": _counts(df["course_difficulty_missing"]),
        "coverage": {
            "covered": int((df["course_is_new"] == 0).sum()),
            "uncovered": int((df["course_is_new"] == 1).sum()),
            "covered_pct": round(float((df["course_is_new"] == 0).mean() * 100), 4),
        },
        "course_history_count": {
            key: _json_value(value)
            for key, value in df["course_history_count"].describe().to_dict().items()
        },
        "difficulty_group_support_count": {
            key: _json_value(value)
            for key, value in df["difficulty_group_support_count"].describe().to_dict().items()
        },
        "stat_null_counts": {
            column: int(df[column].isna().sum()) for column in STAT_OUTPUT_COLUMNS
        },
    }


def _year_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    grouped = (
        df.groupby(["part_year", "difficulty_fallback_level"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    result: Dict[str, Any] = {}
    for year, part in grouped.groupby("part_year"):
        year_total = int(part["count"].sum())
        year_values: Dict[str, Any] = {}
        for _, row in part.iterrows():
            count = int(row["count"])
            year_values[str(int(row["difficulty_fallback_level"]))] = {
                "count": count,
                "pct": round(float(count / year_total * 100), 4),
            }
        result[str(int(year))] = year_values
    return result


def _report_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# B2 temporal course statistics — data report",
        "",
        f"- Build: `{report['build_id']}`",
        f"- Reference run: `{report['reference_run']}`",
        f"- Train semesters: {report['time_contract']['n_train_semesters']}",
        "- Train rule: each semester uses only strictly earlier `part_id` values.",
        "- Valid/test rule: both use the same state fitted on complete train; test never uses valid.",
        "- Fallback levels: 6 (faculty Level 4 retained).",
        f"- Shrinkage: k={report['config']['shrinkage_k']}, toward the direct structural parent.",
        "",
        "## Data gates",
        "",
    ]
    for gate, passed in report["data_gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {gate}")

    lines.extend(["", "## Distributions", ""])
    for split in SPLITS:
        values = report["distributions"][split]
        lines.append(f"### {split} ({values['rows']:,} rows)")
        lines.append("")
        for level, item in values["fallback_levels"].items():
            lines.append(
                f"- Level {level}: {item['count']:,} ({item['pct']:.2f}%) — {item['label']}"
            )
        lines.append(
            f"- New courses: {values['course_is_new'].get('1', {'count': 0})['count']:,}; "
            f"low support: {values['course_low_support'].get('1', {'count': 0})['count']:,}; "
            f"difficulty missing: {values['course_difficulty_missing'].get('1', {'count': 0})['count']:,}."
        )
        lines.append("")

    diploma = report["diploma_gpa_isolation"]
    lines.extend(
        [
            "## diploma_gpa isolation",
            "",
            "- The B2 outputs preserve `diploma_gpa` exactly, including nulls.",
            f"- Nulls: train={diploma['null_counts']['train']}, "
            f"valid={diploma['null_counts']['valid']}, test={diploma['null_counts']['test']}.",
            f"- Reference preprocessing match: **{diploma['reference_preprocessing_match']}**.",
            f"- Training status: **{diploma['training_status']}**.",
            f"- Evidence: {diploma['evidence']}",
            "",
            "No model was trained by this script. B3 was not started.",
        ]
    )
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root)
    build_id = args.build_id or datetime.now().astimezone().strftime(
        "%Y-%m-%d_%H%M%S__b2_temporal_course_stats"
    )
    if getattr(args, "output_dir", None):
        output_dir = Path(args.output_dir)
    elif getattr(args, "rebuild_root", None):
        output_dir = rebuild_split_path(
            args.rebuild_root, "train", "difficulty"
        ).parent
    else:
        output_dir = output_root / build_id
    io_plan = _plan_io(args, output_dir)

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing B2 output: {output_dir}")
    staging_dir = output_dir.parent / f".{output_dir.name}.incomplete"
    if staging_dir.exists():
        raise FileExistsError(f"Refusing to overwrite incomplete B2 output: {staging_dir}")
    staging_dir.mkdir(parents=True)

    config = DifficultyConfig(
        min_support=args.min_support,
        shrinkage_k=args.shrinkage_k,
    )

    row_counts: Dict[str, int] = {}
    distributions: Dict[str, Any] = {}
    diploma_null_counts: Dict[str, int] = {}
    row_index_gate = True
    lineage_gate = True
    new_definition_gate = True
    diploma_gate = True
    valid_test_null_gate = True

    # Train is the largest split. Process and persist it before loading valid
    # or test so peak memory is bounded to one base/final pair.
    train_base_path = io_plan.inputs["base"]["train"]
    staged_train_difficulty_path = IOPlan.staged(
        io_plan.outputs["difficulty"]["train"], staging_dir
    )
    train_final_path = (
        io_plan.inputs["final"]["train"] if io_plan.has_final else None
    )
    staged_train_final_path = (
        IOPlan.staged(io_plan.outputs["final"]["train"], staging_dir)
        if io_plan.has_final
        else None
    )
    train_base = pd.read_parquet(train_base_path, columns=DIFFICULTY_INPUT_COLUMNS)
    train_enriched = build_temporal_train(train_base, config, include_source=False)
    full_train_state = fit_difficulty_state(train_base, config)
    if not train_base.index.equals(train_enriched.index):
        raise AssertionError("train: B2 feature index/order changed")

    part_numeric = pd.to_numeric(train_base["part_id"], errors="raise")
    first_part = part_numeric.min()
    first_rows = train_enriched.loc[part_numeric == first_part]
    first_semester_gate = bool(
        (first_rows["difficulty_fallback_level"] == 6).all()
        and (first_rows["course_history_count"] == 0).all()
        and (first_rows["difficulty_group_support_count"] == 0).all()
        and first_rows[STAT_OUTPUT_COLUMNS].isna().all().all()
    )
    multi_level_gate = bool(train_enriched["difficulty_fallback_level"].nunique() > 1)
    new_definition_gate &= bool(
        (
            train_enriched["course_is_new"]
            == (train_enriched["difficulty_fallback_level"] >= 3).astype("int64")
        ).all()
    )

    _stream_write_enriched(
        train_base_path,
        staged_train_difficulty_path,
        train_enriched,
    )
    if io_plan.has_final:
        _stream_write_enriched(
            train_final_path,
            staged_train_final_path,
            train_enriched,
        )
    # Without a final template the isolation check falls back to the difficulty
    # pair; diploma_gpa is carried unchanged through both, so the claim is the
    # same one, measured on the generation that exists.
    diploma_source_path = train_final_path if io_plan.has_final else train_base_path
    diploma_output_path = (
        staged_train_final_path if io_plan.has_final else staged_train_difficulty_path
    )
    train_diploma_before = pd.read_parquet(
        diploma_source_path, columns=["diploma_gpa"]
    )["diploma_gpa"]
    train_diploma_after = pd.read_parquet(
        diploma_output_path, columns=["diploma_gpa"]
    )["diploma_gpa"]
    diploma_gate &= train_diploma_before.equals(train_diploma_after)
    diploma_null_counts["train"] = int(train_diploma_after.isna().sum())
    row_counts["train"] = int(len(train_base))
    distributions["train"] = _split_distribution(train_enriched)
    train_fallback_by_year = _year_distribution(
        pd.concat([train_base[["part_year"]], train_enriched], axis=1, copy=False)
    )
    save_difficulty_state(
        full_train_state,
        staging_dir / "difficulty_state",
        metadata={
            "fit_source": str(train_base_path),
            "fit_rows": int(len(train_base)),
            "part_id_min": str(train_base.loc[part_numeric.idxmin(), "part_id"]),
            "part_id_max": str(train_base.loc[part_numeric.idxmax(), "part_id"]),
            "valid_and_test_policy": "apply frozen full-train state; never fit on valid/test",
        },
    )

    del train_base, train_enriched, train_diploma_before, train_diploma_after
    gc.collect()

    for split in ("valid", "test"):
        base_path = io_plan.inputs["base"][split]
        staged_difficulty_path = IOPlan.staged(
            io_plan.outputs["difficulty"][split], staging_dir
        )
        final_path = io_plan.inputs["final"][split] if io_plan.has_final else None
        staged_final_path = (
            IOPlan.staged(io_plan.outputs["final"][split], staging_dir)
            if io_plan.has_final
            else None
        )
        base = pd.read_parquet(base_path, columns=DIFFICULTY_INPUT_COLUMNS)
        split_enriched = apply_difficulty_state(base, full_train_state, include_source=False)
        row_index_gate &= len(base) == len(split_enriched) and base.index.equals(split_enriched.index)
        new_definition_gate &= bool(
            (
                split_enriched["course_is_new"]
                == (split_enriched["difficulty_fallback_level"] >= 3).astype("int64")
            ).all()
        )
        valid_test_null_gate &= not split_enriched[STAT_OUTPUT_COLUMNS].isna().any().any()

        _stream_write_enriched(
            base_path,
            staged_difficulty_path,
            split_enriched,
        )
        if io_plan.has_final:
            _stream_write_enriched(
                final_path,
                staged_final_path,
                split_enriched,
            )
        diploma_before = pd.read_parquet(
            final_path if io_plan.has_final else base_path, columns=["diploma_gpa"]
        )["diploma_gpa"]
        diploma_after = pd.read_parquet(
            staged_final_path if io_plan.has_final else staged_difficulty_path,
            columns=["diploma_gpa"],
        )["diploma_gpa"]
        diploma_gate &= diploma_before.equals(diploma_after)
        diploma_null_counts[split] = int(diploma_after.isna().sum())
        row_counts[split] = int(len(base))
        distributions[split] = _split_distribution(split_enriched)
        del base, split_enriched, diploma_before, diploma_after
        gc.collect()

    readback_checks = verify_versioned_outputs(io_plan, staging_dir)
    lineage_gate = all(
        values["status"] == "pass" for values in readback_checks.values()
    )

    reference_contract_path = MODEL_RUNS_DIR / args.reference_run / "feature_contract.json"
    reference_contract = json.loads(reference_contract_path.read_text(encoding="utf-8"))
    reference_features = reference_contract["features"]
    contract_gate, contract_gate_detail = _feature_contract_gate(
        args.feature_contract,
        reference_features,
    )
    contract_gate &= reference_contract["n_features"] == len(reference_features)

    data_gates = {
        "row count, index, and order preserved": row_index_gate,
        "all non-difficulty values and dtypes preserved": lineage_gate,
        "first train semester is no-history Level 6": first_semester_gate,
        "train exercises more than one fallback level": multi_level_gate,
        "valid/test statistics contain no nulls": valid_test_null_gate,
        "course_is_new definition equals no Level-1/2 history": new_definition_gate,
        (
            f"named feature contract {args.feature_contract} is intact and "
            "supersets the reference run by the GPA-trend pair"
        ): contract_gate,
        "diploma_gpa values are byte-for-value unchanged": diploma_gate,
    }
    failed = [name for name, passed in data_gates.items() if not passed]
    if failed:
        raise AssertionError(
            f"B2 data gates failed; incomplete output retained at {staging_dir}: {failed}"
        )

    report: Dict[str, Any] = {
        "build_id": build_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "reference_run": args.reference_run,
        "output_dir": str(output_dir),
        "config": {
            "min_support": config.min_support,
            "shrinkage_k": config.shrinkage_k,
            "target_threshold": config.target_threshold,
            "fallback_levels": LEVEL_LABELS,
        },
        "time_contract": {
            "order_key": "numeric part_id",
            "train_history_rule": "part_id < current part_id",
            "n_train_semesters": int(part_numeric.nunique()),
            "first_train_part_id": str(int(part_numeric.min())),
            "last_train_part_id": str(int(part_numeric.max())),
        },
        "row_counts": row_counts,
        "data_gates": data_gates,
        "feature_contract_gate": contract_gate_detail,
        "io_plan": {
            "mode": io_plan.mode,
            "final_generation_built": io_plan.has_final,
            "inputs": {
                generation: {split: str(path) for split, path in paths.items()}
                for generation, paths in io_plan.inputs.items()
            },
            "outputs": {
                generation: {split: str(path) for split, path in paths.items()}
                for generation, paths in io_plan.outputs.items()
            },
        },
        "readback_checks": readback_checks,
        "distributions": distributions,
        "train_fallback_by_part_year": train_fallback_by_year,
        "diploma_gpa_isolation": {
            "values_changed_by_b2": False,
            "null_counts": diploma_null_counts,
            # Verified against git history before this build: the temporary fill
            # was added on 2026-07-13, after the 2026-07-12 reference run.
            "reference_preprocessing_match": False,
            "training_status": "BLOCKED — reference predates temporary diploma_gpa fill",
            "evidence": (
                "git commit 9994119 (2026-07-13) added the temporary train-median fill; "
                "the reference run was created on 2026-07-12"
            ),
        },
        "training_run": {
            "requested_name": "b2_temporal_course_stats",
            "status": "not run because the mandatory diploma_gpa isolation gate failed",
            "selection_metrics": [
                "valid fail-class average precision",
                "valid AUC",
                "valid Brier",
                "train-valid AUC gap",
            ],
            "test_policy": "descriptive only",
        },
        "b3_started": False,
    }
    (staging_dir / "b2_data_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_value),
        encoding="utf-8",
    )
    (staging_dir / "REPORT.md").write_text(_report_markdown(report), encoding="utf-8")
    staging_dir.rename(output_dir)
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(MODEL_DATA_DIR))
    parser.add_argument("--output-root", default=str(MODEL_DATA_VERSIONS_DIR))
    parser.add_argument("--build-id")
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--shrinkage-k", type=float, default=20.0)
    parser.add_argument("--reference-run", default=REFERENCE_RUN)
    parser.add_argument(
        "--feature-contract",
        required=True,
        choices=list(B2_ALLOWED_CONTRACTS),
        help=(
            "Contract the outputs are built for. Checked by name; never "
            "defaulted, so a build always records which contract it claims."
        ),
    )
    parser.add_argument(
        "--rebuild-root",
        type=Path,
        help=(
            "Version root of 2026-08_temporal_rebuild_v1. Fills in the base "
            "inputs and the difficulty/final outputs from src.rebuild_paths. "
            "Final templates are never auto-resolved in this mode."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Full path of the build directory. Overrides --output-root/"
            "--build-id. Every output parquet must sit directly inside it."
        ),
    )
    for split in SPLITS:
        parser.add_argument(
            f"--{split}-base",
            type=Path,
            help=f"Full path of the {split} base template (input).",
        )
    for split in SPLITS:
        parser.add_argument(
            f"--{split}-final",
            type=Path,
            help=(
                f"Full path of the {split} final template (input). All three "
                "or none; omit them to build the difficulty generation alone."
            ),
        )
    for split in SPLITS:
        parser.add_argument(
            f"--{split}-difficulty-out",
            type=Path,
            help=f"Full path of the {split} difficulty output.",
        )
    for split in SPLITS:
        parser.add_argument(
            f"--{split}-final-out",
            type=Path,
            help=f"Full path of the {split} final output.",
        )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    output_dir = build(parse_args(argv))
    print(f"B2 data build completed: {output_dir}")
    print("Data gates passed. Model training was not run because diploma_gpa isolation failed.")


def default_namespace(**overrides: Any) -> argparse.Namespace:
    """A namespace carrying every path knob at its inactive default.

    In-process callers (the GPA-trend wrapper) construct B2 arguments directly.
    Going through here means a new path flag cannot be silently missed by them.
    """
    values: Dict[str, Any] = {
        "input_dir": str(MODEL_DATA_DIR),
        "output_root": str(MODEL_DATA_VERSIONS_DIR),
        "build_id": None,
        "min_support": 20,
        "shrinkage_k": 20.0,
        "reference_run": REFERENCE_RUN,
        "feature_contract": None,
        "rebuild_root": None,
        "output_dir": None,
    }
    for split in SPLITS:
        values[f"{split}_base"] = None
        values[f"{split}_final"] = None
        values[f"{split}_difficulty_out"] = None
        values[f"{split}_final_out"] = None
    unknown = sorted(set(overrides).difference(values))
    if unknown:
        raise TypeError(f"Unknown B2 argument(s): {unknown}")
    values.update(overrides)
    return argparse.Namespace(**values)


if __name__ == "__main__":
    main()
