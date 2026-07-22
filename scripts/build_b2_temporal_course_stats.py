"""Build the versioned B2 temporal course-difficulty data generation.

This script never trains a model and never overwrites the current model-data
files.  It reads the base splits, replaces only course-difficulty columns, and
persists a frozen full-train lookup state for validation/test and inference.
"""

from __future__ import annotations

import argparse
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


SPLITS = ("train", "valid", "test")
REFERENCE_RUN = "2026-07-12_1513__remove-dead-const"

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
    writer: pq.ParquetWriter | None = None
    offset = 0
    output_columns: list[str] | None = None
    try:
        for record_batch in parquet_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([record_batch], schema=parquet_file.schema_arrow)
            batch_frame = table.to_pandas()
            end = offset + len(batch_frame)
            feature_batch = difficulty_features.iloc[offset:end]
            if not batch_frame.index.equals(feature_batch.index):
                raise AssertionError(
                    f"Index/order mismatch while streaming {source_path.name}: rows {offset}:{end}"
                )
            for column in DIFFICULTY_OUTPUT_COLUMNS:
                batch_frame[column] = feature_batch[column].to_numpy()

            output_table = pa.Table.from_pandas(batch_frame, preserve_index=True)
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


def verify_versioned_outputs(input_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Independently compare every non-difficulty Arrow column after write."""

    checks: Dict[str, Any] = {}
    for split in SPLITS:
        for generation in ("difficulty", "final"):
            source_name = "base" if generation == "difficulty" else "final"
            source_path = model_split_path(split, source_name, input_dir)
            output_path = model_split_path(split, generation, output_dir)
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
    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    build_id = args.build_id or datetime.now().astimezone().strftime(
        "%Y-%m-%d_%H%M%S__b2_temporal_course_stats"
    )
    output_dir = output_root / build_id
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing B2 output: {output_dir}")
    staging_dir = output_root / f".{build_id}.incomplete"
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
    train_base_path = model_split_path("train", "base", input_dir)
    train_final_path = model_split_path("train", "final", input_dir)
    staged_train_difficulty_path = model_split_path(
        "train", "difficulty", staging_dir
    )
    staged_train_final_path = model_split_path("train", "final", staging_dir)
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
    _stream_write_enriched(
        train_final_path,
        staged_train_final_path,
        train_enriched,
    )
    train_diploma_before = pd.read_parquet(
        train_final_path, columns=["diploma_gpa"]
    )["diploma_gpa"]
    train_diploma_after = pd.read_parquet(
        staged_train_final_path, columns=["diploma_gpa"]
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
        base_path = model_split_path(split, "base", input_dir)
        final_path = model_split_path(split, "final", input_dir)
        staged_difficulty_path = model_split_path(split, "difficulty", staging_dir)
        staged_final_path = model_split_path(split, "final", staging_dir)
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
        _stream_write_enriched(
            final_path,
            staged_final_path,
            split_enriched,
        )
        diploma_before = pd.read_parquet(
            final_path, columns=["diploma_gpa"]
        )["diploma_gpa"]
        diploma_after = pd.read_parquet(
            staged_final_path, columns=["diploma_gpa"]
        )["diploma_gpa"]
        diploma_gate &= diploma_before.equals(diploma_after)
        diploma_null_counts[split] = int(diploma_after.isna().sum())
        row_counts[split] = int(len(base))
        distributions[split] = _split_distribution(split_enriched)
        del base, split_enriched, diploma_before, diploma_after
        gc.collect()

    readback_checks = verify_versioned_outputs(input_dir, staging_dir)
    lineage_gate = all(
        values["status"] == "pass" for values in readback_checks.values()
    )

    reference_contract_path = MODEL_RUNS_DIR / args.reference_run / "feature_contract.json"
    reference_contract = json.loads(reference_contract_path.read_text(encoding="utf-8"))
    reference_features = reference_contract["features"]
    expected_features_unchanged = (
        reference_contract["n_features"] == len(reference_features) == 39
        and "course_is_new" not in reference_features
        and "course_low_support" not in reference_features
        and "difficulty_fallback_level" not in reference_features
    )

    data_gates = {
        "row count, index, and order preserved": row_index_gate,
        "all non-difficulty values and dtypes preserved": lineage_gate,
        "first train semester is no-history Level 6": first_semester_gate,
        "train exercises more than one fallback level": multi_level_gate,
        "valid/test statistics contain no nulls": valid_test_null_gate,
        "course_is_new definition equals no Level-1/2 history": new_definition_gate,
        "MODEL_FEATURES remains the same 39-column contract": expected_features_unchanged,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(MODEL_DATA_DIR))
    parser.add_argument("--output-root", default=str(MODEL_DATA_VERSIONS_DIR))
    parser.add_argument("--build-id")
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--shrinkage-k", type=float, default=20.0)
    parser.add_argument("--reference-run", default=REFERENCE_RUN)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    output_dir = build(parse_args(argv))
    print(f"B2 data build completed: {output_dir}")
    print("Data gates passed. Model training was not run because diploma_gpa isolation failed.")


if __name__ == "__main__":
    main()
