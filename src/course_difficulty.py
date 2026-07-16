"""Leakage-safe temporal course-difficulty feature construction.

The six-level fallback hierarchy is the same hierarchy introduced in
``02_course_difficulty.ipynb``:

1. degree + course
2. course across degrees
3. degree + requirement type + rounded credits
4. faculty + requirement type + rounded credits
5. requirement type + rounded credits
6. all prior training history

Training rows are enriched semester by semester.  Every row in semester ``t``
uses a state fitted only on semesters strictly before ``t``.  Validation and
test rows use one frozen state fitted on the complete training split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


STAT_OUTPUT_COLUMNS = [
    "course_pass_rate_historical",
    "course_avg_mark_historical",
    "course_retake_rate_historical",
]

AUDIT_OUTPUT_COLUMNS = [
    "course_history_count",
    "difficulty_group_support_count",
    "difficulty_fallback_level",
    "course_is_new",
    "course_low_support",
    "course_difficulty_missing",
]

DIFFICULTY_OUTPUT_COLUMNS = STAT_OUTPUT_COLUMNS + AUDIT_OUTPUT_COLUMNS

# Remove both the legacy columns and B2 columns before an idempotent rebuild.
STALE_DIFFICULTY_COLUMNS = DIFFICULTY_OUTPUT_COLUMNS

RAW_STAT_COLUMNS = [
    "support_count",
    "sum_pass",
    "sum_mark",
    "retake_support_count",
    "sum_retake",
]

LEVEL_KEY_COLUMNS = {
    1: "degree_course_key",
    2: "course_id",
    3: "l3_key",
    4: "l4_key",
    5: "l5_key",
}

LEVEL_LABELS = {
    1: "degree_course_key",
    2: "course_id",
    3: "degree_id+requirement_type_id+rounded_course_credits",
    4: "faculty_id+requirement_type_id+rounded_course_credits",
    5: "requirement_type_id+rounded_course_credits",
    6: "global_previous_train_history",
}


@dataclass(frozen=True)
class DifficultyConfig:
    """Configuration whose values are persisted with the fitted state."""

    min_support: int = 20
    shrinkage_k: float = 20.0
    target_threshold: float = 50.0
    version: str = "b2_temporal_course_stats_v1"


@dataclass
class DifficultyState:
    """Frozen sufficient statistics and smoothed lookup tables."""

    tables: Dict[int, pd.DataFrame]
    global_stats: Dict[str, float | int]
    config: DifficultyConfig
    degree_to_faculty: Dict[str, str]


def _drop_stale_columns(df: pd.DataFrame) -> pd.DataFrame:
    stale = [column for column in STALE_DIFFICULTY_COLUMNS if column in df.columns]
    if stale:
        return df.drop(columns=stale, errors="ignore").copy(deep=False)
    # A shallow frame is safe: all builders only append/replace difficulty
    # columns. Sharing immutable source blocks avoids duplicating large splits.
    return df.copy(deep=False)


def _course_id_from_key(df: pd.DataFrame) -> pd.Series:
    return df["degree_course_key"].astype("string").str.rsplit("__", n=1).str[-1]


def _rounded_credits(df: pd.DataFrame) -> pd.Series:
    result = pd.Series(pd.NA, index=df.index, dtype="Int64")
    present = df["course_credits"].notna()
    result.loc[present] = (
        df.loc[present, "course_credits"].astype("float64").round().astype("int64")
    )
    return result


_COMPOSITE_KEY_CHUNK_ROWS = 50_000


def _composite_key(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    """Build composite keys without pandas' memory-heavy row aggregation.

    ``DataFrame.agg(..., axis=1)`` transposes its input internally.  On the
    complete training set that created a large temporary object and exhausted
    RAM.  Chaining vectorized ``Series.str.cat`` calls in positional chunks
    preserves the original key and missing-value semantics while bounding all
    temporary allocations to ``_COMPOSITE_KEY_CHUNK_ROWS`` rows.
    """

    columns = list(columns)
    result = pd.Series(pd.NA, index=df.index, dtype="string")

    # Preserve the legacy aggregation result for an empty column list.
    if not columns:
        result.iloc[:] = ""
        return result

    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        # The legacy df[columns] access raised KeyError even for an empty frame.
        raise KeyError(f"Composite-key columns not found: {missing_columns}")

    for start in range(0, len(df), _COMPOSITE_KEY_CHUNK_ROWS):
        stop = min(start + _COMPOSITE_KEY_CHUNK_ROWS, len(df))
        block = df.iloc[start:stop].loc[:, columns]
        complete_positions = np.flatnonzero(
            block.notna().all(axis=1).to_numpy()
        )
        if complete_positions.size == 0:
            continue

        complete = block.iloc[complete_positions]
        joined = complete.iloc[:, 0].astype("string")
        for position in range(1, len(columns)):
            # Passing the extension array avoids index alignment, which also
            # makes this safe for non-default and duplicate DataFrame indexes.
            joined = joined.str.cat(
                complete.iloc[:, position].astype("string").array,
                sep="__",
            )
        result.iloc[start + complete_positions] = joined.array

    return result


def build_level_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Return the five non-global keys without mutating ``df``."""

    required = {
        "degree_course_key",
        "degree_id",
        "faculty_id",
        "requirement_type_id",
        "course_credits",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns required for difficulty keys: {missing}")

    keys = pd.DataFrame(index=df.index)
    keys["degree_course_key"] = df["degree_course_key"].astype("string")
    keys["course_id"] = _course_id_from_key(df)
    keys["course_credits_key"] = _rounded_credits(df)

    key_frame = pd.DataFrame(index=df.index)
    key_frame["degree_id"] = df["degree_id"].astype("string")
    key_frame["faculty_id"] = df["faculty_id"].astype("string")
    key_frame["requirement_type_id"] = df["requirement_type_id"].astype("string")
    key_frame["course_credits_key"] = keys["course_credits_key"].astype("string")

    keys["l3_key"] = _composite_key(
        key_frame, ["degree_id", "requirement_type_id", "course_credits_key"]
    )
    keys["l4_key"] = _composite_key(
        key_frame, ["faculty_id", "requirement_type_id", "course_credits_key"]
    )
    keys["l5_key"] = _composite_key(
        key_frame, ["requirement_type_id", "course_credits_key"]
    )
    return keys


def _validate_training_frame(df: pd.DataFrame) -> None:
    """ Raise if the training frame is missing required columns or has non-numeric part_id."""
    required = {
        "part_id",
        "final_mark",
        "attempt_number",
        "degree_course_key",
        "degree_id",
        "faculty_id",
        "requirement_type_id",
        "course_credits",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns required for course difficulty: {missing}")

    part_numeric = pd.to_numeric(df["part_id"], errors="coerce")
    if part_numeric.isna().any():
        raise ValueError("part_id must be non-null and numerically sortable")


def _sufficient_stats(frame: pd.DataFrame, key: pd.Series) -> pd.DataFrame:
    work = pd.DataFrame(index=frame.index)
    work["key"] = key
    work["mark_present"] = frame["final_mark"].notna().astype("int64")
    work["pass_value"] = (
        (frame["final_mark"] >= 50) & frame["final_mark"].notna()
    ).astype("int64")
    work["mark_value"] = frame["final_mark"].fillna(0.0).astype("float64")
    work["retake_present"] = frame["attempt_number"].notna().astype("int64")
    work["retake_value"] = (
        (frame["attempt_number"] > 1) & frame["attempt_number"].notna()
    ).astype("int64")
    work = work.loc[work["key"].notna()]
    if work.empty:
        return pd.DataFrame(columns=RAW_STAT_COLUMNS).rename_axis("key")

    return work.groupby("key", sort=False).agg(
        support_count=("mark_present", "sum"),
        sum_pass=("pass_value", "sum"),
        sum_mark=("mark_value", "sum"),
        retake_support_count=("retake_present", "sum"),
        sum_retake=("retake_value", "sum"),
    )


def _degree_faculty_map(df: pd.DataFrame) -> Dict[str, str]:
    pairs = (
        df.loc[df["degree_id"].notna() & df["faculty_id"].notna(), ["degree_id", "faculty_id"]]
        .astype("string")
        .drop_duplicates()
    )
    counts = pairs.groupby("degree_id")["faculty_id"].nunique()
    ambiguous = counts[counts > 1]
    if not ambiguous.empty:
        raise ValueError(
            "Cannot define Level-3 -> Level-4 parent: degree_id maps to multiple "
            f"faculty_id values ({ambiguous.index.tolist()[:5]})"
        )
    return dict(zip(pairs["degree_id"].astype(str), pairs["faculty_id"].astype(str)))


def _global_raw(df: pd.DataFrame) -> Dict[str, float | int]:
    mark_present = df["final_mark"].notna()
    retake_present = df["attempt_number"].notna()
    return {
        "support_count": int(mark_present.sum()),
        "sum_pass": float(((df["final_mark"] >= 50) & mark_present).sum()),
        "sum_mark": float(df["final_mark"].fillna(0.0).sum()),
        "retake_support_count": int(retake_present.sum()),
        "sum_retake": float(((df["attempt_number"] > 1) & retake_present).sum()),
    }


def _safe_rate(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _finalize_global(raw: Dict[str, float | int]) -> Dict[str, float | int]:
    return {
        **raw,
        "course_pass_rate_historical": _safe_rate(
            float(raw["sum_pass"]), int(raw["support_count"])
        ),
        "course_avg_mark_historical": _safe_rate(
            float(raw["sum_mark"]), int(raw["support_count"])
        ),
        "course_retake_rate_historical": _safe_rate(
            float(raw["sum_retake"]), int(raw["retake_support_count"])
        ),
    }


def _parent_values(
    table: pd.DataFrame,
    parent: pd.DataFrame | None,
    global_stats: Dict[str, float | int],
) -> pd.DataFrame:
    result = pd.DataFrame(index=table.index)
    for column in STAT_OUTPUT_COLUMNS:
        if parent is not None and "parent_key" in table.columns:
            values = table["parent_key"].map(parent[column])
            result[column] = values.fillna(float(global_stats[column]))
        else:
            result[column] = float(global_stats[column])
    return result

# shrinkage to parent or global stats
def _smooth_table(
    raw_table: pd.DataFrame,
    parent_keys: pd.Series | None,
    parent_table: pd.DataFrame | None,
    global_stats: Dict[str, float | int],
    config: DifficultyConfig,
) -> pd.DataFrame:
    table = raw_table.copy()
    if parent_keys is not None:
        table["parent_key"] = parent_keys.reindex(table.index).astype("string")
    parent_values = _parent_values(table, parent_table, global_stats)
    k = float(config.shrinkage_k)

    for output, numerator, denominator in (
        ("course_pass_rate_historical", "sum_pass", "support_count"),
        ("course_avg_mark_historical", "sum_mark", "support_count"),
        ("course_retake_rate_historical", "sum_retake", "retake_support_count"),
    ):
        n = table[denominator].astype("float64")
        local_sum = table[numerator].astype("float64")
        parent_value = parent_values[output].astype("float64")
        table[output] = (local_sum + k * parent_value) / (n + k)

    for column in ("support_count", "retake_support_count"):
        table[column] = table[column].astype("int64")
    return table


def _finalize_state_from_raw(
    raw_tables: Dict[int, pd.DataFrame],
    global_raw: Dict[str, float | int],
    degree_to_faculty: Dict[str, str],
    config: DifficultyConfig,
) -> DifficultyState:
    """Create smoothed tables from cumulative sufficient statistics."""

    global_stats = _finalize_global(global_raw)
    # Direct structural parents: L1->L2, L3->L4->L5->L6.  Course_id is
    # cross-degree, so L2 has no unique L3 parent and correctly shrinks to L6.
    l5_parent = None
    l4_index = pd.Series(
        raw_tables[4].index.astype(str), index=raw_tables[4].index, dtype="string"
    )
    l4_parts = l4_index.str.rsplit("__", n=2, expand=True)
    l4_parent = (l4_parts[1] + "__" + l4_parts[2]).astype("string")

    l3_index = pd.Series(
        raw_tables[3].index.astype(str), index=raw_tables[3].index, dtype="string"
    )
    l3_parts = l3_index.str.split("__", n=2, expand=True)
    if raw_tables[3].empty:
        l3_parent = pd.Series(index=raw_tables[3].index, dtype="string")
    else:
        l3_degree = l3_parts[0].astype("string")
        l3_req = l3_parts[1].astype("string")
        l3_credits = l3_parts[2].astype("string")
        l3_faculty = l3_degree.map(degree_to_faculty).astype("string")
        l3_parent = l3_faculty + "__" + l3_req + "__" + l3_credits

    l1_index = pd.Series(
        raw_tables[1].index.astype(str), index=raw_tables[1].index, dtype="string"
    )
    l1_parent = l1_index.str.rsplit("__", n=1).str[-1].astype("string")

    tables: Dict[int, pd.DataFrame] = {}
    tables[5] = _smooth_table(raw_tables[5], l5_parent, None, global_stats, config)
    tables[4] = _smooth_table(raw_tables[4], l4_parent, tables[5], global_stats, config)
    tables[3] = _smooth_table(raw_tables[3], l3_parent, tables[4], global_stats, config)
    tables[2] = _smooth_table(raw_tables[2], None, None, global_stats, config)
    tables[1] = _smooth_table(raw_tables[1], l1_parent, tables[2], global_stats, config)

    for level, table in tables.items():
        table.index = table.index.astype("string")
        table.index.name = LEVEL_KEY_COLUMNS[level]
    return DifficultyState(tables, global_stats, config, degree_to_faculty)


def fit_difficulty_state(
    df_train_history: pd.DataFrame,
    config: DifficultyConfig | None = None,
) -> DifficultyState:
    """Fit a frozen six-level state from training history only."""

    config = config or DifficultyConfig()
    _validate_training_frame(df_train_history)
    clean = _drop_stale_columns(df_train_history)
    keys = build_level_keys(clean)
    raw_tables = {
        level: _sufficient_stats(clean, keys[key_column])
        for level, key_column in LEVEL_KEY_COLUMNS.items()
    }
    return _finalize_state_from_raw(
        raw_tables,
        _global_raw(clean),
        _degree_faculty_map(clean),
        config,
    )


def _add_raw_tables(existing: pd.DataFrame, addition: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        result = addition.copy()
    elif addition.empty:
        result = existing.copy()
    else:
        result = existing.add(addition, fill_value=0.0)
    result = result.loc[:, RAW_STAT_COLUMNS]
    for column in ("support_count", "sum_pass", "retake_support_count", "sum_retake"):
        result[column] = result[column].astype("int64")
    result["sum_mark"] = result["sum_mark"].astype("float64")
    result.index = result.index.astype("string")
    return result


class _DifficultyAccumulator:
    """Incrementally maintain history so each semester needs one small groupby."""

    def __init__(self, config: DifficultyConfig) -> None:
        self.config = config
        self.raw_tables = {
            level: pd.DataFrame(columns=RAW_STAT_COLUMNS).rename_axis(key)
            for level, key in LEVEL_KEY_COLUMNS.items()
        }
        self.global_raw: Dict[str, float | int] = {
            "support_count": 0,
            "sum_pass": 0.0,
            "sum_mark": 0.0,
            "retake_support_count": 0,
            "sum_retake": 0.0,
        }
        self.degree_to_faculty: Dict[str, str] = {}

    @property
    def has_history(self) -> bool:
        return int(self.global_raw["support_count"]) > 0

    def state(self) -> DifficultyState:
        if not self.has_history:
            return empty_difficulty_state(self.config)
        return _finalize_state_from_raw(
            self.raw_tables,
            self.global_raw,
            self.degree_to_faculty,
            self.config,
        )

    def update(self, frame: pd.DataFrame) -> None:
        clean = _drop_stale_columns(frame)
        keys = build_level_keys(clean)
        for level, key_column in LEVEL_KEY_COLUMNS.items():
            addition = _sufficient_stats(clean, keys[key_column])
            self.raw_tables[level] = _add_raw_tables(self.raw_tables[level], addition)

        increment = _global_raw(clean)
        for column in self.global_raw:
            self.global_raw[column] = self.global_raw[column] + increment[column]

        new_mapping = _degree_faculty_map(clean)
        for degree, faculty in new_mapping.items():
            previous = self.degree_to_faculty.get(degree)
            if previous is not None and previous != faculty:
                raise ValueError(
                    f"degree_id {degree!r} changed faculty_id from {previous!r} to {faculty!r}"
                )
            self.degree_to_faculty[degree] = faculty


def empty_difficulty_state(config: DifficultyConfig | None = None) -> DifficultyState:
    """Return the no-history state used for the first training semester."""

    config = config or DifficultyConfig()
    empty_tables = {
        level: pd.DataFrame(columns=RAW_STAT_COLUMNS + STAT_OUTPUT_COLUMNS).rename_axis(key)
        for level, key in LEVEL_KEY_COLUMNS.items()
    }
    global_stats = _finalize_global(
        {
            "support_count": 0,
            "sum_pass": 0.0,
            "sum_mark": 0.0,
            "retake_support_count": 0,
            "sum_retake": 0.0,
        }
    )
    return DifficultyState(empty_tables, global_stats, config, {})


def apply_difficulty_state(
    df: pd.DataFrame,
    state: DifficultyState,
    *,
    include_source: bool = True,
) -> pd.DataFrame:
    """Apply a frozen state without using any targets from ``df``.

    ``include_source=False`` returns only the nine B2 output columns and is used
    by the memory-bounded versioned builder.
    """

    clean = _drop_stale_columns(df)
    keys = build_level_keys(clean)
    n_rows = len(clean)

    supports: Dict[int, pd.Series] = {}
    mapped_stats: Dict[int, Dict[str, pd.Series]] = {}
    hits: Dict[int, np.ndarray] = {}
    already_hit = np.zeros(n_rows, dtype=bool)

    for level in range(1, 6):
        table = state.tables[level]
        key = keys[LEVEL_KEY_COLUMNS[level]]
        support = key.map(table["support_count"]) if not table.empty else pd.Series(np.nan, index=clean.index)
        supports[level] = support
        available = support.notna().to_numpy()
        hits[level] = available & ~already_hit
        already_hit |= available
        mapped_stats[level] = {
            column: key.map(table[column]) if not table.empty else pd.Series(np.nan, index=clean.index)
            for column in STAT_OUTPUT_COLUMNS
        }

    hits[6] = ~already_hit
    selected_level = np.select(
        [hits[level] for level in range(1, 6)],
        [level for level in range(1, 6)],
        default=6,
    ).astype("int64")

    feature_values: Dict[str, np.ndarray] = {}
    for column in STAT_OUTPUT_COLUMNS:
        values = np.full(n_rows, float(state.global_stats[column]), dtype="float64")
        for level in range(5, 0, -1):
            level_values = mapped_stats[level][column].to_numpy(dtype="float64", na_value=np.nan)
            values[hits[level]] = level_values[hits[level]]
        feature_values[column] = values

    l1_support = supports[1].fillna(0).to_numpy(dtype="int64")
    l2_support = supports[2].fillna(0).to_numpy(dtype="int64")
    course_history = np.where(hits[1], l1_support, np.where(hits[2], l2_support, 0))

    group_support = np.full(n_rows, int(state.global_stats["support_count"]), dtype="int64")
    for level in range(5, 0, -1):
        values = supports[level].fillna(0).to_numpy(dtype="int64")
        group_support[hits[level]] = values[hits[level]]

    course_is_new = (~supports[1].notna() & ~supports[2].notna()).astype("int64").to_numpy()
    course_low_support = (
        (course_history > 0) & (course_history < int(state.config.min_support))
    ).astype("int64")

    feature_values["course_history_count"] = course_history.astype("int64")
    feature_values["difficulty_group_support_count"] = group_support
    feature_values["difficulty_fallback_level"] = selected_level
    feature_values["course_is_new"] = course_is_new
    feature_values["course_low_support"] = course_low_support
    feature_values["course_difficulty_missing"] = (
        (course_is_new == 1) | (course_low_support == 1)
    ).astype("int64")

    features = pd.DataFrame(feature_values, index=clean.index, copy=False)
    out = (
        pd.concat([clean, features], axis=1)
        if include_source
        else features
    )

    if len(out) != len(df) or not out.index.equals(df.index):
        raise AssertionError("Difficulty enrichment changed row count, order, or index")
    return out


def build_temporal_train(
    df_train: pd.DataFrame,
    config: DifficultyConfig | None = None,
    *,
    include_source: bool = True,
) -> pd.DataFrame:
    """Enrich train so semester ``t`` sees only semesters strictly before it."""

    config = config or DifficultyConfig()
    _validate_training_frame(df_train)
    clean = _drop_stale_columns(df_train)
    part_numeric = pd.to_numeric(clean["part_id"], errors="raise")
    semester_values = sorted(part_numeric.unique().tolist())

    result_columns: Dict[str, np.ndarray] = {
        column: np.full(len(clean), np.nan, dtype="float64") for column in STAT_OUTPUT_COLUMNS
    }
    result_columns.update(
        {column: np.zeros(len(clean), dtype="int64") for column in AUDIT_OUTPUT_COLUMNS}
    )

    accumulator = _DifficultyAccumulator(config)
    for semester in semester_values:
        positions = np.flatnonzero(part_numeric.to_numpy() == semester)
        current = clean.iloc[positions]
        state = accumulator.state()
        enriched = apply_difficulty_state(current, state, include_source=False)
        for column in DIFFICULTY_OUTPUT_COLUMNS:
            result_columns[column][positions] = enriched[column].to_numpy()
        # Current targets are admitted only after every row in this semester
        # has been enriched from the pre-semester snapshot.
        accumulator.update(current)

    features = pd.DataFrame(result_columns, index=clean.index, copy=False)
    out = (
        pd.concat([clean, features], axis=1)
        if include_source
        else features
    )

    if len(out) != len(df_train) or not out.index.equals(df_train.index):
        raise AssertionError("Temporal train enrichment changed row count, order, or index")
    return out


def save_difficulty_state(
    state: DifficultyState,
    output_dir: str | Path,
    *,
    metadata: Dict[str, object] | None = None,
) -> Path:
    """Persist every lookup level and its fitting contract."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite fitted state: {output_dir}")
    output_dir.mkdir(parents=True)

    table_files: Dict[str, str] = {}
    for level in range(1, 6):
        filename = f"level_{level}.parquet"
        state.tables[level].reset_index().to_parquet(output_dir / filename, index=False)
        table_files[str(level)] = filename

    manifest = {
        "artifact": "course_difficulty_state",
        "version": state.config.version,
        "config": asdict(state.config),
        "fallback_levels": {str(key): value for key, value in LEVEL_LABELS.items()},
        "shrinkage_parents": {
            "1": "2",
            "2": "6",
            "3": "4",
            "4": "5",
            "5": "6",
            "6": None,
        },
        "global_stats": state.global_stats,
        "degree_to_faculty": state.degree_to_faculty,
        "table_files": table_files,
        "metadata": metadata or {},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_dir


def load_difficulty_state(input_dir: str | Path) -> DifficultyState:
    """Load a state created by :func:`save_difficulty_state`."""

    input_dir = Path(input_dir)
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    config = DifficultyConfig(**manifest["config"])
    tables: Dict[int, pd.DataFrame] = {}
    for level_text, filename in manifest["table_files"].items():
        level = int(level_text)
        key_column = LEVEL_KEY_COLUMNS[level]
        table = pd.read_parquet(input_dir / filename)
        if table[key_column].duplicated().any():
            raise ValueError(f"Duplicate keys in persisted difficulty level {level}")
        tables[level] = table.set_index(key_column)
        tables[level].index = tables[level].index.astype("string")
    return DifficultyState(
        tables=tables,
        global_stats=manifest["global_stats"],
        config=config,
        degree_to_faculty={str(k): str(v) for k, v in manifest["degree_to_faculty"].items()},
    )
