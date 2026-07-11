"""Schema-driven ID casting for the cleaning notebooks.

This is the only module that knows both the normalization mechanics
(cleaning_utils) and the per-table schema (schemas). Cleaning notebooks call
normalize_ids(df, table=TABLE_...) once, after their drop/null-handling steps,
to cast every registered ID column in place — no duplicate `_key` columns.
"""

import pandas as pd

from src.cleaning_utils import normalize_id_series
from src.schemas import ID_COLUMN_RULES, TABLE_ID_COLUMNS


def normalize_ids(df, table, skip=None):
    """Cast every ID column registered for `table` per schemas.ID_COLUMN_RULES.

    `skip`: optional iterable of column names to exclude from this call.
    Existence is still validated even for skipped columns
    (skip means "don't cast", not "this column doesn't exist here").
    Any skip usage is printed, never silent.
    """
    skip = set(skip or [])
    if skip:
        print(f"{table}: skipping ID normalization for {sorted(skip)}")

    registered = TABLE_ID_COLUMNS[table]

    # Allowlist check (mirrors the feature_contract.json philosophy): an
    # *_id column that reaches the cast point without a registry entry means
    # a new or renamed ID slipped in — fail loudly instead of passing it
    # through with an unmanaged dtype.
    unregistered = [
        column for column in df.columns
        if column.endswith("_id") and column not in registered
    ]
    if unregistered:
        raise AssertionError(
            f"{table}: ID-like columns not registered in TABLE_ID_COLUMNS: "
            f"{unregistered}. Register them in src/schemas.py before cleaning."
        )

    for column in registered:
        if column not in df.columns:
            raise AssertionError(f"{table}: expected ID column '{column}' missing")
        if column in skip:
            continue
        rule = ID_COLUMN_RULES[column]
        if rule.dtype == "string":
            df[column] = normalize_id_series(df[column])
        elif rule.dtype == "Int64":
            # Categorical codes may arrive as dotted-suffix floats
            # (e.g. 15.111 where .111 is university identity, not a decimal).
            # A bare astype("Int64") raises on non-integer floats, so verify
            # the suffix is uniform, strip it, then cast — same guard as the
            # diploma-merge stage.
            as_string = normalize_id_series(df[column])
            suffixes = set(
                as_string.str.extract(r"\.([^.]+)$", expand=False).dropna().unique()
            )
            if len(suffixes) > 1:
                raise AssertionError(
                    f"{table}.{column}: multiple university suffixes {suffixes} — "
                    "STOP (rows span multiple universities)."
                )
            if suffixes:
                print(
                    f"{table}.{column}: uniform suffix "
                    f"'.{next(iter(suffixes))}' stripped before Int64 cast"
                )
            base = as_string.str.split(".").str[0]
            casted = pd.to_numeric(base, errors="coerce").astype("Int64")
            if int(casted.isna().sum()) != int(as_string.isna().sum()):
                raise AssertionError(
                    f"{table}.{column}: Int64 cast introduced new nulls — investigate."
                )
            df[column] = casted
        else:
            raise AssertionError(
                f"{table}.{column}: unhandled canonical dtype '{rule.dtype}'"
            )
    return df
