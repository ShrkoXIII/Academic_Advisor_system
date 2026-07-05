"""Shared cleaning helpers for ID-heavy academic source tables.

The raw Oracle extracts mix numeric IDs, dotted composite IDs, decimals, and
null-like strings. These helpers normalize those values before joins and
diagnostics so downstream feature engineering works with stable keys.
"""

from decimal import Decimal

import numpy as np
import pandas as pd


__all__ = [
    "add_id_components",
    "append_reason",
    "integer_like_report",
    "is_integer_like_numeric",
    "normalize_id_columns",
    "normalize_id_series",
    "normalize_id_to_string",
]


def is_integer_like_numeric(series: pd.Series) -> pd.Series:
    # Treat integer-like values as safe ID candidates while preserving a mask
    # for fractional values that would be unsafe to coerce silently.
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna() & ((numeric % 1) == 0)


def integer_like_report(df: pd.DataFrame, column: str) -> dict:
    # Summarize conversion risk before normalizing an ID-like column.
    numeric = pd.to_numeric(df[column], errors="coerce")
    non_null_numeric = numeric.dropna()
    fractional_mask = numeric.notna() & ~is_integer_like_numeric(df[column])

    return {
        "column": column,
        "source_dtype": str(df[column].dtype),
        "non_null_count": int(df[column].notna().sum()),
        "numeric_count": int(non_null_numeric.shape[0]),
        "non_numeric_or_null_count": int(df[column].shape[0] - non_null_numeric.shape[0]),
        "fractional_count": int(fractional_mask.sum()),
        "fractional_ratio": (
            float(fractional_mask.sum() / len(non_null_numeric))
            if len(non_null_numeric) > 0
            else 0
        ),
    }


def _strip_useless_decimal_zero(text: str) -> str:
    # Oracle exports sometimes turn integer IDs into "123.0"; strip only the
    # meaningless decimal form so true dotted IDs keep their suffix.
    integer_part, separator, decimal_part = text.partition(".")
    if separator and decimal_part and set(decimal_part) == {"0"} and integer_part.lstrip("+-").isdigit():
        return integer_part
    return text


def normalize_id_to_string(value):
    # Normalize one value through explicit type branches so IDs do not change
    # depending on pandas' inferred dtype for a source file.
    if pd.isna(value):
        return pd.NA

    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, Decimal):
        text = format(value, "f")
    elif isinstance(value, (np.integer, int)):
        text = str(value)
    elif isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return pd.NA
        text = np.format_float_positional(value, trim="-")
    else:
        text = str(value).strip()

    if text == "" or text.lower() in {"nan", "none", "null", "<na>"}:
        return pd.NA

    # Preserve meaningful dotted suffixes, but remove decimal artifacts.
    return _strip_useless_decimal_zero(text)


def normalize_id_series(series: pd.Series) -> pd.Series:
    """Normalize an ID-like series while preserving meaningful decimal suffixes."""
    return series.map(normalize_id_to_string).astype("string")


def normalize_id_columns(
    df: pd.DataFrame,
    columns=None,
    *,
    column_map=None,
    copy: bool = True,
) -> pd.DataFrame:
    """Normalize ID columns in a dataframe.

    Use columns for same-name updates and column_map for source-to-target keys.
    """
    # Copy by default so notebook and pipeline callers can inspect before/after
    # data without accidental mutation.
    result = df.copy() if copy else df

    if isinstance(columns, str):
        columns = [columns]

    # Normalize columns that keep their original names.
    for column in columns or []:
        result[column] = normalize_id_series(result[column])

    # Normalize columns that need to be materialized under canonical names.
    mappings = column_map.items() if isinstance(column_map, dict) else column_map
    for source_column, target_column in mappings or []:
        result[target_column] = normalize_id_series(result[source_column])

    return result


def add_id_components(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    # Keep raw, string, base, suffix, and key variants together so data-quality
    # notebooks can diagnose ID joins without recomputing each representation.
    result = df.copy()

    raw_col = f"{id_col}_raw"
    str_col = f"{id_col}_str"
    base_col = f"{id_col}_base"
    suffix_col = f"{id_col}_suffix"
    key_col = f"{id_col}_key"

    result[raw_col] = result[id_col]
    result[str_col] = result[id_col].apply(normalize_id_to_string).astype("string")

    # Split only once because the dotted suffix carries university/source
    # identity in this project.
    split_parts = result[str_col].str.split(".", n=1, expand=True)

    result[base_col] = pd.to_numeric(split_parts[0], errors="coerce").astype("Int64")

    if split_parts.shape[1] > 1:
        result[suffix_col] = split_parts[1].astype("string")
    else:
        result[suffix_col] = pd.Series(pd.NA, index=result.index, dtype="string")

    result[key_col] = result[str_col]

    return result


def append_reason(df: pd.DataFrame, mask: pd.Series, column: str, reason: str) -> None:
    # Accumulate audit reasons in-place so cleaning rules can preserve every
    # reason a row was changed or flagged.
    current = df.loc[mask, column].astype("string").fillna("")
    separator = np.where(current.str.len() > 0, " | ", "")
    df.loc[mask, column] = current + separator + reason
