from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd


__all__ = [
    "add_id_components",
    "append_reason",
    "integer_like_report",
    "is_integer_like_numeric",
    "normalize_id_to_string",
]


def is_integer_like_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna() & ((numeric % 1) == 0)


def integer_like_report(df: pd.DataFrame, column: str) -> dict:
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


def normalize_id_to_string(value):
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()

    if text == "" or text.lower() in {"nan", "none", "null", "<na>"}:
        return pd.NA

    try:
        decimal_value = Decimal(text)
        normalized = format(decimal_value, "f")

        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")

        return normalized

    except InvalidOperation:
        return text


def add_id_components(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    result = df.copy()

    raw_col = f"{id_col}_raw"
    str_col = f"{id_col}_str"
    base_col = f"{id_col}_base"
    suffix_col = f"{id_col}_suffix"
    key_col = f"{id_col}_key"

    result[raw_col] = result[id_col]
    result[str_col] = result[id_col].apply(normalize_id_to_string).astype("string")

    split_parts = result[str_col].str.split(".", n=1, expand=True)

    result[base_col] = pd.to_numeric(split_parts[0], errors="coerce").astype("Int64")

    if split_parts.shape[1] > 1:
        result[suffix_col] = split_parts[1].astype("string")
    else:
        result[suffix_col] = pd.Series(pd.NA, index=result.index, dtype="string")

    result[key_col] = result[str_col]

    return result


def append_reason(df: pd.DataFrame, mask: pd.Series, column: str, reason: str) -> None:
    current = df.loc[mask, column].astype("string").fillna("")
    separator = np.where(current.str.len() > 0, " | ", "")
    df.loc[mask, column] = current + separator + reason
