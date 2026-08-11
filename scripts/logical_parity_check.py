"""Logical parity check between two parquet artifacts.

Implements the "B. Logical parity" comparison defined in
``docs/data_governance_plan.md`` §4:

    Compare: row counts; unique keys + duplicate counts; schema + dtypes;
    values sort-normalized on stable keys; null patterns; distributions.
    Requires determinism: fixed seeds, stable sort, declared float tolerance.
    Any delta beyond these fails.

Byte comparison is deliberately never attempted here. Parquet is not
byte-stable across runs (compression settings, row-group ordering, page
layout can all differ between two writes of logically identical data), so a
byte-level hash difference proves nothing about the data and byte equality
is not required by this script.

"Sort-normalised on stable keys" is implemented as an outer merge on the
declared natural key rather than a positional sort-then-zip comparison --
merging on the key is a strictly stronger normalisation (row order in either
input is fully irrelevant) and it directly yields the key-only-in-A /
key-only-in-B sets that a sort-then-zip approach would have to derive
separately.

This script reports differences. It does not decide whether a difference is
acceptable, does not adjust any threshold, and does not pick a winner
between the two inputs -- that judgement belongs to the project owner.

Usage
-----
    python scripts/logical_parity_check.py PATH_A PATH_B \\
        --keys student_course_id \\
        [--float-tol 1e-6] \\
        [--label-a "live"] [--label-b "staging"] \\
        [--sample-mismatches 10]

Prints a self-contained Markdown section to stdout. Compose a combined
report by concatenating the output of one invocation per artifact pair.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    # Local Markdown renderer so the script has no dependency on the
    # optional `tabulate` package that pandas.DataFrame.to_markdown needs.
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body_lines = []
    for _, row in df.iterrows():
        cells = ["" if pd.isna(v) else str(v) for v in row]
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + body_lines)


def load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def check_row_counts(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    return {
        "rows_a": len(df_a),
        "rows_b": len(df_b),
        "delta": len(df_b) - len(df_a),
        "match": len(df_a) == len(df_b),
    }


def check_schema(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    cols_a, cols_b = set(df_a.columns), set(df_b.columns)
    common = sorted(cols_a & cols_b)
    only_a = sorted(cols_a - cols_b)
    only_b = sorted(cols_b - cols_a)
    dtype_mismatches = []
    for col in common:
        da, db = str(df_a[col].dtype), str(df_b[col].dtype)
        if da != db:
            dtype_mismatches.append((col, da, db))
    return {
        "common_columns": common,
        "only_a": only_a,
        "only_b": only_b,
        "dtype_mismatches": dtype_mismatches,
        "match": not only_a and not only_b and not dtype_mismatches,
    }


def check_keys(df_a: pd.DataFrame, df_b: pd.DataFrame, keys: list[str]) -> dict:
    out = {}
    for label, df in (("a", df_a), ("b", df_b)):
        n = len(df)
        nk = df[keys].drop_duplicates().shape[0]
        out[label] = {"rows": n, "unique_keys": nk, "duplicates": n - nk}
    out["match"] = out["a"]["duplicates"] == 0 and out["b"]["duplicates"] == 0
    return out


def check_nulls(df_a: pd.DataFrame, df_b: pd.DataFrame, common_cols: list[str]) -> dict:
    rows = []
    any_delta = False
    for col in common_cols:
        na_a = int(df_a[col].isna().sum())
        na_b = int(df_b[col].isna().sum())
        pct_a = na_a / len(df_a) * 100 if len(df_a) else 0.0
        pct_b = na_b / len(df_b) * 100 if len(df_b) else 0.0
        delta = na_b - na_a
        if delta != 0:
            any_delta = True
        rows.append((col, na_a, pct_a, na_b, pct_b, delta))
    return {"rows": rows, "match": not any_delta}


def _canonical_key(s: pd.Series) -> pd.Series:
    # Merge keys must be mergeable even when A and B disagree on the key's
    # *dtype* (e.g. float64 123.0 on one side, string "123" on the other --
    # a real dtype mismatch, already surfaced separately by check_schema).
    # This canonicalisation exists only so the value-level comparison below
    # can still run; it does not hide the dtype mismatch, which is reported
    # verbatim in its own section.
    def one(v):
        if pd.isna(v):
            return "__NULL__"
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v).strip()

    return s.map(one)


def check_values(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    keys: list[str],
    common_cols: list[str],
    float_tol: float,
    sample_mismatches: int,
) -> dict:
    value_cols = [c for c in common_cols if c not in keys]
    a = df_a[keys + value_cols].copy()
    b = df_b[keys + value_cols].copy()
    for k in keys:
        a[k] = _canonical_key(a[k])
        b[k] = _canonical_key(b[k])
    a.columns = list(keys) + [f"{c}__A" for c in value_cols]
    b.columns = list(keys) + [f"{c}__B" for c in value_cols]

    merged = a.merge(b, on=keys, how="outer", indicator=True)

    key_only_a = int((merged["_merge"] == "left_only").sum())
    key_only_b = int((merged["_merge"] == "right_only").sum())
    both = merged[merged["_merge"] == "both"]

    per_column = []
    for col in value_cols:
        ca, cb = merged[f"{col}__A"], merged[f"{col}__B"]
        both_na = ca.isna() & cb.isna()
        if pd.api.types.is_float_dtype(ca) or pd.api.types.is_float_dtype(cb):
            a_num = pd.to_numeric(ca, errors="coerce")
            b_num = pd.to_numeric(cb, errors="coerce")
            close = np.isclose(
                a_num.to_numpy(dtype="float64", na_value=np.nan),
                b_num.to_numpy(dtype="float64", na_value=np.nan),
                atol=float_tol,
                rtol=0.0,
                equal_nan=False,
            )
            match = both_na.to_numpy() | close
        else:
            eq = (ca.astype(str) == cb.astype(str)).to_numpy()
            match = both_na.to_numpy() | eq
        # only score rows present on both sides -- key-only rows are already
        # reported separately and have no counterpart value to compare
        both_mask = (merged["_merge"] == "both").to_numpy()
        mismatches = int((~match & both_mask).sum())
        sample = None
        if mismatches and sample_mismatches > 0:
            mism_idx = merged.index[(~match) & both_mask][:sample_mismatches]
            sample = merged.loc[mism_idx, list(keys) + [f"{col}__A", f"{col}__B"]]
        per_column.append({"column": col, "mismatches": mismatches, "sample": sample})

    return {
        "key_only_a": key_only_a,
        "key_only_b": key_only_b,
        "per_column": per_column,
        "match": key_only_a == 0
        and key_only_b == 0
        and all(pc["mismatches"] == 0 for pc in per_column),
    }


def check_distributions(df_a: pd.DataFrame, df_b: pd.DataFrame, common_cols: list[str]) -> dict:
    rows = []
    for col in common_cols:
        if not (pd.api.types.is_numeric_dtype(df_a[col]) and pd.api.types.is_numeric_dtype(df_b[col])):
            continue
        sa, sb = df_a[col].describe(), df_b[col].describe()
        rows.append(
            {
                "column": col,
                "a": {k: sa.get(k) for k in ("count", "mean", "std", "min", "50%", "max")},
                "b": {k: sb.get(k) for k in ("count", "mean", "std", "min", "50%", "max")},
            }
        )
    return {"rows": rows}


def render_markdown(
    label_a: str,
    label_b: str,
    path_a: Path,
    path_b: Path,
    keys: list[str],
    float_tol: float,
    row_result: dict,
    schema_result: dict,
    key_result: dict,
    null_result: dict,
    value_result: dict,
    dist_result: dict,
) -> str:
    lines = []
    lines.append(f"## {path_a.name}")
    lines.append("")
    lines.append(f"- A ({label_a}): `{path_a}`")
    lines.append(f"- B ({label_b}): `{path_b}`")
    lines.append(f"- Natural key: `{', '.join(keys)}`")
    lines.append(f"- Float tolerance: `{float_tol}`")
    lines.append("")

    # Row counts
    lines.append("### Row counts")
    lines.append("")
    lines.append(f"- A: {_fmt_int(row_result['rows_a'])}")
    lines.append(f"- B: {_fmt_int(row_result['rows_b'])}")
    lines.append(f"- Delta (B - A): {row_result['delta']:+,}")
    lines.append(f"- **{'MATCH' if row_result['match'] else 'DIFFERS'}**")
    lines.append("")

    # Schema
    lines.append("### Schema and dtypes")
    lines.append("")
    lines.append(f"- Common columns: {len(schema_result['common_columns'])}")
    lines.append(f"- Only in A: {schema_result['only_a'] or 'none'}")
    lines.append(f"- Only in B: {schema_result['only_b'] or 'none'}")
    if schema_result["dtype_mismatches"]:
        lines.append("- Dtype mismatches on common columns:")
        for col, da, db in schema_result["dtype_mismatches"]:
            lines.append(f"  - `{col}`: A=`{da}` vs B=`{db}`")
    else:
        lines.append("- Dtype mismatches on common columns: none")
    lines.append(f"- **{'MATCH' if schema_result['match'] else 'DIFFERS'}**")
    lines.append("")

    # Keys
    lines.append("### Unique-key and duplicate counts")
    lines.append("")
    lines.append("| Side | Rows | Unique keys | Duplicates |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| A | {_fmt_int(key_result['a']['rows'])} | {_fmt_int(key_result['a']['unique_keys'])} | {_fmt_int(key_result['a']['duplicates'])} |"
    )
    lines.append(
        f"| B | {_fmt_int(key_result['b']['rows'])} | {_fmt_int(key_result['b']['unique_keys'])} | {_fmt_int(key_result['b']['duplicates'])} |"
    )
    lines.append(f"- **{'MATCH' if key_result['match'] else 'DIFFERS'}**")
    lines.append("")

    # Nulls
    lines.append("### Null patterns per column (common columns)")
    lines.append("")
    delta_rows = [r for r in null_result["rows"] if r[5] != 0]
    if delta_rows:
        lines.append("| Column | Nulls A | % A | Nulls B | % B | Delta |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for col, na_a, pct_a, na_b, pct_b, delta in delta_rows:
            lines.append(
                f"| `{col}` | {_fmt_int(na_a)} | {pct_a:.3f}% | {_fmt_int(na_b)} | {pct_b:.3f}% | {delta:+,} |"
            )
    else:
        lines.append("No null-count delta on any common column.")
    lines.append(f"- **{'MATCH' if null_result['match'] else 'DIFFERS'}**")
    lines.append("")

    # Values
    lines.append("### Values, sort-normalised on the natural key (outer merge)")
    lines.append("")
    lines.append(f"- Keys only in A (missing from B): {_fmt_int(value_result['key_only_a'])}")
    lines.append(f"- Keys only in B (missing from A): {_fmt_int(value_result['key_only_b'])}")
    mismatched_cols = [pc for pc in value_result["per_column"] if pc["mismatches"] > 0]
    if mismatched_cols:
        lines.append("- Columns with value mismatches on shared keys:")
        lines.append("")
        lines.append("| Column | Mismatched rows |")
        lines.append("|---|---:|")
        for pc in mismatched_cols:
            lines.append(f"| `{pc['column']}` | {_fmt_int(pc['mismatches'])} |")
        lines.append("")
        for pc in mismatched_cols:
            if pc["sample"] is not None and len(pc["sample"]):
                lines.append(f"Sample mismatches for `{pc['column']}`:")
                lines.append("")
                lines.append(_df_to_markdown_table(pc["sample"]))
                lines.append("")
    else:
        lines.append("- No value mismatches on any shared column for rows present in both.")
    lines.append(f"- **{'MATCH' if value_result['match'] else 'DIFFERS'}**")
    lines.append("")

    # Distributions
    lines.append("### Distribution summaries (numeric common columns)")
    lines.append("")
    if dist_result["rows"]:
        lines.append("| Column | count A | mean A | std A | min A | median A | max A | count B | mean B | std B | min B | median B | max B |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in dist_result["rows"]:
            a, b = r["a"], r["b"]
            def f(v):
                return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.4f}" if isinstance(v, float) else str(v)
            lines.append(
                f"| `{r['column']}` | {f(a['count'])} | {f(a['mean'])} | {f(a['std'])} | {f(a['min'])} | {f(a['50%'])} | {f(a['max'])} "
                f"| {f(b['count'])} | {f(b['mean'])} | {f(b['std'])} | {f(b['min'])} | {f(b['50%'])} | {f(b['max'])} |"
            )
    else:
        lines.append("No shared numeric columns.")
    lines.append("")

    overall = (
        row_result["match"]
        and schema_result["match"]
        and key_result["match"]
        and null_result["match"]
        and value_result["match"]
    )
    lines.append(f"### Overall: {'PARITY MATCH' if overall else 'PARITY DIFFERS'}")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    # Several source columns carry Arabic text (course/degree names); make
    # stdout tolerant of whatever the active console codepage is rather than
    # letting an unexpected glyph abort a long batch run.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path_a", type=Path)
    parser.add_argument("path_b", type=Path)
    parser.add_argument("--keys", required=True, help="Comma-separated natural-key column names")
    parser.add_argument("--float-tol", type=float, default=1e-6)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--sample-mismatches", type=int, default=10)
    args = parser.parse_args(argv)

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]

    df_a = load(args.path_a)
    df_b = load(args.path_b)

    for k in keys:
        if k not in df_a.columns or k not in df_b.columns:
            print(f"ERROR: key column {k!r} missing from A or B", file=sys.stderr)
            return 2

    row_result = check_row_counts(df_a, df_b)
    schema_result = check_schema(df_a, df_b)
    key_result = check_keys(df_a, df_b, keys)
    null_result = check_nulls(df_a, df_b, schema_result["common_columns"])
    value_result = check_values(
        df_a, df_b, keys, schema_result["common_columns"], args.float_tol, args.sample_mismatches
    )
    dist_result = check_distributions(df_a, df_b, schema_result["common_columns"])

    md = render_markdown(
        args.label_a,
        args.label_b,
        args.path_a,
        args.path_b,
        keys,
        args.float_tol,
        row_result,
        schema_result,
        key_result,
        null_result,
        value_result,
        dist_result,
    )
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
