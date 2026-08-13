"""Raising guards shared across the data pipeline.

These helpers own the whole guard: the condition, the message, and the raise.
A call site is one line, so a module reads as the computation it performs
rather than as a wall of error strings.

Design rules, all deliberate:

* **The exception class stays with the call site**, passed as ``error=``. The
  guards consolidated here span ``KeyError``, ``ValueError`` and
  ``AssertionError``, and which one a caller raises is part of that caller's
  contract, not of the check.
* **No keyword has a default.** ``label``, ``error`` and ``check_index`` are
  all keyword-only and all required. A default is how a site silently acquires
  the wrong exception class, an anonymous message, or an index comparison it
  never asked for.
* **The message text is owned here** and may evolve, as long as it keeps
  naming the label and the offending items. Nothing outside this module
  should compose these strings.
* **The missing-column list is sorted here.** Several callers build their
  required set as a ``set`` literal, and Python randomises string hashing per
  process, so an unsorted list would produce a different message on every run.
  Sorting makes the output deterministic and lets callers pass a set directly.

This project does not run under ``python -O``; that is a recorded decision.
Guards that used to be bare ``assert`` statements therefore convert to
``error=AssertionError`` calls without losing enforcement.
"""

from __future__ import annotations

from typing import Iterable, Type

import pandas as pd


__all__ = ["require_columns", "assert_shape_preserved"]


def require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    label: str,
    error: Type[BaseException],
) -> None:
    """Raise ``error`` when any of ``columns`` is absent from ``df``.

    Message::

        <label>: missing required columns: ['a', 'b']

    ``columns`` may be any iterable, including a ``set`` -- the reported list
    is sorted, so the message is stable across processes either way.
    """
    present = set(df.columns)
    missing = sorted(column for column in columns if column not in present)
    if missing:
        raise error(f"{label}: missing required columns: {missing}")


def assert_shape_preserved(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    label: str,
    check_index: bool,
    error: Type[BaseException],
) -> None:
    """Raise ``error`` when ``after`` is no longer row-aligned with ``before``.

    Reports the two failure modes separately, with the counts, so the message
    says what actually happened::

        <label>: row count changed: 450465 -> 450464
        <label>: row order or index changed (row count 450465 unchanged)

    ``check_index`` selects whether index identity is part of the contract. It
    is required rather than defaulted, so a length-only site says so out loud.
    """
    if len(after) != len(before):
        raise error(
            f"{label}: row count changed: {len(before)} -> {len(after)}"
        )
    if check_index and not after.index.equals(before.index):
        raise error(
            f"{label}: row order or index changed "
            f"(row count {len(before)} unchanged)"
        )
