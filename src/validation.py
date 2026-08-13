"""Detection helpers shared by the repo's data guards.

These functions DETECT. They never raise.

Every call site keeps its own ``raise`` statement, its own exception class,
and its own message text, literally in place. That is deliberate: the guards
being consolidated here span four exception types (``KeyError``,
``ValueError``, ``AssertionError``, and bare ``assert``) and every one of them
has a distinct message. A helper that raised on the caller's behalf would have
to be parameterised by exception class and message anyway, and a bare-``assert``
site routed through it would start firing under ``python -O`` where today it is
stripped. Detection-only keeps all of that unchanged.

So the call-site shape is::

    missing = find_missing_columns(df, REQUIRED)
    if missing:
        raise KeyError(f"...{missing}")          # unchanged, in place

    if shape_changed(df, out, check_index=True):
        raise AssertionError("...")              # unchanged, in place

    assert not shape_changed(a, b, check_index=False), "..."   # still stripped by -O
"""

from __future__ import annotations

from typing import Iterable, List

import pandas as pd


__all__ = ["find_missing_columns", "shape_changed"]


def find_missing_columns(
    df: pd.DataFrame, columns: Iterable[str]
) -> List[str]:
    """Return the entries of ``columns`` absent from ``df``, in argument order.

    The result follows the order of ``columns``, not the order of
    ``df.columns``. Call sites whose current message shows an alphabetically
    sorted list are therefore expected to pass ``sorted(required)`` -- three
    of them build ``required`` as a set literal and print
    ``sorted(required - set(df.columns))``, and passing the set unsorted would
    change their message text.

    Returns a list so the caller can test it for truthiness and interpolate it
    into a message exactly as before.
    """
    existing = set(df.columns)
    return [column for column in columns if column not in existing]


def shape_changed(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    check_index: bool,
) -> bool:
    """Return True when ``after`` is no longer row-aligned with ``before``.

    Mirrors the guard idiom it replaces::

        len(after) != len(before) or not after.index.equals(before.index)

    ``check_index`` is keyword-only and has NO default, on purpose. A site that
    compares lengths only must say so explicitly, and cannot silently acquire
    an index comparison -- or lose one -- because a default changed later.
    ``tests/test_validation.py`` asserts the absence of that default.
    """
    if len(after) != len(before):
        return True
    if check_index and not after.index.equals(before.index):
        return True
    return False
