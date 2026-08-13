<<<<<<< HEAD
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
=======
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
>>>>>>> 770c7a147a9b3b0664121b3c8c165bf6cbfc57a9
"""

from __future__ import annotations

<<<<<<< HEAD
from typing import Iterable, Type
=======
from typing import Iterable, List
>>>>>>> 770c7a147a9b3b0664121b3c8c165bf6cbfc57a9

import pandas as pd


<<<<<<< HEAD
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
=======
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
>>>>>>> 770c7a147a9b3b0664121b3c8c165bf6cbfc57a9
