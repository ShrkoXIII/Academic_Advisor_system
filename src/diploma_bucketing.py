"""Train-fitted bucketing of ``diploma_type_id``.

The rule lived only inside ``note_books/model_eng/03_diploma_type_bucketing.ipynb``
and could therefore not be re-fitted or tested. It is extracted here verbatim:

* rank the **non-null** ``diploma_type_id`` values of TRAIN by frequency and keep
  the top five as their own categories, under their own raw codes;
* every other code seen in TRAIN maps to ``rare_bucket_label`` (6);
* a null, or a code never seen in TRAIN, maps to ``unseen_bucket_label`` (-1);
* refuse to run if any real code equals a reserved label, which would make the
  bucket and the code indistinguishable.

Nothing here is new behaviour. ``tests/test_diploma_bucketing.py`` re-fits the
superseded TRAIN split and asserts the result reproduces the live map at
``data/artifacts/diploma_type_bucket_map.json`` field for field, which is what
makes the extraction checkable rather than merely plausible.

Decisions_Log.md 2026-08-03 Amendment 3 requires the map to be re-fitted on the
rebuild's TRAIN and persisted under its version root; the live map is not
modified. The fitting rule itself is unchanged by that decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import pandas as pd


DIPLOMA_TYPE_COLUMN = "diploma_type_id"
DIPLOMA_BUCKET_COLUMN = "diploma_type_bucket"

RARE_BUCKET_LABEL = 6
UNSEEN_BUCKET_LABEL = -1
TOP_CODE_COUNT = 5

ARTIFACT_NAME = "diploma_type_bucket_map"
TRAIN_MANIFEST_REF = "docs/manifests/baseline_7bc_2026-07-08.json"

NULL_POLICY = "null diploma_type_id -> unseen bucket label"
UNSEEN_POLICY = "non-null codes never seen in train -> unseen bucket label"
RESERVED_LABEL_GUARANTEE = (
    "Step B.3 asserts no real code equals a reserved label in any split"
)
FIT_NOTE = "fitted on the train split ONLY (difficulty generation)"


@dataclass(frozen=True)
class DiplomaBucketMap:
    """The fitted state: which codes keep their identity and which collapse."""

    top_codes: tuple[int, ...]
    code_to_bucket: Mapping[int, int]
    rare_bucket_label: int
    unseen_bucket_label: int
    categories: tuple[int, ...]

    def apply(self, series: pd.Series) -> pd.Series:
        """Map raw ``diploma_type_id`` values to bucket integers."""

        def map_one(value: Any) -> int:
            if pd.isna(value):
                return self.unseen_bucket_label
            return self.code_to_bucket.get(int(value), self.unseen_bucket_label)

        return series.map(map_one)

    def as_categorical(self, series: pd.Series) -> pd.Categorical:
        """Apply the map and pin the category set, as the notebook did."""
        return pd.Categorical(self.apply(series), categories=list(self.categories))


def assert_no_reserved_label_collision(
    frames: Mapping[str, pd.DataFrame],
    *,
    rare_bucket_label: int = RARE_BUCKET_LABEL,
) -> None:
    """Refuse to bucket when a real code equals the rare-bucket label.

    A real ``diploma_type_id`` of 6 would be indistinguishable from the rare
    bucket, so the label has to move before anything is fitted.
    """
    colliding = {
        name: int((frame[DIPLOMA_TYPE_COLUMN] == rare_bucket_label).sum())
        for name, frame in frames.items()
    }
    offenders = {name: count for name, count in colliding.items() if count}
    if offenders:
        raise AssertionError(
            f"diploma_type_id == {rare_bucket_label} exists in {offenders}. "
            "A real diploma code would collide with the rare-bucket label; "
            "move the label outside the real code range before proceeding."
        )


def fit_diploma_bucket_map(
    train: pd.DataFrame,
    *,
    rare_bucket_label: int = RARE_BUCKET_LABEL,
    unseen_bucket_label: int = UNSEEN_BUCKET_LABEL,
    top_code_count: int = TOP_CODE_COUNT,
) -> DiplomaBucketMap:
    """Fit the bucket map on TRAIN only. Validation and test are never read."""
    if DIPLOMA_TYPE_COLUMN not in train.columns:
        raise AssertionError(
            f"{DIPLOMA_TYPE_COLUMN} is missing from the train split; the split "
            "was built from a chain that predates the upstream diploma merge."
        )

    # Nulls are excluded from the ranking, exactly as the notebook did: they are
    # routed to the unseen bucket rather than competing for a top-five slot.
    counts = (
        train[DIPLOMA_TYPE_COLUMN]
        .dropna()
        .value_counts()
        .sort_values(ascending=False)
    )
    top_codes = list(counts.index[:top_code_count])
    top_code_set = set(top_codes)

    all_train_codes = {int(value) for value in train[DIPLOMA_TYPE_COLUMN].dropna().unique()}
    rare_codes = all_train_codes - top_code_set

    code_to_bucket: dict[int, int] = {}
    for code in top_code_set:
        code_to_bucket[int(code)] = int(code)
    for code in rare_codes:
        code_to_bucket[int(code)] = rare_bucket_label

    categories = sorted(int(code) for code in top_code_set) + [
        rare_bucket_label,
        unseen_bucket_label,
    ]

    return DiplomaBucketMap(
        top_codes=tuple(int(code) for code in top_codes),
        code_to_bucket=code_to_bucket,
        rare_bucket_label=int(rare_bucket_label),
        unseen_bucket_label=int(unseen_bucket_label),
        categories=tuple(categories),
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unavailable"


def bucket_map_state(
    bucket_map: DiplomaBucketMap,
    *,
    fit_source_path: str | Path,
    fit_rows: int,
    created: str | None = None,
    git_commit: str | None = None,
    train_manifest_ref: str = TRAIN_MANIFEST_REF,
    fit_note: str = FIT_NOTE,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the persisted artifact dict, in the recorded field order.

    ``fit_note``, ``train_manifest_ref``, and ``extra`` exist so a re-fit on a
    different split can describe itself honestly; their defaults reproduce the
    live artifact exactly.
    """
    state = {
        "artifact": ARTIFACT_NAME,
        "created": created
        or datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit if git_commit is not None else _git_commit(),
        "fit_source": {
            "path": str(fit_source_path),
            "rows": int(fit_rows),
            "note": fit_note,
        },
        "train_manifest_ref": train_manifest_ref,
        "top_codes": [int(code) for code in bucket_map.top_codes],
        "code_to_bucket_map": {
            str(int(code)): int(bucket)
            for code, bucket in bucket_map.code_to_bucket.items()
        },
        "rare_bucket_label": int(bucket_map.rare_bucket_label),
        "unseen_bucket_label": int(bucket_map.unseen_bucket_label),
        "null_policy": NULL_POLICY,
        "unseen_policy": UNSEEN_POLICY,
        "reserved_label_guarantee": RESERVED_LABEL_GUARANTEE,
        "categories": [int(code) for code in bucket_map.categories],
    }
    if extra:
        state.update(extra)
    return state


def load_bucket_map(path: str | Path) -> DiplomaBucketMap:
    """Rehydrate a persisted map without re-fitting."""
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    return DiplomaBucketMap(
        top_codes=tuple(int(code) for code in state["top_codes"]),
        code_to_bucket={
            int(code): int(bucket)
            for code, bucket in state["code_to_bucket_map"].items()
        },
        rare_bucket_label=int(state["rare_bucket_label"]),
        unseen_bucket_label=int(state["unseen_bucket_label"]),
        categories=tuple(int(code) for code in state["categories"]),
    )


def save_bucket_map(state: Mapping[str, Any], path: str | Path) -> Path:
    """Write the artifact. Refuses to overwrite an existing map."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite diploma bucket map: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return path
