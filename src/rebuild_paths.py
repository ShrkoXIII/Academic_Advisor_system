"""Path resolution for the ``2026-08_temporal_rebuild_v1`` dataset version.

This module exists because the rebuild's artifacts do **not** follow the live
``df_{split}_{generation}.parquet`` naming that :func:`src.paths.model_split_path`
produces.  Phase 1 wrote ``train_base_candidate.parquet`` and its siblings under
``01_split/``, and the later generations do not exist yet.  Rather than teach the
live path helper a second naming scheme, the rebuild gets its own resolver.

Deliberate properties, all of them governance requirements rather than style
choices:

* **The version root is a required argument.**  There is no default and no
  fallback to ``MODEL_DATA_DIR``.  A caller that does not name a root cannot
  accidentally resolve to live data; this module never imports the live root.
* **Every returned path stays inside the given root.**  Split and generation
  names come from closed vocabularies and the basenames are built here, so no
  caller-supplied string reaches the filesystem as a path component.
* **No returned basename can collide with a live artifact name.**  The live
  names are exactly ``df_{split}_{generation}.parquet``; the rebuild names never
  carry the ``df_`` prefix.  A module-level self-check enforces this at import.
* **Missing roots and missing required inputs raise.**  Silence is the failure
  mode this rebuild cannot afford.

Naming convention (chosen here, documented here)
------------------------------------------------

Phase 1 established ``{prefix}_base_candidate.parquet`` where ``prefix`` is
``train``, ``valid``, or ``test_provisional``.  The later generations extend that
same pattern instead of inventing a second one::

    01_split/       train_base_candidate.parquet
                    valid_base_candidate.parquet
                    test_provisional_base_candidate.parquet

    03_features/    train_difficulty_candidate.parquet
                    train_final_candidate.parquet
                    valid_difficulty_candidate.parquet
                    ...
                    test_provisional_final_candidate.parquet

    04_concurrent/  train_concurrent_candidate.parquet
                    train_final_candidate.parquet
                    ...
                    test_provisional_final_candidate.parquet

    05_dataset/     train_dataset_candidate.parquet
                    valid_dataset_candidate.parquet
                    test_provisional_dataset_candidate.parquet

The ``test_provisional`` prefix is not decoration: TEST is built from ``20251``
only and carries ``test_provisional_20251_only = true`` (Decisions_Log.md,
2026-08-03 Amendment 2, Correction 2).  The ``_candidate`` suffix records that
nothing in this version is promoted.

``final`` appears in two stages because the concurrent builder consumes a
``final`` generation and emits a wider one under the same generation name - the
same overlap the live pipeline resolves by writing each build into its own
directory.  Separating the stages keeps a builder from reading and writing one
path, so ``stage`` selects between them: ``features`` is the pre-concurrent
``final``, ``concurrent`` the post-concurrent one.
"""

from __future__ import annotations

from pathlib import Path


REBUILD_VERSION = "2026-08_temporal_rebuild_v1"

# Subdirectories inside the version root. Phase 1 owns 01_split; the feature
# generations that Phase 3 will write live under 03_features and 04_concurrent.
SPLIT_SUBDIR = "01_split"
FEATURES_SUBDIR = "03_features"
CONCURRENT_SUBDIR = "04_concurrent"
DATASET_SUBDIR = "05_dataset"

STAGE_SUBDIRS: dict[str, str] = {
    "split": SPLIT_SUBDIR,
    "features": FEATURES_SUBDIR,
    "concurrent": CONCURRENT_SUBDIR,
}

# The stage each generation belongs to unless the caller names another one.
DEFAULT_STAGE: dict[str, str] = {
    "base": "split",
    "difficulty": "features",
    "final": "features",
    "concurrent": "concurrent",
}

REBUILD_STAGES: tuple[str, ...] = ("split", "features", "concurrent")

REBUILD_SPLITS: tuple[str, ...] = ("train", "valid", "test")
REBUILD_GENERATIONS: tuple[str, ...] = ("base", "difficulty", "concurrent", "final")

# TEST is provisional for this version, and its filenames say so.
SPLIT_FILE_PREFIX: dict[str, str] = {
    "train": "train",
    "valid": "valid",
    "test": "test_provisional",
}

FILE_SUFFIX = "_candidate.parquet"

DIPLOMA_BUCKET_MAP_NAME = "diploma_type_bucket_map.json"


def rebuild_basename(split: str, generation: str) -> str:
    """Return the rebuild filename for ``split``/``generation`` (no directory)."""
    split = _normalize(split, REBUILD_SPLITS, "split")
    generation = _normalize(generation, REBUILD_GENERATIONS, "generation")
    return f"{SPLIT_FILE_PREFIX[split]}_{generation}{FILE_SUFFIX}"


def _normalize(value: str, allowed: tuple[str, ...], label: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in allowed:
        raise ValueError(
            f"Unknown {label} {value!r}; expected one of {list(allowed)}"
        )
    return normalized


def rebuild_version_root(root: str | Path) -> Path:
    """Validate and return the version root.

    ``root`` is required. There is no default, and no fallback to the live
    model-data directory: refusing to guess is the point of this module.
    """
    if root is None:
        raise ValueError(
            "The rebuild version root is required; this resolver has no default "
            "and never falls back to the live model-data directory."
        )
    resolved = Path(root)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Rebuild version root does not exist: {resolved}"
        )
    if not resolved.is_dir():
        raise NotADirectoryError(
            f"Rebuild version root is not a directory: {resolved}"
        )
    return resolved


def rebuild_split_path(
    root: str | Path,
    split: str,
    generation: str,
    *,
    stage: str | None = None,
    must_exist: bool = False,
) -> Path:
    """Return the path of one rebuild split/generation artifact under ``root``.

    ``stage`` defaults per generation (see :data:`DEFAULT_STAGE`) and only needs
    naming for ``final``, which exists both before and after the concurrent
    build.  Pass ``must_exist=True`` when the caller is about to read the file,
    so a missing input fails here with the path named rather than deeper in a
    builder.
    """
    version_root = rebuild_version_root(root)
    generation = _normalize(generation, REBUILD_GENERATIONS, "generation")
    stage = _normalize(
        DEFAULT_STAGE[generation] if stage is None else stage,
        REBUILD_STAGES,
        "stage",
    )
    path = (
        version_root / STAGE_SUBDIRS[stage] / rebuild_basename(split, generation)
    )
    _assert_within(path, version_root)
    if must_exist and not path.exists():
        raise FileNotFoundError(
            f"Required rebuild artifact is missing: {path}"
        )
    return path


def rebuild_generation_paths(
    root: str | Path,
    generation: str,
    *,
    stage: str | None = None,
    must_exist: bool = False,
) -> dict[str, Path]:
    """Return ``{split: path}`` for one generation across all three splits."""
    return {
        split: rebuild_split_path(
            root, split, generation, stage=stage, must_exist=must_exist
        )
        for split in REBUILD_SPLITS
    }


def rebuild_dataset_path(
    root: str | Path,
    split: str,
    *,
    must_exist: bool = False,
) -> Path:
    """Return the assembled model-facing dataset path for one split.

    This is the end of the chain: one file per split carrying the union of the
    named contracts. It is a distinct stage because it is not a feature
    generation - nothing downstream enriches it further.
    """
    version_root = rebuild_version_root(root)
    split = _normalize(split, REBUILD_SPLITS, "split")
    path = (
        version_root
        / DATASET_SUBDIR
        / f"{SPLIT_FILE_PREFIX[split]}_dataset{FILE_SUFFIX}"
    )
    _assert_within(path, version_root)
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Required rebuild artifact is missing: {path}")
    return path


def rebuild_diploma_bucket_map_path(
    root: str | Path,
    *,
    must_exist: bool = False,
) -> Path:
    """Return the version-local diploma bucket map path.

    Amendment 3 (2026-08-03) requires the map to be refitted on the new TRAIN and
    persisted under the version root.  The live map at
    ``data/artifacts/diploma_type_bucket_map.json`` is never written by anything
    that resolves through here.
    """
    version_root = rebuild_version_root(root)
    path = version_root / DIPLOMA_BUCKET_MAP_NAME
    _assert_within(path, version_root)
    if must_exist and not path.exists():
        raise FileNotFoundError(
            f"Required rebuild artifact is missing: {path}"
        )
    return path


def _assert_within(path: Path, root: Path) -> None:
    """Refuse any resolved path that escapes the version root."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        raise ValueError(
            f"Resolved path {path} falls outside the version root {root}"
        ) from None


def _live_basename(split: str, generation: str) -> str:
    """The live naming this module must never reproduce."""
    return f"df_{split}_{generation}.parquet"


# Import-time guarantee: no rebuild basename can shadow a live artifact name.
_LIVE_BASENAMES = frozenset(
    _live_basename(split, generation)
    for split in REBUILD_SPLITS
    for generation in REBUILD_GENERATIONS
)
for _split in REBUILD_SPLITS:
    for _generation in REBUILD_GENERATIONS:
        _name = rebuild_basename(_split, _generation)
        if _name in _LIVE_BASENAMES or _name.startswith("df_"):
            raise AssertionError(
                f"Rebuild basename {_name!r} collides with the live naming scheme"
            )
del _split, _generation, _name
