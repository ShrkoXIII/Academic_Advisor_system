"""Central filesystem paths for data, model artifacts, and reports."""

import os
from pathlib import Path


# Resolve paths from the installed source file so notebooks and CLI commands can
# run from any working directory without hard-coded absolute paths.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Keep the project data layout in one place; downstream notebooks and jobs should
# import these constants instead of duplicating folder names.
# DATA_DIR can be overridden via ACADEMIC_ADVISOR_DATA_DIR environment variable.
DATA_DIR = Path(os.environ.get("ACADEMIC_ADVISOR_DATA_DIR", PROJECT_ROOT / "data"))

# Captured before the ensure_dir loop below runs, so assert_data_root can tell a
# pre-existing populated root from a tree this import itself just created.
DATA_DIR_FROM_ENV = "ACADEMIC_ADVISOR_DATA_DIR" in os.environ
_DATA_DIR_PREEXISTED = DATA_DIR.exists()
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "preprocessed"
PREPROCESSED_DIR = CLEAN_DIR
FEATURES_DIR = DATA_DIR / "features"
FINAL_DIR = DATA_DIR / "final"
AUDIT_DIR = DATA_DIR / "audit"
MODEL_DATA_DIR = DATA_DIR / "model_data"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
MERGE_DIR = PREPROCESSED_DIR / "merge"
ARCHIVE_DIR = DATA_DIR / "archive"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Canonical pipeline artifacts. Keep file names and reusable subdirectories here
# so scripts and notebooks share one filesystem contract.
SELECTED_MODEL_POPULATION_PATH = FEATURES_DIR / "selected_model_population.parquet"
FEATURE_ENGINEERED_PRIMARY_PATH = FEATURES_DIR / "feature_engineered_primary.parquet"
MODEL_DATA_VERSIONS_DIR = MODEL_DATA_DIR / "versions"
DIPLOMA_TYPE_BUCKET_MAP_PATH = ARTIFACTS_DIR / "diploma_type_bucket_map.json"
MODEL_RUNS_DIR = MODELS_DIR / "runs"
GPA_TREND_REPORTS_DIR = REPORTS_DIR / "gpa_trend"

MODEL_SPLITS = frozenset({"train", "valid", "test"})
MODEL_DATA_GENERATIONS = frozenset({"base", "difficulty", "final"})


def model_split_path(
    split: str,
    generation: str,
    root: str | Path | None = None,
) -> Path:
    """Return a validated model-split artifact path under ``root``.

    ``root`` defaults to the live model-data directory, while versioned builders
    can pass their staging or published directory without duplicating filenames.
    """
    split = str(split).strip().lower()
    generation = str(generation).strip().lower()
    if split not in MODEL_SPLITS:
        raise ValueError(
            f"Unknown model split {split!r}; expected one of {sorted(MODEL_SPLITS)}"
        )
    if generation not in MODEL_DATA_GENERATIONS:
        raise ValueError(
            "Unknown model-data generation "
            f"{generation!r}; expected one of {sorted(MODEL_DATA_GENERATIONS)}"
        )
    split_root = MODEL_DATA_DIR if root is None else Path(root)
    return split_root / f"df_{split}_{generation}.parquet"


def ensure_dir(path: Path) -> Path:
    # Return the path after creation so callers can compose this in expressions.
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> Path:
    # Create parent directories for file outputs without assuming the file exists.
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def assert_data_root(*required_paths) -> None:
    """Refuse to run against a wrong or freshly created empty data root.

    The import-time ensure_dir loop below materializes missing folders, which
    would otherwise mask an unset or mistyped ACADEMIC_ADVISOR_DATA_DIR by
    silently creating an empty tree. Pipeline entry points call this with their
    required input artifacts before any read/write (governance contract 12).
    """
    if not _DATA_DIR_PREEXISTED:
        raise RuntimeError(
            f"Data root {DATA_DIR} did not exist before src.paths was imported "
            f"(ACADEMIC_ADVISOR_DATA_DIR is {'set' if DATA_DIR_FROM_ENV else 'NOT set'}). "
            "Refusing to run against a freshly created empty tree - point "
            "ACADEMIC_ADVISOR_DATA_DIR at the populated data root and restart."
        )
    for required in required_paths:
        required = Path(required)
        if not required.exists() or required.stat().st_size == 0:
            raise RuntimeError(f"Required data artifact missing or empty: {required}")


# Materialize the expected project folders at import time because notebooks use
# these constants interactively and benefit from idempotent setup.
for directory in (
    DATA_DIR,
    RAW_DIR,
    CLEAN_DIR,
    FEATURES_DIR,
    FINAL_DIR,
    AUDIT_DIR,
    MODEL_DATA_DIR,
    MODEL_DATA_VERSIONS_DIR,
    ARTIFACTS_DIR,
    MODELS_DIR,
    MODEL_RUNS_DIR,
    REPORTS_DIR,
    GPA_TREND_REPORTS_DIR,
):
    ensure_dir(directory)
