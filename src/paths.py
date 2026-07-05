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


def ensure_dir(path: Path) -> Path:
    # Return the path after creation so callers can compose this in expressions.
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> Path:
    # Create parent directories for file outputs without assuming the file exists.
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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
    ARTIFACTS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
):
    ensure_dir(directory)
