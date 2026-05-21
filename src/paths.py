from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "preprocessed"
PREPROCESSED_DIR = CLEAN_DIR
FEATURES_DIR = DATA_DIR / "features"
FINAL_DIR = DATA_DIR / "final"
REPORTS_DIR = PROJECT_ROOT / "reports"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


for directory in (
    DATA_DIR,
    RAW_DIR,
    CLEAN_DIR,
    FEATURES_DIR,
    FINAL_DIR,
    REPORTS_DIR,
):
    ensure_dir(directory)
