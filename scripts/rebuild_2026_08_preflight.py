"""Phase 1 preflight for ``2026-08_temporal_rebuild_v1``: immutable baseline capture.

Writes, in this order and before any candidate dataset exists:

* ``00_preflight/current_artifacts_baseline_manifest.csv``
* ``00_preflight/preflight_environment.json``
* ``00_preflight/pinned_version_constants.csv``
* ``00_preflight/preflight_report.md``

The manifest is the ONLY proof of data immutability available to this project.
``DATA_DIR`` resolves outside the Git tree by design, so ``git status`` cannot
witness changes there; SHA-256 can. Once written the manifest is never
regenerated -- a later phase re-hashes and diffs against it.

Reads only. Creates no dataset and touches nothing under the live model-data
directory.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import paths as P  # noqa: E402

REBUILD_VERSION = "2026-08_temporal_rebuild_v1"
FROZEN_VERSION = "2026-07-26_batched_fixes__registration_roster_concurrent"

VERSION_ROOT = P.MODEL_DATA_VERSIONS_DIR / REBUILD_VERSION
PREFLIGHT_DIR = VERSION_ROOT / "00_preflight"

MANIFEST_CSV = PREFLIGHT_DIR / "current_artifacts_baseline_manifest.csv"
ENVIRONMENT_JSON = PREFLIGHT_DIR / "preflight_environment.json"
PINNED_CSV = PREFLIGHT_DIR / "pinned_version_constants.csv"
REPORT_MD = PREFLIGHT_DIR / "preflight_report.md"

MANIFEST_FIELDS = [
    "absolute_path",
    "relative_logical_role",
    "file_size_bytes",
    "modified_timestamp",
    "sha256",
    "exists",
]

# A dataset-version string looks like 2026-07-26_... or 2026-08_...; match the
# literal so a constant pointing at ANY version is inventoried, not just the
# frozen one.
VERSION_STRING = re.compile(r"\b(20\d\d-\d\d(?:-\d\d)?_[A-Za-z0-9_.-]+)\b")

SCAN_SUFFIXES = {".py", ".ipynb", ".md", ".json", ".cfg", ".toml", ".ini", ".txt"}
SKIP_DIR_PARTS = {".git", ".venv", "__pycache__", ".ipynb_checkpoints", "node_modules"}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path, role: str) -> dict[str, object]:
    """Hash one artifact. Byte-level only -- no column or outcome is interpreted.

    Hashing df_test_*.parquet is deliberate and is NOT a TEST read: it produces
    a fixed-length digest of the file's bytes and exposes no row, label, or
    schema to this process or to any model.
    """
    exists = path.is_file()
    if not exists:
        return {
            "absolute_path": str(path),
            "relative_logical_role": role,
            "file_size_bytes": "",
            "modified_timestamp": "",
            "sha256": "",
            "exists": False,
        }
    stat = path.stat()
    return {
        "absolute_path": str(path),
        "relative_logical_role": role,
        "file_size_bytes": stat.st_size,
        "modified_timestamp": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(timespec="seconds"),
        "sha256": sha256_of(path),
        "exists": True,
    }


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    def add(path: Path, role: str) -> None:
        key = str(path).lower()
        if key in seen:
            return
        seen.add(key)
        rows.append(record(path, role))

    def add_tree(root: Path, role_prefix: str) -> None:
        if not root.exists():
            return
        for path in sorted(root.rglob("*")):
            if path.is_file():
                add(path, f"{role_prefix}/{path.relative_to(root).as_posix()}")

    # Source extracts and every intermediate the rebuild could read.
    add_tree(P.RAW_DIR, "raw")
    add_tree(P.CLEAN_DIR, "preprocessed")
    add_tree(P.FEATURES_DIR, "features")
    add_tree(P.FINAL_DIR, "final")
    add_tree(P.ARTIFACTS_DIR, "artifacts")
    add_tree(P.AUDIT_DIR, "audit")

    # Live split generations -- the artifacts most at risk of accidental
    # overwrite, plus their frozen counterparts.
    for split in ("train", "valid", "test"):
        for generation in ("base", "difficulty", "concurrent", "final"):
            add(
                P.MODEL_DATA_DIR / f"df_{split}_{generation}.parquet",
                f"live_model_data/df_{split}_{generation}",
            )
    add_tree(P.MODEL_DATA_DIR / "versions", "model_data_versions")
    for extra in sorted(P.MODEL_DATA_DIR.glob("*")):
        if extra.is_file():
            add(extra, f"live_model_data/{extra.name}")

    # Model binaries, contracts, and run provenance.
    add_tree(P.MODELS_DIR, "models")

    # Governance and yardstick documents.
    add(PROJECT_ROOT / "Decisions_Log.md", "governance/Decisions_Log.md")
    add(PROJECT_ROOT / "CLAUDE.md", "governance/CLAUDE.md")
    add(P.MODEL_RUNS_DIR / "NOISE_BAND.md", "governance/NOISE_BAND.md")
    add(PROJECT_ROOT / "docs" / "pipeline_rules.md", "governance/pipeline_rules.md")

    return rows


def scan_pinned_versions() -> list[dict[str, object]]:
    """Inventory every hardcoded dataset-version string in the repository."""
    found: list[dict[str, object]] = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SCAN_SUFFIXES:
            continue
        if SKIP_DIR_PARTS & set(path.parts):
            continue
        # Never scan our own outputs back into the inventory.
        if VERSION_ROOT in path.parents:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in VERSION_STRING.finditer(line):
                value = match.group(1)
                # A version string is only interesting when it names a real
                # dataset version directory or the rebuild version itself.
                if not (
                    value == REBUILD_VERSION
                    or (P.MODEL_DATA_VERSIONS_DIR / value).exists()
                ):
                    continue
                name_match = re.search(
                    r"([A-Z_][A-Z0-9_]*)\s*(?::[^=]+)?=\s*[^=]*" + re.escape(value),
                    line,
                )
                found.append(
                    {
                        "file": path.relative_to(PROJECT_ROOT).as_posix(),
                        "line": lineno,
                        "constant_name": name_match.group(1) if name_match else "",
                        "value": value,
                        "recommendation": (
                            "keep_pinned_to_frozen"
                            if value == FROZEN_VERSION
                            else "rebuild_version_self_reference"
                            if value == REBUILD_VERSION
                            else "review_parameterise_later"
                        ),
                    }
                )
    return found


def main() -> int:
    if MANIFEST_CSV.exists():
        raise SystemExit(
            f"STOP: {MANIFEST_CSV} already exists. The baseline manifest is "
            "captured once and never regenerated; refusing to overwrite it."
        )
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)

    started = datetime.now().astimezone()
    rows = collect_rows()

    with MANIFEST_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    pinned = scan_pinned_versions()
    with PINNED_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file", "line", "constant_name", "value", "recommendation"],
        )
        writer.writeheader()
        writer.writerows(pinned)

    environment = {
        "rebuild_version": REBUILD_VERSION,
        "frozen_reference_version": FROZEN_VERSION,
        "execution_timestamp": started.isoformat(timespec="seconds"),
        "repository_root": str(PROJECT_ROOT),
        "data_dir": str(P.DATA_DIR),
        "data_dir_from_env": P.DATA_DIR_FROM_ENV,
        "academic_advisor_data_dir_env": __import__("os").environ.get(
            "ACADEMIC_ADVISOR_DATA_DIR"
        ),
        "model_data_dir": str(P.MODEL_DATA_DIR),
        "model_data_versions_dir": str(P.MODEL_DATA_VERSIONS_DIR),
        "version_root": str(VERSION_ROOT),
        "current_version_txt": (
            (P.MODEL_DATA_DIR / "CURRENT_VERSION.txt").read_text(encoding="utf-8")
            if (P.MODEL_DATA_DIR / "CURRENT_VERSION.txt").is_file()
            else None
        ),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "manifest_rows": len(rows),
        "manifest_files_hashed": sum(1 for r in rows if r["exists"]),
        "manifest_files_missing": sum(1 for r in rows if not r["exists"]),
        "pinned_version_constants_found": len(pinned),
    }
    ENVIRONMENT_JSON.write_text(
        json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total_bytes = sum(int(r["file_size_bytes"] or 0) for r in rows if r["exists"])
    lines = [
        "# Phase 1 preflight — `2026-08_temporal_rebuild_v1`",
        "",
        "Read-only baseline capture. No candidate dataset existed when this ran.",
        "",
        "## Why a hash manifest",
        "",
        "`DATA_DIR` resolves outside the Git tree, so `git status` cannot witness",
        "changes to the data. SHA-256 can. This manifest is captured once, before any",
        "Phase 1 artifact is written, and is never regenerated. A later phase re-hashes",
        "the same paths and diffs against this file to prove nothing was mutated.",
        "",
        "## Coverage",
        "",
        f"- Artifacts recorded: **{len(rows)}**",
        f"- Present and hashed: **{environment['manifest_files_hashed']}**",
        f"- Recorded as absent: **{environment['manifest_files_missing']}**",
        f"- Total bytes hashed: **{total_bytes:,}**",
        "",
        "Covered: raw extracts, cleaned/merged datasets, feature datasets, every live",
        "split generation, every dataset version directory (including the frozen",
        f"`{FROZEN_VERSION}`), mapping files, model binaries, generated feature",
        "contracts, run provenance, and the governance documents.",
        "",
        "### TEST hashing is not a TEST read",
        "",
        "`df_test_*.parquet` files are hashed. Hashing reads bytes and emits a digest;",
        "it exposes no row, no column, no label, and no schema to this process or to any",
        "model. Declaration 1 item 6 forbids *reading* the frozen version's",
        "`df_test_final.parquet` — no such read occurs here or anywhere in Phase 1.",
        "",
        "## Resolved environment",
        "",
        f"- `DATA_DIR` = `{P.DATA_DIR}`",
        f"- resolved from `ACADEMIC_ADVISOR_DATA_DIR`: **{P.DATA_DIR_FROM_ENV}**",
        f"- `MODEL_DATA_DIR` = `{P.MODEL_DATA_DIR}`",
        f"- `MODEL_DATA_VERSIONS_DIR` = `{P.MODEL_DATA_VERSIONS_DIR}`",
        f"- version root = `{VERSION_ROOT}`",
        "",
        "## Pinned dataset-version constants",
        "",
        f"Found **{len(pinned)}** hardcoded dataset-version references; see",
        "`pinned_version_constants.csv`. None was changed in this phase. An",
        "uninventoried hardcoded version is a silent path to reading the wrong dataset.",
        "",
        "## Path policy — why `model_split_path` is not used for candidates",
        "",
        "`src/paths.py` validates `generation` against exactly `base`, `difficulty`,",
        "`concurrent`, `final`, and always yields the basename `df_{split}_{generation}",
        ".parquet`. Phase 1 candidates are neither a live generation nor safe to name",
        "with a live basename: writing `df_train_final.parquet` into any root invites",
        "exactly the confusion this rebuild is guarding against. Rather than widen the",
        "validated vocabulary — which would be a production change made for a candidate's",
        "convenience — Phase 1 writes distinct basenames directly under its versioned",
        "phase directories, resolved from `MODEL_DATA_VERSIONS_DIR`. `model_split_path`",
        "is left untouched.",
        "",
        "## Prerequisites",
        "",
        "Verified before this script ran; evidence is in the task report.",
        "",
        "| Prerequisite | Result |",
        "|---|---|",
        "| A — supplying a version dir changes what is read | PASS |",
        "| B1 — case-insensitive filename collision | PASS (resolved `d297af8`) |",
        "| B2 — candidate basenames vs live basenames | PASS (no collision) |",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"manifest rows      : {len(rows)}")
    print(f"  hashed           : {environment['manifest_files_hashed']}")
    print(f"  absent           : {environment['manifest_files_missing']}")
    print(f"  bytes hashed     : {total_bytes:,}")
    print(f"pinned constants   : {len(pinned)}")
    print(f"wrote              : {MANIFEST_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
