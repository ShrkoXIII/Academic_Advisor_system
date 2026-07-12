"""Copy the root-level legacy baseline into a hash-verified persistent run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from src import experiment_tracking
from src.paths import MODELS_DIR


REQUIRED_ARTIFACTS = [
    "m1_pass_model.lgbm",
    "m2_grade_model.lgbm",
    "m1_feature_importance.csv",
    "m2_feature_importance.csv",
    "metrics.json",
    "feature_contract.json",
    "training_curves.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="baseline-39f")
    parser.add_argument("--note", default="Migrated existing root-level 39-feature baseline artifacts.")
    parser.add_argument("--source", default=str(MODELS_DIR))
    parser.add_argument("--out", default=str(MODELS_DIR))
    args = parser.parse_args(argv)

    source = Path(args.source)
    missing = [name for name in REQUIRED_ARTIFACTS if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Legacy baseline is missing required artifacts: {missing}")
    contract = json.loads((source / "feature_contract.json").read_text(encoding="utf-8"))
    if contract.get("n_features") != 39 or len(contract.get("features", [])) != 39:
        raise ValueError("Legacy baseline migration requires exactly the 39-feature contract.")
    results = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    context = experiment_tracking.resolve_output(Path(args.out), args.run_name, args.note, None)

    source_hashes = {}
    for name in REQUIRED_ARTIFACTS:
        src = source / name
        dst = context.output_dir / name
        source_hashes[name] = sha256(src)
        shutil.copy2(src, dst)
        if sha256(dst) != source_hashes[name]:
            raise RuntimeError(f"SHA-256 verification failed after copying {name}")

    report = (
        f"# {context.case_name}\n\n"
        f"**Run ID:** {context.run_id}\n"
        f"**Date:** {context.created_at.isoformat(timespec='seconds')}\n"
        "**Features:** 39\n"
        "**Compared with:** none\n\n"
        "## What changed\n\n"
        f"- {context.note}\n\n"
        "## Why\n\n"
        "Preserve the existing 39-feature baseline before new experiments are added.\n\n"
        "## Main result\n\n"
        "- Result delta: not calculated\n\n"
        "## Segment result\n\n"
        "- Legacy metrics did not persist validation segment AUCs.\n\n"
        "## Important flags\n\n"
        "- Original root artifacts were copied, not moved; all seven copied files passed SHA-256 verification.\n"
    )
    (context.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    valid_m1 = results["m1_pass_classifier"]["valid"]
    valid_m2 = results["m2_grade_regressor"]["valid"]
    experiment_tracking.append_leaderboard_row(
        context.artifact_root / "runs" / "leaderboard.csv",
        {
            "run_id": context.run_id,
            "case_name": context.case_name,
            "created_at": context.created_at.isoformat(timespec="seconds"),
            "n_features": 39,
            "m1_valid_auc": valid_m1["auc"],
            "m1_valid_fail_precision": valid_m1["fail_precision"],
            "m1_valid_fail_recall": valid_m1["fail_recall"],
            "m1_valid_fail_f1": valid_m1["fail_f1"],
            "m2_valid_mae": valid_m2["mae"],
            "m2_valid_r2": valid_m2["r2"],
            "baseline_run_id": "",
            "one_line_change": context.note,
        },
    )
    print(f"Migrated baseline to {context.output_dir}")
    print("Original root artifacts were not changed.")


if __name__ == "__main__":
    main()
