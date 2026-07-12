"""Small, append-only tracking helpers for model-training experiments."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


LEADERBOARD_FIELDS = [
    "run_id",
    "case_name",
    "created_at",
    "n_features",
    "m1_valid_auc",
    "m1_valid_fail_precision",
    "m1_valid_fail_recall",
    "m1_valid_fail_f1",
    "m2_valid_mae",
    "m2_valid_r2",
    "baseline_run_id",
    "one_line_change",
]


@dataclass(frozen=True)
class RunContext:
    artifact_root: Path
    output_dir: Path
    persistent: bool
    run_id: Optional[str]
    case_name: Optional[str]
    created_at: datetime
    note: str
    compare_to: Optional[str]


def normalize_case_name(value: str) -> str:
    """Return the readable lowercase slug used in persistent run IDs."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    if not slug:
        raise ValueError("--run-name must contain at least one letter or number.")
    return slug


def resolve_output(
    artifact_root: Path,
    run_name: Optional[str],
    note: Optional[str],
    compare_to: Optional[str],
    now: Optional[datetime] = None,
) -> RunContext:
    """Allocate a never-overwritten persistent run or the disposable quick folder."""
    artifact_root = Path(artifact_root)
    created_at = now or datetime.now().astimezone()
    note = (note or "").strip()

    if not run_name:
        if compare_to:
            raise ValueError("--compare-to requires --run-name; quick runs are disposable.")
        out_dir = artifact_root / "quick" / "latest"
        out_dir.mkdir(parents=True, exist_ok=True)
        return RunContext(
            artifact_root=artifact_root,
            output_dir=out_dir,
            persistent=False,
            run_id=None,
            case_name=None,
            created_at=created_at,
            note=note,
            compare_to=None,
        )

    case_name = normalize_case_name(run_name)
    runs_dir = artifact_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{created_at.strftime('%Y-%m-%d_%H%M')}__{case_name}"

    suffix = 1
    while True:
        run_id = stem if suffix == 1 else f"{stem}__{suffix:02d}"
        out_dir = runs_dir / run_id
        try:
            out_dir.mkdir()
            break
        except FileExistsError:
            suffix += 1

    return RunContext(
        artifact_root=artifact_root,
        output_dir=out_dir,
        persistent=True,
        run_id=run_id,
        case_name=case_name,
        created_at=created_at,
        note=note,
        compare_to=compare_to,
    )


def load_baseline_metrics(context: RunContext) -> Optional[Dict[str, Any]]:
    """Load a requested persistent baseline without modifying it."""
    if not context.compare_to:
        return None
    if not context.persistent:
        raise ValueError("Only persistent runs may compare against a baseline.")
    run_id = context.compare_to
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("--compare-to must be a run ID, not a path.")
    path = context.artifact_root / "runs" / run_id / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Baseline run metrics not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _value(metrics: Optional[Dict[str, Any]], *keys: str) -> Optional[float]:
    current: Any = metrics
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return float(current) if current is not None else None


def _delta(old: Optional[float], new: Optional[float]) -> str:
    if old is None or new is None:
        return "not calculated"
    return f"{new - old:+.4f}"


def _metric_line(label: str, old: Optional[float], new: Optional[float]) -> str:
    if old is None or new is None:
        return f"- {label}: not calculated"
    return f"- {label}: {old:.4f} -> {new:.4f} ({_delta(old, new)})"


def _segment_auc(metrics: Optional[Dict[str, Any]], name: str) -> Optional[float]:
    return _value(metrics, "segments", "valid", name, "auc")


def _report_text(
    context: RunContext,
    results: Dict[str, Any],
    segment_metrics: Dict[str, Any],
    baseline: Optional[Dict[str, Any]],
    n_features: int,
    flags: Iterable[str],
) -> str:
    current = {**results, "segments": segment_metrics}
    compared_with = context.compare_to or "none"
    note = context.note or "No note supplied."
    why = (
        "Evaluate whether pre-admission diploma GPA and diploma-type categories improve validation performance, especially for cold-start students."
        if context.case_name == "add-diploma-signals"
        else "Record this isolated training experiment for reproducible comparison."
    )
    old_recall = _value(baseline, "m1_pass_classifier", "valid", "fail_recall")
    old_f1 = _value(baseline, "m1_pass_classifier", "valid", "fail_f1")
    new_recall = _value(current, "m1_pass_classifier", "valid", "fail_recall")
    new_f1 = _value(current, "m1_pass_classifier", "valid", "fail_f1")
    if None in {old_recall, old_f1, new_recall, new_f1}:
        recall_f1 = "- M1 valid fail recall/F1: not calculated"
    else:
        recall_f1 = (
            f"- M1 valid fail recall/F1: {old_recall:.4f}/{old_f1:.4f} -> "
            f"{new_recall:.4f}/{new_f1:.4f} "
            f"({_delta(old_recall, new_recall)}/{_delta(old_f1, new_f1)})"
        )
    flags = list(flags)
    flag_lines = "\n".join(f"- {flag}" for flag in flags) if flags else "- None"

    return f"""# {context.case_name}

**Run ID:** {context.run_id}
**Date:** {context.created_at.isoformat(timespec='seconds')}
**Features:** {n_features}
**Compared with:** {compared_with}

## What changed

- {note}

## Why

{why}

## Main result

{_metric_line('M1 valid AUC', _value(baseline, 'm1_pass_classifier', 'valid', 'auc'), _value(current, 'm1_pass_classifier', 'valid', 'auc'))}
{recall_f1}
{_metric_line('M2 valid MAE', _value(baseline, 'm2_grade_regressor', 'valid', 'mae'), _value(current, 'm2_grade_regressor', 'valid', 'mae'))}

## Segment result

{_metric_line('First-semester valid AUC', _segment_auc(baseline, 'first_semester'), _segment_auc(current, 'first_semester'))}
{_metric_line('Cold-start GPA valid AUC', _segment_auc(baseline, 'cold_start_gpa'), _segment_auc(current, 'cold_start_gpa'))}

## Important flags

{flag_lines}
"""


def finalize_persistent_run(
    context: RunContext,
    results: Dict[str, Any],
    segment_metrics: Dict[str, Any],
    n_features: int,
    flags: Iterable[str],
    baseline: Optional[Dict[str, Any]],
) -> None:
    """Write compact run metadata only after all core model artifacts exist."""
    if not context.persistent:
        return
    payload = {**results, "segments": segment_metrics}
    (context.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (context.output_dir / "REPORT.md").write_text(
        _report_text(context, results, segment_metrics, baseline, n_features, flags),
        encoding="utf-8",
    )
    row = {
        "run_id": context.run_id,
        "case_name": context.case_name,
        "created_at": context.created_at.isoformat(timespec="seconds"),
        "n_features": n_features,
        "m1_valid_auc": _value(results, "m1_pass_classifier", "valid", "auc"),
        "m1_valid_fail_precision": _value(results, "m1_pass_classifier", "valid", "fail_precision"),
        "m1_valid_fail_recall": _value(results, "m1_pass_classifier", "valid", "fail_recall"),
        "m1_valid_fail_f1": _value(results, "m1_pass_classifier", "valid", "fail_f1"),
        "m2_valid_mae": _value(results, "m2_grade_regressor", "valid", "mae"),
        "m2_valid_r2": _value(results, "m2_grade_regressor", "valid", "r2"),
        "baseline_run_id": context.compare_to or "",
        "one_line_change": context.note,
    }
    append_leaderboard_row(context.artifact_root / "runs" / "leaderboard.csv", row)


def append_leaderboard_row(path: Path, row: Dict[str, Any]) -> None:
    """Append one compact row and reject an incompatible pre-existing CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        with path.open("r", newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), [])
        if header != LEADERBOARD_FIELDS:
            raise ValueError(f"Leaderboard header mismatch in {path}")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEADERBOARD_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in LEADERBOARD_FIELDS})
