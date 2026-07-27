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
    if new is None:
        return f"- {label}: not calculated"
    if old is None:
        return f"- {label}: {new:.4f} (no baseline)"
    return f"- {label}: {old:.4f} -> {new:.4f} ({_delta(old, new)})"


def _segment_auc(metrics: Optional[Dict[str, Any]], name: str) -> Optional[float]:
    return _value(metrics, "segments", "valid", name, "auc")


def _segment_block(
    baseline: Optional[Dict[str, Any]],
    current: Dict[str, Any],
) -> str:
    """Report every segment already present in the run, with baseline deltas.

    Segment definitions live in model_training._segment_masks; this only
    reports what that produced. It never defines a new segment.
    """
    segments = (current.get("segments") or {}).get("valid") or {}
    if not segments:
        return "- No segment metrics recorded."
    lines = []
    for name in sorted(segments):
        entry = segments[name] or {}
        n = entry.get("n")
        status = entry.get("status")
        if status and status != "ok":
            lines.append(f"- {name} (n={n}): {status}")
            continue
        lines.append(
            _metric_line(
                f"{name} valid AUC (n={n})",
                _segment_auc(baseline, name),
                _segment_auc(current, name),
            )
        )
    # Two segments have historically had identical populations; surface it.
    first = segments.get("first_semester") or {}
    cold = segments.get("cold_start_gpa") or {}
    if first.get("n") is not None and first.get("n") == cold.get("n"):
        same_auc = first.get("auc") == cold.get("auc")
        lines.append(
            f"- **NOTE:** `first_semester` and `cold_start_gpa` have identical "
            f"population size (n={first.get('n')})"
            + (" and identical AUC" if same_auc else " but different AUC")
            + " — treat them as one segment until the definitions are separated."
        )
    return "\n".join(lines)


def _auc_gap(metrics: Optional[Dict[str, Any]]) -> Optional[float]:
    stored = _value(
        metrics, "m1_pass_classifier", "valid", "train_valid_auc_gap"
    )
    if stored is not None:
        return stored
    train_auc = _value(metrics, "m1_pass_classifier", "train", "auc")
    valid_auc = _value(metrics, "m1_pass_classifier", "valid", "auc")
    if train_auc is None or valid_auc is None:
        return None
    return train_auc - valid_auc


def _settings_block(
    run_settings: Optional[Dict[str, Any]],
    baseline: Optional[Dict[str, Any]],
) -> str:
    """Render run settings and flag any that differ from the baseline run."""
    if not run_settings:
        return "- Not recorded for this run."
    baseline_settings = (baseline or {}).get("run_settings") or {}
    lines = []
    for key in sorted(run_settings):
        value = run_settings[key]
        old = baseline_settings.get(key)
        if baseline_settings and old != value:
            lines.append(f"- {key}: `{old}` -> `{value}`  **DIFFERS FROM BASELINE**")
        else:
            lines.append(f"- {key}: `{value}`")
    if baseline_settings:
        differing = [
            key for key in run_settings
            if key in baseline_settings and baseline_settings[key] != run_settings[key]
        ]
        controlled = [k for k in differing if k in {"feature_contract", "feature_count"}]
        uncontrolled = [k for k in differing if k not in controlled]
        lines.append("")
        lines.append(
            f"- Intended delta: {controlled or 'none'}; "
            f"unintended differences: {uncontrolled or 'none'}"
        )
    return "\n".join(lines)


def _report_text(
    context: RunContext,
    results: Dict[str, Any],
    segment_metrics: Dict[str, Any],
    baseline: Optional[Dict[str, Any]],
    n_features: int,
    flags: Iterable[str],
    run_settings: Optional[Dict[str, Any]] = None,
) -> str:
    current = {**results, "segments": segment_metrics}
    compared_with = context.compare_to or "none"
    note = context.note or "No note supplied."
    why = (
        "Evaluate whether pre-admission diploma GPA and diploma-type categories improve validation performance, especially for cold-start students."
        if context.case_name == "add-diploma-signals"
        else "Record this isolated training experiment for reproducible comparison."
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

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

{_metric_line('M1 valid fail-class Average Precision', _value(baseline, 'm1_pass_classifier', 'valid', 'fail_avg_precision'), _value(current, 'm1_pass_classifier', 'valid', 'fail_avg_precision'))}
{_metric_line('M1 valid AUC', _value(baseline, 'm1_pass_classifier', 'valid', 'auc'), _value(current, 'm1_pass_classifier', 'valid', 'auc'))}
{_metric_line('M1 valid Brier', _value(baseline, 'm1_pass_classifier', 'valid', 'brier'), _value(current, 'm1_pass_classifier', 'valid', 'brier'))}
{_metric_line('M1 train-valid AUC gap', _auc_gap(baseline), _auc_gap(current))}

## M2 regressor

{_metric_line('M2 valid MAE', _value(baseline, 'm2_grade_regressor', 'valid', 'mae'), _value(current, 'm2_grade_regressor', 'valid', 'mae'))}
{_metric_line('M2 valid RMSE', _value(baseline, 'm2_grade_regressor', 'valid', 'rmse'), _value(current, 'm2_grade_regressor', 'valid', 'rmse'))}
{_metric_line('M2 valid R2', _value(baseline, 'm2_grade_regressor', 'valid', 'r2'), _value(current, 'm2_grade_regressor', 'valid', 'r2'))}

## Fail-class at the locked reporting threshold (VALID)

{_metric_line('M1 valid fail precision', _value(baseline, 'm1_pass_classifier', 'valid', 'fail_precision'), _value(current, 'm1_pass_classifier', 'valid', 'fail_precision'))}
{_metric_line('M1 valid fail recall', _value(baseline, 'm1_pass_classifier', 'valid', 'fail_recall'), _value(current, 'm1_pass_classifier', 'valid', 'fail_recall'))}
{_metric_line('M1 valid fail F1', _value(baseline, 'm1_pass_classifier', 'valid', 'fail_f1'), _value(current, 'm1_pass_classifier', 'valid', 'fail_f1'))}

## Segment result (VALID; existing segment definitions only)

{_segment_block(baseline, current)}

## Run settings

{_settings_block(run_settings, baseline)}

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
    run_settings: Optional[Dict[str, Any]] = None,
) -> None:
    """Write compact run metadata only after all core model artifacts exist.

    ``run_settings`` records the feature contract, reporting threshold, seed and
    input paths inside metrics.json, so two runs can never be compared without
    the settings that produced them.
    """
    if not context.persistent:
        return
    payload: Dict[str, Any] = {**results, "segments": segment_metrics}
    if run_settings:
        payload["run_settings"] = run_settings
    (context.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (context.output_dir / "REPORT.md").write_text(
        _report_text(
            context, results, segment_metrics, baseline, n_features, flags,
            run_settings,
        ),
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
