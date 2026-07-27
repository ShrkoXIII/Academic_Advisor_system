# new-difficulty-logic

**Run ID:** 2026-07-16_1008__new-difficulty-logic
**Date:** 2026-07-16T10:08:22+03:00
**Features:** 39
**Compared with:** none

## What changed

- six-level temporal course-difficulty logic

## Why

Record this isolated training experiment for reproducible comparison.

## Main result

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

- M1 valid fail-class Average Precision: 0.3222 (no baseline)
- M1 valid AUC: 0.8086 (no baseline)
- M1 valid Brier: 0.0807 (no baseline)
- M1 train-valid AUC gap: 0.0622 (no baseline)

## Segment result

- First-semester valid AUC: 0.7366 (no baseline)
- Cold-start GPA valid AUC: 0.7366 (no baseline)

## Important flags

- diploma_gpa has no dedicated missing-value indicator; source-level median fill remains unchanged and unmatched diploma records may be null.
