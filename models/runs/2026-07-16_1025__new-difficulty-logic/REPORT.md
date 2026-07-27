# new-difficulty-logic

**Run ID:** 2026-07-16_1025__new-difficulty-logic
**Date:** 2026-07-16T10:25:48+03:00
**Features:** 39
**Compared with:** 2026-07-12_1513__remove-dead-const

## What changed

- Six-level temporal course-difficulty logic using versioned B2 data

## Why

Record this isolated training experiment for reproducible comparison.

## Main result

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

- M1 valid fail-class Average Precision: not calculated
- M1 valid AUC: 0.7820 -> 0.8086 (+0.0266)
- M1 valid Brier: 0.0855 -> 0.0807 (-0.0048)
- M1 train-valid AUC gap: 0.1239 -> 0.0622 (-0.0617)

## Segment result

- First-semester valid AUC: 0.6368 -> 0.7366 (+0.0998)
- Cold-start GPA valid AUC: 0.6368 -> 0.7366 (+0.0998)

## Important flags

- diploma_gpa has no dedicated missing-value indicator; source-level median fill remains unchanged and unmatched diploma records may be null.
