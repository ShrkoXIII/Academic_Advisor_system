# new-difficulty-logic-0-85

**Run ID:** 2026-07-16_1439__new-difficulty-logic-0-85
**Date:** 2026-07-16T14:39:06+03:00
**Features:** 39
**Compared with:** 2026-07-16_1008__new-difficulty-logic

## What changed

- Six-level temporal course-difficulty logic using versioned B2 data

## Why

Record this isolated training experiment for reproducible comparison.

## Main result

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

- M1 valid fail-class Average Precision: 0.3222 -> 0.3222 (+0.0000)
- M1 valid AUC: 0.8086 -> 0.8086 (+0.0000)
- M1 valid Brier: 0.0807 -> 0.0807 (+0.0000)
- M1 train-valid AUC gap: 0.0622 -> 0.0622 (+0.0000)

## Segment result

- First-semester valid AUC: 0.7366 -> 0.7366 (+0.0000)
- Cold-start GPA valid AUC: 0.7366 -> 0.7366 (+0.0000)

## Important flags

- diploma_gpa has no dedicated missing-value indicator; source-level median fill remains unchanged and unmatched diploma records may be null.
