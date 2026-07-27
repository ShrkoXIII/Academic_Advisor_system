# gpa-trend-feature

**Run ID:** 2026-07-21_1224__gpa-trend-feature
**Date:** 2026-07-21T12:24:14+03:00
**Features:** 41
**Compared with:** 2026-07-16_1025__new-difficulty-logic

## What changed

- Isolated GPA trend delta plus missing indicator; versioned 2026-07-21 data

## Why

Record this isolated training experiment for reproducible comparison.

## Main result

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

- M1 valid fail-class Average Precision: 0.3222 -> 0.3220 (-0.0002)
- M1 valid AUC: 0.8086 -> 0.8092 (+0.0006)
- M1 valid Brier: 0.0807 -> 0.0808 (+0.0001)
- M1 train-valid AUC gap: 0.0622 -> 0.0554 (-0.0068)

## Segment result

- First-semester valid AUC: 0.7366 -> 0.7329 (-0.0037)
- Retake-attempt valid AUC: 0.6726 -> 0.6768 (+0.0042)
- Cold-start GPA valid AUC: 0.7366 -> 0.7329 (-0.0037)

The first-semester feature is always undefined, so an improvement was not expected there.
The retake segment, where GPA trend is usually available, improved by 0.0042 AUC.

## Diagnostic threshold 0.80

This is a post-training readability diagnostic only; it was not used for selection and no
threshold code changed.

- VALID fail precision 0.3307, recall 0.4195, F1 0.3698, warning rate 13.12%.
- TEST (descriptive) fail precision 0.2771, recall 0.3480, F1 0.3085, warning rate 11.25%.

## Important flags

- diploma_gpa has no dedicated missing-value indicator; source-level median fill remains unchanged and unmatched diploma records may be null.
