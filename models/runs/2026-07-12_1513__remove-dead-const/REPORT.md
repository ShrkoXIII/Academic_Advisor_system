# remove-dead-const

**Run ID:** 2026-07-12_1513__remove-dead-const
**Date:** 2026-07-12T15:13:19+03:00
**Features:** 39
**Compared with:** 2026-07-12_1208__baseline-39f__02

## What changed

- remove const while keep diploma.

## Why

has zero gain. 
Record this isolated training experiment for reproducible comparison.

## Main result

- M1 valid AUC: 0.7807 -> 0.7820 (+0.0013)
- M1 valid fail recall/F1: 0.0342/0.0603 -> 0.0361/0.0641 (+0.0019/+0.0038)
- M2 valid MAE: 10.0180 -> 9.9064 (-0.1116)

## Segment result

- First-semester valid AUC: not calculated
- Cold-start GPA valid AUC: not calculated

## Important flags

- diploma_gpa has no dedicated missing-value indicator; source-level median fill remains unchanged and unmatched diploma records may be null.