# Pre-registered M1 R2 covered/uncovered decision plan

**Status: locked before covered/uncovered predictions are loaded or scored.**

This plan resolves one question on the immutable
`2026-07-26_batched_fixes__registration_roster_concurrent` VALID split:

> Does the existing `baseline_41`, `num_leaves=31` R2 challenger provide a
> consistent product-relevant benefit specifically on model-facing uncovered
> course rows versus the existing `baseline_41`, `num_leaves=127` incumbent?

The burden of proof is on R2. This plan is not a training authorization.
Every model is an existing frozen LightGBM binary. TEST remains
`closed_not_read`.

## Frozen comparison

Seeds: 42, 52, 62, 72, 82.

For every seed:

- control: `baseline_41`, current default parameters, `num_leaves=127`;
- challenger: `baseline_41`, identical parameters except `num_leaves=31`.

Exact run folders are taken from
`models/runs/R2_CONFIRMATION_5SEED_REPORT.json`. Before any prediction is
loaded, every pair must pass all existing parity checks:

- immutable TRAIN and VALID hashes match;
- feature contract and ordered features match;
- root and derived seeds match;
- categorical levels match;
- diploma fill behavior/value match;
- reporting threshold matches at 0.80;
- parameters differ only in `num_leaves` (`127 → 31`);
- both M1 binaries exist;
- TEST policy is `closed_not_read`.

Any parity failure stops the task without a decision.

`concurrent_43` is rejected for M1 and is not scored here.
`concurrent_44` is historical only and is not used.
M2 is not rescored or reconsidered.

## Locked segments

- complete VALID: every VALID row;
- covered: `course_difficulty_missing == 0`;
- uncovered: `course_difficulty_missing == 1`;
- `never_in_train`: uncovered and `course_id` absent from TRAIN;
- `thin_history`: uncovered, `course_id` present in TRAIN, and
  `course_history_count < 20`.

`never_in_train` and `thin_history` are reported separately and are not
combined when interpreting cause.

## Locked metrics and delta directions

For every seed, arm, and segment:

- row count;
- fail rate;
- ROC AUC;
- fail-class average precision;
- Brier score;
- fail precision, recall, and F1 at the locked pass-probability threshold 0.80;
- confusion matrix `(TN, FP, FN, TP)`, where fail is label 0.

Every paired delta is `R2 minus control`.

- AUC, fail AP, fail precision, fail recall, and fail F1: positive is
  beneficial.
- Brier: negative is beneficial.

Fail AP is never treated as proof by itself; failure prevalence is reported
for every segment.

Across seeds, report mean, median, population SD, minimum, maximum, and
beneficial/harmful/zero counts.

## Locked R2 adoption rule for M1

R2 is adopted only if **all** of the following are true:

1. On uncovered VALID rows:
   - AUC improves in at least 4 of 5 seeds;
   - Brier improves in at least 4 of 5 seeds;
   - the five-seed mean AUC delta is beneficial;
   - the five-seed mean Brier delta is beneficial.
2. Fail-class average precision on uncovered rows must not show a repeated
   harmful direction:
   - harmful in no more than 2 of 5 seeds.
3. On covered VALID rows, R2 must not show systematic harm:
   - AUC harmful in no more than 2 of 5 seeds;
   - Brier harmful in no more than 2 of 5 seeds.
4. On complete VALID, the existing primary M1 guardrails remain satisfied:
   - no harmful degradation beyond the documented comparison band for VALID
     AUC, fail AP, or Brier.
5. The result must not depend on seed 42 alone.
6. If the rule is not fully met, the incumbent `num_leaves=127` wins.

The documented comparison band is read verbatim from
`models/runs/NOISE_BAND.md`:

- VALID AUC delta: `[-0.000382, +0.001042]`;
- VALID fail-AP delta: `[-0.002045, +0.001544]`;
- VALID Brier delta: `[-0.000108, +0.000119]`.

For clause 4, a harmful breach means:

- AUC delta below `-0.000382`;
- fail-AP delta below `-0.002045`;
- Brier delta above `+0.000119`.

Clause 5 passes only if clauses 1–4 remain true after removing seed 42 and
applying the count requirements proportionally to the four remaining seeds:

- uncovered AUC and Brier beneficial in at least 3 of 4;
- uncovered fail AP harmful in no more than 2 of 4;
- covered AUC and Brier harmful in no more than 2 of 4;
- four-seed mean uncovered AUC beneficial and Brier beneficial;
- complete-VALID mean guardrails remain unbreached.

This operationalization is locked here before results and prevents the phrase
“must not depend on seed 42 alone” from being interpreted after viewing them.

## Decision outputs

If every clause passes:

```text
ADOPT_R2_FOR_M1
```

Otherwise:

```text
KEEP_DEFAULT_127_FOR_M1
```

Regardless of the M1 verdict:

```text
This read-only result does not change M2.
M2 remains concurrent_43 with num_leaves=127.
```

No default, model binary, dataset, promotion marker, deployment path,
inference/recommendation wiring, or TEST policy changes as a result of this
analysis.
