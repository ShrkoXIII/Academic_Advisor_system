# Phase 3 — predecessor-prior PILOT

```
STATUS: PILOT — PENDING/UNREVIEWED MAPPINGS — NOT FOR PROMOTION
This run measures whether the predecessor-prior mechanism is worth reviewing.
It does not authorize freezing, promoting, or wiring any model, regardless of
result. Human review of course_link_proposed.csv remains a precondition for
any production use.
```

## Governance status

- Every row of `course_link_proposed.csv` and `course_split_candidates.csv` is still `approval_status = pending`. **Zero rows have been human-reviewed**, and this run changed none of them.
- This pilot exists to answer one question: *is the predecessor-prior mechanism worth the human review effort it would take to approve it?* It is not a substitute for that review.
- Nothing here authorizes freezing M1/M2, opening TEST, promoting a dataset version, or wiring anything into production.
- TEST was never read, globbed, or stat-ed. `Decisions_Log.md` was not edited (a ready-to-copy entry is at the end). Nothing was pushed.

### Conflicts with `CLAUDE.md`, flagged rather than resolved silently

1. `CLAUDE.md` §3 names the **regularization pass** as the only active workstream and §8 forbids improvising outside a scoped prompt. This task is a different, explicitly authorized workstream that permits a dataset write and model work. The prompt was followed; the conflict is recorded here.
2. `CLAUDE.md` §5 says *"never copy datasets into new folders"*. The task requires `df_train_final.parquet` inside the new version directory and requires it to be byte-identical. It is a byte copy, verified by SHA-256 against the frozen file. This does create a second physical copy of TRAIN; it is a pilot artifact and should be deleted rather than promoted.
3. `CLAUDE.md` §6 selection metrics are unchanged, but the acceptance clauses below were pre-registered by the task prompt, not by `docs/EXPERIMENT_REGULARIZATION_PLAN.md`, which governs a different experiment.

## Mechanism recap

For a VALID row whose `course_id` appears in the Phase 2T link table as a `new_course_id` with `relationship_type` in `{successor, consolidated_into}` (i.e. it carries a `weight_hint`), the two historical difficulty estimates are replaced by the weight-blended TRAIN estimate of that course's predecessors:

- **Per-predecessor estimate**, the precedence Phase 0's `answer_q6` already validated: substitute the predecessor's `course_id` into the row's own `degree_course_key` and use the **Level-1** (degree+course) TRAIN estimate when that substituted key has support in `fit_difficulty_state(TRAIN)`; otherwise use the predecessor's **Level-2** (course-across-degrees) TRAIN estimate.
- Across the 456 distinct (degree, new course) pairs there were 772 predecessor contributions: **1 resolved at Level-1** and **771 at Level-2**. The new-plan degrees almost never carry TRAIN history for an old-plan course, so the mechanism is in practice a Level-2 substitution.
- **Blend**: `successor` (one predecessor, `weight_hint = 1.0`) takes that predecessor's estimate directly; `consolidated_into` takes the `weight_hint`-weighted average across *all* contributing predecessors.
- **Shrinkage identity**: because this is a single frozen-TRAIN snapshot, an eligible course's own history is structurally zero, so `(n_new*local + k*prior)/(n_new+k)` collapses to the prior term **exactly** at `n_new = 0`. This substitution is not an approximation of that formula — it *is* that formula at `n_new = 0`.

Changed on eligible rows: `course_pass_rate_historical`, `course_avg_mark_historical`. Deliberately **unchanged on every row**: `course_difficulty_missing` (still 1), `course_history_count` (still 0), `difficulty_fallback_level`, `course_retake_rate_historical`, `difficulty_group_support_count`, `course_is_new`, `course_low_support`.

### Downstream propagation (one change, not two)

Concurrent-group features are mechanically derived from `d = 1 - course_pass_rate_historical`, so they were rebuilt through the **unmodified** `src.concurrent_group_features` builder over the frozen registration roster with only its inputs changed. Before the rebuild, the builder was shown to reproduce all eight frozen concurrent columns **exactly** from the frozen roster, so any difference afterwards is attributable to this one change and nothing else.

### Audit-only columns

Six columns were added, using the names from the original Section 9 spec: `course_history_count_predecessor`, `course_cross_plan_prior_used`, `course_cross_plan_prior_weight`, `course_cross_plan_relationship_type`, `course_identity_confidence`, `course_difficulty_source_level`. **None of them is in the M1 or M2 feature list**, and the build asserts that before writing anything.

## Eligible-set confirmation

| Quantity | Value |
|---|---:|
| Eligible (weighted-link) courses | **80** |
| Eligible VALID rows | **16,269** |
| — of which `successor` courses | 66 |
| — of which `consolidated_into` courses | 14 |
| Scope `shared` / `specific` courses | 21 / 59 |
| Courses with ≥1 credit-changed weighted predecessor | 20 |
| Roster occurrences substituted (peer side) | 17,350 |

Phase 2T reported **82 courses / 17,036 VALID rows** for the *union* of weighted, structural, and manual relationships. This pilot's eligibility rule excludes `split_from` / `merged_from`, which is exactly the difference and it reconciles to the row: the two structural courses `1183.111` and `1192.111` account for **2 courses / 767 rows**, and `80 + 2 = 82`, `16,269 + 767 = 17,036`. The recount is therefore not a discrepancy — it is the preregistered exclusion, and no unexplained drift exists.

## Pre-training verification (Section 3)

| # | Check | Result |
|---|---|---|
| 0 | TRAIN byte-identical to frozen TRAIN | **PASS** — SHA-256 `8aaff32aeac5b37506b24584d56913b5…` identical |
| 0 | `fit_difficulty_state(TRAIN)` reproduces all 9 frozen VALID difficulty columns | **PASS** — 0 mismatches |
| 0 | Unmodified concurrent builder reproduces all 8 frozen concurrent columns | **PASS** — 0 mismatches |
| 1 | Rows differing in the two substituted columns ⊆ eligible set | **PASS** — **0** rows changed outside the eligible set |
| 1 | Every eligible row carries its assigned value | **PASS** — 16,147 of 16,269 moved numerically; **122 were exact no-ops** (see note) |
| 2 | No unexposed covered / untouched-relationship row changed in any locked contract column | **PASS** — 0 leaks |
| 3 | M1 `baseline_41` and M2 `concurrent_43` feature **lists** identical to the locked contracts | **PASS** — byte-identical; only *values* changed |

**Note on the 122 no-op rows.** All belong to course `1201.111`. Its Level-5 fallback group (`requirement_type 4` + 4 credits, TRAIN support 71) consists of exactly the same TRAIN rows as its predecessor `1036.111`'s Level-2 group (support 71). The frozen fallback value therefore already *was* the predecessor estimate, and the substitution is a genuine no-op — a set identity, not a tolerance artifact. Those rows are still counted as directly eligible everywhere below; their predictions simply cannot move.

## Row sets (addendum §2 / §3)

| Row set | n |
|---|---:|
| VALID total | 156,097 |
| `directly_eligible_rows` | 16,269 |
| `propagation_exposed_rows` (exposed only, not directly eligible) | 15,885 |
| Completely unexposed | 123,943 |
| Never-in-TRAIN (all 182 link-table courses) | 25,627 |
| `covered_unexposed` (Clause-0 sanity) | 121,382 |
| `untouched_uncovered_unexposed` (Clause-0 sanity) | 1,306 |

Changed rows per column, pilot VALID vs frozen VALID:

| Column | Changed rows |
|---|---:|
| `course_pass_rate_historical` (direct substitution) | 16,147 |
| `course_avg_mark_historical` (direct substitution) | 16,147 |
| `concurrent_peer_difficulty_mean` (propagation) | 30,452 |
| `concurrent_peer_difficulty_max` (propagation) | 17,977 |
| `concurrent_peer_difficulty_missing` (propagation) | 0 |
| `concurrent_peer_set_empty` (propagation) | 0 |
| `concurrent_peer_difficulty_values_missing` (propagation) | 0 |
| `concurrent_peer_observed_count` (propagation) | 0 |
| `concurrent_peer_weak_ratio` (propagation) | 0 |
| `concurrent_peer_same_req_type_ratio` (propagation) | 0 |
| every other frozen column | **0** |

## Experimental design — paired, not two trainings

The pilot TRAIN frame is byte-identical to the frozen TRAIN frame and the contracts and hyperparameters are unchanged, so for a given seed the two arms have identical training inputs and training a second nominal model would measure nothing. Per the addendum, **one locked model artifact per seed was evaluated on both VALID frames**, with the artifact's SHA-256 recorded for both arms.

This is not a shortcut: an eligible course is *never-in-TRAIN* by construction, so no retraining on this data could ever learn from the substituted values. The mechanism is inherently an inference-time intervention on VALID, and the paired evaluation is its complete measurement.

Locked specs, not retuned: **M1 = `baseline_41`, `num_leaves=127`**; **M2 = `concurrent_43`, `num_leaves=127`**; seeds 42/52/62/72/82; reporting threshold 0.80.

Existing 5-seed runs at exactly these specs on the frozen version were found and reused. Each was verified before use: recorded contract name, seed, `num_leaves` / `min_child_samples` / `reg_lambda`, `dataset_version`, `test_policy = closed_not_read`, and recorded TRAIN/VALID SHA-256 matching the frozen files on disk — and then, evidence-first, **its recorded frozen-VALID metrics were recomputed from the booster on disk and had to match**.

| Seed | M1 source run (`baseline_41`) | M2 source run (`concurrent_43`) |
|---|---|---|
| 42 | `2026-07-26_1551__baseline-41-gpa-trend-control` | `2026-07-27_1327__seed42-concurrent-43-drop-dead-missing-flag` |
| 52 | `2026-07-27_1027__seed52-baseline-41-gpa-trend-control` | `2026-07-27_1328__seed52-concurrent-43-drop-dead-missing-flag` |
| 62 | `2026-07-27_1031__seed62-baseline-41-gpa-trend-control` | `2026-07-27_1329__seed62-concurrent-43-drop-dead-missing-flag` |
| 72 | `2026-07-27_1035__seed72-baseline-41-gpa-trend-control` | `2026-07-27_1330__seed72-concurrent-43-drop-dead-missing-flag` |
| 82 | `2026-07-27_1038__seed82-baseline-41-gpa-trend-control` | `2026-07-27_1331__seed82-concurrent-43-drop-dead-missing-flag` |

**Provenance caveat, stated rather than buried.** Every reused run recorded a dirty working tree at training time (seeds 42, 52, 62, 72, 82 for M1, and the M2 runs likewise), so bit-exact retrainability cannot be *proved* from the artifacts alone. It does not affect this measurement: both arms use the same booster file, so the comparison is internally valid for any locked-spec model. It would matter for a model-vs-model comparison, which this is not.

## Segmented results — M1 (`baseline_41`, threshold-independent)

Sign convention: delta = with-prior − baseline. Higher-better: AUC, fail-AP. Lower-better: Brier. "Beyond band?" counts seeds whose delta falls outside the published `NOISE_BAND.md` range **in either direction**.

| Segment | n | Metric | Baseline (5-seed mean) | With prior (5-seed mean) | Mean delta | Seeds improved | Beyond band? |
|---|---:|---|---:|---:|---:|---:|---|
| Whole VALID frame (model level) | 156,097 | `auc` | 0.809523 | 0.809039 | -0.000484 | 0/5 | 3/5 |
| Whole VALID frame (model level) | 156,097 | `fail_avg_precision` | 0.322717 | 0.323218 | +0.000501 | 3/5 | 0/5 |
| Whole VALID frame (model level) | 156,097 | `brier` | 0.080749 | 0.080710 | -0.000039 | 4/5 | 0/5 |
| 1. Overall uncovered — all 182 never-in-TRAIN courses | 25,627 | `auc` | 0.766385 | 0.763815 | -0.002570 | 0/5 | 4/5 |
| 1. Overall uncovered — all 182 never-in-TRAIN courses | 25,627 | `fail_avg_precision` | 0.337007 | 0.338350 | +0.001343 | 2/5 | 3/5 |
| 1. Overall uncovered — all 182 never-in-TRAIN courses | 25,627 | `brier` | 0.108339 | 0.108102 | -0.000237 | 4/5 | 5/5 |
| 2. Affected segment — the directly eligible rows | 16,269 | `auc` | 0.765004 | 0.761177 | -0.003827 | 0/5 | 4/5 |
| 2. Affected segment — the directly eligible rows | 16,269 | `fail_avg_precision` | 0.352728 | 0.353654 | +0.000925 | 2/5 | 5/5 |
| 2. Affected segment — the directly eligible rows | 16,269 | `brier` | 0.114970 | 0.114597 | -0.000373 | 4/5 | 5/5 |
| 3a. Affected ∩ scope = shared | 6,781 | `auc` | 0.795912 | 0.785373 | -0.010539 | 0/5 | 5/5 |
| 3a. Affected ∩ scope = shared | 6,781 | `fail_avg_precision` | 0.405819 | 0.395102 | -0.010716 | 1/5 | 4/5 |
| 3a. Affected ∩ scope = shared | 6,781 | `brier` | 0.116003 | 0.115723 | -0.000279 | 3/5 | 5/5 |
| 3b. Affected ∩ scope = specific | 9,488 | `auc` | 0.741772 | 0.743432 | +0.001660 | 3/5 | 5/5 |
| 3b. Affected ∩ scope = specific | 9,488 | `fail_avg_precision` | 0.317558 | 0.324065 | +0.006507 | 4/5 | 4/5 |
| 3b. Affected ∩ scope = specific | 9,488 | `brier` | 0.114232 | 0.113792 | -0.000439 | 5/5 | 4/5 |
| 4a. Affected ∩ credit-changed predecessor | 8,302 | `auc` | 0.757302 | 0.748868 | -0.008435 | 0/5 | 5/5 |
| 4a. Affected ∩ credit-changed predecessor | 8,302 | `fail_avg_precision` | 0.372851 | 0.366104 | -0.006747 | 1/5 | 4/5 |
| 4a. Affected ∩ credit-changed predecessor | 8,302 | `brier` | 0.127615 | 0.127719 | +0.000104 | 2/5 | 4/5 |
| 4b. Affected ∩ credit-unchanged predecessors | 7,967 | `auc` | 0.769323 | 0.771286 | +0.001962 | 4/5 | 4/5 |
| 4b. Affected ∩ credit-unchanged predecessors | 7,967 | `fail_avg_precision` | 0.328253 | 0.340657 | +0.012404 | 5/5 | 5/5 |
| 4b. Affected ∩ credit-unchanged predecessors | 7,967 | `brier` | 0.101793 | 0.100923 | -0.000870 | 5/5 | 4/5 |
| 5. SANITY — covered rows, propagation-unexposed | 121,382 | `auc` | 0.820814 | 0.820814 | +0.000000 | 0/5 | 0/5 |
| 5. SANITY — covered rows, propagation-unexposed | 121,382 | `fail_avg_precision` | 0.325609 | 0.325609 | +0.000000 | 0/5 | 0/5 |
| 5. SANITY — covered rows, propagation-unexposed | 121,382 | `brier` | 0.074944 | 0.074944 | +0.000000 | 0/5 | 0/5 |
| 6. SANITY — untouched uncovered rows, propagation-unexposed | 1,306 | `auc` | 0.778968 | 0.778968 | +0.000000 | 0/5 | 0/5 |
| 6. SANITY — untouched uncovered rows, propagation-unexposed | 1,306 | `fail_avg_precision` | 0.190649 | 0.190649 | +0.000000 | 0/5 | 0/5 |
| 6. SANITY — untouched uncovered rows, propagation-unexposed | 1,306 | `brier` | 0.055947 | 0.055947 | +0.000000 | 0/5 | 0/5 |
| SANITY (superset) — every completely unexposed row | 123,943 | `auc` | 0.820084 | 0.820084 | +0.000000 | 0/5 | 0/5 |
| SANITY (superset) — every completely unexposed row | 123,943 | `fail_avg_precision` | 0.324954 | 0.324954 | +0.000000 | 0/5 | 0/5 |
| SANITY (superset) — every completely unexposed row | 123,943 | `brier` | 0.075110 | 0.075110 | +0.000000 | 0/5 | 0/5 |
| DIAGNOSTIC — propagation-exposed only, not directly eligible | 15,885 | `auc` | 0.755905 | 0.755905 | +0.000000 | 0/5 | 0/5 |
| DIAGNOSTIC — propagation-exposed only, not directly eligible | 15,885 | `fail_avg_precision` | 0.265355 | 0.265355 | +0.000000 | 0/5 | 0/5 |
| DIAGNOSTIC — propagation-exposed only, not directly eligible | 15,885 | `brier` | 0.089696 | 0.089696 | +0.000000 | 0/5 | 0/5 |

`train_valid_auc_gap` is reported **only at the overall model level** — there is no corresponding TRAIN population for a never-in-TRAIN course, so a segment-level gap would be `N/A — no corresponding TRAIN segment`.

| Seed | Gap baseline | Gap with prior | Delta |
|---|---:|---:|---:|
| 42 | 0.055384 | 0.055564 | +0.000180 |
| 52 | 0.068735 | 0.069526 | +0.000791 |
| 62 | 0.061577 | 0.061780 | +0.000203 |
| 72 | 0.058200 | 0.058631 | +0.000431 |
| 82 | 0.046400 | 0.047215 | +0.000815 |
| **mean** | | | **+0.000484** |

The gap moves only because VALID AUC moved; TRAIN AUC is identical by construction, so a *widening* gap here is the same fact as a falling VALID AUC, not independent evidence.

## Segmented results — M2 (`concurrent_43`)

| Segment | n | Metric | Baseline (5-seed mean) | With prior (5-seed mean) | Mean delta | Seeds improved | Beyond band? |
|---|---:|---|---:|---:|---:|---:|---|
| Whole VALID frame (model level) | 156,097 | `mae` | 9.5630 | 9.5824 | +0.0194 | 0/5 | 0/5 |
| Whole VALID frame (model level) | 156,097 | `rmse` | 12.8474 | 12.8662 | +0.0187 | 0/5 | 0/5 |
| Whole VALID frame (model level) | 156,097 | `r2` | 0.3527 | 0.3508 | -0.0019 | 0/5 | 0/5 |
| 1. Overall uncovered — all 182 never-in-TRAIN courses | 25,627 | `mae` | 11.3131 | 11.4319 | +0.1188 | 0/5 | 5/5 |
| 1. Overall uncovered — all 182 never-in-TRAIN courses | 25,627 | `rmse` | 14.8020 | 14.9028 | +0.1008 | 0/5 | 3/5 |
| 1. Overall uncovered — all 182 never-in-TRAIN courses | 25,627 | `r2` | 0.2759 | 0.2660 | -0.0099 | 0/5 | 3/5 |
| 2. Affected segment — the directly eligible rows | 16,269 | `mae` | 11.7868 | 11.9649 | +0.1781 | 0/5 | 5/5 |
| 2. Affected segment — the directly eligible rows | 16,269 | `rmse` | 15.3240 | 15.4652 | +0.1412 | 0/5 | 5/5 |
| 2. Affected segment — the directly eligible rows | 16,269 | `r2` | 0.2870 | 0.2738 | -0.0132 | 0/5 | 5/5 |
| 3a. Affected ∩ scope = shared | 6,781 | `mae` | 12.3848 | 12.6381 | +0.2533 | 0/5 | 5/5 |
| 3a. Affected ∩ scope = shared | 6,781 | `rmse` | 16.1872 | 16.3242 | +0.1370 | 0/5 | 4/5 |
| 3a. Affected ∩ scope = shared | 6,781 | `r2` | 0.3234 | 0.3119 | -0.0115 | 0/5 | 3/5 |
| 3b. Affected ∩ scope = specific | 9,488 | `mae` | 11.3594 | 11.4839 | +0.1244 | 0/5 | 5/5 |
| 3b. Affected ∩ scope = specific | 9,488 | `rmse` | 14.6759 | 14.8206 | +0.1447 | 0/5 | 4/5 |
| 3b. Affected ∩ scope = specific | 9,488 | `r2` | 0.2422 | 0.2272 | -0.0150 | 0/5 | 5/5 |
| 4a. Affected ∩ credit-changed predecessor | 8,302 | `mae` | 12.6922 | 12.9442 | +0.2521 | 0/5 | 5/5 |
| 4a. Affected ∩ credit-changed predecessor | 8,302 | `rmse` | 16.3385 | 16.5514 | +0.2129 | 0/5 | 5/5 |
| 4a. Affected ∩ credit-changed predecessor | 8,302 | `r2` | 0.2719 | 0.2528 | -0.0191 | 0/5 | 5/5 |
| 4b. Affected ∩ credit-unchanged predecessors | 7,967 | `mae` | 10.8434 | 10.9445 | +0.1011 | 0/5 | 5/5 |
| 4b. Affected ∩ credit-unchanged predecessors | 7,967 | `rmse` | 14.1899 | 14.2454 | +0.0554 | 1/5 | 1/5 |
| 4b. Affected ∩ credit-unchanged predecessors | 7,967 | `r2` | 0.3051 | 0.2997 | -0.0054 | 1/5 | 1/5 |
| 5. SANITY — covered rows, propagation-unexposed | 121,382 | `mae` | 9.1052 | 9.1052 | +0.0000 | 0/5 | 0/5 |
| 5. SANITY — covered rows, propagation-unexposed | 121,382 | `rmse` | 12.2772 | 12.2772 | +0.0000 | 0/5 | 0/5 |
| 5. SANITY — covered rows, propagation-unexposed | 121,382 | `r2` | 0.3849 | 0.3849 | +0.0000 | 0/5 | 0/5 |
| 6. SANITY — untouched uncovered rows, propagation-unexposed | 1,306 | `mae` | 9.6236 | 9.6236 | +0.0000 | 0/5 | 0/5 |
| 6. SANITY — untouched uncovered rows, propagation-unexposed | 1,306 | `rmse` | 12.9744 | 12.9744 | +0.0000 | 0/5 | 0/5 |
| 6. SANITY — untouched uncovered rows, propagation-unexposed | 1,306 | `r2` | 0.2821 | 0.2821 | +0.0000 | 0/5 | 0/5 |
| SANITY (superset) — every completely unexposed row | 123,943 | `mae` | 9.1352 | 9.1352 | +0.0000 | 0/5 | 0/5 |
| SANITY (superset) — every completely unexposed row | 123,943 | `rmse` | 12.3191 | 12.3191 | +0.0000 | 0/5 | 0/5 |
| SANITY (superset) — every completely unexposed row | 123,943 | `r2` | 0.3826 | 0.3826 | +0.0000 | 0/5 | 0/5 |
| DIAGNOSTIC — propagation-exposed only, not directly eligible | 15,885 | `mae` | 10.6238 | 10.6318 | +0.0080 | 0/5 | 0/5 |
| DIAGNOSTIC — propagation-exposed only, not directly eligible | 15,885 | `rmse` | 14.0480 | 14.0583 | +0.0103 | 2/5 | 0/5 |
| DIAGNOSTIC — propagation-exposed only, not directly eligible | 15,885 | `r2` | 0.2089 | 0.2078 | -0.0012 | 2/5 | 0/5 |

## Threshold-dependent guards at the locked 0.80 cut

Per the addendum, fail recall and fail F1 are **mandatory secondary non-regression guards**, not readability-only. Pass-class recall/F1 are shown for readability.

| Segment | Metric | Baseline (5-seed mean) | With prior (5-seed mean) | Mean delta | Seeds not declining |
|---|---|---:|---:|---:|---:|
| Whole VALID frame (model level) | `fail_precision` | 0.3302 | 0.3286 | -0.0016 | 1/5 |
| Whole VALID frame (model level) | `fail_recall` | 0.4223 | 0.4288 | +0.0065 | 4/5 |
| Whole VALID frame (model level) | `fail_f1` | 0.3704 | 0.3720 | +0.0016 | 4/5 |
| Whole VALID frame (model level) | `recall` | 0.9011 | 0.8989 | -0.0022 | 1/5 |
| Whole VALID frame (model level) | `f1` | 0.9158 | 0.9150 | -0.0008 | 1/5 |
| 1. Overall uncovered — all 182 never-in-TRAIN courses | `fail_precision` | 0.3280 | 0.3216 | -0.0063 | 1/5 |
| 1. Overall uncovered — all 182 never-in-TRAIN courses | `fail_recall` | 0.4735 | 0.5025 | +0.0289 | 4/5 |
| 1. Overall uncovered — all 182 never-in-TRAIN courses | `fail_f1` | 0.3860 | 0.3918 | +0.0057 | 4/5 |
| 1. Overall uncovered — all 182 never-in-TRAIN courses | `recall` | 0.8385 | 0.8246 | -0.0138 | 1/5 |
| 1. Overall uncovered — all 182 never-in-TRAIN courses | `f1` | 0.8708 | 0.8649 | -0.0059 | 1/5 |
| 2. Affected segment — the directly eligible rows | `fail_precision` | 0.3359 | 0.3265 | -0.0094 | 1/5 |
| 2. Affected segment — the directly eligible rows | `fail_recall` | 0.4982 | 0.5405 | +0.0423 | 4/5 |
| 2. Affected segment — the directly eligible rows | `fail_f1` | 0.3996 | 0.4068 | +0.0072 | 4/5 |
| 2. Affected segment — the directly eligible rows | `recall` | 0.8211 | 0.7990 | -0.0221 | 1/5 |
| 2. Affected segment — the directly eligible rows | `f1` | 0.8588 | 0.8492 | -0.0097 | 1/5 |
| 3a. Affected ∩ scope = shared | `fail_precision` | 0.3797 | 0.3491 | -0.0307 | 1/5 |
| 3a. Affected ∩ scope = shared | `fail_recall` | 0.5396 | 0.6244 | +0.0848 | 4/5 |
| 3a. Affected ∩ scope = shared | `fail_f1` | 0.4418 | 0.4474 | +0.0056 | 4/5 |
| 3a. Affected ∩ scope = shared | `recall` | 0.8288 | 0.7787 | -0.0502 | 1/5 |
| 3a. Affected ∩ scope = shared | `f1` | 0.8645 | 0.8417 | -0.0227 | 1/5 |
| 3b. Affected ∩ scope = specific | `fail_precision` | 0.3054 | 0.3066 | +0.0012 | 3/5 |
| 3b. Affected ∩ scope = specific | `fail_recall` | 0.4662 | 0.4757 | +0.0094 | 4/5 |
| 3b. Affected ∩ scope = specific | `fail_f1` | 0.3684 | 0.3726 | +0.0042 | 3/5 |
| 3b. Affected ∩ scope = specific | `recall` | 0.8156 | 0.8133 | -0.0023 | 1/5 |
| 3b. Affected ∩ scope = specific | `f1` | 0.8548 | 0.8542 | -0.0006 | 2/5 |
| 4a. Affected ∩ credit-changed predecessor | `fail_precision` | 0.3538 | 0.3451 | -0.0086 | 2/5 |
| 4a. Affected ∩ credit-changed predecessor | `fail_recall` | 0.5167 | 0.5465 | +0.0299 | 4/5 |
| 4a. Affected ∩ credit-changed predecessor | `fail_f1` | 0.4177 | 0.4226 | +0.0048 | 4/5 |
| 4a. Affected ∩ credit-changed predecessor | `recall` | 0.8006 | 0.7833 | -0.0173 | 1/5 |
| 4a. Affected ∩ credit-changed predecessor | `f1` | 0.8416 | 0.8341 | -0.0075 | 1/5 |
| 4b. Affected ∩ credit-unchanged predecessors | `fail_precision` | 0.3126 | 0.3040 | -0.0087 | 1/5 |
| 4b. Affected ∩ credit-unchanged predecessors | `fail_recall` | 0.4730 | 0.5322 | +0.0592 | 4/5 |
| 4b. Affected ∩ credit-unchanged predecessors | `fail_f1` | 0.3754 | 0.3866 | +0.0112 | 4/5 |
| 4b. Affected ∩ credit-unchanged predecessors | `recall` | 0.8414 | 0.8146 | -0.0268 | 1/5 |
| 4b. Affected ∩ credit-unchanged predecessors | `f1` | 0.8757 | 0.8640 | -0.0118 | 1/5 |
| 5. SANITY — covered rows, propagation-unexposed | `fail_precision` | 0.3335 | 0.3335 | +0.0000 | 5/5 |
| 5. SANITY — covered rows, propagation-unexposed | `fail_recall` | 0.4184 | 0.4184 | +0.0000 | 5/5 |
| 5. SANITY — covered rows, propagation-unexposed | `fail_f1` | 0.3711 | 0.3711 | +0.0000 | 5/5 |
| 5. SANITY — covered rows, propagation-unexposed | `recall` | 0.9112 | 0.9112 | +0.0000 | 5/5 |
| 5. SANITY — covered rows, propagation-unexposed | `f1` | 0.9237 | 0.9237 | +0.0000 | 5/5 |
| 6. SANITY — untouched uncovered rows, propagation-unexposed | `fail_precision` | 0.2432 | 0.2432 | +0.0000 | 5/5 |
| 6. SANITY — untouched uncovered rows, propagation-unexposed | `fail_recall` | 0.2386 | 0.2386 | +0.0000 | 5/5 |
| 6. SANITY — untouched uncovered rows, propagation-unexposed | `fail_f1` | 0.2407 | 0.2407 | +0.0000 | 5/5 |
| 6. SANITY — untouched uncovered rows, propagation-unexposed | `recall` | 0.9496 | 0.9496 | +0.0000 | 5/5 |
| 6. SANITY — untouched uncovered rows, propagation-unexposed | `f1` | 0.9490 | 0.9490 | +0.0000 | 5/5 |
| SANITY (superset) — every completely unexposed row | `fail_precision` | 0.3332 | 0.3332 | +0.0000 | 5/5 |
| SANITY (superset) — every completely unexposed row | `fail_recall` | 0.4179 | 0.4179 | +0.0000 | 5/5 |
| SANITY (superset) — every completely unexposed row | `fail_f1` | 0.3708 | 0.3708 | +0.0000 | 5/5 |
| SANITY (superset) — every completely unexposed row | `recall` | 0.9110 | 0.9110 | +0.0000 | 5/5 |
| SANITY (superset) — every completely unexposed row | `f1` | 0.9235 | 0.9235 | +0.0000 | 5/5 |
| DIAGNOSTIC — propagation-exposed only, not directly eligible | `fail_precision` | 0.2996 | 0.2996 | +0.0000 | 5/5 |
| DIAGNOSTIC — propagation-exposed only, not directly eligible | `fail_recall` | 0.3445 | 0.3445 | +0.0000 | 5/5 |
| DIAGNOSTIC — propagation-exposed only, not directly eligible | `fail_f1` | 0.3196 | 0.3196 | +0.0000 | 5/5 |
| DIAGNOSTIC — propagation-exposed only, not directly eligible | `recall` | 0.8999 | 0.8999 | +0.0000 | 5/5 |
| DIAGNOSTIC — propagation-exposed only, not directly eligible | `f1` | 0.9085 | 0.9085 | +0.0000 | 5/5 |

## Clause 0 — mandatory sanity, checked before any other verdict

Row-level prediction identity on the propagation-**unexposed** sanity segments, using the same model artifact for both arms. Legitimate concurrent propagation is excluded from these segments by construction and is reported separately as `indirect_propagation_only`; it is not classified as leakage.

| Sanity segment | n | Max abs prediction difference | Rows over tolerance |
|---|---:|---:|---:|
| `covered_unexposed` | 121,382 | 0.0e+00 | **0** |
| `untouched_uncovered_unexposed` | 1,306 | 0.0e+00 | **0** |
| `completely_unexposed` | 123,943 | 0.0e+00 | **0** |

Tolerance `1e-12` (the repository's existing `CHANGE_ATOL`). Across all 5 seeds × both models × all three sanity segments the maximum absolute prediction difference is **exactly 0.0** and **0 rows** exceed tolerance — the predictions are bit-identical, not merely within tolerance. The additional aggregate `NOISE_BAND.md` check is trivially satisfied: every metric delta on these segments is exactly zero.

**Clause 0 verdict: PASS.** The run is valid and Clauses 1–6 may be read.

One further structural confirmation: on `indirect_propagation_only` (15,885 rows) **M1's** deltas are exactly zero in every seed, because `baseline_41` contains no concurrent feature and therefore cannot see the propagation at all. Only M2 responds there.

## Clauses 1–6, on the affected segment

Bounds come from `models/runs/NOISE_BAND.md`; none was invented here. `NOISE_BAND.md` publishes no band for fail recall or fail F1, so for those the only non-invented rule is the sign of the paired delta.

| Clause | Metric | Rule | Per-seed deltas (42/52/62/72/82) | Mean | Verdict |
|---|---|---|---|---:|---|
| 1 | M1 AUC | improves in ≥4/5 seeds beyond the band | -0.00081 / -0.00601 / -0.00033 / -0.00370 / -0.00828 | -0.00383 | **FAIL** |
| 2 | M1 Brier | 5-seed mean does not worsen | -0.00096 / +0.00039 / -0.00072 / -0.00030 / -0.00027 | -0.00037 | **PASS** |
| 3 | M1 fail recall @0.80 | does not decline in ≥4/5 seeds | +0.0660 / -0.0109 / +0.0540 / +0.0290 / +0.0733 | +0.0423 | **PASS** |
| 4 | M1 fail F1 @0.80 | does not decline in ≥4/5 seeds | +0.0107 / +0.0040 / +0.0150 / -0.0057 / +0.0120 | +0.0072 | **PASS** |
| 5 | M2 MAE | improves in ≥4/5 seeds beyond the band | +0.1502 / +0.1874 / +0.2557 / +0.1617 / +0.1356 | +0.1781 | **FAIL** |
| 6 | M2 RMSE | 5-seed mean does not worsen | +0.1080 / +0.1565 / +0.2563 / +0.0839 / +0.1012 | +0.1412 | **FAIL** |

**3 of 6 pass, 3 fail.**

Reading this honestly matters more than the tally. The three clauses that pass (2, 3, 4) are **non-regression guards** — they only say the mechanism did not make calibration or fail-catching worse. The two clauses that actually test whether the mechanism *helps* (1 and 5) both fail, and they fail **with the sign inverted**, not merely by landing inside the band:

- **M1 AUC fell in 5/5 seeds** on the affected segment (mean -0.00383), with 4/5 seeds outside the published band on the worsening side.
- **M2 MAE rose in 5/5 seeds** on the affected segment (mean +0.1781), which is roughly **4×** the band's worst observed noise excursion (+0.046520). M2 RMSE rose in 5/5 as well.

Clause 2's pass is real and worth noting: **Brier improved while AUC fell.** The substituted prior moved absolute risk levels in a helpful direction on these rows while degrading the *ranking* among them.

## Why the direction came out this way

This is a model-free diagnostic: it compares each prior directly against what actually happened on the affected rows, with no model in the loop.

| Segment | n | Actual pass rate | Prior pass, frozen → pilot | Actual mean mark | Prior mark, frozen → pilot |
|---|---:|---:|---|---:|---|
| `affected` | 16,269 | 0.8474 | 0.8485 → 0.8164 | 68.725 | 66.253 → 65.371 |
| `affected_credit_changed` | 8,302 | 0.8274 | 0.8536 → 0.8120 | 68.142 | 66.653 → 65.450 |
| `affected_credit_unchanged` | 7,967 | 0.8682 | 0.8433 → 0.8209 | 69.332 | 65.837 → 65.288 |
| `affected_scope_shared` | 6,781 | 0.8404 | 0.8633 → 0.8077 | 70.472 | 67.927 → 65.032 |
| `affected_scope_specific` | 9,488 | 0.8523 | 0.8380 → 0.8226 | 67.476 | 65.056 → 65.612 |

Row-level accuracy of the prior itself, against the realized outcome:

| Segment | MAE(prior mark vs actual mark) frozen → pilot | MAE(prior pass vs actual pass) frozen → pilot |
|---|---|---|
| `affected` | 14.249 → **14.746** | 0.2513 → **0.2713** |
| `affected_credit_changed` | 14.795 → **15.739** | 0.2592 → **0.2914** |
| `affected_credit_unchanged` | 13.681 → **13.712** | 0.2431 → **0.2503** |
| `affected_scope_shared` | 14.856 → **16.277** | 0.2383 → **0.2811** |
| `affected_scope_specific` | 13.816 → **13.652** | 0.2606 → **0.2642** |

The premise does not hold on this data. On the affected rows the **existing Level-4/Level-5 fallback prior was already closer to the truth** than the predecessor prior: the frozen fallback put the pass rate at 0.8485 against an actual 0.8474 — nearly exact — while the predecessor prior moves it down to 0.8164, away from the outcome. At row level the pass-rate prior gets less accurate in **all five** sub-segments and the mark prior in **four of five** — the one exception is `affected_scope_specific`, whose mark prior improves slightly (13.816 → 13.652) while its pass-rate prior still degrades.

Interpretation, offered as a hypothesis and not as a finding: **the old-plan predecessor courses were systematically harder than their new-plan successors turned out to be.** The mechanism faithfully imports that old difficulty, which is precisely why it hurts. The effect is strongest exactly where one would expect a weaker identity claim — the credit-changed subset (prior-mark MAE 14.795 → 15.739) degrades far more than the credit-unchanged subset (13.681 → 13.712, essentially flat).

That sub-segment structure is the pilot's most useful output for a reviewer: it is consistent across seeds, it points at the credit-changed and `shared`-scope links as the damaging ones, and it is visible in the audit columns without any model.

## Worked examples

### `1175.111` — `successor`, 917 VALID rows

| Field | Before (frozen) | After (pilot) |
|---|---|---|
| `course_pass_rate_historical` | 0.838550 | 0.784862 |
| `course_avg_mark_historical` | 65.0623 | 64.7634 |
| `difficulty_fallback_level` | 5 | 5 (unchanged by design) |
| `course_difficulty_missing` | 1 | 1 (unchanged by design) |
| `course_history_count` | 0 | 0 (unchanged by design) |
| `course_history_count_predecessor` (audit) | not present | 1,524 |
| `course_cross_plan_prior_weight` (audit) | not present | 1.000000 |
| `course_difficulty_source_level` (audit) | not present | `predecessor_prior` |
| `course_identity_confidence` (audit) | not present | `in_lineage` |

Predecessors and the weights actually used in the blend:

| Predecessor | Weight | TRAIN support | Level used |
|---|---:|---:|---:|
| `502.111` | 1.000000 | 1,524 | 2 |

The single declared predecessor was used: confirmed. The course's 10 distinct degree contexts collapse to one post-substitution value because the Level-1 substituted key has no TRAIN support in the new-plan degrees, so all of them fall to the shared Level-2 estimate.

### `1422.111` — `consolidated_into`, 686 VALID rows

| Field | Before (frozen) | After (pilot) |
|---|---|---|
| `course_pass_rate_historical` | 0.780618, 0.792937, 0.888625, 0.921250, 0.927293, 0.973408 | 0.911589 |
| `course_avg_mark_historical` | 60.0190, 62.5508, 68.8896, 70.3357, 70.6627, 78.8604 | 70.0584 |
| `difficulty_fallback_level` | 4, 5 | 4, 5 (unchanged by design) |
| `course_difficulty_missing` | 1 | 1 (unchanged by design) |
| `course_history_count` | 0 | 0 (unchanged by design) |
| `course_history_count_predecessor` (audit) | not present | 9,739 |
| `course_cross_plan_prior_weight` (audit) | not present | 0.950200 |
| `course_difficulty_source_level` (audit) | not present | `predecessor_prior` |
| `course_identity_confidence` (audit) | not present | `in_lineage` |

Predecessors and the weights actually used in the blend:

| Predecessor | Weight | TRAIN support | Level used |
|---|---:|---:|---:|
| `967.111` | 0.950200 | 9,254 | 2 |
| `150.111` | 0.019715 | 192 | 2 |
| `398.111` | 0.015299 | 149 | 2 |
| `173.111` | 0.009139 | 89 | 2 |
| `584.111` | 0.005647 | 55 | 2 |

All 5 declared contributing predecessors were used in the blend: confirmed. The course's 15 distinct degree contexts collapse to one post-substitution value because the Level-1 substituted key has no TRAIN support in the new-plan degrees, so all of them fall to the shared Level-2 estimate.

### `1419.111` — `successor`, 2 VALID rows

| Field | Before (frozen) | After (pilot) |
|---|---|---|
| `course_pass_rate_historical` | 0.826701 | 0.909210 |
| `course_avg_mark_historical` | 64.1759 | 74.5389 |
| `difficulty_fallback_level` | 5 | 5 (unchanged by design) |
| `course_difficulty_missing` | 1 | 1 (unchanged by design) |
| `course_history_count` | 0 | 0 (unchanged by design) |
| `course_history_count_predecessor` (audit) | not present | 48 |
| `course_cross_plan_prior_weight` (audit) | not present | 1.000000 |
| `course_difficulty_source_level` (audit) | not present | `predecessor_prior` |
| `course_identity_confidence` (audit) | not present | `in_lineage` |

Predecessors and the weights actually used in the blend:

| Predecessor | Weight | TRAIN support | Level used |
|---|---:|---:|---:|
| `49.111` | 1.000000 | 48 | 2 |

The single declared predecessor was used: confirmed.

### `1093.111` — Clause-0 covered-row sanity example (VALID row 0)

| Field | Before (frozen) | After (pilot) |
|---|---|---|
| `degree_course_key` | 2.111__1093.111 | 2.111__1093.111 |
| `course_pass_rate_historical` | 0.9499318537859929 | 0.9499318537859929 |
| `course_avg_mark_historical` | 66.19314192659223 | 66.19314192659223 |
| `course_difficulty_missing` | 0 | 0 |
| `course_history_count` | 223 | 223 |
| `difficulty_fallback_level` | 1 | 1 |
| `course_cross_plan_prior_used` (audit) | not present | False |
| `course_difficulty_source_level` (audit) | not present | `1` |

Zero change in every field, and its model prediction is bit-identical in both arms for all five seeds and both models.

## Verdict

# `MIXED`

**This verdict does not authorize promotion.** It does not approve any proposal row, does not validate any individual link, does not permit freezing M1 or M2, does not permit opening TEST, and does not permit wiring anything into production.

`MIXED` is the category the pre-registered arithmetic produces: Clause 0 holds and 3 of 6 clauses pass, which is a split rather than a majority failure. The substance is more one-sided than the label: both improvement clauses failed in the wrong direction across all five seeds, and M2's degradation is several times the published noise band, while the passing clauses are non-regression guards.

The direct answer to the question this pilot was built to ask — *is the predecessor-prior mechanism worth the human review effort?* — is: **not as specified, applied to all weighted links.** On this VALID frame the existing fallback prior is already the better estimate, and importing predecessor difficulty makes both models worse on the rows it touches.

What the pilot does support spending review effort on, if anything: the credit-unchanged / `specific`-scope subset is roughly neutral rather than harmful, and the harm concentrates in the credit-changed and `shared`-scope links. A reviewer who wants to salvage the idea should look there first — but note that **M2 MAE worsened in 5/5 seeds even on the credit-unchanged subset**, so no sub-segment identified here is positive on the M2 side.

### Methodological caveats a reader must carry

- `NOISE_BAND.md` was measured on the **full VALID frame** from *contract-change* deltas across seeds. It is applied here to a 16,269-row **segment** under a *feature-value* change with a *fixed* model. Segment metrics are noisier than full-frame metrics, so the band is, if anything, too permissive here — which makes the failures of Clauses 1 and 5 harder to dismiss, not easier. `CLAUDE.md` §11 already warns the band is the best available yardstick, not an exact one.
- The paired design removes seed-to-seed *training* variation entirely. The five "seeds" here vary only the fixed instrument, not the intervention, so 5/5 consistency is strong evidence about the intervention's direction but is **not** an independent replication in the sense the multi-seed contract experiments used.
- "Overall uncovered" is reported as the 25,627 rows of the 182 link-table courses, per the task. The frozen VALID actually holds 26,882 rows with `course_difficulty_missing == 1`; the extra 1,255 are low-support courses that are not never-in-TRAIN and appear in no link table.

## If the human approves the pending rows

Nothing in this pilot may be reused as a candidate. To turn any part of it into a real model candidate, **this exact pipeline must be re-run restricted to only the rows with `review_decision = approved`**:

1. Reviewers record a decision on each `course_link_proposed.csv` row. The eligibility filter then becomes `relationship_type ∈ {successor, consolidated_into}` **and** `review_decision = approved`, not the current `approval_status = pending` population.
2. `scripts/phase3_predecessor_prior_pilot_build.py` is re-run against that restricted set, producing a different eligible-set size and a different dataset version. Weight renormalization for a consolidation that loses a predecessor is an **open design question**, deliberately unanswered here: this pilot stops rather than redistributing a dropped predecessor's weight.
3. The paired evaluation is re-run on the restricted frame. The clause results in this report **do not carry over** — they describe the unreviewed population, which is a different population.
4. Only then can the mechanism be discussed as a candidate, and only through the ordinary freeze/promotion gates, which this run does not touch.

This pilot deliberately built its dataset version under a `_PENDING_REVIEW` name so it cannot be mistaken for a promotable artifact. It should be deleted, not promoted.

## Artifacts

| Artifact | Path |
|---|---|
| Pilot dataset version | `data/model_data/versions/2026-07-30_predecessor_prior_pilot_PENDING_REVIEW/` |
| Build script | `scripts/phase3_predecessor_prior_pilot_build.py` |
| Paired-evaluation script | `scripts/phase3_predecessor_prior_pilot_evaluate.py` |
| Clause/report script | `scripts/phase3_predecessor_prior_pilot_report.py` |
| Raw paired metrics | `models/runs/phase3_predecessor_prior_pilot/phase3_pilot_evaluation.json` |
| Clause evaluation | `models/runs/phase3_predecessor_prior_pilot/PHASE3_PILOT_CLAUSES.json` |
| Per-arm run directories | `models/runs/<stamp>__predecessor_prior_pilot_seed{N}_{baseline,withprior}/` |

## Governance entry (ready to copy — `Decisions_Log.md` was NOT edited)

> Phase 3 ran the predecessor-prior mechanism as a pilot on unreviewed (`pending`) proposal rows, explicitly to measure whether the mechanism justifies the human review effort before that review is spent. The pilot's verdict does not constitute approval of any row and does not authorize promotion; it is an input to the decision of whether to proceed with human review at all.
>
> Result: `MIXED`. Clause 0 passed at the strongest level — predictions on every propagation-unexposed row are bit-identical between arms across 5 seeds and both models. Of the six numbered clauses, the two that test benefit failed with the sign inverted (M1 AUC fell in 5/5 seeds on the affected segment; M2 MAE rose in 5/5, roughly 4× the published noise band), while the three non-regression guards (Brier, fail recall, fail F1) passed. A model-free diagnostic shows why: on these rows the existing Level-4/5 fallback prior is already closer to the realized outcome than the predecessor prior, i.e. the old-plan courses were harder than their successors proved to be. The harm concentrates in credit-changed and `shared`-scope links. No proposal row was approved, no version promoted, no model frozen, TEST untouched.

