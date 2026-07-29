# Phase 2R — mapping tables rebuilt from TRAIN membership

Status: **proposal and diagnostic tables only**. Every relationship is pending; no mapping was approved or applied.

## Validation gates

| Gate | Result |
|---|---:|
| Normalized known-name matches | 66 / 67 |
| Expected sole normalized-name non-match | `510.111 -> 1183.111` |
| Never-in-TRAIN VALID course IDs | 182 |
| Never-in-TRAIN VALID rows | 25,627 |
| Known-pair final-category total | 67 |
| VALID `course_history_count` rows checked | 156,097 |
| VALID `course_history_count` mismatches | 0 |
| Proposal approval status | `pending` only |
| Proposed links applied to difficulty | No |

The VALID projection contained only the explicitly requested identifier, catalog-key, semester, and frozen history-count columns. No VALID outcome column was loaded. No model artifact was loaded, trained, tuned, or rescored.

## 1. Corrected degree-generation census

- TRAIN-present degrees: **20**
- VALID-only degrees: **25**
- Catalog-only degrees: **13**

Complete VALID-only degree list:

`26.111`, `27.111`, `29.111`, `30.111`, `31.111`, `33.111`, `34.111`, `35.111`, `36.111`, `37.111`, `39.111`, `40.111`, `41.111`, `42.111`, `44.111`, `45.111`, `46.111`, `47.111`, `48.111`, `50.111`, `52.111`, `53.111`, `54.111`, `55.111`, `56.111`

Catalog-only degrees (reported separately and excluded from the current-new lineage):

`9.111`, `12.111`, `17.111`, `25.111`, `38.111`, `51.111`, `57.111`, `58.111`, `59.111`, `60.111`, `61.111`, `64.111`, `65.111`

### Diagnostic comparison with the former numeric proxy

| Classification | Degrees |
|---|---:|
| Former catalog `numeric degree_id >= 40` proxy | 23 |
| Correct VALID-only membership definition | 25 |
| Added by correction | 11 |
| Removed by correction | 9 |

Added: `26.111`, `27.111`, `29.111`, `30.111`, `31.111`, `33.111`, `34.111`, `35.111`, `36.111`, `37.111`, `39.111`

Removed: `49.111`, `51.111`, `57.111`, `58.111`, `59.111`, `60.111`, `61.111`, `64.111`, `65.111`

`49.111` is TRAIN-present and is therefore **old**, irrespective of its numeric value: yes.

## 2. Degree lineage rebuilt against TRAIN-present predecessors

The table contains the top three deterministic candidates for each of the 25 VALID-only degrees (75 proposal rows). Ranking is overlap-of-new, Jaccard, degree-name similarity, then normalized old-degree ID.

| New degree | Rank-1 TRAIN degree | Shared keys | Overlap of new | Jaccard | Name similarity |
|---|---|---:|---:|---:|---:|
| `26.111` | `49.111` | 45 | 0.608 | 0.429 | 0.571 |
| `27.111` | `49.111` | 45 | 0.608 | 0.429 | 0.571 |
| `29.111` | `49.111` | 47 | 0.635 | 0.456 | 0.571 |
| `30.111` | `49.111` | 50 | 0.676 | 0.500 | 0.625 |
| `31.111` | `49.111` | 66 | 0.892 | 0.786 | 1.000 |
| `33.111` | `8.111` | 31 | 0.517 | 0.337 | 0.500 |
| `34.111` | `8.111` | 31 | 0.525 | 0.341 | 1.000 |
| `35.111` | `7.111` | 37 | 0.627 | 0.435 | 0.556 |
| `36.111` | `11.111` | 41 | 0.683 | 0.500 | 0.600 |
| `37.111` | `10.111` | 36 | 0.600 | 0.414 | 0.439 |
| `39.111` | `10.111` | 27 | 0.711 | 0.365 | 0.327 |
| `40.111` | `10.111` | 19 | 0.475 | 0.226 | 0.327 |
| `41.111` | `2.111` | 74 | 0.881 | 0.796 | 1.000 |
| `42.111` | `1.111` | 81 | 0.890 | 0.818 | 0.850 |
| `44.111` | `13.111` | 70 | 0.875 | 0.795 | 1.000 |
| `45.111` | `49.111` | 60 | 0.789 | 0.652 | 0.625 |
| `46.111` | `49.111` | 55 | 0.724 | 0.567 | 0.571 |
| `47.111` | `49.111` | 55 | 0.724 | 0.567 | 0.571 |
| `48.111` | `49.111` | 57 | 0.750 | 0.600 | 0.571 |
| `50.111` | `24.111` | 60 | 0.857 | 0.769 | 1.000 |
| `52.111` | `10.111` | 28 | 0.452 | 0.289 | 0.439 |
| `53.111` | `8.111` | 23 | 0.377 | 0.228 | 1.000 |
| `54.111` | `8.111` | 23 | 0.371 | 0.225 | 0.500 |
| `55.111` | `11.111` | 33 | 0.532 | 0.359 | 0.600 |
| `56.111` | `7.111` | 29 | 0.475 | 0.305 | 0.556 |

Previously missed ranges now receiving TRAIN-era candidates:

- IDs 26–31 present in the VALID-only set: `26.111`, `27.111`, `29.111`, `30.111`, `31.111`; all have three lineage candidates: yes.
- IDs 33–39 present in the VALID-only set: `33.111`, `34.111`, `35.111`, `36.111`, `37.111`, `39.111`; all have three lineage candidates: yes.

## 3. Corrected course-generation census

TRAIN-present course IDs: **811**. VALID course IDs absent from TRAIN: **182**, covering **25,627** VALID rows.

The former `course_id >= 1150` rule appears here only as a diagnostic:

| Diagnostic | Course IDs |
|---|---:|
| Numeric proxy set | 162 |
| Correct membership set | 182 |
| Membership-new IDs missed by proxy | 20 |
| Proxy-new IDs removed by membership | 0 |

Membership-new IDs missed by the proxy: `99.111`, `101.111`, `102.111`, `103.111`, `104.111`, `105.111`, `106.111`, `107.111`, `108.111`, `109.111`, `110.111`, `111.111`, `113.111`, `114.111`, `115.111`, `116.111`, `117.111`, `447.111`, `489.111`, `492.111`

Proxy-new IDs removed: —

## 4. Normalization and split handling

The unchanged normalization gate produced **66/67** exact known-pair matches and one correct non-match. The non-match is not forced through successor matching.

`510.111 بنيان الحواسيب → 1183.111 بنيان الحواسيب1 | 1192.111 بنيان الحواسيب2` is recorded as a split with `credit_change = +3` and `approval_status = pending`.

The catalog-wide membership detector produced 4 structural candidate rows. Only the task-specified known split is marked for automatic ordinary-match exclusion; every other structural pattern remains a pending diagnostic candidate.

## 5. Course-link proposal census

| Relationship type | Proposal rows | Distinct new courses touched |
|---|---:|---:|
| `candidate_below_support` | 5 | 5 |
| `consolidated_into` | 38 | 12 |
| `manual_candidate` | 1 | 1 |
| `name_only_review_candidate` | 3 | 3 |
| `none` | 134 | 134 |
| `split_from` | 2 | 2 |
| `successor` | 28 | 28 |

All **182** membership-new course IDs appear. Candidate rows below support and name-only review rows remain visible and unweighted.

### One coherent census for the 67 known pairs

| Final category | Pairs |
|---|---:|
| automatic eligible link | 34 |
| split_or_merge | 1 |
| manual proposal | 1 |
| candidate_below_support | 2 |
| name_only_review_candidate | 1 |
| unresolved | 28 |
| **Total** | **67** |

Every known pair occurs in exactly one category. Pairs not automatically eligible:

| New course | Old course | Final category | TRAIN support | Reason |
|---|---|---|---:|---|
| `1175.111` التحليل الرياضي1 | `502.111` التحليل الرياضي 1 | `unresolved` | 1524 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1172.111` الرياضيات المتقطعة | `431.111` الرياضيات المتقطعة | `unresolved` | 1929 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1174.111` الفيزياء1 | `501.111` الفيزياء 1 | `unresolved` | 1910 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1179.111` الفيزياء2 | `457.111` الفيزياء 2 | `unresolved` | 362 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1173.111` الجبر الخطي ونظرية المصفوفات | `432.111` الجبر الخطي ونظرية المصفوفات | `unresolved` | 1933 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1178.111` الدارات المنطقية | `434.111` الدارات المنطقية | `unresolved` | 1396 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1183.111` بنيان الحواسيب1 | `510.111` بنيان الحواسيب | `split_or_merge` | 1165 | Task-confirmed split/merge; excluded from ordinary successor matching. |
| `1184.111` الخوارزميات وبنى المعطيات1 | `516.111` الخوارزميات وبنى المعطيات 1 | `unresolved` | 1088 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1182.111` الإحصاء والاحتمالات | `504.111` الإحصاء والاحتمالات | `unresolved` | 886 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1187.111` نظرية الحوسبة | `521.111` نظرية الحوسبة | `unresolved` | 483 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1195.111` نظرية المعلومات | `437.111` نظرية المعلومات | `unresolved` | 650 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1190.111` المعادلات التفاضلية والتحويلات | `426.111` المعادلات التفاضلية والتحويلات | `unresolved` | 292 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1164.111` التحليل العددي | `503.111` التحليل العددي | `unresolved` | 1031 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1196.111` شبكات الحاسوب | `466.111` شبكات الحاسوب | `unresolved` | 548 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1235.111` تصميم المترجمات | `523.111` تصميم المترجمات | `unresolved` | 319 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1166.111` إدارة المشاريع | `505.111` إدارة المشاريع | `unresolved` | 624 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1246.111` قواعد البيانات المتقدمة | `533.111` قواعد البيانات المتقدمة | `unresolved` | 223 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1205.111` معالجة الصور وتحليلها | `451.111` معالجة الصور وتحليلها | `unresolved` | 57 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1249.111` نظم المعلومات الإدارية | `545.111` نظم المعلومات الإدارية | `unresolved` | 135 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1216.111` الإشارات والنظم | `458.111` الإشارات والنظم | `unresolved` | 260 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1198.111` مشروع فصلي | `538.111` مشروع  فصلي | `unresolved` | 206 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1165.111` مهارات التواصل | `662.111` مهارات التواصل | `unresolved` | 1596 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1252.111` المتحكمات الصغرية والنظم المضمنة | `461.111` المتحكمات الصغرية والنظم المضمنة | `unresolved` | 177 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1255.111` مدخل إلى الروبوتية | `445.111` مدخل إلى الروبوتية | `candidate_below_support` | 3 | Raw TRAIN support is 3, below min_support=20. |
| `1271.111` برمجة التطبيقات الشبكية | `417.111` برمجة التطبيقات الشبكية | `unresolved` | 54 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1212.111` معالجة اللغات الطبيعية | `448.111` معالجة اللغات الطبيعية | `candidate_below_support` | 1 | Raw TRAIN support is 1, below min_support=20. |
| `1214.111` النظم الخبيرة | `441.111` النظم الخبيرة | `unresolved` | 119 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1167.111` إدارة المؤسسات | `509.111` إدارة المؤسسات | `unresolved` | 461 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1339.111` إنكليزي الأعمال | `74.111` إنكليزي الأعمال | `name_only_review_candidate` | 178 | Normalized name matches in lineage scope, but the narrow credits/type key does not. |
| `1419.111` نقود ومصارف | `49.111` نقود ومصارف | `manual proposal` | 48 | Explicit single-course exception; automatic lineage scope did not recover it safely. |
| `1224.111` الاتصالات الرقمية | `463.111` الاتصالات الرقمية | `unresolved` | 168 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1231.111` الاتصالات النقالة واللاسلكية | `474.111` الاتصالات النقالة واللاسلكية | `unresolved` | 206 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |
| `1277.111` نظم الزمن الحقيقي | `536.111` نظم الزمن الحقيقي | `unresolved` | 353 | The normalized names match, but the reviewed old course was not present in the rank-1 TRAIN-lineage catalog scope for this new course. |

### Required worked examples

1. Split: `510.111 → 1183.111|1192.111`; `relationship_type = split_from`, no weight.
2. Shared consolidation: `967.111 → 1422.111`; TRAIN support **9,254**, computed weight **0.950200**. The weight was derived from current TRAIN volumes, not hard-coded.
3. Manual pending proposal: `49.111 → 1419.111`; TRAIN support **48**, relationship `manual_candidate`, weight `1.0`.
4. Corrected 26–31 lineage example: `26.111` → `49.111` (rank 1, overlap 0.608).
5. Corrected 33–39 lineage example: `33.111` → `8.111` (rank 1, overlap 0.517).
6. Truly new course with no eligible predecessor: `99.111` سلوك المستهلك; relationship `none`.

## 6. Coverage comparison

Coverage means a weighted eligible successor/consolidation/manual proposal or a task-confirmed structural relationship. Review-only and below-support candidates are not counted as covered.

The previous measurement uses the course-ID set marked covered in the prior numeric-threshold `course_link_proposed.csv`, re-aggregated against the direct membership census of 25,627 VALID rows. This keeps the current row denominator explicit and explains the small difference from the older rounded 32.6% report. The corrected eligibility figure also enforces the new narrow-key and review-only rules, so it is not a one-variable causal estimate of the generation-proxy effect.

| Measurement | Course IDs covered | VALID rows covered | Coverage of 25,627 |
|---|---:|---:|---:|
| Previous numeric-threshold scoped coverage | 33 | 8,370 | 32.7% |
| Corrected TRAIN-membership scoped coverage | 43 | 7,619 | 29.7% |
| Global name-key diagnostic upper bound | 83 | 17,814 | 69.5% |

Corrected exclusive row census:

| Contribution/status | Course IDs | VALID rows |
|---|---:|---:|
| `shared` | 21 | 6,781 |
| `specific` | 20 | 71 |
| `split_or_merge` | 2 | 767 |
| `name_only_review` | 3 | 10 |
| `below_support_only` | 2 | 26 |
| `unresolved` | 134 | 17,972 |

Below-support candidate exposure is non-additive: **5** candidate links touch **5** new courses / **1,099** VALID rows. This includes `893.111` at support 2, which is visible but excluded from consolidated weights.

## 7. Temporal difficulty prototype

The prototype contains **138,712** rows across all **811** distinct TRAIN courses, with Level 1 (degree + course), Level 2 (course across degrees), pre-semester snapshots using strictly earlier semesters, and a final `TRAIN_END_STATE` snapshot from all TRAIN rows.

`TRAIN_END_STATE` reproduced frozen VALID `course_history_count` for **156,097** rows with **0 mismatches**.

The table is a prototype only: `link_used = null` and `link_weight = null` on every row. No proposed mapping was applied.

## Governance entry (ready to copy)

> Any factual claim used as a hard gate in an implementation prompt must cite a specific repository artifact and line range. Otherwise it must be stated as a hypothesis to verify, not as an established fact.

The earlier numeric generation proxy was an unsupported assumption that reduced measured coverage and was caught by the known-answer validation set.

No decision log was edited. Generation, validation, and reporting stop here.
