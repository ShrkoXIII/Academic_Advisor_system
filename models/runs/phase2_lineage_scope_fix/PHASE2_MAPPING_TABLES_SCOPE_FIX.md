# Phase 2S — lineage search scope fix

Status: **proposal and diagnostic tables only**. Every proposal remains `approval_status = pending`; no relationship was approved or applied.

## Validation and safety gates

| Gate | Result |
|---|---:|
| Normalization known-pair matches | 66 / 67 |
| Sole normalization non-match | `510.111 -> 1183.111` |
| Never-in-TRAIN VALID course IDs | 182 |
| Never-in-TRAIN VALID rows | 25,627 |
| VALID-only degrees | 25 |
| Measured recoverable pairs linked | 26 / 26 |
| Six-category known-pair total | 67 |
| Temporal rows checked | 156,097 |
| Temporal mismatches | 0 |
| Proposal statuses | `pending` only |
| Difficulty links applied | No |

VALID was loaded from a fixed explicit projection containing identifiers, faculty, catalog keys, semester, and frozen `course_history_count`; `final_mark` and every defined VALID outcome column were absent at runtime. No TEST path or model artifact was accessed, and no training, tuning, or rescoring was performed.

## 1. Ancestry eligibility

A TRAIN-present degree is ancestry-eligible only when at least **5** distinct courses in its catalog are TRAIN-present anywhere. Ineligible degrees remain old for generation; they are barred only from predecessor candidacy.

| Degree | Degree name | TRAIN enrolment rows | Catalog courses | Old courses in catalog | Eligible |
|---|---|---:|---:|---:|---:|
| `1.111` | دكتور في طب الأسنان | 115,336 | 89 | 86 | yes |
| `2.111` | دكتور في الطب | 89,748 | 84 | 69 | yes |
| `3.111` | هندسة البرمجيات ونظم المعلومات | 25,317 | 75 | 70 | yes |
| `4.111` | هندسة الذكاء الصنعي | 1,532 | 75 | 67 | yes |
| `5.111` | هندسة الحاسوب و التحكم | 752 | 77 | 66 | yes |
| `6.111` | هندسة الاتصالات و الشبكات | 15,410 | 77 | 68 | yes |
| `7.111` | التسويق و الإمداد | 1,436 | 63 | 42 | yes |
| `8.111` | اداره الموارد البشرية | 13,307 | 63 | 59 | yes |
| `10.111` | إدارة المؤسسات المالية و المصرفية | 4,202 | 63 | 60 | yes |
| `11.111` | المحاسبة و التدقيق | 8,998 | 63 | 58 | yes |
| `13.111` | الصيدلة و الكيمياء الصيدلية | 116,593 | 78 | 75 | yes |
| `15.111` | دكتور في الطب البشري | 2,834 | 88 | 50 | yes |
| `16.111` | دكتور في طب الأسنان | 9,626 | 81 | 78 | yes |
| `18.111` | هندسة الحاسوب و المعلوماتية | 2,494 | 62 | 59 | yes |
| `19.111` | إدارة الأعمال | 6,773 | 61 | 58 | yes |
| `20.111` | هندسة البترول | 14,209 | 66 | 61 | yes |
| `21.111` | هندسة حاسوب (اختصاص عام) | 5,831 | 42 | 37 | yes |
| `22.111` | إدارة (اختصاص عام) | 5,246 | 41 | 38 | yes |
| `24.111` | إجازة في هندسة البترول | 10,819 | 68 | 65 | yes |
| `49.111` | هندسة الاتصالات 2023 | 2 | 76 | 0 | no |

Excluded degrees: **1** — `49.111`. Degree `49.111` is correctly excluded with 76 catalog courses and zero TRAIN-present catalog courses.

## 2. Degree lineage over the eligible pool

The output retains three deterministic candidates for each of all 25 VALID-only degrees (75 rows). Ranking is overlap-of-new descending, Jaccard descending, degree-name similarity descending, then normalized old degree ID ascending.

| New degree | Phase 2R rank 1 | Phase 2S rank 1 | Changed |
|---|---|---|---:|
| `26.111` | `49.111` | `3.111` | yes |
| `27.111` | `49.111` | `4.111` | yes |
| `29.111` | `49.111` | `6.111` | yes |
| `30.111` | `49.111` | `5.111` | yes |
| `31.111` | `49.111` | `6.111` | yes |
| `33.111` | `8.111` | `8.111` | no |
| `34.111` | `8.111` | `8.111` | no |
| `35.111` | `7.111` | `7.111` | no |
| `36.111` | `11.111` | `11.111` | no |
| `37.111` | `10.111` | `10.111` | no |
| `39.111` | `10.111` | `10.111` | no |
| `40.111` | `10.111` | `10.111` | no |
| `41.111` | `2.111` | `2.111` | no |
| `42.111` | `1.111` | `1.111` | no |
| `44.111` | `13.111` | `13.111` | no |
| `45.111` | `49.111` | `5.111` | yes |
| `46.111` | `49.111` | `3.111` | yes |
| `47.111` | `49.111` | `4.111` | yes |
| `48.111` | `49.111` | `6.111` | yes |
| `50.111` | `24.111` | `24.111` | no |
| `52.111` | `10.111` | `10.111` | no |
| `53.111` | `8.111` | `8.111` | no |
| `54.111` | `8.111` | `8.111` | no |
| `55.111` | `11.111` | `11.111` | no |
| `56.111` | `7.111` | `7.111` | no |

Changed rank 1: **9 degrees** — `26.111`, `27.111`, `29.111`, `30.111`, `31.111`, `45.111`, `46.111`, `47.111`, `48.111`.

Unchanged rank 1: **16 degrees** — `33.111`, `34.111`, `35.111`, `36.111`, `37.111`, `39.111`, `40.111`, `41.111`, `42.111`, `44.111`, `50.111`, `52.111`, `53.111`, `54.111`, `55.111`, `56.111`.

The prompt preregistered eight informatics degrees; repository evidence shows the same defect also affected `30.111`, so nine rank-1 choices change in total. None now resolves to ancestry-ineligible `49.111`.

## 3. Union search scope and confidence labels

For a specific course, `lineage_scope_used` is the sorted union of the top-three eligible lineage candidates for all catalogued new degrees. An ancestry-eligible TRAIN-present placement can contribute itself directly; ineligible `49.111` cannot. Shared courses continue to use all TRAIN-present catalog courses.

Every catalog-wide normalized-name match is emitted. Faculty is absent from the cleaned catalog but available in both TRAIN and the outcome-free VALID projection, so out-of-lineage confidence uses old TRAIN versus new VALID course-faculty set intersection.

Section 4's explicit `in_lineage` name-key weighting rule controls the automatic relationship here. The former specific-course credits/type narrow gate cannot remain an eligibility gate: 13 of the preregistered 26 recoverable pairs (including required `502.111 → 1175.111`) fail it, which would recover only 13 and trigger the task's fewer-than-20 stop condition. No known-pair-specific exception was introduced.

| Scope confidence | Proposal rows | Distinct new courses |
|---|---:|---:|
| `in_lineage` | 117 | 82 |
| `same_faculty` | 0 | 0 |
| `cross_faculty` | 26 | 19 |

All proposal rows with an old-course match, counted by `lineage_rank_matched`:

| Lineage rank | Proposal rows | Distinct new courses |
|---|---:|---:|
| `1` | 66 | 60 |
| `2` | 0 | 0 |
| `3` | 0 | 0 |
| `null` | 79 | 42 |

The null bucket contains shared-scope, out-of-lineage, structural, manual, and any direct-self matches for which no ranked lineage candidate supplied the row.

The preregistered 26-pair recovery reproduced **26 of 26 pairs / 7,335 VALID rows**. Ranks for those exact pairs: rank 1: 26. The eligibility filter promoted the true informatics ancestors, so rank 1 supplies all 26 rather than rank 2 supplying the majority under an unfiltered ranking.

Recovered exact pairs: `503.111 → 1164.111`, `505.111 → 1166.111`, `509.111 → 1167.111`, `431.111 → 1172.111`, `432.111 → 1173.111`, `501.111 → 1174.111`, `502.111 → 1175.111`, `434.111 → 1178.111`, `457.111 → 1179.111`, `504.111 → 1182.111`, `516.111 → 1184.111`, `521.111 → 1187.111`, `426.111 → 1190.111`, `437.111 → 1195.111`, `466.111 → 1196.111`, `538.111 → 1198.111`, `451.111 → 1205.111`, `441.111 → 1214.111`, `458.111 → 1216.111`, `463.111 → 1224.111`, `474.111 → 1231.111`, `523.111 → 1235.111`, `533.111 → 1246.111`, `545.111 → 1249.111`, `461.111 → 1252.111`, `536.111 → 1277.111`

The two preregistered residual reviewed pairs are now visible, but remain unweighted `cross_faculty` review evidence: `662.111 → 1165.111` and `417.111 → 1271.111`. Those new courses also have different in-lineage same-name candidates (`439.111` and `544.111` respectively); that does not recover the reviewed exact pairs.

## 4. Unchanged normalization, support, split, and manual rules

The unchanged normalization gate is **66/67**. Its sole non-match remains `510.111 -> 1183.111`, represented by the unchanged split `510.111 → 1183.111|1192.111` with `credit_change = +3`.

`min_support = 20` remains in force. Below-support candidates are visible and unweighted, including `893.111` at support 2. Weighted `consolidated_into` groups sum to 1.0.

The task-required manual proposal `49.111 → 1419.111` remains `match_method = manual`, pending, and explicitly records that degree `49.111` is ancestry-ineligible. The identically numbered old course `49.111` is catalogued under degree `10.111`; the pair-specific automatic generation path is suppressed so the required manual governance status is preserved.

### Course-link relationship census

| Relationship type | Proposal rows | Distinct new courses |
|---|---:|---:|
| `candidate_below_support` | 9 | 7 |
| `consolidated_into` | 42 | 14 |
| `manual_candidate` | 1 | 1 |
| `name_only_review_candidate` | 26 | 19 |
| `none` | 95 | 95 |
| `split_from` | 2 | 2 |
| `successor` | 65 | 65 |

All **182** never-in-TRAIN VALID course IDs are present.

### Six-category known-pair census

| Exclusive category | Pairs |
|---|---:|
| `automatic eligible link` | 61 |
| `split_or_merge` | 1 |
| `manual proposal` | 1 |
| `candidate_below_support` | 2 |
| `name_only_review_candidate` | 2 |
| `unresolved` | 0 |
| **Total** | **67** |

Every one of the 67 reviewed pairs appears in exactly one category. Non-automatic reviewed pairs:

| New course | Old course | Category | TRAIN support | Reason |
|---|---|---|---:|---|
| `1183.111` بنيان الحواسيب1 | `510.111` بنيان الحواسيب | `split_or_merge` | 1165 | Task-confirmed split/merge; excluded from ordinary matching. |
| `1165.111` مهارات التواصل | `662.111` مهارات التواصل | `name_only_review_candidate` | 1596 | Exact name-key match is visible but outside lineage scope; it remains unweighted review evidence. |
| `1255.111` مدخل إلى الروبوتية | `445.111` مدخل إلى الروبوتية | `candidate_below_support` | 3 | Raw TRAIN support is 3, below min_support=20. |
| `1271.111` برمجة التطبيقات الشبكية | `417.111` برمجة التطبيقات الشبكية | `name_only_review_candidate` | 54 | Exact name-key match is visible but outside lineage scope; it remains unweighted review evidence. |
| `1212.111` معالجة اللغات الطبيعية | `448.111` معالجة اللغات الطبيعية | `candidate_below_support` | 1 | Raw TRAIN support is 1, below min_support=20. |
| `1419.111` نقود ومصارف | `49.111` نقود ومصارف | `manual proposal` | 48 | Explicit pending manual proposal retained by task requirement. |

## 5. Required coverage comparison

| Measurement | Course IDs | VALID rows | % of 25,627 |
|---|---:|---:|---:|
| Phase 2R (rank-1 scope) | 43 | 7,619 | 29.7% |
| Phase 2S (union scope + ancestry filter) | 82 | 17,036 | 66.5% |
| Global name-key diagnostic upper bound | 83 | 17,814 | 69.5% |

Pre-registered coverage check: **PASS**. The actual 66.5% exceeds the central estimate because the literal Section 4 rule makes every support-eligible in-lineage name-key match weightable, including courses outside the 67-pair validation set. No parameter was adjusted to obtain this result.

Exclusive contribution/status census:

| Contribution/status | Course IDs | VALID rows | % of 25,627 |
|---|---:|---:|---:|
| `shared` | 21 | 6,781 | 26.5% |
| `specific` | 59 | 9,488 | 37.0% |
| `split_or_merge` | 2 | 767 | 3.0% |
| `name_only_review` | 3 | 1,545 | 6.0% |
| `below_support_only` | 2 | 26 | 0.1% |
| `unresolved` | 95 | 7,020 | 27.4% |

Below-support exposure is non-additive: 9 candidate rows touch 7 courses / 1,132 VALID rows.

## 6. Required worked examples

1. Degree `26.111`: Phase 2R rank 1 was `49.111`; after excluding the zero-ancestor sibling degree, Phase 2S rank 1 is `3.111`.

2. `502.111 → 1175.111`: new catalog degrees `26.111|27.111|29.111|30.111|31.111|45.111|46.111|47.111|48.111|49.111`; union scope `3.111|4.111|5.111|6.111`; matched rank `1`; confidence `in_lineage`; weight `1.0`.

3. `967.111 → 1422.111`: old TRAIN support 9,254; weighted-group support 9,739; volume-derived weight `0.950200225896` (still near 0.95).

4. `510.111 → 1183.111|1192.111` remains the pending split with `credit_change = +3`; ordinary successor matching remains disabled for it.

5. Cross-faculty visibility: `662.111 → 1165.111` is now emitted with `scope_confidence = cross_faculty`, `relationship_type = name_only_review_candidate`, `match_method = cross_faculty_name_only_review_candidate`, and null weight. New VALID faculties are `167.111`; old TRAIN faculties are `2.111`.

6. Still unresolved course: `99.111` سلوك المستهلك. It has no TRAIN-present catalog course with the same normalized name key after the unchanged structural exclusions, so its relationship is `none`.

## 7. Temporal difficulty prototype

The unchanged prototype contains **138,712** rows and covers all **811** TRAIN courses. `TRAIN_END_STATE` reproduces frozen VALID `course_history_count` at **0 mismatches over 156,097 rows**. `link_used` and `link_weight` are null on every row.

## Governance entry (ready to copy)

> A prompt specification must be checked against the findings of prior phases before it is issued. The Phase 2R rank-1 scoping rule directly contradicted Phase 0 Q4, which had already established that no old→new degree relationship in this data is one-to-one. The contradiction was authored in the prompt, not introduced by the implementation, and cost 7,335 VALID rows of coverage. It was caught only because the 67 human-reviewed pairs act as a known-answer validation set.

Ranking by catalog overlap measures **similarity**, not ancestry. A same-generation sibling scores highest precisely because it shares the new courses. Ancestry therefore requires the explicit eligibility constraint used here; similarity alone selects the wrong degree.

No decision log was edited. Generation, validation, and reporting stop here; no Phase 3 action is proposed.
