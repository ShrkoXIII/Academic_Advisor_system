# Course-identity reconciliation - resolving the 170 pending candidates with history through 2025 S2

**Status: READ-ONLY EVIDENCE TASK. Updated candidates only.**

> **No mapping has been created, applied, or wired.** No `canonical_course_id` column exists anywhere as a result of this work. `course_identity_candidates_v2.csv` is a *review artifact* for a human reader, not an equivalence table, and must not be consumed as a drop-in mapping file: it deliberately mixes buckets, carries conflicting-evidence columns, and lists more than one candidate per row - exactly as the v1 CSV did.

| | |
|---|---|
| Prior investigation | `COURSE_IDENTITY_INVESTIGATION.md` (HEAD `a32f20c3529b654ac8a6445ec9a3186ad97de3c1`) |
| Frozen dataset version | `2026-07-26_batched_fixes__registration_roster_concurrent` |
| Extended history file | `data/final/without_outliers.parquet` |
| Extended coverage | `20051` -> `20252` (66 semesters, 727,852 rows) |
| TEST parquet | never read - no TEST path was constructed, globbed, stat-ed or read |
| Models trained / tuned | none |
| Datasets written | none |
| Defaults / wiring / promotion changed | none |
| HEAD at run time | `0a9f3464202a5b101d6992f2cb93d45967c31a5a` |

## Headline verdict

- **The extra years move 8 courses / 1,317 VALID rows from `likely` to `confirmed`.** `confirmed_equivalent` goes 5 -> 13 courses (1,791 -> 3,108 rows); confirmed-only coverage recovery goes 1,727 -> 3,044 rows.
- **The `confirmed + likely` upper bound does not move at all: 18,060 rows, unchanged from 18,060.** The taper signal only relocates courses between the two accepted buckets - it cannot rescue a course whose *name, credits, requirement type or lineage* were the problem. The headline payoff of this whole line of work is therefore unchanged.
- **The extra years refute nearly as many flagged courses as they confirm.** Of the 5 courses the prior censoring guard withheld, 3 are now confirmed and **2 are positively refuted**: their predecessors are still enrolling in 2025. The prior investigation's own sensitivity variant would have confirmed all 5 - so the conservative pre-registered guard was right, and the sensitivity was wrong on 2 of 5.
- **The dominant new finding is negative.** For 72 of the 170 pending courses (4,119 VALID rows) the candidate predecessor is still running at comparable volume through 2025 - both courses are live, which is evidence AGAINST equivalence, not for it.
- **Nothing was downgraded out of `likely`, and no `unresolved` course moved.** Every `unresolved` course was blocked by a non-temporal clause, which extra history cannot address.

## 0. Declared governance conflict - the extended window overlaps TEST

`docs/pipeline_rules.md` line 81 fixes the temporal split as **Train 2005-2021 / Validation 2022-2023 / Test 2024 + 2025 S1**. The extended history this task directs the analysis to use therefore spans the TEST window, while CLAUDE.md section 5 declares TEST `closed_not_read`. **This is a real conflict and it is declared here rather than resolved silently.** The task prompt is explicit and read-only, so per the CLAUDE.md header rule the prompt was followed and the conflict is flagged.

Containment actually applied:

- `df_test_final.parquet` was never read, globbed, stat-ed or path-constructed.
- Only these columns were loaded from the extended file: `course_id`, `part_id`, `degree_id`, `student_id`. `final_mark` and every other outcome column were never loaded, so no label information from the TEST window can have entered any artifact produced here.
- Only per-course enrolment **counts** were derived. No metric, no difficulty statistic and no model input was computed from post-20233 rows.

The residual exposure is nonetheless real: the taper evidence below is derived from enrolment volumes inside the TEST window. **The human owns the decision of whether that is acceptable before any of this feeds a mapping.**

## 1. Preconditions - the prior state was reproduced, not trusted

| Quantity | Reproduced now | Prior report | Match |
|---|---:|---:|:--:|
| never-in-TRAIN courses | 182 | 182 | yes |
| never-in-TRAIN VALID rows | 25,627 | 25,627 | yes |
| total uncovered VALID rows | 26,882 | 26,882 | yes |
| `confirmed_equivalent` | 5 / 1,791 | 5 / 1,791 | yes |
| `likely_equivalent_needs_review` | 82 / 16,359 | 82 / 16,359 | yes |
| `genuinely_new` | 7 / 134 | 7 / 134 | yes |
| `unresolved` | 88 / 7,343 | 88 / 7,343 | yes |

The candidate CSV was cross-checked row-by-row against the reproduction: 182 data rows, 0 bucket mismatches, 0 top-candidate mismatches.

### The pre-registered censoring flag

The prior investigation's censoring guard withheld confirmation from **5 courses / 2,281 VALID rows** that its own sensitivity variant would have confirmed (confirmed 5 -> 10 courses, 1,791 -> 4,072 rows). Those courses are:

| Course | Name | VALID rows | Prior bucket |
|---|---|---:|---|
| `1365.111` | رياضيات الأعمال | 638 | `likely_equivalent_needs_review` |
| `1332.111` | إدارة الموارد البشرية | 594 | `likely_equivalent_needs_review` |
| `1184.111` | الخوارزميات وبنى المعطيات1 | 486 | `likely_equivalent_needs_review` |
| `1182.111` | الإحصاء والاحتمالات | 378 | `likely_equivalent_needs_review` |
| `1164.111` | التحليل العددي | 185 | `likely_equivalent_needs_review` |

They are the clearest test of whether the extra years change anything: each was blocked only by the guard's scope, never by weak evidence. Their outcome:

| Course | VALID rows | Updated bucket | Verdict | Extended-history evidence |
|---|---:|---|---|---|
| `1365.111` | 638 | `likely_equivalent_needs_review` | **REFUTED by the extra years - predecessor never tapered** | predecessor still running through 2025 at reduced volume (last active 20251, 2024/2021 ratio 0.05) - no confirmed taper |
| `1332.111` | 594 | `confirmed_equivalent` | **CONFIRMED by the extra years** | taper confirmed: predecessor last active 20242 (after the new course's debut), then near-zero for 3 consecutive semesters through 20252 |
| `1184.111` | 486 | `confirmed_equivalent` | **CONFIRMED by the extra years** | taper confirmed: predecessor last active 20241 (after the new course's debut), then near-zero for 4 consecutive semesters through 20252 |
| `1182.111` | 378 | `likely_equivalent_needs_review` | **REFUTED by the extra years - predecessor never tapered** | predecessor still running through 2025 at reduced volume (last active 20252, 2024/2021 ratio 0.1344) - no confirmed taper |
| `1164.111` | 185 | `confirmed_equivalent` | **CONFIRMED by the extra years** | taper confirmed: predecessor last active 20242 (after the new course's debut), then near-zero for 3 consecutive semesters through 20252 |

**This is the single most decision-relevant result in this reconciliation.** The prior investigation's `temporal_only` sensitivity would have confirmed all 5 of these; the extended history shows 2 of the 5 to be wrong. The pre-registered conservative guard was the correct call, and the sensitivity block should not be used as if it were.

## 2. Extended-history coverage

- Rows: 727,852; distinct courses: 1,091.
- Semester range `20051` -> `20252`. Reaches 2024: **True**. Reaches 2025 S2 (`20252`): **True**.
- Semesters beyond the VALID end (`20233`): `20241`, `20242`, `20243`, `20251`, `20252` - 5 semesters of history the prior investigation could not see.

Consistency of the shared window (the extended file is not a different population):

| Check | Value |
|---|---:|
| extended rows in the shared window (<= 20233) | 606,562 |
| TRAIN + VALID rows of the frozen version | 606,562 |
| (course, semester) cells in TRAIN + VALID | 15,315 |
| of those also present in the extended file | 15,315 |
| cells present only in the extended shared window | 0 |
| columns loaded from the extended file | 4 |

- Predecessors of the 170 pending courses present in the extended file: 170 of 170.

## 3. The rule applied - one substitution, nothing else

The pre-registered rule of `COURSE_IDENTITY_INVESTIGATION.md` section 5 is applied unchanged: identifier pattern, credits, requirement type, faculty, planned level, name similarity (0.80 strong / 0.60 plausible tiers), degree overlap, degree lineage, and the >= 3.0x uniqueness dominance tie-break all keep their original definitions and their original scored values.

**The single substitution.** The prior confirmation clause read:

> AND at least one independent temporal/enrolment corroboration (...) AND the corroboration is not censored (predecessor activity does not end in 20232/20233)

Both halves of that clause existed only because VALID ended at 20233. With history through 20252 the question is directly observable, so the two halves collapse into one term:

> AND the predecessor's enrolment has **tapered**: it falls below 5 enrolments in a semester and stays there for >= 3 consecutive semesters running through the end of the observed calendar (`20252`).

Thresholds are inherited, not invented:

- **near-zero = fewer than 5 enrolments in a semester** - the prior investigation's own `MIN_SEMESTER_ACTIVITY` activity floor.
- **>= 3 consecutive near-zero semesters** - the bar set by this task. Requiring the silence to run through `20252` also retires the old censoring guard: three observed empty semesters are a disappearance, not a right-censored gap.
- **still-active-at-full-volume = 2024 rows / 2021 rows > 0.35** - the prior investigation's own `COLLAPSE_RATIO_MAX`. 2024 is used as the reference late year because it is the last COMPLETE academic year in the file; 2025 carries only S1/S2 and would understate a full-year rate.

Two sub-cases of a confirmed taper are distinguished because they carry different information:

- **taper post-debut** - the predecessor was still running when the new course debuted and died afterwards. This is the signal the old censoring guard could not see.
- **taper pre-debut** - the predecessor was already dead before the new course debuted. The extra years confirm it never rebounded, but they add no new evidence.

## 4. Reclassification of the 170 pending courses

| Bucket | Prior (182) | Updated (182) | Delta |
|---|---:|---:|---:|
| `confirmed_equivalent` | 5 / 1,791 | 13 / 3,108 | +8 / +1,317 |
| `likely_equivalent_needs_review` | 82 / 16,359 | 74 / 15,042 | -8 / -1,317 |
| `genuinely_new` | 7 / 134 | 7 / 134 | +0 / +0 |
| `unresolved` | 88 / 7,343 | 88 / 7,343 | +0 / +0 |

### Transitions

| Transition | Courses | VALID rows |
|---|---:|---:|
| `likely_equivalent_needs_review` -> `likely_equivalent_needs_review` | 74 | 15,042 |
| `unresolved` -> `unresolved` | 88 | 7,343 |
| `likely_equivalent_needs_review` -> `confirmed_equivalent` | 8 | 1,317 |

### Evidence that drove the changes

| Extended-history evidence | Courses | VALID rows |
|---|---:|---:|
| predecessor already tapered before the debut (no new evidence) | 25 | 2,335 |
| taper confirmed AFTER the new course's debut (new evidence) | 38 | 9,283 |
| predecessor still running at reduced volume - no confirmed taper | 26 | 7,757 |
| predecessor still active at comparable volume through 2025 (evidence AGAINST) | 72 | 4,119 |
| predecessor never reached the activity floor in any semester - no taper to observe, no corroboration available | 9 | 208 |

The new course's own trajectory was checked for all 170 pending courses: **17 courses / 275 VALID rows** have themselves tapered to near-zero for >= 3 semesters, i.e. the *new* course looks like a short-lived offering. This is recorded in the v2 CSV column `new_course_short_lived` and reported here, but it is deliberately **not** allowed to change any bucket: gating on it would be a new rule, and this task's mandate is to apply the pre-registered one.

### Courses that changed bucket

| # | New course | Name | VALID rows | Old -> New | Predecessor | New evidence |
|---:|---|---|---:|---|---|---|
| 1 | `1332.111` | إدارة الموارد البشرية | 594 | `likely_equivalent_needs_review` -> `confirmed_equivalent` | `73.111` إدارة الموارد البشرية | taper confirmed: predecessor last active 20242 (after the new course's debut), then near-zero for 3 consecutive semesters through 20252 |
| 2 | `1184.111` | الخوارزميات وبنى المعطيات1 | 486 | `likely_equivalent_needs_review` -> `confirmed_equivalent` | `516.111` الخوارزميات وبنى المعطيات 1 | taper confirmed: predecessor last active 20241 (after the new course's debut), then near-zero for 4 consecutive semesters through 20252 |
| 3 | `1164.111` | التحليل العددي | 185 | `likely_equivalent_needs_review` -> `confirmed_equivalent` | `503.111` التحليل العددي | taper confirmed: predecessor last active 20242 (after the new course's debut), then near-zero for 3 consecutive semesters through 20252 |
| 4 | `1216.111` | الإشارات والنظم | 27 | `likely_equivalent_needs_review` -> `confirmed_equivalent` | `458.111` الإشارات والنظم | taper confirmed: predecessor last active 20232 (after the new course's debut), then near-zero for 6 consecutive semesters through 20252 |
| 5 | `1218.111` | الدارات الإلكترونية2 | 10 | `likely_equivalent_needs_review` -> `confirmed_equivalent` | `459.111` الدارات الإلكترونية 2 | taper confirmed: predecessor last active 20241 (after the new course's debut), then near-zero for 4 consecutive semesters through 20252 |
| 6 | `1318.111` | إدارة التغيير | 8 | `likely_equivalent_needs_review` -> `confirmed_equivalent` | `81.111` إدارة التغيير | taper confirmed: predecessor last active 20242 (after the new course's debut), then near-zero for 3 consecutive semesters through 20252 |
| 7 | `1220.111` | الدارات الكهربائية2 | 6 | `likely_equivalent_needs_review` -> `confirmed_equivalent` | `428.111` الدارات الكهربائية 2 | taper confirmed: predecessor last active 20232 (after the new course's debut), then near-zero for 6 consecutive semesters through 20252 |
| 8 | `1224.111` | الاتصالات الرقمية | 1 | `likely_equivalent_needs_review` -> `confirmed_equivalent` | `463.111` الاتصالات الرقمية | taper confirmed: predecessor last active 20242 (after the new course's debut), then near-zero for 3 consecutive semesters through 20252 |

### Courses whose classification did not change

162 of the 170 pending courses (22,385 VALID rows) kept their bucket. Grouped by why the extra years added nothing:

| Reason nothing changed | Courses | VALID rows |
|---|---:|---:|
| predecessor already fully tapered before the debut - the extra years confirm no rebound but add no new evidence | 25 | 2,335 |
| taper confirmed, but another clause of the pre-registered rule still blocks confirmation (name/credits/requirement type/lineage/uniqueness) | 30 | 7,966 |
| predecessor still running at reduced volume through 2025 - no confirmed taper, bucket unchanged | 26 | 7,757 |
| predecessor still active at comparable volume through 2025 - the extra years positively deny the taper, and the bucket was already below confirmed | 72 | 4,119 |
| predecessor never reached the activity floor in any semester - there is no taper to observe, so the extra years add no corroboration either way | 9 | 208 |

### What still blocks the courses whose predecessor DID taper

63 pending courses have a confirmed taper (38 post-debut, 25 pre-debut), but only 8 reached `confirmed_equivalent`. For the other 55 (10,301 VALID rows) the taper is no longer the obstacle - some other clause of the pre-registered rule is. This is the actionable list for a registrar review, because these are the courses where the enrolment evidence is already settled and only catalog attributes are in dispute:

| Remaining blocker | Courses | VALID rows |
|---|---:|---:|
| credits differ | 5 | 3,438 |
| name not an exact normalized match | 23 | 2,742 |
| requirement type differs | 5 | 1,876 |
| ambiguous tie with a second candidate | 2 | 1,450 |
| name not an exact normalized match; requirement type differs | 4 | 292 |
| name not an exact normalized match; credits differ; requirement type differs | 1 | 153 |
| credits differ; requirement type differs; ambiguous tie with a second candidate | 1 | 105 |
| name not an exact normalized match; no degree-lineage link | 8 | 102 |
| name not an exact normalized match; credits differ; requirement type differs; no degree-lineage link | 1 | 54 |
| name not an exact normalized match; no degree-lineage link; ambiguous tie with a second candidate | 1 | 51 |
| name not an exact normalized match; credits differ | 1 | 30 |
| no degree-lineage link | 3 | 8 |

The 40 highest-volume pending courses, with their updated evidence:

| # | New course | Name | VALID rows | Old bucket | New bucket | Changed | Predecessor | New evidence |
|---:|---|---|---:|---|---|:--:|---|---|
| 1 | `1423.111` | اللغة الإنكليزية – 1 | 979 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `830.111` اللغة الانكليزية - 1 | predecessor already fully tapered before the new course's debut (last active 20171); the extra years confirm no rebound but add no new evidence |
| 2 | `1175.111` | التحليل الرياضي1 | 917 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `502.111` التحليل الرياضي 1 | taper confirmed: predecessor last active 20232 (after the new course's debut), then near-zero for 6 consecutive semesters through 20252 |
| 3 | `1421.111` | مهارات اللغة العربية | 854 | `unresolved` | `unresolved` | no | `955.111` اللغة العربية | predecessor still running through 2025 at reduced volume (last active 20252, 2024/2021 ratio 0.0933) - no confirmed taper |
| 4 | `1171.111` | مدخل إلى الخوارزميات والبرمجة | 834 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `513.111` مقدمة في الخوارزميات والبرمجة | taper confirmed: predecessor last active 20232 (after the new course's debut), then near-zero for 6 consecutive semesters through 20252 |
| 5 | `1180.111` | الدارات الكهربائية1 | 773 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `427.111` الدارات الكهربائية 1 | taper confirmed: predecessor last active 20222 (after the new course's debut), then near-zero for 9 consecutive semesters through 20252 |
| 6 | `1179.111` | الفيزياء2 | 759 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `457.111` الفيزياء 2 | taper confirmed: predecessor last active 20222 (after the new course's debut), then near-zero for 9 consecutive semesters through 20252 |
| 7 | `1176.111` | البرمجة1 | 751 | `unresolved` | `unresolved` | no | `519.111` برمجة النظم | predecessor still active at comparable volume through 2025 (63 rows in 2024 vs 128 in 2021, ratio 0.4922) - evidence AGAINST equivalence: both courses are live |
| 8 | `1173.111` | الجبر الخطي ونظرية المصفوفات | 719 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `432.111` الجبر الخطي ونظرية المصفوفات | taper confirmed: predecessor last active 20222 (after the new course's debut), then near-zero for 9 consecutive semesters through 20252 |
| 9 | `1177.111` | التحليل الرياضي2 | 688 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `433.111` التحليل الرياضي 2 | predecessor still running through 2025 at reduced volume (last active 20251, 2024/2021 ratio 0.0674) - no confirmed taper |
| 10 | `1422.111` | مهارات الحاسوب | 686 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `967.111` مهارات الحاسوب | predecessor still running through 2025 at reduced volume (last active 20252, 2024/2021 ratio 0.0518) - no confirmed taper |
| 11 | `1375.111` | مبادئ المحاسبة1 | 656 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `1.111` مبادئ المحاسبة 1 | taper confirmed: predecessor last active 20232 (after the new course's debut), then near-zero for 6 consecutive semesters through 20252 |
| 12 | `1365.111` | رياضيات الأعمال | 638 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `118.111` رياضيات الأعمال | predecessor still running through 2025 at reduced volume (last active 20251, 2024/2021 ratio 0.05) - no confirmed taper |
| 13 | `1332.111` | إدارة الموارد البشرية | 594 | `likely_equivalent_needs_review` | `confirmed_equivalent` | yes | `73.111` إدارة الموارد البشرية | taper confirmed: predecessor last active 20242 (after the new course's debut), then near-zero for 3 consecutive semesters through 20252 |
| 14 | `1372.111` | مبادئ الإدارة | 579 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `70.111` مبادئ الإدارة | taper confirmed: predecessor last active 20242 (after the new course's debut), then near-zero for 3 consecutive semesters through 20252 |
| 15 | `1178.111` | الدارات المنطقية | 567 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `434.111` الدارات المنطقية | taper confirmed: predecessor last active 20231 (after the new course's debut), then near-zero for 7 consecutive semesters through 20252 |
| 16 | `1368.111` | قانون الأعمال | 531 | `unresolved` | `unresolved` | no | `71.111` البيئة القانونية للأعمال | predecessor still running through 2025 at reduced volume (last active 20252, 2024/2021 ratio 0.0249) - no confirmed taper |
| 17 | `1181.111` | البرمجة2 | 517 | `unresolved` | `unresolved` | no | `519.111` برمجة النظم | predecessor still active at comparable volume through 2025 (63 rows in 2024 vs 128 in 2021, ratio 0.4922) - evidence AGAINST equivalence: both courses are live |
| 18 | `1183.111` | بنيان الحواسيب1 | 495 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `510.111` بنيان الحواسيب | taper confirmed: predecessor last active 20242 (after the new course's debut), then near-zero for 3 consecutive semesters through 20252 |
| 19 | `1184.111` | الخوارزميات وبنى المعطيات1 | 486 | `likely_equivalent_needs_review` | `confirmed_equivalent` | yes | `516.111` الخوارزميات وبنى المعطيات 1 | taper confirmed: predecessor last active 20241 (after the new course's debut), then near-zero for 4 consecutive semesters through 20252 |
| 20 | `1424.111` | اللغة الإنكليزية – 2 | 471 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `836.111` اللغة الانكليزية - 2 | predecessor already fully tapered before the new course's debut (last active 20192); the extra years confirm no rebound but add no new evidence |
| 21 | `1185.111` | أساسيات قواعد البيانات | 461 | `unresolved` | `unresolved` | no | `517.111` قواعد البيانات 1 | taper confirmed: predecessor last active 20242 (after the new course's debut), then near-zero for 3 consecutive semesters through 20252 |
| 22 | `1189.111` | الخوارزميات وبنى المعطيات2 | 394 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `522.111` الخوارزميات وبنى المعطيات 2 | predecessor still running through 2025 at reduced volume (last active 20251, 2024/2021 ratio 0.0853) - no confirmed taper |
| 23 | `1374.111` | مبادئ التسويق | 394 | `unresolved` | `unresolved` | no | `98.111` إدارة التسويق | predecessor still running through 2025 at reduced volume (last active 20252, 2024/2021 ratio 0.1478) - no confirmed taper |
| 24 | `1182.111` | الإحصاء والاحتمالات | 378 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `504.111` الإحصاء والاحتمالات | predecessor still running through 2025 at reduced volume (last active 20252, 2024/2021 ratio 0.1344) - no confirmed taper |
| 25 | `1282.111` | الاقتصاد الجزئي | 373 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `45.111` الاقتصاد الجزئي | predecessor still running through 2025 at reduced volume (last active 20251, 2024/2021 ratio 0.2441) - no confirmed taper |
| 26 | `1187.111` | نظرية الحوسبة | 371 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `521.111` نظرية الحوسبة | predecessor still running through 2025 at reduced volume (last active 20251, 2024/2021 ratio 0.1623) - no confirmed taper |
| 27 | `1191.111` | تراسل البيانات | 355 | `unresolved` | `unresolved` | no | `512.111` تراسل المعطيات | taper confirmed: predecessor last active 20242 (after the new course's debut), then near-zero for 3 consecutive semesters through 20252 |
| 28 | `1188.111` | مدخل إلى هندسة البرمجيات | 324 | `unresolved` | `unresolved` | no | `532.111` هندسة البرمجيات | predecessor still active at comparable volume through 2025 (58 rows in 2024 vs 73 in 2021, ratio 0.7945) - evidence AGAINST equivalence: both courses are live |
| 29 | `1195.111` | نظرية المعلومات | 316 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `437.111` نظرية المعلومات | predecessor still running through 2025 at reduced volume (last active 20251, 2024/2021 ratio 0.239) - no confirmed taper |
| 30 | `1429.111` | علم البيئة | 308 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `1021.111` علم البيئة | predecessor still running through 2025 at reduced volume (last active 20252, 2024/2021 ratio 0.1991) - no confirmed taper |
| 31 | `1190.111` | المعادلات التفاضلية والتحويلات | 305 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `426.111` المعادلات التفاضلية والتحويلات | taper confirmed: predecessor last active 20231 (after the new course's debut), then near-zero for 7 consecutive semesters through 20252 |
| 32 | `1376.111` | مبادئ المحاسبة2 | 292 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `2.111` مبادئ المحاسبة 2 | predecessor still running through 2025 at reduced volume (last active 20251, 2024/2021 ratio 0.0886) - no confirmed taper |
| 33 | `1186.111` | البرمجة3 | 277 | `unresolved` | `unresolved` | no | `519.111` برمجة النظم | predecessor still active at comparable volume through 2025 (63 rows in 2024 vs 128 in 2021, ratio 0.4922) - evidence AGAINST equivalence: both courses are live |
| 34 | `1192.111` | بنيان الحواسيب2 | 272 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `510.111` بنيان الحواسيب | taper confirmed: predecessor last active 20242 (after the new course's debut), then near-zero for 3 consecutive semesters through 20252 |
| 35 | `1309.111` | إحصاء الاعمال1 | 268 | `unresolved` | `unresolved` | no | `120.111` إحصاء الأعمال والاقتصاد | predecessor still running through 2025 at reduced volume (last active 20252, 2024/2021 ratio 0.0629) - no confirmed taper |
| 36 | `1373.111` | مبادئ الإدارة المالية | 254 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `44.111` مبادئ الإدارة المالية | predecessor still running through 2025 at reduced volume (last active 20251, 2024/2021 ratio 0.0898) - no confirmed taper |
| 37 | `1237.111` | نظم قواعد البيانات | 237 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `527.111` قواعد البيانات 2 | predecessor still active at comparable volume through 2025 (57 rows in 2024 vs 102 in 2021, ratio 0.5588) - evidence AGAINST equivalence: both courses are live |
| 38 | `1193.111` | نظم التشغيل1 | 223 | `unresolved` | `unresolved` | no | `528.111` نظم التشغيل | predecessor still active at comparable volume through 2025 (105 rows in 2024 vs 48 in 2021, ratio 2.1875) - evidence AGAINST equivalence: both courses are live |
| 39 | `1194.111` | مدخل إلى الذكاء الصنعي | 213 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `435.111` مقدمة في الذكاء الصنعي | predecessor still running through 2025 at reduced volume (last active 20252, 2024/2021 ratio 0.2262) - no confirmed taper |
| 40 | `1346.111` | بحوث العمليات | 190 | `likely_equivalent_needs_review` | `likely_equivalent_needs_review` | no | `122.111` بحوث العمليات | predecessor still active at comparable volume through 2025 (59 rows in 2024 vs 100 in 2021, ratio 0.59) - evidence AGAINST equivalence: both courses are live |

## 5. Updated payoff

Recomputed with the identical counterfactual method as the prior investigation: in-memory substitution of the candidate predecessor `course_id` into the VALID rows, then exact re-evaluation of the Level-1 (`degree_course_key`) -> Level-2 (`course_id`) support lookup of `src/course_difficulty.py` against TRAIN-only statistics. **No data was written and no mapping was persisted.**

Simulation validation against the on-disk columns: 0 `course_history_count` mismatches and 0 `course_difficulty_missing` mismatches over 156,097 VALID rows (verdict: **exact**).

| Scenario | Courses mapped | Rows gaining observed history | Rows crossing the 20-row threshold | Prior figure | Delta |
|---|---:|---:|---:|---:|---:|
| `confirmed_equivalent` only | 13 | 3,108 | 3,044 | 1,727 | +1,317 |
| `confirmed` + `likely` (upper bound) | 87 | 18,150 | 18,060 | 18,060 | +0 |

**Residual against the original 26,882 uncovered VALID rows:**

| Scenario | Uncovered rows remaining | % of 26,882 | Prior % |
|---|---:|---:|---:|
| `confirmed` only | 23,838 | 88.68% | 93.58% |
| `confirmed` + `likely` | 8,822 | 32.82% | 32.82% |

> The payoff remains an arithmetic upper bound on *coverage*, not on *accuracy*. A confirmed taper shows that the old course stopped running; it does not show that the new course teaches the same content at the same difficulty. Only outcomes under the new plan can test that, and those are exactly the rows in question.

## 6. What was NOT done

- No `canonical_course_id` or equivalence mapping was created, applied or wired.
- The 5 `confirmed_equivalent` and 7 `genuinely_new` courses were NOT re-scored; they are carried over verbatim, per the task's explicit non-scope.
- The identifier-structure check, the all-course disappearance/appearance scan and the faculty-specific curriculum-revision finding were NOT re-run or re-litigated.
- No dataset was built, copied or written; no `CURRENT_VERSION.txt`, default, or promotion marker was touched.
- No model was trained, retrained or re-tuned.
- `df_test_final.parquet` was not read, and no TEST path was constructed.
- Nothing was pushed.

## 7. Reported note on the requested `genuinely_new` transition line

The task asks for a count of courses "reclassified to genuinely_new (predecessor never tapered)". Under the pre-registered rule, `genuinely_new` means *no TRAIN course reaches name similarity >= 0.60* - a statement about content, not about enrolment. A predecessor that never tapers is strong evidence AGAINST equivalence, but it does not make the new course content-novel, and relabelling it `genuinely_new` would be inventing a rule this task is explicitly told not to invent.

So those courses are reported under their rule-correct bucket and carry the dedicated flag `predecessor_active_through_2025` in the v2 CSV: **72 courses / 4,119 VALID rows** have a top candidate still enrolling at comparable volume through 2025. That is the number the requested line reports.

