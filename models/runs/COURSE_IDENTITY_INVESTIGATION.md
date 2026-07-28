# Course-identity investigation - are the 182 'new' VALID courses renumbered old courses?

**Status: READ-ONLY EVIDENCE TASK. Candidates only.**

> **No mapping has been created, applied, or wired.** No `canonical_course_id` column exists anywhere as a result of this work. `course_identity_candidates.csv` is a *review artifact* for a human reader, not an equivalence table, and must not be consumed as a drop-in mapping file: it deliberately mixes buckets, carries conflicting-evidence columns, and lists more than one candidate per row.

| | |
|---|---|
| Dataset version | `2026-07-26_batched_fixes__registration_roster_concurrent` |
| Splits read | `df_train_final.parquet`, `df_valid_final.parquet` |
| TEST | `closed_not_read` - no TEST path was constructed, globbed, stat-ed or read |
| Models trained / tuned | none |
| Datasets written | none |
| Defaults / wiring / promotion changed | none |
| HEAD at run time | `a32f20c` |

## Headline verdict

- **The 182 courses are not renumbered old courses.** 162 of 182 ids were allocated above the previous maximum with no mechanical relation to any old id; there is no renumbering fingerprint.
- **What actually happened is a curriculum revision.** New degree programmes (25 VALID-only degrees, several literally named with a `2023` suffix) were opened under two faculty codes that barely exist in TRAIN (167.111: 2 TRAIN rows; 177.111: 0), each with a freshly numbered course catalog.
- **Many new courses do have a content predecessor.** 5 courses (1,791 VALID rows) meet the pre-registered confirmation bar and 82 more (16,359 rows) are likely but need review - typically the same normalized name, credits and requirement type inside a linked degree, with the old course's enrolment collapsing to a teach-out tail.
- **But equivalence here is curricular, not clerical.** The predecessor usually still exists and still runs. Borrowing its difficulty statistics is a modelling decision about content similarity, not a correction of a broken identifier - and it is the human's decision, not this task's.
- **Best case coverage recovery:** 18,060 of the 25,627 never-in-TRAIN rows under `confirmed + likely`, leaving 8,822 of the original 26,882 uncovered rows (32.82%).

## 1. Reproduction of the never-in-TRAIN population

| Quantity | Obtained | Expected (diagnostic `a32f20c`) | Match |
|---|---:|---:|:--:|
| Distinct never-in-TRAIN course_ids | 182 | 182 | yes |
| VALID rows | 25,627 | 25,627 | yes |
| Total uncovered VALID rows | 26,882 | 26,882 | yes |

Recomputed independently from the frozen parquets with the identical definition (`course_difficulty_missing == 1` and `course_id` absent from TRAIN).

## 2. Attribute inventory (performed BEFORE any matching was designed)

Sources inventoried:

- `data/raw/v_acd_degree_course.parquet`
- `data/preprocessed/V_ACD_DEGREE_COURSE/clean_v_acd_degree_course.parquet`
- `data/raw/v_acs_grade.parquet (grade dictionary - no course attributes)`
- `data/raw/v_add_academic_info.parquet (student-level - no course attributes)`
- `data/raw/v_add_student_degree_status.parquet (student-level)`
- `data/raw/v_crg_student_course_raw.parquet (enrolment facts, not a catalog)`

`V_ACD_DEGREE_COURSE` is the only course-level catalog in the data root: 4,006 degree-course rows, 1,503 distinct course_ids, 58 degrees. The cleaned parquet is a strict column subset of the raw view (9 of 17 columns) and adds no attribute, so the raw view was used.

**All 182 of the 182 new courses are present in the catalog**, as are 804 of the 811 TRAIN courses.

| Attribute | Exists | Null rate: catalog | Null rate: 182 new | Null rate: TRAIN courses | Distinct |
|---|:--:|---:|---:|---:|---:|
| `degree_course_id` | yes | 0.0% | 0.0% | 0.0% | 4006 |
| `course_id` | yes | 0.0% | 0.0% | 0.0% | 1503 |
| `degree_id` | yes | 0.0% | 0.0% | 0.0% | 58 |
| `faculty_id` | yes | 61.9% | 68.7% | 74.3% | 8 |
| `course_type_id` | yes | 0.0% | 0.0% | 0.0% | 3 |
| `requirement_type_id` | yes | 0.0% | 0.0% | 0.0% | 6 |
| `requirement_type_sl` | yes | 0.0% | 0.0% | 0.0% | 6 |
| `course_name_sl` | yes | 0.0% | 0.0% | 0.0% | 1208 |
| `course_official_sl` | yes | 0.0% | 0.0% | 0.0% | 1210 |
| `degree_name_sl` | yes | 0.0% | 0.0% | 0.0% | 56 |
| `year_order` | yes | 0.2% | 0.0% | 0.0% | 6 |
| `semester_order` | yes | 0.2% | 0.0% | 0.0% | 2 |
| `required_credits` | yes | 92.6% | 95.6% | 92.9% | 23 |
| `course_credits` | yes | 0.0% | 0.0% | 0.0% | 9 |
| `active` | yes | 0.0% | 0.0% | 0.0% | 1 |
| `credits_count` | yes | 0.0% | 0.0% | 0.0% | 32 |
| `req_degree_id` | yes | 60.1% | 37.9% | 31.8% | 49 |

**Attributes that do NOT exist** (so the corresponding evidence is unavailable):

- `prerequisites` - no prerequisite column exists in V_ACD_DEGREE_COURSE or its cleaned parquet; no prerequisite table exists in the data root
- `dates` - no created/updated/effective-from/effective-to column exists
- `active_flag` - column 'active' exists but is constant 'A' for all 4006 rows -> carries zero information; retired courses are not marked
- `course_code` - no alphanumeric course code (e.g. 'CS101') exists; the only identifier is the numeric course_id
- `faculty_id` - present but 61.9% null in the catalog; the model data carries a non-null per-row faculty_id, which is used instead

### Text attributes

- `course_name_sl`: UTF-8, Arabic script stored as native codepoints (verified: no mojibake, no cp1256 double-encoding). Scripts: {'arabic_only': 3993, 'mixed': 13}. 1208 distinct raw values collapse to 1136 after normalization; 22 values carry irregular leading/trailing/double whitespace. Latin transliteration present: False.
- `course_official_sl`: UTF-8, Arabic script stored as native codepoints (verified: no mojibake, no cp1256 double-encoding). Scripts: {'arabic_only': 3993, 'mixed': 13}. 1210 distinct raw values collapse to 1136 after normalization; 21 values carry irregular leading/trailing/double whitespace. Latin transliteration present: False.
- `requirement_type_sl`: UTF-8, Arabic script stored as native codepoints (verified: no mojibake, no cp1256 double-encoding). Scripts: {'arabic_only': 4006}. 6 distinct raw values collapse to 6 after normalization; 0 values carry irregular leading/trailing/double whitespace. Latin transliteration present: False.

**Inventory conclusion.** Matching is possible: the catalog carries a course name for every one of the 182 new courses and for 804 of 811 TRAIN courses, plus credits, requirement type and planned level. Prerequisites, dates and an informative active flag do NOT exist, so prerequisite-structure evidence is unavailable.

## 3. Identifier-structure check

'<numeric_course_id>.<university_id>'; university_id is the constant '111' for every row in TRAIN and VALID. The numeric part is the only varying component - there is no department prefix, no alphabetic code and no check digit.

| | TRAIN courses | The 182 new courses |
|---|---|---|
| Distinct courses | 811 | 182 |
| Numeric id range | 1 - 1093 | 99 - 1433 |

Block histogram (numeric id / 100):

| Block | TRAIN | New |
|---|---:|---:|
| 0-99 | 80 | 1 |
| 100-199 | 59 | 16 |
| 200-299 | 89 | 0 |
| 300-399 | 99 | 0 |
| 400-499 | 85 | 3 |
| 500-599 | 85 | 0 |
| 600-699 | 51 | 0 |
| 700-799 | 17 | 0 |
| 800-899 | 73 | 0 |
| 900-999 | 83 | 0 |
| 1000-1099 | 90 | 0 |
| 1100-1199 | 0 | 36 |
| 1200-1299 | 0 | 57 |
| 1300-1399 | 0 | 49 |
| 1400-1499 | 0 | 20 |

**Pattern found.** 162 of 182 new ids (89.0%) are strictly greater than the TRAIN maximum (1093) and occupy three dense contiguous runs (1163-1260, 1267-1409, 1418-1433). This is monotone append-only allocation. There is NO added prefix, NO added digit, NO shifted department block and NO arithmetic offset relating any new id to an old id, i.e. no mechanical renumbering signature.

- **162 of 182 (89.01%)** new ids lie above the TRAIN maximum -> sequential allocation appended after the previous maximum - the signature of NEW catalog records, not of an identifier rewrite of existing records.
- **20 of 182** lie inside the TRAIN id range (99, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 447, 489, 492) -> ids that fall inside the TRAIN id range but were never enrolled during TRAIN - dormant/late-activated catalog slots, not renumbered courses.

> **This is the single most important negative result of the investigation.** A renumbering migration leaves a mechanical fingerprint - a constant offset, an added prefix, a widened field. None is present. The ids were *allocated*, not *rewritten*. Whatever equivalence exists is curricular, not clerical.

## 4. Disappearance analysis

- TRAIN courses: 811; still present in VALID: 504; absent from VALID: 307 (25,338 TRAIN rows).
- Absent AND active at the boundary (last TRAIN semester 20212/20213): **26 courses, 3,897 TRAIN rows**.
- Enrolment-collapse cohort (>= 20 rows in the final TRAIN academic year (2021*) and post-boundary rows/year <= 35% of that): **29 courses, 31,283 TRAIN rows**.

**Complementarity verdict.** Disappearance does NOT balance appearance. Only 26 courses (3897 TRAIN rows) vanish outright at the boundary, against 182 new courses carrying 25,627 VALID rows. Volume collapse of surviving old courses, not outright disappearance, is where the transferred enrolment is visible.

The dominant teach-out signature: the old course keeps a small tail in valid rather than vanishing outright, so a binary disappear/appear test badly understates the replacement. For example, the old first-year courses of degree `3.111` collapse from roughly 1,500-2,100 TRAIN rows each to 23-68 VALID rows while their same-named new-plan counterparts absorb 700-900 VALID rows each - a teach-out tail, not a disappearance.

### Censoring limitation (explicit)

VALID ends at 20233. A course whose activity stops in 20232 or 20233 cannot be distinguished from one that has simply not been offered again yet. Such candidates are FLAGGED (censored_predecessor / censored_debut) and are never scored as if the stop were confirmed.

Affected in this run: 52 new courses debut in 20232/20233 (939 VALID rows) and are flagged `censored_debut`; 122 top candidates have a predecessor whose last active semester falls in the censored window and are flagged `censored_predecessor`. Confirmation is withheld from all of them by rule.

### Degree lineage used to constrain candidates

Lineage links a VALID-only degree to a TRAIN degree when EITHER at least 5 students moved from the old degree to the new one, OR the normalized degree names are >= 0.60 similar after stripping the `2023` re-issue suffix. Both signals are reported so the link can be checked independently.

| New degree | Name | VALID rows | Predecessor(s) | Migrating students |
|---|---|---:|---|---|
| `26.111` | الهندسة المعلوماتية/هندسة البرمجيات ونظم المعلومات | 5,678 | `21.111`, `3.111`, `4.111`, `18.111`, `49.111`, `6.111` | 21.111:101, 3.111:75, 4.111:9 |
| `27.111` | الهندسة المعلوماتية/هندسة الذكاء الصنعي وعلوم البيانات | 2,966 | `4.111`, `21.111`, `18.111` | 4.111:42, 21.111:41 |
| `29.111` | الهندسة المعلوماتية/هندسة أمن النظم والشبكات الحاسوبية | 3,303 | `21.111`, `3.111`, `6.111` | 21.111:62, 3.111:14, 6.111:5 |
| `30.111` | هندسة التحكم والروبوت | 1,313 | `21.111`, `5.111`, `49.111`, `6.111`, `20.111`, `18.111`, `4.111` | 21.111:9, 5.111:6 |
| `31.111` | هندسة الاتصالات | 924 | `6.111`, `49.111`, `20.111`, `4.111`, `21.111`, `5.111`, `3.111` | 6.111:6 |
| `33.111` | إدارة الأعمال | 448 | `19.111`, `22.111`, `8.111` | - |
| `34.111` | إدارة الموارد البشرية | 1,339 | `8.111`, `19.111` | - |
| `35.111` | التسويق | 454 | - | - |
| `36.111` | المحاسبة | 1,305 | `11.111` | - |
| `37.111` | التمويل والمصارف | 351 | `7.111` | - |
| `39.111` | العلوم الإدارية (اختصاص عام) | 874 | `22.111` | - |
| `40.111` | العلوم الإدارية (اختصاص عام) 2023 | 1,430 | `22.111` | - |
| `41.111` | دكتور في الطب 2023 | 3,125 | `2.111`, `15.111`, `1.111`, `16.111` | - |
| `42.111` | إجازة دكتور في طب الأسنان 2023 | 2,765 | `1.111`, `16.111`, `2.111`, `15.111` | - |
| `44.111` | الصيدلة و الكيمياء الصيدلية 2023 | 2,149 | `13.111` | - |
| `45.111` | هندسة التحكم والروبوت 2023 | 862 | `49.111`, `6.111`, `20.111`, `18.111`, `4.111` | - |
| `46.111` | الهندسة المعلوماتية/هندسة البرمجيات ونظم المعلومات 2023 | 706 | `3.111`, `18.111`, `49.111`, `6.111` | - |
| `47.111` | الهندسة المعلوماتية/هندسة الذكاء الصنعي وعلوم البيانات 2023 | 855 | `4.111`, `18.111` | - |
| `48.111` | الهندسة المعلوماتية/هندسة أمن النظم والشبكات الحاسوبية 2023 | 862 | `6.111` | - |
| `50.111` | إجازة في هندسة البترول2023 | 855 | `24.111`, `20.111` | - |
| `52.111` | التمويل والمصارف 2023 | 125 | `7.111` | - |
| `53.111` | إدارة الموارد البشرية 2023 | 436 | `8.111`, `19.111` | - |
| `54.111` | إدارة الأعمال2023 | 154 | `19.111`, `22.111`, `8.111` | - |
| `55.111` | المحاسبة 2023 | 496 | `11.111` | - |
| `56.111` | التسويق2023 | 149 | - | - |

## 5. Matching rule - FIXED BEFORE the classification was produced

**Name similarity alone is NEVER sufficient. Every bucket above 'unresolved' requires a degree-lineage link plus at least two non-name structural attributes.**

Normalization applied to every course name before comparison:

- Unicode NFKC compound normalization
- Arabic-Indic and extended Arabic-Indic digits folded to ASCII 0-9
- Arabic diacritics (U+064B-U+065F, U+0670, U+06D6-U+06ED) removed
- tatweel (U+0640) removed
- alef variants (أ إ آ ٱ) folded to ا
- alef maqsura (ى) folded to ي; waw/ya hamza (ؤ ئ) folded to و/ي
- ta marbuta (ة) folded to ه
- bare hamza (ء) dropped
- all dash/slash/underscore variants (including en-dash U+2013) folded to space
- punctuation removed
- digit/letter boundaries split with a space so 'الفيزياء1' == 'الفيزياء 1'
- Latin text casefolded
- whitespace collapsed and trimmed

A second, looser key additionally strips the Arabic definite article `ال` from each token; similarity is the maximum over the strict and loose keys of (character sequence ratio, token-set Jaccard).

Signals scored per (new course, candidate predecessor) pair: identifier pattern; disappearance/appearance complementarity; predecessor enrolment collapse; enrolment-volume continuity; credits equal; requirement type equal; faculty equal; degree-lineage link; planned level (year_order, semester_order) equal; normalized name similarity; degree overlap. Prerequisite structure could not be scored - the attribute does not exist (section 2).

### Minimum evidence combination per bucket

**`confirmed_equivalent`**

- normalized course name is an EXACT match (after the documented normalization)
- AND the predecessor is offered in the same degree, or in a degree that is a documented predecessor of a degree the new course is offered in
- AND course_credits are equal
- AND requirement_type_id is equal
- AND at least one independent temporal/enrolment corroboration: the predecessor's enrolment collapsed after the TRAIN boundary, OR its last active semester precedes/equals the new course's debut
- AND the match is unique: no second candidate reaches the same evidence weight unless the top candidate carries >= 3.0x its TRAIN volume
- AND the corroboration is not censored (predecessor activity does not end in 20232/20233)

**`likely_equivalent_needs_review`**

- degree-lineage link present, AND either
- (a) exact normalized name with exactly one of credits/requirement_type conflicting, or with no uncensored temporal corroboration, or
- (b) name similarity >= 0.8 with BOTH credits and requirement_type equal

**`genuinely_new`**

- no TRAIN course reaches name similarity >= 0.6 against the new course, in any degree - i.e. no plausible predecessor exists by content

**`unresolved`**

- a plausible candidate exists but the evidence combination reaches neither bucket above: missing degree lineage, conflicting credits AND requirement type, ambiguous tie between candidates, or censoring blocks the temporal judgement

## 6. Classification

| Bucket | Courses | VALID rows | % of 25,627 |
|---|---:|---:|---:|
| `confirmed_equivalent` | 5 | 1,791 | 7.0% |
| `likely_equivalent_needs_review` | 82 | 16,359 | 63.8% |
| `genuinely_new` | 7 | 134 | 0.5% |
| `unresolved` | 88 | 7,343 | 28.7% |
| **total** | **182** | **25,627** | 100.0% |

### Sensitivity to the scope of the censoring guard (NOT the reported result)

As pre-registered, the guard withholds confirmation whenever the predecessor's last active semester falls in 20232/20233 - even when the corroborating evidence is a whole-VALID-window enrolment collapse, which is not itself right-censored. The pre-registered rule is reported as authoritative and was NOT changed after results were seen; this block quantifies how much of the conservatism the guard's scope accounts for.

Variation: the censoring guard is applied only to the temporal-complementarity signal; a whole-window enrolment collapse is accepted as uncensored corroboration.

| Bucket | Courses (pre-registered) | Courses (sensitivity) | VALID rows (pre-registered) | VALID rows (sensitivity) |
|---|---:|---:|---:|---:|
| `confirmed_equivalent` | 5 | 10 | 1,791 | 4,072 |
| `likely_equivalent_needs_review` | 82 | 77 | 16,359 | 14,078 |
| `genuinely_new` | 7 | 7 | 134 | 134 |
| `unresolved` | 88 | 88 | 7,343 | 7,343 |

Coverage recovery under the sensitivity: 4,008 rows from `confirmed` alone (vs 1,727 under the pre-registered rule); the `confirmed + likely` upper bound is 18,060 either way, because the guard only moves courses between the two accepted buckets.

**The pre-registered rule remains authoritative.** The sensitivity is reported so the reader can see that the difference between `confirmed` and `likely` here is largely a judgement about censoring scope, not about evidence strength - which is precisely the kind of call the registrar's equivalence table would settle.

Full per-course detail is in `course_identity_candidates.csv` (sorted by VALID row count descending). The 40 highest-volume courses:

| # | New course | Name | VALID rows | Debut | Bucket | Candidate predecessor | Matched | Conflicted |
|---:|---|---|---:|---|---|---|---|---|
| 1 | `1423.111` | اللغة الإنكليزية – 1 | 979 | 20231 | `likely_equivalent_needs_review` | `830.111` اللغة الانكليزية - 1 | name_exact, credits, requirement_type, degree_lineage, planned_level, temporal_complementarity | faculty, predecessor_volume_collapse, volume_continuity |
| 2 | `1175.111` | التحليل الرياضي1 | 917 | 20221 | `likely_equivalent_needs_review` | `502.111` التحليل الرياضي 1 | name_exact, requirement_type, degree_lineage, predecessor_volume_collapse, volume_continuity | credits, faculty, planned_level, temporal_complementarity |
| 3 | `1172.111` | الرياضيات المتقطعة | 876 | 20221 | `confirmed_equivalent` | `431.111` الرياضيات المتقطعة | name_exact, credits, requirement_type, degree_lineage, predecessor_volume_collapse, volume_continuity | faculty, planned_level, temporal_complementarity |
| 4 | `1421.111` | مهارات اللغة العربية | 854 | 20231 | `unresolved` | `955.111` اللغة العربية | credits, requirement_type, degree_lineage | name_exact, faculty, planned_level, predecessor_volume_collapse, temporal_complementarity, volume_continuity |
| 5 | `1171.111` | مدخل إلى الخوارزميات والبرمجة | 834 | 20221 | `likely_equivalent_needs_review` | `513.111` مقدمة في الخوارزميات والبرمجة | credits, requirement_type, degree_lineage, planned_level, predecessor_volume_collapse, volume_continuity | name_exact, faculty, temporal_complementarity |
| 6 | `1174.111` | الفيزياء1 | 791 | 20221 | `confirmed_equivalent` | `501.111` الفيزياء 1 | name_exact, credits, requirement_type, degree_lineage, planned_level, predecessor_volume_collapse, volume_continuity | faculty, temporal_complementarity |
| 7 | `1180.111` | الدارات الكهربائية1 | 773 | 20221 | `likely_equivalent_needs_review` | `427.111` الدارات الكهربائية 1 | name_exact, credits, degree_lineage, predecessor_volume_collapse | requirement_type, faculty, planned_level, temporal_complementarity, volume_continuity |
| 8 | `1179.111` | الفيزياء2 | 759 | 20222 | `likely_equivalent_needs_review` | `457.111` الفيزياء 2 | name_exact, credits, degree_lineage, planned_level, predecessor_volume_collapse, temporal_complementarity | requirement_type, faculty, volume_continuity |
| 9 | `1176.111` | البرمجة1 | 751 | 20221 | `unresolved` | `519.111` برمجة النظم | credits, requirement_type, degree_lineage, volume_continuity | name_exact, faculty, planned_level, predecessor_volume_collapse, temporal_complementarity |
| 10 | `1173.111` | الجبر الخطي ونظرية المصفوفات | 719 | 20221 | `likely_equivalent_needs_review` | `432.111` الجبر الخطي ونظرية المصفوفات | name_exact, requirement_type, degree_lineage, planned_level, predecessor_volume_collapse, volume_continuity | credits, faculty, temporal_complementarity |
| 11 | `1177.111` | التحليل الرياضي2 | 688 | 20221 | `likely_equivalent_needs_review` | `433.111` التحليل الرياضي 2 | name_exact, requirement_type, degree_lineage, predecessor_volume_collapse, volume_continuity | credits, faculty, planned_level, temporal_complementarity |
| 12 | `1422.111` | مهارات الحاسوب | 686 | 20231 | `likely_equivalent_needs_review` | `967.111` مهارات الحاسوب | name_exact, credits, requirement_type, degree_lineage, planned_level | faculty, predecessor_volume_collapse, temporal_complementarity, volume_continuity |
| 13 | `1375.111` | مبادئ المحاسبة1 | 656 | 20221 | `likely_equivalent_needs_review` | `1.111` مبادئ المحاسبة 1 | name_exact, requirement_type, degree_lineage, planned_level, predecessor_volume_collapse, volume_continuity | credits, faculty, temporal_complementarity |
| 14 | `1365.111` | رياضيات الأعمال | 638 | 20221 | `likely_equivalent_needs_review` | `118.111` رياضيات الأعمال | name_exact, credits, requirement_type, degree_lineage, planned_level, predecessor_volume_collapse, volume_continuity | faculty, temporal_complementarity |
| 15 | `1332.111` | إدارة الموارد البشرية | 594 | 20221 | `likely_equivalent_needs_review` | `73.111` إدارة الموارد البشرية | name_exact, credits, requirement_type, degree_lineage, predecessor_volume_collapse, volume_continuity | faculty, planned_level, temporal_complementarity |
| 16 | `1372.111` | مبادئ الإدارة | 579 | 20221 | `likely_equivalent_needs_review` | `70.111` مبادئ الإدارة | name_exact, requirement_type, degree_lineage, planned_level, predecessor_volume_collapse, volume_continuity | credits, faculty, temporal_complementarity |
| 17 | `1178.111` | الدارات المنطقية | 567 | 20221 | `likely_equivalent_needs_review` | `434.111` الدارات المنطقية | name_exact, requirement_type, degree_lineage, predecessor_volume_collapse, volume_continuity | credits, faculty, planned_level, temporal_complementarity |
| 18 | `1368.111` | قانون الأعمال | 531 | 20221 | `unresolved` | `71.111` البيئة القانونية للأعمال | credits, requirement_type, degree_lineage, predecessor_volume_collapse, volume_continuity | name_exact, faculty, planned_level, temporal_complementarity |
| 19 | `1181.111` | البرمجة2 | 517 | 20221 | `unresolved` | `519.111` برمجة النظم | credits, requirement_type, degree_lineage, volume_continuity | name_exact, faculty, planned_level, predecessor_volume_collapse, temporal_complementarity |
| 20 | `1183.111` | بنيان الحواسيب1 | 495 | 20221 | `likely_equivalent_needs_review` | `510.111` بنيان الحواسيب | credits, requirement_type, degree_lineage, predecessor_volume_collapse, volume_continuity | name_exact, faculty, planned_level, temporal_complementarity |
| 21 | `1184.111` | الخوارزميات وبنى المعطيات1 | 486 | 20221 | `likely_equivalent_needs_review` | `516.111` الخوارزميات وبنى المعطيات 1 | name_exact, credits, requirement_type, degree_lineage, planned_level, predecessor_volume_collapse, volume_continuity | faculty, temporal_complementarity |
| 22 | `1424.111` | اللغة الإنكليزية – 2 | 471 | 20232 | `likely_equivalent_needs_review` | `836.111` اللغة الانكليزية - 2 | name_exact, credits, requirement_type, degree_lineage, planned_level, temporal_complementarity | faculty, predecessor_volume_collapse, volume_continuity |
| 23 | `1185.111` | أساسيات قواعد البيانات | 461 | 20221 | `unresolved` | `517.111` قواعد البيانات 1 | credits, requirement_type, degree_lineage, volume_continuity | name_exact, faculty, planned_level, predecessor_volume_collapse, temporal_complementarity |
| 24 | `1189.111` | الخوارزميات وبنى المعطيات2 | 394 | 20221 | `likely_equivalent_needs_review` | `522.111` الخوارزميات وبنى المعطيات 2 | name_exact, credits, degree_lineage, planned_level, volume_continuity | requirement_type, faculty, predecessor_volume_collapse, temporal_complementarity |
| 25 | `1374.111` | مبادئ التسويق | 394 | 20221 | `unresolved` | `98.111` إدارة التسويق | credits, requirement_type, degree_lineage, volume_continuity | name_exact, faculty, planned_level, predecessor_volume_collapse, temporal_complementarity |
| 26 | `1182.111` | الإحصاء والاحتمالات | 378 | 20221 | `likely_equivalent_needs_review` | `504.111` الإحصاء والاحتمالات | name_exact, credits, requirement_type, degree_lineage, predecessor_volume_collapse, volume_continuity | faculty, planned_level, temporal_complementarity |
| 27 | `1282.111` | الاقتصاد الجزئي | 373 | 20222 | `likely_equivalent_needs_review` | `45.111` الاقتصاد الجزئي | name_exact, requirement_type, degree_lineage, planned_level, volume_continuity | credits, faculty, predecessor_volume_collapse, temporal_complementarity |
| 28 | `1187.111` | نظرية الحوسبة | 371 | 20221 | `likely_equivalent_needs_review` | `521.111` نظرية الحوسبة | name_exact, credits, degree_lineage, planned_level, volume_continuity | requirement_type, faculty, predecessor_volume_collapse, temporal_complementarity |
| 29 | `1191.111` | تراسل البيانات | 355 | 20221 | `unresolved` | `512.111` تراسل المعطيات | credits, requirement_type, degree_lineage, volume_continuity | name_exact, faculty, planned_level, predecessor_volume_collapse, temporal_complementarity |
| 30 | `1188.111` | مدخل إلى هندسة البرمجيات | 324 | 20221 | `unresolved` | `532.111` هندسة البرمجيات | credits, degree_lineage, volume_continuity | name_exact, requirement_type, faculty, planned_level, predecessor_volume_collapse, temporal_complementarity |
| 31 | `1195.111` | نظرية المعلومات | 316 | 20221 | `likely_equivalent_needs_review` | `437.111` نظرية المعلومات | name_exact, credits, requirement_type, degree_lineage, planned_level, volume_continuity | faculty, predecessor_volume_collapse, temporal_complementarity |
| 32 | `1429.111` | علم البيئة | 308 | 20231 | `likely_equivalent_needs_review` | `1021.111` علم البيئة | name_exact, credits, requirement_type, degree_lineage | faculty, planned_level, predecessor_volume_collapse, temporal_complementarity, volume_continuity |
| 33 | `1190.111` | المعادلات التفاضلية والتحويلات | 305 | 20221 | `likely_equivalent_needs_review` | `426.111` المعادلات التفاضلية والتحويلات | name_exact, credits, degree_lineage, planned_level | requirement_type, faculty, predecessor_volume_collapse, temporal_complementarity, volume_continuity |
| 34 | `1376.111` | مبادئ المحاسبة2 | 292 | 20222 | `likely_equivalent_needs_review` | `2.111` مبادئ المحاسبة 2 | name_exact, requirement_type, degree_lineage, volume_continuity | credits, faculty, planned_level, predecessor_volume_collapse, temporal_complementarity |
| 35 | `1186.111` | البرمجة3 | 277 | 20221 | `unresolved` | `519.111` برمجة النظم | credits, requirement_type, degree_lineage, volume_continuity | name_exact, faculty, planned_level, predecessor_volume_collapse, temporal_complementarity |
| 36 | `1192.111` | بنيان الحواسيب2 | 272 | 20221 | `likely_equivalent_needs_review` | `510.111` بنيان الحواسيب | credits, requirement_type, degree_lineage, predecessor_volume_collapse, volume_continuity | name_exact, faculty, planned_level, temporal_complementarity |
| 37 | `1309.111` | إحصاء الاعمال1 | 268 | 20222 | `unresolved` | `120.111` إحصاء الأعمال والاقتصاد | credits, requirement_type, degree_lineage, planned_level, volume_continuity | name_exact, faculty, predecessor_volume_collapse, temporal_complementarity |
| 38 | `1373.111` | مبادئ الإدارة المالية | 254 | 20222 | `likely_equivalent_needs_review` | `44.111` مبادئ الإدارة المالية | name_exact, requirement_type, degree_lineage, volume_continuity | credits, faculty, planned_level, predecessor_volume_collapse, temporal_complementarity |
| 39 | `1237.111` | نظم قواعد البيانات | 237 | 20221 | `likely_equivalent_needs_review` | `527.111` قواعد البيانات 2 | credits, requirement_type, degree_lineage, volume_continuity | name_exact, faculty, planned_level, predecessor_volume_collapse, temporal_complementarity |
| 40 | `1193.111` | نظم التشغيل1 | 223 | 20221 | `unresolved` | `528.111` نظم التشغيل | credits, degree_lineage, volume_continuity | name_exact, requirement_type, faculty, planned_level, predecessor_volume_collapse, temporal_complementarity |

## 7. Payoff quantification

counterfactual in-memory substitution of the candidate predecessor course_id into the VALID rows, then exact re-evaluation of the Level-1 (degree_course_key) -> Level-2 (course_id) support lookup of src/course_difficulty.py against TRAIN-only statistics. No data was written and no mapping was persisted.

**Simulation validation.** Before use, the re-implementation of the Level-1 -> Level-2 support lookup was checked against the on-disk columns: 0 `course_history_count` mismatches and 0 `course_difficulty_missing` mismatches over 156,097 VALID rows (verdict: **exact**).

| Scenario | Courses mapped | Rows gaining observed history | Rows crossing the 20-row threshold | never-in-TRAIN rows still uncovered |
|---|---:|---:|---:|---:|
| `confirmed_equivalent` only | 5 | 1,791 | 1,727 | 23,900 |
| `confirmed` + `likely` (upper bound) | 87 | 18,150 | 18,060 | 7,567 |

**Residual against the original 26,882 uncovered VALID rows:**

| Scenario | Uncovered rows remaining | % of the original 26,882 |
|---|---:|---:|
| `confirmed` only | 25,155 | 93.58% |
| `confirmed` + `likely` | 8,822 | 32.82% |

> The payoff is an arithmetic upper bound on *coverage*, not on *accuracy*. Borrowing an old course's pass-rate statistics for a revised course assumes the revision did not change difficulty. Nothing in this data can test that assumption; only outcomes under the new plan can, and those are exactly the rows in question.

## 8. Scope of the change

**Verdict: faculty-specific.**

| Faculty | New courses | VALID rows | Of which equivalent candidates (courses / rows) |
|---|---:|---:|---|
| `167.111` | 85 | 16,303 | 44 / 12,463 |
| `177.111` | 66 | 6,029 | 35 / 3,944 |
| `2.111` | 8 | 2,510 | 7 / 1,656 |
| `7.111` | 17 | 529 | 0 / 0 |
| `5.111` | 3 | 102 | 0 / 0 |
| `4.111` | 1 | 87 | 1 / 87 |
| `3.111` | 2 | 67 | 0 / 0 |

Faculty 167.111 (informatics/communications engineering) and 177.111 (business administration) hold the overwhelming majority of the new courses and rows, and both are effectively absent from TRAIN (2 rows and 0 rows respectively). The remaining faculties contribute small elective additions. The event is a curriculum revision concentrated in two faculties that were re-coded, not a university-wide identifier migration: university-requirement course ids (955, 956, 962, 967, 1015-1021, 1038, 1160-1162) are REUSED UNCHANGED inside the new degrees, which a system-wide id migration would not do.

## 9. Questions only the university registrar can answer

1. Does an official course-equivalence / course-substitution table exist for the 2022 and 2023 curricula, and can it be exported? This investigation can only produce candidates; the registrar's table is the only authoritative source.
2. Was there a formal curriculum revision or accreditation cycle effective in semester 20221, and a second one effective in 20231 (the degree names literally carry a '2023' suffix)? Are these two separate revisions or one phased rollout?
3. Which faculties and degrees did each revision cover? Specifically: were faculty codes 167 (informatics/communications engineering) and 177 (business) newly created, or are they re-codings of the previous faculty codes 5 and 7?
4. Are the new-plan courses (ids 1163+) intended to be academically equivalent to the old-plan courses of the same name, or was content/assessment also revised? Equivalent identifiers do not imply equivalent pass rates.
5. Are old-plan courses being taught out on a published schedule, and until when? This decides whether their historical statistics remain representative.
6. Were students migrated from old degrees to new degrees administratively, and if so, were their completed old-plan courses credited as the new-plan equivalents?
7. Why do course ids 99, 101-117, 447, 489 and 492 exist inside the historical id range but carry no enrolment before 20221 - dormant catalog slots, or ids reused after an earlier course was deleted?

## 10. What was NOT done

- No `canonical_course_id` or equivalence mapping was created, applied or wired.
- No dataset was built, copied or written; no `CURRENT_VERSION.txt`, default, or promotion marker was touched.
- No model was trained, retrained or re-tuned.
- `df_test_final.parquet` was not read, and no TEST path was constructed.
- Nothing was pushed.

