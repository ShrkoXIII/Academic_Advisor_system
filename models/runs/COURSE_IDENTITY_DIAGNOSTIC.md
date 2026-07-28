# Course identity diagnostic

**Status: diagnostic candidates only. No mapping was created or accepted.**

This report reads only the immutable TRAIN/VALID parquets and explicit course-catalog sources. TEST remained `closed_not_read`. Dotted ID suffixes are preserved as identity. Similarity can never create `confirmed_equivalent`; only official university evidence can.

## Headline

- Distinct never-in-TRAIN courses: **182** covering **25,627 VALID rows**.
- Official equivalence/replacement source: **not found**.
- Likely renumbering/content-predecessor candidates needing review: **67 courses / 13,686 rows**.
- If every likely candidate were later confirmed, at most **13,686 uncovered VALID rows** could receive historical course identity.
- Genuinely new: **11 courses / 216 rows**.
- Unresolved: **104 courses / 11,725 rows**.

## Freeze-blocking gate

**MODEL_FREEZE_BLOCKED_BY_COURSE_IDENTITY**

61 likely candidates have exact normalized Arabic-name matches plus multiple structural/temporal signals, covering 10,151 VALID rows. Without an official equivalence source, accepting or rejecting these identity links requires human/university review.

No numerical threshold was invented for the freeze gate. The gate is triggered by direct evidence: numerous high-volume courses have exact normalized Arabic-name matches plus multiple structural matches, yet no official source exists to adjudicate identity. A human/university mapping review is required before a final model-specification freeze.

## Preconditions and target reproduction

| Measure | Recomputed | Required |
| --- | --- | --- |
| VALID model-facing uncovered rows | 26,882 | 26,882 |
| never_in_train rows | 25,627 | 25,627 |
| thin_history rows | 1,255 | 1,255 |
| distinct never-in-TRAIN course IDs | 182 | recompute (inherited ≈182) |

## Source inventory

| Evidence field | Available | Actual source |
| --- | --- | --- |
| course_names_arabic | yes | course_name_sl and course_official_sl |
| course_names_english | no | unavailable |
| university | yes | university_id in TRAIN/VALID and dotted ID suffix |
| degree | yes | degree_id and degree_name_sl |
| faculty | yes | faculty_id in TRAIN/VALID and raw catalog |
| credits | yes | course_credits |
| course_type | yes | course_type_id in raw catalog |
| requirement_type | yes | requirement_type_id and requirement_type_sl |
| planned_year_or_level | yes | year_order in raw catalog |
| planned_semester | yes | semester_order in raw catalog |
| prerequisites | no | unavailable |
| equivalent_or_replacement_course_ids | no | unavailable |
| active_inactive_dates | no | unavailable |
| active_flag | yes | active in raw catalog (non-temporal flag only) |
| curriculum_or_version_identifier | no | unavailable |

The canonical cleaned `V_ACD_DEGREE_COURSE` representation is the identity base. The raw representation supplements fields removed during cleaning. Arabic names are retained. No English-name column exists. Prerequisite similarity is blank in the CSV because no prerequisite source exists; it is not fabricated as zero.

## Deterministic candidate score

Score components are persisted for every candidate in JSON: exact name 30; strong-name similarity up to 20; same university 5; degree 10; faculty 5; credits 8; course type 5; requirement type 7; planned level 5; strict temporal replacement 10. Official mapping evidence would add 100 and is the only route to `confirmed_equivalent`.

A candidate is `likely_renumbered_needs_review` only when normalized name similarity is at least 0.85, score is at least 55, and at least two structural/temporal signals match. A high score remains review-only without official evidence.

## Classification summary

| Status | Courses | VALID rows | % never-in-TRAIN rows |
| --- | --- | --- | --- |
| confirmed_equivalent | 0 | 0 | 0.00% |
| likely_renumbered_needs_review | 67 | 13,686 | 53.40% |
| genuinely_new | 11 | 216 | 0.84% |
| unresolved | 104 | 11,725 | 45.75% |

## Top likely renumbering candidates

| New course | New Arabic name | Rows | Old candidate | Old Arabic name | Name sim | Score | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1423.111 | اللغة الإنكليزية – 1 | 979 | 391.111 | اللغة الإنكليزية 1 | 1.000 | 75.0 | exact normalized name; same credits; same requirement type; same planned level; similarity cannot confirm equivalence without an official source |
| 1175.111 | التحليل الرياضي1 | 917 | 502.111 | التحليل الرياضي 1 | 0.970 | 56.4 | same degree; same faculty; same course type; same requirement type; same planned level; similarity cannot confirm equivalence without an official source |
| 1172.111 | الرياضيات المتقطعة | 876 | 431.111 | الرياضيات المتقطعة | 1.000 | 90.0 | exact normalized name; same degree; same faculty; same credits; same requirement type; same planned level; similarity cannot confirm equivalence without an official source |
| 1174.111 | الفيزياء1 | 791 | 501.111 | الفيزياء 1 | 0.947 | 63.9 | same degree; same faculty; same credits; same course type; same requirement type; same planned level; similarity cannot confirm equivalence without an official source |
| 1179.111 | الفيزياء2 | 759 | 457.111 | الفيزياء 2 | 0.947 | 56.9 | same degree; same faculty; same credits; same course type; same planned level; similarity cannot confirm equivalence without an official source |
| 1173.111 | الجبر الخطي ونظرية المصفوفات | 719 | 432.111 | الجبر الخطي ونظرية المصفوفات | 1.000 | 87.0 | exact normalized name; same degree; same faculty; same course type; same requirement type; same planned level; similarity cannot confirm equivalence without an official source |
| 1422.111 | مهارات الحاسوب | 686 | 967.111 | مهارات الحاسوب | 1.000 | 95.0 | exact normalized name; same degree; same faculty; same credits; same course type; same requirement type; same planned level; similarity cannot confirm equivalence without an official source |
| 1365.111 | رياضيات الأعمال | 638 | 118.111 | رياضيات الأعمال | 1.000 | 85.0 | exact normalized name; same faculty; same credits; same course type; same requirement type; same planned level; similarity cannot confirm equivalence without an official source |
| 1332.111 | إدارة الموارد البشرية | 594 | 73.111 | إدارة الموارد البشرية | 1.000 | 80.0 | exact normalized name; same faculty; same credits; same course type; same requirement type; similarity cannot confirm equivalence without an official source |
| 1372.111 | مبادئ الإدارة | 579 | 70.111 | مبادئ الإدارة | 1.000 | 77.0 | exact normalized name; same faculty; same course type; same requirement type; same planned level; similarity cannot confirm equivalence without an official source |
| 1178.111 | الدارات المنطقية | 567 | 434.111 | الدارات المنطقية | 1.000 | 72.0 | exact normalized name; same faculty; same course type; same requirement type; similarity cannot confirm equivalence without an official source |
| 1183.111 | بنيان الحواسيب1 | 495 | 510.111 | بنيان الحواسيب | 0.966 | 59.3 | same degree; same faculty; same credits; same requirement type; same planned level; similarity cannot confirm equivalence without an official source |
| 1184.111 | الخوارزميات وبنى المعطيات1 | 486 | 516.111 | الخوارزميات وبنى المعطيات 1 | 0.981 | 64.6 | same degree; same faculty; same credits; same course type; same requirement type; same planned level; similarity cannot confirm equivalence without an official source |
| 1424.111 | اللغة الإنكليزية – 2 | 471 | 392.111 | اللغة الإنكليزية 2 | 1.000 | 75.0 | exact normalized name; same credits; same requirement type; same planned level; similarity cannot confirm equivalence without an official source |
| 1182.111 | الإحصاء والاحتمالات | 378 | 504.111 | الإحصاء والاحتمالات | 1.000 | 85.0 | exact normalized name; same faculty; same credits; same course type; same requirement type; same planned level; similarity cannot confirm equivalence without an official source |
| 1282.111 | الاقتصاد الجزئي | 373 | 45.111 | الاقتصاد الجزئي | 1.000 | 72.0 | exact normalized name; same faculty; same requirement type; same planned level; similarity cannot confirm equivalence without an official source |
| 1187.111 | نظرية الحوسبة | 371 | 521.111 | نظرية الحوسبة | 1.000 | 78.0 | exact normalized name; same faculty; same credits; same course type; same planned level; similarity cannot confirm equivalence without an official source |
| 1195.111 | نظرية المعلومات | 316 | 437.111 | نظرية المعلومات | 1.000 | 85.0 | exact normalized name; same faculty; same credits; same course type; same requirement type; same planned level; similarity cannot confirm equivalence without an official source |
| 1429.111 | علم البيئة | 308 | 1021.111 | علم البيئة | 1.000 | 95.0 | exact normalized name; same degree; same faculty; same credits; same course type; same requirement type; same planned level; similarity cannot confirm equivalence without an official source |
| 1190.111 | المعادلات التفاضلية والتحويلات | 305 | 426.111 | المعادلات التفاضلية والتحويلات | 1.000 | 78.0 | exact normalized name; same faculty; same credits; same course type; same planned level; similarity cannot confirm equivalence without an official source |
| 1373.111 | مبادئ الإدارة المالية | 254 | 44.111 | مبادئ الإدارة المالية | 1.000 | 67.0 | exact normalized name; same faculty; same requirement type; similarity cannot confirm equivalence without an official source |
| 1346.111 | بحوث العمليات | 190 | 518.111 | بحوث العمليات | 1.000 | 75.0 | exact normalized name; same credits; same course type; same requirement type; similarity cannot confirm equivalence without an official source |
| 1164.111 | التحليل العددي | 185 | 503.111 | التحليل العددي | 1.000 | 95.0 | exact normalized name; same degree; same faculty; same credits; same course type; same requirement type; same planned level; similarity cannot confirm equivalence without an official source |
| 1425.111 | مدخل الى القانون | 185 | 1019.111 | مدخل الى القانون | 1.000 | 95.0 | exact normalized name; same degree; same faculty; same credits; same course type; same requirement type; same planned level; similarity cannot confirm equivalence without an official source |
| 1196.111 | شبكات الحاسوب | 149 | 466.111 | شبكات الحاسوب | 1.000 | 73.0 | exact normalized name; same faculty; same credits; same course type; similarity cannot confirm equivalence without an official source |

Full per-course review data is in `models/runs/COURSE_IDENTITY_CANDIDATES.csv`. It is not a mapping table.

## Guardrail confirmations

- No `canonical_course_id` mapping was created or accepted.
- No dataset or parquet was written or modified.
- TEST was not constructed or read.
- No model was trained or scored.
- No source/default/promotion/inference/recommendation wiring changed.
