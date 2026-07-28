# Course identity: 67-candidate degree verification

**Diagnosis only. No course mapping was accepted.**

The verification uses every catalog row for each old and new course in the canonical and raw `V_ACD_DEGREE_COURSE` representations, plus immutable TRAIN and VALID enrolment history. TEST remained `closed_not_read`; no model was trained or scored.

## Direct answers

1. Same university: **67 / 67**.
2. At least one exact shared normalized `degree_id`: **0 / 67**.
3. Same university + same degree + different `course_id`: **0 / 67**.
4. Exact normalized Arabic-name match + same degree: **0 / 67**.
5. Same degree plus matching credits, course type, requirement type, and planned year: **0 / 67**.
6. Direct or one-semester-gap temporal replacement with no enrolment overlap: **0 / 67**.
7. Strict same-degree renumbering candidates: **0 / 67**.
8. Same university but no shared degree: **67 / 67**.
9. Partial degree overlap: **0 / 67**.
10. Ambiguous/insufficient or tied-best cases: **0 / 67**.

## Direct-answer VALID-row prevalence

| group | candidates | VALID rows |
| --- | --- | --- |
| same university | 67 | 13,686 |
| same exact degree | 0 | 0 |
| same university + same degree + different course_id | 0 | 0 |
| exact name + same degree | 0 | 0 |
| strict same-degree candidate | 0 | 0 |
| same university, different degree | 67 | 13,686 |
| partial degree overlap | 0 | 0 |
| ambiguous | 0 | 0 |

A strict result is still only a candidate. It requires the same exact university and degree sets, a different course ID, name similarity of at least 0.85, exact agreement on faculty/credits/course type/requirement type/planned year/planned semester, and direct or one-semester-gap replacement without overlap. Similarity never creates a confirmed equivalence.

## Why the exact-degree result differs from the earlier candidate score

The earlier diagnostic built a course-level degree set by pooling `degree_id` values from enrolment rows with catalog rows. That broad profile was useful for candidate discovery, but it did not establish that the old and new catalog records belonged to the same degree. This verification uses every actual catalog row and compares full normalized `degree_id` strings directly. Under that stricter question, all 67 pairs have disjoint old/new catalog degree sets.

Temporal history also argues against a simple ID-only replacement: 64 pairs have both IDs enrolled in at least one common semester, and the remaining 3 have a long gap. None has direct or one-semester-gap replacement without overlap.

## Degree relationship and VALID-row prevalence

| diagnostic conclusion | candidates | VALID rows |
| --- | --- | --- |
| STRICT_SAME_DEGREE_RENUMBERING_CANDIDATE | 0 | 0 |
| SAME_DEGREE_BUT_STRUCTURAL_DIFFERENCE | 0 | 0 |
| PARTIAL_DEGREE_OVERLAP | 0 | 0 |
| SAME_UNIVERSITY_DIFFERENT_DEGREE | 67 | 13,686 |
| DIFFERENT_UNIVERSITY | 0 | 0 |
| INSUFFICIENT_EVIDENCE | 0 | 0 |

## Temporal evidence

| temporal signal | candidates | VALID rows |
| --- | --- | --- |
| LONG_GAP | 3 | 1,463 |
| OVERLAPPING_ENROLMENT | 64 | 12,223 |

## All 67 best-per-course cases

| new_course_id | old_course_id | new course | old course | university comparison | degree comparison | same degree | different ID | structural matches | temporal evidence | diagnostic conclusion | VALID rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1423.111 | 391.111 | اللغة الإنكليزية – 1 | اللغة الإنكليزية 1 | 111 → 111 | 18.111 → 40.111\|41.111\|42.111\|44.111\|45.111\|46.111\|47.111\|48.111\|49.111\|50.111\|51.111\|52.111\|53.111\|54.111\|55.111\|56.111\|57.111\|58.111\|59.111\|60.111\|61.111\|64.111\|65.111 | False | True | credits=True; type=False; req=True; year=True; semester=True | LONG_GAP | SAME_UNIVERSITY_DIFFERENT_DEGREE | 979 |
| 1175.111 | 502.111 | التحليل الرياضي1 | التحليل الرياضي 1 | 111 → 111 | 21.111\|3.111\|4.111\|5.111\|6.111 → 26.111\|27.111\|29.111\|30.111\|31.111\|45.111\|46.111\|47.111\|48.111\|49.111 | False | True | credits=False; type=True; req=True; year=True; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 917 |
| 1172.111 | 431.111 | الرياضيات المتقطعة | الرياضيات المتقطعة | 111 → 111 | 21.111\|3.111\|4.111\|5.111\|6.111 → 26.111\|27.111\|29.111\|30.111\|31.111\|45.111\|46.111\|47.111\|48.111\|49.111 | False | True | credits=True; type=False; req=True; year=True; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 876 |
| 1174.111 | 501.111 | الفيزياء1 | الفيزياء 1 | 111 → 111 | 21.111\|3.111\|4.111\|5.111\|6.111 → 26.111\|27.111\|29.111\|30.111\|31.111\|45.111\|46.111\|47.111\|48.111\|49.111 | False | True | credits=True; type=True; req=True; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 791 |
| 1179.111 | 457.111 | الفيزياء2 | الفيزياء 2 | 111 → 111 | 5.111\|6.111 → 26.111\|27.111\|29.111\|30.111\|31.111\|45.111\|46.111\|47.111\|48.111\|49.111 | False | True | credits=True; type=True; req=False; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 759 |
| 1173.111 | 432.111 | الجبر الخطي ونظرية المصفوفات | الجبر الخطي ونظرية المصفوفات | 111 → 111 | 21.111\|3.111\|4.111\|5.111\|6.111 → 26.111\|27.111\|29.111\|30.111\|31.111\|45.111\|46.111\|47.111\|48.111\|49.111 | False | True | credits=False; type=True; req=True; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 719 |
| 1422.111 | 967.111 | مهارات الحاسوب | مهارات الحاسوب | 111 → 111 | 1.111\|10.111\|11.111\|12.111\|13.111\|2.111\|21.111\|22.111\|24.111\|25.111\|26.111\|27.111\|29.111\|3.111\|30.111\|31.111\|33.111\|34.111\|35.111\|36.111\|37.111\|38.111\|39.111\|4.111\|5.111\|6.111\|7.111\|8.111\|9.111 → 40.111\|41.111\|42.111\|44.111\|45.111\|46.111\|47.111\|48.111\|49.111\|50.111\|51.111\|52.111\|53.111\|54.111\|55.111\|56.111\|57.111\|58.111\|59.111\|60.111\|61.111\|64.111\|65.111 | False | True | credits=True; type=True; req=True; year=False; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 686 |
| 1365.111 | 118.111 | رياضيات الأعمال | رياضيات الأعمال | 111 → 111 | 10.111\|11.111\|12.111\|22.111\|7.111\|8.111\|9.111 → 33.111\|34.111\|35.111\|36.111\|37.111\|38.111\|39.111\|40.111\|51.111\|52.111\|53.111\|54.111\|55.111\|56.111 | False | True | credits=True; type=True; req=True; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 638 |
| 1332.111 | 73.111 | إدارة الموارد البشرية | إدارة الموارد البشرية | 111 → 111 | 10.111\|11.111\|12.111\|22.111\|7.111\|8.111\|9.111 → 33.111\|34.111\|35.111\|36.111\|37.111\|38.111\|39.111\|40.111\|51.111\|52.111\|53.111\|54.111\|55.111\|56.111 | False | True | credits=True; type=True; req=True; year=False; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 594 |
| 1372.111 | 70.111 | مبادئ الإدارة | مبادئ الإدارة | 111 → 111 | 10.111\|11.111\|12.111\|22.111\|7.111\|8.111\|9.111 → 33.111\|34.111\|35.111\|36.111\|37.111\|38.111\|39.111\|40.111\|51.111\|52.111\|53.111\|54.111\|55.111\|56.111 | False | True | credits=False; type=True; req=True; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 579 |
| 1178.111 | 434.111 | الدارات المنطقية | الدارات المنطقية | 111 → 111 | 21.111\|3.111\|4.111\|5.111\|6.111 → 26.111\|27.111\|29.111\|30.111\|31.111\|45.111\|46.111\|47.111\|48.111\|49.111 | False | True | credits=False; type=True; req=True; year=False; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 567 |
| 1183.111 | 510.111 | بنيان الحواسيب1 | بنيان الحواسيب | 111 → 111 | 21.111\|3.111\|4.111\|5.111\|6.111 → 26.111\|27.111\|29.111\|30.111\|31.111\|45.111\|46.111\|47.111\|48.111\|49.111 | False | True | credits=True; type=False; req=True; year=True; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 495 |
| 1184.111 | 516.111 | الخوارزميات وبنى المعطيات1 | الخوارزميات وبنى المعطيات 1 | 111 → 111 | 21.111\|3.111\|4.111\|5.111\|6.111 → 26.111\|27.111\|29.111\|30.111\|31.111\|45.111\|46.111\|47.111\|48.111\|49.111 | False | True | credits=True; type=True; req=True; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 486 |
| 1424.111 | 392.111 | اللغة الإنكليزية – 2 | اللغة الإنكليزية 2 | 111 → 111 | 18.111 → 40.111\|41.111\|42.111\|44.111\|45.111\|46.111\|47.111\|48.111\|49.111\|50.111\|51.111\|52.111\|53.111\|54.111\|55.111\|56.111\|57.111\|58.111\|59.111\|60.111\|61.111\|64.111\|65.111 | False | True | credits=True; type=False; req=True; year=False; semester=False | LONG_GAP | SAME_UNIVERSITY_DIFFERENT_DEGREE | 471 |
| 1182.111 | 504.111 | الإحصاء والاحتمالات | الإحصاء والاحتمالات | 111 → 111 | 21.111\|3.111\|4.111\|5.111\|6.111 → 26.111\|27.111\|29.111\|30.111\|31.111\|45.111\|46.111\|47.111\|48.111\|49.111 | False | True | credits=True; type=True; req=True; year=True; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 378 |
| 1282.111 | 45.111 | الاقتصاد الجزئي | الاقتصاد الجزئي | 111 → 111 | 10.111\|11.111\|12.111\|22.111\|7.111\|8.111\|9.111 → 33.111\|34.111\|35.111\|36.111\|37.111\|38.111\|39.111\|40.111\|51.111\|52.111\|53.111\|54.111\|55.111\|56.111 | False | True | credits=False; type=False; req=True; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 373 |
| 1187.111 | 521.111 | نظرية الحوسبة | نظرية الحوسبة | 111 → 111 | 3.111\|4.111 → 26.111\|27.111\|29.111\|30.111\|31.111\|45.111\|46.111\|47.111\|48.111\|49.111 | False | True | credits=True; type=True; req=False; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 371 |
| 1195.111 | 437.111 | نظرية المعلومات | نظرية المعلومات | 111 → 111 | 21.111\|3.111\|4.111\|5.111\|6.111 → 26.111\|27.111\|29.111\|30.111\|31.111\|45.111\|46.111\|47.111\|48.111\|49.111 | False | True | credits=True; type=True; req=True; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 316 |
| 1429.111 | 1021.111 | علم البيئة | علم البيئة | 111 → 111 | 1.111\|10.111\|11.111\|12.111\|13.111\|2.111\|21.111\|22.111\|24.111\|25.111\|26.111\|27.111\|29.111\|3.111\|30.111\|31.111\|33.111\|34.111\|35.111\|36.111\|37.111\|38.111\|39.111\|4.111\|5.111\|6.111\|7.111\|8.111\|9.111 → 40.111\|41.111\|42.111\|44.111\|45.111\|46.111\|47.111\|48.111\|49.111\|50.111\|51.111\|52.111\|53.111\|54.111\|55.111\|56.111\|57.111\|58.111\|59.111\|60.111\|61.111\|64.111\|65.111 | False | True | credits=True; type=True; req=True; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 308 |
| 1190.111 | 426.111 | المعادلات التفاضلية والتحويلات | المعادلات التفاضلية والتحويلات | 111 → 111 | 5.111\|6.111 → 26.111\|27.111\|29.111\|30.111\|31.111\|45.111\|46.111\|47.111\|48.111\|49.111 | False | True | credits=True; type=True; req=False; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 305 |
| 1373.111 | 44.111 | مبادئ الإدارة المالية | مبادئ الإدارة المالية | 111 → 111 | 10.111\|11.111\|12.111\|22.111\|7.111\|8.111\|9.111 → 33.111\|34.111\|35.111\|36.111\|37.111\|38.111\|39.111\|40.111\|51.111\|52.111\|53.111\|54.111\|55.111\|56.111 | False | True | credits=False; type=False; req=True; year=False; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 254 |
| 1346.111 | 518.111 | بحوث العمليات | بحوث العمليات | 111 → 111 | 21.111\|3.111\|4.111\|5.111\|6.111 → 33.111\|34.111\|35.111\|36.111\|37.111\|38.111\|39.111\|40.111\|51.111\|52.111\|53.111\|54.111\|55.111\|56.111 | False | True | credits=True; type=True; req=True; year=False; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 190 |
| 1164.111 | 503.111 | التحليل العددي | التحليل العددي | 111 → 111 | 21.111\|3.111\|4.111\|5.111\|6.111 → 26.111\|27.111\|29.111\|30.111\|31.111\|45.111\|46.111\|47.111\|48.111\|49.111 | False | True | credits=True; type=True; req=True; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 185 |
| 1425.111 | 1019.111 | مدخل الى القانون | مدخل الى القانون | 111 → 111 | 1.111\|10.111\|11.111\|12.111\|13.111\|2.111\|21.111\|22.111\|24.111\|25.111\|26.111\|27.111\|29.111\|3.111\|30.111\|31.111\|33.111\|34.111\|35.111\|36.111\|37.111\|38.111\|39.111\|4.111\|5.111\|6.111\|7.111\|8.111\|9.111 → 40.111\|41.111\|42.111\|44.111\|45.111\|46.111\|47.111\|48.111\|49.111\|50.111\|51.111\|52.111\|53.111\|54.111\|55.111\|56.111\|57.111\|58.111\|59.111\|60.111\|61.111\|64.111\|65.111 | False | True | credits=True; type=True; req=True; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 185 |
| 1196.111 | 466.111 | شبكات الحاسوب | شبكات الحاسوب | 111 → 111 | 3.111\|5.111\|6.111 → 26.111\|27.111\|29.111\|30.111\|31.111\|45.111\|46.111\|47.111\|48.111\|49.111 | False | True | credits=True; type=True; req=False; year=False; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 149 |
| 1235.111 | 523.111 | تصميم المترجمات | تصميم المترجمات | 111 → 111 | 3.111\|4.111 → 26.111\|27.111\|29.111\|46.111\|47.111\|48.111 | False | True | credits=True; type=True; req=False; year=False; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 140 |
| 1166.111 | 505.111 | إدارة المشاريع | إدارة المشاريع | 111 → 111 | 21.111\|3.111\|4.111\|5.111\|6.111 → 26.111\|27.111\|29.111\|30.111\|31.111\|45.111\|46.111\|47.111\|48.111\|49.111 | False | True | credits=True; type=True; req=True; year=False; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 112 |
| 1283.111 | 48.111 | الاقتصاد الكلي | الاقتصاد الكلي | 111 → 111 | 10.111\|11.111\|12.111\|22.111\|7.111\|8.111\|9.111 → 33.111\|34.111\|35.111\|36.111\|37.111\|38.111\|39.111\|40.111\|51.111\|52.111\|53.111\|54.111\|55.111\|56.111 | False | True | credits=False; type=False; req=True; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 109 |
| 1303.111 | 59.111 | المالية العامة | المالية العامة | 111 → 111 | 10.111 → 33.111\|34.111\|35.111\|36.111\|37.111\|38.111\|39.111\|40.111\|51.111\|52.111\|53.111\|54.111\|55.111\|56.111 | False | True | credits=True; type=True; req=False; year=False; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 105 |
| 1412.111 | 545.111 | نظم المعلومات الإدارية | نظم المعلومات الإدارية | 111 → 111 | 3.111 → 33.111\|34.111\|35.111\|36.111\|37.111\|38.111\|39.111\|40.111\|51.111\|52.111\|53.111\|54.111\|55.111\|56.111 | False | True | credits=True; type=True; req=False; year=False; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 91 |
| 1246.111 | 533.111 | قواعد البيانات المتقدمة | قواعد البيانات المتقدمة | 111 → 111 | 3.111\|4.111 → 26.111\|27.111\|46.111\|47.111 | False | True | credits=True; type=True; req=False; year=True; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 90 |
| 1426.111 | 1018.111 | علم الاجتماع | علم اجتماع | 111 → 111 | 1.111\|10.111\|11.111\|12.111\|13.111\|2.111\|21.111\|22.111\|24.111\|25.111\|26.111\|27.111\|29.111\|3.111\|30.111\|31.111\|33.111\|34.111\|35.111\|36.111\|37.111\|38.111\|39.111\|4.111\|5.111\|6.111\|7.111\|8.111\|9.111 → 40.111\|41.111\|42.111\|44.111\|45.111\|46.111\|47.111\|48.111\|49.111\|50.111\|51.111\|52.111\|53.111\|54.111\|55.111\|56.111\|57.111\|58.111\|59.111\|60.111\|61.111\|64.111\|65.111 | False | True | credits=True; type=True; req=True; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 87 |
| 1205.111 | 451.111 | معالجة الصور وتحليلها | معالجة الصور وتحليلها | 111 → 111 | 3.111\|4.111\|5.111 → 26.111\|27.111\|46.111\|47.111 | False | True | credits=True; type=True; req=False; year=False; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 55 |
| 1409.111 | 24.111 | نظرية المحاسبة | نظرية المحاسبة | 111 → 111 | 11.111 → 33.111\|34.111\|35.111\|36.111\|37.111\|38.111\|39.111\|40.111\|51.111\|52.111\|53.111\|54.111\|55.111\|56.111 | False | True | credits=True; type=True; req=False; year=False; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 52 |
| 1249.111 | 545.111 | نظم المعلومات الإدارية | نظم المعلومات الإدارية | 111 → 111 | 3.111 → 26.111\|27.111\|46.111 | False | True | credits=True; type=False; req=True; year=False; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 32 |
| 1216.111 | 458.111 | الإشارات والنظم | الإشارات والنظم | 111 → 111 | 5.111\|6.111 → 29.111\|30.111\|31.111\|45.111\|48.111\|49.111 | False | True | credits=True; type=True; req=True; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 27 |
| 1198.111 | 538.111 | مشروع فصلي | مشروع  فصلي | 111 → 111 | 3.111 → 26.111\|27.111\|29.111\|30.111\|31.111\|45.111\|46.111\|47.111\|48.111\|49.111 | False | True | credits=True; type=True; req=False; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 25 |
| 1324.111 | 72.111 | إدارة الجودة | إدارة الجودة | 111 → 111 | 10.111\|11.111\|12.111\|22.111\|7.111\|8.111\|9.111 → 33.111\|34.111\|35.111\|36.111\|37.111\|38.111\|39.111\|40.111\|51.111\|52.111\|53.111\|54.111\|55.111\|56.111 | False | True | credits=False; type=False; req=False; year=True; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 25 |
| 1165.111 | 662.111 | مهارات التواصل | مهارات التواصل | 111 → 111 | 2.111\|41.111 → 26.111\|27.111\|29.111\|30.111\|31.111\|45.111\|46.111\|47.111\|48.111\|49.111 | False | True | credits=True; type=True; req=False; year=False; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 21 |
| 1252.111 | 461.111 | المتحكمات الصغرية والنظم المضمنة | المتحكمات الصغرية والنظم المضمنة | 111 → 111 | 5.111\|6.111 → 29.111\|30.111\|31.111\|45.111\|48.111\|49.111 | False | True | credits=True; type=True; req=False; year=False; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 14 |
| 1255.111 | 445.111 | مدخل إلى الروبوتية | مدخل إلى الروبوتية | 111 → 111 | 4.111 → 27.111\|30.111\|45.111\|47.111 | False | True | credits=True; type=True; req=True; year=False; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 14 |
| 1271.111 | 417.111 | برمجة التطبيقات الشبكية | برمجة التطبيقات الشبكية | 111 → 111 | 18.111 → 29.111\|48.111 | False | True | credits=True; type=True; req=True; year=False; semester=True | LONG_GAP | SAME_UNIVERSITY_DIFFERENT_DEGREE | 13 |
| 1212.111 | 448.111 | معالجة اللغات الطبيعية | معالجة اللغات الطبيعية | 111 → 111 | 4.111 → 27.111\|47.111 | False | True | credits=True; type=True; req=False; year=False; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 12 |
| 1214.111 | 441.111 | النظم الخبيرة | النظم الخبيرة | 111 → 111 | 3.111\|4.111 → 27.111\|47.111 | False | True | credits=True; type=True; req=False; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 11 |
| 1299.111 | 82.111 | السلوك التنظيمي | السلوك التنظيمي | 111 → 111 | 8.111 → 33.111\|34.111\|53.111\|54.111 | False | True | credits=True; type=True; req=True; year=False; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 11 |
| 1378.111 | 8.111 | محاسبة المؤسسات المالية والمصرفية | محاسبة المؤسسات المالية والمصرفية | 111 → 111 | 10.111\|11.111 → 36.111\|37.111\|52.111\|55.111 | False | True | credits=True; type=False; req=True; year=False; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 11 |
| 1318.111 | 81.111 | إدارة التغيير | إدارة التغيير | 111 → 111 | 8.111 → 33.111\|34.111\|53.111\|54.111 | False | True | credits=True; type=True; req=True; year=True; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 8 |
| 1167.111 | 509.111 | إدارة المؤسسات | إدارة المؤسسات | 111 → 111 | 21.111\|3.111\|4.111\|5.111\|6.111 → 26.111\|27.111\|29.111\|30.111\|31.111\|45.111\|46.111\|47.111\|48.111\|49.111 | False | True | credits=True; type=True; req=True; year=False; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 7 |
| 1386.111 | 12.111 | محاسبة مالية خاصة | محاسبة مالية خاصة | 111 → 111 | 11.111 → 36.111\|55.111 | False | True | credits=True; type=False; req=True; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 7 |
| 1403.111 | 20.111 | معايير التدقيق الدولية | معايير التدقيق الدولية | 111 → 111 | 11.111 → 36.111\|55.111 | False | True | credits=True; type=True; req=True; year=False; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 6 |
| 1291.111 | 23.111 | التدقيق الداخلي | التدقيق الداخلي | 111 → 111 | 11.111 → 36.111\|55.111 | False | True | credits=True; type=True; req=True; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 5 |
| 1339.111 | 74.111 | إنكليزي الأعمال | إنكليزي الأعمال | 111 → 111 | 10.111\|11.111\|12.111\|22.111\|7.111\|8.111\|9.111 → 34.111\|35.111\|53.111\|56.111 | False | True | credits=True; type=True; req=False; year=False; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 3 |
| 1366.111 | 51.111 | رياضيات مالية | رياضيات مالية | 111 → 111 | 10.111 → 37.111\|52.111 | False | True | credits=True; type=False; req=True; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 3 |
| 1430.111 | 1015.111 | اللغة الانكليزية - 3 | اللغة الانكليزية 3 | 111 → 111 | 1.111\|10.111\|11.111\|12.111\|13.111\|2.111\|21.111\|22.111\|24.111\|25.111\|26.111\|27.111\|29.111\|3.111\|30.111\|31.111\|33.111\|34.111\|35.111\|36.111\|37.111\|38.111\|39.111\|4.111\|5.111\|6.111\|7.111\|8.111\|9.111 → 40.111\|41.111\|42.111\|44.111\|45.111\|46.111\|47.111\|48.111\|49.111\|50.111\|51.111\|52.111\|53.111\|54.111\|55.111\|56.111\|57.111\|58.111\|59.111\|60.111\|61.111\|64.111\|65.111 | False | True | credits=True; type=True; req=True; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 3 |
| 1300.111 | 100.111 | العلاقات العامة | العلاقات العامة | 111 → 111 | 7.111\|8.111 → 35.111\|56.111 | False | True | credits=True; type=True; req=True; year=False; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 2 |
| 1306.111 | 53.111 | المصارف الاسلامية | المصارف الاسلامية | 111 → 111 | 10.111 → 37.111\|52.111 | False | True | credits=True; type=True; req=True; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 2 |
| 1419.111 | 49.111 | نقود ومصارف | نقود ومصارف | 111 → 111 | 10.111 → 33.111\|37.111\|52.111\|54.111 | False | True | credits=True; type=True; req=False; year=True; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 2 |
| 1224.111 | 463.111 | الاتصالات الرقمية | الاتصالات الرقمية | 111 → 111 | 5.111\|6.111 → 31.111\|49.111 | False | True | credits=True; type=True; req=True; year=True; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 1 |
| 1231.111 | 474.111 | الاتصالات النقالة واللاسلكية | الاتصالات النقالة واللاسلكية | 111 → 111 | 3.111\|6.111 → 31.111\|49.111 | False | True | credits=True; type=True; req=False; year=True; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 1 |
| 1277.111 | 536.111 | نظم الزمن الحقيقي | نظم الزمن الحقيقي | 111 → 111 | 3.111\|5.111\|6.111 → 29.111\|48.111 | False | True | credits=True; type=True; req=False; year=False; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 1 |
| 1304.111 | 16.111 | المحاسبة الدولية | المحاسبة الدولية | 111 → 111 | 11.111 → 36.111\|55.111 | False | True | credits=True; type=False; req=True; year=True; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 1 |
| 1305.111 | 15.111 | المحاسبة الضريبية | المحاسبة الضريبية | 111 → 111 | 11.111 → 36.111\|55.111 | False | True | credits=True; type=False; req=True; year=False; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 1 |
| 1379.111 | 7.111 | محاسبة إدارية | محاسبة إدارية | 111 → 111 | 11.111\|12.111 → 36.111\|55.111 | False | True | credits=True; type=False; req=False; year=False; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 1 |
| 1382.111 | 17.111 | محاسبة توحيد الأعمال | محاسبة توحيد الأعمال | 111 → 111 | 11.111 → 36.111\|55.111 | False | True | credits=True; type=False; req=True; year=True; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 1 |
| 1383.111 | 6.111 | محاسبة حكومية ومؤسسات غير ربحية | محاسبة حكومية  ومؤسسات غير ربحية | 111 → 111 | 11.111 → 36.111\|55.111 | False | True | credits=True; type=False; req=True; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 1 |
| 1398.111 | 25.111 | مشروع تخرج في المحاسبة والتدقيق | مشروع  تخرج في المحاسبة والتدقيق | 111 → 111 | 11.111 → 36.111\|55.111 | False | True | credits=True; type=True; req=True; year=True; semester=True | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 1 |
| 1402.111 | 18.111 | مشكلات محاسبية معاصرة | مشكلات محاسبية معاصرة | 111 → 111 | 11.111 → 36.111\|55.111 | False | True | credits=True; type=False; req=True; year=False; semester=False | OVERLAPPING_ENROLMENT | SAME_UNIVERSITY_DIFFERENT_DEGREE | 1 |

## Source and integrity controls

- Candidate input: `models\runs\COURSE_IDENTITY_CANDIDATES.csv`.
- Canonical catalog: `data\preprocessed\V_ACD_DEGREE_COURSE\clean_v_acd_degree_course.parquet`.
- Raw supplement: `data\raw\v_acd_degree_course.parquet`.
- TRAIN: `data\model_data\versions\2026-07-26_batched_fixes__registration_roster_concurrent\df_train_final.parquet`.
- VALID: `data\model_data\versions\2026-07-26_batched_fixes__registration_roster_concurrent\df_valid_final.parquet`.
- Catalog rows were retained at `degree_course_id` grain; no course was reduced to its first catalog row.
- `same_degree` is the intersection of actual normalized full `degree_id` strings. Names, faculty, and dotted suffixes cannot make `same_degree` true.
- Meaningful dotted ID suffixes were preserved as strings. Catalog university identity uses the degree suffix and was cross-checked against TRAIN/VALID `university_id` where available.
- Every input hash matched before and after generation.
- No mapping was accepted; no dataset or source was changed; no model was trained or scored; TEST remained `closed_not_read`.
- Model freeze remains blocked pending human/university review.
