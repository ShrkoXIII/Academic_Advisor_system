# ملخص مرحلة KNN

آخر تحديث: 2026-08-23

## الحالة الحالية

مرحلة KNN **منجزة جزئيًا وليست مغلقة بعد**.

اكتمل بناء History V2 من مصدر المستوى الرسمي، واكتمل KNN V2 الذي يبحث داخل نفس الدرجة والمستوى ويعيد snapshot واحدة فقط لكل طالب تاريخي. ما زال يلزم تقييم VALID، تشابه الخطط، ثم الربط بنتائج M1/M2.

## ما تم بناؤه

### 1. جدولا KNN History

بُنيا من TRAIN فقط، دون قراءة VALID أو TEST ودون تدريب نموذج:

```text
data/artifacts/knn/2026-08-23_history_v2_level/
    student_semester_outcomes.parquet
    student_semester_courses.parquet
```

#### `student_semester_outcomes.parquet`

- الحبة: صف واحد لكل `university_id + student_id + degree_id + part_id`.
- عدد الصفوف: 130,277.
- يحتوي حالة الطالب قبل الفصل ونتيجة الفصل بعد اكتماله.
- يحتوي GPA الفصل، التغير عن الفصل السابق، التغير التراكمي، وعدد/حالة المواد.
- يحتوي `academic_level_before`, `academic_level_after`, `academic_level_delta`, و`academic_level_advanced`.
- `start_level_name_short` الرسمي طابق `start_level_ord` القديم في 100% من 130,277 سجلًا.
- مستوى البداية والنهاية بلا قيم مفقودة؛ توجد 34,155 حالة تقدم و65 حالة انخفاض رسمي محفوظة للمراجعة دون تعديل.
- لا توجد مفاتيح فصلية مكررة.

#### `student_semester_courses.parquet`

- الحبة: صف واحد لكل `student_course_id` في TRAIN.
- عدد الصفوف: 603,068.
- يحتوي المادة والعلامة والساعات والمحاولة الحالية.
- يحتوي ملخص المحاولات السابقة لنفس المادة قبل الصف الحالي.
- صفوف الإعادة: 86,500.
- صفوف لديها رسوب سابق في المادة: 73,110.
- لا يوجد `student_course_id` مكرر.

القيم الفارغة في `course_last_attempt_part` وحقول آخر محاولة صحيحة في أول محاولة للمادة؛ لا يوجد فصل محاولة سابق في هذه الحالة.

### 2. KNN V2 GPA + academic-level nearest

الـartifact الحالي:

```text
data/artifacts/knn/2026-08-23_history_v2_level/knn_v2_gpa_level_nearest.pkl
```

محتواه:

```text
Returning historical cases: 109,715
Cold-start historical cases: 13,773
Degrees covered: 45
Returning degree-level groups: 166
Cold-start degree-level groups: 100
Semester rows: 130,277
Course rows: 603,068
```

طريقة البحث الحالية:

```text
Returning:
    نفس degree_id
    نفس academic_level
    ثم أقرب cumulative_gpa_before

Cold-start:
    نفس degree_id
    نفس start academic_level
    ثم أقرب diploma_gpa
```

يمكن تمرير `exclude_student_id` لمنع الطالب الحالي من الظهور كجار لنفسه، وبعد ترتيب GPA يحتفظ KNN بأقرب snapshot واحدة لكل `student_id`. لا يوجد fallback إلى مستوى آخر في هذه النسخة؛ ضعف الدعم يعيد عددًا أقل ولا يوسع المقارنة بصمت.

نسخة V1 وartifact الخاص بها ما زالا محفوظين ولم تتم إعادة الكتابة فوقهما.

## الملفات البرمجية

```text
src/knn_history_helpers.py
scripts/build_knn_history_tables.py
scripts/build_knn_history_tables_v2.py
tests/test_knn_history_helpers.py

src/knn_advisor_v2.py
scripts/build_knn_v2.py
scripts/build_knn_v2_level.py
tests/test_knn_advisor_v2.py
tests/test_knn_advisor_v2_level.py
```

ملفَا السكربت يحتويان تدفق القراءة والبناء والحفظ فقط. التجميع، التحقق، وترتيب التاريخ موجود في ملفات `src/`.

## إعادة البناء

بناء جدولي History في نسخة جديدة:

```powershell
& ".venv/Scripts/python.exe" scripts/build_knn_history_tables_v2.py `
  --status-path "data/raw/v_add_student_degree_status_v2.parquet" `
  --output-dir "data/artifacts/knn/<new_history_version>"
```

بناء KNN V2 من نسخة History:

```powershell
& ".venv/Scripts/python.exe" scripts/build_knn_v2_level.py `
  --history-dir "data/artifacts/knn/<history_version>" `
  --output "data/artifacts/knn/<history_version>/knn_v2_gpa_level_nearest.pkl"
```

لا تعيد السكربتات الكتابة فوق artifact موجود؛ يجب اختيار اسم نسخة جديد.

## الاستعمال الحالي

### Returning

```python
from src.knn_advisor_v2 import KNNAdvisorV2Level

advisor = KNNAdvisorV2Level.load(
    "data/artifacts/knn/2026-08-23_history_v2_level/knn_v2_gpa_level_nearest.pkl"
)

neighbours = advisor.find_nearest_gpa(
    degree_id="13.111",
    academic_level=2,
    gpa=2.27,
    cold_start=False,
    k=20,
    exclude_student_id="current_student_id",
)

evidence = advisor.summarize(neighbours)
neighbour_courses = advisor.courses_for_neighbours(neighbours)
```

### Cold-start

```python
neighbours = advisor.find_nearest_gpa(
    degree_id="13.111",
    academic_level=1,
    gpa=81.5,
    cold_start=True,
    k=20,
)
```

في مسار cold-start تمثل `gpa` معدل الدبلوم، وليس GPA جامعيًا.

## ماذا يعيد KNN؟

الناتج الحالي هو أقرب **حالات طالب/فصل** تاريخية، وليس توصية مواد جاهزة. يحتوي كل جار على:

```text
student_id
degree_id
part_id
matched_gpa
gpa_distance
matched_academic_level
level_fallback_used
term_gpa
term_gpa_delta
cumulative_gpa_delta
term_gpa_improved
cumulative_gpa_improved
any_course_failed
all_courses_passed
```

يعيد `summarize()` متوسط نتائج الجيران ونسب التحسن والرسوب، ويعيد `courses_for_neighbours()` المواد التي أخذها هؤلاء في الفصول المسترجعة.

## ما لم يكتمل بعد

### 1. تقييم VALID

يبقى TRAIN قاعدة الجيران، ويستخدم VALID كطلبات مخفية النتائج. يجب قياس:

```text
MAE لتوقع term_gpa
MAE لتوقع cumulative_gpa_delta
جودة توقع تحسن GPA
جودة توقع الرسوب
coverage والدعم
median GPA distance
```

ويجب تجربة `K = 5, 10, 20, 30, 50` واختيار القيمة باستخدام VALID فقط.

### 2. تشابه خطة المواد

بعد استرجاع الجيران، نقارن خطة الطالب بمواد فصولهم باستخدام Jaccard، ثم نحسب evidence من الحالات ذات التشابه والدعم الكافيين. في الاختبار المحلي تستخدم مواد الطالب المسجلة فعليًا في VALID كخطة كانت معروفة عند التسجيل، مع إخفاء نتائجها.

### 3. الدمج مع M1/M2

لا يدخل KNN كميزات جديدة إلى النموذجين الحاليين. الدمج يكون بعد التنبؤ وعلى مستوى الخطة:

```text
M1/M2 predictions
        +
GPA result processing
        +
KNN plan evidence
        ↓
Plan Ranker
```

نبدأ بإرجاع نتائج النموذج وKNN جنبًا إلى جنب. لا نعتمد وزن KNN في الترتيب إلا إذا أظهر backtest على VALID تحسنًا مقارنة بـ`models only`.

## بوابة إغلاق مرحلة KNN

لا تعتبر المرحلة مكتملة إلا عند تحقق جميع البنود التالية:

- [x] بناء جدولي History من TRAIN.
- [x] فصل Returning عن Cold-start.
- [x] فلترة الجيران داخل نفس `degree_id`.
- [x] دعم استبعاد الطالب الحالي.
- [x] استرجاع نتائج الفصول ومواد الجيران.
- [x] فلترة `academic_level` بالرقم الرسمي ودون fallback صامت.
- [x] أقرب snapshot واحد لكل طالب تاريخي.
- [ ] تقييم قيم K على VALID.
- [ ] إضافة Plan Similarity وsupport/confidence.
- [ ] مقارنة `models only` مع `models + KNN`.
- [ ] ربط KNN بالـAPI وPlan Ranker.

## التحقق الحالي

```text
KNN/History focused tests: 9 PASS
Full project suite: 269 tests — OK (3 environment-gated tests skipped)
Development freeze verification: PASS
```

لم تتغير بيانات TRAIN/VALID المجمدة أو نماذج M1/M2 أثناء هذه المرحلة.
