---
aliases:
  - KNN History Helpers
  - دوال تاريخ KNN
tags:
  - academic-advisor
  - knn
  - knn/data
  - knn/leakage
  - knn/pipeline/02-helpers
---

# `knn_history_helpers.py`: تجهيز التاريخ ومنع التسريب

> [!abstract] التنقل
> السابق: [[knn_readme/01_build_knn_history_tables_v2|بناء جداول التاريخ]] ·
> الأعلى: [[knn_readme/README|دليل KNN]] · التالي:
> [[knn_readme/03_build_knn_v2_level|عمل fit لموديلات sklearn]]

## الفكرة

هذا الملف يحتوي المنطق الفعلي الذي يستعمله سكربت بناء التاريخ. فصل الدوال عن سكربت التشغيل يجعلها قابلة للاختبار ويمنع تحوّل سكربت CLI إلى ملف طويل يصعب تدقيقه.

ينقسم الملف منطقياً إلى أربع وحدات:

| الوحدة          | الدوال                                                     | الهدف                                               |
| --------------- | ---------------------------------------------------------- | --------------------------------------------------- |
| التحقق والتطبيع | `require_columns`, `normalize_identifier`, `assert_unique` | رفض البيانات غير المطابقة للعقد                     |
| الدمج الرسمي    | `attach_official_references`                               | إضافة AGPA والمستوى وحالة الدرجة من المصادر الرسمية |
| تاريخ المقرر    | `build_student_semester_courses`                           | حساب المحاولات السابقة من دون رؤية النتيجة الحالية  |
| نتيجة الفصل     | `build_student_semester_outcomes`                          | إنتاج صف واحد لكل فصل مع المدخلات والأهداف          |

## مفاتيح البيانات

```python
SEMESTER_KEY = ["university_id", "student_id", "degree_id", "part_id"]
COURSE_TIMELINE_KEY = [
    "university_id", "student_id", "degree_id", "course_id"
]
```

| المفتاح | معناه |
|---|---|
| `SEMESTER_KEY` | يحدد فصلاً واحداً لطالب داخل تخصص وجامعة |
| `COURSE_TIMELINE_KEY` | يحدد تسلسل محاولات مقرر واحد لطالب داخل التخصص |

وجود `degree_id` في المفتاح يمنع خلط سجل الطالب إذا غيّر تخصصه، ووجود `university_id` يمنع اصطدام المعرفات بين الجامعات.

## التحقق من الأعمدة

```python
def require_columns(frame, columns, *, name):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")
```

الفكرة ليست تحسين شكل رسالة الخطأ فقط. الموديل يجب ألا يكمل إذا اختفى عمود مثل `end_agpa_points` أو `finish_status`، لأن التعويض الصامت سيغير معنى الهدف.

		## تطبيع المعرفات

```python
def normalize_identifier(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
    )
```

قد يصل المعرّف مرة كرقم `111.0` ومرة كنص `"111"`. التطبيع يمنع فشل الدمج بسبب اختلاف شكلي لا يحمل معنى حقيقياً، مع إبقاء المعرف نصاً حتى لا تضيع suffixes أو الأصفار ذات الدلالة.

## إضافة المراجع الرسمية

```python
result = result.merge(
    status_ref,
    on="student_status_id",
    how="left",
    validate="many_to_one",
)
result = result.merge(
    grade_ref,
    on="grade_id",
    how="left",
    validate="many_to_one",
)
```

`validate="many_to_one"` يعني أن عدة مقررات يمكن أن تشير إلى حالة طالب أو درجة واحدة، لكن المرجع نفسه يجب أن يحتوي صفاً واحداً فقط لكل ID.

بعد الدمج يتم اشتقاق الهدفين:

```python
result["is_passed"] = result["finish_status"].eq("P").astype("int8")
result["is_failed"] = (
    result["finish_status"].isin(["F", "FE"])
    | result["grade_show"].eq("F")
).astype("int8")
```

| العمود      | القيمة 1 تعني                        |
| ----------- | ------------------------------------ |
| `is_passed` | المرجع الرسمي أعطى `finish_status=P` |
| `is_failed` | الحالة `F/FE` أو عرض الدرجة `F`      |
|             |                                      |

هذا أفضل من قاعدة عشوائية مثل `final_mark >= 50` لأن قاعدة النجاح تأتي من مرجع الدرجات نفسه.

## التحقق من المستوى الرسمي

```python
derived_start_level = pd.to_numeric(
    result["start_level_ord"], errors="coerce"
).astype("Int64")
mismatch = result["start_level_name_short"].ne(derived_start_level)
if mismatch.any():
    raise ValueError(...)
```

هنا تتم مقارنة المستوى الموجود في TRAIN بالمستوى الرسمي القادم من حالة الطالب. إذا اختلفا لا يتم اختيار أحدهما بصمت؛ يتوقف البناء حتى يتم التحقيق في مصدر الاختلاف.

## منع تسريب محاولة المقرر الحالية

أهم فكرة في `build_student_semester_courses` هي استعمال `shift(1)` أو طرح القيمة الحالية من التجميع التراكمي.

```python
result["course_last_mark_prior"] = grouped["final_mark"].shift(1)

result["__best_mark_including"] = grouped["final_mark"].cummax()
result["course_best_mark_prior"] = result.groupby(
    COURSE_TIMELINE_KEY, dropna=False, sort=False
)["__best_mark_including"].shift(1)
```

مثال:

| المحاولة | العلامة الحالية | `course_last_mark_prior` | `course_best_mark_prior` |
|---:|---:|---:|---:|
| 1 | 42 | NaN | NaN |
| 2 | 57 | 42 | 42 |
| 3 | 65 | 57 | 57 |

المحاولة الثانية لا ترى علامتها `57` في تاريخها السابق، والمحاولة الثالثة ترى المحاولتين السابقتين فقط.

## حساب متوسط العلامات السابقة

```python
prior_mark_sum = grouped["__mark_value"].cumsum() - result["__mark_value"]
prior_mark_count = grouped["__mark_seen"].cumsum() - result["__mark_seen"]
result["course_mean_mark_prior"] = prior_mark_sum.div(
    prior_mark_count.where(prior_mark_count.gt(0))
)
```

طرح الصف الحالي من `cumsum` يحقق النتيجة نفسها التي يحققها `shift`، لكنه يسمح بحساب sum وcount بكفاءة على البيانات الكبيرة.

## تجميع نتيجة الفصل

```python
grouped = enriched.groupby(SEMESTER_KEY, dropna=False, sort=False)
snapshots = grouped[snapshot_columns].first().reset_index()
observed = grouped.agg(
    observed_course_count=("student_course_id", "count"),
    semester_average_mark=("final_mark", "mean"),
    passed_course_count=("is_passed", "sum"),
    failed_course_count=("is_failed", "sum"),
).reset_index()
```

يتم فصل نوعين من الأعمدة:

| النوع | طريقة التجميع | مثال |
|---|---|---|
| حالة ثابتة داخل الفصل | `first()` | GPA قبل الفصل، المستوى، عدد الساعات المسجلة |
| نتيجة عبر المقررات | `count/sum/mean` | عدد المقررات الراسبة ومتوسط العلامة |

ثم يتم تكوين هدف التصنيف:

```python
result["any_course_failed"] = (
    result["failed_course_count"].gt(0).astype("int8")
)
```

أي أن المصنف لا يتنبأ بنجاح مقرر واحد؛ بل يتنبأ إن كان الفصل سيحتوي **أي مقرر راسب**.

## أهداف الانحدار

```python
"gpa_points": "term_gpa",
...
result["term_gpa_delta"] = (
    result["term_gpa"] - result["previous_valid_term_gpa"]
)
```

| الهدف | الوحدة | استخدامه الحالي |
|---|---|---|
| `term_gpa` | نقاط GPA | هدف `KNeighborsRegressor` الأول |
| `semester_average_mark` | علامة من نطاق الدرجات | هدف `KNeighborsRegressor` الثاني |
| `term_gpa_delta` | فرق نقاط | evidence وصفي، وليس خرج regressor الحالي |
| `cumulative_gpa_delta` | فرق نقاط cumulative | evidence وصفي |

## ضمان grain الجدول

```python
assert_unique(output, SEMESTER_KEY, name="student_semester_outcomes")
```

إذا ظهر أكثر من صف لنفس الطالب/التخصص/الفصل يتوقف البناء. هذا الضمان مهم لأن وجود التكرار سيعطي الفصل وزناً مضاعفاً أثناء تدريب KNN.

## اختبارات مرتبطة

الاختبارات في `tests/test_knn_history_helpers.py` تتحقق من:

- أن المحاولة الأولى ليس لها تاريخ سابق مصطنع.
- أن تاريخ المقرر لا يرى النتيجة الحالية.
- أن نتيجة الفصل تستعمل AGPA الرسمي.

تشغيلها:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_knn_history_helpers -v
```

## روابط مرتبطة

- تُستعمل النتائج كمدخلات في [[knn_readme/03_build_knn_v2_level|سكربت التدريب]].
- الدوال نفسها تعيد بناء أهداف VALID في [[knn_readme/05_evaluate_knn_v2_level|سكربت التقييم]].
- تعريف الأهداف التي يتعلمها [[knn_readme/04_knn_advisor_v2_level|المصنف والمنحدر]].
- العودة إلى [[knn_readme/README|خريطة KNN الرئيسية]].
