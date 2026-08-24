---
aliases:
  - KNN Recommendation Integration
  - دمج KNN مع التوصيات
tags:
  - academic-advisor
  - knn
  - knn/recommendation
  - knn/pipeline/06-serving
---

# ربط KNN مع `recommendation.py`

> [!abstract] التنقل
> السابق: [[knn_readme/05_evaluate_knn_v2_level|تقييم KNN]] · الأعلى:
> [[knn_readme/README|دليل KNN]] · مرجع النسخ:
> [[knn_readme/07_legacy_and_active_files|الملفات القديمة والفعالة]]

## الفكرة

M1 وM2 يتنبآن بأداء كل مقرر في الخطة، بينما KNN يضيف سياقاً على مستوى الطالب والفصل: ماذا حدث لطلاب من نفس التخصص والمستوى ولديهم GPA قريبة؟

يستعمل recommendation واجهتين مختلفتين من KNN:

| الواجهة | الغرض |
|---|---|
| `predict` | الحصول على احتمال الرسوب ومعدل وعلامة الفصل المتوقعين |
| `find_nearest_gpa` | إرجاع الجيران ومقرراتهم حتى يكون الـ evidence قابلاً للشرح |

## تحميل الـ artifact

```python
knn = KNNAdvisorV2Level.load(knn_index_path)
```

المسار الحالي في مثال التحميل هو:

```text
data/artifacts/knn/2026-08-24_sklearn_v3/
knn_v2_level_sklearn_k20.pkl
```

بعد التحميل يحتوي `Recommender` ثلاثة أجزاء أساسية:

| الجزء | المسؤولية |
|---|---|
| `StudentScorer` | توقع علامة واحتمال نجاح كل مقرر بواسطة M1/M2 |
| `KNNAdvisorV2Level` | توقع نتيجة الفصل وإرجاع طلاب مشابهين |
| `GradeScale 3.111` | تحويل العلامة المتوقعة إلى نقاط GPA الرسمية |

## تحديد state الطالب

```python
resolved_gpa, resolved_level, exclude_student_id = _resolve_knn_query(
    df_history=df_history,
    snapshot=snapshot,
    cold_start=cold_start,
    knn_gpa=knn_gpa,
    academic_level=academic_level,
)
```

| الحالة | GPA | المستوى |
|---|---|---|
| Returning | آخر AGPA رسمي بعد أحدث فصل | المستوى الرسمي بعد أحدث فصل |
| Cold-start | `diploma_gpa` | المستوى الأول أو المستوى الممرر صراحة |

يمكن للـ caller تمرير `knn_gpa` و`academic_level` صراحة، لكن القيم الافتراضية تأتي من snapshot الرسمي، لا من رقم عشوائي.

## استرجاع الجيران

```python
neighbours = self.knn.find_nearest_gpa(
    degree_id=degree_id,
    academic_level=resolved_level,
    gpa=resolved_gpa,
    cold_start=cold_start,
    k=knn_k,
    exclude_student_id=exclude_student_id,
)
```

هذا الاستدعاء يستعمل `kneighbors` من المصنف المـfit، ثم يرجع DataFrame يحتوي النتائج التاريخية. يتم استبعاد `student_id` الحالي من قائمة الشرح.

## التنبؤ المباشر

```python
knn_prediction = self.knn.predict(
    degree_id=degree_id,
    academic_level=resolved_level,
    gpa=resolved_gpa,
    cold_start=cold_start,
    k=knn_k,
)
```

الناتج يدخل إلى recommendation بهذه المعاني:

| خرج KNN | استعماله |
|---|---|
| `failure_probability` | evidence رئيسي لخطر الفصل |
| `predicted_any_course_failed` | قرار المصنف 0/1 للعرض أو التدقيق |
| `predicted_term_gpa` | تقدير نتيجة الفصل من الطلاب المشابهين |
| `predicted_semester_average_mark` | تقدير متوسط علامة الفصل |
| `support` | عدد الجيران الذي استعمله estimator |

## تحويل التوقع إلى evidence

```python
evidence = {
    "knn_pct_any_failed": predicted_failure_probability,
    "knn_predicted_any_course_failed": prediction.get(
        "predicted_any_course_failed"
    ),
    "knn_failure_probability": predicted_failure_probability,
    "knn_predicted_term_gpa": predicted_term_gpa,
    "knn_predicted_semester_average_mark": predicted_average_mark,
}
```

تبقى معلومات الجيران الوصفية بجانب prediction:

- `knn_support`
- `knn_route`
- `knn_mean_matched_gpa`
- `knn_median_gpa_distance`
- `knn_avg_pass_rate`
- تغير GPA لدى الجيران

## تداخل مقررات الخطة

```python
overlap_ratio = len(plan_set & historical_set) / len(plan_set)
if overlap_ratio >= 0.5:
    similar_plan_marks.append(float(marks.mean()))
```

إذا كان الجار أخذ نصف مقررات الخطة المقترحة على الأقل، تدخل علامته المتوسطة في:

```text
knn_similar_plan_avg_mark
```

هذا الحقل تفسيري حالياً؛ يتم إرجاعه للمستخدم، لكنه لا يدخل بعد بصورة مباشرة في معادلة ترتيب الخطط.

## دخول احتمال KNN إلى composite

```python
failure_probability = knn_evidence.get("knn_failure_probability", np.nan)
if not pd.isna(failure_probability):
    knn_bonus = ((1.0 - float(failure_probability)) - 0.8) * 0.5
```

أمثلة:

| احتمال الرسوب من KNN | احتمال الأمان `1-p` | `knn_bonus` |
|---:|---:|---:|
| 0.10 | 0.90 | +0.05 |
| 0.20 | 0.80 | 0.00 |
| 0.40 | 0.60 | -0.10 |
| 0.70 | 0.30 | -0.25 |

ثم تدخل القيمة في المعادلة:

```python
composite = (
    0.40 * (expected_agpa / 4.0)
    - 0.30 * risk
    + 0.20 * (1.0 - workload_ratio * 0.5)
    + 0.10 * grad_progress
    + knn_bonus
)
```

| العنصر | مصدره |
|---|---|
| `expected_agpa` | علامات M2 محولة بمقياس ACS_GRADE 3.111 |
| `risk` | احتمالات نجاح M1 لكل مقرر |
| `workload_ratio` | ساعات الخطة مقارنة بالحد |
| `grad_progress` | تقدم الخطة نحو ساعات التخرج المتبقية |
| `knn_bonus` | احتمال الرسوب من KNN |

## ملاحظة مهمة في ترتيب الخطط

توقع KNN الحالي يعتمد على حالة الطالب فقط، وليس على مقررات كل خطة. لذلك `knn_bonus` نفسها تضاف إلى كل الخطط الخاصة بالطالب في الاستدعاء الواحد. النتيجة:

- تؤثر KNN في الدرجة المطلقة التي نعرضها للخطة.
- لكنها لا تغيّر ترتيب خطتين للطالب نفسه ما دام الجزء الخاص بـ KNN متطابقاً.

لجعل KNN يغيّر ترتيب الخطط، يجب إدخال evidence خاص بالخطة، مثل تداخل المقررات أو حمل الخطة، في prediction أو في bonus بعد التحقق منه على VALID.

## حالة عدم وجود تغطية

إذا لم توجد مجموعة تخصص ومستوى مطابقة:

```python
{
    "covered": False,
    "support": 0,
    "failure_probability": None,
    ...
}
```

لا تتوقف recommendation. يكون KNN bonus صفراً أو يستعمل evidence الوصفي إن توفر، وتستمر الخطة بالاعتماد على M1/M2 وبقية المحاور.

## اختبار التكامل

`tests/test_recommendation_knn_v2.py` يتحقق من:

- تمرير التخصص والمستوى وGPA الصحيحة.
- استعمال diploma GPA في cold-start.
- استدعاء `predict` وليس `find_nearest_gpa` فقط.
- ظهور احتمال الرسوب ومعدل KNN في الخطة الناتجة.
- استمرار recommendation عند عدم وجود مجموعة KNN.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_recommendation_knn_v2 -v
```

## روابط مرتبطة

- التوقع صادر من [[knn_readme/04_knn_advisor_v2_level|KNNAdvisorV2Level]].
- جودة التوقع موثقة في [[knn_readme/05_evaluate_knn_v2_level|تقييم VALID]].
- الـ artifact بُنيت في [[knn_readme/03_build_knn_v2_level|مرحلة fit]].
- العودة إلى [[knn_readme/README|خريطة KNN الرئيسية]].
