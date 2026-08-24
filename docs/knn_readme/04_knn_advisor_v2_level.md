---
aliases:
  - KNNAdvisorV2Level
  - قلب موديل KNN
tags:
  - academic-advisor
  - knn
  - knn/model
  - knn/sklearn
  - knn/pipeline/04-model
---

# `knn_advisor_v2.py`: قلب موديل KNN الحالي

> [!abstract] التنقل
> السابق: [[knn_readme/03_build_knn_v2_level|بناء وتدريب الموديل]] · الأعلى:
> [[knn_readme/README|دليل KNN]] · التالي:
> [[knn_readme/05_evaluate_knn_v2_level|تقييم الموديل]]

## الفكرة العامة

الـ class الفعال هو `KNNAdvisorV2Level`. يجمع ثلاث وظائف في واجهة واحدة:

| الوظيفة | واجهة sklearn المستعملة | الناتج |
|---|---|---|
| تصنيف خطر الرسوب | `KNeighborsClassifier.predict/predict_proba` | label واحتمال رسوب |
| توقع معدل وعلامة الفصل | `KNeighborsRegressor.predict` | `term_gpa` ومتوسط العلامة |
| تفسير النتيجة | `kneighbors` | صفوف الطلاب التاريخيين الأقرب |

أما `KNNAdvisorV2` الموجود في الملف نفسه فهو نسخة توافق قديمة لا تنفذ sklearn fit، ولا تستخدمها recommendation الحالية.

## استيراد موديلات sklearn

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
```

المصنف مناسب للهدف الثنائي `any_course_failed`، والمنحدر يدعم multi-output ولذلك يتنبأ بهدفين مستمرين في استدعاء واحد.

## حزمة موديلات المجموعة

```python
@dataclass
class _SklearnKNNGroup:
    positions: np.ndarray
    classifier: KNeighborsClassifier
    regressor: KNeighborsRegressor
```

كل مفتاح مجموعة يشير إلى:

- مواضع صفوف الطلاب في جدول النتائج.
- مصنف fitted.
- منحدر fitted.

مفتاح المجموعة نفسه هو:

```python
(degree_id, academic_level)
```

ويتم وضعه داخل route منفصل للطالب المستمر أو cold-start.

## تكوين بيانات التدريب

```python
x_train = features.to_numpy(dtype=float).reshape(-1, 1)
classifier_target = targets["any_course_failed"].astype(int).to_numpy()
regression_targets = targets[
    ["term_gpa", "semester_average_mark"]
].to_numpy(dtype=float)
```

الأبعاد هي:

| المصفوفة | الشكل | المعنى |
|---|---|---|
| `x_train` | `(n_students, 1)` | GPA قبل الفصل أو diploma GPA |
| `classifier_target` | `(n_students,)` | 0 أو 1 لوجود رسوب |
| `regression_targets` | `(n_students, 2)` | معدل الفصل ومتوسط العلامة |

## عمل fit

```python
classifier = KNeighborsClassifier(
    n_neighbors=fitted_k,
    weights=weights,
    metric=metric,
    algorithm="auto",
)
regressor = KNeighborsRegressor(
    n_neighbors=fitted_k,
    weights=weights,
    metric=metric,
    algorithm="auto",
)
classifier.fit(x_train, classifier_target)
regressor.fit(x_train, regression_targets)
```

`fitted_k` يساوي الأقل بين K المطلوبة وعدد طلاب المجموعة:

```python
fitted_k = min(int(n_neighbors), len(frame))
```

هذا يمنع خطأ sklearn عندما تحتوي مجموعة نادرة على أقل من 20 طالباً.

## لماذا آخر snapshot لكل طالب؟

```python
unique_students = (
    outcomes.iloc[group["__position"].to_numpy(dtype=np.int64)]
    .drop_duplicates("student_id", keep="last")
)
```

لو تركنا عشرة فصول للطالب نفسه وصفاً واحداً لطالب آخر، سيحصل الأول على عشرة أصوات. السياسة الحالية تحتفظ بآخر snapshot موثوق لكل طالب داخل التخصص والمستوى والـ route، فيصبح وزن كل طالب صفاً واحداً.

هذا قرار تصميم يجب تثبيته في report كل artifact، لأنه يغيّر population التدريب مقارنة بالنسخة اليدوية القديمة.

## التنبؤ لطالب واحد

```python
predicted_class = int(group.classifier.predict(x_query)[0])
failure_probability = float(
    self._failure_probability(group.classifier, x_query)[0]
)
predicted_regression = group.regressor.predict(x_query)[0]
```

الناتج:

```python
{
    "covered": True,
    "support": 20,
    "knn_route": "returning_degree_level_cumulative_gpa",
    "predicted_any_course_failed": 0,
    "failure_probability": 0.25,
    "predicted_term_gpa": 2.73,
    "predicted_semester_average_mark": 74.8,
}
```

| الحقل | تفسيره |
|---|---|
| `predicted_any_course_failed` | قرار المصنف النهائي 0/1 |
| `failure_probability` | نسبة الجيران المصنفين كفصول فيها رسوب عند uniform weights |
| `predicted_term_gpa` | متوسط GPA الهدف لدى الجيران |
| `predicted_semester_average_mark` | متوسط هدف العلامة لدى الجيران |
| `support` | K الفعلية للمجموعة، وقد تقل عن 20 |

## معالجة مجموعة ذات class واحد

بعض المجموعات الصغيرة قد تحتوي فقط على طلاب لم يرسبوا. عندها `predict_proba` لا يحتوي عمود class `1`. الدالة التالية تعيد احتمال رسوب صفراً بدلاً من اختيار عمود خاطئ:

```python
probabilities = classifier.predict_proba(x)
failure_columns = np.flatnonzero(classifier.classes_ == 1)
if len(failure_columns) == 0:
    return np.zeros(len(x), dtype=float)
```

## التنبؤ الدفعي

```python
def predict_frame(self, queries: pd.DataFrame) -> pd.DataFrame:
    ...
```

هذه الواجهة تجمع الاستعلامات حسب route والتخصص والمستوى، ثم ترسل عدة GPA دفعة واحدة إلى estimator نفسها:

```python
regression = group.regressor.predict(x_query)
classification = group.classifier.predict(x_query)
probability = self._failure_probability(group.classifier, x_query)
```

تستخدمها مرحلة التقييم لأنها أسرع بكثير من استدعاء `predict` منفرداً لآلاف الطلاب، مع بقاء الحساب نفسه.

## إرجاع الجيران للشرح

```python
distances, local_indices = group.classifier.kneighbors(
    np.asarray([[query_gpa]], dtype=float),
    n_neighbors=request_count,
    return_distance=True,
)
```

هذه العملية لا تنتج classification label؛ بل تعيد index ومسافة كل جار. بعدها تُحوّل المواضع إلى صفوف أصلية:

```python
selected_positions = group.positions[local_indices[0]]
candidates = self._outcomes.iloc[selected_positions].copy()
```

الـ recommendation يحتاج هذه الصفوف لمعرفة:

- الفصول الأقرب.
- المسافة في GPA.
- المقررات التي أخذها الجار.
- العلامات التاريخية للمقررات المتداخلة مع الخطة.

## استبعاد الطالب من قائمة الشرح

```python
if exclude_student_id is not None:
    candidates = candidates.loc[
        candidates["student_id"].ne(normalized_student)
    ].copy()
```

الاستبعاد مطبق على `find_nearest_gpa` حتى لا يظهر الطالب نفسه كـ «طالب مشابه» في الشرح.

لكن توجد ملاحظة مهمة: واجهة sklearn الأصلية `predict` تستقبل GPA فقط، ولذلك لا تستبعد ID الطالب من عملية التصويت إذا كان لديه snapshot أقدم في TRAIN. تقييم VALID الحالي يصرح بهذه السياسة في `metrics.json`. إذا أردنا تقييماً student-disjoint بالكامل، يجب إنشاء split حسب الطالب أو تصميم estimator بمدخلات استبعاد مختلفة.

## الحفظ والتحميل

```python
pickle.dump(
    {
        "version": self.VERSION,
        "outcomes": self._outcomes,
        "courses": self._courses,
        "route_positions": self._route_positions,
        "group_models": self._group_models,
        "n_neighbors": self.n_neighbors,
        "weights": self.weights,
        "metric": self.metric,
    },
    handle,
)
```

يجب تحميل Pickle من مصدر موثوق فقط؛ Pickle ليس format آمناً لملف مجهول أو منزل من الإنترنت.

## التحقق من K عند predict

```python
if k is not None and int(k) != self.n_neighbors:
    raise ValueError(...)
```

هذا يمنع حالة يكون فيها `kneighbors` على K=10 بينما `predict` ما زال يستعمل estimator مدربة بإعداد K=20.

## الاختبارات

الملف `tests/test_knn_advisor_v2_level.py` يتحقق من:

- أن الأنواع المحفوظة هي `KNeighborsClassifier` و`KNeighborsRegressor` فعلاً.
- أن `predict` يعمل للتصنيف والانحدار.
- أن `predict_frame` يغطي الحالات ويعيد missing للمجموعة غير الموجودة.
- أن المستوى لا يحصل على fallback صامت.
- أن artifact تعمل بعد save/load.
- أن K مختلفة عن K التدريب تُرفض.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_knn_advisor_v2_level -v
```

## روابط مرتبطة

- يتم عمل fit من [[knn_readme/03_build_knn_v2_level|سكربت البناء]].
- يتم قياس `predict` في [[knn_readme/05_evaluate_knn_v2_level|تقرير VALID]].
- تستعمله [[knn_readme/06_recommendation_integration|Recommendation]] للتوقع والشرح.
- مقارنة `KNeighborsClassifier` بالنسخ السابقة في [[knn_readme/07_legacy_and_active_files|Legacy vs Active]].
- العودة إلى [[knn_readme/README|خريطة KNN الرئيسية]].
