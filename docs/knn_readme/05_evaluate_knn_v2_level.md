---
aliases:
  - KNN VALID Evaluation
  - تقييم KNN
tags:
  - academic-advisor
  - knn
  - knn/evaluation
  - knn/metrics
  - knn/pipeline/05-evaluation
---

# `evaluate_knn_v2_level.py`: تقييم التصنيف والانحدار

> [!abstract] التنقل
> السابق: [[knn_readme/04_knn_advisor_v2_level|قلب الموديل]] · الأعلى:
> [[knn_readme/README|دليل KNN]] · التالي:
> [[knn_readme/06_recommendation_integration|الربط مع Recommendation]]

## الفكرة

هذا السكربت لا يدرب الموديل. هو يحمل artifact مـfit من TRAIN، يعيد بناء النتائج الرسمية لمجموعة `VALID`، ثم يستدعي واجهات sklearn الأصلية عبر `predict_frame` ويقارن التوقع بالحقيقة.

`TEST` لا يُقرأ. الهدف هو اختيار التصميم وفهم نقاط الضعف باستعمال VALID، ثم إبقاء TEST للتقييم النهائي بعد تثبيت كل القرارات.

## ما الذي يتم تقييمه؟

| القسم | Actual | Prediction | نوع المسألة |
|---|---|---|---|
| Failure | `any_course_failed` | `predict` و`predict_proba` | تصنيف ثنائي |
| GPA | `term_gpa` | خرج regressor الأول | انحدار |
| Mark | `semester_average_mark` | خرج regressor الثاني | انحدار |

ولهذا فإن نتيجة التقرير السابقة لم تكن للمعدل فقط ولا للتصنيف فقط؛ التقرير يحتوي الأقسام الثلاثة.

## بناء حالات VALID

```python
valid = pd.read_parquet(valid_path, columns=TRAIN_COLUMNS)
status = pd.read_parquet(
    status_path, columns=STATUS_COLUMNS + LEVEL_STATUS_COLUMNS
)
grades = pd.read_parquet(grades_path, columns=GRADE_COLUMNS)
enriched = attach_official_references(valid, status, grades)
outcomes = build_student_semester_outcomes(enriched)
```

يتم استعمال الدوال نفسها التي استُعملت لبناء TRAIN. هذا يمنع اختلاف تعريف الهدف بين التدريب والتقييم، مثل اعتبار حالة درجة ناجحة في TRAIN وفاشلة في VALID بسبب منطق مختلف.

## تحديد route الاستعلام

```python
returning = outcomes["is_first_active_semester"].eq(0) & cumulative_gpa.gt(0)
cold_start = outcomes["is_first_active_semester"].eq(1) & diploma_gpa.notna()
```

ثم يتم اختيار GPA المناسبة:

```python
eligible["query_gpa"] = np.where(
    eligible["evaluation_route"].eq("cold_start"),
    eligible["diploma_gpa"],
    eligible["cumulative_gpa_before"],
)
```

| route | GPA الاستعلام |
|---|---|
| Returning | GPA التراكمي قبل الفصل المراد توقعه |
| Cold-start | GPA الشهادة قبل أول فصل جامعي |

## استدعاء sklearn predict

```python
model_predictions = advisor.predict_frame(model_queries)
```

داخل `predict_frame` تُنفذ فعلياً:

```python
group.classifier.predict(x_query)
group.classifier.predict_proba(x_query)
group.regressor.predict(x_query)
```

هذا مهم لأنه يعني أن التقرير الحالي لا يحسب المتوسط يدوياً بدلاً من الموديل؛ هو يقيس الـ estimators المحفوظة نفسها.

## مقاييس التصنيف

```python
return {
    "roc_auc": roc_auc_score(observed, probability),
    "average_precision": average_precision_score(observed, probability),
    "brier": brier_score_loss(observed, probability),
    "accuracy": accuracy_score(observed, predicted),
    "balanced_accuracy": balanced_accuracy_score(observed, predicted),
    "precision": precision_score(observed, predicted),
    "recall": recall_score(observed, predicted),
    "f1": f1_score(observed, predicted),
}
```

| المقياس | ماذا يجيب؟ | ملاحظة في بياناتنا |
|---|---|---|
| Accuracy | كم قراراً صحيحاً إجمالاً؟ | قد تكون مضللة بسبب كثرة «لا رسوب» |
| ROC-AUC | هل الحالات الأخطر تأخذ احتمالاً أعلى؟ | أنسب لفهم جودة الترتيب |
| Average Precision | جودة اكتشاف class الرسوب | يقارن عملياً مع prevalence الرسوب |
| Precision | من الحالات التي حذرنا منها، كم واحدة رسبت؟ | الحالي 53.57% |
| Recall | من كل حالات الرسوب، كم واحدة اكتشفنا؟ | الحالي منخفض: 19.28% |
| F1 | توازن Precision وRecall | الحالي 28.36% |
| Brier | جودة الاحتمال نفسه | الأقل أفضل |

### لماذا Accuracy وحدها لا تكفي؟

نسبة الحالات التي لا تحتوي رسوباً تقارب 74%. مصنف بسيط يقول دائماً «لا رسوب» يحصل على Accuracy قريبة من موديل KNN لكنه لا يكتشف أي خطر.

```text
KNN Accuracy              = 74.92%
Always-no-failure Accuracy = 74.26%
KNN Recall for failure     = 19.28%
```

لذلك AUC وAverage Precision وRecall ضرورية بجانب Accuracy.

## مقاييس الانحدار

```python
return {
    "mae": mean_absolute_error(observed, predicted),
    "rmse": np.sqrt(mean_squared_error(observed, predicted)),
    "r2": r2_score(observed, predicted),
    "bias_pred_minus_observed": (predicted - observed).mean(),
}
```

| المقياس | التفسير |
|---|---|
| MAE | متوسط حجم الخطأ بوحدة الهدف الأصلية |
| RMSE | يعاقب الأخطاء الكبيرة أكثر من MAE |
| R² | مقدار التباين المفسر؛ الصفر يعادل تقريباً توقع المتوسط والسالب أسوأ منه |
| Bias | هل الموديل يرفع أو يخفض التوقعات بصورة منتظمة؟ |

أمثلة القراءة:

- `GPA MAE=0.506`: متوسط الفرق المطلق نحو نصف نقطة GPA.
- `Mark MAE=8.34`: متوسط الفرق المطلق نحو 8.34 درجات.
- لا يجوز مقارنة الرقمين مباشرة لأن وحدتي القياس مختلفتان.

## Coverage وSupport

| المقياس | المعنى |
|---|---|
| `query_rows` | الحالات التي لديها state صالح لاستعلام KNN |
| `covered_rows` | الحالات التي وجد لها موديل مطابق للتخصص والمستوى والـ route |
| `coverage` | `covered_rows / query_rows` |
| `full_k_coverage` | نسبة الحالات التي وجدت مجموعة فيها 20 طالباً على الأقل |
| `support` | K الفعلية للمجموعة |

النتيجة الحالية:

| العنصر | القيمة |
|---|---:|
| كل صفوف فصول VALID | 16,042 |
| الحالات القابلة لتكوين query | 14,673 |
| الحالات التي غطاها موديل | 14,368 |
| Coverage من الحالات القابلة للاستعلام | 97.92% |
| Full K=20 coverage | 88.30% |

## النتيجة الحالية بالتفصيل

### التصنيف

| الشريحة | AUC | AP | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| All | 73.61% | 44.94% | 74.92% | 53.57% | 19.28% | 28.36% |
| Returning | 75.42% | 48.24% | 75.78% | 60.81% | 16.85% | 26.39% |
| Cold-start | 63.44% | 38.24% | 66.23% | 36.52% | 44.34% | 40.06% |

### الانحدار

| الشريحة | GPA MAE | GPA RMSE | GPA R² | Mark MAE | Mark RMSE | Mark R² |
|---|---:|---:|---:|---:|---:|---:|
| All | 0.506 | 0.691 | 0.295 | 8.34 | 11.66 | 0.305 |
| Returning | 0.477 | 0.657 | 0.345 | 7.81 | 10.95 | 0.355 |
| Cold-start | 0.799 | 0.967 | -0.144 | 13.73 | 17.34 | -0.039 |

الخلاصة الإحصائية:

- Returning يحمل إشارة مفيدة، خصوصاً في AUC والانحدار.
- قرار المصنف الأصلي محافظ جداً؛ Recall الرسوب منخفض.
- Cold-start أضعف، وR² السالب يعني أن انحداره لا يصلح منفرداً حالياً.
- لا يتم تغيير threshold داخل هذا السكربت؛ القرار المعروض هو `KNeighborsClassifier.predict` الأصلي.

## سياسة التسريب الحالية

Artifact مبنية من TRAIN فقط، ونتائج VALID تستخدم targets للتقييم فقط. لكن temporal VALID قد تحتوي الطالب نفسه الذي لديه snapshot أقدم داخل TRAIN، وواجهة sklearn `predict` الحالية لا تستقبل `student_id` كي تستبعده من التصويت.

هذا ليس تسريباً لنتيجة فصل VALID نفسها، لكنه ليس تقييماً student-disjoint. إذا كان المطلوب قياس التعميم على طلاب لم يرهم الموديل مطلقاً، نحتاج split منفصلاً حسب `student_id`.

## التشغيل

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_knn_v2_level.py `
  --knn-path data\artifacts\knn\2026-08-24_sklearn_v3\knn_v2_level_sklearn_k20.pkl `
  --output-dir reports\knn\2026-08-24_knn_sklearn_v3_valid_k20
```

السكربت يرفض الكتابة إذا كان مجلد التقرير موجوداً. اختر version جديداً لكل تجربة.

## مخرجات التقرير

| الملف | الاستخدام |
|---|---|
| `metrics.json` | ملخص المقاييس، hashes المدخلات، metadata وسياسة التقييم |
| `predictions.parquet` | توقع كل صف، مناسب لتحليل الأخطاء والشرائح والـ threshold لاحقاً |

## روابط مرتبطة

- تعريف أهداف VALID يأتي من [[knn_readme/02_knn_history_helpers|دوال التاريخ]].
- الـ artifact المقاسة بُنيت في [[knn_readme/03_build_knn_v2_level|مرحلة fit]].
- واجهات `predict` موضحة في [[knn_readme/04_knn_advisor_v2_level|قلب الموديل]].
- تفسير أثر النتيجة في [[knn_readme/06_recommendation_integration|Recommendation]].
- العودة إلى [[knn_readme/README|خريطة KNN الرئيسية]].
