---
aliases:
  - KNN sklearn Model Builder
  - تدريب KNN sklearn
tags:
  - academic-advisor
  - knn
  - knn/training
  - knn/sklearn
  - knn/pipeline/03-fit
---

# `build_knn_v2_level.py`: عمل fit وحفظ موديلات sklearn

> [!abstract] التنقل
> السابق: [[knn_readme/02_knn_history_helpers|تجهيز التاريخ]] · الأعلى:
> [[knn_readme/README|دليل KNN]] · التالي:
> [[knn_readme/04_knn_advisor_v2_level|قلب موديل KNN]]

## الفكرة

هذا هو سكربت التدريب الفعلي. يقرأ جدولي التاريخ الناتجين من المرحلة السابقة، يستدعي `KNNAdvisorV2Level.build(...)`، ثم يحفظ المصنفات والمنحدرات بعد عمل `fit` داخل ملف Pickle واحد.

السكربت صغير عمداً؛ منطق التدريب موجود في `src/knn_advisor_v2.py` حتى يمكن اختباره واستعماله من Python من دون CLI.

## المدخلات والمخرجات

| النوع | المسار الافتراضي | الوصف |
|---|---|---|
| Input | `.../2026-08-23_history_v2_level/student_semester_outcomes.parquet` | ميزات وأهداف كل فصل تاريخي |
| Input | `.../2026-08-23_history_v2_level/student_semester_courses.parquet` | تفاصيل المقررات لأغراض الشرح |
| Output | `.../2026-08-24_sklearn_v3/knn_v2_level_sklearn_k20.pkl` | حزمة الموديلات المـfit |

## خيارات التدريب

```python
parser.add_argument("--n-neighbors", type=int, default=20)
parser.add_argument(
    "--weights", choices=["uniform", "distance"], default="uniform"
)
parser.add_argument("--metric", default="euclidean")
```

| الخيار | القيمة الحالية | المعنى |
|---|---|---|
| `n_neighbors` | 20 | عدد الجيران الذين يصوتون أو يدخلون في المتوسط |
| `weights` | `uniform` | كل جار يحصل على الوزن نفسه |
| `metric` | `euclidean` | المسافة؛ مع ميزة GPA واحدة تعادل فرق GPA المطلق في الترتيب |

إذا استُعمل `weights="distance"` تصبح الحالات الأقرب في GPA أكثر تأثيراً، لكن يجب بناء artifact جديد وتقييمه على VALID قبل اعتماده.

## قراءة تاريخ التدريب

```python
outcomes_path = args.history_dir / "student_semester_outcomes.parquet"
courses_path = args.history_dir / "student_semester_courses.parquet"
assert_data_root(outcomes_path, courses_path)

outcomes = pd.read_parquet(outcomes_path)
courses = pd.read_parquet(courses_path)
```

الموديل يتعلم من `outcomes`. أما `courses` فلا تدخل في `X` أو `y`، بل تُحفظ كي يستطيع recommendation لاحقاً شرح المقررات التي أخذها الجيران وحساب تداخل خطة مقترحة معهم.

## استدعاء التدريب

```python
advisor = KNNAdvisorV2Level.build(
    outcomes,
    courses,
    n_neighbors=args.n_neighbors,
    weights=args.weights,
    metric=args.metric,
)
```

داخل `build` يحدث الآتي:

1. التحقق من الأعمدة.
2. تطبيع IDs.
3. تقسيم البيانات إلى returning وcold-start.
4. تقسيم كل route حسب `degree_id + academic_level`.
5. اختيار آخر snapshot تاريخي لكل طالب داخل المجموعة.
6. إنشاء `KNeighborsClassifier` و`KNeighborsRegressor` لكل مجموعة.
7. استدعاء `fit` على كل estimator.

هذا يعني أن artifact ليس موديل sklearn عالمي واحد؛ بل قاموس من الموديلات المحلية المتخصصة.

## لماذا موديل لكل تخصص ومستوى؟

إذا درّبنا موديل GPA واحداً على كل الطلاب، قد يصبح أقرب جار لطالب في المستوى الثالث طالباً في المستوى الأول أو تخصصاً مختلفاً. التقسيم يجعل معنى المسافة أكثر منطقية:

```text
distance is compared only after:
route matches
AND degree_id matches
AND academic_level matches
```

| حالة الاستعلام | هل تدخل المجموعة؟ |
|---|---|
| نفس التخصص، نفس المستوى، GPA قريب | نعم |
| نفس التخصص، مستوى مختلف | لا |
| تخصص مختلف، نفس GPA | لا |
| cold-start مقابل returning | لا |

## حفظ artifact

```python
advisor.save(args.output)
```

الملف المحفوظ يحتوي:

| المفتاح | المحتوى |
|---|---|
| `version` | عقد النسخة `knn_v2_level_sklearn_v3` |
| `outcomes` | صفوف TRAIN المستخدمة في الشرح وإرجاع الجيران |
| `courses` | مقررات الجيران |
| `route_positions` | مواضع الصفوف لكل مجموعة |
| `group_models` | المصنف والمنحدر المـfit لكل مجموعة |
| `n_neighbors` | قيمة K التي تدرب عليها الموديل |
| `weights`, `metric` | إعدادات المسافة والتصويت |

وجود version يمنع تحميل artifact قديم بالـ class الجديدة وكأنه متوافق.

## metadata بعد البناء الحالي

| الحقل | القيمة |
|---|---:|
| Backend | sklearn |
| K | 20 |
| Returning training snapshots | 40,741 |
| Cold-start training snapshots | 13,251 |
| Returning groups | 166 |
| Cold-start groups | 100 |
| Raw semester rows before unique-student policy | 130,277 |

## التشغيل

```powershell
.\.venv\Scripts\python.exe scripts\build_knn_v2_level.py `
  --history-dir data\artifacts\knn\2026-08-23_history_v2_level `
  --output data\artifacts\knn\2026-08-24_sklearn_v3\knn_v2_level_sklearn_k20.pkl `
  --n-neighbors 20 `
  --weights uniform `
  --metric euclidean
```

## تغيير K بصورة صحيحة

لا تغيّر `knn_k` في recommendation فقط. `KNeighborsClassifier.predict` يستعمل قيمة K الموجودة في الموديل المـfit. لتجربة `K=10` مثلاً:

```powershell
.\.venv\Scripts\python.exe scripts\build_knn_v2_level.py `
  --output data\artifacts\knn\EXPERIMENT_K10\knn_sklearn_k10.pkl `
  --n-neighbors 10
```

بعد ذلك قيّم artifact الجديدة في report جديد. الكود يرفض حالياً تمرير `k` مختلفة عن `n_neighbors` المسجلة حتى لا يعطي neighbor evidence من K وتنبؤاً من K أخرى.

## حدود مسؤولية السكربت

- يعمل `fit` لكنه لا يقيس الدقة.
- لا يقرأ VALID أو TEST.
- لا يقرر threshold تصنيف.
- لا يربط الموديل بخطط المقررات.
- مسؤوليته إنتاج artifact قابلة للتحميل والتنبؤ.

## روابط مرتبطة

- يقرأ ناتج [[knn_readme/01_build_knn_history_tables_v2|History Builder V2]].
- يستدعي `build/fit` في [[knn_readme/04_knn_advisor_v2_level|KNNAdvisorV2Level]].
- الـ artifact الناتجة تُقاس في [[knn_readme/05_evaluate_knn_v2_level|تقييم VALID]].
- النسخة التي استبدلها موضحة في [[knn_readme/07_legacy_and_active_files|دليل Legacy]].
- العودة إلى [[knn_readme/README|خريطة KNN الرئيسية]].
