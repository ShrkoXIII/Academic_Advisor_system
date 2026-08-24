---
aliases:
  - KNN Legacy vs Active
  - نسخ KNN القديمة والحالية
tags:
  - academic-advisor
  - knn
  - knn/legacy
  - knn/migration
---

# ملفات KNN القديمة والملفات الفعالة

> [!abstract] التنقل
> الأعلى: [[knn_readme/README|دليل KNN]] · المسار الفعال يبدأ من:
> [[knn_readme/01_build_knn_history_tables_v2|History V2]] وينتهي في:
> [[knn_readme/06_recommendation_integration|Recommendation]]

## لماذا توجد عدة نسخ؟

المشروع مر بعدة مراحل. بعض الملفات ما زالت موجودة للتوثيق والتوافق وإعادة التجارب القديمة، لكنها ليست المسار الذي تستعمله recommendation الحالية.

## جدول الحالة

| الملف | التقنية | مستوى المطابقة | الحالة الحالية |
|---|---|---|---|
| `src/knn_advisor.py` | `NearestNeighbors` + `StandardScaler` على 10 ميزات | عالمي | Legacy، غير مربوط بالـ recommendation الحالية |
| `scripts/build_knn_history_tables.py` | بناء History V1 | تخصص، من دون level الرسمي V2 | Legacy |
| `scripts/build_knn_v2.py` | `KNNAdvisorV2` lookup مخصص | تخصص + GPA | Legacy |
| `src/knn_advisor_v2.py::KNNAdvisorV2` | فرز يدوي حسب فرق GPA | تخصص فقط | Compatibility |
| `scripts/build_knn_history_tables_v2.py` | History V2 رسمي | تخصص + مستوى | فعال |
| `scripts/build_knn_v2_level.py` | sklearn classifier/regressor | تخصص + مستوى + route | فعال |
| `src/knn_advisor_v2.py::KNNAdvisorV2Level` | fit/predict/kneighbors | تخصص + مستوى + route | فعال |
| `scripts/evaluate_knn_v2_level.py` | sklearn native evaluation | VALID | فعال |

## `src/knn_advisor.py`

هذه كانت محاولة مبكرة تستخدم sklearn فعلاً:

```python
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filled)

nn = NearestNeighbors(
    n_neighbors=n_neighbors,
    metric=metric,
    algorithm="ball_tree",
)
nn.fit(X_scaled)
```

ميزاتها كانت أوسع من GPA:

- GPA signals.
- المستوى.
- عدد الفصول السابقة.
- تاريخ الرسوب.
- الانقطاعات.
- الساعات المكتسبة.

لكن `NearestNeighbors` لا يحتوي `predict` لأنه index بحث، وليس مصنفاً أو regressor. كانت النتائج تُحسب يدوياً من صفوف الجيران. لذلك لا تحقق الشرط الحالي الذي يطلب classifier/regressor مع `fit` و`predict`.

## `build_knn_history_tables.py`: History V1

المقطع الرئيس:

```python
student_status = pd.read_parquet(
    args.status_path,
    columns=STATUS_COLUMNS,
)
```

هذه النسخة لا تطلب `LEVEL_STATUS_COLUMNS`. لذلك لا تضمن المطابقة على المستوى الرسمي قبل وبعد الفصل، وتكتب إلى:

```text
data/artifacts/knn/2026-08-23_history_v1
```

لا تستخدمها عند بناء موديل sklearn الحالي.

## `build_knn_v2.py`: GPA lookup حسب التخصص

```python
advisor = KNNAdvisorV2.build(outcomes, courses)
advisor.save(args.output)
```

`KNNAdvisorV2` في هذه النسخة لا ينشئ `KNeighborsClassifier` أو `KNeighborsRegressor`. هو يحفظ مواضع الصفوف ثم ينفذ:

```python
candidates["gpa_distance"] = (
    candidates["matched_gpa"] - query_gpa
).abs()
```

لذلك artifact التالية قديمة وليست موديل sklearn للتنبؤ:

```text
data/artifacts/knn/2026-08-23_history_v1/knn_v2_gpa_nearest.pkl
```

## الفرق بين `NearestNeighbors` والموديلات الحالية

| الصنف | لديه `fit`؟ | لديه `kneighbors`؟ | لديه `predict`؟ | الاستخدام |
|---|---:|---:|---:|---|
| `NearestNeighbors` | نعم | نعم | لا | البحث عن صفوف قريبة فقط |
| `KNeighborsClassifier` | نعم | نعم | نعم | تصنيف الرسوب واحتماله |
| `KNeighborsRegressor` | نعم | نعم | نعم | توقع GPA والعلامة |

## كيف أعرف artifact الصحيحة؟

بعد التحميل اطبع metadata:

```python
from src.knn_advisor_v2 import KNNAdvisorV2Level

advisor = KNNAdvisorV2Level.load(
    "data/artifacts/knn/2026-08-24_sklearn_v3/"
    "knn_v2_level_sklearn_k20.pkl"
)
print(advisor.metadata)
```

يجب أن ترى:

```text
version: knn_v2_level_sklearn_v3
backend: sklearn
classifier: sklearn.neighbors.KNeighborsClassifier
regressor: sklearn.neighbors.KNeighborsRegressor
fit_called: True
n_neighbors: 20
```

إذا كانت version هي `knn_v2_gpa_level_nearest_v2` أو لا يظهر `backend=sklearn` فهذا artifact lookup قديمة.

## هل نحذف الملفات القديمة؟

ليس ضرورياً الآن. حذفها قد يكسر تجارب أو مراجع سابقة. الأفضل:

- إبقاؤها كـ legacy.
- عدم وضع مساراتها في كود التشغيل الحالي.
- استعمال أسماء versions واضحة.
- حذفها لاحقاً فقط بعد التأكد أنه لا يوجد مستهلك لها وتسجيل قرار migration.

## المسار المعتمد المختصر

```text
build_knn_history_tables_v2.py
        -> History V2 level
build_knn_v2_level.py
        -> sklearn artifact V3 K20
evaluate_knn_v2_level.py
        -> VALID report
recommendation.py
        -> production integration
```

## روابط المسار الفعال

- [[knn_readme/01_build_knn_history_tables_v2|بناء History V2]]
- [[knn_readme/02_knn_history_helpers|دوال تجهيز التاريخ]]
- [[knn_readme/03_build_knn_v2_level|بناء sklearn artifact]]
- [[knn_readme/04_knn_advisor_v2_level|المصنف والمنحدر الحاليان]]
- [[knn_readme/05_evaluate_knn_v2_level|التقييم الحالي]]
- [[knn_readme/06_recommendation_integration|الربط الإنتاجي]]
- العودة إلى [[knn_readme/README|خريطة KNN الرئيسية]].
