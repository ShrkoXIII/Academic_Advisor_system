---
aliases:
  - KNN Documentation Hub
  - دليل KNN
tags:
  - academic-advisor
  - knn
  - knn/index
---

# دليل KNN في نظام المرشد الأكاديمي

> [!info] خريطة Obsidian
> هذه الصفحة هي عقدة البداية. جميع صفحات KNN مرتبطة بها وبالمرحلة السابقة
> واللاحقة، لذلك سيظهر المسار كاملاً داخل Graph View.
>
> لكي تُحل المسارات كما هي، ضع مجلد `knn_readme` نفسه في جذر الـ Vault، أو
> افتح مجلد `docs` كـ Vault. لا تنقل ملفات Markdown منفردة من دون المجلد.

هذا المجلد هو نقطة الدخول لفهم قسم KNN كاملاً: كيف تُبنى بياناته من `TRAIN`، وكيف يتم عمل `fit` لموديلات sklearn، وكيف تتم عملية `predict`، وكيف تُقاس النتائج، ثم كيف تدخل النتيجة في `recommendation`.

## الفكرة الرئيسة

KNN هنا لا يتنبأ بهدف واحد فقط. لكل مجموعة طلاب متجانسة توجد ثلاثة تنبؤات:

| نوع المهمة | موديل sklearn                       | الهدف                                                                | أهم المقاييس                                                |
| ---------- | ----------------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------- |
| تصنيف      | `KNeighborsClassifier`              | هل رسب الطالب في مقرر واحد على الأقل خلال الفصل؟ `any_course_failed` | Accuracy, ROC-AUC, Average Precision, Precision, Recall, F1 |
| انحدار     | `KNeighborsRegressor`               | معدل الفصل `term_gpa`                                                | MAE, RMSE, R²                                               |
| انحدار     | نفس `KNeighborsRegressor` كخرج ثانٍ | متوسط علامات الفصل `semester_average_mark`                           | MAE, RMSE, R²                                               |

لذلك عبارة **دقة KNN** يجب أن تُقرأ حسب الهدف:

- `Accuracy = 74.92%` تخص تصنيف وجود رسوب، ولا تخص المعدل.
- `GPA MAE = 0.506` يخص التنبؤ بمعدل الفصل.
- `Mark MAE = 8.34` يخص التنبؤ بمتوسط العلامة.

## المسار الكامل

```text
TRAIN course rows
    |
    v
إضافة مراجع الحالة والدرجة الرسمية
    |
    +--> student_semester_courses.parquet
    |
    +--> student_semester_outcomes.parquet
              |
              v
تقسيم حسب route + degree_id + academic_level
              |
              v
KNeighborsClassifier.fit(X, any_course_failed)
KNeighborsRegressor.fit(X, [term_gpa, semester_average_mark])
              |
              v
knn_v2_level_sklearn_k20.pkl
              |
       +------+------+
       |             |
       v             v
VALID evaluation   recommendation
predict/proba      predict + kneighbors
```

## مسارا الطالب

| المسار | متى يستخدم؟ | ميزة الإدخال `X` | شرط المطابقة |
|---|---|---|---|
| Returning | الطالب لديه سجل جامعي سابق | `cumulative_gpa_before` | نفس `degree_id` ونفس `academic_level` |
| Cold-start | أول فصل نشط للطالب | `diploma_gpa` | نفس `degree_id` ونفس `academic_level` |

لا يوجد fallback إلى تخصص أو مستوى مختلف. إذا لم توجد مجموعة مطابقة تكون الحالة `covered=False`، ويستمر نظام التوصية من دون KNN.

## الملفات الفعالة حالياً

| الترتيب | الملف البرمجي | صفحة الشرح | المسؤولية |
|---:|---|---|---|
| 1 | `scripts/build_knn_history_tables_v2.py` | [بناء جداول التاريخ](01_build_knn_history_tables_v2.md) | تحويل TRAIN إلى جدول نتائج الفصول وجدول مقررات الفصول |
| 2 | `src/knn_history_helpers.py` | [دوال تجهيز التاريخ](02_knn_history_helpers.md) | الدمج الرسمي، منع التسريب، التجميع والتحقق |
| 3 | `scripts/build_knn_v2_level.py` | [بناء موديلات sklearn](03_build_knn_v2_level.md) | قراءة التاريخ واستدعاء `fit` وحفظ الـ artifact |
| 4 | `src/knn_advisor_v2.py` | [قلب موديل KNN](04_knn_advisor_v2_level.md) | تعريف المصنف والمنحدر وواجهات `predict` و`kneighbors` |
| 5 | `scripts/evaluate_knn_v2_level.py` | [تقييم KNN](05_evaluate_knn_v2_level.md) | تطبيق الموديلات على VALID وحساب مقاييس التصنيف والانحدار |
| 6 | `src/recommendation.py` | [الربط مع Recommendation](06_recommendation_integration.md) | تحويل توقع KNN إلى evidence واستعماله في الخطة |
| 7 | ملفات KNN السابقة | [النسخ القديمة وما هو الفعال](07_legacy_and_active_files.md) | منع الخلط بين lookup القديم والموديل الحالي |

## خريطة روابط Obsidian

- البيانات التاريخية: [[knn_readme/01_build_knn_history_tables_v2|بناء تاريخ KNN]]
- منطق تجهيز البيانات: [[knn_readme/02_knn_history_helpers|دوال التاريخ ومنع التسريب]]
- تدريب sklearn: [[knn_readme/03_build_knn_v2_level|عمل fit وحفظ الموديل]]
- قلب الموديل: [[knn_readme/04_knn_advisor_v2_level|predict وpredict_proba وkneighbors]]
- التقييم: [[knn_readme/05_evaluate_knn_v2_level|مقاييس التصنيف والانحدار]]
- التوصيات: [[knn_readme/06_recommendation_integration|ربط KNN مع Recommendation]]
- تاريخ النسخ: [[knn_readme/07_legacy_and_active_files|الملفات القديمة والفعالة]]

العلاقات الأساسية في الرسم:

```text
01 History Builder <--> 02 Helpers <--> 03 Fit
                              |             |
                              v             v
                         05 Evaluation <--> 04 Model
                                              |
                                              v
                                      06 Recommendation

07 Legacy/Active يرتبط بجميع مراحل المسار لتوضيح النسخة الصحيحة.
```

## الـ artifacts الحالية

| Artifact | المحتوى | مصدر البيانات |
|---|---|---|
| `data/artifacts/knn/2026-08-23_history_v2_level/student_semester_outcomes.parquet` | صف واحد لكل طالب/تخصص/فصل مع الحالة السابقة والنتيجة اللاحقة | TRAIN فقط |
| `data/artifacts/knn/2026-08-23_history_v2_level/student_semester_courses.parquet` | صف لكل مقرر تاريخي مع محاولة الطالب ونتيجتها | TRAIN فقط |
| `data/artifacts/knn/2026-08-24_sklearn_v3/knn_v2_level_sklearn_k20.pkl` | موديلات sklearn المـfit مع بيانات الشرح والفهارس | الجداول السابقة |
| `reports/knn/2026-08-24_knn_sklearn_v3_valid_k20/metrics.json` | نتائج القياس المفصلة | VALID فقط |
| `reports/knn/2026-08-24_knn_sklearn_v3_valid_k20/predictions.parquet` | توقع كل حالة في VALID | VALID فقط |

## النتيجة الحالية على VALID

| الشريحة | AUC للتصنيف | Accuracy | Recall للرسوب | GPA MAE | Mark MAE |
|---|---:|---:|---:|---:|---:|
| جميع الحالات المغطاة | 73.61% | 74.92% | 19.28% | 0.506 | 8.34 |
| Returning | 75.42% | 75.78% | 16.85% | 0.477 | 7.81 |
| Cold-start | 63.44% | 66.23% | 44.34% | 0.799 | 13.73 |

ملاحظتان مهمتان:

1. خط الأساس الذي يقول دائماً «لا يوجد رسوب» يحقق Accuracy مقدارها `74.26%` على الحالات المغطاة، لذلك لا يجوز تقييم المصنف بالـ Accuracy وحدها.
2. `R²` للـ cold-start سلبي في تنبؤ المعدل والعلامة؛ لذلك هذه الشريحة ليست مناسبة للاعتماد على KNN منفرداً.

## أوامر التشغيل بالترتيب

```powershell
# 1) بناء تاريخ جديد. استخدم output-dir جديداً لأن السكربت يرفض الاستبدال.
.\.venv\Scripts\python.exe scripts\build_knn_history_tables_v2.py `
  --output-dir data\artifacts\knn\NEW_HISTORY_VERSION

# 2) عمل fit وحفظ موديل sklearn جديد.
.\.venv\Scripts\python.exe scripts\build_knn_v2_level.py `
  --history-dir data\artifacts\knn\NEW_HISTORY_VERSION `
  --output data\artifacts\knn\NEW_MODEL_VERSION\knn_sklearn_k20.pkl `
  --n-neighbors 20

# 3) قياس الموديل على VALID، من دون قراءة TEST.
.\.venv\Scripts\python.exe scripts\evaluate_knn_v2_level.py `
  --knn-path data\artifacts\knn\NEW_MODEL_VERSION\knn_sklearn_k20.pkl `
  --output-dir reports\knn\NEW_VALID_REPORT
```

## قواعد تمنع الأخطاء

| القاعدة | السبب |
|---|---|
| بناء التاريخ والموديل من TRAIN فقط | منع تسريب نتائج VALID أو TEST إلى الجيران |
| عدم استبدال artifact قائم | الحفاظ على إمكانية إعادة النتائج ومقارنة الإصدارات |
| تثبيت `K` داخل اسم artifact | `predict` يعتمد على قيمة `n_neighbors` التي فُت بها الموديل |
| عدم مقارنة MAE العلامة مع Accuracy | الأول انحدار بوحدة درجات، والثاني تصنيف بنسبة |
| إبقاء TEST مغلقاً أثناء التطوير | استخدامه مرة واحدة بعد تثبيت التصميم والـ threshold |
| قراءة نتيجة cold-start منفصلة | سلوكها أضعف بوضوح من returning |

## الخلاصة

قسم KNN الحالي هو حزمة موديلات sklearn حقيقية، وليس مجرد فرز يدوي:

- يتم عمل `fit` داخل كل مجموعة تخصص ومستوى.
- يتم استدعاء `predict` و`predict_proba` للتنبؤ.
- يتم استدعاء `kneighbors` لإظهار الحالات التاريخية المشابهة وشرح القرار.
- التصنيف والانحدار هدفان مستقلان، حتى لو كانا يستخدمان الجيران أنفسهم.
- KNN حالياً evidence مساند لـ M1/M2، وليس بديلاً عنهما، خصوصاً في cold-start.
