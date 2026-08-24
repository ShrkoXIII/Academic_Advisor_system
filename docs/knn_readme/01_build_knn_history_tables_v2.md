---
aliases:
  - KNN History Builder V2
  - بناء تاريخ KNN
tags:
  - academic-advisor
  - knn
  - knn/data
  - knn/pipeline/01-history
---

# `build_knn_history_tables_v2.py`: بناء تاريخ KNN

> [!abstract] التنقل
> الأعلى: [[knn_readme/README|دليل KNN]] · التالي:
> [[knn_readme/02_knn_history_helpers|دوال تجهيز التاريخ]] · مرتبط أيضاً:
> [[knn_readme/03_build_knn_v2_level|بناء موديلات sklearn]]

## الفكرة

هذا هو أول سكربت في KNN الفعال. وظيفته ليست تدريب موديل، بل تحويل صفوف المقررات الموجودة في `TRAIN` إلى جدولين واضحين:

| الجدول | مستوى الصف | لماذا يحتاجه KNN؟ |
|---|---|---|
| `student_semester_outcomes.parquet` | صف واحد لكل طالب/تخصص/فصل | يحتوي `X` قبل الفصل ونتائج `y` بعد الفصل |
| `student_semester_courses.parquet` | صف لكل مقرر أكمله الطالب | يشرح ما هي المقررات التي أخذها الجيران وكيف كانت نتائجهم |

السكربت يقرأ `TRAIN` فقط. لا يقرأ `VALID` ولا `TEST`، لأن أي نتيجة من هاتين المجموعتين داخل تاريخ الجيران ستكون تسريباً.

## المدخلات الافتراضية

| الوسيط | الملف | المحتوى |
|---|---|---|
| `--train-path` | `train_dataset_candidate.parquet` | صفوف مقررات TRAIN وميزات ما قبل الفصل |
| `--status-path` | `v_add_student_degree_status_v2.parquet` | AGPA والمستوى الرسمي قبل وبعد الفصل |
| `--grades-path` | `v_acs_grade.parquet` | معنى `grade_id` وحالة النجاح أو الرسوب |
| `--output-dir` | `2026-08-23_history_v2_level` | مكان حفظ جدولي التاريخ |

## المقطع الأول: تعريف المصادر

```python
DEFAULT_TRAIN = (
    PROJECT_ROOT
    / "data"
    / "model_data"
    / "versions"
    / "2026-08_temporal_rebuild_v2"
    / "05_dataset"
    / "train_dataset_candidate.parquet"
)
DEFAULT_STATUS = RAW_DIR / "v_add_student_degree_status_v2.parquet"
DEFAULT_GRADES = RAW_DIR / "v_acs_grade.parquet"
```

الفكرة هنا أن حالة الطالب ونوع الدرجة لا يؤخذان من افتراضات داخل KNN. يتم ربط صف TRAIN بمصدر الحالة الرسمي ومصدر الدرجات الرسمي قبل بناء الأهداف.

استخدام ملف الحالة `V2` مهم لأنه يضيف:

- `start_level_name_short`
- `end_level_name_short`

وهما اللذان يسمحان ببناء موديلات منفصلة لكل مستوى أكاديمي.

## المقطع الثاني: حماية المدخلات والمخرجات

```python
semester_path = args.output_dir / "student_semester_outcomes.parquet"
courses_path = args.output_dir / "student_semester_courses.parquet"
assert_data_root(args.train_path, args.status_path, args.grades_path)
ensure_new_output_files([semester_path, courses_path])
```

هذا المقطع يقوم بحمايتين:

1. `assert_data_root` يتأكد أن الملفات المقروءة تقع داخل مسار البيانات المسموح.
2. `ensure_new_output_files` يرفض الكتابة فوق نسخة موجودة.

رفض الاستبدال مقصود. إذا تغيّر المنطق أو المصدر، يجب إنشاء version جديد حتى نعرف أي تاريخ بُني منه كل موديل.

## المقطع الثالث: القراءة الانتقائية

```python
train = pd.read_parquet(args.train_path, columns=TRAIN_COLUMNS)
student_status = pd.read_parquet(
    args.status_path, columns=STATUS_COLUMNS + LEVEL_STATUS_COLUMNS
)
grades = pd.read_parquet(args.grades_path, columns=GRADE_COLUMNS)
```

لا تتم قراءة كل أعمدة الملفات. القوائم الموجودة في `knn_history_helpers.py` تعمل كعقد بيانات واضح؛ إذا كان عمود مطلوب مفقوداً يفشل البناء بدلاً من إنتاج artifact ناقص بصمت.

| مجموعة الأعمدة | أمثلة |
|---|---|
| مفاتيح الهوية | `student_id`, `degree_id`, `part_id`, `course_id` |
| الحالة قبل الفصل | `start_agpa_points`, `start_level_ord`, `diploma_gpa` |
| بيانات الفصل | `semester_reg_credits`, `semester_reg_courses` |
| النتيجة | `final_mark`, `gpa_points`, `grade_id` |
| المرجع الرسمي | `end_agpa_points`, `start_level_name_short`, `finish_status` |

## المقطع الرابع: الإثراء وبناء الجدولين

```python
enriched = attach_official_references(train, student_status, grades)
semester_courses = build_student_semester_courses(enriched)
semester_outcomes = build_student_semester_outcomes(enriched)
```

تسلسل التنفيذ مهم:

1. `attach_official_references` يحدد النجاح والرسوب ويضيف AGPA والمستوى الرسمي.
2. `build_student_semester_courses` يبني تاريخ المقرر ومحاولاته السابقة.
3. `build_student_semester_outcomes` يجمع صفوف المقررات إلى نتيجة فصل واحدة.

تفاصيل هذه الدوال موجودة في [دليل knn_history_helpers](02_knn_history_helpers.md).

## المقطع الخامس: الحفظ

```python
args.output_dir.mkdir(parents=True, exist_ok=True)
semester_outcomes.to_parquet(semester_path, index=False)
semester_courses.to_parquet(courses_path, index=False)
```

يتم استعمال Parquet للأسباب التالية:

- يحافظ على أنواع الأعمدة أفضل من CSV.
- أسرع في القراءة الانتقائية.
- مناسب للبيانات الكبيرة.
- يمكن إعادة استخدام الجداول من دون تكرار الدمج الرسمي في كل تجربة موديل.

## شكل `student_semester_outcomes`

| نوع الحقل | أعمدة نموذجية | زمن توفره |
|---|---|---|
| مفتاح الفصل | `student_id`, `degree_id`, `part_id` | معروف |
| حالة قبل الفصل | `cumulative_gpa_before`, `academic_level_before`, `diploma_gpa` | قبل القرار |
| حمل الفصل | `registered_course_count`, `registered_credit_count` | وقت التسجيل |
| أهداف لاحقة | `any_course_failed`, `term_gpa`, `semester_average_mark` | بعد انتهاء الفصل |
| تغيرات | `term_gpa_delta`, `cumulative_gpa_delta` | بعد انتهاء الفصل |

KNN يستخدم حقول ما قبل الفصل كمدخلات، ويستخدم النتائج اللاحقة كـ labels أثناء `fit`.

## طريقة التشغيل

```powershell
.\.venv\Scripts\python.exe scripts\build_knn_history_tables_v2.py `
  --train-path data\model_data\versions\2026-08_temporal_rebuild_v2\05_dataset\train_dataset_candidate.parquet `
  --status-path data\raw\v_add_student_degree_status_v2.parquet `
  --grades-path data\raw\v_acs_grade.parquet `
  --output-dir data\artifacts\knn\NEW_HISTORY_VERSION
```

إذا كان `output-dir` يحتوي الملفات من قبل، سيطلب السكربت اختيار مسار جديد بدلاً من الكتابة فوقها.

## ما لا يفعله السكربت

- لا يستورد `KNeighborsClassifier`.
- لا ينفذ `fit` أو `predict`.
- لا يختار `K`.
- لا يقيس Accuracy.
- لا يقرأ `VALID` أو `TEST`.

مسؤوليته الوحيدة هي إنتاج تاريخ موثوق يستطيع سكربت التدريب استعماله.

## روابط مرتبطة

- يستخدم دوال [[knn_readme/02_knn_history_helpers|تجهيز التاريخ ومنع التسريب]].
- يجهز مدخلات [[knn_readme/03_build_knn_v2_level|مرحلة fit]].
- الفرق عن History V1 موضح في [[knn_readme/07_legacy_and_active_files|الملفات القديمة والفعالة]].
- العودة إلى [[knn_readme/README|خريطة KNN الرئيسية]].
