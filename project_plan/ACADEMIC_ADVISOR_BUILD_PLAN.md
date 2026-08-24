# خطة بناء نظام Academic Advisor

> **الحالة:** خطة تنفيذ مقترحة، ولا تعتمد أي تغيير في البيانات أو النماذج تلقائيًا.  
> **تاريخ الإنشاء:** 2026-08-22  
> **النطاق:** من الحالة الحالية إلى نظام توصية متكامل قابل للربط مع الواجهة الرسمية.  
> **المبدأ:** تثبيت ما يعمل، بناء القواعد والتوصية حوله، ثم تحسين النماذج بعد قياس جودة الخطة كاملة.

---

## 1. الهدف النهائي

النظام المطلوب   لا يكتفي بتوقع علامة الطالب أو احتمال نجاحه في كل مقرر، بل يجب أن:

1. يستقبل حالة الطالب الحالية من النظام الرسمي.
2. يحدد المقررات المطروحة والمؤهل لها أكاديميًا.
3. يتوقع العلامة واحتمال النجاح لكل مقرر.
4. يولد خططًا فصلية صالحة من المقررات المؤهلة.
5. يحسب معدل الفصل المتوقع والمعدل التراكمي المتوقع بعد كل خطة.
6. يعطي الأولوية للخطط المتوقع أن تحسن المعدل، مع إبقاء خطر الرسوب مقبولًا.
7. يستخدم نتائج طلاب تاريخيين مشابهين وخطط مواد مشابهة كدليل إضافي عبر KNN.
8. يوضح لماذا اختيرت الخطة وما حدود الثقة فيها.
9. يعمل من خلال API واضحة ومحددة المصدر والتوقيت.

السؤال الذي يجب أن يجيب عنه المنتج هو:

> ما أفضل مجموعة مقررات مؤهلة ومطروحة لهذا الطالب في الفصل القادم، بحيث تزيد فرصة تحسن معدله، تقلل خطر الرسوب، وتدفعه نحو التخرج، وما الدليل التاريخي من طلاب مشابهين؟

---

## 2. القرارات المعمارية أثناء البناء

### 2.1 تجميد النماذج مؤقتًا

- لا إعادة تدريب أثناء بناء الـpipeline الوظيفي.
- تستخدم نسخة LightGBM الحالية كمرجع V0 إلى أن يكتمل الاختبار التاريخي للخطط.
- يبقى XGBoost مرشحًا محفوظًا للترقية اللاحقة، ولا يرقى تلقائيًا للإنتاج.
- لا يتغير feature contract أو تعريف TRAIN/VALID/TEST أثناء بناء طبقات المنتج.
- يجب أن يكون محرك التوصية مستقلًا عن نوع النموذج من خلال `ModelAdapter` موحد.

### 2.2 قواعد حتمية

- الأهلية، المتطلبات السابقة، الساعات، GPA، والإعادة تنفذ بقواعد برمجية حتمية.
- M1 وM2 يقدمان التنبؤات فقط.
- KNN يقدم دليلًا تاريخيًا ولا يستبدل النموذج.
- LLM، إن أضيف، يصيغ الشرح فقط ولا يغير القواعد أو التنبؤ أو ترتيب الخطة.

### 2.3 منع التسرب الزمني

- كل معلومة تستخدم في قرار فصل معين يجب أن تكون متاحة قبل تسجيل ذلك الفصل.
- في الاختبار التاريخي لا تستخدم نتائج الفصل المستهدف أو أي فصل لاحق.
- إحصاءات المقررات وKNN تبنى من فصول مكتملة تاريخيًا فقط.
- في الإنتاج تحدث الجداول التاريخية بعد إغلاق الفصل رسميًا، لا أثناءه.

### 2.4 الخصوصية والتدقيق

- لا تعرض معرفات أو أسماء الطلاب التاريخيين في ناتج KNN.
- تعرض النتائج على شكل إحصاءات مجمعة مع عدد حالات الدعم.
- تحمل كل توصية versions للنموذج والقواعد والجداول التي استعملتها.
- لا تحفظ معلومات شخصية أكثر مما يحتاجه التشغيل والتدقيق المصرح به.

---

## 3. المعمارية المستهدفة

```text
Official University System
        |
        | student request + official reference feeds
        v
API Contract / Validation Layer
        |
        +--> Student Snapshot Builder
        |
        +--> Eligibility Engine <---- curriculum / offerings / prerequisites
        |          |
        |          v
        |     Eligible Courses
        |          |
        v          v
Model Adapter --> Course Predictions (M1 + M2)
                       |
                       v
                 Plan Generator
                       |
                       +--> Exact plan-context prediction
                       +--> Official AGPA Engine
                       +--> KNN Similar-Plan Evidence
                       +--> Risk / Progress / Workload
                       |
                       v
                 Plan Ranker
                       |
                       v
Recommendation API Response + Explanation + Audit Metadata
```

كل طبقة تملك مسؤولية واحدة ومخرجات قابلة للاختبار. لا تحتوي طبقة العرض أو API على منطق أكاديمي مخفي.

---

## 4. تقسيم البيانات حسب المصدر والتوقيت

### 4.1 بيانات تصل عند كل طلب

| الحقل | الغرض |
|---|---|
| `student_id` | مفتاح الطلب والتدقيق الداخلي |
| `degree_id` | اختيار الخطة الدراسية والقواعد الصحيحة |
| `target_part_id` | الفصل المطلوب التخطيط له |
| `current_agpa` | المعدل التراكمي الرسمي الحالي |
| `previous_term_gpa` | مقارنة معدل الخطة بالفصل السابق |
| `cumulative_quality_points` | بسط حساب المعدل الرسمي إن كان متاحًا |
| `completed_gpa_credits` | مقام حساب المعدل الرسمي |
| `completed_degree_credits` | قياس التقدم نحو التخرج |
| `academic_level` | الأهلية وتشابه KNN |
| `academic_status` | انتظام، إنذار، إيقاف، أو حالة خاصة |
| `max_allowed_credits` | الحد الرسمي لعبء الطالب |
| `course_history` | المحاولات السابقة ونتائجها وحالاتها |
| `in_progress_courses` | منع التكرار ومعالجة التزامن |
| `diploma_gpa`, `diploma_type_id` | دعم cold-start عند غياب التاريخ الجامعي |

يفضل إرسال الحقائق الرسمية الخام بدل ميزات مشتقة مثل `fail_credit_ratio_capped`. يحسب النظام الميزات المشتقة داخليًا بنفس منطق التدريب.

إذا أرسل النظام الرسمي `current_agpa` وتوفرت النقاط والساعات اللازمة لإعادة حسابه، يحسب النظام نسخة تدقيقية ويبلغ عن الاختلاف بدل استبدال القيمة الرسمية بصمت.

### 4.2 جداول مرجعية تتم مزامنتها دوريًا

| الجدول المقترح | المحتوى | التحديث |
|---|---|---|
| `advisor_degree_course` | مقررات كل درجة، نوع المتطلب، الساعات | عند تغيير الخطة |
| `advisor_course_prerequisite` | المتطلبات السابقة والمتزامنة | عند تغيير الخطة |
| `advisor_term_course_offer` | المقررات المطروحة في الفصل | أثناء التحضير والتسجيل |
| `advisor_grade_scale` | العلامات وgrade IDs ونقاط GPA والنجاح | عند تغيير النظام |
| `advisor_academic_policy` | الساعات، الإعادة، الإنذار، الاستثناءات | عند تغيير اللوائح |
| `advisor_course_equivalence` | معادلات واستبدالات المقررات | عند اعتماد المعادلات |
| `advisor_degree_requirement` | الساعات المطلوبة لكل مجموعة | عند تغيير الخطة |

كل نسخة مرجعية تحتاج `effective_from_part_id` و`effective_to_part_id` و`source_updated_at` و`received_at` وversion أو hash رسمي.

### 4.3 حسابات Offline

- إحصاءات صعوبة ونجاح المقررات التاريخية.
- difficulty state وfallbacks.
- جداول نتائج الطالب-الفصل والطالب-الفصل-المقرر.
- KNN indexes وscalers وmedian fills.
- diploma bucket map.
- model artifacts وfeature contracts.
- فهارس المقررات والمتطلبات والعروض.

### 4.4 حسابات Query-time

- التحقق من payload وأنواع المفاتيح.
- بناء student snapshot.
- حساب الميزات المشتقة من الماضي.
- تحديد الأهلية وأسباب الرفض.
- توقع M1 وM2 للمقررات.
- توليد الخطط الصالحة.
- إعادة التنبؤ بسياق الحمل الحقيقي وميزات concurrent.
- حساب معدل الفصل والمعدل التراكمي المتوقع.
- استرجاع دليل KNN الخاص بكل خطة.
- ترتيب الخطط وإنشاء الشرح وmetadata التدقيق.

---

## 5. عقد API المقترح

### 5.1 الطلب

```http
POST /v1/recommendations
```

```json
{
  "request_id": "uuid",
  "student": {
    "student_id": "...",
    "degree_id": "40.111",
    "target_part_id": "20261",
    "current_agpa": 2.84,
    "previous_term_gpa": 2.72,
    "cumulative_quality_points": 207.32,
    "completed_gpa_credits": 73,
    "completed_degree_credits": 75,
    "academic_level": 3,
    "academic_status": "regular",
    "max_allowed_credits": 18,
    "diploma_gpa": 86.4,
    "diploma_type_id": 15
  },
  "course_history": [
    {
      "course_id": "101.111",
      "part_id": "20252",
      "course_credits": 3,
      "grade_id": "...",
      "final_mark": 77,
      "finish_status": "..."
    }
  ],
  "in_progress_courses": []
}
```

### 5.2 الاستجابة

```json
{
  "request_id": "uuid",
  "student_state": {
    "current_agpa": 2.84,
    "previous_term_gpa": 2.72,
    "cold_start": false
  },
  "plans": [
    {
      "courses": ["201.111", "220.111", "310.111"],
      "total_credits": 9,
      "expected_term_gpa": 3.18,
      "delta_vs_previous_term": 0.46,
      "expected_new_agpa": 2.91,
      "expected_cumulative_delta": 0.07,
      "average_pass_probability": 0.88,
      "minimum_pass_probability": 0.76,
      "graduation_progress": 0.05,
      "knn": {
        "support": 34,
        "median_plan_similarity": 0.67,
        "improved_rate": 0.706,
        "median_term_gpa_delta": 0.22,
        "any_fail_rate": 0.147,
        "confidence": "medium"
      }
    }
  ],
  "rejected_courses": [],
  "versions": {
    "model": "...",
    "feature_contract": "...",
    "knn_index": "...",
    "academic_rules": "...",
    "reference_data": "..."
  }
}
```

### 5.3 قواعد العقد

- IDs تصل كسلاسل نصية ولا تحول إلى float.
- كل طلب يحمل `request_id` لدعم التدقيق ومنع التكرار.
- القيم الرسمية والقيم المحسوبة لا تحمل الاسم نفسه.
- الحقول الناقصة تعطي validation error أو fallback موثقًا.
- لا تسقط history rows بصمت.
- الاستجابة تعرض version كل artifact مستخدم.
- تحدد صيغة timestamps وtimezone قبل الربط النهائي.

---

## 6. محرك GPA وهدف تحسين المعدل

### 6.1 قيم منفصلة

1. `previous_term_gpa`: معدل آخر فصل مكتمل.
2. `current_agpa`: المعدل التراكمي قبل الخطة.
3. `expected_term_gpa`: معدل الخطة المتوقع.
4. `expected_new_agpa`: المعدل التراكمي المتوقع بعد الخطة.
5. `delta_vs_previous_term = expected_term_gpa - previous_term_gpa`.
6. `expected_cumulative_delta = expected_new_agpa - current_agpa`.

ارتفاع معدل الفصل عن الفصل السابق لا يضمن ارتفاع المعدل التراكمي؛ لذلك تعرض المقارنتان معًا، ويكون الهدف الأساسي رفع التراكمي ما لم تعتمد سياسة منتج مختلفة.

### 6.2 الحساب الرسمي

الصيغة العامة قبل قواعد الإعادة:

```text
expected_new_agpa =
    (current_quality_points + expected_plan_quality_points)
    /
    (completed_gpa_credits + plan_gpa_credits)
```

التنفيذ النهائي يأتي من `advisor_grade_scale` وقواعد الجامعة الرسمية، خصوصًا في الإعادة، استبدال المحاولة، المقررات التي لا تدخل في المعدل، الانسحاب، الغياب، incomplete، واختلاف grade scale.

لا يعتمد `_mark_to_gpa_points` الحالي كمرجع رسمي.

### 6.3 مساهمة المادة

```text
marginal_agpa_uplift =
    expected_new_agpa(plan)
    - expected_new_agpa(plan without this course)
```

هذا يبين المادة التي يتوقع أن تدفع المعدل للأعلى داخل الخطة، لكن لا يسمح باختيار مواد سهلة لا تفيد التخرج؛ الأهلية والتقدم الدراسي قيود مستقلة.

### 6.4 عند غياب خطة محسنة

- لا يعد النظام بزيادة غير موجودة.
- إذا لم تحقق أي خطة `expected_cumulative_delta > 0`، يعرض أفضل خطة متاحة مع تحذير واضح.
- تعرض المخاطر وعدم اليقين بدل نقطة توقع مفردة فقط عندما تتوفر آلية موثوقة لذلك.

---

## 7. KNN كدليل نتائج خطة مشابهة

### 7.1 الدور

يجيب KNN عن:

> طلاب كانوا في حالة أكاديمية مشابهة قبل التسجيل، وأخذوا مجموعة مواد مشابهة، ماذا كانت علاماتهم؟ وهل ارتفع معدل فصلهم ومعدلهم التراكمي أم انخفض؟

لا يستخدم كمصدر حقيقة أو بديل عن M1/M2، بل كدليل مستقل يمكن أن يقوي أو يضعف الثقة بخطة معينة.

### 7.2 جدول `advisor_student_semester_outcome`

صف واحد لكل `(student_id, degree_id, part_id)`:

```text
student_id
degree_id
part_id
start_level_ord
start_agpa
previous_term_gpa
start_completed_credits
prior_registered_semesters
prior_fail_credits
prior_fail_ratio
prior_interruption_count
part_semester
semester_course_count
semester_registered_credits
term_gpa
end_agpa
term_gpa_delta
cumulative_agpa_delta
passed_credits
all_courses_passed
any_course_failed
```

حقول `start_*` و`prior_*` فقط تدخل في البحث. حقول النتائج تبقى منفصلة ولا تدخل في متجه KNN.

### 7.3 جدول `advisor_student_semester_course_outcome`

صف واحد لكل `(student_id, degree_id, part_id, course_id)`:

```text
student_id
degree_id
part_id
course_id
course_credits
attempt_number_at_start
final_mark
grade_id
finish_status
passed
```

### 7.4 returning students

فلترة أولية حسب:

- نفس `degree_id`، أو degree family معتمدة فقط عند نقص الدعم.
- مستوى دراسي قريب.
- نوع الفصل.
- نطاق ساعات منجزة قريب.
- حالة returning نفسها.

ثم distance حسب معلومات ما قبل التسجيل:

- `start_agpa` و`previous_term_gpa`.
- الساعات المنجزة وعدد الفصول السابقة.
- تاريخ الرسوب والانقطاع.
- الحمل الدراسي المعتاد.

### 7.5 cold-start students

لا يستخدم median fill لإيهام وجود GPA جامعي سابق. يبنى مسار منفصل يعتمد على:

- نفس الدرجة ومستوى البداية.
- `diploma_gpa` و`diploma_type_bucket`.
- فصل ونظام القبول إن توفر.

وتحمل الاستجابة `cold_start_knn=true` حتى لا تختلط بنتائج returning.

### 7.6 تشابه الخطة

بعد استرجاع الطلاب المشابهين، يعاد ترتيبهم حسب:

```text
jaccard_similarity =
    |proposed_courses intersect historical_courses|
    /
    |proposed_courses union historical_courses|
```

يمكن اختبار أوزان للساعات أو للمقررات الإجبارية لاحقًا، لكن لا يعتمد وزن قبل backtest.

### 7.7 مخرجات KNN لكل خطة

- support وعدد الحالات المستخدمة.
- توزيع distances.
- متوسط/وسيط plan similarity.
- علامات المقررات المشتركة.
- معدل الفصل التاريخي.
- `term_gpa_delta` و`cumulative_agpa_delta`.
- نسبة الطلاب الذين تحسنوا.
- نسبة الرسوب في مقرر واحد على الأقل.
- نسبة النجاح في جميع المقررات.
- confidence مبني على support والتشابه.

### 7.8 حدود الدعم

- لا يستنتج اتجاه من حالتين أو ثلاث.
- `minimum_support` و`minimum_similarity` يسجلان في configuration ويختبران تاريخيًا.
- الدعم الضعيف يعيد `insufficient_evidence`، ولا يحول NaN إلى bonus أو penalty.
- تختبر حساسية النتيجة لقيم K المختلفة قبل اعتماد K النهائي.

### 7.9 فجوات التنفيذ الحالي

يحتاج `src/knn_advisor.py` و`src/recommendation.py` لاحقًا إلى:

- hard filters للدرجة والحالة.
- فصل cold-start عن returning.
- حساب GPA قبل/بعد وdelta.
- plan-specific evidence يؤثر فعلًا في ترتيب كل خطة بدل bonus عام ثابت للطالب.
- مقياس تشابه symmetric بدل شرط overlap الأحادي الحالي.
- تعريف نجاح رسمي موحد بدل `final_mark >= 50`.
- artifact metadata واضحة للـmedians والـcutoff والنسخة.

---

## 8. محرك الأهلية

الأهلية شرط سابق للتنبؤ، وليست وزنًا داخل الترتيب.

### 8.1 القواعد

- المقرر ضمن درجة الطالب أو معادل معتمد.
- المقرر مطروح في الفصل المستهدف.
- المتطلبات السابقة ناجحة وفق التعريف الرسمي.
- المتطلبات المتزامنة موجودة في الخطة عند الحاجة.
- المقرر غير ناجح سابقًا، إلا إذا كانت الإعادة مسموحة.
- المقرر غير مسجل حاليًا بما يمنع التكرار.
- لا يخالف المستوى أو الحالة الأكاديمية.
- الخطة ضمن الحد الرسمي للساعات وعدد المقررات.
- متطلبات التخرج والمجموعات الإجبارية محفوظة.

### 8.2 الناتج

```text
course_id
eligible
reason_codes[]
missing_reference_data[]
rule_version
```

المعلومة الناقصة لا تتحول تلقائيًا إلى قبول. السياسة تحدد هل توقف التوصية أو تعرض المادة كـ`needs_advisor_review`.

---

## 9. التنبؤ وتوليد وترتيب الخطط

### 9.1 Model Adapter

```text
predict_courses(snapshot, eligible_courses, target_part)
predict_plan(snapshot, plan, target_part)
```

الناتج الموحد:

```text
course_id
predicted_mark
pass_probability
model_version
feature_contract
fallback_flags
```

لا يعرف Plan Generator إن كان النموذج LightGBM أو XGBoost.

### 9.2 مرحلتا التنبؤ

1. تنبؤ أولي بالمقررات لتقليص فضاء البحث.
2. تنبؤ نهائي لكل خطة مع الحمل الحقيقي وعدد المقررات وميزات concurrent الصحيحة.

لا يعتمد تنبؤ مقرر منفرد كتنبؤ نهائي بعد دخوله ضمن خطة كاملة.

### 9.3 توليد الخطط

- يبدأ من eligible offerings فقط.
- يستخدم الساعات الرسمية، وأي fallback يحمل flag واضحًا.
- يمنع التكرارات.
- يحترم minimum وmaximum load.
- يسمح بعدد مقررات واقعي وفق بيانات الجامعة.
- يضمن تقدمًا دراسيًا معقولًا.
- يمنع هيمنة الخطط الصغيرة لمجرد انخفاض خطرها.
- يكون deterministic تحت seed/config محددة.

### 9.4 ترتيب الخطط

الترتيب المنطقي الأولي:

1. صلاحية أكاديمية كاملة.
2. `expected_cumulative_delta` موجب أو أعلى ما يمكن تحقيقه.
3. `expected_term_gpa` أعلى من `previous_term_gpa`.
4. خطر الرسوب ضمن المستوى المقبول.
5. تقدم نحو التخرج.
6. دليل KNN خاص بالخطة ودعمه كافٍ.
7. عبء متوازن.

لا تثبت أوزان composite نهائية قبل backtest. تحفظ الأوزان في configuration ويقارن أكثر من ترتيب، ويمكن استخدام Pareto frontier لمنع وزن واحد من إخفاء خطة أفضل بوضوح.

---

## 10. Set-level Backtesting

هذه بوابة المنتج الأساسية.

### 10.1 المحاكاة

لكل طالب وفصل تاريخي:

1. تقطع البيانات عند بداية الفصل.
2. تبني snapshot من الماضي فقط.
3. تسترجع المقررات التي كانت مطروحة ومؤهلة وقتها.
4. تولد الخطط كما لو أن القرار يتخذ وقتها.
5. تمنع KNN وإحصاءات المقررات من رؤية الفصل نفسه أو المستقبل.
6. تقارن التوصيات بالنتائج اللاحقة المتاحة.

### 10.2 المقاييس

- نسبة الخطط الخالية من مخالفات الأهلية.
- نسبة الطلبات التي أنتجت خطة.
- Top-K course coverage.
- expected مقابل actual term GPA عند وجود مقارنة عادلة.
- expected مقابل actual cumulative AGPA delta.
- fail-risk calibration على مستوى المقرر والخطة.
- نسبة الخطط الموصى بها ذات uplift موجب فعليًا ضمن الحالات القابلة للمقارنة.
- التقدم الدراسي والساعات المجتازة.
- نتائج منفصلة لـcold-start وreturning والدرجات والمستويات.
- KNN support coverage وجودة الإشارة.
- latency وحجم فضاء الخطط.

### 10.3 Baselines

- اختيار أعلى pass probability فقط.
- اختيار أعلى predicted mark فقط.
- اختيار أسرع تقدم نحو التخرج فقط.
- النظام من دون KNN.
- النظام مع KNN.

لا يعتمد KNN إلا إذا حسن مقاييس الخطة أو حسن التفسير والثقة دون زيادة أخطاء القرار.

---

## 11. مراحل التنفيذ وبوابات القبول

### المرحلة 0 — Safety Freeze

**المهام**

- تسجيل dataset/model/contract paths وhashes.
- تحديد النموذج المرجعي V0 داخل manifest.
- تشغيل الاختبارات الحالية.
- إنشاء Golden Prediction Fixtures لحالات cold-start وreturning وfallback.
- تسجيل مخرجات recommendation الحالية.

**القبول**

- يمكن إعادة إنتاج المقاييس والتنبؤات الحالية.
- لم يتغير أي parquet أو model artifact.
- توجد نقطة رجوع واضحة.

### المرحلة 1 — عقد البيانات والـAPI

**المهام**

- مراجعة الحقول المتاحة من الواجهة الرسمية.
- الاتفاق على الحقول الديناميكية والجداول المرجعية.
- تعريف JSON schemas وversioning وerror codes.
- إعداد fixtures بلا معلومات شخصية.
- تحديد authority لكل قيمة، خصوصًا GPA والساعات والحالة.

**القبول**

- يبنى Student Snapshot كامل من payload تجريبي.
- لكل حقل مصدر وdtype وسياسة missing واضحة.

### المرحلة 2 — التنظيف الآمن: الموجة الأولى

**المهام**

- manifest لكل `src/` و`scripts/` وnotebooks والتقارير.
- فحص imports والاستدعاءات والاختبارات المرتبطة.
- حذف generated caches القابلة لإعادة البناء فقط.
- نقل التجارب المغلقة إلى archive مع hashes وREADME.
- عدم حذف أي data/model/report evidence.

**القبول**

- الاختبارات وGolden Predictions مطابقة.
- current pipeline entry points تعمل.
- لا import أو رابط مكسور.

### المرحلة 3 — Official GPA Engine

**المهام**

- grade-scale loader مؤرخ.
- term GPA وcumulative AGPA.
- قواعد الإعادة والحالات الخاصة بعد اعتمادها رسميًا.
- expected deltas وmarginal uplift.
- unit tests لحالات معتمدة يدويًا.

**القبول**

- تطابق كامل مع عينات الجامعة.
- لا يبقى الحساب التقريبي في مسار الإنتاج.

### المرحلة 4 — Eligibility and Candidate Engine

**المهام**

- reference repositories للمقررات والمتطلبات والعروض.
- تنفيذ القواعد مع reason codes.
- policy versioning.
- قوائم eligible وrejected وneeds-review.

**القبول**

- صفر مخالفة معروفة في fixtures المعتمدة.
- كل قرار قابل للتفسير.

### المرحلة 5 — Model Adapter and Serve-time Parity

**المهام**

- فصل تحميل النموذج عن Recommender.
- توحيد LightGBM output contract.
- تثبيت categorical levels وimputation من artifact.
- اختبار course-level وplan-level inference.
- إبقاء XGBoost adapter مؤجلًا أو disabled.

**القبول**

- نتائج LightGBM تطابق Golden Predictions.
- contract mismatch يفشل قبل التنبؤ برسالة واضحة.

### المرحلة 6 — Historical Outcome Store and KNN v2

**المهام**

- بناء الجدولين التاريخيين.
- فصل pre-registration state عن outcomes.
- index منفصل لـreturning وcold-start.
- hard filters وتشابه الخطة.
- support/confidence وGPA deltas.
- عدم كشف هوية الجيران.

**القبول**

- اختبارات عدم التسرب تمر.
- كل نتيجة تحمل cutoff/version/support.
- KNN يختلف بين خطط الطالب المختلفة عندما تختلف موادها.

### المرحلة 7 — Plan Generator and Ranker v2

**المهام**

- توليد خطط من eligible offerings.
- استخدام real credits.
- إعادة التنبؤ بسياق الخطة الكامل.
- دمج AGPA uplift وrisk وprogress وKNN evidence.
- منع small-plan bias.
- إرجاع خيار محسن للمعدل، وخيار منخفض المخاطر، وخيار متوازن عند الإمكان.

**القبول**

- كل خطة صالحة أكاديميًا.
- الحساب deterministic.
- تفسير ترتيب كل خطة ممكن من الحقول المعادة.

### المرحلة 8 — Set-level Backtest

**المهام**

- simulator زمني.
- مقارنة النظام مع baselines.
- قياس KNN on/off.
- تحليل cold-start وreturning والدرجات والمستويات.
- تحديد thresholds والأوزان من VALID/backtest وفق الحوكمة.

**القبول**

- تقرير قرار واضح لكل مكون.
- لا استخدام للمستقبل.
- لا ترقية بلا تحسن قابل للدفاع عنه.

### المرحلة 9 — Recommendation Service API

**المهام**

- endpoint وvalidation.
- تحميل artifacts مرة واحدة عند بدء الخدمة.
- request tracing وtimeouts وstructured errors.
- version metadata في كل response.
- fallback آمن عند نقص الجداول أو ضعف KNN.

**القبول**

- contract tests مع الجهة الرسمية.
- integration tests end-to-end.
- لا تعاد خطة غير مؤهلة عند خطأ جزئي.

### المرحلة 10 — Explanation and Monitoring

**المهام**

- شرح rule-based أولًا.
- LLM output-only اختياريًا.
- مراقبة drift وcoverage وcalibration وlatency وfallbacks وKNN support.
- audit trail للنسخ والقرارات.

**القبول**

- الشرح لا يغير القرار.
- كل recommendation قابلة لإعادة الإنتاج.

### المرحلة 11 — Model Upgrade and Final Cleanup

**المهام**

- مقارنة LightGBM وXGBoost بعدة seeds.
- اختبار M1 وM2 مستقلين.
- تقييم `concurrent_43` على مستوى الخطة.
- إعادة backtest لكل ترقية.
- الموجة الثانية من التنظيف بعد استقرار المسارات.

**القبول**

- لا ترقية بسبب metric منفرد صغير.
- الترقية تحسن المنتج أو المعايرة/الاستقرار دون كسر segment مهم.
- الحذف النهائي يحتاج موافقة صريحة منفصلة.

---

## 12. خطة التنظيف دون ضرر

### 12.1 تصنيف الملفات

كل ملف يصنف إلى واحدة فقط:

1. `ACTIVE_PIPELINE`: جزء من المسار الحالي.
2. `ACTIVE_LIBRARY`: كود reusable مستورد حاليًا.
3. `SAFETY_TOOL`: parity أو validation أو audit ضروري.
4. `EXPERIMENT_CANDIDATE`: تجربة قد تستخدم لاحقًا، مثل XGBoost.
5. `HISTORICAL_EVIDENCE`: يعيد إنتاج قرار سابق.
6. `ARCHIVE_CANDIDATE`: one-off انتهت ولا caller حالي لها.
7. `GENERATED`: cache أو artifact قابل لإعادة البناء.
8. `MANUAL_REVIEW`: الغرض غير محسوم ولا ينقل تلقائيًا.

### 12.2 مرشحو الأرشفة المعروفون

بعد فحص imports والاختبارات والمراجع:

- `scripts/diploma_gpa_diagnostic.py`.
- `scripts/Diagnose concurrent group v2.py`.
- مجموعة `scripts/course_identity_*` المغلقة.
- مجموعة `scripts/phase0_*`, `phase1_name_key_layer.py`, و`phase2_*` المغلقة.
- مجموعة `scripts/phase3_predecessor_prior_pilot_*`.
- `scripts/migrate_legacy_baseline.py`.
- مولدات تقارير التجارب المنتهية بعد ربطها بتقاريرها النهائية.
- `src/merge_diploma.py` بعد فترة compatibility وفحص عدم وجود caller.

هذه قائمة مرشحين وليست أمر حذف.

### 12.3 ملفات نشطة أو محمية

- rebuild v2 producers ومساعدوها.
- `scripts/build_model_population_v2.py`.
- أدوات audit/parity إلى أن تستبدل باختبارات أحدث.
- `scripts/train_xgboost_baseline.py` و`evaluate_xgboost_run.py` كتجربة مرشحة.
- كل ما داخل `data/`.
- كل model run وmetrics/report provenance.
- `Decisions_Log.md` ولا يعاد كتابة تاريخه.

### 12.4 Generated cleanup

بعد التأكد من أنها ليست tracked أو مستخدمة:

- project-level `__pycache__`.
- project-level `*.egg-info`.
- الملفات المؤقتة ذات الأسماء المشوهة في الجذر بعد فحص محتواها.

لا تنظف caches داخل `.venv` ملفًا ملفًا؛ البيئة إما تبقى أو يعاد بناؤها كاملة في مهمة منفصلة.

### 12.5 بروتوكول النقل والحذف

1. inventory يتضمن path وsize وhash وstatus وreason وcallers وoutputs.
2. dependency scan وtest-reference scan.
3. archive manifest.
4. نقل دفعة صغيرة فقط.
5. تحديث المراجع التي يجب أن تبقى صالحة.
6. تشغيل unit/integration tests وGolden Predictions.
7. مقارنة مخرجات pipeline المرجعية.
8. إبقاء الملفات في archive خلال مرحلة استقرار واحدة على الأقل.
9. الحذف الدائم يحتاج قائمة صريحة وموافقة المالك.

أي اختلاف غير مفسر يوقف دفعة التنظيف.

---

## 13. استراتيجية الاختبارات

### Unit tests

- grade scale وAGPA math.
- repeat-course policies.
- prerequisites/corequisites.
- eligibility reason codes.
- plan constraints.
- KNN filters وdistance وplan similarity وsupport.
- cold-start routing.

### Contract tests

- API request/response schemas.
- ID dtypes.
- version metadata.
- missing-field policies.
- model feature order وcategorical levels.

### Integration tests

- payload رسمي تجريبي إلى recommendation كاملة.
- returning وcold-start.
- مقرر غير مطروح أو prerequisite ناقص.
- weak KNN support.
- unknown course/difficulty fallback.

### Regression tests

- Golden Predictions للنموذج المثبت.
- hashes للبيانات والـartifacts.
- ترتيب deterministic لخطط fixtures.
- عدم تغير النتائج بعد التنظيف.

### Temporal tests

- لا صف بعد cutoff.
- KNN لا يرى outcome للفصل المستهدف.
- course statistics تستخدم الماضي فقط.
- reference table version صحيحة زمنيًا.

---

## 14. تعريف اكتمال V1

تعتبر V1 مكتملة عندما:

- يستقبل النظام payload متفقًا عليه من الواجهة الرسمية.
- يبني حالة الطالب دون notebook يدوي.
- يعيد مواد مؤهلة ومطروحة فقط.
- يستخدم grade scale وساعات رسمية.
- يحسب معدل الفصل والتراكمي المتوقع بصورة صحيحة.
- يعرض الفرق عن الفصل السابق والفرق التراكمي الحالي.
- يولد عدة خطط صالحة ويبين trade-offs بينها.
- يعرض KNN evidence خاصًا بكل خطة مع support/confidence.
- يفصل cold-start عن returning.
- يمر set-level backtest بلا تسرب.
- يسجل versions لكل نموذج وقاعدة وجدول مستخدم.
- يعيد إنتاج recommendation من audit metadata.
- لا يعتمد LLM لاتخاذ قرار أكاديمي.

---

## 15. أول حزمة تنفيذ

لا تبدأ بكتابة KNN v2 أو API server مباشرة. الحزمة الأولى هي:

1. Safety Freeze manifest.
2. Golden Prediction fixtures.
3. API field inventory يرسل للجهة الرسمية.
4. الحصول على grade-scale وقواعد الإعادة والـprerequisites الرسمية.
5. إعداد cleanup inventory فقط دون نقل في المهمة نفسها.

بعد نجاح هذه الحزمة يبدأ Official GPA Engine، لأنه تعتمد عليه أهداف رفع المعدل ونتائج KNN وترتيب الخطط.

---

## 16. المراجع الحالية

- `CLAUDE.md`: قرارات نماذج تاريخية، وبعض حالته أقدم من rebuild v2؛ لا يعتمد وحده للحالة الحالية.
- `Decisions_Log.md`: سجل قرارات append-only.
- `docs/manifests/project_pipeline_routes_2026-08.md`: خريطة البيانات والتقارير.
- `docs/manifests/codebase_map_2026-08.md`: جرد source/scripts وقت إنشائه.
- `docs/audit/script_inventory_2026_08.md`: جرد يحتاج تحديثًا قبل التنظيف.
- `src/inference.py`: التنبؤ الحالي للمقررات والخطط.
- `src/recommendation.py`: توليد وترتيب الخطط الحالي.
- `src/knn_advisor.py`: KNN الحالي المطلوب تحويله إلى plan-outcome evidence.
- `src/feature_contracts.py`: عقود الميزات.
- `src/model_training.py`: مسار LightGBM.
- `scripts/train_xgboost_baseline.py`: تجربة XGBoost المعزولة.

هذه الوثيقة هي خطة البناء المقترحة من هذه النقطة، ولا تعدل القرارات التاريخية ولا تعتمد أوزانًا أو thresholds إنتاجية قبل الـbacktest.
