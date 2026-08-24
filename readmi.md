# Academic Advisor Pipeline

توثيق KNN المفصل، بما فيه بناء البيانات و`fit/predict` والتقييم والربط مع
التوصيات، موجود في [docs/knn_readme/README.md](docs/knn_readme/README.md).

Clean Tables
↓
Student Snapshot تمثيل حالة الطالب قبل بداية الفصل.  (student_id + semester_id) 
↓
Candidate Courses    candidate_courses = requestable AND offered  student_id + semester_id + course_id
↓
Course Features  تمثيل خصائص المادة نفسها.  course_id 
↓
Student-Course Features تمثيل علاقة الطالب بالمادة student_id + semester_id + course_id 
↓
Course-Level Training Dataset 
↓
Model 1: Grade / Points Prediction
↓
Model 2: Pass / Risk Prediction
↓
Course Prediction Table for all the candidate in the semester 
↓
Plan Candidate Generation 
Plan Generation based on the rule System 
↓
Similar Students Layer student_id + semester_id  student_semester_snapshot  student_semester_course_options  student_semester_outcome
↓
knn soft evidence 
↓
Plan Scoring
↓
Final Recommendation
{ 
1. Current Student Snapshot
        ↓
2. Rules Engine
   - requestable courses
   - offered courses
   - prerequisites
        ↓
3. Candidate Courses
        ↓
4. Model 1: Grade / Points Prediction
        ↓
5. Model 2: Pass / Risk Prediction
        ↓
6. Plan Candidate Generation
   - generate N possible course plans
        ↓
7. KNN Similar Historical Cases
   - find similar students
   - compare their choices and outcomes
        ↓
8. Plan Scoring
   - expected AGPA Y
   - risk
   - workload
   - graduation progress
   - similar-student evidence
        ↓
9. Final Ranked Recommendations
}
