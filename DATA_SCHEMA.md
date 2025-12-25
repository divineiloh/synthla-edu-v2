# SYNTHLA-EDU V2 Data Schemas

This document describes the data schemas for OULAD and ASSISTments datasets, including all feature engineering steps.

---

## OULAD Dataset Schema

### Source Tables
The OULAD dataset is built from 7 CSV files that are merged and aggregated to student-level:

1. **studentInfo.csv** - Student demographics and outcomes
2. **studentAssessment.csv** - Assessment submissions and scores
3. **studentVle.csv** - Virtual Learning Environment interactions
4. **vle.csv** - VLE resource metadata
5. **assessments.csv** - Assessment metadata
6. **courses.csv** - Course metadata
7. **studentRegistration.csv** - Student registration dates

### Final Student-Level Features (27 columns)

#### Demographics & Identifiers
- **`code_module`** (categorical) - Course module code (e.g., AAA, BBB)
- **`code_presentation`** (categorical) - Course presentation (e.g., 2013J, 2014B)
- **`id_student`** (int) - Unique student identifier
- **`gender`** (categorical) - Student gender: M/F
- **`region`** (categorical) - Geographic region (UK regions)
- **`highest_education`** (categorical) - Highest education level: HE Qualification, A Level, Lower Than A Level, Post Graduate Qualification
- **`imd_band`** (categorical) - Index of Multiple Deprivation band: 0-10%, 10-20%, ..., 90-100%
- **`age_band`** (categorical) - Age group: 0-35, 35-55, 55<=
- **`num_of_prev_attempts`** (int) - Number of previous course attempts: 0, 1, 2, 3+
- **`studied_credits`** (int) - Total credits being studied

#### Registration Features (Engineered)
- **`date_registration`** (int) - Days from course start to registration (negative = early, positive = late)
- **`date_unregistration`** (float) - Days from course start to unregistration (NaN if completed)
- **`registration_timing`** (categorical) - Derived: early, on_time, late

#### VLE Activity Features (Aggregated from studentVle.csv)
- **`total_clicks`** (int) - Total VLE interactions
- **`unique_dates_active`** (int) - Number of unique days with activity
- **`avg_clicks_per_day`** (float) - Mean clicks per active day
- **`activity_span_days`** (int) - Days between first and last interaction
- **`clicks_per_resource_type`** (dict/encoded) - Clicks by resource type (homepage, oucontent, resource, etc.)

#### Assessment Features (Aggregated from studentAssessment.csv)
- **`num_assessments_submitted`** (int) - Total assessments submitted
- **`avg_assessment_score`** (float) - Mean score across all assessments
- **`weighted_avg_score`** (float) - Weighted by assessment weight
- **`assessments_on_time`** (int) - Count of on-time submissions
- **`assessments_late`** (int) - Count of late submissions
- **`earliest_submission_day`** (int) - First submission relative to course start
- **`latest_submission_day`** (int) - Last submission relative to course start

#### Target Variables
- **`final_result`** (categorical) - Outcome: Pass, Fail, Withdrawn, Distinction
- **`dropout`** (binary) - Classification target: 0 (Pass/Distinction), 1 (Fail/Withdrawn)
- **`final_grade`** (float) - Regression target: weighted average assessment score (0-100)

### Feature Engineering Steps

1. **Merge** studentInfo with studentRegistration (on id_student, code_module, code_presentation)
2. **Aggregate VLE clicks** by student (sum, mean, count unique dates)
3. **Aggregate assessments** by student (mean score, count submissions, timeliness)
4. **Weight assessments** by their importance (from assessments.csv)
5. **Create binary dropout** target: Fail/Withdrawn = 1, Pass/Distinction = 0
6. **Handle missing values**:
   - `date_unregistration`: NaN for completers
   - VLE features: 0 for students with no activity
   - Assessment features: NaN for non-submitters (imputed with median)

### Data Statistics
- **Rows**: 32,593 students
- **Train/Test Split**: 70/30 stratified by dropout
- **Class Balance**: ~32% dropout rate
- **Missing Data**: <5% in most columns

---

## ASSISTments Dataset Schema

### Source Table
Single CSV file: **assistments_2009_2010.csv** (interaction-level data)

### Aggregation Strategy
**Interaction → Student-Level**: Group by `user_id` and aggregate interaction metrics.

### Final Student-Level Features (15-20 columns)

#### Student Identifiers
- **`user_id`** (int) - Unique student identifier
- **`problem_log_id`** (int, dropped after aggregation) - Individual problem attempt ID

#### Interaction Metadata
- **`skill_id`** (int) - Problem skill/concept ID
- **`problem_id`** (int) - Specific problem ID
- **`assistment_id`** (int) - ASSISTment system ID
- **`order_id`** (int) - Sequence order of problem

#### Performance Features (Original)
- **`correct`** (binary) - Whether first attempt was correct: 0/1
- **`original`** (binary) - Whether this is an original problem (vs. scaffolding): 0/1
- **`attempt_count`** (int) - Number of attempts on this problem

#### Temporal Features
- **`ms_first_response`** (int) - Milliseconds to first response
- **`start_time`** (timestamp) - Problem start timestamp
- **`end_time`** (timestamp) - Problem completion timestamp

#### Hint/Help Features
- **`hint_count`** (int) - Number of hints requested
- **`hint_total`** (int) - Total hints available
- **`scaffold_count`** (int) - Number of scaffolding problems

#### Student-Level Aggregated Features (Post-Aggregation)
- **`total_problems_attempted`** (int) - Count of problems attempted
- **`num_correct_first_attempt`** (int) - Count of correct first attempts
- **`student_pct_correct`** (float) - **Fraction correct (0.0-1.0, not percentage)** - stored as decimal, not 0-100 scale
- **`avg_attempt_count`** (float) - Mean attempts per problem
- **`total_hints_used`** (int) - Sum of all hints requested
- **`avg_hints_per_problem`** (float) - Mean hints per problem
- **`avg_response_time_ms`** (float) - Mean milliseconds to first response
- **`total_active_time_ms`** (int) - Sum of (end_time - start_time)
- **`num_original_problems`** (int) - Count of original problems (not scaffolding)
- **`num_scaffold_problems`** (int) - Count of scaffolding problems

#### Target Variables
- **`high_accuracy`** (binary) - Classification target: 1 if student_pct_correct >= 0.5 (50%), else 0
- **`student_pct_correct`** (float) - **Regression target: fraction correct (0.0-1.0 scale, NOT 0-100 percentage)**

### Feature Engineering Steps

1. **Parse timestamps** (start_time, end_time) to datetime
2. **Group by user_id** and aggregate:
   - Count total problems
   - Sum correct answers
   - Calculate percentage correct
   - Average attempts, hints, response times
3. **Create derived features**:
   - `student_pct_correct` = (num_correct / total_problems) — **stored as fraction 0.0-1.0, NOT percentage 0-100**
   - `avg_response_time_ms` = mean of ms_first_response
4. **Handle missing values**:
   - `ms_first_response`: Fill with median
   - `hint_count`: Fill with 0 (no hints requested)
5. **Filter students**: Minimum 10 problems attempted (quality threshold)

### Data Statistics
- **Original Rows**: ~500,000 interactions
- **Aggregated Rows**: ~4,000-5,000 students
- **Train/Test Split**: 70/30 with group-aware splitting (by user_id)
- **Target Distribution**: Student accuracy ranges 0-100%

---

## Cross-Dataset Consistency

### Shared Properties
Both datasets are aggregated to **student-level** with:
- **Binary classification** target (dropout/correctness)
- **Continuous regression** target (grade/percentage)
- **Stratified splitting** on classification target
- **No data leakage** (test students never seen in training)

### Target Variable Mapping

| Dataset | Classification Target | Regression Target |
|---------|----------------------|-------------------|
| OULAD | `dropout` (0/1) | `final_grade` (0-100) |
| ASSISTments | `high_accuracy` (binary) | `student_pct_correct` (0-100) |

### Feature Categories

| Category | OULAD | ASSISTments |
|----------|-------|-------------|
| Demographics | ✓ (gender, age, education) | ✗ (not available) |
| Activity Metrics | ✓ (VLE clicks, dates) | ✓ (problems attempted, time) |
| Performance | ✓ (assessment scores) | ✓ (correctness, attempts) |
| Help-Seeking | ✗ | ✓ (hints, scaffolds) |
| Temporal | ✓ (registration, submission timing) | ✓ (response times) |

---

## Usage in SYNTHLA-EDU V2

### Data Loading
```python
from synthla_edu_v2 import build_dataset

# Load OULAD (student-level table)
df_oulad, schema_oulad = build_dataset("oulad", "data/raw")

# Load ASSISTments (aggregated to student-level)
df_assist, schema_assist = build_dataset("assistments", "data/raw")
```

### Schema Dictionary Structure
```python
schema = {
    "id_cols": ["id_student"],  # or ["user_id"]
    "target_cols": ["dropout", "final_grade"],  # or ["high_accuracy", "student_pct_correct"]
    "categorical_cols": ["gender", "region", ...],
    "numeric_cols": ["total_clicks", "avg_assessment_score", ...],
    "group_col": "id_student"  # For group-aware splitting
}
```

---

## Data Quality Notes

### OULAD
- **Missing registrations**: Some students lack registration dates (filled with 0)
- **Withdrawn students**: May have partial VLE/assessment data
- **Multiple attempts**: Students retaking courses have duplicate entries (different presentations)

### ASSISTments
- **Incomplete attempts**: Some problems have no end_time (filtered out)
- **Outlier response times**: Some exceed 1 hour (capped at 99th percentile)
- **Low-activity students**: Students with <10 problems excluded from analysis

---

## References

1. **OULAD**: Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data*, 4, 170171.
2. **ASSISTments**: Feng, M., Heffernan, N., & Koedinger, K. (2009). Addressing the assessment challenge with an online system that tutors as it assesses. *User Modeling and User-Adapted Interaction*, 19(3), 243-266.
