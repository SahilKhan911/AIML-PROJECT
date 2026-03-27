# Use Case - Early Academic Risk Detection

## Problem
Faculty teams need to identify struggling students early, but manual spreadsheet analysis is slow and inconsistent.

## User
- Academic coordinator
- Mentor / class advisor

## Input
- Student performance CSV containing scores, attendance, and study-behavior fields.

## Workflow
1. **Upload** the latest class dataset.
2. **Explore Data (EDA Tab)** to visualize correlation heatmaps and identify anomalies or missing values.
3. **Run supervised analytics** (`G3` pass/fail recommended for screening) in the Web App tab.
4. **Review Context (SHAP)** to understand specific positive/negative learning drivers influencing each student's outcome.
5. **Compare Models (Benchmark Tab)** if wanting to evaluate and rank algorithm baseline performances.
6. **Export** the recommendation report and assign personalized feature-aware interventions based dynamically on low scores or heavy absences.

## Output
- Exploratory data visualization and target correlation breakdowns.
- Student-level risk classification coupled with granular SHAP accountability.
- Feature-aware study recommendations per student directly extracting weak spots.
- Downloadable CSV report for mentoring workflow.

## Success Criteria
- Fast triage of at-risk learners leveraging pre-computed data distributions.
- Highly actionable recommendation generation (moving beyond generics).
- Intuitive explainability of *why* specific learners struggle.
- Measurable week-over-week progress tracking.
