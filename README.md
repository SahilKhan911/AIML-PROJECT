# Project 2 - Milestone 1

Intelligent Learning Analytics System (classical ML only).

## What this app does
- Upload student performance CSV data.
- **Exploratory Data Analysis (EDA)** tab to visualize numeric distributions, correlation heatmaps, and missing values before running ML.
- Preprocess data (missing value handling + scaling + encoding).
- Run boosted supervised ML with hyperparameter tuning and cross-validation (Logistic/RandomForest/GradientBoosting for classification, Linear/RandomForest/GradientBoosting for regression) or K-Means clustering.
- **SHAP-based model explainability** showing global feature importance and granular per-student explanations.
- Classify learners into `At-risk`, `Average`, `High-performing`.
- Generate **personalized, feature-aware study recommendations** that directly reference the student's weakest signals by name (e.g., low grades, high absences).
- Run **reproducible benchmarks** natively in the web app via the specific Benchmark tab across datasets and objectives.
- Show evaluation metrics and downloadable results.

## Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Reproducible benchmark
You can run benchmarks directly from the **Benchmark tab** in the web app, which allows dynamic loading of sample datasets or custom CSVs to evaluate models across `Pass/Fail`, `3-Level Risk`, and `Regression` objectives. 

Alternatively, run the automated benchmark script:
```bash
source .venv/bin/activate
python scripts_benchmark.py
```
This script evaluates models and produces comprehensive outputs:
- Console performance metrics
- A detailed `reports/benchmark_results.csv`
- A styled `reports/benchmark_report.html`

## Recommended datasets
- https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
- https://archive.ics.uci.edu/ml/datasets/student+performance

## Local datasets included
- `data/raw/uci_student/student-mat.csv`
- `data/raw/uci_student/student-por.csv`

Notes:
- UCI files are semicolon-delimited; the app now auto-detects delimiters.
- Kaggle dataset requires Kaggle authentication/API key, so it is not auto-downloaded in this setup.

## Milestone 1 deliverable docs
- [docs/milestone1_report.md](docs/milestone1_report.md)
- [docs/use_case.md](docs/use_case.md)
- [docs/deployment.md](docs/deployment.md)

## Deployable website support
- Streamlit app entrypoint: `app.py`
- Streamlit config: `.streamlit/config.toml`
- Render config: `render.yaml`
