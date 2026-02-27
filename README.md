# Project 2 - Milestone 1

Intelligent Learning Analytics System (classical ML only).

## What this app does
- Upload student performance CSV data.
- Preprocess data (missing value handling + scaling + encoding).
- Run boosted supervised ML with model selection (Logistic/RandomForest/GradientBoosting for classification, Linear/RandomForest/GradientBoosting for regression) or K-Means clustering.
- Classify learners into `At-risk`, `Average`, `High-performing`.
- Generate rule-based study recommendations.
- Show evaluation metrics and downloadable results.

## Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Reproducible benchmark
```bash
source .venv/bin/activate
python scripts_benchmark.py
```
This runs boosted models on UCI student datasets for:
- `G3` pass/fail (`>= 10` pass)
- `G3` 3-level risk classification

## Recommended datasets
- https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
- https://archive.ics.uci.edu/ml/datasets/student+performance

## Local datasets included
- `/Users/sahilkhan/Desktop/AIML Project/data/raw/uci_student/student-mat.csv`
- `/Users/sahilkhan/Desktop/AIML Project/data/raw/uci_student/student-por.csv`

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
