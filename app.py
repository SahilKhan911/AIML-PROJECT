from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from src.ml_pipeline import cluster_students, generate_recommendation, train_supervised


st.set_page_config(page_title="Project 2 - Milestone 1", layout="wide")
st.title("Intelligent Learning Analytics System")
st.caption("Project 2 | Milestone 1: ML-Based Learning Analytics")

st.markdown(
    """
A deployable learning analytics website that analyzes student performance,
flags risk categories, and generates study recommendations.
"""
)

DATA_DIR = Path(__file__).parent / "data" / "raw" / "uci_student"


def read_csv_auto(source: str | io.BytesIO | io.StringIO | Path) -> pd.DataFrame:
    # Try common separators so UCI semicolon CSVs load correctly.
    for sep in [None, ";", ",", "\t", "|"]:
        try:
            if sep is None:
                frame = pd.read_csv(source, sep=None, engine="python")
            else:
                frame = pd.read_csv(source, sep=sep)
            if frame.shape[1] > 1:
                return frame
        except Exception:
            continue
    return pd.read_csv(source)


def render_use_case() -> None:
    st.subheader("Use Case: Early Academic Risk Detection")
    st.markdown(
        """
**Scenario**
A college analytics coordinator receives weekly student performance exports and needs to quickly identify who is at risk before mid-sem exams.

**Actor**
Academic coordinator / class mentor.

**Input**
CSV with assessment scores, study behavior, and attendance indicators.

**System Action**
1. Preprocesses numeric + categorical fields.
2. Trains/evaluates ML models.
3. Classifies students (`At-risk`, `Average`, `High-performing`) or predicts pass/fail.
4. Produces per-student recommendations.

**Output**
Prioritized intervention list + downloadable recommendation report for mentors.
"""
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Primary Goal", "Early intervention")
    with c2:
        st.metric("Target Users", "Mentors / faculty")
    with c3:
        st.metric("Decision Speed", "Minutes, not days")

    with st.expander("Expected classroom workflow"):
        st.markdown(
            """
- Upload latest class performance CSV.
- Run `G3` pass/fail objective for quick screening.
- Filter predicted `At-risk` students.
- Share recommendations with mentors and track improvement next week.
"""
        )


def render_analytics() -> None:
    st.subheader("Analytics Dashboard")

    sample_choice = st.selectbox(
        "Load sample data or upload your own",
        options=[
            "None (I will upload)",
            "UCI - student-mat.csv",
            "UCI - student-por.csv",
        ],
    )
    uploaded = st.file_uploader("Upload student performance CSV", type=["csv"])

    df: pd.DataFrame | None = None
    if sample_choice == "UCI - student-mat.csv":
        sample_path = DATA_DIR / "student-mat.csv"
        if sample_path.exists():
            df = read_csv_auto(sample_path)
        else:
            st.warning("Bundled sample not found. Please upload a CSV.")
    elif sample_choice == "UCI - student-por.csv":
        sample_path = DATA_DIR / "student-por.csv"
        if sample_path.exists():
            df = read_csv_auto(sample_path)
        else:
            st.warning("Bundled sample not found. Please upload a CSV.")

    if df is None and uploaded:
        try:
            df = read_csv_auto(uploaded)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read CSV: {exc}")
            return

    if df is None:
        st.info("Upload a CSV or load a sample to run analytics.")
        return

    if df.empty:
        st.warning("The selected dataset is empty.")
        return

    st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    st.dataframe(df.head(20), use_container_width=True)

    with st.expander("Column summary"):
        st.write(df.dtypes.astype(str).rename("dtype").to_frame())

    mode = st.radio(
        "Choose analysis mode",
        options=["Supervised prediction + classification", "Clustering (no target column)"],
        horizontal=True,
    )

    if mode == "Supervised prediction + classification":
        default_idx = 0
        for key in ["G3", "final_score", "overall_score", "exam_score", "risk_level", "performance"]:
            if key in df.columns:
                default_idx = list(df.columns).index(key)
                break

        target_col = st.selectbox("Select target column", options=list(df.columns), index=default_idx)
        is_numeric_target = pd.api.types.is_numeric_dtype(df[target_col])

        objective_options = ["Auto (best by target type)"]
        if is_numeric_target:
            objective_options += [
                "Regression (continuous score)",
                "Classification: Pass/Fail from target",
                "Classification: 3-level risk from target",
            ]
        else:
            objective_options += ["Classification (original labels)"]

        objective_choice = st.selectbox("Prediction objective", objective_options)

        pass_threshold = 10.0
        if objective_choice == "Classification: Pass/Fail from target":
            pass_threshold = st.number_input(
                "Pass threshold (target >= threshold => Pass)",
                min_value=float(df[target_col].min()),
                max_value=float(df[target_col].max()),
                value=10.0,
                step=0.5,
            )

        tune_hyperparams = st.checkbox(
            "Enable hyperparameter tuning + CV model selection",
            value=True,
            help="Tests Logistic/RandomForest/GradientBoosting and keeps the best model.",
        )
        cv_folds = st.slider("CV folds", min_value=3, max_value=7, value=3)

        if st.button("Run supervised analytics", type="primary"):
            objective_map = {
                "Auto (best by target type)": "auto",
                "Regression (continuous score)": "regression",
                "Classification: Pass/Fail from target": "pass_fail",
                "Classification: 3-level risk from target": "risk_3level",
                "Classification (original labels)": "classification",
            }
            objective = objective_map[objective_choice]

            try:
                artifacts = train_supervised(
                    df,
                    target_col,
                    objective=objective,
                    tune_hyperparams=tune_hyperparams,
                    cv_folds=cv_folds,
                    pass_threshold=pass_threshold,
                )
                out = artifacts.output_frame.copy()
                out["study_recommendation"] = out.apply(generate_recommendation, axis=1)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Analytics failed: {exc}")
                return

            include_true_label = st.checkbox(
                "Include original target column in output table",
                value=False,
                help="Keep this off to avoid exposing ground-truth labels in shared outputs.",
            )
            if not include_true_label and target_col in out.columns:
                out = out.drop(columns=[target_col])

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Model type", artifacts.mode)
            with c2:
                st.metric("Target", artifacts.target_column)
            with c3:
                st.metric("Best model", artifacts.model_name)

            st.subheader("Model Metrics")
            st.json(artifacts.metrics)

            if artifacts.mode == "regression" and "predicted_risk_level" in out:
                st.subheader("Predicted Risk Distribution")
                st.bar_chart(out["predicted_risk_level"].value_counts())

            if artifacts.mode == "classification" and "predicted_category" in out:
                st.subheader("Predicted Category Distribution")
                st.bar_chart(out["predicted_category"].value_counts())
                if artifacts.confusion_matrix_df is not None:
                    st.subheader("Confusion Matrix (test set)")
                    st.dataframe(artifacts.confusion_matrix_df, use_container_width=True)
                    st.bar_chart(artifacts.confusion_matrix_df)

            st.subheader("Predictions + Recommendations")
            st.dataframe(out, use_container_width=True)

            buffer = io.StringIO()
            out.to_csv(buffer, index=False)
            st.download_button(
                label="Download results CSV",
                data=buffer.getvalue(),
                file_name="learning_analytics_results.csv",
                mime="text/csv",
            )

    else:
        k = st.slider("Number of clusters", min_value=2, max_value=6, value=3)

        if st.button("Run clustering analytics", type="primary"):
            try:
                out, metrics = cluster_students(df, n_clusters=k)
                out["study_recommendation"] = out.apply(generate_recommendation, axis=1)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Clustering failed: {exc}")
                return

            st.subheader("Cluster Quality")
            st.json(metrics)

            st.subheader("Cluster Risk Distribution")
            st.bar_chart(out["cluster_risk_level"].value_counts())

            st.subheader("Clustered Learners + Recommendations")
            st.dataframe(out, use_container_width=True)

            buffer = io.StringIO()
            out.to_csv(buffer, index=False)
            st.download_button(
                label="Download clustered results CSV",
                data=buffer.getvalue(),
                file_name="learning_analytics_clustered_results.csv",
                mime="text/csv",
            )


use_case_tab, analytics_tab = st.tabs(["Use Case", "Web App"])

with use_case_tab:
    render_use_case()

with analytics_tab:
    render_analytics()

st.markdown("---")
st.caption(
    "Milestone 1 scope: classical ML only (no LLMs, no agentic workflows). "
    "Deployable on Streamlit Community Cloud, Hugging Face Spaces, or Render."
)
