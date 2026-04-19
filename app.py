from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.ml_pipeline import (
    cluster_students,
    explain_model,
    generate_recommendation,
    train_supervised,
    _engineer_features,
)
from src.agentic_coach import AgenticStudyCoach, generate_pdf_report


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

    st.subheader("System Architecture Diagram")
    st.graphviz_chart('''
    digraph G {
        rankdir="LR";
        node [shape=box, style=filled, color=lightblue, fontname="Helvetica"];
        
        Data [label="Student Data (CSV)"]
        Prep [label="Data Preprocessing\n(Missing Values, Scaling)"]
        Feat [label="Feature Engineering\n(Risk Bucketing, Derived Metrics)"]
        Model [label="ML Pipeline\n(Regression / Classification / Clustering)"]
        Eval [label="Model Evaluation\n(Metrics, Cross-Validation)"]
        Recs [label="Generative Rule-based\nStudy Recommendations"]
        UI [label="Streamlit\nDashboard"]
        
        Data -> Prep -> Feat -> Model -> Eval -> Recs -> UI
    }
    ''')


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

def render_limitations() -> None:
    st.subheader("Model Performance & Limitations Analysis")
    st.markdown(
        """
### Analysis of Performance
- **Data Quality Dependance**: The classical ML models (Logistic Regression, Random Forest, Gradient Boosting) perform extremely well when input datasets (like the UCI Student dataset) provide consistent historical features (like `G1` and `G2`).
- **Feature Significance**: Regular attendance, prior grades, and study time play the most predictive roles in risk determination.
- **Clustering Insights**: K-Means successfully partitions learners into At-Risk, Average, and High-Performing without need for explicit target labels, scoring solid silhouette scores on purely numeric columns.

### Key Limitations
1. **Limited Context**: The models only know what is in the CSV. Socio-emotional parameters, external life events, and detailed cognitive obstacles are completely uncaptured.
2. **Deterministic Fallbacks**: Traditional ML can flag an "at-risk" student, but the recommendations are strictly rule-based (e.g., "Review mistakes if test scores < 50"). It lacks the semantic reasoning of an LLM.
3. **Imbalanced Classes**: In typical educational settings, severe failures are the minority class. Even with balanced class weights, predicting the edge cases (extreme risk) remains challenging.
4. **Generalization Constraints**: Models trained on one demographic or institution's grading system often experience drift when applied to a different system without re-training.
"""
    )

def _load_dataframe() -> pd.DataFrame | None:
    """Shared data-loading logic used by both the EDA and Web App tabs.

    Stores the loaded DataFrame in ``st.session_state['shared_df']`` so
    that a CSV loaded in one tab is automatically available in the other.
    """
    sample_choice = st.selectbox(
        "Load sample data or upload your own",
        options=[
            "None (I will upload)",
            "UCI - student-mat.csv",
            "UCI - student-por.csv",
        ],
        key=f"sample_select_{st.session_state.get('_active_tab', 'default')}",
    )
    uploaded = st.file_uploader(
        "Upload student performance CSV",
        type=["csv"],
        key=f"file_upload_{st.session_state.get('_active_tab', 'default')}",
    )

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
            return None

    if df is not None:
        st.session_state["shared_df"] = df
    elif "shared_df" in st.session_state:
        df = st.session_state["shared_df"]

    return df


# ---------------------------------------------------------------------------
# EDA Tab
# ---------------------------------------------------------------------------

def render_eda() -> None:
    """Exploratory Data Analysis tab."""
    st.subheader("Exploratory Data Analysis")

    st.session_state["_active_tab"] = "eda"
    df = _load_dataframe()

    if df is None:
        st.info("Upload a CSV or load a sample to explore the data.")
        return
    if df.empty:
        st.warning("The selected dataset is empty.")
        return

    st.write(f"**Rows:** {len(df)}  |  **Columns:** {len(df.columns)}")

    # ---- 1. Numeric column distributions --------------------------------
    st.markdown("---")
    st.markdown("### 📊 Numeric Column Distributions")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns found in this dataset.")
    else:
        default_selection = numeric_cols[:6]
        selected_cols = st.multiselect(
            "Select numeric columns to visualise (max 6)",
            options=numeric_cols,
            default=default_selection,
            max_selections=6,
        )
        if selected_cols:
            for col_name in selected_cols:
                counts = df[col_name].dropna().value_counts().sort_index()
                st.markdown(f"**{col_name}**")
                st.bar_chart(counts)
        else:
            st.info("Pick at least one column above to see its distribution.")

    # ---- 2. Correlation heatmap -----------------------------------------
    st.markdown("---")
    st.markdown("### 🔗 Correlation Heatmap")

    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns to compute correlations.")
    else:
        corr = df[numeric_cols].corr(numeric_only=True)

        def _corr_color(val: float) -> str:
            """Return a CSS background colour that goes red ➜ white ➜ green."""
            v = np.clip(val, -1, 1)
            # -1 → red (220,60,60), 0 → white (255,255,255), +1 → green (60,179,113)
            if v < 0:
                t = v + 1  # 0..1
                r = int(220 + (255 - 220) * t)
                g = int(60 + (255 - 60) * t)
                b = int(60 + (255 - 60) * t)
            else:
                t = v  # 0..1
                r = int(255 + (60 - 255) * t)
                g = int(255 + (179 - 255) * t)
                b = int(255 + (113 - 255) * t)
            return f"background-color: rgb({r},{g},{b}); color: {'#fff' if abs(v) > 0.7 else '#000'}"

        styled_corr = corr.style.map(_corr_color).format("{:.2f}")
        st.dataframe(styled_corr, use_container_width=True)

    # ---- 3. Target column breakdown -------------------------------------
    st.markdown("---")
    st.markdown("### 🎯 Target Column Breakdown")

    target_col = st.selectbox(
        "Pick a target column (optional)",
        options=["— none —"] + list(df.columns),
        index=0,
        key="eda_target_col",
    )

    if target_col != "— none —":
        st.markdown("**Value counts**")
        value_counts = df[target_col].value_counts()
        st.bar_chart(value_counts)

        if numeric_cols:
            st.markdown("**Mean of numeric columns grouped by target**")
            # Exclude the target itself from the numeric column list if present
            group_cols = [c for c in numeric_cols if c != target_col]
            if group_cols:
                grouped = df.groupby(target_col)[group_cols].mean(numeric_only=True)
                st.dataframe(grouped, use_container_width=True)
            else:
                st.info("No other numeric columns to group by.")
    else:
        st.info("Select a target column above to see its breakdown.")

    # ---- 4. Missing value summary ---------------------------------------
    st.markdown("---")
    st.markdown("### ⚠️ Missing Value Summary")

    null_counts = df.isnull().sum()
    missing = null_counts[null_counts > 0]
    if missing.empty:
        st.success("No missing values found in the dataset! 🎉")
    else:
        missing_df = pd.DataFrame(
            {
                "Column": missing.index,
                "Missing Count": missing.values,
                "Missing %": (missing.values / len(df) * 100).round(2),
            }
        )
        missing_df = missing_df.sort_values("Missing Count", ascending=False).reset_index(drop=True)
        st.dataframe(missing_df, use_container_width=True)


# ---------------------------------------------------------------------------
# Web App Tab (Analytics)
# ---------------------------------------------------------------------------

def _clear_supervised_state() -> None:
    """Remove cached supervised-mode results from session state."""
    for key in ("artifacts", "supervised_out", "analytics_target_col",
                "analytics_objective", "shap_global", "shap_student"):
        st.session_state.pop(key, None)


def _clear_cluster_state() -> None:
    """Remove cached clustering-mode results from session state."""
    for key in ("cluster_out", "cluster_metrics"):
        st.session_state.pop(key, None)


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

    # -- Clear stale results when switching modes -------------------------
    if mode == "Supervised prediction + classification":
        _clear_cluster_state()
    else:
        _clear_supervised_state()

    # =====================================================================
    # Supervised prediction + classification
    # =====================================================================
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

        # -- Staleness guard: clear cached results if settings changed ----
        if (
            st.session_state.get("analytics_target_col") != target_col
            or st.session_state.get("analytics_objective") != objective_choice
        ):
            _clear_supervised_state()

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

        # -- Button: train model & cache results --------------------------
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
                _clear_supervised_state()
                return

            # Persist in session state
            st.session_state["artifacts"] = artifacts
            st.session_state["supervised_out"] = out
            st.session_state["analytics_target_col"] = target_col
            st.session_state["analytics_objective"] = objective_choice

            # Pre-compute SHAP so changing the student row is instant
            try:
                work_df = df.copy().dropna(how="all").drop_duplicates()
                x_for_shap = work_df.drop(columns=[target_col])
                x_for_shap = _engineer_features(x_for_shap)
                global_imp, student_shap = explain_model(
                    artifacts, x_for_shap, max_display=10
                )
                st.session_state["shap_global"] = global_imp
                st.session_state["shap_student"] = student_shap
            except Exception:  # noqa: BLE001
                st.session_state.pop("shap_global", None)
                st.session_state.pop("shap_student", None)

        # -- Render persisted results (survives widget reruns) ------------
        artifacts = st.session_state.get("artifacts")
        out = st.session_state.get("supervised_out")

        if artifacts is not None and out is not None:
            cached_target = st.session_state.get("analytics_target_col", "")

            include_true_label = st.checkbox(
                "Include original target column in output table",
                value=False,
                help="Keep this off to avoid exposing ground-truth labels in shared outputs.",
            )
            display_out = out.copy()
            if not include_true_label and cached_target in display_out.columns:
                display_out = display_out.drop(columns=[cached_target])

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Model type", artifacts.mode)
            with c2:
                st.metric("Target", artifacts.target_column)
            with c3:
                st.metric("Best model", artifacts.model_name)

            st.subheader("Model Metrics")
            st.json(artifacts.metrics)

            if artifacts.mode == "regression" and "predicted_risk_level" in display_out:
                st.subheader("Predicted Risk Distribution")
                st.bar_chart(display_out["predicted_risk_level"].value_counts())

            if artifacts.mode == "classification" and "predicted_category" in display_out:
                st.subheader("Predicted Category Distribution")
                st.bar_chart(display_out["predicted_category"].value_counts())
                if artifacts.confusion_matrix_df is not None:
                    st.subheader("Confusion Matrix (test set)")
                    st.dataframe(artifacts.confusion_matrix_df, use_container_width=True)
                    st.bar_chart(artifacts.confusion_matrix_df)

            st.subheader("Predictions + Recommendations")
            st.dataframe(display_out, use_container_width=True)

            buffer = io.StringIO()
            display_out.to_csv(buffer, index=False)
            st.download_button(
                label="Download results CSV",
                data=buffer.getvalue(),
                file_name="learning_analytics_results.csv",
                mime="text/csv",
            )

            # ---- SHAP explainability ------------------------------------
            with st.expander("Model explainability (SHAP)"):
                try:
                    global_imp = st.session_state.get("shap_global")
                    student_shap = st.session_state.get("shap_student")

                    if global_imp is None or student_shap is None:
                        st.warning(
                            "Explainability unavailable for this model/data combination."
                        )
                    else:
                        # -- Global feature importance bar chart -----------
                        st.markdown("#### Top 10 Global Feature Importances")
                        chart_df = global_imp.set_index("feature")
                        st.bar_chart(chart_df)

                        # -- Per-student explanation -----------------------
                        st.markdown("#### Explain a Specific Student")
                        row_options = list(student_shap.index)
                        selected_row = st.selectbox(
                            "Explain a specific student (row index)",
                            options=row_options,
                            index=0,
                        )

                        student_vals = student_shap.loc[selected_row]
                        top_pos = student_vals.nlargest(5)
                        top_neg = student_vals.nsmallest(5)

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("**Top 5 positive drivers**")
                            pos_df = top_pos.reset_index()
                            pos_df.columns = ["Feature", "SHAP value"]
                            st.dataframe(pos_df, use_container_width=True)
                        with col_b:
                            st.markdown("**Top 5 negative drivers**")
                            neg_df = top_neg.reset_index()
                            neg_df.columns = ["Feature", "SHAP value"]
                            st.dataframe(neg_df, use_container_width=True)

                        # Plain-English explanation
                        pred_col = None
                        for c in ["predicted_category", "predicted_risk_level"]:
                            if c in artifacts.output_frame.columns:
                                pred_col = c
                                break
                        if pred_col is not None and selected_row in artifacts.output_frame.index:
                            prediction = artifacts.output_frame.loc[selected_row, pred_col]
                        else:
                            prediction = "this outcome"

                        pos_names = ", ".join(top_pos.head(3).index.tolist())
                        neg_names = ", ".join(top_neg.head(3).index.tolist())
                        st.info(
                            f"This student is predicted **{prediction}** mainly because of "
                            f"high influence from **{pos_names}** (positive drivers) "
                            f"and **{neg_names}** (negative drivers)."
                        )
                except Exception:  # noqa: BLE001
                    st.warning(
                        "Explainability unavailable for this model/data combination."
                    )

    # =====================================================================
    # Clustering (no target column)
    # =====================================================================
    else:
        k = st.slider("Number of clusters", min_value=2, max_value=6, value=3)

        # -- Button: run clustering & cache results -----------------------
        if st.button("Run clustering analytics", type="primary"):
            try:
                out, metrics = cluster_students(df, n_clusters=k)
                out["study_recommendation"] = out.apply(generate_recommendation, axis=1)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Clustering failed: {exc}")
                _clear_cluster_state()
                return

            st.session_state["cluster_out"] = out
            st.session_state["cluster_metrics"] = metrics

        # -- Render persisted clustering results --------------------------
        cluster_out = st.session_state.get("cluster_out")
        cluster_metrics = st.session_state.get("cluster_metrics")

        if cluster_out is not None and cluster_metrics is not None:
            st.subheader("Cluster Quality")
            st.json(cluster_metrics)

            st.subheader("Cluster Risk Distribution")
            st.bar_chart(cluster_out["cluster_risk_level"].value_counts())

            st.subheader("Clustered Learners + Recommendations")
            st.dataframe(cluster_out, use_container_width=True)

            buffer = io.StringIO()
            cluster_out.to_csv(buffer, index=False)
            st.download_button(
                label="Download clustered results CSV",
                data=buffer.getvalue(),
                file_name="learning_analytics_clustered_results.csv",
                mime="text/csv",
            )

def render_agentic_coach() -> None:
    st.subheader("Agentic AI Study Coach")
    st.markdown("Diagnose learning gaps and generate a personalized study plan using LLMs and live web search.")
    
    provider = st.radio("Select API Provider", ["Google Gemini", "Groq"], horizontal=True)
    api_key = st.text_input(f"Enter your {provider} API Key", type="password")
    
    student_record = st.text_area("Student Record / Performance Notes", 
                                  "Example: Student scores: G1=9, G2=10, absences=14, studytime=2. Struggles with math fundamentals.")
    student_goal = st.text_input("Student Goal", "Pass the upcoming midterms with at least 60%.")
    
    if st.button("Generate Study Plan", type="primary"):
        if not api_key:
            st.error(f"Please enter a valid {provider} API key.")
            return
            
        with st.spinner("Agent is reasoning, searching, and generating plan..."):
            try:
                coach = AgenticStudyCoach(api_key=api_key, provider=provider)
                markdown_plan = coach.run(student_record=student_record, student_goal=student_goal)
                
                st.success("Plan generated successfully!")
                
                st.markdown("### Personalized AI Study Plan")
                st.markdown(markdown_plan)
                
                pdf_bytes = generate_pdf_report(markdown_plan)
                st.download_button(
                    label="📥 Download Study Plan as PDF",
                    data=pdf_bytes,
                    file_name="AI_Study_Plan.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# ---------------------------------------------------------------------------
# Benchmark Tab
# ---------------------------------------------------------------------------

_BENCHMARK_OBJECTIVES = [
    ("pass_fail", "Pass/Fail", {"pass_threshold": 10.0}),
    ("risk_3level", "3-Level Risk", {}),
    ("regression", "Regression", {}),
]

_BENCHMARK_DATASETS = [
    DATA_DIR / "student-mat.csv",
    DATA_DIR / "student-por.csv",
]


def _benchmark_row(
    dataset: str,
    objective: str,
    model_name: str,
    metrics: dict,
) -> dict:
    return {
        "dataset": dataset,
        "objective": objective,
        "model_name": model_name,
        "accuracy": metrics.get("accuracy", ""),
        "f1_weighted": metrics.get("f1_weighted", ""),
        "cv_best_accuracy": metrics.get("cv_best_accuracy", ""),
        "mae": metrics.get("mae", ""),
        "r2": metrics.get("r2", ""),
        "cv_best_r2": metrics.get("cv_best_r2", ""),
    }



def _benchmark_summaries(rows: list[dict]) -> list[str]:
    summaries: list[str] = []
    for row in rows:
        ds = Path(row["dataset"]).name
        obj = row["objective"]
        model = row["model_name"]
        if obj == "regression":
            r2 = row.get("r2", "")
            mae = row.get("mae", "")
            if r2 != "" and mae != "":
                summaries.append(
                    f"{ds} | {obj.replace('_', ' ').title()} → "
                    f"{model} achieved R²={float(r2):.4f}, MAE={float(mae):.4f}"
                )
            else:
                summaries.append(f"{ds} | {obj} → {model}")
        else:
            acc = row.get("accuracy", "")
            if acc != "":
                pct = float(acc) * 100
                label = "Pass/Fail" if "pass" in obj else "3-Level Risk"
                summaries.append(
                    f"{ds} | {label} → {model} won with {pct:.1f}% accuracy"
                )
            else:
                summaries.append(f"{ds} | {obj} → {model}")
    return summaries


def render_benchmark() -> None:
    st.subheader("Reproducible Benchmark")
    st.markdown(
        "Run the reproducible benchmark on the bundled UCI student datasets "
        "(math + Portuguese). Tests all 3 objectives: **Pass/Fail**, "
        "**3-Level Risk**, and **Regression**."
    )

    sample_choice = st.selectbox(
        "Load sample data or upload your own",
        options=[
            "None (I will upload)",
            "UCI - student-mat.csv",
            "UCI - student-por.csv",
        ],
        key="benchmark_sample",
    )
    uploaded = st.file_uploader("Upload student performance CSV", type=["csv"], key="benchmark_uploader")

    df: pd.DataFrame | None = None
    ds_name = ""
    if sample_choice == "UCI - student-mat.csv":
        sample_path = DATA_DIR / "student-mat.csv"
        if sample_path.exists():
            df = read_csv_auto(sample_path)
            ds_name = sample_path.name
        else:
            st.warning("Bundled sample not found. Please upload a CSV.")
    elif sample_choice == "UCI - student-por.csv":
        sample_path = DATA_DIR / "student-por.csv"
        if sample_path.exists():
            df = read_csv_auto(sample_path)
            ds_name = sample_path.name
        else:
            st.warning("Bundled sample not found. Please upload a CSV.")

    if df is None and uploaded:
        try:
            df = read_csv_auto(uploaded)
            ds_name = uploaded.name
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read CSV: {exc}")
            return

    if df is None:
        st.info("Upload a CSV or load a sample to run benchmark.")
        return

    if df.empty:
        st.warning("The selected dataset is empty.")
        return

    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    default_idx = 0
    if "G3" in df.columns:
        default_idx = list(df.columns).index("G3")
    target_col = st.selectbox(
        "Select target column",
        options=list(df.columns),
        index=default_idx,
        key="benchmark_target",
    )

    # -- Buttons -----------------------------------------------------------
    col_run, col_clear = st.columns([1, 1])
    run_clicked = col_run.button("Run Benchmark", type="primary")
    clear_clicked = col_clear.button("Clear & re-run")

    if clear_clicked:
        st.session_state.pop("benchmark_results", None)
        run_clicked = True  # treat clear as immediate re-run

    # -- Execute benchmark ------------------------------------------------
    if run_clicked:
        try:
            total = len(_BENCHMARK_OBJECTIVES)
            progress = st.progress(0, text="Starting benchmark…")
            rows: list[dict] = []
            step = 0

            for obj_key, obj_label, extra_kwargs in _BENCHMARK_OBJECTIVES:
                step += 1
                progress.progress(
                    step / total,
                    text=f"[{step}/{total}] {ds_name} — {obj_label}…",
                )
                arts = train_supervised(
                    df,
                    target_column=target_col,
                    objective=obj_key,
                    tune_hyperparams=True,
                    cv_folds=3,
                    **extra_kwargs,
                )
                rows.append(
                    _benchmark_row(
                        ds_name, obj_key, arts.model_name, arts.metrics
                    )
                )

            progress.progress(1.0, text="Benchmark complete ✅")
            st.session_state["benchmark_results"] = rows

        except Exception as exc:  # noqa: BLE001
            st.error(f"Benchmark failed: {exc}")
            return

    # -- Display persisted results ----------------------------------------
    results = st.session_state.get("benchmark_results")
    if results is None:
        st.info("Click **Run Benchmark** to start.")
        return

    results_df = pd.DataFrame(results)

    st.subheader("Results")
    st.dataframe(results_df, use_container_width=True)

    # -- Plain-English summaries ------------------------------------------
    st.subheader("Summary")
    for line in _benchmark_summaries(results):
        st.markdown(f"- {line}")

    # -- CSV download -----------------------------------------------------
    csv_buf = io.StringIO()
    results_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="Download benchmark CSV",
        data=csv_buf.getvalue(),
        file_name="benchmark_results.csv",
        mime="text/csv",
    )


use_case_tab, eda_tab, model_analysis_tab, analytics_tab, benchmark_tab, agentic_tab = st.tabs(
    ["Use Case & Architecture", "EDA", "Model Analysis & Limitations", "Classical ML App", "Benchmark", "Agentic Study Coach"]
)

with use_case_tab:
    render_use_case()

with eda_tab:
    render_eda()

with model_analysis_tab:
    render_limitations()

with analytics_tab:
    render_analytics()

with agentic_tab:
    render_agentic_coach()

with benchmark_tab:
    render_benchmark()

st.markdown("---")
st.caption(
    "Milestone 1 & 2 scope: Classical ML UI + LangGraph/LLM Agentic workflows. "
    "Deployable on Streamlit Community Cloud, Hugging Face Spaces, or Render."
)
