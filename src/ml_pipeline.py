from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RISK_LABELS = ["At-risk", "Average", "High-performing"]


@dataclass
class ModelArtifacts:
    mode: str
    target_column: str
    pipeline: Pipeline
    metrics: Dict[str, float]
    output_frame: pd.DataFrame
    model_name: str
    confusion_matrix_df: pd.DataFrame | None = None


def _feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def _preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def _to_risk_bucket(values: np.ndarray) -> np.ndarray:
    q1, q2 = np.quantile(values, [0.33, 0.66])

    def mapper(v: float) -> str:
        if v <= q1:
            return "At-risk"
        if v <= q2:
            return "Average"
        return "High-performing"

    return np.array([mapper(v) for v in values])


def _engineer_features(x: pd.DataFrame) -> pd.DataFrame:
    work = x.copy()

    if {"G1", "G2"}.issubset(work.columns):
        work["avg_prior_grade"] = (work["G1"] + work["G2"]) / 2
        work["grade_delta_g2_g1"] = work["G2"] - work["G1"]

    if {"studytime", "failures"}.issubset(work.columns):
        work["study_efficiency"] = work["studytime"] / (work["failures"] + 1)

    if "absences" in work.columns and pd.api.types.is_numeric_dtype(work["absences"]):
        q99 = work["absences"].quantile(0.99)
        work["absences_capped"] = work["absences"].clip(upper=q99)

    numeric_cols = work.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) >= 2:
        work["numeric_mean_all"] = work[numeric_cols].mean(axis=1)
        work["numeric_std_all"] = work[numeric_cols].std(axis=1).fillna(0.0)

    return work


def _derive_class_target(
    y: pd.Series,
    objective: str,
    pass_threshold: float = 10.0,
) -> pd.Series:
    if objective == "classification":
        return y.astype(str)

    numeric_y = pd.to_numeric(y, errors="coerce")
    if numeric_y.isna().all():
        raise ValueError("Selected target is not numeric, cannot derive class labels.")

    if objective == "pass_fail":
        return np.where(numeric_y >= pass_threshold, "Pass", "At-risk")

    if objective == "risk_3level":
        q1, q2 = np.quantile(numeric_y.dropna(), [0.33, 0.66])

        def mapper(v: float) -> str:
            if v <= q1:
                return "At-risk"
            if v <= q2:
                return "Average"
            return "High-performing"

        return pd.Series([mapper(v) for v in numeric_y], index=y.index)

    raise ValueError(f"Unsupported objective: {objective}")


def _safe_stratify(y: pd.Series) -> pd.Series | None:
    counts = pd.Series(y).value_counts()
    if (counts < 2).any() or len(counts) < 2:
        return None
    return y


def _fit_best_classifier(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    tune: bool,
    cv_folds: int,
    random_state: int,
    x_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> Tuple[Pipeline, str, float | None]:
    candidates = [
        (
            "LogisticRegression",
            LogisticRegression(max_iter=3000),
            {
                "model__C": np.logspace(-2, 2, 10),
                "model__solver": ["lbfgs", "liblinear"],
                "model__class_weight": [None, "balanced"],
            },
        ),
        (
            "RandomForestClassifier",
            RandomForestClassifier(random_state=random_state, class_weight="balanced"),
            {
                "model__n_estimators": [150, 300],
                "model__max_depth": [None, 8, 12],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2],
                "model__class_weight": [None, "balanced"],
            },
        ),
        (
            "GradientBoostingClassifier",
            GradientBoostingClassifier(random_state=random_state),
            {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
                "model__subsample": [1.0],
            },
        ),
    ]

    best_pipeline: Pipeline | None = None
    best_model_name = ""
    best_cv: float | None = None
    best_score = -np.inf

    for name, model, params in candidates:
        base = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        if tune:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            search = GridSearchCV(
                estimator=base,
                param_grid=params,
                scoring="accuracy",
                cv=cv,
                n_jobs=1,
            )
            search.fit(x_train, y_train)
            fitted = search.best_estimator_
            cv_score = float(search.best_score_)
        else:
            fitted = base.fit(x_train, y_train)
            cv_score = None

        if x_val is not None and y_val is not None:
            score = float(accuracy_score(y_val, fitted.predict(x_val)))
        else:
            score = float(fitted.score(x_train, y_train))

        if score > best_score:
            best_score = score
            best_pipeline = fitted
            best_model_name = name
            best_cv = cv_score

    assert best_pipeline is not None
    return best_pipeline, best_model_name, best_cv


def _fit_best_regressor(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    tune: bool,
    cv_folds: int,
    random_state: int,
) -> Tuple[Pipeline, str, float | None]:
    candidates = [
        (
            "LinearRegression",
            LinearRegression(),
            {},
        ),
        (
            "RandomForestRegressor",
            RandomForestRegressor(random_state=random_state),
            {
                "model__n_estimators": [100, 200, 300, 500],
                "model__max_depth": [None, 4, 8, 12, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        (
            "GradientBoostingRegressor",
            GradientBoostingRegressor(random_state=random_state),
            {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.7, 0.85, 1.0],
            },
        ),
    ]

    best_pipeline: Pipeline | None = None
    best_model_name = ""
    best_cv: float | None = None
    best_score = -np.inf

    for name, model, params in candidates:
        base = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        if tune and params:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            search = RandomizedSearchCV(
                estimator=base,
                param_distributions=params,
                n_iter=12,
                scoring="r2",
                cv=cv,
                random_state=random_state,
                n_jobs=1,
            )
            search.fit(x_train, y_train)
            score = float(search.best_score_)
            fitted = search.best_estimator_
            cv_score = score
        else:
            fitted = base.fit(x_train, y_train)
            score = float(fitted.score(x_train, y_train))
            cv_score = None

        if score > best_score:
            best_score = score
            best_pipeline = fitted
            best_model_name = name
            best_cv = cv_score

    assert best_pipeline is not None
    return best_pipeline, best_model_name, best_cv


def train_supervised(
    df: pd.DataFrame,
    target_column: str,
    random_state: int = 42,
    objective: str = "auto",
    tune_hyperparams: bool = True,
    cv_folds: int = 5,
    pass_threshold: float = 10.0,
) -> ModelArtifacts:
    work_df = df.copy().dropna(how="all").drop_duplicates()

    y = work_df[target_column]
    x = work_df.drop(columns=[target_column])
    x = _engineer_features(x)

    numeric_cols, categorical_cols = _feature_columns(x)
    if not numeric_cols and not categorical_cols:
        raise ValueError("No feature columns available after removing the target column.")

    preprocess = _preprocessor(numeric_cols, categorical_cols)

    if objective == "auto":
        objective = "regression" if pd.api.types.is_numeric_dtype(y) else "classification"

    metrics: Dict[str, float] = {}

    if objective == "regression":
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=random_state
        )
        pipeline, model_name, cv_score = _fit_best_regressor(
            x_train,
            y_train,
            preprocess,
            tune_hyperparams,
            cv_folds,
            random_state,
        )
        y_pred = pipeline.predict(x_test)
        metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
        metrics["r2"] = float(r2_score(y_test, y_pred))
        if cv_score is not None:
            metrics["cv_best_r2"] = float(cv_score)

        all_preds = pipeline.predict(x)
        risk_labels = _to_risk_bucket(all_preds)

        output_frame = work_df.copy()
        output_frame["predicted_score"] = all_preds
        output_frame["predicted_risk_level"] = risk_labels

        return ModelArtifacts(
            mode="regression",
            target_column=target_column,
            pipeline=pipeline,
            metrics=metrics,
            output_frame=output_frame,
            model_name=model_name,
            confusion_matrix_df=None,
        )

    y_cls = _derive_class_target(y, objective, pass_threshold=pass_threshold)
    y_cls = pd.Series(y_cls, index=work_df.index)

    stratify = _safe_stratify(y_cls)
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x,
        y_cls,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )

    if objective == "pass_fail":
        model_name = "GradientBoostingClassifier"
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", GradientBoostingClassifier(random_state=random_state)),
            ]
        )
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_score = float(cross_val_score(pipeline, x_train_full, y_train_full, cv=cv, scoring="accuracy").mean())
        pipeline.fit(x_train_full, y_train_full)
    else:
        stratify_train = _safe_stratify(y_train_full)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full,
            y_train_full,
            test_size=0.2,
            random_state=random_state,
            stratify=stratify_train,
        )
        pipeline, model_name, cv_score = _fit_best_classifier(
            x_train,
            y_train,
            preprocess,
            tune_hyperparams,
            cv_folds,
            random_state,
            x_val=x_val,
            y_val=y_val,
        )
        pipeline.fit(x_train_full, y_train_full)

    y_pred = pipeline.predict(x_test)
    metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
    metrics["f1_weighted"] = float(f1_score(y_test, y_pred, average="weighted"))
    if cv_score is not None:
        metrics["cv_best_accuracy"] = float(cv_score)
    class_labels = sorted(pd.Series(y_cls).astype(str).unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {label}" for label in class_labels],
        columns=[f"Pred: {label}" for label in class_labels],
    )

    all_preds = pipeline.predict(x)
    output_frame = work_df.copy()
    output_frame["predicted_category"] = all_preds

    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        proba = pipeline.predict_proba(x)
        if proba.shape[1] == 2:
            positive_cls = pipeline.named_steps["model"].classes_[1]
            output_frame[f"predicted_prob_{positive_cls}"] = proba[:, 1]

    return ModelArtifacts(
        mode="classification",
        target_column=target_column,
        pipeline=pipeline,
        metrics=metrics,
        output_frame=output_frame,
        model_name=model_name,
        confusion_matrix_df=cm_df,
    )


def cluster_students(df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> Tuple[pd.DataFrame, Dict[str, float]]:
    work_df = df.copy().dropna(how="all").drop_duplicates()
    work_df = _engineer_features(work_df)
    numeric_df = work_df.select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        raise ValueError("Clustering needs at least 2 numeric feature columns.")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(numeric_df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(x_scaled)
    work_df["cluster"] = labels

    sil = silhouette_score(x_scaled, labels)
    cluster_mean = work_df.groupby("cluster")[numeric_df.columns].mean().mean(axis=1)
    sorted_clusters = cluster_mean.sort_values().index.tolist()
    mapping = {
        sorted_clusters[0]: "At-risk",
        sorted_clusters[1]: "Average",
        sorted_clusters[-1]: "High-performing",
    }
    work_df["cluster_risk_level"] = work_df["cluster"].map(mapping)

    return work_df, {"silhouette_score": float(sil)}


def generate_recommendation(row: pd.Series) -> str:
    risk_col = None
    for candidate in ["predicted_risk_level", "cluster_risk_level", "predicted_category"]:
        if candidate in row and pd.notna(row[candidate]):
            risk_col = str(row[candidate]).strip()
            break

    low_perf_signals = []
    improve_signals = []

    for c in row.index:
        lc = c.lower()
        if any(k in lc for k in ["score", "accuracy", "quiz", "test", "exam", "g1", "g2", "g3"]):
            try:
                value = float(row[c])
            except (TypeError, ValueError):
                continue
            if value < 50 and "g" not in lc:
                low_perf_signals.append(c)
            elif "g" in lc and value < 10:
                low_perf_signals.append(c)
            elif ("g" in lc and value < 14) or ("g" not in lc and value < 70):
                improve_signals.append(c)

    for c in row.index:
        lc = c.lower()
        if "time" in lc:
            try:
                value = float(row[c])
            except (TypeError, ValueError):
                continue
            if value < 2:
                improve_signals.append(c)

    if risk_col and ("at-risk" in risk_col.lower() or "fail" in risk_col.lower()):
        return (
            "High-priority intervention: focus on fundamentals, schedule daily revision blocks, "
            "and take a weekly checkpoint quiz."
        )

    if low_perf_signals:
        topics = ", ".join(low_perf_signals[:3])
        return (
            f"Target weak areas ({topics}). Use active recall + practice sets, and review mistakes within 24 hours."
        )

    if improve_signals:
        topics = ", ".join(improve_signals[:3])
        return (
            f"Performance is moderate. Improve consistency in {topics} with spaced practice 4 days/week."
        )

    return "Maintain current pace and increase challenge with mixed-topic mock tests twice per week."
