from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ml_pipeline import train_supervised


def run() -> None:
    datasets = [
        Path("data/raw/uci_student/student-mat.csv"),
        Path("data/raw/uci_student/student-por.csv"),
    ]

    print("Benchmarking boosted supervised pipeline...\n")
    for path in datasets:
        df = pd.read_csv(path, sep=";")
        print(f"=== {path} ===")

        pass_fail = train_supervised(
            df,
            target_column="G3",
            objective="pass_fail",
            tune_hyperparams=True,
            cv_folds=3,
            pass_threshold=10.0,
        )
        print(
            f"G3 pass/fail -> model={pass_fail.model_name}, "
            f"acc={pass_fail.metrics['accuracy']:.4f}, "
            f"f1={pass_fail.metrics['f1_weighted']:.4f}"
        )

        risk3 = train_supervised(
            df,
            target_column="G3",
            objective="risk_3level",
            tune_hyperparams=True,
            cv_folds=3,
        )
        print(
            f"G3 3-level risk -> model={risk3.model_name}, "
            f"acc={risk3.metrics['accuracy']:.4f}, "
            f"f1={risk3.metrics['f1_weighted']:.4f}"
        )
        print()


if __name__ == "__main__":
    run()
