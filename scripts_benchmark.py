from __future__ import annotations

import platform
import sys
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from src.ml_pipeline import train_supervised


# ---------------------------------------------------------------------------
# Helpers — CSV + HTML report generation
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "dataset",
    "objective",
    "model_name",
    "accuracy",
    "f1_weighted",
    "cv_best_accuracy",
    "mae",
    "r2",
    "cv_best_r2",
]


def _row_dict(
    dataset: str,
    objective: str,
    model_name: str,
    metrics: dict,
) -> dict:
    """Build a flat dict for one benchmark run."""
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


def _write_csv(rows: list[dict], path: Path) -> None:
    df = pd.DataFrame(rows, columns=_CSV_COLUMNS)
    df.to_csv(path, index=False)


def _fmt(val: object, higher_is_better: bool = True) -> str:
    """Format a metric value for display."""
    if val == "" or val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{float(val):.4f}"


def _build_html(rows: list[dict], timestamp: str) -> str:
    """Return a self-contained HTML report string."""

    # --- Identify best / worst per metric column -------------------------
    metric_cols = ["accuracy", "f1_weighted", "cv_best_accuracy", "mae", "r2", "cv_best_r2"]
    higher_is_better = {
        "accuracy": True,
        "f1_weighted": True,
        "cv_best_accuracy": True,
        "mae": False,
        "r2": True,
        "cv_best_r2": True,
    }

    # Collect numeric values per metric for best/worst detection
    col_values: dict[str, list[tuple[int, float]]] = {m: [] for m in metric_cols}
    for idx, row in enumerate(rows):
        for m in metric_cols:
            v = row.get(m, "")
            if v != "" and v is not None:
                try:
                    col_values[m].append((idx, float(v)))
                except (TypeError, ValueError):
                    pass

    best_cells: set[tuple[int, str]] = set()
    worst_cells: set[tuple[int, str]] = set()

    for m in metric_cols:
        vals = col_values[m]
        if len(vals) < 2:
            continue
        hib = higher_is_better[m]
        best_idx = max(vals, key=lambda t: t[1] if hib else -t[1])[0]
        worst_idx = min(vals, key=lambda t: t[1] if hib else -t[1])[0]
        if best_idx != worst_idx:
            best_cells.add((best_idx, m))
            worst_cells.add((worst_idx, m))

    # --- Build table rows ------------------------------------------------
    table_rows = ""
    for idx, row in enumerate(rows):
        cells = f"<td>{escape(str(row['dataset']))}</td>"
        cells += f"<td>{escape(str(row['objective']))}</td>"
        cells += f"<td>{escape(str(row['model_name']))}</td>"
        for m in metric_cols:
            val = _fmt(row.get(m, ""))
            if (idx, m) in best_cells:
                style = ' style="background:#d4edda;font-weight:600"'
            elif (idx, m) in worst_cells:
                style = ' style="background:#f8d7da"'
            else:
                style = ""
            cells += f"<td{style}>{val}</td>"
        table_rows += f"<tr>{cells}</tr>\n"

    # --- Model selection summaries ---------------------------------------
    summaries = _model_selection_summaries(rows)
    summary_html = "\n".join(f"<p>{escape(s)}</p>" for s in summaries)

    # --- Assemble --------------------------------------------------------
    py_ver = platform.python_version()
    sk_ver = sklearn.__version__

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Learning Analytics — Benchmark Report</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 2rem; color: #222; background: #fafafa;
  }}
  h1 {{ color: #1a1a2e; }}
  h2 {{ color: #16213e; margin-top: 2rem; }}
  table {{
    border-collapse: collapse; width: 100%; margin-top: 1rem;
    font-size: 0.92rem;
  }}
  th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: center; }}
  th {{ background: #1a1a2e; color: #fff; }}
  tr:nth-child(even) {{ background: #f4f4f8; }}
  .footer {{
    margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #ddd;
    font-size: 0.82rem; color: #888;
  }}
</style>
</head>
<body>
<h1>Learning Analytics — Benchmark Report</h1>
<p><strong>Run timestamp:</strong> {escape(timestamp)}</p>

<h2>Results Table</h2>
<table>
<thead>
<tr>
  <th>Dataset</th><th>Objective</th><th>Model</th>
  <th>Accuracy</th><th>F1 (weighted)</th><th>CV Accuracy</th>
  <th>MAE</th><th>R²</th><th>CV R²</th>
</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>

<h2>Model Selection Summary</h2>
{summary_html}

<div class="footer">
  Python {escape(py_ver)} &middot; scikit-learn {escape(sk_ver)}
</div>
</body>
</html>"""


def _model_selection_summaries(rows: list[dict]) -> list[str]:
    """Generate plain-English summaries grouped by dataset + objective."""
    from itertools import groupby

    summaries: list[str] = []

    # Group by (dataset, objective)
    key_fn = lambda r: (r["dataset"], r["objective"])
    sorted_rows = sorted(rows, key=key_fn)

    for (dataset, objective), group in groupby(sorted_rows, key=key_fn):
        group_list = list(group)
        if len(group_list) < 1:
            continue

        entry = group_list[0]
        model = entry["model_name"]
        ds_name = Path(dataset).name

        if objective == "regression":
            r2 = entry.get("r2", "")
            mae = entry.get("mae", "")
            if r2 != "" and mae != "":
                summaries.append(
                    f"For {ds_name} regression, {model} achieved "
                    f"R²={float(r2):.4f} with MAE={float(mae):.4f}."
                )
            else:
                summaries.append(
                    f"For {ds_name} regression, {model} was selected."
                )
        else:
            acc = entry.get("accuracy", "")
            if acc != "":
                pct = float(acc) * 100
                summaries.append(
                    f"For {ds_name} {objective}, {model} won "
                    f"with {pct:.1f}% accuracy."
                )
            else:
                summaries.append(
                    f"For {ds_name} {objective}, {model} was selected."
                )

    return summaries


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run() -> None:
    datasets = [
        Path("data/raw/uci_student/student-mat.csv"),
        Path("data/raw/uci_student/student-por.csv"),
    ]

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

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
        all_rows.append(
            _row_dict(str(path), "pass_fail", pass_fail.model_name, pass_fail.metrics)
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
        all_rows.append(
            _row_dict(str(path), "risk_3level", risk3.model_name, risk3.metrics)
        )

        regression = train_supervised(
            df,
            target_column="G3",
            objective="regression",
            tune_hyperparams=True,
            cv_folds=3,
        )
        print(
            f"G3 regression  -> model={regression.model_name}, "
            f"mae={regression.metrics['mae']:.4f}, "
            f"r2={regression.metrics['r2']:.4f}"
        )
        all_rows.append(
            _row_dict(str(path), "regression", regression.model_name, regression.metrics)
        )
        print()

    # --- Write reports ---------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    csv_path = reports_dir / "benchmark_results.csv"
    _write_csv(all_rows, csv_path)
    print(f"CSV report saved to {csv_path}")

    html_path = reports_dir / "benchmark_report.html"
    html_path.write_text(_build_html(all_rows, timestamp), encoding="utf-8")
    print(f"HTML report saved to {html_path}")


if __name__ == "__main__":
    run()
