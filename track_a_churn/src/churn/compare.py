"""Export the side-by-side comparison of every training run from MLflow to results/.

Writes results/runs_comparison.csv, results/runs_comparison.md and results/runs_comparison.png.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from churn import config
from churn.tracking import setup_mlflow

METRICS = [
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_roc_auc",
    "test_pr_auc",
    "cv_roc_auc_mean",
    "cv_f1_mean",
    "overfit_gap_roc_auc",
    "fit_seconds",
    "predict_ms_per_1k_rows",
]
PARAMS = [
    "model__C",
    "model__l1_ratio",
    "model__class_weight",
    "model__n_estimators",
    "model__max_depth",
    "model__learning_rate",
    "model__scale_pos_weight",
]


def runs_table() -> pd.DataFrame:
    exp_id = setup_mlflow(config.EXPERIMENT_TRAINING)
    runs = mlflow.search_runs([exp_id], filter_string="attributes.status = 'FINISHED'")
    runs = runs.sort_values(f"metrics.{config.SELECTION_METRIC}", ascending=False)
    table = pd.DataFrame({"run": runs["tags.mlflow.runName"], "family": runs["tags.model_family"]})
    for p in PARAMS:
        col = f"params.{p}"
        table[p.removeprefix("model__")] = runs[col].fillna("-") if col in runs else "-"
    for m in METRICS:
        table[m] = runs[f"metrics.{m}"].round(4)
    table["run_id"] = runs["run_id"].str[:8]
    return table.reset_index(drop=True)


def _markdown(table: pd.DataFrame) -> str:
    cols = ["run", "family", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc", "test_pr_auc", "cv_roc_auc_mean", "overfit_gap_roc_auc", "fit_seconds", "predict_ms_per_1k_rows"]
    header = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    rows = []
    best = {c: table[c].max() for c in ("test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc", "test_pr_auc", "cv_roc_auc_mean")}
    for _, r in table.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            text = f"{v:.3f}" if isinstance(v, float) else str(v)
            cells.append(f"**{text}**" if c in best and v == best[c] else text)
        rows.append("| " + " | ".join(cells) + " |")
    return header + "\n".join(rows) + "\n"


def _plot(table: pd.DataFrame) -> None:
    metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.8 / len(table)
    for i, (_, row) in enumerate(table.iterrows()):
        xs = [j + i * width for j in range(len(metrics))]
        ax.bar(xs, [row[m] for m in metrics], width=width, label=row["run"])
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(metrics))], [m.removeprefix("test_") for m in metrics])
    ax.set_ylim(0.4, 0.9)
    ax.set_ylabel("hold-out score")
    ax.set_title("Telco churn: all MLflow training runs, hold-out metrics")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "runs_comparison.png", dpi=130)
    plt.close(fig)


def main() -> None:
    export()


def export() -> pd.DataFrame:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = runs_table()
    table.to_csv(config.RESULTS_DIR / "runs_comparison.csv", index=False)
    (config.RESULTS_DIR / "runs_comparison.md").write_text(_markdown(table))
    _plot(table)
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    main()
