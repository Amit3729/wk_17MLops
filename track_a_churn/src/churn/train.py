"""Train the candidate models and log each one as an MLflow run.

Usage:  uv run churn-train            # all candidates
        uv run churn-train --only xgb_shallow_weighted
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn import config
from churn.data import prepare_splits, train_test
from churn.features import build_preprocessor
from churn.tracking import setup_mlflow


@dataclass
class Candidate:
    name: str
    family: str
    params: dict[str, Any]
    factory: Callable[..., BaseEstimator]
    rationale: str
    extra_params: dict[str, Any] = field(default_factory=dict)  # computed at fit time (e.g. scale_pos_weight)


def candidates(pos_weight: float) -> list[Candidate]:
    """Six configurations spanning three families; hyperparameters differ in more than just the seed."""
    return [
        Candidate(
            "logreg_baseline",
            "logistic_regression",
            {"C": 1.0, "l1_ratio": 0.0, "class_weight": None, "max_iter": 2000},
            LogisticRegression,
            "Unweighted L2 baseline: shows what optimising accuracy on an imbalanced target looks like.",
        ),
        Candidate(
            "logreg_l1_balanced",
            "logistic_regression",
            {"C": 0.1, "l1_ratio": 1.0, "solver": "saga", "class_weight": "balanced", "max_iter": 5000},
            LogisticRegression,
            "Stronger L1 regularisation plus class weighting to trade precision for churner recall.",
        ),
        Candidate(
            "rf_shallow_balanced",
            "random_forest",
            {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 5, "class_weight": "balanced_subsample", "n_jobs": -1},
            RandomForestClassifier,
            "Depth-limited, weighted forest: non-linear interactions without memorising the training set.",
        ),
        Candidate(
            "rf_deep_unweighted",
            "random_forest",
            {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1, "class_weight": None, "n_jobs": -1},
            RandomForestClassifier,
            "Fully grown trees, no weighting: the overfitting end of the forest spectrum.",
        ),
        Candidate(
            "xgb_shallow_weighted",
            "xgboost",
            {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
            XGBClassifier,
            "Slow, shallow boosting with scale_pos_weight = neg/pos to counter the imbalance.",
            {"scale_pos_weight": round(pos_weight, 4), "eval_metric": "logloss", "n_jobs": -1},
        ),
        Candidate(
            "xgb_deep_unweighted",
            "xgboost",
            {"n_estimators": 600, "max_depth": 7, "learning_rate": 0.1, "subsample": 1.0, "colsample_bytree": 1.0},
            XGBClassifier,
            "Deep, fast-learning boosting with no imbalance correction: the high-variance boosting setting.",
            {"scale_pos_weight": 1.0, "eval_metric": "logloss", "n_jobs": -1},
        ),
    ]


# Types skops must be told to trust when MLflow serialises the pipeline. These models are built by
# this script, so trusting them is safe; nothing else is allowlisted.
SKOPS_TRUSTED_TYPES = {
    "logistic_regression": [],
    "random_forest": ["sklearn.tree._tree.Tree"],
    "xgboost": ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
}


def _seeded(params: dict[str, Any], estimator_cls: type) -> dict[str, Any]:
    params = dict(params)
    if "random_state" in estimator_cls().get_params():
        params["random_state"] = config.SEED
    return params


def evaluate(y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "log_loss": log_loss(y_true, proba),
    }


def _plot_artifacts(name: str, y_test: pd.Series, proba: np.ndarray, out: Path) -> None:
    pred = (proba >= 0.5).astype(int)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["No churn", "Churn"], cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix: {name}")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_title("ROC curve (hold-out)")
    fig.tight_layout()
    fig.savefig(out / "roc_curve.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    PrecisionRecallDisplay.from_predictions(y_test, proba, name=name, ax=ax)
    ax.set_title("Precision-recall curve (hold-out)")
    fig.tight_layout()
    fig.savefig(out / "pr_curve.png", dpi=120)
    plt.close(fig)


def _feature_importance(pipe: Pipeline) -> pd.DataFrame | None:
    names = pipe.named_steps["prep"].get_feature_names_out()
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return None
    return pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False)


def train_one(cand: Candidate, X_train, X_test, y_train, y_test, data_info: dict[str, Any]) -> str:
    params = _seeded({**cand.params, **cand.extra_params}, cand.factory)
    pipe = Pipeline([("prep", build_preprocessor()), ("model", cand.factory(**params))])

    with mlflow.start_run(run_name=cand.name) as run:
        mlflow.set_tags({"model_family": cand.family, "candidate": cand.name, "rationale": cand.rationale})
        mlflow.log_params({f"model__{k}": v for k, v in params.items()})
        mlflow.log_params(data_info)

        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=1)

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        proba = pipe.predict_proba(X_test)[:, 1]
        latency_ms_per_1k = (time.perf_counter() - t0) * 1000 / len(X_test) * 1000

        test = evaluate(y_test, proba)
        train = evaluate(y_train, pipe.predict_proba(X_train)[:, 1])
        mlflow.log_metrics({f"test_{k}": v for k, v in test.items()})
        mlflow.log_metrics({f"train_{k}": v for k, v in train.items() if k in ("f1", "roc_auc")})
        mlflow.log_metrics(
            {
                "cv_roc_auc_mean": cv_auc.mean(),
                "cv_roc_auc_std": cv_auc.std(),
                "cv_f1_mean": cv_f1.mean(),
                "cv_f1_std": cv_f1.std(),
                "overfit_gap_roc_auc": train["roc_auc"] - test["roc_auc"],
                "fit_seconds": fit_seconds,
                "predict_ms_per_1k_rows": latency_ms_per_1k,
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _plot_artifacts(cand.name, y_test, proba, out)
            report = classification_report(y_test, (proba >= 0.5).astype(int), target_names=["No churn", "Churn"], output_dict=True)
            (out / "classification_report.json").write_text(json.dumps(report, indent=2))
            fi = _feature_importance(pipe)
            if fi is not None:
                fi.to_csv(out / "feature_importance.csv", index=False)
            mlflow.log_artifacts(str(out), artifact_path="evaluation")

        signature = infer_signature(X_test, pipe.predict(X_test))
        model_info = mlflow.sklearn.log_model(
            sk_model=pipe,
            name="model",
            signature=signature,
            input_example=X_test.head(3),
            skops_trusted_types=SKOPS_TRUSTED_TYPES[cand.family] or None,
        )
        mlflow.set_tag("model_uri", model_info.model_uri)  # MLflow 3 logged-model URI, used by churn.register
        print(
            f"{cand.name:<22} f1={test['f1']:.3f} roc_auc={test['roc_auc']:.3f} "
            f"recall={test['recall']:.3f} acc={test['accuracy']:.3f}  run_id={run.info.run_id}"
        )
        return run.info.run_id


def main(argv: list[str] | None = None) -> None:
    train_all(argv)


def train_all(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="train only these candidate names")
    args = parser.parse_args(argv)

    setup_mlflow(config.EXPERIMENT_TRAINING)
    splits = prepare_splits()
    X_train, X_test, y_train, y_test = train_test(splits["reference"])
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    data_info = {
        "data_rows_train": len(X_train),
        "data_rows_test": len(X_test),
        "data_churn_rate_train": round(float(y_train.mean()), 4),
        "data_sha256": config.RAW_CSV_SHA256[:12],
        "data_split_seed": config.SEED,
    }

    run_ids = []
    for cand in candidates(pos_weight):
        if args.only and cand.name not in args.only:
            continue
        run_ids.append(train_one(cand, X_train, X_test, y_train, y_test, data_info))
    return run_ids


if __name__ == "__main__":
    main()
