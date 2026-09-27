"""MLflow connection helpers."""

from __future__ import annotations

import mlflow

from churn import config


def setup_mlflow(experiment: str) -> str:
    """Point MLflow at the project's SQLite store and create the experiment with a local artifact root."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        config.MLFLOW_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        exp_id = mlflow.create_experiment(experiment, artifact_location=(config.MLFLOW_ARTIFACT_ROOT / experiment).as_uri())
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment)
    return exp_id
