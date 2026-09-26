"""Training/serving tests against a throwaway MLflow store, so they never touch the project's mlflow.db."""

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def isolated(tmp_path_factory):
    root = tmp_path_factory.mktemp("mlops")
    mp = pytest.MonkeyPatch()
    from churn import config

    mp.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{root / 'mlflow.db'}")
    mp.setattr(config, "MLFLOW_ARTIFACT_ROOT", root / "mlartifacts")
    mp.setattr(config, "RESULTS_DIR", root / "results")
    yield config
    mp.undo()


def test_train_register_serve(isolated):
    from churn import register, train

    run_ids = train.train_all(["--only", "logreg_baseline", "logreg_l1_balanced"])
    assert len(run_ids) == 2

    import mlflow

    runs = mlflow.search_runs(experiment_names=[isolated.EXPERIMENT_TRAINING])
    for metric in ("test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"):
        assert runs[f"metrics.{metric}"].notna().all()
    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(run_ids[0], "evaluation")}
    assert {"evaluation/confusion_matrix.png", "evaluation/roc_curve.png"} <= artifacts

    version = register.register_best()
    mv = mlflow.MlflowClient().get_model_version(isolated.REGISTERED_MODEL_NAME, version)
    assert mv.current_stage == "Production"

    serve = importlib.reload(importlib.import_module("churn.serve"))
    client = TestClient(serve.app)
    assert client.get("/health").json()["stage"] == "Production"
    example = serve.Customer.model_config["json_schema_extra"]["example"]
    body = client.post("/predict", json=example).json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert client.post("/predict", json={**example, "Contract": "Weekly"}).status_code == 422
