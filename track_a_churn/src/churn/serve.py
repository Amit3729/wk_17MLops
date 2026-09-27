"""FastAPI wrapper that loads the Production model from the MLflow Model Registry.

Run:   uv run churn-serve                      # http://127.0.0.1:8000/docs
       CHURN_MODEL_URI=models:/telco-churn-classifier@staging uv run churn-serve
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel, Field

from churn import config

MODEL_URI = os.environ.get("CHURN_MODEL_URI", f"models:/{config.REGISTERED_MODEL_NAME}@{config.PRODUCTION_ALIAS}")
YesNo = Literal["Yes", "No"]


class Customer(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: YesNo
    Partner: YesNo
    Dependents: YesNo
    tenure: float = Field(ge=0)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "No", "tenure": 2,
                "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
                "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.5, "TotalCharges": 191.0,
            }
        }
    }


class Prediction(BaseModel):
    churn_probability: float
    churn: bool
    threshold: float


class Batch(BaseModel):
    customers: list[Customer]


@lru_cache(maxsize=1)
def load_model():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(MODEL_URI)
    meta = {"model_uri": MODEL_URI}
    if MODEL_URI.startswith("models:/") and "@" in MODEL_URI:
        name, alias = MODEL_URI.removeprefix("models:/").split("@")
        mv = MlflowClient().get_model_version_by_alias(name, alias)
        meta |= {"name": name, "version": mv.version, "stage": mv.current_stage, "run_id": mv.run_id}
    return model, meta


app = FastAPI(title="Telco churn model", description="Serves the MLflow-registered Production churn classifier.")
THRESHOLD = 0.5


@app.get("/health")
def health() -> dict:
    try:
        _, meta = load_model()
    except Exception as exc:  # surfaced to the caller so a missing registry is obvious
        raise HTTPException(503, f"model not loadable from {MODEL_URI}: {exc}") from exc
    return {"status": "ok", **meta}


def _score(customers: list[Customer]) -> list[Prediction]:
    model, _ = load_model()
    frame = pd.DataFrame([c.model_dump() for c in customers])[config.FEATURES]
    proba = model.predict_proba(frame)[:, 1]
    return [Prediction(churn_probability=round(float(p), 4), churn=bool(p >= THRESHOLD), threshold=THRESHOLD) for p in proba]


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    return _score([customer])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(batch: Batch) -> list[Prediction]:
    return _score(batch.customers)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
