"""Single source of truth for paths, names, seeds and thresholds.

Every value here is referenced from the README, so changing one changes the documented behaviour.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("CHURN_PROJECT_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "telco_churn.csv"
RAW_CSV_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)
RAW_CSV_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"

REFERENCE_CSV = DATA_DIR / "reference.csv"  # 70% "training-time" data
CURRENT_CLEAN_CSV = DATA_DIR / "current_clean.csv"  # 30% "production" data, untouched (control)
CURRENT_DRIFTED_CSV = DATA_DIR / "current_drifted.csv"  # same 30% with injected drift

REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = PROJECT_ROOT / "results"

# MLflow: SQLite backend (required for the Model Registry) + local artifact store.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
MLFLOW_ARTIFACT_ROOT = PROJECT_ROOT / "mlartifacts"
EXPERIMENT_TRAINING = "telco-churn-training"
EXPERIMENT_MONITORING = "telco-churn-monitoring"
REGISTERED_MODEL_NAME = "telco-churn-classifier"
PRODUCTION_ALIAS = "production"
STAGING_ALIAS = "staging"

SEED = 42
REFERENCE_FRACTION = 0.70  # reference/current split for monitoring
TEST_FRACTION = 0.20  # hold-out inside the reference set for model selection
CV_FOLDS = 5

TARGET = "Churn"
ID_COLUMN = "customerID"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Model selection: churn is imbalanced (~26.5% positive), so accuracy is not used to pick a winner.
SELECTION_METRIC = "test_f1"
TIEBREAK_METRIC = "test_roc_auc"

# Synthetic drift injected into the current set (see churn.data.inject_drift).
DRIFT_MONTHLY_CHARGES_OFFSET = (10.0, 30.0)  # uniform per-row price increase, in $
DRIFT_MONTH_TO_MONTH_SHARE = 0.80  # natural share is ~55%
DRIFT_LABEL_FLIP_RATE = 0.08  # share of "No" labels flipped to "Yes"

# Retraining policy used by churn.drift and the Airflow DAG.
DRIFT_SHARE_THRESHOLD = 0.30  # retrain if >= 30% of features drift
MEAN_MONTHLY_CHARGES_DELTA_LIMIT = 5.0  # custom metric test bound, in $
SEGMENT_CHURN_SHIFT_LIMIT = 0.05  # custom metric test bound (absolute churn-rate change)
F1_DEGRADATION_LIMIT = 0.05  # retrain if production F1 drops by more than this on labelled current data
AUC_DEGRADATION_LIMIT = 0.05  # ... or if ROC-AUC does (prevalence-independent, so it catches what F1 hides)
