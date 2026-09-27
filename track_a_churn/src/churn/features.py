"""Feature preprocessing shared by every model, so runs differ only in the estimator and its hyperparameters."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import config


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.CATEGORICAL_FEATURES),
        ]
    )
