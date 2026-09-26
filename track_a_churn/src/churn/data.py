"""Data loading, cleaning, the reference/current split and synthetic drift injection."""

from __future__ import annotations

import hashlib
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churn import config


def download_raw(force: bool = False) -> None:
    """Fetch the IBM Telco CSV (if missing) and verify its checksum so every clone trains on identical bytes."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if force or not config.RAW_CSV.exists():
        urllib.request.urlretrieve(config.RAW_CSV_URL, config.RAW_CSV)
    digest = hashlib.sha256(config.RAW_CSV.read_bytes()).hexdigest()
    if digest != config.RAW_CSV_SHA256:
        raise RuntimeError(f"{config.RAW_CSV} checksum mismatch: {digest}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the dataset's known quirks: blank TotalCharges for tenure-0 customers and a 0/1 SeniorCitizen flag."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip().replace("", np.nan))
    # The 11 blank rows all have tenure == 0: they have not been billed yet.
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).astype(str)
    df["tenure"] = df["tenure"].astype(float)
    df[config.TARGET] = (df[config.TARGET] == "Yes").astype(int)
    return df


def load_clean() -> pd.DataFrame:
    download_raw()
    return clean(pd.read_csv(config.RAW_CSV))


def split_reference_current(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random 70/30 split, stratified on the target so the clean current set is a fair control."""
    reference, current = train_test_split(
        df,
        train_size=config.REFERENCE_FRACTION,
        stratify=df[config.TARGET],
        random_state=config.SEED,
    )
    return reference.reset_index(drop=True), current.reset_index(drop=True)


def inject_drift(current: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """Return a copy of ``current`` with three deliberate, documented shifts.

    1. Categorical skew: resample rows so Contract == "Month-to-month" makes up 80% (naturally ~55%).
    2. Numeric shift: MonthlyCharges += Uniform(10, 30) per row (a simulated price rise).
    3. Label drift: flip 8% of "No churn" labels to "Churn".
    Row count is preserved so reference/current sizes stay comparable.
    """
    rng = np.random.default_rng(seed)
    n = len(current)

    m2m = current[current["Contract"] == "Month-to-month"]
    other = current[current["Contract"] != "Month-to-month"]
    n_m2m = int(round(n * config.DRIFT_MONTH_TO_MONTH_SHARE))
    drifted = pd.concat(
        [
            m2m.sample(n=n_m2m, replace=n_m2m > len(m2m), random_state=seed),
            other.sample(n=n - n_m2m, replace=False, random_state=seed),
        ]
    )
    drifted = drifted.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    low, high = config.DRIFT_MONTHLY_CHARGES_OFFSET
    drifted["MonthlyCharges"] = (drifted["MonthlyCharges"] + rng.uniform(low, high, size=n)).round(2)

    negatives = drifted.index[drifted[config.TARGET] == 0]
    flip = rng.choice(negatives, size=int(round(len(negatives) * config.DRIFT_LABEL_FLIP_RATE)), replace=False)
    drifted.loc[flip, config.TARGET] = 1
    return drifted


def prepare_splits() -> dict[str, pd.DataFrame]:
    """Materialise reference / clean current / drifted current CSVs under data/ and return them."""
    df = load_clean()
    reference, current = split_reference_current(df)
    drifted = inject_drift(current)
    reference.to_csv(config.REFERENCE_CSV, index=False)
    current.to_csv(config.CURRENT_CLEAN_CSV, index=False)
    drifted.to_csv(config.CURRENT_DRIFTED_CSV, index=False)
    return {"reference": reference, "current_clean": current, "current_drifted": drifted}


def train_test(reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Hold-out split inside the reference set; every model run sees exactly the same rows."""
    X = reference[config.FEATURES]
    y = reference[config.TARGET]
    return train_test_split(X, y, test_size=config.TEST_FRACTION, stratify=y, random_state=config.SEED)
