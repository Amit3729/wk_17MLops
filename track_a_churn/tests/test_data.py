import pandas as pd

from churn import config
from churn.data import clean, inject_drift, load_clean, split_reference_current


def test_clean_fixes_known_quirks():
    df = load_clean()
    assert len(df) == 7043
    assert df["TotalCharges"].isna().sum() == 0
    assert set(df[config.TARGET].unique()) == {0, 1}
    assert set(df["SeniorCitizen"].unique()) == {"No", "Yes"}


def test_split_is_70_30_and_stratified():
    ref, cur = split_reference_current(load_clean())
    assert len(ref) + len(cur) == 7043
    assert abs(len(ref) / 7043 - config.REFERENCE_FRACTION) < 0.001
    assert abs(ref[config.TARGET].mean() - cur[config.TARGET].mean()) < 0.005
    assert set(ref[config.ID_COLUMN]).isdisjoint(cur[config.ID_COLUMN])


def test_inject_drift_applies_each_perturbation():
    _, cur = split_reference_current(load_clean())
    drifted = inject_drift(cur)
    assert len(drifted) == len(cur)
    share = (drifted["Contract"] == "Month-to-month").mean()
    assert abs(share - config.DRIFT_MONTH_TO_MONTH_SHARE) < 0.001
    low, _ = config.DRIFT_MONTHLY_CHARGES_OFFSET
    assert drifted["MonthlyCharges"].mean() - cur["MonthlyCharges"].mean() > low
    assert drifted[config.TARGET].mean() > cur[config.TARGET].mean()


def test_inject_drift_is_deterministic():
    _, cur = split_reference_current(load_clean())
    pd.testing.assert_frame_equal(inject_drift(cur), inject_drift(cur))


def test_clean_handles_blank_total_charges():
    raw = pd.DataFrame(
        {"TotalCharges": [" ", "10.5"], "SeniorCitizen": [0, 1], "tenure": [0, 1], config.TARGET: ["No", "Yes"]}
    )
    out = clean(raw)
    assert out["TotalCharges"].tolist() == [0.0, 10.5]
