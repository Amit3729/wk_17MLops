import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.tests import lte

from churn import config
from churn.drift import retrain_decision
from churn.monitoring_metrics import MeanDifference, SegmentRateShift


def _ds(df):
    return Dataset.from_pandas(
        df, data_definition=DataDefinition(numerical_columns=["MonthlyCharges", "Churn"], categorical_columns=["Contract"])
    )


def _run(metric, ref, cur):
    snap = Report([metric], include_tests=True).run(_ds(cur), _ds(ref)).dict()
    return snap["metrics"][0]["value"], [t["status"] for t in snap["tests"]]


REF = pd.DataFrame({"MonthlyCharges": [10.0, 20.0, 30.0, 40.0], "Churn": [0, 1, 0, 0], "Contract": ["M", "M", "Y", "Y"]})
CUR = pd.DataFrame({"MonthlyCharges": [20.0, 30.0, 40.0, 50.0], "Churn": [1, 1, 0, 1], "Contract": ["M", "M", "Y", "Y"]})


def test_mean_difference_value_and_test():
    value, statuses = _run(MeanDifference(column="MonthlyCharges", tests=[lte(5.0)]), REF, CUR)
    assert value == 10.0
    assert [str(getattr(s, "value", s)) for s in statuses] == ["FAIL"]


def test_segment_rate_shift_overall_and_segment():
    overall, _ = _run(SegmentRateShift(target="Churn"), REF, CUR)
    assert overall == 0.5  # 0.25 -> 0.75
    seg, _ = _run(SegmentRateShift(target="Churn", segment_column="Contract", segment_value="Y"), REF, CUR)
    assert seg == 0.5  # 0.0 -> 0.5


def _quality(f1, auc):
    return {"f1": f1, "roc_auc": auc}


def test_retrain_decision_quiet_when_nothing_moves():
    dd = {"share_drifted": 0.0, "n_drifted": 0, "n_features": 19}
    retrain, reasons = retrain_decision(dd, {"tests": []}, _quality(0.62, 0.84), _quality(0.62, 0.84))
    assert not retrain and reasons == []


def test_retrain_decision_catches_auc_drop_that_f1_hides():
    dd = {"share_drifted": 0.1, "n_drifted": 2, "n_features": 19}
    retrain, reasons = retrain_decision(dd, {"tests": []}, _quality(0.62, 0.84), _quality(0.65, 0.75))
    assert retrain and any("ROC-AUC" in r for r in reasons)


def test_retrain_decision_on_feature_share():
    dd = {"share_drifted": config.DRIFT_SHARE_THRESHOLD, "n_drifted": 6, "n_features": 19}
    retrain, _ = retrain_decision(dd, {"tests": []}, _quality(0.62, 0.84), _quality(0.62, 0.84))
    assert retrain
