"""Custom Evidently metrics that go beyond the built-in drift presets.

Both are SingleValueMetric subclasses, so they render in the HTML report and accept Evidently tests
(e.g. ``tests=[lte(5.0)]``) exactly like built-in metrics.
"""

from __future__ import annotations

from typing import Optional

from evidently.core.metric_types import SingleValueCalculation, SingleValueMetric


class MeanDifference(SingleValueMetric):
    """|mean(current[column]) - mean(reference[column])|, in the column's own units (e.g. dollars).

    Evidently's drift tests answer "did the distribution change?"; this answers "by how much, in money?",
    which is the number a pricing or finance team would actually act on.
    """

    column: str


class MeanDifferenceCalculation(SingleValueCalculation[MeanDifference]):
    def calculate(self, context, current_data, reference_data):
        if reference_data is None:
            raise ValueError("MeanDifference needs reference data")
        cur = current_data.as_dataframe()[self.metric.column].mean()
        ref = reference_data.as_dataframe()[self.metric.column].mean()
        return self.result(abs(float(cur - ref))), None

    def display_name(self) -> str:
        return f"Absolute change in mean {self.metric.column} (current vs reference)"


class SegmentRateShift(SingleValueMetric):
    """|rate(current) - rate(reference)| of ``target == positive`` inside one segment (or overall).

    Global churn can look stable while one contract type churns much more; this isolates that segment.
    """

    target: str
    positive: int = 1
    segment_column: Optional[str] = None
    segment_value: Optional[str] = None


class SegmentRateShiftCalculation(SingleValueCalculation[SegmentRateShift]):
    def _rate(self, df) -> float:
        m = self.metric
        if m.segment_column is not None:
            df = df[df[m.segment_column] == m.segment_value]
        return float((df[m.target] == m.positive).mean())

    def calculate(self, context, current_data, reference_data):
        if reference_data is None:
            raise ValueError("SegmentRateShift needs reference data")
        shift = self._rate(current_data.as_dataframe()) - self._rate(reference_data.as_dataframe())
        return self.result(abs(shift)), None

    def display_name(self) -> str:
        m = self.metric
        where = f" where {m.segment_column} == '{m.segment_value}'" if m.segment_column else ""
        return f"Absolute shift in {m.target} rate{where} (current vs reference)"
