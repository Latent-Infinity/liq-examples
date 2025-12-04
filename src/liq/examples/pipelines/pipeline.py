"""Feature pipeline setup for BTC_USDT example."""

from __future__ import annotations

import polars as pl

from liq.features.pipeline import FeaturePipeline


def fit_pipeline(df: pl.DataFrame) -> FeaturePipeline:
    """Fit feature pipeline on midrange returns."""
    mid = (df["high"] + df["low"]) / 2
    rets = mid.pct_change().fill_null(0).to_list()
    pipeline = FeaturePipeline(model_type="nn", d=0.3)
    pipeline.fit(rets)
    return pipeline
