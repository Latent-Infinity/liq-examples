"""LSTM example model placeholder.

Intent:
- Sequence model on scaled midrange-return windows (1m base, optional 5m context).
- Forward midrange return sign classification with embargoed splits.
- Designed to be wired through liq-runner orchestration.

Implementation to follow per `quant/docs/model-example-plan.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import polars as pl

from liq.types import OrderRequest


@dataclass
class LSTMModel:
    """Stub wrapper for future LSTM integration."""

    lookback: int = 60
    horizon: int = 1

    def fit(self, df: pl.DataFrame) -> "LSTMModel":
        # TODO: implement scaling + sequence batching + training
        return self

    def predict_orders(self, df: pl.DataFrame, symbol: str) -> List[OrderRequest]:
        # TODO: convert model outputs to orders (long/short/flat) with throttling
        return []

