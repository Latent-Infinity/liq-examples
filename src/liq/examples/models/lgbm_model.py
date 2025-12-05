"""LightGBM example model placeholder.

Intent:
- Tabular model on midrange-based features (no scaling required).
- Forward midrange return sign classification with embargoed splits.
- Designed to be wired through liq-runner orchestration.

Implementation to follow per `quant/docs/model-example-plan.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import polars as pl

from liq.types import OrderRequest


@dataclass
class LightGBMModel:
    """Stub classifier wrapper for future LightGBM integration."""

    params: Dict[str, Any] | None = None

    def fit(self, df: pl.DataFrame) -> "LightGBMModel":
        # TODO: implement training using prepared feature matrix and labels
        return self

    def predict_orders(self, df: pl.DataFrame, symbol: str) -> List[OrderRequest]:
        # TODO: convert model scores to orders (long/short/flat) with throttling
        return []

