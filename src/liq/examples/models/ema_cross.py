"""EMA crossover strategy using midrange."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List

import polars as pl

from liq.types import OrderRequest
from liq.types.enums import OrderSide, OrderType, TimeInForce


@dataclass
class EMACrossModel:
    fast_window: int = 5
    slow_window: int = 20

    def predict(self, df: pl.DataFrame, symbol: str) -> List[OrderRequest]:
        if df.height < self.slow_window:
            return []
        mid = (df["high"] + df["low"]) / 2
        fast = mid.ewm_mean(span=self.fast_window, adjust=False).alias("fast")
        slow = mid.ewm_mean(span=self.slow_window, adjust=False).alias("slow")
        enriched = df.select(["timestamp"]).with_columns([fast, slow])

        # Find the first bullish crossover (fast crosses above slow) to generate a buy.
        fast_series = enriched["fast"]
        slow_series = enriched["slow"]
        cross_index = None
        for i in range(1, len(fast_series)):
            prev_diff = fast_series[i - 1] - slow_series[i - 1]
            curr_diff = fast_series[i] - slow_series[i]
            if prev_diff <= 0 and curr_diff > 0:
                cross_index = i
                break

        if cross_index is None:
            return []

        ts = enriched["timestamp"][cross_index]
        return [
            OrderRequest(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1"),
                time_in_force=TimeInForce.DAY,
                timestamp=ts if isinstance(ts, datetime) else datetime.now(),
            )
        ]
