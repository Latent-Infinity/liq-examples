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
    fast_window: int = 10
    slow_window: int = 30

    def predict(self, df: pl.DataFrame, symbol: str) -> List[OrderRequest]:
        if df.height < self.slow_window:
            return []
        mid = (df["high"] + df["low"]) / 2
        fast = mid.ewm_mean(span=self.fast_window, adjust=False).alias("fast")
        slow = mid.ewm_mean(span=self.slow_window, adjust=False).alias("slow")
        enriched = df.select(["timestamp"]).with_columns([fast, slow])
        last = enriched.tail(2)
        if last.height < 2:
            return []
        prev_fast, prev_slow = last["fast"][0], last["slow"][0]
        curr_fast, curr_slow = last["fast"][1], last["slow"][1]
        crossed_up = prev_fast <= prev_slow and curr_fast > curr_slow
        if not crossed_up:
            return []
        ts = df["timestamp"][-1]
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
