"""Baseline buy-and-hold model."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import List

import polars as pl

from liq.types import OrderRequest
from liq.types.enums import OrderSide, OrderType, TimeInForce


def buy_and_hold(df: pl.DataFrame, symbol: str) -> List[OrderRequest]:
    """Generate a single market buy at first bar."""
    if df.is_empty():
        return []
    first_ts = df["timestamp"][0]
    return [
        OrderRequest(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1"),
            time_in_force=TimeInForce.DAY,
            timestamp=first_ts if isinstance(first_ts, datetime) else datetime.now(),
        )
    ]
