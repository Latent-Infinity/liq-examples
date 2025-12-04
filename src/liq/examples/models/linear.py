"""Linear regression signal model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List

import polars as pl
from sklearn.linear_model import LinearRegression

from liq.types import OrderRequest
from liq.types.enums import OrderSide, OrderType, TimeInForce


@dataclass
class LinearSignalModel:
    lookback: int = 3
    threshold: float = 0.0

    def fit(self, df: pl.DataFrame) -> "LinearSignalModel":
        # Fit on lagged midrange returns
        if df.height <= self.lookback:
            return self
        features = []
        targets = []
        mid = (df["high"] + df["low"]) / 2
        rets = mid.pct_change().fill_null(0).to_list()
        for i in range(self.lookback, len(rets)):
            features.append(rets[i - self.lookback : i])
            targets.append(rets[i])
        self._model = LinearRegression().fit(features, targets)
        return self

    def predict(self, df: pl.DataFrame, symbol: str) -> List[OrderRequest]:
        if not hasattr(self, "_model"):
            return []
        mid = (df["high"] + df["low"]) / 2
        rets = mid.pct_change().fill_null(0).to_list()
        if len(rets) < self.lookback:
            return []
        latest_feats = [rets[-i] for i in range(self.lookback, 0, -1)]
        pred = float(self._model.predict([latest_feats])[0])
        if pred <= self.threshold:
            return []
        last_ts = df["timestamp"][-1]
        return [
            OrderRequest(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1"),
                time_in_force=TimeInForce.DAY,
                timestamp=last_ts if isinstance(last_ts, datetime) else datetime.now(),
            )
        ]
