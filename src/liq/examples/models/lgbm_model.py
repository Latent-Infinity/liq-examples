"""LightGBM example model placeholder.

Intent:
- Tabular model on midrange-based features (no scaling required).
- Forward midrange return sign classification with embargoed splits.
- Designed to be wired through liq-runner orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl

from liq.core import OrderRequest
from liq.core.enums import OrderSide, OrderType, TimeInForce


try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


@dataclass
class LightGBMModel:
    """LightGBM wrapper with minimal defaults for examples."""

    params: Dict[str, Any] | None = None
    model: Any | None = None

    def fit(self, features: pl.DataFrame, label_col: str = "label") -> Tuple["LightGBMModel", Dict[str, float]]:
        if lgb is None:
            raise ImportError("lightgbm is not installed; install to use LightGBMModel")
        X, y = _split_features_labels(features, label_col)
        dtrain = lgb.Dataset(X, label=y)
        params = self.params or {"objective": "binary", "learning_rate": 0.1, "num_leaves": 15, "min_data_in_leaf": 5}
        self.model = lgb.train(params, dtrain, num_boost_round=20)
        preds = self.model.predict(X)
        preds_sign = np.where(preds > 0.55, 1, np.where(preds < 0.45, -1, 0))
        acc = float((preds_sign == y).mean()) if len(y) else 0.0
        return self, {"accuracy": acc}

    def save(self, path: str) -> None:
        """Persist the trained model to disk."""
        if lgb is None:
            raise ImportError("lightgbm is not installed; cannot save model")
        if not self.model:
            raise ValueError("No model to save")
        self.model.save_model(path)

    @classmethod
    def load(cls, path: str) -> "LightGBMModel":
        """Load a persisted LightGBM model from disk."""
        if lgb is None:
            raise ImportError("lightgbm is not installed; cannot load model")
        booster = lgb.Booster(model_file=path)
        inst = cls()
        inst.model = booster
        return inst

    def predict_orders(
        self,
        df: pl.DataFrame,
        symbol: str,
        *,
        max_signals: int | None = None,
        cooldown: int = 10,
        threshold_hi: float = 0.55,
        threshold_lo: float = 0.45,
    ) -> List[Any]:
        if not self.model:
            return []
        X, _ = _split_features_labels(df, "label")
        preds = self.model.predict(X)
        orders: list[OrderRequest] = []
        last_idx = None
        for i, p in enumerate(preds):
            if last_idx is not None and (i - last_idx) < cooldown:
                continue
            ts = df["timestamp"][i]
            mid_price = df["mid"][i] if "mid" in df.columns else None
            if p > threshold_hi:
                side = OrderSide.BUY
            elif p < threshold_lo:
                side = OrderSide.SELL
            else:
                continue
            orders.append(
                OrderRequest(
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=Decimal("1"),
                    time_in_force=TimeInForce.DAY,
                    timestamp=ts if isinstance(ts, datetime) else datetime.now(),
                    metadata={"score": float(p), "mid": float(mid_price) if mid_price is not None else None},
                )
            )
            last_idx = i
            if max_signals is not None and len(orders) >= max_signals:
                break
        return orders


def _split_features_labels(features: pl.DataFrame, label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    cols = [c for c in features.columns if c not in (label_col, "timestamp")]
    X = features.select(cols).to_numpy()
    y = features[label_col].to_numpy()
    return X, y
