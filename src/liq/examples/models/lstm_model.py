"""LSTM example model placeholder.

Intent:
- Sequence model on scaled midrange-return windows (1m base, optional 5m context).
- Forward midrange return sign classification with embargoed splits.
- Designed to be wired through liq-runner orchestration.

Implementation to follow per `quant/docs/model-example-plan.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, List, Tuple

import numpy as np
import polars as pl

from liq.core import OrderRequest
from liq.core.enums import OrderSide, OrderType, TimeInForce

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


@dataclass
class LSTMModel:
    """Tiny LSTM wrapper for example use."""

    lookback: int = 60
    horizon: int = 1
    hidden_size: int = 16
    epochs: int = 3
    lr: float = 0.01
    model: any | None = None  # torch.nn.Module

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMModel":
        if torch is None:
            raise ImportError("torch is not installed; install to use LSTMModel")
        torch.set_num_threads(1)
        X = X.astype(np.float32)
        # Simple single-layer LSTM with mean over time -> linear head
        self.model = _TinyLSTM(input_size=X.shape[2], hidden_size=self.hidden_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y.astype(np.int64))
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self.model(X_tensor)
            loss = criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()
        return self

    def predict_orders(
        self,
        X: np.ndarray,
        timestamps: list,
        symbol: str,
        *,
        threshold_hi: float = 0.2,
        threshold_lo: float = -0.2,
        max_signals: int | None = None,
        cooldown: int = 10,
        mids: list[float] | None = None,
    ) -> List[OrderRequest]:
        if not self.model or torch is None:
            return []
        X_t = torch.from_numpy(X.astype(np.float32))
        logits = self.model(X_t)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        # map probs to score in [-1,1]: p_pos - p_neg
        scores = probs[:, 2] - probs[:, 0]
        orders: list[OrderRequest] = []
        last_idx = None
        for i, score in enumerate(scores):
            if last_idx is not None and (i - last_idx) < cooldown:
                continue
            ts = timestamps[i + self.lookback - 1]  # align window end with timestamp
            mid = mids[i + self.lookback - 1] if mids else None
            if score > threshold_hi:
                side = OrderSide.BUY
            elif score < threshold_lo:
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
                    metadata={"score": float(score), "mid": float(mid) if mid is not None else None},
                )
            )
            last_idx = i
            if max_signals is not None and len(orders) >= max_signals:
                break
        return orders


class _TinyLSTM(nn.Module):  # pragma: no cover - simple helper
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 3)  # classes: -1,0,1 mapped to indices 0,1,2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        return self.head(pooled)
