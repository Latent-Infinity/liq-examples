"""LSTM example model placeholder.

Intent:
- Sequence model on scaled midrange-return windows (1m base, optional 5m context).
- Forward midrange return sign classification with embargoed splits.
- Designed to be wired through liq-runner orchestration.

Implementation to follow per `quant/docs/model-example-plan.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import polars as pl

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

    def predict_orders(self, df: pl.DataFrame, symbol: str) -> List[Any]:
        # Strategy conversion is implemented elsewhere; keep stub here.
        return []


class _TinyLSTM(nn.Module):  # pragma: no cover - simple helper
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 3)  # classes: -1,0,1 mapped to indices 0,1,2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        return self.head(pooled)
