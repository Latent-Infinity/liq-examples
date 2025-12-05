"""LightGBM example model placeholder.

Intent:
- Tabular model on midrange-based features (no scaling required).
- Forward midrange return sign classification with embargoed splits.
- Designed to be wired through liq-runner orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl


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

    def predict_orders(self, df: pl.DataFrame, symbol: str) -> List[Any]:
        # Strategy conversion is implemented elsewhere; keep stub here.
        return []


def _split_features_labels(features: pl.DataFrame, label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    cols = [c for c in features.columns if c not in (label_col, "timestamp")]
    X = features.select(cols).to_numpy()
    y = features[label_col].to_numpy()
    return X, y
