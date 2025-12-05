import polars as pl
import sys
from pathlib import Path

# Ensure sibling library paths are discoverable (liq-types, etc.)
ROOT = Path(__file__).resolve().parents[2]  # quant
for rel in ("liq-examples/src", "liq-types/src"):
    candidate = (ROOT / rel).resolve()
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
import pytest

from liq.examples.data.fixtures import btc_usdt_fixture
from liq.examples.models.lgbm_model import LightGBMModel
from liq.examples.models.lstm_model import LSTMModel
from liq.examples.pipelines.features import compute_features, make_sequence_windows


def test_lgbm_fit_returns_metrics_if_installed() -> None:
    feats = compute_features(btc_usdt_fixture(), horizon=1, include_5m=False)
    try:
        model, metrics = LightGBMModel().fit(feats)
    except ImportError:
        pytest.skip("lightgbm not installed")
    assert "accuracy" in metrics
    assert model.model is not None


def test_lstm_fit_runs_on_sequence_if_installed() -> None:
    feats = compute_features(btc_usdt_fixture(), horizon=1, include_5m=False)
    feature_cols = ["ret_1", "vol_10", "ema_fast", "ema_slow", "vol_ratio"]
    X, y = make_sequence_windows(feats, feature_cols, "label", lookback=5)
    if X.shape[0] == 0:
        pytest.skip("not enough data for windows")
    # Cap for test speed
    X, y = X[:10], y[:10]
    try:
        model = LSTMModel(lookback=5, epochs=1)
        model.fit(X, y)
    except ImportError:
        pytest.skip("torch not installed")
    # model.model should exist if torch is available
    assert hasattr(model, "model") and model.model is not None
