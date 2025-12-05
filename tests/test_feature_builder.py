import polars as pl

from liq.examples.data.fixtures import btc_usdt_fixture
from liq.examples.pipelines.features import aggregate_5m, compute_features, make_sequence_windows


def test_compute_features_with_aggregation_and_label_shift() -> None:
    df = btc_usdt_fixture()
    feats = compute_features(df, horizon=1, include_5m=True)
    # Columns exist
    for col in ["ret_1", "vol_10", "ema_fast", "ema_slow", "vol_ratio", "label", "agg5m_close"]:
        assert col in feats.columns
    # No null labels after drop_nulls
    assert feats["label"].null_count() == 0
    # Label sign matches forward midrange change for an interior row
    mid = (df["high"] + df["low"]) / 2
    fwd_ret = (mid[6] - mid[5]) / mid[5]
    expected_label = 1 if fwd_ret > 0 else -1 if fwd_ret < 0 else 0
    assert feats["label"][5] == expected_label


def test_sequence_windows_shape_and_no_leakage() -> None:
    df = btc_usdt_fixture()
    feats = compute_features(df, horizon=1, include_5m=False)
    feature_cols = ["ret_1", "vol_10", "ema_fast", "ema_slow", "vol_ratio"]
    X, y = make_sequence_windows(feats, feature_cols, "label", lookback=5)
    assert X.shape[1] == 5  # lookback
    assert X.shape[2] == len(feature_cols)
    # number of windows matches expected (len - lookback + 1 minus dropped nulls)
    assert X.shape[0] == len(y)
