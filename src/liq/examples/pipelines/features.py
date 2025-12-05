"""Feature builders for model examples (LightGBM, LSTM)."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import polars as pl


def aggregate_5m(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate 1m bars to 5m bars and shift by one window to avoid look-ahead."""
    grouped = df.group_by_dynamic(
        index_column="timestamp",
        every="5m",
        period="5m",
        closed="right",
        label="right",
    ).agg(
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
    ).sort("timestamp")
    # Shift aggregates so only completed 5m bars are used at or after their end time
    shifted = grouped.with_columns(
        [
            pl.col(col).shift(1)
            for col in grouped.columns
            if col != "timestamp"
        ]
    )
    rename_map = {c: f"agg5m_{c}" for c in shifted.columns if c != "timestamp"}
    return shifted.rename(rename_map)


def compute_features(
    df: pl.DataFrame,
    *,
    horizon: int = 1,
    include_5m: bool = True,
) -> pl.DataFrame:
    """Build leakage-safe tabular features and labels."""
    df = df.sort("timestamp")
    mid = (df["high"] + df["low"]) / 2
    ret_1 = mid.pct_change().fill_null(0).alias("ret_1")
    vol_10 = ret_1.rolling_std(window_size=10, min_samples=5).alias("vol_10")
    ema_fast = mid.ewm_mean(span=5, adjust=False).alias("ema_fast")
    ema_slow = mid.ewm_mean(span=20, adjust=False).alias("ema_slow")
    vol_ratio = (df["volume"] / df["volume"].rolling_mean(window_size=20, min_samples=5)).alias("vol_ratio")

    base = df.select(
        "timestamp",
        ret_1,
        vol_10,
        ema_fast,
        ema_slow,
        vol_ratio,
        mid.alias("mid"),
    )

    if include_5m:
        agg = aggregate_5m(df)
        base = base.join_asof(agg, on="timestamp", strategy="backward")

    # Label: forward midrange return sign
    fwd_mid = base["mid"].shift(-horizon)
    label = (
        pl.when(fwd_mid > base["mid"])
        .then(1)
        .when(fwd_mid < base["mid"])
        .then(-1)
        .otherwise(0)
        .alias("label")
    )
    features = base.with_columns(label)
    features = features.drop_nulls(subset=["label"])
    return features


def make_sequence_windows(
    features: pl.DataFrame,
    feature_cols: Iterable[str],
    label_col: str,
    *,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create rolling sequence windows for LSTM training."""
    feats = features.select(list(feature_cols))
    labels = features[label_col]
    max_start = len(features) - lookback
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    for i in range(max_start):
        label_val = labels[i + lookback - 1]
        if label_val is None:
            continue
        window = feats.slice(i, lookback).to_numpy()
        X_list.append(window)
        y_list.append(int(label_val))
    if not X_list:
        return np.empty((0, lookback, len(list(feature_cols)))), np.empty((0,))
    return np.stack(X_list), np.array(y_list)
