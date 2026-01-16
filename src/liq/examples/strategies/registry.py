"""Strategy registry for examples.

Keeps run_example.py slim and makes adding/removing strategies trivial.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Dict, Optional

import polars as pl

from liq.examples.models.baseline import buy_and_hold
from liq.examples.models.linear import LinearSignalModel
from liq.examples.signals import signals_to_orders
from liq.signals import Signal
from liq.core.enums import TimeInForce
from liq.features.indicators.zigzag import zigzag_pivots


@dataclass
class StrategyConfig:
    cooldown_bars: int = 60
    max_signals: Optional[int] = None
    zigzag_pct: float = 0.01


StrategyFunc = Callable[[pl.DataFrame, str, StrategyConfig], list]


def baseline_strategy(df: pl.DataFrame, symbol: str, cfg: StrategyConfig) -> list:
    return buy_and_hold(df, symbol)


def linear_strategy(df: pl.DataFrame, symbol: str, cfg: StrategyConfig) -> list:
    return LinearSignalModel().fit(df).predict(df, symbol)


def ema_long_short_strategy(df: pl.DataFrame, symbol: str, cfg: StrategyConfig) -> list:
    from liq.examples.models.ema_cross import EMACrossModel

    return EMACrossModel(
        allow_short=True,
        cooldown_bars=cfg.cooldown_bars,
        max_signals=cfg.max_signals,
    ).predict(df, symbol)


def ema_bracket_strategy(df: pl.DataFrame, symbol: str, cfg: StrategyConfig) -> list:
    from liq.examples.models.ema_cross import EMACrossModel

    return EMACrossModel(
        allow_short=True,
        take_profit_pct=0.01,
        stop_loss_pct=0.005,
        cooldown_bars=cfg.cooldown_bars,
        max_signals=cfg.max_signals,
    ).predict(df, symbol)


def zigzag_strategy(df: pl.DataFrame, symbol: str, cfg: StrategyConfig) -> list:
    signals = zigzag_pivots(
        df["timestamp"].to_list(),
        ((df["high"] + df["low"]) / 2).to_list(),
        pct=cfg.zigzag_pct,
        symbol=symbol,
    )
    return signals_to_orders(
        signals,
        default_quantity=Decimal("1"),
        default_time_in_force=TimeInForce.DAY,
    )


def build_registry() -> Dict[str, StrategyFunc]:
    return {
        "baseline": baseline_strategy,
        "linear": linear_strategy,
        "ema_long_short": ema_long_short_strategy,
        "ema_bracket": ema_bracket_strategy,
        "zigzag": zigzag_strategy,
    }


def get_strategy(name: str) -> StrategyFunc:
    registry = build_registry()
    if name not in registry:
        raise ValueError(f"Unknown strategy '{name}'")
    return registry[name]
