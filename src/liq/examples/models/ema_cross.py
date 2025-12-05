"""EMA crossover strategy using midrange."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List

import polars as pl

from liq.core import OrderRequest
from liq.core.enums import OrderSide, OrderType, TimeInForce


@dataclass
class EMACrossModel:
    fast_window: int = 5
    slow_window: int = 20
    quantity: Decimal = Decimal("1")
    allow_short: bool = False
    take_profit_pct: float | None = None
    stop_loss_pct: float | None = None
    cooldown_bars: int = 60  # enforce spacing between signals
    max_signals: int | None = None

    def predict(self, df: pl.DataFrame, symbol: str) -> List[OrderRequest]:
        if df.height < self.slow_window:
            return []
        mid = (df["high"] + df["low"]) / 2
        fast = mid.ewm_mean(span=self.fast_window, adjust=False).alias("fast")
        slow = mid.ewm_mean(span=self.slow_window, adjust=False).alias("slow")
        enriched = df.select(["timestamp", "close"]).with_columns([fast, slow])

        fast_series = enriched["fast"]
        slow_series = enriched["slow"]
        orders: list[OrderRequest] = []
        last_signal_idx: int | None = None
        signal_count = 0

        for i in range(1, len(fast_series)):
            if last_signal_idx is not None and (i - last_signal_idx) < self.cooldown_bars:
                continue
            if self.max_signals is not None and signal_count >= self.max_signals:
                break

            prev_diff = fast_series[i - 1] - slow_series[i - 1]
            curr_diff = fast_series[i] - slow_series[i]
            ts = enriched["timestamp"][i]
            close_p = enriched["close"][i]

            crossed_up = prev_diff <= 0 and curr_diff > 0
            crossed_down = prev_diff >= 0 and curr_diff < 0

            if crossed_up:
                orders.extend(self._entry_and_brackets(symbol, OrderSide.BUY, ts, close_p))
                last_signal_idx = i
                signal_count += 1
            elif crossed_down and self.allow_short:
                orders.extend(self._entry_and_brackets(symbol, OrderSide.SELL, ts, close_p))
                last_signal_idx = i
                signal_count += 1

        return orders

    def _entry_and_brackets(
        self,
        symbol: str,
        side: OrderSide,
        ts: datetime,
        close_price: float,
    ) -> List[OrderRequest]:
        """Create market entry plus optional TP/SL bracket."""
        entry = OrderRequest(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=self.quantity,
            time_in_force=TimeInForce.DAY,
            timestamp=ts if isinstance(ts, datetime) else datetime.now(),
        )

        orders = [entry]

        if self.take_profit_pct is None or self.stop_loss_pct is None:
            return orders

        cp = Decimal(str(close_price))
        tp_mult = Decimal(str(self.take_profit_pct))
        sl_mult = Decimal(str(self.stop_loss_pct))

        if side == OrderSide.BUY:
            tp_price = cp * (Decimal("1") + tp_mult)
            sl_price = cp * (Decimal("1") - sl_mult)
            tp_side = OrderSide.SELL
            sl_side = OrderSide.SELL
        else:
            tp_price = cp * (Decimal("1") - tp_mult)
            sl_price = cp * (Decimal("1") + sl_mult)
            tp_side = OrderSide.BUY
            sl_side = OrderSide.BUY

        orders.append(
            OrderRequest(
                symbol=symbol,
                side=tp_side,
                order_type=OrderType.LIMIT,
                quantity=self.quantity,
                limit_price=tp_price,
                time_in_force=TimeInForce.DAY,
                timestamp=ts if isinstance(ts, datetime) else datetime.now(),
                tags={"type": "take_profit"},
            )
        )
        orders.append(
            OrderRequest(
                symbol=symbol,
                side=sl_side,
                order_type=OrderType.STOP,
                quantity=self.quantity,
                stop_price=sl_price,
                time_in_force=TimeInForce.DAY,
                timestamp=ts if isinstance(ts, datetime) else datetime.now(),
                tags={"type": "stop_loss"},
            )
        )
        return orders
