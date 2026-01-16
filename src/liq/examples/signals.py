"""Helpers to convert stack-level signals into orders for examples."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Iterable, List

from liq.core import OrderRequest
from liq.core.enums import OrderSide, OrderType, TimeInForce
from liq.signals import Signal


def signals_to_orders(
    signals: Iterable[Signal],
    *,
    default_quantity: Decimal = Decimal("1"),
    default_time_in_force: TimeInForce = TimeInForce.DAY,
    price_lookup: dict[datetime, float] | None = None,
    risk_fraction: float | None = None,
    initial_equity: float | None = None,
    max_signals_per_day: int | None = None,
    cooldown_bars: int | None = None,
) -> List[OrderRequest]:
    """Convert signals to orders with optional frequency controls."""
    orders: list[OrderRequest] = []
    per_day_counts: dict[datetime, int] = {}
    last_dir_ts: dict[str, dict[str, datetime]] = {}
    for sig in signals:
        if sig.direction == "flat":
            continue
        side = OrderSide.BUY if sig.direction == "long" else OrderSide.SELL
        ts = sig.normalized_timestamp()
        if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
            ts = ts.replace(tzinfo=timezone.utc)
        day = ts.date()
        # Enforce per-day cap
        if max_signals_per_day is not None:
            count = per_day_counts.get(day, 0)
            if count >= max_signals_per_day:
                continue
        # Enforce same-direction cooldown
        if cooldown_bars is not None:
            dir_key = "long" if side == OrderSide.BUY else "short"
            last_ts = last_dir_ts.get(sig.symbol, {}).get(dir_key)
            if last_ts and (ts - last_ts).total_seconds() < cooldown_bars * 60 * 15:  # assuming 15m bars
                continue
        qty = default_quantity
        if risk_fraction and initial_equity and price_lookup:
            px = price_lookup.get(ts)
            if px and px > 0:
                qty = Decimal(str((initial_equity * risk_fraction) / px))
        orders.append(
            OrderRequest(
                symbol=sig.symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=qty,
                time_in_force=default_time_in_force,
                timestamp=ts,
                confidence=sig.strength,
                metadata=sig.metadata or None,
            )
        )
        per_day_counts[day] = per_day_counts.get(day, 0) + 1
        last_dir_ts.setdefault(sig.symbol, {})["long" if side == OrderSide.BUY else "short"] = ts
    return orders
