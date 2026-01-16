"""Run the BTC_USDT end-to-end example using Binance public data."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from time import perf_counter

import polars as pl
from rich.console import Console
import typer
import numpy as np

# Add sibling library src paths when running from repo (no install)
ROOT = Path(__file__).resolve().parents[3]
for rel in (
    "src",
    "../liq-metrics/src",
    "../liq-features/src",
    "../liq-data/src",
    "../liq-sim/src",
    "../liq-signals/src",
    "../liq-store/src",
    "../liq-risk/src",
):
    candidate = (ROOT / rel).resolve()
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from liq.data.settings import load_symbol_data, get_settings
from liq.examples.data.fixtures import btc_usdt_fixture, btc_usdt_year_fixture
from liq.examples.models.baseline import buy_and_hold
from liq.examples.pipelines.pipeline import fit_pipeline
from liq.examples.strategies.registry import StrategyConfig, get_strategy
from liq.metrics import summarize_drift, summarize_labels, summarize_qa
from liq.data.qa import run_bar_qa
from liq.features.drift import ks_drift
from liq.features.labels import TripleBarrierConfig, triple_barrier_labels
from liq.sim.config import ProviderConfig, SimulatorConfig
from liq.sim.simulator import Simulator
from liq.core.enums import OrderSide, OrderType
from uuid import UUID
from liq.core import OrderRequest, Bar
from liq.core import Fill
from liq.signals import FileSignalProvider, Signal
from liq.risk.sizers import CryptoFractionalSizer
from liq.risk.config import RiskConfig, MarketState
from liq.risk.engine import RiskEngine
from liq.core.portfolio import PortfolioState
from liq.core.position import Position

app = typer.Typer(help="BTC_USDT end-to-end example")
console = Console()
status = console.status


def _performance_metrics(equity_curve: list[tuple[datetime, Decimal]]) -> dict[str, float]:
    """Compute performance metrics: Sharpe, Sortino, Calmar, total return, max drawdown, hit rate."""
    if len(equity_curve) < 2:
        return {"sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "total_return": 0.0, "max_drawdown": 0.0, "hit_rate": 0.0}
    eq = np.array([float(v[1]) for v in equity_curve], dtype=float)
    rets = np.diff(eq) / eq[:-1]
    mean_ret = rets.mean()
    std_ret = rets.std(ddof=1)
    downside = rets[rets < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    # 1m bars ~ 525600 periods per year
    sharpe = float((mean_ret / std_ret) * np.sqrt(525600)) if std_ret != 0 else 0.0
    sortino = float((mean_ret / downside_std) * np.sqrt(525600)) if downside_std != 0 else 0.0
    total_return = float(eq[-1] / eq[0] - 1.0)
    peaks = np.maximum.accumulate(eq)
    drawdowns = (eq - peaks) / peaks
    max_dd = float(drawdowns.min()) if len(drawdowns) else 0.0
    calmar = float(total_return / abs(max_dd)) if max_dd < 0 else 0.0
    hits = (rets > 0).sum()
    hit_rate = float(hits / len(rets)) if len(rets) else 0.0
    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
    }


def _trade_metrics(fills: list[Fill]) -> dict[str, float]:
    """Compute trade-level metrics from fills (realized PnL based)."""
    if not fills:
        return {
            "trades": 0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "win_rate": 0.0,
            "pnl_sum": 0.0,
        }
    pnls = [float(f.realized_pnl) for f in fills]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else float("inf") if wins else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    win_rate = float(len(wins) / len(pnls)) if pnls else 0.0
    return {
        "trades": len(pnls),
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_rate": win_rate,
        "pnl_sum": float(sum(pnls)),
    }


def _build_bar_lookup(df: pl.DataFrame, symbol: str) -> dict[datetime, Bar]:
    lookup: dict[datetime, Bar] = {}
    for row in df.to_dicts():
        ts = row["timestamp"]
        lookup[ts] = Bar(
            symbol=symbol,
            timestamp=ts,
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
        )
    return lookup


def _size_with_risk(
    signals: list[Signal],
    bars: list[Bar],
    risk_fraction: float,
    initial_capital: Decimal,
    max_position_pct: float = 0.1,
    gross_leverage_cap: float = 2.0,
) -> list[OrderRequest]:
    """Size signals using RiskEngine + CryptoFractionalSizer with fractional-friendly constraints."""

    signals = sorted(signals, key=lambda s: s.timestamp)
    bar_map = {b.timestamp: b for b in bars}

    cash = initial_capital
    positions: dict[str, Position] = {}
    orders: list[OrderRequest] = []
    step = Decimal("0.0001")

    class FractionalMaxPositionConstraint:
        def apply(self, orders: list[OrderRequest], portfolio_state: PortfolioState, market_state: MarketState, risk_config: RiskConfig) -> list[OrderRequest]:
            result: list[OrderRequest] = []
            equity = portfolio_state.equity
            max_position_value = equity * Decimal(str(risk_config.max_position_pct))
            for order in orders:
                if order.side == OrderSide.SELL:
                    result.append(order)
                    continue
                bar = market_state.current_bars.get(order.symbol)
                if bar is None:
                    continue
                price = bar.close
                existing = portfolio_state.positions.get(order.symbol)
                existing_val = abs(existing.market_value) if existing else Decimal("0")
                remaining = max_position_value - existing_val
                if remaining <= 0:
                    continue
                order_val = order.quantity * price
                if order_val <= remaining:
                    result.append(order)
                else:
                    qty = (remaining / price).quantize(step, rounding=ROUND_DOWN)
                    if qty >= step:
                        result.append(order.model_copy(update={"quantity": qty}))
            return result

    class FractionalGrossLeverageConstraint:
        def apply(self, orders: list[OrderRequest], portfolio_state: PortfolioState, market_state: MarketState, risk_config: RiskConfig) -> list[OrderRequest]:
            equity = portfolio_state.equity
            max_exposure = equity * Decimal(str(risk_config.max_gross_leverage))
            current_exposure = sum((abs(pos.market_value) for pos in portfolio_state.positions.values()), Decimal("0"))
            sells = [o for o in orders if o.side == OrderSide.SELL]
            buys = [o for o in orders if o.side == OrderSide.BUY]
            result: list[OrderRequest] = list(sells)
            if not buys:
                return result

            order_values: list[tuple[OrderRequest, Decimal]] = []
            total_new = Decimal("0")
            for order in buys:
                bar = market_state.current_bars.get(order.symbol)
                if bar is None:
                    continue
                val = order.quantity * bar.close
                total_new += val
                order_values.append((order, val))
            if not order_values:
                return result

            remaining = max_exposure - current_exposure
            if remaining <= 0:
                return result
            if total_new <= remaining:
                result.extend(o for o, _ in order_values)
                return result

            scale = remaining / total_new
            for order, val in order_values:
                bar = market_state.current_bars.get(order.symbol)
                if bar is None:
                    continue
                scaled_val = val * scale
                qty = (scaled_val / bar.close).quantize(step, rounding=ROUND_DOWN)
                if qty >= step:
                    result.append(order.model_copy(update={"quantity": qty}))
            return result

    risk_cfg = RiskConfig(
        max_position_pct=max_position_pct,
        max_gross_leverage=gross_leverage_cap,
        risk_per_trade=risk_fraction,
        min_position_value=Decimal("1"),
    )
    engine = RiskEngine(
        sizer=CryptoFractionalSizer(fraction=risk_fraction, min_qty=step, step_qty=step),
        constraints=[
            FractionalMaxPositionConstraint(),
            FractionalGrossLeverageConstraint(),
        ],
    )

    def equity_and_gross(ts: datetime) -> tuple[Decimal, Decimal]:
        pos_val = Decimal("0")
        gross = Decimal("0")
        for pos in positions.values():
            bar = bar_map.get(ts)
            if bar and bar.symbol == pos.symbol:
                mv = pos.quantity * bar.close
                pos_val += mv
                gross += abs(mv)
        return cash + pos_val, gross

    for sig in signals:
        bar = bar_map.get(sig.timestamp)
        if bar is None or bar.close <= 0:
            continue

        # Mark existing positions to current price for accurate exposure
        marked_positions = {
            sym: pos.model_copy(update={"current_price": bar.close})
            for sym, pos in positions.items()
        }

        portfolio_state = PortfolioState(
            cash=cash,
            unsettled_cash=Decimal("0"),
            positions=marked_positions,
            realized_pnl=Decimal("0"),
            timestamp=bar.timestamp,
        )
        market_state = MarketState(
            current_bars={sig.symbol: bar},
            volatility={},
            liquidity={},
            sector_map=None,
            correlations=None,
            regime=None,
            timestamp=bar.timestamp,
        )
        res = engine.process_signals([sig], portfolio_state, market_state, risk_cfg)
        if not res.orders:
            continue
        for order in res.orders:
            orders.append(order)
            fill_notional = order.quantity * bar.close
            if order.side == OrderSide.BUY:
                cash -= fill_notional
                prev = positions.get(order.symbol)
                if prev:
                    new_qty = prev.quantity + order.quantity
                    avg_price = ((prev.quantity * prev.average_price) + fill_notional) / new_qty
                    positions[order.symbol] = Position(
                        symbol=order.symbol,
                        quantity=new_qty,
                        average_price=avg_price,
                        realized_pnl=prev.realized_pnl,
                        timestamp=bar.timestamp,
                        current_price=bar.close,
                    )
                else:
                    positions[order.symbol] = Position(
                        symbol=order.symbol,
                        quantity=order.quantity,
                        average_price=bar.close,
                        realized_pnl=Decimal("0"),
                        timestamp=bar.timestamp,
                        current_price=bar.close,
                    )
            else:
                # Short proceeds collateralized; do not expand spendable cash
                prev = positions.get(order.symbol)
                if prev:
                    new_qty = prev.quantity - order.quantity
                    positions[order.symbol] = Position(
                        symbol=order.symbol,
                        quantity=new_qty,
                        average_price=prev.average_price,
                        realized_pnl=prev.realized_pnl,
                        timestamp=bar.timestamp,
                        current_price=bar.close,
                    )
                else:
                    positions[order.symbol] = Position(
                        symbol=order.symbol,
                        quantity=-order.quantity,
                        average_price=bar.close,
                        realized_pnl=Decimal("0"),
                        timestamp=bar.timestamp,
                        current_price=bar.close,
                    )

    return orders


def _serialize_fill(fill: Fill) -> dict:
    """Convert Fill to JSON-serializable dict."""
    data = fill.__dict__.copy()
    for k, v in list(data.items()):
        if isinstance(v, UUID):
            data[k] = str(v)
        elif isinstance(v, (Decimal,)):
            data[k] = str(v)
        elif isinstance(v, datetime):
            data[k] = v.isoformat()
        elif isinstance(v, Path):
            data[k] = str(v)
    return data


def _to_bars(df: pl.DataFrame, symbol: str) -> list[Bar]:
    return [
        Bar(
            symbol=symbol,
            timestamp=row["timestamp"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
        )
        for row in df.to_dicts()
    ]

def _aggregate_for_sim(df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    """Optionally downsample bars for simulation to reduce runtime."""
    if timeframe == "1m":
        return df
    raise ValueError("Unsupported sim timeframe; use 1m")


@app.command()
def run(
    use_fixture: bool = typer.Option(False, help="Use small fixture instead of fetching"),
    use_synthetic_year: bool = typer.Option(False, help="Use 1-year synthetic fixture (no network; not default)"),
    start: str = typer.Option(None, help="Start date YYYY-MM-DD (default: 1 year ago)"),
    end: str = typer.Option(None, help="End date YYYY-MM-DD (default: today)"),
    provider: str = typer.Option("binance_us", help="Provider: binance|binance_us|coinbase"),
    strategy: str = typer.Option(
        "baseline",
        help="Strategy: baseline|linear|ema_long_short|ema_bracket|zigzag",
    ),
    cooldown_bars: int = typer.Option(60, help="Min bars between EMA signals"),
    max_signals: int | None = typer.Option(None, help="Max signals (caps runtime) for ML/EMA; None=unbounded"),
    export_json: str = typer.Option(None, help="Optional path to export fills/equity summary as JSON"),
    signals_path: str = typer.Option(None, help="Path to pre-generated signals file (.csv/.json/.jsonl)"),
    profile_json: str = typer.Option(None, help="Optional path to write timing/profile summary as JSON"),
    risk_fraction: float = typer.Option(0.05, help="Fraction of equity per trade for sizing (liq-risk fixed-fractional)"),
    risk_max_position_pct: float = typer.Option(0.1, help="Max position size as fraction of equity"),
    gross_leverage_cap: float = typer.Option(2.0, help="Max gross leverage cap used for sizing"),
    zigzag_pct: float = typer.Option(0.01, help="Zigzag reversal threshold (fraction, e.g., 0.01=1%)"),
) -> None:
    """Run the full pipeline: data -> features -> model -> sim -> metrics."""
    timings: dict[str, float] = {}

    def record(label: str, start: float) -> float:
        end = perf_counter()
        timings[label] = end - start
        return perf_counter()

    t0 = perf_counter()
    # Select symbol based on provider formatting
    trade_symbol = "BTC_USDT" if provider in ("binance", "binance_us") else "BTC-USD"
    initial_capital = Decimal("1000000")

    # Data
    with console.status("[yellow]Loading data...[/yellow]"):
        if use_fixture:
            df = btc_usdt_fixture()
        elif use_synthetic_year:
            df = btc_usdt_year_fixture()
        else:
            today = date.today()
            end_dt = date.fromisoformat(end) if end else today
            start_dt = date.fromisoformat(start) if start else (end_dt - timedelta(days=365))

            # Use liq-data's load_symbol_data which accesses data via liq-store
            df = None
            provider_candidates = [provider, provider.replace("_us", "")]

            # Try liq-data loader (uses liq-store exclusively)
            for prov in provider_candidates:
                try:
                    df_candidate = load_symbol_data(
                        provider=prov,
                        symbol=trade_symbol,
                        timeframe="1m",
                        start=start_dt,
                        end=end_dt,
                    )
                    if df_candidate is not None and df_candidate.height > 0:
                        df = df_candidate.sort("timestamp")
                        break
                except FileNotFoundError:
                    continue
                except Exception:
                    continue

            if df is None:
                data_root = get_settings().data_root
                raise RuntimeError(
                    f"No data found for {provider}/{trade_symbol}/1m in liq-store. "
                    f"Please ingest data via liq-data:\n"
                    f"  liq-data fetch {provider.replace('_us', '')} {trade_symbol} "
                    f"--start {start_dt} --end {end_dt}\n"
                    f"Data root: {data_root}"
                )
    t0 = record("data_load", t0)
    console.print(f"[green]Loaded bars: {df.height}[/green]")

    # QA metrics
    with console.status("[yellow]Running QA...[/yellow]"):
        qa = run_bar_qa(df)
    t0 = record("qa", t0)
    console.print("[cyan]QA metrics[/cyan]", summarize_qa(qa))

    # Features/pipeline for baseline metrics
    with console.status("[yellow]Building features...[/yellow]"):
        mid = (df["high"] + df["low"]) / 2
        rets = mid.pct_change().fill_null(0).to_list()
        pipeline = fit_pipeline(df)
        transformed = pipeline.transform(rets)
    t0 = record("feature_pipeline", t0)

    # Drift (compare end slice vs train slice)
    split = max(5, len(transformed) // 2)
    drift_res = ks_drift(transformed[split:], transformed[:split], feature="rets", threshold=0.05)
    t0 = record("drift", t0)
    console.print("[cyan]Drift[/cyan]", summarize_drift([drift_res.statistic]))

    # Labels for sanity (triple barrier on closes)
    with console.status("[yellow]Computing labels...[/yellow]"):
        cfg = TripleBarrierConfig(take_profit=0.01, stop_loss=0.02, max_holding=5)
        label_df = df.with_columns(
            [
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
            ]
        )
        labels = triple_barrier_labels(label_df, cfg)
    t0 = record("labels", t0)
    console.print("[cyan]Labels[/cyan]", summarize_labels(labels))

    # Signals: either load from file or generate from a strategy/model
    loaded_signals: list[Signal] = []
    if signals_path:
        provider = FileSignalProvider(Path(signals_path))
        loaded_signals = list(provider.generate_signals())
        console.print(f"[cyan]Loaded {len(loaded_signals)} signals from {signals_path}[/cyan]")

    baseline_orders = buy_and_hold(df, trade_symbol)
    model_orders: list[OrderRequest] = []
    if loaded_signals:
        bars_for_sizing = _to_bars(df, trade_symbol)
        model_orders = _size_with_risk(
            loaded_signals,
            bars_for_sizing,
            risk_fraction,
            initial_capital,
            risk_max_position_pct,
            gross_leverage_cap=gross_leverage_cap,
        )
    else:
        cfg = StrategyConfig(
            cooldown_bars=cooldown_bars,
            max_signals=max_signals,
            zigzag_pct=zigzag_pct,
        )
        try:
            strategy_fn = get_strategy(strategy)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise SystemExit(1)
        with console.status(f"[yellow]Running strategy {strategy}...[/yellow]"):
            model_orders = strategy_fn(df, trade_symbol, cfg)
    t0 = record("signals_generation", t0)
    console.print(
        f"[yellow]Orders[/yellow] baseline={len(baseline_orders)}, strategy={strategy} -> {len(model_orders)}"
    )
    df_for_sim = df
    all_orders = baseline_orders + model_orders
    console.print(
        f"[yellow]Running simulation on {len(bars:=_to_bars(_aggregate_for_sim(df_for_sim, '1m'), trade_symbol))} bars and {len(all_orders)} orders; "
        f"may take a few minutes for long ranges[/yellow]"
    )

    # Sim config (Binance-like)
    provider_cfg = ProviderConfig(
        name="binance_us",
        asset_classes=["crypto"],
        fee_model="TieredMakerTaker",
        fee_params={"maker_bps": "8", "taker_bps": "10"},
        slippage_model="SpreadBased",
        slippage_params={},
        short_enabled=True,
        margin_type="RegT",
        initial_margin_rate=Decimal("0.5"),
        maintenance_margin_rate=Decimal("0.4"),
        settlement_days=0,
    )
    sim_cfg = SimulatorConfig(min_order_delay_bars=0, initial_capital=int(initial_capital))
    sim = Simulator(provider_config=provider_cfg, config=sim_cfg)
    result = sim.run(all_orders, bars)
    t0 = record("simulation", t0)
    console.print(f"[green]Fills: {len(result.fills)}[/green]")
    if result.equity_curve:
        console.print(f"[green]Final equity: {result.equity_curve[-1][1]}[/green]")
        perf = _performance_metrics(result.equity_curve)
        perf["start_ts"] = result.equity_curve[0][0].isoformat()
        perf["end_ts"] = result.equity_curve[-1][0].isoformat()
        trade_perf = _trade_metrics(result.fills)
        console.print(f"[cyan]Performance (all)[/cyan] {perf}")
        console.print(f"[cyan]Trade metrics[/cyan] {trade_perf}")
    if export_json or profile_json:
        export_path = Path(export_json) if export_json else Path(profile_json)
        summary = {
            "strategy": strategy,
            "provider": provider,
            "orders_submitted": len(all_orders),
            "fills": [_serialize_fill(f) for f in result.fills],
            "equity_curve": [(ts.isoformat(), str(eq)) for ts, eq in result.equity_curve],
            "timings": timings,
            "performance": perf if result.equity_curve else {},
            "trade_metrics": trade_perf if result.fills else {},
            "risk": {
                "risk_fraction": risk_fraction,
                "max_position_pct": risk_max_position_pct,
                "gross_leverage_cap": gross_leverage_cap,
            },
        }
        export_path.write_text(json.dumps(summary, indent=2))
        console.print(f"[cyan]Exported run summary to {export_path}[/cyan]")
    else:
        console.print(f"[cyan]Timings (s)[/cyan] {timings}")


if __name__ == "__main__":
    app()
