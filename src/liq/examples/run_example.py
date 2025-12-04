"""Run the BTC_USDT end-to-end example using Binance public data."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
from rich.console import Console
import typer

# Add sibling library src paths when running from repo (no install)
ROOT = Path(__file__).resolve().parents[3]
for rel in ("src", "../liq-metrics/src", "../liq-features/src", "../liq-data/src", "../liq-sim/src"):
    candidate = (ROOT / rel).resolve()
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from liq.data.providers.binance import BinanceProvider
from liq.data.providers.coinbase import CoinbaseProvider
from liq.examples.data.fixtures import btc_usdt_fixture, btc_usdt_year_fixture
from liq.examples.models.baseline import buy_and_hold
from liq.examples.models.linear import LinearSignalModel
from liq.examples.pipelines.pipeline import fit_pipeline
from liq.features.pipeline import FeaturePipeline
from liq.metrics import summarize_drift, summarize_labels, summarize_qa
from liq.data.qa import run_bar_qa
from liq.features.drift import ks_drift
from liq.features.labels import TripleBarrierConfig, triple_barrier_labels
from liq.sim.config import ProviderConfig, SimulatorConfig
from liq.sim.simulator import Simulator
from liq.types.enums import OrderSide, OrderType, TimeInForce
from liq.types import OrderRequest, Bar

app = typer.Typer(help="BTC_USDT end-to-end example")
console = Console()


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


@app.command()
def run(
    use_fixture: bool = typer.Option(False, help="Use small fixture instead of fetching"),
    use_synthetic_year: bool = typer.Option(False, help="Use 1-year synthetic fixture (no network; not default)"),
    start: str = typer.Option(None, help="Start date YYYY-MM-DD (default: 1 year ago)"),
    end: str = typer.Option(None, help="End date YYYY-MM-DD (default: today)"),
    provider: str = typer.Option("binance_us", help="Provider: binance|binance_us|coinbase"),
    strategy: str = typer.Option("baseline", help="Strategy: baseline|linear|ema_long_short|ema_bracket"),
) -> None:
    """Run the full pipeline: data -> features -> model -> sim -> metrics."""
    # Select symbol based on provider formatting
    trade_symbol = "BTC_USDT" if provider in ("binance", "binance_us") else "BTC-USD"

    # Data
    if use_fixture:
        df = btc_usdt_fixture()
    elif use_synthetic_year:
        df = btc_usdt_year_fixture()
    else:
        today = date.today()
        end_dt = date.fromisoformat(end) if end else today
        start_dt = date.fromisoformat(start) if start else (end_dt - timedelta(days=365))
        if provider == "binance":
            provider_client = BinanceProvider(use_us=False)
            df = provider_client.fetch_bars(
                trade_symbol,
                start=start_dt,
                end=end_dt,
                timeframe="1m",
            )
        elif provider == "binance_us":
            provider_client = BinanceProvider(use_us=True)
            df = provider_client.fetch_bars(
                trade_symbol,
                start=start_dt,
                end=end_dt,
                timeframe="1m",
            )
        elif provider == "coinbase":
            provider_client = CoinbaseProvider()
            df = provider_client.fetch_bars(
                trade_symbol,
                start=start_dt,
                end=end_dt,
                timeframe="1m",
            )
        else:
            raise ValueError("Unsupported provider; choose binance, binance_us, or coinbase")
    console.print(f"[green]Loaded bars: {df.height}[/green]")

    # QA metrics
    qa = run_bar_qa(df)
    console.print("[cyan]QA metrics[/cyan]", summarize_qa(qa))

    # Features/pipeline
    mid = (df["high"] + df["low"]) / 2
    rets = mid.pct_change().fill_null(0).to_list()
    pipeline = fit_pipeline(df)
    transformed = pipeline.transform(rets)

    # Drift (compare end slice vs train slice)
    split = max(5, len(transformed) // 2)
    drift_res = ks_drift(transformed[split:], transformed[:split], feature="rets", threshold=0.05)
    console.print("[cyan]Drift[/cyan]", summarize_drift([drift_res.statistic]))

    # Labels for sanity (triple barrier on closes)
    cfg = TripleBarrierConfig(take_profit=0.01, stop_loss=0.02, max_holding=5)
    labels = triple_barrier_labels(df, cfg)
    console.print("[cyan]Labels[/cyan]", summarize_labels(labels))

    # Orders: baseline + selected strategy
    baseline_orders = buy_and_hold(df, trade_symbol)
    model_orders: list[OrderRequest] = []
    if strategy == "linear":
        model_orders = LinearSignalModel().fit(df).predict(df, trade_symbol)
    elif strategy == "ema_long_short":
        from liq.examples.models.ema_cross import EMACrossModel
        model_orders = EMACrossModel(allow_short=True).predict(df, trade_symbol)
    elif strategy == "ema_bracket":
        from liq.examples.models.ema_cross import EMACrossModel
        model_orders = EMACrossModel(
            allow_short=True,
            take_profit_pct=0.01,
            stop_loss_pct=0.005,
        ).predict(df, trade_symbol)
    console.print(
        f"[yellow]Orders[/yellow] baseline={len(baseline_orders)}, strategy={strategy} -> {len(model_orders)}"
    )
    all_orders = baseline_orders + model_orders

    # Sim config (Binance-like)
    provider_cfg = ProviderConfig(
        name="binance",
        asset_classes=["crypto"],
        fee_model="ZeroCommission",
        slippage_model="VolumeWeighted",
        slippage_params={"base_bps": "0", "volume_impact": "0"},
        settlement_days=0,
    )
    sim_cfg = SimulatorConfig(min_order_delay_bars=0, initial_capital=1_000_000)
    sim = Simulator(provider_config=provider_cfg, config=sim_cfg)
    bars = _to_bars(df, trade_symbol)
    result = sim.run(all_orders, bars)
    console.print(f"[green]Fills: {len(result.fills)}[/green]")
    if result.equity_curve:
        console.print(f"[green]Final equity: {result.equity_curve[-1][1]}[/green]")


if __name__ == "__main__":
    app()
