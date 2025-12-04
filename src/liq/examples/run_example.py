"""Run the BTC_USDT end-to-end example using Binance public data."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
from rich.console import Console
import typer

from liq.data.providers.binance import BinanceProvider
from liq.examples.data.fixtures import btc_usdt_fixture
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


def _to_bars(df: pl.DataFrame) -> list[Bar]:
    return [
        Bar(
            symbol="BTC_USDT",
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
    start: str = typer.Option("2024-01-01", help="Start date YYYY-MM-DD"),
    end: str = typer.Option("2024-01-07", help="End date YYYY-MM-DD"),
    use_us: bool = typer.Option(False, help="Use binance.us endpoint"),
) -> None:
    """Run the full pipeline: data -> features -> model -> sim -> metrics."""
    # Data
    if use_fixture:
        df = btc_usdt_fixture()
    else:
        provider = BinanceProvider(use_us=use_us)
        df = provider.fetch_bars(
            "BTC_USDT",
            start=date.fromisoformat(start),
            end=date.fromisoformat(end),
            timeframe="1m",
        )
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

    # Orders: baseline + model
    baseline_orders = buy_and_hold(df, "BTC_USDT")
    model = LinearSignalModel().fit(df)
    model_orders = model.predict(df, "BTC_USDT")
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
    bars = _to_bars(df)
    result = sim.run(all_orders, bars)
    console.print(f"[green]Fills: {len(result.fills)}[/green]")
    if result.equity_curve:
        console.print(f"[green]Final equity: {result.equity_curve[-1][1]}[/green]")


if __name__ == "__main__":
    app()
