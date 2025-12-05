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
from liq.examples.models.lgbm_model import LightGBMModel
from liq.examples.models.lstm_model import LSTMModel
from liq.examples.pipelines.features import compute_features, make_sequence_windows
from liq.examples.pipelines.pipeline import fit_pipeline
from liq.features.pipeline import FeaturePipeline
from liq.metrics import summarize_drift, summarize_labels, summarize_qa
from liq.data.qa import run_bar_qa
from liq.features.drift import ks_drift
from liq.features.labels import TripleBarrierConfig, triple_barrier_labels
from liq.sim.config import ProviderConfig, SimulatorConfig
from liq.sim.simulator import Simulator
from liq.core.enums import OrderSide, OrderType, TimeInForce
from liq.core import OrderRequest, Bar
from liq.core import Fill

app = typer.Typer(help="BTC_USDT end-to-end example")
console = Console()
status = console.status


def _serialize_fill(fill: Fill) -> dict:
    """Convert Fill to JSON-serializable dict."""
    data = fill.__dict__.copy()
    for k, v in list(data.items()):
        if isinstance(v, (Decimal,)):
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


@app.command()
def run(
    use_fixture: bool = typer.Option(False, help="Use small fixture instead of fetching"),
    use_synthetic_year: bool = typer.Option(False, help="Use 1-year synthetic fixture (no network; not default)"),
    start: str = typer.Option(None, help="Start date YYYY-MM-DD (default: 1 year ago)"),
    end: str = typer.Option(None, help="End date YYYY-MM-DD (default: today)"),
    provider: str = typer.Option("binance_us", help="Provider: binance|binance_us|coinbase"),
    strategy: str = typer.Option("baseline", help="Strategy: baseline|linear|ema_long_short|ema_bracket|lgbm|lstm"),
    cooldown_bars: int = typer.Option(60, help="Min bars between EMA signals"),
    max_signals: int = typer.Option(2000, help="Max EMA signals (caps runtime)"),
    export_json: str = typer.Option(None, help="Optional path to export fills/equity summary as JSON"),
) -> None:
    """Run the full pipeline: data -> features -> model -> sim -> metrics."""
    # Select symbol based on provider formatting
    trade_symbol = "BTC_USDT" if provider in ("binance", "binance_us") else "BTC-USD"

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
    with console.status("[yellow]Running QA...[/yellow]"):
        qa = run_bar_qa(df)
    console.print("[cyan]QA metrics[/cyan]", summarize_qa(qa))

    # Features/pipeline for baseline metrics
    with console.status("[yellow]Building features...[/yellow]"):
        mid = (df["high"] + df["low"]) / 2
        rets = mid.pct_change().fill_null(0).to_list()
        pipeline = fit_pipeline(df)
        transformed = pipeline.transform(rets)

    # Drift (compare end slice vs train slice)
    split = max(5, len(transformed) // 2)
    drift_res = ks_drift(transformed[split:], transformed[:split], feature="rets", threshold=0.05)
    console.print("[cyan]Drift[/cyan]", summarize_drift([drift_res.statistic]))

    # Labels for sanity (triple barrier on closes)
    with console.status("[yellow]Computing labels...[/yellow]"):
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
        with console.status("[yellow]Generating EMA long/short signals...[/yellow]"):
            model_orders = EMACrossModel(
                allow_short=True,
                cooldown_bars=cooldown_bars,
                max_signals=max_signals,
            ).predict(df, trade_symbol)
    elif strategy == "ema_bracket":
        from liq.examples.models.ema_cross import EMACrossModel
        with console.status("[yellow]Generating EMA bracket signals...[/yellow]"):
            model_orders = EMACrossModel(
                allow_short=True,
                take_profit_pct=0.01,
                stop_loss_pct=0.005,
                cooldown_bars=cooldown_bars,
                max_signals=max_signals,
            ).predict(df, trade_symbol)
    elif strategy in {"lgbm", "lstm"}:
        with console.status("[yellow]Preparing ML features...[/yellow]"):
            feats = compute_features(df, horizon=1, include_5m=True)
        if strategy == "lgbm":
            try:
                model, metrics = LightGBMModel().fit(feats)
                console.print(f"[cyan]LGBM metrics[/cyan] {metrics}")
                model_orders = model.predict_orders(feats, trade_symbol, max_signals=max_signals, cooldown=cooldown_bars)
            except ImportError:
                console.print("[red]lightgbm not installed; skipping strategy[/red]")
                model_orders = []
        else:  # lstm
            try:
                feature_cols = [c for c in feats.columns if c not in ("timestamp", "label")]
                X, y = make_sequence_windows(feats, feature_cols, "label", lookback=60)
                if X.shape[0] == 0:
                    raise ValueError("Not enough data for LSTM windows")
                model = LSTMModel(lookback=60, epochs=1)
                model.fit(X, y)
                timestamps = feats["timestamp"].to_list()
                mids = feats["mid"].to_list()
                model_orders = model.predict_orders(
                    X,
                    timestamps,
                    trade_symbol,
                    max_signals=max_signals,
                    cooldown=cooldown_bars,
                    mids=mids,
                )
                console.print(f"[cyan]LSTM[/cyan] training complete; signals={len(model_orders)}")
            except ImportError:
                console.print("[red]torch not installed; skipping strategy[/red]")
                model_orders = []
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")
                model_orders = []
    console.print(
        f"[yellow]Orders[/yellow] baseline={len(baseline_orders)}, strategy={strategy} -> {len(model_orders)}"
    )
    all_orders = baseline_orders + model_orders
    console.print(
        f"[yellow]Running simulation on {len(bars:=_to_bars(df, trade_symbol))} bars and {len(all_orders)} orders; "
        f"may take a few minutes for long ranges[/yellow]"
    )

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
    result = sim.run(all_orders, bars)
    console.print(f"[green]Fills: {len(result.fills)}[/green]")
    if result.equity_curve:
        console.print(f"[green]Final equity: {result.equity_curve[-1][1]}[/green]")
    if export_json:
        export_path = Path(export_json)
        summary = {
            "strategy": strategy,
            "provider": provider,
            "orders_submitted": len(all_orders),
            "fills": [_serialize_fill(f) for f in result.fills],
            "equity_curve": [(ts.isoformat(), str(eq)) for ts, eq in result.equity_curve],
        }
        export_path.write_text(json.dumps(summary, indent=2))
        console.print(f"[cyan]Exported run summary to {export_path}[/cyan]")


if __name__ == "__main__":
    app()
