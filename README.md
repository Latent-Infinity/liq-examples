# liq-examples

Example end-to-end workflows for the LIQ Stack. Keeps examples isolated from core libraries to preserve separation of concerns. Data fetching relies on `liq-data`; credentials should be provided via `.env` where required. Prefer free/no-auth providers (e.g., Coinbase public, Binance public) for 1m granularity demos when possible.

## Structure
- `src/liq/examples`: Example scripts/pipelines (data → features → model → sim → metrics)
- `tests`: Integration tests using small fixtures

Advanced model research (LightGBM, LSTM, rolling retrain) lives in a separate
internal repo to keep examples public and minimal.

## Install
```bash
uv venv
uv pip install -e .
```
Requires the core LIQ libraries (`liq-data`, `liq-features`, `liq-metrics`, `liq-sim`,
`liq-signals`, `liq-store`, `liq-risk`) to be installed or available on `PYTHONPATH`.

## Running
1) Provide any needed credentials via `.env` (or select a no-auth provider/symbol).
2) Run example scripts/CLIs to execute the pipeline end-to-end.

CLI entry point:
```bash
liq-examples --help
```

### Development
```bash
uv sync --extra dev
uv run ruff check src tests
uv run ty check src tests
```

### BTC_USDT Examples

Uses Binance public data (no auth required) or a small fixture:
```bash
# From repo root
cd quant/liq-examples
uv run liq-examples --use-fixture

# Buy-and-hold on a small live slice (baseline)
uv run liq-examples --start 2024-01-01 --end 2024-01-02 --provider coinbase --strategy baseline

# EMA long/short on ~1 year of live data (midrange EMAs, shorting allowed). Adjust cooldown and max-signals to control runtime.
uv run liq-examples --start 2024-01-01 --end 2024-12-31 --provider binance_us --strategy ema_long_short --cooldown-bars 60 --max-signals 2000

# EMA with take-profit/stop-loss brackets (long and short)
uv run liq-examples --start 2024-01-01 --end 2024-12-31 --provider binance_us --strategy ema_bracket --cooldown-bars 60 --max-signals 2000

# Zigzag reversal signals (simple demo)
uv run liq-examples --start 2024-01-01 --end 2024-12-31 --provider binance_us --strategy zigzag --zigzag-pct 0.01

# Synthetic 1-year fixture remains available for offline runs (not default)
uv run liq-examples --use-synthetic-year --strategy ema_long_short

# Pre-generated signals replay: `--signals-path path/to/signals.csv|json`
```

### GP & Evolution Examples

Genetic programming (liq-gp) and strategy evolution (liq-evolution) examples:

```bash
cd quant/liq-examples

# Pure liq-gp symbolic regression — evolves sin(x) approximation
uv run python -m liq.examples.strategies.gp_basics

# Seed catalog exploration + warm-start evolution
uv run python -m liq.examples.strategies.evolution_seeds

# Two-stage fitness: fast F1 screening + parsimony refinement
uv run python -m liq.examples.strategies.evolution_two_stage

# Regime-aware operational presets
uv run python -m liq.examples.strategies.evolution_regime_presets
```

### Data source (store-only)
- All examples load bars from `liq-store` (Parquet). Populate via `liq-data`:
```bash
BINANCE_USE_US=1 DATA_ROOT=/tmp/liq_cache \
uv run python -m liq.data.cli fetch binance BTC_USDT --start 2024-01-01 --end 2024-12-31 --timeframe 1m
```
- **One venue, one storage key.** Binance bars live under `binance/BTC_USDT/1m` and nowhere else,
  including when `BINANCE_USE_US=1` produced them: that tree was measured against a live
  `api.binance.us` pull and is Binance.US data, which is what the `binance` key is declared to
  hold. **Do not copy or alias bars into a `binance_us/...` path** — this line used to say
  "copy/alias as needed", and duplicating bars under a second key makes the same minutes reachable
  under two fold governances, which is exactly what fold governance exists to prevent. Nothing needs
  copying: `--provider binance_us` already falls back to the `binance` key, and the two strings
  resolve to the same governed dataset. `liq-data` refuses an ingest whose venue disagrees with the
  key (`StorageVenueMismatchError`), and `fetch binance_us` is not a thing — no provider factory
  builds it. See `liq-docs/plans/oracle-binance-us-market-identity-2026-09-24.md`.
Ensure `DATA_ROOT` points to your stored data (e.g., `export DATA_ROOT=/path/to/liq-data/data`).
