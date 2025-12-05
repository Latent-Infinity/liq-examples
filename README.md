# liq-examples

Example end-to-end workflows for the LIQ Stack. Keeps examples isolated from core libraries to preserve separation of concerns. Data fetching relies on `liq-data`; credentials should be provided via `.env` where required. Prefer free/no-auth providers (e.g., Coinbase public, Binance public) for 1m granularity demos when possible.

## Structure
- `src/liq/examples`: Example scripts/pipelines (data → features → model → sim → metrics)
- `tests`: Integration tests using small fixtures

## Running
1) Provide any needed credentials via `.env` (or select a no-auth provider/symbol).
2) Run example scripts/CLIs to execute the pipeline end-to-end.

### BTC_USDT Examples

Uses Binance public data (no auth required) or a small fixture:
```bash
# From repo root (adds sibling libs to PYTHONPATH)
cd quant/liq-examples
PYTHONPATH=src:../liq-metrics/src:../liq-features/src:../liq-data/src:../liq-sim/src \
  python -m liq.examples.run_example --use-fixture

# Buy-and-hold on a small live slice (baseline)
PYTHONPATH=src:../liq-metrics/src:../liq-features/src:../liq-data/src:../liq-sim/src \
  python -m liq.examples.run_example --start 2024-01-01 --end 2024-01-02 --provider coinbase --strategy baseline

# EMA long/short on ~1 year of live data (midrange EMAs, shorting allowed). Adjust cooldown and max-signals to control runtime.
PYTHONPATH=src:../liq-metrics/src:../liq-features/src:../liq-data/src:../liq-sim/src \
  python -m liq.examples.run_example --start 2024-01-01 --end 2024-12-31 --provider binance_us --strategy ema_long_short --cooldown-bars 60 --max-signals 2000

# EMA with take-profit/stop-loss brackets (long and short)
PYTHONPATH=src:../liq-metrics/src:../liq-features/src:../liq-data/src:../liq-sim/src \
  python -m liq.examples.run_example --start 2024-01-01 --end 2024-12-31 --provider binance_us --strategy ema_bracket --cooldown-bars 60 --max-signals 2000

# Synthetic 1-year fixture remains available for offline runs (not default)
PYTHONPATH=src:../liq-metrics/src:../liq-features/src:../liq-data/src:../liq-sim/src \
  python -m liq.examples.run_example --use-synthetic-year --strategy ema

# Model examples (in progress; optional deps):
# - LightGBM tabular (1m/5m features, no scaling): `--strategy lgbm` (requires lightgbm)
# - LSTM sequence (scaled sequences, 1m + optional 5m context): `--strategy lstm` (requires torch)
# Both will be orchestrated through liq-runner configs once finalized.
```
