# liq-examples

Example end-to-end workflows for the LIQ Stack. Keeps examples isolated from core libraries to preserve separation of concerns. Data fetching relies on `liq-data`; credentials should be provided via `.env` where required. Prefer free/no-auth providers (e.g., Coinbase public, Binance public) for 1m granularity demos when possible.

## Structure
- `src/liq/examples`: Example scripts/pipelines (data → features → model → sim → metrics)
- `tests`: Integration tests using small fixtures

## Running
1) Provide any needed credentials via `.env` (or select a no-auth provider/symbol).
2) Run example scripts/CLIs to execute the pipeline end-to-end.

### BTC_USDT Example

Uses Binance public data (no auth required) or a small fixture:
```bash
# From repo root (adds sibling libs to PYTHONPATH)
cd quant/liq-examples
PYTHONPATH=src:../liq-metrics/src:../liq-features/src:../liq-data/src:../liq-sim/src \
  python -m liq.examples.run_example --use-fixture

# Fetch binance public BTC_USDT 1m (adjust dates as desired)
PYTHONPATH=src:../liq-metrics/src:../liq-features/src:../liq-data/src:../liq-sim/src \
  python -m liq.examples.run_example --start 2024-01-01 --end 2024-01-07
```
