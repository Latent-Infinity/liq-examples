# liq-examples

Example end-to-end workflows for the LIQ Stack. Keeps examples isolated from core libraries to preserve separation of concerns. Data fetching relies on `liq-data`; credentials should be provided via `.env` where required. Prefer free/no-auth providers (e.g., Coinbase public, Binance public) for 1m granularity demos when possible.

## Structure
- `src/liq/examples`: Example scripts/pipelines (data → features → model → sim → metrics)
- `tests`: Integration tests using small fixtures

## Running
1) Provide any needed credentials via `.env` (or select a no-auth provider/symbol).
2) Run example scripts/CLIs to execute the pipeline end-to-end.
