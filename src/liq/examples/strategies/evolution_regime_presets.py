"""Regime-aware operational presets example.

Demonstrates loading and inspecting pre-tuned regime presets:

- ``list_regime_presets()`` — enumerate available presets
- ``get_regime_preset("baseline")`` — load a named preset
- Inspect: ``operator_mode``, ``seed_templates``, ``stability_controls``
- Override with a small ``LiqGPConfig`` for a quick demo run
- Run evolution with seed programs from the preset's recommended templates

Usage:
    python -m liq.examples.strategies.evolution_regime_presets
"""

from __future__ import annotations

import numpy as np

from liq.evolution import (
    LabelFitnessEvaluator,
    build_strategy_seeds,
    build_trading_registry,
    evolve,
    get_regime_preset,
    list_regime_presets,
    prepare_evaluation_context,
)
from liq.evolution.config import PrimitiveConfig
from liq.gp.config import GPConfig as LiqGPConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_ohlcv(n: int = 50, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    spread = rng.uniform(0.001, 0.005, n)
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.uniform(1e6, 1e7, n)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def _make_labels(close: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(close))
    labels[:-1] = (close[1:] > close[:-1]).astype(float)
    return labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_regime_presets_demo(preset_name: str = "baseline") -> None:
    """Load a regime preset, inspect it, and run a small evolution.

    Args:
        preset_name: Name of the preset to load (e.g. ``"baseline"``).
    """
    # -- 1. List available presets -------------------------------------
    available = list_regime_presets()
    print(f"Available presets: {available}")

    # -- 2. Load the named preset --------------------------------------
    preset = get_regime_preset(preset_name)
    print(f"\nPreset            : {preset.name}")
    print(f"Description       : {preset.description}")
    print(f"Operator mode     : {preset.operator_mode}")
    print(f"Seed templates    : {preset.seed_templates}")
    print(f"Stability controls: {preset.stability_controls}")

    # -- 3. Build a small config for demo ------------------------------
    # Use LiqGPConfig directly with conservative settings to keep
    # the example fast and avoid indicator-parameter edge cases.
    gp_config = LiqGPConfig(
        population_size=50,
        max_depth=4,
        generations=5,
        seed=42,
        tournament_size=3,
        elitism_count=2,
        constant_opt_enabled=False,
        semantic_dedup_enabled=False,
        simplification_enabled=False,
    )

    # -- 4. Prepare data, evaluator, and seed programs -----------------
    registry = build_trading_registry(PrimitiveConfig())

    # Build seed programs from the preset's recommended templates
    seed_programs = build_strategy_seeds(preset.seed_templates, registry)
    ohlcv = _synthetic_ohlcv()
    context = prepare_evaluation_context(ohlcv)
    context["labels"] = _make_labels(ohlcv["close"])

    evaluator = LabelFitnessEvaluator(metric="f1", top_k=0.5)

    # -- 5. Run evolution with preset config ---------------------------
    print(f"\nEvolving with preset '{preset_name}' ({len(seed_programs)} seeds) …")
    result = evolve(
        registry=registry,
        config=gp_config,
        evaluator=evaluator,
        context=context,
        seed_programs=seed_programs,
    )

    best_fitness = result.fitness_history[-1].best_fitness[0]
    print(f"Best fitness: {best_fitness:.4f}")
    print(f"Pareto front size: {len(result.pareto_front)}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_regime_presets_demo()
