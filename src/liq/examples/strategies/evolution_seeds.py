"""Seed catalog and warm-start example.

Demonstrates liq-evolution's curated seed catalog:

- ``list_known_strategy_seeds()`` — enumerate all 51+ seeds
- ``list_seed_templates_by_role(role)`` — filter by detector / expert / risk
- ``get_seed_template("ema_crossover")`` — inspect metadata
- ``build_strategy_seeds(names, registry)`` — build programs from templates
- ``evolve(..., seed_programs=seeds)`` — warm-start evolution with curated seeds

Usage:
    python -m liq.examples.strategies.evolution_seeds
"""

from __future__ import annotations

import numpy as np

from liq.evolution import (
    LabelFitnessEvaluator,
    SeedTemplateRole,
    build_strategy_seeds,
    build_trading_registry,
    evolve,
    get_seed_template,
    list_known_strategy_seeds,
    list_seed_templates_by_role,
    prepare_evaluation_context,
)
from liq.evolution.config import PrimitiveConfig
from liq.gp.config import GPConfig as LiqGPConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_ohlcv(n: int = 50, seed: int = 42) -> dict[str, np.ndarray]:
    """Generate synthetic OHLCV data with a trending close series."""
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
    """Binary labels: 1.0 where next bar's close is higher."""
    labels = np.zeros(len(close))
    labels[:-1] = (close[1:] > close[:-1]).astype(float)
    return labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_seed_catalog_demo(
    population_size: int = 50,
    generations: int = 5,
    seed: int = 42,
) -> None:
    """Explore the seed catalog, then warm-start an evolution run.

    Args:
        population_size: Programs per generation.
        generations: Number of generations.
        seed: RNG seed.
    """
    # -- 1. Explore the catalog ----------------------------------------
    all_seeds = list_known_strategy_seeds()
    print(f"Total seeds available: {len(all_seeds)}")

    for role in SeedTemplateRole:
        names = list_seed_templates_by_role(role)
        print(f"  {role.value:>10s}: {len(names)} seeds")

    # -- 2. Inspect a specific template --------------------------------
    template = get_seed_template("ema_crossover")
    print(f"\nTemplate: {template.name}")
    print(f"  Role           : {template.block_role.value}")
    print(f"  Arity          : {template.arity}")
    print(f"  Expected inputs: {template.expected_inputs}")
    print(f"  Regime hints   : {template.regime_hints}")

    # -- 3. Build seed programs ----------------------------------------
    registry = build_trading_registry(PrimitiveConfig())

    seed_names = ["ema_crossover", "rsi_oversold"]
    # Only use seeds that exist in the catalog
    available = set(all_seeds)
    seed_names = [s for s in seed_names if s in available]

    programs = build_strategy_seeds(seed_names, registry)
    print(f"\nBuilt {len(programs)} seed programs: {seed_names}")

    # -- 4. Warm-start evolution ---------------------------------------
    ohlcv = _synthetic_ohlcv(seed=seed)
    context = prepare_evaluation_context(ohlcv)
    context["labels"] = _make_labels(ohlcv["close"])

    # Build liq-gp config directly (mirrors liq-evolution's own integration
    # tests) with semantic_dedup_enabled=False to avoid FeatureContext
    # caching conflicts when fingerprinting uses a sub-sampled context.
    gp_config = LiqGPConfig(
        population_size=population_size,
        max_depth=4,
        generations=generations,
        seed=seed,
        tournament_size=3,
        elitism_count=2,
        constant_opt_enabled=False,
        semantic_dedup_enabled=False,
        simplification_enabled=False,
    )

    evaluator = LabelFitnessEvaluator(metric="f1", top_k=0.5)

    print(f"\nEvolving with {len(programs)} warm-start seeds …")
    result = evolve(
        registry=registry,
        config=gp_config,
        evaluator=evaluator,
        context=context,
        seed_programs=programs,
    )

    best = result.fitness_history[-1].best_fitness[0]
    print(f"Best fitness after {generations} generations: {best:.4f}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_seed_catalog_demo()
