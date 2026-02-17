"""GP strategy evolution example.

Demonstrates evolving a trading signal strategy using liq-evolution's
genetic programming pipeline with synthetic OHLCV data.

Usage:
    python -m liq.examples.strategies.gp_strategy_example
"""

from __future__ import annotations

import numpy as np

from liq.evolution import (
    EvolutionConfig,
    Genome,
    LabelFitnessEvaluator,
    build_gp_config,
    build_trading_registry,
    evolve,
    prepare_evaluation_context,
    serialize_genome,
)
from liq.evolution.config import PrimitiveConfig


def generate_synthetic_ohlcv(n: int = 500, seed: int = 42) -> dict[str, np.ndarray]:
    """Generate synthetic OHLCV data with a trending close series."""
    rng = np.random.default_rng(seed)

    # Random walk for close prices
    returns = rng.normal(0.0002, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(returns))

    # Derive OHLC from close
    spread = rng.uniform(0.001, 0.005, n)
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.uniform(1e6, 1e7, n)

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def make_labels(close: np.ndarray) -> np.ndarray:
    """Create binary labels: 1.0 where next bar's close is higher."""
    labels = np.zeros(len(close))
    labels[:-1] = (close[1:] > close[:-1]).astype(float)
    return labels


def run_evolution() -> None:
    """Run a GP evolution and print the results."""
    # Build registry with default trading primitives (no liq-ta needed)
    registry = build_trading_registry(PrimitiveConfig())

    # Generate synthetic data
    ohlcv = generate_synthetic_ohlcv(n=500, seed=42)

    # Prepare evaluation context (adds derived series)
    context = prepare_evaluation_context(ohlcv)
    context["labels"] = make_labels(ohlcv["close"])

    # Configure evolution
    evo_config = EvolutionConfig(
        population_size=100,
        generations=20,
        max_depth=6,
        seed=42,
    )
    gp_config = build_gp_config(evo_config)

    # Evolve using label-based fitness (F1 score)
    evaluator = LabelFitnessEvaluator(metric="f1")
    result = evolve(
        registry=registry,
        config=gp_config,
        evaluator=evaluator,
        context=context,
    )

    # Results
    best_fitness = result.fitness_history[-1].best_fitness[0]
    print(f"Best fitness: {best_fitness:.4f}")
    print(f"Best program: {result.best_program}")
    print(f"Generations: {len(result.fitness_history)}")

    # Serialize the winning strategy
    genome = Genome(entry_program=result.best_program)
    payload = serialize_genome(genome)
    print(f"Serialized genome keys: {list(payload.keys())}")


if __name__ == "__main__":
    run_evolution()
