"""Two-stage fitness evaluation example.

Demonstrates wiring a fast screening stage (label-based F1) with a
refinement stage that adds a parsimony bonus for smaller programs:

- ``LabelFitnessEvaluator(metric="f1")`` as Stage A (fast screening)
- Custom ``RefinementEvaluator`` as Stage B (parsimony bonus)
- ``TwoStageFitnessEvaluator(stage_a, stage_b, top_k=5)`` — wired together
- Gate metadata inspection: ``reason_code``, ``lineage``, ``source_stage``

Usage:
    python -m liq.examples.strategies.evolution_two_stage
"""

from __future__ import annotations

from typing import Any

import numpy as np

from liq.evolution import (
    LabelFitnessEvaluator,
    TwoStageFitnessEvaluator,
    build_trading_registry,
    evaluate,
    evolve,
    prepare_evaluation_context,
)
from liq.evolution.config import PrimitiveConfig
from liq.gp import FitnessResult
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
# Stage B: refinement evaluator (label fitness + parsimony bonus)
# ---------------------------------------------------------------------------


class RefinementEvaluator:
    """Re-score programs with a parsimony bonus for smaller trees.

    fitness = label_score + parsimony_weight / program_size
    """

    def __init__(self, parsimony_weight: float = 0.01) -> None:
        self._label_eval = LabelFitnessEvaluator(metric="f1", top_k=0.5)
        self._parsimony_weight = parsimony_weight

    def evaluate(
        self,
        programs: list[Any],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        base_results = self._label_eval.evaluate(programs, context)
        refined: list[FitnessResult] = []
        for prog, base in zip(programs, base_results, strict=True):
            bonus = self._parsimony_weight / max(prog.size, 1)
            refined.append(
                FitnessResult(
                    objectives=(base.objectives[0] + bonus,),
                    metadata={
                        **base.metadata,
                        "parsimony_bonus": bonus,
                        "source_stage": "B",
                    },
                )
            )
        return refined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_two_stage_demo(
    population_size: int = 50,
    generations: int = 5,
    seed: int = 42,
) -> None:
    """Run evolution with two-stage fitness and inspect gate metadata.

    Args:
        population_size: Programs per generation.
        generations: Number of generations.
        seed: RNG seed.
    """
    registry = build_trading_registry(PrimitiveConfig())

    ohlcv = _synthetic_ohlcv(seed=seed)
    context = prepare_evaluation_context(ohlcv)
    context["labels"] = _make_labels(ohlcv["close"])

    # -- Build the two-stage evaluator ---------------------------------
    stage_a = LabelFitnessEvaluator(metric="f1", top_k=0.5)
    stage_b = RefinementEvaluator(parsimony_weight=0.01)

    two_stage = TwoStageFitnessEvaluator(
        stage_a=stage_a,
        stage_b=stage_b,
        top_k=5,
    )

    # -- Configure & run -----------------------------------------------
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

    print(f"Two-stage evolution  (pop={population_size}, gens={generations}, top_k=5)")
    result = evolve(
        registry=registry,
        config=gp_config,
        evaluator=two_stage,
        context=context,
    )

    # -- Inspect results -----------------------------------------------
    best = result.best_program
    output = evaluate(best, context)
    print(f"\nBest program size   : {best.size}")
    print(f"Best fitness        : {result.fitness_history[-1].best_fitness[0]:.4f}")
    print(f"Output range        : [{output.min():.4f}, {output.max():.4f}]")

    # Run both stages on the winner to show metadata
    a_results = stage_a.evaluate([best], context)
    b_results = stage_b.evaluate([best], context)
    print(f"Stage A score       : {a_results[0].objectives[0]:.4f}")
    print(f"Stage B score       : {b_results[0].objectives[0]:.4f}")
    if b_results[0].metadata:
        print(f"Stage B metadata    : {b_results[0].metadata}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_two_stage_demo()
