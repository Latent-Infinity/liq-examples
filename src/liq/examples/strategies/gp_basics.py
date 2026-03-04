"""Pure liq-gp symbolic regression example.

Evolves an approximation of ``sin(x)`` using only the liq-gp engine — no
liq-evolution dependency. Demonstrates:

- ``PrimitiveRegistry`` — building a custom registry with math ops
- ``GPConfig`` — small config with constant optimization and simplification
- Custom ``RegressionEvaluator`` — negative MSE fitness
- ``evolve()`` — core loop with a generation callback
- ``serialize`` / ``deserialize`` — program round-trip
- ``simplify`` — post-evolution algebraic simplification

Usage:
    python -m liq.examples.strategies.gp_basics
"""

from __future__ import annotations

import numpy as np

from liq.gp import (
    EvolutionResult,
    FitnessConfig,
    FitnessResult,
    GenerationStats,
    GPConfig,
    PrimitiveRegistry,
    Series,
    deserialize,
    evaluate,
    evolve,
    serialize,
    simplify,
)

# ---------------------------------------------------------------------------
# 1. Build a custom primitive registry
# ---------------------------------------------------------------------------


def _build_math_registry() -> PrimitiveRegistry:
    """Register basic math ops + terminals for symbolic regression."""
    reg = PrimitiveRegistry()

    # Binary operators -------------------------------------------------
    reg.register(
        "add",
        lambda a, b: a + b,
        input_types=(Series, Series),
        output_type=Series,
        category="arithmetic",
    )
    reg.register(
        "sub",
        lambda a, b: a - b,
        input_types=(Series, Series),
        output_type=Series,
        category="arithmetic",
    )
    reg.register(
        "mul",
        lambda a, b: a * b,
        input_types=(Series, Series),
        output_type=Series,
        category="arithmetic",
    )

    def _protected_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(np.abs(b) < 1e-10, 1.0, a / b)
        return np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)

    reg.register(
        "div",
        _protected_div,
        input_types=(Series, Series),
        output_type=Series,
        category="arithmetic",
    )

    # Unary operators --------------------------------------------------
    reg.register(
        "neg",
        lambda a: -a,
        input_types=(Series,),
        output_type=Series,
        category="arithmetic",
    )

    # Terminals --------------------------------------------------------
    reg.register("x", lambda: None, input_types=(), output_type=Series, category="terminal")

    return reg


# ---------------------------------------------------------------------------
# 2. Custom fitness evaluator (negative MSE)
# ---------------------------------------------------------------------------


class RegressionEvaluator:
    """Evaluate programs as regressors: fitness = -MSE vs. target."""

    def __init__(self, target: np.ndarray) -> None:
        self._target = target

    def evaluate(
        self,
        programs: list,
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        results: list[FitnessResult] = []
        for prog in programs:
            try:
                output = evaluate(prog, context)
                mse = float(np.mean((output - self._target) ** 2))
            except Exception:
                mse = 1e12
            results.append(FitnessResult(objectives=(-mse,)))
        return results


# ---------------------------------------------------------------------------
# 3. Main entry point
# ---------------------------------------------------------------------------


def run_gp_basics(
    population_size: int = 50,
    generations: int = 5,
    seed: int = 42,
) -> EvolutionResult:
    """Evolve an approximation of sin(x) and return the result.

    Args:
        population_size: Number of programs per generation.
        generations: How many generations to run.
        seed: RNG seed for reproducibility.

    Returns:
        The ``EvolutionResult`` from the GP engine.
    """
    # -- data ----------------------------------------------------------
    rng = np.random.default_rng(seed)
    x = rng.uniform(-np.pi, np.pi, 200)
    target = np.sin(x)
    context: dict[str, np.ndarray] = {"x": x}

    # -- registry ------------------------------------------------------
    registry = _build_math_registry()

    # -- config --------------------------------------------------------
    config = GPConfig(
        population_size=population_size,
        generations=generations,
        max_depth=6,
        seed=seed,
        constant_opt_enabled=True,
        simplification_enabled=True,
        fitness=FitnessConfig(
            objectives=["neg_mse"],
            objective_directions=["maximize"],
        ),
    )

    # -- evaluator -----------------------------------------------------
    evaluator = RegressionEvaluator(target)

    # -- evolve --------------------------------------------------------
    def _on_generation(stats: GenerationStats) -> None:
        print(
            f"  Gen {stats.generation:>3d}  "
            f"best={stats.best_fitness[0]:+.6f}  "
            f"mean={stats.mean_fitness[0]:+.6f}  "
            f"size={stats.best_program_size}"
        )

    print(f"Evolving sin(x) approximation  (pop={population_size}, gens={generations})")
    result = evolve(
        registry=registry,
        config=config,
        evaluator=evaluator,
        context=context,
        callback=_on_generation,
    )

    # -- post-processing -----------------------------------------------
    simplified = simplify(result.best_program)
    print(f"\nBest program size : {result.best_program.size}")
    print(f"Simplified size   : {simplified.size}")

    # Serialize / deserialize round-trip
    payload = serialize(simplified)
    restored = deserialize(payload, registry)
    assert restored.size == simplified.size, "round-trip size mismatch"
    print("Serialize → deserialize round-trip OK")

    return result


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_gp_basics()
