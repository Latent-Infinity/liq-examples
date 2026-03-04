"""Smoke tests for GP and Evolution example modules."""

from __future__ import annotations

from liq.examples.strategies.evolution_regime_presets import run_regime_presets_demo
from liq.examples.strategies.evolution_seeds import run_seed_catalog_demo
from liq.examples.strategies.evolution_two_stage import run_two_stage_demo
from liq.examples.strategies.gp_basics import run_gp_basics


def test_gp_basics_smoke() -> None:
    """run_gp_basics completes and returns a valid EvolutionResult."""
    result = run_gp_basics(population_size=50, generations=3, seed=42)
    assert result.best_program is not None
    assert len(result.fitness_history) == 3
    assert result.best_program.size >= 1


def test_evolution_seeds_smoke() -> None:
    """run_seed_catalog_demo completes without error."""
    run_seed_catalog_demo(population_size=50, generations=3, seed=42)


def test_evolution_two_stage_smoke() -> None:
    """run_two_stage_demo completes without error."""
    run_two_stage_demo(population_size=50, generations=3, seed=42)


def test_evolution_regime_presets_smoke() -> None:
    """run_regime_presets_demo completes without error."""
    run_regime_presets_demo("baseline")
