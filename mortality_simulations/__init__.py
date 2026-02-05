"""Mortality Monte Carlo Simulations package."""

from mortality_simulations.simulation import (
    analyze_simulation_confidence,
    check_threading_config,
    compute_portfolio_moments,
    estimate_quantile_ci_width,
    estimate_required_simulations,
    generate_confidence_summary,
    get_optimal_params,
    print_confidence_summary,
    stochastic_runs_hybrid,
)

__version__ = "0.1.0"
__all__ = [
    "stochastic_runs_hybrid",
    "get_optimal_params",
    "check_threading_config",
    "analyze_simulation_confidence",
    "estimate_required_simulations",
    "generate_confidence_summary",
    "print_confidence_summary",
    "compute_portfolio_moments",
    "estimate_quantile_ci_width",
]
