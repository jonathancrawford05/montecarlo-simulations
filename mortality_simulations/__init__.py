"""
Mortality Monte Carlo Simulations package.

This package provides mortality simulation tools with two execution backends:

1. **Local (multiprocessing)** — Default, runs on a single machine.
   Import directly from this package::

       from mortality_simulations import stochastic_runs_hybrid

2. **Distributed (Spark)** — For cluster execution on Databricks/Spark.
   Import from the spark_simulation module::

       from mortality_simulations.spark_simulation import stochastic_runs_spark

The Spark backend requires PySpark and is not imported by default to avoid
adding PySpark as a required dependency.
"""

from mortality_simulations.simulation import (
    analyze_simulation_confidence,
    check_threading_config,
    compute_portfolio_moments,
    estimate_quantile_ci_width,
    estimate_required_simulations,
    generate_confidence_summary,
    get_optimal_params,
    plan_simulation_count,
    print_confidence_summary,
    stochastic_runs_hybrid,
)

__version__ = "0.1.0"
__all__ = [
    # Simulation — local (multiprocessing)
    "stochastic_runs_hybrid",
    "get_optimal_params",
    "check_threading_config",
    # Confidence analysis — planning (analytical, no simulation needed)
    "plan_simulation_count",
    "compute_portfolio_moments",
    "estimate_quantile_ci_width",
    # Confidence analysis — validation (from simulation results)
    "analyze_simulation_confidence",
    "estimate_required_simulations",
    "generate_confidence_summary",
    "print_confidence_summary",
]


def _spark_available() -> bool:
    """Check if PySpark is available for import."""
    try:
        import pyspark  # noqa: F401
        return True
    except ImportError:
        return False
