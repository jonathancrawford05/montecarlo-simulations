# Mortality Monte Carlo Simulations

A high-performance Python package for running Monte Carlo mortality simulations using hybrid parallel processing with vectorized batch operations.

## Overview

This project implements stochastic mortality simulations designed to efficiently process large datasets (millions of rows) by combining:

- **Multiprocessing**: Distributes trials across CPU cores
- **Vectorized batching**: NumPy-based batch operations within each worker
- **Auto-tuning**: Automatically selects optimal parameters based on data size
- **Joint life modeling**: Supports last-to-die policies with mortality correlation and contagion

## Project Structure

```
mortality_simulations/
├── mortality_simulations/     # Main package
│   ├── __init__.py
│   ├── joint_life.py          # Joint life simulation with contagion
│   └── simulation.py          # Core simulation + confidence analysis
├── tests/                     # Unit tests
├── benchmarks/                # Performance benchmarks and calibration
├── examples/                  # Jupyter notebooks
│   └── confidence_analysis.ipynb
├── docs/                      # Documentation
│   ├── joint_life.md          # Joint life modeling guide
│   ├── calibration.md         # Calibration guide and benchmark results
│   └── confidence_analysis.md # Statistical background and methodology
├── data/                      # Sample data (gitignored)
├── pyproject.toml             # Poetry configuration
└── README.md
```

## Installation

### Prerequisites

- Python 3.10+
- Poetry

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd montecarlo-simulations

# Install dependencies with Poetry
poetry install

# Activate the virtual environment
poetry shell
```

## Usage

### Basic Usage (Recommended)

The simulation automatically selects optimal parameters based on your data size:

```python
import pandas as pd
from mortality_simulations import stochastic_runs_hybrid

# Prepare your data
data = pd.DataFrame({
    'volume': [...],           # Claim volumes
    'baseline_qx': [...],      # Baseline mortality rates
    'shocked_qx': [...],       # Shocked mortality rates
})

# Run simulations - auto-tuning selects optimal parameters
results = stochastic_runs_hybrid(
    data=data,
    n_trials=1000,
    volume_col='volume',
    baseline_qx_col='baseline_qx',
    shocked_qx_col='shocked_qx',
)

# Results contain:
# - claim_volume_baseline / claim_volume_shocked
# - claim_count_baseline / claim_count_shocked
# - claim_count_baseline_10PLUS / claim_count_shocked_10PLUS
# - volume_baseline_10PLUS / volume_shocked_10PLUS
```

### Manual Parameter Override

For fine-grained control, you can specify parameters explicitly:

```python
results = stochastic_runs_hybrid(
    data=data,
    n_trials=1000,
    volume_col='volume',
    baseline_qx_col='baseline_qx',
    shocked_qx_col='shocked_qx',
    n_processes=2,      # Override auto-selection
    batch_size=25,      # Override auto-selection
)
```

### Multi-Year Projection

Provide per-year qx columns and (optionally) per-year volumes. The simulator
returns total losses plus per-year breakdowns via `*_by_year` keys.

```python
from mortality_simulations import stochastic_runs_multi_year

data = pd.DataFrame({
    "volume": [...],
    "baseline_qx_1": [...],
    "baseline_qx_2": [...],
    "baseline_qx_3": [...],
    "shocked_qx_1": [...],
    "shocked_qx_2": [...],
    "shocked_qx_3": [...],
})

results = stochastic_runs_multi_year(
    data=data,
    n_trials=1000,
    volume_col="volume",
    baseline_qx_cols=["baseline_qx_1", "baseline_qx_2", "baseline_qx_3"],
    shocked_qx_cols=["shocked_qx_1", "shocked_qx_2", "shocked_qx_3"],
    n_processes="auto",
)
```

### Inspecting Optimal Parameters

```python
from mortality_simulations import get_optimal_params

# Get recommended parameters for your data size
n_processes, batch_size = get_optimal_params(n_rows=1_000_000)
print(f"Recommended: {n_processes} processes, batch_size={batch_size}")
```

## Auto-Tuning

The package includes calibrated defaults based on benchmarks run on a 10-core MacBook:

| Data Size | n_processes | batch_size | Rationale |
|-----------|-------------|------------|-----------|
| < 500K rows | 1 | 50 | Multiprocessing overhead exceeds benefit |
| 500K - 5M rows | 2 | 25 | Parallelism helps, memory manageable |
| > 5M rows | 2 | 10 | Memory-constrained, small batches critical |

### Key Insights

- **Small data**: Single process is faster because multiprocessing overhead (process spawning, data serialization) exceeds computation time
- **Large data**: Memory pressure dominates; smaller batch sizes prevent swap thrashing
- **More cores ≠ faster**: Beyond 2 processes, coordination overhead and memory bandwidth become bottlenecks

For detailed benchmark results and guidance on calibrating for your system, see [docs/calibration.md](docs/calibration.md).

## Joint Life Simulation

For policies covering two lives with "last-to-die" payout (claim triggers when both lives have died):

```python
from mortality_simulations import stochastic_runs_joint_life

# Data must have paired lives (A and B) per policy
results = stochastic_runs_joint_life(
    data=joint_life_data,
    n_trials=1000,
    policy_col='policy_number',
    life_id_col='life_id',          # 'A' or 'B'
    volume_col='volume',
    baseline_qx_col='baseline_qx',
    shocked_qx_col='shocked_qx',
    rho=0.15,                       # Mortality correlation (Gaussian copula)
    contagion_multiplier=1.5,       # Survivor's qx increases when partner dies
)
```

### Mixed Portfolio (Single + Joint Lives)

```python
from mortality_simulations import stochastic_runs_portfolio

results = stochastic_runs_portfolio(
    data=mixed_data,
    n_trials=1000,
    volume_col='volume',
    baseline_qx_col='baseline_qx',
    shocked_qx_col='shocked_qx',
    life_status_col='life_status',  # 'single' or 'joint'
    policy_col='policy_number',
    life_id_col='life_id',
    rho=0.15,
    contagion_multiplier=1.5,
)
# Returns combined results plus separate single_life_* and joint_life_* breakdowns
```

For detailed documentation on joint life modeling, correlation estimation, and contagion effects, see [docs/joint_life.md](docs/joint_life.md).

## Confidence Analysis

The package includes tools for quantifying uncertainty in simulation results — specifically, how confident you can be in tail quantiles (e.g., the 95th percentile) used for risk management.

There are two workflows:

### Planning: How many simulations do I need?

Use `plan_simulation_count` **before** running simulations. It computes exact analytical moments from the portfolio data (no simulation required) and projects CI widths for different simulation counts:

```python
from mortality_simulations import plan_simulation_count

plan = plan_simulation_count(
    data["volume"].values,
    data["shocked_qx"].values,
    target_ci_width_relative=1.0,  # Target: 1% CI width for the 95th pctile
)

print(f"Recommended: {plan['recommended_n']:,} simulations")
for s in plan["scenarios"]:
    print(f"  {s['n_simulations']:>10,} sims -> CI width {s['ci_width_relative']:.2f}%")
```

This is deterministic — same portfolio, same answer every time. It accounts for heterogeneous mortality rates and volume concentration via the Cornish-Fisher expansion. See [docs/confidence_analysis.md](docs/confidence_analysis.md) for the statistical methodology.

### Validation: Did my simulation achieve the target?

After running simulations, use `generate_confidence_summary` to verify the precision achieved, optionally comparing against the analytical projection:

```python
from mortality_simulations import (
    stochastic_runs_hybrid,
    generate_confidence_summary,
    print_confidence_summary,
)

results = stochastic_runs_hybrid(data, n_trials=100_000, ...)

summary = generate_confidence_summary(
    results,
    "claim_volume_shocked",
    portfolio_data={
        "volumes": data["volume"].values,
        "qx": data["shocked_qx"].values,
    },
)
print_confidence_summary(summary)
```

This prints a side-by-side comparison of empirical and analytical CI widths at 10K, 100K, and 1M simulations, with use-case recommendations.

### Function Reference

| Function | Purpose |
|----------|---------|
| `plan_simulation_count` | **Planning.** Deterministic simulation count recommendation from portfolio data |
| `compute_portfolio_moments` | Exact mean, variance, skewness of the aggregate claim distribution |
| `estimate_quantile_ci_width` | Analytical CI width for a quantile at any simulation count |
| `analyze_simulation_confidence` | Empirical CI for a quantile from simulation results (order statistics) |
| `estimate_required_simulations` | Extrapolate required n from a pilot run (warns if pilot < 1,000) |
| `generate_confidence_summary` | Full report with projections and recommendations |
| `print_confidence_summary` | Formatted console output |

For worked examples, see [examples/confidence_analysis.ipynb](examples/confidence_analysis.ipynb).

## Development

### Running Tests

```bash
poetry run pytest
```

### Running Benchmarks

```bash
# Full calibration suite
poetry run python benchmarks/benchmark_simulation.py
```

## License

[Add your license here]
