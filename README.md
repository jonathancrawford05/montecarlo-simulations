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
│   ├── simulation.py          # Single life simulation
│   └── joint_life.py          # Joint life simulation with contagion
├── tests/                     # Unit tests
├── benchmarks/                # Performance benchmarks and calibration
├── docs/                      # Documentation
│   ├── calibration.md         # Performance calibration guide
│   └── joint_life.md          # Joint life modeling guide
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
