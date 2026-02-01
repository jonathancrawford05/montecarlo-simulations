# Mortality Monte Carlo Simulations

A high-performance Python package for running Monte Carlo mortality simulations using hybrid parallel processing with vectorized batch operations.

## Overview

This project implements stochastic mortality simulations designed to efficiently process large datasets (millions of rows) by combining:

- **Multiprocessing**: Distributes trials across CPU cores
- **Vectorized batching**: NumPy-based batch operations within each worker
- **Memory-efficient design**: Configurable batch sizes to manage memory usage

## Project Structure

```
mortality_simulations/
├── mortality_simulations/     # Main package
│   ├── __init__.py
│   └── simulation.py          # Core simulation functions
├── tests/                     # Unit tests
├── benchmarks/                # Performance benchmarks and calibration
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

```python
import pandas as pd
from mortality_simulations import stochastic_runs_hybrid

# Prepare your data
data = pd.DataFrame({
    'volume': [...],           # Claim volumes
    'baseline_qx': [...],      # Baseline mortality rates
    'shocked_qx': [...],       # Shocked mortality rates
})

# Run simulations
results = stochastic_runs_hybrid(
    data=data,
    n_trials=1000,
    volume_col='volume',
    baseline_qx_col='baseline_qx',
    shocked_qx_col='shocked_qx',
    n_processes=None,          # Uses all available cores
    batch_size=50,             # Adjust based on memory constraints
)

# Results contain:
# - claim_volume_baseline / claim_volume_shocked
# - claim_count_baseline / claim_count_shocked
# - claim_count_baseline_10PLUS / claim_count_shocked_10PLUS
# - volume_baseline_10PLUS / volume_shocked_10PLUS
```

## Configuration

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_processes` | `cpu_count()` | Number of parallel workers |
| `batch_size` | 50 | Trials per batch (lower = less memory) |

### Memory Guidelines

For large datasets, adjust `batch_size` based on available RAM:

- **20M rows, batch_size=50**: ~8GB per worker
- **20M rows, batch_size=25**: ~4GB per worker

## Development

### Running Tests

```bash
poetry run pytest
```

### Running Benchmarks

```bash
poetry run pytest benchmarks/ --benchmark-only
```

## License

[Add your license here]
