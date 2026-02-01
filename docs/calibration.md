# Performance Calibration Guide

This document explains the performance characteristics of the mortality simulation, presents benchmark results, and provides guidance for calibrating on your own system.

## Overview

The `stochastic_runs_hybrid` function uses a hybrid parallelization strategy:

1. **Multiprocessing**: Distributes Monte Carlo trials across CPU cores
2. **Vectorized batching**: Each worker processes trials in NumPy-vectorized batches

The optimal configuration depends on three competing factors:

| Factor | Small Data | Large Data |
|--------|------------|------------|
| **Multiprocessing overhead** | Dominates | Negligible |
| **Computation time** | Fast | Slow |
| **Memory pressure** | Low | High |

## Benchmark Results

### Test Environment

- **Hardware**: MacBook with 10-core Apple Silicon
- **Memory**: 16GB unified memory
- **Python**: 3.11
- **NumPy**: 1.26.x

### Results by Data Size

#### 10,000 rows (Small)

| Processes | Batch Size | Time (s) | Trials/s |
|-----------|------------|----------|----------|
| **1** | **50** | **0.29** | **341.0** |
| 1 | 25 | 0.30 | 337.7 |
| 2 | 100 | 0.39 | 257.8 |
| 4 | 25 | 0.54 | 183.5 |
| 10 | 25 | 0.93 | 108.0 |

**Finding**: Single process is 3x faster than using all 10 cores. Multiprocessing overhead (process spawning, data serialization, IPC) exceeds the computation time.

#### 100,000 rows (Small-Medium)

| Processes | Batch Size | Time (s) | Trials/s |
|-----------|------------|----------|----------|
| **1** | **50** | **0.38** | **260.2** |
| 1 | 25 | 0.40 | 251.7 |
| 2 | 50 | 0.48 | 208.5 |
| 4 | 100 | 0.59 | 168.1 |
| 10 | 50 | 1.00 | 99.9 |

**Finding**: Still overhead-dominated. Single process remains optimal.

#### 1,000,000 rows (Medium)

| Processes | Batch Size | Time (s) | Trials/s |
|-----------|------------|----------|----------|
| **2** | **25** | **1.02** | **98.5** |
| 4 | 50 | 1.02 | 97.7 |
| 2 | 100 | 1.04 | 96.0 |
| 1 | 25 | 1.30 | 77.2 |
| 10 | 25 | 1.48 | 67.8 |

**Finding**: Parallelism finally helps (~28% speedup with 2 processes). But 10 processes is slower than 1 due to coordination overhead.

#### 20,000,000 rows (Large)

| Processes | Batch Size | Time (s) | Trials/s |
|-----------|------------|----------|----------|
| **2** | **10** | **14.84** | **6.7** |
| 1 | 10 | 24.22 | 4.1 |
| 1 | 25 | 26.84 | 3.7 |
| 1 | 50 | 38.39 | 2.6 |
| 2 | 25 | 40.96 | 2.4 |
| 10 | 10 | 64.26 | 1.6 |
| 1 | 100 | 134.54 | 0.7 |

**Finding**: Memory pressure dominates. Larger batch sizes cause severe performance degradation (9x slower with batch_size=100 vs 10). More processes hurt because each needs memory for its batches.

## Analysis

### Why More Cores ≠ Faster

Three factors limit parallelization benefits:

1. **Process overhead**: Spawning processes, serializing/deserializing NumPy arrays via pickle, and inter-process communication all take time.

2. **Memory bandwidth**: All cores share the same memory bus. At large scales, cores compete for memory access rather than compute.

3. **NumPy efficiency**: NumPy already uses optimized BLAS/LAPACK libraries that may internally parallelize operations.

### Memory Usage Model

Each batch allocates:

```
Memory per batch ≈ n_rows × batch_size × (8 bytes for float64 + 4 bytes for booleans)
                 ≈ n_rows × batch_size × 12 bytes
```

For 20M rows with batch_size=50:
```
20,000,000 × 50 × 12 = 12 GB per batch
```

With 2 processes, peak memory approaches 24GB, exceeding 16GB RAM and causing swap thrashing.

### Calibrated Defaults

Based on the benchmarks, we selected these thresholds:

| Data Size | n_processes | batch_size | Rationale |
|-----------|-------------|------------|-----------|
| < 500K rows | 1 | 50 | Overhead exceeds parallelism benefit |
| 500K - 5M rows | 2 | 25 | Sweet spot: parallelism helps, memory OK |
| > 5M rows | 2 | 10 | Memory-constrained: small batches critical |

## Calibrating for Your System

### When to Re-calibrate

Re-run benchmarks if your system differs significantly:

- Different CPU architecture (Intel vs Apple Silicon vs AMD)
- Different core count
- Different RAM amount (especially if < 16GB or > 64GB)
- Different workload characteristics

### Running the Benchmark Suite

```bash
# Full calibration suite
poetry run python benchmarks/benchmark_simulation.py

# Custom data sizes (edit the script)
# Modify row_sizes in run_calibration_suite()
```

### Interpreting Results

Look for these patterns:

1. **Overhead crossover**: The data size where 2 processes first beats 1 process
2. **Memory wall**: Where larger batch sizes start hurting performance
3. **Diminishing returns**: Where adding more processes stops helping

### Adjusting Thresholds

Edit `mortality_simulations/simulation.py`:

```python
def get_optimal_params(n_rows: int) -> tuple[int, int]:
    if n_rows < YOUR_SMALL_THRESHOLD:
        return (1, YOUR_SMALL_BATCH)
    elif n_rows < YOUR_MEDIUM_THRESHOLD:
        return (YOUR_MEDIUM_PROCESSES, YOUR_MEDIUM_BATCH)
    else:
        return (YOUR_LARGE_PROCESSES, YOUR_LARGE_BATCH)
```

### System-Specific Recommendations

#### High-Memory Systems (64GB+)

You may be able to use larger batch sizes at scale:

```python
# Example for 64GB system
if n_rows < 500_000:
    return (1, 50)
elif n_rows < 10_000_000:
    return (4, 50)  # More parallelism, larger batches
else:
    return (4, 25)  # Still comfortable with 25
```

#### Low-Memory Systems (8GB)

Use more conservative batch sizes:

```python
# Example for 8GB system
if n_rows < 500_000:
    return (1, 25)
elif n_rows < 2_000_000:
    return (2, 10)
else:
    return (1, 5)  # Minimize memory, sacrifice parallelism
```

#### Many-Core Systems (32+ cores)

Test whether more processes help at medium scales:

```python
# Example for 32-core workstation
if n_rows < 500_000:
    return (1, 50)
elif n_rows < 5_000_000:
    return (4, 25)  # Try 4 processes
else:
    return (4, 10)
```

## Advanced Tuning

### Manual Override

You can always specify parameters explicitly:

```python
results = stochastic_runs_hybrid(
    data=df,
    n_trials=1000,
    volume_col="volume",
    baseline_qx_col="baseline_qx",
    shocked_qx_col="shocked_qx",
    n_processes=4,    # Override auto
    batch_size=15,    # Override auto
)
```

### Profiling Memory Usage

Use `memory-profiler` to monitor actual memory consumption:

```bash
poetry run python -m memory_profiler your_script.py
```

Or in code:

```python
from memory_profiler import profile

@profile
def run_simulation():
    results = stochastic_runs_hybrid(...)
```

### Monitoring During Runs

Watch system resources during long simulations:

```bash
# macOS
top -pid $(pgrep -f python)

# Linux
htop -p $(pgrep -f python | tr '\n' ',')
```

## Troubleshooting

### Simulation is Slow

1. Check batch size isn't too large for available RAM
2. Verify you're not running other memory-intensive applications
3. Try reducing `n_processes` if memory is constrained

### Out of Memory Errors

1. Reduce `batch_size` (try 5 or even 1)
2. Reduce `n_processes` to 1
3. Process data in chunks if necessary

### Inconsistent Timing

1. Ensure no background processes are competing for resources
2. Run benchmarks multiple times and take the median
3. Check for thermal throttling on laptops
