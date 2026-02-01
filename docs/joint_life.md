# Joint Life Modeling Guide

This guide covers the joint life mortality simulation with contagion effects, including the mathematical framework, parameter estimation, and usage examples.

## Overview

Joint life policies cover two lives (typically spouses) with "last-to-die" payout - the claim triggers only when **both** lives have died. This creates:

1. **Lower claim frequency**: P(both die) < P(either dies)
2. **Mortality correlation**: Deaths may be correlated (shared environment, lifestyle)
3. **Contagion effect**: When one dies, the survivor's mortality increases ("widow/widower effect")

## Mathematical Framework

### Gaussian Copula

We model mortality correlation using a **Gaussian copula**:

1. Generate correlated standard normals:
   ```
   Z_A ~ N(0,1)
   Z_B = ρ * Z_A + √(1-ρ²) * Z_independent
   ```

2. Transform to correlated uniforms via normal CDF:
   ```
   U_A = Φ(Z_A)
   U_B = Φ(Z_B)
   ```

3. Apply mortality:
   ```
   Life A dies if: U_A < qx_A
   Life B dies if: U_B < qx_B
   ```

The correlation parameter **ρ** (rho) controls the dependency:
- ρ = 0: Independent deaths
- ρ > 0: Positive correlation (contagion - deaths tend to occur together)
- ρ < 0: Negative correlation (rare in mortality)

### Last-to-Die Logic

For a joint life policy:
```python
claim_triggers = life_A_dead AND life_B_dead
```

Possible outcomes per trial:
- Both alive → No claim
- A dead, B alive → No claim (yet)
- A alive, B dead → No claim (yet)
- Both dead → **Claim**

### Contagion Mortality Adjustment

When one life dies, the survivor's mortality rate increases:

```python
qx_survivor_adjusted = qx_survivor * contagion_multiplier
```

Literature suggests:
- Immediate effect (0-6 months): multiplier ≈ 1.5-2.0
- Medium term (6-24 months): multiplier ≈ 1.2-1.5
- Long term (2+ years): multiplier ≈ 1.0-1.1

## Usage

### Basic Joint Life Simulation

```python
import pandas as pd
from mortality_simulations import stochastic_runs_joint_life

# Prepare data with paired lives
data = pd.DataFrame({
    'policy_number': ['POL1', 'POL1', 'POL2', 'POL2'],
    'life_id': ['A', 'B', 'A', 'B'],
    'volume': [1_000_000, 1_000_000, 2_000_000, 2_000_000],
    'baseline_qx': [0.02, 0.015, 0.03, 0.025],
    'shocked_qx': [0.04, 0.03, 0.06, 0.05],
})

results = stochastic_runs_joint_life(
    data=data,
    n_trials=1000,
    policy_col='policy_number',
    life_id_col='life_id',
    volume_col='volume',
    baseline_qx_col='baseline_qx',
    shocked_qx_col='shocked_qx',
    rho=0.15,                  # Moderate correlation
    contagion_multiplier=1.5,  # 50% mortality increase for survivor
)

# Results include:
# - claim_volume_baseline/shocked: Total claim amounts (last-to-die)
# - claim_count_baseline/shocked: Number of claims (both lives dead)
# - both_alive_baseline/shocked: Policies where both survive
# - one_dead_baseline/shocked: Policies where exactly one died
```

### Mixed Portfolio (Single + Joint Lives)

```python
from mortality_simulations import stochastic_runs_portfolio

# Data with mixed life types
data = pd.DataFrame({
    'policy_number': ['S1', 'S2', 'J1', 'J1', 'J2', 'J2'],
    'life_id': ['A', 'A', 'A', 'B', 'A', 'B'],
    'life_status': ['single', 'single', 'joint', 'joint', 'joint', 'joint'],
    'volume': [500_000, 750_000, 1_000_000, 1_000_000, 2_000_000, 2_000_000],
    'baseline_qx': [0.02, 0.025, 0.02, 0.015, 0.03, 0.025],
    'shocked_qx': [0.04, 0.05, 0.04, 0.03, 0.06, 0.05],
})

results = stochastic_runs_portfolio(
    data=data,
    n_trials=1000,
    volume_col='volume',
    baseline_qx_col='baseline_qx',
    shocked_qx_col='shocked_qx',
    life_status_col='life_status',
    policy_col='policy_number',
    life_id_col='life_id',
    rho=0.15,
    contagion_multiplier=1.5,
)

# Results include:
# - Combined totals: claim_volume_baseline, claim_count_baseline, etc.
# - Single life segment: single_life_claim_volume_baseline, etc.
# - Joint life segment: joint_life_claim_volume_baseline, etc.
```

## Parameter Estimation

### Estimating ρ (Correlation)

#### From Literature

Use published widow/widower effect studies:

| Hazard Ratio | Approximate ρ | Description |
|--------------|---------------|-------------|
| 1.0 | 0.00 | No correlation |
| 1.2 | 0.05 | Mild (shared environment) |
| 1.5 | 0.15 | Moderate (recommended default) |
| 2.0 | 0.30 | Strong (immediate widow effect) |
| 3.0 | 0.50 | Very strong |

#### From Observed Data

If you have joint policy outcomes:

```python
import numpy as np
from scipy.optimize import minimize_scalar
from mortality_simulations import generate_correlated_uniforms

def estimate_rho_from_data(observed_both_died, observed_qx_A, observed_qx_B):
    """
    Estimate rho from observed joint outcomes using MLE.

    Parameters:
    - observed_both_died: Array of booleans (True if both died)
    - observed_qx_A: Expected mortality for life A
    - observed_qx_B: Expected mortality for life B
    """
    n_policies = len(observed_both_died)
    p_both_observed = observed_both_died.mean()

    def objective(rho):
        # Simulate under this rho
        n_sim = 100_000
        U_A, U_B = generate_correlated_uniforms(n_policies, n_sim, rho)

        dead_A = U_A < observed_qx_A[:, np.newaxis]
        dead_B = U_B < observed_qx_B[:, np.newaxis]

        p_both_simulated = (dead_A & dead_B).mean()

        return (p_both_simulated - p_both_observed)**2

    result = minimize_scalar(objective, bounds=(-0.5, 0.5), method='bounded')
    return result.x
```

### Estimating Contagion Multiplier

From survival analysis on widow/widower data:

```python
from lifelines import CoxPHFitter

# Fit Cox model to estimate hazard ratio
cph = CoxPHFitter()
cph.fit(
    survival_data,
    duration_col='time_since_spouse_death',
    event_col='died',
    formula='spouse_died_recently'
)

hazard_ratio = np.exp(cph.params_['spouse_died_recently'])
# Use hazard_ratio directly as contagion_multiplier
```

## Data Requirements

### Input DataFrame Structure

| Column | Type | Description |
|--------|------|-------------|
| `policy_number` | str | Unique policy identifier |
| `life_id` | str | 'A' or 'B' for joint lives |
| `life_status` | str | 'single' or 'joint' |
| `volume` | float | Claim amount |
| `baseline_qx` | float | Baseline mortality rate (0-1) |
| `shocked_qx` | float | Stressed mortality rate (0-1) |

### Joint Life Pairing Rules

1. Each joint policy must have exactly 2 rows (life A and life B)
2. Both lives share the same `policy_number`
3. Policies with only one life are excluded (with warning)
4. Volume is taken from life A (assumed same for both)

## Performance Considerations

### Memory Usage

Joint life simulation uses the same batching strategy as single life:

```
Memory per batch ≈ n_policies × batch_size × 24 bytes
```

(Higher than single life due to paired uniform generation)

### Auto-tuning

The joint life simulation reuses the same auto-tuning thresholds:

| Policies | n_processes | batch_size |
|----------|-------------|------------|
| < 500K | 1 | 50 |
| 500K-5M | 2 | 25 |
| > 5M | 2 | 10 |

### Benchmarking

Run the joint life benchmark:

```bash
poetry run python benchmarks/benchmark_joint_life.py
```

## Sensitivity Analysis

### Correlation Impact

Test different ρ values to understand risk sensitivity:

```python
rho_values = [0.0, 0.1, 0.2, 0.3]
for rho in rho_values:
    results = stochastic_runs_joint_life(
        data=data,
        n_trials=1000,
        rho=rho,
        contagion_multiplier=1.0,  # Isolate correlation effect
        ...
    )
    print(f"ρ={rho}: Mean claims = {np.mean(results['claim_count_baseline']):.1f}")
```

### Contagion Impact

Test different contagion multipliers:

```python
multipliers = [1.0, 1.5, 2.0]
for mult in multipliers:
    results = stochastic_runs_joint_life(
        data=data,
        n_trials=1000,
        rho=0.15,
        contagion_multiplier=mult,
        ...
    )
    print(f"Multiplier={mult}: Mean claims = {np.mean(results['claim_count_baseline']):.1f}")
```

## Limitations

1. **Single-period model**: Current implementation assumes single-period mortality. For multi-period with time-varying contagion, extend the model.

2. **Symmetric contagion**: Both lives experience the same contagion multiplier. Could be extended for age/gender-specific effects.

3. **Simultaneous evaluation**: Deaths are evaluated simultaneously, not sequentially. For sequential modeling (first death triggers contagion for remainder of period), modify the worker function.

4. **Binary contagion**: Contagion is either on (partner dead) or off. For graded effects based on time-since-death, extend `apply_contagion()`.

## References

- Frees, E.W., Carriere, J., & Valdez, E. (1996). "Annuity Valuation with Dependent Mortality"
- Denuit, M., Dhaene, J., Le Bailly de Tilleghem, C., & Teghem, S. (2001). "Measuring the Impact of Dependence among Insured Lifelengths"
- Luciano, E., Spreeuw, J., & Vigna, E. (2008). "Modelling Stochastic Mortality for Dependent Lives"
