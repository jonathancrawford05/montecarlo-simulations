# Confidence Analysis Methodology

This document describes the statistical foundations of the confidence analysis tools in `mortality_simulations`, covering the analytical model, the Cornish-Fisher expansion, pilot stability, and guidance on choosing between the planning and validation workflows.

## The Problem

Monte Carlo simulation produces *estimates* of distributional quantities like the 95th percentile of total claim volume. These estimates carry sampling uncertainty: run the simulation again with a different random seed and you get a different answer. The confidence analysis tools quantify this uncertainty and answer the question: **how many simulations are needed for a given level of precision?**

## Analytical Model

### Portfolio as a Weighted Poisson Binomial

Each life *i* in the portfolio is modelled as an independent Bernoulli trial:

```
D_i ~ Bernoulli(qx_i)       (death indicator)
```

The total claim volume for one simulation trial is:

```
S = sum(v_i * D_i)           (aggregate claim volume)
```

where `v_i` is the claim volume and `qx_i` is the mortality rate for life *i*. Since the `qx_i` values differ across lives, this is a *weighted Poisson binomial* — a sum of independent but non-identically distributed Bernoulli random variables, each weighted by its volume.

### Exact Moments

The moments of *S* can be computed in closed form, with no simulation required:

| Moment | Formula | Interpretation |
|--------|---------|----------------|
| Mean | `mu = sum(v_i * qx_i)` | Expected total claims |
| Variance | `sigma^2 = sum(v_i^2 * qx_i * (1 - qx_i))` | Spread of total claims |
| Third central moment | `mu_3 = sum(v_i^3 * qx_i * (1-qx_i) * (1-2*qx_i))` | Asymmetry |
| Skewness | `gamma = mu_3 / sigma^3` | Standardised asymmetry |
| Coefficient of variation | `CV = sigma / mu` | Relative spread |

These are implemented in `compute_portfolio_moments()`.

### Why Heterogeneity Matters

Consider two portfolios with 1,000 lives each, both with the same average `qx`:

- **Uniform**: All lives at `qx = 0.02`, all volumes at 5M
- **Concentrated**: 10 "whale" policies at 100-500M, varied `qx` from 0.001 to 0.06

Both have similar means, but the concentrated portfolio has:

- **Higher CV** (1.15 vs 0.22) due to volume concentration in `sum(v_i^2 * ...)`
- **Higher skewness** (1.72 vs 0.22) due to the `v_i^3` term
- **Wider CI** at any given simulation count (5.4% vs 1.5% at 10K sims)

The analytical approach captures this because the variance and skewness formulae use the individual `v_i` and `qx_i` values, not just their averages.

## Quantile Estimation

### Normal Approximation

By the Central Limit Theorem, for large portfolios *S* is approximately normal:

```
S ~ N(mu, sigma^2)
```

The *p*-th quantile under this approximation is:

```
xi_p = mu + z_p * sigma
```

where `z_p = Phi^{-1}(p)` is the standard normal quantile.

### Cornish-Fisher Expansion

For skewed distributions (positive skewness is typical for mortality portfolios with low `qx`), the normal approximation underestimates the upper tail. The Cornish-Fisher expansion corrects for this:

```
xi_p ≈ mu + sigma * w_p

where:  w_p = z_p + (z_p^2 - 1) * gamma / 6
```

This shifts the quantile estimate upward when `gamma > 0` and `z_p > 0` (upper tail). For the 95th percentile with typical mortality portfolio skewness (0.1-0.5), this correction is usually 0.5-3% of the quantile value.

This is implemented in `estimate_quantile_ci_width()`.

## Confidence Interval for the Sample Quantile

### Standard Error

The standard error of the sample *p*-th quantile from *n* iid simulation draws is:

```
SE(xi_hat_p) = sqrt(p * (1-p) / n) / f(xi_p)
```

where `f(xi_p)` is the probability density at the quantile point. Under the Cornish-Fisher corrected distribution:

```
f_CF(xi_p) = phi(z_p) / (sigma * (1 + 2 * z_p * gamma / 6))
```

Combining these:

```
SE(xi_hat_p) = sqrt(p * (1-p) / n) * sigma * (1 + 2 * z_p * gamma / 6) / phi(z_p)
```

### Confidence Interval Width

The 95% CI width for the quantile is:

```
CI_width = 2 * z_0.975 * SE(xi_hat_p)
```

And as a percentage of the quantile estimate:

```
CI_width_relative = CI_width / xi_p * 100
```

### Scaling Law

Since `SE ∝ 1/sqrt(n)`, the CI width scales as `1/sqrt(n)`:

- To halve the CI width, quadruple the simulations
- To achieve 10x tighter CI, use 100x more simulations

This scaling is exact asymptotically and holds regardless of the underlying distribution.

## Empirical Confidence Interval (Order Statistics)

For existing simulation results, the CI for the *p*-th quantile uses the exact binomial method. Given *n* sorted simulation outcomes `X_(1), ..., X_(n)`, find order statistics `X_(j)` and `X_(k)` such that:

```
P(X_(j) <= xi_p <= X_(k)) >= 1 - alpha
```

The indices *j* and *k* are found via:

```
j = BinomialPPF(alpha/2, n, p)
k = BinomialPPF(1 - alpha/2, n, p)
```

This is **distribution-free** — it makes no assumptions about the shape of the aggregate claim distribution. It is implemented in `analyze_simulation_confidence()`.

## Pilot Stability

### The Problem

`estimate_required_simulations()` measures the empirical CI width from a pilot run and extrapolates to other simulation counts. But the pilot CI width is itself a random variable — it depends on which particular draws you got.

### Empirical Analysis

We ran 20 replicates at each pilot size for a 1,000-life portfolio, targeting 1% CI width:

| Pilot Size | CV of Estimated n | Range of Estimated n |
|------------|-------------------|---------------------|
| 100 | 60.5% | 9,680 - 90,426 |
| 500 | 46.2% | 4,494 - 45,350 |
| 1,000 | 31.3% | 10,327 - 33,236 |
| 5,000 | 23.7% | 10,001 - 25,067 |
| 10,000 | 20.7% | 9,171 - 23,348 |

The mean converges by ~1,000, but individual estimates can vary by 2-3x until the pilot reaches 5,000+.

### Recommendations

| Pilot Size | Stability | Guidance |
|------------|-----------|----------|
| < 1,000 | Low | Warning issued. Use `plan_simulation_count()` instead |
| 1,000 - 4,999 | Moderate | Usable for rough planning. Results ± ~30% |
| >= 5,000 | High | Reliable extrapolation. Results ± ~20% |

`estimate_required_simulations()` automatically classifies the pilot and warns when below 1,000.

### Why `plan_simulation_count()` Is Preferred for Planning

`plan_simulation_count()` uses the analytical moments directly. It is:

- **Deterministic**: Same portfolio, same answer every time
- **Free**: No simulation runs needed
- **Exact** (up to CLT + Cornish-Fisher approximation): Uses the full portfolio structure

The pilot-based approach is better suited for **validation** — confirming that a completed simulation run achieved the desired precision.

## Two Workflows

### 1. Planning (Before Simulation)

```
Portfolio data (volumes, qx)
    |
    v
compute_portfolio_moments()  -->  exact moments
    |
    v
plan_simulation_count()  -->  recommended n for target CI width
    |
    v
stochastic_runs_hybrid(n_trials=recommended_n)
```

### 2. Validation (After Simulation)

```
Simulation results
    |
    v
analyze_simulation_confidence()  -->  empirical CI from order statistics
    |
    v
generate_confidence_summary(portfolio_data=...)  -->  empirical + analytical comparison
    |
    v
print_confidence_summary()  -->  formatted report
```

## Limitations

1. **CLT approximation**: The normal/Cornish-Fisher approach assumes the portfolio is large enough for the CLT to apply. For very small portfolios (< 50 lives) or extreme `qx` values, the approximation may be poor.

2. **Independence assumption**: The model assumes independent lives. Correlated mortality (e.g., pandemic events, joint life policies) would increase the true variance beyond what the analytical formulae predict. For joint life policies, use the `joint_life` module which models correlation explicitly.

3. **Single-period model**: The Bernoulli assumption applies to a single projection period. Multi-period projections with decremented portfolios would need separate analysis per period.

4. **Skewness only**: The Cornish-Fisher expansion used here corrects for skewness (third moment) but not kurtosis (fourth moment). For extremely heavy-tailed portfolios, a higher-order expansion or bootstrap methods may be more appropriate.
