# Dynamic Portfolio Optimization: LP, MILP, & Stochastic

## Overview

This project presents a comprehensive out-of-sample comparison of three portfolio optimization paradigms applied to a high-dimensional universe of 8,405 global assets (equities, commodities, and currency pairs). The study evaluates trade-offs between structural concentration and uncertainty management over a volatile testing period (2022–2025).

## Methodology

All models utilize Mean Absolute Deviation (MAD) as the risk measure:

- Linear Programming (LP): Unconstrained deterministic optimization.

- Mixed-Integer Linear Programming (MILP): Cardinality constrained (Limited to max 10 assets).

- Stochastic Optimization: Scenario-based approach using bootstrapped data.

## Key Findings

- Best Efficiency (MILP): The MILP strategy achieved the highest Sharpe Ratio (5.28) with a stable 5.84% return. Constraints acted as effective noise filters.

- Highest Returns (Stochastic): The Stochastic model yielded the highest return (14.61%) but suffered from high volatility due to crypto exposure.

- Dynamic vs. Static: Static implementations failed (0% to -22% returns). Dynamic rebalancing was proven essential for profitability in non-stationary markets.
