# Dynamic Portfolio Optimization: LP vs. MILP vs. Stochastic

## Overview
This project presents a comprehensive, out-of-sample comparison of three Mean Absolute Deviation (MAD) optimization paradigms applied to a high-dimensional global asset universe. 

The study tests these strategies on **8,405 assets** (equities, commodities, crypto-pairs) over a volatile testing period (2022–2025), aiming to determine the most effective method for managing risk and return in large-scale financial markets.

## Methodologies
The project implements and backtests three distinct optimization models:

1.  **Linear Programming (LP):** Unconstrained MAD minimization focusing on diversification.
2.  **Mixed-Integer Linear Programming (MILP):** Adds cardinality constraints ($K \le 10$) to enforce sparsity and reduce transaction costs.
3.  **Stochastic Optimization:** Minimizes Expected MAD across bootstrapped scenarios to account for parameter uncertainty.

> **Note:** All strategies utilize a **Dynamic Rebalancing** framework (rolling window). The study proved that static implementations failed significantly (0% to -22% returns) in this environment.

## Key Performance Results
The results reveal a distinct trade-off between structural concentration (MILP) and uncertainty management (Stochastic).

| Strategy | Annualized Return | Annualized MAD | Sharpe Ratio | Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **Dynamic MILP** | 5.84% | 0.54% | **5.28** | **Most Efficient.** Constraints acted as noise filters, selecting stable assets. |
| **Dynamic LP** | 3.98% | **0.38%** | 4.76 | **Safest.** Achieved the lowest absolute risk through broad diversification. |
| **Dynamic Stochastic** | **14.61%** | 1.90% | 1.47 | **Highest Return.** Aggressively targeted high-volatility assets (e.g., Crypto). |

## Core Insights
* **Constraints as Filters:** Contrary to standard theory, the cardinality constraint in **MILP** improved risk-adjusted performance by filtering out the "noise" of thousands of marginal assets.
* **Volatility in Stochastic Models:** While the **Stochastic** model captured the highest upside (driven by crypto exposure), it suffered from overfitting to high-variance scenarios, resulting in the lowest Sharpe ratio.
* **Adaptivity is Critical:** Dynamic rebalancing is essential. Static portfolios collapsed under out-of-sample volatility.

## Mathematical Foundation
The core objective function for the LP formulation minimizes the Mean Absolute Deviation:

$$\text{min } \frac{1}{T} \sum_{t=1}^{T} \big| R_{p,t} - \mu_p \big|$$

Where $R_{p,t}$ is the portfolio return at time $t$ and $\mu_p$ is the mean portfolio return.
