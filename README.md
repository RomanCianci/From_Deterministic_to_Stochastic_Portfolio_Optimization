# From Deterministic to Stochastic portfolio Optimization

![Build Status](https://img.shields.io/badge/status-ready-blue)
![License](https://img.shields.io/badge/license-MIT-blue)
![Methodology](https://img.shields.io/badge/method-LP%20%7C%20MILP%20%7C%20Stochastic-orange)

This repository contains a comparative analysis of portfolio optimization techniques applied to a global asset universe. We compare **Linear Programming (LP)**, **Mixed-Integer Linear Programming (MILP)**, and **Scenario-Based Stochastic Optimization** to evaluate the trade-offs between stability, sparsity, and risk-adjusted returns.

## 📂 Repository Structure

* `code/`: Contains all Python scripts for the optimization models.
    * `Linear_Portfolio_Optimization.py`: Basic Mean Absolute Deviation (MAD) model.
    * `Integer_Programming_for_Constraints.py`: MILP model with cardinality constraints ($K \le 10$).
    * `Stochastic_Optimization.py`: Scenario-based optimization using bootstrapped data.
    * `Dynamic_Rebalancing.py`: Contains the turnover-constrained optimization for rolling backtests.
    * `Sensitivity_Analysis.py`: Generates Efficient Frontiers and performs weight stability analysis.
    * `Portfolio_Comparison.py`: Main driver script for running backtests and comparing strategies.
* `data/`: Placeholder directory for raw and filtered historical asset price data.
* `requirements.txt`: Lists all necessary Python dependencies (pandas, pulp, numpy, etc.).
* `README.md`: Project overview and documentation.

## 🚀 How to Run

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Ensure Data Setup** (See `Data Sources` below).

3.  **Run the main comparison:**
    The primary backtesting and comparison logic is contained in `Portfolio_Comparison.py`.
    ```bash
    cd code
    python Portfolio_Comparison.py
    ```
    *(Note: Running this script requires significant computational resources due to the large asset universe and MILP component.)*

## 💾 Data Sources and Setup

The analysis relies on a large proprietary dataset (approx. 1 GB) of daily historical prices from **Stooq** spanning **World, U.S., U.K., and Japan** asset classes.

**The raw data is NOT included in this repository due to size and licensing constraints.**

To reproduce the full analysis, the raw data files must be downloaded directly from the Stooq archive: **<https://stooq.com/db/h/>**

The downloaded data must then be organized into the following explicit structure under the project root (`From_Deterministic_to_Stochastic_Portfolio_Optimization/`), matching the paths defined in the code:

````

└── data/
├── d\_world\_txt/
├── d\_us\_txt/
├── d\_uk\_txt/
└── d\_jp\_txt/

```

## 📊 Key Results

* **Dynamic MILP** achieved the highest Sharpe Ratio by filtering noise and isolating stable assets.
* **Stochastic Optimization** yielded the highest returns but with high volatility, suggesting overfitting to high-variance opportunities.
* The complete failure of static implementations emphasizes that **dynamic rebalancing** is essential in non-stationary, high-dimensional markets.

## 📜 License
**The project is released under the MIT License.**