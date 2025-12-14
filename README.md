# Portfolio Optimization: Deterministic vs. Stochastic Approaches

This repository contains a comparative analysis of portfolio optimization techniques applied to a global asset universe. We compare **Linear Programming (LP)**, **Mixed-Integer Linear Programming (MILP)**, and **Stochastic Optimization** to evaluate the trade-offs between stability, sparsity, and risk-adjusted returns.

## 📂 Repository Structure

* `code/`: Contains all Python scripts for the optimization models.
    * `Linear_Portfolio_Optimization.py`: Basic MAD model.
    * `Integer_Programming_for_Constraints.py`: MILP model with cardinality constraints ($K \le 10$).
    * `Stochastic_Optimization.py`: Scenario-based optimization using bootstrapped data.
    * `Dynamic_Rebalancing.py`: Rolling-window backtesting engine.
    * `Sensitivity_Analysis.py`: Efficient frontier generation.
* `data/`: Contains historical asset price data (CSVs).

## 🚀 How to Run

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the models:**
    Navigate to the code directory and run the desired script.
    ```bash
    cd code
    python Linear_Portfolio_Optimization.py
    ```

## 📊 Key Results

* **Dynamic MILP** achieved the highest Sharpe Ratio (5.28) by filtering noise.
* **Stochastic Optimization** yielded the highest returns (14.61%) but with higher volatility.
* Static portfolios failed significantly, highlighting the importance of dynamic rebalancing.

## 📜 License
MIT License