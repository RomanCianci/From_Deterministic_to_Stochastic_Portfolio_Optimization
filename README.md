# From Deterministic to Stochastic Portfolio Optimization

![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![Status](https://img.shields.io/badge/status-complete-green)

A comparative analysis of **Linear Programming (LP)**, **Mixed-Integer Linear Programming (MILP)**, and **Scenario-Based Stochastic Optimization** applied to high-dimensional asset universes ($N \gg T$). This project demonstrates how cardinality constraints can act as effective regularizers in volatile markets, often outperforming unconstrained stochastic approaches.

## 📂 Repository Structure

* **`code/`**: Python implementations of the optimization models.
    * `Portfolio_Comparison.py`: **Main driver script.** Runs the rolling-window backtest and generates performance metrics.
    * `Linear_Portfolio_Optimization.py`: Implementation of the base Mean Absolute Deviation (MAD) LP model.
    * `Integer_Programming_for_Constraints.py`: MILP formulation with cardinality constraints ($K \le 10$).
    * `Stochastic_Optimization.py`: Scenario-based optimization using bootstrapped return paths.
    * `Dynamic_Rebalancing.py`: Rolling-window engine for out-of-sample testing.
    * `Sensitivity_Analysis.py`: Tools for efficient frontier plotting and parameter stability checks.
    * `generate_toy_data.py`: Script to generate synthetic data for the "Toy Model" demo.
* **`data/`**:
    * `sample/`: **Toy dataset** for immediate reproducibility and code verification.
    * *(Note: The full 1GB Stooq dataset is excluded due to size constraints. See "Full Replication" below.)*
* **`paper/`**:
    * `Final_Report.pdf`: The full scientific manuscript detailing methodology and results.
    * `source/`: LaTeX source files for the report.

## 🚀 Quick Start (Toy Model / Demo Mode)

To verify the code functionality immediately without downloading the massive Stooq dataset, use the built-in **Demo Mode**. This uses a synthetic dataset to test the optimization pipelines.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/RomanCianci/From_Deterministic_to_Stochastic_Portfolio_Optimization.git](https://github.com/RomanCianci/From_Deterministic_to_Stochastic_Portfolio_Optimization.git)
   cd From_Deterministic_to_Stochastic_Portfolio_Optimization/code
````

2.  **Install dependencies:**

    ```bash
    pip install -r ../requirements.txt
    ```

3.  **Generate the toy data:**

    ```bash
    python generate_toy_data.py
    ```

    *(This creates a `../data/sample/generated_data` folder with synthetic assets).*

4.  **Run the backtest in Demo Mode:**

    ```bash
    python Portfolio_Comparison.py --demo
    ```

    *This will run the LP, MILP, and Stochastic models on the small dataset and output performance metrics to the console.*

## 📈 Full Replication (Real Data)

To reproduce the full paper results ($N=8,405$ assets):

1.  **Download Data:** Get the historical daily data from the [Stooq Database](https://stooq.com/db/h/).
2.  **Organize Folders:** Extract the data into the `data/` directory so it matches the following structure:
    ```text
    data/
    ├── d_world_txt/
    ├── d_us_txt/
    ├── d_uk_txt/
    └── d_jp_txt/
    ```
3.  **Run the Analysis:**
    ```bash
    python Portfolio_Comparison.py
    ```
    *(Note: This requires significant RAM and CPU time due to the size of the asset universe and the complexity of the MILP solver.)*

## 📊 Key Results

Our analysis of the 2022-2025 volatile period reveals:

1.  **Dynamic MILP** achieved the highest risk-adjusted efficiency (**Sharpe Ratio 3.09**), effectively using cardinality constraints ($K \le 10$) to filter out high-volatility noise.
2.  **Dynamic Stochastic** delivered the highest raw returns (**22.07% annualized**) but suffered from significant drawdown risk and turnover, highlighting the "Curse of Dimensionality" in scenario generation.
3.  **Unconstrained LP** consistently underperformed, confirming that in high-dimensional settings ($N \gg T$), sparsity is not just a constraint but a necessary regularizer.

## 👥 Contributors

  * **Roman CIANCI**: Sourced the Stooq dataset, implemented the MILP model with cardinality constraints, and drafted the Introduction/Literature Review.
  * **Timothé COMPAGNION**: Formulated the Stochastic Optimization model, implemented the bootstrap scenario generation, and wrote the Results analysis.
  * **Robin LEBREVELEC**: Developed the rolling-window backtesting engine, performed the sensitivity analysis, and managed the repository/documentation.

## 📜 License

This project is released under the MIT License.
