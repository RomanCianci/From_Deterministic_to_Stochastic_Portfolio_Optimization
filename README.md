Thanks. I need you to correct the README : # From Deterministic to Stochastic Portfolio Optimization

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Status](https://img.shields.io/badge/status-complete-green)

A comparative analysis of **Linear Programming (LP)**, **Mixed-Integer Linear Programming (MILP)**, and **Scenario-Based Stochastic Optimization** applied to high-dimensional asset universes ($N \gg T$). This project demonstrates how cardinality constraints can act as effective regularizers in volatile markets, often outperforming unconstrained stochastic approaches.

---

## 📂 Repository Structure

* **`code/`**: Python implementations of the optimization models.

  * `utils.py` – **Shared utility functions** for data loading, cleaning, merging, and common solver logic.
  * `Portfolio_Comparison.py` – **Main driver script**: runs rolling-window backtests for all models (LP, MILP, Stochastic), generates performance metrics, and plots results.
  * `Linear_Portfolio_Optimization.py` – Base Mean Absolute Deviation (MAD) LP model. Calculates and exports the efficient frontier.
  * `Integer_Programming_for_Constraints.py` – MILP formulation with cardinality constraints ($K \le 10$) and buy-in thresholds.
  * `Stochastic_Optimization.py` – Scenario-based optimization using bootstrapped return paths to minimize Expected MAD.
  * `Dynamic_Rebalancing.py` – Demonstrates dynamic rebalancing logic with turnover constraints over multiple periods.
  * `Sensitivity_Analysis.py` – Tools for plotting comparative efficient frontiers and analyzing weight stability.
  * `result_tables.py` – Parses the backtest CSV to calculate metrics.
  * `generate_toy_data.py` – Generates synthetic data for the *Toy Model* demo.

* **`data/`**:

  * `sample/` – **Toy dataset** for immediate reproducibility and code verification.
    *(Note: The full ~1 GB Stooq dataset is excluded due to size constraints; see **Full Replication** below.)*

* **`paper/`**:

  * `Final_Report.pdf` – Full scientific manuscript detailing methodology and results.
  * `source/` – LaTeX source files for the report.

---

## 🚀 Quick Start (Toy Model / Demo Mode)

Verify functionality without downloading the full Stooq dataset using the built-in **Demo Mode**, which relies on synthetic data.

### 1. Clone the repository

```bash
git clone https://github.com/RomanCianci/From_Deterministic_to_Stochastic_Portfolio_Optimization.git
cd From_Deterministic_to_Stochastic_Portfolio_Optimization/code
```

### 2. Install dependencies

```bash
pip install -r ../requirements.txt
```

### 3. Generate the toy data

```bash
python generate_toy_data.py
```

This creates a `../data/sample/generated/` directory containing synthetic asset return series.

### 4. Run the backtest in Demo Mode

```bash
python Portfolio_Comparison.py --demo
```

The script runs LP, MILP, and Stochastic models on the toy dataset and generates performance plots (wealth curves and portfolio composition).

---

## 📈 Full Replication (Real Data)

To reproduce the full empirical results reported in the paper (N = 8,405 assets):

### 1. Download the data

Download historical daily price data from the [Stooq Database](https://stooq.com/db/h/).

### 2. Organize the folders

Extract the data into the `data/` directory with the following structure:

```
data/
├── d_world_txt/
├── d_us_txt/
├── d_uk_txt/
└── d_jp_txt/
```

### 3. Run the analysis

```bash
python Portfolio_Comparison.py
```

> **Note:** Full replication requires substantial RAM and CPU time due to the size of the asset universe and the computational complexity of the MILP solver.

---

## 📊 Key Results

An empirical analysis of the 2022–2025 volatile period shows:

1. **Dynamic MILP** achieved the most robust performance (Sharpe Ratio: 0.64, Annualized Return: 3.42%), utilizing cardinality constraints ($K \le 10$) to filter out "toxic" assets and delivering **2x the return** of the unconstrained LP.
2. **Dynamic Stochastic Optimization** failed in high dimensions ($N \gg T$), suffering a -41.25% annualized loss due to overfitting to noisy data artifacts in scenario generation.
3. **Unconstrained LP** remained profitable but diluted, yielding a negligible annualized return of 1.67%, confirming that over-diversification limits potential in high-dimensional settings.

---

## 👥 Contributors

* **Roman Cianci** Sourced the high-dimensional Stooq dataset; implemented the MILP model with cardinality constraints and the stochastic optimization model; developed the rolling-window backtesting engine; managed the GitHub repository and co-authored Methodology and Discussion.
* **Timothé Compagnion** Implemented bootstrap-based scenario generation; conducted sensitivity analysis; authored the Results section; created the Efficient Frontier visualization.
* **Robin Lebrevelec** Developed the Two-Stage Screening Heuristic; performed computational cost analysis; drafted the Introduction and Literature Review.

---

## 📜 License

This project is released under the **MIT License**.