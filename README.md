# From Deterministic to Stochastic Portfolio Optimization

A comparative analysis of **Linear Programming (LP)**, **Mixed-Integer Linear Programming (MILP)**, and **Scenario-Based Stochastic Optimization** applied to high-dimensional asset universes (). This project demonstrates how cardinality constraints act as effective  regularizers, providing essential noise filtering in volatile markets where unconstrained models often overfit.

---

## 📂 Repository Structure

* **`code/`**: Python implementations of the optimization models.
* 
`utils.py` – **Shared utility functions** for data loading, Point-in-Time alignment, and MAD solver logic.


* 
`Portfolio_Comparison.py` – **Main driver script**: Runs rigorous rolling-window backtests using a **Point-in-Time (PIT)** framework to eliminate look-ahead bias.


* 
`Linear_Portfolio_Optimization.py` – Base Mean Absolute Deviation (MAD) LP model.


* 
`Integer_Programming_for_Constraints.py` – MILP formulation with cardinality constraints ().


* 
`Stochastic_Optimization.py` – Scenario-based optimization using bootstrapped returns.


* 
`Dynamic_Rebalancing.py` – Demonstrates multi-period rebalancing with turnover constraints.


* 
`Sensitivity_Analysis.py` – Tools for analyzing efficient frontiers and weight stability.




* **`data/`**:
* `sample/` – **Toy dataset** for immediate reproducibility.


(Note: The full ~1 GB Stooq dataset containing 8,405 global assets is excluded from Git due to size; see Replication below).




* **`paper/`**:
* 
`Portfolio_Optimization_Paper.pdf` – Full scientific manuscript with corrected PIT results.





---

## 🚀 Quick Start (Toy Model / Demo Mode)

### 1. Clone the repository

```bash
git clone https://github.com/RomanCianci/From_Deterministic_to_Stochastic_Portfolio_Optimization.git
cd From_Deterministic_to_Stochastic_Portfolio_Optimization/code

```

### 2. Install dependencies

```bash
pip install -r ../requirements.txt

```

### 3. Run the backtest in Demo Mode

```bash
python Portfolio_Comparison.py --demo

```

The script runs LP, MILP, and Stochastic models on synthetic data and generates performance plots.

---

## 📈 Full Replication (Real Data)

To reproduce the empirical results reported in the paper (N = 8,405 assets):

1. 
**Download the data**: Sourced from the [Stooq Database](https://stooq.com/db/h/).


2. 
**Organize folders**: Extract into `data/` under `d_world_txt/`, `d_us_txt/`, `d_uk_txt/`, and `d_jp_txt/`.


3. **Run the analysis**:
```bash
python Portfolio_Comparison.py

```



Note: This utilizes a Two-Stage Screening Heuristic to maintain computational tractability.



---

## 📊 Key Results

The empirical analysis of the 2022–2025 period, corrected for survivorship and look-ahead bias, shows:

1. 
**Dynamic MILP** achieved superior risk-adjusted performance (**Sharpe Ratio: 0.64**), successfully using cardinality constraints () to filter out "toxic" assets.


2. 
**Dynamic LP** remained profitable but was diluted by over-diversification, yielding an annualized return of **1.67%**—roughly half that of the MILP strategy.


3. 
**Stochastic Optimization** failed in high dimensions (), suffering a **-41.25% annualized loss** due to overfitting to noisy data artifacts in scenario generation.



---

## 👥 Contributors

* 
**Roman Cianci**: Dataset sourcing, MILP/Stochastic implementation, backtesting engine, and co-authored methodology.


* 
**Timothé Compagnion**: Bootstrap scenario generation, results authorship, and frontier visualization.


* 
**Robin Lebrevelec**: Two-Stage Screening Heuristic, computational analysis, and introduction/literature review.



---

## 📜 License

This project is released under the **MIT License**.