# Portfolio Optimization using MAD & MILP

This project implements a portfolio optimization model using **Mean Absolute Deviation (MAD)** as the risk metric. It explores both **Linear Programming (LP)** for standard optimization and **Mixed-Integer Linear Programming (MILP)** to enforce real-world constraints such as cardinality (limiting the number of assets) and buy-in thresholds.

Additionally, the project includes a sensitivity analysis to evaluate the robustness of the optimal portfolios via efficient frontiers and bootstrapping.

## 📂 Project Structure

* `Linear_Portfolio_Optimization.py`: Basic implementation of the MAD optimization model using Linear Programming.
* `Integer_Programming_for_Constraints.py`: Extends the basic model using MILP to add constraints on the maximum number of assets ($K$) and minimum/maximum weights.
* `Sensitivity_Analysis.py`: Performs robustness checks, generates efficient frontiers, and runs bootstrap simulations to test weight stability.

---

## 1. Linear Optimization Model (MAD)

The objective is to minimize the absolute portfolio risk while achieving a target return, using the MAD risk metric.

$$
\begin{aligned}
& \text{Minimize} & & \text{MAD} = \frac{1}{T} \sum_{t=1}^{T} | R_{p,t} - \mu_p | \\
& \text{Subject to} & & \mu_p \ge \mu_{\text{target}} \\
& & & \sum_{i=1}^n w_i = 1
\end{aligned}
$$

To solve this efficiently using linear programming, we linearize the absolute value by introducing auxiliary variables $z_t$:

$$
z_t \geq |R_{p,t} - \mu_p| \implies \begin{cases} z_t \geq R_{p,t} - \mu_p \\ z_t \geq -(R_{p,t} - \mu_p) \end{cases}
$$

The final **Linear Programming (LP)** formulation is:

$$
\begin{aligned}
& \text{Minimize} & & \frac{1}{T} \sum_{t=1}^{T} z_t \\
& \text{Subject to} & & \sum_{i=1}^n w_i = 1 \\
& & & \mu_p \ge \mu_{\text{target}} \\
& & & z_t \geq R_{p,t} - \mu_p, \quad \forall t \in [1, T] \\
& & & z_t \geq \mu_p - R_{p,t}, \quad \forall t \in [1, T] \\
& & & w_i \geq 0, \quad \mu_{\text{target}} \geq 0
\end{aligned}
$$

---

## 2. Integer Programming for Constraints

To make the portfolio realistic, we introduce **Mixed-Integer Linear Programming (MILP)** constraints:
1.  **Cardinality:** Use a maximum of $K$ assets.
2.  **Buy-in Thresholds:** If an asset is selected, its weight must be between $w_i^{min}$ and $w_i^{max}$.

We define binary variables $y_i$:
$$
y_i = \begin{cases} 1 & \text{if asset } i \text{ is selected} \\ 0 & \text{otherwise} \end{cases}
$$

The additional constraints are:
$$
\begin{cases} 
\sum_{i} y_i \le K \\ 
y_i \cdot w_i^{min} \leq w_i \le y_i \cdot w_i^{max} \\
y_i \in \{0, 1\}
\end{cases}
$$

---

## 3. Sensitivity Analysis

To validate the models, we perform the following analyses:

### Efficient Frontier
We calculate the minimum MAD for $n$ different target returns and plot the evolution of risk vs. return.

### Weight Stability
We measure the average asset weight variation between two optimal solutions on the frontier to quantify how sensitive the portfolio composition is to slight changes in target return:

$$
s = \frac{1}{P-1}\sum_{p=1}^{P-1} \left(\sum_{i=1}^n|w_{i,p+1} - w_{i,p}|\right)
$$

### Bootstrap Analysis (Robustness)
To ensure optimal weights do not overfit the specific historical sample, we generate $B$ artificial samples using bootstrapping (random sampling with replacement). We then calculate the standard deviation of the optimal weights for each asset across these samples:

$$
\sigma_{w_i} = \sqrt{ \frac{1}{B-1} \sum_{b=1}^B (w_{b,i} - \bar{w}_i)^2}
$$

A low $\sigma_{w_i}$ indicates a robust allocation, whereas a high $\sigma_{w_i}$ suggests instability and overfitting.

---

## 📊 Asset Universe

The analysis is performed on a diversified set of assets spanning equities, bonds, commodities, and currencies.

| Ticker | Company / Asset Name | Sector / Class | Role in Portfolio |
| :--- | :--- | :--- | :--- |
| **SPX** | S&P 500 Index | Global Market | Benchmark Index |
| **JPM** | JPMorgan Chase | Finance | High Beta / Cyclical |
| **XOM** | Exxon Mobil | Energy | Inflation Hedge |
| **WMT** | Walmart | Consumer Defensive | Low Volatility / Defensive |
| **TSLA** | Tesla | Consumer Cyclical | High Growth / High Volatility |
| **UNH** | UnitedHealth Group | Healthcare | Stable Defensive |
| **V** | Visa | Technology | Payments / Transactions |
| **NVDA** | NVIDIA | Semiconductors | Tech Growth |
| **DIS** | Walt Disney | Media / Entertainment | Cyclical Media |
| **EFA** | MSCI EAFE ETF | International Equities | Developed Markets Diversification |
| **VWO** | Vanguard Emerging Markets | Emerging Markets | High Risk / High Growth |
| **TM** | Toyota Motor Corp. | Automotive (Japan) | Industrial Exposure |
| **GLD** | SPDR Gold Shares | Commodities (Gold) | Safe Haven |
| **USO** | US Oil Fund | Commodities (Oil) | Energy Correlation |
| **BND** | Vanguard Total Bond Market | US Bonds | Risk-Free Proxy / Low Correlation |
| **VNQ** | Vanguard Real Estate | Real Estate (REITs) | Sector Diversification / Income |
| **UUP** | Invesco DB US Dollar Index | Currency (USD) | Counter-cyclical / Crisis Hedge |

## 🛠 Requirements

* Python 3.x
* `pandas`
* `numpy`
* `matplotlib`
* `pulp` (Linear Solver)
* `tqdm`
