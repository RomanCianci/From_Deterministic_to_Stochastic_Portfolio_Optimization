import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils

df_results = pd.read_csv("backtest_results_final.csv", index_col=0, parse_dates=True)

df_results = df_results.replace([np.inf, -np.inf], 0.0).fillna(0.0)

print("\n" + "="*30)

metrics = []

for col in df_results.columns:
    series = df_results[col]
    
    # Metrics
    ann_ret = series.mean() * 252
    ann_vol = series.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
    wealth = (1 + series).cumprod()
    max_dd = ((wealth - wealth.cummax()) / wealth.cummax()).min()
    
    metrics.append({
        "Strategy": col,
        "Ann Return": f"{ann_ret:.2%}",
        "Ann Vol": f"{ann_vol:.2%}",
        "Sharpe": f"{sharpe:.2f}",
        "Max DD": f"{max_dd:.2%}"
    })

    print(f"\n[{col}]")
    print(f"Annualized Return: {ann_ret:.2%}")
    print(f"Annualized Vol:    {ann_vol:.2%}")
    print(f"Sharpe Ratio:      {sharpe:.2f}")
    print(f"Max Drawdown:      {max_dd:.2%}")


plt.figure(figsize=(10, 6))
for col in df_results.columns:
    wealth = (1 + df_results[col]).cumprod() * 100
    plt.plot(wealth, label=col)

plt.title("Out-of-Sample Performance")
plt.ylabel("Wealth ($)")
plt.xlabel("Date")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("figure1_repaired.png", dpi=300)





