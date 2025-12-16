import pandas as pd
import numpy as np
import pulp as pl
from typing import Dict, List, Callable, Tuple
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm 
import utils


def filter_by_data_completeness(prices: pd.DataFrame, min_ratio: float, start_date: str, end_date: str) -> pd.DataFrame:
    if prices.empty:
        return prices

    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index, errors="coerce")
    
    if not prices.index.is_monotonic_increasing:
        prices.sort_index(inplace=True)

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    mask = (prices.index >= start) & (prices.index <= end)
    window_prices = prices.loc[mask]

    if window_prices.empty:
        return pd.DataFrame()

    required_count = len(window_prices.index)
    thresh = int(np.ceil(required_count * min_ratio))

    print(f"Calculating valid assets (Threshold: {thresh}/{required_count})...")
    non_nan_counts = window_prices.count()
    keepers = non_nan_counts[non_nan_counts >= thresh].index.tolist()

    filtered_prices = prices[keepers].copy()
    print(f"Filtering Results: {prices.shape[1]} initial -> {len(keepers)} kept.")
    return filtered_prices 

def get_milp_filtered_assets(returns: pd.DataFrame, n_limit: int) -> pd.DataFrame:
    if returns.empty:
        return returns
    
    returns_clean = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    volatility = returns_clean.std()
    zero_vol_assets = volatility[volatility < 1e-10].index
    
    if not zero_vol_assets.empty:
        if len(returns_clean.columns) - len(zero_vol_assets) < 1:
            return pd.DataFrame() 
        returns_clean = returns_clean.drop(columns=zero_vol_assets)
        volatility = returns_clean.std() 

    mean_returns = returns_clean.mean()
    score = mean_returns / volatility
    score_series = pd.Series(score, index=returns_clean.columns)
    score_series = score_series.dropna().replace([np.inf, -np.inf], 0)
    
    limit = min(n_limit, len(score_series))
    if limit == 0:
        return pd.DataFrame()

    milp_assets = score_series.sort_values(ascending=False).head(limit).index.tolist()
    return returns_clean[milp_assets]

def optimize_stochastic_mad_lp(scenarios: List[pd.DataFrame], probabilities: List[float], target_return: float, allow_short: bool = False) -> Dict[str, float]:
    scenarios_clean = [utils.preprocess_returns_for_solver(s) for s in scenarios]
    if any(s.empty for s in scenarios_clean):
        raise ValueError("Empty scenario.")
        
    assets = scenarios_clean[0].columns.tolist()
    num_scenarios = len(scenarios_clean)
    
    prob = pl.LpProblem("Stochastic_MAD", pl.LpMinimize)
    w = pl.LpVariable.dicts("w", assets, lowBound=0 if not allow_short else None)
    
    z = {}
    for s in range(num_scenarios):
        for t in range(len(scenarios_clean[s])):
             z[(s, t)] = pl.LpVariable(f"z_{s}_{t}", lowBound=0)

    mu_s = {s: pl.LpVariable(f"mu_s_{s}") for s in range(num_scenarios)}

    MAX_WEIGHT = 0.15
    for a in assets:
        prob += w[a] <= MAX_WEIGHT

    for s, scenario_returns in enumerate(scenarios_clean):
        avg_r_s = scenario_returns.mean()
        prob += mu_s[s] == pl.lpSum([avg_r_s[a] * w[a] for a in assets])

        ret_values = scenario_returns.values
        asset_indices = range(len(assets))

        for t in range(len(scenario_returns)):
            R_p_t = pl.lpSum([ret_values[t][i] * w[assets[i]] for i in asset_indices])
            prob += z[(s, t)] >= R_p_t - mu_s[s]
            prob += z[(s, t)] >= -(R_p_t - mu_s[s])

    objective = pl.lpSum([probabilities[s] * (1.0 / len(scenarios_clean[s])) * pl.lpSum([z[(s, t)] for t in range(len(scenarios_clean[s]))]) for s in range(num_scenarios)])
    prob += objective

    expected_return = pl.lpSum([probabilities[s] * mu_s[s] for s in range(num_scenarios)])
    prob += expected_return >= target_return
    prob += pl.lpSum([w[a] for a in assets]) == 1

    prob.solve(pl.PULP_CBC_CMD(msg=False, timeLimit=25))
    
    if pl.LpStatus[prob.status] != "Optimal":
        return {a: 0.0 for a in assets}

    weights = {a: w[a].value() if w[a].value() is not None else 0.0 for a in assets}
    return weights


def optimization_backtest(returns: pd.DataFrame, training_length: int, rebalance_frequency: int, target_return: float, transaction_cost: float, optimization_type: Callable, **optimizer_kwargs) -> Tuple[pd.Series, pd.DataFrame]:

    current_weights = np.zeros(returns.shape[1])
    portfolio_returns = []
    
    weight_history = []
    rebalance_dates = []

    all_assets = returns.columns
    asset_filter_limit = optimizer_kwargs.pop('asset_filter_limit', None)

    num_iterations = (len(returns) - training_length) // rebalance_frequency
    
    for k in tqdm(range(num_iterations), desc=f"Backtesting {optimization_type.__name__}"):
        i = training_length + k * rebalance_frequency
        train_start = max(0, i - training_length)
        train_end = i
        
        current_date = returns.index[i]

        training_returns = returns.iloc[train_start:train_end]
        current_returns = training_returns.copy()

        if asset_filter_limit:
            current_returns = get_milp_filtered_assets(training_returns, asset_filter_limit)

        is_stochastic = optimization_type == optimize_stochastic_mad_lp

        try:
            if is_stochastic:
                num_scenarios = optimizer_kwargs.get("num_scenarios", 10)
                scenarios_list = [
                    current_returns.sample(n=len(current_returns), replace=True).reset_index(drop=True)
                    for _ in range(num_scenarios)
                ]
                probabilities = [1.0 / num_scenarios] * num_scenarios
                optimal_weights_dict = optimization_type(scenarios=scenarios_list, probabilities=probabilities, target_return=target_return, **{k: v for k, v in optimizer_kwargs.items() if k != "num_scenarios"})
            else:
                optimal_weights_dict = optimization_type(current_returns, target_return, **optimizer_kwargs)

        except Exception as e:
            print(f"Error at step {k}: {e}")
            optimal_weights_dict = {}

        new_weights = np.array([optimal_weights_dict.get(a, 0.0) for a in all_assets])
        
        # Store weights
        weight_history.append(new_weights)
        rebalance_dates.append(current_date)

        # Transaction Costs
        turnover = np.sum(np.abs(new_weights - current_weights))
        cost = transaction_cost * turnover

        # Apply to Test Window
        test_start = i
        test_end = min(i + rebalance_frequency, len(returns))
        testing_returns = returns.iloc[test_start:test_end]

        if testing_returns.empty:
            current_weights = new_weights
            continue

        period_portfolio_returns = testing_returns.values.dot(new_weights)
        if period_portfolio_returns.size > 0:
            period_portfolio_returns[0] -= cost

        portfolio_returns.extend(period_portfolio_returns.tolist())
        current_weights = new_weights

    start_idx = training_length
    end_idx = training_length + len(portfolio_returns)
    idx = returns.index[start_idx:end_idx]
    
    ret_series = pd.Series(portfolio_returns, index=idx)
    weight_df = pd.DataFrame(weight_history, index=rebalance_dates, columns=all_assets)
    
    return ret_series, weight_df



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run with toy data')
    args = parser.parse_args()

    BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    TARGET_RETURN = 0.01 / 100        
    TRAIN_TEST_SPLIT = 0.8      
    DYNAMIC_REBALANCE_FREQ = 100       
    DYNAMIC_TRAIN_WINDOW = 252      
    DYNAMIC_COST = 0.005       
    MILP_K = 10                       
    MILP_LOWER_BOUND = 0.05           
    MILP_UPPER_BOUND = 0.7  
    LP_ASSET_LIMIT = 200            
    MILP_ASSET_LIMIT = 100           
    STOCHASTIC_ASSET_LIMIT = 100      
    STOCHASTIC_SCENARIOS = 30       
    
    if args.demo:
        print("--- DEMO MODE ---")
        sample_folder = os.path.join(BASE_DIR, 'sample', 'generated')
        ASSET_BASE_PATHS = [sample_folder]
        MIN_START_DATE = "2022-01-01" 
        MIN_END_DATE = "2023-06-01"   
        MIN_DATA_COMPLETENESS = 0.0  
    else:
        BASE_PATH = BASE_DIR
        ASSET_BASE_PATHS = [
            # World (d_world_txt)
            f"{BASE_PATH}/d_world_txt/data/daily/world/bonds",
            f"{BASE_PATH}/d_world_txt/data/daily/world/cryptocurrencies",
            f"{BASE_PATH}/d_world_txt/data/daily/world/currencies/major",
            f"{BASE_PATH}/d_world_txt/data/daily/world/currencies/other",
            f"{BASE_PATH}/d_world_txt/data/daily/world/money market",
            f"{BASE_PATH}/d_world_txt/data/daily/world/stooq stocks indices",
            f"{BASE_PATH}/d_world_txt/data/daily/world/indices",
            f"{BASE_PATH}/d_world_txt/data/daily/us",
            f"{BASE_PATH}/d_world_txt/data/daily/uk",
            f"{BASE_PATH}/d_world_txt/data/daily/jp",
            f"{BASE_PATH}/d_world_txt/data/daily/macro",

            # Japan (d_jp_txt)
            f"{BASE_PATH}/d_jp_txt/data/daily/jp/tse corporate bonds",
            f"{BASE_PATH}/d_jp_txt/data/daily/jp/tse etfs",
            f"{BASE_PATH}/d_jp_txt/data/daily/jp/tse futures",
            f"{BASE_PATH}/d_jp_txt/data/daily/jp/tse indices",
            f"{BASE_PATH}/d_jp_txt/data/daily/jp/tse options",
            f"{BASE_PATH}/d_jp_txt/data/daily/jp/tse stocks",
            f"{BASE_PATH}/d_jp_txt/data/daily/jp/tse treasury bonds",

            # UK (d_uk_txt)
            f"{BASE_PATH}/d_uk_txt/data/daily/uk/lse etfs/1",
            f"{BASE_PATH}/d_uk_txt/data/daily/uk/lse etfs/2",
            f"{BASE_PATH}/d_uk_txt/data/daily/uk/lse etfs/3",
            f"{BASE_PATH}/d_uk_txt/data/daily/uk/lse stocks",
            f"{BASE_PATH}/d_uk_txt/data/daily/uk/lse stocks intl/1",
            f"{BASE_PATH}/d_uk_txt/data/daily/uk/lse stocks intl/2",

            # USA (d_us_txt)
            f"{BASE_PATH}/d_us_txt/data/daily/us/nasdaq etfs",
            f"{BASE_PATH}/d_us_txt/data/daily/us/nasdaq stocks/1",
            f"{BASE_PATH}/d_us_txt/data/daily/us/nasdaq stocks/2",
            f"{BASE_PATH}/d_us_txt/data/daily/us/nasdaq stocks/3",
            f"{BASE_PATH}/d_us_txt/data/daily/us/nyse etfs/1", 
            f"{BASE_PATH}/d_us_txt/data/daily/us/nyse etfs/2",
            f"{BASE_PATH}/d_us_txt/data/daily/us/nyse stocks/1",
            f"{BASE_PATH}/d_us_txt/data/daily/us/nyse stocks/2",
            f"{BASE_PATH}/d_us_txt/data/daily/us/nysemkt stocks",
        ]
        MIN_START_DATE = "2008-01-01"
        MIN_END_DATE = "2025-07-01"
        MIN_DATA_COMPLETENESS = 0.5 

    dfs = utils.load_stooq_assets_glob_all(ASSET_BASE_PATHS, min_rows=252)
    raw_prices = utils.align_and_merge_prices(dfs)
    
    prices = filter_by_data_completeness(raw_prices, MIN_DATA_COMPLETENESS, start_date=MIN_START_DATE, end_date=MIN_END_DATE)
    returns = utils.compute_returns(prices).fillna(0.0)
    returns = returns.loc[MIN_START_DATE:] 
    split_point = int(len(returns) * TRAIN_TEST_SPLIT)
    out_of_sample_returns = returns.iloc[split_point:]

    print("\n--- 2. Dynamic Portfolios Backtest ---\n")

    results_data = {}
    weights_data = {}

    # A) LP
    lp_params = {"asset_filter_limit": LP_ASSET_LIMIT}
    print("Running LP...")
    ret_lp, w_lp = optimization_backtest(returns, DYNAMIC_TRAIN_WINDOW, DYNAMIC_REBALANCE_FREQ, TARGET_RETURN, DYNAMIC_COST, utils.optimize_mad_lp, **lp_params)
    results_data['Dynamic LP'] = ret_lp
    weights_data['Dynamic LP'] = w_lp

    # B) MILP
    milp_params = {"K": MILP_K, "lower_bound": MILP_LOWER_BOUND, "upper_bound": MILP_UPPER_BOUND, "asset_filter_limit": MILP_ASSET_LIMIT}
    print("Running MILP...")
    ret_milp, w_milp = optimization_backtest(returns, DYNAMIC_TRAIN_WINDOW, DYNAMIC_REBALANCE_FREQ, TARGET_RETURN, DYNAMIC_COST, utils.optimize_mad_milp, **milp_params)
    results_data['Dynamic MILP'] = ret_milp
    weights_data['Dynamic MILP'] = w_milp

    # C) Stochastic
    stoch_params = {"num_scenarios": STOCHASTIC_SCENARIOS, "asset_filter_limit": STOCHASTIC_ASSET_LIMIT}
    print(f"Running Stochastic ({STOCHASTIC_SCENARIOS} scenarios)...")
    ret_stoch, w_stoch = optimization_backtest(returns, DYNAMIC_TRAIN_WINDOW, DYNAMIC_REBALANCE_FREQ, TARGET_RETURN, DYNAMIC_COST, optimize_stochastic_mad_lp, **stoch_params)
    results_data['Dynamic Stochastic'] = ret_stoch
    weights_data['Dynamic Stochastic'] = w_stoch



    print("\nGenerating Figure 1 (Wealth)...")
    common_idx = results_data['Dynamic MILP'].index.intersection(out_of_sample_returns.index)
    
    milp_curve = (1 + results_data['Dynamic MILP'].loc[common_idx]).cumprod() * 100
    stoch_curve = (1 + results_data['Dynamic Stochastic'].loc[common_idx]).cumprod() * 100
    
    eq_weights = np.full(returns.shape[1], 1/returns.shape[1])
    bench_rets = returns.loc[common_idx].dot(eq_weights)
    bench_curve = (1 + bench_rets).cumprod() * 100
    
    def get_dd(series): return (series - series.cummax()) / series.cummax()
    dd_milp = get_dd(milp_curve)
    dd_stoch = get_dd(stoch_curve)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(milp_curve, label='Dynamic MILP', color='#1f77b4', linewidth=2)
    ax1.plot(stoch_curve, label='Dynamic Stochastic', color='#2ca02c', alpha=0.7)
    ax1.plot(bench_curve, label='Benchmark (Eq Weight)', color='gray', linestyle='--')
    ax1.set_ylabel('Wealth ($)')
    ax1.set_title('Out-of-Sample Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.fill_between(dd_milp.index, dd_milp, 0, color='#1f77b4', alpha=0.3, label='MILP Drawdown')
    ax2.plot(dd_stoch, color='#2ca02c', linewidth=1, label='Stoch Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure1_wealth_drawdown.png', dpi=300)
    print("Saved figure1_wealth_drawdown.png")



    print("\nGenerating Figure 3 (MILP Composition)...")
    
    w_df = weights_data['Dynamic MILP'].copy()
    
    selected_assets = w_df.columns[(w_df > 0.001).any()]
    w_df = w_df[selected_assets]
    usage_sum = w_df.sum().sort_values(ascending=False)
    top_assets = usage_sum.head(15).index 
    w_plot = w_df[top_assets]
    
    plt.figure(figsize=(12, 6))
    plt.stackplot(w_plot.index, w_plot.T, labels=w_plot.columns, alpha=0.8)
    plt.title('Dynamic MILP Portfolio Composition (Top 15 Assets)', fontsize=14)
    plt.ylabel('Allocation Weight')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    plt.tight_layout()
    plt.savefig('figure3_milp_composition.png', dpi=300)
    print("Saved figure3_milp_composition.png")
