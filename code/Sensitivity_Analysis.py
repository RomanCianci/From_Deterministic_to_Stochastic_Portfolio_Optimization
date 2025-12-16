import pandas as pd
import numpy as np
import pulp as pl
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
import utils

def optimize_stochastic_mad_lp(scenarios: List[pd.DataFrame], probabilities: List[float], target_return: float, allow_short: bool = False, max_individual_weight: float = 0.10) -> Dict[str, float]:
    scenarios_clean = [utils.preprocess_returns_for_solver(s) for s in scenarios]
    
    common_assets = set(scenarios_clean[0].columns)
    for s in scenarios_clean[1:]:
        common_assets = common_assets.intersection(set(s.columns))
    
    assets = sorted(list(common_assets))
    num_scenarios = len(scenarios_clean)
    
    if not assets:
         return {a: 0.0 for a in scenarios[0].columns} 
        
    scenarios_clean = [s[assets] for s in scenarios_clean]
    
    objective_function = pl.LpProblem("stochastic_portfolio_optimization", pl.LpMinimize)
    low_bound_w = 0 if not allow_short else None
    w = pl.LpVariable.dicts("w", assets, lowBound=low_bound_w)
    
    z = {}
    for s in range(num_scenarios):
        T_s = len(scenarios_clean[s])
        for t in range(T_s):
            z[(s, t)] = pl.LpVariable(f"z_{s}_{t}", lowBound=0)

    mu_s = {s: pl.LpVariable(f"mu_s_{s}") for s in range(num_scenarios)}

    # max individual weight
    for a in assets:
        objective_function += w[a] <= max_individual_weight, f"max_weight_{a}"

    for s, scenario_returns in enumerate(scenarios_clean):
        T_s = len(scenario_returns)
        avg_r_s = scenario_returns.mean()
        objective_function += mu_s[s] == pl.lpSum([avg_r_s[a] * w[a] for a in assets]), f"mean_return_{s}"

        ret_values = scenario_returns.values
        asset_indices = range(len(assets))

        for t in range(T_s):
            R_p_t = pl.lpSum([ret_values[t][i] * w[assets[i]] for i in asset_indices])
            objective_function += z[(s, t)] >= R_p_t - mu_s[s]
            objective_function += z[(s, t)] >= -(R_p_t - mu_s[s])

    # objective
    objective = pl.lpSum([
        probabilities[s] * (1.0 / len(scenarios_clean[s])) * pl.lpSum([z[(s, t)] for t in range(len(scenarios_clean[s]))]) 
        for s in range(num_scenarios)
    ])
    objective_function += objective, "expected_mad"

    expected_portfolio_return = pl.lpSum([probabilities[s] * mu_s[s] for s in range(num_scenarios)])
    objective_function += expected_portfolio_return >= target_return, "target_return"

    objective_function += pl.lpSum([w[a] for a in assets]) == 1, "sum_of_weights"

    objective_function.solve(pl.PULP_CBC_CMD(msg=False, timeLimit=60))
    status: str = pl.LpStatus[objective_function.status]
    
    if status != "Optimal":
        return {a: 0.0 for a in scenarios[0].columns}

    weights = {a: w[a].value() if w[a].value() is not None else 0.0 for a in assets}
    
    full_weights = {a: 0.0 for a in scenarios[0].columns}
    full_weights.update(weights)
    return full_weights

def generate_bootstrapped_scenarios(returns: pd.DataFrame, num_scenarios: int) -> Tuple[List[pd.DataFrame], List[float]]:
    T_max = len(returns)
    scenarios_list = [
        returns.sample(n=T_max, replace=True).reset_index(drop=True) 
        for _ in range(num_scenarios)
    ]
    probabilities = [1.0 / num_scenarios] * num_scenarios
    return scenarios_list, probabilities

def efficient_frontier_lp(returns, target_returns, allow_short=False):
    mad_evolution = []
    for target_return in tqdm(target_returns, desc="efficient_frontier_lp"):
        w = utils.optimize_mad_lp(returns, target_return, allow_short=allow_short)
        w_array = np.array([w.get(a, 0.0) for a in returns.columns])
        expected_return = float(np.dot(returns.mean(axis=0).values, w_array))
        mad = utils.compute_mad(returns, w_array)
        mad_evolution.append({"target_return": target_return, "expected_return": expected_return, "mad": mad, "weights": w})
    return mad_evolution

def efficient_frontier_milp(returns, target_returns, K=3, lower_bounds=0.01, upper_bounds=0.5, allow_short=False):
    mad_evolution = []
    for target_return in tqdm(target_returns, desc="efficient_frontier_milp"):
        w = utils.optimize_mad_milp(returns, target_return, K=K, lower_bound=lower_bounds, upper_bound=upper_bounds, allow_short=allow_short)
        w_array = np.array([w.get(a, 0.0) for a in returns.columns])
        expected_return = float(np.dot(returns.mean(axis=0).values, w_array))
        mad = utils.compute_mad(returns, w_array)
        mad_evolution.append({"target_return": target_return, "expected_return": expected_return, "mad": mad, "weights": w, "K": K})
    return mad_evolution

def efficient_frontier_stochastic(returns: pd.DataFrame, target_returns: List[float], num_scenarios: int = 10, max_weight: float = 0.10, allow_short: bool = False):
    mad_evolution = []
    for target_return in tqdm(target_returns, desc="efficient_frontier_stochastic"):
        scenarios_list, probabilities = generate_bootstrapped_scenarios(returns, num_scenarios)
        
        w = optimize_stochastic_mad_lp(scenarios=scenarios_list, 
                                       probabilities=probabilities, 
                                       target_return=target_return, 
                                       allow_short=allow_short,
                                       max_individual_weight=max_weight)
        
        w_array = np.array([w.get(a, 0.0) for a in returns.columns])
        expected_return = float(np.dot(returns.mean(axis=0).values, w_array))
        mad = utils.compute_mad(returns, w_array)
        
        mad_evolution.append({
            "target_return": target_return, 
            "expected_return": expected_return, 
            "mad": mad, 
            "weights": w, 
            "num_scenarios": num_scenarios
        })
    return mad_evolution

def plot_efficient_frontiers(mad_evolution_lp, mad_evolution_milp=None, mad_evolution_stochastic=None):
    plt.figure(figsize=(10,6))
    
    mad_lp = [r["mad"] for r in mad_evolution_lp if r["mad"] > 1e-6]
    expected_return_lp = [r["expected_return"] for r in mad_evolution_lp if r["mad"] > 1e-6]
    plt.plot(mad_lp, expected_return_lp, '-o', label="MAD LP")

    if mad_evolution_milp is not None:
        mad_milp = [r["mad"] for r in mad_evolution_milp if r["mad"] > 1e-6]
        expected_return_milp = [r["expected_return"] for r in mad_evolution_milp if r["mad"] > 1e-6]
        plt.plot(mad_milp, expected_return_milp, '-s', label="MAD MILP")

    if mad_evolution_stochastic is not None:
        mad_stoch = [r["mad"] for r in mad_evolution_stochastic if r["mad"] > 1e-6]
        expected_return_stoch = [r["expected_return"] for r in mad_evolution_stochastic if r["mad"] > 1e-6]
        plt.plot(mad_stoch, expected_return_stoch, '-^', label="Stochastic MAD")

    plt.xlabel("MAD")
    plt.ylabel("Expected return")
    plt.title("Efficient frontier - Evolution of MAD based on expected return")
    plt.legend()
    plt.grid(True)
    plt.show()

def weight_variance(returns, target_return, number_of_bootstraps=10, solver="lp", milp_config=None, stochastic_config=None):
    N = returns.shape[1]
    W = np.zeros((number_of_bootstraps, N))
    T_max = len(returns)    
    asset_names = list(returns.columns)

    for b in range(number_of_bootstraps):
        i = np.random.choice(T_max, size=T_max, replace=True)
        sample = returns.iloc[i]

        if solver == "lp":
            w = utils.optimize_mad_lp(sample, target_return)
        elif solver == "milp":
            config = milp_config or {}
            w = utils.optimize_mad_milp(sample, target_return, K=config.get("K", 3), lower_bound=config.get("lower_bound", 0.01), upper_bound=config.get("upper_bound", 0.5))
        elif solver == "stochastic":
            config = stochastic_config or {}
            num_scenarios = config.get("num_scenarios", 10)
            max_weight = config.get("max_weight", 0.10)
            scenarios_list, probabilities = generate_bootstrapped_scenarios(sample, num_scenarios)
            w = optimize_stochastic_mad_lp(scenarios=scenarios_list, probabilities=probabilities, target_return=target_return, max_individual_weight=max_weight)

        W[b, :] = np.array([w.get(a, 0.0) for a in asset_names])

    variances = dict(zip(asset_names, W.var(axis=0)))
    sorted_variances = dict(sorted(variances.items(), key=lambda item: item[1], reverse=True))
    return sorted_variances

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run with toy data')
    args = parser.parse_args()

    BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    if args.demo:
        print("Warning : We are currently running on demo mode (toy data)")
        BASE_PATH = os.path.join(BASE_DIR, 'sample', 'generated')
        WORLD_BASE_PATHS = [BASE_PATH]
    else:
        BASE_PATH = BASE_DIR
        WORLD_BASE_PATHS = [
            f"{BASE_PATH}/d_world_txt/data/daily/world/bonds",
            f"{BASE_PATH}/d_world_txt/data/daily/world/cryptocurrencies",
            f"{BASE_PATH}/d_world_txt/data/daily/world/currencies/major",
            f"{BASE_PATH}/d_world_txt/data/daily/world/currencies/other",
            f"{BASE_PATH}/d_world_txt/data/daily/world/money market",
            f"{BASE_PATH}/d_world_txt/data/daily/world/stooq stocks indices",
            f"{BASE_PATH}/d_world_txt/data/daily/world/indices",
        ]
    
    dfs = utils.load_stooq_assets_glob_all(WORLD_BASE_PATHS)
    prices = utils.align_and_merge_prices(dfs)
    
    RECENT_START_DATE = "2018-01-01"
    prices = prices.loc[prices.index >= RECENT_START_DATE].copy()
    
    returns = utils.compute_returns(prices).dropna(axis=1, how='all').dropna(how='all')
    
    if returns.empty:
        print("Error: No common data period found...")
        
    else:
        print(f"Data aligned. Using {returns.shape[1]} assets over {returns.shape[0]} periods.")
        
        mean_returns = returns.mean()
        min_ret = mean_returns.min()
        target_returns = list(np.linspace(max(0, min_ret), 0.0005, 10)) 
        print("\n--- Computing LP Frontier ---")
        returns_lp = returns.iloc[:, :500] 
        lp_frontier = efficient_frontier_lp(returns_lp, target_returns, allow_short=False)

        print("\n--- Computing MILP Frontier ---")
        N_MILP_LIMIT = 50
        vol = returns.std()
        score = returns.mean() / vol
        milp_assets = score.sort_values(ascending=False).head(N_MILP_LIMIT).index.tolist()
        returns_milp = returns[milp_assets]
        
        milp_frontier = efficient_frontier_milp(returns_milp, target_returns, K=5, lower_bounds=0.01, upper_bounds=0.5, allow_short=False)
        
        print("\n--- Computing Stochastic Frontier ---")
        returns_stoch = returns[milp_assets]
        stochastic_frontier = efficient_frontier_stochastic(returns_stoch, target_returns, num_scenarios=50, max_weight=0.10, allow_short=False)
        plot_efficient_frontiers(lp_frontier, milp_frontier, stochastic_frontier)
        target_for_bootstrap = target_returns[len(target_returns)//2]
        number_of_bootstraps = 5

        print(f"\nCalculating weight variance (LP) for target {target_for_bootstrap:.6f}...")
        weight_var_lp = weight_variance(returns_lp, target_for_bootstrap, number_of_bootstraps=number_of_bootstraps, solver="lp")
        
        print(f"\nCalculating weight variance (MILP) for target {target_for_bootstrap:.6f}...")
        weight_var_milp = weight_variance(returns_milp, target_for_bootstrap, number_of_bootstraps=number_of_bootstraps, solver="milp", milp_config={"K":5})
        
        print(f"\nCalculating weight variance (Stochastic) for target {target_for_bootstrap:.6f}...")
        weight_var_stoch = weight_variance(returns_stoch, target_for_bootstrap, number_of_bootstraps=number_of_bootstraps, solver="stochastic", stochastic_config={"num_scenarios":10})

        print("\nTop 5 Variance Assets (LP):")
        for k, v in list(weight_var_lp.items())[:5]: print(f"{k}: {v:.6f}")