import pandas as pd
import numpy as np
import pulp as pl
from typing import Dict, List
import argparse
import os
import utils

def optimize_stochastic_mad_lp(scenarios: List[pd.DataFrame], probabilities: List[float], target_return: float, allow_short: bool = False) -> Dict[str, float]:
    scenarios_clean = [utils.preprocess_returns_for_solver(s) for s in scenarios]
    common_assets = set(scenarios_clean[0].columns)
    for s in scenarios_clean[1:]:
        common_assets = common_assets.intersection(set(s.columns))
    
    assets = sorted(list(common_assets))
    if not assets:
         return {a: 0.0 for a in scenarios[0].columns} 
         
    scenarios_clean = [s[assets] for s in scenarios_clean]
    num_scenarios = len(scenarios_clean)
    
    objective_function = pl.LpProblem("stochastic_portfolio_optimization", pl.LpMinimize)
    
    # Decision variables
    w = pl.LpVariable.dicts("w", assets, lowBound=0 if not allow_short else None)
    z = {}
    for s in range(num_scenarios):
        for t in range(len(scenarios_clean[s])):
             z[(s, t)] = pl.LpVariable(f"z_{s}_{t}", lowBound=0)

    mu_s = {s: pl.LpVariable(f"mu_s_{s}") for s in range(num_scenarios)}


    # max individual weight (Diversification constraint)
    MAX_INDIVIDUAL_WEIGHT = 0.50
    for a in assets:
        objective_function += w[a] <= MAX_INDIVIDUAL_WEIGHT, f"max_weight_{a}"

    for s, scenario_returns in enumerate(scenarios_clean):
        
        # mean return per scenario
        avg_r_s = scenario_returns.mean()
        objective_function += mu_s[s] == pl.lpSum([avg_r_s[a] * w[a] for a in assets]), f"mean_return_{s}"

        # mad linearization per scenario
        for t, (_, row) in enumerate(scenario_returns.iterrows()):
            R_p_t = pl.lpSum([row[a] * w[a] for a in assets])
            objective_function += z[(s, t)] >= R_p_t - mu_s[s]
            objective_function += z[(s, t)] >= -(R_p_t - mu_s[s])

    # objective: Expected MAD
    objective = pl.lpSum([probabilities[s] * (1.0 / len(scenarios_clean[s])) * pl.lpSum([z[(s, t)] for t in range(len(scenarios_clean[s]))]) for s in range(num_scenarios)])
    objective_function += objective, "expected_mad"

    # target return (Expected across scenarios)
    expected_portfolio_return = pl.lpSum([probabilities[s] * mu_s[s] for s in range(num_scenarios)])
    objective_function += expected_portfolio_return >= target_return, "target_return"

    # sum of weights
    objective_function += pl.lpSum([w[a] for a in assets]) == 1, "sum_of_weights"

    # Solve
    objective_function.solve(pl.PULP_CBC_CMD(msg=False, timeLimit=120)) 
    status: str = pl.LpStatus[objective_function.status]
    
    if status != "Optimal":
        print(f"Warning ! Infeasible (status: {status}) for {target_return:.4f}.")
        return {a: 0.0 for a in assets}

    # optimal weights
    weights = {a: w[a].value() if w[a].value() is not None else 0.0 for a in assets}
    full_weights = {a: 0.0 for a in scenarios[0].columns} 
    full_weights.update(weights)
    
    return full_weights


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
    returns = utils.compute_returns(prices)

    if returns.empty:
        print("Error: No common data period found after aligning. Cannot proceed.")
        
    else:
        print(f"Data aligned. Using {returns.shape[1]} assets over {returns.shape[0]} periods.")

        num_scenarios = 10 
        T_sample = len(returns) 
        
        scenarios = []
        probabilities = [1.0 / num_scenarios] * num_scenarios
        
        print(f"\nGenerating {num_scenarios} scenarios by bootstrap (length {T_sample})...")
        
        for _ in range(num_scenarios):
            random_indices = np.random.choice(T_sample, size=T_sample, replace=True)
            scenarios.append(returns.iloc[random_indices].reset_index(drop=True))
        
        target_return_stochastic = 0.01 / 100 

        print("Stochastic optimization...\n")
        stochastic_weights = optimize_stochastic_mad_lp(scenarios, probabilities, target_return_stochastic)
        print("\n--- Optimal Stochastic Weights (Minimizing Expected MAD) ---")
        
        w_vec = np.array([stochastic_weights.get(a, 0.0) for a in returns.columns])
        
        expected_return_achieved = returns.mean().values.dot(w_vec)
        mad_achieved = utils.compute_mad(returns, w_vec)
        
        print(f"Target Expected Return: {target_return_stochastic:.6f}")
        print(f"Achieved Expected Return (on historical data): {expected_return_achieved:.6f}")
        print(f"Achieved MAD (on historical data): {mad_achieved:.6f}")
        print(f"Sum of Weights Check: {np.sum(w_vec):.4f}")
        
        print("\nTop 10 Weight Allocations:")
        count = 0
        for asset, weight in sorted(stochastic_weights.items(), key=lambda item: -item[1]):
            if weight > 0.001 and count < 10: 
                print(f"- {asset} : {weight:.4f}")
                count += 1