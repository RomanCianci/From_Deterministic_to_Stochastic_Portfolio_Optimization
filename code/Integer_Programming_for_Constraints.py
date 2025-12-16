import pandas as pd
import numpy as np
import argparse
import os
from typing import Dict
import utils

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
        print("Error : No common data period found...")
        
    else:

        TARGET_RETURN = 0.05 / 100
        ALLOW_SHORT = False
        
        print(f"\nOptimizing MAD to achieve a {TARGET_RETURN:.4f}% return...")
        weights: Dict[str, float] = utils.optimize_mad_lp(returns=returns, target_return=TARGET_RETURN, allow_short=ALLOW_SHORT)

        print("\n--- Optimal portfolio weights (LP) ---\n")
        
        count_lp = 0
        for asset, w in weights.items():
            if w > 1e-4:
                count_lp += 1
        print(f"LP used {count_lp} assets.")

        w = np.array([weights.get(a, 0.0) for a in returns.columns])
        expected_return = returns.mean().values.dot(w)
        mad = utils.compute_mad(returns, w)

        print("\n--- Optimal portfolio metrics (LP) ---\n")
        print(f"* Expected return = {expected_return:.6f} (target = {TARGET_RETURN:.6f})")
        print(f"* MAD = {mad:.6f}\n")

        print("-"*50)
        print("\n--- Optimal portfolio weights (MILP) ---\n")

        N_MILP_LIMIT = 50 
        mean_returns = returns.mean()
        volatility = returns.std()
        score = mean_returns / volatility
        
        milp_assets = score.sort_values(ascending=False).head(N_MILP_LIMIT).index.tolist()
        returns_milp = returns[milp_assets]
        
        print(f"Reducing universe to {len(milp_assets)} assets for MILP calculation.")

        K = 5                
        lower_bound = 0.05   
        upper_bound = 0.7   

        w_milp = utils.optimize_mad_milp(returns_milp, target_return=TARGET_RETURN, K=K, lower_bound=lower_bound, upper_bound=upper_bound, allow_short=ALLOW_SHORT)

        for asset, w in w_milp.items():
            if w > 1e-4:
                print(f"* {asset} : {w:.4f}")

        w_vec_milp = [w_milp.get(a, 0.0) for a in returns_milp.columns]
        expected_return_milp = returns_milp.mean().values @ w_vec_milp
        mad_milp = utils.compute_mad(returns_milp, w_vec_milp)

        print("\n--- Optimal portfolio metrics (MILP) ---\n")
        print(f"* Expected return = {expected_return_milp:.6f}")
        print(f"* MAD = {mad_milp:.6f}\n")