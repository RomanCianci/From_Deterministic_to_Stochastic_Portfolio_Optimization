import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt 
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
    
    print("Loading ALL world assets from multiple Stooq directories...")
    dfs: Dict[str, pd.DataFrame] = utils.load_stooq_assets_glob_all(WORLD_BASE_PATHS)
    
    if not dfs:
        print("Error : No asset data was loaded. Cannot proceed with optimization.")
        exit()
        
    assets_list = list(dfs.keys()) 
    print(f"Total assets available for optimization: {len(assets_list)}")
        
    prices = utils.align_and_merge_prices(dfs)
    returns = utils.compute_returns(prices)

    if returns.empty:
        print("Error: No common time period found after aligning and merging prices. Cannot proceed.")
        exit()
            
    mean_returns = returns.mean()
    min_return_for_frontier = np.max(mean_returns.min() * 1.05, 0) 
    max_return_for_frontier = mean_returns.max() * 0.95
    
    NUM_POINTS = 50
    target_returns = np.linspace(min_return_for_frontier, max_return_for_frontier, NUM_POINTS)
    
    ALLOW_SHORT = False 

    frontier_results = []
    
    print(f"\nCalculating MAD Efficient Frontier (Total {NUM_POINTS} points)...")
    
    for i, target in enumerate(target_returns):
        
        if (i + 1) % 10 == 0 or i == 0 or i == NUM_POINTS - 1:
            print(f"-> Calculating point {i+1}/{NUM_POINTS} (Target: {target:.6f})...")

        weights: Dict[str, float] = utils.optimize_mad_lp(returns=returns, target_return=target, allow_short=ALLOW_SHORT)

        w = np.array([weights.get(a, 0.0) for a in returns.columns])
        expected_return = returns.mean().values.dot(w)
        mad = utils.compute_mad(returns, w)
        
        result_entry = {'Target_Return': target, 'Expected_Return': expected_return, 'MAD': mad,}

        for asset, weight_value in weights.items():
            result_entry[f'W_{asset}'] = weight_value
            
        frontier_results.append(result_entry)
        
    frontier_df = pd.DataFrame(frontier_results)
    
    export_filename = "mad_efficient_frontier_data.csv"
    export_path = os.path.join(BASE_PATH, export_filename)
    
    try:
        frontier_df.to_csv(export_path, index=False)
        print(f"\n--- SUCCESS: Data exported to {export_path} ---")
    except Exception as e:
        print(f"\n--- ERROR: Failed to export data to CSV. Reason: {e} ---")

    print("\n\n--- COMPLETE MAD EFFICIENT FRONTIER DATA (MAD, Expected Return, and ALL WEIGHTS) ---\n")
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(frontier_df.to_string())
        
    print("\n\n--- Starting Plotting of Efficient Frontier ---")

    plt.figure(figsize=(12, 8))
    plt.scatter(frontier_df['MAD'], frontier_df['Expected_Return'], c=frontier_df['Expected_Return'], cmap='viridis', marker='o', s=40) 

    min_mad_point = frontier_df.iloc[frontier_df['MAD'].idxmin()]
    plt.scatter(min_mad_point['MAD'], min_mad_point['Expected_Return'], color='red', marker='*', s=300, label='Minimum MAD Portfolio')
    plt.title('MAD Efficient Frontier (Risk vs. Expected Return)', fontsize=16)
    plt.xlabel('Risk (Mean Absolute Deviation - MAD)', fontsize=14)
    plt.ylabel('Expected Portfolio Return', fontsize=14)
    plt.colorbar(label='Expected Return')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show() 
    
    print("\n--- Process Complete ---")