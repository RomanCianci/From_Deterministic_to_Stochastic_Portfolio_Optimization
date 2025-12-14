import pandas as pd
import numpy as np
import pulp as pl
from typing import Dict, Optional, List
import glob
import argparse
import os
import matplotlib.pyplot as plt 







def compute_mad(returns: pd.DataFrame, w: np.ndarray) -> float:
    """
    Computes the Mean Average Deviation of the portfolio.

    Args : 
        returns (DataFrame) : all the returns from the assets.
        w (ndarray) : weights of the assets.
    
    Returns : 
        Mean Average Deviation of the portfolio.    
    """

    w = np.array(w)
        
    portfolio_returns = returns.values.dot(w)
    mean_return = portfolio_returns.mean()
    mad = np.mean(np.abs(portfolio_returns - mean_return))

    return mad

def optimize_mad_lp(returns: pd.DataFrame, target_return: float, lower_bounds: Optional[Dict[str, float]] = None, upper_bounds: Optional[Dict[str, float]] = None, allow_short: bool = False) -> Dict[str, float]:
    """
    This solves the linearized version of our problem. 
    
    Args:
        returns (DataFrame) : all the returns from the assets.
        target_return (float) : the minimum return we aim for.
        lower_bounds (Optional[Dict[str, float]]) : minimum weight per asset.
        upper_bounds (Optional[Dict[str, float]]) : maximum weight per asset.
        allow_short (bool) : True if we want to be able to short, False otherwise. 
        
    Returns:
        Dictionnary of optimal weights per asset -> {asset: weight}.
    """

    assets = list(returns.columns)
    T = len(returns)
    objective_function = pl.LpProblem("lp_portfolio_optimization", pl.LpMinimize)
    low_bound_w = None if allow_short else 0
    
    # Decision variables
    w = pl.LpVariable.dicts("w", assets, lowBound=low_bound_w)
    z = pl.LpVariable.dicts("z", list(range(T)), lowBound=0)
    mu_p = pl.LpVariable("mu_p")

    # Objective function
    objective_function += (1.0 / T) * pl.lpSum([z[i] for i in range(T)]), "objective_function"


    # --- CONSTRAINTS ---

    # mu_p
    mean_asset_returns = returns.mean(axis=0)
    objective_function += mu_p == pl.lpSum([mean_asset_returns[a] * w[a] for a in assets]), "mean_portfolio_return"

    # mad linearization
    for t, (_, return_values) in enumerate(returns.iterrows()):
        
        # R_p_t
        R_p_t = pl.lpSum([return_values[asset] * w[asset] for asset in assets]) 
        
        # z_t : positive case
        objective_function += z[t] >= R_p_t - mu_p, f"mad_linearization_positive_case_{t}"
        
        # z_t : negative case
        objective_function += z[t] >= mu_p - R_p_t, f"mad_linearization_negative_case_{t}"

    # sum of weigths = 1
    objective_function += pl.lpSum([w[a] for a in assets]) == 1, "sum_of_weigths"
    
    # target return
    objective_function += mu_p >= target_return, "target_return"
    
    # min weigth
    if lower_bounds:
        for i, lower_bound in lower_bounds.items():
            if lower_bound < 0 and not allow_short:
                raise ValueError(f"Your lower bound '{lower_bound}' is negative. Shorting is not allowed here.")
            objective_function += w[i] >= lower_bound, f"lower_bound_{i}"
            
    # max weight
    if upper_bounds:
        for i, upper_bound in upper_bounds.items():
            objective_function += w[i] <= upper_bound, f"upper_bound_{i}"


    # Solve
    objective_function.solve(pl.PULP_CBC_CMD(msg=False))
    status: str = pl.LpStatus[objective_function.status]

    if status != "Optimal":
        return {a: 0.0 for a in assets}

    # optimal weights
    w_vals: Dict[str, float] = {
        a: w[a].value() if w[a].value() is not None else 0.0 
        for a in assets
    }
    
    return w_vals

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily returns from prices.

    Args : 
        prices (DataFrame) : prices of the assets.
    
    Returns : 
        Dataframe of daily returns for each asset.
    """

    return prices.pct_change().dropna(how="all")

def clean_asset_file(file_path: str, ticker: str, parse_dates: bool = True, dayfirst: bool = False) -> pd.DataFrame:
    """
    Load an asset CSV/TXT file (Stooq format) and clean it.

    Args : 
        file_path (str) : path of the CSV file.
        ticker (str) : ticker of the asset.
        parse_dates (bool) : True if should convert date column in datetime.
        dayfirst (bool) : True if DD/MM/YY, False if YYYY-MM-DD or YYYYMMDD. (Set to False for Stooq)

    Returns : 
        Cleaned CSV file of dates/prices for the asset.
    """
    
    DATE_COL_STOOQ = "<DATE>"
    CLOSE_COL_STOOQ = "<CLOSE>"
    DATE_COL_FINAL = "Date"

    try:
        df = pd.read_csv(file_path, sep=",", encoding="utf-8-sig")
    except FileNotFoundError:
        print(f"File {file_path} wasn't found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading {file_path} : {e}")
        return pd.DataFrame()

    if DATE_COL_STOOQ not in df.columns or CLOSE_COL_STOOQ not in df.columns:
        print(f"Error: Stooq columns ('{DATE_COL_STOOQ}', '{CLOSE_COL_STOOQ}') not found in {file_path}. Headers might be missing or wrong.")
        return pd.DataFrame()

    df = df.rename(columns={DATE_COL_STOOQ: DATE_COL_FINAL})
    price_col = CLOSE_COL_STOOQ
    df = df[[DATE_COL_FINAL, price_col]].rename(columns={price_col: ticker})
    
    if parse_dates:
        df[DATE_COL_FINAL] = pd.to_datetime(df[DATE_COL_FINAL], format="%Y%m%d", errors='coerce')

    df = df.dropna(subset=[DATE_COL_FINAL])
    df = df.set_index(DATE_COL_FINAL).sort_index()

    return df

def align_and_merge_prices(price_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge and align asset price DataFrames.

    Args : 
        prices (Dict[str, pd.DataFrame]) : prices for each asset.

    Returns :    
        Merged and aligned asset price DataFrames.
    """
    
    if not price_dfs:
        return pd.DataFrame()

    first_ticker = next(iter(price_dfs))
    merged_df = price_dfs[first_ticker].reset_index().sort_values("Date")
    
    for ticker, ticker_df in price_dfs.items():
        if ticker != first_ticker:
            ticker_df = ticker_df.reset_index().sort_values("Date")
            merged_df = pd.merge_asof(merged_df, ticker_df, on="Date", direction="nearest", tolerance=pd.Timedelta(days=7))

    prices = merged_df.set_index("Date")
    prices = prices.dropna(how='any').sort_index() 

    return prices

def load_stooq_assets_glob_all(base_paths: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Finds and loads ALL Stooq asset data from multiple directories recursively.

    Args:
        base_paths (List[str]): List of base directories to search (e.g., bonds, crypto, indices).

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of DataFrames for ALL found and successfully loaded assets.
    """

    dfs: Dict[str, pd.DataFrame] = {}
    found_files: Dict[str, str] = {}
    
    print("Searching for ALL assets in provided directories...")

    for base_path in base_paths:
        search_pattern_txt = os.path.join(base_path, "**/*.txt")
        search_pattern_csv = os.path.join(base_path, "**/*.csv")
        
        file_list = glob.glob(search_pattern_txt, recursive=True) + glob.glob(search_pattern_csv, recursive=True)
        
        for file_path in file_list:
            filename_with_ext = os.path.basename(file_path)
            ticker, ext = os.path.splitext(filename_with_ext)
            
            if ticker not in found_files:
                found_files[ticker] = file_path
    
    if not found_files:
        print("Error: No asset files were found based on the provided paths.")
        return {}
        
    total_assets = len(found_files)
    print(f"Found {total_assets} potential asset files. Starting data cleaning...")
    
    count_loaded = 0
    for i, (ticker, file_path) in enumerate(found_files.items()):
        
        if (i + 1) % 100 == 0 or (i + 1) == total_assets:
            print(f"Processing asset {i + 1}/{total_assets}...")

        try:
            df = clean_asset_file(file_path, ticker, parse_dates=True, dayfirst=False)
            
            if not df.empty and ticker in df.columns:
                dfs[ticker] = df
                count_loaded += 1
            
        except Exception as e:
            print(f"CRITICAL ERROR processing {ticker} from {file_path}: {e}. Skipping.")


    print(f"Successfully loaded data for {len(dfs)}/{total_assets} assets.")
    return dfs







if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run with toy data')
    args = parser.parse_args()

    BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

    if args.demo:
        print("Warning : We are currently running on demo mode (toy data)")
        BASE_PATH = os.path.join(BASE_DIR, 'sample')
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
    dfs: Dict[str, pd.DataFrame] = load_stooq_assets_glob_all(WORLD_BASE_PATHS)
    
    if not dfs:
        print("Error : No asset data was loaded. Cannot proceed with optimization.")
        exit()
        
    assets_list = list(dfs.keys()) 
    print(f"Total assets available for optimization: {len(assets_list)}")
        
    prices = align_and_merge_prices(dfs)
    returns = compute_returns(prices)

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

        weights: Dict[str, float] = optimize_mad_lp(returns=returns, target_return=target, allow_short=ALLOW_SHORT)

        w = np.array([weights.get(a, 0.0) for a in returns.columns])
        expected_return = returns.mean().values.dot(w)
        mad = compute_mad(returns, w)
        
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




