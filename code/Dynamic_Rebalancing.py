import pandas as pd
import numpy as np
import pulp as pl
from typing import Dict, Optional, List
import matplotlib.pyplot as plt
import glob
import argparse
import os




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
    Load an asset CSV/TXT file and clean it.

    Args : 
        file_path (str) : path of the CSV file.
        ticker (str) : ticker of the asset.
        parse_dates (bool) : True if should convert date column in datetime.
        dayfirst (bool) : True if DD/MM/YY, False otherwise.

    Returns : 
        Cleaned CSV file of dates/prices for the asset.
    """
    DATE_COL_STOOQ = "<DATE>"
    CLOSE_COL_STOOQ = "<CLOSE>"
    DATE_COL_FINAL = "Date"
    try:
        df = pd.read_csv(file_path, sep=",", encoding="utf-8-sig")
    except: return pd.DataFrame()

    if DATE_COL_STOOQ not in df.columns or CLOSE_COL_STOOQ not in df.columns: return pd.DataFrame()

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
    if not price_dfs: return pd.DataFrame()
    first_ticker = next(iter(price_dfs))
    merged_df = price_dfs[first_ticker].reset_index().sort_values("Date")
    for ticker, ticker_df in price_dfs.items():
        if ticker != first_ticker:
            ticker_df = ticker_df.reset_index().sort_values("Date")
            merged_df = pd.merge_asof(merged_df, ticker_df, on="Date", direction="nearest", tolerance=pd.Timedelta(days=7))
    prices = merged_df.set_index("Date")
    return prices.dropna(how='any').sort_index()

def load_stooq_assets_glob_all(base_paths: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Finds and loads ALL Stooq asset data from multiple directories recursively.
    """
    dfs = {}
    found_files = {}
    print("Searching for assets...")
    for base_path in base_paths:
        file_list = glob.glob(os.path.join(base_path, "**/*.txt"), recursive=True) + glob.glob(os.path.join(base_path, "**/*.csv"), recursive=True)
        for file_path in file_list:
            ticker = os.path.splitext(os.path.basename(file_path))[0]
            if ticker not in found_files: found_files[ticker] = file_path
            
    print(f"Found {len(found_files)} potential asset files.")
    
    loaded_count = 0
    for i, (ticker, file_path) in enumerate(found_files.items()):
        if (i+1)%100==0: print(f"Loaded {i+1}...")
        try:
            df = clean_asset_file(file_path, ticker)
            if not df.empty: 
                dfs[ticker] = df
                loaded_count += 1
        except: pass
    
    print(f"Successfully loaded data for {loaded_count} assets.")
    return dfs

# New function

def optimize_dynamic_mad_lp(returns: pd.DataFrame, target_return: float, periods: int, max_turnover: float, transaction_cost: float = 0.0):
    """
    This solves the linearized version of our problem with turnover constraints.
    
    Args:
        returns (DataFrame) : all the returns from the assets.
        target_return (float) : the minimum return we aim for.
        periods (int) : number of iterations to optimize the portfolio.
        max_turnover (float) : maximum allowed turnover per period.
        transaction_cost (float) : % to pay for changing weights of the assets.
        
    Returns:
        List of optimal weights per period.
    """

    N = len(returns)
    period_len = N // periods
    assets = returns.columns.tolist()
    weights_periods = []
    
    min_bounds = {a: 0.0 for a in assets}
    max_bounds = {a: 1.0 for a in assets}
    allow_short = False

    current_weights = {a: 1.0/len(assets) for a in assets} 

    print(f"\n--- Starting Dynamic Optimization ({periods} periods, Max Turnover: {max_turnover:.3f}) ---")

    for i in range(periods):
        start = i * period_len
        end = (i+1) * period_len if i < periods - 1 else N
        period_returns = returns.iloc[start:end]
        
        # --- CONSTRAINTS ---
        
        T_period = len(period_returns)
        assets_period = list(period_returns.columns)
        
        objective_function = pl.LpProblem(f"dynamic_opt_period_{i+1}", pl.LpMinimize)
        low_bound_w = None if allow_short else 0
        
        # Decision variables
        w = pl.LpVariable.dicts("w", assets_period, lowBound=low_bound_w)
        z = pl.LpVariable.dicts("z", list(range(T_period)), lowBound=0)
        mu_p = pl.LpVariable("mu_p")
        d = pl.LpVariable.dicts("d", assets_period, lowBound=0) 

        # objective function
        objective_function += (1.0 / T_period) * pl.lpSum([z[t] for t in range(T_period)]), "objective_function"

        # mean return
        mean_asset_returns = period_returns.mean(axis=0)
        objective_function += mu_p == pl.lpSum([mean_asset_returns[a] * w[a] for a in assets_period]), "mean_portfolio_return"
        objective_function += mu_p >= target_return, "target_return"
        
        # mad linearization
        for t, (_, return_values) in enumerate(period_returns.iterrows()):
            R_p_t = pl.lpSum([return_values[asset] * w[asset] for asset in assets_period])  
            objective_function += z[t] >= R_p_t - mu_p
            objective_function += z[t] >= mu_p - R_p_t

        # sum of weigths = 1
        objective_function += pl.lpSum([w[a] for a in assets_period]) == 1, "sum_of_weigths"

        # turnover constraint, linearization 
        for a in assets_period:
            w_prev = current_weights.get(a, 0.0)
            objective_function += d[a] >= w[a] - w_prev, f"turnover_positive_{a}"
            objective_function += d[a] >= w_prev - w[a], f"turnover_negative_{a}"
            
        objective_function += pl.lpSum([d[a] for a in assets_period]) <= max_turnover, "max_total_turnover"

        # min/max weigth
        for a in assets_period:
            objective_function += w[a] >= min_bounds[a]
            objective_function += w[a] <= max_bounds[a]
        
        # solve
        objective_function.solve(pl.PULP_CBC_CMD(msg=False))
        
        if pl.LpStatus[objective_function.status] == "Optimal":
            w_new = {a: w[a].value() if w[a].value() is not None else 0.0 for a in assets_period}
            turnover_actual = sum(abs(w_new.get(a, 0.0) - current_weights.get(a, 0.0)) for a in assets_period)
        else:
            print(f"Period {i+1}: Infeasible. Keeping previous weights.")
            w_new = current_weights
            turnover_actual = 0.0

        cost = transaction_cost * turnover_actual
        print(f"Period {i+1}/{periods} : Actual Turnover = {turnover_actual:.6f}, Cost = {cost:.6f}")

        weights_periods.append(w_new)
        current_weights = w_new

    return weights_periods





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
    

    dfs = load_stooq_assets_glob_all(WORLD_BASE_PATHS)
    prices = align_and_merge_prices(dfs)
    returns = compute_returns(prices)

    if returns.empty:
        print("Error : No common data period found...")
        
    else:

        target_return = 0.05 / 100
        periods = 20
        transaction_cost = 0.001
        MAX_TURNOVER = 0.5            

        weights_dynamic = optimize_dynamic_mad_lp(returns, target_return, periods, MAX_TURNOVER, transaction_cost)
        
        periods_list = list(range(len(weights_dynamic)))
        assets = returns.columns
        
        plt.figure(figsize=(12, 6))
        plotted_assets = 0
        asset_history = {a: [weights_dynamic[p].get(a, 0.0) for p in periods_list] for a in assets}

        for a in assets:
            if max(asset_history[a]) > 0.05: 
                plt.plot(periods_list, asset_history[a], label=a)
                plotted_assets += 1
        
        plt.xlabel("Period")
        plt.ylabel("Weight")
        plt.title(f"Dynamic Rebalancing Weights Over Time (Max Turnover: {MAX_TURNOVER})")
        
        if 0 < plotted_assets < 20: 
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()








