import pandas as pd
import numpy as np
import pulp as pl
from typing import Dict, List, Optional
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

def preprocess_returns_for_solver(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares returns data for PuLP by handling NaN/Inf values.
    
    Args:
        returns (DataFrame) : raw returns data.
        
    Returns:
        DataFrame : clean returns data.
    """

    returns_clean = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    std_dev = returns_clean.std(axis=0)
    zero_vol_assets = std_dev[std_dev < 1e-10].index
    if not zero_vol_assets.empty:
        returns_clean = returns_clean.drop(columns=zero_vol_assets)
        
    return returns_clean

def optimize_stochastic_mad_lp(scenarios: List[pd.DataFrame], probabilities: List[float], target_return: float, allow_short: bool = False) -> Dict[str, float]:
    """
    This optimizes our problem using multiple different future scenarios. 
    
    Args:
        scenarios (List[pd.DataFrame]) : different future scenarios. 
        probabilities (List[float]) : probability that a specific scenario happens.
        target_return (float) : the minimum return we aim for.
        allow_short (bool) : True if we want to be able to short, False otherwise. 
        
    Returns:
        Dictionnary of optimal weights per asset -> {asset: weight}.
    """
    
    scenarios_clean = [preprocess_returns_for_solver(s) for s in scenarios]
    common_assets = set(scenarios_clean[0].columns)
    for s in scenarios_clean[1:]:
        common_assets = common_assets.intersection(set(s.columns))
    
    assets = sorted(list(common_assets))
    if not assets:
         return {a: 0.0 for a in scenarios[0].columns} 
         
    scenarios_clean = [s[assets] for s in scenarios_clean]
    num_scenarios = len(scenarios_clean)
    

    # Objective function
    objective_function = pl.LpProblem("stochastic_portfolio_optimization", pl.LpMinimize)
    
    # Decision variables
    w = pl.LpVariable.dicts("w", assets, lowBound=0 if not allow_short else None)
    z = {}
    for s in range(num_scenarios):
        for t in range(len(scenarios_clean[s])):
             z[(s, t)] = pl.LpVariable(f"z_{s}_{t}", lowBound=0)

    mu_s = {s: pl.LpVariable(f"mu_s_{s}") for s in range(num_scenarios)}


    # --- CONSTRAINTS ---

    # max individual weight (Diversification constraint)
    MAX_INDIVIDUAL_WEIGHT = 0.10 
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
    prices = prices.dropna(how='any').sort_index() 

    return prices

def load_stooq_assets_glob_all(base_paths: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Finds and loads all Stooq asset data from multiple directories recursively.
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
        
        target_return_stochastic = 0.05 / 100 

        print("Stochastic optimization...\n")
        stochastic_weights = optimize_stochastic_mad_lp(scenarios, probabilities, target_return_stochastic)
        print("\n--- Optimal Stochastic Weights (Minimizing Expected MAD) ---")
        
        w_vec = np.array([stochastic_weights.get(a, 0.0) for a in returns.columns])
        
        expected_return_achieved = returns.mean().values.dot(w_vec)
        mad_achieved = compute_mad(returns, w_vec)
        
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

















