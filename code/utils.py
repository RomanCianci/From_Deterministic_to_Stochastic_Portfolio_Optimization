import pandas as pd
import numpy as np
import pulp as pl
import os
import glob
import gc
from typing import Dict, List, Optional


def compute_mad(returns: pd.DataFrame, w: np.ndarray) -> float:
    """Computes the Mean Absolute Deviation of the portfolio."""
    w = np.array(w)
    portfolio_returns = returns.values.dot(w)
    mean_return = portfolio_returns.mean()
    mad = np.mean(np.abs(portfolio_returns - mean_return))
    return mad

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Computes daily returns from prices."""
    return prices.pct_change().dropna(how="all")

def clean_asset_file(file_path: str, ticker: str, parse_dates: bool = True) -> pd.DataFrame:
    """Load an asset CSV/TXT file and clean it."""
    DATE_COL_STOOQ = "<DATE>"
    CLOSE_COL_STOOQ = "<CLOSE>"
    DATE_COL_FINAL = "Date"

    try:
        df = pd.read_csv(file_path, sep=",", encoding="utf-8-sig")
    except:
        return pd.DataFrame()

    if DATE_COL_STOOQ not in df.columns or CLOSE_COL_STOOQ not in df.columns:
        return pd.DataFrame()

    df = df.rename(columns={DATE_COL_STOOQ: DATE_COL_FINAL})
    price_col = CLOSE_COL_STOOQ
    
    if df[price_col].dtype != np.float32:
        df[price_col] = df[price_col].astype(np.float32)
        
    df = df[[DATE_COL_FINAL, price_col]].rename(columns={price_col: ticker})
    
    if parse_dates:
        df[DATE_COL_FINAL] = pd.to_datetime(df[DATE_COL_FINAL], format="%Y%m%d", errors='coerce')

    df = df.dropna(subset=[DATE_COL_FINAL])
    df = df.set_index(DATE_COL_FINAL).sort_index()
    return df

def align_and_merge_prices(price_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge and align asset price DataFrames using concat (faster than merge_asof loop)."""
    if not price_dfs:
        return pd.DataFrame()

    print(f"Aligning and merging {len(price_dfs)} assets...")
    dfs_prepared = []
    
    for ticker, df in price_dfs.items():
        if df.empty:
            continue
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
            
        if df.columns[0] != ticker:
            df = df.rename(columns={df.columns[0]: ticker})
            
        dfs_prepared.append(df)

    if not dfs_prepared:
        return pd.DataFrame()

    merged = pd.concat(dfs_prepared, axis=1, join="outer")
    merged.sort_index(inplace=True)
    
    del dfs_prepared
    gc.collect()
    
    
    return merged.dropna(how='all').fillna(method='ffill')

def load_stooq_assets_glob_all(base_paths: List[str], min_rows: int = 0) -> Dict[str, pd.DataFrame]:
    """Finds and loads Stooq asset data from multiple directories recursively."""
    dfs = {}
    found_files = {}
    print("Searching for assets...")
    for base_path in base_paths:
        file_list = glob.glob(os.path.join(base_path, "**/*.txt"), recursive=True) + \
                    glob.glob(os.path.join(base_path, "**/*.csv"), recursive=True)
        for file_path in file_list:
            ticker = os.path.splitext(os.path.basename(file_path))[0]
            if ticker not in found_files:
                found_files[ticker] = file_path
            
    print(f"Found {len(found_files)} potential asset files.")
    
    loaded_count = 0
    total = len(found_files)
    
    for i, (ticker, file_path) in enumerate(found_files.items()):
        if (i+1) % 100 == 0:
            print(f"Processing {i+1}/{total}...")
        try:
            df = clean_asset_file(file_path, ticker)
            if not df.empty and len(df) >= min_rows: 
                dfs[ticker] = df
                loaded_count += 1
        except:
            pass
    
    print(f"Successfully loaded data for {loaded_count} assets.")
    return dfs

def preprocess_returns_for_solver(returns: pd.DataFrame) -> pd.DataFrame:
    """Prepares returns data for PuLP by handling NaN/Inf values."""
    returns_clean = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    std_dev = returns_clean.std(axis=0)
    zero_vol_assets = std_dev[std_dev < 1e-10].index
    if not zero_vol_assets.empty:
        returns_clean = returns_clean.drop(columns=zero_vol_assets)
    return returns_clean


def optimize_mad_lp(returns: pd.DataFrame, target_return: float, lower_bounds: Optional[Dict[str, float]] = None, upper_bounds: Optional[Dict[str, float]] = None, allow_short: bool = False) -> Dict[str, float]:
    """Solves the Linearized MAD problem using LP."""
    returns = preprocess_returns_for_solver(returns)
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

    # Constraints
    mean_asset_returns = returns.mean(axis=0)
    objective_function += mu_p == pl.lpSum([mean_asset_returns[a] * w[a] for a in assets]), "mean_portfolio_return"

    ret_values = returns.values
    asset_indices = range(len(assets))

    for t in range(T):
        R_p_t = pl.lpSum([ret_values[t][i] * w[assets[i]] for i in asset_indices]) 
        objective_function += z[t] >= R_p_t - mu_p
        objective_function += z[t] >= mu_p - R_p_t

    objective_function += pl.lpSum([w[a] for a in assets]) == 1, "sum_of_weights"
    objective_function += mu_p >= target_return, "target_return"
    
    if lower_bounds:
        for i, lb in lower_bounds.items():
            objective_function += w[i] >= lb
    if upper_bounds:
        for i, ub in upper_bounds.items():
            objective_function += w[i] <= ub

    # Solve
    objective_function.solve(pl.PULP_CBC_CMD(msg=False))
    
    if pl.LpStatus[objective_function.status] != "Optimal":
        return {a: 0.0 for a in assets}

    return {a: w[a].value() if w[a].value() is not None else 0.0 for a in assets}

def optimize_mad_milp(returns: pd.DataFrame, target_return: float, K: int = 3, lower_bound: float = 0.01, upper_bound: float = 0.5, allow_short: bool = False) -> Dict[str, float]:
    """Solves the MAD problem with Cardinality (K) and Semicontinuous constraints using MILP."""
    returns = preprocess_returns_for_solver(returns)
    assets = list(returns.columns)
    T = len(returns)
    
    objective_function = pl.LpProblem("milp_portfolio_optimization", pl.LpMinimize)
    
    # Decision variables
    w = pl.LpVariable.dicts("w", assets, lowBound=None if allow_short else 0)
    z = pl.LpVariable.dicts("z", list(range(T)), lowBound=0)
    y = pl.LpVariable.dicts("y", assets, cat="Binary") 
    mu_p = pl.LpVariable("mu_p", lowBound=None)
    
    objective_function += (1.0 / T) * pl.lpSum([z[t] for t in range(T)]), "MAD_obj"
    mean_asset_returns = returns.mean()
    objective_function += mu_p == pl.lpSum([mean_asset_returns[a] * w[a] for a in assets]), "mu_def"
    
    ret_values = returns.values
    asset_indices = range(len(assets))

    for t in range(T):
        R_p_t = pl.lpSum([ret_values[t][i] * w[assets[i]] for i in asset_indices])
        objective_function += z[t] >= R_p_t - mu_p
        objective_function += z[t] >= mu_p - R_p_t
    
    objective_function += mu_p >= target_return
    objective_function += pl.lpSum([w[a] for a in assets]) == 1
    
    # Constraints with binary variables
    for a in assets:
        objective_function += w[a] <= upper_bound * y[a]
        objective_function += w[a] >= lower_bound * y[a]
    
    objective_function += pl.lpSum([y[a] for a in assets]) <= K
    
    objective_function.solve(pl.PULP_CBC_CMD(msg=False, timeLimit=30))
    
    if pl.LpStatus[objective_function.status] not in ["Optimal", "Feasible", "UserLimit"]:
        return {a: 0.0 for a in assets}
        
    return {a: w[a].value() if w[a].value() else 0.0 for a in assets}