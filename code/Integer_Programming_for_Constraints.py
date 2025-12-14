import pandas as pd
import numpy as np
import pulp as pl
from typing import Dict, Optional, List
import glob
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
    objective_function = pl.LpProblem("linear_portfolio_optimization", pl.LpMinimize)
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


    # solve
    objective_function.solve(pl.PULP_CBC_CMD(msg=False))
    status: str = pl.LpStatus[objective_function.status]

    if status != "Optimal":
        print(f"Warning ! Infeasible (status: {status}) for {target_return:.4f}.")
        return {a: 0.0 for a in assets}

    # optimal weights
    w_vals: Dict[str, float] = {
        a: w[a].value() if w[a].value() is not None else 0.0 
        for a in assets
    }
    
    return w_vals

def optimize_mad_milp(returns: pd.DataFrame, target_return: float, K: int = 3, lower_bound: float = 0.01, upper_bound: float = 0.5, allow_short: bool = False) -> Dict[str, float]:
    """
    This solves the new problem using MILP. 
    
    Args:
        returns (DataFrame) : all the returns from the assets.
        target_return (float) : the minimum return we aim for.
        lower_bound (float) : minimum weight for an asset.
        upper_bound (float]) : maximum weight for an asset.
        allow_short (bool) : True if we want to be able to short, False otherwise. 
        
    Returns:
        Dictionnary of optimal weights per asset -> {asset: weight}.
    """
    
    assets = list(returns.columns)
    T = len(returns)
    
    objective_function = pl.LpProblem("milp_portfolio_optimization", pl.LpMinimize)
    
    # Decision variables
    w = pl.LpVariable.dicts("w", assets, lowBound=None if allow_short else 0)
    z = pl.LpVariable.dicts("z", list(range(T)), lowBound=0)
    y = pl.LpVariable.dicts("y", assets, cat="Binary")  
    mu_p = pl.LpVariable("mu_p", lowBound=None)
    
    # Objective function
    objective_function += (1.0 / T) * pl.lpSum([z[t] for t in range(T)]), "MAD_obj"


    # --- CONSTRAINTS ---

    # mu_p
    mean_asset_returns = returns.mean()
    objective_function += mu_p == pl.lpSum([mean_asset_returns[a] * w[a] for a in assets]), "mu_def"
    
    # mad linearization
    for t, (_, return_values) in enumerate(returns.iterrows()):

        # R_p_t
        R_p_t = pl.lpSum([return_values[a] * w[a] for a in assets])

        # z_t : positive case
        objective_function += z[t] >= R_p_t - mu_p

        # z_t : negative case
        objective_function += z[t] >= mu_p - R_p_t
    
    # target return
    objective_function += mu_p >= target_return, "target_return"
    
    # sum of weights
    objective_function += pl.lpSum([w[a] for a in assets]) == 1, "sum_of_weights"
    
    # max weigth, min weigth
    for a in assets:
        objective_function += w[a] <= upper_bound * y[a]
        objective_function += w[a] >= lower_bound * y[a]
    
    # cardinality
    objective_function += pl.lpSum([y[a] for a in assets]) <= K, "cardinality"
    
    # Solve
    objective_function.solve(pl.PULP_CBC_CMD(msg=False, timeLimit=300))
    status: str = pl.LpStatus[objective_function.status]

    if status != "Optimal":
        print(f"Warning ! Infeasible or Time Limit (status: {status}).")
        return {a: 0.0 for a in assets}
    
    # optimal weights
    w = {a: w[a].value() if w[a].value() else 0.0 for a in assets}
    y = {a: y[a].value() for a in assets}
    
    print(f"The optimal portfolio uses {int(sum(y.values()))} assets (≤ K={K})\n")
    
    return w

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
    Finds and loads ALL Stooq asset data from multiple directories recursively.
    
    Args:
        base_paths (List[str]): List of base directories to search.
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of loaded dataframes.
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

    import os
    BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
    
    WORLD_BASE_PATHS = [
        f"{BASE_PATH}\\d_world_txt\\data\\daily\\world\\bonds",
        f"{BASE_PATH}\\d_world_txt\\data\\daily\\world\\cryptocurrencies",
        f"{BASE_PATH}\\d_world_txt\\data\\daily\\world\\currencies\\major",
        f"{BASE_PATH}\\d_world_txt\\data\\daily\\world\\currencies\\other",
        f"{BASE_PATH}\\d_world_txt\\data\\daily\\world\\money market",
        f"{BASE_PATH}\\d_world_txt\\data\\daily\\world\\stooq stocks indices",
        f"{BASE_PATH}\\d_world_txt\\data\\daily\\world\\indices",
    ]
    
    
    dfs = load_stooq_assets_glob_all(WORLD_BASE_PATHS)
    prices = align_and_merge_prices(dfs)
    returns = compute_returns(prices)

    if returns.empty:
        print("Error : No common data period found...")
        
    else:

        TARGET_RETURN = 0.05 / 100
        ALLOW_SHORT = False
        
        print(f"\nOptimizing MAD to achieve a {TARGET_RETURN:.4f}% return...")
        weights: Dict[str, float] = optimize_mad_lp(returns=returns, target_return=TARGET_RETURN, allow_short=ALLOW_SHORT)

        print("\n--- Optimal portfolio weights (LP) ---\n")
        
        count_lp = 0
        for asset, w in weights.items():
            if w > 1e-4:
                count_lp += 1
        print(f"LP used {count_lp} assets.")

        w = np.array([weights.get(a, 0.0) for a in returns.columns])
        expected_return = returns.mean().values.dot(w)
        mad = compute_mad(returns, w)

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

        w_milp = optimize_mad_milp(returns_milp, target_return=TARGET_RETURN, K=K, lower_bound=lower_bound, upper_bound=upper_bound, allow_short=ALLOW_SHORT)

        for asset, w in w_milp.items():
            if w > 1e-4:
                print(f"* {asset} : {w:.4f}")

        w_vec_milp = [w_milp.get(a, 0.0) for a in returns_milp.columns]
        expected_return_milp = returns_milp.mean().values @ w_vec_milp
        mad_milp = compute_mad(returns_milp, w_vec_milp)

        print("\n--- Optimal portfolio metrics (MILP) ---\n")
        print(f"* Expected return = {expected_return_milp:.6f}")
        print(f"* MAD = {mad_milp:.6f}\n")









