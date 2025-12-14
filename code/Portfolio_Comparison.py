import pandas as pd
import numpy as np
import pulp as pl
from typing import Dict, List, Optional, Callable
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm 
import gc  


DATE_COL_STOOQ = "<DATE>"
CLOSE_COL_STOOQ = "<CLOSE>"
DATE_COL_FINAL = "Date"

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
    Load an asset CSV file and clean it.

    Args : 
        file_path (str) : path of the CSV file.
        ticker (str) : ticker of the asset.
        parse_dates (bool) : True if should convert date column in datetime.
        dayfirst (bool) : True if DD/MM/YY, False otherwise.

    Returns : 
        Cleaned CSV file of dates/prices for the asset.
    """

    try:
        df = pd.read_csv(file_path, sep=",", encoding="utf-8-sig")
    except: 
        return pd.DataFrame() 

    if DATE_COL_STOOQ not in df.columns or CLOSE_COL_STOOQ not in df.columns: 
        return pd.DataFrame() 

    df = df.rename(columns={DATE_COL_STOOQ: DATE_COL_FINAL})
    price_col = CLOSE_COL_STOOQ
    df[price_col] = df[price_col].astype(np.float32)
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

    print("Preparing dataframes for merge...")
    dfs_prepared = []
    
    for ticker, df in tqdm(price_dfs.items(), desc="Aligning"):
        if df.empty:
            continue

        df_local = df
        if not isinstance(df_local.index, pd.DatetimeIndex):
            df_local.index = pd.to_datetime(df_local.index, errors="coerce")
        
        colname = df_local.columns[0]
        if colname != ticker:
            df_local = df_local.rename(columns={colname: ticker})

        if df_local[ticker].dtype != 'float32':
             df_local[ticker] = df_local[ticker].astype(np.float32)

        dfs_prepared.append(df_local[[ticker]])

    if not dfs_prepared:
        return pd.DataFrame()

    print(f"Merging {len(dfs_prepared)} assets... (This may take RAM)")
    
    merged = pd.concat(dfs_prepared, axis=1, join="outer")
    del dfs_prepared
    gc.collect()

    merged.sort_index(inplace=True)
    return merged

def load_stooq_assets_glob_all(base_paths: List[str], min_rows: int = 252) -> Dict[str, pd.DataFrame]:
    """
    Loads Stooq assets band keeps those with enough data.

    Args : 
        base_paths (List[str]) : paths to load the assets.
        min_rows (int) : have at least this number of data.
    
    Returns :
        Asset dataframes with at least min_rows of data.
    """
    dfs = {}
    found_files = {}
    print("Searching for assets...")

    for base_path in base_paths:
        file_list = glob.glob(os.path.join(base_path, "**/*.txt"), recursive=True) \
                    + glob.glob(os.path.join(base_path, "**/*.csv"), recursive=True)
        for file_path in file_list:
            ticker = os.path.splitext(os.path.basename(file_path))[0]
            if ticker not in found_files:
                found_files[ticker] = file_path

    print(f"Found {len(found_files)} potential asset files.")
    loaded_count = 0

    sorted_items = sorted(found_files.items())

    for i, (ticker, file_path) in enumerate(sorted_items):
        if (i + 1) % 1000 == 0:
            print(f"Scanned {i+1} files...")

        try:
            df = clean_asset_file(file_path, ticker)
            if df.empty or len(df) < min_rows:
                continue

            dfs[ticker] = df
            loaded_count += 1

        except Exception:
            continue 

    print(f"Successfully loaded data for {loaded_count} assets (after filtering).")
    return dfs

def filter_by_data_completeness(prices: pd.DataFrame, min_ratio: float, start_date: str = "2008-01-01", end_date: str = "2025-12-31") -> pd.DataFrame:
    """
    Keeps assets with enough data on the selected window.

    Args : 
        prices (Dataframe) : prices of the asset.
        min_ratio (float) : % of data to have in the window.
        start_date (str) : start date for the window.
        end_date (str) : end data for the window.
    
    Returns : 
        Filtered dataframe.
    """

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
        print(f"No dates found in the requested window {start_date} -> {end_date}.")
        return pd.DataFrame()

    required_count = len(window_prices.index)
    thresh = int(np.ceil(required_count * min_ratio))

    print(f"Calculating valid assets (Threshold: {thresh}/{required_count})...")
    non_nan_counts = window_prices.count()
    keepers = non_nan_counts[non_nan_counts >= thresh].index.tolist()

    filtered_prices = prices[keepers].copy()

    print(f"Filtering Results: {prices.shape[1]} initial assets -> " f"{len(keepers)} assets kept (need >= {thresh}/{required_count} non-NA in {start_date}->{end_date}).")

    return filtered_prices 

def get_milp_filtered_assets(returns: pd.DataFrame, n_limit: int) -> pd.DataFrame:
    """
    FIlters assets based on sharpe ratio for MILP.

    Args : 
        returns (DataFrame) : daily returns.
        n_limit (int) : number of assets to keep.

    Returns : 
        Filtered assets.
    """

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

def preprocess_returns_for_solver(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Removes assets that cause numerical issues.
    """
    
    returns_clean = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    std_dev = returns_clean.std(axis=0)
    zero_vol_assets = std_dev[std_dev < 1e-10].index
    
    if not zero_vol_assets.empty:
        returns_clean = returns_clean.drop(columns=zero_vol_assets)
        
    mean_returns = returns_clean.mean(axis=0)
    invalid_assets = mean_returns[mean_returns.isna() | mean_returns.isin([np.inf, -np.inf])].index
    
    if not invalid_assets.empty:
        returns_clean = returns_clean.drop(columns=invalid_assets)
    
    returns_final = returns_clean.dropna(how='all')

    if returns_final.empty:
         raise ValueError("Optimization returns data is empty after cleaning numerical issues.")
         
    return returns_final

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

    returns = preprocess_returns_for_solver(returns)
    
    assets = list(returns.columns)
    T = len(returns)
    objective_function = pl.LpProblem("linear_portfolio_optimization", pl.LpMinimize)
    low_bound_w = None if allow_short else 0
    w = pl.LpVariable.dicts("w", assets, lowBound=low_bound_w)
    z = pl.LpVariable.dicts("z", list(range(T)), lowBound=0)
    mu_p = pl.LpVariable("mu_p")

    objective_function += (1.0 / T) * pl.lpSum([z[i] for i in range(T)]), "objective_function"
    
    mean_asset_returns = returns.mean(axis=0) 
    
    objective_function += mu_p == pl.lpSum([mean_asset_returns[a] * w[a] for a in assets]), "mean_portfolio_return"

    ret_values = returns.values
    asset_indices = range(len(assets))

    for t in range(T):
        R_p_t = pl.lpSum([ret_values[t][i] * w[assets[i]] for i in asset_indices])  
        objective_function += z[t] >= R_p_t - mu_p
        objective_function += z[t] >= mu_p - R_p_t

    objective_function += pl.lpSum([w[a] for a in assets]) == 1, "sum_of_weigths"
    objective_function += mu_p >= target_return, "target_return"
    
    if lower_bounds:
        for i, lb in lower_bounds.items(): objective_function += w[i] >= lb
    if upper_bounds:
        for i, ub in upper_bounds.items(): objective_function += w[i] <= ub

    objective_function.solve(pl.PULP_CBC_CMD(msg=False, timeLimit=15))
    if pl.LpStatus[objective_function.status] != "Optimal": return {a: 0.0 for a in assets}
    return {a: w[a].value() if w[a].value() is not None else 0.0 for a in assets}

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

    returns = preprocess_returns_for_solver(returns)
    
    assets = list(returns.columns)
    T = len(returns)
    objective_function = pl.LpProblem("milp_portfolio_optimization", pl.LpMinimize)
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
    for a in assets:
        objective_function += w[a] <= upper_bound * y[a]
        objective_function += w[a] >= lower_bound * y[a]
    objective_function += pl.lpSum([y[a] for a in assets]) <= K
    
    MILP_TIME_LIMIT_SEC = 15 
    objective_function.solve(pl.PULP_CBC_CMD(msg=False, timeLimit=MILP_TIME_LIMIT_SEC))
    
    if pl.LpStatus[objective_function.status] != "Optimal": 
        if pl.LpStatus[objective_function.status] in ["Feasible", "UserLimit"]:
             w_vals = {a: w[a].value() if w[a].value() else 0.0 for a in assets}
             return w_vals
        return {a: 0.0 for a in assets}
    
    return {a: w[a].value() if w[a].value() else 0.0 for a in assets}

def optimize_stochastic_mad_lp(scenarios: List[pd.DataFrame], probabilities: List[float], target_return: float, allow_short: bool = False) -> Dict[str, float]:
    """
    This optimizes our problem using multiple different future scenarios. 
    
    Args:
        scenarios (List[pd.DataFrame]) : different future scenarios. 
        returns (DataFrame) : all the returns from the assets.
        probabilities (List[float]) : probability that a specific scenario happens.
        target_return (float) : the minimum return we aim for.
        allow_short (bool) : True if we want to be able to short, False otherwise. 
        
    Returns:
        Dictionnary of optimal weights per asset -> {asset: weight}.
    """
    
    scenarios_clean = [preprocess_returns_for_solver(s) for s in scenarios]
    
    if any(s.empty for s in scenarios_clean):
        raise ValueError("One or more scenarios became empty after cleaning numerical issues. Cannot run stochastic optimization.")
        
    assets = scenarios_clean[0].columns.tolist()
    num_scenarios = len(scenarios_clean)
    
    objective_function = pl.LpProblem("stochastic_portfolio_optimization", pl.LpMinimize)
    w = pl.LpVariable.dicts("w", assets, lowBound=0 if not allow_short else None)
    
    z = {}
    for s in range(num_scenarios):
        for t in range(len(scenarios_clean[s])):
             z[(s, t)] = pl.LpVariable(f"z_{s}_{t}", lowBound=0)

    mu_s = {s: pl.LpVariable(f"mu_s_{s}") for s in range(num_scenarios)}

    MAX_INDIVIDUAL_WEIGHT = 0.10 
    for a in assets:
        objective_function += w[a] <= MAX_INDIVIDUAL_WEIGHT, f"max_weight_{a}"

    for s, scenario_returns in enumerate(scenarios_clean):
        avg_r_s = scenario_returns.mean()
        objective_function += mu_s[s] == pl.lpSum([avg_r_s[a] * w[a] for a in assets]), f"mean_return_{s}"

        ret_values = scenario_returns.values
        asset_indices = range(len(assets))

        for t in range(len(scenario_returns)):
            R_p_t = pl.lpSum([ret_values[t][i] * w[assets[i]] for i in asset_indices])
            objective_function += z[(s, t)] >= R_p_t - mu_s[s]
            objective_function += z[(s, t)] >= -(R_p_t - mu_s[s])

    objective = pl.lpSum([probabilities[s] * (1.0 / len(scenarios_clean[s])) * pl.lpSum([z[(s, t)] for t in range(len(scenarios_clean[s]))]) for s in range(num_scenarios)])
    objective_function += objective, "expected_mad"

    expected_portfolio_return = pl.lpSum([probabilities[s] * mu_s[s] for s in range(num_scenarios)])
    objective_function += expected_portfolio_return >= target_return, "target_return"

    objective_function += pl.lpSum([w[a] for a in assets]) == 1, "sum_of_weights"

    STOCHASTIC_TIME_LIMIT_SEC = 30
    objective_function.solve(pl.PULP_CBC_CMD(msg=False, timeLimit=STOCHASTIC_TIME_LIMIT_SEC))
    status: str = pl.LpStatus[objective_function.status]
    
    if status != "Optimal":
        print(f"Warning ! Infeasible (status: {status}) for {target_return:.4f}.")
        return {a: 0.0 for a in assets}

    weights = {a: w[a].value() if w[a].value() is not None else 0.0 for a in assets}
    return weights

def optimization_backtest(returns: pd.DataFrame, training_length: int, rebalance_frequency: int, target_return: float, transaction_cost: float, optimization_type: Callable, **optimizer_kwargs) -> pd.Series:
    """
    Simulates a dynamic rebalacing using the selecter optimiser.

    Args:
        returns (DataFrame) : all returns from the period.
        training_length (int) : length of the training window in days.
        rebalance_frequency (int) : frequency of rebalancing in days.
        target_return (float) : the minimum return we aim for.
        transaction_cost (float) : % to pay for changing weights of the assets.
        optimization_type (Callable) : the type of optimization we want to backtest.
        **optimizer_kwargs : additional arguments specific to the chosen optimization type.
    
    Returns:
        A series of daily returns of the dynamic portfolio.
    """

    current_weights = np.zeros(returns.shape[1])
    portfolio_returns = []

    all_assets = returns.columns
    asset_filter_limit = optimizer_kwargs.pop('asset_filter_limit', None)

    num_iterations = (len(returns) - training_length) // rebalance_frequency
    for k in tqdm(range(num_iterations), desc=f"Backtesting {optimization_type.__name__}"):
        i = training_length + k * rebalance_frequency
        train_start = max(0, i - training_length)
        train_end = i

        training_returns = returns.iloc[train_start:train_end]
        current_returns = training_returns.copy()

        if asset_filter_limit:
            current_returns = get_milp_filtered_assets(training_returns, asset_filter_limit)

        is_stochastic = optimization_type == optimize_stochastic_mad_lp

        try:
            if is_stochastic:
                num_scenarios = optimizer_kwargs.get("num_scenarios", 10)
                if current_returns.empty:
                    raise ValueError("Empty returns for stochastic scenarios after filtering.")

                scenarios_list = [
                    current_returns.sample(n=len(current_returns), replace=True).reset_index(drop=True)
                    for _ in range(num_scenarios)
                ]
                probabilities = [1.0 / num_scenarios] * num_scenarios
                optimal_weights = optimization_type(scenarios=scenarios_list, probabilities=probabilities, target_return=target_return, **{k: v for k, v in optimizer_kwargs.items() if k != "num_scenarios"})
            else:
                if current_returns.empty:
                    raise ValueError("Empty returns after filtering.")
                optimal_weights = optimization_type(current_returns, target_return, **optimizer_kwargs)

        except Exception as e:
            print(f"[Backtest] Skipping window {k} (i={i}) due to error: {e}. Asset check: {current_returns.shape[1]} assets.")
            continue

        new_weights = np.array([optimal_weights.get(a, 0.0) for a in all_assets])

        turnover = np.sum(np.abs(new_weights - current_weights))
        cost = transaction_cost * turnover

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
    return pd.Series(portfolio_returns, index=idx)




if __name__ == "__main__":
    
    import os
    BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
    
    ASSET_BASE_PATHS = [

        # World (d_world_txt)
        f"{BASE_PATH}\\d_world_txt\\data\\daily\\world\\bonds",
        f"{BASE_PATH}\\d_world_txt\\data\\daily\\world\\cryptocurrencies",
        f"{BASE_PATH}\\d_world_txt\\data\\daily\\world\\currencies\\major",
        f"{BASE_PATH}\\d_world_txt\\data\\daily\\world\\currencies\\other",
        f"{BASE_PATH}\\d_world_txt\\data\\daily\\world\\money market",
        f"{BASE_PATH}\\d_world_txt\\data\\daily\\world\\stooq stocks indices",
        f"{BASE_PATH}\\d_world_txt\\data\\daily\\world\\indices",
        f"{BASE_PATH}\\d_world_txt\\data\\daily\\us",
        f"{BASE_PATH}\\d_world_txt\\data\\daily\\uk",
        f"{BASE_PATH}\\d_world_txt\\data\\daily\\jp",
        f"{BASE_PATH}\\d_world_txt\\data\\daily\\macro",

        # Japan (d_jp_txt)
        f"{BASE_PATH}\\d_jp_txt\\data\\daily\\jp\\tse corporate bonds",
        f"{BASE_PATH}\\d_jp_txt\\data\\daily\\jp\\tse etfs",
        f"{BASE_PATH}\\d_jp_txt\\data\\daily\\jp\\tse futures",
        f"{BASE_PATH}\\d_jp_txt\\data\\daily\\jp\\tse indices",
        f"{BASE_PATH}\\d_jp_txt\\data\\daily\\jp\\tse options",
        f"{BASE_PATH}\\d_jp_txt\\data\\daily\\jp\\tse stocks",
        f"{BASE_PATH}\\d_jp_txt\\data\\daily\\jp\\tse treasury bonds",

        # UK (d_uk_txt)
        f"{BASE_PATH}\\d_uk_txt\\data\\daily\\uk\\lse etfs\\1",
        f"{BASE_PATH}\\d_uk_txt\\data\\daily\\uk\\lse etfs\\2",
        f"{BASE_PATH}\\d_uk_txt\\data\\daily\\uk\\lse etfs\\3",
        f"{BASE_PATH}\\d_uk_txt\\data\\daily\\uk\\lse stocks",
        f"{BASE_PATH}\\d_uk_txt\\data\\daily\\uk\\lse stocks intl\\1",
        f"{BASE_PATH}\\d_uk_txt\\data\\daily\\uk\\lse stocks intl\\2",

        # USA (d_us_txt)
        f"{BASE_PATH}\\d_us_txt\\data\\daily\\us\\nasdaq etfs",
        f"{BASE_PATH}\\d_us_txt\\data\\daily\\us\\nasdaq stocks\\1",
        f"{BASE_PATH}\\d_us_txt\\data\\daily\\us\\nasdaq stocks\\2",
        f"{BASE_PATH}\\d_us_txt\\data\\daily\\us\\nasdaq stocks\\3",
        f"{BASE_PATH}\\d_us_txt\\data\\daily\\us\\nyse etfs\\1", 
        f"{BASE_PATH}\\d_us_txt\\data\\daily\\us\\nyse etfs\\2",
        f"{BASE_PATH}\\d_us_txt\\data\\daily\\us\\nyse stocks\\1",
        f"{BASE_PATH}\\d_us_txt\\data\\daily\\us\\nyse stocks\\2",
        f"{BASE_PATH}\\d_us_txt\\data\\daily\\us\\nysemkt stocks",

    ]
    
    MIN_DATA_COMPLETENESS = 0.5       
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
    STOCHASTIC_SCENARIOS = 10       
    MIN_START_DATE = "2008-01-01"
    MIN_END_DATE = "2025-07-01"
    
    dfs = load_stooq_assets_glob_all(ASSET_BASE_PATHS)
    raw_prices = align_and_merge_prices(dfs)
    del dfs
    gc.collect()
    prices = filter_by_data_completeness(raw_prices, MIN_DATA_COMPLETENESS, start_date=MIN_START_DATE, end_date=MIN_END_DATE)
    del raw_prices
    gc.collect()
    returns = compute_returns(prices)

    if returns.empty:
        print("Error: No common data period found after filtering. Cannot proceed.")
        exit()
        
    returns = returns.loc[MIN_START_DATE:] 

    if returns.empty:
        print("Error: Returns is empty after restricting to the modern window (2008-01-01). Cannot proceed.")
        exit()
        
    returns = returns.fillna(0.0)

    print(f"\nFINAL DATASET STATS:")
    print(f"Total Data Period: {returns.index.min().date()} to {returns.index.max().date()}")
    print(f"Final Assets Used: {returns.shape[1]}")
        
    split_point = int(len(returns) * TRAIN_TEST_SPLIT)
    in_sample_returns = returns.iloc[:split_point]
    out_of_sample_returns = returns.iloc[split_point:]
    
    print(f"\nTraining period (static/in-sample) : {in_sample_returns.index.min().date()} to {in_sample_returns.index.max().date()}")
    print(f"Testing period (out-of-sample) : {out_of_sample_returns.index.min().date()} to {out_of_sample_returns.index.max().date()}\n")


    # Static Portfolios
    
    
    print("--- 1. Static Portfolios (Optimized on In-Sample Data) ---\n")
    
    static_weights = {}
    
    # LP 
    print(f"* LP Optimization (Filtered to {LP_ASSET_LIMIT} assets)...")
    lp_in_sample_returns = get_milp_filtered_assets(in_sample_returns, LP_ASSET_LIMIT)
    static_weights['LP'] = optimize_mad_lp(lp_in_sample_returns, TARGET_RETURN)
    
    # MILP 
    print(f"* MILP Optimization (K={MILP_K}, Filtered to {MILP_ASSET_LIMIT} assets)...")
    milp_in_sample_returns = get_milp_filtered_assets(in_sample_returns, MILP_ASSET_LIMIT)
    static_weights['MILP'] = optimize_mad_milp(milp_in_sample_returns, TARGET_RETURN, K=MILP_K, lower_bound=MILP_LOWER_BOUND, upper_bound=MILP_UPPER_BOUND)

    # Stochastic 
    print(f"* Stochastic Optimization (Filtered to {STOCHASTIC_ASSET_LIMIT} assets, {STOCHASTIC_SCENARIOS} scenarios)...")
    stochastic_in_sample_returns = get_milp_filtered_assets(in_sample_returns, STOCHASTIC_ASSET_LIMIT)
    
    scenarios_list = [stochastic_in_sample_returns.sample(n=len(stochastic_in_sample_returns), replace=True).reset_index(drop=True) for _ in range(STOCHASTIC_SCENARIOS)]
    scenario_probabilities = [1.0 / STOCHASTIC_SCENARIOS] * STOCHASTIC_SCENARIOS
    static_weights['Stochastic'] = optimize_stochastic_mad_lp(scenarios=scenarios_list, probabilities=scenario_probabilities, target_return=TARGET_RETURN)

    weights_df = pd.DataFrame({k: {a: w.get(a, 0.0) for a in returns.columns} for k, w in static_weights.items()}).fillna(0)
    print("\n--- Optimal static portfolio weights (Top 5 Assets per Strategy) ---\n")
    
    display_weights = {}
    for col in weights_df.columns:
        display_weights[col] = weights_df[col].sort_values(ascending=False).head(5)
    print(pd.DataFrame(display_weights).round(4).fillna('-'))

    results = []
    for name, weights_dict in static_weights.items():
        w_vec = np.array([weights_dict.get(a, 0.0) for a in out_of_sample_returns.columns])
        if np.sum(w_vec) < 0.99: w_vec = np.zeros_like(w_vec) 
        
        oos_returns = out_of_sample_returns.dot(w_vec)
        realized_return = oos_returns.mean()
        realized_mad = np.mean(np.abs(oos_returns - realized_return))
        sharpe_ratio = (realized_return / oos_returns.std()) * np.sqrt(252) if oos_returns.std() > 0 else 0
        results.append({"Strategy": name, "Annualized return (%)": realized_return * 252 * 100, "Annualized MAD (%)": realized_mad * np.sqrt(252) * 100, "Sharpe ratio": sharpe_ratio,})

    static_results_df = pd.DataFrame(results).set_index("Strategy")
    print("\n--- Out-of-sample performance (static) ---\n")
    print(static_results_df.round(4))


    # Dynamic Portfolios 

    
    print("\n\n--- 2. Dynamic Portfolios (Backtested) ---\n")

    dynamic_returns = {}
    
    # Dynamic LP 
    lp_params = {"asset_filter_limit": LP_ASSET_LIMIT}
    dynamic_returns['Dynamic LP'] = optimization_backtest(returns, DYNAMIC_TRAIN_WINDOW, DYNAMIC_REBALANCE_FREQ, TARGET_RETURN, DYNAMIC_COST, optimization_type=optimize_mad_lp, **lp_params)

    # Dynamic MILP 
    milp_params = {"K": MILP_K, "lower_bound": MILP_LOWER_BOUND, "upper_bound": MILP_UPPER_BOUND, "asset_filter_limit": MILP_ASSET_LIMIT}
    dynamic_returns['Dynamic MILP'] = optimization_backtest(returns, DYNAMIC_TRAIN_WINDOW, DYNAMIC_REBALANCE_FREQ, TARGET_RETURN, DYNAMIC_COST, optimization_type=optimize_mad_milp, **milp_params)

    # Dynamic Stochastic 
    stochastic_params = {"num_scenarios": STOCHASTIC_SCENARIOS, "asset_filter_limit": STOCHASTIC_ASSET_LIMIT}
    dynamic_returns['Dynamic Stochastic'] = optimization_backtest(returns, DYNAMIC_TRAIN_WINDOW, DYNAMIC_REBALANCE_FREQ, TARGET_RETURN, DYNAMIC_COST, optimization_type=optimize_stochastic_mad_lp, **stochastic_params)


    dynamic_results = []
    for name, returns_series in dynamic_returns.items():
        oos_returns = returns_series.loc[out_of_sample_returns.index.min():out_of_sample_returns.index.max()]
        
        realized_return = oos_returns.mean()
        realized_mad = np.mean(np.abs(oos_returns - realized_return))
        sharpe_ratio = (realized_return / oos_returns.std()) * np.sqrt(252) if oos_returns.std() > 0 else 0
        dynamic_results.append({"Strategy": name, "Annualized return (%)": realized_return * 252 * 100, "Annualized MAD (%)": realized_mad * np.sqrt(252) * 100, "Sharpe ratio": sharpe_ratio,})
    dynamic_results_df = pd.DataFrame(dynamic_results).set_index("Strategy")

    print("\n--- Out-of-sample performance (dynamic) ---\n")
    print(dynamic_results_df.round(4))
    

    plt.figure(figsize=(14, 8))
    
    equal_weights = np.array([1.0 / returns.shape[1]] * returns.shape[1])
    bh_returns = returns.loc[out_of_sample_returns.index.min():out_of_sample_returns.index.max()].dot(equal_weights)
    bh_cumulative = (1 + bh_returns).cumprod() * 100
    plt.plot(bh_cumulative.index, bh_cumulative, label="Benchmark (Buy & Hold)", linestyle='--', color='gray')


    for name, returns_series in dynamic_returns.items():
        oos_returns = returns_series.loc[out_of_sample_returns.index.min():out_of_sample_returns.index.max()]
        cumulative_performance = (1 + oos_returns).cumprod() * 100 
        plt.plot(cumulative_performance.index, cumulative_performance, label=name)
            
    plt.title("Evolution of portfolio value from $100 (Dynamic Strategy Comparison)", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Portfolio value ($)")
    plt.legend()
    plt.grid(True)
    plt.show()













