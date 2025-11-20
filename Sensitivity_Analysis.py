import pandas as pd
import numpy as np
import pulp as pl
from typing import Dict, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm


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

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily returns from prices.

    Args : 
        prices (DataFrame) : prices of the assets.
    
    Returns : 
        Dataframe of daily returns for each asset.
    """
    return prices.pct_change().dropna(how="all")

def clean_asset_file(file_path: str, ticker: str, parse_dates: bool = True, dayfirst: bool = True) -> pd.DataFrame:
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
        df = pd.read_csv(file_path, sep=None, engine="python", encoding="utf-8-sig", parse_dates=["Date"] if parse_dates else None, dayfirst=dayfirst)
    except FileNotFoundError:
        print(f"File {file_path} wasn't found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error downloadinf {file_path} : {e}")
        return pd.DataFrame()

    price_col = df.columns[1] 
    df = df[["Date", price_col]].rename(columns={price_col: ticker})
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=dayfirst, errors='coerce')
    df = df.dropna(subset=["Date"])
    df = df.set_index("Date").sort_index()

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
            merged_df = pd.merge_asof(merged_df, ticker_df, on="Date", direction="nearest", tolerance=pd.Timedelta(days=3))

    prices = merged_df.set_index("Date")
    prices = prices.dropna(how='any').sort_index() 

    return prices

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
    objective_function.solve(pl.PULP_CBC_CMD(msg=False))
    status: str = pl.LpStatus[objective_function.status]

    if status != "Optimal":
        print(f"Warning ! Infeasible (status: {status}) for {target_return:.4f}.")
        return {a: 0.0 for a in assets}
    
    # optimal weights
    w = {a: w[a].value() if w[a].value() else 0.0 for a in assets}
    y = {a: y[a].value() for a in assets}
    
    return w

# New functions

def efficient_frontier_lp(returns, target_returns, allow_short=False, lower_bounds=None, upper_bounds=None):
    """
    Computes the efficient frontier using LP.

    Args :
        returns (DataFrame) : all the returns from the assets.
        target_returns (List[float]) : minimum returns we aim for.       
        allow_short (bool) : True if we want to be able to short, False otherwise. 
        lower_bounds (Optional[Dict[str, float]]) : minimum weight per asset.
        upper_bounds (Optional[Dict[str, float]]) : maximum weight per asset.
    
    Returns :
        List of dictionnaries where each dictionnary represents a point on the efficient frontier. 
    """

    mad_evolution = []

    for target_return in tqdm(target_returns, desc="efficient_frontier_lp"):
        w = optimize_mad_lp(returns, target_return, lower_bounds=lower_bounds, upper_bounds=upper_bounds, allow_short=allow_short)
        w_array = np.array([w[a] for a in returns.columns])
        expected_return = float(np.dot(returns.mean(axis=0).values, w_array))
        mad = compute_mad(returns, w_array)
        mad_evolution.append({"target_return": target_return, "expected_return": expected_return, "mad": mad, "weights": w})

    return mad_evolution

def efficient_frontier_milp(returns, target_returns, K=3, lower_bounds=0.01, upper_bounds=0.5, allow_short=False):
    """
    Computes the efficient frontier using MILP.
    
    Args :
        returns (DataFrame) : all the returns from the assets.
        target_returns (List[float]) : minimum returns we aim for.       
        allow_short (bool) : True if we want to be able to short, False otherwise. 
        lower_bounds (Optional[Dict[str, float]]) : minimum weight per asset.
        upper_bounds (Optional[Dict[str, float]]) : maximum weight per asset.
    
    Returns :
        List of dictionnaries where each dictionnary represents a point on the efficient frontier. 
    """
    
    mad_evolution = []
    for target_return in tqdm(target_returns, desc="efficient_frontier_milp"):
        w = optimize_mad_milp(returns, target_return, K=K, lower_bound=lower_bounds, upper_bound=upper_bounds, allow_short=allow_short)
        w_array = np.array([w[a] for a in returns.columns])
        expected_return = float(np.dot(returns.mean(axis=0).values, w_array))
        mad = compute_mad(returns, w_array)
        mad_evolution.append({"target_return": target_return, "expected_return": expected_return, "mad": mad, "weights": w})

    return mad_evolution

def plot_efficient_frontiers(mad_evolution_lp, mad_evolution_milp=None):
    """
    Plots the 2 efficient frontiers.

    Args : 
        mad_evolution_milp (List[Dict]) : points of the efficient frontier using LP.
        mad_evolution_milp (List[Dict]) : points of the efficient frontier using MILP.
    
    Returns : 
        Plots the evolution of MAD based on expected return for LP and MILP.
    """

    plt.figure(figsize=(8,6))
    mad_lp = [r["mad"] for r in mad_evolution_lp]
    expected_return_lp = [r["expected_return"] for r in mad_evolution_lp]
    plt.plot(mad_lp, expected_return_lp, '-o', label="MAD LP")

    if mad_evolution_milp is not None:
        mad_milp = [r["mad"] for r in mad_evolution_milp]
        expected_return_milp = [r["expected_return"] for r in mad_evolution_milp]
        plt.plot(mad_milp, expected_return_milp, '-s', label="MAD MILP")

    plt.xlabel("MAD")
    plt.ylabel("Expected return")
    plt.title("Efficient frontier - Evolution of MAD based on expected return")
    plt.legend()
    plt.grid(True)
    plt.show()

def asset_weight_variation(points, tickers):
    """
    Compute average distance between adjacent frontier weights.

    Args : 
        points (List[Dict]) : points of the efficient frontier.
        tickers (List[str]) : asset names.

    Return : 
        Average distance between adjacent frontier weights and individual distances.
    """

    weights = [np.array([p["weights"][t] for t in tickers]) for p in points]
    distances = [np.sum(np.abs(weights[i+1] - weights[i])) for i in range(len(weights) - 1)]

    return np.mean(distances), distances

def count_assets(rows, threshold=1e-5):
    counts = [sum(1 for v in r["weights"].values() if abs(v) > threshold) for r in rows]
    return counts

def weight_variance(returns, target_return, number_of_bootstraps=100, solver="lp", milp_config=None):
    """
    Boostraps and re-solves to estimate weights variance.

    Args : 
        returns (DataFrame) : all the returns from the assets.
        target_return (float) : minimum return we aim for.       
        number_of_bootstraps (int) : number of bootstraps.
        solver ("lp" or "milp") : type of solver.
        milp_config (Dict) : MILP arguments if solver=="lp".

    Returns :
        Dictionnary of the assets and the bootstrap weight variance.
    """

    N = returns.shape[1]
    W = np.zeros((number_of_bootstraps, N))
    T = len(returns)    
    asset_names = list(returns.columns)

    for b in range(number_of_bootstraps):
        i = np.random.choice(T, size=T, replace=True)
        sample = returns.iloc[i]

        if solver == "lp":
            w = optimize_mad_lp(sample, target_return)
        else:
            config = milp_config or {}
            w = optimize_mad_milp(sample, target_return, K=config.get("K", 3),lower_bound=config.get("lower_bound", 0.01), upper_bound=config.get("upper_bound", 0.5))
        
        W[b, :] = np.array([w[a] for a in asset_names])

    variances = dict(zip(asset_names, W.var(axis=0)))

    return variances




if __name__ == "__main__":

    BASE_PATH = r"C:\Users\romsc\OneDrive - De Vinci\UVic\Courses\Optimization\Portfolio_Optimization"
    ASSETS = ["SPX", "AAPL", "MSFT", "GOOG", "JPM", "XOM", "WMT", "TSLA", "UNH", "V", "NVDA", "DIS", "EFA", "VWO", "TM", "GLD", "USO", "BND", "VNQ", "UUP"]

    dfs = {}
    for ticker in ASSETS:
        file_path = f"{BASE_PATH}\\{ticker}.csv"
        
        df = clean_asset_file(file_path, ticker)
        dfs[ticker] = df


    if len(dfs) != len(ASSETS):
        print("Error : Couldn't load 1 df per asset...")
        
    else:
        
        prices = align_and_merge_prices(dfs)
        returns = compute_returns(prices)
        mean_return = returns.mean().mean()
        target_returns = list(np.linspace(0.0005, 0.0025, 15))


        # frontiers
        lp_frontier = efficient_frontier_lp(returns, target_returns, allow_short=False)
        milp_K = 2
        milp_frontier = efficient_frontier_milp(returns, target_returns, K=milp_K, lower_bounds=0.05, upper_bounds=0.7, allow_short=False)

        plot_efficient_frontiers(lp_frontier, milp_frontier)


        # weight stability
        asset_names = list(returns.columns)
        lp_avg_dist, lp_dists = asset_weight_variation(lp_frontier, asset_names)
        milp_avg_dist, milp_dists = asset_weight_variation(milp_frontier, asset_names)
        lp_counts = count_assets(lp_frontier)
        milp_counts = count_assets(milp_frontier)

        print(f"\nLP average adjacent weight variation : {lp_avg_dist:.4f}")
        print(f"MILP average adjacent weight variation : {milp_avg_dist:.4f}\n")
        print("Assets used per target return (MILP) :", milp_counts)


        # weight variance
        target_for_bootstrap = target_returns[len(target_returns)//2]
        number_of_bootstraps = 10

        print(f"\nCalculating weigth variance (LP) for {number_of_bootstraps} scenarios...")
        weigth_variance_lp = weight_variance(returns, target_for_bootstrap, number_of_bootstraps=number_of_bootstraps, solver="lp")
        print(f"Weight variance (LP) for target return {target_for_bootstrap} :\n")
        print(weigth_variance_lp)

        print(f"\nCalculating weigth variance (MILP) for {number_of_bootstraps} scenarios...") 
        weigth_variance_milp = weight_variance(returns, target_for_bootstrap, number_of_bootstraps=number_of_bootstraps, solver="milp", milp_config={"K":milp_K, "lower_bound":0.05, "upper_bound":0.7})
        print(f"Weight variance (MILP) for target return {target_for_bootstrap} :\n")
        print(weigth_variance_milp)














