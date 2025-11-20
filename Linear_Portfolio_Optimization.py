import pandas as pd
import numpy as np
import pulp as pl
from typing import Dict, Optional



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

        TARGET_RETURN = 0.05 / 100
        ALLOW_SHORT = False

        
        print(f"\nOptimizing MAD to achieve a {TARGET_RETURN:.4f}% return...")
        weights: Dict[str, float] = optimize_mad_lp(returns=returns, target_return=TARGET_RETURN, allow_short=ALLOW_SHORT)

        print("\n--- Optimal portfolio weights ---\n")
        for asset, w in weights.items():
            print(f"* {asset} : {w:.4f}")

            
        w = np.array([weights.get(a, 0.0) for a in returns.columns])
        expected_return = returns.mean().values.dot(w)
        mad = compute_mad(returns, w)

        print("\n--- Optimal portfolio metrics ---\n")
        print(f"* Expected return = {expected_return:.6f} (target = {TARGET_RETURN:.6f})")
        print(f"* MAD = {mad:.6f}\n")






















