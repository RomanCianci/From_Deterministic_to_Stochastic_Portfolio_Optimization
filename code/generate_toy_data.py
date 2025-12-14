import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample')
ASSETS = ['US_EQ_1', 'JP_EQ_1', 'UK_BOND_1', 'WLD_COMM_1']
DAYS = 500  

def generate_asset_history(name, seed):
    np.random.seed(seed)
    dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(DAYS)]
    dates = [d for d in dates if d.weekday() < 5] 
    
    returns = np.random.normal(0.0005, 0.015, len(dates))
    price_path = 100 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        '<DATE>': [d.strftime('%Y%m%d') for d in dates],
        '<CLOSE>': price_path,
        '<OPEN>': price_path,
        '<HIGH>': price_path,
        '<LOW>': price_path,
        '<VOL>': 1000
    })
    
    folder = os.path.join(DATA_ROOT, "generated")
    os.makedirs(folder, exist_ok=True)
    
    file_path = os.path.join(folder, f"{name.lower()}.txt")
    df.to_csv(file_path, index=False)
    print(f"Generated: {file_path}")

if __name__ == "__main__":
    print("Generating Stooq-compatible Toy Dataset...")
    for i, asset in enumerate(ASSETS):
        generate_asset_history(asset, seed=i)
    print("Done.")