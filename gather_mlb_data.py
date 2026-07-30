import os, sys, time
import pandas as pd
from pybaseball import *

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Baseball Projects', 'savant_2026_07272026.csv')

def load_pitch_data(start_year=2021, end_year=2025):
    # Load/build database
    data_csv = path
    os.makedirs(os.path.dirname(data_csv), exist_ok=True)
    if not os.path.exists(data_csv):
        dfs = []
        for year in range(start_year, end_year+1):
            print(f"Pulling {year}...")
            try:
                df = statcast(
                    start_dt=f"{year}-03-01",
                    end_dt=f"{year}-10-31",
                    verbose=False
                )
                dfs.append(df)
                time.sleep(5)  # be nice to Savant
            except Exception as e:
                print(f"Failed for {year}: {e}")

        data = pd.concat(dfs, ignore_index=True)
        data.to_csv(data_csv)
    else:
        data = pd.read_csv(data_csv, index_col=0)

    return data

df = load_pitch_data(start_year=2023, end_year=2026)