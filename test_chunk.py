import os
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()
fred = Fred(api_key=os.getenv('FRED_API_KEY'))
series_id = 'DFF'
start = "2003-01-01"
end = "2024-01-01"

start_dt = pd.to_datetime(start)
end_dt = pd.to_datetime(end)

date_ranges = []
current_start = start_dt
while current_start < end_dt:
    current_end = min(current_start + pd.DateOffset(years=5) - pd.Timedelta(days=1), end_dt)
    date_ranges.append((current_start, current_end))
    current_start = current_end + pd.Timedelta(days=1)

dfs = []
for c_start, c_end in date_ranges:
    print(f"Downloading chunk {c_start.date()} to {c_end.date()}...")
    chunk_df = fred.get_series_all_releases(
        series_id, 
        realtime_start=c_start.strftime("%Y-%m-%d"), 
        realtime_end=c_end.strftime("%Y-%m-%d")
    )
    dfs.append(chunk_df)

final_df = pd.concat(dfs, ignore_index=True)
print("Total rows:", len(final_df))
