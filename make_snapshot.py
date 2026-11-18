from market_data import fetch_smile
import yfinance as yf 
import pandas as pd
from datetime import datetime

data = fetch_smile("AAPL", 45, 14, 2, 3)
smile_table = data["smile"]
smile_table.to_csv("data/AAPL-2026-11-17_smile.csv", index=False)
live_data = pd.DataFrame([{"ticker": data["ticker"], "spot": data["spot"], "rate": data["rate"], "maturity": data["time"], "expiry": data["expiry"], "retrieved at": datetime.now()}])
live_data.to_csv("data/AAPL-2026-11-17_data.csv", index=False)
raw_data = yf.Ticker("AAPL").option_chain(data["expiry"])
raw_data.calls.to_csv("data/AAPL-2026-11-17_calls.csv", index=False)
raw_data.puts.to_csv("data/AAPL-2026-11-17_puts.csv", index=False)

print(data["spot"])