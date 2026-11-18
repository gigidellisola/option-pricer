"""Download an option chain, filter it, and build an implied volatility smile.

Data comes from Yahoo Finance through yfinance. It is unofficial and delayed, 
so it is suitable for illustration only.
"""

import yfinance as yf 
from datetime import datetime, timedelta 
import pandas as pd 
from implied_vol import implied_vol 

def choose_expiry(expirations, target_days, min_days):
    """Pick the listed expiry closest to a target time to expiry.
    
    Only expiries at least min_days away are considered. Assumes expiry at
    16:00 in the machine's local time zone (US options expire at 16:00 
    Eastern). 
    
    Args:
        expirations: List of dates as "YYYY-MM-DD" strings.
        target_days: Preferred days to expiry.
        min_days: Minimum days to expiry.
        
    Returns:
        Tuple (expiry string, T in years, using calendar days / 365), or 
        (None, None) if no expiry qualifies.
    """
    valid_expirations = []
    lengths = []
    today = datetime.now()
    target = target_days * 24 * 60 * 60
    for exp_str in expirations: 
        expiry = datetime.strptime(exp_str, "%Y-%m-%d")
        # assumes expiry at 16:00 (US Eastern)
        expiry += timedelta(hours=16)
        secs_to_expiry = (expiry - today).total_seconds()
        if secs_to_expiry >= (min_days * 24 * 60 * 60):
            valid_expirations.append(exp_str)
            lengths.append(secs_to_expiry)
    if len(valid_expirations) != 0:
        distances = [abs(x - target) for x in lengths]
        loc = distances.index(min(distances))
        chosen_expiry = valid_expirations[loc]
        secs_to_expiry = lengths[loc]
        T = secs_to_expiry / (365 * 24 * 60 * 60)
        return chosen_expiry, T
    else: 
        return None, None

def get_risk_free_rate():
    """Return the latest 13-week US Treasury bill yield (^IRX) as a decimal. 
    
    This is a short-rate proxy used for all maturities. It is a quoted yield, 
    not a continuously compounded rate. Requires network access.
    """
    treasury = yf.Ticker("^IRX")
    rate_data = treasury.history(period="5d")
    r = rate_data["Close"].iloc[-1] / 100 
    return r 

def filter_liquid(options, S, min_volume, min_open_interest):
    """Keep liquid, out-of-the-money options and compute their midpoints.
    
    Keeps calls with strike at or above S and puts with strike below S (the
    out-of-the-money side is usually more liquid and carries less early 
    exercise value). Other conditions: strike within 25% of S, bid and ask 
    both positive, volume and open interest strictly above the minimums, and 
    a bid-ask spread under 30% of the midpoint. Adds a "midpoint" column, 
    (bid + ask) / 2, to the tables it is given.
    
    Args:
        options: Object with .calls and .puts DataFrames (yfinance format).
        S: Spot price. 
        
    Returns: 
        Tuple (liquid_calls, liquid_puts, listed_calls, listed_puts, 
        filtered_call_count, filtered_put_count).
    """
    calls = options.calls
    puts = options.puts
    listed_calls = len(calls)
    listed_puts = len(puts)
    calls["midpoint"] = (calls["bid"] + calls["ask"]) / 2
    puts["midpoint"] = (puts["bid"] + puts["ask"]) / 2
    # out-of-the-money options only: more liquid, less early-exercise value. 
    liquid_calls = calls[(calls["volume"] > min_volume) & (calls["openInterest"] > min_open_interest) & (calls["bid"] > 0) & (calls["ask"] > 0) & (calls["strike"] >= S) & (abs(calls["strike"] - S) / S <= 0.25) & (((calls["ask"] - calls["bid"]) / calls["midpoint"]) < 0.3)].copy()
    liquid_puts = puts[(puts["volume"] > min_volume) & (puts["openInterest"] > min_open_interest) & (puts["bid"] > 0) & (puts["ask"] > 0) & (puts["strike"] < S) & (abs(puts["strike"] - S) / S <= 0.25) & (((puts["ask"] - puts["bid"]) / puts["midpoint"]) < 0.3)].copy()
    filtered_call_count = len(liquid_calls)
    filtered_put_count = len(liquid_puts)
    return liquid_calls, liquid_puts, listed_calls, listed_puts, filtered_call_count, filtered_put_count

def add_implied_vol(liquid_calls, liquid_puts, listed_calls, listed_puts, filtered_call_count, filtered_put_count, S, T, r):
    """Compute implied volatility from each option's midpoint price. 
    
    Options for which the solver returns NaN are dropped. 
    
    Returns: 
        Tuple: (valid_calls, valid_puts, counts). counts is a dictionary with 
        the listed, filtered, NaN, and valid counts for calls and puts. 
        """
    if len(liquid_calls) == 0: 
        liquid_calls["volatility"] = []
    else: 
        liquid_calls["volatility"] = liquid_calls.apply(lambda row: implied_vol(S, row["strike"], T, r, row["midpoint"], "call"), axis=1)
    if len(liquid_puts) == 0:
        liquid_puts["volatility"] = []
    else: 
        liquid_puts["volatility"] = liquid_puts.apply(lambda row: implied_vol(S, row["strike"], T, r, row["midpoint"], "put"), axis=1)
    nan_count_calls = liquid_calls["volatility"].isna().sum()
    nan_count_puts = liquid_puts["volatility"].isna().sum()
    valid_liquid_calls = liquid_calls.dropna(subset=["volatility"])
    valid_liquid_puts = liquid_puts.dropna(subset=["volatility"])
    valid_count_calls = len(valid_liquid_calls)
    valid_count_puts = len(valid_liquid_puts)
    return valid_liquid_calls, valid_liquid_puts, {"call_listed_count": round(listed_calls), "put_listed_count": round(listed_puts), "filtered_call_count": round(filtered_call_count), "filtered_put_count": round(filtered_put_count), "call_nan_count": round(nan_count_calls), "put_nan_count": round(nan_count_puts), "call_valid_count": round(valid_count_calls), "put_valid_count": round(valid_count_puts)}

def build_smile(valid_liquid_calls, valid_liquid_puts):
    """Combine calls and puts into one table sorted by strike. 
    
    Returns:
        DataFrame with columns "strike", "volatility", and "option type".
    """
    smile_calls = valid_liquid_calls[["strike", "volatility"]].copy()
    smile_calls["option type"] = "call"
    smile_puts = valid_liquid_puts[["strike", "volatility"]].copy()
    smile_puts["option type"] = "put"
    smile_table = pd.concat([smile_calls, smile_puts], ignore_index=True)
    smile_table = smile_table.sort_values("strike")
    return smile_table

def fetch_smile(stock, target_days, min_days, min_volume, min_open_interest):
    """Build the implied volatility smile for one stock at one expiry. 
    
    Downloads the option chain, chooses an expiry, filters for liquidity 
    and solves for implied volatility. Requires network access.
    
    Returns:    
        Dictionary with keys ticker, spot, rate, time, expiry, smile, counts, 
        and message. message is None on success. Otherwise it describes the 
        failure: no valid expiry, no options passing the filter or solver, or
        fewer than 5 valid points.
    """
    ticker = yf.Ticker(stock)
    expirations = ticker.options 
    S = ticker.fast_info["lastPrice"]
    r = get_risk_free_rate()
    chosen_expiry, T = choose_expiry(expirations, target_days, min_days)
    if chosen_expiry is None: 
        smile_table = pd.DataFrame(columns=["strike", "volatility", "option type"])
        counts = {"call_listed_count": 0, "put_listed_count": 0, "filtered_call_count": 0, "filtered_put_count": 0, "call_nan_count": 0, "put_nan_count": 0, "call_valid_count": 0, "put_valid_count": 0}
        return {"ticker": stock, "spot": S, "rate": r, "time": T, "expiry": chosen_expiry, "smile": smile_table, "counts": counts, "message": "no valid expiry within timeframe"}
    else: 
        options = ticker.option_chain(chosen_expiry)
        liquid_calls, liquid_puts, listed_calls, listed_puts, filtered_call_count, filtered_put_count = filter_liquid(options, S, min_volume, min_open_interest)
        valid_liquid_calls, valid_liquid_puts, counts = add_implied_vol(liquid_calls, liquid_puts, listed_calls, listed_puts, filtered_call_count, filtered_put_count, S, T, r)
        smile_table = build_smile(valid_liquid_calls, valid_liquid_puts)
        if len(smile_table) == 0:
            message = "no options passed the liquidity filter or solver failed"
        elif len(smile_table) < 5:
            message = "too few valid options to build smile"
        else: 
            message = None
        return {"ticker": stock, "spot": S, "rate": float(r), "time": T, "expiry": chosen_expiry, "smile": smile_table, "counts": counts, "message": message}


    




