import pytest
from datetime import datetime, timedelta
import yfinance as yf 
import numpy as np
import pandas as pd
import types
import argparse
import market_data
from market_data import choose_expiry, filter_liquid, add_implied_vol, build_smile, fetch_smile
from black_scholes import black_scholes

def test_expiry():
    target_days, min_days = 45, 14
    today = datetime.today()
    expirations = [(today + timedelta(days=d)).strftime("%Y-%m-%d") for d in (10, 30, 50, 90)]
    chosen, T = choose_expiry(expirations, target_days, min_days)
    assert chosen == (today + timedelta(days=50)).strftime("%Y-%m-%d")
    assert T == pytest.approx(50 / 365, abs=0.005)
    expirations_invalid = [(today + timedelta(days=d)).strftime("%Y-%m-%d") for d in (1, 2, 3 ,4)]
    assert choose_expiry(expirations_invalid, target_days, min_days) == (None, None)

def test_filter():
    calls_data = {"strike": [105, 90, 110, 112, 108, 109, 115, 130, 100], "bid": [2.0, 12.0, 0.0, 1.0, 1.5, 1.4, 0.5, 0.05, 5.0], "ask": [2.2, 12.4, 1.0, 0.0, 1.7, 1.6, 1.0, 0.06, 5.2], "volume": [50, 50, 50, 50, 2, 50, 50, 50, 50], "openInterest": [100, 100, 100, 100, 100, 3, 100, 100, 100]}
    puts_data = {"strike": [95, 110, 100, 90, 88, 92, 91, 85, 70], "bid": [2.0, 12.0, 5.0, 0.0, 1.0, 1.5, 1.4, 0.5, 0.05], "ask": [2.2, 12.4, 5.2, 1.0, 0.0, 1.7, 1.6, 1.0, 0.06], "volume": [50, 50, 50, 50, 50, 2, 50, 50, 50], "openInterest": [100, 100, 100, 100, 100, 100, 3, 100, 100]}
    calls_table = pd.DataFrame(calls_data)
    puts_table = pd.DataFrame(puts_data)
    options = types.SimpleNamespace(calls=calls_table, puts=puts_table)
    liquid_calls, liquid_puts, _, _, filtered_call_count, filtered_put_count = filter_liquid(options, 100, 2, 3)
    assert liquid_calls["strike"].tolist() == [105, 100]
    assert liquid_puts["strike"].tolist() == [95]
    assert liquid_calls["midpoint"].tolist() == pytest.approx([2.1, 5.1], abs=1e-4)
    assert liquid_puts["midpoint"].tolist() == pytest.approx([2.1], abs=1e-4)
    assert filtered_call_count == 2 
    assert filtered_put_count == 1

@pytest.mark.parametrize("one_side_table", [(pd.DataFrame({"strike": [], "midpoint": []}), pd.DataFrame({"strike": [95, 90], "midpoint": [5.774751, 	4.024075]})), (pd.DataFrame({"strike": [105, 110], "midpoint": [9.556256, 7.635066]}), pd.DataFrame({"strike": [], "midpoint": []}))])
def test_add_implied_vol(one_side_table):
    S, T, r, sigma = 100, 1.0, 0.04, 0.25 
    call_strikes = [105, 110, 115, 120]
    call_midpoints = []
    put_strikes = [95, 90, 85]
    put_midpoints = []
    for strike in call_strikes:
        price, _, _ = black_scholes(S, strike, T, r, sigma, "call")
        call_midpoints.append(price)
    for strike in put_strikes:
        price, _, _ = black_scholes(S, strike, T, r, sigma, "put")
        put_midpoints.append(price)
    call_midpoints[3] = 0.0  
    calls = pd.DataFrame({"strike": call_strikes, "midpoint": call_midpoints})
    puts = pd.DataFrame({"strike": put_strikes, "midpoint": put_midpoints})
    valid_calls, valid_puts, counts = add_implied_vol(calls, puts, 10, 8, 4, 3, S, T, r)
    assert counts["call_nan_count"] == 1
    assert counts["call_valid_count"] == 3
    assert counts["put_nan_count"] == 0 
    assert counts["put_valid_count"] == 3
    assert counts["call_listed_count"] == 10 
    assert counts["put_listed_count"] == 8 
    assert counts["filtered_call_count"] == 4
    assert counts["filtered_put_count"] == 3
    assert 120 not in valid_calls["strike"]
    assert (valid_calls["volatility"] - sigma).abs().max() < 1e-3
    assert (valid_puts["volatility"] - sigma).abs().max() < 1e-3
    empty_calls, empty_puts = pd.DataFrame({"strike": [], "midpoint": []}), pd.DataFrame({"strike": [], "midpoint": []})
    valid_calls, valid_puts, counts = add_implied_vol(empty_calls, empty_puts, 10, 8, 4, 3, S, T, r)
    assert counts["call_nan_count"] == 0
    assert counts["call_valid_count"] == 0
    assert counts["put_nan_count"] == 0 
    assert counts["put_valid_count"] == 0
    assert counts["call_listed_count"] == 10 
    assert counts["put_listed_count"] == 8 
    assert counts["filtered_call_count"] == 4
    assert counts["filtered_put_count"] == 3
    assert len(valid_calls) == 0 
    assert len(valid_puts) == 0 
    assert "volatility" in valid_calls.columns
    assert "volatility" in valid_puts.columns
    calls, puts = one_side_table 
    valid_calls, valid_puts, counts = add_implied_vol(calls, puts, 10, 8, 4, 3, S, T, r)
    assert counts["call_nan_count"] == 0 
    assert counts["put_nan_count"] == 0 
    assert counts["call_listed_count"] == 10 
    assert counts["put_listed_count"] == 8 
    assert counts["filtered_call_count"] == 4
    assert counts["filtered_put_count"] == 3
    if len(calls) == 0:
        assert valid_calls.shape == (0, 3)
        assert valid_puts.shape == (2, 3)
        assert counts["call_valid_count"] == 0 
        assert counts["put_valid_count"] == 2 
        assert (valid_puts["volatility"] - sigma).abs().max() < 1e-3
    if len(puts) == 0:
        assert valid_calls.shape == (2, 3)
        assert valid_puts.shape == (0, 3)
        assert counts["call_valid_count"] == 2
        assert counts["put_valid_count"] == 0 
        assert (valid_calls["volatility"] - sigma).abs().max() < 1e-3  

def test_build_smile():
    calls = pd.DataFrame({"strike": [110, 105], "volatility": [0.31, 0.30]})
    puts = pd.DataFrame({"strike": [95, 90], "volatility": [0.27, 0.28]})   
    smile_table = build_smile(calls, puts)
    assert smile_table["strike"].tolist() == [90, 95, 105, 110]
    assert list(smile_table.columns) == ["strike", "volatility", "option type"]
    assert smile_table["option type"].tolist() == ["put", "put", "call", "call"]
    assert smile_table.iloc[0][["strike", "volatility"]].tolist() == [90, 0.28]
    one_call = pd.DataFrame({"strike": [110], "volatility": [0.31]})
    calls_empty = pd.DataFrame({"strike": [], "volatility": []})
    puts_empty = pd.DataFrame({"strike": [], "volatility": []})
    smile_table_asym = build_smile(one_call, puts_empty)
    smile_table_empty = build_smile(calls_empty, puts_empty)
    assert smile_table_asym["strike"].tolist() == [110]
    assert list(smile_table_asym.columns) == ["strike", "volatility", "option type"]
    assert smile_table_asym["option type"].tolist() == ["call"]
    assert len(smile_table_empty) == 0
    assert list(smile_table_empty.columns) == ["strike", "volatility", "option type"]


RATE = 0.04
EXPS = [(datetime.today() + timedelta(days=d)).strftime("%Y-%m-%d") for d in (10, 30, 50, 90)]
PLUS50 = (datetime.today() + timedelta(days=50)).strftime("%Y-%m-%d")

def fake_smile(volume=50, rate=RATE):
    S, T, sigma = 100, 50 / 365, 0.25 
    call_strikes = [102, 105, 108]
    put_strikes = [98, 95, 92]
    call_midpoints, put_midpoints = [], []
    for strike in call_strikes:
        midpoint, _, _ = black_scholes(S, strike, T, rate, sigma, "call")
        call_midpoints.append(midpoint)
    for strike in put_strikes:
        midpoint, _, _ = black_scholes(S, strike, T, rate, sigma, "put")
        put_midpoints.append(midpoint)
    call_bids = np.array(call_midpoints) * 0.95 
    put_bids = np.array(put_midpoints) * 0.95 
    call_asks = np.array(call_midpoints) * 1.05 
    put_asks = np.array(put_midpoints) * 1.05 
    call_smile = pd.DataFrame({"strike": call_strikes, "bid": call_bids, "ask": call_asks, "volume": volume, "openInterest": 100})
    put_smile = pd.DataFrame({"strike": put_strikes, "bid": put_bids, "ask": put_asks, "volume": volume, "openInterest": 100})
    return call_smile, put_smile


def make_args(ticker="AAPL", target_days=45, min_days=14, **others):
    return argparse.Namespace(ticker=ticker, target_days=target_days, min_days=min_days, **others)

def install_fakes(monkeypatch, volume=50):
    requests = []
    chain = fake_smile(volume=volume)
    def fake_option_chain(expiry):
        requests.append(expiry)
        calls, puts = fake_smile(volume=volume, rate=RATE)
        return types.SimpleNamespace(calls=calls, puts=puts)
    fake_ticker = types.SimpleNamespace(options=EXPS, fast_info={"lastPrice": 100.0}, option_chain=fake_option_chain)
    monkeypatch.setattr(market_data.yf, "Ticker", lambda symbol: fake_ticker)
    monkeypatch.setattr(market_data, "get_risk_free_rate", lambda: RATE)
    return requests 

def test_smile_normal(monkeypatch):
    requests = install_fakes(monkeypatch)
    result = fetch_smile("TEST", 45, 14, 2, 3)
    smile = result["smile"]
    counts = result["counts"]
    assert result["message"] is None 
    assert result["expiry"] == PLUS50 
    assert requests == [PLUS50]
    assert result["spot"] == 100.0 and result["rate"] == RATE
    assert result["time"] == pytest.approx(50 / 365, abs=0.005)
    assert len(result["smile"]) == 6
    assert counts["call_listed_count"] == 3
    assert counts["put_listed_count"] == 3
    assert counts["filtered_call_count"] == 3
    assert counts["filtered_put_count"] == 3
    assert counts["call_nan_count"] == 0 
    assert counts["put_nan_count"] == 0
    assert counts["call_valid_count"] == 3
    assert counts["put_valid_count"] == 3
    for vol in smile["volatility"].tolist():
        assert 0.2 < vol < 0.3

def test_smile_no_expiry(monkeypatch):
    requests = install_fakes(monkeypatch)
    result = fetch_smile("TEST", 45, 5000, 2, 3)
    assert result["expiry"] is None 
    assert result["message"] is not None
    assert len(result["smile"]) == 0 
    assert requests == []
    for count, value in result["counts"].items():
        assert value == 0 

def test_smile_nothing_liquid(monkeypatch):
    requests = install_fakes(monkeypatch, volume=0)
    result = fetch_smile("TEST", 45, 14, 2, 3)
    counts = result["counts"]
    assert result["message"] is not None
    assert len(result["smile"]) == 0 
    assert requests == [PLUS50]
    assert counts["call_listed_count"] == 3
    assert counts["put_listed_count"] == 3
    assert counts["filtered_call_count"] == 0
    assert counts["filtered_put_count"] == 0
    assert counts["call_nan_count"] == 0 
    assert counts["put_nan_count"] == 0
    assert counts["call_valid_count"] == 0
    assert counts["put_valid_count"] == 0

def test_smile_same_keys(monkeypatch):
    requests = install_fakes(monkeypatch)
    normal = fetch_smile("TEST", 45, 14, 2, 3)
    no_expiry = fetch_smile("TEST", 45, 5000, 2, 3)
    assert set(normal) == set(no_expiry)
    assert set(normal["counts"]) == set(no_expiry["counts"])



