import pytest
import argparse
import pandas as pd
import matplotlib as plt 
from main import get_parser, prices_printer, convergence_printer, variance_printer, greeks_printer, exotics_printer, market_printer, run_prices, run_convergence, run_variance, run_greeks, run_exotics, run_market, convergence_plotter, variance_plotter, greeks_panels_plotter, greeks_value_plotter, exotics_paths, exotics_barrier, market_plotter

plt.use("Agg")

def test_pricer(capsys):
    args = get_parser().parse_args(["prices"])
    section = run_prices(args)
    prices_printer(section, args)
    out = capsys.readouterr().out
    assert 'None' not in out 
    assert "=== PRICES ===" in out 

def test_convergence(capsys):
    args = get_parser().parse_args(["convergence"])
    section = run_convergence(args)
    convergence_printer(section, args)
    out = capsys.readouterr().out
    assert 'None' not in out 
    assert "=== CONVERGENCE ===" in out 

def test_variance(capsys):
    args = get_parser().parse_args(["variance"])
    section = run_variance(args)
    variance_printer(section, args)
    out = capsys.readouterr().out
    assert 'None' not in out 
    assert "=== VARIANCE REDUCTION ===" in out 

def test_greeks(capsys):
    args = get_parser().parse_args(["greeks"])
    section = run_greeks(args)
    greeks_printer(section, args)
    out = capsys.readouterr().out
    assert 'None' not in out 
    assert "=== GREEKS ===" in out 

def test_exotics(capsys):
    args = get_parser().parse_args(["exotics", "--trials", "2000"])
    section = run_exotics(args)
    exotics_printer(section, args)
    out = capsys.readouterr().out
    assert 'None' not in out 
    assert "=== EXOTICS ===" in out 

COUNTS_ZERO = {"call_listed_count": 0, "put_listed_count": 0, "filtered_call_count": 0, "filtered_put_count": 0, "call_nan_count": 0, "put_nan_count": 0, "call_valid_count": 0, "put_valid_count": 0}
COUNTS_ILLIQUID = {"call_listed_count": 3, "put_listed_count": 3, "filtered_call_count": 0, "filtered_put_count": 0, "call_nan_count": 0, "put_nan_count": 0, "call_valid_count": 0, "put_valid_count": 0}
COUNTS_NORMAL = {"call_listed_count": 3, "put_listed_count": 3, "filtered_call_count": 2, "filtered_put_count": 2, "call_nan_count": 0, "put_nan_count": 0, "call_valid_count": 2, "put_valid_count": 2}
EMPTY_SMILE = pd.DataFrame(columns=["strike", "volatility", "option type"])
GOOD_SMILE = pd.DataFrame({"strike": [90, 95, 98, 102, 105, 110], "volatility": [0.30, 0.29, 0.28, 0.27, 0.26, 0.25], "option type": ["put", "put", "put", "call", "call", "call"]})

CASES = [({"error": "no connection"}, "could not fetch market data: no connection"), ({"ticker": "TEST", "spot": 100.0, "rate": 0.04, "time": None, "expiry": None, "smile": EMPTY_SMILE, "counts": COUNTS_ZERO, "message": "no valid expiry within timeframe"}, "no valid expiry within timeframe"), ({"ticker": "TEST", "spot": 100.0, "rate": 0.04, "time": 0.137, "expiry": "2027-01-05", "smile": EMPTY_SMILE, "counts": COUNTS_ILLIQUID, "message": "no options passed the liquidity filter or solver failed"}, "no options passed the liquidity filter"), ({"ticker": "TEST", "spot": 100.0, "rate": 0.04, "time": 0.137, "expiry": "2027-01-05", "smile": GOOD_SMILE, "counts": COUNTS_NORMAL, "message": None}, "Smile: 6 points, strikes 90.00 to 110.00")]
IDS = ["error", "no_expiry", "nothing_liquid", "normal"]

@pytest.mark.parametrize("section, phrase", CASES, ids=IDS)
def test_market(capsys, section, phrase):
    market_printer(section, argparse.Namespace(show=False))
    out = capsys.readouterr().out
    assert "=== MARKET ===" in out 
    assert phrase in out
    assert 'None' not in out 

def test_convergence_plot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = get_parser().parse_args(["convergence"])
    convergence_plotter(run_convergence(args), args)
    path = tmp_path / "images" / "convergence.png"
    assert path.exists()
    assert path.stat().st_size > 0
    
def test_variance_plot(tmp_path, monkeypatch):   
    monkeypatch.chdir(tmp_path)
    args = get_parser().parse_args(["variance"])
    variance_plotter(run_variance(args), args)
    path = tmp_path / "images" / "variance.png"
    assert path.exists()
    assert path.stat().st_size > 0

@pytest.mark.parametrize("plotter, file_name", [(greeks_panels_plotter, "greeks_panels.png"), (greeks_value_plotter, "greeks_value.png")])
def test_greeks_plot(plotter, file_name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = get_parser().parse_args(["greeks"])
    plotter(run_greeks(args), args)
    path = tmp_path / "images" / file_name
    assert path.exists()
    assert path.stat().st_size > 0

@pytest.mark.parametrize("plotter, file_name", [(exotics_paths, "exotics_paths.png"), (exotics_barrier, "exotics_barrier.png")])
def test_exotics_plot(plotter, file_name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = get_parser().parse_args(["exotics"])
    plotter(run_exotics(args), args)
    path = tmp_path / "images" / file_name
    assert path.exists()
    assert path.stat().st_size > 0

CASES1 = [{"error": "no connection"}, {"ticker": "TEST", "spot": 100.0, "rate": 0.04, "time": None, "expiry": None, "smile": EMPTY_SMILE, "counts": COUNTS_ZERO, "message": "no valid expiry within timeframe"}, {"ticker": "TEST", "spot": 100.0, "rate": 0.04, "time": 0.137, "expiry": "2027-01-05", "smile": EMPTY_SMILE, "counts": COUNTS_ILLIQUID, "message": "no options passed the liquidity filter or solver failed"}, {"ticker": "TEST", "spot": 100.0, "rate": 0.04, "time": 0.137, "expiry": "2027-01-05", "smile": GOOD_SMILE, "counts": COUNTS_NORMAL, "message": None}]

@pytest.mark.parametrize("section", CASES1, ids=IDS)
def test_market_plot(tmp_path, section, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = get_parser().parse_args(["market"])
    market_plotter(section, args)
    path = tmp_path / "images" / "market_smile.png"
    if "error" in section: 
        assert path.exists() == False 
    elif section['message'] is not None: 
        assert path.exists() == False
    elif len(section['smile']) == 0:
        assert path.exists() == False
    elif len(section['smile']) < 5:
        assert path.exists() == False
    else:
        assert path.exists()
        assert path.stat().st_size > 0