import pytest
import numpy as np
import math
import main
import argparse
from main import get_parser, run_prices, run_convergence, run_variance, run_greeks, run_exotics

def test_prices():
    args1 = get_parser().parse_args(["prices", "--seed", "42"]) 
    args2 = get_parser().parse_args(["prices", "--seed", "43"])
    result1 = run_prices(args1)
    result2 = run_prices(args2)
    assert result1["black_scholes_price"] == pytest.approx(10.4506, abs=1e-4)
    assert result1["se_gap"] < 3
    assert result1["black_scholes_price"] == result2["black_scholes_price"]
    assert result1["monte_carlo_price"] != result2["monte_carlo_price"]

def test_convergence():
    args = get_parser().parse_args(["convergence"])
    result = run_convergence(args)
    assert len(result) == 6 
    mc = result["monte_carlo_prices"]
    se = result["standard_errors"]
    bs = result["black_scholes_price"]
    assert abs(mc[-1] - bs) < 3 * se[-1]
    scaled_errors = result["scaled_errors"]
    for i in [-1, -2]:
        assert abs(scaled_errors[i] - 14.71) < 0.05 * 14.71

def test_variance():
    args = get_parser().parse_args(["variance"])
    result = run_variance(args)
    assert len(result) == 11
    with pytest.raises(ValueError):
        get_parser().parse_args(["variance"])
        args.trials = 101 
        run_variance(args)
    variance_factor = (result["plain_se"] / result["control_se"]) ** 2
    assert variance_factor == pytest.approx(1 / (1 - (result["correlation"] ** 2)), rel=1e-6)
    args_1 = get_parser().parse_args(["prices"])
    prices = run_prices(args_1)
    assert result["reference_price"] == prices["black_scholes_price"]

def test_greeks():
    args = get_parser().parse_args(["greeks"])
    args1 = get_parser().parse_args(["greeks", "--spot-min", "0.005"])
    args2 = get_parser().parse_args(["greeks", "--spot-min", "0.5", "--spot-max", "0.25"])
    args3 = get_parser().parse_args(["greeks", "--maturity", "0.00001"])
    args4 = get_parser().parse_args(["greeks", "--sigma", "0.00001"])
    args5 = get_parser().parse_args(["greeks"])
    args5.strike = -1
    args6 = get_parser().parse_args(["greeks", "--spot", "0.005"])
    with pytest.raises(ValueError):
        run_greeks(args1)
    with pytest.raises(ValueError):
        run_greeks(args2)
    with pytest.raises(ValueError):
        run_greeks(args3)
    with pytest.raises(ValueError):
        run_greeks(args4)
    with pytest.raises(ValueError):
        run_greeks(args5)
    with pytest.raises(ValueError):
        run_greeks(args6)
    result = run_greeks(args)
    grid = result["spot grid"]
    assert len(result) == 6
    assert len(grid) == 81
    assert grid[0] == 60 
    assert grid[-1] == 140 
    jumps = np.diff([grid])
    assert np.allclose(jumps, 1.0, rtol=0, atol=1e-9) == True
    assert grid[40] == pytest.approx(100.0, abs=1e-9)
    call_table, put_table = result["call greeks"], result["put greeks"]
    call_spots, put_spots = call_table["spot price"].tolist(), put_table["spot price"].tolist()
    call_gamma, put_gamma = call_table["analytical gamma"].tolist(), put_table["analytical gamma"].tolist()
    call_vega, put_vega = call_table["analytical vega"].tolist(), put_table["analytical vega"].tolist()
    call_theta, put_theta = call_table["analytical theta"].tolist(), put_table["analytical theta"].tolist()
    call_theta_fd, put_theta_fd = call_table["finite difference theta"].tolist(), put_table["finite difference theta"].tolist()
    assert np.array_equal(call_spots, grid, equal_nan=False) == True
    assert np.array_equal(put_spots, grid, equal_nan=False) == True 
    assert np.array_equal(call_gamma, put_gamma, equal_nan=False) == True 
    assert np.array_equal(call_vega, put_vega, equal_nan=False) == True 
    theta_parity = args.rate * args.strike * math.exp(-args.rate * args.maturity)
    for i in range(len(grid)):
        assert put_theta[i] - call_theta[i] == pytest.approx(theta_parity, abs=1e-6)
        assert put_theta_fd[i] - call_theta_fd[i] == pytest.approx(theta_parity, abs=1e-4)
    max_differences = result["max differences"]
    call_diffs, put_diffs = max_differences["max_diff_call"], max_differences["max_diff_put"]
    greeks = ["delta", "gamma", "vega", "theta"]
    tols = [1e-5, 1e-5, 1e-5, 1e-3]
    for greek, tol in zip(greeks, tols): 
        assert call_diffs[greek] < tol 
        assert put_diffs[greek] < tol 
    greeks_at_spot = result["greeks for spot input"]
    call_greek_spot = greeks_at_spot["spot_greeks_call"]
    put_greek_spot = greeks_at_spot["spot_greeks_put"]
    assert len(call_greek_spot) == 8 
    assert len(put_greek_spot) == 8 
    assert result["strike"] == args.strike 

def test_exotics():
    args = get_parser().parse_args(["exotics", "--trials", "5000"])
    args1 = get_parser().parse_args(["exotics", "--trials", "5000"])
    args2 = get_parser().parse_args(["exotics", "--trials", "5000"])
    args1.barrier = -100 
    with pytest.raises(ValueError):
        run_exotics(args1)
    args2.spot, args2.barrier = 80, 100
    with pytest.raises(ValueError):
        run_exotics(args2)
    result = run_exotics(args)
    assert result["parity_check_barrier"] == True
    assert abs(result["gap"]) < 1e-9 
    assert result["asian_price"] < result["vanilla_price"]
    barrier_sensitivity = result["barrier_sensitivity"]
    for i in range(1, len(barrier_sensitivity)):
        assert barrier_sensitivity[i] <= barrier_sensitivity[i - 1]
    sample_path = result["path_sample"]
    assert len(sample_path) == 20 
    for i in range(len(sample_path)):
        assert len(sample_path[i]) == 252
    assert abs(result["vanilla_price"] - result["reference_price"]) / result["vanilla_se"] == result["gap_in_se"]

def test_defaults():
    a = run_prices(get_parser().parse_args(["prices"]))
    b = run_prices(get_parser().parse_args(["prices", "--steps", "1"]))
    assert a["monte_carlo_price"] == b["monte_carlo_price"]
    c = run_exotics(get_parser().parse_args(["exotics"]))
    d = run_exotics(get_parser().parse_args(["exotics", "--steps", "252"]))
    c_path, d_path = c["path_sample"], d["path_sample"]
    assert len(c) == len(d)
    for i in range(len(c_path)):
        assert len(c_path[i]) == len(d_path[i])

def test_market(monkeypatch):
    calls = []
    answer = "ABC"
    def recorder(*received):
        calls.append(received)
        return answer
    monkeypatch.setattr(main, "fetch_smile", recorder)
    result = main.run_market(argparse.Namespace(ticker="AAPL", target_days=45, min_days=14, min_volume=2, min_open_interest=3))
    assert calls == [("AAPL", 45, 14, 2, 3)]
    assert result is answer

def test_all(monkeypatch):
    monkeypatch.setattr(main, "run_prices", lambda args: {"A": "prices"})
    monkeypatch.setattr(main, "run_convergence", lambda args: {"B": "convergence"})
    monkeypatch.setattr(main, "run_variance", lambda args: {"C": "variance"})
    monkeypatch.setattr(main, "run_greeks", lambda args: {"D": "greeks"})
    monkeypatch.setattr(main, "run_exotics", lambda args: {"E": "exotics"})
    def broken_market(args):
        raise RuntimeError("no connection")
    monkeypatch.setattr(main, "run_market", broken_market)
    result = main.run_all(get_parser().parse_args(["all"]))
    assert set(result) == {"prices", "convergence", "variance", "greeks", "exotics", "market"}
    assert result["market"] == {"error": "no connection"}
    assert result["prices"] == {"A": "prices"}
    







    





    
