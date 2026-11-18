import pytest
import argparse
import numpy as np 
from main import get_parser, positive_float, positive_int, non_negative_percent, option, run_prices, run_convergence, run_variance, run_greeks, run_exotics, run_market, pick_steps, generator

def test_positive_float():
    assert positive_float("1.5") == float(1.5)
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float("abc")
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float("0")
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float("-1")
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float("inf")
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float("nan")

def test_positive_percent():
    assert non_negative_percent("0") == float(0)
    assert non_negative_percent("1") == float(1)
    with pytest.raises(argparse.ArgumentTypeError):
        non_negative_percent("1.5")
    with pytest.raises(argparse.ArgumentTypeError):
        non_negative_percent("-0.1")

def test_positive_int():
    assert positive_int("5") == int(5)
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int("5.5")
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int("-3")

def test_option():
    assert option("Put") == "put"
    assert option("CALL") == "call"
    with pytest.raises(argparse.ArgumentTypeError):
        option("option") 

def test_prices():
    args = get_parser().parse_args(["prices"])
    assert args.command == "prices"
    assert args.func == run_prices
    assert args.spot == 100 
    assert args.strike == 100
    assert args.maturity == 1.0
    assert args.rate == 0.05
    assert args.sigma == 0.2
    assert args.trials == 20000
    assert args.option_type == "call"
    assert args.seed == 42
    assert args.steps == None

def test_convergence():
    args = get_parser().parse_args(["convergence"])
    assert args.command == "convergence"
    assert args.func == run_convergence
    assert args.spot == 100 
    assert args.strike == 100
    assert args.maturity == 1.0
    assert args.rate == 0.05
    assert args.sigma == 0.2
    assert args.option_type == "call"
    assert args.seed == 42
    assert args.steps == None
    if args.show in args:
        assert args.show == True
    else:
        assert args.show == False

def test_variance():
    args = get_parser().parse_args(["variance"])
    assert args.command == "variance"
    assert args.func == run_variance
    assert args.spot == 100 
    assert args.strike == 100
    assert args.maturity == 1.0
    assert args.rate == 0.05
    assert args.sigma == 0.2
    assert args.option_type == "call"
    assert args.seed == 42
    assert args.steps == None
    if args.show in args:
        assert args.show == True
    else:
        assert args.show == False


def test_greeks():
    args = get_parser().parse_args(["greeks"])
    assert args.command == "greeks"
    assert args.func == run_greeks
    assert args.spot == 100 
    assert args.strike == 100
    assert args.maturity == 1.0
    assert args.rate == 0.05
    assert args.sigma == 0.2
    assert args.spot_min == 60
    assert args.spot_max == 140
    if args.show in args:
        assert args.show == True
    else:
        assert args.show == False 

def test_exotics():
    args = get_parser().parse_args(["exotics"])
    assert args.command == "exotics"
    assert args.func == run_exotics
    assert args.spot == 100 
    assert args.strike == 100
    assert args.maturity == 1.0
    assert args.rate == 0.05
    assert args.sigma == 0.2
    assert args.option_type == "call"
    assert args.seed == 42
    assert args.steps == None
    assert args.trials == 20000
    assert args.barrier == 90
    if args.show in args:
        assert args.show == True
    else:
        assert args.show == False

def test_market():
    args = get_parser().parse_args(["market"])
    assert args.command == "market"
    assert args.func == run_market
    assert args.target_days == 45
    assert args.min_days == 14
    assert args.ticker == "AAPL"
    assert args.min_volume == 2
    assert args.min_open_interest == 3
    if args.show in args:
        assert args.show == True
    else:
        assert args.show == False

@pytest.mark.parametrize("command, flag, invalid_input", [("prices", "--barrier", "90"), ("convergence", "--trials", "100"), ("variance", "--ticker", "AAPL"), ("greeks", "--trials", "100"), ("exotics", "--spot-min", "50"), ("market", "--spot", "100")])
def test_parents(command, flag, invalid_input):
    with pytest.raises(SystemExit) as e:
        get_parser().parse_args([command, flag, invalid_input])
    assert e.value.code == 2

@pytest.mark.parametrize("command, value, bad_input", [("prices", "--strike", "-1"), ("prices", "--sigma", "0"), ("prices", "--rate", "1.5"), ("prices", "--option-type", "straddle"), ("prices", "--trials", "0"), ("prices", "--trials", "abc"), ("prices", "--steps", "0")])
def test_bad_input(command, value, bad_input):
    with pytest.raises(SystemExit) as e:
        get_parser().parse_args([command, value, bad_input])
    assert e.value.code == 2 

def test_no_command():
    with pytest.raises(SystemExit) as e: 
        get_parser().parse_args([])
    assert e.value.code == 2

def test_steps():
    assert pick_steps(None, 252) == 252
    assert pick_steps(5, 252) == 5 

def test_generator():
    n_trials, n_steps = 20000, 252
    x = generator(42).standard_normal((n_trials, n_steps))
    y = generator(42).standard_normal((n_trials, n_steps))
    assert np.array_equal(x, y) == True 









        
    