import pytest
import numpy as np
import statistics
import math
from math import e
from monte_carlo import asian_payoff_calculator, knock_in_payoff, knock_out_payoff, european_payoff_calculator, stock_price_vectorized

def test_handbuilt_path():
    K, T, r = 100, 1.0, 0 
    asian_path = np.array([[100., 110., 120.], [80., 90., 100.]])
    asian_price_call = asian_payoff_calculator(asian_path, K, r, T, "call")
    asian_price_put = asian_payoff_calculator(asian_path, K, r, T, "put")
    assert asian_price_call == 5
    assert asian_price_put == 5
    asian_price_r1 = asian_payoff_calculator(asian_path, K, 0.05, T, "call")
    assert asian_price_r1 == pytest.approx(4.756147, abs=1e-6)
    barrier_path = np.array([[100., 95., 110.], [100., 85., 120]])
    assert knock_out_payoff(barrier_path, 90, K, r, T, "call") == 5
    assert knock_in_payoff(barrier_path, 90, K, r, T, "call") == 10
    boundary_path = np.array([[100., 90., 110.]])
    assert knock_out_payoff(boundary_path, 90, K, r, T, "call") == 0 
    assert knock_in_payoff(boundary_path, 90, K, r, T, "call") == 10
    assert knock_out_payoff(boundary_path, 89.99, K, r, T, "call") == 10 
    assert knock_in_payoff(boundary_path, 89.99, K, r, T, "call") == 0

@pytest.mark.parametrize("barrier, option_type", [(80, "call"), (90, "call"), (95, "call"), (80, "put"), (90, "put"), (95, "put")])
def test_parity(barrier, option_type):
    S, K, T, r, sigma, n_steps, n_trials = 100, 100, 1.0, 0.05, 0.2, 252, 5000
    rng = np.random.default_rng(42)
    path = stock_price_vectorized(r, sigma, T, n_trials, n_steps, S, rng=rng)
    vanilla_price, _ = european_payoff_calculator(path, K, r, T, option_type)
    assert knock_out_payoff(path, barrier, K, r, T, option_type) + knock_in_payoff(path, barrier, K, r, T, option_type) == pytest.approx(vanilla_price, abs=1e-9)

def test_extremes():
    S, K, T, r, sigma, n_steps, n_trials = 100, 100, 1.0, 0.05, 0.2, 252, 10000
    rng = np.random.default_rng(42)
    path = stock_price_vectorized(r, sigma, T, n_trials, n_steps, S, rng=rng)
    vanilla_price, _ = european_payoff_calculator(path, K, r, T, "call")
    assert knock_out_payoff(path, 1, K, r, T, "call") == pytest.approx(vanilla_price, abs=1e-6)
    assert knock_in_payoff(path, 1, K, r, T, "call") == 0 
    assert knock_in_payoff(path, 100000, K, r, T, "call") == pytest.approx(vanilla_price, abs=1e-6)
    assert knock_out_payoff(path, 100000, K, r, T, "call") == 0 

def test_monotonic():
    S, K, T, r, sigma, n_steps, n_trials = 100, 100, 1.0, 0.05, 0.2, 252, 10000
    rng = np.random.default_rng(42)
    path = stock_price_vectorized(r, sigma, T, n_trials, n_steps, S, rng=rng)
    vanilla_price, _ = european_payoff_calculator(path, K, r, T, "call")
    grid = np.linspace(S / 2, S - 1, 10)
    knock_out_prices = []
    for barrier in grid:
        price = knock_out_payoff(path, barrier, K, r, T, "call")
        knock_out_prices.append(price)
    for i in range(9):
        assert knock_out_prices[i + 1] <= knock_out_prices[i] + 1e-12
    for i in range(10):
        assert knock_out_prices[i] <= vanilla_price + 1e-12

def test_asian():
    S, K, T, r, sigma, n_steps, n_trials = 100, 100, 1.0, 0.05, 0.2, 252, 20000
    rng = np.random.default_rng(42)
    path = stock_price_vectorized(r, sigma, T, n_trials, n_steps, S, rng=rng)
    vanilla_price, _ = european_payoff_calculator(path, K, r, T, "call")
    assert asian_payoff_calculator(path, K, r, T, "call") < vanilla_price

def test_touch_fraction():
    S, K, T, r, sigma, n_trials, barrier = 100, 100, 1.0, 0.05, 0.2, 20000, 90
    rng = np.random.default_rng(42)
    path_12 = stock_price_vectorized(r, sigma, T, n_trials, 12, S, rng=rng)
    path_252 = stock_price_vectorized(r, sigma, T, n_trials, 252, S, rng=rng)
    fraction_12 = np.mean(path_12.min(axis=1) <= barrier)
    fraction_252 = np.mean(path_252.min(axis=1) <= barrier)
    assert fraction_252 > fraction_12

@pytest.mark.parametrize("n_steps", [12, 252])
def test_stretch(n_steps):
    S, K, T, r, sigma, n_trials, barrier = 100, 100, 1.0, 0.05, 0.2, 20000, 90
    rng = np.random.default_rng(42)
    path_12 = stock_price_vectorized(r, sigma, T, n_trials, n_steps, S, rng=rng)
    fraction = np.mean(path_12.min(axis=1) <= barrier)
    miu = r - ((sigma ** 2) / 2)
    sd = sigma * math.sqrt(T)
    dt = T / n_steps
    B_eff = barrier * (e ** (-0.5826 * sigma * math.sqrt(dt)))
    b = math.log(B_eff / S)
    P = statistics.NormalDist().cdf((b - (miu * T)) / sd) + (np.exp((2 * miu * b) / (sigma ** 2)) * statistics.NormalDist().cdf((b + (miu * T)) / sd))
    assert abs(fraction - P) < 3 * math.sqrt(P * (1 - P) / n_trials)







  