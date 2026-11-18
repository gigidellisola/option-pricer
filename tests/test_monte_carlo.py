import pytest
import numpy as np
import math 
from math import e 
from monte_carlo import stock_price_vectorized, stock_price_antithetic, european_payoff_calculator, trial_list
from black_scholes import black_scholes

@pytest.mark.parametrize("n_trials, n_steps", [(100, 100), (1000, 252), (20000, 252), (55, 10), (101, 252)])
def test_stock_price_vectorized(n_trials, n_steps):
    rng = np.random.default_rng(42)
    paths = stock_price_vectorized(0.05, 0.2, 1.0, n_trials, n_steps, 100, rng=rng)
    antithetic_paths = stock_price_antithetic(0.05, 0.2, 1.0, n_trials, n_steps, 100, rng=rng)
    assert paths.shape == (n_trials, n_steps)
    if n_trials % 2 == 0:  
        assert antithetic_paths.shape == (n_trials, n_steps)
    else:
        assert antithetic_paths.shape == (2 * (n_trials // 2), n_steps)
    
def test_prices_pos():
    rng = np.random.default_rng(42)
    paths = stock_price_vectorized(0.05, 0.2, 1.0, 1000, 252, 100, rng=rng)
    antithetic_paths = stock_price_antithetic(0.05, 0.2, 1.0, 1000, 252, 100, rng=rng)
    for i in range(1000):
        for j in range(252):
            assert paths[i][j] > 0 
            assert antithetic_paths[i][j] > 0 

@pytest.mark.parametrize("n_trials, n_days, start_price", [(10, 252, 100), (100, 100, 110), (1000, 252, 100), (1000, 80, 90)])
def test_sig_zero(n_trials, n_days, start_price):
    r, T, K = 0.05, 1.0, 100
    rng = np.random.default_rng(42)
    future_price = start_price * (e ** (r * T))
    paths = stock_price_vectorized(r, 0, T, n_trials, n_days, start_price, rng=rng)
    for i in range(n_trials):
        final_price = paths[i][-1]
        assert final_price == pytest.approx(future_price, abs=1e-9)
    call_price, _ = european_payoff_calculator(paths, K, r, T, "call")
    future_call = max(start_price - (K * (e ** (-r * T))), 0)
    assert call_price == pytest.approx(future_call, abs=1e-9)


@pytest.mark.parametrize("n_trials, n_steps, start_price", [(1000, 252, 100), (10000, 252, 100)])
def test_mirror(n_trials, n_steps, start_price):
    rng = np.random.default_rng(42)
    r, sigma, T = 0.05, 0.2, 1.0
    half = n_trials // 2
    antithetic_paths = stock_price_antithetic(r, sigma, T, n_trials, n_steps, start_price, rng=rng)
    path = antithetic_paths[:half]
    path_mirror = antithetic_paths[half:]
    relationship = 2 * (r - ((sigma ** 2) / 2)) * T
    final_path = []
    final_path_mirror = []
    for i in range(half):
        final_path.append(path[i][-1])
        final_path_mirror.append(path_mirror[i][-1])
    path_log = np.log(np.array(final_path) / start_price)
    path_mirror_log = np.log(np.array(final_path_mirror) / start_price)
    for i in range(len(path_log)):
        assert path_log[i] + path_mirror_log[i] == pytest.approx(relationship, abs=1e-9)

@pytest.mark.parametrize("start_price", [100, 90, 80, 110, 120])
def test_terminal_movement(start_price):
    rng = np.random.default_rng(42)
    r, sigma, T, n_trials, n_steps, = 0.05, 0.2, 1.0, 100000, 1
    path = stock_price_vectorized(r, sigma, T, n_trials, n_steps, start_price, rng=rng)
    final = path[:-1]
    X = np.log(np.array(final) / start_price)
    mean_S = final.mean()
    mean_X = X.mean()
    std_X = X.std(ddof=1)
    mean_S_se = final.std(ddof=1) / math.sqrt(n_trials)
    mean_X_se = sigma * math.sqrt(T) / math.sqrt(n_trials)
    std_X_se = sigma * math.sqrt(T) / math.sqrt(2 * n_trials)
    assert mean_X == pytest.approx(r - ((sigma ** 2) / 2), abs=3 * mean_X_se)
    assert mean_S == pytest.approx(start_price * (e ** (r * T)), abs=3 * mean_S_se)
    assert std_X == pytest.approx(sigma * math.sqrt(T), abs=3 * std_X_se)

@pytest.mark.parametrize("r, T, option_type, expected_price, expected_se", [(0, 1.0, "call", 7.5, math.sqrt(275/3) / 2), (0, 1.0, "put", 2.5, 2.5), (0.05, 1, "call", 7.134221, 4.553664), (0.05, 1, "put", 2.378074, 2.378074), (0.05, 2, "call", 6.786281, 4.331579)])
def test_hand_built_payoff(r, T, option_type, expected_price, expected_se):
    price_path = np.array([[90], [100], [110], [120]])
    assert (expected_price, expected_se) == pytest.approx(european_payoff_calculator(price_path, 100, r, T, option_type), abs=1e-6)
    with pytest.raises(ValueError):
        european_payoff_calculator(price_path, 100, r, T, "straddle")

@pytest.mark.parametrize("K, sigma, option_type", [(90, 0.2, "call"), (90, 0.2, "put"), (110, 0.3, "call"), (110, 0.3, "put"), (100, 0.5, "call"), (100, 0.5, "put"), (120, 0.2, "call"), (120, 0.2, "put")])
def test_mc_vs_bs(K, sigma, option_type):
    r, T, n_trials, n_steps, start_price = 0.05, 1.0, 100000, 1, 100
    rng = np.random.default_rng(42)
    path = stock_price_vectorized(r, sigma, T, n_trials, n_steps, start_price, rng=rng)
    price_mc, se = european_payoff_calculator(path, K, r, T, option_type)
    price_bs, _, _ = black_scholes(start_price, K, T, r, sigma, option_type)
    assert abs(price_mc - price_bs) < 4 * se

def test_parity():
    r, sigma, T, n_trials, n_steps, start_price, K = 0.05, 0.2, 1.0, 20000, 252, 100, 100
    rng = np.random.default_rng(42)
    path = stock_price_vectorized(r, sigma, T, n_trials, n_steps, start_price, rng=rng)
    final_price = path[:, -1]
    call_price, _ = european_payoff_calculator(path, K, r, T, "call")
    put_price, _ = european_payoff_calculator(path, K, r, T, "put")
    parity = ((e ** (-r * T)) * final_price.mean()) - (K * (e ** (-r * T)))
    assert call_price - put_price == pytest.approx(parity, abs=1e-9)
    se = (((e ** (-r * T)) * final_price).std(ddof=1)) / math.sqrt(n_trials)
    assert abs(start_price - ((e ** (-r * T)) * final_price.mean())) < (3 * se)

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_steps(option_type):
    r, sigma, T, n_trials, start_price, K = 0.05, 0.2, 1.0, 20000, 100, 100
    rng = np.random.default_rng(42)
    path_1 = stock_price_vectorized(r, sigma, T, n_trials, 1, start_price, rng=rng)
    path_252 = stock_price_vectorized(r, sigma, T, n_trials, 252, start_price, rng=rng)
    price_1, se_1 = european_payoff_calculator(path_1, K, r, T, option_type)
    price_252, se_252 = european_payoff_calculator(path_252, K, r, T, option_type)
    assert abs(price_1 - price_252) < 3 * math.sqrt((se_1 ** 2) + (se_252 ** 2))

def test_scaling():
    rng = np.random.default_rng(42)
    trial_counts, S, K, T, r, sigma, n_steps = [5000, 10000, 100000], 100, 100, 1.0, 0.05, 0.2, 1
    _, standard_errors = trial_list(trial_counts, S, K, T, r, sigma, "call", n_steps, rng=rng)
    for i in range(len(trial_counts)):
        assert abs((standard_errors[i] * math.sqrt(trial_counts[i])) - 14.72) < (0.05 * 14.72) 

def test_trial_list():
    trial_counts = [100, 10000]
    S, K, T, r, sigma, option_type = 100, 100, 1.0, 0.05, 0.2, "call"
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(42)
    prices_x, ses_x = trial_list(trial_counts, S, K, T, r, sigma, option_type, 1, rng=rng_x)
    prices_y, ses_y = trial_list(trial_counts, S, K, T, r, sigma, option_type, 1, rng=rng_y)
    assert prices_x == prices_y
    assert ses_x == ses_y 
    rng = np.random.default_rng(42)
    assert trial_list(trial_counts, S, K, T, r, sigma, option_type, rng=rng) != trial_list(trial_counts, S, K, T, r, sigma, option_type, rng=rng)
    _, ses = trial_list(trial_counts, S, K, T, r, sigma, option_type, 1, rng=rng)
    assert (ses[1] / ses[0]) < 0.3

@pytest.mark.parametrize("function", [stock_price_vectorized, stock_price_antithetic])
def test_reproducibility(function):
    r, sigma, T, n_trials, n_steps, start_price = 0.05, 0.2, 1.0, 10000, 252, 100
    a1 = function(r, sigma, T, n_trials, n_steps, start_price, rng=np.random.default_rng(42))
    a2 = function(r, sigma, T, n_trials, n_steps, start_price, rng=np.random.default_rng(42))
    assert np.array_equal(a1, a2) 
    b1 = function(r, sigma, T, n_trials, n_steps, start_price, rng=np.random.default_rng(42))
    b2 = function(r, sigma, T, n_trials, n_steps, start_price, rng=np.random.default_rng(43))
    assert np.array_equal(b1, b2) == False 
    rng=np.random.default_rng(42)
    c1 = function(r, sigma, T, n_trials, n_steps, start_price, rng=rng)
    c2 = function(r, sigma, T, n_trials, n_steps, start_price, rng=rng)
    assert np.array_equal(c1, c2) == False 
    d1 = function(r, sigma, T, n_trials, n_steps, start_price)
    d2 = function(r, sigma, T, n_trials, n_steps, start_price)   
    assert np.array_equal(d1, d2) == False








    







    
                   
