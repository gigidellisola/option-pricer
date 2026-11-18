import pytest
from monte_carlo import stock_price_vectorized, stock_price_antithetic, european_payoff_calculator, euro_antithetic_payoff, european_control_variate
import numpy as np 
from black_scholes import black_scholes
import math 

def test_control_variate():
    rng = np.random.default_rng(42)
    r, sigma, T, n_trials, n_steps, start_price, K = 0.05, 0.2, 1.0, 20000, 1, 100, 100
    price_path = stock_price_vectorized(r, sigma, T, n_trials, n_steps, start_price, rng=rng)
    hedge_ratio, corr, mean, price, se_control = european_control_variate(price_path, K, r, T, start_price, "call")
    price_bs, _, _ = black_scholes(start_price, K, T, r, sigma, "call")
    _, se_plain = european_payoff_calculator(price_path, K, r, T, "call")
    variance_factor = (se_plain / se_control) ** 2
    assert variance_factor == pytest.approx(1 / (1 - (corr **2)), rel=1e-6)
    X = price_path[:,-1] * np.exp(-r * T)
    se_X = X.std(ddof=1) / math.sqrt(len(X))
    assert abs(mean - start_price) < (3 * se_X)
    assert abs(price - price_bs) < (3 * se_control)
    assert se_control < se_plain
    assert corr > 0.8 
    assert 0 < hedge_ratio < 2

def test_antithetic():
    path = np.array([[80.], [120.], [90.], [110.]])
    K, r, T, sigma, n_trials, n_steps, start_price = 100, 0, 1.0, 0.2, 20000, 1, 100
    antithetic_price, antithetic_se = euro_antithetic_payoff(path, K, r, T, "call")
    assert antithetic_price == 7.5
    assert antithetic_se == 7.5 
    rng = np.random.default_rng(42)
    plain_path = stock_price_vectorized(r, sigma, T, n_trials, n_steps, start_price, rng=rng)
    antithetic_path = stock_price_antithetic(r, sigma, T, n_trials, n_steps, start_price, rng=rng)
    price_bs, _, _ = black_scholes(start_price, K, T, r, sigma, "call")
    _, plain_se = european_payoff_calculator(plain_path, K, r, T, "call")
    anti_price, anti_se = euro_antithetic_payoff(antithetic_path, K, r, T, "call")
    assert abs(anti_price - price_bs) < (3 * anti_se)
    assert anti_se < plain_se

def test_compare():
    rng = np.random.default_rng(42)
    K, r, T, sigma, n_trials, n_steps, start_price = 100, 0, 1.0, 0.2, 20000, 1, 100
    plain_path = stock_price_vectorized(r, sigma, T, n_trials, n_steps, start_price, rng=rng)
    antithetic_path = stock_price_antithetic(r, sigma, T, n_trials, n_steps, start_price, rng=rng)
    _, plain_se = european_payoff_calculator(plain_path, K, r, T, "call")
    _, _, _, _, control_se = european_control_variate(plain_path, K, r, T, start_price, "call")
    _, anti_se = euro_antithetic_payoff(antithetic_path, K, r, T, "call")
    control_factor = (plain_se / control_se) ** 2
    anti_factor = (plain_se / anti_se) ** 2
    assert control_factor > anti_factor > 1 



