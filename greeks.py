"""Option Greeks, computed analytically and by finite differences. 

Units: vega is per 1.00 change in volatility (divide by 100 for 1% move), 
theta is per year (divide by 365 for per day), and gamma is the change in 
delta per $1 move in the spot price. No dividends.
"""

import math 
from scipy.stats import norm 
from black_scholes import black_scholes 

def finite_diff_delta(S, K, T, r, sigma, option_type, h_delta):
    """Estimate delta by a central difference in the spot price. 
    
    Computes (V(S + h) - V(S - h)) / (2h), where V is the Black-Scholes price. 
    The error shrinks in proportion to h squared. Requires S > h_delta.
    """
    bs_s_plus_h, _, _ = black_scholes(S + h_delta, K, T, r, sigma, option_type)
    bs_s_minus_h, _, _  = black_scholes(S - h_delta, K, T, r, sigma, option_type)
    delta = (bs_s_plus_h - bs_s_minus_h) / (2 * h_delta)
    return delta 

def finite_diff_gamma(S, K, T, r, sigma, option_type, h_gamma):
    """Estimate gamma by a second difference in the spot price. 
    
    Computes (V(S + h) - 2 V(S) + V(S - h)) / h^2. Gamma measures how quickly 
    delta changes as the spot price moves. Requires S > h_gamma. 
    """
    bs_s, _, _ = black_scholes(S, K, T, r, sigma, option_type)
    bs_s_plus_h, _, _ = black_scholes(S + h_gamma, K, T, r, sigma, option_type)
    bs_s_minus_h, _, _  = black_scholes(S - h_gamma, K, T, r, sigma, option_type)
    gamma = (bs_s_plus_h - (2 * bs_s) + bs_s_minus_h) / (h_gamma ** 2)
    return gamma 

def finite_diff_vega(S, K, T, r, sigma, option_type, h_vega):
    """Estimate vega by a central difference in volatility. 
    
    The result is the price change per 1.00 change in sigma. Requires 
    sigma > h_vega. 
    """
    bs_sigma_plus, _, _ = black_scholes(S, K, T, r, sigma + h_vega, option_type)
    bs_sigma_minus, _, _ = black_scholes(S, K, T, r, sigma - h_vega, option_type)
    vega = (bs_sigma_plus - bs_sigma_minus) / (2 * h_vega)
    return vega 

def finite_diff_theta(S, K, T, r, sigma, option_type, h_theta):
    """Estimate theta by a one-sided difference in time to expiry. 
    
    Computes (V(T - h) - V(T)) / h, the change in price as time passes, 
    per year. This difference is only first-order accurate, so theta is 
    the least precise of the four finite-difference Greeks. Requires
    T > h_theta.
    """
    bs_s, _, _ = black_scholes(S, K, T, r, sigma, option_type)
    bs_t, _, _ = black_scholes(S, K, T - h_theta, r, sigma, option_type)
    theta = (bs_t - bs_s) / h_theta
    return theta 

def finite_diff_greeks(S, K, T, r, sigma, option_type, h_delta=0.01, h_gamma=0.01, h_vega=0.0001, h_theta=0.0001):
    """Estimate all four Greeks by finite differences. 
    
    Returns: 
        Tuple (delta, gamma, vega, theta), in that order, using given bump
        sizes. 
    """
    bump_delta = finite_diff_delta(S, K, T, r, sigma, option_type, h_delta)
    bump_gamma = finite_diff_gamma(S, K, T, r, sigma, option_type, h_gamma)
    bump_vega = finite_diff_vega(S, K, T, r, sigma, option_type, h_vega)
    bump_theta = finite_diff_theta(S, K, T, r, sigma, option_type, h_theta)
    return bump_delta, bump_gamma, bump_vega, bump_theta

def analytical_greeks(S, K, T, r, sigma, option_type):
    """Compute delta, gamma, vega, and theta from closed-form formulas.
    
    Gamma and vega are identical for calls and puts. Theta is usually 
    negative for a long option (time decay).
    
    Returns:
        Tuple (delta, gamma, vega, theta) of floats.
        
    Raises:
        ValueError: If the inputs are invalid (raised by black_scholes).
    """
    _, d1, d2 = black_scholes(S, K, T, r, sigma, option_type)
    if option_type.lower() == "call":
        analytical_delta = norm.cdf(d1)
        analytical_theta = (-S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - (r * K * math.exp(-r * T) * norm.cdf(d2))
    elif option_type.lower() == "put":
        analytical_delta = norm.cdf(d1) - 1 
        analytical_theta = (-S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + (r * K * math.exp(-r * T) * norm.cdf(-d2))
    else: 
        raise ValueError("option type must be 'call' or 'put'")
    analytical_gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    analytical_vega = S * norm.pdf(d1) * math.sqrt(T)
    return float(analytical_delta), float(analytical_gamma), float(analytical_vega), float(analytical_theta)


