"""Implied volatility solver: Newton-Raphson with a bisection fallback."""

import math 
from scipy.stats import norm 
from black_scholes import black_scholes 

def implied_vol(S, K, T, r, market_price, option_type, initial_guess=0.3, tolerance=0.0001, max_iterations=100): 
    """Find the volatility at which Black-Scholes reproduces a market price.
    
    First checks that the price lies inside the no-arbitrage bounds (intrinsic
    value of the discounted strike at the bottom, S for a call or the discounted
    strike for a put at the top). Then runs Newton-Raphson from initial_guess.
    If Newton's method leaves the range [0.01, 5], meets a near-zero vega or 
    fails to converge, it falls back to implied_vol_bisection over the same range.
    Assumes European exercise and no dividends. 
    
    Args:
        market_price: Observed option price. 
        initial_guess: Starting volatility for Newton's method.
        tolerance: Accepted price error, in price units (not in volatility).
        max_iterations: Iteration limit for each method.
        
    Returns:
        Implied volatility as a decimal (0.20 = 20%), or float("nan") if no
        volatility in [0.01, 5] reproduces the price. 
        
    Raises:
        ValueError: If option_type is not "call" or "put".
    """
    discounted_strike = K * math.exp(-r * T)
    if option_type.lower() == "call":
        lower = max(S - discounted_strike, 0)
        upper = S
    elif option_type.lower() == "put":
        lower = max(discounted_strike - S, 0)
        upper = discounted_strike
    else: 
        raise ValueError("option type must be 'call' or 'put'")
    if market_price <= lower + tolerance or market_price >= upper: 
        return float("nan")
    sigma = initial_guess
    bs_price, d1, _ = black_scholes(S, K, T, r, sigma, option_type)
    analytical_vega = S * norm.pdf(d1) * math.sqrt(T)
    for _ in range(max_iterations): 
        # search range: 1% to 500% annualised volatility.
        if 0.01 <= sigma <= 5:
            if abs(float(market_price) - float(bs_price)) > tolerance:
                if analytical_vega < 1e-8:
                    return implied_vol_bisection(S, K, T, r, market_price, option_type, 0.01, 5, tolerance, max_iterations)
                else:
                    sigma -= ((bs_price - market_price) / analytical_vega)
                    if not 0.01 <= sigma <= 5:
                        return implied_vol_bisection(S, K, T, r, market_price, option_type, 0.01, 5, tolerance, max_iterations)
                    else:
                        bs_price, d1, _ = black_scholes(S, K, T, r, sigma, option_type)
                        analytical_vega = S * norm.pdf(d1) * math.sqrt(T)
            else: 
                return sigma 
        else: 
            return implied_vol_bisection(S, K, T, r, market_price, option_type, 0.01, 5, tolerance, max_iterations)
    return implied_vol_bisection(S, K, T, r, market_price, option_type, 0.01, 5, tolerance, max_iterations)

def implied_vol_bisection(S, K, T, r, market_price, option_type, low_vol=0.01, high_vol=5, tolerance=0.0001, max_iterations=100):
    """Find implied volatility by repeatedly halving a bracket. 
    
    The Black-Scholes price rises with volatility, so if the market price lies 
    between the model prices at low_vol and high_vol, a solution exists inside 
    the bracket and halving will find it. Stops when the model price is within
    tolerance of the market price, so the volatility is only as precise as that
    price tolerance allows (less so where vega is small).
    
    Returns:
        Implied volatility, or float("nan") if the market price is outside the
        range of model prices over the bracket or the iteration limit is hit. 
        This function does not check the no-arbitrage bounds itself.
    """
    high_price, _, _ = black_scholes(S, K, T, r, high_vol, option_type)
    low_price, _, _ = black_scholes(S, K, T, r, low_vol, option_type)
    if not low_price <= market_price <= high_price:
        return float("nan")
    else: 
        mid_vol = (high_vol + low_vol) / 2 
        price, _, _ = black_scholes(S, K, T, r, mid_vol, option_type)
        for _ in range(max_iterations):
            if abs(float(market_price) - float(price)) > tolerance:
                mid_vol = (high_vol + low_vol) / 2 
                price, _, _ = black_scholes(S, K, T, r, mid_vol, option_type)
                if price > market_price: 
                    high_vol = mid_vol 
                elif price < market_price:
                    low_vol = mid_vol 
                else: 
                    return mid_vol
            else: 
                return mid_vol
        return float("nan")
