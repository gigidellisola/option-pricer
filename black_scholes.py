"""Black-Scholes closed-form pricing for European options."""

from scipy.stats import norm 
import math 

def black_scholes(S, K, T, r, sigma, option_type):
    """Price a European call or put with the Black-Scholes formula.
    
    Assumes constant volatility and interest rate, lognormally distributed 
    prices, European exercise and no dividends. The rate r can be negative, 
    though the command line interface restricts --rate to [0, 1]. 
    
    Args:
        S: Spot price of the underlying.
        K: Strike price.
        T: Time to expiry in years.
        r: Annualised risk-free rate, continuously compounded (0.05 = 5%).
        sigma: Annualised volatility (0.20 = 20%).
        option_type: "call" or "put" (case-insensitive).
        
    Returns:
        Tuple (price, d1, d2) of floats. d1 and d2 are returned so that 
        other functions can reuse them.
        
    Raises:
        ValueError: if S, K, T, or sigma is not positive, or if option_type
        is not "call" or "put". 
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("number inputs must be positive")
    d1 = ((math.log(S / K) + (r + ((sigma ** 2) / 2)) * T) / (sigma * math.sqrt(T)))
    d2 = d1 - (sigma * math.sqrt(T))
    if option_type.lower() == "call": 
        price = ((S * norm.cdf(d1)) - (K * math.exp(-r * T) * norm.cdf(d2)))
    elif option_type.lower() == "put":
        price = ((K * math.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))) 
    else:
        raise ValueError("option type must be 'call' or 'put'")
    return float(price), float(d1), float(d2)
