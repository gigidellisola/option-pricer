"""Monte Carlo simulation of stock prices and option payoffs. 

Prices follow geometric Brownian motion under the risk-neutral measure, 
with drift equal to the risk-free rate r. T is in years. The number of time
steps is the total over the option's life, not per year. Payoffs are discounted
at exp(-rT).
"""

import numpy as np 
import math 

def payoff_by_type(option_type, K, final_price):
    """Return the undiscounted call or put payoff for each price.
    
    Raises:
        ValueError: If option_type is not "call" or "put". 
    """
    if option_type.lower() == "call":
        payoff = np.maximum(final_price - K, 0) 
    elif option_type.lower() == "put":
        payoff = np.maximum(K - final_price, 0)
    else: 
        raise ValueError("option type must be 'call' or 'put'")
    return payoff

def stock_price_vectorized(r, sigma, T, n_trials, n_steps, start_price, rng=None):
    """Simulate geometric Brownian motion paths. 
    
    Each step multiplies the price by exp((r - sigma^2 / 2) dt + sigma sqrt(dt) Z)
    with Z standard normal. This is the exact solution of the model, so there is no
    discretisation error in the terminal price.
    
    Args:
        n_trials: Number of paths. 
        n_steps: Number of time steps per path. 
        start_price: Price at time zero. 
        rng: Optional numpy Generator, for reproducible results. 
        
    Returns: 
        Array of shape (n_trials, n_steps) with the prices at steps 1 to 
        n_steps. The starting price is not included.
    """
    if rng is None: 
        rng = np.random.default_rng() 
    dt = T / n_steps
    Z = rng.standard_normal((n_trials, n_steps))
    multiplier = (((r - ((sigma ** 2) / 2)) * dt) + (sigma * math.sqrt(dt) * Z))
    growth_factor = np.exp(multiplier)
    stock_position = np.cumprod(growth_factor, axis=1)
    stock_price = stock_position * start_price 
    return stock_price

def stock_price_antithetic(r, sigma, T, n_trials, n_steps, start_price, rng=None): 
    """Simulate paths in antithetic pairs for variance reduction. 
    
    The first half of the rows use random draws Z and the second half use -Z, 
    so row i and row i + n_trials // 2 are mirror images. If n_trials is odd, 
    one path is dropped, so the result has 2 * (n_trials // 2) rows. 
    """
    if rng is None: 
        rng = np.random.default_rng()
    dt = T / n_steps
    half = n_trials // 2
    Z = rng.standard_normal((half, n_steps))
    Z = np.concatenate((Z, -Z), axis=0)
    multiplier = (((r - ((sigma ** 2) / 2)) * dt) + (sigma * math.sqrt(dt) * Z))
    growth_factor = np.exp(multiplier)
    stock_position = np.cumprod(growth_factor, axis=1)
    stock_price = stock_position * start_price 
    return stock_price

def european_payoff_calculator(price_path, K, r, T, option_type):
    """Price a European option from simulated paths. 
    
    Uses the last column of price_path as the terminal price. 
    
    Returns:
        Tuple (discounted mean payoff, standard error of the mean).
    """
    final_price = price_path[:, -1]
    payoff = payoff_by_type(option_type, K, final_price)
    discount_factor = np.exp(-r * T)
    discounted_payoff = discount_factor * payoff 
    price = np.mean(discounted_payoff)
    standard_error = np.std(discounted_payoff, ddof=1) / np.sqrt(len(discounted_payoff))
    return float(price), float(standard_error)

def trial_list(trial_counts, S, K, T, r, sigma, option_type, n_steps=1, rng=None):
    """Price a European option at several trial counts. 
    
    Each count uses fresh, independent paths. Used to show how the standard
    error shrinks as the number of trials grows. 
    
    Returns:
        Tuple (prices, standard_errors), two lists in the order of
        trial_counts.
    """
    prices = []
    standard_errors = []
    if rng is None: 
        rng = np.random.default_rng()
    for n in trial_counts:
        path = stock_price_vectorized(r, sigma, T, n, n_steps, S, rng=rng)
        price, se = european_payoff_calculator(path, K, r, T, option_type)
        prices.append(price)
        standard_errors.append(se)
    return prices, standard_errors 

def euro_antithetic_payoff(price_path, K, r, T, option_type):
    """Price a European option from antithetic paths. 
    
    Expects paths from stock_price_antithetic, with the first half of 
    the rows paired with the second half. Averages each pair first, 
    then takes the standard error across the pairs, because pairs are 
    independent but the two paths within a pair are not.
    
    Returns:
        Tuple (price, standard error). 
    """
    final_price = price_path[:, -1]
    half = price_path.shape[0] // 2
    payoff = payoff_by_type(option_type, K, final_price)
    discount_factor = np.exp(-r * T)
    discounted_payoff = discount_factor * payoff 
    first_half = discounted_payoff[:half]
    second_half = discounted_payoff[half:]
    pair_avgs = (first_half + second_half) / 2
    price = np.mean(pair_avgs)
    standard_error = np.std(pair_avgs, ddof=1) / np.sqrt(half)
    return float(price), float(standard_error)

def european_control_variate(price_path, K, r, T, start_price, option_type):
    """Price a European option using the stock price as a control variate.
    
    The control is the discounted terminal price, whose risk-neutral 
    expected value is start_price. The payoff is adjusted by beta times
    the control's sampling error, where beta is the sample covariance
    divided by the variance of the control. Beta is estimated from the 
    same sample, which adds a small bias. The expected variance reduction
    factor is 1 / (1 - rho^2), where rho is the correlation. 
    
    Returns:
        Tuple (hedge_ratio, correlation, control_mean, price, standard_error).
        control_mean should be close to start_price.
    """
    final_price = price_path[:, -1]
    payoff = payoff_by_type(option_type, K, final_price)
    discount_factor = np.exp(-r * T)
    # E[discounted terminal price] = start_price under the risk-neutral measure.
    X = final_price * discount_factor 
    Y = discount_factor * payoff 
    covariance = np.cov(X, Y)[0, 1]
    hedge_ratio = covariance / np.var(X, ddof=1)
    corr = np.corrcoef(X, Y)[0, 1]
    adjusted_payoff = Y - hedge_ratio * (X - start_price)
    price = np.mean(adjusted_payoff)
    standard_error = np.std(adjusted_payoff, ddof=1) / np.sqrt(len(price_path)) 
    return float(hedge_ratio), float(corr), float(np.mean(X)), float(price), float(standard_error)
    
def asian_payoff_calculator(price_path, K, r, T, option_type):
    """Price an arithmetic-average Asian option. 
    
    The payoff uses the average of all simulated prices on each path 
    (the starting price is excluded), then is discounted. Returns the 
    price only, with no standard error.
    """
    final_price = price_path.mean(axis=1)
    payoff = payoff_by_type(option_type, K, final_price)
    average_payoff = np.sum(payoff) / len(payoff)
    discount_factor = np.exp(-r * T)
    discounted_value = float(average_payoff * discount_factor)
    return float(discounted_value)

def knock_out_payoff(price_path, barrier, K, r, T, option_type):
    """Price a down-and-out barrier option. 
    
    The payoff is zero if the path falls to or below the barrier at 
    any simulated step. The barrier is monitored only at the 
    simulation steps, so the price is higher than it would be 
    under continuous monitoring. No rebate. Returns the price only. 
    """
    final_price = np.array(price_path[:, -1])
    payoff = payoff_by_type(option_type, K, final_price)
    # discrete monitoring misses crossings between steps.
    touched = price_path.min(axis=1) <= barrier 
    payoff[touched] = 0 
    average_payoff = np.sum(payoff) / len(payoff)
    discount_factor = np.exp(-r * T)
    discounted_value = float(average_payoff * discount_factor)
    return float(discounted_value)

def knock_in_payoff(price_path, barrier, K, r, T, option_type):
    """Price a down-and-in barrier option. 
    
    Pays the vanilla payoff only if the path falls to or below the 
    barrier at some simulated step. On the same paths, knock-in plus
    knock-out equals the vanilla price exactly, which makes a useful
    consistency check. Returns the price only. 
    """
    final_price = np.array(price_path[:, -1])
    payoff = payoff_by_type(option_type, K, final_price)
    untouched = price_path.min(axis=1) > barrier 
    payoff[untouched] = 0
    average_payoff = np.sum(payoff) / len(payoff)
    discount_factor = np.exp(-r * T)
    discounted_value = float(average_payoff * discount_factor)
    return float(discounted_value)

 