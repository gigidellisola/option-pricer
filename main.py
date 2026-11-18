"""Command-line interface for the option pricing toolkit. 

Usage: python3 main.py <command> [options]. Commands: prices, convergence,
variance, greeks, exotics, market, and all. Run with --help for the options. 
"""

import argparse
import numpy as np 
import math 
import pandas as pd 
import os 
from datetime import datetime 
import matplotlib.pyplot as plt 
from black_scholes import black_scholes 
from greeks import finite_diff_greeks, analytical_greeks
from monte_carlo import stock_price_vectorized, stock_price_antithetic, european_payoff_calculator, trial_list, euro_antithetic_payoff, european_control_variate, asian_payoff_calculator, knock_in_payoff, knock_out_payoff
from market_data import fetch_smile 

def positive_float(text):
    """Argparse type: parse a finite number greater than zero."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number")
    if not value > 0:
        raise argparse.ArgumentTypeError("Input must be positive")
    elif not math.isfinite(value):
        raise argparse.ArgumentTypeError("Input must be a finite number")
    else:
        return value

def non_negative_percent(text):
    """Argparse type: parse a finite decimal from 0 to 1 (0.05 means 5%)."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number")
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("Input must be a finite number")
    elif not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("Input must be a decimal")
    else:
        return value

def option(text):
    """Argparse type: accept "call" or "put" in any case, return lowercase."""
    if not text.lower() in ["call", "put"]:
        raise argparse.ArgumentTypeError("Option type must be 'call' or 'put'")
    else: 
        return text.lower()

def positive_int(text):
    """Argparse type: parse any integer greater than zero."""
    try: 
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("Input must be an integer")
    if value <= 0: 
        raise argparse.ArgumentTypeError("Input must be positive")
    else: 
        return value 


def get_parser():
    """Build the command-line parser with its subcommands. 
    
    Each subcommand stores its run function in args.func. Options 
    shared by several commands are defined once and reused.
    """
    parser = argparse.ArgumentParser(description="Stock Analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    pricing = argparse.ArgumentParser(add_help=False)
    pricing.add_argument("--spot", type=positive_float, default=100, help="Spot price, default $100")
    pricing.add_argument("--strike", type=positive_float, default=100, help="Strike price, default $100")
    pricing.add_argument("--maturity", type=positive_float, default=1.0, help="Time to maturity (in years), default 1 year")
    pricing.add_argument("--rate", type=non_negative_percent, default=0.05, help="Nominal risk-free interest rate, continuously compounded (expressed as decimal), default = 0.05")
    pricing.add_argument("--sigma", type=positive_float, default=0.2, help="Annualised volatility, default 0.2")
    simulation = argparse.ArgumentParser(add_help=False)
    simulation.add_argument("--option-type", type=option, default="call", help="Option type; 'call' or 'put', default 'call'")
    simulation.add_argument("--seed", type=int, default=42, help="random seed, default 42")
    simulation.add_argument("--steps", type=positive_int, default=None, help="Number of time steps")
    trials_group = argparse.ArgumentParser(add_help=False)
    trials_group.add_argument("--trials", type=positive_int, default=20000, help="Trials, default 20,000 (for variance command, use even trial count)")
    plotting = argparse.ArgumentParser(add_help=False)
    plotting.add_argument("--show", action="store_true", help="Show plot")
    barrier_group = argparse.ArgumentParser(add_help=False)
    barrier_group.add_argument("--barrier", type=positive_float, default=90, help="Barrier price for knock out/in options, default $90")
    greeks_range = argparse.ArgumentParser(add_help=False)
    greeks_range.add_argument("--spot-min", type=positive_float, default=60, help="Minimum spot price for greeks, default $60")
    greeks_range.add_argument("--spot-max", type=positive_float, default=140, help="Maximum spot price for greeks, default $140")
    market_group = argparse.ArgumentParser(add_help=False)
    market_group.add_argument("--ticker", type=str, default="AAPL", help="Stock for market data analysis, default AAPL")
    market_group.add_argument("--target-days", type=positive_int, default=45, help="Target days to expiry for market data, default 45")
    market_group.add_argument("--min-days", type=positive_int, default=14, help="Minimum days to expiry for market data, default 14")
    market_group.add_argument("--min-volume", type=positive_int, default=2, help="Minimum volume of options trading today")
    market_group.add_argument("--min-open-interest", type=positive_int, default=3, help="Minimum open interest of option")
    prices = sub.add_parser("prices", parents=[pricing, simulation, trials_group])
    prices.set_defaults(func=run_prices)
    convergence = sub.add_parser("convergence", parents=[pricing, simulation, plotting])
    convergence.set_defaults(func=run_convergence)
    variance = sub.add_parser("variance", parents=[pricing, simulation, trials_group, plotting])
    variance.set_defaults(func=run_variance)
    greeks = sub.add_parser("greeks", parents=[pricing, plotting, greeks_range])
    greeks.set_defaults(func=run_greeks)
    exotics = sub.add_parser("exotics", parents=[pricing, simulation, trials_group, plotting, barrier_group])
    exotics.set_defaults(func=run_exotics)
    market = sub.add_parser("market", parents=[plotting, market_group])
    market.set_defaults(func=run_market)
    all_cmd = sub.add_parser("all", parents=[pricing, simulation, trials_group, plotting, barrier_group, greeks_range, market_group])
    all_cmd.set_defaults(func=run_all)
    return parser

def pick_steps(input_step, step_default):
    """Return the user's step count, or the default if none was given."""
    if input_step is None: 
        return step_default 
    return input_step 

def generator(x):
    """Return a numpy random Generator seeded with x."""
    return np.random.default_rng(x)

def run_prices(args):
    """Price one option by Black-Scholes and by Monte Carlo.
    
    Returns:
        Dictionary with black_scholes_price, monte_carlo_price, 
        standard_error, and se_gap (the gap between the two prices in 
        standard errors).
    """
    S, K, T, r, sigma, option_type, seed, steps, trials = args.spot, args.strike, args.maturity, args.rate, args.sigma, args.option_type, args.seed, args.steps, args.trials
    n_steps = pick_steps(steps, 1)
    rng = generator(seed)
    black_scholes_price, _, _ = black_scholes(S, K, T, r, sigma, option_type)
    path = stock_price_vectorized(r, sigma, T, trials, n_steps, S, rng=rng)
    monte_carlo_price, standard_error = european_payoff_calculator(path, K, r, T, option_type)
    se_gap = (abs(black_scholes_price - monte_carlo_price) / standard_error)
    return {"black_scholes_price": black_scholes_price, "monte_carlo_price": monte_carlo_price, "standard_error": standard_error, "se_gap": se_gap}

def run_convergence(args):
    """Run Monte Carlo at 100 to 100,000 trials and compare with Black-Scholes. 
    
    Returns:
        Dictionary with the trial counts, prices, standard errors, the 
        Black-Scholes price and the scaled errors (standard error times the 
        square root of N, which should be roughly constant).
    """
    S, K, T, r, sigma, option_type, seed, steps = args.spot, args.strike, args.maturity, args.rate, args.sigma, args.option_type, args.seed, args.steps
    n_steps = pick_steps(steps, 1)
    rng = generator(seed)
    trial_range = [100, 500, 1000, 5000, 10000, 100000]
    trial_range_str = ["100", "500", "1,000", "5,000", "10,000", "100,000"]
    scaled_errors = []
    monte_carlo_prices, standard_errors = trial_list(trial_range, S, K, T, r, sigma, option_type, n_steps, rng=rng)
    black_scholes_price, _, _ = black_scholes(S, K, T, r, sigma, option_type)
    for i in range(len(trial_range)): 
        scaled_error = standard_errors[i] * math.sqrt(trial_range[i])
        scaled_errors.append(scaled_error)
    return {"trial_range": trial_range, "trial_range_str": trial_range_str, "monte_carlo_prices": monte_carlo_prices, "standard_errors": standard_errors, "black_scholes_price": black_scholes_price, "scaled_errors": scaled_errors}

def run_variance(args):
    """Compare plain, control variate, and antithetic Monte Carlo estimators. 
    
    All use the same number of trials, which must be even. 
    
    Returns:
        Dictionary with each estimator's price and standard error, the variance
        reduction factors (plain variance divided by each method's variance), 
        the hedge ratio, the correlation, and the Black-Scholes reference price.
        
    Raises:
        ValueError: If the number of trials is odd. 
    """
    S, K, T, r, sigma, option_type, seed, trials, steps = args.spot, args.strike, args.maturity, args.rate, args.sigma, args.option_type, args.seed, args.trials, args.steps
    n_steps = pick_steps(steps, 1)
    rng = generator(seed)
    if not trials % 2 == 0: 
        raise ValueError("trial number must be even")
    plain_path = stock_price_vectorized(r, sigma, T, trials, n_steps, S, rng=rng)
    plain_price, plain_se = european_payoff_calculator(plain_path, K, r, T, option_type)
    hedge_ratio, corr, _, control_price, control_se = european_control_variate(plain_path, K, r, T, S, option_type)
    antithetic_path = stock_price_antithetic(r, sigma, T, trials, n_steps, S, rng=rng)
    antithetic_price, antithetic_se = euro_antithetic_payoff(antithetic_path, K, r, T, option_type)
    variance_reduction_control = (plain_se ** 2) / (control_se ** 2)
    variance_reduction_antithetic = (plain_se ** 2) / (antithetic_se ** 2)
    reference_price, _, _ = black_scholes(S, K, T, r, sigma, option_type)
    return {"plain_price": plain_price, "plain_se": plain_se, "control_price": control_price, "control_se": control_se, "variance_reduction_control": variance_reduction_control, "antithetic_price": antithetic_price, "antithetic_se": antithetic_se, "variance_reduction_antithetic": variance_reduction_antithetic, "hedge_ratio": hedge_ratio, "correlation": corr, "reference_price": reference_price}

def run_greeks(args):
    """Compute analytical and finite-difference Greeks across a range of spots.
    
    Uses 81 spot prices between spot_min and spot_max, for calls and puts, and 
    also at the input spot. 
    
    Returns: 
        Dictionary with the spot grid, call and put tables, the maximum 
        difference between the two methods for each Greek, the Greeks at 
        the input spot, and the strike. 
        
    Raises:
        ValueError: If the inputs are too small for the finite-difference 
        bump sizes or the spot range is invalid. 
    """
    S, K, T, r, sigma, spot_min, spot_max = args.spot, args.strike, args.maturity, args.rate, args.sigma, args.spot_min, args.spot_max
    if not (spot_min > 0.01 and spot_min < spot_max):
        raise ValueError("Spot minimum must be positive and less than spot maximum")
    if not T > 0.0001:
        raise ValueError("T must be greater than h_theta")
    if not sigma > 0.0001:
        raise ValueError("sigma must be greater than h_vega")
    if not K > 0:
        raise ValueError("strike must be positive")
    if not S > 0.01: 
        raise ValueError("spot price must be above $0.01")
    spot_grid = np.linspace(spot_min, spot_max, 81) 
    call = []
    put = []
    for spot in spot_grid: 
        delta_analytical, gamma_analytical, vega_analytical, theta_analytical = analytical_greeks(spot, K, T, r, sigma, "call")
        delta_fd, gamma_fd, vega_fd, theta_fd = finite_diff_greeks(spot, K, T, r, sigma, "call")
        price, _, _ = black_scholes(spot, K, T, r, sigma, "call")
        call.append({"spot price": spot, "Black Scholes price": price, "analytical delta": delta_analytical, "finite difference delta": delta_fd, "analytical gamma": gamma_analytical, "finite difference gamma": gamma_fd, "analytical vega": vega_analytical, "finite difference vega": vega_fd, "analytical theta": theta_analytical, "finite difference theta": theta_fd})
    for spot in spot_grid:
        delta_analytical, gamma_analytical, vega_analytical, theta_analytical = analytical_greeks(spot, K, T, r, sigma, "put")
        delta_fd, gamma_fd, vega_fd, theta_fd = finite_diff_greeks(spot, K, T, r, sigma, "put")
        price, _, _ = black_scholes(spot, K, T, r, sigma, "put")
        put.append({"spot price": spot, "Black Scholes price": price, "analytical delta": delta_analytical, "finite difference delta": delta_fd, "analytical gamma": gamma_analytical, "finite difference gamma": gamma_fd, "analytical vega": vega_analytical, "finite difference vega": vega_fd, "analytical theta": theta_analytical, "finite difference theta": theta_fd})
    call_table = pd.DataFrame(call)
    put_table = pd.DataFrame(put)
    call_delta_analytical_spot, call_gamma_analytical_spot, call_vega_analytical_spot, call_theta_analytical_spot = analytical_greeks(S, K, T, r, sigma, "call")
    call_delta_fd_spot, call_gamma_fd_spot, call_vega_fd_spot, call_theta_fd_spot = finite_diff_greeks(S, K, T, r, sigma, "call")
    put_delta_analytical_spot, put_gamma_analytical_spot, put_vega_analytical_spot, put_theta_analytical_spot = analytical_greeks(S, K, T, r, sigma, "put")
    put_delta_fd_spot, put_gamma_fd_spot, put_vega_fd_spot, put_theta_fd_spot = finite_diff_greeks(S, K, T, r, sigma, "put")
    greeks_at_spot = {"spot_greeks_call": {"spot_delta_anl": call_delta_analytical_spot, "spot_delta_fd": call_delta_fd_spot, "spot_gamma_anl": call_gamma_analytical_spot, "spot_gamma_fd": call_gamma_fd_spot, "spot_vega_anl": call_vega_analytical_spot, "spot_vega_fd": call_vega_fd_spot, "spot_theta_anl": call_theta_analytical_spot, "spot_theta_fd": call_theta_fd_spot}, "spot_greeks_put": {"spot_delta_anl": put_delta_analytical_spot, "spot_delta_fd": put_delta_fd_spot, "spot_gamma_anl": put_gamma_analytical_spot, "spot_gamma_fd": put_gamma_fd_spot, "spot_vega_anl": put_vega_analytical_spot, "spot_vega_fd": put_vega_fd_spot, "spot_theta_anl": put_theta_analytical_spot, "spot_theta_fd": put_theta_fd_spot}}
    call_max_diff_delta = max(abs(call_table["analytical delta"] - call_table["finite difference delta"]))
    call_max_diff_gamma = max(abs(call_table["analytical gamma"] - call_table["finite difference gamma"]))
    call_max_diff_vega = max(abs(call_table["analytical vega"] - call_table["finite difference vega"]))
    call_max_diff_theta = max(abs(call_table["analytical theta"] - call_table["finite difference theta"]))
    put_max_diff_delta = max(abs(put_table["analytical delta"] - put_table["finite difference delta"]))
    put_max_diff_gamma = max(abs(put_table["analytical gamma"] - put_table["finite difference gamma"]))
    put_max_diff_vega = max(abs(put_table["analytical vega"] - put_table["finite difference vega"]))
    put_max_diff_theta = max(abs(put_table["analytical theta"] - put_table["finite difference theta"]))
    max_differences = {"max_diff_call": {"delta": call_max_diff_delta, "gamma": call_max_diff_gamma, "vega": call_max_diff_vega, "theta": call_max_diff_theta}, "max_diff_put": {"delta": put_max_diff_delta, "gamma": put_max_diff_gamma, "vega": put_max_diff_vega, "theta": put_max_diff_theta}}
    return {"spot grid": spot_grid.tolist(), "call greeks": call_table, "put greeks": put_table, "max differences": max_differences, "greeks for spot input": greeks_at_spot, "strike": K}
    
def run_exotics(args):
    """Price vanilla, Asian, knock-in and knock-out options on the same paths.
    
    Returns: 
        Dictionary with the prices, the knock-in plus knock-out consistency
        gap and flag, whether the Asian price is below vanilla, the knock-out
        price across ten barrier levels, 20 sample paths, and the 
        Black-Scholes reference price. 
        
    Raises:
        ValueError: If the barrier is not between 0 and the spot price. 
    """
    S, K, T, r, sigma, option_type, seed, trials, steps, barrier = args.spot, args.strike, args.maturity, args.rate, args.sigma, args.option_type, args.seed, args.trials, args.steps, args.barrier
    n_steps = pick_steps(steps, 252)
    rng = generator(seed)
    tol = 1e-9
    if not barrier < S:
        raise ValueError("barrier value must be below spot price")
    if not barrier > 0: 
        raise ValueError("barrier value must be positive")
    path = stock_price_vectorized(r, sigma, T, trials, n_steps, S, rng=rng)
    path_sample = path[:20].tolist()
    vanilla_price, vanilla_se = european_payoff_calculator(path, K, r, T, option_type)
    asian_price = asian_payoff_calculator(path, K, r, T, option_type)
    knock_in_price = knock_in_payoff(path, barrier, K, r, T, option_type)
    knock_out_price = knock_out_payoff(path, barrier, K, r, T, option_type)
    gap = (knock_in_price + knock_out_price) - vanilla_price
    reference_price, _, _ = black_scholes(S, K, T, r, sigma, option_type)
    gap_in_se = abs(vanilla_price - reference_price) / vanilla_se
    if abs(gap) < tol:
        parity_check_barrier = True
    else: 
        parity_check_barrier = False
    if asian_price <= vanilla_price:
        asian_below_vanilla = True
    else: 
        asian_below_vanilla = False
    barrier_grid = np.linspace(S / 2, S - 1, 10)
    barrier_sensitivity = []
    for bar in barrier_grid:
        knock_out_p = knock_out_payoff(path, bar, K, r, T, option_type)
        barrier_sensitivity.append(knock_out_p)
    return {"vanilla_price": vanilla_price, "vanilla_se": vanilla_se, "asian_price": asian_price, "knock_in_price": knock_in_price, "knock_out_price": knock_out_price, "parity_check_barrier": parity_check_barrier, "gap": gap, "asian_below_vanilla": asian_below_vanilla, "barrier_sensitivity": barrier_sensitivity, "barrier_grid": barrier_grid.tolist(), "barrier": barrier, "path_sample": path_sample, "reference_price": reference_price, "gap_in_se": gap_in_se}

def run_market(args):
    """Build the implied volatility smile for the chosen ticker."""
    ticker, target_days, min_days, min_volume, min_open_interest = args.ticker, args.target_days, args.min_days, args.min_volume, args.min_open_interest
    market = fetch_smile(ticker, target_days, min_days, min_volume, min_open_interest)
    return market

def run_all(args):
    """Run every section. 
    
    If the market section fails (for example with no internet), the error is
    stored as {"error": message} so the other sections still run and print.
    """
    prices = run_prices(args)
    convergence  = run_convergence(args)
    variance = run_variance(args)
    greeks = run_greeks(args)
    exotics = run_exotics(args)
    try: 
        market = run_market(args)
    except Exception as error: 
        market = {"error": str(error)}
    return {"prices": prices, "convergence": convergence, "variance": variance, "greeks": greeks, "exotics": exotics, "market": market}


def prices_printer(section, args):
    """Print the prices section, with a 95% interval around the Monte Carlo price."""
    print()
    print("=== PRICES ===")
    print()
    print("Inputs")
    print(f"  option type: {args.option_type}")
    print(f"  spot: {args.spot} | strike: {args.strike}") 
    print(f"  maturity: {args.maturity} | rate: {args.rate}")
    print(f"  sigma: {args.sigma} | trials: {args.trials}")
    print(f"  steps: {pick_steps(args.steps, 1)} | seed: {args.seed}")
    print("Results")
    print(f"  Black-Scholes price: {section['black_scholes_price']:.4f}")
    print(f"  Monte Carlo price: {section['monte_carlo_price']:.4f}   (std error {section['standard_error']:.4f})")
    print(f"  Gap: {section['se_gap']:.2f} standard errors")
    interval_minus, interval_plus = section['monte_carlo_price'] - (1.96 * section['standard_error']), section['monte_carlo_price'] + (1.96 * section['standard_error'])
    if interval_minus <= section['black_scholes_price'] <= interval_plus:
        validity = "yes"
    else:
        validity = "no"
    print(f"  95% interval: [{interval_minus:.4f}, {interval_plus:.4f}]")
    print(f"  Contains the Black-Scholes price: {validity}")
    print()

def convergence_printer(section, args):
    """Print the convergence section."""
    print()
    print("=== CONVERGENCE ===")
    print()
    print(f"  option type: {args.option_type} | steps: {pick_steps(args.steps, 1)} | seed: {args.seed}")
    print(f"  Black-Scholes Reference price: {section['black_scholes_price']:.4f}")
    print()
    mc_array = np.array(section['monte_carlo_prices'])
    error_vs_bs = mc_array - section['black_scholes_price']
    se_array = np.array(section['standard_errors'])
    gap_std = abs(error_vs_bs) / se_array 
    convergence_table = pd.DataFrame({"N": section['trial_range_str'], "MC price": section['monte_carlo_prices'], "Std error": section['standard_errors'], "Error vs. BS": error_vs_bs, "Gap (std errs)": gap_std, "SE x sqrt(N)": section['scaled_errors']})
    print(convergence_table.to_string(index=False))
    print()

def variance_printer(section, args):
    """Print the variance reduction section."""
    reference_price = section['reference_price']
    print()
    print("=== VARIANCE REDUCTION ===")
    print()
    print(f"  option type: {args.option_type} | trials: {args.trials:,} | steps: {pick_steps(args.steps, 1)} | seed: {args.seed}")
    print(f"  Black-Scholes Reference price: {reference_price:.4f}")
    print()
    gap_se = []
    prices = [section['plain_price'], section['antithetic_price'], section['control_price']]
    std_errs = [section['plain_se'], section['antithetic_se'], section['control_se']]
    predicted_factor = 1 / (1 - (section['correlation'] ** 2))
    for i in range(3):
        gap = abs(prices[i] - reference_price) / std_errs[i]
        gap_se.append(gap)
    variance_frame = pd.DataFrame({"Method": ["Plain", "Antithetic", "Control variate"], "Price": [section['plain_price'], section['antithetic_price'], section['control_price']], "Std error": std_errs, "Variance reduction": [1.0, section['variance_reduction_antithetic'], section['variance_reduction_control']], "Gap (std errs)": gap_se})
    formats = {"Price": "{:.4f}".format, "Std error": "{:.4f}".format, "Variance reduction": "{:.2f}".format, "Gap (std errs)": "{:.2f}".format}
    print(variance_frame.to_string(index=False, formatters=formats))
    print()
    print(f"  Control variate: hedge ratio: {section['hedge_ratio']:.4f}, correlation: {section['correlation']:.4f}")
    print(f"  Predicted factor 1 / (1 - corr^2) = {predicted_factor:.2f}  (table shows {section['variance_reduction_control']:.2f})")
    print()

def greeks_printer(section, args):
    """Print the Greeks section."""
    print()
    print("=== GREEKS ===")
    print()
    print(f"  spot: {args.spot} | strike: {args.strike} | maturity: {args.maturity} | rate: {args.rate} | sigma: {args.sigma}")
    print(f"  grid: spot {args.spot_min} to {args.spot_max} (81 points)")
    print("  vega is per 1.00 change in sigma; theta is per year")
    print()
    print(f"  CALL at spot {args.spot}")
    greeks_at_spot = section['greeks for spot input']
    spot_call = greeks_at_spot['spot_greeks_call']
    spot_put = greeks_at_spot['spot_greeks_put']
    call_analytical = [spot_call['spot_delta_anl'], spot_call['spot_gamma_anl'], spot_call['spot_vega_anl'], spot_call['spot_theta_anl']]
    put_analytical = [spot_put['spot_delta_anl'], spot_put['spot_gamma_anl'], spot_put['spot_vega_anl'], spot_put['spot_theta_anl']]
    call_fd = [spot_call['spot_delta_fd'], spot_call['spot_gamma_fd'], spot_call['spot_vega_fd'], spot_call['spot_theta_fd']]
    put_fd = [spot_put['spot_delta_fd'], spot_put['spot_gamma_fd'], spot_put['spot_vega_fd'], spot_put['spot_theta_fd']]
    call_abs_diff = [abs(spot_call['spot_delta_anl'] - spot_call['spot_delta_fd']), abs(spot_call['spot_gamma_anl'] - spot_call['spot_gamma_fd']), abs(spot_call['spot_vega_anl'] - spot_call['spot_vega_fd']), abs(spot_call['spot_theta_anl'] - spot_call['spot_theta_fd'])]
    put_abs_diff = [abs(spot_put['spot_delta_anl'] - spot_put['spot_delta_fd']), abs(spot_put['spot_gamma_anl'] - spot_put['spot_gamma_fd']), abs(spot_put['spot_vega_anl'] - spot_put['spot_vega_fd']), abs(spot_put['spot_theta_anl'] - spot_put['spot_theta_fd'])]
    call_greeks_frame = pd.DataFrame({"Greek": ["delta", "gamma", "vega", "theta"], "Analytical": call_analytical, "Finite Diff": call_fd, "Abs Diff": call_abs_diff})
    put_greeks_frame = pd.DataFrame({"Greek": ["delta", "gamma", "vega", "theta"], "Analytical": put_analytical, "Finite Diff": put_fd, "Abs Diff": put_abs_diff})
    formats = {"Analytical": "{:.6f}".format, "Finite Diff": "{:.6f}".format, "Abs Diff": "{:.2e}".format}
    print(call_greeks_frame.to_string(index=False, formatters=formats))
    print()
    print(f"  PUT at spot {args.spot}")
    print(put_greeks_frame.to_string(index=False, formatters=formats))
    print()
    print("   Maximum |analytical - finite difference| over the spot grid")
    max_diffs = section["max differences"]
    max_diffs_call = max_diffs["max_diff_call"]
    max_diffs_put = max_diffs["max_diff_put"]
    call_diffs = [max_diffs_call["delta"], max_diffs_call["gamma"], max_diffs_call["vega"], max_diffs_call["theta"]]
    put_diffs = [max_diffs_put["delta"], max_diffs_put["gamma"], max_diffs_put["vega"], max_diffs_put["theta"]]
    max_diff_frame = pd.DataFrame({"Greek": ["delta", "gamma", "vega", "theta"], "Call": call_diffs, "Put": put_diffs})
    print(max_diff_frame.to_string(index=False, float_format="{:.2e}".format))
    print()

def exotics_printer(section, args):
    """Print the exotics section."""
    print()
    print("=== EXOTICS ===")
    print()
    print(f"  option type: {args.option_type} | spot: {args.spot} | strike: {args.strike}")
    print(f"  maturity: {args.maturity} | rate: {args.rate} sigma: {args.sigma}")
    print(f"  barrier: {args.barrier} (down barrier) | trials: {args.trials:,} | steps: {pick_steps(args.steps, 252)} | seed: {args.seed}")
    print()
    print("   Prices")
    print(f"  Black-Scholes reference: {section['reference_price']:.4f}, gap: {section['gap_in_se']:.2f} standard errors")
    print(f"  Vanilla (European)  {section['vanilla_price']:.4f}  (std error {section['vanilla_se']:.4f})")
    print(f"  Asian (arithmetic)  {section['asian_price']:.4f}")
    print(f"  Knock-out           {section['knock_out_price']:.4f}")
    print(f"  Knock-in            {section['knock_in_price']:.4f}")
    print()
    print("  Consistency checks")
    if section['parity_check_barrier']:
        knock_check = "PASS"
    else:
        knock_check = "FAIL"
    if section['asian_below_vanilla']:
        asian_check = "yes"
    else:
        asian_check = "no"
    print(f"  Knock-out + knock-in - vanilla = {section['gap']:.2e}  {knock_check}")
    print()
    print("  Observations")
    print(f"  Asian below vanilla: {asian_check} (expected for call options)")
    print()
    print("   Knock-out price by barrier level")
    print()
    knock_out_price = np.array(section['barrier_sensitivity'])
    share_of_vanilla = (knock_out_price / section['vanilla_price']) * 100
    barrier_frame = pd.DataFrame({"Barrier": section['barrier_grid'], "Knock-out price": section['barrier_sensitivity'], "Share of vanilla": share_of_vanilla})
    formats = {"Barrier": "{:.2f}".format, "Knock-out price": "{:.2f}".format, "Share of vanilla": "{:.1f}%".format}
    print(barrier_frame.to_string(index=False, formatters=formats))
    print()

def market_printer(section, args):
    """Print the market section, or the reason no smile could be built."""
    if "error" in section: 
        print()
        print("=== MARKET ===")
        print()
        print(f"  could not fetch market data: {section['error']}")
        return 
    elif section['message'] is not None and section['expiry'] is not None: 
        print()
        print("=== MARKET ===")
        print()
        print(f"  ticker: {section['ticker']} | expiry: {section['expiry']} | time to expiry: {section['time']:.4f} years")
        print(f"  spot: {section['spot']:.2f} | risk-free rate: {(section['rate'] * 100):.2f}%")
        print()
        counts = section['counts']
        removed_filter_call = round(counts['call_listed_count'] - counts['filtered_call_count'])
        removed_filter_put = round(counts['put_listed_count'] - counts['filtered_put_count'])
        call_count = [counts['call_listed_count'], removed_filter_call, counts['filtered_call_count'], counts['call_nan_count'], counts['call_valid_count']]
        put_count = [counts['put_listed_count'], removed_filter_put, counts['filtered_put_count'], counts['put_nan_count'], counts['put_valid_count']]
        market_data = pd.DataFrame({"Data quality": ["listed", "removed by liquidity filter", "liquid", "solver failed", "usable (valid vol)"], "Calls": call_count, "Puts": put_count})
        print(market_data.to_string(index=False))
        if section['message'] == "no options passed the liquidity filter or solver failed":
            if (market_data["Calls"][0] + market_data["Puts"][0]) == (market_data["Calls"][1] + market_data["Puts"][1]):
                print("no options passed the liquidity filter")
            elif (market_data["Calls"][2] + market_data["Puts"][2]) == (market_data["Calls"][3] + market_data["Puts"][3]):
                print("solver failed")
        else:
            print(section['message'])
        print()
    elif section['expiry'] is None: 
        print()
        print("=== MARKET ===")
        print()
        print(f"  ticker: {section['ticker']}")
        print(f"  {section['message']}")
        print()
    else: 
        days = section['time'] * 365 
        print("=== MARKET ===")
        print()
        print(f"  ticker: {section['ticker']} | expiry: {section['expiry']} | time to expiry: {section['time']:.4f} years (about {round(days)} days)")
        print(f"  spot: {section['spot']:.2f} | risk-free rate: {(section['rate'] * 100):.2f}%")
        print()
        counts = section['counts']
        removed_filter_call = round(counts['call_listed_count'] - counts['filtered_call_count'])
        removed_filter_put = round(counts['put_listed_count'] - counts['filtered_put_count'])
        call_count = [counts['call_listed_count'], removed_filter_call, counts['filtered_call_count'], counts['call_nan_count'], counts['call_valid_count']]
        put_count = [counts['put_listed_count'], removed_filter_put, counts['filtered_put_count'], counts['put_nan_count'], counts['put_valid_count']]
        market_data = pd.DataFrame({"Data quality": ["listed", "removed by liquidity filter", "liquid", "solver failed", "usable (valid vol)"], "Calls": call_count, "Puts": put_count})
        print(market_data.to_string(index=False))
        print()
        smile_table = section['smile']
        vols = np.array(smile_table['volatility']) * 100 
        print(f"  Smile: {len(smile_table)} points, strikes {smile_table.iloc[0]['strike']:.2f} to {smile_table.iloc[-1]['strike']:.2f}")
        print(f"  Implied vol: {vols.min():.1f}% to {vols.max():.1f}%")
        print()

def save_figure(fig, name, show, folder="images"):
    """Save a figure as a PNG, optionally show it, then close it.
    
    Creates the folder if it does not exist. 
    
    Returns:
        The path of the saved file. 
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show: 
        plt.show()
    plt.close(fig)
    return path

def convergence_plotter(section, args):
    """Plot price against trial count and standard error against trial count."""
    trial_range = section['trial_range']
    monte_carlo_prices, standard_errors = section['monte_carlo_prices'], section['standard_errors']
    scaled_errors, black_scholes_price = section['scaled_errors'], section['black_scholes_price']
    show = getattr(args, "show", False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    errs = 1.96 * np.array(standard_errors)
    ax1.errorbar(trial_range, monte_carlo_prices, yerr=errs, fmt="o-", capsize=3, label="Monte Carlo price ±1.96SE")
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of trials (log scale)")
    ax1.set_ylabel("Option price")
    ax1.set_title(f"Convergence of price ({args.option_type})")
    ax1.axhline(black_scholes_price, linestyle="--", color="gray", label="Black-Scholes price")
    ax1.legend()
    ax2.plot(trial_range, standard_errors, marker="o", label="standard error")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of trials (log scale)")
    ax2.set_ylabel("Standard error")
    ax2.set_title(f"Convergence of standard error ({args.option_type})")
    c = np.median(scaled_errors)
    reference = c / np.sqrt(np.array(trial_range))
    ax2.plot(trial_range, reference, linestyle="--", color="gray", label="c/√N reference")
    ax2.legend()
    fig.suptitle(f"spot: {args.spot} | strike: {args.strike} | maturity: {args.maturity} | rate: {args.rate} | sigma: {args.sigma} | steps: {pick_steps(args.steps, 1)} | seed: {args.seed}")
    fig.tight_layout()
    save_figure(fig, "convergence", show)

def variance_plotter(section, args):
    """Plot the variance reduction factor of each method (plain = 1)."""
    names = ["Plain", "Antithetic", "Control Variate"]
    heights = [1.0, section['variance_reduction_antithetic'], section['variance_reduction_control']]
    show = getattr(args, "show", False)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, heights)
    ax.bar_label(bars, fmt="%.2fx")
    ax.set_ylabel("Variance Reduction Factor (plain = 1.0)")
    ax.set_title(f"Number of trials: {args.trials:,}")
    fig.suptitle("Variance Reduction")
    save_figure(fig, "variance", show)

def greeks_panels_plotter(section, args):
    """Plot analytical against finite-difference Greek across the spot range."""
    call_table = section['call greeks']
    put_table = section['put greeks']
    show = getattr(args, "show", False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, greek in zip(axes.flatten(), ["delta", "gamma", "vega", "theta"]):
        ax.plot(call_table['spot price'], call_table[f'analytical {greek}'], color="C0", label="call analytical")
        ax.scatter(call_table['spot price'].iloc[::4], call_table[f'finite difference {greek}'].iloc[::4], color="C0", label="call finite difference")
        ax.plot(put_table['spot price'], put_table[f'analytical {greek}'], linestyle="--", color="C1", label="put analytical")
        ax.scatter(put_table['spot price'].iloc[::4], put_table[f'finite difference {greek}'].iloc[::4], marker="x", color="C1", label="put finite difference")
        ax.legend()
        ax.set_title(greek.title())
        ax.set_xlabel("spot prices")
        ax.grid(True, alpha=0.3)
        if greek in ["gamma", "vega"]:
            ax.text(0.02, 0.98, "call and put coincide\n(put-call parity)", transform=ax.transAxes, fontsize=9, va="top")
    fig.suptitle("Analytical vs. finite difference greeks")
    axes[0, 0].set_ylabel("Delta (\\$ per \\$1 of spot)")
    axes[0, 1].set_ylabel("Gamma (change in delta per $1)")
    axes[1, 0].set_ylabel("Vega ($ per 1.00 of sigma)")
    axes[1, 1].set_ylabel("Theta ($ per year)")
    fig.tight_layout()
    save_figure(fig, "greeks_panels", show)

def greeks_value_plotter(section, args):
    """Plot option value against spot, with the payoff at expiry."""
    show = getattr(args, "show", False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    spots = section['spot grid']
    call = section['call greeks']
    put = section['put greeks']
    bs_price_call = call['Black Scholes price']
    bs_price_put = put['Black Scholes price']
    ax1.plot(spots, bs_price_call, label="Black-Scholes price")
    ax1.plot(spots, np.maximum(np.array(spots) - args.strike, 0), label="Payoff at expiry")
    ax1.axvline(args.strike, linestyle=":")
    ax1.set_title("Call option")
    ax1.set_xlabel("Spot price")
    ax1.set_ylabel("Option value (USD)")
    ax2.plot(spots, bs_price_put, label="Black-Scholes price")
    ax2.plot(spots, np.maximum(args.strike - np.array(spots), 0), label="Payoff at expiry")
    ax2.axvline(args.strike, linestyle=":")
    ax2.set_title("Put option")
    ax2.set_xlabel("Spot price")
    ax2.set_ylabel("Option value (USD)")
    ax1.legend()
    ax2.legend()
    fig.suptitle(f"strike: {args.strike} | maturity: {args.maturity} | rate: {args.rate} | sigma: {args.sigma}")
    fig.tight_layout()
    save_figure(fig, "greeks_value", show)

def greeks_plotter(section, args):
    """Draw both Greeks figures."""
    return greeks_panels_plotter(section, args), greeks_value_plotter(section, args)

def exotics_barrier(section, args):
    """Plot the knock-out price against barrier level."""
    show = getattr(args, "show", False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(section['barrier_grid'], section['barrier_sensitivity'], marker="o", label="Knock-out price")
    ax.axhline(section['vanilla_price'], label="Vanilla price (simulated, ceiling)", linestyle="--", color="gray")
    ax.scatter(args.barrier, section['knock_out_price'], label=f"chosen barrier {args.barrier}", color="red")
    ax.set_xlabel("Barrier level (USD)")
    ax.set_ylabel("Knock-out price (USD)")
    ax.legend()
    ax.set_title(f"option type: {args.option_type} | trials: {args.trials:,} | steps: {pick_steps(args.steps, 252)}")
    fig.suptitle(f"Knock-out {args.option_type} price vs. barrier level")
    save_figure(fig, "exotics_barrier", show)

def exotics_paths(section, args):
    """Plot sample paths, coloured by whether they touched the barrier."""
    show = getattr(args, "show", False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    paths = section['path_sample']
    times = range(1, len(paths[0]) + 1)
    barrier_hit_count = 0
    for path in paths:
        if min(path) <= args.barrier:
            color = "tab:red"
            barrier_hit_count += 1
        else: 
            color = "C0"
        ax.plot(times, path, color=color, linewidth=0.8, alpha=0.5)
    ax.plot([], [], color="tab:red", label="reached barrier")
    ax.plot([], [], color="C0", label="stayed above barrier")
    ax.text(0.02, 0.98, f"{barrier_hit_count} of {len(section['path_sample'])} sample paths touched the barrier", transform=ax.transAxes, fontsize=9, va="top")
    ax.axhline(args.barrier, color="red", linestyle="--", label="barrier")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Stock price (USD)")
    fig.suptitle(f"{len(section['path_sample'])} sample paths, barrier {args.barrier} ({args.option_type}, knock-out)")
    ax.set_title(f"steps: {pick_steps(args.steps, 252)} | seed: {args.seed}")
    ax.legend()
    save_figure(fig, "exotics_paths", show)

def exotics_plotter(section, args):
    """Draw both exotics figures."""
    return exotics_barrier(section, args), exotics_paths(section, args)
   
def market_plotter(section, args):
    """Plot the implied volatility smile, if a smile was built."""
    if "error" in section:
        return
    elif section['message'] is not None:
        return 
    elif len(section['smile']) == 0:
        return 
    show = getattr(args, "show", False)
    fig, ax = plt.subplots(figsize=(8, 5))
    smile = pd.DataFrame(section['smile'])
    calls = smile[smile["option type"] == "call"]
    puts = smile[smile["option type"] == "put"]
    x_calls = calls["strike"]
    x_puts = puts["strike"]
    y_calls = np.array(calls["volatility"]) * 100 
    y_puts = np.array(puts["volatility"]) * 100 
    ax.scatter(x_calls, y_calls, color="C0", label="call options")
    ax.scatter(x_puts, y_puts, color="C1", label="put options")
    ax.axvline(section['spot'], linestyle="--", color="gray", label=f"spot price {section['spot']:.2f}")
    ax.set_xlabel("Strike (USD)")
    ax.set_ylabel("Implied volatility (%)")
    ax.legend()
    fig.suptitle(f"{section['ticker']} implied volatility smile, expiry {section['expiry']}")
    ax.set_title(f"spot: {section['spot']:.2f} | days to expiry: {round(section['time'] * 365)} | {len(smile)} points run time | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "market_smile", show)
    
PRINTERS = {"prices": prices_printer, "convergence": convergence_printer, "variance": variance_printer, "greeks": greeks_printer, "exotics": exotics_printer, "market": market_printer}
PLOTTERS = {"convergence": convergence_plotter, "variance": variance_plotter, "greeks": greeks_plotter, "exotics": exotics_plotter, "market": market_plotter}

def main():
    """Parse the command line, run the chosen command, then print and plot."""
    parser = get_parser()
    args = parser.parse_args()
    result = args.func(args)
    if args.command == "all":
        results = result 
    else:
        results = {args.command: result}
    for name, section in results.items():
        if name in PRINTERS:
            PRINTERS[name](section, args)
        else:
            print(name, section)
        if name in PLOTTERS:
            PLOTTERS[name](section, args)


if __name__ == "__main__":
    main()