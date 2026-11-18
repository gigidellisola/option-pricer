import pytest 
from implied_vol import implied_vol, implied_vol_bisection 
from black_scholes import black_scholes 
import math 
from math import e 
from scipy.optimize import brentq

@pytest.mark.parametrize("K, T, r, option_type, vol, price", [(100, 1.0, 0.05, "call", 0.20, 10.4506), (100, 1.0, 0.05, "put", 0.20, 5.5735), (110, 0.5, 0.05, "call", 0.3, 5.5871), (90, 0.5, 0.05, "put", 0.25, 2.2150), (90, 1.0, 0.05, "call", 0.40, 22.9848), (110, 0.25, 0.05, "put", 0.15, 9.1653), (100, 1.0, 0.05, "call", 0.05, 5.2833), (100, 1.0, 0.05, "call", 2.00, 69.0575), (100, 1.0, -0.01, "call", 0.20, 7.5131), (105, 0.5, 0.00, "put", 0.35, 12.7889), (100, 3.0, 0.03, "call", 0.22, 19.1597)])
@pytest.mark.parametrize("solver", [implied_vol, implied_vol_bisection])
def test_round_trip(K, T, r, option_type, vol, price, solver):
    S = 100
    bs_price, _, _ = black_scholes(S, K, T, r, vol, option_type)
    imp_vol = solver(S, K, T, r, bs_price, option_type)
    assert vol == pytest.approx(imp_vol, abs=1e-4)
    assert price == pytest.approx(bs_price, abs=1e-4)

@pytest.mark.parametrize("K, T, r, option_type, vol, price", [(100, 1.0, 0.05, "call", 0.20, 10.4506), (100, 1.0, 0.05, "put", 0.20, 5.5735), (110, 0.5, 0.05, "call", 0.3, 5.5871), (90, 0.5, 0.05, "put", 0.25, 2.2150), (90, 1.0, 0.05, "call", 0.40, 22.9848), (110, 0.25, 0.05, "put", 0.15, 9.1653), (100, 1.0, 0.05, "call", 0.05, 5.2833), (100, 1.0, 0.05, "call", 2.00, 69.0575), (100, 1.0, -0.01, "call", 0.20, 7.5131), (105, 0.5, 0.00, "put", 0.35, 12.7889), (100, 3.0, 0.03, "call", 0.22, 19.1597)])
def test_brentq(K, T, r, option_type, vol, price):
    vol_mine = implied_vol(100, K, T, r, price, option_type)
    f = lambda x: black_scholes(100, K, T, r, x, option_type)[0] - price 
    vol_ref = brentq(f, 0.01, 5, xtol=1e-12)
    price_ref = black_scholes(100, K, T, r, vol_ref, option_type)[0]
    price_mine = black_scholes(100, K, T, r, vol_mine, option_type)[0]
    assert price_ref == pytest.approx(price_mine, abs=2e-4)
    assert vol_mine == pytest.approx(vol, abs=1e-4)


@pytest.mark.parametrize("K, T, option_type, vol, price", [(150, 0.5, "call", 1.2916018514238568, 23.4765), (130, 0.1, "put", 0.80, 31.7721), (150, 0.1, "call", 1.20, 3.4165), (130, 0.05, "call", 1.20, 2.6784), (70, 0.05, "put", 0.80, 0.1248)])
def test_fallback(K, T, option_type, vol, price):
    S, r = 100, 0.04
    imp_vol = implied_vol(S, K, T, r, price, option_type)
    assert imp_vol == pytest.approx(vol, abs=1e-3)

@pytest.mark.parametrize("K, option_type", [(90, "call"), (100, "put")])
def test_nan_return(K, option_type):
    S, T, r = 100, 1.0, 0.05
    if option_type == "call":
        prices = [14.3894, 14.0, 0.0, 100.0, 105.0]
        for price in prices:
            assert math.isnan(implied_vol(S, K, T, r, price, "call"))
        prices_bis = [14.0, 0.0, 100.0, 105.0]
        for price in prices_bis:
            assert math.isnan(implied_vol_bisection(S, K, T, r, price, "call"))
        floor = max(S - (K * (e ** (-r * T))), 0)
    else:
        prices = [0.0, 95.1229, 97.0]
        for price in prices: 
            assert math.isnan(implied_vol(S, K, T, r, price, "put"))
        floor = max((K * (e ** (-r * T))) - S, 0)
    assert 0.01 <= implied_vol_bisection(100, 90, T, r, 14.3894, "call") <= 5
    price, _, _ = black_scholes(S, K, T, r, 0.2, option_type)
    assert math.isnan(implied_vol(S, K, T, r, floor - 1, option_type))
    assert math.isnan(implied_vol(S, K, T, r, -1, option_type))
    assert math.isnan(implied_vol(S, K, T, r, price, option_type, max_iterations=1))
    price_far_sig, _, _ = black_scholes(S, K, T, r, 0.0001, option_type)
    assert math.isnan(implied_vol(S, K, T, r, price_far_sig, option_type)) 
    with pytest.raises(ValueError):
        implied_vol(S, K, T, r, price, "option")


@pytest.mark.parametrize("solver", [implied_vol, implied_vol_bisection])
def test_nan_floor(solver):
    assert math.isnan(solver(100, 100, 1.0, 0, 0.20, "call"))
    assert math.isnan(solver(100, 100, 1.0, 0.00, 99.5, "call"))
