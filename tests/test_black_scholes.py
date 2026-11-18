import pytest 
import math 
from black_scholes import black_scholes 

@pytest.mark.parametrize("S, K, T, r, sigma, option_type, expected", [(100, 100, 1.0, 0.05, 0.2, "call", (10.450583, 0.35, 0.15)), (100, 100, 1.0, 0.05, 0.2, "CALL", (10.450583, 0.35, 0.15)), (100, 100, 1.0, 0.05, 0.2, "put", (5.573526, 0.35, 0.15))])
def test_black_scholes(S, K, T, r, sigma, option_type, expected):
    assert black_scholes(S, K, T, r, sigma, option_type) == pytest.approx(expected, abs=1e-4)

def test_inputs():
    with pytest.raises(ValueError):
        black_scholes(100, 100, 1.0, 0.05, 0.2, "option")
    with pytest.raises(ValueError):
        black_scholes(-5, 100, 1.0, 0.05, 0.2, "option")  
    with pytest.raises(ValueError):
        black_scholes(100, -5, 1.0, 0.05, 0.2, "option") 
    with pytest.raises(ValueError):
        black_scholes(100, 100, -1, 0.05, 0.2, "option") 
    with pytest.raises(ValueError):
        black_scholes(-5, 100, 1.0, 0.05, -1, "option") 


def test_put_call_parity():
    C, _, _ = black_scholes(100, 100, 1.0, 0.05, 0.2, "call")
    P, _, _ = black_scholes(100, 100, 1.0, 0.05, 0.2, "put")
    S, K, T, r = 100, 100, 1.0, 0.05
    assert (C - P) - (S - (K * math.exp(-r * T))) == pytest.approx(0, 1e-9)

def test_bounds():
    price_call, _, _ = black_scholes(100, 100, 1.0, 0.05, 0.2, "call")
    price_put, _, _ = black_scholes(100, 100, 1.0, 0.05, 0.2, "put")
    S, K, T, r = 100, 100, 1.0, 0.05
    call_max = max(S - (K * math.exp(-r * T)), 0)
    put_max = max((K * math.exp(-r * T)) - S, 0)
    assert S >= price_call >= call_max
    assert (K * math.exp(-r * T)) >= price_put >= put_max

def test_direction():
    assert black_scholes(90, 100, 1.0, 0.05, 0.2, "call") < black_scholes(100, 100, 1.0, 0.05, 0.2, "call")
    assert black_scholes(100, 100, 1.0, 0.05, 0.1, "call") < black_scholes(100, 100, 1.0, 0.05, 0.2, "call")
    assert black_scholes(100, 100, 0.9, 0.05, 0.2, "call") < black_scholes(100, 100, 1.0, 0.05, 0.2, "call")
    assert black_scholes(100, 90, 1.0, 0.05, 0.2, "call") > black_scholes(100, 100, 1.0, 0.05, 0.2, "call")

def test_scaling():
    price0, _, _ = black_scholes(300, 300, 1.0, 0.05, 0.2, "call")
    price1, _, _ = black_scholes(100, 100, 1.0, 0.05, 0.2, "call")
    assert price0 - (3 * price1) == pytest.approx(0, abs=1e-9)
    price0, _, _ = black_scholes(300, 300, 1.0, 0.05, 0.2, "put")
    price1, _, _ = black_scholes(100, 100, 1.0, 0.05, 0.2, "put")
    assert price0 - (3 * price1) == pytest.approx(0, abs=1e-9)

def test_identity():
    sigma, T = 0.2, 1.0
    _, d1, d2 = black_scholes(100, 100, T, 0.05, sigma, "call")
    assert d2 - (d1 - (sigma * math.sqrt(T))) == pytest.approx(0, abs=1e-4)
    sigma, T = 0.1, 1.0 
    _, d1, d2 = black_scholes(100, 100, T, 0.05, sigma, "call")
    assert d2 - (d1 - (sigma * math.sqrt(T))) == pytest.approx(0, abs=1e-4)

@pytest.mark.parametrize("S, option_type", [(400, "call"), (20, "put"), (400, "put"), (20, "call")])
def test_limits_spot(S, option_type):
    K, T, r, sigma = 100, 1.0, 0.05, 0.2
    if option_type == "call":
        price_limit = max(S - (K * math.exp(-r * T)), 0)
    else: 
        price_limit = max((K * math.exp(-r * T)) - S, 0)
    price, _, _ = black_scholes(S, K, T, r, sigma, option_type)
    assert price == pytest.approx(price_limit, abs=1e-6)

@pytest.mark.parametrize("S, T, sigma, option_type", [(100, 1.0, 1e-4, "call"), (90, 1.0, 1e-4, "put"), (100, 1.0, 50, "call"), (110, 1e-6, 0.2, "call"), (90, 1e-6, 0.2, "put"), (100, 1e-6, 0.2, "call"), (100, 1e-6, 0.2, "put")])
def test_limits(S, T, sigma, option_type):
    K, r = 100, 0.05 
    price, _, _ = black_scholes(S, K, T, r, sigma, option_type)
    if sigma == 1e-4:
        if option_type == "call":
            price_limit = max(S - (K * math.exp(-r * T)), 0)
        else: 
            price_limit = max((K * math.exp(-r * T)) - S, 0)
        assert price == pytest.approx(price_limit, abs=1e-3)
    elif sigma == 50:
        if option_type == "call":
            price_limit = S
        else: 
            price_limit = K * math.exp(-r * T)
        assert price == pytest.approx(price_limit, abs=1e-6)
    if T == 1e-6 and S != 100:
        if option_type == "call":
            price_limit = max(S - (K * math.exp(-r * T)), 0)
        else: 
            price_limit = max((K * math.exp(-r * T)) - S, 0)
        assert price == pytest.approx(price_limit, abs=1e-4)
    elif T == 1e-6 and S == 100:
        assert price == pytest.approx(0.008, abs=0.01)
    
    

    



