import pytest
from greeks import finite_diff_greeks, analytical_greeks
from black_scholes import black_scholes
import math 

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_analytical_greeks(option_type):
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    delta, gamma, vega, theta = analytical_greeks(S, K, T, r, sigma, option_type)
    with pytest.raises(ValueError):
        analytical_greeks(S, K, T, r, sigma, "option")
    assert gamma == pytest.approx(0.018762017345846895, rel=1e-9)
    assert vega == pytest.approx(37.52403469169379, rel=1e-9)
    if option_type == "call":
        assert delta == pytest.approx(0.6368306511756191, rel=1e-9)
        assert theta == pytest.approx(-6.414027546438197, rel=1e-9)
    else:
        assert delta == pytest.approx(-0.3631693488243809, rel=1e-9)
        assert theta == pytest.approx(-1.657880423934626, rel=1e-9) 

@pytest.mark.parametrize("S, option_type", [(100, "call"), (100, "put"), (90, "call"), (90, "put"), (110, "call"), (110, "put")])
def test_comparison(S, option_type):
    K, T, r, sigma = 100, 1.0, 0.05, 0.2
    delta_an, gamma_an, vega_an, theta_an = analytical_greeks(S, K, T, r, sigma, option_type)
    delta_fd, gamma_fd, vega_fd, theta_fd = finite_diff_greeks(S, K, T, r, sigma, option_type)
    assert delta_an == pytest.approx(delta_fd, 1e-5)
    assert gamma_an == pytest.approx(gamma_fd, 1e-5)
    assert vega_an == pytest.approx(vega_fd, 1e-5)
    assert theta_an == pytest.approx(theta_fd, 1e-3)

def test_relationships():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    call_delta_an, call_gamma_an, call_vega_an, call_theta_an = analytical_greeks(S, K, T, r, sigma, "call")
    put_delta_an, put_gamma_an, put_vega_an, put_theta_an = analytical_greeks(S, K, T, r, sigma, "put")
    call_delta_fd, call_gamma_fd, call_vega_fd, call_theta_fd = finite_diff_greeks(S, K, T, r, sigma, "call")
    put_delta_fd, put_gamma_fd, put_vega_fd, put_theta_fd = finite_diff_greeks(S, K, T, r, sigma, "put")
    theta_gap = -r * K * math.exp(-r * T)
    assert call_delta_an - put_delta_an == pytest.approx(1, 1e-12)
    assert call_delta_fd - put_delta_fd == pytest.approx(1, 1e-9)
    assert call_gamma_an - put_gamma_an == pytest.approx(0, 1e-12)
    assert call_gamma_fd - put_gamma_fd == pytest.approx(0, 1e-8)
    assert call_vega_an - put_vega_an == pytest.approx(0, 1e-12)
    assert call_vega_fd - put_vega_fd == pytest.approx(0, 1e-8)
    assert call_theta_an - put_theta_an == pytest.approx(theta_gap, 1e-9)
    assert call_theta_fd - put_theta_fd == pytest.approx(theta_gap, 1e-4)

@pytest.mark.parametrize("S, option_type", [(100, "call"), (100, "put"), (60, "put")])
def test_ranges(S, option_type):
    K, T, r, sigma = 100, 1.0, 0.05, 0.2
    delta_an, gamma_an, vega_an, theta_an = analytical_greeks(S, K, T, r, sigma, option_type)
    delta_fd, gamma_fd, vega_fd, theta_fd = finite_diff_greeks(S, K, T, r, sigma, option_type)
    if option_type == "call":
        assert 0 < delta_an < 1
        assert 0 < delta_fd < 1
        assert theta_an < 0 
        assert theta_fd < 0 
    else: 
        assert -1 < delta_an < 0 
        assert -1 < delta_fd < 0
        if S == 60: 
            assert theta_an > 0 
            assert theta_fd > 0  
    assert gamma_an > 0 
    assert gamma_fd > 0 
    assert vega_an > 0 
    assert vega_fd > 0 

@pytest.mark.parametrize("sigma0, sigma1, T0, T1, option_type", [(0.2, 0.21, 1.0, 1.0, "call"), (0.2, 0.21, 1.0, 1.0, "put"), (0.2, 0.2, 1.0, 0.99726027397, "call"), (0.2, 0.2, 1.0, 0.99726027397, "put")])
def test_units(sigma0, sigma1, T0, T1, option_type):
    S, K, r = 100, 100, 0.05 
    if T0 == T1:
        _, _, vega_an0, _ = analytical_greeks(S, K, T0, r, sigma0, option_type)
        _, _, vega_fd0, _ = finite_diff_greeks(S, K, T0, r, sigma0, option_type)
        _, _, vega_an1, _ = analytical_greeks(S, K, T0, r, sigma1, option_type)
        _, _, vega_fd1, _ = finite_diff_greeks(S, K, T0, r, sigma1, option_type)   
        vega_change_an = ((vega_an0 + vega_an1) / 2) * 0.01
        vega_change_fd = ((vega_fd0 + vega_fd1) / 2) * 0.01
        price0, _, _ = black_scholes(S, K, T0, r, sigma0, option_type)
        price1, _, _ = black_scholes(S, K, T0, r, sigma1, option_type)
        assert price1 - price0 == pytest.approx(vega_change_an, rel=0.01)
        assert price1 - price0 == pytest.approx(vega_change_fd, rel=0.01)
    else: 
        _, _, _, theta_an0 = analytical_greeks(S, K, T0, r, sigma0, option_type)
        _, _, _, theta_fd0 = finite_diff_greeks(S, K, T0, r, sigma0, option_type)
        _, _, _, theta_an1 = analytical_greeks(S, K, T1, r, sigma0, option_type)
        _, _, _, theta_fd1 = finite_diff_greeks(S, K, T1, r, sigma0, option_type)  
        theta_change_an = ((theta_an0 + theta_an1) / 2) / 365
        theta_change_fd = ((theta_fd0 + theta_fd1) / 2) / 365
        price0, _, _ = black_scholes(S, K, T0, r, sigma0, option_type)
        price1, _, _ = black_scholes(S, K, T1, r, sigma0, option_type)
        assert price1 - price0 == pytest.approx(theta_change_an, rel=0.01)
        assert price1 - price0 == pytest.approx(theta_change_fd, rel=0.01)

@pytest.mark.parametrize("S, K, T, r, sigma, option_type, expected", [(100, 100, 1.0, 0.05, 0.2, "call", (0.6368306511756191, 0.018762017345846895, 37.52403469169379, -6.414027546438197)), (90, 100, 1.0, 0.05, 0.2, "call", (0.42983173188954316, 0.021819747579660095, 35.34799107904935, -5.214480797467002)), (100, 90, 1.0, 0.05, 0.2, "call", (0.8097030607754923, 0.013581289746314732, 27.16257949262946, -5.929800832719607)), (100, 100, 1.0, 0.03, 0.2, "call", (0.5987063256829237, 0.019333405840142464, 38.66681168028493, -5.380398043561674)), (100, 100, 1.0, 0.05, 0.1, "call", (0.7088403132116536, 0.03429438550193839, 34.29438550193839, -4.918672955714081)), (100, 100, 1.0, 0.05, 0.2, "put", (-0.3631693488243809, 0.018762017345846895, 37.52403469169379, -1.657880423934626)), (90, 100, 1.0, 0.05, 0.2, "put", (-0.5701682681104568, 0.021819747579660095, 35.34799107904935, -0.45833367496343147)), (100, 90, 1.0, 0.05, 0.2, "put", (-0.19029693922450774, 0.013581289746314732, 27.16257949262946, -1.649268422466394)), (100, 100, 1.0, 0.03, 0.2, "put", (-0.4012936743170763, 0.019333405840142464, 38.66681168028493, -2.4690614429161495)), (100, 100, 1.0, 0.05, 0.1, "put", (-0.29115968678834636, 0.03429438550193839, 34.29438550193839, -0.16252583321051062)) ])  
def test_changes(S, K, T, r, sigma, option_type, expected):
    assert analytical_greeks(S, K, T, r, sigma, option_type) == pytest.approx(expected, rel=0.0003)
    assert finite_diff_greeks(S, K, T, r, sigma, option_type) == pytest.approx(expected, rel=0.0003) 







