"""pytest tests/test_bs.py"""

from __future__ import annotations
import math
import pytest
from bs import (
    OptionType, price, delta, gamma, vega, theta, rho,
    vanna, volga, charm, implied_vol, put_call_parity_check,
    compute_all, fd_greeks, pnl_attribution,
)

CALL = OptionType.CALL
PUT  = OptionType.PUT
S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.20


class TestKnownValues:
    def test_call_atm_haug(self):
        # Haug Table 1-1: S=60, K=65, T=0.25, r=0.08, q=0, σ=0.30 → C≈2.1334
        c = price(60, 65, 0.25, 0.08, 0.0, 0.30, CALL)
        assert abs(c - 2.1334) < 0.001

    def test_call_itm(self):
        # S=100, K=90, T=0.5, r=0.06, q=0, σ=0.25 → verified numerically
        c = price(100, 90, 0.5, 0.06, 0.0, 0.25, CALL)
        assert abs(c - 14.7639) < 0.001

    def test_expiry_call_intrinsic(self):
        assert abs(price(110, 100, 0.0, 0.05, 0.0, 0.20, CALL) - 10.0) < 1e-10

    def test_expiry_put_otm_zero(self):
        assert abs(price(110, 100, 0.0, 0.05, 0.0, 0.20, PUT)) < 1e-10


class TestPutCallParity:
    @pytest.mark.parametrize("spot,strike,texp,rate,div,vol", [
        (100, 100, 1.0, 0.05, 0.02, 0.20),
        (80,  100, 0.5, 0.03, 0.00, 0.30),
        (150, 100, 2.0, 0.06, 0.04, 0.15),
    ])
    def test_parity(self, spot, strike, texp, rate, div, vol):
        assert abs(put_call_parity_check(spot, strike, texp, rate, div, vol)) < 1e-10


class TestGreeksVsFD:
    """All analytics vs central finite differences."""

    def _fd(self):
        return fd_greeks(S, K, T, r, q, sigma, CALL)

    def test_delta_call(self):
        assert abs(delta(S, K, T, r, q, sigma, CALL) - self._fd()["delta_fd"]) < 1e-6

    def test_delta_put(self):
        fd_p = fd_greeks(S, K, T, r, q, sigma, PUT)
        assert abs(delta(S, K, T, r, q, sigma, PUT) - fd_p["delta_fd"]) < 1e-6

    def test_gamma(self):
        assert abs(gamma(S, K, T, r, q, sigma) - self._fd()["gamma_fd"]) < 1e-6

    def test_vega_raw(self):
        # Both analytic and FD are raw (per unit sigma decimal)
        assert abs(vega(S, K, T, r, q, sigma) - self._fd()["vega_fd"]) < 1e-5

    def test_theta_per_day(self):
        assert abs(theta(S, K, T, r, q, sigma, CALL) - self._fd()["theta_fd"]) < 1e-6

    def test_rho_raw(self):
        # Both raw (per unit r decimal)
        assert abs(rho(S, K, T, r, q, sigma, CALL) - self._fd()["rho_fd"]) < 1e-4

    def test_vanna_call(self):
        h = 1e-4
        van_fd = (delta(S, K, T, r, q, sigma+h, CALL) -
                  delta(S, K, T, r, q, sigma-h, CALL)) / (2*h)
        assert abs(vanna(S, K, T, r, q, sigma) - van_fd) < 1e-5

    def test_volga(self):
        h = 1e-4
        vlg_fd = (vega(S, K, T, r, q, sigma+h) -
                  vega(S, K, T, r, q, sigma-h)) / (2*h)
        assert abs(volga(S, K, T, r, q, sigma) - vlg_fd) < 1e-3

    def test_charm_call(self):
        h = 1e-5
        # charm = -dDelta/dT = dDelta/dt
        chm_fd = -(delta(S, K, T+h, r, q, sigma, CALL) -
                   delta(S, K, T-h, r, q, sigma, CALL)) / (2*h)
        # analytic is per day; multiply by 365 to get per year
        assert abs(charm(S, K, T, r, q, sigma, CALL)*365 - chm_fd) < 1e-6

    def test_charm_put(self):
        h = 1e-5
        chm_fd = -(delta(S, K, T+h, r, q, sigma, PUT) -
                   delta(S, K, T-h, r, q, sigma, PUT)) / (2*h)
        assert abs(charm(S, K, T, r, q, sigma, PUT)*365 - chm_fd) < 1e-6


class TestGreekProperties:
    def test_call_delta_in_01(self):
        assert 0 < delta(S, K, T, r, q, sigma, CALL) < 1

    def test_put_delta_in_m1_0(self):
        assert -1 < delta(S, K, T, r, q, sigma, PUT) < 0

    def test_put_call_delta_relation(self):
        # Δ_call − Δ_put = e^{−qT}
        dc = delta(S, K, T, r, q, sigma, CALL)
        dp = delta(S, K, T, r, q, sigma, PUT)
        assert abs(dc - dp - math.exp(-q * T)) < 1e-10

    def test_gamma_positive(self):
        assert gamma(S, K, T, r, q, sigma) > 0

    def test_vega_positive(self):
        assert vega(S, K, T, r, q, sigma) > 0

    def test_theta_call_sign(self):
        # For most calls, theta is negative (time decay)
        assert theta(S, K, T, r, q, sigma, CALL) < 0


class TestImpliedVol:
    @pytest.mark.parametrize("sig_true", [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80])
    def test_round_trip_call(self, sig_true):
        mkt = price(S, K, T, r, q, sig_true, CALL)
        iv  = implied_vol(mkt, S, K, T, r, q, CALL)
        assert abs(iv - sig_true) < 1e-8

    @pytest.mark.parametrize("sig_true", [0.10, 0.25, 0.40])
    def test_round_trip_put(self, sig_true):
        mkt = price(S, K, T, r, q, sig_true, PUT)
        iv  = implied_vol(mkt, S, K, T, r, q, PUT)
        assert abs(iv - sig_true) < 1e-8

    def test_deep_otm_call(self):
        mkt = price(50, 100, 1.0, 0.05, 0.0, 0.30, CALL)
        assert abs(implied_vol(mkt, 50, 100, 1.0, 0.05, 0.0, CALL) - 0.30) < 1e-6

    def test_below_intrinsic_raises(self):
        with pytest.raises(ValueError):
            implied_vol(max(S-K, 0) - 1.0, S, K, T, r, q, CALL)


class TestPnLAttribution:
    def test_flat_zero_pnl(self):
        attr = pnl_attribution(S, S, sigma, sigma, 0.0, K, T, r, q, CALL)
        assert abs(attr.total_pnl) < 1e-10

    def test_components_sum_to_total(self):
        attr = pnl_attribution(S, S*1.02, sigma, sigma*0.98, 1.0, K, T, r, q, CALL)
        approx = (attr.delta_pnl + attr.gamma_pnl + attr.vega_pnl
                  + attr.theta_pnl + attr.residual)
        assert abs(approx - attr.total_pnl) < 1e-10
