"""Scenario analysis, P&L attribution, Greek profiles, FD verification."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from .core import price, delta, gamma, vega, theta, OptionType, OptionGreeks, compute_all


@dataclass
class ScenarioMatrix:
    spot_shocks: List[float]
    vol_shocks:  List[float]
    prices:      np.ndarray
    pnls:        np.ndarray
    base_price:  float


def scenario_matrix(S, K, T, r, q, sigma, option_type, notional=1.0,
                    spot_shocks=None, vol_shocks=None) -> ScenarioMatrix:
    if spot_shocks is None:
        spot_shocks = [-0.30, -0.20, -0.15, -0.10, -0.05, 0.0,
                        0.05,  0.10,  0.15,  0.20,  0.30]
    if vol_shocks is None:
        vol_shocks = [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]
    base = price(S, K, T, r, q, sigma, option_type) * notional
    ns, nv = len(spot_shocks), len(vol_shocks)
    prices = np.zeros((ns, nv))
    pnls   = np.zeros((ns, nv))
    for i, ds in enumerate(spot_shocks):
        for j, dv in enumerate(vol_shocks):
            pv = price(S*(1+ds), K, T, r, q, max(1e-6, sigma+dv), option_type) * notional
            prices[i, j] = pv
            pnls[i, j]   = pv - base
    return ScenarioMatrix(spot_shocks, vol_shocks, prices, pnls, base)


@dataclass
class PnLAttribution:
    """ΔV ≈ Δ·ΔS + ½Γ·ΔS² + ν·Δσ + Θ·Δt + residual"""
    total_pnl:  float
    delta_pnl:  float
    gamma_pnl:  float
    vega_pnl:   float
    theta_pnl:  float
    residual:   float


def pnl_attribution(S0, S1, sigma0, sigma1, dt_days, K, T, r, q,
                    option_type, notional=1.0) -> PnLAttribution:
    g = compute_all(S0, K, T, r, q, sigma0, option_type)
    dS = S1 - S0
    dsigma = sigma1 - sigma0
    dt_years = dt_days / 365.0

    d_pnl  = g.delta * dS * notional
    g_pnl  = 0.5 * g.gamma * dS**2 * notional
    v_pnl  = g.vega * dsigma * notional           # vega raw * dsigma decimal
    th_pnl = g.theta * dt_days * notional          # theta per day * days

    T1    = max(T - dt_years, 0)
    pv0   = g.price * notional
    pv1   = price(S1, K, T1, r, q, sigma1, option_type) * notional
    total = pv1 - pv0
    residual = total - d_pnl - g_pnl - v_pnl - th_pnl

    return PnLAttribution(total, d_pnl, g_pnl, v_pnl, th_pnl, residual)


@dataclass
class GreekProfile:
    spots: np.ndarray
    deltas: np.ndarray
    gammas: np.ndarray
    vegas:  np.ndarray
    thetas: np.ndarray


def greek_profile(S, K, T, r, q, sigma, option_type,
                  spot_range=None, n_points=100) -> GreekProfile:
    if spot_range is None:
        spot_range = (S * 0.5, S * 1.5)
    spots  = np.linspace(spot_range[0], spot_range[1], n_points)
    d = np.array([delta(s, K, T, r, q, sigma, option_type) for s in spots])
    g = np.array([gamma(s, K, T, r, q, sigma) for s in spots])
    v = np.array([vega(s,  K, T, r, q, sigma) for s in spots])
    th = np.array([theta(s, K, T, r, q, sigma, option_type) for s in spots])
    return GreekProfile(spots, d, g, v, th)


def fd_greeks(S, K, T, r, q, sigma, option_type,
              h_S=None, h_sigma=1e-4, h_T=1e-5, h_r=1e-4) -> dict:
    """Central finite-difference Greeks for verification."""
    h_S = h_S or S * 1e-4
    def pv(s=S, k=K, t=T, ri=r, qi=q, sig=sigma):
        return price(s, k, t, ri, qi, sig, option_type)

    d_fd   = (pv(s=S+h_S)    - pv(s=S-h_S))    / (2*h_S)
    g_fd   = (pv(s=S+h_S)    - 2*pv() + pv(s=S-h_S)) / h_S**2
    v_fd   = (pv(sig=sigma+h_sigma) - pv(sig=sigma-h_sigma)) / (2*h_sigma)  # raw, per unit sigma
    th_fd  = -(pv(t=T+h_T)   - pv(t=T-h_T))    / (2*h_T) / 365.0  # per day
    rh_fd  = (pv(ri=r+h_r)   - pv(ri=r-h_r))   / (2*h_r)   # raw, per unit r

    return {"delta_fd": d_fd, "gamma_fd": g_fd, "vega_fd": v_fd,
            "theta_fd": th_fd, "rho_fd": rh_fd}
