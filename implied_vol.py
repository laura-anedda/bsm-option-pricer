"""
Implied volatility solver.

Primary method: Newton-Raphson using the analytic vega derivative.
Fallback: Brent's method on a bracketed interval.
Initial guess: Brenner-Subrahmanyam (1988) ATM approximation, refined
by Manaster-Koehler (1982) seed for OTM options.

Reference: Jaeckel (2015), "Let's be rational".
"""

from __future__ import annotations

import math
from typing import Optional

from scipy.optimize import brentq

from .core import price, vega, OptionType, d1 as _d1_fn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MAX_ITER_NEWTON   = 100
_NEWTON_TOL        = 1e-10    # vol tolerance
_VEGA_FLOOR        = 1e-12    # stop Newton if vega too small
_SIGMA_LO          = 1e-6
_SIGMA_HI          = 20.0     # 2000% vol upper bound
_PRICE_TOL         = 1e-10    # treat prices below this as zero


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: OptionType,
    initial_guess: Optional[float] = None,
) -> float:
    """
    Solve for σ such that BSM(σ) = market_price.

    Parameters
    ----------
    market_price : float
        Observed mid price (must be above intrinsic, below parity).
    initial_guess : float, optional
        Starting vol for Newton-Raphson.  If None, uses Brenner-Subrahmanyam.

    Returns
    -------
    float
        Implied volatility in decimal form (0.20 = 20%).

    Raises
    ------
    ValueError
        If price is outside no-arbitrage bounds or solver fails to converge.
    """
    _validate_price(market_price, S, K, T, r, q, option_type)

    sigma0 = initial_guess if initial_guess is not None else _initial_guess(
        market_price, S, K, T, r, q, option_type
    )

    # --- Newton-Raphson ---
    sigma = _newton(market_price, S, K, T, r, q, option_type, sigma0)
    if sigma is not None:
        return sigma

    # --- Brent fallback ---
    return _brent(market_price, S, K, T, r, q, option_type)


# ---------------------------------------------------------------------------
# Newton-Raphson
# ---------------------------------------------------------------------------

def _newton(
    target: float,
    S: float, K: float, T: float,
    r: float, q: float,
    option_type: OptionType,
    sigma0: float,
) -> Optional[float]:
    sigma = max(_SIGMA_LO, min(sigma0, _SIGMA_HI))

    for _ in range(_MAX_ITER_NEWTON):
        pv  = price(S, K, T, r, q, sigma, option_type)
        res = pv - target
        if abs(res) < _PRICE_TOL:
            return sigma

        v = vega(S, K, T, r, q, sigma)
        if v < _VEGA_FLOOR:
            return None   # degenerate; fall through to Brent

        step  = res / v
        sigma -= step

        sigma = max(_SIGMA_LO, min(sigma, _SIGMA_HI))

        if abs(step) < _NEWTON_TOL:
            return sigma

    return None


# ---------------------------------------------------------------------------
# Brent fallback
# ---------------------------------------------------------------------------

def _brent(
    target: float,
    S: float, K: float, T: float,
    r: float, q: float,
    option_type: OptionType,
) -> float:
    def f(sigma: float) -> float:
        return price(S, K, T, r, q, sigma, option_type) - target

    # Expand bracket if needed
    lo, hi = _SIGMA_LO, _SIGMA_HI
    if f(lo) * f(hi) > 0:
        raise ValueError(
            f"Implied vol solver: no bracket found for price={target:.6f}. "
            f"f({lo})={f(lo):.6f}, f({hi})={f(hi):.6f}"
        )
    return brentq(f, lo, hi, xtol=_NEWTON_TOL, maxiter=500)


# ---------------------------------------------------------------------------
# Initial guess: Brenner-Subrahmanyam ATM + Manaster-Koehler OTM refinement
# ---------------------------------------------------------------------------

def _initial_guess(
    market_price: float,
    S: float, K: float, T: float,
    r: float, q: float,
    option_type: OptionType,
) -> float:
    """
    Brenner & Subrahmanyam (1988):
        σ₀ ≈ √(2π/T) · C / S  (ATM call approximation)
    """
    F  = S * math.exp((r - q) * T)
    df = math.exp(-r * T)

    # ATM approximation: works well for near-the-money options
    atm_approx = math.sqrt(2 * math.pi / T) * market_price / (F * df) if T > 1e-9 else 0.3

    # Manaster-Koehler seed: iterate once using d1 inversion
    if abs(math.log(F / K)) < 2.0 and T > 1e-6:
        try:
            sigma_mk = abs(math.log(F / K)) / math.sqrt(T) * 1.5
            return max(_SIGMA_LO, min(max(atm_approx, sigma_mk), 5.0))
        except Exception:
            pass

    return max(_SIGMA_LO, min(atm_approx, 5.0)) if atm_approx > 0 else 0.3


# ---------------------------------------------------------------------------
# No-arbitrage validation
# ---------------------------------------------------------------------------

def _validate_price(
    market_price: float,
    S: float, K: float, T: float,
    r: float, q: float,
    option_type: OptionType,
) -> None:
    phi = option_type.phi
    dfq = math.exp(-q * T)
    dfr = math.exp(-r * T)

    # European no-arbitrage bounds under continuous dividend yield:
    #   call: max(S e^{-qT} - K e^{-rT}, 0) <= C <= S e^{-qT}
    #   put : max(K e^{-rT} - S e^{-qT}, 0) <= P <= K e^{-rT}
    # Do not use spot intrinsic max(phi*(S-K), 0) as the lower bound: a
    # European dividend-paying call can legitimately trade below spot intrinsic.
    lb = max(phi * (S * dfq - K * dfr), 0.0)
    ub = S * dfq if option_type == OptionType.CALL else K * dfr

    if market_price < lb - 1e-6:
        raise ValueError(
            f"Price {market_price:.6f} violates European lower bound ({lb:.6f})."
        )
    if market_price > ub + 1e-6:
        raise ValueError(
            f"Price {market_price:.6f} violates European upper bound ({ub:.6f})."
        )
