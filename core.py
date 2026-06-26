"""
Black-Scholes-Merton pricing engine.

European options on a dividend-paying underlying (Merton 1973).
All rates and vol are continuously compounded / annualised decimals.
Time T is in years.

Notation
--------
S   : spot price
K   : strike
T   : time to expiry (years)
r   : continuously-compounded risk-free rate
q   : continuously-compounded dividend yield (0 for non-div paying)
σ   : annualised implied / model volatility
φ   : +1 for call, −1 for put
F   : risk-neutral forward  F = S · e^{(r−q)T}
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from scipy.stats import norm

_N   = norm.cdf
_n   = norm.pdf
_EPS = 1e-12


class OptionType(Enum):
    CALL = "call"
    PUT  = "put"

    @property
    def phi(self) -> int:
        return 1 if self == OptionType.CALL else -1


# ---------------------------------------------------------------------------
# d1 / d2 / forward
# ---------------------------------------------------------------------------

def d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """d₁ = [ln(S/K) + (r − q + ½σ²)T] / (σ√T)"""
    if T <= _EPS or sigma <= _EPS:
        return math.inf if S * math.exp((r - q) * T) >= K else -math.inf
    return (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """d₂ = d₁ − σ√T"""
    return d1(S, K, T, r, q, sigma) - sigma * math.sqrt(T)


def forward(S: float, T: float, r: float, q: float) -> float:
    return S * math.exp((r - q) * T)


def moneyness(S: float, K: float, T: float, r: float, q: float) -> float:
    """Log-moneyness: ln(F/K)."""
    return math.log(forward(S, T, r, q) / K)


# ---------------------------------------------------------------------------
# Price
# ---------------------------------------------------------------------------

def price(S: float, K: float, T: float, r: float, q: float,
          sigma: float, option_type: OptionType) -> float:
    """
    BSM European option price.

        C = S·e^{−qT}·Φ(d₁) − K·e^{−rT}·Φ(d₂)
        P = K·e^{−rT}·Φ(−d₂) − S·e^{−qT}·Φ(−d₁)
    """
    if T <= _EPS:
        return max(option_type.phi * (S - K), 0.0)
    phi  = option_type.phi
    _d1  = d1(S, K, T, r, q, sigma)
    _d2  = d2(S, K, T, r, q, sigma)
    return phi * (S * math.exp(-q * T) * _N(phi * _d1)
                  - K * math.exp(-r * T) * _N(phi * _d2))


def intrinsic(S: float, K: float, option_type: OptionType) -> float:
    return max(option_type.phi * (S - K), 0.0)


def time_value(S: float, K: float, T: float, r: float, q: float,
               sigma: float, option_type: OptionType) -> float:
    return price(S, K, T, r, q, sigma, option_type) - intrinsic(S, K, option_type)


# ---------------------------------------------------------------------------
# First-order Greeks
# ---------------------------------------------------------------------------

def delta(S: float, K: float, T: float, r: float, q: float,
          sigma: float, option_type: OptionType) -> float:
    """
    ∂V/∂S.

        Δ_call = e^{−qT} · Φ(d₁)
        Δ_put  = −e^{−qT} · Φ(−d₁)
    """
    if T <= _EPS:
        return option_type.phi * (1.0 if option_type.phi * (S - K) > 0 else 0.0)
    phi = option_type.phi
    return phi * math.exp(-q * T) * _N(phi * d1(S, K, T, r, q, sigma))


def gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    ∂²V/∂S².  Identical for call and put.

        Γ = e^{−qT} · φ(d₁) / (S · σ · √T)
    """
    if T <= _EPS or sigma <= _EPS:
        return 0.0
    return math.exp(-q * T) * _n(d1(S, K, T, r, q, sigma)) / (S * sigma * math.sqrt(T))


def vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    ∂V/∂σ.  Identical for call and put.

        ν = S · e^{−qT} · φ(d₁) · √T

    Returned as raw value (per unit σ decimal).
    For $ per 1 vol-point (1% move), divide by 100.
    """
    if T <= _EPS:
        return 0.0
    return S * math.exp(-q * T) * _n(d1(S, K, T, r, q, sigma)) * math.sqrt(T)


def theta(S: float, K: float, T: float, r: float, q: float,
          sigma: float, option_type: OptionType, per_day: bool = True) -> float:
    """
    dV/dt = −dV/dT  (time decay; negative = option loses value as time passes).

        Θ_call = −S·e^{−qT}·φ(d₁)·σ/(2√T) − r·K·e^{−rT}·Φ(d₂)  + q·S·e^{−qT}·Φ(d₁)
        Θ_put  = −S·e^{−qT}·φ(d₁)·σ/(2√T) + r·K·e^{−rT}·Φ(−d₂) − q·S·e^{−qT}·Φ(−d₁)

    Unified via φ: Θ = −S·e^{−qT}·φ(d₁)·σ/(2√T)
                       − φ·r·K·e^{−rT}·Φ(φ·d₂)
                       + φ·q·S·e^{−qT}·Φ(φ·d₁)

    per_day=True divides by 365 to give $ decay per calendar day.
    """
    if T <= _EPS:
        return 0.0
    phi  = option_type.phi
    _d1  = d1(S, K, T, r, q, sigma)
    _d2  = d2(S, K, T, r, q, sigma)
    th = (- S * math.exp(-q * T) * _n(_d1) * sigma / (2.0 * math.sqrt(T))
          - phi * r * K * math.exp(-r * T) * _N(phi * _d2)
          + phi * q * S * math.exp(-q * T) * _N(phi * _d1))
    return th / 365.0 if per_day else th


def rho(S: float, K: float, T: float, r: float, q: float,
        sigma: float, option_type: OptionType) -> float:
    """
    ∂V/∂r  (raw, per unit r decimal).
    Divide by 100 for $ per 1% rate move.

        ρ_call = K · T · e^{−rT} · Φ(d₂)
        ρ_put  = −K · T · e^{−rT} · Φ(−d₂)
    """
    if T <= _EPS:
        return 0.0
    phi = option_type.phi
    return phi * K * T * math.exp(-r * T) * _N(phi * d2(S, K, T, r, q, sigma))


# ---------------------------------------------------------------------------
# Second-order / cross Greeks
# ---------------------------------------------------------------------------

def vanna(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    ∂²V / (∂S ∂σ) = ∂Δ/∂σ = ∂ν/∂S.  Identical for call and put.

        Vanna = −e^{−qT} · φ(d₁) · d₂ / σ

    Derivation: Δ = φ·e^{−qT}·Φ(φ·d₁)
    ∂Δ/∂σ = φ·e^{−qT}·φ(d₁)·∂d₁/∂σ  (φ² = 1, cancels)
           = e^{−qT}·φ(d₁)·∂d₁/∂σ

    ∂d₁/∂σ = −d₂/σ  (standard result: differentiate d₁ = ln(S/K)/(σ√T) + (r−q)T/(σ√T) + ½σ√T)
    → Vanna = −e^{−qT}·φ(d₁)·d₂/σ
    """
    if T <= _EPS or sigma <= _EPS:
        return 0.0
    _d1 = d1(S, K, T, r, q, sigma)
    _d2 = d2(S, K, T, r, q, sigma)
    return -math.exp(-q * T) * _n(_d1) * _d2 / sigma


def volga(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    ∂²V/∂σ² = ∂ν/∂σ.  Also called Vomma.  Identical for call and put.

        Volga = ν · d₁ · d₂ / σ

    where ν = S·e^{−qT}·φ(d₁)·√T  (raw vega, per unit σ decimal).

    Derivation: ν = S·e^{−qT}·φ(d₁)·√T
    ∂ν/∂σ = S·e^{−qT}·√T · φ'(d₁) · ∂d₁/∂σ
           = S·e^{−qT}·√T · (−d₁·φ(d₁)) · (−d₂/σ)
           = ν · d₁·d₂/σ
    """
    if T <= _EPS or sigma <= _EPS:
        return 0.0
    _d1 = d1(S, K, T, r, q, sigma)
    _d2 = d2(S, K, T, r, q, sigma)
    v   = vega(S, K, T, r, q, sigma)
    return v * _d1 * _d2 / sigma


def charm(S: float, K: float, T: float, r: float, q: float,
          sigma: float, option_type: OptionType) -> float:
    """
    ∂²V / (∂S ∂t) = dΔ/dt.  Delta decay as calendar time passes.

    Derivation (call, φ=+1):
        Δ_call = e^{−qT}·Φ(d₁)
        dΔ/dT  = −q·e^{−qT}·Φ(d₁) + e^{−qT}·φ(d₁)·∂d₁/∂T

    ∂d₁/∂T = [2(r−q)T − d₂·σ·√T] / (2σT^{3/2})  (derived by differentiating d₁)

    Since dT = −dt:
        Charm_call = −dΔ/dT = q·e^{−qT}·Φ(d₁)  − e^{−qT}·φ(d₁)·∂d₁/∂T

    For put (φ=−1):
        Δ_put = −e^{−qT}·Φ(−d₁)
        dΔ_put/dT = q·e^{−qT}·Φ(−d₁) − e^{−qT}·φ(d₁)·∂d₁/∂T  [φ(−d₁)=φ(d₁)]
        Charm_put = −dΔ_put/dT = −q·e^{−qT}·Φ(−d₁) + e^{−qT}·φ(d₁)·∂d₁/∂T

    Unified:
        Charm = e^{−qT} · [ φ·q·Φ(φ·d₁) − φ(d₁)·∂d₁/∂T ]

    Divided by 365 to give per-calendar-day delta decay.
    """
    if T <= _EPS or sigma <= _EPS:
        return 0.0
    phi   = option_type.phi
    _d1   = d1(S, K, T, r, q, sigma)
    _d2   = d2(S, K, T, r, q, sigma)
    sqrtT = math.sqrt(T)

    dd1_dT = (2.0 * (r - q) * T - _d2 * sigma * sqrtT) / (2.0 * sigma * T ** 1.5)
    c = math.exp(-q * T) * (phi * q * _N(phi * _d1) - _n(_d1) * dd1_dT)
    return c / 365.0


def speed(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    ∂³V/∂S³ = ∂Γ/∂S.

        Speed = −(Γ/S) · (d₁/(σ√T) + 1)
    """
    if T <= _EPS or sigma <= _EPS:
        return 0.0
    _d1 = d1(S, K, T, r, q, sigma)
    g   = gamma(S, K, T, r, q, sigma)
    return -g / S * (_d1 / (sigma * math.sqrt(T)) + 1)


# ---------------------------------------------------------------------------
# Put-call parity
# ---------------------------------------------------------------------------

def put_call_parity_check(S: float, K: float, T: float,
                          r: float, q: float, sigma: float) -> float:
    """C − P = S·e^{−qT} − K·e^{−rT}.  Returns residual (should be ~0)."""
    C = price(S, K, T, r, q, sigma, OptionType.CALL)
    P = price(S, K, T, r, q, sigma, OptionType.PUT)
    return (C - P) - (S * math.exp(-q * T) - K * math.exp(-r * T))


# ---------------------------------------------------------------------------
# Convenience container
# ---------------------------------------------------------------------------

@dataclass
class OptionGreeks:
    price:          float
    delta:          float
    gamma:          float
    vega:           float   # raw, per unit σ decimal. For per-1%-vol-pt: divide by 100
    theta:          float   # per calendar day
    rho:            float   # raw, per unit r decimal. For per-1%-rate-pt: divide by 100
    vanna:          float   # ∂Δ/∂σ  (same units as delta, per unit σ)
    volga:          float   # ∂²V/∂σ² (per unit σ²)
    charm:          float   # dΔ/dt per calendar day
    speed:          float   # ∂Γ/∂S
    intrinsic_value: float
    time_value:     float
    d1:             float
    d2:             float
    forward:        float
    moneyness_log:  float   # ln(F/K)


def compute_all(S: float, K: float, T: float, r: float, q: float,
                sigma: float, option_type: OptionType) -> OptionGreeks:
    """Compute price + full Greek set."""
    _d1 = d1(S, K, T, r, q, sigma)
    _d2 = d2(S, K, T, r, q, sigma)
    return OptionGreeks(
        price           = price(S, K, T, r, q, sigma, option_type),
        delta           = delta(S, K, T, r, q, sigma, option_type),
        gamma           = gamma(S, K, T, r, q, sigma),
        vega            = vega(S, K, T, r, q, sigma),
        theta           = theta(S, K, T, r, q, sigma, option_type, per_day=True),
        rho             = rho(S, K, T, r, q, sigma, option_type),
        vanna           = vanna(S, K, T, r, q, sigma),
        volga           = volga(S, K, T, r, q, sigma),
        charm           = charm(S, K, T, r, q, sigma, option_type),
        speed           = speed(S, K, T, r, q, sigma),
        intrinsic_value = intrinsic(S, K, option_type),
        time_value      = time_value(S, K, T, r, q, sigma, option_type),
        d1              = _d1,
        d2              = _d2,
        forward         = forward(S, T, r, q),
        moneyness_log   = moneyness(S, K, T, r, q),
    )
