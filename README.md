# Black-Scholes-Merton Pricer

European option pricing engine for dividend-paying underlyings.

Merton (1973) extension for continuous dividend yield, analytical Greek set including
second-order cross-Greeks, Newton-Raphson implied volatility solver with Brent
fallback, finite-difference verification, P&L attribution, and spot-volatility
scenario analysis.

---

## Architecture

```
bs/
├── core.py         BSM formula, all Greeks, put-call parity, OptionGreeks container
├── implied_vol.py  Newton-Raphson IV solver with Brent fallback
├── scenarios.py    S×σ P&L matrix, P&L attribution, greek profiles, FD verification
└── reporting.py    Formatted console output

dashboard.html      Interactive browser dashboard
main.py             End-to-end pricing and risk analytics demo
tests/test_bs.py    Pytest suite for pricing, Greeks, IV and P&L checks
```

---

## Pricing Model

### Black-Scholes-Merton (1973)

```
C = S·e^{−qT}·Φ(d₁) − K·e^{−rT}·Φ(d₂)
P = K·e^{−rT}·Φ(−d₂) − S·e^{−qT}·Φ(−d₁)

d₁ = [ln(S/K) + (r − q + ½σ²)T] / (σ√T)
d₂ = d₁ − σ√T
F  = S·e^{(r−q)T}        (risk-neutral forward)
```

`q > 0` handles dividend-paying equities, FX options where the foreign rate
acts as dividend yield, and futures-style pricing when `q = r`.

### Greeks

| Greek   | Formula                                  | Units                                           |
| ------- | ---------------------------------------- | ----------------------------------------------- |
| Delta Δ | `±e^{−qT}·Φ(±d₁)`                        | price change per 1 unit spot move               |
| Gamma Γ | `e^{−qT}·ϕ(d₁)/(Sσ√T)`                   | delta change per 1 unit spot move               |
| Vega ν  | `S·e^{−qT}·ϕ(d₁)·√T`                     | raw, per unit σ; divide by 100 for 1 vol pt     |
| Theta Θ | see below                                | price decay per calendar day                    |
| Rho ρ   | `±K·T·e^{−rT}·Φ(±d₂)`                    | raw, per unit r; divide by 100 for 1% rate move |
| Vanna   | `−e^{−qT}·ϕ(d₁)·d₂/σ`                    | ∂Δ/∂σ                                           |
| Volga   | `ν·d₁·d₂/σ`                              | ∂ν/∂σ, using raw ν                              |
| Charm   | `e^{−qT}[φqΦ(φd₁) − ϕ(d₁)·∂d₁/∂T] / 365` | dΔ/dt per calendar day                          |
| Speed   | `−Γ/S·(d₁/(σ√T)+1)`                      | ∂Γ/∂S                                           |

where `φ = +1` for calls and `φ = −1` for puts.

Theta:

```
Θ = −S·e^{−qT}·ϕ(d₁)·σ/(2√T)
    − φ·r·K·e^{−rT}·Φ(φd₂)
    + φ·q·S·e^{−qT}·Φ(φd₁)
```

Charm uses:

```
∂d₁/∂T = [2(r−q)T − d₂σ√T] / [2σT^(3/2)]
Charm = e^{−qT} · [φqΦ(φd₁) − ϕ(d₁)·∂d₁/∂T] / 365
```

---

## Implied Vol Solver

Primary method: Newton-Raphson using the analytical vega derivative.

Seed: Brenner-Subrahmanyam ATM approximation, with a Manaster-Koehler-style
refinement for non-ATM cases.

Fallback: Brent's method on `[1e-6, 20.0]`.

Convergence settings:

```
Maximum Newton iterations: 100
Volatility tolerance:      1e-10
Price tolerance:           1e-10
Vega floor:                1e-12
```

The solver validates European no-arbitrage bounds under continuous dividend
yield:

```
Call lower bound = max(S·e^{−qT} − K·e^{−rT}, 0)
Put lower bound  = max(K·e^{−rT} − S·e^{−qT}, 0)

Call upper bound = S·e^{−qT}
Put upper bound  = K·e^{−rT}
```

Round-trip tests recover the input volatility for standard call and put cases.

---

## P&L Scenario Matrix

The scenario matrix reprices the option across spot and volatility shocks.

For each scenario:

```
S_scenario = S · (1 + spot shock)
σ_scenario = max(1e−6, σ + vol shock)

P&L = V(S_scenario, σ_scenario) − V(S, σ)
```

Volatility shocks are absolute volatility-point changes. For example, if the
base volatility is `18%`, a `+10 vol pts` shock reprices the option at `28%`
volatility, not at `19.8%`.

---

## P&L Attribution

First and second-order decomposition:

```
ΔV ≈ Δ·ΔS  +  ½Γ·ΔS²  +  ν·Δσ  +  Θ·Δt  +  residual
```

where:

```
ΔS     = S₁ − S₀
Δσ     = σ₁ − σ₀
Δt     = number of calendar days
ν      = raw vega, per unit σ
Θ      = theta per calendar day
```

The residual captures the part of the full repricing move not explained by the
delta-gamma-vega-theta approximation.

---

## Results (SPX 460 Call, 45 DTE, Notional = 100)

```
════════════════════════════════════════════════════════════════════════
  BLACK-SCHOLES PRICER  SPX 460 Call  45 DTE
  CALL | S=450.0000  K=460.0000  T=0.123288Y  σ=18.00%  r=5.30%  q=1.30%
════════════════════════════════════════════════════════════════════════

  PRICING
  ──────────────────────────────────────────────────────────────────────
  Price                         797.418130
  Intrinsic                       0.000000
  Time Value                    797.418130
  Forward F                     452.224659
  d₁                             -0.238127
  d₂                             -0.301329
  ln(F/K)                        -0.017043

  GREEKS  (notional=100)
  ──────────────────────────────────────────────────────────────────────
  Delta  ∂V/∂S                   40.524137
  Gamma  ∂²V/∂S²                  1.361307
  Vega   ∂V/∂σ (raw)           6117.488329
  Vega   per 1 vol pt            61.174883
  Theta  dV/dt (per day)        -14.117638
  Rho    ∂V/∂r (raw)           2149.945113
  Rho    per 1% rate move        21.499451

  SECOND-ORDER GREEKS
  Vanna  ∂Δ/∂σ                   64.814056
  Volga  ∂²V/∂σ²               2438.651494
  Charm  dΔ/dt                   -0.195318
  Speed  ∂Γ/∂S                    0.008373
```

**P&L attribution: +5% spot, −2 vol pts, 1-day hold**

```
  Delta P&L        +911.7931
  Gamma P&L        +344.5807
  Vega P&L         -122.3498
  Theta P&L         -14.1176
  Residual           +8.6891
  ──────────────────────────
  Total P&L       +1128.5955
```

---

## Dashboard

The project includes an interactive HTML dashboard for visualising:

```
Option price and payoff vs spot
Delta and gamma vs spot
Vega and theta vs spot
Price vs implied volatility
Spot-volatility P&L matrix
```

The dashboard also displays put-call parity residuals and implied volatility
round-trip checks.

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

### Library usage

```python
from bs import OptionType, compute_all, implied_vol, scenario_matrix, pnl_attribution

# Price + full Greek set
g = compute_all(
    S=450,
    K=460,
    T=45/365,
    r=0.053,
    q=0.013,
    sigma=0.18,
    option_type=OptionType.CALL,
)

print(f"Price: {g.price:.4f}")
print(f"Delta: {g.delta:.4f}")
print(f"Gamma: {g.gamma:.6f}")
print(f"Vega raw: {g.vega:.4f}")
print(f"Vega per 1 vol pt: {g.vega / 100:.4f}")

# Implied volatility
iv = implied_vol(
    market_price=7.97,
    S=450,
    K=460,
    T=45/365,
    r=0.053,
    q=0.013,
    option_type=OptionType.CALL,
)

# Scenario matrix
mat = scenario_matrix(
    S=450,
    K=460,
    T=45/365,
    r=0.053,
    q=0.013,
    sigma=0.18,
    option_type=OptionType.CALL,
    notional=100,
)

# P&L attribution
attr = pnl_attribution(
    S0=450,
    S1=450 * 1.05,
    sigma0=0.18,
    sigma1=0.16,
    dt_days=1,
    K=460,
    T=45/365,
    r=0.053,
    q=0.013,
    option_type=OptionType.CALL,
    notional=100,
)
```

---

## References

1. Black & Scholes (1973). *The pricing of options and corporate liabilities.* JPE 81(3).
2. Merton (1973). *Theory of rational option pricing.* Bell Journal of Economics 4(1).
3. Brenner & Subrahmanyam (1988). *A simple formula to compute the implied standard deviation.* Financial Analysts Journal.
4. Manaster & Koehler (1982). *The calculation of implied variances from the Black-Scholes model.* Journal of Finance.
5. Jaeckel (2015). *Let's Be Rational.*
