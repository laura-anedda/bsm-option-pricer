from __future__ import annotations
from .core import OptionGreeks, OptionType
from .scenarios import ScenarioMatrix, PnLAttribution

_W = 72

def print_option_sheet(S, K, T, r, q, sigma, option_type, g: OptionGreeks,
                       notional=1.0, label=""):
    tag = f"  {label}" if label else ""
    print(f"\n{'═'*_W}")
    print(f"  BLACK-SCHOLES PRICER{tag}")
    print(f"  {option_type.value.upper()} | S={S:.4f}  K={K:.4f}  T={T:.6f}Y  "
          f"σ={sigma*100:.2f}%  r={r*100:.2f}%  q={q*100:.2f}%")
    print(f"{'═'*_W}")

    N = notional
    print(f"\n  PRICING")
    print(f"  {'─'*(_W-2)}")
    print(f"  {'Price':25s}  {g.price * N:>14.6f}")
    print(f"  {'Intrinsic':25s}  {g.intrinsic_value * N:>14.6f}")
    print(f"  {'Time Value':25s}  {g.time_value * N:>14.6f}")
    print(f"  {'Forward F':25s}  {g.forward:>14.6f}")
    print(f"  {'d₁':25s}  {g.d1:>14.6f}")
    print(f"  {'d₂':25s}  {g.d2:>14.6f}")
    print(f"  {'ln(F/K)':25s}  {g.moneyness_log:>14.6f}")

    print(f"\n  GREEKS  (notional={N:g})")
    print(f"  {'─'*(_W-2)}")
    print(f"  {'Delta  ∂V/∂S':25s}  {g.delta * N:>14.6f}   [$ per $1 spot move]")
    print(f"  {'Gamma  ∂²V/∂S²':25s}  {g.gamma * N:>14.6f}   [$ per $1² spot move]")
    print(f"  {'Vega   ∂V/∂σ (raw)':25s}  {g.vega * N:>14.6f}   [$ per unit σ; /100 for per 1%]")
    print(f"  {'Vega   per 1 vol pt':25s}  {g.vega * N / 100.0:>14.6f}   [$ per 1% σ move]")
    print(f"  {'Theta  dV/dt (per day)':25s}  {g.theta * N:>14.6f}   [$ per calendar day]")
    print(f"  {'Rho    ∂V/∂r (raw)':25s}  {g.rho * N:>14.6f}   [$ per unit r; /100 for per 1%]")
    print(f"  {'Rho    per 1% rate move':25s}  {g.rho * N / 100.0:>14.6f}   [$ per 1% r move]")

    print(f"\n  SECOND-ORDER GREEKS")
    print(f"  {'Vanna  ∂Δ/∂σ':25s}  {g.vanna * N:>14.6f}   [per unit σ]")
    print(f"  {'Volga  ∂²V/∂σ²':25s}  {g.volga * N:>14.6f}   [per unit σ²]")
    print(f"  {'Charm  dΔ/dt':25s}  {g.charm * N:>14.6f}   [delta decay per day]")
    print(f"  {'Speed  ∂Γ/∂S':25s}  {g.speed * N:>14.6f}")
    print(f"\n{'═'*_W}\n")


def print_scenario_matrix(mat: ScenarioMatrix, label="P&L SCENARIO MATRIX"):
    print(f"\n  {label}  (base={mat.base_price:.4f})\n")
    hdr = f"  {'Spot \\ Vol':>10s}"
    for dv in mat.vol_shocks:
        # vol_shocks are absolute volatility-point changes: +0.10 = +10 vol pts
        hdr += f"  {dv*100:>+7.1f}vpt "
    print(hdr)
    print("  " + "─" * 70)
    for i, ds in enumerate(mat.spot_shocks):
        row = f"  {ds*100:>+9.1f}%S "
        for j in range(len(mat.vol_shocks)):
            pnl = mat.pnls[i, j]
            row += f"  {pnl:>+9.2f} "
        print(row)
    print()


def print_pnl_attribution(attr: PnLAttribution, label=""):
    tag = f" — {label}" if label else ""
    print(f"\n  P&L ATTRIBUTION{tag}")
    print(f"  {'─'*40}")
    print(f"  {'Delta P&L':20s}  {attr.delta_pnl:>+12.4f}")
    print(f"  {'Gamma P&L':20s}  {attr.gamma_pnl:>+12.4f}")
    print(f"  {'Vega P&L':20s}  {attr.vega_pnl:>+12.4f}")
    print(f"  {'Theta P&L':20s}  {attr.theta_pnl:>+12.4f}")
    print(f"  {'Residual':20s}  {attr.residual:>+12.4f}")
    print(f"  {'─'*36}")
    print(f"  {'Total P&L':20s}  {attr.total_pnl:>+12.4f}\n")
