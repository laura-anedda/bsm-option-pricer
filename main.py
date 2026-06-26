"""BSM European option pricing demo."""

from __future__ import annotations
from datetime import date
from bs import (
    OptionType, compute_all, implied_vol,
    put_call_parity_check, scenario_matrix, pnl_attribution,
    print_option_sheet, print_scenario_matrix, print_pnl_attribution,
    fd_greeks,
)


def main():
    S, K, T = 450.00, 460.00, 45 / 365
    r, q, sigma = 0.0530, 0.013, 0.18
    N = 100

    # ── Call ──────────────────────────────────────────────────────
    call = compute_all(S, K, T, r, q, sigma, OptionType.CALL)
    print_option_sheet(S, K, T, r, q, sigma, OptionType.CALL, call,
                       notional=N, label="SPX 460 Call  45 DTE")

    # ── Put ───────────────────────────────────────────────────────
    put = compute_all(S, K, T, r, q, sigma, OptionType.PUT)
    print_option_sheet(S, K, T, r, q, sigma, OptionType.PUT, put,
                       notional=N, label="SPX 460 Put   45 DTE")

    # ── Put-call parity ───────────────────────────────────────────
    pcp = put_call_parity_check(S, K, T, r, q, sigma)
    print(f"  Put-call parity residual: {pcp:.2e}\n")

    # ── FD Greek verification ─────────────────────────────────────
    fd = fd_greeks(S, K, T, r, q, sigma, OptionType.CALL)
    print("  Greek verification (analytic vs finite-difference):")
    print(f"  {'Greek':10s}  {'Analytic':>14s}  {'FD':>14s}  {'Error':>12s}")
    print(f"  {'─'*56}")
    checks = [
        ("Delta",  call.delta,           fd["delta_fd"]),
        ("Gamma",  call.gamma,           fd["gamma_fd"]),
        ("Vega",   call.vega,            fd["vega_fd"]),    # both raw
        ("Theta",  call.theta,           fd["theta_fd"]),
        ("Rho",    call.rho,             fd["rho_fd"]),     # both raw
    ]
    for name, analytic, numerical in checks:
        err = analytic - numerical
        flag = "" if abs(err) < 1e-5 else "  ◄"
        print(f"  {name:10s}  {analytic:>14.8f}  {numerical:>14.8f}  {err:>+12.2e}{flag}")
    print()

    # ── IV round-trip ─────────────────────────────────────────────
    iv = implied_vol(call.price, S, K, T, r, q, OptionType.CALL)
    print(f"  IV round-trip:  σ_input={sigma*100:.4f}%  price={call.price:.6f}"
          f"  IV={iv*100:.6f}%  err={abs(iv-sigma)*1e6:.2e} µvol\n")

    # ── Scenario matrix ───────────────────────────────────────────
    mat = scenario_matrix(S, K, T, r, q, sigma, OptionType.CALL, notional=N)
    print_scenario_matrix(mat, label=f"CALL N={N}  base={mat.base_price:.2f}")

    # ── P&L attribution ───────────────────────────────────────────
    attr = pnl_attribution(
        S0=S, S1=S*1.05, sigma0=sigma, sigma1=sigma-0.02,
        dt_days=1, K=K, T=T, r=r, q=q,
        option_type=OptionType.CALL, notional=N,
    )
    print_pnl_attribution(attr, label="+5% spot, −2 vol pts, 1-day")


if __name__ == "__main__":
    main()
