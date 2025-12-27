# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 14:28:30 2025

@author: demaria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from _auxFunc import load_params, make_forcing, build_canonical_params


def simulate_from_params(p):
    m1, m2 = p["m1"], p["m2"]
    k1, k2 = p["k1"], p["k2"]
    c1, c2 = p["c1"], p["c2"]
    cd, kc = p["cd"], p["kc"]

    F_ext = make_forcing(p)

    y0 = (p["x1_0"], p["x1d_0"], p["x2_0"], p["x2d_0"])
    t_eval = np.linspace(p["t0"], p["t_end"], int(p["n_points"]))

    def rhs(t, y):
        x1, x1d, x2, x2d = y
        x1dd = (-k1*x1 - c1*x1d + cd*(x2d - x1d) + kc*(x2 - x1) + F_ext(t)) / m1
        x2dd = (-k2*x2 - c2*x2d - cd*(x2d - x1d) - kc*(x2 - x1)) / m2
        return [x1d, x1dd, x2d, x2dd]

    sol = solve_ivp(
        rhs,
        (p["t0"], p["t_end"]),
        y0,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.t, sol.y


def objective_J_of_k2(k2, p_base, reg_k2):
    p = dict(p_base)
    p["k2"] = float(k2)
    t, y = simulate_from_params(p)
    x1 = y[0]
    
    integrand = x1**2
    J_x1 = np.trapezoid(integrand, t)
    J_k2 = np.trapezoid([float(k2)]*len(t),t)
    
    J_tot = J_x1 + J_k2*reg_k2
    return J_tot


def sweep_k2(param_file, k2_min, k2_max, reg_k2, n_k2=80, logspace=True):
    p_base = load_params(param_file)

    if logspace:
        k2_grid = np.logspace(np.log10(k2_min), np.log10(k2_max), n_k2)
    else:
        k2_grid = np.linspace(k2_min, k2_max, n_k2)

    J_grid = np.empty_like(k2_grid)

    for i, k2 in enumerate(k2_grid):
        J_grid[i] = objective_J_of_k2(k2, p_base, reg_k2)

    i_best = int(np.argmin(J_grid))
    k2_best = float(k2_grid[i_best])
    J_best = float(J_grid[i_best])

    return p_base, k2_grid, J_grid, k2_best, J_best


if __name__ == "__main__":
    param_file = "params.txt"
    k2_min = 1.0
    k2_max = 1e3
    n_k2 = 100
    reg_k2 = 1e-6
    
    
    p_base, k2_grid, J_grid, k2_best, J_best = sweep_k2(
        param_file, k2_min, k2_max, reg_k2 , n_k2=n_k2, logspace=True
    )

    print(f"Best (from sweep) k2 = {k2_best}")
    print(f"Best (from sweep) J  = {J_best}")
    _ = build_canonical_params(load_params(param_file), verbose=True)

    # --- Plot cost function ---
    plt.figure()
    plt.plot(k2_grid, J_grid, marker="o", linewidth=1)
    plt.xscale("log")
    plt.xlabel("k2")
    plt.ylabel(r"J(k2) = ∫ x1(t)^2 dt")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    # --- Optional: simulate & plot trajectories at best k2 from sweep ---
    p_best = dict(p_base)
    p_best["k2"] = k2_best
    t, y = simulate_from_params(p_best)
    x1, x1d, x2, x2d = y

    plt.figure()
    plt.plot(t, x1, label="x1 (best from sweep)")
    plt.plot(t, x2, label="x2 (best from sweep)")
    plt.xlabel("t [s]")
    plt.ylabel("Displacement [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(t, x1d, label="x1d (best from sweep)")
    plt.plot(t, x2d, label="x2d (best from sweep)")
    plt.xlabel("t [s]")
    plt.ylabel("Velocity [m/s]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()
