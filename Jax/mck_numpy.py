# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 09:32:37 2025

@author: demaria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from _auxFunctions import load_params
from typing import Callable, Tuple

# -------------------------------------------------
# 1) Parameters
# -------------------------------------------------
params = load_params("./Jax/data.txt")
m1, k1, c1 = params["m1"], params["k1"], params["c1"]
m2, k2_nominal, c2 = params["m2"], params["k2"], params["c2"]
cd = params["cd"]

# external forcing on m1
F0 = 1.0       # [N] amplitude

# time grid (fixed for all simulations)
T_end = 10.0          # [s] total time
n_pts = 4000
t_eval = np.linspace(0.0, T_end, n_pts)

# initial conditions: [x1, x1d, x2, x2d]
y0 = [0.0, 0.0, 0.0, 0.0]
w_k = 1e-8

# -------------------------------------------------
# 2) Forcing and (constant) k2
# -------------------------------------------------
def F_ext(t: float) -> float:
    """External force on m1 (pulse until 0.5s)."""
    return F0 if t < 0.5 else 0.0


def k2_effective(t: float, x: np.ndarray, k2_value: float) -> float:
    """Currently constant stiffness; placeholder for time-varying laws."""
    return k2_value


# -------------------------------------------------
# 3) ODE system for a given k2
# -------------------------------------------------
def make_rhs(k2_value: float) -> Callable:
    """Return rhs(t, y) for the given k2."""
    def rhs(t: float, y: np.ndarray):
        x1, x1d, x2, x2d = y
        k2 = k2_effective(t, y, k2_value)
        F = F_ext(t)

        # m1 * x1dd = -k1*x1 - c1*x1d + cd*(x2d - x1d) + F
        # m2 * x2dd = -k2*x2 - c2*x2d - cd*(x2d - x1d)
        x1dd = (-k1 * x1 - c1 * x1d + cd * (x2d - x1d) + F) / m1
        x2dd = (-k2 * x2 - c2 * x2d - cd * (x2d - x1d)) / m2

        return [x1d, x1dd, x2d, x2dd]

    return rhs


def J_cost(t: np.ndarray, x: np.ndarray, k_value: float, w_k: float) -> float:
    """
    Quadratic cost: integral x^2 dt + w_k * integral k^2 dt.
    For constant k, integral k^2 dt = k^2 * T.
    """
    if t.size == 0:
        return np.inf
    J_x = np.trapz(x**2, t)
    # more efficient analytic expression for constant k
    J_k = w_k * (k_value**2) * (t[-1] - t[0])
    return float(J_x + J_k)


# -------------------------------------------------
# Simulation and objective
# -------------------------------------------------
def simulate(k2_value: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the system for a given k2_value and return (t, x1, x2)."""
    rhs = make_rhs(k2_value)
    sol = solve_ivp(rhs, (0.0, T_end), y0, t_eval=t_eval, rtol=1e-7, atol=1e-9)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed for k2 = {k2_value}")
    t = sol.t
    x1 = sol.y[0, :]
    x2 = sol.y[2, :]
    return t, x1, x2


def objective_k2(k2_value: float) -> float:
    """Objective to minimize w.r.t k2 (returns large penalty for invalid k2)."""
    if not np.isfinite(k2_value) or k2_value <= 0.0:
        return 1e12
    try:
        t, x1, _ = simulate(k2_value)
    except Exception:
        return 1e12
    return J_cost(t, x1, k2_value, w_k=w_k)


# -------------------------------------------------
# Main optimization + plotting (encapsulated)
# -------------------------------------------------
def run_optimization():
    k2_min = 1.0
    k2_max = 600.0

    res = minimize_scalar(
        objective_k2,
        bounds=(k2_min, k2_max),
        method="bounded",
        options={"xatol": 1e-2},
    )

    k2_opt = float(res.x)
    print("Optimization result:")
    print(f"  k2_nominal = {k2_nominal:.3e} [N/m]")
    print(f"  k2_opt     = {k2_opt:.3e} [N/m]")
    print(f"  min J(k2)  = {res.fun:.6e}")
    print(f"  success    = {res.success}, message = {res.message}")

    # cost curve
    n_samples = 60
    k2_grid = np.linspace(k2_min, k2_max, n_samples)
    J_vals = np.array([objective_k2(k) for k in k2_grid])
    J_nominal = objective_k2(k2_nominal)

    plt.figure()
    plt.plot(k2_grid, J_vals, marker="o", linestyle="-", label="J(k2)")
    plt.axvline(k2_opt, linestyle="--", label=f"k2_opt = {k2_opt:.2f}")
    plt.xlabel("k2 [N/m]")
    plt.ylabel("J(k2)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # final simulation with optimal k2
    t, x1, x2 = simulate(k2_opt)
    plt.figure()
    plt.plot(t, x1, label="x1 (m1)")
    plt.plot(t, x2, label="x2 (m2)")
    plt.xlabel("time [s]")
    plt.ylabel("displacement [m]")
    plt.grid(True)
    plt.legend()
    plt.title(f"Response with optimized k2 = {k2_opt:.3e} N/m")

    # spare sim
    sim_k = 200
    t, x1_nom, x2_nom = simulate(sim_k)
    plt.figure()
    plt.plot(t, x1_nom, "-", label="x1")
    plt.plot(t, x2_nom, "--", label="x2")
    plt.xlabel("time [s]")
    plt.ylabel("displacement [m]")
    plt.grid(True)
    plt.title("k constant = " + str(sim_k))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_optimization()