# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 09:32:37 2025

Two-mass–spring–damper system with state-dependent stiffness k2(t, x)
of the form k2 = alpha * (x1_dot - x2_dot)^2, clipped between k2_min and k2_max.

We minimize a cost functional J(alpha) to identify the optimal alpha.

@author: demaria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from _auxFunctions import load_params
# %%


# -------------------------------------------------
# 1) Parameters
# -------------------------------------------------
params = load_params("./data.txt")
m1, k1, c1 = params["m1"], params["k1"], params["c1"]
m2, k2_nominal, c2 = params["m2"], params["k2"], params["c2"]
cd = params["cd"]

# External forcing on m1
F0 = 1.0       # [N] amplitude

# Time grid (fixed for all simulations)
T_end = 10.0          # [s] total time
n_pts = 4000
t_eval = np.linspace(0.0, T_end, n_pts)

# Initial conditions: [x1, x1d, x2, x2d]
y0 = [0.0, 0.0, 0.0, 0.0]

# Bounds on k2 and on alpha
k2_min = 1.0       # [N/m]
k2_max = 600.0     # [N/m]
alpha_min = 0.0
alpha_max = 1.0e4  # you can tighten/relax this range if needed

# Weight for stiffness penalization in the cost
w_k = 1.0e-8


# -------------------------------------------------
# 2) Forcing
# -------------------------------------------------
def F_ext(t):
    """External force on m1."""
    return F0 if t < 1.0 else 0.0

# %%


# -------------------------------------------------
# 3) k2(t, x; alpha)
# -------------------------------------------------
def k2_effective(state, alpha):
    """
    Compute k2 = alpha * (x1_dot - x2_dot)^2, clipped between k2_min and k2_max.

    state = [x1, x1d, x2, x2d]
    """
    x1, x1d, x2, x2d = state
    v_rel = x1d - x2d
    # k2 = alpha * v_rel**2
    # k2 = alpha * v_rel
    # k2 = alpha * x1
    # k2 = alpha * np.abs(x2-x1)**2
    k2= alpha *np.abs(v_rel)
    # Clip to [k2_min, k2_max]
    k2 = np.clip(k2, k2_min, k2_max)
    return k2


# -------------------------------------------------
# 4) ODE: controlled system
# -------------------------------------------------
def rhs_controlled(t, y, alpha):
    """
    System with time-varying / state-dependent k2(t, x).

    y = [x1, x1_dot, x2, x2_dot]
    """
    x1, x1d, x2, x2d = y
    k2 = k2_effective(y, alpha)
    F = F_ext(t)

    # m1 * x1dd = -k1*x1 - c1*x1d + cd*(x2d - x1d) + F_ext
    x1dd = (-k1 * x1 - c1 * x1d + cd * (x2d - x1d) + F) / m1

    # m2 * x2dd = -k2*x2 - c2*x2d - cd*(x2d - x1d)
    x2dd = (-k2 * x2 - c2 * x2d - cd * (x2d - x1d)) / m2

    return [x1d, x1dd, x2d, x2dd]


# -------------------------------------------------
# 5) Simulation helper
# -------------------------------------------------
def simulate_controlled(alpha):
    """
    Simulate the system for a given alpha.

    Returns:
        t      : time vector
        x1,x2  : displacements
        x1d,x2d: velocities
        v_rel  : x2d - x1d
        k2_t   : k2(t) along the trajectory
    """
    def fun(t, y):
        return rhs_controlled(t, y, alpha)

    sol = solve_ivp(
        fun,
        (0.0, T_end),
        y0,
        t_eval=t_eval,
        rtol=1e-7,
        atol=1e-9
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed for alpha = {alpha}. Message: {sol.message}")

    t = sol.t
    x1 = sol.y[0, :]
    x1d = sol.y[1, :]
    x2 = sol.y[2, :]
    x2d = sol.y[3, :]

    v_rel = x2d - x1d

    # Reconstruct k2(t) along the trajectory
    k2_t = np.empty_like(t)
    for i in range(t.size):
        state_i = sol.y[:, i]
        k2_t[i] = k2_effective(state_i, alpha)

    return t, x1, x2, x1d, x2d, v_rel, k2_t


# -------------------------------------------------
# 6) Cost function
# -------------------------------------------------
def J_cost(t, x, k2_t, w_k):
    """
    Cost functional:
        J = ∫ x(t)^2 dt + w_k * ∫ k2(t)^2 dt
    """
    J_x = np.trapezoid(x**2, t)
    J_k = w_k * np.trapz(k2_t**2, t)
    return J_x + J_k


# -------------------------------------------------
# 7) Objective in terms of alpha
# -------------------------------------------------
def objective_alpha(alpha):
    """
    Objective J(alpha) to be minimized.

    Smaller J(alpha) means smaller oscillations of x1 and/or smaller k2 usage.
    """
    # Enforce non-negative alpha explicitly (safety)
    if alpha < 0:
        return 1e20

    try:
        t, x1, x2, x1d, x2d, v_rel, k2_t = simulate_controlled(alpha)
    except Exception:
        # If integration fails, return a big penalty
        return 1e20

    J = J_cost(t, x1, k2_t, w_k=w_k)

    # Guard against NaNs or inf
    if not np.isfinite(J):
        return 1e20

    return J


# -------------------------------------------------
# 8) Optimization + post-processing
# -------------------------------------------------
if __name__ == "__main__":
    # Optimize alpha
    res = minimize_scalar(
        objective_alpha,
        bounds=(alpha_min, alpha_max),
        method="bounded",
        options={"xatol": 1e-2}
    )

    alpha_opt = res.x

    print("Optimization result:")
    print(f"  k2_nominal = {k2_nominal:.3e} [N/m]")
    print(f"  alpha_opt  = {alpha_opt:.3e}")
    print(f"  min J(a)   = {res.fun:.6e}")
    print(f"  success    = {res.success}, message = {res.message}")

    # Re-simulate with optimal alpha
    t, x1, x2, x1d, x2d, v_rel, k2_t = simulate_controlled(alpha_opt)

    # -------------------------------------------------
    # Simple plots (optional)
    # -------------------------------------------------
    plt.figure()
    plt.plot(t, x1, label="x1(t)")
    plt.plot(t, x2, label="x2(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [m]")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.title("Displacements with optimal alpha")

    plt.figure()
    plt.plot(t, k2_t)
    plt.xlabel("Time [s]")
    plt.ylabel("k2(t) [N/m]")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.title(f"Effective stiffness k2(t) with optimal alpha=  {round(alpha_opt)}")

    plt.show()
