# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 09:32:37 2025

@author: demaria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from _auxFunctions import load_params, l1_cost
from scipy.linalg import solve_continuous_are

# -------------------------------------------------
# 1) Parameters
# -------------------------------------------------
params = load_params("./data.txt")
m1, k1, c1 = params["m1"], params["k1"], params["c1"]
m2, k2_nominal, c2 = params["m2"], params["k2"], params["c2"]
cd = params["cd"]
k2_nominal=2000
# external forcing on m1
F0 = 10.0     # [N] amplitude

# time grid (fixed for all simulations)
T_end = 5.0          # [s] total time
n_pts = 500
t_eval = np.linspace(0.0, T_end, n_pts)

# initial conditions: [x1, x1d, x2, x2d]
y0 = [0.0, 0.0, 0.0, 0.0]


# -------------------------------------------------
# 2) Forcing
# -------------------------------------------------
def F_ext(t):
    """External force on m1."""
    return F0 if t <3.0 else 0


# -------------------------------------------------
# 3bis) LQR design (linearize around k2_nominal)
# -------------------------------------------------
# Build linear state-space x = [x1, x1d, x2, x2d]
# x1'  = x1d
# x1'' = (-k1*x1 - c1*x1d + cd*(x2d - x1d))/m1
# x2'  = x2d
# x2'' = (-k2_nominal*x2 - c2*x2d - cd*(x2d - x1d))/m2 + u/m2
A = np.zeros((4,4))
A[0,1] = 1.0
A[1,0] = -k1 / m1
A[1,1] = - (c1 + cd) / m1
A[1,3] = cd / m1
A[2,3] = 1.0
A[3,0] = 0.0
A[3,2] = -k2_nominal / m2
A[3,1] = cd / m2
A[3,3] = - (c2 + cd) / m2

B = np.zeros((4,1))
B[3,0] = 1.0 / m2  # control enters as an additive force on mass 2 for LQR design

# LQR weights (tune as needed)
Q = np.diag([100.0, 1.0, 1.0, 1.0])   # penalize displacements more
R = np.array([[1e-2]])                  # control penalty

# solve CARE and compute gain
P = solve_continuous_are(A, B, Q, R)
K_lqr = np.linalg.solve(R, B.T.dot(P))  # 1x4 row vector

# -------------------------------------------------
# 3) Time-varying k2(t, x) using LQR -> map to stiffness
# -------------------------------------------------
# sensible bounds and regularizer relative to nominal
k2_min = max(1.0, 0.1 * float(k2_nominal))
k2_max = max(float(k2_nominal) * 5.0, float(k2_nominal) + 100.0)
eps_div = 1e-6  # avoid division by zero when x2 ~ 0

# smoothing for k2 updates (first-order low-pass)
tau_k2 = 0.05  # [s] time constant for smoothing k2(t) updates
_k2_filter_state = {"t": None, "k2": float(k2_nominal)}

def k2_effective(t, y):
    x1, x1d, x2, x2d = y

    # state vector for LQR
    x_state = np.array([x1, x1d, x2, x2d])

    # LQR recommended additive force u (to be realized via stiffness change)
    u = -float(K_lqr.dot(x_state))  # scalar

    # robust mapping u -> delta_k2 = -u / x2 with protection for small x2
    if abs(x2) > eps_div:
        delta_k2_des = -u / x2
    else:
        # keep sign information, avoid division by tiny x2
        delta_k2_des = -u / (eps_div if u != 0.0 else 1.0)

    # safety: cap extreme delta_k2 before enforcing bounds
    delta_headroom = max(k2_max - float(k2_nominal), float(k2_nominal) - k2_min)
    max_delta_abs = max(10.0 * delta_headroom, 1e3)
    delta_k2_des = float(np.clip(delta_k2_des, -max_delta_abs, max_delta_abs))

    # desired k2 (clipped to feasible range)
    k2_des = float(np.clip(float(k2_nominal) + delta_k2_des, k2_min, k2_max))

    # first-order smoothing to avoid discontinuities for the ODE solver
    state = _k2_filter_state
    if state["t"] is None:
        k2_prev = state["k2"]
        dt = 0.0
    else:
        k2_prev = state["k2"]
        dt = max(1e-8, t - state["t"])

    alpha = dt / (tau_k2 + dt) if dt > 0 else 1.0
    k2_smooth = (1.0 - alpha) * k2_prev + alpha * k2_des
    k2_smooth = float(np.clip(k2_smooth, k2_min, k2_max))

    # update filter state
    state["k2"] = k2_smooth
    state["t"] = t

    return k2_smooth



# -------------------------------------------------
# 4) ODE systems
# -------------------------------------------------
def rhs_controlled(t, y):
    """
    System with time-varying / state-dependent k2(t, x).
    y = [x1, x1_dot, x2, x2_dot]
    """
    x1, x1d, x2, x2d = y

    k2 = k2_effective(t, y)
    F  = F_ext(t)
    
    # m1 * x1dd = -k1*x1 - c1*x1d + cd*(x2d - x1d) + F_ext
    x1dd = (-k1 * x1 - c1 * x1d + cd * (x2d - x1d) + F) / m1

    # m2 * x2dd = -k2*x2 - c2*x2d - cd*(x2d - x1d)
    x2dd = (-k2 * x2 - c2 * x2d - cd * (x2d - x1d)) / m2

    return [x1d, x1dd, x2d, x2dd]


def rhs_constant(t, y, k2_value):
    """
    Same system but with constant k2 = k2_value.
    Useful as a baseline for comparison.
    """
    x1, x1d, x2, x2d = y
    F  = F_ext(t)

    x1dd = (-k1 * x1 - c1 * x1d + cd * (x2d - x1d) + F) / m1
    x2dd = (-k2_value * x2 - c2 * x2d - cd * (x2d - x1d)) / m2

    return [x1d, x1dd, x2d, x2dd]


# -------------------------------------------------
# 5) Simulation helpers
# -------------------------------------------------
def simulate_controlled():
    """
    Simulate system with state-dependent k2(t, x).
    Returns t, x1, x2, x1d, x2d, v_rel, k2_t.
    """
    sol = solve_ivp(
        rhs_controlled,
        (0.0, T_end),
        y0,
        t_eval=t_eval,
        rtol=1e-7,
        atol=1e-9
    )

    if not sol.success:
        raise RuntimeError("solve_ivp failed for controlled k2(t)")

    t = sol.t
    x1 = sol.y[0, :]
    x1d = sol.y[1, :]
    x2 = sol.y[2, :]
    x2d = sol.y[3, :]

    v_rel = x2d - x1d

    # reconstruct k2(t) along the trajectory
    k2_t = np.array([k2_effective(ti, sol.y[:, i]) for i, ti in enumerate(t)])

    return t, x1, x2, x1d, x2d, v_rel, k2_t


def simulate_constant(k2_value):
    """
    Simulate system with a constant k2 = k2_value.
    Returns t, x1, x2, x1d, x2d, v_rel.
    """
    fun = lambda t, y: rhs_constant(t, y, k2_value)

    sol = solve_ivp(
        fun,
        (0.0, T_end),
        y0,
        t_eval=t_eval,
        rtol=1e-7,
        atol=1e-9
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed for constant k2 = {k2_value}")

    t = sol.t
    x1 = sol.y[0, :]
    x1d = sol.y[1, :]
    x2 = sol.y[2, :]
    x2d = sol.y[3, :]

    v_rel = x2d - x1d
    return t, x1, x2, x1d, x2d, v_rel


# -------------------------------------------------
# 6) Run simulations
# -------------------------------------------------
# Controlled k2(t)
t_ctrl, x1_ctrl, x2_ctrl, x1d_ctrl, x2d_ctrl, v_rel_ctrl, k2_t = simulate_controlled()

# Baseline: constant k2 = k2_nominal
t_nom, x1_nom, x2_nom, x1d_nom, x2d_nom, v_rel_nom = simulate_constant(k2_nominal)

# Optional: quantify how much |x2d - x1d| we get (integral L1 cost)
J_rel_ctrl = np.abs(l1_cost(v_rel_ctrl, t_ctrl))
J_rel_nom  = np.abs(l1_cost(v_rel_nom,  t_nom))

print("Integral of |x2d - x1d|:")
print(f"  Controlled k2(t): {J_rel_ctrl:.6e}")
print(f"  Constant  k2 = {k2_nominal:.3e}: {J_rel_nom:.6e}")

# %%


# -------------------------------------------------
# 7) Plots
# -------------------------------------------------
# Displacements
plt.figure()
plt.plot(t_ctrl, x1_ctrl, label="x1 controlled")
# plt.plot(t_ctrl, x2_ctrl, label="x2 controlled")
plt.plot(t_nom, x1_nom, "--", label=f"x1 constant (k2 = {k2_nominal:.2f})")
# plt.plot(t_nom, x2_nom, "--", label=f"x2 constant (k2 = {k2_nominal:.2f})")
plt.xlabel("time [s]")
plt.ylabel("displacement [m]")
plt.grid(True)
plt.legend()
plt.title("Displacements: controlled k2(t) vs constant k2")

# Relative velocity |x2d - x1d|
plt.figure()
plt.plot(t_ctrl, (v_rel_ctrl), label="|x2d - x1d| controlled")
plt.plot(t_nom, (v_rel_nom),  "--", label="|x2d - x1d| constant")
plt.xlabel("time [s]")
plt.ylabel("x2d - x1d [m/s]")
plt.grid(True)
plt.legend()
plt.title("Relative velocity magnitude |x2d - x1d|")

# k2(t) used by controller
plt.figure()
plt.plot(t_ctrl, k2_t)
plt.xlabel("time [s]")
plt.ylabel("k2(t) [N/m]")
plt.grid(True)
plt.title("Effective stiffness k2(t) (state-dependent)")

plt.tight_layout()
plt.show()
# %%

print('Uncontrolled system cost function: ',l1_cost(t_nom,x1_nom))
print('Controlled system cost function: ',l1_cost(t_ctrl,x1_ctrl))
print('Controlled system cost function [%]: ',l1_cost(t_ctrl,x1_ctrl)/l1_cost(t_nom,x1_nom))
