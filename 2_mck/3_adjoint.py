# -*- coding: utf-8 -*-
"""
Two-mass–spring–damper system with state-dependent stiffness k2(t, x)
of the form k2 = k2_min + alpha * |x1_dot - x2_dot|,
clipped between [k2_min, k2_max].

We minimize a cost functional J(alpha) using an adjoint-based gradient.

Created on Thu Nov 27 10:40:59 2025
@author: demaria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from _auxFunctions import load_params

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
y0 = np.array([0.0, 0.0, 0.0, 0.0])

# Bounds on k2 and on alpha
k2_min = 1.0       # [N/m]
k2_max = 600.0     # [N/m]
alpha_min = 0.0
alpha_max = 400

# Weight for stiffness penalization in the cost
w_k = 1.0e-8

# Small epsilon for derivative logic
_eps = 1e-8


# -------------------------------------------------
# 2) Forcing
# -------------------------------------------------
def F_ext(t):
    """External force on m1."""
    return F0 if t <1.0 else 0.0


# -------------------------------------------------
# 3) k2 and its derivatives
# -------------------------------------------------
def k2_and_derivs(state, alpha):
    """
    Compute k2 and its derivatives wrt v1, v2, alpha.

    state = [x1, v1, x2, v2]
    k2_raw = k2_min + alpha * |v_rel|,
    v_rel = v1 - v2,
    k2 = min(k2_raw, k2_max).
    """
    x1, v1, x2, v2 = state
    v_rel = v1 - v2
    v_abs = np.abs(v_rel)

    # Raw stiffness before clipping
    k2_raw = k2_min + alpha * v_abs
    # Clip only upper bound
    k2 = np.minimum(k2_raw, k2_max)

    # Defaults: saturated at max or v_rel ~ 0 -> zero derivatives
    dk_dv1 = 0.0
    dk_dv2 = 0.0
    dk_dalpha = 0.0

    # If not saturated at the top and v_rel != 0, we have nonzero derivatives
    if (k2_raw < k2_max - _eps) and (v_abs > _eps):
        s = np.sign(v_rel)
        dk_dv1 = alpha * s
        dk_dv2 = -alpha * s
        dk_dalpha = v_abs

    return k2, dk_dv1, dk_dv2, dk_dalpha


def k2_effective(state, alpha):
    """Wrapper: only k2(t, x; alpha)."""
    k2, _, _, _ = k2_and_derivs(state, alpha)
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

    return np.array([x1d, x1dd, x2d, x2dd])


# -------------------------------------------------
# 5) Forward simulation
# -------------------------------------------------
def simulate_forward(alpha):
    """
    Simulate the system for a given alpha.

    Returns:
        sol   : OdeSolution object (with dense_output)
        t     : time vector
        Y     : states array shape (4, N)
        k2_t  : k2(t) along the trajectory
    """
    def fun(t, y):
        return rhs_controlled(t, y, alpha)

    sol = solve_ivp(
        fun,
        (0.0, T_end),
        y0,
        t_eval=t_eval,
        rtol=1e-7,
        atol=1e-9,
        dense_output=True
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed for alpha = {alpha}. Message: {sol.message}")

    t = sol.t
    Y = sol.y  # shape (4, N)

    # Reconstruct k2(t) along the trajectory
    k2_t = np.empty_like(t)
    for i in range(t.size):
        state_i = Y[:, i]
        k2_t[i] = k2_effective(state_i, alpha)

    return sol, t, Y, k2_t


# -------------------------------------------------
# 6) Cost function
# -------------------------------------------------
def J_cost(t, x1, k2_t, w_k):
    """
    Cost functional:
        J = ∫ [ x1(t)^2 + w_k * k2(t)^2 ] dt
    """
    J_x = np.trapezoid(x1**2, t)
    J_k = w_k * np.trapz(k2_t**2, t)
    return J_x + J_k


# -------------------------------------------------
# 7) Jacobian df/dy and df/dalpha
# -------------------------------------------------
def df_dy_and_df_dalpha(t, y, alpha):
    """
    Compute Jacobian df/dy (4x4) and df/dalpha (4,).

    y = [x1, v1, x2, v2]
    """
    x1, v1, x2, v2 = y
    k2, dk_dv1, dk_dv2, dk_dalpha = k2_and_derivs(y, alpha)

    A = np.zeros((4, 4))
    df_dalpha = np.zeros(4)

    # f1 = v1
    A[0, 0] = 0.0
    A[0, 1] = 1.0
    A[0, 2] = 0.0
    A[0, 3] = 0.0

    # f2 = (-k1 x1 - c1 v1 + cd (v2 - v1) + F) / m1
    A[1, 0] = -k1 / m1
    A[1, 1] = -(c1 + cd) / m1
    A[1, 2] = 0.0
    A[1, 3] = cd / m1

    # f3 = v2
    A[2, 0] = 0.0
    A[2, 1] = 0.0
    A[2, 2] = 0.0
    A[2, 3] = 1.0

    # f4 = (-k2 x2 - c2 v2 - cd (v2 - v1)) / m2
    #     = (-k2 x2 - (c2 + cd) v2 + cd v1) / m2
    A[3, 0] = 0.0
    A[3, 1] = (-x2 * dk_dv1 + cd) / m2
    A[3, 2] = -k2 / m2
    A[3, 3] = (-x2 * dk_dv2 - (c2 + cd)) / m2

    # df/dalpha: only in f4 through k2
    df_dalpha[3] = -(x2 * dk_dalpha) / m2

    return A, df_dalpha


# -------------------------------------------------
# 8) Cost and gradient via adjoint
# -------------------------------------------------
def compute_cost_and_gradient(alpha):
    """
    Compute J(alpha) and dJ/dalpha using an adjoint ODE.
    """

    # ----- Forward solve -----
    sol, t, Y, k2_t = simulate_forward(alpha)
    x1 = Y[0, :]
    J = J_cost(t, x1, k2_t, w_k=w_k)

    # ----- Adjoint solve (backward in time) -----
    p_T = np.zeros(4)   # p(T) = 0

    def rhs_adjoint(t_adj, p):
        """
        Adjoint ODE:
            dp/dt = - (df/dy)^T p - dL/dy
        """
        y = sol.sol(t_adj).reshape(4,)
        x1, v1, x2, v2 = y

        k2, dk_dv1, dk_dv2, _ = k2_and_derivs(y, alpha)
        A, _ = df_dy_and_df_dalpha(t_adj, y, alpha)

        # L = x1^2 + w_k k2^2
        dL_dy = np.zeros(4)
        dL_dy[0] = 2.0 * x1
        dL_dy[1] = 2.0 * w_k * k2 * dk_dv1
        dL_dy[2] = 0.0
        dL_dy[3] = 2.0 * w_k * k2 * dk_dv2

        dpdt = -A.T @ p - dL_dy
        return dpdt

    sol_adj = solve_ivp(
        rhs_adjoint,
        (T_end, 0.0),
        p_T,
        t_eval=t[::-1],     # backwards times
        rtol=1e-7,
        atol=1e-9
    )

    if not sol_adj.success:
        raise RuntimeError(f"Adjoint solve failed for alpha = {alpha}. Message: {sol_adj.message}")

    # Reorder adjoint to forward time
    p_backward = sol_adj.y  # shape (4, N) at t[::-1]
    p_forward = np.fliplr(p_backward).T  # shape (N, 4), aligned with t

    # ----- Gradient dJ/dalpha -----
    integrand = np.zeros_like(t)

    for i, (ti, pi) in enumerate(zip(t, p_forward)):
        y = Y[:, i]
        k2, dk_dv1, dk_dv2, dk_dalpha = k2_and_derivs(y, alpha)
        _, df_dalpha = df_dy_and_df_dalpha(ti, y, alpha)

        # ∂L/∂alpha = 2 w_k k2 dk_dalpha
        dL_dalpha = 2.0 * w_k * k2 * dk_dalpha

        integrand[i] = dL_dalpha + np.dot(pi, df_dalpha)

    dJ_dalpha = np.trapz(integrand, t)

    return J, dJ_dalpha


# -------------------------------------------------
# 9) Simple gradient-descent optimization on alpha
# -------------------------------------------------
if __name__ == "__main__":
    # Try a non-trivial initial guess
    alpha = 300
    lr = 1.0e6      # learning rate for gradient descent
    max_iter = 300
    ALPHAS= []
    Js = []
    print("Gradient-descent optimization on alpha (adjoint-based gradient)")
    for it in range(max_iter):
        J, dJ = compute_cost_and_gradient(alpha)
        ALPHAS.append(alpha)
        Js.append(J)
        print(f"iter {it:03d} | alpha = {alpha} | J = {J:.6e} | dJ/dalpha = {dJ:.6e}")

        alpha_new = alpha - lr * dJ
        alpha_new = np.clip(alpha_new, alpha_min, alpha_max)

        if np.abs(alpha_new - alpha) < 1e-12:
            alpha = alpha_new
            print("Converged (small alpha update).")
            break

        alpha = alpha_new

    alpha_opt = alpha
    print("\nAdjoint-based optimization result:")
    print(f"  k2_nominal = {k2_nominal:.3e} [N/m]")
    print(f"  alpha_opt  = {alpha_opt:.3e}")

    # ----- Re-simulate with optimal alpha -----
    sol_opt, t, Y_opt, k2_t_opt = simulate_forward(alpha_opt)
    x1_opt = Y_opt[0, :]
    x2_opt = Y_opt[2, :]

    # -------------------------------------------------
    # Plots
    # -------------------------------------------------
    plt.figure()
    plt.plot(t, x1_opt, label="x1(t)")
    plt.plot(t, x2_opt, label="x2(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [m]")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.title(f"Displacements with optimal alpha = {alpha_opt:.2e}")

    plt.figure()
    plt.plot(t, k2_t_opt)
    plt.xlabel("Time [s]")
    plt.ylabel("k2(t) [N/m]")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.title(f"Effective stiffness k2(t) with optimal alpha = {alpha_opt:.2e}")

    plt.show()
    
    fig, ax1 = plt.subplots()
    
    iters = np.arange(len(ALPHAS))
    
    # Left y-axis: alpha
    ax1.plot(iters, np.array(ALPHAS), marker='o',label='alpha')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('Alpha')
    ax1.grid(True)
    
    # Right y-axis: J
    ax2 = ax1.twinx()
    ax2.plot(iters, Js,'r', marker='x',label='J')
    ax2.set_ylabel('J')
    plt.legend()
    plt.title('Alpha and J vs iteration')
    plt.xlim(0,80)
    plt.show()
