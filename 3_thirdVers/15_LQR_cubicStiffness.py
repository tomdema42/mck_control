# -*- coding: utf-8 -*-
"""
Full-state continuous-time LQR for the 2-DOF mass-spring-damper system,
then simulation on the NONLINEAR plant (Duffing cubic stiffness on DOF1 / left wall / m1):

Duffing term:
    x1dd += -(k3/m1) * x1^3

Control (full state):
    u(t) = -K x(t),   K = [K1,K2,K3,K4]

External forcing:
    applied on DOF1 acceleration: +F(t)/m1 in x1dd

@author: demaria
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

from _auxFunc import load_params, make_forcing


# =============================================================================
# Model (linear part)
# =============================================================================
def build_linear_system(m1, m2, k1, k2, c1, c2, kc, cd):
    """
    State x = [x1, x1d, x2, x2d]
    Linear dynamics:
        xdot = A x + B u + b(F)
    where u acts on x2dd only.
    """
    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-(k1 + kc) / m1, -(c1 + cd) / m1, kc / m1, cd / m1],
            [0.0, 0.0, 0.0, 1.0],
            [kc / m2, cd / m2, -(k2 + kc) / m2, -(c2 + cd) / m2],
        ]
    )

    B = np.array([0.0, 0.0, 0.0, 1.0 / m2]).reshape(4, 1)

    def forcing_vec(F_scalar):
        return np.array([0.0, F_scalar / m1, 0.0, 0.0])

    return A, B, forcing_vec


# =============================================================================
# LQR (Q built to match your running cost)
# =============================================================================
def build_Q(w_x1, w_x1d, w_e, w_ed):
    """
    Your running cost:
        w_x1 x1^2 + w_x1d x1d^2 + w_e (x1+x2)^2 + w_ed (x1d+x2d)^2
    corresponds to x^T Q x with:

    Q00 = w_x1 + w_e
    Q11 = w_x1d + w_ed
    Q22 = w_e
    Q33 = w_ed
    Q02 = Q20 = w_e
    Q13 = Q31 = w_ed
    """
    Q = np.zeros((4, 4))
    Q[0, 0] = w_x1 + w_e
    Q[1, 1] = w_x1d + w_ed
    Q[2, 2] = w_e
    Q[3, 3] = w_ed
    Q[0, 2] = w_e
    Q[2, 0] = w_e
    Q[1, 3] = w_ed
    Q[3, 1] = w_ed
    return Q


def lqr_gain(A, B, Q, r_u):
    """
    Continuous-time LQR:
        minimize ∫ (x^T Q x + u^T R u) dt
        u = -K x
    """
    R = np.array([[r_u]])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)  # shape (1,4)
    return K.ravel(), P


# =============================================================================
# Nonlinear plant (Duffing on DOF1) + simulation
# =============================================================================
def simulate_nonlinear_duffing_lqr(
    A,
    B,
    forcing_vec,
    F_ext,       # callable F(t)
    K,           # (4,)
    k3,          # Duffing coefficient
    m1,          # used in Duffing term
    t_end,
    x0,
    n_eval=2000,
    use_control=True,
    method="RK45",
    rtol=1e-7,
    atol=1e-9,
):
    """
    Nonlinear dynamics:
        xdot = A x + B u + b(F) + g_duff(x)
    g_duff affects x1dd only:
        x1dd += -(k3/m1) x1^3
    """
    t_eval = np.linspace(0.0, t_end, n_eval)

    def rhs(t, x):
        if use_control:
            u = -float(K @ x)
        else:
            u = 0.0

        F = float(F_ext(t))
        xdot = A @ x + (B[:, 0] * u) + forcing_vec(F)

        # Duffing on DOF1: x1dd is state index 1
        xdot[1] += -(k3 / m1) * (x[0] ** 3)

        return xdot

    sol = solve_ivp(rhs, (0.0, t_end), np.array(x0), t_eval=t_eval, method=method, rtol=rtol, atol=atol)

    X = sol.y.T  # (n_eval,4)
    if use_control:
        u = -(X @ K)
    else:
        u = np.zeros(len(t_eval))

    return sol.t, X, u


def running_cost_nodes(X, u, w_x1, w_x1d, w_e, w_ed, r_u):
    x1 = X[:, 0]
    x1d = X[:, 1]
    x2 = X[:, 2]
    x2d = X[:, 3]
    e = x1 + x2
    ed = x1d + x2d
    L = (
        w_x1 * x1**2
        + w_x1d * x1d**2
        + w_e * e**2
        + w_ed * ed**2
        + r_u * u**2
    )
    return L


def trapz_cost(t, L):
    return np.trapz(L, t)


# =============================================================================
# Plotting
# =============================================================================
def plot_states_and_control(t, X_passive, X_cl, u_cl):
    plt.figure(figsize=(12, 6))
    plt.plot(t, X_passive[:, 0], label="x1 passive")
    plt.plot(t, X_passive[:, 2], label="x2 passive")
    plt.plot(t, X_cl[:, 0], "--", label="x1 closed-loop (LQR)")
    plt.plot(t, X_cl[:, 2], "--", label="x2 closed-loop (LQR)")
    plt.xlabel("t [s]")
    plt.ylabel("Displacement [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(t, u_cl, label="u(t) = -Kx (LQR)")
    plt.xlabel("t [s]")
    plt.ylabel("Control force [N]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    start = time.time()

    # -----------------------------
    # Load parameters + forcing
    # -----------------------------
    p = load_params("params.txt")

    m1, m2 = p["m1"], p["m2"]
    k1, k2 = p["k1"], p["k2"]
    c1, c2 = p["c1"], p["c2"]
    cd, kc = p["cd"], p["kc"]

    F_ext = make_forcing(p)

    # -----------------------------
    # Duffing coefficient on DOF1
    # -----------------------------
    k3 = 60_000  # change as you want

    # -----------------------------
    # LQR weights (match your cost)
    # -----------------------------
    w_x1, w_x1d = 1.0, 0.1
    w_e, w_ed = 50.0, 2.0
    r_u = 0.05

    # -----------------------------
    # Build A,B and compute LQR
    # -----------------------------
    A, B, forcing_vec = build_linear_system(m1, m2, k1, k2, c1, c2, kc, cd)
    Q = build_Q(w_x1, w_x1d, w_e, w_ed)

    K, P = lqr_gain(A, B, Q, r_u)


    # -----------------------------
    # Simulate (nonlinear plant)
    # -----------------------------
    t_end = 10.0
    x0 = (0.0, 0.0, 0.0, 0.0)

    t, X_passive, u_passive = simulate_nonlinear_duffing_lqr(
        A, B, forcing_vec, F_ext, K, k3, m1,
        t_end=t_end, x0=x0, n_eval=2000, use_control=False
    )

    t, X_cl, u_cl = simulate_nonlinear_duffing_lqr(
        A, B, forcing_vec, F_ext, K, k3, m1,
        t_end=t_end, x0=x0, n_eval=2000, use_control=True
    )

    # -----------------------------
    # Compare costs on trajectories
    # -----------------------------
    L_passive = running_cost_nodes(X_passive, u_passive, w_x1, w_x1d, w_e, w_ed, r_u)
    L_cl = running_cost_nodes(X_cl, u_cl, w_x1, w_x1d, w_e, w_ed, r_u)

    J_passive = trapz_cost(t, L_passive)
    J_cl = trapz_cost(t, L_cl)

    print("\nTrajectory costs (evaluated on the nonlinear Duffing plant):")
    print("J_passive =", J_passive)
    print("J_LQR     =", J_cl)

    print("\nTotal Time:", time.time() - start)

    # -----------------------------
    # Plots
    # -----------------------------
    plot_states_and_control(t, X_passive, X_cl, u_cl)
