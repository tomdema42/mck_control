# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 14:30:42 2026

@author: demaria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are


# -----------------------------
# Forcing (impulse-like pulse)
# -----------------------------
def make_forcing(F0=10.0, w=10.0):
    # def F_ext(t):
    #     return F0 * np.sin(w * t)
    def F_ext(t):
        t0 = 1.0
        dt = 1.0
        return F0 if (t0 <= t < t0 + dt) else 0.0
    return F_ext


# -----------------------------
# LQR helper
# -----------------------------
def lqr_gain(A, B, Q, R):
    """
    Continuous-time LQR:
      minimize ∫ (x^T Q x + u^T R u) dt
      u = -K x
    """
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)  # K = R^-1 B^T P
    return K


# -----------------------------
# Simulation (passive + LQR)
# -----------------------------
def simulate_2dof_with_lqr(
    m1=1.0, m2=0.2,
    k1=80.0, k2=20.0,
    c1=0.8, c2=0.2,
    kc=30.0, cd=0.5,
    F0=5.0, w_forcing=8.0,
    u_max=100.0,
    t_end=15.0, max_step=1e-2,
    y0=(0.0, 0.0, 0.0, 0.0),
    # weights for Q via outputs: [x1, x1d, e=x1+x2, ed=x1d+x2d]
    w_x1=1.0, w_x1d=0.1, w_e=20.0, w_ed=1.0,
    r_u=0.01,
):
    """
    State y = [x1, x1d, x2, x2d]

    Passive dynamics:
      x1dd = (-k1*x1 - c1*x1d + cd*(x2d-x1d) + kc*(x2-x1) + F_ext(t)) / m1
      x2dd = (-k2*x2 - c2*x2d - cd*(x2d-x1d) - kc*(x2-x1)) / m2

    With control u on mass 2:
      x2dd = passive_terms/m2 + u/m2

    LQR is designed on the *linear* state-space model (same as the ODE, no approximation),
    and we use it as u = -K y (then saturate).
    """

    F_ext = make_forcing(F0=F0, w=w_forcing)

    def clip(u):
        return np.clip(u, -u_max, u_max)

    # --- Build state-space matrices A, B for the unforced (F_ext=0) plant ---
    # x = [x1, v1, x2, v2]
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-(k1 + kc) / m1, -(c1 + cd) / m1,  kc / m1,         cd / m1],
        [0.0, 0.0, 0.0, 1.0],
        [ kc / m2,         cd / m2,        -(k2 + kc) / m2, -(c2 + cd) / m2],
    ])

    B = np.array([
        [0.0],
        [0.0],
        [0.0],
        [1.0 / m2],
    ])

    # --- Build Q to penalize: x1, x1d, e=x1+x2, ed=x1d+x2d ---
    # z = C x, cost = z^T W z = x^T (C^T W C) x
    C = np.array([
        [1.0, 0.0, 0.0, 0.0],  # x1
        [0.0, 1.0, 0.0, 0.0],  # x1d
        [1.0, 0.0, 1.0, 0.0],  # e = x1 + x2
        [0.0, 1.0, 0.0, 1.0],  # ed = x1d + x2d
    ])
    W = np.diag([w_x1, w_x1d, w_e, w_ed])
    Q = C.T @ W @ C

    R = np.array([[r_u]])

    K = lqr_gain(A, B, Q, R)  # shape (1,4)

    # store control history
    u_log_t = []
    u_log = []

    def rhs_passive(t, y):
        x1, x1d, x2, x2d = y
        x1dd = (-k1 * x1 - c1 * x1d + cd * (x2d - x1d) + kc * (x2 - x1) + F_ext(t)) / m1
        x2dd = (-k2 * x2 - c2 * x2d - cd * (x2d - x1d) - kc * (x2 - x1)) / m2
        return [x1d, x1dd, x2d, x2dd]

    def rhs_lqr(t, y):
        x1, x1d, x2, x2d = y

        # LQR control law
        u = -float(K @ np.array([x1, x1d, x2, x2d]))
        u = float(clip(u))

        u_log_t.append(t)
        u_log.append(u)

        # dynamics with control
        x1dd = (-k1 * x1 - c1 * x1d + cd * (x2d - x1d) + kc * (x2 - x1) + F_ext(t)) / m1
        x2dd = (-k2 * x2 - c2 * x2d - cd * (x2d - x1d) - kc * (x2 - x1) + u) / m2

        return [x1d, x1dd, x2d, x2dd]

    sol_passive = solve_ivp(
        rhs_passive, (0.0, t_end), y0,
        max_step=max_step, rtol=1e-7, atol=1e-9
    )

    sol_lqr = solve_ivp(
        rhs_lqr, (0.0, t_end), y0,
        max_step=max_step, rtol=1e-7, atol=1e-9
    )

    return sol_passive, sol_lqr, (np.array(u_log_t), np.array(u_log)), K


# -----------------------------
# Run + plots
# -----------------------------
if __name__ == "__main__":
    sol0, sol1, (tu, u), K = simulate_2dof_with_lqr(
        m1=0.3, m2=0.1,
        k1=11.83, k2=20.0,
        c1=0.11, c2=0.05,
        kc=0.0, cd=1.6,
        F0=1.0, w_forcing=8.0,
        u_max=10.0,
        t_end=10.0,
        y0=(0.0, 0.0, 0.0, 0.0),
        # try tuning these:
        w_x1=1.0, w_x1d=0.1, w_e=50.0, w_ed=2.0,
        r_u=0.05,
    )

    print("LQR gain K =", K)

    t0 = sol0.t
    x1_0, x1d_0, x2_0, x2d_0 = sol0.y

    t1 = sol1.t
    x1_1, x1d_1, x2_1, x2d_1 = sol1.y

    plt.figure(figsize=(12, 6))
    plt.plot(t0, x1_0, label="x1 passive")
    plt.plot(t0, x2_0, label="x2 passive")
    plt.plot(t1, x1_1, "--", label="x1 LQR")
    plt.plot(t1, x2_1, "--", label="x2 LQR")
    plt.xlabel("t [s]")
    plt.ylabel("Displacement [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(tu, u, label="u(t)")
    plt.xlabel("t [s]")
    plt.ylabel("Control force [N]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
