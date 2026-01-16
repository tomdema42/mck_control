# -*- coding: utf-8 -*-
"""
2-DOF mass-spring-damper + LQR control on mass 2

State: y = [x1, x1d, x2, x2d]
Control: u(t) = sat( -K y )

Main fix vs your version:
- DO NOT log u(t) inside rhs() when using solve_ivp (it evaluates rhs many times internally).
- Instead: integrate with a fixed output grid (t_eval), then compute u(t) from the stored trajectory.

@author: demaria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

from _auxFunc import load_params, make_forcing


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
    K = np.linalg.solve(R, B.T @ P)
    return K


# -----------------------------
# Model matrices
# -----------------------------
def build_state_space(m1, m2, k1, k2, c1, c2, kc, cd):
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
    return A, B


def build_Q_from_outputs(w_x1, w_x1d, w_e, w_ed):
    # z = C x, cost = z^T W z = x^T (C^T W C) x
    C = np.array([
        [1.0, 0.0, 0.0, 0.0],  # x1
        [0.0, 1.0, 0.0, 0.0],  # x1d
        [1.0, 0.0, 1.0, 0.0],  # e = x1 + x2
        [0.0, 1.0, 0.0, 1.0],  # ed = x1d + x2d
    ])
    W = np.diag([w_x1, w_x1d, w_e, w_ed])
    Q = C.T @ W @ C
    return Q


# -----------------------------
# Saturation
# -----------------------------
def saturate(u, u_max, mode="clip"):
    """
    mode="clip" : hard saturation (np.clip)
    mode="tanh" : smooth saturation (u_max * tanh(u/u_max))
    """
    if u_max is None:
        return float(u)

    if mode == "tanh":
        return float(u_max * np.tanh(u / u_max))

    return float(np.clip(u, -u_max, u_max))


# -----------------------------
# Simulation (passive + LQR)
# -----------------------------
def simulate_2dof_with_lqr(
    m1, m2,
    k1, k2,
    c1, c2,
    kc, cd,
    F_ext,
    u_max,
    t_end,
    y0,
    w_x1, w_x1d, w_e, w_ed,
    r_u,
    n_eval=4000,
    method="RK45",
    max_step=1e-2,
    sat_mode="clip",
    rtol=1e-7,
    atol=1e-9,
):
    """
    Passive dynamics:
      x1dd = (-k1*x1 - c1*x1d + cd*(x2d-x1d) + kc*(x2-x1) + F_ext(t)) / m1
      x2dd = (-k2*x2 - c2*x2d - cd*(x2d-x1d) - kc*(x2-x1)) / m2

    With control u on mass 2:
      x2dd = passive_terms/m2 + u/m2

    LQR designed on the same linear model, applied as u = sat(-K y).
    u(t) is computed AFTER integration from the saved trajectory (no rhs logging).
    """
    A, B = build_state_space(m1, m2, k1, k2, c1, c2, kc, cd)
    Q = build_Q_from_outputs(w_x1, w_x1d, w_e, w_ed)
    R = np.array([[r_u]])

    K = lqr_gain(A, B, Q, R)  # u = -K y

    def rhs_passive(t, y):
        x1, x1d, x2, x2d = y
        x1dd = (-k1 * x1 - c1 * x1d + cd * (x2d - x1d) + kc * (x2 - x1) + F_ext(t)) / m1
        x2dd = (-k2 * x2 - c2 * x2d - cd * (x2d - x1d) - kc * (x2 - x1)) / m2
        return [x1d, x1dd, x2d, x2dd]

    def rhs_lqr(t, y):
        x1, x1d, x2, x2d = y

        u = -float(K @ np.array([x1, x1d, x2, x2d]))
        u = saturate(u, u_max, mode=sat_mode)

        x1dd = (-k1 * x1 - c1 * x1d + cd * (x2d - x1d) + kc * (x2 - x1) + F_ext(t)) / m1
        x2dd = (-k2 * x2 - c2 * x2d - cd * (x2d - x1d) - kc * (x2 - x1) + u) / m2
        return [x1d, x1dd, x2d, x2dd]

    t_eval = np.linspace(0.0, t_end, n_eval)

    sol_passive = solve_ivp(
        rhs_passive,
        (0.0, t_end),
        y0,
        t_eval=t_eval,
        method=method,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
    )

    sol_lqr = solve_ivp(
        rhs_lqr,
        (0.0, t_end),
        y0,
        t_eval=t_eval,
        method=method,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
    )

    # Compute u(t) *after* integration on the same time grid
    Y = sol_lqr.y  # shape (4, n_eval)
    u = -(K @ Y).ravel()
    if u_max is not None:
        if sat_mode == "tanh":
            u = u_max * np.tanh(u / u_max)
        else:
            u = np.clip(u, -u_max, u_max)

    return sol_passive, sol_lqr, t_eval, u, K


# -----------------------------
# Run + plots
# -----------------------------
if __name__ == "__main__":
    param_file = "params.txt"
    p = load_params(param_file)

    m1, m2 = p["m1"], p["m2"]
    k1, k2 = p["k1"], p["k2"]
    c1, c2 = p["c1"], p["c2"]
    cd, kc = p["cd"], p["kc"]

    F_ext = make_forcing(p)

    u_max = 10.0
    t_end = p["t_end"]
    y0 = [0.0, 0.0, 0.0, 0.0]

    # Penalization factors
    w_x1, w_x1d = 1.0, 0.1
    w_e, w_ed = 50.0, 2.0
    r_u = 0.05

    sol0, sol1, tu, u, K = simulate_2dof_with_lqr(
        m1, m2,
        k1, k2,
        c1, c2,
        kc, cd,
        F_ext,
        u_max,
        t_end,
        y0,
        w_x1, w_x1d, w_e, w_ed,
        r_u,
        n_eval=4000,
        method="RK45",
        max_step=1e-2,
        sat_mode="clip",   # try "tanh" if you want smooth saturation
    )

    print("LQR gain K (u = -K y) =\n", K)

    # Passive
    t0 = sol0.t
    x1_0, x1d_0, x2_0, x2d_0 = sol0.y

    # LQR
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
    plt.plot(tu, u, ".", label="u(t) computed after solve")
    plt.xlabel("t [s]")
    plt.ylabel("Control force [N]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
