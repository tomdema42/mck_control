# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 14:30:42 2026

@author: demaria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from _auxFunc import load_params, make_forcing
chSize = 18
plt.rcParams.update({ 'font.size': chSize,  })        # Base font size


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
    m1, m2,
    k1, k2,
    c1, c2,
    kc, cd,
    F_ext,
    u_max,
    t_end,
    y0,
    # weights for Q via outputs: [x1, x1d, e=x1+x2, ed=x1d+x2d]
    w_x1, w_x1d, w_e, w_ed,
    r_u,
    max_step=1e-2,
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


    def rhs_passive(t, y):
        x1, x1d, x2, x2d = y
        x1dd = (-k1 * x1 - c1 * x1d + cd * (x2d - x1d) + kc * (x2 - x1) + F_ext(t)) / m1
        x2dd = (-k2 * x2 - c2 * x2d - cd * (x2d - x1d) - kc * (x2 - x1)) / m2
        return [x1d, x1dd, x2d, x2dd]

    def rhs_lqr(t, y):
        x1, x1d, x2, x2d = y

        # LQR control law
        u = -float(K @ np.array([x1, x1d, x2, x2d]))

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
    Y = sol_lqr.y
    u_export = -(K @ Y).ravel()
    
    return sol_passive, sol_lqr, u_export, K


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

    u_max = 10
    t_end = p['t_end']
    y0 = (0.0, 0.0, 0.0, 0.0)
    
    #Penalization factors
    w_x1, w_x1d =1.0, 0.1 # Penalization on x1 displacement and velocity
    w_e, w_ed  = 50.0, 2.0 # Enforcement on the antiphase e = x1+x2
    
    r_u  = 1.05 #Penalization on the control ui^2
    
    sol0, sol1, u_export, K = simulate_2dof_with_lqr(
        m1, m2,
        k1, k2,
        c1, c2,
        kc, cd,
        F_ext,
        u_max,
        t_end,
        y0,
        # try tuning these:
        w_x1, w_x1d, w_e, w_ed,
        r_u,
    )

    print("LQR gain K =", -K)

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
    
    t_u = np.linspace(0,t_end,len(u_export)) 
    plt.figure(figsize=(12, 4))
    plt.plot(t_u,u_export,'.', label="u(t)")
    plt.xlabel("t [s]")
    plt.ylabel("Control force [N]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()