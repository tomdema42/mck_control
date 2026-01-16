# -*- coding: utf-8 -*-
"""
2-DOF mass-spring-damper simulation with a GIVEN state-feedback gain K.

State: y = [x1, x1d, x2, x2d]
Control: u(t) = sat( -K y )

Key point:
- Do NOT log u inside rhs() when using solve_ivp.
- Integrate with t_eval, then compute u(t) afterward from the saved trajectory.

@author: demaria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from _auxFunc import load_params, make_forcing


def saturate(u, u_max, mode="clip"):
    if u_max is None:
        return float(u)
    if mode == "tanh":
        return float(u_max * np.tanh(u / u_max))
    return float(np.clip(u, -u_max, u_max))


def simulate_2dof_given_K(
    K,
    m1, m2,
    k1, k2,
    c1, c2,
    kc, cd,
    F_ext,
    t_end,
    y0,
    u_max=10.0,
    n_eval=4000,
    method="RK45",
    max_step=1e-2,
    sat_mode="clip",
    rtol=1e-7,
    atol=1e-9,
):
    """
    Simulate closed-loop dynamics with a given K.

    K:
      - shape (4,) or (1,4)
      - uses u = -K @ y

    Returns:
      t_eval, Y (4,n), u (n,), sol (solve_ivp object)
    """
    K = np.asarray(K).reshape(1, 4)

    def rhs(t, y):
        x1, x1d, x2, x2d = y

        u = -float(K @ np.array([x1, x1d, x2, x2d]))
        u = saturate(u, u_max, mode=sat_mode)

        x1dd = (-k1 * x1 - c1 * x1d + cd * (x2d - x1d) + kc * (x2 - x1) + F_ext(t)) / m1
        x2dd = (-k2 * x2 - c2 * x2d - cd * (x2d - x1d) - kc * (x2 - x1) + u) / m2

        return [x1d, x1dd, x2d, x2dd]

    t_eval = np.linspace(0.0, t_end, n_eval)

    sol = solve_ivp(
        rhs,
        (0.0, t_end),
        y0,
        t_eval=t_eval,
        method=method,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
    )

    Y = sol.y  # (4, n_eval)

    # compute u(t) AFTER integration on the same grid
    u = -(K @ Y).ravel()
    if u_max is not None:
        if sat_mode == "tanh":
            u = u_max * np.tanh(u / u_max)
        else:
            u = np.clip(u, -u_max, u_max)

    return t_eval, Y, u, sol


if __name__ == "__main__":
    # --- load params ---
    param_file = "params.txt"
    p = load_params(param_file)

    m1, m2 = p["m1"], p["m2"]
    k1, k2 = p["k1"], p["k2"]
    c1, c2 = p["c1"], p["c2"]
    cd, kc = p["cd"], p["kc"]

    F_ext = make_forcing(p)

    t_end = p["t_end"]
    y0 = [0.0, 0.0, 0.0, 0.0]
    u_max = 10.0

    # Example K (replace with yours)
    # If you have K from LQR code, it is already (1,4) typically.
    K_adj = [11.65391309, 20.57484076,  0.09192578, 10.18879521]
    K_lqr = [29.621664,    7.29122777, 16.82680442 , 5.68928951]
    
    K_in = K_lqr

    t, Y, u, sol = simulate_2dof_given_K(
        K=K_in,
        m1=m1, m2=m2,
        k1=k1, k2=k2,
        c1=c1, c2=c2,
        kc=kc, cd=cd,
        F_ext=F_ext,
        t_end=t_end,
        y0=y0,
        u_max=u_max,
        n_eval=4000,
        sat_mode="clip",  # or "tanh"
    )

    x1, x1d, x2, x2d = Y

    plt.figure(figsize=(12, 6))
    plt.plot(t, x1, label="x1")
    plt.plot(t, x2, label="x2")
    plt.xlabel("t [s]")
    plt.ylabel("Displacement [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(t, u, label="u(t)")
    plt.xlabel("t [s]")
    plt.ylabel("Control force [N]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
