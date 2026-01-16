# -*- coding: utf-8 -*-
"""
Simulate a 2-DOF mass-spring-damper system with LQR-derived adaptive stiffness k2(t).

@author: demaria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from _auxFunc import load_params, make_forcing


def simulate_from_file(param_file):

    # ---------------------------------------------------------
    # Load parameters
    # ---------------------------------------------------------
    p = load_params(param_file)

    m1, m2 = p["m1"], p["m2"]
    k1 = p["k1"]
    k2_nom = p["k2_nom"]          # <-- must be in params.txt
    c1, c2 = p["c1"], p["c2"]
    cd, kc = p["cd"], p["kc"]

    # Bounds for k2
    k2_min = p.get("k2_min", 10.0)
    k2_max = p.get("k2_max", 500.0)

    F_ext = make_forcing(p)

    # Initial conditions
    y0 = (
        p["x1_0"],
        p["x1d_0"],
        p["x2_0"],
        p["x2d_0"],
    )

    t_eval = np.linspace(p["t0"], p["t_end"], int(p["n_points"]))

    # ---------------------------------------------------------
    # Build linearized model for LQR design
    # ---------------------------------------------------------
    A = np.array([
        [0,            1,              0,            0],
        [-(k1+kc)/m1, -(c1+cd)/m1,     kc/m1,        cd/m1],
        [0,            0,              0,            1],
        [kc/m2,        cd/m2, -(k2_nom+kc)/m2, -(c2+cd)/m2]
    ])

    B = np.array([
        [0],
        [0],
        [0],
        [1/m2]
    ])

    # LQR cost matrices
    Q = np.diag([1e3, 0.0, 0.0, 0.0])   # penalize x1 displacement
    R = np.array([[100]])             # penalize control effort

    # Solve Riccati equation
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P     # 1x4 gain

    # ---------------------------------------------------------
    # LQR-based k2(t) law
    # ---------------------------------------------------------
    def k2_lqr(y):
        x1, x1d, x2, x2d = y
        x_vec = np.array([[x1], [x1d], [x2], [x2d]])

        u = float(-K @ x_vec)  # optimal control

        eps = 1e-4
        if abs(x2) < eps:
            return k2_nom

        k2 = k2_nom + u / x2
        return np.clip(k2, k2_min, k2_max)

    # ---------------------------------------------------------
    # Full nonlinear dynamics
    # ---------------------------------------------------------
    def rhs(t, y):
        x1, x1d, x2, x2d = y

        k2 = k2_lqr(y)

        x1dd = (-k1*x1 - c1*x1d + cd*(x2d - x1d) + kc*(x2 - x1) + F_ext(t)) / m1
        x2dd = (-k2*x2 - c2*x2d - cd*(x2d - x1d) - kc*(x2 - x1)) / m2

        return [x1d, x1dd, x2d, x2dd]

    # ---------------------------------------------------------
    # Integrate
    # ---------------------------------------------------------
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

    return sol.t, sol.y, K, k2_lqr


if __name__ == "__main__":

    t, y, K, k2_lqr = simulate_from_file("params.txt")

    x1, x1d, x2, x2d = y

    print("LQR gain K =", K)

    plt.figure()
    plt.plot(t, x1, label="x1")
    plt.plot(t, x2, label="x2")
    plt.xlabel("t [s]")
    plt.ylabel("Displacement [m]")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.figure()
    plt.plot(t, x1d, label="x1d")
    plt.plot(t, x2d, label="x2d")
    plt.xlabel("t [s]")
    plt.ylabel("Velocity [m/s]")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.show()
    # Compute k2(t) over the trajectory
    k2_time = np.array([k2_lqr(y[:, i]) for i in range(y.shape[1])])
    
    plt.figure()
    plt.plot(t, k2_time, label="k2(t)")
    plt.xlabel("t [s]")
    plt.ylabel("k2(t)")
    plt.title("Evolution of k2(t)")
    plt.grid()
    plt.tight_layout()
    plt.show()
    
