# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 13:10:22 2025

@author: demaria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from _auxFunc import load_params, make_forcing


def simulate_from_file(param_file, alpha):

    p = load_params(param_file)

    m1, m2 = p["m1"], p["m2"]
    k1 = p["k1"]
    c1, c2 = p["c1"], p["c2"]
    cd, kc = p["cd"], p["kc"]

    F_ext = make_forcing(p)

    y0 = (
        p["x1_0"],
        p["x1d_0"],
        p["x2_0"],
        p["x2d_0"],
    )

    t_eval = np.linspace(p["t0"], p["t_end"], int(p["n_points"]))

    def rhs(t, y):
        x1, x1d, x2, x2d = y

        # nonlinear stiffness defined by alpha
        k2 = alpha * np.abs(x2d-x1d)

        x1dd = (-k1*x1 - c1*x1d + cd*(x2d - x1d) + kc*(x2 - x1) + F_ext(t)) / m1
        x2dd = (-k2*x2 - c2*x2d - cd*(x2d - x1d) - kc*(x2 - x1)) / m2

        return [x1d, x1dd, x2d, x2dd]

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

    return sol.t, sol.y


if __name__ == "__main__":

    
    alpha = 300

    t, y = simulate_from_file("params.txt", alpha)

    x1, x1d, x2, x2d = y

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
