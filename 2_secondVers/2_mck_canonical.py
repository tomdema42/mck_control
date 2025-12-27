# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 13:30:15 2025


NOT WITH KC

@author: demaria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from _auxFunc import load_params,make_forcing,build_canonical_params





def simulate_canonical_from_file(param_file):
    p = load_params(param_file)

    omega1, omega2, zeta1, zeta2, zeta_d1, zeta_d2 = build_canonical_params(p)

    # forcing (still appears as F/m1 in canonical form)
    F_ext = make_forcing(p)
    m1 = p["m1"]

    # initial conditions
    y0 = (
        p.get("x1_0", 0.0),
        p.get("x1d_0", 0.0),
        p.get("x2_0", 0.0),
        p.get("x2d_0", 0.0),
    )

    t0 = p.get("t0", 0.0)
    t_end = p.get("t_end", 10.0)
    n_points = int(p.get("n_points", 4000))
    t_eval = np.linspace(t0, t_end, n_points)

    def rhs(t, y):
        x1, x1d, x2, x2d = y

        # canonical coupled form
        x1dd = (
            -2.0 * (zeta1 + zeta_d1) * omega1 * x1d
            - (omega1 ** 2) * x1
            + 2.0 * zeta_d1 * omega1 * x2d
            + F_ext(t) / m1
        )

        x2dd = (
            -2.0 * (zeta2 + zeta_d2) * omega2 * x2d
            - (omega2 ** 2) * x2
            + 2.0 * zeta_d2 * omega2 * x1d
        )

        return [x1d, x1dd, x2d, x2dd]

    sol = solve_ivp(
        rhs,
        (t0, t_end),
        y0,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    meta = {
        "omega1": omega1,
        "omega2": omega2,
        "zeta1": zeta1,
        "zeta2": zeta2,
        "zeta_d1": zeta_d1,
        "zeta_d2": zeta_d2,
        "zeta_eff1": zeta1 + zeta_d1,
        "zeta_eff2": zeta2 + zeta_d2,
    }

    return sol.t, sol.y, meta


if __name__ == "__main__":
    t, y, meta = simulate_canonical_from_file("params.txt")
    x1, x1d, x2, x2d = y
    print("Canonical parameters:")
    for k in ["omega1", "omega2", "zeta1", "zeta2", "zeta_d1", "zeta_d2", "zeta_eff1", "zeta_eff2"]:
        print(f"  {k}: {meta[k]}")


    plt.figure()
    plt.plot(t, x1, label="x1")
    plt.plot(t, x2, label="x2")
    plt.xlabel("t [s]")
    plt.ylabel("displacement")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.figure()
    plt.plot(t, x1d, label="x1d")
    plt.plot(t, x2d, label="x2d")
    plt.xlabel("t [s]")
    plt.ylabel("velocity")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.show()
