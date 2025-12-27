# -*- coding: utf-8 -*-
"""
Optimize the parameter alpha defining the nonlinear stiffness k2 in a 2-DOF mass-spring-damper system
to minimize the objective function J(alpha) = ∫ x1(t) dt + lambda_k2 * ∫ k2(t) dt,
where k2(t) = alpha * |x2d(t) - x1d(t)|.

@author: demaria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from _auxFunc import load_params, make_forcing


def simulate_from_file(param_file, alpha):

    p = load_params(param_file)

    m1, m2 = p["m1"], p["m2"]
    k1 = p["k1"]
    k2_nom = p['k2_nom']
    c1, c2 = p["c1"], p["c2"]
    cd, kc = p["cd"], p["kc"]

    F_ext = make_forcing(p)

    y0 = (p["x1_0"], p["x1d_0"], p["x2_0"], p["x2d_0"])
    t_eval = np.linspace(p["t0"], p["t_end"], int(p["n_points"]))

    def rhs(t, y):
        x1, x1d, x2, x2d = y
        
        k2 = k2_nom +  alpha * np.abs(x2d - x1d)

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

    # reconstruct k2(t)
    x1d = sol.y[1]
    x2d = sol.y[3]
    k2_t = alpha * np.abs(x2d - x1d)

    return sol.t, sol.y, k2_t


def cost_alpha(alpha, param_file, lambda_k2):

    if alpha < 0:
        return np.inf

    try:
        t, y, k2_t = simulate_from_file(param_file, alpha)
        x1 = y[0]

        J_x1 = np.trapezoid(x1, t)
        J_k2 = np.trapezoid(k2_t, t)

        J = J_x1 + lambda_k2 * J_k2

        if not np.isfinite(J):
            return np.inf

        return J

    except Exception:
        return np.inf


if __name__ == "__main__":

    param_file = "params.txt"
    lambda_k2 = 1e-6     # <<< tuning parameter

    alpha_bounds = (0.0, 2000.0)

    res = minimize_scalar(
        cost_alpha,
        bounds=alpha_bounds,
        args=(param_file, lambda_k2),
        method="bounded",
        options={"xatol": 1e-3, "maxiter": 80},
    )

    if not res.success:
        raise RuntimeError(res.message)

    alpha_best = res.x
    print(f"Optimal alpha = {alpha_best:.6g}")

    # Final simulation
    t, y, k2_t = simulate_from_file(param_file, alpha_best)
    x1, x1d, x2, x2d = y

    # Displacements
    plt.figure()
    plt.plot(t, x1, label="x1")
    plt.plot(t, x2, label="x2")
    plt.xlabel("t [s]")
    plt.ylabel("Displacement [m]")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Velocities
    plt.figure()
    plt.plot(t, x1d, label="x1d")
    plt.plot(t, x2d, label="x2d")
    plt.xlabel("t [s]")
    plt.ylabel("Velocity [m/s]")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # k2(t)
    plt.figure()
    plt.plot(t, k2_t, label="k2(t)")
    plt.xlabel("t [s]")
    plt.ylabel("Nonlinear stiffness k2")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.show()
