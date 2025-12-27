# -*- coding: utf-8 -*-
"""
Optimize the stiffness k2 of a 2-DOF mass-spring-damper system to minimize the objective function
J(k2) = ∫ x1(t)^2 dt + reg_k2 * ∫ k2 dt.

@author: demaria
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from _auxFunc import load_params, make_forcing, build_canonical_params


def simulate_from_params(p):
    m1, m2 = p["m1"], p["m2"]
    k1, k2 = p["k1"], p["k2"]
    c1, c2 = p["c1"], p["c2"]
    cd, kc = p["cd"], p["kc"]

    F_ext = make_forcing(p)

    y0 = (p["x1_0"], p["x1d_0"], p["x2_0"], p["x2d_0"])
    t_eval = np.linspace(p["t0"], p["t_end"], int(p["n_points"]))

    def rhs(t, y):
        x1, x1d, x2, x2d = y
        
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


def objective_J_of_k2(k2, p_base,reg_k2):
    p = dict(p_base)          # shallow copy
    p["k2"] = float(k2)       # override decision variable

    t, y = simulate_from_params(p)
    x1 = y[0]  
    
    integrand = x1**2
    J_x1 = np.trapezoid(integrand, t)
    J_k2 = np.trapezoid([float(k2)]*len(t),t)
    
    J_tot = J_x1 + J_k2*reg_k2
    return J_tot


def optimize_k2(param_file, k2_bounds,reg_k2, mode="x1_sq"):
    p_base = load_params(param_file)

    def fun(k2):
        return objective_J_of_k2(k2, p_base,reg_k2)

    res = minimize_scalar(fun, bounds=k2_bounds, method="bounded")

    if not res.success:
        raise RuntimeError(res.message)

    k2_star = float(res.x)
    J_star = float(res.fun)

    # simulate optimal trajectory (useful to plot)
    p_opt = dict(p_base)
    p_opt["k2"] = k2_star
    t, y = simulate_from_params(p_opt)

    return k2_star, J_star, t, y


if __name__ == "__main__":
    # choose bounds for k2 (example)
    param_file = "params.txt"
    k2_min = 1
    k2_max = 1e6
    reg_k2 = 1e-6
    k2_star, J_star, t, y = optimize_k2(param_file, (k2_min, k2_max),reg_k2)
    x1, x1d, x2, x2d = y

    print(f"Optimal k2 = {k2_star}")
    print(f"Objective J(k2*) = {J_star}")
    _ = build_canonical_params(load_params(param_file),verbose=True)
    
    plt.figure()
    plt.plot(t, x1, label="x1 (optimal)")
    plt.plot(t, x2, label="x2 (optimal)")
    plt.xlabel("t [s]")
    plt.ylabel("Displacement [m]")
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(t, x1d, label="x1d (optimal)")
    plt.plot(t, x2d, label="x2d (optimal)")
    plt.xlabel("t [s]")
    plt.ylabel("Velocity [m/s]")
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.show()
