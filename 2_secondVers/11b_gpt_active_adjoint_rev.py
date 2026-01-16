# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 14:46:09 2026

@author: demaria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from _auxFunc import load_params, make_forcing
from _auxFunc import rk4_step, rk4_step_adjoint

# -----------------------------
# Main: adjoint optimal control
# -----------------------------
def simulate_2dof_with_adjoint_optimization(
        m1, m2,
        k1, k2,
        c1, c2,
        kc, cd,
        F_ext,
        u_max,
        t_end,
        y0,
        N,
        w_x1, w_x1d, w_e, w_ed,
        r_u,
        max_iter
):
    """
    Optimal control via adjoint:
      minimize J = ∫ [ w_x1 x1^2 + w_x1d x1d^2 + w_e (x1+x2)^2 + w_ed (x1d+x2d)^2 + r_u u^2 ] dt
      subject to dynamics, with u piecewise-constant and -u_max <= u_i <= u_max

    Returns:
      t, X_passive, X_opt, u_opt, info
    """

    t = np.linspace(0.0, t_end, N + 1)
    dt = t[1] - t[0]

    # dynamics f(t, x, u)
    def f(ti, x, ui):
        x1, x1d, x2, x2d = x
        x1dd = (-k1 * x1 - c1 * x1d + cd * (x2d - x1d) + kc * (x2 - x1) + F_ext(ti)) / m1
        x2dd = (-k2 * x2 - c2 * x2d - cd * (x2d - x1d) - kc * (x2 - x1) + ui) / m2
        return np.array([x1d, x1dd, x2d, x2dd])

    # running cost L(x,u)
    def L(x, ui):
        x1, x1d, x2, x2d = x
        e = x1 + x2
        ed = x1d + x2d
        return (
            w_x1 * x1*x1
            + w_x1d * x1d*x1d
            + w_e * e*e
            + w_ed * ed*ed
            + r_u * ui*ui
        )

    # dL/dx
    def dL_dx(x):
        x1, x1d, x2, x2d = x
        e = x1 + x2
        ed = x1d + x2d

        d_x1  = 2*w_x1*x1 + 2*w_e*e
        d_x1d = 2*w_x1d*x1d + 2*w_ed*ed
        d_x2  = 2*w_e*e
        d_x2d = 2*w_ed*ed
        return np.array([d_x1, d_x1d, d_x2, d_x2d])

    # dL/du
    def dL_du(ui):
        return 2.0 * r_u * ui

    # A = df/dx 
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-(k1+kc)/m1, -(c1+cd)/m1,  kc/m1,       cd/m1],
        [0.0, 0.0, 0.0, 1.0],
        [ kc/m2,       cd/m2,      -(k2+kc)/m2, -(c2+cd)/m2],
    ])

    # B = df/du
    B = np.array([0.0, 0.0, 0.0, 1.0/m2])

    # adjoint dynamics: lam_dot = - (dL/dx + A^T lam)
    def lam_dot(ti, lam, x, ui):
        return -(dL_dx(x) + A.T @ lam)

    def forward_simulate(u_vals):
        X = np.zeros((N + 1, 4))
        X[0] = np.array(y0)
        for i in range(N):
            X[i+1] = rk4_step(f, t[i], X[i], dt, u_vals[i])
        return X

    def passive_simulate():
        X = np.zeros((N + 1, 4))
        X[0] = np.array(y0)
        for i in range(N):
            X[i+1] = rk4_step(f, t[i], X[i], dt, 0.0)
        return X

    def adjoint_backward(X, u_vals):
        Lam = np.zeros((N + 1, 4))
        Lam[N] = 0.0
        for i in range(N, 0, -1):
            Lam[i-1] = rk4_step_adjoint(lam_dot, t[i], Lam[i], -dt, X[i], u_vals[i-1])
        return Lam

    # objective + gradient wrt u directly
    def J_and_grad(u_vals):
        X = forward_simulate(u_vals)
        Lam = adjoint_backward(X, u_vals)

        J = 0.0
        for i in range(N):
            J += 0.5 * dt * (L(X[i], u_vals[i]) + L(X[i+1], u_vals[i]))

        g_u = np.zeros(N)
        for i in range(N):
            g_u[i] = dt * (dL_du(u_vals[i]) + Lam[i] @ B)
        return J, g_u

    u0 = np.zeros(N)
    bounds = [(-u_max, u_max)] * N

    res = minimize(
        fun=lambda u: J_and_grad(u)[0],
        x0=u0,
        jac=lambda u: J_and_grad(u)[1],
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-9, "gtol": 1e-6},
    )

    u_opt = res.x

    X_passive = passive_simulate()
    X_opt = forward_simulate(u_opt)

    info = {"success": res.success, "message": res.message, "nit": res.nit, "J": res.fun}
    return t, X_passive, X_opt, u_opt, info


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
    t_end = 10.0
    y0 = (0.0, 0.0, 0.0, 0.0)
    
    N = 400                 #Number of discretizations in time, TO change with dt
    #Penalization factors
    w_x1, w_x1d =1.0, 0.1 # Penalization on x1 displacement and velocity
    w_e, w_ed  = 50.0, 2.0 # Enforcement on the antiphase e = x1+x2
    
    r_u  = 0.05 #Penalization on the control ui^2
        
    max_iter = 6000 #maxiter for the minimization.
         
    
    t, X0, X1, u, info = simulate_2dof_with_adjoint_optimization(
        m1, m2,
        k1, k2,
        c1, c2,
        kc, cd,
        F_ext,
        u_max,
        t_end,
        y0,
        N,
        w_x1, w_x1d, w_e, w_ed,
        r_u,
        max_iter,
    )

    print(info)

    x1_0, x1d_0, x2_0, x2d_0 = X0.T
    x1_1, x1d_1, x2_1, x2d_1 = X1.T

    plt.figure(figsize=(12, 6))
    plt.plot(t, x1_0, label="x1 passive")
    plt.plot(t, x2_0, label="x2 passive")
    plt.plot(t, x1_1, "--", label="x1 optimal (adjoint)")
    plt.plot(t, x2_1, "--", label="x2 optimal (adjoint)")
    plt.xlabel("t [s]")
    plt.ylabel("Displacement [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(t[:-1], u, label="u(t) optimal")
    plt.xlabel("t [s]")
    plt.ylabel("Control force [N]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    