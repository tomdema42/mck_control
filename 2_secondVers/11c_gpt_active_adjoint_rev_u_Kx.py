
# -*- coding: utf-8 -*-
"""
Adjoint optimization of a constant state-feedback gain K (NO saturation):
    u(t) = K @ x(t)
where x = [x1, x1d, x2, x2d].

Minimize:
  J = ∫ [ w_x1 x1^2 + w_x1d x1d^2 + w_e (x1+x2)^2 + w_ed (x1d+x2d)^2 + r_u u^2 ] dt

@author: demaria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from _auxFunc import load_params, make_forcing, rk4_step_11c


# -----------------------------
# Main: adjoint optimization of K (NO saturation)
# -----------------------------
def simulate_2dof_with_adjoint_K_optimization(
        m1, m2,
        k1, k2,
        c1, c2,
        kc, cd,
        F_ext,
        t_end,
        y0,
        N,
        w_x1, w_x1d, w_e, w_ed,
        r_u,
        max_iter,
        K0=None
):
    """
    Returns:
      t, X_passive, X_opt, u_opt_nodes, K_opt, info
    """

    t = np.linspace(0.0, t_end, N + 1)
    dt = t[1] - t[0]

    # State-space (linear 2DOF) for x = [x1, x1d, x2, x2d]
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-(k1 + kc) / m1, -(c1 + cd) / m1,  kc / m1,       cd / m1],
        [0.0, 0.0, 0.0, 1.0],
        [ kc / m2,       cd / m2,      -(k2 + kc) / m2, -(c2 + cd) / m2],
    ])
    B = np.array([0.0, 0.0, 0.0, 1.0 / m2])  # input acts on x2dd

    def forcing_vec(ti):
        return np.array([0.0, F_ext(ti) / m1, 0.0, 0.0])

    # Running cost L(x,u)
    def L(x, u):
        x1, x1d, x2, x2d = x
        e = x1 + x2
        ed = x1d + x2d
        return (
            w_x1 * x1 * x1
            + w_x1d * x1d * x1d
            + w_e * e * e
            + w_ed * ed * ed
            + r_u * u * u
        )

    # d/dx of the state-only part (excluding the u^2 term)
    def dL_state_dx(x):
        x1, x1d, x2, x2d = x
        e = x1 + x2
        ed = x1d + x2d
        d_x1 = 2.0 * w_x1 * x1 + 2.0 * w_e * e
        d_x1d = 2.0 * w_x1d * x1d + 2.0 * w_ed * ed
        d_x2 = 2.0 * w_e * e
        d_x2d = 2.0 * w_ed * ed
        return np.array([d_x1, d_x1d, d_x2, d_x2d])

    # Closed-loop dynamics (NO saturation): u = Kx
    def f_closed(ti, x, K):
        u = float(np.dot(K, x))
        return A @ x + B * u + forcing_vec(ti)

    def forward_simulate(K):
        X = np.zeros((N + 1, 4))
        X[0] = np.array(y0)

        for i in range(N):
            X[i + 1] = rk4_step_11c(f_closed, t[i], X[i], dt, K)

        u_nodes = X @ K
        return X, u_nodes

    def passive_simulate():
        Kz = np.zeros(4)
        X, u_nodes = forward_simulate(Kz)
        u_nodes[:] = 0.0
        return X, u_nodes

    # Adjoint dynamics:
    # lam_dot = - [ dL/dx + (df/dx)^T lam ]
    #
    # With u = Kx:
    #   du/dx = K
    #   df/dx = A + B * (du/dx) = A + outer(B, K)
    #   dL/dx = dL_state/dx + 2*r_u*u*(du/dx) = dL_state/dx + 2*r_u*u*K
    def lam_dot(ti, lam, x, K, u):
        A_eff = A + np.outer(B, K)
        dLdx = dL_state_dx(x) + (2.0 * r_u * u) * K
        return -(dLdx + A_eff.T @ lam)

    def adjoint_backward(X, u_nodes, K):
        Lam = np.zeros((N + 1, 4))
        Lam[N] = 0.0  # terminal condition

        for i in range(N, 0, -1):
            Lam[i - 1] = rk4_step_11c(lam_dot, t[i], Lam[i], -dt, X[i], K, u_nodes[i])

        return Lam

    # Objective + gradient wrt K (adjoint)
    #
    # dJ/dK = ∫ (dH/du) * (du/dK) dt
    # where:
    #   dH/du = dL/du + lam^T df/du = 2*r_u*u + lam^T B
    # and:
    #   u = Kx  =>  du/dK = x  (componentwise)
    #
    # so integrand = (2*r_u*u + lam^T B) * x
    def J_and_grad(K):
        X, u_nodes = forward_simulate(K)
        Lam = adjoint_backward(X, u_nodes, K)

        # Cost (trapezoid)
        J = 0.0
        for i in range(N):
            J += 0.5 * dt * (L(X[i], u_nodes[i]) + L(X[i + 1], u_nodes[i + 1]))

        # Gradient (trapezoid)
        gK = np.zeros(4)
        for i in range(N):
            phi_i = 2.0 * r_u * u_nodes[i] + float(np.dot(Lam[i], B))
            phi_ip1 = 2.0 * r_u * u_nodes[i + 1] + float(np.dot(Lam[i + 1], B))
            gK += 0.5 * dt * (phi_i * X[i] + phi_ip1 * X[i + 1])

        return J, gK

    # Avoid computing forward+adjoint twice per BFGS iteration (fun + jac)
    cache = {"K": None, "J": None, "g": None}

    def fun_cached(K):
        if cache["K"] is not None and np.array_equal(K, cache["K"]):
            return cache["J"]
        J, g = J_and_grad(K)
        cache["K"] = np.array(K, copy=True)
        cache["J"] = J
        cache["g"] = g
        return J

    def jac_cached(K):
        if cache["K"] is not None and np.array_equal(K, cache["K"]):
            return cache["g"]
        J, g = J_and_grad(K)
        cache["K"] = np.array(K, copy=True)
        cache["J"] = J
        cache["g"] = g
        return g

    if K0 is None:
        K0 = np.zeros(4)

    res = minimize(
        fun=fun_cached,
        x0=np.array(K0),
        jac=jac_cached,
        method="BFGS",
        options={"maxiter": max_iter, "gtol": 1e-6}
    )

    K_opt = res.x
    X_passive, u_passive = passive_simulate()
    X_opt, u_opt_nodes = forward_simulate(K_opt)

    info = {
        "success": res.success,
        "message": res.message,
        "nit": res.nit,
        "J": res.fun,
        "K_opt": K_opt
    }

    return t, X_passive, X_opt, u_opt_nodes, K_opt, info


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

    # your forcing factory (keep your own implementation inside _auxFunc)
    F_ext = make_forcing(p)

    t_end = 10.0
    y0 = (0.0, 0.0, 0.0, 0.0)

    N = 400
    w_x1, w_x1d = 1.0, 0.1
    w_e, w_ed = 50.0, 2.0
    r_u = 0.05
    
    max_iter = 500
    K0=np.zeros(4)
    K0_lqr  = [-29.621664, -7.29122777, -16.82680442, -5.68928951]
    K0 = K0

    t, X0, X1, u_nodes, K_opt, info = simulate_2dof_with_adjoint_K_optimization(
        m1, m2,
        k1, k2,
        c1, c2,
        kc, cd,
        F_ext,
        t_end,
        y0,
        N,
        w_x1, w_x1d, w_e, w_ed,
        r_u,
        max_iter,
        K0,
    )

    print(info)
    print("K_opt =", K_opt)

    x1_0, x1d_0, x2_0, x2d_0 = X0.T
    x1_1, x1d_1, x2_1, x2d_1 = X1.T

    plt.figure(figsize=(12, 6))
    plt.plot(t, x1_0, label="x1 passive")
    plt.plot(t, x2_0, label="x2 passive")
    plt.plot(t, x1_1, "--", label="x1 closed-loop (K*)")
    plt.plot(t, x2_1, "--", label="x2 closed-loop (K*)")
    plt.xlabel("t [s]")
    plt.ylabel("Displacement [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(t, u_nodes, label="u(t)=Kx")
    plt.xlabel("t [s]")
    plt.ylabel("Control force [N]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
