# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 17:42:19 2026

@author: demaria
"""

# -*- coding: utf-8 -*-
"""
k2 = alpha * u  (u = -K x)
Because k2 depends on u, the plant is nonlinear (u*x2 term).
We provide:
  (1) Linearized LQR around x=0,u=0  -> K_lin (CARE)
  (2) iLQR on the nonlinear plant    -> time-varying gains + best constant K_fit

@author: demaria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from _auxFunc import load_params, make_forcing


# -----------------------------
# LQR helper (continuous-time)
# -----------------------------
def lqr_gain(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)  # u = -K x
    return K


def build_Q_from_weights(w_x1, w_x1d, w_e, w_ed):
    # z = [x1, x1d, e=x1+x2, ed=x1d+x2d]
    C = np.array([
        [1.0, 0.0, 0.0, 0.0],  # x1
        [0.0, 1.0, 0.0, 0.0],  # x1d
        [1.0, 0.0, 1.0, 0.0],  # e
        [0.0, 1.0, 0.0, 1.0],  # ed
    ])
    W = np.diag([w_x1, w_x1d, w_e, w_ed])
    Q = C.T @ W @ C
    return Q


# -----------------------------
# Nonlinear plant: k2 = alpha*u
# -----------------------------
def rhs_k2_alpha_u(t, y, K, alpha, m1, m2, k1, c1, c2, kc, cd, F_ext, u_max=None):
    x1, x1d, x2, x2d = y

    # control law
    u = -float(K @ np.array([x1, x1d, x2, x2d]))
    if u_max is not None:
        u = float(np.clip(u, -u_max, u_max))

    # stiffness law
    k2_t = alpha * u

    # dynamics
    x1dd = (-k1 * x1 - c1 * x1d + cd * (x2d - x1d) + kc * (x2 - x1) + F_ext(t)) / m1

    # NOTE: u appears both as force (+u) and via k2_t*x2 (because k2_t depends on u)
    x2dd = (-k2_t * x2 - c2 * x2d - cd * (x2d - x1d) - kc * (x2 - x1) + u) / m2

    return [x1d, x1dd, x2d, x2dd]


def simulate_nonlinear_with_constant_K(
    K, alpha,
    m1, m2, k1, c1, c2, kc, cd,
    F_ext, u_max,
    t_end, y0,
    max_step=1e-2
):
    def rhs(t, y):
        return rhs_k2_alpha_u(t, y, K, alpha, m1, m2, k1, c1, c2, kc, cd, F_ext, u_max=u_max)

    sol = solve_ivp(rhs, (0.0, t_end), y0, max_step=max_step, rtol=1e-7, atol=1e-9)

    Y = sol.y
    u = -(K @ Y).ravel()
    if u_max is not None:
        u = np.clip(u, -u_max, u_max)

    k2_t = alpha * u
    return sol, u, k2_t


# -----------------------------
# Discrete-time helpers for iLQR
# -----------------------------
def dyn_continuous(t, x, u, alpha, m1, m2, k1, c1, c2, kc, cd, F_ext):
    x1, x1d, x2, x2d = x
    k2_t = alpha * u

    x1dd = (-k1 * x1 - c1 * x1d + cd * (x2d - x1d) + kc * (x2 - x1) + F_ext(t)) / m1
    x2dd = (-k2_t * x2 - c2 * x2d - cd * (x2d - x1d) - kc * (x2 - x1) + u) / m2
    return np.array([x1d, x1dd, x2d, x2dd])


def rk4_step(x, u, t, dt, alpha, m1, m2, k1, c1, c2, kc, cd, F_ext):
    k1v = dyn_continuous(t, x, u, alpha, m1, m2, k1, c1, c2, kc, cd, F_ext)
    k2v = dyn_continuous(t + 0.5 * dt, x + 0.5 * dt * k1v, u, alpha, m1, m2, k1, c1, c2, kc, cd, F_ext)
    k3v = dyn_continuous(t + 0.5 * dt, x + 0.5 * dt * k2v, u, alpha, m1, m2, k1, c1, c2, kc, cd, F_ext)
    k4v = dyn_continuous(t + dt, x + dt * k3v, u, alpha, m1, m2, k1, c1, c2, kc, cd, F_ext)
    return x + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)


def rollout_discrete(x0, U, dt, t0, Q, R, Qf,
                     alpha, m1, m2, k1, c1, c2, kc, cd, F_ext,
                     u_clip=None):
    N = len(U)
    X = np.zeros((N + 1, 4))
    X[0] = x0
    cost = 0.0
    t = t0

    for k in range(N):
        u = float(U[k])
        if u_clip is not None:
            u = float(np.clip(u, -u_clip, u_clip))

        x = X[k]
        cost += dt * (x @ Q @ x + R * (u * u))

        X[k + 1] = rk4_step(x, u, t, dt, alpha, m1, m2, k1, c1, c2, kc, cd, F_ext)
        t += dt

    xN = X[-1]
    cost += xN @ Qf @ xN
    return X, cost


def linearize_discrete_step(x, u, t, dt,
                            alpha, m1, m2, k1, c1, c2, kc, cd, F_ext,
                            eps=1e-6):
    # x_{k+1} = Phi(x_k, u_k)
    x_next = rk4_step(x, u, t, dt, alpha, m1, m2, k1, c1, c2, kc, cd, F_ext)

    A = np.zeros((4, 4))
    for i in range(4):
        dx = np.zeros(4)
        dx[i] = eps
        xp = rk4_step(x + dx, u, t, dt, alpha, m1, m2, k1, c1, c2, kc, cd, F_ext)
        xm = rk4_step(x - dx, u, t, dt, alpha, m1, m2, k1, c1, c2, kc, cd, F_ext)
        A[:, i] = (xp - xm) / (2.0 * eps)

    # scalar u
    up = rk4_step(x, u + eps, t, dt, alpha, m1, m2, k1, c1, c2, kc, cd, F_ext)
    um = rk4_step(x, u - eps, t, dt, alpha, m1, m2, k1, c1, c2, kc, cd, F_ext)
    B = ((up - um) / (2.0 * eps)).reshape(4, 1)

    return A, B, x_next


def ilqr_k2_alpha_u(
    x0, t_end, dt,
    Q, R, Qf,
    alpha, m1, m2, k1, c1, c2, kc, cd, F_ext,
    u_clip=None,
    max_iter=50,
    tol=1e-6,
    mu=1e-6
):
    N = int(np.round(t_end / dt))
    t0 = 0.0

    # initial guess: zero control
    U = np.zeros(N)

    # nominal rollout
    X, J = rollout_discrete(x0, U, dt, t0, Q, R, Qf,
                            alpha, m1, m2, k1, c1, c2, kc, cd, F_ext,
                            u_clip=u_clip)

    K_list = np.zeros((N, 1, 4))   # feedback gains (time-varying)
    kff_list = np.zeros(N)         # feedforward terms

    for it in range(max_iter):
        # linearize along trajectory
        A_list = np.zeros((N, 4, 4))
        B_list = np.zeros((N, 4, 1))

        t = t0
        for k in range(N):
            A_k, B_k, _ = linearize_discrete_step(
                X[k], float(U[k]), t, dt,
                alpha, m1, m2, k1, c1, c2, kc, cd, F_ext
            )
            A_list[k] = A_k
            B_list[k] = B_k
            t += dt

        # backward pass
        Vx = 2.0 * (Qf @ X[-1])
        Vxx = 2.0 * Qf

        for k in range(N - 1, -1, -1):
            xk = X[k]
            uk = float(U[k])

            lx = 2.0 * (Q @ xk) * dt
            lu = 2.0 * (R * uk) * dt
            lxx = 2.0 * Q * dt
            luu = 2.0 * R * dt  # scalar
            # lxu = 0

            A = A_list[k]
            B = B_list[k]

            Qx = lx + A.T @ Vx
            Qu = lu + float(B.T @ Vx)

            Qxx = lxx + A.T @ Vxx @ A
            Qux = (B.T @ Vxx @ A)  # shape (1,4)
            Quu = luu + float(B.T @ Vxx @ B)

            # regularize Quu (scalar)
            Quu_reg = Quu + mu
            inv_Quu = 1.0 / Quu_reg

            kff = -inv_Quu * Qu
            Kfb = -inv_Quu * Qux  # (1,4)

            kff_list[k] = kff
            K_list[k] = Kfb

            # value function update
            Vx = Qx + (Kfb.T * Quu)[:, 0] * kff + (Kfb.T[:, 0] * Qu) + (Qux.T[:, 0] * kff)
            Vxx = Qxx + (Kfb.T @ (Quu * Kfb)) + (Kfb.T @ Qux) + (Qux.T @ Kfb)
            Vxx = 0.5 * (Vxx + Vxx.T)

        # forward line-search
        accepted = False
        for a in [1.0, 0.5, 0.25, 0.1, 0.05]:
            X_new = np.zeros_like(X)
            U_new = np.zeros_like(U)
            X_new[0] = x0
            t = t0
            for k in range(N):
                dx = X_new[k] - X[k]
                u = float(U[k] + a * kff_list[k] + float(K_list[k] @ dx))
                if u_clip is not None:
                    u = float(np.clip(u, -u_clip, u_clip))
                U_new[k] = u
                X_new[k + 1] = rk4_step(X_new[k], u, t, dt, alpha, m1, m2, k1, c1, c2, kc, cd, F_ext)
                t += dt

            _, J_new = rollout_discrete(x0, U_new, dt, t0, Q, R, Qf,
                                        alpha, m1, m2, k1, c1, c2, kc, cd, F_ext,
                                        u_clip=u_clip)

            if J_new < J:
                X, U, J = X_new, U_new, J_new
                accepted = True
                break

        if not accepted:
            mu *= 10.0
        else:
            mu = max(1e-9, mu / 2.0)

        if it > 0 and abs(J_new - J) < tol:
            break

    # Fit a constant K such that u ≈ -K x along the optimized trajectory
    X_fit = X[:-1].T  # (4, N)
    U_fit = U.reshape(-1, 1)  # (N,1)
    # Solve X^T k = -U  (least squares)
    k_vec, *_ = np.linalg.lstsq(X_fit.T, -U_fit, rcond=None)
    K_const = k_vec.T  # (1,4)

    t_grid = np.linspace(0.0, t_end, N + 1)
    return t_grid, X, U, K_list, K_const, J


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    param_file = "params.txt"
    p = load_params(param_file)

    m1, m2 = p["m1"], p["m2"]
    k1 = p["k1"]
    c1, c2 = p["c1"], p["c2"]
    cd, kc = p["cd"], p["kc"]

    F_ext = make_forcing(p)

    # user settings
    u_max = 10.0
    t_end = p["t_end"]
    y0 = np.array([0.0, 0.0, 0.0, 0.0])

    # cost weights (same structure as your current code)
    w_x1, w_x1d = 1.0, 0.1
    w_e, w_ed = 50.0, 2.0
    r_u = 0.05

    # NEW: alpha in k2 = alpha * u
    alpha = 0.2   # tune this

    Q = build_Q_from_weights(w_x1, w_x1d, w_e, w_ed)
    R = float(r_u)
    Qf = Q.copy()

    # -------------------------------------------------------
    # (1) Linearized LQR around x=0, u=0  -> k2=0 at equilibrium
    # -------------------------------------------------------
    # Linearization uses k2=0 (because u=0 at equilibrium => k2=alpha*u=0)
    A_lin = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-(k1 + kc) / m1, -(c1 + cd) / m1,  kc / m1,         cd / m1],
        [0.0, 0.0, 0.0, 1.0],
        [ kc / m2,         cd / m2,        -(0.0 + kc) / m2, -(c2 + cd) / m2],
    ])
    B_lin = np.array([[0.0], [0.0], [0.0], [1.0 / m2]])
    K_lin = lqr_gain(A_lin, B_lin, Q, np.array([[R]]))

    sol_lin, u_lin, k2_lin = simulate_nonlinear_with_constant_K(
        K_lin, alpha,
        m1, m2, k1, c1, c2, kc, cd,
        F_ext, u_max,
        t_end, y0,
        max_step=1e-2
    )

    print("K_lin (LQR on linearization), control law u=-Kx -> K =", K_lin)

    # -------------------------------------------------------
    # (2) iLQR on the nonlinear plant, then fit a constant K
    # -------------------------------------------------------
    dt = 1e-2
    t_grid, X_opt, U_opt, K_tv, K_fit, J_opt = ilqr_k2_alpha_u(
        y0, t_end, dt,
        Q, R, Qf,
        alpha, m1, m2, k1, c1, c2, kc, cd, F_ext,
        u_clip=u_max,
        max_iter=40
    )

    print("K_fit (best constant K fitted from iLQR traj), u=-Kx -> K =", K_fit)
    print("Final iLQR cost J =", J_opt)

    sol_fit, u_fit, k2_fit = simulate_nonlinear_with_constant_K(
        K_fit, alpha,
        m1, m2, k1, c1, c2, kc, cd,
        F_ext, u_max,
        t_end, y0,
        max_step=1e-2
    )

    # -----------------------------
    # Plots: x1, x2 and u(t)
    # -----------------------------
    t1 = sol_lin.t
    x1_lin, x1d_lin, x2_lin, x2d_lin = sol_lin.y

    t2 = sol_fit.t
    x1_fit, x1d_fit, x2_fit, x2d_fit = sol_fit.y

    plt.figure(figsize=(12, 6))
    plt.plot(t1, x1_lin, label="x1 (K_lin)")
    plt.plot(t1, x2_lin, label="x2 (K_lin)")
    plt.plot(t2, x1_fit, "--", label="x1 (K_fit)")
    plt.plot(t2, x2_fit, "--", label="x2 (K_fit)")
    plt.xlabel("t [s]")
    plt.ylabel("Displacement [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(t1, u_lin, label="u(t) (K_lin)")
    plt.plot(t2, u_fit, "--", label="u(t) (K_fit)")
    plt.xlabel("t [s]")
    plt.ylabel("Control force [N]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(t1, k2_lin, label="k2(t)=alpha*u (K_lin)")
    plt.plot(t2, k2_fit, "--", label="k2(t)=alpha*u (K_fit)")
    plt.xlabel("t [s]")
    plt.ylabel("k2(t) [N/m] (from alpha*u)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
