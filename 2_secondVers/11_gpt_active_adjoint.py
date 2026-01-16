# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 14:46:09 2026

@author: demaria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# -----------------------------
# Forcing (impulse-like pulse)
# -----------------------------
def make_forcing(F0=10.0):
    def F_ext(t):
        t0 = 1.0
        dt = 1.0
        return F0 if (t0 <= t < t0 + dt) else 0.0
    return F_ext


# -----------------------------
# RK4 integrators (fixed step)
# -----------------------------
def rk4_step(f, t, y, dt, u):
    k1 = f(t, y, u)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1, u)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2, u)
    k4 = f(t + dt, y + dt * k3, u)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def rk4_step_adjoint(g, t, lam, dt, x, u):
    # integrate lam backward: lam(t-dt) = lam(t) + ...
    # so we call with negative dt
    k1 = g(t, lam, x, u)
    k2 = g(t + 0.5 * dt, lam + 0.5 * dt * k1, x, u)
    k3 = g(t + 0.5 * dt, lam + 0.5 * dt * k2, x, u)
    k4 = g(t + dt, lam + dt * k3, x, u)
    return lam + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# -----------------------------
# Main: adjoint optimal control
# -----------------------------
def simulate_2dof_with_adjoint_optimization(
    m1=0.3, m2=0.1,
    k1=11.83, k2=20.0,
    c1=0.11, c2=0.05,
    kc=0.0, cd=1.6,
    F0=1.0,
    u_max=10.0,
    t_end=10.0,
    y0=(0.0, 0.0, 0.0, 0.0),
    N=400,                 # number of control intervals (piecewise-constant)
    max_iter=200,
    # cost weights
    w_x1=1.0,
    w_x1d=0.1,
    w_e=50.0,              # e = x1 + x2
    w_ed=2.0,              # ed = x1d + x2d
    r_u=0.05,              # control effort
):
    """
    Optimal control via adjoint:
      minimize J = ∫ [ w_x1 x1^2 + w_x1d x1d^2 + w_e (x1+x2)^2 + w_ed (x1d+x2d)^2 + r_u u^2 ] dt
      subject to dynamics, with u piecewise-constant and |u| <= u_max via u = u_max*tanh(v)

    Returns:
      t, X_passive, X_opt, u_opt, info
    """

    F_ext = make_forcing(F0=F0)

    # time grid aligned with control intervals
    t = np.linspace(0.0, t_end, N + 1)
    dt = t[1] - t[0]

    def u_from_v(v):
        return u_max * np.tanh(v)

    def du_dv(v):
        th = np.tanh(v)
        return u_max * (1.0 - th*th)

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

    # A = df/dx (constant here because system is linear)
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
        Lam[N] = 0.0  # no terminal cost
        # integrate backward from t[N] to t[0]
        for i in range(N, 0, -1):
            # use state at i (or midpoint); here use X[i] for simplicity
            Lam[i-1] = rk4_step_adjoint(lam_dot, t[i], Lam[i], -dt, X[i], u_vals[i-1])
        return Lam

    # objective + gradient wrt v (unconstrained variables)
    def J_and_grad(v):
        u_vals = u_from_v(v)

        X = forward_simulate(u_vals)
        Lam = adjoint_backward(X, u_vals)

        # cost (trapezoid)
        J = 0.0
        for i in range(N):
            J += 0.5 * dt * (L(X[i], u_vals[i]) + L(X[i+1], u_vals[i]))
        # gradient wrt u_i: ∫ (dL/du + lam^T df/du) dt ≈ dt*(dL/du + lam·B)
        g_u = np.zeros(N)
        for i in range(N):
            # use Lam at i (or midpoint). This is a standard discrete approximation.
            g_u[i] = dt * (dL_du(u_vals[i]) + Lam[i] @ B)

        # chain rule: du/dv
        g_v = g_u * du_dv(v)
        return J, g_v

    # initial guess: zero control
    v0 = np.zeros(N)

    res = minimize(
        fun=lambda v: J_and_grad(v)[0],
        x0=v0,
        jac=lambda v: J_and_grad(v)[1],
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-9, "gtol": 1e-6},
    )

    v_opt = res.x
    u_opt = u_from_v(v_opt)

    X_passive = passive_simulate()
    X_opt = forward_simulate(u_opt)

    info = {"success": res.success, "message": res.message, "nit": res.nit, "J": res.fun}
    return t, X_passive, X_opt, u_opt, info


# -----------------------------
# Run + plots
# -----------------------------
if __name__ == "__main__":
    t, X0, X1, u, info = simulate_2dof_with_adjoint_optimization(
        m1=0.3, m2=0.1,
        k1=11.83, k2=20.0,
        c1=0.11, c2=0.05,
        kc=0.0, cd=1.6,
        F0=1.0,
        u_max=10.0,
        t_end=10.0,
        y0=(0.0, 0.0, 0.0, 0.0),
        N=400,
        w_x1=1.0, w_x1d=0.1, w_e=50.0, w_ed=2.0,
        r_u=0.05,
        max_iter=200,
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

    plt.figure(figsize=(12, 4))
    plt.plot(t, x1_1 + x2_1, label="e(t) = x1 + x2 (should go to 0)")
    plt.xlabel("t [s]")
    plt.ylabel("Anti-phase error [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
