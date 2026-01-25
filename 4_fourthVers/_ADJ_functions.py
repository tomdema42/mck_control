# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 13:09:00 2026

@author: demaria
"""
import numpy as np
import jax.numpy as jnp


#%%
def make_time_grid(t_end, N):
    t = np.linspace(0.0, t_end, N + 1)
    dt = float(t[1] - t[0])
    return t, dt
# -----------------------------
def running_cost(x, u, w_x1, w_x1d, w_e, w_ed, r_u):
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
def grad_running_cost_x(x, K, w_x1, w_x1d, w_e, w_ed, r_u):
    """
    ∂L/∂x for L(x,u) with u = K·x.
    """
    x1, x1d, x2, x2d = x
    e = x1 + x2
    ed = x1d + x2d

    u = jnp.dot(K, x)

    g_state = jnp.array([
        2.0 * (w_x1 * x1 + w_e * e),
        2.0 * (w_x1d * x1d + w_ed * ed),
        2.0 * (w_e * e),
        2.0 * (w_ed * ed),
    ])

    # d/ dx [ r_u (Kx)^2 ] = 2 r_u u K
    g_control = 2.0 * r_u * u * K
    return g_state + g_control

#%%
def rk4_step_state(x, K, dt, A, B):
    """
    One RK4 step for the forward state dynamics WITHOUT forcing.

    xdot = A x + B u,  u = K @ x
    """
    def f_closed(x_local):
        u_local = jnp.dot(K, x_local)
        return A @ x_local + B * u_local

    k1_ = f_closed(x)
    k2_ = f_closed(x + 0.5 * dt * k1_)
    k3_ = f_closed(x + 0.5 * dt * k2_)
    k4_ = f_closed(x + dt * k3_)

    return x + (dt / 6.0) * (k1_ + 2.0 * k2_ + 2.0 * k3_ + k4_)

def rk4_step_adjoint_rev(lam, g_curr, g_mid, g_next, dt, AclT):
    """
    One RK4 step for the *reversed-time* adjoint integration.

    Original adjoint:
        lam_dot = -Acl^T lam - gradL
        lam(T) = 0

    Reverse time with τ = T - t:
        dlam/dτ =  Acl^T lam + gradL
    """
    def rhs(lam_local, gradL_local):
        return AclT @ lam_local + gradL_local

    k1_ = rhs(lam, g_curr)
    k2_ = rhs(lam + 0.5 * dt * k1_, g_mid)
    k3_ = rhs(lam + 0.5 * dt * k2_, g_mid)
    k4_ = rhs(lam + dt * k3_, g_next)

    return lam + (dt / 6.0) * (k1_ + 2.0 * k2_ + 2.0 * k3_ + k4_)
