# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 13:09:00 2026

@author: demaria
"""
import numpy as np
import jax.numpy as jnp


#%%
def make_time_grid(t_end, N):
    """
    Create time grid from 0 to t_end with N intervals (N+1 points).
    """
    t = np.linspace(0.0, t_end, N + 1)
    dt = float(t[1] - t[0])
    return t, dt
# -----------------------------
def running_cost(s, a, w_s1, w_s1d, w_e, w_ed, r_a):
    """
    Running cost L(s,a) at a single time step.
    """
    s1, s1d, s2, s2d = s
    e = s1 + s2
    ed = s1d + s2d
    return (
        w_s1 * s1 * s1
        + w_s1d * s1d * s1d
        + w_e * e * e
        + w_ed * ed * ed
        + r_a * a * a
    )
def grad_running_cost_s(s, K, w_s1, w_s1d, w_e, w_ed, r_a, reducedState=False):
    """
    ∂L/∂s for L(s,a) with a = K·s.
    """
    s1, s1d, s2, s2d = s
    e = s1 + s2
    ed = s1d + s2d
    if reducedState:
        K = k_full(K)
        a = jnp.dot(K, s)
    else:   
        a = jnp.dot(K, s)

    g_state = jnp.array([
        2.0 * (w_s1 * s1 + w_e * e),
        2.0 * (w_s1d * s1d + w_ed * ed),
        2.0 * (w_e * e),
        2.0 * (w_ed * ed),
    ])

    # d/ ds [ r_a (Ks)^2 ] = 2 r_a a K
    if reducedState:
        g_control = 2.0 * r_a * a * k_full(K)
    else:
        g_control = 2.0 * r_a * a * K
    return g_state + g_control

#%%
def rk4_step_state(s, K, dt, A, B, reducedState=False):
    """
    One RK4 step for the forward state dynamics.

    sdot = A s + B a,  a = K @ s
    """
    if reducedState:
        def f_closed(s_local):
            a_local = jnp.dot(K[:2], s_local[:2])  # K is 2-element gain
            return A @ s_local + B * a_local
    else:
        def f_closed(s_local):
            a_local = jnp.dot(K, s_local)
            return A @ s_local + B * a_local
        

    k1_ = f_closed(s)
    k2_ = f_closed(s + 0.5 * dt * k1_)
    k3_ = f_closed(s + 0.5 * dt * k2_)
    k4_ = f_closed(s + dt * k3_)

    return s + (dt / 6.0) * (k1_ + 2.0 * k2_ + 2.0 * k3_ + k4_)

def rk4_step_adjoint_rev(lam, g_curr, g_mid, g_next, dt, AclT):
    """
    One RK4 step for the *reversed-time* adjoint integration.

    Original adjoint:
        lam_dot = -Acl^T lam - gradL
        lam(T) = 0

    Reverse time with τ = T - t:
        dlam/dτ =  Acl^T lam + gradL
    """
    def rhs_lambda(lam_local, gradL_local):
        return AclT @ lam_local + gradL_local

    k1_ = rhs_lambda(lam, g_curr)
    k2_ = rhs_lambda(lam + 0.5 * dt * k1_, g_mid)
    k3_ = rhs_lambda(lam + 0.5 * dt * k2_, g_mid)
    k4_ = rhs_lambda(lam + dt * k3_, g_next)

    return lam + (dt / 6.0) * (k1_ + 2.0 * k2_ + 2.0 * k3_ + k4_)


def k_full(k):
    """Embed k=[k0,k1] into full-state gain [k0,k1,0,0]."""
    return jnp.array([k[0], k[1], 0.0, 0.0])

def A_cl(A, B, K):
    return A + jnp.outer(B, K)
# %%