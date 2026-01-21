# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 13:21:52 2025

@author: demaria
"""
import numpy as np

def load_params(filename):
    params = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, val = line.split("=")
            params[key.strip()] = val.strip()

    # cast numbers
    for k, v in params.items():
        try:
            params[k] = float(v)
        except ValueError:
            pass

    return params


def make_forcing(params):
    kind = params["forcing_type"]
    F0 = params["F0"]

    if kind == "zero":
        return lambda t: 0.0

    if kind == "step":
        t_on = params["forcing_t_on"]
        return lambda t: F0 if t >= t_on else 0.0
    if kind == "impulse":
        t_off = params['forcing_t_off']
        t_on = params['forcing_t_on']
        return lambda t: F0 if(t_on <= t <  t_off) else 0.0
    if kind == "sine":
        w = 2.0 * np.pi * params["forcing_freq_hz"]
        return lambda t: F0 * np.sin(w * t)

    raise ValueError("Unknown forcing type")
    
def build_canonical_params(p,verbose=False):
    """
    Build canonical parameters:
      omega_n1, omega_n2
      zeta1, zeta2 (local damping)
      zeta_d1, zeta_d2 (coupling damper contribution)
    such that the equations can be written as:

      x1dd + 2( zeta1 + zeta_d1 )*omega1*x1d + omega1^2*x1 - 2*zeta_d1*omega1*x2d = F/m1
      x2dd + 2( zeta2 + zeta_d2 )*omega2*x2d + omega2^2*x2 - 2*zeta_d2*omega2*x1d = 0
    """
    m1, m2 = p["m1"], p["m2"]
    k1, k2 = p["k1"], p["k2"]
    c1, c2 = p["c1"], p["c2"]
    cd = p["cd"]

    omega1 = np.sqrt(k1 / m1)
    omega2 = np.sqrt(k2 / m2)

    zeta1 = c1 / (2.0 * m1 * omega1)
    zeta2 = c2 / (2.0 * m2 * omega2)

    zeta_d1 = cd / (2.0 * m1 * omega1)
    zeta_d2 = cd / (2.0 * m2 * omega2)
    if verbose ==True:
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
        print("Canonical parameters:")
        for k in ["omega1", "omega2", "zeta1", "zeta2", "zeta_d1", "zeta_d2", "zeta_eff1", "zeta_eff2"]:
            print(f"  {k}: {meta[k]}")

    return omega1, omega2, zeta1, zeta2, zeta_d1, zeta_d2

# %%

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
    k1 = g(t, lam, x, u)
    k2 = g(t + 0.5 * dt, lam + 0.5 * dt * k1, x, u)
    k3 = g(t + 0.5 * dt, lam + 0.5 * dt * k2, x, u)
    k4 = g(t + dt, lam + dt * k3, x, u)
    return lam + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# -----------------------------
# Generic RK4 (works for closed-loop + adjoint)
# -----------------------------
def rk4_step_11c(fun, t, y, dt, *args):
    k1 = fun(t, y, *args)
    k2 = fun(t + 0.5 * dt, y + 0.5 * dt * k1, *args)
    k3 = fun(t + 0.5 * dt, y + 0.5 * dt * k2, *args)
    k4 = fun(t + dt,       y + dt * k3,       *args)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
# %% 
def CostFunc_comparison(x, u, w_x1,w_x1d,w_e,w_ed,r_u):
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

