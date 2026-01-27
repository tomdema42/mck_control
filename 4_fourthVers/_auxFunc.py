# -*- coding: utf-8 -*-
"""
Auxiliary functions for the 2-DOF system control.

@author: demaria
"""
import numpy as np
import jax.numpy as jnp
# %%
def load_params(filename):
    """
    Load system parameters from a text file.
    Input: filename - path to the parameter file.
    Output: is a dictionary with parameter names and values.
    """
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
# %%
def build_dyn_system_np(m1, m2, k1, k2, c1, c2, kc, cd):
    """
    Build state-space matrices A, B for the 2-DOF system:
      x = [s1, s1d, s2, s2d]
      x_dot = A x + B u
    where u is the control force on mass 2.
    """
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-(k1 + kc) / m1, -(c1 + cd) / m1,  kc / m1,         cd / m1],
        [0.0, 0.0, 0.0, 1.0],
        [ kc / m2,         cd / m2,        -(k2 + kc) / m2, -(c2 + cd) / m2],
    ])
    B = np.array([[0.0], [0.0], [0.0], [1.0 / m2]])
    return A, B
def build_dyn_system_jnp(m1, m2, k1, k2, c1, c2, kc, cd):
    """
    Build open-loop state-space matrices (A, B) for the 2-DOF system:
        s = [s1, s1d, s2, s2d]

    B is a 4-vector because u is a scalar input acting on x2dd.
    """
    A = jnp.array([
        [0.0, 1.0, 0.0, 0.0],
        [-(k1 + kc) / m1, -(c1 + cd) / m1,  kc / m1,       cd / m1],
        [0.0, 0.0, 0.0, 1.0],
        [ kc / m2,       cd / m2,      -(k2 + kc) / m2, -(c2 + cd) / m2],
    ])
    B = jnp.array([0.0, 0.0, 0.0, 1.0 / m2])  # scalar u injects into x2dd
    return A, B

#%%
def rhs_fullState(t, y, K, m1, m2, k1, k2, c1, c2, kc, cd, Passive = False):
    """
    Right-hand side of the state-space ODE for the 2-DOF system with LQR control.
    Inputs:
        t - time (not used, but required by ODE solvers)
        y - state vector [s1, s1d, s2, s2d]
        K - LQR gain matrix
        m1, m2 - masses
        k1, k2 - spring constants
        c1, c2 - damping coefficients
        kc, cd - coupling spring and damping
        Passive - if True, no control is applied (u=0)
    Outputs:
        dydt - time derivative of the state vector
    """
    s1, s1d, s2, s2d = y
    if Passive:
        a = 0.0
    else:
        # LQR control law
        a = -float(K @ np.array([s1, s1d, s2, s2d]))

    # dynamics with control
    s1dd = (-k1 * s1 - c1 * s1d + cd * (s2d - s1d) + kc * (s2 - s1) ) / m1
    s2dd = (-k2 * s2 - c2 * s2d - cd * (s2d - s1d) - kc * (s2 - s1) + a) / m2

    return [s1d, s1dd, s2d, s2dd]
# %%
