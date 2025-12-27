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
        return lambda t: F0 if t<=t_off else 0.0
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