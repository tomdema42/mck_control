# -*- coding: utf-8 -*-
"""
This script is used to simulate the dynamics of the two-mass-spring-damper system
using numpy and JAX, in order to find the faster configuration

@author: demaria
"""
#%% Imports
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from _auxFunctions import load_params
#%% Parameter and settings
# -------------------------------------------------
# 1) Load of Parameters
# -------------------------------------------------
params = load_params("./data.txt")
m1, k1, c1 = params["m1"], params["k1"], params["c1"]
m2, k2, c2 = params["m2"], params["k2"], params["c2"]
cd = params["cd"]

# external forcing on m1
F0 = 1.0       # [N] amplitude

# time grid (fixed for all simulations)
T_end = 10.0          # [s] total time
n_pts = 4000
t_eval = np.linspace(0.0, T_end, n_pts)

# initial conditions: [x1, x1d, x2, x2d]
y0 = [0.0, 0.0, 0.0, 0.0]
#%%

# -------------------------------------------------
# 2) Forcing and (constant) k2
# -------------------------------------------------
def F_ext(t):
    """External force on m1 (pulse until 0.5s)."""
    return F0 if t < 1.0 else 0.0

# -------------------------------------------------
# 3) ODE system
# -------------------------------------------------
def make_rhs():
    """Return rhs(t, y)"""
    def rhs(t, y):
        x1, x1d, x2, x2d = y
        
        F = F_ext(t)

        # m1 * x1dd = -k1*x1 - c1*x1d + cd*(x2d - x1d) + F
        # m2 * x2dd = -k2*x2 - c2*x2d - cd*(x2d - x1d)
        x1dd = (-k1 * x1 - c1 * x1d + cd * (x2d - x1d) + F) / m1
        x2dd = (-k2 * x2 - c2 * x2d - cd * (x2d - x1d)) / m2

        return [x1d, x1dd, x2d, x2dd]

    return rhs
# -------------------------------------------------
# Simulation and objective
# -------------------------------------------------
def simulate():
    """Simulate the system for a given k2_value and return (t, x1, x2)."""
    rhs = make_rhs()
    sol = solve_ivp(rhs, (0.0, T_end), y0, t_eval=t_eval, rtol=1e-7, atol=1e-9)
    t = sol.t
    x1 = sol.y[0, :]
    x2 = sol.y[2, :]
    return t, x1, x2

# -------------------------------------------------
# Main 
# -------------------------------------------------

if __name__ == "__main__":
    nSims = 1000
    start = time.time()
    for i in range(nSims):
        t, x1, x2 = simulate()
    end = time.time()
    print(f"Average time per simulation over {nSims} runs: {(end - start)/nSims:.6f} seconds")

      
    plt.figure()
    plt.plot(t, x1, label="x1 (m1)")
    plt.plot(t, x2, label="x2 (m2)")
    plt.xlabel("time [s]")
    plt.ylabel("displacement [m]")
    plt.grid(True)
    plt.legend()
    plt.title("Two-mass-spring-damper system response")
    plt.show()