# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 09:32:37 2025

@author: demaria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from _auxFunctions import load_params
# %% DEFAULT


# -------------------------------------------------
# 1) Parameters
# -------------------------------------------------
params = load_params("./data.txt")
m1, k1, c1 = params["m1"], params["k1"], params["c1"]
m2, k2_nominal, c2 = params["m2"], params["k2"], params["c2"]
cd = params["cd"]

# external forcing on m1
F0 = 1.0       # [N] amplitude

# time grid (fixed for all simulations)
T_end = 10.0          # [s] total time
n_pts = 4000
t_eval = np.linspace(0.0, T_end, n_pts)

# initial conditions: [x1, x1d, x2, x2d]
y0 = [0.0, 0.0, 0.0, 0.0]
w_k=1e-8

# -------------------------------------------------
# 2) Forcing and (constant) k2
# -------------------------------------------------
def F_ext(t):
    """External force on m1."""
    return F0 if t <0.5 else 0


# %%



def k2_effective(t, x, k2_value):
    """
    For now: constant stiffness k2_value.
    """
    return k2_value


# -------------------------------------------------
# 3) ODE system for a given k2
# -------------------------------------------------
def make_rhs(k2_value):
    """
    Returns a function rhs(t, y) that uses the given k2_value.
    """

    def rhs(t, y):
        """
        y = [x1, x1_dot, x2, x2_dot]
        """
        x1, x1d, x2, x2d = y

        k2 = k2_effective(t, y, k2_value)
        F  = F_ext(t)

        # Equation for m1:
       
        

        # Equation for m2:
        
        # m1 * x1dd = -k1*x1 - c1*x1d + cd*(x2d - x1d) + F_ext
        # m2 * x2dd = -k2*x2 - c2*x2d - cd*(x2d - x1d)
        
        x1dd = (-k1 * x1 - c1 * x1d + cd * (x2d - x1d) + F) / m1
        x2dd = (-k2 * x2 - c2 * x2d - cd * (x2d - x1d)) / m2

        return [x1d, x1dd, x2d, x2dd]

    return rhs

def J_cost(t,x,k,w_k):
    J_x = np.trapezoid(x**2, t)
    k_full =np.full_like(t,k,dtype=float)
    J_k =np.trapezoid(k_full**2, t)*w_k
    return J_x + J_k
# %%



def simulate(k2_value):
    """
    Simulate the system for a given k2_value.
    Returns t, x1, x2.
    """
    rhs = make_rhs(k2_value)
    sol = solve_ivp(rhs,
                    (0.0, T_end),
                    y0,
                    t_eval=t_eval,
                    rtol=1e-7,
                    atol=1e-9)

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed for k2 = {k2_value}")

    t = sol.t
    x1 = sol.y[0, :]
    x2 = sol.y[2, :]
    return t, x1, x2


# -------------------------------------------------
# 4) Cost functional J(k2) based on x1(t)
# -------------------------------------------------
def objective_k2(k2_value):
    """
    Objective to minimize with respect to k2.
    Here we use an L1-type cost on x1(t) over [0, T_end].

    Smaller J(k2) means smaller oscillations of x1.
    """
    try:
        t, x1, _ = simulate(k2_value)
    except Exception:
        # If integration fails, return a big penalty
        return 1e12

    # Use your helper (assumed signature l1_cost(signal, time))
    # If your l1_cost only takes the signal, adapt this line to: l1_cost(x1)
    J = J_cost(t,x1,k2_value,w_k=w_k)

    return J


# -------------------------------------------------
# 5) Minimize J(k2) over a given interval
# -------------------------------------------------
k2_min = 1
k2_max = 600

res = minimize_scalar(
    objective_k2,
    bounds=(k2_min, k2_max),
    method="bounded",
    options={"xatol": 1e-2},
)

k2_opt = res.x
print("Optimization result:")
print(f"  k2_nominal = {k2_nominal:.3e} [N/m]")
print(f"  k2_opt     = {k2_opt:.3e} [N/m]")
print(f"  min J(k2)  = {res.fun:.6e}")
print(f"  success    = {res.success}, message = {res.message}")

# -------------------------------------------------
# 6) Plot cost function J(k2)
# -------------------------------------------------
n_samples = 60  # number of k2 points where we evaluate J(k2)
k2_grid = np.linspace(k2_min, k2_max, n_samples)

J_vals = []
for k2_val in k2_grid:
    J_vals.append(objective_k2(k2_val))
J_vals = np.array(J_vals)

J_nominal = objective_k2(k2_nominal)

plt.figure()
plt.plot(k2_grid, J_vals, marker='o', linestyle='-', label='J(k2)')
plt.axvline(k2_opt, linestyle='--', label=f'k2_opt = {k2_opt:.2f}')
plt.xlabel("k2 [N/m]")
plt.ylabel("J(k2)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# -------------------------------------------------
# 6) Final simulation with optimal k2
# -------------------------------------------------
t, x1, x2 = simulate(k2_opt)
k2_t = np.full_like(t, k2_opt)


# -------------------------------------------------
# 7) Plots
# -------------------------------------------------
plt.figure()
plt.plot(t, x1, label="x1 (m1)")
plt.plot(t, x2, label="x2 (m2)")
plt.xlabel("time [s]")
plt.ylabel("displacement [m]")
plt.grid(True)
plt.legend()
plt.title(f"Response with optimized k2 = {k2_opt:.3e} N/m")

# %% SPARE SIMULATION OF DYNAMICAL SYSTEM
# 
sim_k = 200
t,x1_nom,x2_nom = simulate(sim_k)
plt.figure()
# plt.plot(t, x1, label="x1 (opt)")
plt.plot(t, x1_nom,'-', label="x1")
plt.plot(t,x2_nom,'--',label='x2')
plt.xlabel("time [s]")
plt.ylabel("displacement [m]")
plt.grid(True)
plt.title('k constant = '+str(sim_k))
plt.legend()
