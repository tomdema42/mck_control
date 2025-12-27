# -*- coding: utf-8 -*-
"""
Two-mass-spring-damper system simulation using JAX (no NumPy / SciPy ODE solver).

- Uses a JIT-compiled fixed-step RK4 integrator implemented with lax.scan.

@author: demaria
"""

# %% Imports
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, lax
import time
import numpy as np
import matplotlib.pyplot as plt
from _auxFunctions import load_params

# %% Parameter and settings
# -------------------------------------------------
# 1) Load Parameters
# -------------------------------------------------
params = load_params("./0_data.txt")
m1, k1, c1 = params["m1"], params["k1"], params["c1"]
m2, k2, c2 = params["m2"], params["k2"], params["c2"]
cd = params["cd"]

# external forcing on m1
F0 = 1.0  # [N] amplitude

# time grid (fixed for all simulations)
T_end = 10.0  # [s]
n_pts = 4000
t_eval = jnp.linspace(0.0, T_end, n_pts)

# initial conditions: [x1, x1d, x2, x2d]
y0 = jnp.array([0.0, 0.0, 0.0, 0.0])


# -------------------------------------------------
# 2) Forcing
# -------------------------------------------------
@jit
def F_ext(t):
    """External force on m1 (pulse until 1.0s)."""
    return jnp.where(t < 1.0, F0, 0.0)


# -------------------------------------------------
# 3) ODE system
# -------------------------------------------------
@jit
def rhs(y, t, p):
    """
    y = [x1, x1d, x2, x2d]
    p = [m1, k1, c1, m2, k2, c2, cd]
    """
    x1, x1d, x2, x2d = y
    m1_, k1_, c1_, m2_, k2_, c2_, cd_ = p

    F = F_ext(t)

    x1dd = (-k1_ * x1 - c1_ * x1d + cd_ * (x2d - x1d) + F) / m1_
    x2dd = (-k2_ * x2 - c2_ * x2d - cd_ * (x2d - x1d)) / m2_

    return jnp.array([x1d, x1dd, x2d, x2dd])


# -------------------------------------------------
# 4) JAX RK4 Integrator (JIT-compiled)
# -------------------------------------------------
@jit
def integrate_rk4(y0, t):
    """
    Fixed-step RK4 over the provided time vector t.
    Returns Y with shape (len(t), 4), including the initial state at t[0].
    """
    dt = t[1] - t[0]

    def step(y, t_i):
        k1_ = rhs(y, t_i, p)
        k2_ = rhs(y + 0.5 * dt * k1_, t_i + 0.5 * dt, p)
        k3_ = rhs(y + 0.5 * dt * k2_, t_i + 0.5 * dt, p)
        k4_ = rhs(y + dt * k3_, t_i + dt, p)
        y_next = y + (dt / 6.0) * (k1_ + 2.0 * k2_ + 2.0 * k3_ + k4_)
        return y_next, y_next

    _, ys_tail = lax.scan(step, y0, t[:-1])
    ys = jnp.vstack([y0[None, :], ys_tail])
    return ys


# -------------------------------------------------
# 5) Simulation wrapper
# -------------------------------------------------
def simulate():
    """Simulate the system and return (t, x1, x2) as JAX arrays."""
    global p
    p = jnp.array([m1, k1, c1, m2, k2, c2, cd])
    Y = integrate_rk4(y0, t_eval)
    x1 = Y[:, 0]
    x2 = Y[:, 2]
    return t_eval, x1, x2


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

    # Matplotlib expects NumPy arrays
    t_np = np.asarray(t)
    x1_np = np.asarray(x1)
    x2_np = np.asarray(x2)

    plt.figure()
    plt.plot(t_np, x1_np, label="x1 (m1)")
    plt.plot(t_np, x2_np, label="x2 (m2)")
    plt.xlabel("time [s]")
    plt.ylabel("displacement [m]")
    plt.grid(True)
    plt.legend()
    plt.title("Two-mass-spring-damper system response (JAX RK4)")
    plt.show()
