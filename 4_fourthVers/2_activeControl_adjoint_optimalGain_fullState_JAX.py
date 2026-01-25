# -*- coding: utf-8 -*-
"""
Classical (continuous-time) adjoint optimization of a constant state-feedback gain K

Control law:
    u(t) = K @ x(t)
where x = [x1, x1d, x2, x2d] and K is a 4-vector.

Running cost:
    L(x,u) = w_x1 x1^2 + w_x1d x1d^2 + w_e (x1+x2)^2 + w_ed (x1d+x2d)^2 + r_u u^2

@author: demaria
"""

import time
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

from _auxFunc import load_params,build_dyn_system_jnp
from _ADJ_functions import make_time_grid, running_cost, grad_running_cost_x
from _ADJ_functions import  rk4_step_state, rk4_step_adjoint_rev
chSize = 18
plt.rcParams.update({'font.size': chSize})

jax.config.update("jax_enable_x64", True)
#%%
# -----------------------------
# Main simulation + optimization
# -----------------------------
def simulate_2dof_with_optax_adam_classical_adjoint(
    m1, m2,
    k1, k2,
    c1, c2,
    kc, cd,
    t_end,
    y0,
    N,
    w_x1, w_x1d, w_e, w_ed,
    r_u,
    max_iter,
    K0,
    lr=1e-2,
    betas=(0.9, 0.999),
    eps=1e-8,
    print_every=25,
):
    """
    Optimize K (4-vector) using classical continuous adjoint gradient and Optax ADAM.

    Returns:
      t: time array (N+1,)
      X_passive: trajectory with K=0 (N+1,4)
      X_opt: trajectory with best K found (N+1,4)
      u_opt: control signal for X_opt (N+1,)
      K_opt: best gain vector (4,)
      info: dict with optimization diagnostics
      K_hist: list of K over iterations (device arrays)
    """
   # -----------------------------
    # Build system dynamics
    A, B = build_dyn_system_jnp(m1, m2, k1, k2, c1, c2, kc, cd)
    # -----------------------------
    # Time grid for rk4 integration
    t, dt = make_time_grid(t_end, N)
    y0 = jnp.array(y0)
    # -----------------------------
    # Core functions for optimization
    # Jitted function (just-in-time compiled)
    # -----------------------------
    @jax.jit
    def cost_and_grad_manual(K):
        #Compute forward trajectory given K
        X = forward_trajectory(K) 
        #Compute cost from trajectory
        J, u, _ = cost_from_traj(K, X)
        #Compute backward adjoint from trajectory
        lam = adjoint_from_traj(K, X) 
        # Compute gradient of cost with respect to K
        gK = grad_wrt_K(K, X, u, lam)
        return J, gK
    
    # -----------------------------
    # Closed-loop system matrix
    def A_cl(K):
        return A + jnp.outer(B, K)
    
    # ----------------------------
    # Forward trajectory given K
    def forward_trajectory(K):
        # RK4 step function
        def step(x, i):
            x_next = rk4_step_state(x, K, dt, A, B)
            return x_next, x_next
        # jax.lax.scan to unroll the loop of N steps
        _, X_next = jax.lax.scan(step, y0, jnp.arange(N))
        # Add initial state
        X = jnp.vstack([y0[None, :], X_next])
        return X
    
    # -----------------------------
    # Compute cost J from trajectory X
    def cost_from_traj(K, X):
        # Control input u = K @ x for each state x in X
        u = X @ K
        # Running cost at each time step, jax.vmap for vectorization
        L = jax.vmap(lambda x_i, u_i: running_cost(x_i, u_i, w_x1, w_x1d, w_e, w_ed, r_u))(X, u)
        # Integral of cost using trapezoidal rule
        J = jnp.sum(0.5 * dt * (L[:-1] + L[1:]))
        return J, u, L
    
    # -----------------------------
    # Compute adjoint trajectory from state trajectory X
    def adjoint_from_traj(K, X):
        # Transpose of closed-loop system matrix
        AclT = A_cl(K).T
        # Compute ∂L/∂x at each time step
        gradL = jax.vmap(lambda x_i: grad_running_cost_x(x_i, K, w_x1, w_x1d, w_e, w_ed, r_u))(X)
        # Reverse the gradient for backward integration
        gradL_rev = gradL[::-1]

        # Initial condition for adjoint at t=T
        lam_T = jnp.zeros((4,))

        # RK4 step function for adjoint in reverse time
        def step(lam, j):
            g_curr = gradL_rev[j]
            g_next = gradL_rev[j + 1]
            g_mid = 0.5 * (g_curr + g_next)
            lam_next = rk4_step_adjoint_rev(lam, g_curr, g_mid, g_next, dt, AclT)
            return lam_next, lam_next
        
        # jax.lax.scan to unroll the loop of N steps
        _, lam_next = jax.lax.scan(step, lam_T, jnp.arange(N))
        #jnp.vstack to add initial condition at t=T
        lam_rev = jnp.vstack([lam_T[None, :], lam_next])
        # Reverse to get lam(t) from lam_rev(τ)
        lam = lam_rev[::-1]
        return lam
    
    # -----------------------------
    # Gradient of cost with respect to K
    def grad_wrt_K(K, X, u, lam):
        
        BTlam = lam @ B  # (N+1,)
        s = 2.0 * r_u * u + BTlam
        g_nodes = X * s[:, None]  # (N+1,4)
         # Integral over time using trapezoidal rule        
        gK = jnp.sum(0.5 * dt * (g_nodes[:-1] + g_nodes[1:]), axis=0)
        return gK

    # -----------------------------
    # Function to simulate trajectory and outputs for given K
    @jax.jit
    def simulate_traj_outputs(K):
        X = forward_trajectory(K)
        J, u, _ = cost_from_traj(K, X)
        return X, u, J

    # -----------------------------
    # Optimizer setup
    # -----------------------------
    K = jnp.array(K0)
    
    # Optax ADAM optimizer
    optimizer = optax.adam(learning_rate=lr, b1=betas[0], b2=betas[1], eps=eps)
    opt_state = optimizer.init(K)
    
    # Optimization history initialization
    J_hist = []
    gnorm_hist = []
    best_J = np.inf
    best_K = None
    best_it = -1
    K_hist = []

    # -----------------------------
    # Optimization loop
    # -----------------------------
    for it in range(1, max_iter + 1):
        # Compute cost and gradient
        J, gK = cost_and_grad_manual(K)

        Jf = float(J)
        gnorm = float(jnp.linalg.norm(gK))
        J_hist.append(Jf)
        gnorm_hist.append(gnorm)
        
        # Store best solution
        if Jf < best_J:
            best_J = Jf
            best_K = np.array(K)
            best_it = it
        # Print progress
        if it == 1 or it % print_every == 0:
            print(
                f"[Adjoint on full state + Optax ADAM] it={it:6d}  "
                f"J={Jf:.6e}  ||g||={gnorm:.3e}  K={np.array(K)}"
            )
        # Optax ADAM update step
        updates, opt_state = optimizer.update(gK, opt_state, params=K)
        K = optax.apply_updates(K, updates)
        K_hist.append(K)

    # Pick best solution seen
    K_opt = best_K
    J_final = best_J
    it_final = best_it

    # Passive (K=0) and optimal trajectories
    X_passive, _, _ = simulate_traj_outputs(jnp.zeros(4))
    X_opt, u_opt, _ = simulate_traj_outputs(jnp.array(K_opt))

    info = {
        "success": True,
        "message": "Adjoint + Optax ADAM completed",
        "nit": it_final,
        "J": J_final,
        "K_opt": -K_opt,
        "J_hist": np.array(J_hist),
        "gnorm_hist": np.array(gnorm_hist),
        "best_iter": best_it,
    }

    return (
        t,
        np.array(X_passive),
        np.array(X_opt),
        np.array(u_opt),
        K_opt,
        info,
        K_hist,
    )


# -----------------------------
# Script entry point (run + plots)
# -----------------------------
if __name__ == "__main__":
    # Load parameters
    param_file = "params.txt"
    p = load_params(param_file)
    # System parameters
    m1, m2 = p["m1"], p["m2"]
    k1, k2 = p["k1"], p["k2"]
    c1, c2 = p["c1"], p["c2"]
    cd, kc = p["cd"], p["kc"]

    # Weights for Q via outputs: [x1, x1d, e=x1+x2, ed=x1d+x2d]
    w_x1 = p['w_x1']    # Penalty on x1
    w_x1d = p['w_x1d']  # Penalty on x1d
    w_e = p['w_e']      # Penalty on e = x1 + x2
    w_ed = p['w_ed']    # Penalty on ed = x1d + x2d
    # Weight for R:
    r_u  = p['r_u']      # Penalty on the control ui^2
    # Simulation time and initial state
    t_end = p['t_end']
    y0 = (p['x1_0'], p['x1d_0'], p['x2_0'], p['x2d_0'])
    
    N = 400 # number of time intervals

    # Optimization hyperparameters
    max_iter = 7_000 # maximum number of ADAM iterations
    lr=2e-2 # learning rate for ADAM
    K0 = np.zeros(4)

    start = time.time()
    t,sol_noControl, sol_control, u_series, K_opt, info, K_hist = simulate_2dof_with_optax_adam_classical_adjoint(
        m1, m2,
        k1, k2,
        c1, c2,
        kc, cd,
        t_end,
        y0,
        N,
        w_x1, w_x1d, w_e, w_ed,
        r_u,
        max_iter,
        K0=K0,
        lr=2e-2,
        print_every=50,
    )
    stop = time.time()
    print("Total exectuion time:", stop - start)
    
    print("\nINFO:")
    for k, v in info.items():
        if k in ("J_hist", "gnorm_hist"):
            continue
        print(f"  {k}: {v}")




    # Unpack trajectories for plotting
    x1_noControl, x1d_noControl, x2_noControl, x2d_noControl = sol_noControl.T
    x1_control, x1d_control, x2_control, x2d_control = sol_control.T

    # Plots
    from _plottingFunc import plot_controlVs_noControl, plot_control_force
    plot_controlVs_noControl(
        t, x1_noControl, x2_noControl,
        t, x1_control, x2_control,
    )
    plot_control_force(t_end, u_series)

    from _plottingFunc import plot_optimization_history,plot_gain_history_fullState
    # Optimization history
    plot_optimization_history(info)

    # Gain history
    plot_gain_history_fullState(K_hist)
