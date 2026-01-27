# -*- coding: utf-8 -*-
"""
Continuous-time adjoint optimization of a constant state-feedback gain K

Control law:
    a(t) = K @ s(t)
where s = [s1, s1d, s2, s2d] and K is a 4-vector.


@author: demaria
"""

import time
import numpy as np

import jax
import jax.numpy as jnp
import optax

from _auxFunc import load_params,build_dyn_system_jnp
from _ADJ_functions import make_time_grid, running_cost, grad_running_cost_s
from _ADJ_functions import  rk4_step_state, rk4_step_adjoint_rev,A_cl
from _plottingFunc import plot_style
plot_style(18)

jax.config.update("jax_enable_x64", True)
#%%
# -----------------------------
# Main simulation + optimization
# -----------------------------
def simulate_adjoint_fullState(
    m1, m2,
    k1, k2,
    c1, c2,
    kc, cd,
    t_end,
    y0,
    N,
    w_s1, w_s1d, w_e, w_ed,
    r_a,
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
      S_passive: trajectory with K=0 (N+1,4)
      S_opt: trajectory with best K found (N+1,4)
      a_opt: control signal for S_opt (N+1,)
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
        S = forward_trajectory(K) 
        #Compute cost from trajectory
        J, a, _ = cost_from_traj(K, S)
        #Compute backward adjoint from trajectory
        lam = adjoint_from_traj(K, S) 
        # Compute gradient of cost with respect to K
        gK = grad_wrt_K(K, S, a, lam)
        return J, gK
      
    # ----------------------------
    # Forward trajectory given K
    def forward_trajectory(K):
        # RK4 step function
        def step(s, i):
            s_next = rk4_step_state(s, K, dt, A, B)
            return s_next, s_next
        # jax.lax.scan to unroll the loop of N steps
        _, S_next = jax.lax.scan(step, y0, jnp.arange(N))
        # Add initial state
        S = jnp.vstack([y0[None, :], S_next])
        return S
    
    # -----------------------------
    # Compute cost J from trajectory S
    def cost_from_traj(K, S):
        # Control input a = K @ s for each state s in S
        a = S @ K
        # Running cost at each time step, jax.vmap for vectorization
        L = jax.vmap(lambda s_i, a_i: running_cost(s_i, a_i, w_s1, w_s1d, w_e, w_ed, r_a))(S, a)
        # Integral of cost using trapezoidal rule
        J = jnp.sum(0.5 * dt * (L[:-1] + L[1:]))
        return J, a, L
    
    # -----------------------------
    # Compute adjoint trajectory from state trajectory S
    def adjoint_from_traj(K, S):
        # Transpose of closed-loop system matrix
        AclT = A_cl(A,B,K).T
        # Compute ∂L/∂s at each time step
        gradL = jax.vmap(lambda s_i: grad_running_cost_s(s_i, K, w_s1, w_s1d, w_e, w_ed, r_a, reducedState=False))(S)
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
    def grad_wrt_K(K, S, a, lam):
        
        BTlam = lam @ B  # (N+1,)
        s = 2.0 * r_a * a + BTlam
        g_nodes = S * s[:, None]  # (N+1,4)
         # Integral over time using trapezoidal rule        
        gK = jnp.sum(0.5 * dt * (g_nodes[:-1] + g_nodes[1:]), axis=0)
        return gK

    # -----------------------------
    # Function to simulate trajectory and outputs for given K
    @jax.jit
    def simulate_traj_outputs(K):
        S = forward_trajectory(K)
        J, a, _ = cost_from_traj(K, S)
        return S, a, J

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
                f"J={Jf:.6e}  K={np.array(K)}"
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
    S_passive, _, _ = simulate_traj_outputs(jnp.zeros(4))
    S_opt, a_opt, _ = simulate_traj_outputs(jnp.array(K_opt))

    info = {
        "success": True,
        "message": "Adjoint + Optax ADAM completed (full State)",
        "nit": it_final,
        "J": J_final,
        "K_opt": K_opt,
        "J_hist": np.array(J_hist),
        "gnorm_hist": np.array(gnorm_hist),
        "best_iter": best_it,
    }

    return (
        t,
        np.array(S_passive),
        np.array(S_opt),
        np.array(a_opt),
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

    # Weights for Q via outputs: [s1, s1d, e=s1+s2, ed=s1d+s2d]
    w_s1 = p['w_s1']    # Penalty on s1
    w_s1d = p['w_s1d']  # Penalty on s1d
    w_e = p['w_e']      # Penalty on e = s1 + s2
    w_ed = p['w_ed']    # Penalty on ed = s1d + s2d
    # Weight for R:
    r_a  = p['r_a']      # Penalty on the control ai^2
    # Simulation time and initial state
    t_end = p['t_end']
    y0 = (p['s1_0'], p['s1d_0'], p['s2_0'], p['s2d_0'])
    
    dt = 0.01
    N = int(t_end/dt)

    # Optimization hyperparameters
    max_iter = 7_000 # maximum number of ADAM iterations
    lr=2e-2 # learning rate for ADAM
    K0 = np.zeros(4)

    start = time.time()
    t,sol_noControl, sol_control, a_series, K_opt, info, K_hist = simulate_adjoint_fullState(
        m1, m2,
        k1, k2,
        c1, c2,
        kc, cd,
        t_end,
        y0,
        N,
        w_s1, w_s1d, w_e, w_ed,
        r_a,
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
    s1_noControl, s1d_noControl, s2_noControl, s2d_noControl = sol_noControl.T
    s1_control, s1d_control, s2_control, s2d_control = sol_control.T

    # Plots
    from _plottingFunc import plot_controlVs_noControl, plot_control_force
    plot_controlVs_noControl(
        t, s1_noControl, s2_noControl,
        t, s1_control, s2_control,
    )
    plot_control_force(t_end, a_series)

    from _plottingFunc import plot_optimization_history,plot_gain_history_fullState
    # Optimization history
    plot_optimization_history(info)

    # Gain history
    plot_gain_history_fullState(K_hist)
