# -*- coding: utf-8 -*-
"""
Continuous-time adjoint optimization of a constant state-feedback gain K

Control law:
    a(t) = K @ s(t)
where s = [s1, s1d, s2, s2d] and K is a 2-vector K = [k1,k2,0,0].


@author: demaria
"""

import time
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

from _auxFunc import load_params,build_dyn_system_jnp
from _ADJ_functions import make_time_grid,k_full


# =============================================================================
# Global plotting / JAX settings
# =============================================================================
plt.rcParams.update({"font.size": 18})
jax.config.update("jax_enable_x64", True)


# =============================================================================
# Core solver + adjoint + optimization
# =============================================================================
def simulate_adjoint_reducedState(
    m1, m2,
    k1, k2,
    c1, c2,
    kc, cd,
    t_end,
    y0, N,
    w_s1, w_s1d,
    w_e,  w_ed,
    r_a,
    max_iter,
    k0,  
    lr=1e-2,
    betas=(0.9, 0.999),
    eps=1e-8,
    print_every=25,
):
    """
    Optimize K (2-vector) using classical continuous adjoint gradient and Optax ADAM.

    Returns:
      t: time array (N+1,)
      S_passive: trajectory with K=0 (N+1,4)
      S_opt: trajectory with best K found (N+1,4)
      a_opt: control signal for S_opt (N+1,)
      K_opt: best gain vector (4,)
      info: dict with optimization diagnostics
      K_hist: list of K over iterations (device arrays)
    """
    # -------------------------------------------------------------------------
    # 1) Time grid (NO forcing)
    # -------------------------------------------------------------------------
    t, dt = make_time_grid(t_end, N)
    y0 = jnp.array(y0)

    # -------------------------------------------------------------------------
    # 2) System matrices (NO forcing mapping)
    # -------------------------------------------------------------------------
    A, B = build_dyn_system_jnp(m1, m2, k1, k2, c1, c2, kc, cd)

    # -------------------------------------------------------------------------
    # 3) Control law + running cost and its gradient w.r.t state
    # ------------------------------------------------------------------------

    def control_a(s, k):
        """Restricted feedback: a = k0*s1 + k1*s1d."""
        return jnp.dot(k, s[:2])

    def running_cost(s, a):
        """Scalar running cost L(s,a)."""
        s1_, s1d_, s2_, s2d_ = s
        e = s1_ + s2_
        ed = s1d_ + s2d_
        return (
            w_s1 * s1_ * s1_
            + w_s1d * s1d_ * s1d_
            + w_e * e * e
            + w_ed * ed * ed
            + r_a * a * a
        )

    def grad_running_cost_s(s, k):
        """
        ∂L/∂s  with a = Ks.
        """
        s1_, s1d_, s2_, s2d_ = s
        e = s1_ + s2_
        ed = s1d_ + s2d_

        a = control_a(s, k)

        g_state = jnp.array(
            [
                2.0 * (w_s1 * s1_ + w_e * e),
                2.0 * (w_s1d * s1d_ + w_ed * ed),
                2.0 * (w_e * e),
                2.0 * (w_ed * ed),
            ]
        )

        g_control = 2.0 * r_a * a * k_full(k)
        return g_state + g_control

    # -------------------------------------------------------------------------
    # 4) Closed-loop dynamics + RK4 forward integrator (NO forcing)
    # -------------------------------------------------------------------------
    def A_cl(k):
        """
        Closed-loop Jacobian ∂f/∂s:
            A_cl = A + B*K_full
        """
        return A + jnp.outer(B, k_full(k))

    def f_closed(s, k):
        """sdot = A s + B a ."""
        a = control_a(s, k)
        return A @ s + B * a

    def rk4_step_state(s, k):
        """One RK4 step from node i -> i+1."""
        k1_ = f_closed(s, k)
        k2_ = f_closed(s + 0.5 * dt * k1_, k)
        k3_ = f_closed(s + 0.5 * dt * k2_, k)
        k4_ = f_closed(s + dt * k3_, k)
        return s + (dt / 6.0) * (k1_ + 2.0 * k2_ + 2.0 * k3_ + k4_)

    def forward_traj(k):
        """
        Forward simulation on the fixed time grid.
        Output:
            S: (N+1,4) state trajectory
        """
        def step(s, _i):
            s_next = rk4_step_state(s, k)
            return s_next, s_next

        _, S_next = jax.lax.scan(step, y0, jnp.arange(N))
        S = jnp.vstack([y0[None, :], S_next])
        return S

    # -------------------------------------------------------------------------
    # 5) Cost evaluation (trapezoidal rule on nodes)
    # -------------------------------------------------------------------------
    def cost_from_traj(k, S):
        """
        Compute cost J from trajectory S.
        Returns:
          J: scalar
          a: (N+1,)   control at nodes
          L: (N+1,)   running cost at nodes
        """
        a = S[:, :2] @ k
        L = jax.vmap(running_cost)(S, a)
        J = jnp.sum(0.5 * dt * (L[:-1] + L[1:]))
        return J, a, L

    # -------------------------------------------------------------------------
    # 6) Classical continuous adjoint (backward integration via time reversal)
    # -------------------------------------------------------------------------
    def adjoint_from_traj(k, S):
        """
        Adjoint ODE:
          lambda_dot = -Acl^T lambda - (∂L/∂s)^T,   lambda(T)=0

        Time reversal:
          τ = T - t  =>  dλ/dτ = +Acl^T λ + ∂L/∂s
        """
        AclT = A_cl(k).T

        gradL = jax.vmap(lambda s: grad_running_cost_s(s, k))(S)  # (N+1,4)
        gradL_rev = gradL[::-1]

        def rhs_rev(lam, g_here):
            return AclT @ lam + g_here

        def rk4_step_adj_rev(lam, g_curr, g_mid, g_next):
            k1_ = rhs_rev(lam, g_curr)
            k2_ = rhs_rev(lam + 0.5 * dt * k1_, g_mid)
            k3_ = rhs_rev(lam + 0.5 * dt * k2_, g_mid)
            k4_ = rhs_rev(lam + dt * k3_, g_next)
            return lam + (dt / 6.0) * (k1_ + 2.0 * k2_ + 2.0 * k3_ + k4_)

        lam_T = jnp.zeros((4,))  # lambda(T)=0  <=>  lambda(τ=0)=0

        def step(lam, j):
            g_curr = gradL_rev[j]
            g_next = gradL_rev[j + 1]
            g_mid = 0.5 * (g_curr + g_next)
            lam_next = rk4_step_adj_rev(lam, g_curr, g_mid, g_next)
            return lam_next, lam_next

        _, lam_next = jax.lax.scan(step, lam_T, jnp.arange(N))
        lam_rev = jnp.vstack([lam_T[None, :], lam_next])  # (N+1,4) in reversed time
        lam = lam_rev[::-1]  # flip back so lam[i] matches t[i]
        return lam

    # -------------------------------------------------------------------------
    # 7) Gradient wrt k (2 parameters) using the adjoint
    # -------------------------------------------------------------------------
    def grad_wrt_k(k, S, a, lam):
        """
        Node-wise integrand:
          gk(t) = [s1, s1d] * ( 2*r_a*a + B^T*lambda )
        """
        BTlam = lam @ B  # (N+1,)
        s = 2.0 * r_a * a + BTlam
        g_nodes = S[:, :2] * s[:, None]  # (N+1,2)
        gk = jnp.sum(0.5 * dt * (g_nodes[:-1] + g_nodes[1:]), axis=0)  # (2,)
        return gk

    # -------------------------------------------------------------------------
    # 8) JIT-ed wrappers
    # -------------------------------------------------------------------------
    @jax.jit
    def cost_and_grad_manual(k):
        S = forward_traj(k)
        J, a, _ = cost_from_traj(k, S)
        lam = adjoint_from_traj(k, S)
        gk = grad_wrt_k(k, S, a, lam)
        return J, gk

    @jax.jit
    def simulate_traj_outputs(k):
        S = forward_traj(k)
        J, a, _ = cost_from_traj(k, S)
        return S, a, J

    # -------------------------------------------------------------------------
    # 9) Optax ADAM loop
    # -------------------------------------------------------------------------
    k = jnp.array(k0)

    optimizer = optax.adam(learning_rate=lr, b1=betas[0], b2=betas[1], eps=eps)
    opt_state = optimizer.init(k)

    J_hist = []
    gnorm_hist = []
    k_hist = []
    K_hist = []

    best_J = np.inf
    best_k = None
    best_it = -1

    for it in range(1, max_iter + 1):
        J, gk = cost_and_grad_manual(k)

        Jf = float(J)
        gnorm = float(jnp.linalg.norm(gk))

        J_hist.append(Jf)
        gnorm_hist.append(gnorm)
        k_hist.append(np.array(k))

        if Jf < best_J:
            best_J = Jf
            best_k = np.array(k)
            best_it = it

        if it % print_every == 0 or it == 1:
            print(
                f"[Adjoint on reduced state + Optax ADAM] "
                f"it={it:4d}  J={Jf:.6e}  k={np.array(k)}"
            )

        updates, opt_state = optimizer.update(gk, opt_state, params=k)
        k = optax.apply_updates(k, updates)
        K_hist.append(k)

    if best_k is None:
        k_opt = np.array(k)
        J_final = float(J_hist[-1]) if J_hist else None
        it_final = len(J_hist)
        best_it = it_final
    else:
        k_opt = best_k
        J_final = best_J
        it_final = best_it

    # -------------------------------------------------------------------------
    # 10) Simulate passive and optimal trajectories
    # -------------------------------------------------------------------------
    S_passive, a_passive, J_passive = simulate_traj_outputs(jnp.zeros(2))
    S_opt, a_opt, J_opt = simulate_traj_outputs(jnp.array(k_opt))

    info = {
        "success": True,
        "message": "Adjoint + Optax ADAM completed (reduced State)",
        "nit": it_final,
        "J": J_final,
        "k_opt": k_opt,
        "J_hist": np.array(J_hist),
        "gnorm_hist": np.array(gnorm_hist),
        "best_iter": best_it,
        "K_full_opt": np.array([k_opt[0], k_opt[1], 0.0, 0.0]),
    }

    return (
        t,
        np.array(S_passive),
        np.array(S_opt),
        np.array(a_opt),
        k_opt,
        info,
        K_hist,
    )




# =============================================================================
# Run
# =============================================================================
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
    max_iter = 2_000 # maximum number of ADAM iterations
    lr=2e-2 # learning rate for ADAM
    K0 = np.zeros(2)

    start = time.time()
    t,sol_noControl, sol_control, a_series, K_opt, info, K_hist = simulate_adjoint_reducedState(
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
        k0=K0,
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

    from _plottingFunc import plot_optimization_history,plot_gain_history_reducedState
    # Optimization history
    plot_optimization_history(info)

    # Gain history
    plot_gain_history_reducedState(K_hist)


