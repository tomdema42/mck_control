# -*- coding: utf-8 -*-
"""
Continuous-time adjoint optimization of a constant state-feedback gain k = [k0, k1]
Nonlinear system with a cubic stiffness acting on m1

@author: demaria
"""

import time
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

from _auxFunc import load_params, build_dyn_system_jnp
from _ADJ_functions import make_time_grid, k_full


# =============================================================================
# Global plotting / JAX settings
# =============================================================================
plt.rcParams.update({"font.size": 18})
jax.config.update("jax_enable_x64", True)

# =============================================================================
# Core solver + adjoint + optimization
# =============================================================================
def simulate_adjoint_reducedState_nonLinear(
    m1,  m2,
    k1, k2,
    c1, c2,
    kc,  cd,
    k3,
    t_end,
    y0,
    N,
    w_s1, w_s1d,
    w_e, w_ed,
    r_a,
    max_iter,
    k0,
    lr=1e-2,
    betas=(0.9, 0.999),
    eps=1e-8,
    print_every=25,
):
    # -------------------------------------------------------------------------
    # 1) Time grid (NO forcing)
    # -------------------------------------------------------------------------
    t, dt = make_time_grid(t_end, N)
    y0 = jnp.array(y0)

    # -------------------------------------------------------------------------
    # 2) Linear system matrices (shared builder)
    # -------------------------------------------------------------------------
    A, B = build_dyn_system_jnp(m1, m2, k1, k2, c1, c2, kc, cd)

    k3_over_m1 = k3 / m1
    dk3ds1_over_m1 = (3.0 * k3) / m1

    # -------------------------------------------------------------------------
    # 3) Control law + running cost and its gradient w.r.t state
    # -------------------------------------------------------------------------
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
        ∂L/∂s with a = k0*s1 + k1*s1d:
          ∂(r_a a^2)/∂s = 2*r_a*a * [k0,k1,0,0]
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
    # 4) Nonlinear closed-loop dynamics + Jacobian wrt s
    # -------------------------------------------------------------------------
    def A_cl(k):
        """
        Linear part of closed-loop Jacobian (constant for given k):
            A_lin_cl = A + B*K_full
        """
        return A + jnp.outer(B, k_full(k))

    def g_nonlin(s):
        """
        Nonlinear term (affects only s1dd):
            s1dd += -(k3/m1) * s1^3
        """
        return jnp.array([0.0, -k3_over_m1 * s[0] ** 3, 0.0, 0.0])

    def f_closed(s, k):
        """
        Nonlinear dynamics (NO forcing):
            sdot = A s + B a + g_nonlin(s)
        """
        a = control_a(s, k)
        return A @ s + B * a + g_nonlin(s)

    def jac_f_s(s, k):
        """
        State Jacobian ∂f/∂s (4x4), evaluated at s:
          ∂f/∂s = (A + outer(B, K_full)) + J_nonlin(s)
        Nonlinear contribution:
          (row=1) = -(3*k3/m1) * s1^2
        """
        J = A_cl(k)
        extra = jnp.zeros((4, 4)).at[1, 0].set(-dk3ds1_over_m1 * s[0] ** 2)
        return J + extra

    # -------------------------------------------------------------------------
    # 5) RK4 forward integrator
    # -------------------------------------------------------------------------
    def rk4_step_state(s, k):
        k1_ = f_closed(s, k)
        k2_ = f_closed(s + 0.5 * dt * k1_, k)
        k3_ = f_closed(s + 0.5 * dt * k2_, k)
        k4_ = f_closed(s + dt * k3_, k)
        return s + (dt / 6.0) * (k1_ + 2.0 * k2_ + 2.0 * k3_ + k4_)

    def forward_traj(k):
        """
        Forward simulation on the fixed grid.
        Returns:
          S: (N+1, 4), with S[i] = s(t[i])
        """
        def step(s, _i):
            s_next = rk4_step_state(s, k)
            return s_next, s_next

        _, S_next = jax.lax.scan(step, y0, jnp.arange(N))
        S = jnp.vstack([y0[None, :], S_next])
        return S

    # -------------------------------------------------------------------------
    # 6) Cost evaluation (trapezoidal rule on nodes)
    # -------------------------------------------------------------------------
    def cost_from_traj(k, S):
        """
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
    # 7) Continuous adjoint with state-dependent Jacobian (time reversal)
    # -------------------------------------------------------------------------
    def adjoint_from_traj(k, S):
        """
        Adjoint ODE:
          lambda_dot = -J_s(t)^T lambda - (∂L/∂s)^T,   lambda(T)=0
        where J_s(t) = ∂f/∂s evaluated along S(t).
        """
        gradL = jax.vmap(lambda s: grad_running_cost_s(s, k))(S)
        Js_nodes = jax.vmap(lambda s: jac_f_s(s, k))(S)

        gradL_rev = gradL[::-1]
        Js_rev = Js_nodes[::-1]
        JsT_rev = jnp.swapaxes(Js_rev, 1, 2)

        def rhs_rev(lam, JT_here, g_here):
            return JT_here @ lam + g_here

        def rk4_step_adj_rev(lam, JT_curr, g_curr, JT_mid, g_mid, JT_next, g_next):
            k1_ = rhs_rev(lam, JT_curr, g_curr)
            k2_ = rhs_rev(lam + 0.5 * dt * k1_, JT_mid, g_mid)
            k3_ = rhs_rev(lam + 0.5 * dt * k2_, JT_mid, g_mid)
            k4_ = rhs_rev(lam + dt * k3_, JT_next, g_next)
            return lam + (dt / 6.0) * (k1_ + 2.0 * k2_ + 2.0 * k3_ + k4_)

        lam_T = jnp.zeros((4,))

        def step(lam, j):
            JT_curr = JsT_rev[j]
            JT_next = JsT_rev[j + 1]
            JT_mid = 0.5 * (JT_curr + JT_next)

            g_curr = gradL_rev[j]
            g_next = gradL_rev[j + 1]
            g_mid = 0.5 * (g_curr + g_next)

            lam_next = rk4_step_adj_rev(lam, JT_curr, g_curr, JT_mid, g_mid, JT_next, g_next)
            return lam_next, lam_next

        _, lam_next = jax.lax.scan(step, lam_T, jnp.arange(N))
        lam_rev = jnp.vstack([lam_T[None, :], lam_next])
        lam = lam_rev[::-1]
        return lam

    # -------------------------------------------------------------------------
    # 8) Gradient wrt k (2 parameters) using the adjoint
    # -------------------------------------------------------------------------
    def grad_wrt_k(k, S, a, lam):
        """
        Node-wise integrand:
          gk(t) = [s1, s1d] * ( 2*r_a*a + B^T*lambda )
        """
        BTlam = lam @ B
        s = 2.0 * r_a * a + BTlam
        g_nodes = S[:, :2] * s[:, None]
        gk = jnp.sum(0.5 * dt * (g_nodes[:-1] + g_nodes[1:]), axis=0)
        return gk

    # -------------------------------------------------------------------------
    # 9) JIT-ed wrappers
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
    # 10) Optax ADAM loop using the manual gradient
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
                f"[Adjoint on reduced state + Optax ADAM (nonlinear)] "
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
    # 11) Simulate passive and optimal trajectories for comparison
    # -------------------------------------------------------------------------
    S_passive, a_passive, J_passive = simulate_traj_outputs(jnp.zeros(2))
    S_opt, a_opt, J_opt = simulate_traj_outputs(jnp.array(k_opt))

    info = {
        "success": True,
        "message": "Adjoint + Optax ADAM completed (reduced State, nonlinear)",
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





#==========================================================
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
    k3 = p["k3"]
    # Weights for Q via outputs: [s1, s1d, e=s1+s2, ed=s1d+s2d]
    w_s1 = p['w_s1']
    w_s1d = p['w_s1d']
    w_e = p['w_e']
    w_ed = p['w_ed']
    # Weight for R:
    r_a  = p['r_a']
    # Simulation time and initial state
    t_end = p['t_end']
    y0 = (p['s1_0'], p['s1d_0'], p['s2_0'], p['s2d_0'])
    
    dt = 0.01
    N = int(t_end/dt)

    # Optimization hyperparameters
    max_iter = 1_000
    lr = 2e-2
    K0 = np.zeros(2)

    start = time.time()
    t, sol_noControl, sol_control, a_series, K_opt, info, K_hist = simulate_adjoint_reducedState_nonLinear(
        m1, m2,
        k1, k2,
        c1, c2,
        kc, cd,
        k3,
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


    # -----------------------------
    # Plots
    # -----------------------------
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

    from _plottingFunc import plot_optimization_history, plot_gain_history_reducedState
    # Optimization history
    plot_optimization_history(info)

    # Gain history
    plot_gain_history_reducedState(K_hist)

