# -*- coding: utf-8 -*-
"""
Classical (continuous-time) adjoint optimization of a *full-state* constant state-feedback law (NO saturation),
WITH BILINEAR STIFFNESS:
    k2(t) = k2_star + alpha * u(t)

State:
    x = [x1, x1d, x2, x2d]

Control law (full-state linear feedback):
    u(t) = K @ x(t) = K1*x1 + K2*x1d + K3*x2 + K4*x2d

Dynamics:
    xdot = A_star x + B u + b(t) + g_bilin(x,u)

- A_star uses k2_star (constant part of stiffness)
- Bilinear term affects ONLY x2dd:
      x2dd += -(alpha/m2) * u * x2

Running cost:
    L(x,u) = w_x1 x1^2 + w_x1d x1d^2 + w_e (x1+x2)^2 + w_ed (x1d+x2d)^2 + r_u u^2

Adjoint (classical, continuous-time, nonlinear dynamics):
    lambda_dot = - (∂f/∂x)^T lambda - (∂L/∂x)^T
    lambda(T) = 0

Gradient wrt K (since K enters only through u):
    dJ/dK = ∫ (∂u/∂K)^T [ ∂L/∂u + (∂f/∂u)^T lambda ] dt
where:
    ∂u/∂K = x
    ∂L/∂u = 2*r_u*u
    ∂f/∂u = (1 - alpha*x2) * B
=>  dJ/dK = ∫ x * ( 2*r_u*u + (1 - alpha*x2)*(B^T*lambda) ) dt

Numerics:
- Forward state: RK4 on fixed grid.
- Adjoint: RK4 backward in time (implemented as forward integration in reversed time).
- Optax ADAM: manual gradients (no jax.grad).

@author: demaria
"""

import time
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

from _auxFunc import load_params, make_forcing

jax.config.update("jax_enable_x64", True)


# =============================================================================
# Plot helpers
# =============================================================================
def plot_cost_landscape_2d_fixed_others(
    cost_only_fn,
    K_opt,
    K_hist=None,
    idx=(0, 1),
    window=(2.0, 2.0),
    n_grid=70,
    use_log=True,
):
    """
    Plot a 2D slice of the cost J over (K[idx[0]], K[idx[1]]) keeping the other K components fixed to K_opt.

    Parameters
    ----------
    cost_only_fn : callable
        JAX-jitted function cost_only_fn(K) -> scalar J.
    K_opt : array-like shape (4,)
        Reference gain around which to slice the landscape.
    K_hist : array-like shape (n_iter,4), optional
        If provided, overlays the optimizer path projected onto the chosen 2D plane.
    idx : tuple(int,int)
        Indices of K to vary (e.g. (0,1) means vary K1 and K2).
    window : tuple(float,float)
        Half-width around K_opt for each varied component.
    n_grid : int
        Grid resolution per axis.
    use_log : bool
        If True plot log10(J - Jmin + eps) for contrast.
    """
    i0, i1 = idx
    d0, d1 = window

    v0 = np.linspace(K_opt[i0] - d0, K_opt[i0] + d0, n_grid)
    v1 = np.linspace(K_opt[i1] - d1, K_opt[i1] + d1, n_grid)
    V0, V1 = np.meshgrid(v0, v1, indexing="xy")

    # Build a batch of K vectors for evaluation on the grid
    K_base = np.array(K_opt, copy=True)
    K_batch = []
    for a, b in zip(V0.ravel(), V1.ravel()):
        K_tmp = K_base.copy()
        K_tmp[i0] = a
        K_tmp[i1] = b
        K_batch.append(K_tmp)
    K_batch = jnp.array(np.array(K_batch))

    @jax.jit
    def batch_cost(Ks):
        return jax.vmap(cost_only_fn)(Ks)

    J_flat = np.array(batch_cost(K_batch))
    J_grid = J_flat.reshape(V0.shape)

    if use_log:
        Z = np.log10(J_grid - J_grid.min() + 1e-12)
        Z_label = "log10(J - Jmin)"
    else:
        Z = J_grid
        Z_label = "J"

    plt.figure(figsize=(7.5, 6))
    cf = plt.contourf(V0, V1, Z, levels=40)
    plt.colorbar(cf, label=Z_label)

    if K_hist is not None and len(K_hist) > 0:
        plt.plot(K_hist[:, i0], K_hist[:, i1], "w.-", lw=1, ms=3, label="ADAM path")

    plt.plot(K_opt[i0], K_opt[i1], "r*", ms=12, label="best K (slice)")
    plt.xlabel(f"K[{i0}]")
    plt.ylabel(f"K[{i1}]")
    plt.title("Cost landscape (2D slice; other K fixed)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# Main solver: forward + adjoint + manual gradient + Optax ADAM
# =============================================================================
def simulate_2dof_adjoint_opt_fullstate_bilinear_k2(
    m1, m2,
    k1, k2_star,          # k2_star is the constant part of k2(t)
    c1, c2,
    kc, cd,
    alpha,                # k2(t) = k2_star + alpha*u
    F_ext,                # python callable F_ext(t)
    t_end,
    x0,
    N,
    w_x1, w_x1d, w_e, w_ed,
    r_u,
    max_iter,
    K0=None,              # initial K = [K1,K2,K3,K4]
    lr=1e-2,
    betas=(0.9, 0.999),
    eps=1e-8,
    print_every=25,
):
    # -------------------------------------------------------------------------
    # 1) Time grid + precompute forcing on nodes and half-steps (for RK4)
    # -------------------------------------------------------------------------
    t = np.linspace(0.0, t_end, N + 1)
    dt = float(t[1] - t[0])
    t_half = t[:-1] + 0.5 * dt

    # F_ext is a python callable -> evaluate in python then convert to JAX arrays
    F_nodes = jnp.array(np.array([F_ext(ti) for ti in t]))
    F_half = jnp.array(np.array([F_ext(ti) for ti in t_half]))

    x0 = jnp.array(x0)

    # -------------------------------------------------------------------------
    # 2) Constant system matrices (built with k2_star only)
    # -------------------------------------------------------------------------
    # x = [x1, x1d, x2, x2d]
    A_star = jnp.array([
        [0.0, 1.0, 0.0, 0.0],
        [-(k1 + kc) / m1, -(c1 + cd) / m1,  kc / m1,        cd / m1],
        [0.0, 0.0, 0.0, 1.0],
        [ kc / m2,        cd / m2,       -(k2_star + kc) / m2, -(c2 + cd) / m2],
    ])

    # Control input u acts only on x2dd equation
    B = jnp.array([0.0, 0.0, 0.0, 1.0 / m2])

    # Unit selectors used in Jacobians / bilinear term
    e3 = jnp.array([0.0, 0.0, 1.0, 0.0])  # selects x2
    e4 = jnp.array([0.0, 0.0, 0.0, 1.0])  # injects into x2dd

    def forcing_vec(F_scalar):
        """b(t) vector in state-space form (only x1dd has external force)."""
        return jnp.array([0.0, F_scalar / m1, 0.0, 0.0])

    # -------------------------------------------------------------------------
    # 3) Control and running cost (and ∂L/∂x)
    # -------------------------------------------------------------------------
    def control_u(x, K):
        """Full-state linear feedback u = K·x."""
        return jnp.dot(K, x)

    def running_cost(x, u):
        """L(x,u) evaluated at one time node."""
        x1_, x1d_, x2_, x2d_ = x
        e = x1_ + x2_
        ed = x1d_ + x2d_
        return (
            w_x1 * x1_ * x1_
            + w_x1d * x1d_ * x1d_
            + w_e * e * e
            + w_ed * ed * ed
            + r_u * u * u
        )

    def grad_running_cost_x(x, K):
        """
        ∂L/∂x (shape (4,)).

        NOTE: L has explicit state terms + r_u*u^2. Since u = K·x,
              ∂(r_u*u^2)/∂x = 2*r_u*u*K.
        """
        x1_, x1d_, x2_, x2d_ = x
        e = x1_ + x2_
        ed = x1d_ + x2d_

        u = control_u(x, K)

        # gradient of state terms
        g_state = jnp.array([
            2.0 * (w_x1 * x1_ + w_e * e),
            2.0 * (w_x1d * x1d_ + w_ed * ed),
            2.0 * (w_e * e),
            2.0 * (w_ed * ed),
        ])

        # gradient of control-penalty term via u(x)=K·x
        g_control = 2.0 * r_u * u * K
        return g_state + g_control

    # -------------------------------------------------------------------------
    # 4) Closed-loop dynamics f(x,K,t) and RK4 forward integration
    # -------------------------------------------------------------------------
    def f_closed(x, K, F_scalar):
        """
        xdot = A_star x + B u + b(t) + bilinear_term

        Bilinear term (ONLY in x2dd):
            x2dd += -(alpha/m2) * u * x2
        """
        u = control_u(x, K)
        bil = e4 * (-(alpha / m2) * u * x[2])
        return A_star @ x + B * u + forcing_vec(F_scalar) + bil

    def rk4_step_state(x, K, Fi, Fh, Fi1):
        """One RK4 step from node i -> i+1 using Fi, Fh, Fi1."""
        k1_ = f_closed(x, K, Fi)
        k2_ = f_closed(x + 0.5 * dt * k1_, K, Fh)
        k3_ = f_closed(x + 0.5 * dt * k2_, K, Fh)
        k4_ = f_closed(x + dt * k3_, K, Fi1)
        return x + (dt / 6.0) * (k1_ + 2.0 * k2_ + 2.0 * k3_ + k4_)

    def forward_traj(K):
        """
        Simulate forward trajectory on the grid.
        Returns X of shape (N+1, 4), containing all state nodes.
        """
        def step(x, i):
            Fi = F_nodes[i]
            Fh = F_half[i]
            Fi1 = F_nodes[i + 1]
            x_next = rk4_step_state(x, K, Fi, Fh, Fi1)
            return x_next, x_next

        _, X_next = jax.lax.scan(step, x0, jnp.arange(N))
        X = jnp.vstack([x0[None, :], X_next])
        return X

    # -------------------------------------------------------------------------
    # 5) Cost on a trajectory (trapezoidal integration)
    # -------------------------------------------------------------------------
    def cost_from_traj(K, X):
        """
        Compute:
          u_nodes = X @ K
          L_nodes = L(X,u)
          J = ∫ L dt via trapezoidal rule
        """
        u = X @ K  # (N+1,)
        L = jax.vmap(running_cost)(X, u)
        J = jnp.sum(0.5 * dt * (L[:-1] + L[1:]))
        return J, u, L

    # -------------------------------------------------------------------------
    # 6) Jacobian ∂f/∂x (state-dependent because of bilinear term)
    # -------------------------------------------------------------------------
    def jac_f_x(x, K):
        """
        f(x) = A_star x + B u + b + e4 * (-(alpha/m2) * u * x2),  u=K·x

        ∂f/∂x = A_star + outer(B, du/dx) + extra_row4
        where:
            du/dx = K
            extra_row4 = -(alpha/m2) * ( x2*K + u*e3 )
        """
        u = control_u(x, K)
        du_dx = K

        J = A_star + jnp.outer(B, du_dx)

        # Only row 4 gets extra terms from bilinear piece
        add_row4 = -(alpha / m2) * (x[2] * du_dx + u * e3)
        return J.at[3, :].add(add_row4)

    # -------------------------------------------------------------------------
    # 7) Classical adjoint integration (backward in time)
    # -------------------------------------------------------------------------
    def adjoint_from_traj(K, X):
        """
        Forward-in-reversed-time implementation.

        Original adjoint (backward in real time):
            lambda_dot = -Jf(x)^T lambda - (∂L/∂x)^T
            lambda(T) = 0

        Define reversed time τ = T - t:
            dλ/dτ = Jf(x)^T λ + ∂L/∂x

        We build arrays in forward time, reverse them, and integrate forward in τ with RK4.
        """
        gradL = jax.vmap(lambda xi: grad_running_cost_x(xi, K))(X)        # (N+1,4)
        JfT = jax.vmap(lambda xi: jac_f_x(xi, K).T)(X)                    # (N+1,4,4)

        gradL_rev = gradL[::-1]
        JfT_rev = JfT[::-1]

        def rhs_rev(lam, Jt, g):
            return Jt @ lam + g

        def rk4_step_adj_rev(lam, Jt_curr, g_curr, Jt_mid, g_mid, Jt_next, g_next):
            k1_ = rhs_rev(lam, Jt_curr, g_curr)
            k2_ = rhs_rev(lam + 0.5 * dt * k1_, Jt_mid, g_mid)
            k3_ = rhs_rev(lam + 0.5 * dt * k2_, Jt_mid, g_mid)
            k4_ = rhs_rev(lam + dt * k3_, Jt_next, g_next)
            return lam + (dt / 6.0) * (k1_ + 2.0 * k2_ + 2.0 * k3_ + k4_)

        lam0 = jnp.zeros((4,))  # corresponds to lambda(T)=0

        def step(lam, j):
            # trapezoid-like mid values (cheap and stable enough for the adjoint RHS)
            g_curr = gradL_rev[j]
            g_next = gradL_rev[j + 1]
            g_mid = 0.5 * (g_curr + g_next)

            Jt_curr = JfT_rev[j]
            Jt_next = JfT_rev[j + 1]
            Jt_mid = 0.5 * (Jt_curr + Jt_next)

            lam_next = rk4_step_adj_rev(lam, Jt_curr, g_curr, Jt_mid, g_mid, Jt_next, g_next)
            return lam_next, lam_next

        _, lam_next = jax.lax.scan(step, lam0, jnp.arange(N))
        lam_rev = jnp.vstack([lam0[None, :], lam_next])  # in reversed time order
        lam = lam_rev[::-1]                              # back to forward time order
        return lam

    # -------------------------------------------------------------------------
    # 8) Manual gradient dJ/dK using (X, u, lambda)
    # -------------------------------------------------------------------------
    def grad_wrt_K(K, X, u, lam):
        """
        dJ/dK = ∫ x * s dt  (trapezoidal rule)

        s(t) = ∂L/∂u + (∂f/∂u)^T λ
             = 2*r_u*u + (1 - alpha*x2) * (B^T λ)
        """
        BTlam = lam @ B                  # (N+1,)
        factor = 1.0 - alpha * X[:, 2]   # (N+1,)
        s = 2.0 * r_u * u + factor * BTlam

        g_nodes = X * s[:, None]         # (N+1,4)
        gK = jnp.sum(0.5 * dt * (g_nodes[:-1] + g_nodes[1:]), axis=0)
        return gK

    # -------------------------------------------------------------------------
    # 9) JIT wrappers (so the optimizer loop stays in python but heavy work is jitted)
    # -------------------------------------------------------------------------
    @jax.jit
    def cost_and_grad_manual(K):
        X = forward_traj(K)
        J, u, _ = cost_from_traj(K, X)
        lam = adjoint_from_traj(K, X)
        gK = grad_wrt_K(K, X, u, lam)
        return J, gK

    @jax.jit
    def cost_only(K):
        X = forward_traj(K)
        J, _, _ = cost_from_traj(K, X)
        return J

    @jax.jit
    def simulate_traj_outputs(K):
        X = forward_traj(K)
        J, u, _ = cost_from_traj(K, X)
        return X, u, J

    # -------------------------------------------------------------------------
    # 10) Optax ADAM loop (manual gradients)
    # -------------------------------------------------------------------------
    if K0 is None:
        K0 = np.zeros(4)
    K = jnp.array(K0)

    optimizer = optax.adam(learning_rate=lr, b1=betas[0], b2=betas[1], eps=eps)
    opt_state = optimizer.init(K)

    J_hist = []
    gnorm_hist = []
    K_hist = []

    best_J = np.inf
    best_K = None
    best_it = -1

    for it in range(1, max_iter + 1):
        J, gK = cost_and_grad_manual(K)
        Jf = float(J)
        gnorm = float(jnp.linalg.norm(gK))

        J_hist.append(Jf)
        gnorm_hist.append(gnorm)
        K_hist.append(np.array(K))

        if Jf < best_J:
            best_J = Jf
            best_K = np.array(K)
            best_it = it

        if it % print_every == 0 or it == 1:
            print(
                f"[Classical Adjoint + ADAM | bilinear k2=k2*+alpha*u | u=K@x] "
                f"it={it:6d}  J={Jf:.6e}  ||g||={gnorm:.3e}  K={np.array(K)}"
            )

        updates, opt_state = optimizer.update(gK, opt_state, params=K)
        K = optax.apply_updates(K, updates)

    # Best solution bookkeeping
    if best_K is None:
        K_opt = np.array(K)
        J_final = float(J_hist[-1]) if J_hist else None
        it_final = len(J_hist)
        best_it = it_final
    else:
        K_opt = best_K
        J_final = best_J
        it_final = best_it

    # Passive and optimal trajectories
    X_passive, u_passive, _ = simulate_traj_outputs(jnp.zeros(4))
    X_opt, u_opt, _ = simulate_traj_outputs(jnp.array(K_opt))

    info = {
        "success": True,
        "message": "Classical adjoint + Optax ADAM completed (bilinear k2=k2*+alpha*u, u=K@x)",
        "nit": it_final,
        "best_iter": best_it,
        "J": J_final,
        "K_opt": K_opt,
        "J_hist": np.array(J_hist),
        "gnorm_hist": np.array(gnorm_hist),
        "K_hist": np.array(K_hist),
        "cost_only": cost_only,
    }

    return (
        t,
        np.array(X_passive),
        np.array(X_opt),
        np.array(u_opt),
        K_opt,
        info,
    )


# =============================================================================
# Example run + plots
# =============================================================================
if __name__ == "__main__":
    start = time.time()

    # --- load parameters / forcing
    param_file = "params.txt"
    p = load_params(param_file)

    m1, m2 = p["m1"], p["m2"]
    k1, k2_star = p["k1"], p["k2"]
    c1, c2 = p["c1"], p["c2"]
    cd, kc = p["cd"], p["kc"]

    F_ext = make_forcing(p)

    # --- bilinear stiffness coefficient
    alpha = 6

    # --- simulation settings
    t_end = 10.0
    x0 = (0.0, 0.0, 0.0, 0.0)
    N = 400

    # --- cost weights
    w_x1, w_x1d = 1.0, 0.1
    w_e, w_ed = 50.0, 2.0
    r_u = 0.05

    # --- optimizer settings
    max_iter = 50_000
    K_init = np.zeros(4)

    # --- run optimization
    t, X_passive, X_opt, u_nodes, K_opt, info = simulate_2dof_adjoint_opt_fullstate_bilinear_k2(
        m1, m2,
        k1, k2_star,
        c1, c2,
        kc, cd,
        alpha,
        F_ext,
        t_end,
        x0,
        N,
        w_x1, w_x1d, w_e, w_ed,
        r_u,
        max_iter,
        K0=K_init,
        lr=2e-2,
        print_every=50,
    )

    print("\nINFO:")
    for kk, vv in info.items():
        if kk in ("J_hist", "gnorm_hist", "K_hist", "cost_only"):
            continue
        print(f"  {kk}: {vv}")
    print("K_opt =", K_opt)

    print("Total Time:", time.time() - start)

    # --- unpack states
    x1_0, x1d_0, x2_0, x2d_0 = X_passive.T
    x1_1, x1d_1, x2_1, x2d_1 = X_opt.T

    # --- state plots
    plt.figure(figsize=(12, 6))
    plt.plot(t, x1_0, label="x1 passive")
    plt.plot(t, x2_0, label="x2 passive")
    plt.plot(t, x1_1, "--", label="x1 closed-loop (K*)")
    plt.plot(t, x2_1, "--", label="x2 closed-loop (K*)")
    plt.xlabel("t [s]")
    plt.ylabel("Displacement [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- control plot
    plt.figure(figsize=(12, 4))
    plt.plot(t, u_nodes, label="u(t)=K@x")
    plt.xlabel("t [s]")
    plt.ylabel("Control force [N]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- k2(t) plot (bilinear stiffness)
    k2_nodes = k2_star + alpha * u_nodes
    plt.figure(figsize=(12, 4))
    plt.plot(t, k2_nodes, label="k2(t)=k2* + alpha*u(t)")
    plt.axhline(k2_star, ls="--", label="k2* (constant)")
    plt.xlabel("t [s]")
    plt.ylabel("Stiffness k2(t) [N/m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- optimization traces
    plt.figure(figsize=(12, 4))
    plt.plot(info["J_hist"])
    plt.xlabel("ADAM iteration")
    plt.ylabel("J")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    K_hist = info["K_hist"]
    plt.plot(K_hist[:, 0], label="K1")
    plt.plot(K_hist[:, 1], label="K2")
    plt.plot(K_hist[:, 2], label="K3")
    plt.plot(K_hist[:, 3], label="K4")
    plt.xlabel("ADAM iteration")
    plt.ylabel("K_i")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- 2D cost landscape slice (example: K1 vs K2)
    plot_cost_landscape_2d_fixed_others(
        cost_only_fn=info["cost_only"],
        K_opt=K_opt,
        K_hist=info["K_hist"],
        idx=(0, 1),
        window=(3, 3),
        n_grid=100,
        use_log=True,
    )
