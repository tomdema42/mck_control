# -*- coding: utf-8 -*-
"""
Classical (continuous-time) adjoint optimization of a *restricted* constant state-feedback law (NO saturation).

Control law (ONLY uses x1 and x1d):
    u(t) = k0 * x1(t) + k1 * x1d(t)
where the full state is x = [x1, x1d, x2, x2d] and the design vector is k = [k0, k1].

Closed-loop dynamics:
    xdot = A x + B u + b(t)

Running cost:
    L(x,u) = w_x1 x1^2 + w_x1d x1d^2 + w_e (x1+x2)^2 + w_ed (x1d+x2d)^2 + r_u u^2

Continuous-time adjoint:
    lambda_dot = - (∂f/∂x)^T lambda - (∂L/∂x)^T
    lambda(T) = 0

For this linear system with constant k:
    ∂f/∂x = A + B*K_full
where K_full = [k0, k1, 0, 0] and (B*K_full) is an outer product.

Gradient wrt k (2 parameters):
    dJ/dk = ∫ [x1, x1d]^T * ( 2*r_u*u + B^T*lambda ) dt

Notes:
- This is the *classical* continuous adjoint (not the exact discrete adjoint of RK4).
- Forward state: RK4 on a fixed grid.
- Adjoint: RK4 backward in time (implemented via time reversal).
- Optax ADAM is used, but gradients are manual (no jax.grad).
- Optionally plots the 2D cost landscape J(k0,k1) and the ADAM path.

@author: demaria
"""

import time
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

from _auxFunc import load_params, make_forcing


# =============================================================================
# Global plotting / JAX settings
# =============================================================================
plt.rcParams.update({"font.size": 18})
jax.config.update("jax_enable_x64", True)


# =============================================================================
# Utilities
# =============================================================================
def precompute_time_and_forcing(F_ext, t_end, N):
    """
    Build a fixed time grid and precompute external forcing on:
      - nodes:    t[i]
      - halfsteps t[i] + dt/2

    Why NumPy here?
    ---------------
    F_ext is a Python callable (not JAX-traceable), so we evaluate it outside JIT
    and then convert the arrays to JAX.
    """
    t = np.linspace(0.0, t_end, N + 1)
    dt = float(t[1] - t[0])

    t_half = t[:-1] + 0.5 * dt

    F_nodes_np = np.array([F_ext(ti) for ti in t])
    F_half_np = np.array([F_ext(ti) for ti in t_half])

    F_nodes = jnp.array(F_nodes_np)
    F_half = jnp.array(F_half_np)

    return t, dt, F_nodes, F_half


def build_linear_system(m1, m2, k1, k2, c1, c2, kc, cd):
    """
    Build the constant matrices for the 2-DOF mass-spring-damper system.

    State:
      x = [x1, x1d, x2, x2d]

    Input:
      u is scalar, applied on x2dd only (through B).
    """
    A = jnp.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-(k1 + kc) / m1, -(c1 + cd) / m1, kc / m1, cd / m1],
            [0.0, 0.0, 0.0, 1.0],
            [kc / m2, cd / m2, -(k2 + kc) / m2, -(c2 + cd) / m2],
        ]
    )

    # u acts only on x2dd => 4th state derivative
    B = jnp.array([0.0, 0.0, 0.0, 1.0 / m2])

    def forcing_vec(F_scalar):
        # External force applied on DOF1 acceleration: +F/m1 in x1dd equation
        return jnp.array([0.0, F_scalar / m1, 0.0, 0.0])

    return A, B, forcing_vec


# =============================================================================
# Core solver + adjoint + optimization
# =============================================================================
def simulate_2dof_with_optax_adam_classical_adjoint_x1x1d_control(
    m1,
    m2,
    k1,
    k2,
    c1,
    c2,
    kc,
    cd,
    F_ext,  # Python callable
    t_end,
    y0,
    N,
    w_x1,
    w_x1d,
    w_e,
    w_ed,
    r_u,
    max_iter,
    k0=None,  # initial k = [k0, k1]
    lr=1e-2,
    betas=(0.9, 0.999),
    eps=1e-8,
    print_every=25,
):
    # -------------------------------------------------------------------------
    # 1) Time grid + forcing values
    # -------------------------------------------------------------------------
    t, dt, F_nodes, F_half = precompute_time_and_forcing(F_ext, t_end, N)
    y0 = jnp.array(y0)

    # -------------------------------------------------------------------------
    # 2) System matrices and forcing mapping
    # -------------------------------------------------------------------------
    A, B, forcing_vec = build_linear_system(m1, m2, k1, k2, c1, c2, kc, cd)

    # -------------------------------------------------------------------------
    # 3) Control law + running cost and its gradient w.r.t state
    # -------------------------------------------------------------------------
    def k_full(k):
        """Embed k=[k0,k1] into full-state gain [k0,k1,0,0]."""
        return jnp.array([k[0], k[1], 0.0, 0.0])

    def control_u(x, k):
        """Restricted feedback: u = k0*x1 + k1*x1d."""
        return jnp.dot(k, x[:2])

    def running_cost(x, u):
        """Scalar running cost L(x,u)."""
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

    def grad_running_cost_x(x, k):
        """
        ∂L/∂x (shape (4,)).

        The term r_u*u^2 contributes:
          ∂(r_u u^2)/∂x = 2*r_u*u * ∂u/∂x
        and since u = k0*x1 + k1*x1d:
          ∂u/∂x = [k0, k1, 0, 0]
        """
        x1_, x1d_, x2_, x2d_ = x
        e = x1_ + x2_
        ed = x1d_ + x2d_

        u = control_u(x, k)

        g_state = jnp.array(
            [
                2.0 * (w_x1 * x1_ + w_e * e),
                2.0 * (w_x1d * x1d_ + w_ed * ed),
                2.0 * (w_e * e),
                2.0 * (w_ed * ed),
            ]
        )

        g_control = 2.0 * r_u * u * k_full(k)
        return g_state + g_control

    # -------------------------------------------------------------------------
    # 4) Closed-loop dynamics + RK4 forward integrator
    # -------------------------------------------------------------------------
    def A_cl(k):
        """
        Closed-loop Jacobian ∂f/∂x:
            A_cl = A + B*K_full
        Since u is scalar and B is (4,), B*K_full is outer(B, K_full) -> (4,4).
        """
        return A + jnp.outer(B, k_full(k))

    def f_closed(x, k, F_scalar):
        """xdot = A x + B u + b(F)."""
        u = control_u(x, k)
        return A @ x + B * u + forcing_vec(F_scalar)

    def rk4_step_state(x, k, Fi, Fh, Fi1):
        """
        One RK4 step from node i -> i+1 using:
          Fi  = F(t_i)
          Fh  = F(t_i + dt/2)
          Fi1 = F(t_{i+1})
        """
        k1_ = f_closed(x, k, Fi)
        k2_ = f_closed(x + 0.5 * dt * k1_, k, Fh)
        k3_ = f_closed(x + 0.5 * dt * k2_, k, Fh)
        k4_ = f_closed(x + dt * k3_, k, Fi1)
        return x + (dt / 6.0) * (k1_ + 2.0 * k2_ + 2.0 * k3_ + k4_)

    def forward_traj(k):
        """
        Forward simulation on the fixed grid.

        Returns:
          X: (N+1, 4), with X[i] = x(t[i])
        """
        def step(x, i):
            Fi = F_nodes[i]
            Fh = F_half[i]
            Fi1 = F_nodes[i + 1]
            x_next = rk4_step_state(x, k, Fi, Fh, Fi1)
            return x_next, x_next

        _, X_next = jax.lax.scan(step, y0, jnp.arange(N))
        X = jnp.vstack([y0[None, :], X_next])
        return X

    # -------------------------------------------------------------------------
    # 5) Cost evaluation (trapezoidal rule on nodes)
    # -------------------------------------------------------------------------
    def cost_from_traj(k, X):
        """
        Returns:
          J: scalar
          u: (N+1,)   control at nodes
          L: (N+1,)   running cost at nodes
        """
        u = X[:, :2] @ k
        L = jax.vmap(running_cost)(X, u)
        J = jnp.sum(0.5 * dt * (L[:-1] + L[1:]))
        return J, u, L

    # -------------------------------------------------------------------------
    # 6) Classical continuous adjoint (backward integration via time reversal)
    # -------------------------------------------------------------------------
    def adjoint_from_traj(k, X):
        """
        Adjoint ODE:
          lambda_dot = -Acl^T lambda - (∂L/∂x)^T,   lambda(T)=0

        Implementation trick (time reversal):
          Define τ = T - t  =>  dλ/dτ = +Acl^T λ + ∂L/∂x
          Integrate forward in τ on the reversed grid, then flip back.
        """
        AclT = A_cl(k).T

        gradL = jax.vmap(lambda x: grad_running_cost_x(x, k))(X)  # (N+1,4)
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
            g_mid = 0.5 * (g_curr + g_next)  # simple midpoint approximation
            lam_next = rk4_step_adj_rev(lam, g_curr, g_mid, g_next)
            return lam_next, lam_next

        _, lam_next = jax.lax.scan(step, lam_T, jnp.arange(N))
        lam_rev = jnp.vstack([lam_T[None, :], lam_next])  # (N+1,4) in reversed time
        lam = lam_rev[::-1]  # flip back so lam[i] matches t[i]
        return lam

    # -------------------------------------------------------------------------
    # 7) Gradient wrt k (2 parameters) using the adjoint
    # -------------------------------------------------------------------------
    def grad_wrt_k(k, X, u, lam):
        """
        Node-wise integrand:
          gk(t) = [x1, x1d] * ( 2*r_u*u + B^T*lambda )

        where:
          2*r_u*u         comes from ∂(r_u u^2)/∂u
          B^T*lambda      comes from ∂(lambda^T f)/∂u with ∂f/∂u = B
        """
        BTlam = lam @ B  # (N+1,)
        s = 2.0 * r_u * u + BTlam  # (N+1,)
        g_nodes = X[:, :2] * s[:, None]  # (N+1,2)
        gk = jnp.sum(0.5 * dt * (g_nodes[:-1] + g_nodes[1:]), axis=0)  # (2,)
        return gk

    # -------------------------------------------------------------------------
    # 8) JIT-ed wrappers (one forward + one adjoint pass per gradient call)
    # -------------------------------------------------------------------------
    @jax.jit
    def cost_and_grad_manual(k):
        X = forward_traj(k)
        J, u, _ = cost_from_traj(k, X)
        lam = adjoint_from_traj(k, X)
        gk = grad_wrt_k(k, X, u, lam)
        return J, gk

    @jax.jit
    def cost_only(k):
        X = forward_traj(k)
        J, _, _ = cost_from_traj(k, X)
        return J

    @jax.jit
    def simulate_traj_outputs(k):
        X = forward_traj(k)
        J, u, _ = cost_from_traj(k, X)
        return X, u, J

    # -------------------------------------------------------------------------
    # 9) Optax ADAM loop using the manual gradient
    # -------------------------------------------------------------------------
    if k0 is None:
        k0 = np.zeros(2)
    k = jnp.array(k0)

    optimizer = optax.adam(learning_rate=lr, b1=betas[0], b2=betas[1], eps=eps)
    opt_state = optimizer.init(k)

    J_hist = []
    gnorm_hist = []
    k_hist = []     # numpy snapshots of k per iteration (easy to plot)
    K_hist = []     # JAX k after update (kept for backward compatibility with your plots)

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
                f"[Classical Adjoint + ADAM | u=k0*x1+k1*x1d] "
                f"it={it:4d}  J={Jf:.6e}  ||g||={gnorm:.3e}  k={np.array(k)}"
            )

        # ADAM update
        updates, opt_state = optimizer.update(gk, opt_state, params=k)
        k = optax.apply_updates(k, updates)
        K_hist.append(k)

    # Prefer best iterate (often better than last for ADAM)
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
    # 10) Simulate passive and optimal trajectories for comparison
    # -------------------------------------------------------------------------
    X_passive, u_passive, J_passive = simulate_traj_outputs(jnp.zeros(2))
    X_opt, u_opt, J_opt = simulate_traj_outputs(jnp.array(k_opt))

    info = {
        "success": True,
        "message": "Classical adjoint + Optax ADAM completed (u depends only on x1,x1d)",
        "nit": it_final,
        "J": J_final,
        "k_opt": k_opt,
        "J_hist": np.array(J_hist),
        "gnorm_hist": np.array(gnorm_hist),
        "k_hist": np.array(k_hist),
        "best_iter": best_it,
        "K_full_opt": np.array([k_opt[0], k_opt[1], 0.0, 0.0]),
        "cost_only": cost_only,
        "J_passive": float(J_passive),
        "J_opt_recomputed": float(J_opt),
    }

    return (
        t,
        np.array(X_passive),
        np.array(X_opt),
        np.array(u_opt),
        k_opt,
        info,
        K_hist,
    )


# =============================================================================
# Plotting helpers
# =============================================================================
def plot_cost_landscape_2d(cost_only_fn, k_opt, k_hist, window=(2.0, 2.0), n_grid=70, use_log=True):
    """
    Plot J(k0,k1) as a filled contour and overlay the optimization path.

    cost_only_fn: callable(k)->J (JAX-jitted ok)
    k_opt: array-like shape (2,)
    k_hist: array shape (iters, 2)
    """
    dk0, dk1 = window

    k0_vals = np.linspace(k_opt[0] - dk0, k_opt[0] + dk0, n_grid)
    k1_vals = np.linspace(k_opt[1] - dk1, k_opt[1] + dk1, n_grid)

    K0, K1 = np.meshgrid(k0_vals, k1_vals, indexing="xy")
    K_flat = jnp.stack([K0.ravel(), K1.ravel()], axis=1)

    @jax.jit
    def batch_cost(K_batch):
        return jax.vmap(cost_only_fn)(K_batch)

    J_flat = np.array(batch_cost(K_flat))
    J_grid = J_flat.reshape(K0.shape)

    if use_log:
        Z = np.log10(J_grid - J_grid.min() + 1e-12)
        Z_label = "log10(J - Jmin)"
    else:
        Z = J_grid
        Z_label = "J"

    plt.figure(figsize=(7.5, 6))
    cf = plt.contourf(K0, K1, Z, levels=40)
    plt.colorbar(cf, label=Z_label)

    if k_hist is not None and len(k_hist) > 0:
        plt.plot(k_hist[:, 0], k_hist[:, 1], "w.-", lw=1, ms=3, label="ADAM path")

    plt.plot(k_opt[0], k_opt[1], "r*", ms=12, label="best k")

    plt.xlabel("k0")
    plt.ylabel("k1")
    plt.title("Cost landscape")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_states(t, X_passive, X_opt):
    x1_0, x1d_0, x2_0, x2d_0 = X_passive.T
    x1_1, x1d_1, x2_1, x2d_1 = X_opt.T

    plt.figure(figsize=(12, 6))
    plt.plot(t, x1_0, label="x1 passive")
    plt.plot(t, x2_0, label="x2 passive")
    plt.plot(t, x1_1, "--", label="x1 closed-loop (k*)")
    plt.plot(t, x2_1, "--", label="x2 closed-loop (k*)")
    plt.xlabel("t [s]")
    plt.ylabel("Displacement [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_control(t, u_nodes):
    plt.figure(figsize=(12, 4))
    plt.plot(t, u_nodes, label="u(t)=k0*x1+k1*x1d")
    plt.xlabel("t [s]")
    plt.ylabel("Control force [N]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cost_history(J_hist):
    plt.figure(figsize=(12, 4))
    plt.plot(J_hist)
    plt.xlabel("ADAM iteration")
    plt.ylabel("J")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_k_history(K_hist):
    K_hist_np = np.array(K_hist)
    plt.figure(figsize=(12, 4))
    plt.plot(K_hist_np[:, 0], label="k0")
    plt.plot(K_hist_np[:, 1], label="k1")
    plt.xlabel("ADAM iteration")
    plt.ylabel("k")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    start = time.time()

    # -----------------------------
    # Load parameters + forcing
    # -----------------------------
    p = load_params("params.txt")

    m1, m2 = p["m1"], p["m2"]
    k1, k2 = p["k1"], p["k2"]
    c1, c2 = p["c1"], p["c2"]
    cd, kc = p["cd"], p["kc"]

    F_ext = make_forcing(p)

    # -----------------------------
    # Simulation + optimization setup
    # -----------------------------
    t_end = 10.0
    y0 = (0.3, 0.5, 0.0, 0.0)

    N = 400
    w_x1, w_x1d = 1.0, 0.1
    w_e, w_ed = 50.0, 2.0
    r_u = 0.05

    max_iter = 6_000
    k_init = np.zeros(2)
    # k_init = np.array([-10.0, -20.0])

    # -----------------------------
    # Run adjoint + ADAM
    # -----------------------------
    t, X0, X1, u_nodes, k_opt, info, K_hist = simulate_2dof_with_optax_adam_classical_adjoint_x1x1d_control(
        m1,
        m2,
        k1,
        k2,
        c1,
        c2,
        kc,
        cd,
        F_ext,
        t_end,
        y0,
        N,
        w_x1,
        w_x1d,
        w_e,
        w_ed,
        r_u,
        max_iter,
        k0=k_init,
        lr=2e-2,
        print_every=50,
    )

    print("\nINFO:")
    for kk, vv in info.items():
        if kk in ("J_hist", "gnorm_hist", "k_hist", "cost_only"):
            continue
        print(f"  {kk}: {vv}")
    print("k_opt =", k_opt)
    print("K_full_opt =", info["K_full_opt"])

    print("Total Time:", time.time() - start)

    # -----------------------------
    # Plots
    # -----------------------------
    plot_states(t, X0, X1)
    plot_control(t, u_nodes)
    plot_cost_history(info["J_hist"])
    plot_k_history(K_hist)

    plot_cost_landscape_2d(
        cost_only_fn=info["cost_only"],
        k_opt=k_opt,
        k_hist=info["k_hist"],
        window=(3, 3),
        n_grid=100,
        use_log=True,
    )
