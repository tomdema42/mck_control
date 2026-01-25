# -*- coding: utf-8 -*-
"""
Classical (continuous-time) adjoint optimization of a *restricted* constant state-feedback law (NO saturation),
WITH BILINEAR STIFFNESS:
    k2(t) = k2_star + alpha * u(t)

Control law (restricted to x1 and x1d):
    u(t) = k0 * x1(t) + k1 * x1d(t)
where the full state is x = [x1, x1d, x2, x2d] and k = [k0, k1].

Closed-loop dynamics:
    xdot = A_star x + B u + b(t) + g_bilin(x,u)
where A_star is built using k2_star (constant part), and the bilinear term affects only x2dd:
    x2dd += -(alpha/m2) * u * x2

Running cost:
    L(x,u) = w_x1 x1^2 + w_x1d x1d^2 + w_e (x1+x2)^2 + w_ed (x1d+x2d)^2 + r_u u^2

Adjoint ODE (nonlinear dynamics):
    lambda_dot = - (∂f/∂x)^T lambda - (∂L/∂x)^T
    lambda(T) = 0

Gradient wrt k (2 params), since k enters only through u:
    dJ/dk = ∫ (∂u/∂k)^T [ ∂L/∂u + (∂f/∂u)^T lambda ] dt
with:
    ∂u/∂k = [x1, x1d]^T
    ∂L/∂u = 2*r_u*u
    ∂f/∂u = (1 - alpha*x2) * B
so:
    dJ/dk = ∫ [x1, x1d]^T * ( 2*r_u*u + (1 - alpha*x2)*(B^T*lambda) ) dt

Numerics:
- Forward state: RK4 on a fixed time grid.
- Adjoint: RK4 backward in time (implemented as forward integration in reverse-time).
- Optax ADAM used for parameter updates; gradients are computed manually.

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
# 1) Model building blocks
# =============================================================================
def build_system_matrices(m1, m2, k1, k2_star, c1, c2, kc, cd):
    """
    Build the constant part of the state matrix A_star using k2_star,
    and the input vector B (u acts on x2dd only).

    State x = [x1, x1d, x2, x2d]
    """
    A_star = jnp.array([
        [0.0, 1.0, 0.0, 0.0],
        [-(k1 + kc) / m1, -(c1 + cd) / m1,  kc / m1,       cd / m1],
        [0.0, 0.0, 0.0, 1.0],
        [ kc / m2,       cd / m2,      -(k2_star + kc) / m2, -(c2 + cd) / m2],
    ])

    # Input goes only into x2dd (4th state derivative)
    B = jnp.array([0.0, 0.0, 0.0, 1.0 / m2])

    # Convenience selectors:
    e3 = jnp.array([0.0, 0.0, 1.0, 0.0])  # selects x2 component
    e4 = jnp.array([0.0, 0.0, 0.0, 1.0])  # selects x2dd equation (row 4)

    return A_star, B, e3, e4


def forcing_vec(F_scalar, m1):
    """
    External force is applied on DOF1 (i.e., equation for x1dd).
    In state form, it enters as:
        x1dd += F(t)/m1
    """
    return jnp.array([0.0, F_scalar / m1, 0.0, 0.0])


def k_full_from_restricted(k):
    """
    Embed restricted gain k=[k0,k1] into the full-state gain [k0,k1,0,0].
    Useful for writing ∂u/∂x compactly.
    """
    return jnp.array([k[0], k[1], 0.0, 0.0])


def control_u(x, k):
    """Restricted control: u = k0*x1 + k1*x1d."""
    return jnp.dot(k, x[:2])


# =============================================================================
# 2) Cost and its derivatives
# =============================================================================
def running_cost(x, u, w_x1, w_x1d, w_e, w_ed, r_u):
    """Scalar running cost L(x,u)."""
    x1, x1d, x2, x2d = x
    e = x1 + x2
    ed = x1d + x2d
    return (
        w_x1 * x1 * x1
        + w_x1d * x1d * x1d
        + w_e * e * e
        + w_ed * ed * ed
        + r_u * u * u
    )


def grad_running_cost_x(x, k, w_x1, w_x1d, w_e, w_ed, r_u):
    """
    Gradient ∂L/∂x (shape (4,)).

    Note: u depends only on x1 and x1d, so the control penalty contributes:
        ∂(r_u u^2)/∂x = 2*r_u*u * ∂u/∂x
        ∂u/∂x = [k0, k1, 0, 0]
    """
    x1, x1d, x2, x2d = x
    e = x1 + x2
    ed = x1d + x2d

    u = control_u(x, k)

    g_state = jnp.array([
        2.0 * (w_x1 * x1 + w_e * e),
        2.0 * (w_x1d * x1d + w_ed * ed),
        2.0 * (w_e * e),
        2.0 * (w_ed * ed),
    ])

    g_control = 2.0 * r_u * u * k_full_from_restricted(k)
    return g_state + g_control


# =============================================================================
# 3) Dynamics with bilinear stiffness term + RK4 integrator
# =============================================================================
def f_closed(x, k, F_scalar, A_star, B, e4, alpha, m1, m2):
    """
    Closed-loop vector field:
        xdot = A_star x + B u + b(t) + bilinear_term

    Bilinear term comes from:
        k2(t) = k2_star + alpha*u
    so in the x2 equation (spring term) you get an extra contribution:
        x2dd += -(alpha/m2) * u * x2
    which is nonlinear in (x,u) and is a product u*x2.
    """
    u = control_u(x, k)
    bilinear_term = e4 * (-(alpha / m2) * u * x[2])
    
    return A_star @ x + B * u + forcing_vec(F_scalar, m1) + bilinear_term


def rk4_step_state(x, k, Fi, Fh, Fi1, dt, A_star, B, e4, alpha, m1, m2):
    """One RK4 step for the state ODE on interval [t_i, t_{i+1}]."""
    k1 = f_closed(x, k, Fi,  A_star, B, e4, alpha, m1, m2)
    k2 = f_closed(x + 0.5 * dt * k1, k, Fh,  A_star, B, e4, alpha, m1, m2)
    k3 = f_closed(x + 0.5 * dt * k2, k, Fh,  A_star, B, e4, alpha, m1, m2)
    k4 = f_closed(x + dt * k3,       k, Fi1, A_star, B, e4, alpha, m1, m2)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def forward_trajectory(k, y0, N, dt, F_nodes, F_half, A_star, B, e4, alpha, m1, m2):
    """
    Forward simulation on the fixed grid with RK4.
    Returns X of shape (N+1,4).
    """
    def step(x, i):
        Fi = F_nodes[i]
        Fh = F_half[i]
        Fi1 = F_nodes[i + 1]
        x_next = rk4_step_state(x, k, Fi, Fh, Fi1, dt, A_star, B, e4, alpha, m1, m2)
        return x_next, x_next

    _, X_next = jax.lax.scan(step, y0, jnp.arange(N))
    return jnp.vstack([y0[None, :], X_next])


def cost_from_traj(k, X, dt, w_x1, w_x1d, w_e, w_ed, r_u):
    """
    Compute:
      - u on nodes
      - L on nodes
      - J via trapezoidal rule
    """
    u = X[:, :2] @ k  # (N+1,)
    L = jax.vmap(lambda x_i, u_i: running_cost(x_i, u_i, w_x1, w_x1d, w_e, w_ed, r_u))(X, u)
    J = jnp.sum(0.5 * dt * (L[:-1] + L[1:]))
    return J, u, L


# =============================================================================
# 4) Jacobian ∂f/∂x for the adjoint (state-dependent because of bilinear term)
# =============================================================================
def jac_f_x(x, k, A_star, B, e3, alpha, m2):
    """
    f(x) = A_star x + B u + b(t) + e4 * (-(alpha/m2) u x2)
    u = k_full·x, with k_full = [k0,k1,0,0]

    ∂f/∂x = A_star + outer(B, ∂u/∂x) + extra_row4

    For the bilinear piece on row 4:
      term = -(alpha/m2) * u * x2
      d/dx (u*x2) = x2 * ∂u/∂x + u * ∂x2/∂x  with ∂x2/∂x = e3
    so:
      extra_row4 = -(alpha/m2) * ( x2 * ∂u/∂x + u * e3 )
    """
    u = control_u(x, k)
    du_dx = k_full_from_restricted(k)  # [k0,k1,0,0]

    J = A_star + jnp.outer(B, du_dx)

    # Only row 4 is affected by the bilinear stiffness term
    add_row4 = -(alpha / m2) * (x[2] * du_dx + u * e3)
    J = J.at[3, :].add(add_row4)
    return J


# =============================================================================
# 5) Classical adjoint (integrate backward in time)
# =============================================================================
def adjoint_from_traj(k, X, N, dt, A_star, B, e3, alpha, m2, w_x1, w_x1d, w_e, w_ed, r_u):
    """
    Classical adjoint:
        lambda_dot = - (Jf(x))^T lambda - (∂L/∂x)^T,   lambda(T)=0

    We integrate it by reversing time:
      Let τ = T - t  =>  dλ/dτ = (Jf(x))^T λ + ∂L/∂x
    Then we can integrate forward in τ using the reversed arrays.
    """
    gradL = jax.vmap(lambda x_i: grad_running_cost_x(x_i, k, w_x1, w_x1d, w_e, w_ed, r_u))(X)  # (N+1,4)
    JfT = jax.vmap(lambda x_i: jac_f_x(x_i, k, A_star, B, e3, alpha, m2).T)(X)                  # (N+1,4,4)

    # Reverse arrays (node 0 in reversed world corresponds to final time T)
    gradL_rev = gradL[::-1]
    JfT_rev = JfT[::-1]

    def rhs_rev(lam, Jt, g):
        """dλ/dτ = Jt @ λ + g"""
        return Jt @ lam + g

    def rk4_step_adj_rev(lam, Jt_curr, g_curr, Jt_mid, g_mid, Jt_next, g_next):
        k1 = rhs_rev(lam,                 Jt_curr, g_curr)
        k2 = rhs_rev(lam + 0.5 * dt * k1, Jt_mid,  g_mid)
        k3 = rhs_rev(lam + 0.5 * dt * k2, Jt_mid,  g_mid)
        k4 = rhs_rev(lam + dt * k3,       Jt_next, g_next)
        return lam + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    lam0 = jnp.zeros((4,))  # corresponds to lambda(T)=0

    def step(lam, j):
        # trapezoid-style midpoint approximations for RK4 stages
        g_curr = gradL_rev[j]
        g_next = gradL_rev[j + 1]
        g_mid = 0.5 * (g_curr + g_next)

        Jt_curr = JfT_rev[j]
        Jt_next = JfT_rev[j + 1]
        Jt_mid = 0.5 * (Jt_curr + Jt_next)

        lam_next = rk4_step_adj_rev(lam, Jt_curr, g_curr, Jt_mid, g_mid, Jt_next, g_next)
        return lam_next, lam_next

    _, lam_next = jax.lax.scan(step, lam0, jnp.arange(N))
    lam_rev = jnp.vstack([lam0[None, :], lam_next])  # (N+1,4) in reversed time
    lam = lam_rev[::-1]  # back to forward time indexing
    return lam


# =============================================================================
# 6) Gradient wrt k (restricted gain) using the adjoint variables
# =============================================================================
def grad_wrt_k(k, X, u, lam, dt, B, alpha, r_u):
    """
    dJ/dk = ∫ [x1,x1d]^T * ( 2*r_u*u + (1 - alpha*x2)*(B^T*lambda) ) dt
    """
    BTlam = lam @ B                 # (N+1,)  equals lambda_4 / m2
    factor = 1.0 - alpha * X[:, 2]  # (N+1,)
    s = 2.0 * r_u * u + factor * BTlam

    g_nodes = X[:, :2] * s[:, None]  # (N+1,2)
    gk = jnp.sum(0.5 * dt * (g_nodes[:-1] + g_nodes[1:]), axis=0)
    return gk


# =============================================================================
# 7) Main optimization routine (ADAM + manual adjoint gradient)
# =============================================================================
def simulate_2dof_with_optax_adam_classical_adjoint_x1x1d_control_bilinear_k2(
    m1, m2,
    k1, k2_star,
    c1, c2,
    kc, cd,
    alpha,
    F_ext,           # python callable, used only for precompute on the fixed grid
    t_end,
    y0,
    N,
    w_x1, w_x1d, w_e, w_ed,
    r_u,
    max_iter,
    k0=None,         # initial k = [k0, k1]
    lr=1e-2,
    betas=(0.9, 0.999),
    eps=1e-8,
    print_every=25,
):
    # -------------------------------------------------------------------------
    # Time grid + forcing sampled on nodes and midpoints (for RK4)
    # -------------------------------------------------------------------------
    t = np.linspace(0.0, t_end, N + 1)
    dt = float(t[1] - t[0])
    t_half = t[:-1] + 0.5 * dt

    F_nodes = jnp.array([F_ext(ti) for ti in t])
    F_half = jnp.array([F_ext(ti) for ti in t_half])

    y0 = jnp.array(y0)

    # -------------------------------------------------------------------------
    # Build constant matrices (depends on k2_star only)
    # -------------------------------------------------------------------------
    A_star, B, e3, e4 = build_system_matrices(m1, m2, k1, k2_star, c1, c2, kc, cd)

    # -------------------------------------------------------------------------
    # JIT-ed helpers (close over constants for speed)
    # -------------------------------------------------------------------------
    @jax.jit
    def cost_and_grad_manual(k):
        X = forward_trajectory(k, y0, N, dt, F_nodes, F_half, A_star, B, e4, alpha, m1, m2)
        J, u, _ = cost_from_traj(k, X, dt, w_x1, w_x1d, w_e, w_ed, r_u)
        lam = adjoint_from_traj(k, X, N, dt, A_star, B, e3, alpha, m2, w_x1, w_x1d, w_e, w_ed, r_u)
        gk = grad_wrt_k(k, X, u, lam, dt, B, alpha, r_u)
        return J, gk

    @jax.jit
    def cost_only(k):
        X = forward_trajectory(k, y0, N, dt, F_nodes, F_half, A_star, B, e4, alpha, m1, m2)
        J, _, _ = cost_from_traj(k, X, dt, w_x1, w_x1d, w_e, w_ed, r_u)
        return J

    @jax.jit
    def simulate_traj_outputs(k):
        X = forward_trajectory(k, y0, N, dt, F_nodes, F_half, A_star, B, e4, alpha, m1, m2)
        J, u, _ = cost_from_traj(k, X, dt, w_x1, w_x1d, w_e, w_ed, r_u)
        return X, u, J

    # -------------------------------------------------------------------------
    # Optimizer setup
    # -------------------------------------------------------------------------
    if k0 is None:
        k0 = np.zeros(2)
    k = jnp.array(k0)

    optimizer = optax.adam(learning_rate=lr, b1=betas[0], b2=betas[1], eps=eps)
    opt_state = optimizer.init(k)

    # History tracking (Python lists -> numpy arrays at the end)
    J_hist = []
    gnorm_hist = []
    k_hist = []
    K_hist = []

    best_J = np.inf
    best_k = None
    best_it = -1

    # -------------------------------------------------------------------------
    # Optimization loop (manual gradients from classical adjoint)
    # -------------------------------------------------------------------------
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
                f"[Classical Adjoint + ADAM | bilinear k2=k2*+alpha*u | u=k0*x1+k1*x1d] "
                f"it={it:4d}  J={Jf:.6e}  ||g||={gnorm:.3e}  k={np.array(k)}"
            )

        # ADAM update step
        updates, opt_state = optimizer.update(gk, opt_state, params=k)
        k = optax.apply_updates(k, updates)
        K_hist.append(np.array(k))

    # Pick best iterate (or last if somehow not set)
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
    # Passive vs optimal trajectories
    # -------------------------------------------------------------------------
    X_passive, u_passive, _ = simulate_traj_outputs(jnp.zeros(2))
    X_opt, u_opt, _ = simulate_traj_outputs(jnp.array(k_opt))

    info = {
        "success": True,
        "message": "Classical adjoint + Optax ADAM completed (bilinear k2=k2*+alpha*u, u depends only on x1,x1d)",
        "nit": it_final,
        "J": J_final,
        "k_opt": k_opt,
        "J_hist": np.array(J_hist),
        "gnorm_hist": np.array(gnorm_hist),
        "k_hist": np.array(k_hist),
        "best_iter": best_it,
        "K_full_opt": np.array([k_opt[0], k_opt[1], 0.0, 0.0]),
        "cost_only": cost_only,
    }

    return (
        t,
        np.array(X_passive),
        np.array(X_opt),
        np.array(u_opt),
        k_opt,
        info,
        np.array(K_hist),
    )


# =============================================================================
# 8) Visualization helpers
# =============================================================================
def plot_cost_landscape_2d(cost_only_fn, k_opt, k_hist, window=(2.0, 2.0), n_grid=70, use_log=True):
    """
    Plot J(k0,k1) in a window around the optimum, plus the optimizer path.
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


# =============================================================================
# 9) Script entry point
# =============================================================================
if __name__ == "__main__":
    start = time.time()

    # -------------------------------------------------------------------------
    # Load parameters and forcing
    # -------------------------------------------------------------------------
    param_file = "params.txt"
    p = load_params(param_file)

    m1, m2 = p["m1"], p["m2"]
    k1, k2_star = p["k1"], p["k2"]     # here p["k2"] is the constant part k2_star
    c1, c2 = p["c1"], p["c2"]
    cd, kc = p["cd"], p["kc"]

    F_ext = make_forcing(p)

    # Bilinear stiffness gain (k2(t) = k2_star + alpha*u)
    alpha = 6

    # -------------------------------------------------------------------------
    # Optimization settings
    # -------------------------------------------------------------------------
    t_end = 10.0
    y0 = (0.0, 0.0, 0.0, 0.0)
    N = 400

    w_x1, w_x1d = 1.0, 0.1
    w_e, w_ed = 50.0, 2.0
    r_u = 0.05

    max_iter = 6_000
    k_init = np.zeros(2)

    # -------------------------------------------------------------------------
    # Run adjoint-based optimization
    # -------------------------------------------------------------------------
    t, X_passive, X_opt, u_nodes, k_opt, info, K_hist = (
        simulate_2dof_with_optax_adam_classical_adjoint_x1x1d_control_bilinear_k2(
            m1, m2,
            k1, k2_star,
            c1, c2,
            kc, cd,
            alpha,
            F_ext,
            t_end,
            y0,
            N,
            w_x1, w_x1d, w_e, w_ed,
            r_u,
            max_iter,
            k0=k_init,
            lr=2e-2,
            print_every=50,
        )
    )

    # -------------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------------
    print("\nINFO:")
    for kk, vv in info.items():
        if kk in ("J_hist", "gnorm_hist", "k_hist", "cost_only"):
            continue
        print(f"  {kk}: {vv}")
    print("k_opt =", k_opt)
    print("K_full_opt =", info["K_full_opt"])

    stop = time.time()
    print("Total Time:", stop - start)

    # -------------------------------------------------------------------------
    # Plot: state trajectories (passive vs closed-loop)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Plot: control u(t)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 4))
    plt.plot(t, u_nodes, label="u(t)=k0*x1+k1*x1d")
    plt.xlabel("t [s]")
    plt.ylabel("Control force [N]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Plot: cost history
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 4))
    plt.plot(info["J_hist"])
    plt.xlabel("ADAM iteration")
    plt.ylabel("J")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Plot: gain history (after each update)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 4))
    plt.plot(K_hist[:, 0], label="k0")
    plt.plot(K_hist[:, 1], label="k1")
    plt.xlabel("ADAM iteration")
    plt.ylabel("k")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Plot: effective k2(t) = k2_star + alpha*u(t)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Plot: 2D cost landscape J(k0,k1) around optimum
    # -------------------------------------------------------------------------
    plot_cost_landscape_2d(
        cost_only_fn=info["cost_only"],
        k_opt=k_opt,
        k_hist=info["k_hist"],
        window=(3, 3),
        n_grid=100,
        use_log=True,
    )
