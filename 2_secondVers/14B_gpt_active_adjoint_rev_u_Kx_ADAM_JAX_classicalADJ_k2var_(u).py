# -*- coding: utf-8 -*-
"""
Classical (continuous-time) adjoint optimization of a *full-state* constant state-feedback law (NO saturation),
WITH BILINEAR STIFFNESS:
    k2(t) = k2_star + alpha * u(t)

Control law (FULL state):
    u(t) = K @ x(t) = K1*x1 + K2*x1d + K3*x2 + K4*x2d
where x = [x1, x1d, x2, x2d] and K = [K1, K2, K3, K4].

State dynamics:
    xdot = A_star x + B u + b(t) + g_bilin(x,u)
where A_star uses k2_star (constant), and the bilinear term affects only x2dd:
    x2dd += -(alpha/m2) * u * x2

Running cost:
    L(x,u) = w_x1 x1^2 + w_x1d x1d^2 + w_e (x1+x2)^2 + w_ed (x1d+x2d)^2 + r_u u^2

Adjoint (classical, continuous-time, nonlinear dynamics):
    lambda_dot = - (∂f/∂x)^T lambda - (∂L/∂x)^T
    lambda(T) = 0

Gradient wrt K (4 params), since K enters only through u:
    dJ/dK = ∫ (∂u/∂K)^T [ ∂L/∂u + (∂f/∂u)^T lambda ] dt
with:
    ∂u/∂K = x
    ∂L/∂u = 2*r_u*u
    ∂f/∂u = (1 - alpha*x2) * B
so:
    dJ/dK = ∫ x * ( 2*r_u*u + (1 - alpha*x2)*(B^T*lambda) ) dt

Numerics:
- Forward state: RK4 on fixed grid.
- Adjoint: RK4 backward in time with time-varying (state-dependent) Jacobian ∂f/∂x.
- Optax ADAM is used, but gradients are manual (no jax.grad).

@author: demaria
"""

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax
import time
from _auxFunc import load_params, make_forcing

jax.config.update("jax_enable_x64", True)


def simulate_2dof_with_optax_adam_classical_adjoint_fullstate_control_bilinear_k2(
    m1, m2,
    k1, k2_star,     # k2_star is the constant part
    c1, c2,
    kc, cd,
    alpha,           # k2(t)=k2_star + alpha*u
    F_ext,           # python callable, used only for precompute
    t_end,
    y0,
    N,
    w_x1, w_x1d, w_e, w_ed,
    r_u,
    max_iter,
    K0=None,         # initial K = [K1,K2,K3,K4]
    lr=1e-2,
    betas=(0.9, 0.999),
    eps=1e-8,
    print_every=25,
):
    # -----------------------------
    # Time grid + forcing precompute
    # -----------------------------
    t = np.linspace(0.0, t_end, N + 1)
    dt = float(t[1] - t[0])
    t_half = t[:-1] + 0.5 * dt

    F_nodes_np = np.array([F_ext(ti) for ti in t])
    F_half_np = np.array([F_ext(ti) for ti in t_half])

    F_nodes = jnp.array(F_nodes_np)
    F_half = jnp.array(F_half_np)

    y0 = jnp.array(y0)

    # -----------------------------
    # Constant matrices (k2_star only)
    # -----------------------------
    A_star = jnp.array([
        [0.0, 1.0, 0.0, 0.0],
        [-(k1 + kc) / m1, -(c1 + cd) / m1,  kc / m1,       cd / m1],
        [0.0, 0.0, 0.0, 1.0],
        [ kc / m2,       cd / m2,      -(k2_star + kc) / m2, -(c2 + cd) / m2],
    ])
    # u acts on x2dd
    B = jnp.array([0.0, 0.0, 0.0, 1.0 / m2])

    # selectors
    e3 = jnp.array([0.0, 0.0, 1.0, 0.0])  # x2 component selector
    e4 = jnp.array([0.0, 0.0, 0.0, 1.0])  # x2dd component selector

    def forcing_vec(F_scalar):
        return jnp.array([0.0, F_scalar / m1, 0.0, 0.0])

    # -----------------------------
    # Control + cost
    # -----------------------------
    def control_u(x, K):
        # full-state linear feedback
        return jnp.dot(K, x)

    def running_cost(x, u):
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
        ∂L/∂x (shape (4,))

        Only the r_u*u^2 term depends on K through u(x)=K·x, so:
          ∂(r_u*u^2)/∂x = 2*r_u*u*K
        """
        x1_, x1d_, x2_, x2d_ = x
        e = x1_ + x2_
        ed = x1d_ + x2d_

        u = control_u(x, K)

        g = jnp.array([
            2.0 * (w_x1 * x1_ + w_e * e),
            2.0 * (w_x1d * x1d_ + w_ed * ed),
            2.0 * (w_e * e),
            2.0 * (w_ed * ed),
        ])

        g = g + 2.0 * r_u * u * K
        return g

    # -----------------------------
    # Dynamics + RK4 (bilinear term)
    # -----------------------------
    def f_closed(x, K, F_scalar):
        """
        xdot = A_star x + B u + b(t) + e4 * (-(alpha/m2) * u * x2)
        """
        u = control_u(x, K)
        bil = e4 * (-(alpha / m2) * u * x[2])
        return A_star @ x + B * u + forcing_vec(F_scalar) + bil

    def rk4_step_state(x, K, Fi, Fh, Fi1):
        k1_ = f_closed(x, K, Fi)
        k2_ = f_closed(x + 0.5 * dt * k1_, K, Fh)
        k3_ = f_closed(x + 0.5 * dt * k2_, K, Fh)
        k4_ = f_closed(x + dt * k3_, K, Fi1)
        return x + (dt / 6.0) * (k1_ + 2.0 * k2_ + 2.0 * k3_ + k4_)

    def forward_traj(K):
        # returns X_nodes (N+1,4)
        def step(x, i):
            Fi = F_nodes[i]
            Fh = F_half[i]
            Fi1 = F_nodes[i + 1]
            x_next = rk4_step_state(x, K, Fi, Fh, Fi1)
            return x_next, x_next

        _, X_next = jax.lax.scan(step, y0, jnp.arange(N))
        X = jnp.vstack([y0[None, :], X_next])
        return X

    def cost_from_traj(K, X):
        u = X @ K  # (N+1,)
        L = jax.vmap(running_cost)(X, u)
        J = jnp.sum(0.5 * dt * (L[:-1] + L[1:]))
        return J, u, L

    # -----------------------------
    # Jacobian ∂f/∂x for adjoint (state-dependent)
    # -----------------------------
    def jac_f_x(x, K):
        """
        f = A_star x + B u + b + e4 * (-(alpha/m2) u x2)
        u = K·x

        ∂f/∂x = A_star + outer(B, du/dx) + extra_row4
        du/dx = K
        extra_row4 (row 4 add-on):
          -(alpha/m2) * ( x2 * K + u * e3 )
        """
        u = control_u(x, K)
        du_dx = K  # (4,)

        J = A_star + jnp.outer(B, du_dx)

        add_row4 = -(alpha / m2) * (x[2] * du_dx + u * e3)
        J = J.at[3, :].add(add_row4)
        return J

    # -----------------------------
    # Classical adjoint (backward, variable Jacobian)
    # -----------------------------
    def adjoint_from_traj(K, X):
        """
        Integrate classical adjoint backward:
          lambda_dot = -Jf(x)^T lambda - (∂L/∂x)^T,  lambda(T)=0

        Reverse time τ = T - t:
          dλ/dτ = Jf(x)^T λ + ∂L/∂x
        """
        gradL = jax.vmap(lambda x: grad_running_cost_x(x, K))(X)  # (N+1,4)
        JfT = jax.vmap(lambda x: jac_f_x(x, K).T)(X)              # (N+1,4,4)

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

        lam0 = jnp.zeros((4,))  # lambda(T)=0 in reverse-time initial condition

        def step(lam, j):
            g_curr = gradL_rev[j]
            g_next = gradL_rev[j + 1]
            g_mid = 0.5 * (g_curr + g_next)

            Jt_curr = JfT_rev[j]
            Jt_next = JfT_rev[j + 1]
            Jt_mid = 0.5 * (Jt_curr + Jt_next)

            lam_next = rk4_step_adj_rev(lam, Jt_curr, g_curr, Jt_mid, g_mid, Jt_next, g_next)
            return lam_next, lam_next

        _, lam_next = jax.lax.scan(step, lam0, jnp.arange(N))
        lam_rev = jnp.vstack([lam0[None, :], lam_next])
        lam = lam_rev[::-1]
        return lam

    # -----------------------------
    # Gradient wrt K (4 params)
    # -----------------------------
    def grad_wrt_K(K, X, u, lam):
        """
        dJ/dK = ∫ x * s dt
        s = 2*r_u*u + (1 - alpha*x2) * (B^T * lambda)
        """
        BTlam = lam @ B                 # (N+1,)
        factor = 1.0 - alpha * X[:, 2]  # (N+1,)
        s = 2.0 * r_u * u + factor * BTlam
        g_nodes = X * s[:, None]        # (N+1,4)
        gK = jnp.sum(0.5 * dt * (g_nodes[:-1] + g_nodes[1:]), axis=0)
        return gK

    # -----------------------------
    # JIT-ed helpers
    # -----------------------------
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

    # -----------------------------
    # Optax optimizer (manual gradients)
    # -----------------------------
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
                f"it={it:4d}  J={Jf:.6e}  ||g||={gnorm:.3e}  K={np.array(K)}"
            )

        updates, opt_state = optimizer.update(gK, opt_state, params=K)
        K = optax.apply_updates(K, updates)

    if best_K is None:
        K_opt = np.array(K)
        J_final = float(J_hist[-1]) if J_hist else None
        it_final = len(J_hist)
        best_it = it_final
    else:
        K_opt = best_K
        J_final = best_J
        it_final = best_it

    # Passive and optimal trajectories (passive = u=0 => K=[0,0,0,0])
    X_passive, u_passive, _ = simulate_traj_outputs(jnp.zeros(4))
    X_opt, u_opt, _ = simulate_traj_outputs(jnp.array(K_opt))

    info = {
        "success": True,
        "message": "Classical adjoint + Optax ADAM completed (bilinear k2=k2*+alpha*u, u=K@x)",
        "nit": it_final,
        "J": J_final,
        "K_opt": K_opt,
        "J_hist": np.array(J_hist),
        "gnorm_hist": np.array(gnorm_hist),
        "K_hist": np.array(K_hist),
        "best_iter": best_it,
        "cost_only": cost_only,
    }

    return (
        t,
        np.array(X_passive),
        np.array(X_opt),
        np.array(u_opt),
        K_opt,
        info,
        np.array(K_hist),
    )


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
    2D slice of J over (K[idx[0]], K[idx[1]]) while keeping other components fixed to K_opt.
    """
    i0, i1 = idx
    d0, d1 = window

    v0 = np.linspace(K_opt[i0] - d0, K_opt[i0] + d0, n_grid)
    v1 = np.linspace(K_opt[i1] - d1, K_opt[i1] + d1, n_grid)

    V0, V1 = np.meshgrid(v0, v1, indexing="xy")

    K_base = np.array(K_opt, copy=True)
    K_flat = []
    for a, b in zip(V0.ravel(), V1.ravel()):
        K_tmp = K_base.copy()
        K_tmp[i0] = a
        K_tmp[i1] = b
        K_flat.append(K_tmp)
    K_flat = jnp.array(np.array(K_flat))

    @jax.jit
    def batch_cost(K_batch):
        return jax.vmap(cost_only_fn)(K_batch)

    J_flat = np.array(batch_cost(K_flat))
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
    plt.title("Cost landscape (2D slice, other K fixed)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# Run + plots
# -----------------------------
if __name__ == "__main__":
    start = time.time()

    param_file = "params.txt"
    p = load_params(param_file)

    m1, m2 = p["m1"], p["m2"]
    k1, k2_star = p["k1"], p["k2"]
    c1, c2 = p["c1"], p["c2"]
    cd, kc = p["cd"], p["kc"]

    F_ext = make_forcing(p)

    alpha = 6

    t_end = 10.0
    y0 = (0.0, 0.0, 0.0, 0.0)

    N = 400
    w_x1, w_x1d = 1.0, 0.1
    w_e, w_ed = 50.0, 2.0
    r_u = 0.05

    max_iter = 50_000
    K_init = np.zeros(4)

    t, X0, X1, u_nodes, K_opt, info, K_hist = simulate_2dof_with_optax_adam_classical_adjoint_fullstate_control_bilinear_k2(
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

    stop = time.time()
    print("Total Time:", stop - start)

    # -----------------------------
    # State plots
    # -----------------------------
    x1_0, x1d_0, x2_0, x2d_0 = X0.T
    x1_1, x1d_1, x2_1, x2d_1 = X1.T

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

    plt.figure(figsize=(12, 4))
    plt.plot(t, u_nodes, label="u(t)=K@x")
    plt.xlabel("t [s]")
    plt.ylabel("Control force [N]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # k2(t) = k2* + alpha*u(t)
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

    plt.figure(figsize=(12, 4))
    plt.plot(info["J_hist"])
    plt.xlabel("ADAM iteration")
    plt.ylabel("J")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # K history (4 components)
    plt.figure(figsize=(12, 4))
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

    # -----------------------------
    # 2D cost landscape slice (example: K1 vs K2, others fixed)
    # -----------------------------
    plot_cost_landscape_2d_fixed_others(
        cost_only_fn=info["cost_only"],
        K_opt=K_opt,
        K_hist=info["K_hist"],
        idx=(0, 1),
        window=(3, 3),
        n_grid=100,
        use_log=True,
    )
