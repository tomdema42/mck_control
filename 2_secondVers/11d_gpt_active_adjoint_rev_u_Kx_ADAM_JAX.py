 # -*- coding: utf-8 -*-
"""
JAX + Optax ADAM optimization of a constant state-feedback gain K (NO saturation):
    u(t) = K @ x(t)
where x = [x1, x1d, x2, x2d].

Minimize (trapezoidal rule on a fixed grid):
  J = ∫ [ w_x1 x1^2 + w_x1d x1d^2 + w_e (x1+x2)^2 + w_ed (x1d+x2d)^2 + r_u u^2 ] dt

Notes:
- Uses JAX autodiff (no manual adjoint).
- Forcing F_ext(t) is precomputed on nodes and half-steps using your existing make_forcing().
- RK4 + lax.scan for fast, differentiable simulation.

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


def simulate_2dof_with_optax_adam(
    m1, m2,
    k1, k2,
    c1, c2,
    kc, cd,
    F_ext,          # python callable, used only for precompute
    t_end,
    y0,
    N,
    w_x1, w_x1d, w_e, w_ed,
    r_u,
    max_iter,
    K0=None,
    lr=1e-2,
    betas=(0.9, 0.999),
    eps=1e-8,
    tol_grad=1e-6,
    print_every=25,
    seed=0,
):
    """
    Returns:
      t, X_passive, X_opt, u_opt_nodes, K_opt, info
    """
    # --- time grid (host/numpy for precompute + plotting) ---
    t = np.linspace(0.0, t_end, N + 1)
    dt = float(t[1] - t[0])
    t_half = t[:-1] + 0.5 * dt

    # --- forcing precompute (host/numpy) ---
    F_nodes_np = np.array([F_ext(ti) for ti in t])
    F_half_np = np.array([F_ext(ti) for ti in t_half])

    # Move to JAX
    F_nodes = jnp.array(F_nodes_np)
    F_half = jnp.array(F_half_np)

    # --- constant matrices (JAX arrays) ---
    A = jnp.array([
        [0.0, 1.0, 0.0, 0.0],
        [-(k1 + kc) / m1, -(c1 + cd) / m1,  kc / m1,       cd / m1],
        [0.0, 0.0, 0.0, 1.0],
        [ kc / m2,       cd / m2,      -(k2 + kc) / m2, -(c2 + cd) / m2],
    ])
    B = jnp.array([0.0, 0.0, 0.0, 1.0 / m2])  # input acts on x2dd

    y0 = jnp.array(y0)

    def forcing_vec(F_scalar):
        return jnp.array([0.0, F_scalar / m1, 0.0, 0.0])

    def running_cost(x, u):
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

    def f_closed(x, K, F_scalar):
        u = jnp.dot(K, x)
        return A @ x + B * u + forcing_vec(F_scalar)

    def rk4_step(x, K, Fi, Fh, Fi1):
        k1 = f_closed(x, K, Fi)
        k2 = f_closed(x + 0.5 * dt * k1, K, Fh)
        k3 = f_closed(x + 0.5 * dt * k2, K, Fh)
        k4 = f_closed(x + dt * k3, K, Fi1)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # -----------------------------
    # Cost-only simulation (fast for training)
    # -----------------------------
    def simulate_cost(K):
        def step(carry, i):
            x, J = carry #x and J at the i step
            Fi = F_nodes[i]
            Fh = F_half[i]
            Fi1 = F_nodes[i + 1]

            u_i = jnp.dot(K, x) 
            x_next = rk4_step(x, K, Fi, Fh, Fi1)
            u_ip1 = jnp.dot(K, x_next)

            J_step = 0.5 * dt * (running_cost(x, u_i) + running_cost(x_next, u_ip1))
            return (x_next, J + J_step), None

        (xN, Jtot), _ = jax.lax.scan(step, (y0, 0.0), jnp.arange(N))
        return Jtot

    value_and_grad = jax.jit(jax.value_and_grad(simulate_cost))

    # # -----------------------------
    # Trajectory simulation (for outputs/plots)
    # TO DO: Change with a normal stuff
    # -----------------------------
    @jax.jit
    def simulate_traj(K):
        def step(x, i):
            Fi = F_nodes[i]
            Fh = F_half[i]
            Fi1 = F_nodes[i + 1]

            u_i = jnp.dot(K, x)
            x_next = rk4_step(x, K, Fi, Fh, Fi1)
            u_ip1 = jnp.dot(K, x_next)

            J_step = 0.5 * dt * (running_cost(x, u_i) + running_cost(x_next, u_ip1))
            return x_next, (x_next, u_ip1, J_step)

        xN, (X_next, u_next, J_steps) = jax.lax.scan(step, y0, jnp.arange(N))
        X = jnp.vstack([y0[None, :], X_next])
        u0 = jnp.dot(K, y0)
        u = jnp.concatenate([jnp.array([u0]), u_next])
        J = jnp.sum(J_steps)
        return X, u, J

    # -----------------------------
    # Optax optimizer
    # -----------------------------
    K = jnp.array(K0)
  
    opt_adam = optax.adam(learning_rate=lr, b1=betas[0], b2=betas[1], eps=eps)

    optimizer = optax.chain(opt_adam)
    opt_state = optimizer.init(K)

    J_hist = []
    gnorm_hist = []
    best_J = np.inf
    best_K = None
    best_it = -1

   

    for it in range(1, max_iter + 1):
        J, gK = value_and_grad(K)
        Jf = float(J)
        gnorm = float(jnp.linalg.norm(gK))

        J_hist.append(Jf)
        gnorm_hist.append(gnorm)

        if Jf < best_J:
            best_J = Jf
            best_K = np.array(K)
            best_it = it

        if  (it % print_every == 0 or it == 1):
            print(f"[Optax ADAM] it={it:4d}  J={Jf:.6e}  ||g||={gnorm:.3e}  K={np.array(K)}")

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

    # Passive and optimal trajectories
    X_passive, u_passive, _ = simulate_traj(jnp.zeros(4))
    X_opt, u_opt, _ = simulate_traj(jnp.array(K_opt))

    info = {
        "success": True,
        "message": "Optax ADAM completed",
        "nit": it_final,
        "J": J_final,
        "K_opt": K_opt,
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
        info
    )


# -----------------------------
# Run + plots
# -----------------------------
if __name__ == "__main__":
    start = time.time()
    param_file = "params.txt"
    p = load_params(param_file)

    m1, m2 = p["m1"], p["m2"]
    k1, k2 = p["k1"], p["k2"]
    c1, c2 = p["c1"], p["c2"]
    cd, kc = p["cd"], p["kc"]

    F_ext = make_forcing(p)

    t_end = 10.0
    y0 = (0.0, 0.0, 0.0, 0.0)

    N = 400
    w_x1, w_x1d = 1.0, 0.1
    w_e, w_ed = 50.0, 2.0
    r_u = 0.05

    max_iter = 2_000
    K0 = np.zeros(4)

    t, X0, X1, u_nodes, K_opt, info = simulate_2dof_with_optax_adam(
        m1, m2,
        k1, k2,
        c1, c2,
        kc, cd,
        F_ext,
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

    print("\nINFO:")
    for k, v in info.items():
        if k in ("J_hist", "gnorm_hist"):
            continue
        print(f"  {k}: {v}")
    print("K_opt =", K_opt)
    stop = time.time()
    print('Total Time: ',stop-start)
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
    plt.plot(t, u_nodes, label="u(t)=Kx")
    plt.xlabel("t [s]")
    plt.ylabel("Control force [N]")
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
