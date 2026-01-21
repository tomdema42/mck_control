# -*- coding: utf-8 -*-
"""
Classical (continuous-time) adjoint optimization of a constant state-feedback gain K (NO saturation)

Control law:
    u(t) = K @ x(t)
where x = [x1, x1d, x2, x2d]  and K is a 4-vector.

State dynamics (open-loop + control input):
    xdot = A x + B u + b(t)
with b(t) built from an external forcing applied on DOF1.

Closed-loop Jacobian wrt state:
    ∂f/∂x = A + B K   (B is a 4-vector, so B K means outer(B, K) giving a 4x4 matrix)

Running cost:
    L(x,u) = w_x1 x1^2 + w_x1d x1d^2 + w_e (x1+x2)^2 + w_ed (x1d+x2d)^2 + r_u u^2

Classical continuous-time adjoint:
    lambda_dot = - (∂f/∂x)^T lambda - (∂L/∂x)^T
    lambda(T) = 0

Gradient wrt K:
    dJ/dK = ∫ [ ∂L/∂K + lambda^T ∂f/∂K ] dt
         = ∫ x(t) * ( 2*r_u*u(t) + B^T lambda(t) ) dt

Notes:
- This is the classical continuous adjoint, not the exact discrete adjoint of RK4.
- Forward trajectory uses RK4 on a fixed grid.
- Adjoint is integrated backward using a reversed-time RK4 on the same grid.
- Optax ADAM performs parameter updates; gradients are computed manually (no jax.grad).

@author: demaria
"""

import time
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

from _auxFunc import load_params, make_forcing
chSize = 18
plt.rcParams.update({ 'font.size': chSize,  })        # Base font size

jax.config.update("jax_enable_x64", True)


# -----------------------------
# Model building blocks
# -----------------------------
def build_state_matrices(m1, m2, k1, k2, c1, c2, kc, cd):
    """
    Build open-loop state-space matrices (A, B) for the 2-DOF system:
        x = [x1, x1d, x2, x2d]

    B is a 4-vector because u is a scalar input acting on x2dd.
    """
    A = jnp.array([
        [0.0, 1.0, 0.0, 0.0],
        [-(k1 + kc) / m1, -(c1 + cd) / m1,  kc / m1,       cd / m1],
        [0.0, 0.0, 0.0, 1.0],
        [ kc / m2,       cd / m2,      -(k2 + kc) / m2, -(c2 + cd) / m2],
    ])
    B = jnp.array([0.0, 0.0, 0.0, 1.0 / m2])  # scalar u injects into x2dd
    return A, B


def precompute_forcing(F_ext, t_end, N, m1):
    """
    Precompute forcing values at RK4 nodes and half-steps (host -> JAX arrays).

    We model forcing as a vector b(t) added to xdot:
        b(t) = [0, F(t)/m1, 0, 0]
    """
    t = np.linspace(0.0, t_end, N + 1)
    dt = float(t[1] - t[0])
    t_half = t[:-1] + 0.5 * dt

    # Host precompute (Python callables are not JIT-friendly)
    F_nodes_np = np.array([F_ext(ti) for ti in t])
    F_half_np = np.array([F_ext(ti) for ti in t_half])

    # Convert to device arrays
    F_nodes = jnp.array(F_nodes_np)
    F_half = jnp.array(F_half_np)

    # Forcing vector constructor (kept as a small function for clarity)
    def forcing_vec(F_scalar):
        return jnp.array([0.0, F_scalar / m1, 0.0, 0.0])

    return t, dt, F_nodes, F_half, forcing_vec


# -----------------------------
# Cost and its state-gradient
# -----------------------------
def running_cost(x, u, w_x1, w_x1d, w_e, w_ed, r_u):
    """
    Scalar running cost L(x,u).
    """
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


def grad_running_cost_x(x, K, w_x1, w_x1d, w_e, w_ed, r_u):
    """
    Compute ∂L/∂x as a 4-vector (same shape as x).

    Important:
    - The state-only terms give straightforward partial derivatives.
    - The control penalty term includes x-dependence via u = K @ x:
        ∂(r_u u^2)/∂x = 2*r_u*u * ∂u/∂x = 2*r_u*u * K
    """
    x1, x1d, x2, x2d = x
    e = x1 + x2
    ed = x1d + x2d

    u = jnp.dot(K, x)

    g_state = jnp.array([
        2.0 * (w_x1 * x1 + w_e * e),
        2.0 * (w_x1d * x1d + w_ed * ed),
        2.0 * (w_e * e),
        2.0 * (w_ed * ed),
    ])

    g_control = 2.0 * r_u * u * K
    return g_state + g_control


# -----------------------------
# RK4 integrators (state + adjoint)
# -----------------------------
def rk4_step_state(x, K, Fi, Fh, Fi1, dt, A, B, forcing_vec):
    """
    One RK4 step for the forward state dynamics.

    xdot = A x + B u + b(t),  u = K @ x
    Forcing is provided as scalar samples Fi (node i), Fh (half), Fi1 (node i+1).
    """

    def f_closed(x_local, F_scalar):
        u_local = jnp.dot(K, x_local)
        return A @ x_local + B * u_local + forcing_vec(F_scalar)

    k1_ = f_closed(x, Fi)
    k2_ = f_closed(x + 0.5 * dt * k1_, Fh)
    k3_ = f_closed(x + 0.5 * dt * k2_, Fh)
    k4_ = f_closed(x + dt * k3_, Fi1)

    return x + (dt / 6.0) * (k1_ + 2.0 * k2_ + 2.0 * k3_ + k4_)


def rk4_step_adjoint_rev(lam, g_curr, g_mid, g_next, dt, AclT):
    """
    One RK4 step for the *reversed-time* adjoint integration.

    Original adjoint:
        lam_dot = -Acl^T lam - gradL
        lam(T) = 0

    Reverse time with τ = T - t:
        dlam/dτ =  Acl^T lam + gradL

    Here we integrate in τ forward, which corresponds to going backward in t.
    """

    def rhs(lam_local, gradL_local):
        return AclT @ lam_local + gradL_local

    k1_ = rhs(lam, g_curr)
    k2_ = rhs(lam + 0.5 * dt * k1_, g_mid)
    k3_ = rhs(lam + 0.5 * dt * k2_, g_mid)
    k4_ = rhs(lam + dt * k3_, g_next)

    return lam + (dt / 6.0) * (k1_ + 2.0 * k2_ + 2.0 * k3_ + k4_)


# -----------------------------
# Main simulation + optimization
# -----------------------------
def simulate_2dof_with_optax_adam_classical_adjoint(
    m1, m2,
    k1, k2,
    c1, c2,
    kc, cd,
    F_ext,          # Python callable (used only for precompute)
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
    """
    # --- build constants ---
    A, B = build_state_matrices(m1, m2, k1, k2, c1, c2, kc, cd)
    t, dt, F_nodes, F_half, forcing_vec = precompute_forcing(F_ext, t_end, N, m1)
    y0 = jnp.array(y0)

    def A_cl(K):
        """
        Closed-loop state Jacobian wrt x:
            Acl = A + B K
        with B K implemented as outer(B, K) (4x1 times 1x4 -> 4x4).
        """
        return A + jnp.outer(B, K)

    def forward_trajectory(K):
        """
        Simulate forward trajectory X at nodes (N+1,4) using RK4 + lax.scan.
        """
        def step(x, i):
            Fi = F_nodes[i]
            Fh = F_half[i]
            Fi1 = F_nodes[i + 1]
            x_next = rk4_step_state(x, K, Fi, Fh, Fi1, dt, A, B, forcing_vec)
            return x_next, x_next

        _, X_next = jax.lax.scan(step, y0, jnp.arange(N))
        X = jnp.vstack([y0[None, :], X_next])
        return X

    def cost_from_traj(K, X):
        """
        Compute trapezoidal integral of L(x,u) on nodes.
          - X is (N+1,4)
          - u is (N+1,)
          - L is (N+1,)
        """
        u = X @ K
        L = jax.vmap(lambda x_i, u_i: running_cost(x_i, u_i, w_x1, w_x1d, w_e, w_ed, r_u))(X, u)
        J = jnp.sum(0.5 * dt * (L[:-1] + L[1:]))
        return J, u, L

    def adjoint_from_traj(K, X):
        """
        Compute lambda(t) at nodes using reversed-time RK4.

        Steps:
          1) compute gradL(t_n) = ∂L/∂x at nodes (N+1,4)
          2) reverse arrays so we integrate from T -> 0 (i.e., τ forward)
          3) RK4 on each reversed segment with midpoint gradL approximated by averaging endpoints
          4) reverse back to align with forward-time indexing
        """
        AclT = A_cl(K).T

        gradL = jax.vmap(lambda x_i: grad_running_cost_x(x_i, K, w_x1, w_x1d, w_e, w_ed, r_u))(X)
        gradL_rev = gradL[::-1]  # node N first

        lam_T = jnp.zeros((4,))  # lambda(T) = 0

        def step(lam, j):
            g_curr = gradL_rev[j]
            g_next = gradL_rev[j + 1]
            g_mid = 0.5 * (g_curr + g_next)
            lam_next = rk4_step_adjoint_rev(lam, g_curr, g_mid, g_next, dt, AclT)
            return lam_next, lam_next

        _, lam_next = jax.lax.scan(step, lam_T, jnp.arange(N))

        lam_rev = jnp.vstack([lam_T[None, :], lam_next])  # (N+1,4) but reversed time
        lam = lam_rev[::-1]  # back to forward-time indexing
        return lam

    def grad_wrt_K(K, X, u, lam):
        """
        Compute dJ/dK using the classical adjoint formula (trapezoidal in time).

        Integrand at nodes:
            gK(t_n) = x(t_n) * ( 2*r_u*u(t_n) + B^T*lambda(t_n) )

        Shapes:
          - X:   (N+1,4)
          - u:   (N+1,)
          - lam: (N+1,4)
          - B:   (4,)
          - gK:  (4,)
        """
        BTlam = lam @ B                 # (N+1,)  -> each node: B dot lambda
        s = 2.0 * r_u * u + BTlam       # (N+1,)
        g_nodes = X * s[:, None]        # (N+1,4)

        gK = jnp.sum(0.5 * dt * (g_nodes[:-1] + g_nodes[1:]), axis=0)
        return gK

    @jax.jit
    def cost_and_grad_manual(K):
        """
        One “evaluate” call for the optimizer:
          - simulate X forward
          - compute J
          - integrate adjoint lambda backward
          - assemble gradient wrt K
        """
        X = forward_trajectory(K)
        J, u, _ = cost_from_traj(K, X)
        lam = adjoint_from_traj(K, X)
        gK = grad_wrt_K(K, X, u, lam)
        return J, gK

    @jax.jit
    def simulate_traj_outputs(K):
        """
        Forward simulation convenience: returns (X, u, J).
        """
        X = forward_trajectory(K)
        J, u, _ = cost_from_traj(K, X)
        return X, u, J

    # -----------------------------
    # Optimizer setup
    # -----------------------------
    if K0 is None:
        K0 = np.zeros(4)
    K = jnp.array(K0)

    optimizer = optax.adam(learning_rate=lr, b1=betas[0], b2=betas[1], eps=eps)
    opt_state = optimizer.init(K)

    J_hist = []
    gnorm_hist = []
    best_J = np.inf
    best_K = None
    best_it = -1
    K_hist = []
    # -----------------------------
    # Optimization loop (Python loop, JAX-evaluated objective/gradient)
    # -----------------------------
    for it in range(1, max_iter + 1):
        J, gK = cost_and_grad_manual(K)

        Jf = float(J)
        gnorm = float(jnp.linalg.norm(gK))
        J_hist.append(Jf)
        gnorm_hist.append(gnorm)
        
        if Jf < best_J:
            best_J = Jf
            best_K = np.array(K)
            best_it = it

        if it == 1 or it % print_every == 0:
            print(
                f"[Classical Adjoint + ADAM] it={it:6d}  "
                f"J={Jf:.6e}  ||g||={gnorm:.3e}  K={np.array(K)}"
            )

        updates, opt_state = optimizer.update(gK, opt_state, params=K)
        K = optax.apply_updates(K, updates)
        K_hist.append(K)
    # Pick best solution seen
    if best_K is None:
        K_opt = np.array(K)
        J_final = float(J_hist[-1]) if J_hist else None
        it_final = len(J_hist)
        best_it = it_final
    else:
        K_opt = best_K
        J_final = best_J
        it_final = best_it

    # Passive (K=0) and optimal trajectories
    X_passive, u_passive, _ = simulate_traj_outputs(jnp.zeros(4))
    X_opt, u_opt, _ = simulate_traj_outputs(jnp.array(K_opt))

    info = {
        "success": True,
        "message": "Classical adjoint + Optax ADAM completed",
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
        info,
        K_hist
    )


# -----------------------------
# Script entry point (run + plots)
# -----------------------------
if __name__ == "__main__":
    start = time.time()

    # Load parameters and external forcing function
    param_file = "params.txt"
    p = load_params(param_file)

    m1, m2 = p["m1"], p["m2"]
    k1, k2 = p["k1"], p["k2"]
    c1, c2 = p["c1"], p["c2"]
    cd, kc = p["cd"], p["kc"]

    F_ext = make_forcing(p)

    # Simulation / optimization setup
    t_end = 10.0
    y0 = (0.0, 0.0, 0.0, 0.0)
    N = 400

    # Cost weights
    w_x1, w_x1d = 1.0, 0.1
    w_e, w_ed = 50.0, 2.0
    r_u = 0.05

    # Optimization hyperparameters
    max_iter = 100_000
    K0 = np.zeros(4)

    t, X0, X1, u_nodes, K_opt, info,K_hist = simulate_2dof_with_optax_adam_classical_adjoint(
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
    print("Total Time:", stop - start)

    # Unpack trajectories for plotting
    x1_0, x1d_0, x2_0, x2d_0 = X0.T
    x1_1, x1d_1, x2_1, x2d_1 = X1.T

    # Displacements
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

    # Control
    plt.figure(figsize=(12, 4))
    plt.plot(t, u_nodes, label="u(t)=Kx")
    plt.xlabel("t [s]")
    plt.ylabel("Control force [N]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optimization history
    plt.figure(figsize=(12, 4))
    plt.plot(info["J_hist"]/min(info["J_hist"]))
    plt.xlabel("ADAM iteration")
    plt.ylabel("J")
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    K_hist_np = np.array(K_hist)
    plt.figure(figsize=(12,4))
    plt.plot(K_hist_np[:,0],label = 'K1')
    plt.plot(K_hist_np[:,1],label = 'K2')
    plt.plot(K_hist_np[:,2],label = 'K3')
    plt.plot(K_hist_np[:,3],label = 'K4')
    plt.xlabel("ADAM iteration")
    plt.ylabel("K_i")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
