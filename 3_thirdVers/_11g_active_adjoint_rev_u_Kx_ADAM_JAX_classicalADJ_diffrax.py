# -*- coding: utf-8 -*-
"""
Classical (continuous-time) adjoint optimization of a constant state-feedback gain K (NO saturation),
NOW USING DIFFRAX FOR BOTH:
  - Forward state integration
  - Backward adjoint integration

Control law:
    u(t) = K @ x(t)
where x = [x1, x1d, x2, x2d] and K is a 4-vector.

State dynamics:
    xdot = A x + B u + b(t)
with b(t) built from an external forcing applied on DOF1:
    b(t) = [0, F(t)/m1, 0, 0]

Closed-loop Jacobian wrt state:
    ∂f/∂x = A + B K   (B is a 4-vector => BK = outer(B, K))

Running cost:
    L(x,u) = w_x1 x1^2 + w_x1d x1d^2 + w_e (x1+x2)^2 + w_ed (x1d+x2d)^2 + r_u u^2

Adjoint (continuous-time):
    lambda_dot = - (∂f/∂x)^T lambda - (∂L/∂x)^T
    lambda(T) = 0

Gradient wrt K:
    dJ/dK = ∫ x(t) * ( 2*r_u*u(t) + B^T lambda(t) ) dt

Notes:
- We keep a FIXED time grid (same spirit as your original code) using ConstantStepSize + SaveAt(ts=...).
- Optax ADAM updates K using manually assembled gradients (no jax.grad).

@author: demaria
"""

import time
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

import diffrax as dfx

from _auxFunc import load_params, make_forcing

jax.config.update("jax_enable_x64", True)

chSize = 18
plt.rcParams.update({"font.size": chSize})


# -----------------------------
# Model building blocks
# -----------------------------
def build_state_matrices(m1, m2, k1, k2, c1, c2, kc, cd):
    """
    Build open-loop state-space matrices (A, B) for the 2-DOF system:
        x = [x1, x1d, x2, x2d]

    B is a 4-vector because u is scalar input acting on x2dd.
    """
    A = jnp.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-(k1 + kc) / m1, -(c1 + cd) / m1, kc / m1, cd / m1],
            [0.0, 0.0, 0.0, 1.0],
            [kc / m2, cd / m2, -(k2 + kc) / m2, -(c2 + cd) / m2],
        ]
    )
    B = jnp.array([0.0, 0.0, 0.0, 1.0 / m2])
    return A, B


def precompute_forcing(F_ext, t_end, N, m1):
    """
    Precompute forcing on a fixed grid and build a Diffrax interpolation.

    We model forcing as:
        b(t) = [0, F(t)/m1, 0, 0]
    """
    t = np.linspace(0.0, t_end, N + 1)
    dt = float(t[1] - t[0])

    F_nodes_np = np.array([F_ext(ti) for ti in t])
    t_jnp = jnp.array(t)
    F_nodes = jnp.array(F_nodes_np)

    # Interpolate forcing values so the ODE RHS can evaluate F(t) at any step time
    F_interp = dfx.LinearInterpolation(ts=t_jnp, ys=F_nodes)

    def b_vec(F_scalar):
        return jnp.array([0.0, F_scalar / m1, 0.0, 0.0])

    return t, dt, t_jnp, F_nodes, F_interp, b_vec


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
    Compute ∂L/∂x as a 4-vector.
    Includes the contribution of u = Kx inside r_u u^2.
    """
    x1, x1d, x2, x2d = x
    e = x1 + x2
    ed = x1d + x2d

    u = jnp.dot(K, x)

    g_state = jnp.array(
        [
            2.0 * (w_x1 * x1 + w_e * e),
            2.0 * (w_x1d * x1d + w_ed * ed),
            2.0 * (w_e * e),
            2.0 * (w_ed * ed),
        ]
    )

    # ∂/∂x (r_u (Kx)^2) = 2 r_u u K
    g_control = 2.0 * r_u * u * K
    return g_state + g_control


# -----------------------------
# Main simulation + optimization
# -----------------------------
def simulate_2dof_with_optax_adam_classical_adjoint_diffrax(
    m1,
    m2,
    k1,
    k2,
    c1,
    c2,
    kc,
    cd,
    F_ext,  # Python callable used only for forcing precompute
    t_end,
    y0,
    N,
    w_x1,
    w_x1d,
    w_e,
    w_ed,
    r_u,
    max_iter,
    K0=None,
    lr=1e-2,
    betas=(0.9, 0.999),
    eps=1e-8,
    print_every=25,
    solver_name="dopri5",
):
    """
    Optimize K (4-vector) using continuous adjoint gradient and Optax ADAM,
    using Diffrax for ODE integrations (forward + adjoint).

    Returns:
      t: time array (N+1,)
      X_passive: trajectory with K=0 (N+1,4)
      X_opt: trajectory with best K found (N+1,4)
      u_opt: control signal (N+1,)
      K_opt: best gain vector (4,)
      info: dict with diagnostics
      K_hist: list of K values over iterations
    """
    A, B = build_state_matrices(m1, m2, k1, k2, c1, c2, kc, cd)
    t, dt, t_jnp, F_nodes, F_interp, b_vec = precompute_forcing(F_ext, t_end, N, m1)

    y0 = jnp.array(y0)

    # Choose Diffrax solver
    if solver_name.lower() == "tsit5":
        solver = dfx.Tsit5()
    else:
        solver = dfx.Dopri5()

    stepsize_controller = dfx.ConstantStepSize()
    # no_adjoint = dfx.NoAdjoint()

    def A_cl(K):
        return A + jnp.outer(B, K)

    # -------- Forward integration (Diffrax) --------
    def forward_trajectory(K):
        """
        Solve xdot = A x + B (Kx) + b(t), saving on the fixed grid t_jnp.
        """

        def vf(ti, x, args):
            K_local = args
            u = jnp.dot(K_local, x)
            F = F_interp.evaluate(ti)
            return A @ x + B * u + b_vec(F)

        term = dfx.ODETerm(vf)

        sol = dfx.diffeqsolve(
            term,
            solver,
            t0=t_jnp[0],
            t1=t_jnp[-1],
            dt0=dt,
            y0=y0,
            args=K,
            saveat=dfx.SaveAt(ts=t_jnp),
            stepsize_controller=stepsize_controller,
            max_steps=N + 32,
            # adjoint=no_adjoint,
        )
        return sol.ys  # (N+1,4)

    # -------- Cost evaluation on nodes --------
    def cost_from_traj(K, X):
        u = X @ K
        L = jax.vmap(lambda x_i, u_i: running_cost(x_i, u_i, w_x1, w_x1d, w_e, w_ed, r_u))(X, u)
        J = jnp.sum(0.5 * dt * (L[:-1] + L[1:]))
        return J, u, L

    # -------- Adjoint integration (Diffrax, backwards in time) --------
    def adjoint_from_traj(K, X):
        """
        Integrate:
            lam_dot = -Acl^T lam - gradL
            lam(T) = 0
        backward from t=T to t=0, saving on the same fixed grid.

        We build gradL(t) on nodes, then interpolate it for RHS evaluation.
        """
        gradL_nodes = jax.vmap(lambda x_i: grad_running_cost_x(x_i, K, w_x1, w_x1d, w_e, w_ed, r_u))(X)
        gradL_interp = dfx.LinearInterpolation(ts=t_jnp, ys=gradL_nodes)

        AclT = A_cl(K).T
        lam_T = jnp.zeros((4,))

        def lam_rhs(ti, lam, args):
            # args unused; kept to match signature
            g = gradL_interp.evaluate(ti)
            return -(AclT @ lam + g)

        term_lam = dfx.ODETerm(lam_rhs)

        # Save in decreasing time order to match backwards integration
        ts_rev = t_jnp[::-1]

        sol_lam = dfx.diffeqsolve(
            term_lam,
            solver,
            t0=t_jnp[-1],
            t1=t_jnp[0],
            dt0=-dt,
            y0=lam_T,
            args=None,
            saveat=dfx.SaveAt(ts=ts_rev),
            stepsize_controller=stepsize_controller,
            max_steps=N + 32,
            # adjoint=no_adjoint,
        )

        lam_rev = sol_lam.ys          # aligned with ts_rev (T -> 0)
        lam = lam_rev[::-1]           # flip back to (0 -> T)
        return lam

    # -------- Gradient assembly wrt K --------
    def grad_wrt_K(K, X, u, lam):
        BTlam = lam @ B               # (N+1,)
        s = 2.0 * r_u * u + BTlam     # (N+1,)
        g_nodes = X * s[:, None]      # (N+1,4)
        gK = jnp.sum(0.5 * dt * (g_nodes[:-1] + g_nodes[1:]), axis=0)
        return gK

    # -------- JIT wrappers --------
    @jax.jit
    def cost_and_grad_manual(K):
        X = forward_trajectory(K)
        J, u, _ = cost_from_traj(K, X)
        lam = adjoint_from_traj(K, X)
        gK = grad_wrt_K(K, X, u, lam)
        return J, gK

    @jax.jit
    def simulate_traj_outputs(K):
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
    # Optimization loop
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
                f"[Adjoint + ADAM + Diffrax] it={it:6d}  "
                f"J={Jf:.6e}  ||g||={gnorm:.3e}  K={np.array(K)}"
            )

        updates, opt_state = optimizer.update(gK, opt_state, params=K)
        K = optax.apply_updates(K, updates)
        K_hist.append(K)

    # Pick best
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
        "message": "Classical adjoint + Optax ADAM (Diffrax integration) completed",
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
        K_hist,
    )


# -----------------------------
# Script entry point (run + plots)
# -----------------------------
if __name__ == "__main__":
    start = time.time()

    # Load parameters and forcing function
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
    max_iter = 4_000
    K0 = np.zeros(4)

    t, X0, X1, u_nodes, K_opt, info, K_hist = simulate_2dof_with_optax_adam_classical_adjoint_diffrax(
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
        K0=K0,
        lr=2e-2,
        print_every=50,
        solver_name="tsit5"#"dopri5",  # "tsit5" also supported
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
    J_hist = info["J_hist"]
    plt.plot(J_hist / np.min(J_hist))
    plt.xlabel("ADAM iteration")
    plt.ylabel("J / min(J)")
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # K history
    K_hist_np = np.array([np.array(k) for k in K_hist])
    plt.figure(figsize=(12, 4))
    plt.plot(K_hist_np[:, 0], label="K1")
    plt.plot(K_hist_np[:, 1], label="K2")
    plt.plot(K_hist_np[:, 2], label="K3")
    plt.plot(K_hist_np[:, 3], label="K4")
    plt.xlabel("ADAM iteration")
    plt.ylabel("K_i")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
