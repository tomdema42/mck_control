# -*- coding: utf-8 -*-
"""
Optimize the parameter alpha defining the nonlinear stiffness k2 in a 2-DOF mass-spring-damper system
to minimize the objective function J(alpha) = ∫ x1(t) dt + lambda_k2 * ∫ k2(t) dt,
where k2(t) = alpha * |x2d(t) - x1d(t)|.

@author: demaria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from _auxFunc import load_params, make_forcing


def simulate_from_file(param_file, alpha):

    p = load_params(param_file)

    m1, m2 = p["m1"], p["m2"]
    k1 = p["k1"]
    c1, c2 = p["c1"], p["c2"]
    cd, kc = p["cd"], p["kc"]

    F_ext = make_forcing(p)

    y0 = (p["x1_0"], p["x1d_0"], p["x2_0"], p["x2d_0"])
    t_eval = np.linspace(p["t0"], p["t_end"], int(p["n_points"]))

    def rhs(t, y):
        x1, x1d, x2, x2d = y
        
        k2 = alpha *np.abs (x1d)

        x1dd = (-k1*x1 - c1*x1d + cd*(x2d - x1d) + kc*(x2 - x1) + F_ext(t)) / m1
        x2dd = (-k2*x2 - c2*x2d - cd*(x2d - x1d) - kc*(x2 - x1)) / m2

        return [x1d, x1dd, x2d, x2dd]

    sol = solve_ivp(
        rhs,
        (p["t0"], p["t_end"]),
        y0,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-8,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    # reconstruct k2(t)
    x1d = sol.y[1]
    x2d = sol.y[3]
    k2_t = alpha * np.abs(x1d-x2d) 

    return sol.t, sol.y, k2_t


def cost_alpha(alpha, param_file, lambda_k2):

    if alpha < 0:
        return np.inf

    try:
        t, y, k2_t = simulate_from_file(param_file, alpha)
        x1 = y[0]

        J_x1 = np.trapezoid(x1**2, t)
        J_k2 = 0#np.trapezoid(k2_t, t)

        J = J_x1 + lambda_k2 * J_k2

        if not np.isfinite(J):
            return np.inf

        return J

    except Exception:
        return np.inf


if __name__ == "__main__":

    param_file = "params.txt"
    lambda_k2 = 1e-3   # <<< tuning parameter

    alpha_bounds = (0.0, 9000)

    res = minimize_scalar(
        cost_alpha,
        bounds=alpha_bounds,
        args=(param_file, lambda_k2),
        method="bounded",
        options={"xatol": 1e-3, "maxiter": 80},
    )

    if not res.success:
        raise RuntimeError(res.message)

    alpha_best = res.x
    print(f"Optimal alpha = {alpha_best:.6g}")
    finalCost =  cost_alpha(alpha_best, param_file, lambda_k2)
    print(f"Cost optimal alpha : {finalCost:.6g}")
    # Final simulation
    t, y, k2_t = simulate_from_file(param_file, alpha_best)
    x1, x1d, x2, x2d = y
    
    print('Integral = ', np.trapezoid(x1**2, t))
    
    # Displacements
    plt.figure()
    plt.plot(t, x1, label="x1")
    plt.plot(t, x2, label="x2")
    plt.xlabel("t [s]")
    plt.ylabel("Displacement [m]")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Velocities
    plt.figure()
    plt.plot(t, x1d, label="x1d")
    plt.plot(t, x2d, label="x2d")
    plt.xlabel("t [s]")
    plt.ylabel("Velocity [m/s]")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # k2(t)
    # plt.figure()
    # plt.plot(t, k2_t, label="k2(t)")
    # plt.xlabel("t [s]")
    # plt.ylabel("Nonlinear stiffness k2")
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()

    plt.show()
# %%

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, PillowWriter
# from pathlib import Path


# def make_gif_2dof(t, x1, x2, k2_t=None, out_path="dyn_system.gif", fps=30, n_frames=400):
#     t = np.asarray(t)
#     x1 = np.asarray(x1)
#     x2 = np.asarray(x2)
#     if k2_t is not None:
#         k2_t = np.asarray(k2_t)

#     # Equilibrium (visual) reference positions so the two masses are separated
#     x1_ref = 0.0
#     x2_ref = 1.0

#     # Sample indices to keep the gif size reasonable
#     n_frames = int(min(n_frames, len(t)))
#     idx = np.linspace(0, len(t) - 1, n_frames).astype(int)

#     x1p = x1_ref + x1[idx]
#     x2p = x2_ref + x2[idx]

#     xmin = min(x1p.min(), x2p.min()) - 0.6
#     xmax = max(x1p.max(), x2p.max()) + 0.6

#     fig, ax = plt.subplots(figsize=(10, 2.6))
#     ax.set_xlim(xmin, xmax)
#     ax.set_ylim(-0.6, 0.6)
#     ax.set_yticks([])
#     ax.set_xlabel("Position")
#     ax.grid(True, axis="x", alpha=0.3)

#     # Track (rail)
#     rail, = ax.plot([xmin, xmax], [0, 0], lw=2)

#     # Wall at the left
#     wall_x = xmin + 0.1
#     wall, = ax.plot([wall_x, wall_x], [-0.25, 0.25], lw=4)

#     # Mass rectangles (as thick line segments for simplicity)
#     m_w = 0.18
#     m_h = 0.18

#     m1_line, = ax.plot([], [], lw=8)  # will draw as a thick segment
#     m2_line, = ax.plot([], [], lw=8)

#     # Connection between masses (spring/damper simplified as a line)
#     link, = ax.plot([], [], lw=2)

#     # Ground connections (to wall / ground) simplified lines
#     g1, = ax.plot([], [], lw=1.5, alpha=0.8)
#     g2, = ax.plot([], [], lw=1.5, alpha=0.8)

#     time_text = ax.text(0.02, 0.90, "", transform=ax.transAxes)

#     # Optional equilibrium markers
#     ax.axvline(x1_ref, ls="--", lw=1, alpha=0.4)
#     ax.axvline(x2_ref, ls="--", lw=1, alpha=0.4)

#     def mass_segment(xc):
#         return [xc - m_w / 2, xc + m_w / 2], [0, 0]

#     def init():
#         m1_line.set_data([], [])
#         m2_line.set_data([], [])
#         link.set_data([], [])
#         g1.set_data([], [])
#         g2.set_data([], [])
#         time_text.set_text("")
#         return m1_line, m2_line, link, g1, g2, time_text

#     def update(frame_i):
#         j = idx[frame_i]
#         x1c = x1_ref + x1[j]
#         x2c = x2_ref + x2[j]

#         m1_line.set_data(*mass_segment(x1c))
#         m2_line.set_data(*mass_segment(x2c))

#         # Link between masses
#         link.set_data([x1c + m_w / 2, x2c - m_w / 2], [0, 0])

#         # Ground connections (visual)
#         g1.set_data([wall_x, x1c - m_w / 2], [0, 0])
#         g2.set_data([x2c + m_w / 2, x2c + m_w / 2 + 0.25], [0, 0])  # small "ground" stub

#         if k2_t is not None:
#             time_text.set_text(f"t = {t[j]:.3f} s    k2 = {k2_t[j]:.3g}")
#         else:
#             time_text.set_text(f"t = {t[j]:.3f} s")

#         return m1_line, m2_line, link, g1, g2, time_text

#     anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True)

#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     writer = PillowWriter(fps=fps)
#     anim.save(str(out_path), writer=writer)
#     plt.close(fig)

#     return str(out_path)


# # --- call it after your final simulation ---
# gif_file = make_gif_2dof(
#     t=t,
#     x1=x1,
#     x2=x2,
#     k2_t=k2_t,                 # optional, shown in the caption
#     out_path="./dyn_system.gif",
#     fps=30,
#     n_frames=400
# )

# print("Saved GIF:", gif_file)
