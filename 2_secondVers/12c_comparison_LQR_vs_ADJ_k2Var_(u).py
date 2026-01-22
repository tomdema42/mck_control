# -*- coding: utf-8 -*-
"""
2-DOF mass-spring-damper simulation with a GIVEN state-feedback gain K,
but with an input-dependent stiffness:

    k2(t) = k2_star + alpha * u(t)

State:   y = [x1, x1d, x2, x2d]
Control: u(t) = -K y

This script compares two gains:
  - K_adj (from adjoint optimization)
  - K_lqr (from LQR)

@author: demaria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from _auxFunc import load_params, make_forcing, CostFunc_comparison


def simulate_2dof_given_K_affine_k2(
    K,
    m1, m2,
    k1, k2_star, alpha,
    c1, c2,
    kc, cd,
    F_ext,
    t_end,
    y0,
    n_eval=4000,
    method="RK45",
    max_step=1e-2,
    rtol=1e-7,
    atol=1e-9,
):
    """
    Simulate closed-loop dynamics with a given K, with:
        k2(t) = k2_star + alpha * u(t)

    Returns:
      t_eval, Y (4,n), u (n,), k2_eff (n,), sol
    """
    K = np.asarray(K).reshape(1, 4)

    def rhs(t, y):
        x1, x1d, x2, x2d = y

        # control
        u = -float(K @ np.array([x1, x1d, x2, x2d]))

        # time-varying stiffness
        k2_eff = k2_star + alpha * u
        
        x1dd = (-k1 * x1 - c1 * x1d + cd * (x2d - x1d) + kc * (x2 - x1) + F_ext(t)) / m1
        x2dd = (-(k2_eff) * x2 - c2 * x2d - cd * (x2d - x1d) - kc * (x2 - x1) + u) / m2

        return [x1d, x1dd, x2d, x2dd]

    t_eval = np.linspace(0.0, t_end, n_eval)

    sol = solve_ivp(
        rhs,
        (0.0, t_end),
        y0,
        t_eval=t_eval,
        method=method,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
    )

    Y = sol.y  # (4, n_eval)

    # recompute u(t) and k2(t) on the same grid
    u = -(K @ Y).ravel()
    k2_eff = k2_star + alpha * u

    return t_eval, Y, u, k2_eff, sol


def compute_metrics(t, Y, u, k2_eff=None):
    x1, x1d, x2, x2d = Y

    dt = t[1] - t[0] if len(t) > 1 else 0.0

    I_x = np.trapezoid(x1**2 + x2**2, t)
    I_xd = np.trapezoid(x1d**2 + x2d**2, t)
    I_u = np.trapezoid(u**2, t)

    peak_x1 = np.max(np.abs(x1))
    peak_x2 = np.max(np.abs(x2))
    peak_u = np.max(np.abs(u))

    xf = np.array([x1[-1], x1d[-1], x2[-1], x2d[-1]])
    final_norm = float(np.linalg.norm(xf))

    out = {
        "I_x (x1^2+x2^2)": float(I_x),
        "I_xd (x1d^2+x2d^2)": float(I_xd),
        "I_u (u^2)": float(I_u),
        "peak|x1|": float(peak_x1),
        "peak|x2|": float(peak_x2),
        "peak|u|": float(peak_u),
        "||x(t_end)||": float(final_norm),
        "dt": float(dt),
    }

    if k2_eff is not None:
        out["min k2(t)"] = float(np.min(k2_eff))
        out["max k2(t)"] = float(np.max(k2_eff))

    return out


def print_metrics_table(metrics_by_label):
    keys = [
        "I_x (x1^2+x2^2)",
        "I_xd (x1d^2+x2d^2)",
        "I_u (u^2)",
        "peak|x1|",
        "peak|x2|",
        "peak|u|",
        "||x(t_end)||",
        "min k2(t)",
        "max k2(t)",
    ]

    labels = list(metrics_by_label.keys())

    # keep only keys that exist in all metrics
    keys = [k for k in keys if all(k in metrics_by_label[lbl] for lbl in labels)]

    col_w = 18
    lab_w = max(10, max(len(lbl) for lbl in labels) + 2)

    header = ("Metric".ljust(col_w) + " | " +
              " | ".join(lbl.ljust(lab_w) for lbl in labels))
    sep = "-" * len(header)

    print("\n" + header)
    print(sep)
    for k in keys:
        row = k.ljust(col_w) + " | " + " | ".join(
            f"{metrics_by_label[lbl][k]:.6g}".ljust(lab_w) for lbl in labels
        )
        print(row)
    print()


def plot_comparison(t, results):
    # results[label] = dict(Y=..., u=..., k2=...)
    plt.figure(figsize=(12, 6))
    for label, r in results.items():
        plt.plot(t, r["Y"][0], label=f"x1 ({label})")
    plt.xlabel("t [s]")
    plt.ylabel("x1 [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xlim(0, 6)

    plt.figure(figsize=(12, 6))
    for label, r in results.items():
        plt.plot(t, r["Y"][2], label=f"x2 ({label})")
    plt.xlabel("t [s]")
    plt.ylabel("x2 [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 4))
    for label, r in results.items():
        plt.plot(t, r["u"], label=f"u ({label})")
    plt.xlabel("t [s]")
    plt.ylabel("u [N]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xlim(0, 6)

    # k2(t) plot
    plt.figure(figsize=(8, 4))
    for label, r in results.items():
        plt.plot(t, r["k2"], label=f"k2(t) ({label})")
    plt.xlabel("t [s]")
    plt.ylabel("k2(t) [N/m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


if __name__ == "__main__":
    chSize = 18
    plt.rcParams.update({"font.size": chSize})

    # Penalization factors (for your CostFunc_comparison)
    w_x1, w_x1d = 1.0, 0.1
    w_e, w_ed = 50.0, 2.0
    r_u = 0.05

    # --- load params ---
    param_file = "params.txt"
    p = load_params(param_file)

    m1, m2 = p["m1"], p["m2"]
    k1 = p["k1"]
    k2_star = p["k2"]   # this is k2*
    c1, c2 = p["c1"], p["c2"]
    cd, kc = p["cd"], p["kc"]

    # choose alpha here
    alpha = 6.0  # <-- set your value (units: (N/m)/N = 1/m)

    F_ext = make_forcing(p)
    t_end = p["t_end"]
    y0 = [0.0, 0.0, 0.0, 0.0]

    # --- your gains ---
   
    K_lqr = [29.621664, 7.29122777, 16.82680442, 5.68928951]   # LQR solution
    # K_adj =  [13.65920891,  3.02540851   ,0,0] #adj with only x1,x1d
    K_adj = [11.42021168 ,10.58120851  , 0.99881109  ,8.26833314] #Adj with the full state x
    cases = {
        "No control": [0,0,0,0],
        "K_adj": K_adj,
        "K_lqr": K_lqr,
    }

    results = {}
    metrics = {}

    for label, K in cases.items():
        t, Y, u, k2_eff, sol = simulate_2dof_given_K_affine_k2(
            K=K,
            m1=m1, m2=m2,
            k1=k1, k2_star=k2_star, alpha=alpha,
            c1=c1, c2=c2,
            kc=kc, cd=cd,
            F_ext=F_ext,
            t_end=t_end,
            y0=y0,
            n_eval=4000,
            max_step=1e-2,
        )

        results[label] = {"Y": Y, "u": u, "k2": k2_eff, "sol": sol}
        metrics[label] = compute_metrics(t, Y, u, k2_eff=k2_eff)

        # your cost function (same call signature as before)
        ciao = CostFunc_comparison(Y, u, w_x1, w_x1d, w_e, w_ed, r_u)
        print("Cost function:", label, np.trapezoid(ciao, t))

    print_metrics_table(metrics)
    plot_comparison(t, results)
    plt.show()
