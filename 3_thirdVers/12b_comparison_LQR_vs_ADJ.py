# -*- coding: utf-8 -*-
"""
2-DOF mass-spring-damper simulation with a GIVEN state-feedback gain K.

State: y = [x1, x1d, x2, x2d]
Control: u(t) = sat( -K y )

This script compares two gains:
  - K_adj (from adjoint optimization)
  - K_lqr (from LQR)

@author: demaria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from _auxFunc import load_params, make_forcing, CostFunc_comparison
# %%


def simulate_2dof_given_K(
    K,
    m1, m2,
    k1, k2,
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
    Simulate closed-loop dynamics with a given K.

    K:
      - shape (4,) or (1,4)
      - uses u = -K @ y

    Returns:
      t_eval, Y (4,n), u (n,), sol (solve_ivp object)
    """
    K = np.asarray(K).reshape(1, 4)

    def rhs(t, y):
        x1, x1d, x2, x2d = y

        u = -float(K @ np.array([x1, x1d, x2, x2d]))
        # u = saturate(u, u_max, mode=sat_mode)

        x1dd = (-k1 * x1 - c1 * x1d + cd * (x2d - x1d) + kc * (x2 - x1) + F_ext(t)) / m1
        x2dd = (-k2 * x2 - c2 * x2d - cd * (x2d - x1d) - kc * (x2 - x1) + u) / m2

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

    # compute u(t) AFTER integration on the same grid
    u = -(K @ Y).ravel()
    return t_eval, Y, u, sol


def compute_metrics(t, Y, u):
    x1, x1d, x2, x2d = Y

    dt = t[1] - t[0] if len(t) > 1 else 0.0

    # energy-like integrals (no arbitrary weights)
    I_x = np.trapezoid(x1**2 + x2**2, t)
    I_xd = np.trapezoid(x1d**2 + x2d**2, t)
    I_u = np.trapezoid(u**2, t)

    # peaks
    peak_x1 = np.max(np.abs(x1))
    peak_x2 = np.max(np.abs(x2))
    peak_u = np.max(np.abs(u))

    # final state norm
    xf = np.array([x1[-1], x1d[-1], x2[-1], x2d[-1]])
    final_norm = float(np.linalg.norm(xf))

    return {
        "I_x (x1^2+x2^2)": float(I_x),
        "I_xd (x1d^2+x2d^2)": float(I_xd),
        "I_u (u^2)": float(I_u),
        "peak|x1|": float(peak_x1),
        "peak|x2|": float(peak_x2),
        "peak|u|": float(peak_u),
        "||x(t_end)||": float(final_norm),
        "dt": float(dt),
    }

def print_metrics_table(metrics_by_label):
    from tabulate import tabulate

    keys = [
        "I_x (x1^2+x2^2)",
        "I_xd (x1d^2+x2d^2)",
        "I_u (u^2)",
        "peak|x1|",
        "peak|x2|",
        "peak|u|",
        "||x(t_end)||",
    ]

    labels = list(metrics_by_label.keys())
    headers = ["Metric"] + labels

    rows = []
    for k in keys:
        rows.append([k] + [metrics_by_label[lbl][k] for lbl in labels])

    print()
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid", floatfmt=".6g"))
    print()

def plot_K_comparison(cases_dict):
    # cases_dict[label] = K (length 4)
    labels = list(cases_dict.keys())
    Ks = np.array([np.asarray(cases_dict[lbl]).reshape(4,) for lbl in labels])  # (n_cases, 4)

    x = np.arange(4)
    width = 0.8 / len(labels)

    plt.figure(figsize=(10, 4))
    for i, lbl in enumerate(labels):
        plt.bar(x + (i - (len(labels)-1)/2)*width, Ks[i], width=width, label=lbl)

    plt.xticks(x, ["K1 (x1)", "K2 (x1d)", "K3 (x2)", "K4 (x2d)"])
    plt.ylabel("Gain value")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()

def plot_comparison(t, results):
    # results[label] = dict(Y=..., u=...)
    # Displacements
    plt.figure(figsize=(12, 6))
    for label, r in results.items():
        x1 = r["Y"][0]
        plt.plot(t, x1, label=f"x1 ({label})")
    plt.xlabel("t [s]")
    plt.ylabel("x1 [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xlim(0,6)

    plt.figure(figsize=(12, 6))
    for label, r in results.items():
        x2 = r["Y"][2]
        plt.plot(t, x2, label=f"x2 ({label})")
    plt.xlabel("t [s]")
    plt.ylabel("x2 [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Control
    plt.figure(figsize=(8, 4))
    for label, r in results.items():
        plt.plot(t, r["u"], label=f"u ({label})")
    plt.xlabel("t [s]")
    plt.ylabel("u [N]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xlim(0,6)

# --- Plot labels / styles (you control everything here) ---
CASE_META = {
    "No control":       {"label": "No control",                "ls": "--", "lw": 2.0},
    "K_lqr":            {"label": r"$K_{\mathrm{LQR}}$",        "ls": "-",  "lw": 2.0},
    "K_adj_fullState":  {"label": r"$K_{\mathrm{adj,full}}$",  "ls": "-",  "lw": 2.0},
    "K_adj":            {"label": r"$K_{\mathrm{adj,red}}$",   "ls": "-",  "lw": 2.0},
}

AX_LABELS = {"t": "t [s]", "x1": "$x_1$ [m]", "u": "u [N]"}


def plot_x1_and_u_single_figure(t, results, case_meta, t_xlim=None):
    # One figure, two stacked axes (shared x)
    fig, (ax_x1, ax_u) = plt.subplots(2, 1, sharex=True, figsize=(12, 7))

    for key, r in results.items():
        meta = case_meta.get(key, {})
        lbl = meta.get("label", key)
        ls = meta.get("ls", "-")
        lw = meta.get("lw", 2.0)

        ax_x1.plot(t, r["Y"][0], label=lbl, linestyle=ls, linewidth=lw)
        ax_u.plot(t, r["u"],     label=lbl, linestyle=ls, linewidth=lw)

    ax_x1.set_ylabel(AX_LABELS["x1"])
    ax_u.set_ylabel(AX_LABELS["u"])
    ax_u.set_xlabel(AX_LABELS["t"])

    ax_x1.grid(True)
    ax_u.grid(True)

    ax_x1.legend()

    if t_xlim is not None:
        ax_u.set_xlim(*t_xlim)

    fig.tight_layout()
    plt.gcf().savefig("./x1_u_vs_t.pdf", bbox_inches="tight")
    return fig
  


if __name__ == "__main__":
    chSize = 18
    plt.rcParams.update({ 'font.size': chSize,  })        # Base font size
    #Penalization factors
    w_x1, w_x1d =1.0, 0.1 # Penalization on x1 displacement and velocity
    w_e, w_ed  = 50.0, 2.0 # Enforcement on the antiphase e = x1+x2
    
    r_u  = 0.05 #Penalization on the control ui^2
    # --- load params ---
    param_file = "params.txt"
    p = load_params(param_file)

    m1, m2 = p["m1"], p["m2"]
    k1, k2 = p["k1"], p["k2"]
    c1, c2 = p["c1"], p["c2"]
    cd, kc = p["cd"], p["kc"]

    F_ext = make_forcing(p)

    t_end = p["t_end"]
    y0 = [0.1, 0.0, 0.0, 0.0]


    # --- your gains ---
    K_lqr = [29.13220515  ,7.44927118 ,19.48431309  ,5.73682962]
    # K_adj_fullState = [14.54161978, 10.12460598,  - 0.20083519,  8.03208862] # Full state adj
    
    K_adj_fullState = [52.69388159 ,11.84605453 ,43.43531895 ,10.28678111] #Adj full state starting from x0 = 0.1 without forcing
    K_adj = [13.65857712 , 3.02411581,0,0] #Only x1, x1d states
    
    cases = {
        "No control": [0,0,0,0],
        "K_lqr": K_lqr,
        "K_adj_fullState": K_adj_fullState,
        "K_adj": K_adj,
    }

    results = {}
    metrics = {}

    # Run both cases with identical settings
    for label, K in cases.items():
        t, Y, u, sol = simulate_2dof_given_K(
            K=K,
            m1=m1, m2=m2,
            k1=k1, k2=k2,
            c1=c1, c2=c2,
            kc=kc, cd=cd,
            F_ext=F_ext,
            t_end=t_end,
            y0=y0,
            n_eval=4000,
            max_step=1e-2,
        )
        results[label] = {"Y": Y, "u": u, "sol": sol}
        metrics[label] = compute_metrics(t, Y, u)
        x1, x1d, x2, x2d = Y
        ciao  = CostFunc_comparison(Y, u,w_x1,w_x1d,w_e,w_ed,r_u)
        
        print('Cost function: ',label,  np.trapezoid(ciao,t))

    print_metrics_table(metrics)
    plot_comparison(t, results)
    plot_K_comparison(cases)
    plot_x1_and_u_single_figure(t, results, CASE_META, t_xlim=(0, 10))

    
    