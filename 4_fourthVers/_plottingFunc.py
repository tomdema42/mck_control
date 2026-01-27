# -*- coding: utf-8 -*-
"""
Plotting functions for the 2-DOF system control.

@author: demaria
"""

import matplotlib.pyplot as plt
import numpy as np




def plot_style(chSize=18):
    """
    Set a consistent Matplotlib style (fonts, grid, figure size, etc.).
    Call once at the top of your script.
    """
    plt.rcParams.update({
        "font.size": chSize,              # Base font size
        "axes.titlesize": chSize + 2,
        "axes.labelsize": chSize,
        "xtick.labelsize": chSize - 2,
        "ytick.labelsize": chSize - 2,
        "legend.fontsize": chSize - 2,
        "figure.titlesize": chSize + 4,

        "figure.figsize": (12, 6),
        "figure.dpi": 120,

        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",

        "lines.linewidth": 2.0,

        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })

#%%
def plot_controlVs_noControl(
    t0, x1_0, x2_0,
    t1, x1_1, x2_1,
):
    """
    Plot displacements with and without control, and control force over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(t0, x1_0,'.-', label="x1 passive")
    plt.plot(t0, x2_0, label="x2 passive")
    plt.plot(t1, x1_1, ".-", label="x1 Control")
    plt.plot(t1, x2_1, "--", label="x2 Control")
    plt.xlabel("time [s]")
    plt.ylabel("Displacement [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return

def plot_control_force(t_end, u_export):
    """
    Plot control force over time.
    """
    t_u = np.linspace(0, t_end, len(u_export))
    plt.figure(figsize=(12, 4))
    plt.plot(t_u,u_export, 'k.-')
    plt.xlabel("time [s]")
    plt.ylabel("Control force [N]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return

def plot_optimization_history(info):
    """
    Plot optimization history: cost function over iterations.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(info["J_hist"] / np.min(info["J_hist"]))
    plt.xlabel("ADAM iteration")
    plt.ylabel("J/min(J)")
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return

def plot_gain_history_fullState(K_hist):
    """
    Plot gain vector history over optimization iterations.
    """
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
    return
def plot_gain_history_reducedState(K_hist):
    """
    Plot gain vector history over optimization iterations.
    """
    K_hist_np = np.array([np.array(k) for k in K_hist])
    plt.figure(figsize=(12, 4))
    plt.plot(K_hist_np[:, 0], label="K1")
    plt.plot(K_hist_np[:, 1], label="K2")
    plt.xlabel("ADAM iteration")
    plt.ylabel("K_i")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
    return
