# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 14:30:42 2026

@author: demaria
"""

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from _auxFunc import load_params, build_dyn_system_np
from _auxFunc import rhs_fullState
from _LQR_functions import build_Q_R_matrix, lqr_gain

chSize = 18
plt.rcParams.update({ 'font.size': chSize,})        # Base font size

# -----------------------------
# Simulation (passive + LQR)
# -----------------------------
def simulate_2dof_with_lqr(
    m1, m2,
    k1, k2,
    c1, c2,
    kc, cd,
    t_end,
    y0,
    w_x1, w_x1d, w_e, w_ed,
    r_u,
):
    """
    Simulate the 2-DOF system with and without LQR control.
    Returns:
      sol_passive: solution without control
      sol_lqr: solution with LQR control
      u_export: control force time series
      K: LQR gain matrix
    """
    # --- Build state-space matrices A, B ---
    A,B = build_dyn_system_np(m1, m2, k1, k2, c1, c2, kc, cd)

    # --- Build Q to penalize: x1, x1d, e=x1+x2, ed=x1d+x2d ---
    Q,R = build_Q_R_matrix(w_x1, w_x1d, w_e, w_ed, r_u)
    # --- Compute LQR gain K ---
    K = lqr_gain(A, B, Q, R) 

    # --- Simulate LQR-controlled systems ---   
    rhs_fullState_lqr = lambda t, y: rhs_fullState(
        t, y, K, m1, m2, k1, k2, c1, c2, kc, cd, Passive=False
    )
    sol_lqr = solve_ivp(
        rhs_fullState_lqr, (0.0, t_end), y0,
        max_step=1e-2, rtol=1e-7, atol=1e-9
    )
    # --- Simulate passive system (no control) ---
    rhs_fullState_noControl = lambda t, y: rhs_fullState(
        t, y, K, m1, m2, k1, k2, c1, c2, kc, cd, Passive=True
    )
    sol_passive = solve_ivp(
        rhs_fullState_noControl, (0.0, t_end), y0,
        max_step=1e-2, rtol=1e-7, atol=1e-9
    )

    # --- Extract control force time series ---
    Y = sol_lqr.y
    u_export = -(K @ Y).ravel()
    
    return sol_passive, sol_lqr, u_export, K


# -----------------------------
# Run + plots
# -----------------------------
if __name__ == "__main__":
    # Load parameters
    param_file = "params.txt"
    p = load_params(param_file)
    # System parameters
    m1, m2 = p["m1"], p["m2"]
    k1, k2 = p["k1"], p["k2"]
    c1, c2 = p["c1"], p["c2"]
    cd, kc = p["cd"], p["kc"]

    # Weights for Q via outputs: [x1, x1d, e=x1+x2, ed=x1d+x2d]
    w_x1 = p['w_x1']    # Penalty on x1
    w_x1d = p['w_x1d']  # Penalty on x1d
    w_e = p['w_e']      # Penalty on e = x1 + x2
    w_ed = p['w_ed']    # Penalty on ed = x1d + x2d
    # Weight for R:
    r_u  = p['r_u']      # Penalty on the control ui^2
    # Simulation time and initial state
    t_end = p['t_end']
    y0 = (p['x1_0'], p['x1d_0'], p['x2_0'], p['x2d_0'])

    sol_noControl, sol_control, u_series, K = simulate_2dof_with_lqr(
        m1, m2,
        k1, k2,
        c1, c2,
        kc, cd,
        t_end,
        y0,
        w_x1, w_x1d, w_e, w_ed,
        r_u,
    )

    print("LQR gain K =", -K)
    # Extract results for plotting
    t_noControl = sol_noControl.t
    x1_noControl, x1d_noControl, x2_noControl, x2d_noControl = sol_noControl.y

    t_control = sol_control.t
    x1_control, x1d_control, x2_control, x2d_control = sol_control.y
    
    # Plots
    from _plottingFunc import plot_controlVs_noControl, plot_control_force
    plot_controlVs_noControl(
        t_noControl, x1_noControl, x2_noControl,
        t_control, x1_control, x2_control,
    )
    plot_control_force(t_end, u_series)
    