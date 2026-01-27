# -*- coding: utf-8 -*-
"""
LQR optimal control of a 2-DOF system: compute optimal gain and simulate
the closed-loop system.

@author: demaria
"""


from scipy.integrate import solve_ivp
from _auxFunc import load_params, build_dyn_system_np
from _auxFunc import rhs_fullState
from _LQR_functions import build_Q_R_matrix, lqr_gain
from _plottingFunc import plot_controlVs_noControl, plot_control_force
from _plottingFunc import plot_style
plot_style(18)

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
    r_a,
):
    """
    Simulate the 2-DOF system with and without LQR control.
    Returns:
      sol_passive: solution without control
      sol_lqr: solution with LQR control
      a_export: control force time series
      K: LQR gain matrix
    """
    # --- Build state-space matrices A, B ---
    A,B = build_dyn_system_np(m1, m2, k1, k2, c1, c2, kc, cd)

    # --- Build Q to penalize: s1, s1d, e=s1+s2, ed=s1d+s2d ---
    Q,R = build_Q_R_matrix(w_x1, w_x1d, w_e, w_ed, r_a)
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
    a_export = -(K @ Y).ravel()
    
    return sol_passive, sol_lqr, a_export, K


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

    # Weights for Q via outputs: [s1, s1d, e=s1+s2, ed=s1d+s2d]
    w_x1 = p['w_x1']    # Penalty on s1
    w_x1d = p['w_x1d']  # Penalty on s1d
    w_e = p['w_e']      # Penalty on e = s1 + s2
    w_ed = p['w_ed']    # Penalty on ed = s1d + s2d
    # Weight for R:
    r_a  = p['r_a']      # Penalty on the control ai^2
    # Simulation time and initial state
    t_end = p['t_end']
    y0 = (p['x1_0'], p['x1d_0'], p['x2_0'], p['x2d_0'])

    sol_noControl, sol_control, a_series, K = simulate_2dof_with_lqr(
        m1, m2,
        k1, k2,
        c1, c2,
        kc, cd,
        t_end,
        y0,
        w_x1, w_x1d, w_e, w_ed,
        r_a,
    )

    print("LQR gain K =", -K)
    # Extract results for plotting
    t_noControl = sol_noControl.t
    s1_noControl, s1d_noControl, s2_noControl, s2d_noControl = sol_noControl.y

    t_control = sol_control.t
    s1_control, s1d_control, s2_control, s2d_control = sol_control.y
    
    # Plots
    
    plot_controlVs_noControl(
        t_noControl, s1_noControl, s2_noControl,
        t_control, s1_control, s2_control,
    )
    plot_control_force(t_end, a_series)
    