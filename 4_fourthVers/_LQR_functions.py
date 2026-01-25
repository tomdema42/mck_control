# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 12:23:42 2026

@author: demaria
"""
import numpy as np
from scipy.linalg import solve_continuous_are


# %%
def lqr_gain(A, B, Q, R):
    """
    Continuous-time LQR:
    Solve for K in the state-feedback law u = -Kx that minimizes the cost.
    Inputs:
      A, B: system dynamics matrices
      Q, R: weighting matrices for states and inputs
    Outputs:
      K: optimal gain matrix
    """
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)  # K = R^-1 B^T P
    return K

#%%
def build_Q_R_matrix(w_x1, w_x1d, w_e, w_ed, r_u):
    """
    Build Q matrix to penalize:
      z = [x1, x1d, e=x1+x2, ed=x1d+x2d]
    and R matrix to penalize control effort.
    Inputs:
      w_x1, w_x1d, w_e, w_ed: weights for states
      r_u: weight for control effort
    Outputs:
      Q, R: weighting matrices
    """
    C = np.array([
        [1.0, 0.0, 0.0, 0.0],  # x1
        [0.0, 1.0, 0.0, 0.0],  # x1d
        [1.0, 0.0, 1.0, 0.0],  # e = x1 + x2
        [0.0, 1.0, 0.0, 1.0],  # ed = x1d + x2d
    ])
    W = np.diag([w_x1, w_x1d, w_e, w_ed])
    Q = C.T @ W @ C
    R = np.array([[r_u]])
    return Q, R