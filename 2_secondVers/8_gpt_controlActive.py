import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# -----------------------------
# Forcing (edit as you want)
# -----------------------------
def make_forcing(F0=10.0, w=10.0):
    # def F_ext(t):
    #     return F0 * np.sin(w * t)
    def F_ext(t):
        t0=1
        dt =1
        return F0 if (t0 <= t < t0 + dt) else 0.0
    return F_ext


# -----------------------------
# Simulation (passive + active)
# -----------------------------
def simulate_2dof_with_control(
    m1=1.0, m2=0.2,
    k1=80.0, k2=20.0,
    c1=0.8, c2=0.2,
    kc=30.0, cd=0.5,
    F0=2.0, w_forcing=8.0,
    w_track=25.0, zeta_track=1.2,   # controller aggressiveness
    u_max=100.0,                    # actuator saturation (N)
    t_end=15.0, max_step=1e-2,
    y0=(0.0, 0.0, 0.0, 0.0),
):
    """
    State y = [x1, x1d, x2, x2d]

    Your original RHS (passive):
      x1dd = (-k1*x1 - c1*x1d + cd*(x2d-x1d) + kc*(x2-x1) + F_ext(t)) / m1
      x2dd = (-k2*x2 - c2*x2d - cd*(x2d-x1d) - kc*(x2-x1)) / m2

    We add an active force u(t) applied to mass 2 (actuator to ground):
      x2dd = passive_terms/m2 + u/m2
    """

    F_ext = make_forcing(F0=F0, w=w_forcing)

    # store control history
    u_log_t = []
    u_log = []

    def clip(u):
        return np.clip(u, -u_max, u_max)

    def rhs_passive(t, y):
        x1, x1d, x2, x2d = y

        x1dd = (-k1 * x1 - c1 * x1d + cd * (x2d - x1d) + kc * (x2 - x1) + F_ext(t)) / m1
        x2dd = (-k2 * x2 - c2 * x2d - cd * (x2d - x1d) - kc * (x2 - x1)) / m2

        return [x1d, x1dd, x2d, x2dd]

    def rhs_controlled(t, y):
        x1, x1d, x2, x2d = y

        # "model" accelerations without control (these are exactly your RHS terms)
        x1dd = (-k1 * x1 - c1 * x1d + cd * (x2d - x1d) + kc * (x2 - x1) + F_ext(t)) / m1
        x2dd_passive = (-k2 * x2 - c2 * x2d - cd * (x2d - x1d) - kc * (x2 - x1)) / m2

        # enforce anti-phase: x2 ~= -x1  <=>  e = x2 + x1 -> 0
        e = x2 + x1
        ed = x2d + x1d

        # desired x2 acceleration so that: e_dd + 2*zeta*w*e_d + w^2*e = 0
        x2dd_des = -x1dd - 2.0 * zeta_track * w_track * ed - (w_track ** 2) * e

        # plant: x2dd = x2dd_passive + u/m2  =>  u = m2*(x2dd_des - x2dd_passive)
        u = m2 * (x2dd_des - x2dd_passive)
        u = float(clip(u))

        # log u
        u_log_t.append(t)
        u_log.append(u)

        x2dd = x2dd_passive + u / m2

        return [x1d, x1dd, x2d, x2dd]

    # run both simulations (passive vs controlled)
    sol_passive = solve_ivp(
        rhs_passive, (0.0, t_end), y0,
        max_step=max_step, rtol=1e-7, atol=1e-9
    )

    sol_controlled = solve_ivp(
        rhs_controlled, (0.0, t_end), y0,
        max_step=max_step, rtol=1e-7, atol=1e-9
    )

    u_log_t = np.array(u_log_t)
    u_log = np.array(u_log)

    return sol_passive, sol_controlled, (u_log_t, u_log)


# -----------------------------
# Run + plots
# -----------------------------
if __name__ == "__main__":
    sol0, sol1, (tu, u) = simulate_2dof_with_control(
        # edit parameters here
        m1=.3, m2=0.1,
        k1=11.83, k2=20.0,
        c1=0.11, c2=0.05,
        kc=0.0, cd=1.6,
        F0=2.0, w_forcing=8.0,
        w_track=25.0, zeta_track=1.2,
        u_max=10.0,
        t_end=10.0,
        y0=(0.0, 0.0, 0.0, 0.0),
    )

    # unpack
    t0 = sol0.t
    x1_0, x1d_0, x2_0, x2d_0 = sol0.y

    t1 = sol1.t
    x1_1, x1d_1, x2_1, x2d_1 = sol1.y

    # velocities (like your plot)
    plt.figure(figsize=(12, 6))
    plt.plot(t0, x1_0, label="x1 passive")
    plt.plot(t0, x2_0, label="x2 passive")
    plt.plot(t1, x1_1, "--", label="x1 controlled")
    plt.plot(t1, x2_1, "--", label="x2 controlled")
    plt.xlabel("t [s]")
    plt.ylabel("Displacement [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # # displacements (to see anti-phase clearly)
    # plt.figure(figsize=(12, 6))
    # plt.plot(t1, x1_1, label="x1 controlled")
    # plt.plot(t1, x2_1, label="x2 controlled")
    # plt.xlabel("t [s]")
    # plt.ylabel("Displacement [m]")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # control force
    plt.figure(figsize=(12, 4))
    plt.plot(tu, u, label="u(t)")
    plt.xlabel("t [s]")
    plt.ylabel("Control force [N]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
