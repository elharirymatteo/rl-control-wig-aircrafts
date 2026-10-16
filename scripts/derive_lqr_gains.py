"""Derive the LQR gain matrices the classical baseline flies: finite-difference linearization of
the CPU plant about level flight, then the continuous-time Riccati equation.

Ported from the development repository's scripts/derive_lqr_gains.py (commit 02ebe3c). Run on
that commit's plant, the port reproduces the previously shipped Airship_V7 and Volantex_Ranger
`*_K_LQR.csv` bit-for-bit. Two variants:

    hold         <plane>_K_LQR.csv               the attitude executor and constant-altitude hold
    acquisition  <plane>_K_LQR_acquisition.csv   the altitude acquisition-and-hold protocol

State and input layout, exactly as `falcons.controllers.lqr.control.LQR_Control.LQR_action`
consumes them:

    x = [ aero actuator states (2 per surface: rate, deflection),
          motor states (1 per motor),
          linear velocity (3), angular velocity (3),
          quaternion-error vector part (3),        # reference attitude is level: error = q[1:]
          position (3),
          k_i * integral of position error (3) ]
    u = [ motor commands (n_motors), aileron, elevator, rudder ]   (raw LQR units, pre-normalize)

Model:
  - actuator and motor blocks: the controller's own internal filters, analytic
    (second order: rate' = -2 zeta wn rate - wn^2 defl + u, defl' = rate; first order:
    y' = -y / T + u);
  - integral block: analytic, d(x_int)/dt = k_i * (target - position);
  - rigid-body block (velocity, rates, quaternion vector, position): central finite differences
    of one plant step, with the plant's own actuator lag burnt in for N_BURN_IN steps under the
    constant command so the sensitivity to u is the settled one, not a single 10 ms sample.

Trim point: forward velocity = the airframe's reference airspeed (default_initial_state), level
attitude, zero rates, altitude TRIM_ALTITUDE, zero commands. The development script linearized
at the reference module's default altitude, 1 m, inside ground effect (h/b ~ 0.14 on the V7);
both LQR protocols fly far above that, so the trim is now taken at 60 m. Integral states whose
k_i is zero are structurally inert and are removed before the Riccati solve (zero gain).

Weights. `hold` is the development repository's per-airframe tuning (the weights that produced
the shipped hold gains). The shipped acquisition gains have no surviving derivation; `acquisition`
uses the (qvz, qz, raero, rmotor) that best approximate them within this Q/R template at the
60 m trim (relative Frobenius error 0.34 on the V7, 0.23 on the Ranger, on the plant they were
designed on) -- an approximation of the original tuning, not a reproduction. The Q/R printed in
earlier drafts of the paper (Q_z = 400, R = diag(1, 1, 800, 800, 800)) saturates the elevator
within one to two seconds on every airframe and was never the shipped design.

Q is diagonal, grouped by physical meaning; per-variant scalars (qvz, qz, raero, rmotor) are in
RECIPES. The x-position weight is exactly zero: the target x is pinned while the aircraft flies
forward, so x error grows without bound and any weight on it saturates the actuators.

    PYTHONPATH=src python scripts/derive_lqr_gains.py all                 # write every CSV
    PYTHONPATH=src python scripts/derive_lqr_gains.py Airship_V7 --variant hold --check
                                                                          # compare, write nothing
Navion and Cirrus_SR22 carry the development repository's placeholder weights; no benchmark
flies an LQR on them, and their gains are written only if the linearized closed loop is stable.
"""
import argparse
import os
import sys

import numpy as np
import scipy.linalg

from falcons.aircraft.config import AircraftConfig
from falcons.controllers.lqr.control import LQR_Control

PLANES = ["Airship_V7", "Volantex_Ranger", "Navion", "Cirrus_SR22"]
TRIM_ALTITUDE = 60.0   # [m] linearization altitude, out of ground effect on every airframe
RB_DIM = 12            # linear velocity, angular velocity, quaternion vector part, position
EPS_RB = np.array([1e-2, 1e-2, 1e-2,     # linear velocity [m/s]
                   1e-3, 1e-3, 1e-3,     # angular velocity [rad/s]
                   1e-4, 1e-4, 1e-4,     # quaternion vector part [-]
                   1e-3, 1e-3, 1e-3])    # position [m]
EPS_U = 1e-3
N_BURN_IN = 150        # plant steps (1.5 s) to settle the plant's actuator lag under a command

# (qvz, qz, raero, rmotor) per variant and airframe.
RECIPES = {
    "hold": {
        "Airship_V7":      dict(qvz=0.1, qz=5.0, raero=30000.0, rmotor=50.0),
        "Volantex_Ranger": dict(qvz=1.0, qz=5.0, raero=1000.0, rmotor=100.0),
        "Navion":          dict(qvz=0.1, qz=100.0, raero=1.0, rmotor=100.0),     # placeholder
        "Cirrus_SR22":     dict(qvz=0.1, qz=100.0, raero=1.0, rmotor=100.0),     # placeholder
    },
    "acquisition": {
        "Airship_V7":      dict(qvz=1.0, qz=1.0, raero=900.0, rmotor=1.0),
        "Volantex_Ranger": dict(qvz=9.0, qz=5.5, raero=1300.0, rmotor=1.0),
    },
}
OUTPUT = {"hold": "{p}_K_LQR.csv", "acquisition": "{p}_K_LQR_acquisition.csv"}


def build_env(plane):
    os.environ["TRAJ_CONST_ALT"] = f"{TRIM_ALTITUDE}"    # the reference module reads it per call
    env = LQR_Control(aircraft_name=plane, targets={"trajectory_type": "constant_altitude"},
                      save_history=False, use_sensor_noise=False, use_estimator=False)
    env.reset()
    env.t = 0.0
    return env


def layout(env):
    n_as, n_mo, n_aero = env.n_aero_states, env.n_motors, env.n_aero_surfaces
    return n_as, n_mo, n_aero, n_as + n_mo + RB_DIM + 3, n_mo + n_aero


def rigid_body_deriv(env, rb, u, n_mo, n_aero):
    """(x_rb(t + dt) - x_rb) / dt from the plant, after burning in the actuator lag under u."""
    frozen = {"position": np.asarray(rb[9:12], float).copy(),
              "linear_vel": np.asarray(rb[0:3], float).copy(),
              "angular_vel": np.asarray(rb[3:6], float).copy(),
              "orientation": np.array([1.0, *rb[6:9]])}
    cmd = env.normalize_actuator_commands(u)
    motor, aero = cmd[0:n_mo], cmd[n_mo:n_mo + n_aero]
    env.airship.actuator_system.reset_all()
    for _ in range(N_BURN_IN):
        env.airship.state = dict(frozen)
        env.airship.step(aero, motor)
    env.airship.state = dict(frozen)
    nxt = env.airship.step(aero, motor)
    y1 = np.concatenate([nxt["linear_vel"], nxt["angular_vel"], nxt["orientation"][1:],
                         nxt["position"]])
    return (y1 - rb) / env.dt


def linearize(env, x_trim, u_trim):
    n_as, n_mo, n_aero, n_x, n_u = layout(env)
    rb0, int0 = n_as + n_mo, n_as + n_mo + RB_DIM
    A, B = np.zeros((n_x, n_x)), np.zeros((n_x, n_u))
    for i in range(n_aero):                          # aero actuator filters
        name = env.aero_actuator_names[i]
        wn, zeta = env.surface_dynamics[name]["omega_n"], env.surface_dynamics[name]["zeta"]
        r, p = 2 * i, 2 * i + 1
        A[r, r], A[r, p], A[p, r] = -2 * zeta * wn, -wn ** 2, 1.0
        B[r, n_mo + i] = 1.0
    for i in range(n_mo):                            # motor filters
        r = n_as + i
        A[r, r] = -1.0 / env.motor_dynamics[env.motor_names[i]]["T_s"]
        B[r, i] = 1.0
    for i in range(3):                               # integral of position error, scaled by k_i
        A[int0 + i, rb0 + 9 + i] = -env.k_i[i]
    rb_trim = x_trim[rb0:rb0 + RB_DIM]
    for j in range(RB_DIM):
        d = np.zeros(RB_DIM); d[j] = EPS_RB[j]
        A[rb0:rb0 + RB_DIM, rb0 + j] = (rigid_body_deriv(env, rb_trim + d, u_trim, n_mo, n_aero)
                                        - rigid_body_deriv(env, rb_trim - d, u_trim, n_mo, n_aero)) \
            / (2 * EPS_RB[j])
    for j in range(n_u):
        d = np.zeros(n_u); d[j] = EPS_U
        B[rb0:rb0 + RB_DIM, j] = (rigid_body_deriv(env, rb_trim, u_trim + d, n_mo, n_aero)
                                  - rigid_body_deriv(env, rb_trim, u_trim - d, n_mo, n_aero)) \
            / (2 * EPS_U)
    return A, B


def weights(env, qvz, qz, raero, rmotor):
    n_as, n_mo, n_aero, _, _ = layout(env)
    q = [14.6, 8.2] * n_aero              # actuator rate, deflection
    q += [0.25] * n_mo                    # motor states
    q += [1.0, 1.0, qvz]                  # linear velocity
    q += [18.2] * 3                       # angular velocity
    q += [1e-5] * 3                       # quaternion vector part: no attitude bias
    q += [0.0, 1.0, qz]                   # position: x untracked, y, z
    q += [0.0, 1.0, qz]                   # integral states reuse the position weights
    return np.array(q), np.array([rmotor] * n_mo + [raero] * n_aero)


def plant(plane):
    """(env, A, B, rb0) at the level trim: the expensive part, shared by every recipe."""
    env = build_env(plane)
    n_as, n_mo, _, n_x, n_u = layout(env)
    rb0 = n_as + n_mo
    x_trim = np.zeros(n_x)
    x_trim[rb0:rb0 + 3] = [env.target_airspeed, 0.0, 0.0]
    x_trim[rb0 + 9:rb0 + 12] = env.target_ref()
    A, B = linearize(env, x_trim, np.zeros(n_u))
    return env, A, B, rb0


def solve(env, A, B, rb0, qvz, qz, raero, rmotor):
    """K and the closed-loop eigenvalues for one (Q, R)."""
    q, r = weights(env, qvz, qz, raero, rmotor)
    int0 = rb0 + RB_DIM
    # zero-weight pure integrators make (A, Q) undetectable and the Riccati solve ill-posed: the
    # integral states whose k_i is zero, and the x position (untracked, and no force depends on it)
    dead = [int0 + i for i in range(3) if env.k_i[i] == 0.0] + [rb0 + 9]
    keep = [i for i in range(A.shape[0]) if i not in dead]
    Ar, Br = A[np.ix_(keep, keep)], B[keep, :]
    P = scipy.linalg.solve_continuous_are(Ar, Br, np.diag(q[keep]), np.diag(r))
    K = np.zeros((B.shape[1], A.shape[0]))
    K[:, keep] = np.linalg.solve(np.diag(r), Br.T @ P)
    return K, np.linalg.eigvals(A - B @ K)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("plane", choices=PLANES + ["all"])
    ap.add_argument("--variant", choices=["hold", "acquisition", "both"], default="both")
    ap.add_argument("--check", action="store_true",
                    help="compare against the stored CSV and write nothing")
    a = ap.parse_args(argv)
    planes = PLANES if a.plane == "all" else [a.plane]
    variants = ["hold", "acquisition"] if a.variant == "both" else [a.variant]
    worst = 0.0
    for p in planes:
        env, A, B, rb0 = plant(p)
        for v in variants:
            if p not in RECIPES[v]:
                print(f"{p} {v}: no recipe -- skipped"); continue
            K, eig = solve(env, A, B, rb0, **RECIPES[v][p])
            out = AircraftConfig(p).dir / OUTPUT[v].format(p=p)
            status = "stable" if eig.real.max() < 1e-9 else "UNSTABLE"
            if a.check:
                Ks = np.loadtxt(out, delimiter=",")
                err = np.linalg.norm(K - Ks) / np.linalg.norm(Ks)
                worst = max(worst, err)
                print(f"{p} {v}: rel. Frobenius difference to {out.name} {err:.3e} ({status})")
            elif status == "stable":
                np.savetxt(out, K, delimiter=",")
                print(f"{p} {v}: wrote {out} ({status}, max Re eig {eig.real.max():.2e})")
            else:
                print(f"{p} {v}: NOT written ({status}, max Re eig {eig.real.max():.2e})")
    return worst


if __name__ == "__main__":
    main()
    sys.exit(0)
