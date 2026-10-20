"""Single-step parity of the three plants: CPU (numpy, float64) and Torch against Warp.

Every airframe, ground effect on and off, gusts off. From one batch of random but in-envelope
states -- wing height h/b in 0.2..6, airspeed 0.8..1.2 x the airframe's cruise, alpha and beta
within +-10 deg, attitude, body rates, actuator states and rates -- each plant takes ONE step of
dt = 0.01 s with the benchmark's integrator (Warp solver_type 1: fixed Dormand-Prince, forces
frozen over the step) under the same commands, and the test compares what the step integrated
(body force and moment) and where it ended (position, velocity, rates, attitude, actuators).

The CPU plant is compared on the full command set. The Torch plant is the altitude env's plant: it
commands elevator and one throttle shared by every motor, with aileron and rudder held at zero
against fixed +-15 deg limits, so it is compared on that subset (aileron/rudder STATES are still
random, inside +-15 deg).

Tolerances. Warp and Torch compute in float32 (unit round-off 6e-8), the CPU plant in float64.
The polynomial coefficients are evaluated with alpha and beta in degrees, so the cubic terms reach
10^3 and the coefficient sums lose about three digits to cancellation; forces and moments then
agree to a few 1e-6 of the aircraft weight (and weight x span) -- the float32 floor of that
evaluation, measured by `parity_errors` below. The bounds are set about 10x above the measured
worst case (TOL), which is still 3-5 orders of magnitude below a modelling difference: before the
CPU plant was aligned (no rate damping, thrust capped at max_thrust and a power model for pistons,
a single motor's reaction torque, scipy RK45) it missed Warp by up to 0.56 of the weight in force
(the A0S thrust cap), 4e-2 of weight x span in moment and 8e-2 rad/s in body rate after one step.

Measured worst case over the 5 airframes x GE on/off (max |difference| / scale):
                force   moment   position[m]  velocity  rates[rad/s]  actuator  attitude[deg]
    CPU   vs Warp 3.0e-6  1.9e-7   7.1e-6       1.9e-7    2.5e-7        7.6e-8    1.6e-5
    Torch vs Warp 1.9e-6  1.8e-7   1.9e-6       6.2e-8    3.0e-7        4.8e-8    5.8e-6
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from falcons.aircraft.config import PLANES, AircraftConfig

N = 64                      # > 4: Warp's buffer setter broadcasts 1-D inputs of length <= 4
DT = 0.01
TOL = {                     # bound on max |difference| / scale (see `scales`), ~10x the measured
    "force": 3e-5,          # worst case over every airframe, GE on and off, both comparisons
    "moment": 3e-6,
    "position": 5e-5,       # metres: float32 spacing is 4-8e-6 m at the 30-70 m altitudes drawn
    "velocity": 3e-6,
    "rates": 3e-6,
    "attitude_deg": 1e-4,
    "actuator": 1e-6,
}


def draw(plane, seed, lateral=True):
    """One batch of in-envelope states and commands, Warp's quaternion order [x, y, z, w]."""
    raw = AircraftConfig(plane).load()
    VP = raw["vehicle_params"]
    surf = VP["actuator_system"]["aero_surfaces"]
    span = VP["wing"]["span"]
    va0 = float(raw["default_initial_state"]["linear_vel"][0])
    rng = np.random.default_rng(seed)
    u = lambda lo, hi, *s: rng.uniform(lo, hi, size=s or (N,))
    h = u(0.2, 6.0) * span                                 # CG altitude; the GE sweep is indexed by it
    va, a, b = va0 * u(0.8, 1.2), np.radians(u(-10, 10)), np.radians(u(-10, 10))
    vel = np.stack([va * np.cos(a) * np.cos(b), va * np.sin(b), va * np.sin(a) * np.cos(b)], 1)
    rpy = np.stack([u(-30, 30), u(-15, 15), u(-180, 180)], 1)
    quat = Rotation.from_euler("ZYX", rpy[:, ::-1], degrees=True).as_quat()   # [x, y, z, w]
    lim = lambda k: (surf[k]["min_deflection"], surf[k]["max_deflection"])
    lat = (lambda k: lim(k)) if lateral else (lambda k: (-15.0, 15.0))
    st = {"elevator": u(*lim("elevator")), "aileron": u(*lat("ailerons")),
          "rudder": u(*lat("rudder")), "throttle_left": u(0, 1),
          "elevator_dot": u(-50, 50), "aileron_dot": u(-50, 50), "rudder_dot": u(-50, 50)}
    cmd = {"elevator": u(-1, 1), "aileron": u(-1, 1), "rudder": u(-1, 1), "throttle_left": u(-1, 1)}
    twin = len(VP["actuator_system"]["motors"]) > 1
    if not lateral:                                      # the Torch plant's command set
        cmd["aileron"], cmd["rudder"] = np.zeros(N), np.zeros(N)
    if twin and lateral:
        st["throttle_right"], cmd["throttle_right"] = u(0, 1), u(-1, 1)
    else:                                                # one throttle for every motor
        st["throttle_right"], cmd["throttle_right"] = st["throttle_left"], cmd["throttle_left"]
    state = {"position": np.stack([np.zeros(N), np.zeros(N), -h], 1), "linear_vel": vel,
             "angular_vel": np.stack([u(-0.5, 0.5), u(-0.5, 0.5), u(-0.5, 0.5)], 1),
             "orientation": quat}
    return state, st, cmd


def _f32(d):
    return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in d.items()}


def step_warp(plane, ge, state, st, cmd):
    import warp as wp
    from falcons.aircraft.params import load_params
    from falcons.sim.warp.aircraft import Aircraft
    wp.init()
    co = load_params(plane)
    VP = co["vehicle_params"]
    VP.WP, VP.J, VP.PP = VP.WP.as_warp_struct(), VP.J.as_warp_struct(), VP.PP.as_warp_struct()
    VP.sensor_system = None
    m = Aircraft(N, "cuda", {"AP": co["aero_params"].as_warp_struct(), "VP": VP,
                             "CL": co["control_limits"], "EP": co["environment_params"]},
                 save_history=False, in_ground_effect=ge)
    m.solver_type = 1
    m.reset(_f32(state), _f32(st))
    m.step(_f32(cmd))
    gust = m.wind_model.total_linear_gusts.numpy()
    assert np.all(gust == 0.0), "gusts must be off for a parity step"
    s = m._state
    return dict(force=m._Fb.numpy(), moment=m._Mb.numpy(), position=s["position"].numpy(),
                velocity=s["linear_vel"].numpy(), rates=s["angular_vel"].numpy(),
                quat=s["orientation"].numpy(),
                actuator=np.stack([m._actuator_states[k].numpy() for k in
                                   ("elevator", "aileron", "rudder", "throttle_left",
                                    "throttle_right")], 1))


def step_cpu(plane, ge, state, st, cmd):
    from falcons.sim.cpu.aircraft import Aircraft
    a = Aircraft(plane, in_ground_effect=ge)
    surfaces = a.actuator_system.actuator_groups["aero_surfaces"].actuators
    motors = a.actuator_system.actuator_groups["motors"].actuators
    assert [s.name for s in surfaces] == ["elevator", "ailerons", "rudder"]
    out = {k: [] for k in ("force", "moment", "position", "velocity", "rates", "quat", "actuator")}
    for i in range(N):
        q = state["orientation"][i]
        a.reset(init_state={"position": state["position"][i].copy(),
                            "linear_vel": state["linear_vel"][i].copy(),
                            "angular_vel": state["angular_vel"][i].copy(),
                            "orientation": np.array([q[3], q[0], q[1], q[2]])})
        for act, k in zip(surfaces, ("elevator", "aileron", "rudder")):
            act.state = np.array([st[k][i], st[k + "_dot"][i]])
        thr = ("throttle_left", "throttle_right")[:len(motors)]
        for act, k in zip(motors, thr):
            act.state = np.array([st[k][i]])
        a.step(np.array([cmd["elevator"][i], cmd["aileron"][i], cmd["rudder"][i]]),
               np.array([cmd[k][i] for k in thr]))
        qn = a.state["orientation"]
        out["force"].append(a.Fb); out["moment"].append(a.Mb)
        out["position"].append(a.state["position"]); out["velocity"].append(a.state["linear_vel"])
        out["rates"].append(a.state["angular_vel"]); out["quat"].append([qn[1], qn[2], qn[3], qn[0]])
        th = [m.state[0] for m in motors]
        out["actuator"].append([s.state[0] for s in surfaces] + [th[0], th[-1]])
    return {k: np.array(v) for k, v in out.items()}


def step_torch(plane, ge, state, st, cmd):
    import torch
    from falcons.sim.torch.altitude import AltitudeEnv
    e = AltitudeEnv(N, "cuda", {"aircraft": plane, "in_ground_effect": ge})
    t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device="cuda")
    e.pos, e.vel, e.omega = t(state["position"]), t(state["linear_vel"]), t(state["angular_vel"])
    e.quat = t(state["orientation"])
    e.elev, e.ail, e.rud, e.thr = t(st["elevator"]), t(st["aileron"]), t(st["rudder"]), t(st["throttle_left"])
    e.elev_d, e.ail_d, e.rud_d = t(st["elevator_dot"]), t(st["aileron_dot"]), t(st["rudder_dot"])
    Fb, Mb = e.physics_step(t(np.stack([cmd["elevator"], cmd["throttle_left"]], 1)))
    c = lambda x: x.detach().cpu().numpy()
    return dict(force=c(Fb), moment=c(Mb), position=c(e.pos), velocity=c(e.vel), rates=c(e.omega),
                quat=c(e.quat), actuator=np.stack([c(e.elev), c(e.ail), c(e.rud), c(e.thr),
                                                   c(e.thr)], 1))


def scales(plane, state):
    """What each difference is measured against: weight, weight x span, 1 m, the airspeed,
    1 rad/s and the 40 deg full travel of a control surface."""
    raw = AircraftConfig(plane).load()
    VP = raw["vehicle_params"]
    W = VP["mass"] * raw["environment_params"]["g"]
    va = np.linalg.norm(state["linear_vel"], axis=1).max()
    return {"force": W, "moment": W * VP["wing"]["span"], "position": 1.0, "velocity": va,
            "rates": 1.0, "actuator": 40.0}


def parity_errors(ref, other, sc):
    """max |difference| / scale per quantity; attitude as the angle of the relative rotation (a
    robust form: 2 arccos of the quaternion dot product turns float32 round-off near 1 into
    hundredths of a degree)."""
    err = {k: float(np.max(np.abs(ref[k] - other[k])) / sc[k]) for k in sc}
    rel = Rotation.from_quat(np.asarray(ref["quat"], float)).inv() \
        * Rotation.from_quat(np.asarray(other["quat"], float))
    err["attitude_deg"] = float(np.degrees(rel.magnitude()).max())
    return err


CASES = [(p, ge) for p in PLANES for ge in (True, False)]


@pytest.mark.cuda
@pytest.mark.parametrize("plane,ge", CASES)
def test_cpu_matches_warp(plane, ge):
    state, st, cmd = draw(plane, seed=1, lateral=True)
    err = parity_errors(step_warp(plane, ge, state, st, cmd), step_cpu(plane, ge, state, st, cmd),
                        scales(plane, state))
    bad = {k: v for k, v in err.items() if v > TOL[k]}
    assert not bad, f"{plane} GE={ge}: {bad}"


@pytest.mark.cuda
@pytest.mark.parametrize("plane,ge", CASES)
def test_torch_matches_warp(plane, ge):
    state, st, cmd = draw(plane, seed=2, lateral=False)
    err = parity_errors(step_warp(plane, ge, state, st, cmd), step_torch(plane, ge, state, st, cmd),
                        scales(plane, state))
    bad = {k: v for k, v in err.items() if v > TOL[k]}
    assert not bad, f"{plane} GE={ge}: {bad}"
