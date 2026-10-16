"""Robustness of the attitude-executor policy (PPO) under four disturbance families -- action
delay, sensor delay, observation noise and Dryden wind -- each swept mild -> medium -> severe with
values grounded in small-UAV hardware and low-altitude turbulence physics. Same maneuver streams
and envelope guard as the clean comparison (benchmark.maneuvers.fly); metrics always come from the
TRUE state, never the disturbed obs.

One executor, PPO, over every attitude seed checkpoint: a disturbance corrupts what the policy
sees or commands and never what is scored, so a cell compares CONDITIONS rather than methods.

The stochastic families (sensor noise, Dryden wind) are flown N_REAL times per (seed, maneuver,
severity), as N_REAL parallel envs with independent noise/gust streams (maneuvers.fly_batch); the
nominal cell and the two delays are deterministic and flown once. A cell pools every run of every
seed: survival as k/n with an interval, the errors as an IQM with a hierarchical bootstrap
interval over seeds (falcons/benchmark/stats.py). robustness_runs.csv keeps every run.

Full derivation and citations: docs/disturbance_model.md.
Run: run(planes) -> <results_dir>/robustness.csv
"""
import csv
import os
import zlib
from pathlib import Path

import numpy as np
import torch

from falcons.benchmark.maneuvers import fly_batch, make_env
from falcons.benchmark.streams import DT, MANEUVERS
from falcons.paths import CKPT_DIR, RESULTS_DIR

SEVERITIES = ["mild", "medium", "severe"]
N_REAL = 100                    # realizations per (seed, maneuver, severity) of a stochastic family
STOCHASTIC = ("obs_noise", "wind")

# --- grounded severity grids -------------------------------------------------
# actuator/command transport lag on top of the modeled servo dynamics: FC compute + link + servo
# deadtime, at DT = 0.01 s = 2/5/10 steps
ACTION_DELAY_MS = {"mild": 20, "medium": 50, "severe": 100}
# obs latency (EKF fusion), 2/4/8 steps; 20 ms is the figure the clean comparison assumes
SENSOR_DELAY_MS = {"mild": 20, "medium": 40, "severe": 80}
# Dryden 6-component gust field, dialed to a target longitudinal intensity Iu = sigma_u / Va;
# a literal MIL W20 of 15-45 kt would exceed airspeed for the 15-28 m/s class
WIND_IU = {"mild": 0.05, "medium": 0.10, "severe": 0.15}     # target sigma_u / Va
# regridded finer: the first sweep (0.10/0.20/0.30) crashed every method at medium+, losing all
# resolution. Iu<=0.15 is the survivable band where the executors actually separate; 0.05-0.15
# spans light -> moderate low-altitude turbulence for the 15-28 m/s class.
# physical sensor-noise floors (SI), mapped to obs-space per channel below
NOISE = {  # (attitude rad, rate rad/s, Va m/s, alpha/beta rad, climb m/s)
    "mild":   dict(att=np.radians(0.5), rate=np.radians(1.0), va=0.2, ab=np.radians(0.5), vz=0.10),
    "medium": dict(att=np.radians(1.0), rate=np.radians(2.0), va=0.5, ab=np.radians(1.0), vz=0.25),
    "severe": dict(att=np.radians(2.0), rate=np.radians(5.0), va=1.0, ab=np.radians(2.0), vz=0.50),
}


def steps_from_ms(ms):
    return int(round(ms / 1000.0 / DT))


def obs_noise_sigma(sev, va_scale, vz_scale):
    """15-dim obs-space std for a severity. Channels 11-14 (prev action) get no sensor noise."""
    n = NOISE[sev]
    q = PI4 = np.pi / 4.0
    s = np.zeros(15, dtype=np.float32)
    s[0] = n["att"] / q          # roll error
    s[1] = n["vz"] / vz_scale    # climb error
    s[2] = n["va"] / va_scale    # Va error
    s[3] = n["att"] / q          # roll
    s[4] = n["att"] / q          # pitch
    s[5] = s[6] = s[7] = n["rate"] / 2.0     # body rates
    s[8] = n["ab"] / 0.349       # alpha
    s[9] = n["ab"] / 0.349       # beta
    s[10] = n["vz"] / vz_scale   # vz
    return s


# --- disturbance wrappers (stateful; one instance per run) -------------------
class ActionDelay:
    """Actuator receives the command from k steps ago (neutral buffer before the loop fills)."""
    def __init__(self, fn, k):
        self.fn, self.k, self.buf = fn, k, []

    def __call__(self, obs):
        a = self.fn(obs)
        self.buf.append(a)
        return self.buf.pop(0) if len(self.buf) > self.k else torch.zeros_like(a)


class SensorDelay:
    """Policy sees the obs from k steps ago (holds the initial obs before the buffer fills).
    obs MUST be cloned: the harness hands out wp.to_torch(e._obs), a view of the one warp obs
    buffer that is overwritten in place every step — buffering the view alone makes the delay a
    silent no-op (identical metrics at every severity)."""
    def __init__(self, fn, k):
        self.fn, self.k, self.buf = fn, k, []

    def __call__(self, obs):
        obs = obs.clone()
        self.buf.append(obs)
        old = self.buf.pop(0) if len(self.buf) > self.k else self.buf[0]
        return self.fn(old)


class ObsNoise:
    """Gaussian sensor noise from a generator of its own, seeded per cell by `noise_seed` so the
    draw is a property of the (aircraft, severity) cell rather than of the process. The golden rows
    this family is gated against were drawn from torch's unseeded global CUDA generator, so they
    are a single draw that no re-run reproduces (repeat-to-repeat spread 0.018 deg of bank RMSE),
    which is why TOLERANCE bounds obs_noise instead of demanding equality."""
    def __init__(self, fn, sigma, seed):
        self.fn = fn
        self.sigma = torch.tensor(sigma, device="cuda")
        self.gen = torch.Generator(device="cuda")
        self.gen.manual_seed(int(seed))

    def __call__(self, obs):
        n = torch.randn(obs.shape, generator=self.gen, device=obs.device, dtype=obs.dtype)
        return self.fn(obs + n * self.sigma)


def noise_seed(aircraft, sev, family="obs_noise", policy_seed=0, maneuver=""):
    """Stable 31-bit seed for one stochastic cell. crc32 rather than hash(): Python salts str
    hashes per process, which would make the draw a property of the process again. The defaults
    give the archive's per-(aircraft, severity) sensor-noise seed."""
    key = f"{aircraft}/{family}/{sev}" + (f"/s{policy_seed}/{maneuver}" if maneuver else "")
    return zlib.crc32(key.encode()) & 0x7FFFFFFF


PI4 = np.pi / 4.0


def calibrate_turb(aircraft):
    """k = sigma_u / W20 from the MIL-F-8785C low-altitude intensity relation the Dryden filter
    itself uses (falcons/sim/warp/dryden.py): sigma_w = 0.1 W20 and
    sigma_u = sigma_w / (0.177 + 0.000823 h_ft)^0.4, evaluated at the maneuvers' spawn altitude.
    W20 = Iu * Va / k then sets the SPECIFICATION intensity sigma_u / Va = Iu.

    An earlier version measured k as the standard deviation of a 5.5 s gust record. At 60 m the
    longitudinal correlation time L_u / Va is about 8 s, so a record that short (started from a
    zero filter state) under-reads sigma_u by about 3x, and every "Iu" it set was in fact about
    three times larger than its label."""
    e = make_env(aircraft, 10)
    h_ft = float(e.spawn_alt) / 0.3048
    return 0.1 / (0.177 + 0.000823 * h_ft) ** 0.4, float(e.base_vel)


METHOD = "PPO"      # the one executor; the name the table prints


def controller(plane, ckpt_dir=CKPT_DIR, seed=0):
    """The study's one executor: the PPO attitude policy at `seed`. A second executor is not an
    option here either -- the shipped checkpoint set carries no SAC+CAPS weights."""
    from falcons.controllers.policies import load_policy
    return load_policy("ppo", "attitude", plane, seed, ckpt_dir=ckpt_dir).act


def _hand_over(ctrl):
    """fly() builds the rollout env itself and asks for the controller only afterwards, so a
    controller is passed as a factory f(env, phi_s, hdot_s, va_s). The disturbance wrappers read
    nothing from the env, so the factory just hands the already-built wrapper back; it is called
    once per rollout, which is exactly the lifetime the delay buffers assume."""
    return lambda env, phi_s, hdot_s, va_s, draw=0: ctrl


def eval_cell(aircraft, fn, dist, sev, seed=0, n_real=N_REAL, maneuvers=MANEUVERS,
              turb_k=None, va=None, va_scale=None, vz_scale=None):
    """Every run of one (policy seed, disturbance, severity): one per maneuver for a
    deterministic family, n_real per maneuver for a stochastic one. Returns run records."""
    out = []
    for mv in maneuvers:
        kw, n = {}, (n_real if dist in STOCHASTIC else 1)
        if dist == "nominal":
            ctrl = fn
        elif dist == "action_delay":
            ctrl = ActionDelay(fn, steps_from_ms(ACTION_DELAY_MS[sev]))
        elif dist == "sensor_delay":
            ctrl = SensorDelay(fn, steps_from_ms(SENSOR_DELAY_MS[sev]))
        elif dist == "obs_noise":
            ctrl = ObsNoise(fn, obs_noise_sigma(sev, va_scale, vz_scale),
                            noise_seed(aircraft, sev, "obs_noise", seed, mv))
        elif dist == "wind":
            ctrl = fn
            kw = dict(turbulence="light", turb_W20=WIND_IU[sev] * va / turb_k,
                      turb_seed=noise_seed(aircraft, sev, "wind", seed, mv))
        for k, r in enumerate(fly_batch(aircraft, mv, _hand_over(ctrl), n, **kw)):
            out.append(dict(aircraft=aircraft, method=METHOD, seed=seed, disturbance=dist,
                            severity=sev, maneuver=mv, realization=k,
                            survived=int(r["survival"] == 1.0), phi_rmse=r["phi_rmse"],
                            hdot_rmse=r["hdot_rmse"], va_rmse=r["va_rmse"]))
    return out


def summarize(runs):
    """One table row from the runs of a cell, pooled over seeds, maneuvers and realizations."""
    from falcons.benchmark.stats import rate, summary
    r0 = runs[0]
    g = np.array([r["seed"] for r in runs])
    surv = np.array([r["survived"] for r in runs], dtype=bool)
    s = rate(surv, g)
    row = dict(aircraft=r0["aircraft"], method=r0["method"], disturbance=r0["disturbance"],
               severity=r0["severity"], n_seeds=len(set(g.tolist())),
               n_real=max(r["realization"] for r in runs) + 1, n=s["n"], n_survive=s["k"],
               n_dropped=s["n"] - s["k"], survival=s["rate"], survival_lo=s["lo"],
               survival_hi=s["hi"])
    for k in ("phi_rmse", "hdot_rmse", "va_rmse"):
        x = summary([r[k] for r in runs if r["survived"]], g[surv])
        row.update({k: x["iqm"], f"{k}_lo": x["lo"], f"{k}_hi": x["hi"]})
    return row


def run(planes, ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR, n_real=N_REAL, seeds=None,
        maneuvers=MANEUVERS, families=("action_delay", "sensor_delay", "obs_noise", "wind")):
    """One row per (aircraft, disturbance, severity): the nominal cell, then the families at three
    severities each. `n_real`, `seeds` (default: every PPO attitude checkpoint), `maneuvers` and
    `families` are the knobs a fast test turns down."""
    import warp as wp
    from falcons.controllers.policies import seeds as ckpt_seeds
    wp.init()
    os.makedirs(results_dir, exist_ok=True)
    rows, runs = [], []
    for ac in planes:
        turb_k, va = calibrate_turb(ac)
        va_scale = vz_scale = max(0.5 * va, 5.0)
        ss = list(seeds) if seeds is not None else ckpt_seeds("ppo", "attitude", ac, ckpt_dir)
        print(f"\n=== {ac}  (Va={va:.1f}, turb k=sigma_u/W20={turb_k:.4f}, seeds {ss}, "
              f"n_real {n_real}) ===", flush=True)
        cells = [("nominal", "-")] + [(d, s) for d in families for s in SEVERITIES]
        got = {c: [] for c in cells}
        for seed in ss:
            fn = controller(ac, ckpt_dir, seed)
            for dist, sev in cells:
                got[(dist, sev)] += eval_cell(ac, fn, dist, sev, seed, n_real, maneuvers,
                                              turb_k, va, va_scale, vz_scale)
        for c in cells:
            row = summarize(got[c])
            rows.append(row); runs += got[c]
            pr = f"{row['phi_rmse']:.2f} [{row['phi_rmse_lo']:.2f}, {row['phi_rmse_hi']:.2f}]" \
                if row["phi_rmse"] == row["phi_rmse"] else "crash"
            print(f"  {METHOD:5s} {c[0]:13s} {c[1]:7s} surv {row['n_survive']}/{row['n']}  "
                  f"phi IQM {pr}", flush=True)
    _write(runs, Path(results_dir) / "robustness_runs.csv", RUN_COLS)
    return _write(rows, Path(results_dir) / "robustness.csv", ROW_COLS)


RUN_COLS = ["aircraft", "method", "seed", "disturbance", "severity", "maneuver", "realization",
            "survived", "phi_rmse", "hdot_rmse", "va_rmse"]
ROW_COLS = ["aircraft", "method", "disturbance", "severity", "n_seeds", "n_real", "n",
            "n_survive", "n_dropped", "survival", "survival_lo", "survival_hi",
            "phi_rmse", "phi_rmse_lo", "phi_rmse_hi", "hdot_rmse", "hdot_rmse_lo", "hdot_rmse_hi",
            "va_rmse", "va_rmse_lo", "va_rmse_hi"]


def _write(rows, out, cols):
    def fmt(v):
        if v is None or (isinstance(v, float) and v != v):
            return ""
        return f"{v:.4f}" if isinstance(v, float) else v
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in cols})
    print(f"wrote {out}")
    return out


# the four-panel per-airframe figure is not here: Task 8 ports it to figures.py.
