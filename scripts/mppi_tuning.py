"""The MPPI altitude tuning sweep recorded in docs/mppi_tuning.md -- the whole tuning budget the
paper reports for the planner, rerunnable as is.

Each configuration (preview horizon, vertical-speed weight, control-cost gamma) flies the
altitude-acquisition protocol of falcons.benchmark.altitude on a reduced grid: commanded altitudes
TARGETS, spawn offsets drawn exactly as the benchmark draws them (the first N_PER_SIGN of each
sign), EPISODE_STEPS long, sampler seed 0. One child process per episode, because the warp
reference bakes the commanded altitude at import (see falcons/controllers/mppi/reference.py).

    PYTHONPATH=src .venv/bin/python scripts/mppi_tuning.py            # the sweep -> CSV
    PYTHONPATH=src .venv/bin/python scripts/mppi_tuning.py --gamma     # control-cost check
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from falcons.aircraft.config import AircraftConfig
from falcons.benchmark import altitude as B
from falcons.envs.configs import PLANE_CONFIGS

PLANES = ["Airship_V7", "Volantex_Ranger"]
TARGETS = [20.0, 60.0, 100.0]
N_PER_SIGN = 2
EPISODE_STEPS = 4000          # 40 s: an 18 m ramp at 2 m/s plus 31 s of hold
HORIZONS = [100, 200, 300]    # 1, 2, 3 s previews
W_VZ = [0.0, 1.0, 3.0]
GAMMA = 0.03
WORKERS = 3
OUT = Path("results/_tuning/mppi_altitude_grid.csv")


def episodes(plane):
    """(target, h0) pairs: the benchmark's own offset draws (falcons.benchmark.altitude)."""
    out = []
    for T in TARGETS:
        rng = np.random.default_rng(0)
        for sgn in (-1.0, +1.0):
            h0s = [float(T + sgn * rng.uniform(B.OFF_LO, B.OFF_HI)) for _ in range(B.N_CL)]
            out += [(T, h) for h in h0s[:N_PER_SIGN]]
    return out


def child(plane, target, h0, horizon, w_vz, gamma):
    from falcons.controllers.mppi.altitude import MPPIAltitude
    scen = {"name": "tune", "trajectory_type": "acquire_ramp", "use_sensor_noise": False}
    d = MPPIAltitude(plane, init_altitude=h0, mppi_seed=0, horizon=horizon,
                     vertical_speed_weight=w_vz, control_cost_gamma=gamma,
                     max_steps=EPISODE_STEPS).run(scen)
    h = -d["positions"][:, 2]
    m = min(len(h), EPISODE_STEPS)
    full = np.full(EPISODE_STEPS, np.nan); full[:m] = h[:m]
    thr = np.full(EPISODE_STEPS, np.nan); thr[:m] = (d["actions"][:m, 0] + 1.0) * 0.5
    met = B.episode_metrics(full, target, h0, thr) if m >= EPISODE_STEPS else None
    print("JSON" + json.dumps(dict(m=met, t_end=m * B.DT)), flush=True)


def one(plane, target, h0, horizon, w_vz, gamma):
    base_vel = float(AircraftConfig(plane).load()["default_initial_state"]["linear_vel"][0])
    env = dict(os.environ, TRAJ_CONST_ALT=f"{target}", TRAJ_ACQ_H0=f"{h0}",
               TRAJ_REF_VA=f"{base_vel}", TRAJ_ACQ_RATE=f"{PLANE_CONFIGS[plane]['ref_rate']}")
    p = subprocess.run([sys.executable, __file__, "--child", plane, f"{target}", f"{h0}",
                        f"{horizon}", f"{w_vz}", f"{gamma}"], capture_output=True, text=True, env=env)
    line = [l for l in p.stdout.splitlines() if l.startswith("JSON")]
    if not line:
        raise RuntimeError(f"child failed {plane} T={target} h0={h0}:\n{p.stderr[-1500:]}")
    return json.loads(line[-1][4:])


def config(plane, horizon, w_vz, gamma):
    eps = episodes(plane)
    with ThreadPoolExecutor(WORKERS) as ex:
        res = list(ex.map(lambda e: one(plane, e[0], e[1], horizon, w_vz, gamma), eps))
    rows = [r["m"] for r in res]
    surv = [r for r in rows if r is not None]
    acq = [r for r in surv if np.isfinite(r["settling"])]
    f = lambda k: float(np.mean([r[k] for r in surv])) if surv else float("nan")
    return dict(plane=plane, horizon_s=horizon * B.DT, w_vz=w_vz, gamma=gamma, episodes=len(rows),
                survival=len(surv) / len(rows), acquired=len(acq) / len(rows),
                rmse_m=f("rmse"), overshoot_m=f("overshoot"),
                settling_s=float(np.mean([r["settling"] for r in acq])) if acq else float("nan"))


def write(rows):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT.exists()
    with open(OUT, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=6)
    ap.add_argument("--gamma", action="store_true", help="control-cost check at --horizon/--w-vz")
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--w-vz", type=float, default=None)
    a = ap.parse_args()
    if a.child:
        p, T, h0, H, W, G = a.child
        return child(p, float(T), float(h0), int(H), float(W), float(G))
    grid = ([(a.horizon, a.w_vz, g) for g in (0.0, GAMMA)] if a.gamma
            else [(H, W, GAMMA) for H in HORIZONS for W in W_VZ])
    for H, W, G in grid:
        rows = [config(p, H, W, G) for p in PLANES]
        write(rows)
        for r in rows:
            print(f"{r['plane']:16s} H {r['horizon_s']:.0f}s w_vz {r['w_vz']:.1f} gamma {r['gamma']:.2f}"
                  f"  surv {r['survival']:.2f} acq {r['acquired']:.2f} rmse {r['rmse_m']:.2f}"
                  f" over {r['overshoot_m']:.2f} t_s {r['settling_s']:.1f}", flush=True)


if __name__ == "__main__":
    main()
