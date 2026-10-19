"""Fly one representative altitude-acquisition episode per controller for the project page.

Same protocol as falcons.benchmark.altitude (same env, same rate-limited acquisition reference,
same checkpoints), but instead of scoring 200 episodes it records the full altitude and throttle
trace of one climb and one descent per (airframe, controller), so the page can replay them.

    .venv/bin/python docs/tools/fly_altitude.py            # -> docs/data/altitude_replay.json
    .venv/bin/python docs/tools/inline_data.py             # copy it into docs/index.html

Classical episodes run as child processes, exactly like the benchmark, because the warp reference
bakes the commanded altitude at build time.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data" / "altitude_replay.json"
PLANES = ["Airship_V7", "Volantex_Ranger"]
TARGET = 60.0
OFFSET = 15.0
STEPS = 6000           # 60 s at dt = 0.01, as in the benchmark
DECIM = 5              # store at 20 Hz


def fly_learned(plane, algo, seed=0):
    import torch
    import warp as wp
    from falcons.benchmark.altitude import STEPS as BSTEPS
    from falcons.controllers.policies import load_policy
    from falcons.envs.altitude import AltitudeEnv
    from falcons.envs.configs import altitude_env_cfg

    assert BSTEPS == STEPS
    alt0 = np.array([TARGET - OFFSET, TARGET + OFFSET], dtype=np.float32)
    tgt = np.full(2, TARGET, dtype=np.float32)
    n = 2
    env = AltitudeEnv(n, "cuda", altitude_env_cfg(plane, 0.0, horizon=STEPS + 1))
    env.reset()
    pos = np.zeros((n, 3), dtype=np.float32); pos[:, 2] = -alt0
    vel = np.zeros((n, 3), dtype=np.float32); vel[:, 0] = env.base_vel
    wp.copy(env._true_target, wp.array(tgt, device="cuda"))
    wp.copy(env._target, wp.array(alt0, device="cuda"))
    wp.copy(env.model._state["position"], wp.array(pos, dtype=wp.vec3f, device="cuda"))
    wp.copy(env.model._state["linear_vel"], wp.array(vel, dtype=wp.vec3f, device="cuda"))
    env._ierr.zero_(); env._refrate.zero_(); env._obs_launch()

    policy = load_policy(algo, "altitude", plane, seed).act
    obs = wp.to_torch(env._obs)
    h = np.full((STEPS, n), np.nan)
    x = np.full((STEPS, n), np.nan)
    thr = np.full((STEPS, n), np.nan)
    live = np.ones(n, dtype=bool)
    with torch.no_grad():
        for t in range(STEPS):
            obs, _, d, _ = env.step(policy(obs).clamp(-1.0, 1.0))
            p = env.model._state["position"].numpy()
            tl = env.model._actuator_states["throttle_left"].numpy()
            h[t, live] = -p[live, 2]; x[t, live] = p[live, 0]; thr[t, live] = tl[live]
            live &= ~d.cpu().numpy().astype(bool)
            if not live.any():
                break
    return [dict(h=h[:, i], x=x[:, i], thr=thr[:, i]) for i in range(n)]


def child(plane, algo, h0, out):
    from falcons.controllers.lqr.altitude import LQRAltitude
    from falcons.controllers.mppi.altitude import MPPIAltitude
    scen = {"name": "acquire", "trajectory_type": "acquire_ramp", "use_sensor_noise": False,
            "use_estimator": False, "wind": None}
    if algo == "lqr":
        d = LQRAltitude(plane, init_altitude=h0).run(scen)
    else:
        d = MPPIAltitude(plane, init_altitude=h0, mppi_seed=0).run(scen)
    m = min(len(d["positions"]), STEPS)
    h = np.full(STEPS, np.nan); h[:m] = -d["positions"][:m, 2]
    x = np.full(STEPS, np.nan); x[:m] = d["positions"][:m, 0]
    thr = np.full(STEPS, np.nan); thr[:m] = (d["actions"][:m, 0] + 1.0) * 0.5
    np.savez(out, h=h, x=x, thr=thr)


def fly_classical(plane, algo):
    from falcons.aircraft.config import AircraftConfig
    from falcons.envs.configs import TRIM_VA
    gains = AircraftConfig(plane).acquisition_gains_path
    runs = []
    for h0 in (TARGET - OFFSET, TARGET + OFFSET):
        out = HERE / f"_tmp_{plane}_{algo}_{int(h0)}.npz"
        env = dict(os.environ, TRAJ_CONST_ALT=f"{TARGET}", TRAJ_ACQ_H0=f"{h0}",
                   TRAJ_REF_VA=f"{TRIM_VA[plane]}", MPPI_DRAW="0")
        if algo == "lqr" and gains is not None:
            env["LQR_GAINS_PATH"] = str(gains)
        p = subprocess.run([sys.executable, __file__, "--child", plane, algo, f"{h0}", str(out)],
                           capture_output=True, text=True, env=env)
        if p.returncode != 0:
            raise RuntimeError(f"{plane} {algo} h0={h0}\n{p.stderr[-2000:]}")
        d = np.load(out)
        runs.append(dict(h=d["h"], x=d["x"], thr=d["thr"]))
        out.unlink()
    return runs


def pack(run):
    def q(a, nd):
        a = a[::DECIM]
        return [None if not np.isfinite(v) else round(float(v), nd) for v in a]
    alive = np.isfinite(run["h"])
    t_end = float(alive.sum()) * 0.01
    return dict(h=q(run["h"], 3), x=q(run["x"], 1), thr=q(run["thr"], 3),
                t_end=round(t_end, 2), survived=bool(alive.all()))


def main():
    from falcons.envs.configs import TRIM_VA, PLANE_CONFIGS
    res = dict(target=TARGET, offset=OFFSET, dt=0.01 * DECIM, steps=STEPS // DECIM, planes={})
    for plane in PLANES:
        cell = {}
        for algo in ("ppo", "sac", "td3"):
            print(plane, algo, flush=True)
            r = fly_learned(plane, algo)
            cell[algo] = dict(climb=pack(r[0]), descent=pack(r[1]))
        for algo in ("lqr", "mppi"):
            print(plane, algo, flush=True)
            r = fly_classical(plane, algo)
            cell[algo] = dict(climb=pack(r[0]), descent=pack(r[1]))
        res["planes"][plane] = dict(trim_va=TRIM_VA[plane], ref_rate=PLANE_CONFIGS[plane]["ref_rate"],
                                    runs=cell)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, separators=(",", ":")))
    print("wrote", OUT, OUT.stat().st_size // 1024, "KiB")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5])
    else:
        main()
