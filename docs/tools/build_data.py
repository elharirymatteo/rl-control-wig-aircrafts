"""Build the JSON the project page reads, from the paper's tables and the shipped flight traces.

    .venv/bin/python docs/tools/build_data.py

Writes docs/data/results.json (benchmark tables, ground effect, robustness, learning curves,
airframes) and docs/data/maneuvers.json (decimated 3D flight traces for the replay viewer).

Benchmark numbers come from tests/golden/, which holds the published tables (PROVENANCE.md);
the flight traces come from results/traces/, the shipped re-flown run. For each
(airframe, maneuver, method) cell the viewer replays the median run by bank RMSE: of three
training seeds for the learned methods, of five sampling draws for MPPI, and the single
deterministic run of the LQR. A run that left the envelope ranks last.
"""
import csv
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "tests" / "golden"
RES = ROOT / "results"
OUT = ROOT / "docs" / "data"

PLANES = ["Airship_V7", "Volantex_Ranger"]
ALL_PLANES = ["Airship_V7", "Airship_A0S", "Volantex_Ranger", "Navion", "Cirrus_SR22"]
MANEUVERS = ["circle", "figure8", "helix", "sturn"]
METHODS = ["lqr", "mppi", "ppo", "sac", "td3"]
REPLAY_HZ = 10


def num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else f


def read_csv(path, keep_str=()):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (v if k in keep_str else num(v)) for k, v in r.items()})
    return rows


def curves():
    out = {}
    for f in sorted((RES / "curves").glob("*.csv")):
        algo, task, rest = f.stem.split("_", 2)
        plane, seed = rest.rsplit("_", 1)
        rows = read_csv(f)
        out.setdefault(task, {}).setdefault(plane, {}).setdefault(algo, []).append(
            dict(seed=int(seed[1:]), steps=[r["env_steps"] for r in rows],
                 rob=[r["rob"] for r in rows],
                 rmse=[r.get("rmse", r.get("phi_rmse")) for r in rows]))
    return out


def airframes():
    out = {}
    for p in ALL_PLANES:
        d = json.load(open(ROOT / "src/falcons/aircraft/data" / p / f"{p}.json"))
        vp = d["vehicle_params"]
        motors = vp["actuator_system"]["motors"]
        m0 = next(iter(motors.values()))
        out[p] = dict(mass=vp["mass"], span=vp["wing"]["span"], area=vp["wing"]["area"],
                      motors=len(motors), power=m0["type"], tau=m0.get("T"),
                      max_thrust=sum(m.get("max_thrust", 0) for m in motors.values()))
    return out


def results():
    alt = read_csv(GOLD / "altitude.csv", keep_str=("aircraft", "method"))
    man = read_csv(GOLD / "maneuvers.csv", keep_str=("aircraft", "maneuver", "method"))
    rob = [r for r in read_csv(GOLD / "robustness.csv",
                               keep_str=("aircraft", "method", "disturbance", "severity"))
           if r["method"] == "PPO"]
    ge_trim = read_csv(GOLD / "ge_trim.csv", keep_str=("aircraft",))
    ge_cl = read_csv(GOLD / "ge_energy.csv", keep_str=("aircraft",))
    return dict(altitude=alt, maneuvers=man, robustness=rob, ge_trim=ge_trim, ge_closed=ge_cl,
                curves=curves(), airframes=airframes())


def trace_file(plane, man, method, suffix=""):
    return RES / "traces" / f"{plane}_{man}_{method}{suffix}.pkl"


def candidates(plane, man, method):
    if method == "mppi":
        runs = pickle.load(open(trace_file(plane, man, "mppi", "_r5"), "rb"))
        return [(f"draw {i + 1} of {len(runs)}", r) for i, r in enumerate(runs)]
    if method == "lqr":
        return [("deterministic", pickle.load(open(trace_file(plane, man, "lqr"), "rb"))[0])]
    out = []
    for s, suf in enumerate(("", "_s1", "_s2")):
        out.append((f"seed {s} of 3", pickle.load(open(trace_file(plane, man, method, suf), "rb"))[0]))
    return out


def median_run(cands):
    key = lambda c: c[1]["phi_rmse"] if c[1]["survival"] >= 1.0 and c[1]["phi_rmse"] is not None \
        else np.inf
    ranked = sorted(cands, key=key)
    return ranked[(len(ranked) - 1) // 2] if len(ranked) > 1 else ranked[0]


def pack_trace(run):
    tr = run["trace"]
    t = np.asarray(tr["t"], dtype=float)
    dt = float(np.median(np.diff(t)))
    k = max(1, int(round(1.0 / (REPLAY_HZ * dt))))
    sl = slice(None, None, k)
    pos = np.asarray(tr["pos"], dtype=float)[sl]
    r1 = lambda a, nd: np.round(np.asarray(a, dtype=float)[sl], nd).tolist()
    return dict(
        n=int(len(t[sl])),
        north=np.round(pos[:, 0], 1).tolist(), east=np.round(pos[:, 1], 1).tolist(),
        alt=np.round(-pos[:, 2], 2).tolist(),
        roll=r1(tr["roll"], 2), phi_cmd=r1(tr["phi_cmd"], 2), yaw=r1(tr["yaw"], 3),
        climb=r1(tr["climb"], 3), hdot_cmd=r1(tr["hdot_cmd"], 3),
        va=r1(tr["Va"], 2), va_cmd=r1(tr["va_cmd"], 2),
    )


def maneuvers():
    out = dict(hz=REPLAY_HZ, cells={})
    for plane in PLANES:
        for man in MANEUVERS:
            cell = {}
            for m in METHODS:
                label, run = median_run(candidates(plane, man, m))
                cell[m] = dict(run=label, survived=bool(run["survival"] >= 1.0),
                               t_fail=run["t_fail"], phi_rmse=run["phi_rmse"],
                               descriptors={k: float(v) for k, v in run["descriptors"].items()},
                               **pack_trace(run))
            out["cells"][f"{plane}/{man}"] = cell
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("results.json", results()), ("maneuvers.json", maneuvers())):
        p = OUT / name
        p.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False))
        print(f"wrote {p.relative_to(ROOT)}  {p.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
