"""The JSBSim cross-check table of the paper: Pearson r, RMSE and worst error per channel.

Both simulators fly the same zero-input free flight (controls at zero, throttle shut) from a level
attitude at the airframe's reference airspeed V0 and alpha = 2 deg, for a few seconds, from
h0 = 1, 10 and 100 m. The JSBSim aircraft carries no ground-effect term (gen_jsbsim.py), so the
FALCON-S plant is flown with its correction off, and once with it on at 1 m, where the two are
meant to part company. Both carry the derivative set's rate damping (CMl_p, CMm_q, CMn_r, ...).

    python tools/jsbsim_validate/gen_jsbsim.py --plane Airship_V7       # once per airframe
    python tools/jsbsim_validate/paper_table.py --planes Airship_V7 Volantex_Ranger
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from validate import FT, body_velocity, rollout  # noqa: F401  (FT re-exported for callers)

HERE = Path(__file__).parent
CONDITIONS = [(1.0, True), (1.0, False), (10.0, False), (100.0, False)]
CHANNELS = [("altitude", "h_falcons", "h_jsbsim", "m"),
            ("pitch rate", "q_falcons_degs", "q_jsbsim_degs", "deg/s"),
            ("airspeed", "V_falcons", "V_jsbsim", "m/s")]


def reference_airspeed(plane):
    from falcons.aircraft.config import AircraftConfig  # noqa: F401
    data = Path(__import__("falcons").__file__).parent / "aircraft" / "data" / plane / f"{plane}.json"
    return float(np.linalg.norm(json.load(open(data))["default_initial_state"]["linear_vel"]))


def compare(frame, f_col, j_col):
    a, b = frame[f_col].to_numpy(float), frame[j_col].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
    return r, float(np.sqrt(np.mean(d ** 2))), float(np.abs(d).max())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--planes", nargs="+", default=["Airship_V7", "Volantex_Ranger"])
    p.add_argument("--aircraft", type=Path, default=HERE / "aircraft")
    p.add_argument("--out", type=Path, default=HERE / "out")
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--alpha", type=float, default=2.0, help="initial angle of attack, deg")
    p.add_argument("--jsb-dt", type=float, default=0.001)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for plane in a.planes:
        v0 = reference_airspeed(plane)
        alpha = np.radians(a.alpha)
        uvw = body_velocity(v0, alpha, 0.0)
        euler = np.array([0.0, alpha, 0.0])                  # level flight path: pitch = alpha
        for h0, ge in CONDITIONS:
            fr = rollout(plane, a.aircraft, h0, euler, uvw, np.zeros(3), a.seconds, a.jsb_dt,
                         ground_effect=ge)
            row = {"plane": plane, "h0": h0, "ground_effect": ge, "seconds": float(fr["t"].iloc[-1])}
            for name, fc, jc, unit in CHANNELS:
                r, rmse, mx = compare(fr, fc, jc)
                row.update({f"{name} r": r, f"{name} rmse": rmse, f"{name} max": mx})
            rows.append(row)
            print(f"{plane:16s} h0={h0:5.1f} GE={'on ' if ge else 'off'} t={row['seconds']:.2f}s  " +
                  "  ".join(f"{n}: r={row[f'{n} r']:.4f} rmse={row[f'{n} rmse']:.3g} max={row[f'{n} max']:.3g}"
                            for n, *_ in CHANNELS), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "paper_table.csv", index=False)
    print(f"wrote {a.out / 'paper_table.csv'}")


if __name__ == "__main__":
    main()
