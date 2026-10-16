"""The LQR attitude executor's bank envelope: constant-bank turns at increasing bank angle.

The gains are linearized about LEVEL trim (scripts/derive_lqr_gains.py), so the executor is only
expected to hold banks near zero. This sweep measures how far: each commanded bank is a
"bank<deg>" stream (falcons/benchmark/streams.py), flown with `maneuvers.fly` on the same plant and
tolerance as the maneuver table. A bank is inside the envelope when the run survives and keeps the
bank error within the +-8.6 deg tolerance for at least 90 % of the scored time.

    PYTHONPATH=src python scripts/lqr_envelope.py        # -> results/lqr_envelope.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np

from falcons.benchmark.maneuvers import controllers, fly, time_in_band
from falcons.paths import RESULTS_DIR

BANKS = [5, 10, 15, 20, 25, 30, 35, 40, 45]
PLANES = ["Airship_V7", "Volantex_Ranger"]
TOL_DEG = 8.6
IN_BAND = 90.0


def main(planes=PLANES, banks=BANKS, out=RESULTS_DIR / "lqr_envelope.csv"):
    import warp as wp
    wp.init()
    rows = []
    for plane in planes:
        (_, factory), = controllers(plane)["LQR"]
        for b in banks:
            r = fly(plane, f"bank{b}", factory)
            tib = 100.0 * time_in_band(r["trace"], TOL_DEG) if r["survival"] else 0.0
            ok = bool(r["survival"] and tib >= IN_BAND)
            rows.append(dict(aircraft=plane, bank_deg=b, survival=r["survival"],
                             phi_rmse_deg=r["phi_rmse"], time_in_band_pct=tib, in_envelope=int(ok)))
            print(f"{plane:16s} bank {b:2d} deg  surv {r['survival']:.0f}  "
                  f"rmse {r['phi_rmse'] if r['phi_rmse'] is not None else float('nan'):6.2f}  "
                  f"in-band {tib:5.1f}%  {'IN' if ok else 'out'}", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(out=Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR / "lqr_envelope.csv")
