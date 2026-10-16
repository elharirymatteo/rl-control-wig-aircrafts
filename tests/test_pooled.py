"""Pooled cell statistics of the three protocols, on synthetic runs (no GPU)."""
import csv

import numpy as np
import pytest

from falcons.benchmark.altitude import POOLED_COLS, pooled as alt_pooled
from falcons.benchmark.robustness import ROW_COLS, summarize
from falcons.benchmark.tables import protocol_table, robustness_table


def _episodes(n_seeds=5, n=40, dead=3):
    rng = np.random.default_rng(0)
    eps = []
    for s in range(n_seeds):
        for i in range(n):
            alive = not (s == 0 and i < dead)
            eps.append(dict(group=s, survived=int(alive), acquired=int(alive and i % 10 != 0),
                            rmse=float(0.3 + 0.01 * rng.standard_normal()) if alive else None,
                            settling=8.0 if alive and i % 10 != 0 else np.inf,
                            overshoot=0.8 if alive else None, energy=0.2 if alive else None))
    return eps


def test_altitude_pooled_counts_and_iqm():
    p = alt_pooled(_episodes())
    assert set(POOLED_COLS) == set(p)
    assert p["n_total"] == 200 and p["n_dropped"] == 3 and p["n_survived"] == 197
    assert p["rmse_lo"] <= p["rmse_iqm"] <= p["rmse_hi"] and abs(p["rmse_iqm"] - 0.3) < 0.01
    assert p["settling_iqm"] == pytest.approx(8.0)


def test_robustness_summarize_pools_seeds_and_realizations():
    runs = [dict(aircraft="Airship_V7", method="PPO", seed=s, disturbance="wind", severity="mild",
                 maneuver="helix", realization=k, survived=int(k != 0),
                 phi_rmse=None if k == 0 else 2.0 + 0.1 * s, hdot_rmse=0.2, va_rmse=1.0)
            for s in range(5) for k in range(10)]
    row = summarize(runs)
    assert set(row) <= set(ROW_COLS) | {"survival"}
    assert row["n"] == 50 and row["n_survive"] == 45 and row["n_dropped"] == 5
    assert row["n_seeds"] == 5 and row["n_real"] == 10
    assert row["phi_rmse_lo"] <= row["phi_rmse"] <= row["phi_rmse_hi"]


def test_tables_switch_to_the_pooled_format(tmp_path):
    alt = dict(aircraft="Airship_V7", method="PPO", n="200", n_seeds="5", survival="0.985",
               acquired="0.9", rmse="0.3", settling="8", overshoot="0.8")
    alt.update({k: f"{v}" for k, v in alt_pooled(_episodes()).items()})
    tex = protocol_table([alt])
    assert "197/200" in tex and "[" in tex and "interquartile" in tex
    rob = dict(aircraft="Airship_V7", method="PPO", disturbance="nominal", severity="-",
               n_real="1", n="20", n_survive="20", phi_rmse="2.0", phi_rmse_lo="1.9",
               phi_rmse_hi="2.1", survival="1.0")
    assert "20/20" in robustness_table([rob]) and "2.0 [1.9, 2.1]" in robustness_table([rob])


def test_maneuver_pooled_counts_departures_and_groups():
    from falcons.benchmark.maneuvers import pooled
    tr = dict(roll=np.zeros(200), phi_cmd=np.ones(200))
    ok = lambda s, e: dict(survival=1.0, group=s, trace=tr, phi=dict(rmse=e),
                          climb=dict(rmse=0.2), airspeed=dict(rmse=1.0))
    reps = [ok(s, 1.0 + 0.1 * s) for s in range(4)] + [dict(survival=0.0, group=4)]
    p = pooled(reps, tol_deg=8.6)
    assert p["n_departed"] == 1 and p["survival_lo"] < 0.8 < p["survival_hi"]
    assert p["phi_rmse_lo"] <= p["phi_rmse_iqm"] <= p["phi_rmse_hi"]
    assert p["time_in_band_iqm"] == pytest.approx(100.0)
