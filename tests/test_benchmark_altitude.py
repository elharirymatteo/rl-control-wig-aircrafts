import csv
from pathlib import Path

import numpy as np
import pytest

from falcons.benchmark import TOLERANCE
from falcons.benchmark.diff import compare_csv
from falcons.benchmark.altitude import episode_metrics, aggregate, run_learned, run

GOLD = Path(__file__).parent / "golden"


def test_episode_metrics_band_is_one_metre_and_overshoot_is_past_target():
    h = np.full(6000, 40.0); h[:100] = np.linspace(30, 41.5, 100); thr = np.full(6000, 0.4)
    m = episode_metrics(h, 40.0, 30.0, thr)
    assert m["overshoot"] == pytest.approx(1.5, abs=1e-6)          # excursion past the target, not |e|
    assert np.isfinite(m["settling"]) and abs(m["rmse"]) < 1e-9    # settled = finite settling time


@pytest.mark.cuda
def test_learned_rows_match_golden_v7_ppo():
    a = aggregate(run_learned("Airship_V7", "ppo", 0))
    with open(GOLD / "altitude.csv") as f:
        gold = next(r for r in csv.DictReader(f)
                    if r["aircraft"] == "Airship_V7" and r["method"] == "PPO")
    # golden row is the seed mean; seed 0 alone must equal it where std is 0 (survival, acquired).
    # The Ranger PPO row loses a seed's episodes, so its mean says nothing about seed 0
    assert float(gold["survival_std"]) == float(gold["acquired_std"]) == 0.0
    assert a["survival"] == float(gold["survival"])
    assert a["survival"] * a["settled_frac"] == pytest.approx(float(gold["acquired"]))


@pytest.mark.slow
@pytest.mark.cuda
def test_full_altitude_table_matches_golden(tmp_path, gate_copy):
    """The golden is the shipped run, whose three MPPI sweeps ran at sampler seeds 0, 1, 2 (the
    `falcons benchmark altitude` default); the MPPI rows are draws and are gated by their spread.

    Three hours of flying produce this CSV; gate_copy parks it under results/_gate/ before the
    comparison, so a failure can be read off the table rather than re-flown."""
    out = run(["Airship_V7", "Volantex_Ranger"], results_dir=tmp_path, mppi_seeds=(0, 1, 2))
    assert compare_csv(gate_copy(out), GOLD / "altitude.csv", TOLERANCE["altitude.csv"]) == []
