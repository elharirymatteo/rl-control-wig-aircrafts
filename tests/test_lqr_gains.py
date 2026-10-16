"""The shipped LQR gains are what scripts/derive_lqr_gains.py produces on the current CPU plant,
and every one of them stabilizes the linearized closed loop it was designed for."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import derive_lqr_gains as D  # noqa: E402

CASES = [(p, v) for v in ("hold", "acquisition") for p in D.PLANES if p in D.RECIPES[v]]


@pytest.fixture(scope="module")
def plants():
    return {}


@pytest.mark.slow
@pytest.mark.parametrize("plane,variant", CASES)
def test_shipped_gains_are_the_derived_gains(plants, plane, variant):
    if plane not in plants:
        plants[plane] = D.plant(plane)
    env, A, B, rb0 = plants[plane]
    K, eig = D.solve(env, A, B, rb0, **D.RECIPES[variant][plane])
    stored = np.loadtxt(D.AircraftConfig(plane).dir / D.OUTPUT[variant].format(p=plane),
                        delimiter=",")
    assert np.linalg.norm(K - stored) / np.linalg.norm(stored) < 1e-6
    ev = eig[np.abs(eig) > 1e-9]                       # drop the removed (structural) modes
    assert ev.real.max() < 0.0
