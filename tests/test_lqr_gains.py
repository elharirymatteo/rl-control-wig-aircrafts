"""The shipped LQR gains are what scripts/derive_lqr_gains.py produces on the current CPU plant,
and every one of them stabilizes the linearized closed loop it was designed for."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import derive_lqr_gains as D  # noqa: E402
from test_controllers import LQR_PENDING  # noqa: E402

pytestmark = pytest.mark.skip(reason=LQR_PENDING)

CASES = [(p, v) for v in D.RECIPES for p in D.PLANES if p in D.RECIPES[v]]


@pytest.fixture(scope="module")
def plants():
    return {}


@pytest.mark.slow
@pytest.mark.parametrize("plane,variant", CASES)
def test_shipped_gains_are_the_derived_gains(plants, plane, variant):
    if plane not in plants:
        plants[plane] = D.plant(plane)
    env, A, B, rb0 = plants[plane]
    K, eig = D.solve(env, A, B, rb0, **D.RECIPES[variant][plane], tied=variant in D.TIED)
    stored = np.loadtxt(D.AircraftConfig(plane).dir / D.OUTPUT[variant].format(p=plane),
                        delimiter=",")
    assert np.linalg.norm(K - stored) / np.linalg.norm(stored) < 1e-6
    ev = eig[np.abs(eig) > 1e-9]                       # drop the removed (structural) modes
    assert ev.real.max() < 0.0


def test_attitude_gains_drive_every_motor_alike():
    """The attitude executor has one throttle channel: every motor row of its gains is the same."""
    for v in D.TIED:
        for plane in D.RECIPES[v]:
            n_mo = D.build_env(plane).n_motors
            K = np.loadtxt(D.AircraftConfig(plane).dir / D.OUTPUT[v].format(p=plane), delimiter=",")
            assert np.allclose(K[:n_mo], K[0]), (plane, v)
