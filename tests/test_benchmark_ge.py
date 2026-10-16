from pathlib import Path

import pytest

from falcons.aircraft.config import PLANES
from falcons.benchmark import TOLERANCE
from falcons.benchmark.diff import compare_csv
from falcons.benchmark.ground_effect import RATIOS, geometry, induced_factor, run_energy, run_trim

GOLD = Path(__file__).parent / "golden"


def test_induced_factor_is_lifting_line_shape():
    """mu_d mu_l^2 multiplies the induced drag: well below one deep in ground effect, rising
    monotonically to one out of it. `geometry` returns (span, TR, AR, cg_z)."""
    span, TR, AR, cg_z = geometry("Airship_V7")
    f = [induced_factor(r + cg_z / span, TR, AR) for r in RATIOS]
    assert f[0] < 0.9                                      # h/b = 0.25: a large induced saving
    assert all(a < b for a, b in zip(f, f[1:]))             # monotone toward one with h/b
    assert abs(f[-1] - 1.0) < 1e-4                         # h/b = 6: out of ground effect


@pytest.mark.cuda
def test_trim_table_matches_golden(tmp_path):
    out = run_trim(PLANES, results_dir=tmp_path)
    assert compare_csv(out, GOLD / "ge_trim.csv", TOLERANCE["ge_trim.csv"]) == []


@pytest.mark.slow
@pytest.mark.cuda
def test_energy_table_matches_golden(tmp_path, gate_copy):
    out = run_energy(PLANES, results_dir=tmp_path)
    assert compare_csv(gate_copy(out), GOLD / "ge_energy.csv", TOLERANCE["ge_energy.csv"]) == []
