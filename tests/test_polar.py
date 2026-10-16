"""The induced-drag factor k that the ground-effect correction scales (falcons/aircraft/polar.py).

The JSON carries k so the plants need no fit at load time; these tests re-derive it from the
polynomial so the stored value cannot drift from the table it came from, and check that the
correction leaves parasitic drag alone in every backend."""
import json

import numpy as np
import pytest

from falcons.aircraft.config import PLANES
from falcons.aircraft.polar import DATA, FIT_ALPHA_DEG, fit, induced_k, longitudinal, physical


def _wing(plane):
    return json.load(open(DATA / plane / f"{plane}.json"))["vehicle_params"]["wing"]


@pytest.mark.parametrize("plane", PLANES)
def test_stored_k_matches_the_fit(plane):
    k, src = induced_k(plane, PLANES)
    w = _wing(plane)
    assert w["induced_drag_k"] == pytest.approx(k, rel=1e-4)     # stored to 6 decimals
    assert w["induced_drag_source"] == src


@pytest.mark.parametrize("plane", PLANES)
def test_parasitic_drag_is_non_negative(plane):
    """C_D - k C_L^2 is the drag the correction leaves untouched; it must be a drag."""
    k = _wing(plane)["induced_drag_k"]
    a = np.arange(FIT_ALPHA_DEG[0], FIT_ALPHA_DEG[1] + 1e-9, 0.5)
    cl, cd = longitudinal(plane, a)
    assert np.min(cd - k * cl ** 2) >= 0.0


def test_fitted_polars_are_parabolic():
    """Every airframe that keeps its own fit is well described by C_D,par + k C_L^2."""
    for p in PLANES:
        if physical(p):
            assert fit(p)[2] > 0.99, p


@pytest.mark.parametrize("plane", PLANES)
def test_aspect_ratio_is_span_squared_over_area(plane):
    w = _wing(plane)
    assert w["aspect_ratio"] == pytest.approx(w["span"] ** 2 / w["area"], abs=1e-3)


def test_cpu_plant_scales_induced_drag_only():
    """At one height the CPU plant's C_D equals C_D,oge + k C_L,oge^2 (mu_d mu_l^2 - 1)."""
    import pandas as pd
    from falcons.benchmark.ground_effect import induced_factor
    from falcons.sim.cpu.physics.aerodynamics import PolynomialAerodynamics
    plane = "Volantex_Ranger"
    raw = json.load(open(DATA / plane / f"{plane}.json"))
    w = raw["vehicle_params"]["wing"]
    model = PolynomialAerodynamics(raw["aero_params"], raw["vehicle_params"],
                                   raw["environment_params"],
                                   pd.read_csv(DATA / plane / f"{plane}_poly.csv"))
    h = 0.5 * w["span"]
    state = {"position": np.array([0.0, 0.0, -h])}
    out = model.get_coefficients_with_ground_effect(np.radians(4.0), 0.0, np.zeros(3), state)
    f = induced_factor((h + w["cg_offset_vector"][2]) / w["span"], w["taper_ratio"], w["aspect_ratio"])
    expect = out["CD_oge"] + w["induced_drag_k"] * out["CL_oge"] ** 2 * (f - 1.0)
    assert out["CD"] == pytest.approx(expect, rel=1e-6)
    assert out["CD"] > out["CD_oge"] - w["induced_drag_k"] * out["CL_oge"] ** 2   # parasitic kept
