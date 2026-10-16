"""Induced-drag factor of each airframe's drag polar, for the ground-effect correction.

The lifting-line correction (Phillips & Hunsaker 2013) scales the INDUCED drag, C_Di = k C_L^2,
not the total drag: parasitic drag does not depend on the trailing-vortex system the ground
image alters. The airframes ship one polynomial for the total C_D, so k is recovered from it by
fitting the parabolic polar

    C_D(alpha) = C_D,par + k C_L(alpha)^2

at zero deflection and zero sideslip over the attached-flow range FIT_ALPHA_DEG. The fitted k is
stored in each airframe's JSON (`vehicle_params.wing.induced_drag_k`); `tests/test_polar.py`
re-fits and checks the stored value, so the JSON cannot drift from the polynomial it came from.

An airframe whose polynomial does not admit a physical polar (negative parasitic drag anywhere
in the range) falls back to the classical k = 1 / (pi e AR), with e the mean Oswald factor of the
airframes that do. Only the Airship A0S takes that path.

    .venv/bin/python -m falcons.aircraft.polar        # print the fit for every airframe
"""
import csv
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
FIT_ALPHA_DEG = (-2.0, 10.0)        # attached flow, below the 13 deg stall angle of every airframe
FIT_STEP_DEG = 0.5
_EXPONENTS = ("alpha", "beta", "delta_e", "delta_a", "delta_r")


def _poly(plane):
    """Rows of the airframe's polynomial table as (exponents, {coefficient: value})."""
    rows = []
    with open(DATA / plane / f"{plane}_poly.csv") as f:
        for r in csv.DictReader(f):
            rows.append(([float(r[k]) for k in _EXPONENTS],
                         {c: float(r[c]) for c in ("CD", "CL")}))
    return rows


def longitudinal(plane, alpha_deg):
    """C_L and C_D at zero sideslip and zero deflections, alpha in degrees (the table's units)."""
    a = np.asarray(alpha_deg, dtype=float)
    cl, cd = np.zeros_like(a), np.zeros_like(a)
    for e, c in _poly(plane):
        if any(e[1:]):                       # beta or a deflection: zero at this point
            continue
        cl += c["CL"] * a ** e[0]
        cd += c["CD"] * a ** e[0]
    return cl, cd


def fit(plane):
    """(C_D,par, k, R^2, min parasitic drag over the range) of the parabolic polar fit."""
    a = np.arange(FIT_ALPHA_DEG[0], FIT_ALPHA_DEG[1] + 1e-9, FIT_STEP_DEG)
    cl, cd = longitudinal(plane, a)
    A = np.stack([np.ones_like(cl), cl ** 2], axis=1)
    (cd0, k), *_ = np.linalg.lstsq(A, cd, rcond=None)
    r2 = 1.0 - np.sum((cd - A @ [cd0, k]) ** 2) / np.sum((cd - cd.mean()) ** 2)
    return float(cd0), float(k), float(r2), float(np.min(cd - k * cl ** 2))


def physical(plane):
    """A polar is usable when its parasitic part stays non-negative over the whole range."""
    return fit(plane)[3] >= 0.0


def aspect_ratio(plane):
    import json
    w = json.load(open(DATA / plane / f"{plane}.json"))["vehicle_params"]["wing"]
    return w["span"] ** 2 / w["area"]


def oswald(plane):
    """Oswald efficiency implied by a fitted k: e = 1 / (pi AR k)."""
    return 1.0 / (np.pi * aspect_ratio(plane) * fit(plane)[1])


def induced_k(plane, planes):
    """The k to store for `plane`: its polar fit, or the Oswald fallback when that is unphysical."""
    if physical(plane):
        return fit(plane)[1], "polar_fit"
    e = float(np.mean([oswald(p) for p in planes if physical(p)]))
    return 1.0 / (np.pi * e * aspect_ratio(plane)), f"oswald_e{e:.2f}"


if __name__ == "__main__":
    from falcons.aircraft.config import PLANES
    for p in PLANES:
        cd0, k, r2, mn = fit(p)
        kk, src = induced_k(p, PLANES)
        print(f"{p:16s} fit: CD_par {cd0:+.4f}  k {k:.4f}  R2 {r2:.4f}  min CD_par {mn:+.4f}"
              f"  ->  k {kk:.4f} ({src})")
