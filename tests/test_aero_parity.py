"""Every backend must compute the same coefficients.

The aerodynamic model exists four times -- numpy (the CPU reference plant), warp, torch, and
through warp again in the MPPI rollout kernels. Four transcriptions of one equation set is four
chances to drop a term, and nothing else in the suite compares them: `test_parity.py` covers only
the Dryden wind model, and notes there that the angular gusts are not even fed to the aero model.

The CPU implementation is the reference. Warp runs in float32, so it is compared at float32
tolerance; torch is driven in float64 where it can be.

Every airframe with usable data is exercised, not just one. The rate terms non-dimensionalise by
span on p/r and by MAC on q, and with a single airframe a span/MAC mix-up is a constant factor
that agrees across all four backends and so goes unseen. The airframes here differ by 3x in span
(Volantex 1.62 m, Airship_V7 5.07 m) and 4.6x in MAC, which separates the two.
"""
import numpy as np
import pytest

from conftest import ONLINE_PLANES
from falcons.aircraft.config import AircraftConfig
from falcons.aircraft.derivatives import GE_MODELS, ph_mu_d, ph_mu_l, ph_relative
from falcons.aircraft.params import DerivativeAeroParameters
from falcons.sim.aero_contract import AeroInputs
from falcons.sim.cpu.physics.aerodynamics import DerivativeAerodynamics

pytestmark = pytest.mark.skipif(not ONLINE_PLANES,
                                reason="no airframe has usable OpenVSP derivative CSVs")


def build_cpu(ac, ge_enable=True):
    cfg = AircraftConfig(ac).load()
    wing = cfg["vehicle_params"]["wing"]
    return DerivativeAerodynamics(DerivativeAeroParameters.from_config(cfg),
                                  span=wing["span"], mac=wing["mac"], ge_enable=ge_enable)


def geometry(ac):
    wing = AircraftConfig(ac).load()["vehicle_params"]["wing"]
    return float(wing["span"]), float(wing["mac"])


# A spread of conditions: in deep ground effect, mid-sweep, far above the table, at the run point,
# and with every control and rate channel excited so no term can be silently dropped.
#
# Heights are h/b, not metres, because the sweep is measured on an h/b grid running 0.20 -> 2.00.
# A fixed metre height would sit deep in ground effect for Volantex and out of it for Airship_V7,
# so the cases would stop meaning what their names say as soon as the airframe changed.
#
# (alpha, beta, v, elevator, aileron, rudder, p, q, r, h_over_b, label)
RAW_CASES = [
    (None, 0.0, None, None, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, "run point, far out of GE"),
    (0.05, 0.02, 15.0, 0.1, -0.05, 0.03, 0.4, -0.3, 0.2, 0.216, "deep GE, all channels live"),
    (-0.03, -0.04, 22.0, -0.2, 0.1, -0.08, -0.5, 0.6, -0.4, 0.617, "mid-sweep"),
    (0.12, 0.01, 8.0, 0.3, 0.02, 0.01, 0.1, 0.2, 0.05, 2.0, "exactly the table top"),
    (0.02, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0617, "below the V floor and the GE floor"),
]
IDS = [c[-1] for c in RAW_CASES]


def cases_for(ac):
    """The case table resolved against one airframe: heights scaled by its span, and the `None`
    fields of case 0 filled from its own measured operating point."""
    span, _ = geometry(ac)
    p = DerivativeAeroParameters.from_config(AircraftConfig(ac).load())
    out = []
    for alpha, beta, v, elev, ail, rud, pp, q, r, hb, _ in RAW_CASES:
        out.append(AeroInputs(
            alpha=p.alpha_run if alpha is None else alpha, beta=beta,
            v=p.v_ref if v is None else v,
            elevator=p.de_run if elev is None else elev, aileron=ail, rudder=rud,
            p=pp, q=q, r=r, h=hb * span))
    return out


@pytest.mark.parametrize("ac", ONLINE_PLANES)
@pytest.mark.parametrize("ge", [True, False])
@pytest.mark.parametrize("case", range(len(RAW_CASES)), ids=IDS)
def test_cpu_model_is_finite_and_ordered(ac, case, ge):
    """Guards the reference itself: no NaN from the V floor or the table clamps."""
    c = build_cpu(ac, ge_enable=ge).coefficients(cases_for(ac)[case])
    assert np.all(np.isfinite(c.as_array()))


@pytest.mark.parametrize("ac", ONLINE_PLANES)
def test_free_air_is_reached_above_the_table(ac):
    """The anchoring claim, end to end: at and above the sweep's top the in-ground-effect model
    must return exactly what the free-air model returns."""
    ige, oge = build_cpu(ac, True), build_cpu(ac, False)
    span, mac = geometry(ac)
    base = cases_for(ac)[3]
    top = base._replace(h=ige.p.hc[-1] * mac)        # the top row itself
    np.testing.assert_allclose(ige.coefficients(top).as_array(),
                               oge.coefficients(top).as_array(), rtol=0, atol=1e-12)
    far = base._replace(h=100.0 * span)              # far above: clamped to the same row
    np.testing.assert_allclose(ige.coefficients(far).as_array(),
                               oge.coefficients(far).as_array(), rtol=0, atol=1e-12)


@pytest.mark.parametrize("ac", ONLINE_PLANES)
def test_ground_effect_is_off_below_the_table_floor_but_held(ac, monkeypatch):
    """With the sweep alone, the increment freezes below its floor rather than extrapolating."""
    monkeypatch.setenv("FALCONS_GE_MODEL", "openvsp")
    m = build_cpu(ac, True)
    base = cases_for(ac)[1]
    at_floor = base._replace(h=m.p.hc_floor * m.mac)
    below = base._replace(h=0.0)
    np.testing.assert_allclose(m.coefficients(below).as_array(),
                               m.coefficients(at_floor).as_array(), rtol=0, atol=1e-12)


@pytest.mark.parametrize("ac", ONLINE_PLANES)
def test_ground_effect_raises_lift_near_the_ground(ac):
    m, f = build_cpu(ac, True), build_cpu(ac, False)
    u = cases_for(ac)[1]
    assert m.coefficients(u).CL > f.coefficients(u).CL


# ─── warp

@pytest.mark.cuda
@pytest.mark.parametrize("ac", ONLINE_PLANES)
@pytest.mark.parametrize("ge", [True, False])
@pytest.mark.parametrize("ge_model", GE_MODELS)
def test_warp_matches_the_cpu_reference(ac, ge, ge_model, monkeypatch):
    import warp as wp

    monkeypatch.setenv("FALCONS_GE_MODEL", ge_model)

    from falcons.aircraft.params import AerodynamicsParametersStruct
    from falcons.sim.warp.aerodynamics import compute_all_coeffs

    wp.init()
    cfg = AircraftConfig(ac).load()
    params = DerivativeAeroParameters.from_config(cfg)
    AP = params.as_warp_struct()
    span, mac = geometry(ac)
    cases = cases_for(ac)

    n = len(cases)
    fields = ["alpha", "beta", "v", "elevator", "aileron", "rudder", "p", "q", "r", "h"]
    ins = {f: wp.array(np.array([getattr(c, f) for c in cases], dtype=np.float32),
                       dtype=wp.float32, device="cuda") for f in fields}
    out = wp.zeros((n, 6), dtype=wp.float32, device="cuda")

    @wp.kernel
    def run(AP: AerodynamicsParametersStruct,
            span: wp.float32, mac: wp.float32,
            alpha: wp.array(dtype=wp.float32), beta: wp.array(dtype=wp.float32),
            v: wp.array(dtype=wp.float32), elevator: wp.array(dtype=wp.float32),
            aileron: wp.array(dtype=wp.float32), rudder: wp.array(dtype=wp.float32),
            p: wp.array(dtype=wp.float32), q: wp.array(dtype=wp.float32),
            r: wp.array(dtype=wp.float32), h: wp.array(dtype=wp.float32),
            ge_enable: bool, out: wp.array2d(dtype=wp.float32)):
        i = wp.tid()
        C_D, C_S, C_L, Cl, Cm, Cn, _cdf, _clf = compute_all_coeffs(
            AP, span, mac, alpha[i], beta[i], v[i],
            elevator[i], aileron[i], rudder[i], p[i], q[i], r[i], h[i], ge_enable)
        out[i, 0] = C_D
        out[i, 1] = C_S
        out[i, 2] = C_L
        out[i, 3] = Cl
        out[i, 4] = Cm
        out[i, 5] = Cn

    wp.launch(run, dim=n,
              inputs=[AP, span, mac] + [ins[f] for f in fields] + [ge, out],
              device="cuda")
    wp.synchronize()

    ref = np.array([build_cpu(ac, ge_enable=ge).coefficients(c).as_array() for c in cases])
    np.testing.assert_allclose(out.numpy(), ref, rtol=2e-5, atol=2e-7)


# ─── torch

@pytest.mark.skipif(len(ONLINE_PLANES) < 2, reason="needs two airframes to collide")
def test_torch_table_cache_survives_one_airframe_replacing_another():
    """The torch backend caches the derivative set as tensors. That cache must be keyed on the
    DATA, not on the parameter object's identity.

    `id()` is a memory address and CPython reuses them: build one airframe's parameters, drop
    them, build another's, and the second object can land at the freed address and be handed the
    first airframe's tables. Nothing raises -- the plant flies, with the wrong aeroplane's
    aerodynamics. This is what a benchmark sweep over airframes does in one process, and it needed
    a garbage collection between two airframes to appear, so a single-airframe parity test was
    blind to it.
    """
    import torch

    from falcons.sim.torch.altitude import _tables

    for ac in ONLINE_PLANES:
        params = DerivativeAeroParameters.from_config(AircraftConfig(ac).load())
        free = _tables(params, "cpu", torch.float64)[0].numpy()
        np.testing.assert_allclose(free, params.free, rtol=0, atol=0,
                                   err_msg=f"{ac} was handed another airframe's derivative set")
        del params          # free it, so the next airframe can be allocated at this address


@pytest.mark.parametrize("ac", ONLINE_PLANES)
@pytest.mark.parametrize("ge_model", GE_MODELS)
def test_torch_matches_the_cpu_reference(ac, ge_model, monkeypatch):
    import torch

    monkeypatch.setenv("FALCONS_GE_MODEL", ge_model)

    from falcons.sim.torch.altitude import torch_coefficients

    cfg = AircraftConfig(ac).load()
    params = DerivativeAeroParameters.from_config(cfg)
    span, mac = geometry(ac)
    cases = cases_for(ac)

    for ge in (True, False):
        cols = {f: torch.tensor([getattr(c, f) for c in cases], dtype=torch.float64)
                for f in AeroInputs._fields}
        got = torch_coefficients(params, span, mac,
                                 ge_enable=ge, device="cpu", dtype=torch.float64, **cols)
        ref = np.array([build_cpu(ac, ge_enable=ge).coefficients(c).as_array() for c in cases])
        np.testing.assert_allclose(np.stack([g.numpy() for g in got], axis=-1), ref,
                                   rtol=1e-11, atol=1e-13)


# ─── the optional Phillips-Hunsaker ground-effect model ─────────────────────────────────────────

def _run_point(ac, h_over_b, ge_model, monkeypatch):
    """Free-air and in-ground-effect coefficients at the VSPAERO run point, h/b above ground."""
    monkeypatch.setenv("FALCONS_GE_MODEL", ge_model)
    cfg = AircraftConfig(ac).load()
    p = DerivativeAeroParameters.from_config(cfg)
    span = cfg["vehicle_params"]["wing"]["span"]
    u = AeroInputs(alpha=p.alpha_run, beta=0.0, v=p.v_ref, elevator=p.de_run, aileron=0.0,
                   rudder=0.0, p=0.0, q=0.0, r=0.0, h=h_over_b * span)
    return p, build_cpu(ac, ge_enable=False).coefficients(u), build_cpu(ac).coefficients(u)


@pytest.mark.parametrize("ac", ONLINE_PLANES)
def test_phillips_hunsaker_scales_lift_and_induced_drag_only(ac, monkeypatch):
    p, free, ige = _run_point(ac, 0.25, "phillips_hunsaker", monkeypatch)
    ml = ph_mu_l(0.25, p.taper_ratio, p.aspect_ratio)
    md = ph_mu_d(0.25, p.taper_ratio, p.aspect_ratio)
    assert ige.CL == pytest.approx(free.CL * ml, rel=1e-12)
    assert ige.CD == pytest.approx(free.CD + p.k_ind_free * free.CL ** 2 * (md * ml * ml - 1.0),
                                   rel=1e-12)
    assert (ige.CS, ige.Cl, ige.Cm, ige.Cn) == (free.CS, free.Cl, free.Cm, free.Cn)


@pytest.mark.parametrize("ac", ONLINE_PLANES)
@pytest.mark.parametrize("h_over_b", [0.3, 0.5, 1.0, 2.0])
def test_the_two_models_agree_on_lift_and_drag_inside_the_sweep(ac, h_over_b, monkeypatch):
    """Inside the measured sweep the lifting-line correlation and the VSPAERO ground image give the
    same change in lift and drag at the run point, within 3 % (the paper's claim)."""
    _, free, vsp = _run_point(ac, h_over_b, "openvsp", monkeypatch)
    _, _, ph = _run_point(ac, h_over_b, "phillips_hunsaker", monkeypatch)
    assert vsp.CL / free.CL == pytest.approx(ph.CL / free.CL, rel=0.03)
    assert vsp.CD / free.CD == pytest.approx(ph.CD / free.CD, rel=0.03)


# ─── the hybrid: the sweep where measured, the correlation's trend below its floor ──────────────

@pytest.mark.parametrize("ac", ONLINE_PLANES)
@pytest.mark.parametrize("h_over_b", [0.2001, 0.25, 0.5, 1.0, 3.0])
def test_hybrid_is_the_sweep_at_and_above_its_floor(ac, h_over_b, monkeypatch):
    hb_floor = DerivativeAeroParameters.from_config(AircraftConfig(ac).load()).hc[0] \
        * geometry(ac)[1] / geometry(ac)[0]
    hb = max(h_over_b, hb_floor)
    base = cases_for(ac)[1]._replace(h=hb * geometry(ac)[0])
    monkeypatch.setenv("FALCONS_GE_MODEL", "openvsp")
    vsp = build_cpu(ac).coefficients(base).as_array()
    monkeypatch.setenv("FALCONS_GE_MODEL", "hybrid")
    np.testing.assert_allclose(build_cpu(ac).coefficients(base).as_array(), vsp, rtol=0, atol=1e-12)


@pytest.mark.parametrize("ac", ONLINE_PLANES)
def test_hybrid_continues_the_floor_with_the_correlation_trend(ac, monkeypatch):
    monkeypatch.setenv("FALCONS_GE_MODEL", "hybrid")
    m = build_cpu(ac)
    span, mac = geometry(ac)
    hb_floor = m.p.hc[0] * mac / span
    base = cases_for(ac)[1]
    floor = m.coefficients(base._replace(h=hb_floor * span))
    just_below = m.coefficients(base._replace(h=(hb_floor - 1e-7) * span))
    np.testing.assert_allclose(just_below.as_array(), floor.as_array(), rtol=1e-5)  # no step
    deep = m.coefficients(base._replace(h=0.1 * span))
    ml, dd = ph_relative(0.1, hb_floor, m.p.taper_ratio, m.p.aspect_ratio)
    assert deep.CL == pytest.approx(floor.CL * ml, rel=1e-12)
    assert deep.CL > floor.CL                          # lift keeps rising toward the surface
    assert (deep.CS, deep.Cl, deep.Cm, deep.Cn) == pytest.approx(
        (floor.CS, floor.Cl, floor.Cm, floor.Cn), rel=1e-12)          # moments held at the floor
