"""The ground-effect training study's aggregation (falcons.benchmark.ge_training).

Synthetic episodes only: the aggregation is where the development study went wrong (errors
averaged over crashed rollouts), so it is tested on data whose answer is known."""
import math

import numpy as np
import pytest

from falcons.benchmark import ge_training as G


def _ep(plane, seed, train, ege, band, rmse, survived):
    return dict(plane=plane, seed=seed, train=train, eval_ge=ege, band=band, episode=0,
                target_m=1.5, survived=int(survived), rmse_m=rmse if survived else float("nan"),
                max_err_m=rmse * 2 if survived else float("nan"))


def _cell(train, ege, rmse, n=20, dropped=0, seeds=(0, 1, 2)):
    rows = []
    for s in seeds:
        rows += [_ep("Airship_V7", s, train, ege, "near", rmse, True) for _ in range(n - dropped)]
        rows += [_ep("Airship_V7", s, train, ege, "near", rmse, False) for _ in range(dropped)]
    return rows


def _two_by_two(on_on=0.2, on_off=1.0, off_on=0.9, off_off=0.2, dropped=0):
    return (_cell("GE_ON", 1, on_on) + _cell("GE_ON", 0, on_off)
            + _cell("GE_OFF", 1, off_on, dropped=dropped) + _cell("GE_OFF", 0, off_off))


def test_errors_use_survivors_only_and_drops_are_counted():
    per_seed, agg, _ = G.aggregate(_two_by_two(dropped=5))
    row = next(r for r in agg if r["train"] == "GE_OFF" and r["eval_ge"] == 1)
    assert (row["n"], row["n_survived"], row["n_dropped"]) == (60, 45, 15)
    assert row["survival"] == pytest.approx(0.75)
    assert row["rmse_iqm"] == pytest.approx(0.9)            # crashed rows (NaN) never enter
    s0 = next(r for r in per_seed if r["train"] == "GE_OFF" and r["eval_ge"] == 1 and r["seed"] == 0)
    assert (s0["n_survived"], s0["n_dropped"]) == (15, 5)


def test_effects_match_their_definitions():
    _, _, eff = G.aggregate(_two_by_two(on_on=0.2, on_off=1.0, off_on=0.9, off_off=0.2))
    e = {r["effect"]: r for r in eff}
    assert e["deploy_penalty"]["value"] == pytest.approx(0.9 - 0.2)
    assert e["interaction"]["value"] == pytest.approx((1.0 - 0.2) - (0.2 - 0.9))
    # constant cells: every bootstrap replicate equals the point
    assert e["interaction"]["lo"] == pytest.approx(e["interaction"]["value"])


def test_cell_with_no_survivors_reports_nan_not_zero():
    rows = _two_by_two()
    rows = [dict(r, survived=0, rmse_m=float("nan")) if (r["train"], r["eval_ge"]) == ("GE_OFF", 0)
            else r for r in rows]
    _, agg, eff = G.aggregate(rows)
    row = next(r for r in agg if r["train"] == "GE_OFF" and r["eval_ge"] == 0)
    assert row["survival"] == 0.0 and math.isnan(row["rmse_iqm"])
    assert all(r["effect"] != "interaction" or math.isnan(r["value"]) for r in eff)


def test_bootstrap_is_deterministic_and_brackets_the_point():
    rng = np.random.default_rng(1)
    cells = {0: {s: rng.normal(1.0, 0.2, 50) for s in range(3)}}
    a = G.bootstrap(cells, lambda m: G.iqm(m[0]))
    b = G.bootstrap(cells, lambda m: G.iqm(m[0]))
    assert a == b
    assert a[1] <= a[0] <= a[2]


def test_iqm_ignores_the_tails():
    assert G.iqm([0, 1, 1, 1, 1, 100]) == pytest.approx(1.0)


def test_bands_scale_with_span():
    for p in ("Airship_V7", "Volantex_Ranger"):
        b = G.span(p)
        lo, hi = G.band(p, "near")
        assert lo == pytest.approx(0.20 * b) and hi == pytest.approx(0.46 * b)
        clo, chi = G.band(p, "control")
        assert clo == 30.0 and chi - clo == pytest.approx(hi - lo)


@pytest.mark.slow
@pytest.mark.cuda
def test_smoke_two_by_two(tmp_path):
    """Pipeline end to end on one seed: a short fine-tune, a handful of episodes per cell."""
    agg, eff = G.run(["Airship_V7"], [0], children=tmp_path / "children", results_dir=tmp_path,
                     steps=131_072, episodes=8, bands=("near",))
    assert len(agg) == 4
    assert {r["effect"] for r in eff} == {"deploy_penalty", "interaction"}
    assert (tmp_path / "ge_training_episodes.csv").exists()
