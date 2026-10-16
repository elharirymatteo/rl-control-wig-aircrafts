"""falcons.benchmark.stats on synthetic data: IQM, the two bootstrap schemes, rate intervals."""
import numpy as np
import pytest

from falcons.benchmark.stats import bootstrap, counts, iqm, rate, summary, wilson


def test_iqm_is_the_mean_of_the_middle_half():
    assert iqm([1, 2, 3, 4, 5, 6, 7, 8]) == pytest.approx(4.5)          # mean of 3..6
    assert iqm([0, 0, 0, 0, 1000]) == pytest.approx(0.0)                # one outlier ignored
    assert iqm([3.0]) == 3.0
    assert np.isnan(iqm([]))


def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    x = np.random.default_rng(1).normal(5.0, 1.0, 200)
    a = bootstrap(x, n_boot=500)
    assert a == bootstrap(x, n_boot=500)                                  # fixed generator seed
    est, lo, hi = a
    assert lo <= est <= hi and hi - lo < 1.0


def test_hierarchical_interval_carries_the_between_seed_spread():
    """Five seeds whose episodes agree inside a seed but disagree between seeds: the stratified
    scheme (seeds fixed) is narrow, the hierarchical one (seeds resampled) is not."""
    rng = np.random.default_rng(2)
    centres = [0.2, 0.3, 0.25, 3.0, 0.28]
    x = np.concatenate([c + 0.01 * rng.standard_normal(100) for c in centres])
    g = np.repeat(np.arange(5), 100)
    _, slo, shi = bootstrap(x, g, mode="stratified", n_boot=1000)
    _, hlo, hhi = bootstrap(x, g, mode="hierarchical", n_boot=1000)
    assert (hhi - hlo) > 5 * (shi - slo)


def test_empty_groups_are_dropped_and_empty_input_is_nan():
    assert np.isnan(bootstrap([], None)[0])
    est, lo, hi = bootstrap([1.0, 1.0], [0, 0])
    assert est == lo == hi == 1.0


def test_wilson_and_degenerate_rates():
    lo, hi = wilson(0, 20)
    assert lo == 0.0 and 0.0 < hi < 0.2                                  # 0/20 still has an upper bound
    r = rate([1] * 50)
    assert r["k"] == 50 and r["n"] == 50 and r["rate"] == 1.0 and r["lo"] < 1.0 and r["hi"] == 1.0


def test_rate_with_groups_uses_the_bootstrap():
    flags = [1] * 100 + [0] * 100 + [1] * 100
    r = rate(flags, np.repeat([0, 1, 2], 100), n_boot=1000)
    assert r["k"] == 200 and r["n"] == 300
    assert r["lo"] < 2 / 3 < r["hi"] and r["hi"] - r["lo"] > 0.3        # whole-seed failures


def test_summary_and_counts_skip_missing_values():
    s = summary([1.0, None, 3.0, np.nan], [0, 0, 1, 1], n_boot=200)
    assert s["n"] == 2 and s["iqm"] == pytest.approx(2.0)
    assert counts([True, False, True]) == dict(n_total=3, n_ok=2, n_dropped=1)
