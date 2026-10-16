"""Aggregate statistics for the benchmark tables: interquartile mean, bootstrap confidence
intervals, and run counts.

A benchmark cell pools runs that come in GROUPS: the episodes of one training seed, the rollouts of
one MPPI sampling draw, the gust realizations flown by one seed. Runs inside a group share the
policy, so they are not independent of each other, and the between-group spread (a lucky or an
unlucky training run) is usually the larger part of the uncertainty. Two resampling schemes are
provided:

- "hierarchical" (default): resample the groups with replacement, then the runs inside each drawn
  group with replacement. This keeps the between-seed variance in the interval; with five seeds it
  is the honest one.
- "stratified": resample runs within each group and keep every group, the scheme of Agarwal et al.
  (2021, "Deep RL at the edge of the statistical precipice", rliable) with the group as stratum.
  It measures the precision of the pooled estimate for THESE seeds only.

Point estimates are the interquartile mean (IQM, the 25% trimmed mean that rliable recommends),
robust to the one diverging run that drags a plain mean. Rates (survival, acquisition) are
reported as counts k/n with a confidence interval: the bootstrap of the indicator mean when there
is more than one group, the Wilson score interval otherwise or when the bootstrap is degenerate
(every run survived or none did, where a resampling interval collapses to a point).

Every resampling uses a fixed generator seed, so a table is reproducible bit for bit.
"""
import numpy as np

N_BOOT = 5000
ALPHA = 0.05            # 95 % intervals
SEED = 0


def iqm(x):
    """Interquartile mean: the mean of the middle 50 % of the values (25 % cut from each end,
    fractional cut points as in scipy.stats.trim_mean). NaN for an empty input."""
    a = np.sort(np.asarray(x, dtype=float).ravel())
    n = a.size
    if n == 0:
        return float("nan")
    lo = int(np.floor(0.25 * n))
    return float(a[lo:n - lo].mean()) if n - 2 * lo > 0 else float(a.mean())


def _groups(values, groups):
    v = np.asarray(values, dtype=float).ravel()
    if groups is None:
        return [v]
    g = np.asarray(groups).ravel()
    if g.size != v.size:
        raise ValueError(f"values ({v.size}) and groups ({g.size}) differ in length")
    return [v[g == k] for k in dict.fromkeys(g.tolist())]


def bootstrap(values, groups=None, stat=iqm, n_boot=N_BOOT, alpha=ALPHA, seed=SEED,
              mode="hierarchical"):
    """(estimate, lo, hi): `stat` of the pooled values and its percentile bootstrap interval.

    `groups` labels each value with its group (seed, draw); None means one group. Empty groups are
    dropped (a seed whose every run departed contributes nothing to an error metric). NaN for an
    empty input; a single value has no spread and returns it as both bounds."""
    parts = [p for p in _groups(values, groups) if p.size]
    if not parts:
        return float("nan"), float("nan"), float("nan")
    est = stat(np.concatenate(parts))
    rng = np.random.default_rng(seed)
    G = len(parts)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, G, G) if (mode == "hierarchical" and G > 1) else np.arange(G)
        draw = [parts[i][rng.integers(0, parts[i].size, parts[i].size)] for i in pick]
        reps[b] = stat(np.concatenate(draw))
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return float(est), float(lo), float(hi)


def wilson(k, n, alpha=ALPHA):
    """Wilson score interval for k successes in n trials; (nan, nan) for n = 0."""
    if n == 0:
        return float("nan"), float("nan")
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def rate(flags, groups=None, n_boot=N_BOOT, alpha=ALPHA, seed=SEED):
    """dict(k, n, rate, lo, hi) for boolean outcomes (survived, acquired)."""
    f = np.asarray(flags, dtype=float).ravel()
    n, k = int(f.size), int(f.sum())
    if n == 0:
        return dict(k=0, n=0, rate=float("nan"), lo=float("nan"), hi=float("nan"))
    ng = len([p for p in _groups(f, groups) if p.size])
    if ng > 1 and 0 < k < n:
        est, lo, hi = bootstrap(f, groups, stat=np.mean, n_boot=n_boot, alpha=alpha, seed=seed)
        if hi > lo:
            return dict(k=k, n=n, rate=est, lo=lo, hi=hi)
    lo, hi = wilson(k, n, alpha)
    return dict(k=k, n=n, rate=k / n, lo=lo, hi=hi)


def summary(values, groups=None, **kw):
    """dict(iqm, lo, hi, n) of a metric over the runs that produced one (NaN/None dropped)."""
    v = np.asarray([np.nan if x is None else x for x in np.asarray(values, dtype=object).ravel()],
                   dtype=float)
    g = None if groups is None else np.asarray(groups).ravel()
    ok = np.isfinite(v)
    est, lo, hi = bootstrap(v[ok], None if g is None else g[ok], **kw)
    return dict(iqm=est, lo=lo, hi=hi, n=int(ok.sum()))


def counts(flags):
    """dict(n_total, n_ok, n_dropped) for boolean outcomes."""
    f = np.asarray(flags, dtype=bool).ravel()
    return dict(n_total=int(f.size), n_ok=int(f.sum()), n_dropped=int((~f).sum()))
