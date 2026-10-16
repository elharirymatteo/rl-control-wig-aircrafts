"""Does modelling ground effect during TRAINING matter for the controller? A 2x2 study.

Each PPO altitude parent (`checkpoints/ppo_altitude_<plane>_s<seed>.pt`, a 30-90 m tracker) is
fine-tuned twice on a near-ground band: once with the ground-effect correction on (GE_ON) and once
with it off (GE_OFF). The two arms share the parent, every hyperparameter, the step count, the band,
the spawn and the seed; the only difference is `in_ground_effect`. Each child is then flown with the
correction on and off -- the 2x2 {train GE} x {eval GE} -- in the near-ground band, and in a 30 m
band as a forgetting check.

Ported from the development study, with three fixes:
  - Survival comes first and errors average SURVIVING episodes only. The env auto-resets a
    terminated episode inside `step`, so a crashed env goes on flying a fresh episode; the
    development evaluator kept accumulating its error, mixing crashed rollouts into the RMSE. Here
    an `alive` mask ends each episode at its first termination, and the dropped count is reported.
  - Parents are the benchmark checkpoints themselves, one per seed, so every child of a seed shares
    the same, published parent.
  - The 30 m band is a forgetting check, not a difference-in-differences control: near-ground
    fine-tuning degrades the 30 m regime in BOTH arms, so it cannot isolate ground effect.
Altitude error is taken from the simulator state against the commanded altitude, not from the
observation (which is clipped).

Statistics: interquartile mean over the surviving episodes pooled across seeds, with a percentile
bootstrap that resamples episodes within each seed (stratified, Agarwal et al. 2021); survival is a
pooled rate with a Wilson interval. The bootstrap here is self-contained; the benchmark-wide
`falcons.benchmark.stats` (if present) implements the same estimators.

Outputs in <results_dir>:
  ge_training_episodes.csv  one row per episode (the raw data every number is computed from)
  ge_training.csv           one row per (plane, seed, train arm, eval GE, band)
  ge_training_agg.csv       one row per (plane, band, train arm, eval GE): survival and RMSE IQM, CIs
  ge_training_effects.csv   per (plane, band): deploy penalty and interaction term, with CIs

Run: falcons benchmark ge-training [--planes ...] [--seeds ...] [--parents DIR]
"""
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from falcons.paths import CKPT_DIR, RESULTS_DIR

PLANES = ["Airship_V7", "Volantex_Ranger"]
SEEDS = [0, 1, 2, 3, 4]
ARMS = {"GE_ON": True, "GE_OFF": False}
NEAR_HB = (0.20, 0.46)          # commanded height over span of the near-ground band
CONTROL_LO = 30.0               # control band starts here [m], same width as the near band
FT_STEPS = 20_000_000           # fine-tuning budget per child
LR0, STD0, STD1 = 2e-4, 0.5, 0.2  # warm-start schedule: gentle LR, moderate exploration
HORIZON, SETTLE = 1500, 500     # 15 s episodes, settled window = last 10 s
EPISODES = 200
EVAL_SEED = 10_000              # np.random seed base: every cell of a seed sees the same targets
N_BOOT = 2000


# ───── bands ─────
def span(plane):
    from falcons.aircraft.config import AircraftConfig
    return float(AircraftConfig(plane).load()["vehicle_params"]["wing"]["span"])


def band(plane, name):
    """(lo, hi) commanded altitude [m] of the `near` or `control` band for `plane`."""
    b = span(plane)
    lo, hi = NEAR_HB[0] * b, NEAR_HB[1] * b
    if name == "near":
        return lo, hi
    if name == "control":
        return CONTROL_LO, CONTROL_LO + (hi - lo)
    raise ValueError(name)


def spawn_of(lo, hi):
    """Initial offset from the target: inside the band, as in the development study."""
    return min(0.6, 0.45 * (hi - lo))


def env_cfg(plane, lo, hi, ge, horizon):
    from falcons.envs.configs import altitude_env_cfg
    cfg = altitude_env_cfg(plane, spawn_of(lo, hi), horizon=horizon, in_ground_effect=ge)
    cfg.update(target_mode="dynamic", target_lo=lo, target_hi=hi)
    return cfg


def child_path(plane, seed, arm, out_dir):
    return Path(out_dir) / f"ppo_altitude_{plane}_s{seed}_{arm.lower().replace('_', '')}.pt"


def _sha1(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


# ───── training ─────
def train(plane, seed, arm, parents=CKPT_DIR, out_dir=None, steps=FT_STEPS):
    """Fine-tune the seed's altitude parent on the near band with ground effect `arm`."""
    import warp as wp
    from falcons.controllers.policies import ckpt_path
    from falcons.envs.altitude import AltitudeEnv
    from falcons.train.ppo import Agent, DEVICE, N_ENVS, train_loop
    wp.init()
    out_dir = Path(out_dir or Path(parents) / "ge_training")
    parent = ckpt_path("ppo", "altitude", plane, seed, parents)
    torch.manual_seed(seed)
    np.random.seed(seed)
    lo, hi = band(plane, "near")
    cfg = env_cfg(plane, lo, hi, ARMS[arm], HORIZON)
    env = AltitudeEnv(N_ENVS, DEVICE, cfg)
    agent = Agent(env.num_obs, env.num_act).to(DEVICE)
    agent.load_state_dict(torch.load(parent, map_location=DEVICE))
    t0 = time.time()
    train_loop(env, agent, total_steps=steps, lr0=LR0, std0=STD0, std1=STD1,
               log_prefix=f"GE-training {plane} s{seed} {arm}")
    out = child_path(plane, seed, arm, out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), out)
    manifest = dict(plane=plane, seed=seed, arm=arm, in_ground_effect=ARMS[arm], steps=steps,
                    parent=str(parent), parent_sha1=_sha1(parent), band=[lo, hi],
                    lr0=LR0, std0=STD0, std1=STD1, n_envs=N_ENVS, seconds=round(time.time() - t0),
                    child=str(out), cfg={k: cfg[k] for k in sorted(cfg)})
    with open(out.parent / "manifest.jsonl", "a") as f:
        f.write(json.dumps(manifest) + "\n")
    print(f"WROTE {out}  ({manifest['seconds']} s)")
    return out


# ───── evaluation ─────
@torch.no_grad()
def evaluate(child, plane, band_name, eval_ge, episodes=EPISODES, seed=0):
    """Fly `episodes` episodes of the band with the correction `eval_ge`; one dict per episode.

    `survived` is False for an episode that terminated (crash, stall, numerical guard) before the
    horizon; its error fields are NaN, so nothing downstream can average them by accident."""
    import warp as wp
    from falcons.envs.altitude import AltitudeEnv
    from falcons.train.ppo import Agent, DEVICE
    wp.init()
    lo, hi = band(plane, band_name)
    np.random.seed(EVAL_SEED + seed)            # targets and spawns identical across the 2x2
    env = AltitudeEnv(episodes, DEVICE, env_cfg(plane, lo, hi, eval_ge, HORIZON + 1))
    agent = Agent(env.num_obs, env.num_act).to(DEVICE)
    agent.load_state_dict(torch.load(child, map_location=DEVICE))
    agent.eval()
    obs = env.reset()
    target = wp.to_torch(env._true_target).clone()
    alive = torch.ones(episodes, dtype=torch.bool, device=DEVICE)
    sq = torch.zeros(episodes, device=DEVICE)
    mx = torch.zeros(episodes, device=DEVICE)
    cnt = 0
    for t in range(HORIZON):
        obs, _, done, _ = env.step(agent.actor_mean(obs).clamp(-1.0, 1.0))
        alive &= ~done                          # horizon is HORIZON+1: done here means terminated
        if t >= SETTLE:
            h = -wp.to_torch(env.model._state["position"])[:, 2]
            e = (h - target).abs()
            sq += torch.where(alive, e * e, torch.zeros_like(e))
            mx = torch.where(alive, torch.maximum(mx, e), mx)
            cnt += 1
    rmse = torch.sqrt(sq / cnt)
    nan = float("nan")
    return [dict(episode=i, target_m=float(target[i]), survived=int(alive[i]),
                 rmse_m=float(rmse[i]) if alive[i] else nan,
                 max_err_m=float(mx[i]) if alive[i] else nan)
            for i in range(episodes)]


# ───── statistics (self-contained) ─────
def iqm(x):
    """Interquartile mean: the mean of the middle 50% (scipy.stats.trim_mean(x, 0.25))."""
    x = np.sort(np.asarray(x, dtype=float))
    if x.size == 0:
        return float("nan")
    from scipy.stats import trim_mean
    return float(trim_mean(x, 0.25))


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def bootstrap(cells, fn, n_boot=N_BOOT, seed=0):
    """Percentile CI of fn({cell: pooled samples}) under resampling WITHIN each (cell, seed).

    cells: {cell: {seed: 1-d array}}. Returns (point, lo, hi)."""
    def pooled(sample):
        return {c: np.concatenate([v for v in s.values()]) if s else np.array([])
                for c, s in sample.items()}
    point = fn(pooled(cells))
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        rs = {c: {k: (v[rng.integers(0, len(v), len(v))] if len(v) else v) for k, v in s.items()}
              for c, s in cells.items()}
        reps.append(fn(pooled(rs)))
    reps = np.asarray(reps, dtype=float)
    reps = reps[np.isfinite(reps)]
    if reps.size == 0 or not np.isfinite(point):
        return float(point), float("nan"), float("nan")
    return float(point), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def deploy_penalty(m):
    """RMSE a GE-off-trained policy loses in the GE-on world, against a GE-on-trained one."""
    return m[("GE_OFF", 1)] - m[("GE_ON", 1)]


def interaction(m):
    """(train ON: eval off - eval on) - (train OFF: eval off - eval on); > 0 = GE-specific."""
    return (m[("GE_ON", 0)] - m[("GE_ON", 1)]) - (m[("GE_OFF", 0)] - m[("GE_OFF", 1)])


def aggregate(episodes):
    """episode rows -> (per-seed rows, aggregate rows, effect rows). Errors use survivors only."""
    per_seed, agg, effects = [], [], []
    key = lambda r: (r["plane"], r["band"], r["train"], int(r["eval_ge"]))
    groups = {}
    for r in episodes:
        groups.setdefault(key(r), {}).setdefault(int(r["seed"]), []).append(r)
    for (plane, bnd, arm, ege), by_seed in sorted(groups.items()):
        for s, rows in sorted(by_seed.items()):
            surv = [r for r in rows if int(r["survived"])]
            e = np.array([float(r["rmse_m"]) for r in surv])
            per_seed.append(dict(plane=plane, band=bnd, train=arm, eval_ge=ege, seed=s,
                                 n=len(rows), n_survived=len(surv), n_dropped=len(rows) - len(surv),
                                 rmse_mean=float(e.mean()) if e.size else np.nan,
                                 rmse_iqm=iqm(e),
                                 max_err_mean=float(np.mean([float(r["max_err_m"]) for r in surv]))
                                 if surv else np.nan))
        n = sum(len(v) for v in by_seed.values())
        k = sum(int(r["survived"]) for v in by_seed.values() for r in v)
        lo, hi = wilson(k, n)
        cells = {0: {s: np.array([float(r["rmse_m"]) for r in v if int(r["survived"])])
                     for s, v in by_seed.items()}}
        p, plo, phi = bootstrap(cells, lambda m: iqm(m[0]))
        agg.append(dict(plane=plane, band=bnd, train=arm, eval_ge=ege, n_seeds=len(by_seed),
                        n=n, n_survived=k, n_dropped=n - k, survival=k / n if n else np.nan,
                        survival_lo=lo, survival_hi=hi, rmse_iqm=p, rmse_lo=plo, rmse_hi=phi))
    for plane, bnd in sorted({(k[0], k[1]) for k in groups}):
        cells = {}
        for arm in ARMS:
            for ege in (0, 1):
                g = groups.get((plane, bnd, arm, ege), {})
                cells[(arm, ege)] = {s: np.array([float(r["rmse_m"]) for r in v if int(r["survived"])])
                                     for s, v in g.items()}
        if len(cells) < 4 or any(not c for c in cells.values()):
            continue
        stat = lambda fn: (lambda m: fn({c: iqm(v) for c, v in m.items()}))
        for name, fn in (("deploy_penalty", deploy_penalty), ("interaction", interaction)):
            p, lo, hi = bootstrap(cells, stat(fn))
            effects.append(dict(plane=plane, band=bnd, effect=name, value=p, lo=lo, hi=hi))
    return per_seed, agg, effects


def _write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path}")


# ───── the study ─────
def run(planes=None, seeds=None, parents=CKPT_DIR, children=None, results_dir=RESULTS_DIR,
        steps=FT_STEPS, episodes=EPISODES, bands=("near", "control")):
    """Train the missing children, fly the 2x2 in every band, write the four CSVs."""
    from falcons.controllers.policies import ckpt_path
    planes, seeds = planes or PLANES, SEEDS if seeds is None else seeds
    children = Path(children or Path(parents) / "ge_training")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for plane in planes:
        for seed in seeds:
            if not ckpt_path("ppo", "altitude", plane, seed, parents).exists():
                print(f"{plane} s{seed}: no parent in {parents} -- skipped")
                continue
            for arm in ARMS:
                c = child_path(plane, seed, arm, children)
                if not c.exists():
                    train(plane, seed, arm, parents, children, steps)
            for arm in ARMS:
                c = child_path(plane, seed, arm, children)
                for bnd in bands:
                    for ege in (True, False):
                        t0 = time.time()
                        ep = evaluate(c, plane, bnd, ege, episodes, seed)
                        surv = [e for e in ep if e["survived"]]
                        rm = np.mean([e["rmse_m"] for e in surv]) if surv else float("nan")
                        print(f"{plane:16s} s{seed} train {arm:6s} eval GE {'on ' if ege else 'off'} "
                              f"{bnd:7s} surv {len(surv)}/{len(ep)}  RMSE {rm:.3f} m  "
                              f"({time.time() - t0:.0f} s)", flush=True)
                        rows += [dict(plane=plane, seed=seed, train=arm, eval_ge=int(ege), band=bnd,
                                      **e) for e in ep]
    per_seed, agg, effects = aggregate(rows)
    _write(results_dir / "ge_training_episodes.csv", rows)
    _write(results_dir / "ge_training.csv", per_seed)
    _write(results_dir / "ge_training_agg.csv", agg)
    _write(results_dir / "ge_training_effects.csv", effects)
    return agg, effects
