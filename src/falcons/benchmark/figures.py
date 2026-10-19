"""Every figure the benchmark reports, one function per figure, all writing into
<results_dir>/figures and returning the file they wrote.

    altitude_viz(algo, planes)  altitude tracking, one figure per algorithm, one row per airframe
    attitude_exec(planes)       the command interface: commanded bank fan + all three channels
    maneuver_facet(plane)       small multiples of bank tracking, maneuvers x methods
    maneuver_render(planes)     top-down ground tracks of the four maneuvers, coloured by bank
    ge_energy()                 ground effect: thrust change and tracking accuracy against h/b
    robustness_ppo()            the PPO executor under four disturbance families
    throughput()                env-steps/s against the number of parallel environments
    learning_curves()           evaluation metric against env steps, IQM over training seeds

Every figure is drawn at the size it is printed (TEXT_W wide, or HALF_W for a half-width figure)
so the paper includes it at width=\\textwidth (or its natural width) with no scaling: 8 pt text,
7 pt ticks and annotations. Each writes a vector PDF and a PNG beside it for quick viewing.

Run: render() -> <results_dir>/figures/*
"""
import csv
import functools
import pickle
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from falcons.paths import CKPT_DIR, RESULTS_DIR

DT = 0.01
TEXT_W = 6.5           # in; TMLR \textwidth, the printed width of the full-width figures
HALF_W = 3.2           # in; a half-width figure (two side by side, or wrapped in text)
PNG_DPI = 200          # the quick-look PNG written next to every PDF
BANK_LIM = 50.0       # deg; the bank colour scale, shared by attitude_exec and maneuver_render
# The printed style. Sizes are in points at the printed size, which is the canvas size.
RC = {"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8,
      "ytick.labelsize": 8, "legend.fontsize": 8, "figure.titlesize": 9,
      "figure.labelsize": 9, "axes.linewidth": 0.7, "lines.linewidth": 1.2,
      "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.5,
      "ytick.minor.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
      "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.major.pad": 2,
      "ytick.major.pad": 2, "axes.labelpad": 2, "axes.titlepad": 3,
      "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
      "legend.handlelength": 1.8, "legend.columnspacing": 1.2, "legend.borderaxespad": 0.2,
      "grid.linewidth": 0.4, "grid.alpha": 0.5, "grid.color": "0.8",
      # TrueType rather than Type 3 fonts in the PDF: what camera-ready checkers ask for
      "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300}
# One palette across every figure: the tracking figures name methods, the ground-effect figure names
# airframes. LQR purple, MPPI red, PPO (CAPS) blue, SAC green, TD3 orange.
METHOD_COLORS = {"LQR": "#8250c4", "MPPI": "#d1495b", "PPO": "#2a78d6", "SAC": "#008300",
                 "TD3": "#e8601c"}
PLANE_COLORS = {"Airship_V7": "#2a78d6", "Airship_A0S": "#d1495b", "Volantex_Ranger": "#008300",
                "Navion": "#8250c4", "Cirrus_SR22": "#e8601c"}
# The two airframes the attitude figures report, in panel order, with their printed names.
PRETTY = {"Airship_V7": "Airship V7", "Volantex_Ranger": "Volantex Ranger"}
SHAPE_PRETTY = {"circle": "circle", "figure8": "figure-8", "helix": "helix", "sturn": "s-turn"}


PLANES = tuple(PRETTY)


def _figdir(results_dir):
    d = Path(results_dir) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _paper(fn):
    """Draw the figure under the printed style RC, leaving the global rcParams alone."""
    @functools.wraps(fn)
    def wrapped(*a, **k):
        with plt.rc_context(RC):
            return fn(*a, **k)
    return wrapped


def _rasterize_dense(fig, max_points=300):
    """Keep the PDF light enough for Overleaf: data artists with many vertices (long traces, fill
    bands, line collections, dense scatters) are rasterized at savefig.dpi (300) inside the PDF,
    while text, axes, ticks, legends and sparse markers stay vector, so the type stays sharp."""
    from matplotlib.collections import Collection
    for ax in fig.axes:
        for ln in ax.lines:
            if len(ln.get_xdata()) > max_points:
                ln.set_rasterized(True)
        for c in ax.collections:
            if not isinstance(c, Collection):
                continue
            n = len(c.get_paths()) + (len(c.get_offsets()) if c.get_offsets() is not None else 0)
            big = n > max_points or any(len(p.vertices) > max_points for p in c.get_paths())
            if big:
                c.set_rasterized(True)


def _save(fig, results_dir, name):
    """<name>.pdf at the canvas size (never bbox_inches="tight": that would change the printed
    size), plus <name>.png for a quick look. A constrained layout solves at draw time against
    text extents that depend on the dpi, so it is settled at the output dpi first."""
    out = _figdir(results_dir) / f"{name}.pdf"
    _rasterize_dense(fig)
    fig.set_dpi(300)
    fig.canvas.draw()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=PNG_DPI)
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"wrote {out}  ({w:.2f} x {h:.2f} in)")
    return out


def _letter(ax, s):
    """Panel letter (and an optional short label), at the panel's upper left."""
    ax.set_title(s, loc="left", fontsize=8, pad=3)


# ───────────────────────────── altitude dashboard ─────────────────────────────
# constants for this dashboard alone; `N` is its Monte Carlo batch, not a batch anywhere else.
BAND = 1.0        # settled = |alt - target| < BAND [m]
BASE = 60.0       # spawn altitude [m]
K = 1000          # 10 s of stable convergence before advancing to the next target
M = 5             # altitude waypoints per sequence
N = 256           # Monte Carlo batch
NTRACK = 10       # episodes recorded in full (trajectories)
H = 13000         # 130 s total


def _altitude_run(algo, plane, ckpt_dir):
    """Sequential altitude waypoints tracked by one learned policy, over N Monte Carlo episodes,
    NTRACK of them recorded in full."""
    import torch
    import warp as wp
    from falcons.envs.altitude import AltitudeEnv
    from falcons.envs.configs import PLANE_CONFIGS
    from falcons.controllers.policies import load_policy

    with torch.no_grad():
        rr = PLANE_CONFIGS[plane]["ref_rate"]
        env = AltitudeEnv(N, "cuda", {"target_mode": "dynamic", "spawning_distance": 0.0,
                                      "aircraft": plane, "ref_rate": rr, "horizon": H + 1,
                                      "i_clamp": PLANE_CONFIGS[plane].get("i_clamp", 25.0)})
        act = load_policy(algo, "altitude", plane, 0, ckpt_dir=ckpt_dir).act

        # per-env altitude sequences: realistic climb/descent steps
        seq = np.zeros((N, M), np.float32); cur = np.full(N, BASE)
        np.random.seed(42)
        for j in range(M):
            up = np.random.rand(N) < 0.5
            step = np.where(up, np.random.uniform(4, 8, N), -np.random.uniform(4, 18, N))
            cur = np.clip(cur + step, 25, 95); seq[:, j] = cur

        env.reset()
        pos0 = np.zeros((N, 3), np.float32); pos0[:, 2] = -BASE
        vel0 = np.zeros((N, 3), np.float32); vel0[:, 0] = env.base_vel
        wp.copy(env._true_target, wp.array(seq[:, 0].copy(), device="cuda"))
        wp.copy(env._target,      wp.array(np.full(N, BASE, np.float32), device="cuda"))
        wp.copy(env.model._state["position"],   wp.array(pos0, dtype=wp.vec3f, device="cuda"))
        wp.copy(env.model._state["linear_vel"], wp.array(vel0, dtype=wp.vec3f, device="cuda"))
        env._ierr.zero_(); env._refrate.zero_(); env._obs_launch()
        obs = wp.to_torch(env._obs)

        idx = np.zeros(N, int); sc = np.zeros(N, int); done = np.zeros(N, bool)
        frozen = np.zeros(NTRACK, bool)
        # full trajectory: (north, east, alt, alt_cmd, elevator, throttle)
        traj = [[] for _ in range(NTRACK)]
        inj = [[(0, float(seq[i, 0]))] for i in range(NTRACK)]   # injection events
        endstep = np.full(NTRACK, H)
        # MC arrays for all N episodes. An episode that departs (crash/stall) is auto-reset by the
        # env; what it flies after that is a different episode, so from its departure on it is
        # masked out of the error (NaN) and captures no further targets.
        err_all = np.full((H, N), np.nan)
        dead = np.zeros(N, bool)

        for t in range(H):
            cur_tgt = seq[np.arange(N), np.minimum(idx, M - 1)]
            wp.copy(env._true_target, wp.array(cur_tgt, device="cuda"))

            p   = env.model._state["position"].numpy()
            ast = env.model._actuator_states
            el  = ast["elevator"].numpy()
            th  = ast["throttle_left"].numpy()
            alt = -p[:, 2]

            for i in range(NTRACK):
                if not frozen[i]:
                    traj[i].append((p[i, 0], p[i, 1], alt[i], cur_tgt[i], el[i], th[i]))

            err_all[t] = np.where(dead, np.nan, np.abs(alt - cur_tgt))

            live = ~done & ~dead
            hit = (np.abs(alt - cur_tgt) < BAND) & live
            sc[hit] += 1; sc[~hit & live] = 0
            adv = (sc >= K) & live; idx[adv] += 1; sc[adv] = 0
            for i in range(NTRACK):
                if adv[i] and not frozen[i] and idx[i] < M:
                    inj[i].append((t, float(seq[i, idx[i]])))
            done |= idx >= M

            obs, _, d, _ = env.step(act(obs).clamp(-1.0, 1.0))
            dd = d.cpu().numpy()
            dead |= dd.astype(bool)
            for i in range(NTRACK):
                if dd[i] and not frozen[i]:
                    frozen[i] = True; endstep[i] = min(endstep[i], t)

        # settled RMSE: last 50 s of the recorded episodes that flew the whole horizon
        settled_rmse = []
        for i in range(NTRACK):
            tr = np.array(traj[i])
            if not frozen[i] and len(tr) > 5000:
                e = np.abs(tr[-5000:, 2] - tr[-5000:, 3])
                settled_rmse.append(np.sqrt(np.mean(e ** 2)))

        return dict(traj=traj, inj=inj, endstep=endstep, seq=seq,
                    err_all=err_all,
                    completion=(idx >= M).mean(), survival=1.0 - dead.mean(),
                    mean_legs=idx.clip(max=M).mean(),
                    settled_rmse=np.nanmean(settled_rmse) if settled_rmse else np.nan)


N_SHOW = 6        # episodes drawn in the tracking panel


def _altitude_rollout(algo, plane, ckpt_dir, results_dir):
    """The altitude Monte Carlo of `_altitude_run`, reduced to what the figure draws and cached
    under <results_dir>/traces, so the figure re-renders without flying. The per-episode error of
    all N episodes is kept as its 10/50/90 % quantiles over episodes at each step."""
    path = Path(results_dir) / "traces" / f"altitude_viz_{algo}_{plane}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    import warp as wp
    from falcons.envs.configs import PLANE_CONFIGS
    wp.init()
    d = _altitude_run(algo, plane, ckpt_dir)
    err = d.pop("err_all")
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # a step every episode has left
        d["err_q"] = np.nanpercentile(err, [10, 50, 90], axis=1).astype(np.float32)
    d["traj"] = [np.asarray(t, np.float32) for t in d["traj"]]
    d["ref_rate"] = float(PLANE_CONFIGS[plane]["ref_rate"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(d, f)
    return d


@_paper
def altitude_viz(algo, planes=PLANES, ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR):
    """Sequential altitude waypoints tracked by one learned policy (seed 0), over a 256-episode
    Monte Carlo per airframe. One row per airframe (a single airframe name gives a one-row
    figure), three panels per row:
      (a) commanded staircase (dashed) and achieved altitude of N_SHOW episodes; x = departure
      (b) |altitude error| over the N episodes, each up to its departure (if any): median and
          10-90 % band, log scale, with the +-1 m capture band
      (c) elevator and throttle of the longest of those episodes
    The run statistics (completion, captured targets, settled RMSE) are printed for the caption,
    not drawn. The rollout is cached (`_altitude_rollout`); a cold cache flies on the GPU."""
    planes = (planes,) if isinstance(planes, str) else tuple(planes)
    ds = [(p, _altitude_rollout(algo, p, ckpt_dir, results_dir)) for p in planes]
    col = METHOD_COLORS[algo.upper()]
    ep_c = plt.cm.viridis(np.linspace(0.0, 0.82, N_SHOW))
    tm = np.arange(H) * DT
    n = len(ds)
    fig = plt.figure(figsize=(TEXT_W, 0.45 + 1.68 * n), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.04)
    gs = fig.add_gridspec(n, 3, width_ratios=[1.15, 1.0, 0.9])
    letters = iter("abcdefghijkl")
    for r, (p, d) in enumerate(ds):
        axa = fig.add_subplot(gs[r, 0])
        axb = fig.add_subplot(gs[r, 1], sharex=axa)
        sub = gs[r, 2].subgridspec(2, 1)
        axe = fig.add_subplot(sub[0], sharex=axa)
        axt = fig.add_subplot(sub[1], sharex=axa)

        # (a) commanded staircase and achieved altitude
        shown = [t for t in d["traj"] if len(t) > 50][:N_SHOW]
        for tr, c in zip(shown, ep_c):
            t = np.arange(len(tr)) * DT
            axa.plot(t, tr[:, 3], color=c, lw=0.8, ls=(0, (3.5, 1.5)))
            axa.plot(t, tr[:, 2], color=c, lw=1.1)
            if len(tr) < H - 1:                       # the episode departed before the horizon
                axa.plot(t[-1], tr[-1, 2], "x", color=c, ms=4.5, mew=1.2)
        axa.set_ylabel("altitude [m]")
        _letter(axa, f"({next(letters)})  {PRETTY.get(p, p.replace('_', ' '))}")

        # (b) Monte Carlo error: median and 10-90 % band over the N episodes, log scale
        lo, med, hi = (np.maximum(q, 1e-3) for q in d["err_q"])
        axb.fill_between(tm, lo, hi, color=col, alpha=0.22, lw=0)
        axb.plot(tm, med, color=col, lw=1.1)
        axb.axhline(BAND, color="0.15", ls=":", lw=1.0)
        axb.set_yscale("log")
        axb.set_ylim(max(1e-3, 0.7 * float(np.nanmin(lo))), 1.4 * float(np.nanmax(hi)))
        axb.set_ylabel("|altitude error| [m]")
        _letter(axb, f"({next(letters)})")

        # (c) controls of one episode, two thin stacked axes
        # the longest of the shown episodes, in its colour from (a)
        k = max(range(len(shown)), key=lambda i: len(shown[i]))
        tr = shown[k]
        t = np.arange(len(tr)) * DT
        axe.plot(t, tr[:, 4], color=ep_c[k], lw=0.8)
        axt.plot(t, tr[:, 5], color=ep_c[k], lw=0.8)
        axe.set_ylabel("elevator\n[deg]")
        axt.set_ylabel("throttle")
        axt.set_ylim(-0.03, 1.03); axt.set_yticks([0, 0.5, 1]); axt.set_yticklabels(["0", ".5", "1"])
        axe.tick_params(labelbottom=False)
        _letter(axe, f"({next(letters)})")

        for a in (axa, axb, axe, axt):
            a.grid(True)
        for a in (axa, axb, axt):
            if r < n - 1:
                a.tick_params(labelbottom=False)
            else:
                a.set_xlabel("time [s]")
        axa.set_xlim(0, H * DT)
        axa.xaxis.set_major_locator(MultipleLocator(30))
        print(f"  {algo}/{p}: survival {d.get('survival', float('nan')) * 100:.0f}%, "
              f"completion {d['completion'] * 100:.0f}%, captured "
              f"{d['mean_legs']:.2f}/{M}, settled RMSE {d['settled_rmse']:.3f} m "
              f"(N={N}, ref rate {d.get('ref_rate', float('nan')):.1f} m/s)")

    fig.legend(handles=[
        Line2D([], [], color="0.3", lw=1.1, label="achieved"),
        Line2D([], [], color="0.3", lw=0.8, ls=(0, (3.5, 1.5)), label="commanded"),
        Line2D([], [], color="0.3", lw=0, marker="x", ms=4.5, mew=1.2, label="departure"),
        Line2D([], [], color=col, lw=1.1, label=f"median |error| ({N} episodes)"),
        Patch(color=col, alpha=0.22, lw=0, label="10–90 %"),
        Line2D([], [], color="0.15", lw=1.0, ls=":", label=f"±{BAND:g} m capture band")],
        loc="outside upper center", ncol=3, handlelength=1.6, columnspacing=1.0)
    name = f"fig_altitude_viz_{algo}" if n > 1 else f"fig_altitude_viz_{algo}_{planes[0]}"
    return _save(fig, results_dir, name)


# ───────────────────────────── attitude executor ─────────────────────────────
AE_N = 64          # demo batch: the fan of commanded bank angles
AE_H = 4500        # 45 s (a ~30 deg-bank turn is about one full circle in 30 s)
CH = [(3, 4, "$\\phi$", "$^\\circ$"), (5, 6, "$\\dot h$", "m/s"),
      (7, 8, "$V_a$", "m/s")]               # (achieved col, commanded col, symbol, unit)
CH_C = ["#2a78d6", "#008300", "#e8601c"]


def _attitude_demo(plane, ckpt_dir):
    """The scripted demo episodes behind the command-interface figure: each episode holds a coordinated
    turn at its own bank angle (hdot* = 0, so altitude is held) for the first two thirds, then
    levels and climbs. PPO seed 0 drives it.

    Columns per step, as recorded:
    (N, E, alt, roll_deg, phi_cmd_deg, hdot, hdot_cmd, Va, Va_cmd, elevator, aileron, rudder,
     throttle).
    """
    import torch
    import warp as wp
    from falcons.envs.attitude import AttitudeEnv
    from falcons.envs.configs import attitude_env_cfg
    from falcons.controllers.policies import load_policy

    def rolldeg(q):
        qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return np.degrees(np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2)))

    def climb_rate(q, vbody):
        # world vertical speed vz = (R(q) v_body)[2]; climb rate = -vz  (NED z-down)
        t = 2.0 * np.cross(q[:, :3], vbody)
        vw = vbody + q[:, 3:4] * t + np.cross(q[:, :3], t)
        return -vw[:, 2]

    with torch.no_grad():
        env = AttitudeEnv(AE_N, "cuda", attitude_env_cfg(plane, horizon=AE_H + 1))
        act = load_policy("ppo", "attitude", plane, 0, ckpt_dir=ckpt_dir).act

        banks = np.linspace(-env.phi_max_full, env.phi_max_full, AE_N).astype(np.float32)
        climb = min(1.0, env.hdot_max_full)                   # gentle climb within envelope
        zero = np.zeros(AE_N, np.float32)
        phi3 = np.stack([banks, banks, zero], axis=1)
        hd3 = np.stack([zero, zero, np.full(AE_N, climb, np.float32)], axis=1)
        va3 = np.full((AE_N, 3), env.base_vel, np.float32)

        env.reset()
        wp.copy(env._phi3, wp.array(phi3, device="cuda"))
        wp.copy(env._hdot3, wp.array(hd3, device="cuda"))
        wp.copy(env._va3, wp.array(va3, device="cuda"))
        env._select_launch(); env._obs_launch()
        obs = wp.to_torch(env._obs)

        # record NTRACK episodes SPREAD across the bank fan (so both left and right turns show)
        track_idx = np.linspace(0, AE_N - 1, NTRACK).astype(int)
        frozen = np.zeros(NTRACK, bool)
        traj = [[] for _ in range(NTRACK)]
        for t in range(AE_H):
            s = env.model._state
            p = s["position"].numpy(); q = s["orientation"].numpy(); vb = s["linear_vel"].numpy()
            ast = env.model._actuator_states
            roll = rolldeg(q); climb_r = climb_rate(q, vb)
            Va = env.model._Va.numpy()
            phi_cmd = np.degrees(wp.to_torch(env._phi_tgt).cpu().numpy())
            hdot_cmd = wp.to_torch(env._hdot_tgt).cpu().numpy()
            va_cmd = wp.to_torch(env._va_tgt).cpu().numpy()
            el = ast["elevator"].numpy(); ai = ast["aileron"].numpy()
            ru = ast["rudder"].numpy(); th = ast["throttle_left"].numpy()
            for k, i in enumerate(track_idx):
                if not frozen[k]:
                    traj[k].append((p[i, 0], p[i, 1], -p[i, 2], roll[i], phi_cmd[i],
                                    climb_r[i], hdot_cmd[i], Va[i], va_cmd[i],
                                    el[i], ai[i], ru[i], th[i]))
            obs, _, d, _ = env.step(act(obs).clamp(-1.0, 1.0))
            dd = d.cpu().numpy()
            for k, i in enumerate(track_idx):
                if dd[i]:
                    frozen[k] = True
    # np.array over the recorded tuples, with no dtype of its own: every field is a float32 scalar
    # off a warp array, so the row array is float32
    return [np.array(t) for t in traj]


def _attitude_rollout(plane, ckpt_dir, results_dir):
    """Recorded demo episodes for `plane`, cached on disk so a shipped cache replays without
    flying. The headline numbers are computed from the demo itself (`rob` is the fraction of the
    recorded episodes that ran the whole horizon); they are printed, never drawn."""
    path = Path(results_dir) / "traces" / f"attitude_exec_{plane}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    traj = _attitude_demo(plane, ckpt_dir)
    kept = [t for t in traj if len(t) > 50]
    rmse = lambda ia, ic: float(np.sqrt(np.mean(np.concatenate(
        [(t[:, ia] - t[:, ic]) ** 2 for t in kept]))))
    out = dict(traj=kept, dt=DT, rob=float(np.mean([len(t) >= AE_H for t in traj])),
               phi=rmse(3, 4), hdot=rmse(5, 6), va=rmse(7, 8))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def _fan(ax, traj):
    """Ground tracks over the commanded bank fan, coloured by achieved bank."""
    for tr in traj:
        pts = tr[:, :2].reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="coolwarm", lw=1.0,
                            norm=plt.Normalize(-BANK_LIM, BANK_LIM))
        lc.set_array(tr[:-1, 3])
        ax.add_collection(lc)
    ax.scatter([traj[0][0, 0]], [traj[0][0, 1]], c="k", s=14, marker="^", zorder=5)
    # Frame on the turns. The shallowest commanded banks fly almost straight and leave the area the
    # turning episodes occupy; autoscaling to them shrinks every circle to a dot. The radius of the
    # widest CLOSED turn sets the window, and the shallow tracks simply run out of frame.
    closed = [t for t in traj if np.hypot(*(t[-1, :2] - t[0, :2])) < 0.5 * np.ptp(t[:, :2])]
    span = max((np.abs(t[:, :2] - traj[0][0, :2]).max() for t in (closed or traj)), default=1.0)
    ax.set_xlim(traj[0][0, 0] - 1.15 * span, traj[0][0, 0] + 1.15 * span)
    ax.set_ylim(traj[0][0, 1] - 1.15 * span, traj[0][0, 1] + 1.15 * span)
    ax.set_aspect("equal")
    ax.set_xlabel("North [m]"); ax.set_ylabel("East [m]")
    ax.grid(True)


def _channels(ax, traj, dt):
    """The three commanded channels of one episode, each in its own lane.

    Lanes are half-height (traces span at most +-0.5 of the 2.0 lane pitch), which keeps a clear
    strip above every lane for its label whatever shape the channel takes, and each is normalised
    by its own range because the three carry different units. The label carries the physical
    numbers the normalisation removes.
    """
    # The steepest commanded turn that runs the whole episode. The fan spans the airframe's full
    # bank envelope and its outermost members depart within seconds, so picking on commanded bank
    # alone lands on a two-second trace.
    full = max(len(t) for t in traj)
    mid = max([t for t in traj if len(t) > 0.95 * full], key=lambda t: abs(t[len(t) // 4, 4]))
    t = np.arange(len(mid)) * dt
    for lane, ((ia, ic, lab, unit), col) in enumerate(zip(CH, CH_C)):
        a, c = mid[:, ia], mid[:, ic]
        span = max(np.ptp(np.concatenate([a, c])), 1e-6)
        base, ref = 2.0 * (2 - lane), np.median(c)
        ax.plot(t, base + 0.5 * (a - ref) / span, color=col, lw=1.1)
        ax.plot(t, base + 0.5 * (c - ref) / span, color=col, lw=0.9, ls="--", alpha=0.75)
        lo, hi = float(c[len(c) // 4]), float(c[-1])
        ax.text(0.0, base + 0.62, f"{lab}: {lo:.0f}$\\rightarrow${hi:.0f} {unit}",
                color=col, fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("time [s]")
    ax.set_yticks([]); ax.set_ylim(-0.9, 5.7)
    ax.grid(True, axis="x")
    ax.spines["left"].set_visible(False)


def _colour_key(fig, ax):
    """Bank-angle key inside the fan panel, where it cannot be read as a label belonging to the
    panel beside it."""
    cax = ax.inset_axes([0.05, 0.10, 0.36, 0.04])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="coolwarm",
                                            norm=plt.Normalize(-BANK_LIM, BANK_LIM)),
                      cax=cax, orientation="horizontal", ticks=[-BANK_LIM, 0, BANK_LIM])
    cb.set_label("bank [deg]", fontsize=8, labelpad=1)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=8, pad=1, length=2)
    cb.outline.set_linewidth(0.4)


@_paper
def attitude_exec(planes, ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR):
    """The command interface itself: (A) top-down ground tracks over a fan of commanded bank angles
    spanning the airframe's envelope, coloured by achieved bank -- the executor as a continuum, not
    four demos; (B) all three commanded channels (phi*, hdot*, Va*) against what was achieved, on
    one time axis, each normalised by its own commanded step so they share a vertical scale.

    One airframe -> a half-width two-panel figure. Two -> the airframes side by side across the
    page width, each contributing its fan and its channel lanes."""
    ds = [(ac, _attitude_rollout(ac, ckpt_dir, results_dir)) for ac in planes]
    n = len(ds)
    # constrained_layout, not tight_layout: the fan panels carry an equal aspect ratio, so their
    # drawn box is smaller than the cell they sit in, and tight_layout places the neighbouring
    # x-labels against the nominal cell instead and pushes them off the canvas.
    fig, ax = plt.subplots(1, 2 * n, figsize=(TEXT_W if n > 1 else HALF_W, 2.75 if n > 1 else 2.3),
                           layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.05, hspace=0.0)
    for k, (ac, d) in enumerate(ds):
        _fan(ax[2 * k], d["traj"])
        _channels(ax[2 * k + 1], d["traj"], d["dt"])
        if n > 1:                        # name the pair, since two airframes now share the figure
            ax[2 * k].set_title(PRETTY.get(ac, ac.replace("_", " ")), fontsize=8, loc="left",
                                pad=3)
            if k:
                ax[2 * k].set_ylabel("")
        print(f"  {ac}: rob {d['rob']:.3f}, phi {d['phi']:.2f} deg, "
              f"hdot {d['hdot']:.2f} m/s, Va {d['va']:.2f} m/s")
    # one bank key for the whole figure, under it: inside a fan panel it covered the tracks
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="coolwarm",
                                            norm=plt.Normalize(-BANK_LIM, BANK_LIM)),
                      ax=list(ax), location="bottom", shrink=0.32, aspect=30, pad=0.02,
                      ticks=[-BANK_LIM, -BANK_LIM / 2, 0, BANK_LIM / 2, BANK_LIM])
    cb.set_label("achieved bank [deg]  (< 0 left, > 0 right)", labelpad=1)
    cb.outline.set_linewidth(0.4)
    name = f"fig_attitude_exec_{planes[0]}" if n == 1 else "fig_attitude_exec"
    return _save(fig, results_dir, name)


# ───────────────────────────── maneuver small multiples ─────────────────────────────
KDASH = "0.15"        # commanded dashed line
BAND_GREY = "0.6"     # tolerance-band fill


def _maneuver_runs(plane, ckpt_dir, results_dir):
    """Every (maneuver, method) cell of one airframe, read back from the rollout cache
    falcons.benchmark.maneuvers writes. Nothing re-flies while that cache is warm."""
    from falcons.benchmark.maneuvers import SHAPES, cell, controllers
    from falcons.envs.configs import attitude_env_cfg
    tol_deg = np.degrees(attitude_env_cfg(plane).get("sig_phi", 0.15))   # tracking tolerance band
    methods = list(controllers(plane, ckpt_dir))
    cache = Path(results_dir) / "traces"
    runs = {sh: {m: cell(plane, sh, m, ckpt_dir=ckpt_dir, cache_dir=cache, refresh=False,
                         tol_deg=tol_deg) for m in methods} for sh in SHAPES}
    return runs, methods, tol_deg


@_paper
def maneuver_facet(plane, ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR):
    """Small-multiples grid of BANK tracking: rows = maneuvers, columns = methods, one trace per
    cell over the commanded reference + tolerance band. This is the section's only time-series
    figure. Above each panel: bank nRMSE / time in the tolerance band of the trace drawn (or the
    time of the crash); the key above the grid says so.

    Why not the overlay: SAC and TD3 are bang-bang on these streams (ReLU actors, no CAPS in the
    loss), so on a shared axis their traces cover the three tight trackers entirely. Faceting keeps
    every method at full opacity and turns the jitter into a legible per-method finding.

    Why bank only: the climb channel is commanded non-zero in the helix alone (streams.py) --- the
    other three maneuvers command hdot* = 0 throughout, so a climb facet spends fifteen of twenty
    panels showing a hold at zero. Its one finding, that SAC and TD3 inject 1-3 m/s of vertical
    oscillation while tracking bank, is the hdot_rmse column of the table.
    """
    from falcons.benchmark.maneuvers import SHAPES, time_in_band
    runs_by_shape, methods, tol_deg = _maneuver_runs(plane, ckpt_dir, results_dir)
    key, cmd = "roll", "phi_cmd"
    fig, ax = plt.subplots(len(SHAPES), len(methods), figsize=(TEXT_W, 5.0),
                           sharex="row", sharey="row", squeeze=False, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.03)
    for i, sh in enumerate(SHAPES):
        runs = runs_by_shape[sh]
        ref = max((runs[m]["trace"] for m in methods
                   if runs[m]["trace"] is not None and len(runs[m]["trace"]["t"])),
                  key=lambda d: len(d["t"]), default=None)
        for j, m in enumerate(methods):
            a = ax[i, j]
            if ref is not None:
                a.fill_between(ref["t"], ref[cmd] - tol_deg, ref[cmd] + tol_deg,
                               color=BAND_GREY, alpha=0.25, lw=0)
                a.plot(ref["t"], ref[cmd], color=KDASH, lw=0.9, ls=(0, (3, 1.5)))
            r = runs[m]
            tr = r["trace"]
            if tr is not None and len(tr["t"]):
                a.plot(tr["t"], tr[key], color=METHOD_COLORS[m], lw=0.9)
            if r["survival"] == 0:
                note, nc = f"crash @ {r['t_fail']:.0f} s", "#a33"
            elif r["phi"] is not None:
                nr = r["phi"]["nrmse"]
                note = (f"{nr * 100:.0f}% / " if nr is not None else "") \
                    + f"{time_in_band(tr, tol_deg) * 100:.0f}%"
                nc = "0.2"
            else:
                note, nc = "", "0.2"
            a.set_title(note, loc="right", fontsize=8, color=nc, pad=2)
            if i == 0:
                a.set_title(m, loc="left", fontsize=8, fontweight="bold",
                            color=METHOD_COLORS[m], pad=2)
            if j == 0:
                a.set_ylabel(SHAPE_PRETTY.get(sh, sh))
            a.grid(True)
    fig.supylabel("bank [deg]")
    fig.supxlabel("time [s]")
    fig.legend(handles=[
        Line2D([], [], color=KDASH, lw=0.9, ls=(0, (3, 1.5)), label="commanded"),
        Patch(color=BAND_GREY, alpha=0.25, lw=0, label=f"±{tol_deg:.1f}° tolerance band"),
        Line2D([], [], color="none", label="above panels: bank nRMSE / time in band")],
        loc="outside upper center", ncol=3, handlelength=1.8)
    return _save(fig, results_dir, f"fig_maneuver_facet_{plane}")


# ───────────────────────────── ground tracks ─────────────────────────────
def _ppo_track(plane, maneuver, ckpt_dir, results_dir):
    """The PPO seed-0 rollout of one cell, with the flight path the simulator actually flew, or
    None when there is no track to draw.

    `cell` owns the rollout cache and its file naming, so it is asked first -- and then the seed-0
    entry is read back out of the cache rather than taken as `cell` returned it: a cell flown over
    several training seeds comes back aggregated, and the trace an aggregate keeps is the
    median-RMSE seed. This figure draws one flight per panel, and which flight that is should not
    depend on how the other seeds happened to score.

    A run cached before `fly` recorded `pos`/`yaw` carries every channel the table scores and no
    path; rather than rewrite that cache -- the table reproduces itself from it byte for byte, and
    the shipped trace set is exactly such a cache -- that cell is re-flown at PPO seed 0 and used
    in memory. Nothing is written back.
    """
    from falcons.benchmark.maneuvers import ALGO, cell, controllers, fly
    from falcons.envs.configs import attitude_env_cfg
    tol_deg = np.degrees(attitude_env_cfg(plane).get("sig_phi", 0.15))
    cache = Path(results_dir) / "traces"
    entries = controllers(plane, ckpt_dir)["PPO"]
    suffix, seed0 = next(((s, f) for s, f in entries if s == ""), entries[0])   # "" is seed 0
    cell(plane, maneuver, "PPO", ckpt_dir=ckpt_dir, cache_dir=cache, refresh=False,
         tol_deg=tol_deg)                                       # fills the cache; nothing re-flies
    with open(cache / f"{plane}_{maneuver}_{ALGO['PPO']}{suffix}.pkl", "rb") as f:
        got = pickle.load(f)
    trace = (got[0] if isinstance(got, list) else got).get("trace")
    if trace is None or "pos" not in trace:
        print(f"  {plane}/{maneuver}: cached run predates the pos channel -- re-flying PPO seed 0")
        trace = fly(plane, maneuver, seed0)["trace"]
    # a run that departed on its first step leaves an empty trace: there is nothing to plot
    return trace if trace is not None and np.asarray(trace["pos"]).shape[0] else None


def _nice(x):
    """The largest 1-2-2.5-5 x 10^k length not above x: a scale-bar length."""
    k = 10.0 ** np.floor(np.log10(x))
    return max(m * k for m in (1, 2, 2.5, 5) if m * k <= x)


def _ground_track(ax, tr, norm):
    """One maneuver from above: east right, north up, equal scale, coloured by achieved bank, with
    its start (open circle) and end (filled square) and a scale bar in place of tick labels."""
    pos = np.asarray(tr["pos"])
    north, east, roll = pos[:, 0], pos[:, 1], np.asarray(tr["roll"])
    pts = np.column_stack([east, north]).reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap="coolwarm", norm=norm,
                        lw=1.4, capstyle="round")
    lc.set_array(roll[:-1])
    ax.add_collection(lc)
    ax.plot(east[0], north[0], "o", ms=4.5, mfc="white", mec="k", mew=0.9, zorder=5)
    ax.plot(east[-1], north[-1], "s", ms=3.8, color="k", zorder=5)
    span = max(np.ptp(east), np.ptp(north), 1.0)
    pad = 0.08 * span
    ax.set_xlim(east.min() - pad, east.max() + pad)
    # extra room at the bottom for the scale bar, below the track
    ax.set_ylim(north.min() - pad - 0.16 * span, north.max() + pad)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    # scale bar, lower right, about a quarter of the track's extent
    L = _nice(0.3 * span)
    x1, y0 = east.max() + pad, north.min() - pad - 0.12 * span
    ax.plot([x1 - L, x1], [y0, y0], color="k", lw=1.6, solid_capstyle="butt")
    ax.text(x1 - L / 2, y0 + 0.015 * span, f"{L:g} m", ha="center", va="bottom", fontsize=8)
    alt = -pos[:, 2]
    return float(alt[-1] - alt[0])


@_paper
def maneuver_render(planes, ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR):
    """Qualitative strip for the maneuver section: the top-down ground track of every named
    maneuver flown by PPO (seed 0), coloured by achieved bank, one row per airframe and one column
    per maneuver, one shared bank-angle colourbar. Each panel is scaled to its own track, so each
    carries a scale bar, and the altitude gained over the maneuver (Δh), which a top-down view hides.
    Drawn straight from the recorded rollouts in the maneuver cache."""
    from falcons.benchmark.maneuvers import SHAPES
    norm = plt.Normalize(-BANK_LIM, BANK_LIM)
    fig, ax = plt.subplots(len(planes), len(SHAPES), figsize=(TEXT_W, 1.45 + 0.95 * len(planes)),
                           squeeze=False, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
    for r, ac in enumerate(planes):
        for c, sh in enumerate(SHAPES):
            a = ax[r, c]
            tr = _ppo_track(ac, sh, ckpt_dir, results_dir)
            if r == 0:
                a.set_title(SHAPE_PRETTY.get(sh, sh), fontsize=8, pad=3)
            if c == 0:
                a.set_ylabel(PRETTY.get(ac, ac.replace("_", " ")), fontsize=8)
            if tr is None:
                a.set_xticks([]); a.set_yticks([])
                a.text(0.5, 0.5, "departed", transform=a.transAxes, ha="center", fontsize=8)
                continue
            dh = _ground_track(a, tr, norm)
            a.text(0.02, 0.98, f"Δh {dh:+.0f} m".replace("-", "\u2212"), transform=a.transAxes,
                   ha="left", va="top",
                   fontsize=8)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="coolwarm", norm=norm), ax=ax, location="right",
                      shrink=0.75, aspect=22, pad=0.01, ticks=[-50, -25, 0, 25, 50])
    cb.set_label("bank [deg]  (< 0 left, > 0 right)")
    cb.outline.set_linewidth(0.4)
    fig.legend(handles=[
        Line2D([], [], ls="", marker="o", ms=4.5, mfc="white", mec="k", mew=0.9, label="start"),
        Line2D([], [], ls="", marker="s", ms=3.8, color="k", label="end")],
        loc="outside lower center", ncol=2)
    return _save(fig, results_dir, "fig_maneuver_render")


# ───────────────────────────── ground effect ─────────────────────────────
RATIOS = [0.25, 0.5, 1.0, 2.0, 3.0, 6.0]


@_paper
def ge_energy(results_dir=RESULTS_DIR):
    """Full-width figure, the two panels side by side. Left: change in thrust from ground effect,
    for the closed-loop policy (solid), for the open-loop trim of `ground_effect.run_trim`
    (dash-dot, every airframe at every band because it needs no policy) and for the lifting-line
    prediction (dotted). All three are thrust ratios, so they sit on one axis; the throttle-squared
    control-effort metric stays in the CSV but is not plotted against the prediction, being a
    different quantity. Right: tracking accuracy of the two arms, log-scaled because the three
    airframes' errors span two decades. No titles: the caption carries the description.

    Drawn at its final printed width so it is included at \\textwidth without rescaling; scaling
    a figure rescales every font in it.
    """
    d = Path(results_dir)
    with open(d / "ge_energy.csv") as f:
        rows = [{k: (v if k == "aircraft" else float(v)) for k, v in r.items()}
                for r in csv.DictReader(f)]

    trim = {}
    tp = d / "ge_trim.csv"
    if tp.exists():
        with open(tp) as f:
            for r in csv.DictReader(f):
                trim.setdefault(r["aircraft"], []).append(
                    (float(r["h_over_b"]), float(r["delta_thrust_pct"]), float(r["theory_pct"])))

    fig, ax = plt.subplots(1, 2, figsize=(TEXT_W, 2.45), sharex=True, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, wspace=0.06)
    planes = []
    for ac in dict.fromkeys(r["aircraft"] for r in rows):
        c = PLANE_COLORS.get(ac)
        rs = [r for r in rows if r["aircraft"] == ac and r["hold_valid"]]
        if not rs:
            continue          # no policy holds any band: its trim curve would have no counterpart
        planes.append((ac, c, rs[0]["span"]))
        x = [r["h_over_b"] for r in rs]
        ax[0].plot(x, [r["delta_thrust_pct"] for r in rs], "o-", color=c, lw=1.4, ms=3)
        ax[1].plot(x, [r["rmse_ge_on"] for r in rs], "o-", color=c, lw=1.4, ms=3)
        ax[1].plot(x, [r["rmse_ge_off"] for r in rs], "o--", color=c, lw=1.0, ms=2.5, alpha=0.6)
        t = sorted(trim.get(ac, []))
        if t:
            ax[0].plot([v[0] for v in t], [v[1] for v in t], "-.", color=c, lw=1.2)
            ax[0].plot([v[0] for v in t], [v[2] for v in t], ":", color=c, lw=1.0, alpha=0.8)
    ax[0].axhline(0, color="0.5", lw=0.8)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    # Label the bands themselves. The default log locator puts a single decade tick at h/b = 1
    # across this 0.25-6 range, which leaves the reader no way to place the deep-ground-effect end.
    ax[1].set_xticks(RATIOS); ax[1].set_xticklabels([f"{r:g}" for r in RATIOS])
    ax[1].set_xticks([], minor=True)
    for a in ax:
        a.set_xlabel("height over span  $h/b$", fontsize=9)
    ax[0].set_ylabel("thrust change [%]", fontsize=9)
    ax[1].set_ylabel("settled altitude RMSE [m]", fontsize=9)
    ax[1].set_ylim(top=ax[1].get_ylim()[1] * 3.2)      # headroom for the airframe key
    # Two keys, one per panel: the left names the three measurements, the right the airframes,
    # so neither panel carries a legend for something it does not draw.
    ax[0].legend(handles=[Line2D([], [], color="0.25", lw=1.4, marker="o", ms=3, label="policy"),
                          Line2D([], [], color="0.25", lw=1.2, ls="-.", label="open-loop trim"),
                          Line2D([], [], color="0.25", lw=1.0, ls=":", label="lifting-line")],
                 fontsize=8, frameon=False, loc="lower right", handlelength=2.0,
                 borderpad=0.1, labelspacing=0.25)
    ax[1].legend(handles=[Line2D([], [], color=c, lw=1.4,
                                 label=f"{a.replace('_', ' ')} ($b$={b:.2f} m)")
                          for a, c, b in planes]
                 + [Line2D([], [], color="0.25", lw=1.0, ls="--", alpha=0.6, label="GE off")],
                 fontsize=8, frameon=False, loc="upper left", handlelength=2.0,
                 borderpad=0.1, labelspacing=0.25)
    for a in ax:
        a.grid(True)
        a.tick_params(labelsize=8)
    return _save(fig, results_dir, "fig_ge_energy")


# ───────────────────────────── robustness ─────────────────────────────
X = ["nom", "mild", "medium", "severe"]
# Order is the finding: noise costs nothing, the two latencies cost the same, wind is the wall.
FAM = {"obs_noise": ("sensor noise", "#2a78d6"), "sensor_delay": ("sensor delay", "#008300"),
       "action_delay": ("action delay", "#e8601c"), "wind": ("turbulence", "#d1495b")}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


@_paper
def robustness_ppo(results_dir=RESULTS_DIR):
    """Full-width robustness figure: PPO only, both airframes, one line per disturbance family
    against severity. `benchmark.robustness` writes the full sweep; this figure needs the compact
    version, with the families as lines so that the ordering between them is the thing the reader
    sees. When the CSV carries the bootstrap interval of the bank RMSE (phi_rmse_lo/_hi), it is
    drawn as error bars, the families nudged apart along x so the bars do not overlap. Reads the
    stored CSV, so it never re-flies anything."""
    with open(Path(results_dir) / "robustness.csv") as f:
        rs = [r for r in csv.DictReader(f) if r["method"] == "PPO"]
    idx = {(r["aircraft"], r["disturbance"], r["severity"]): r for r in rs}
    has_ci = any(np.isfinite(_num(r.get("phi_rmse_lo"))) for r in rs)

    fig, ax = plt.subplots(1, 2, figsize=(TEXT_W, 2.45), sharey=True)
    for a, ac in zip(ax, PRETTY):
        nom = idx[(ac, "nominal", "-")]
        for k, (fam, (lab, c)) in enumerate(FAM.items()):
            pts = [(0, nom)]
            for i, sev in enumerate(["mild", "medium", "severe"], start=1):
                r = idx.get((ac, fam, sev))
                if r is None or not np.isfinite(_num(r["phi_rmse"])):
                    continue                      # departed: no tracking metric to plot
                pts.append((i, r))
            xs = np.array([i for i, _ in pts], float)
            ys = np.array([_num(r["phi_rmse"]) for _, r in pts])
            if has_ci:
                lo = np.array([_num(r.get("phi_rmse_lo")) for _, r in pts])
                hi = np.array([_num(r.get("phi_rmse_hi")) for _, r in pts])
                err = np.vstack([np.where(np.isfinite(lo), ys - lo, 0.0),
                                 np.where(np.isfinite(hi), hi - ys, 0.0)])
                a.errorbar(xs + 0.07 * (k - 1.5), ys, yerr=np.clip(err, 0, None), fmt="o-",
                           color=c, lw=1.4, ms=3.5, elinewidth=0.9, capsize=1.8, label=lab)
            else:
                a.plot(xs, ys, "o-", color=c, lw=1.4, ms=3.5, label=lab)
        # Survival is the second axis of the result. Rather than spend a panel on it, flag the cells
        # that lost runs along the top of the plot, one row per family so flags at the same
        # severity never collide; a line that simply stops is a family whose runs all departed,
        # with no RMSE left to draw, and its flag says so.
        flagged = [(fam, c) for fam, (_, c) in FAM.items()
                   if any(_num(idx[(ac, fam, sv)]["survival"]) < 1.0
                          for sv in ("mild", "medium", "severe") if (ac, fam, sv) in idx)]
        for row, (fam, c) in enumerate(flagged):
            for i, sev in enumerate(["mild", "medium", "severe"], start=1):
                r = idx.get((ac, fam, sev))
                sv = _num(r["survival"]) if r else 1.0
                if sv >= 1.0:
                    continue
                txt = "all departed" if sv == 0.0 else \
                    f"s={sv:.2f}" if round(sv, 2) < 1.0 else f"s={sv:.3f}"
                a.text(i, 0.97 - 0.085 * row, txt, transform=a.get_xaxis_transform(),
                       ha="center", va="top", fontsize=8, color=c)
        a.set_title(PRETTY[ac], fontsize=9, pad=3)
        a.set_xticks(range(4)); a.set_xticklabels(X, fontsize=8)
        a.set_xlim(-0.3, 3.3)
        a.set_xlabel("severity", fontsize=9)
        a.tick_params(axis="y", labelsize=8)
        a.grid(True)
    ax[0].set_ylabel("bank RMSE [deg]", fontsize=9)
    # headroom: the flag rows take the top 30 % of the panels, the data stays below them
    top = np.nanmax([_num(r.get("phi_rmse_hi")) if np.isfinite(_num(r.get("phi_rmse_hi")))
                     else _num(r["phi_rmse"]) for r in rs])
    ax[0].set_ylim(0, 1.45 * top)
    # The survival flags take the top of both panels; the Ranger's centre left is empty at every
    # severity.
    ax[1].legend(fontsize=8, frameon=False, loc="center left", handlelength=2.0,
                 borderpad=0.1, labelspacing=0.3)
    fig.tight_layout(pad=0.4)
    return _save(fig, results_dir, "fig_robustness_ppo")


# ───────────────────────────── throughput ─────────────────────────────
NEURALPLANE = (1_048_576, 5.0e6)      # (envs, env-steps/s) as reported by NeuralPlane
BACKENDS = {"warp": ("Warp, GPU", "#2a78d6", "o", "-"),
            "torch": ("PyTorch, GPU", "#e8601c", "s", "--"),
            "cpu": ("NumPy, CPU (1 env)", "0.25", "^", "-")}


@_paper
def throughput(results_dir=RESULTS_DIR):
    """Half-width: simulator throughput (env-steps/s) against the number of parallel environments,
    log-log, one line per backend of `benchmark.throughput`, error bars = the 95 % CI of the
    repeated timings, and NeuralPlane's reported figure as a reference point."""
    with open(Path(results_dir) / "throughput.csv") as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(HALF_W, 2.4), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    present = dict.fromkeys(r["backend"] for r in rows)
    for b in [k for k in BACKENDS if k in present] + [k for k in present if k not in BACKENDS]:
        lab, c, mk, ls = BACKENDS.get(b, (b, "0.5", "o", "-"))
        rs = sorted((r for r in rows if r["backend"] == b), key=lambda r: int(r["n_envs"]))
        x = np.array([int(r["n_envs"]) for r in rs])
        y = np.array([float(r["steps_per_s_mean"]) for r in rs])
        e = np.array([float(r["ci95"]) for r in rs])
        ax.errorbar(x, y, yerr=e, color=c, marker=mk, ms=3.5, ls=ls, lw=1.1, elinewidth=0.9,
                    capsize=2, label=lab)
    ax.plot(*NEURALPLANE, marker="*", ms=8, color="k", ls="")
    ax.annotate("NeuralPlane\n(reported)", NEURALPLANE, xytext=(0, -7),
                textcoords="offset points", ha="center", va="top", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.5, 4e6)
    ax.set_xlabel("parallel environments")
    ax.set_ylabel("env steps / s")
    ax.grid(True, which="major")
    ax.legend(loc="upper left")
    return _save(fig, results_dir, "fig_throughput")


# ───────────────────────────── learning curves ─────────────────────────────
# task -> (curve column, axis label, log y): the metric each task's evaluation reports
CURVE_METRIC = {"altitude": ("rmse", "altitude RMSE [m]", True),
                "attitude": ("phi_rmse", "bank RMSE [deg]", False)}
ROB_MIN = 0.5         # below this survival rate the error of an evaluation is marked


def _read_curves(curves_dir):
    """{(algo, task, plane): [ {column: array}, ... one per seed ]} from <algo>_<task>_<plane>_s<k>.csv"""
    runs = {}
    for f in sorted(Path(curves_dir).glob("*_s*.csv")):
        algo, task, rest = f.stem.split("_", 2)
        plane, _seed = rest.rsplit("_s", 1)
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        cols = {k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows])
                for k in rows[0]}
        order = np.argsort(cols["env_steps"], kind="stable")
        runs.setdefault((algo, task, plane), []).append({k: v[order] for k, v in cols.items()})
    return runs


def _seed_band(runs, key):
    """IQM over seeds with its 95 % bootstrap interval at every evaluation point. Seeds whose
    evaluation grids differ are interpolated (in log steps) onto the union of the grids, over the
    range every seed covers."""
    from falcons.benchmark.stats import bootstrap, iqm
    lo_x = max(r["env_steps"][0] for r in runs)
    hi_x = min(r["env_steps"][-1] for r in runs)
    grid = np.unique(np.concatenate([r["env_steps"] for r in runs]))
    grid = grid[(grid >= lo_x) & (grid <= hi_x)]
    lg = np.log(grid)
    Y = np.array([np.interp(lg, np.log(r["env_steps"]), r[key]) for r in runs])
    R = np.array([np.interp(lg, np.log(r["env_steps"]), r["rob"]) for r in runs])
    est, lo, hi = np.array([bootstrap(Y[:, j], n_boot=2000) for j in range(len(grid))]).T
    rob = np.array([iqm(R[:, j]) for j in range(len(grid))])
    return grid, est, lo, hi, rob


@_paper
def learning_curves(results_dir=RESULTS_DIR, planes=PLANES):
    """Learning curves: rows = task (altitude: settled altitude RMSE [m] of evaluate_altitude;
    attitude: bank RMSE [deg] of evaluate_attitude), columns = airframe, one line per algorithm =
    IQM over training seeds with a 95 % bootstrap band. x is env steps on a log scale (PPO trains
    for 1.5e8 steps, SAC and TD3 for 3e7). Open circles mark evaluations in which fewer than half
    the episodes survived: the error there is averaged over episodes that departed."""
    runs = _read_curves(Path(results_dir) / "curves")
    algos = [a for a in ("ppo", "sac", "td3") if any(k[0] == a for k in runs)]
    tasks = [t for t in CURVE_METRIC if any(k[1] == t for k in runs)]
    fig, ax = plt.subplots(len(tasks), len(planes), figsize=(TEXT_W, 0.5 + 1.75 * len(tasks)),
                           sharex=True, squeeze=False, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.04)
    flagged = False
    for i, task in enumerate(tasks):
        key, ylab, logy = CURVE_METRIC[task]
        for j, p in enumerate(planes):
            a = ax[i, j]
            for algo in algos:
                rs = runs.get((algo, task, p))
                if not rs:
                    continue
                c = METHOD_COLORS[algo.upper()]
                x, y, lo, hi, rob = _seed_band(rs, key)
                a.fill_between(x, lo, hi, color=c, alpha=0.2, lw=0)
                a.plot(x, y, color=c, lw=1.2)
                bad = rob < ROB_MIN
                if bad.any():
                    flagged = True
                    a.plot(x[bad], y[bad], "o", ms=3.5, mfc="white", mec=c, mew=0.9)
                print(f"  {task}/{p}/{algo}: {len(rs)} seeds, final IQM {key} {y[-1]:.3g} "
                      f"[{lo[-1]:.3g}, {hi[-1]:.3g}], rob {rob[-1]:.2f}")
            a.set_xscale("log")
            if logy:
                a.set_yscale("log")
            a.grid(True, which="major")
            if j == 0:
                a.set_ylabel(ylab)
            if i == 0:
                a.set_title(PRETTY.get(p, p.replace("_", " ")), fontsize=8, pad=3)
            if i == len(tasks) - 1:
                a.set_xlabel("environment steps")
            _letter(a, f"({'abcdefgh'[i * len(planes) + j]})")
    handles = [Line2D([], [], color=METHOD_COLORS[a.upper()], lw=1.2, label=a.upper())
               for a in algos]
    handles.append(Patch(color="0.5", alpha=0.25, lw=0, label="95 % bootstrap CI of the IQM"))
    if flagged:
        handles.append(Line2D([], [], ls="", marker="o", ms=3.5, mfc="white", mec="0.3",
                              mew=0.9, label=f"survival < {ROB_MIN:.0%}"))
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))
    return _save(fig, results_dir, "fig_learning_curves")


PAIR = ("Airship_V7", "Volantex_Ranger")
FIGURES = {
    **{f"altitude_viz_{a}": (lambda a=a: (lambda ckpt_dir, results_dir: altitude_viz(a, PAIR, ckpt_dir, results_dir)))()
       for a in ("ppo", "sac", "td3")},
    "attitude_exec": lambda ckpt_dir, results_dir: attitude_exec(PAIR, ckpt_dir, results_dir),
    "maneuver_facet_Airship_V7": lambda ckpt_dir, results_dir: maneuver_facet("Airship_V7", ckpt_dir, results_dir),
    "maneuver_render": lambda ckpt_dir, results_dir: maneuver_render(PAIR, ckpt_dir, results_dir),
    "ge_energy": lambda ckpt_dir, results_dir: ge_energy(results_dir),
    "robustness_ppo": lambda ckpt_dir, results_dir: robustness_ppo(results_dir),
    "throughput": lambda ckpt_dir, results_dir: throughput(results_dir),
    "learning_curves": lambda ckpt_dir, results_dir: learning_curves(results_dir),
}


def render(names=None, ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR):
    (Path(results_dir) / "figures").mkdir(parents=True, exist_ok=True)
    return [FIGURES[n](ckpt_dir, results_dir) for n in (names or FIGURES)]
