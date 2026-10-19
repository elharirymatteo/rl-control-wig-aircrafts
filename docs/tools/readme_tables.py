"""Write the result tables of README.md from the shipped CSVs, so the README cannot drift from the
numbers in results/.

    .venv/bin/python docs/tools/readme_tables.py

Each table lives between `<!-- TABLE:<name> -->` and `<!-- /TABLE:<name> -->` in README.md and is
replaced whole; the prose around it is not touched. A table whose CSV is missing is left as it is.
Cells follow results/tables.tex: interquartile mean over the surviving episodes with its 95 %
bootstrap interval in brackets, survival as a rate with its interval.
"""
import csv
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
README = ROOT / "README.md"

PLANES = ["Airship_V7", "Volantex_Ranger"]
METHODS = ["PPO", "SAC", "TD3", "LQR", "MPPI"]
MANEUVERS = [("circle", "Circle"), ("figure8", "Figure-8"), ("helix", "Helix"), ("sturn", "S-turn")]
PRETTY = {"Airship_V7": "Airship V7", "Volantex_Ranger": "Volantex Ranger",
          "Airship_A0S": "Airship A0S", "Navion": "Navion", "Cirrus_SR22": "Cirrus SR22"}
DIST = [("obs_noise", "Sensor noise"), ("sensor_delay", "Sensor delay"),
        ("action_delay", "Action delay"), ("wind", "Turbulence")]


def rows(name):
    p = RES / name
    if not p.exists():
        return None
    with open(p) as f:
        return list(csv.DictReader(f))


def f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def fmt(v, nd):
    return "–" if v is None else f"{v:.{nd}f}"


def auto(v):
    """Digits by magnitude, as the paper prints errors that span seven orders."""
    if v is None:
        return "–"
    a = abs(v)
    if a == 0:
        return "0"
    if a < 1e-3:
        m, e = f"{v:.1e}".split("e")
        return f"{m}×10<sup>{int(e)}</sup>"
    return f"{v:.3f}" if a < 1 else f"{v:.2f}" if a < 10 else f"{v:.1f}"


def ci(v, lo, hi, nd=None):
    v, lo, hi = f(v), f(lo), f(hi)
    if v is None:
        return "–"
    g = (lambda x: fmt(x, nd)) if nd is not None else auto
    return g(v) if lo is None or hi is None else f"{g(v)} [{g(lo)}, {g(hi)}]"


def rate(k, n, lo=None, hi=None):
    k, n = f(k), f(n)
    if not n:
        return "–"
    s = f"{k / n:.2f}"
    if f(lo) is not None and f(hi) is not None:
        s += f" [{f(lo):.2f}, {f(hi):.2f}]"
    return s


def table(head, body, align=None):
    align = align or ["---"] + ["--:"] * (len(head) - 1)
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return "\n".join(out)


def by(rs, *keys):
    return {tuple(r[k] for k in keys): r for r in rs}


# ─── one function per table ──────────────────────────────────────────────────────────────────────

def t_altitude():
    rs = rows("altitude.csv")
    if not rs:
        return None
    body = []
    for r in sorted(rs, key=lambda r: (PLANES.index(r["aircraft"]) if r["aircraft"] in PLANES else 9,
                                       METHODS.index(r["method"]))):
        n = f(r.get("n_total")) or f(r["n"])
        surv = rate(r.get("n_survived"), n, r.get("survival_lo"), r.get("survival_hi")) \
            if r.get("n_survived") else fmt(f(r["survival"]), 2)
        acq = rate(r.get("n_acquired"), n, r.get("acquired_lo"), r.get("acquired_hi")) \
            if r.get("n_acquired") else fmt(f(r["acquired"]), 2)
        body.append([PRETTY[r["aircraft"]], r["method"], int(n), surv, acq,
                     ci(r.get("rmse_iqm"), r.get("rmse_lo"), r.get("rmse_hi")),
                     ci(r.get("settling_iqm"), r.get("settling_lo"), r.get("settling_hi"), 1),
                     ci(r.get("overshoot_iqm"), r.get("overshoot_lo"), r.get("overshoot_hi"), 2)])
    return table(["Airframe", "Method", "Episodes", "Survival", "Acquired (±1 m)",
                  "Settled error [m]", "Settling time [s]", "Overshoot [m]"], body,
                 ["---", "---"] + ["--:"] * 6)


def t_maneuvers():
    rs = rows("maneuvers.csv")
    if not rs:
        return None
    cell = by(rs, "aircraft", "maneuver", "method")
    body = []
    for p in PLANES:
        for m in METHODS:
            row = [PRETTY[p], m]
            for k, _ in MANEUVERS:
                r = cell.get((p, k, m))
                if r is None:
                    row.append("–")
                    continue
                c = ci(r.get("phi_rmse_iqm") or r["phi_rmse_deg"], r.get("phi_rmse_lo"),
                       r.get("phi_rmse_hi"), 1)
                dep = int(f(r.get("n_departed")) or 0)
                row.append(c + (f" · {dep} departed" if dep else ""))
            body.append(row)
    return table(["Airframe", "Method"] + [n for _, n in MANEUVERS], body,
                 ["---", "---"] + ["--:"] * 4)


def t_ground_effect():
    trim, cl = rows("ge_trim.csv"), rows("ge_energy.csv")
    if not trim:
        return None
    closed = by(cl or [], "aircraft", "h_over_b")
    body = []
    for r in trim:
        c = closed.get((r["aircraft"], r["h_over_b"]))
        body.append([PRETTY.get(r["aircraft"], r["aircraft"]), fmt(f(r["h_over_b"]), 2),
                     fmt(f(r["delta_thrust_pct"]), 1), fmt(f(r["theory_pct"]), 1),
                     fmt(f(c["delta_thrust_pct"]), 1) if c and c.get("hold_valid") in ("1", "True") else "–",
                     auto(f(c["rmse_ge_on"])) if c else "–", auto(f(c["rmse_ge_off"])) if c else "–"])
    return table(["Airframe", "h/b", "ΔT trim [%]", "ΔT theory [%]", "ΔT under PPO [%]",
                  "Error GE on [m]", "Error GE off [m]"], body)


def t_robustness():
    rs = rows("robustness.csv")
    if not rs:
        return None
    cell = by([r for r in rs if r["method"] == "PPO"], "aircraft", "disturbance", "severity")
    body = []
    for p in PLANES:
        nom = cell.get((p, "nominal", "-"))
        if nom:
            body.append([PRETTY[p], "Nominal", "–", ci(nom["phi_rmse"], nom.get("phi_rmse_lo"),
                                                       nom.get("phi_rmse_hi"), 1), fmt(f(nom["survival"]), 2)])
        for d, name in DIST:
            for sev in ("mild", "medium", "severe"):
                r = cell.get((p, d, sev))
                if r is None:
                    continue
                body.append([PRETTY[p], name, sev,
                             ci(r["phi_rmse"], r.get("phi_rmse_lo"), r.get("phi_rmse_hi"), 1),
                             fmt(f(r["survival"]), 2)])
    return table(["Airframe", "Disturbance", "Severity", "Bank error [deg]", "Survival"], body,
                 ["---", "---", "---", "--:", "--:"])


def t_ge_training():
    agg, eff = rows("ge_training_agg.csv"), rows("ge_training_effects.csv")
    if not agg:
        return None
    band = {"near": "near ground", "control": "30 m"}
    arm = {"GE_ON": "with GE", "GE_OFF": "without GE"}
    order = lambda r: (PLANES.index(r["plane"]), list(band).index(r["band"]),
                       list(arm).index(r["train"]), -int(float(r["eval_ge"])))
    body = [[PRETTY.get(r["plane"], r["plane"]), band.get(r["band"], r["band"]),
             arm.get(r["train"], r["train"]), "on" if r["eval_ge"] in ("1", "True") else "off",
             rate(r["n_survived"], r["n"], r["survival_lo"], r["survival_hi"]),
             ci(r["rmse_iqm"], r["rmse_lo"], r["rmse_hi"])] for r in sorted(agg, key=order)]
    out = table(["Airframe", "Band", "Fine-tuned", "Flown with GE", "Survival", "Altitude error [m]"],
                body, ["---", "---", "---", "---", "--:", "--:"])
    if eff:
        eff = sorted(eff, key=lambda r: (PLANES.index(r["plane"]), list(band).index(r["band"])))
        out += "\n\n" + table(["Airframe", "Band", "Effect", "Value [m]"],
                              [[PRETTY.get(r["plane"], r["plane"]), band.get(r["band"], r["band"]),
                                r["effect"].replace("_", " "), ci(r["value"], r["lo"], r["hi"])]
                               for r in eff], ["---", "---", "---", "--:"])
    return out


def t_lqr_envelope():
    rs = rows("lqr_envelope.csv")
    if not rs:
        return None
    banks = sorted({int(float(r["bank_deg"])) for r in rs})
    cell = {(r["aircraft"], int(float(r["bank_deg"]))): r for r in rs}
    body = []
    for p in PLANES:
        row = [PRETTY[p]]
        for b in banks:
            r = cell.get((p, b))
            row.append("–" if r is None else
                       (fmt(f(r["phi_rmse_deg"]), 2) if f(r["survival"]) == 1.0 else "departs"))
        body.append(row)
    return table(["Bank error [deg] at"] + [f"{b}°" for b in banks], body)


def t_jsbsim():
    rs = rows("jsbsim/paper_table.csv")
    if not rs:
        return None
    body = [[PRETTY[r["plane"]], f"{float(r['h0']):g} m", "on" if r["ground_effect"] == "True" else "off",
             f"{float(r['altitude r']):.4f}", f"{float(r['pitch rate r']):.4f}",
             f"{float(r['airspeed r']):.4f}", auto(f(r["altitude rmse"])),
             auto(f(r["airspeed rmse"]))] for r in rs]
    return table(["Airframe", "h₀", "Ground effect", "r altitude", "r pitch rate", "r airspeed",
                  "RMSE altitude [m]", "RMSE airspeed [m/s]"], body,
                 ["---", "--:", "---"] + ["--:"] * 5)


def t_throughput():
    rs = rows("throughput.csv")
    if not rs:
        return None
    name = {"cpu": "NumPy, CPU", "torch": "PyTorch, GPU", "warp": "Warp, GPU"}
    body = [[name.get(r["backend"], r["backend"]), f"{int(float(r['n_envs'])):,}",
             f"{float(r['steps_per_s_mean']):,.0f}"] for r in rs]
    return table(["Backend", "Parallel envs", "Env-steps / s"], body, ["---", "--:", "--:"])


TABLES = dict(altitude=t_altitude, maneuvers=t_maneuvers, ground_effect=t_ground_effect,
              robustness=t_robustness, ge_training=t_ge_training, lqr_envelope=t_lqr_envelope,
              jsbsim=t_jsbsim, throughput=t_throughput)


def main():
    text = README.read_text(encoding="utf-8")
    for name, fn in TABLES.items():
        pat = re.compile(rf"(<!-- TABLE:{name} -->\n)(?:.*?\n)?(<!-- /TABLE:{name} -->)", re.S)
        if not pat.search(text):
            continue
        body = fn()
        if body is None:
            print(f"  {name}: no CSV, left as is")
            continue
        text = pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)
        print(f"  {name}: {body.count(chr(10)) - 1} rows")
    README.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
