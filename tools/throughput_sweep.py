"""The wide throughput sweep behind the paper's throughput figure and the project page.

`falcons benchmark throughput` measures the widths RL training uses (up to 16,384 environments) and
the CPU plant at one environment. This script extends both ends, on an otherwise idle machine:

    GPU   warp and torch from 64 to 2,097,152 environments, through the benchmark's own
          `throughput()` (same warm-up, step count, trial count and CUDA fencing)
    CPU   the single-aircraft NumPy plant, one process per environment, from 1 to 32 processes;
          the processes start each trial together on a barrier, and the aggregate rate is
          n * steps / (slowest process's elapsed time)

    .venv/bin/python tools/throughput_sweep.py --out DIR [--plane Airship_V7] [--skip-gpu] [--skip-cpu]

Writes DIR/throughput.csv (GPU sweep plus the one-env CPU row) and DIR/cpu_scaling.csv, both with
the benchmark's columns: backend, n_envs, steps_per_s_mean, steps_per_s_std, ci95.
"""
import os

# one BLAS/OpenMP thread per process, set before numpy loads, so the CPU series measures one
# process per environment and not a thread pool fighting itself
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import multiprocessing as mp
import time
from pathlib import Path

WIDE_SIZES = (64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 2097152)
CPU_PROCS = (1, 2, 4, 8, 16, 24, 32)
CPU_SCALING_STEPS = 1000     # per process per trial: long enough that barrier release jitter is noise


def _cpu_worker(plane, steps, warmup, trials, barrier, out):
    import numpy as np
    from falcons.benchmark.throughput import CPU_THROTTLE
    from falcons.sim.cpu.aircraft import Aircraft
    env = Aircraft(plane)
    env.reset()
    aero = np.zeros(env.get_aero_action_size())
    motor = np.full(env.get_motor_action_size(), CPU_THROTTLE)
    for _ in range(warmup):
        env.step(aero, motor)
    for k in range(trials):
        barrier.wait()
        t0 = time.perf_counter()
        for _ in range(steps):
            env.step(aero, motor)
        out.put((k, time.perf_counter() - t0))


def cpu_scaling(plane, procs=CPU_PROCS, steps=CPU_SCALING_STEPS, trials=None):
    from falcons.benchmark.throughput import CPU_WARMUP, R_TRIALS, stats
    trials = trials or R_TRIALS
    ctx = mp.get_context("fork")
    rows = []
    for n in procs:
        barrier, out = ctx.Barrier(n), ctx.Queue()
        ps = [ctx.Process(target=_cpu_worker, args=(plane, steps, CPU_WARMUP, trials, barrier, out))
              for _ in range(n)]
        for p in ps:
            p.start()
        elapsed = [[] for _ in range(trials)]
        for _ in range(n * trials):
            k, dt = out.get()
            elapsed[k].append(dt)
        for p in ps:
            p.join()
        rates = [n * steps / max(e) for e in elapsed]
        mu, sd, ci = stats(rates)
        rows.append(("cpu", n, mu, sd, ci))
        print(f"cpu  {n:3d} processes  {mu:12,.0f} env-steps/s  (sd {sd:,.0f})", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plane", default="Airship_V7")
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--skip-cpu", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.skip_cpu:
        # before the GPU sweep, so no CUDA context exists when the workers fork
        rows = cpu_scaling(a.plane)
        with open(a.out / "cpu_scaling.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["backend", "n_envs", "steps_per_s_mean", "steps_per_s_std", "ci95"])
            w.writerows(rows)
        print(f"wrote {a.out / 'cpu_scaling.csv'}", flush=True)
    if not a.skip_gpu:
        from falcons.benchmark.throughput import throughput
        throughput(a.plane, sizes=WIDE_SIZES, results_dir=a.out)


if __name__ == "__main__":
    main()
