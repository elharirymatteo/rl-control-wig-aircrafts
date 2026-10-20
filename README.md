# FALCON-S

A simulation framework and controller benchmark for fixed-wing flight close to the ground.

Six-degree-of-freedom rigid-body dynamics with OpenVSP aerodynamics and ground effect, actuator dynamics,
sensor noise and Dryden turbulence, across four airframes. Three independent plants — a NumPy
reference, a PyTorch twin and an NVIDIA Warp implementation that steps tens of thousands of
environments in parallel — validated against each other to single-step parity. Reinforcement
learning (PPO, SAC, TD3) and classical control (LQR, MPPI) drive the same command interface through
the same models, so a comparison measures the controller and not the task encoding.

**Project page:** [docs/index.html](docs/index.html), with the benchmark tables, the paper figures
and an interactive replay of the recorded flights. The page draws them with JavaScript; in a
viewer that blocks scripts by default, such as the anonymized repository's, press **JS on**. The
[Results](#results) section below carries the same content as plain tables and figures.

**Contents:** [Results](#results) · [Requirements](#requirements) · [Install](#install) ·
[Commands](#commands) · [Reproducing the shipped results](#reproducing-the-shipped-results) ·
[Layout](#layout) · [Tests](#tests)

## Results

Five controllers fly two airframes, the **Airship V7** wing-in-ground-effect prototype and the
**Volantex Ranger** RC aircraft, under identical conditions: same plant, actuators, sensors,
reference and scoring. Learned policies pool five training seeds; MPPI pools its sampling draws;
the LQR is deterministic. Errors are interquartile means over the surviving runs with a 95 %
bootstrap interval, and survival counts every run, so no cell hides the runs it lost. The tables
are generated from `results/*.csv` by `docs/tools/readme_tables.py`.

### Altitude acquisition and hold

Each episode starts 12–18 m off a commanded altitude between 20 and 100 m and follows a reference
that ramps to it at the airframe's rate limit. *Survival*: no crash or envelope exit. *Acquired*:
reaches the target and stays within ±1 m.

<!-- TABLE:altitude -->
<!-- /TABLE:altitude -->

<!-- OBS:altitude -->

![PPO flying a sequence of altitude targets on both airframes](results/figures/fig_altitude_viz_ppo.png)

### Attitude-command tracking

A maneuver is a stream of bank, climb-rate and airspeed commands (circle, figure-of-eight, helix,
s-turn) built from each airframe's envelope; every executor gets the same stream. Cells are bank
RMSE in degrees.

<!-- TABLE:maneuvers -->
<!-- /TABLE:maneuvers -->

<!-- OBS:maneuvers -->

![Bank tracking on the Airship V7, maneuvers by executor](results/figures/fig_maneuver_facet_Airship_V7.png)

The LQR's bank envelope, one constant-bank turn per angle (`scripts/lqr_envelope.py`):

<!-- TABLE:lqr_envelope -->
<!-- /TABLE:lqr_envelope -->

### Ground effect

Ground effect comes from OpenVSP: the airframe's derivatives computed at eleven heights between
h/b = 0.2 and 2. Below h/b = 0.2 the measured values are continued with the Phillips–Hunsaker
lifting-line correlation (the default *hybrid* model; either source alone can be selected with
`aero_params.ge_model`). The table compares the thrust needed to fly level with ground effect on
and off, at a trim solved without a controller and under the PPO altitude policy.

<!-- TABLE:ground_effect -->
<!-- /TABLE:ground_effect -->

<!-- OBS:ground_effect -->

![Thrust change and tracking accuracy against height over span](results/figures/fig_ge_energy.png)

The two ground-effect sources, where both apply (`results/ge_models/`):

<!-- TABLE:ge_models -->
<!-- /TABLE:ge_models -->

<!-- OBS:ge_models -->

**Does training in ground effect matter?** Each PPO altitude policy is fine-tuned twice near the
ground (h/b 0.20–0.46), with and without ground effect, then flown with and without it.

<!-- TABLE:ge_training -->
<!-- /TABLE:ge_training -->

<!-- OBS:ge_training -->

### Robustness

The PPO attitude executor flies the four maneuvers under one disturbance at a time, at three
severities in physical units ([docs/disturbance_model.md](docs/disturbance_model.md)). Metrics come
from the true state, never the disturbed observation. The second table is the same executor trained
through random 0–100 ms delays (`falcons train --latency-ms 100`).

<!-- TABLE:robustness -->
<!-- /TABLE:robustness -->

<!-- TABLE:robustness_dr -->
<!-- /TABLE:robustness_dr -->

<!-- OBS:robustness -->

![The PPO executor under four disturbance families](results/figures/fig_robustness_ppo.png)

### Training

![Learning curves, IQM over five seeds](results/figures/fig_learning_curves.png)

<!-- TABLE:training -->
<!-- /TABLE:training -->

### Simulator throughput

Env-steps per second on one RTX 4090 (`falcons benchmark throughput`). The full sweep and the
NeuralPlane comparison are on the [project page](docs/index.html#throughput).

<!-- TABLE:throughput -->
<!-- /TABLE:throughput -->

### Cross-simulator validation

Zero-input free flight from identical initial conditions in FALCON-S and JSBSim
(`tools/jsbsim_validate`), which shares the derivative set but has no ground effect: with it off
the two should match, with it on at 1 m they should part.

<!-- TABLE:jsbsim -->
<!-- /TABLE:jsbsim -->

## Requirements

Python 3.10 and an NVIDIA GPU with a CUDA 12.6 driver. The Warp plant is CUDA-only; `falcons check`
and the NumPy reference plant run without one.

## Install

```bash
git clone <this repository> falcon-s && cd falcon-s
bash scripts/setup_venv.sh      # venv → requirements.lock → editable install → falcons check
```

A Docker image is also provided: `docker build -t falcons . && docker run --rm --gpus all falcons`.

## Commands

```bash
falcons check                                     # versions, GPU, load checkpoints, step the plant

# train a policy: 3 algorithms × 2 tasks × 5 airframes × any seed
falcons train --algo ppo --task altitude  --plane Airship_V7 --seed 0
falcons train --algo sac --task attitude  --plane Volantex_Ranger --seed 1
scripts/train_seeds.sh                            # the 62 shipped runs, idempotent

# evaluate one controller on one task, learned or classical
falcons eval --algo ppo  --task altitude --plane Airship_V7 --plot
falcons eval --algo mppi --task attitude --plane Volantex_Ranger --maneuver circle

# sweep a whole protocol across airframes and methods
falcons benchmark altitude                        # acquisition and hold         3 h 35
falcons benchmark maneuvers --refresh             # four named attitude streams    25 min
falcons benchmark ground-effect                   # thrust vs height over span      3 min
falcons benchmark robustness                      # noise, latency, turbulence     44 min
falcons benchmark ge-training                     # train GE x eval GE, 2 x 2      17 min
falcons benchmark throughput                      # step rate and cross-plant parity 7 min
PYTHONPATH=src python scripts/lqr_envelope.py     # the LQR's bank envelope          4 min

falcons tables                                    # results/*.csv → results/tables.tex
falcons figures [name …]                          # → results/figures/
falcons reproduce --out DIR                       # re-fly everything and diff against results/
```

`eval` is the inner loop `benchmark` composes, so a cell printed by `eval` is the cell that lands in
the table. Runtimes are those of the shipped run, on an RTX 4090 that was shared with another job;
an idle GPU is faster. Throughput must be measured on an idle GPU.

## Layout

```
src/falcons/
  aircraft/     four airframes: geometry, OpenVSP derivative sets, LQR gains
  sim/          the three plants — cpu, torch, warp
  envs/         altitude and attitude RL environments
  controllers/  LQR, MPPI, and the checkpoint loader
  train/        PPO, SAC, TD3
  benchmark/    the five protocols, the tables, the figures, reproduce
checkpoints/    62 trained policies (5 seeds × 3 algorithms × 2 tasks × 2 airframes, + 2)
results/        tables, figures and training curves from the shipped run
docs/           the project page (index.html) and the tools that build its data
```

The disturbance models used by the robustness sweep are documented in
[docs/disturbance_model.md](docs/disturbance_model.md).

## Tests

```bash
.venv/bin/pytest -q -m "not slow"     # about a minute
.venv/bin/pytest -q -m slow           # ~5 h: every protocol re-flown and diffed
```

Three tests need a rollout-cache fixture that is not in the repository; without it the fast tier
skips them. Point `FALCONS_TRACE_FIXTURE` at the cache to run them.

## Reproducing the shipped results

`falcons reproduce` re-flies every protocol from the checkpoints and diffs the outcome against
`results/`. Deterministic rows — every learned policy and the LQR — return bit-identical. MPPI is a
sampling planner, so its cells are draws and are compared within their measured spread.

## License

MIT — see `LICENSE`.
