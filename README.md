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
| Airframe | Method | Episodes | Survival | Acquired (±1 m) | Settled error [m] | Settling time [s] | Overshoot [m] |
|---|---|--:|--:|--:|--:|--:|--:|
| Airship V7 | PPO | 1000 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.388 [0.351, 0.418] | 7.4 [7.3, 7.5] | 0.71 [0.69, 0.72] |
| Airship V7 | SAC | 1000 | 1.00 [1.00, 1.00] | 1.00 [0.99, 1.00] | 0.009 [0.005, 0.020] | 7.2 [7.2, 7.3] | 0.56 [0.52, 0.63] |
| Airship V7 | TD3 | 1000 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.008 [0.001, 0.021] | 7.2 [7.2, 7.3] | 0.63 [0.58, 0.67] |
| Airship V7 | LQR | 50 | 1.00 [0.93, 1.00] | 1.00 [0.93, 1.00] | 9.2×10<sup>-14</sup> [5.9×10<sup>-14</sup>, 1.2×10<sup>-13</sup>] | 9.7 [9.3, 10.0] | 1.54 [1.52, 1.55] |
| Airship V7 | MPPI | 150 | 0.98 [0.95, 1.00] | 0.98 [0.95, 1.00] | 0.052 [0.048, 0.057] | 7.4 [7.1, 7.6] | 0.17 [0.15, 0.20] |
| Volantex Ranger | PPO | 1000 | 0.95 [0.84, 1.00] | 0.95 [0.84, 1.00] | 0.018 [0.015, 0.020] | 8.6 [8.4, 8.8] | 1.12 [1.06, 1.25] |
| Volantex Ranger | SAC | 1000 | 0.91 [0.73, 1.00] | 0.91 [0.73, 1.00] | 0.002 [0.001, 0.003] | 7.2 [7.1, 7.3] | 0.45 [0.41, 0.48] |
| Volantex Ranger | TD3 | 1000 | 1.00 [1.00, 1.00] | 1.00 [0.99, 1.00] | 0.003 [3.0×10<sup>-4</sup>, 0.011] | 7.2 [7.1, 7.3] | 0.45 [0.36, 0.47] |
| Volantex Ranger | LQR | 50 | 1.00 [0.93, 1.00] | 1.00 [0.93, 1.00] | 7.1×10<sup>-14</sup> [4.7×10<sup>-14</sup>, 1.0×10<sup>-13</sup>] | 7.7 [7.5, 8.0] | 0.96 [0.91, 1.00] |
| Volantex Ranger | MPPI | 150 | 1.00 [0.98, 1.00] | 1.00 [0.98, 1.00] | 0.067 [0.061, 0.074] | 7.4 [7.2, 7.6] | 0.26 [0.23, 0.29] |
<!-- /TABLE:altitude -->

<!-- OBS:altitude -->
- **Every controller acquires on both airframes.** At least 91 % of episodes reach and hold the
  target for every method, 98–100 % on the V7.
- **The off-policy agents settle tightest, the LQR is exact.** Once there, SAC and TD3 hold within
  millimetres and PPO within centimetres. On a constant reference the task is the linear regulation
  problem the LQR gains solve, so its error is round-off; it pays with the largest overshoot.
- **The Ranger misses are single seeds.** One PPO seed crashes on half of its climbs and one SAC
  seed on 89 of its 100 descents; the other seeds acquire every episode.
- **Training has to cover the protocol.** Trained only up to 10 m from the target, the V7 policies
  settled off target after the protocol's 12–18 m climbs, so they train to 18 m. PPO on the Ranger
  is the exception: its curriculum stalls past 10 m under that airframe's narrow altitude reward.
<!-- /OBS:altitude -->

![PPO flying a sequence of altitude targets on both airframes](results/figures/fig_altitude_viz_ppo.png)

### Attitude-command tracking

A maneuver is a stream of bank, climb-rate and airspeed commands (circle, figure-of-eight, helix,
s-turn) built from each airframe's envelope; every executor gets the same stream. Cells are bank
RMSE in degrees.

<!-- TABLE:maneuvers -->
| Airframe | Method | Circle | Figure-8 | Helix | S-turn |
|---|---|--:|--:|--:|--:|
| Airship V7 | PPO | 1.7 [1.6, 1.8] | 2.1 [2.1, 2.2] | 0.7 [0.7, 0.8] | 2.2 [2.2, 2.3] |
| Airship V7 | SAC | 7.6 [4.1, 12.4] | 7.8 [5.5, 10.5] | 7.8 [3.5, 9.2] · 1 departed | 6.9 [6.5, 7.4] · 2 departed |
| Airship V7 | TD3 | 7.1 [4.6, 16.3] | 7.2 [4.8, 12.3] | 5.4 [3.1, 10.6] | 5.9 [4.2, 6.9] |
| Airship V7 | LQR | 3.0 [3.0, 3.0] | 3.2 [3.2, 3.2] | 1.4 [1.4, 1.4] | 2.7 [2.7, 2.7] |
| Airship V7 | MPPI | 1.3 [1.0, 2.3] | 0.6 [0.4, 1.2] | 0.3 [0.2, 0.6] | 0.2 [0.1, 0.4] |
| Volantex Ranger | PPO | 1.1 [1.0, 1.2] | 1.8 [1.6, 1.9] | 0.5 [0.4, 0.5] | 2.1 [1.9, 2.3] |
| Volantex Ranger | SAC | 3.2 [2.8, 3.5] | 4.4 [3.5, 5.4] | 2.9 [2.5, 4.9] | 5.3 [3.9, 6.4] |
| Volantex Ranger | TD3 | 3.8 [3.2, 4.9] | 5.5 [3.9, 8.3] | 3.9 [3.2, 4.4] | 5.8 [4.6, 7.2] |
| Volantex Ranger | LQR | 1.6 [1.6, 1.6] | 2.1 [2.1, 2.1] | 1.7 [1.7, 1.7] | 2.2 [2.2, 2.2] |
| Volantex Ranger | MPPI | 0.5 [0.3, 1.1] | 0.5 [0.3, 1.0] | 0.2 [0.1, 0.2] | 0.2 [0.1, 0.3] |
<!-- /TABLE:maneuvers -->

<!-- OBS:maneuvers -->
- **MPPI is the most accurate executor, PPO is close behind, the LQR holds every stream.**
- **SAC and TD3 fly bang-bang.** Their actuators sit at the stops most of the time, so the bank
  chatters around the command (facet figure below). Stronger smoothing, a control-effort cost and
  different activations did not change that; a small regularizer on the actor's pre-tanh output
  did stop one TD3 seed in five from freezing at full deflection, where it departed every stream.
- **SAC on the V7 is the only cell that loses runs** (3 of 20).
<!-- /OBS:maneuvers -->

![Bank tracking on the Airship V7, maneuvers by executor](results/figures/fig_maneuver_facet_Airship_V7.png)

The LQR's bank envelope, one constant-bank turn per angle (`scripts/lqr_envelope.py`):

<!-- TABLE:lqr_envelope -->
| Bank error [deg] at | 5° | 10° | 15° | 20° | 25° | 30° | 35° | 40° | 45° |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Airship V7 | 0.29 | 0.53 | 0.70 | 0.79 | 0.91 | 1.41 | 2.62 | 4.89 | 9.20 |
| Volantex Ranger | 0.35 | 0.69 | 1.01 | 1.29 | 1.52 | 1.73 | 1.94 | 2.24 | 2.73 |
<!-- /TABLE:lqr_envelope -->

### Ground effect

Ground effect comes from OpenVSP: the airframe's derivatives computed at eleven heights between
h/b = 0.2 and 2. Below h/b = 0.2 the measured values are continued with the Phillips–Hunsaker
lifting-line correlation (the default *hybrid* model; either source alone can be selected with
`aero_params.ge_model`). The table compares the thrust needed to fly level with ground effect on
and off, at a trim solved without a controller and under the PPO altitude policy.

<!-- TABLE:ground_effect -->
| Airframe | h/b | ΔT trim [%] | ΔC_D at fixed α [%] | ΔT under PPO [%] | Error GE on [m] | Error GE off [m] |
|---|--:|--:|--:|--:|--:|--:|
| Airship V7 | 0.25 | -13.4 | -6.4 | -17.7 | 0.085 | 0.175 |
| Airship V7 | 0.50 | -4.8 | -2.3 | -7.5 | 0.128 | 0.176 |
| Airship V7 | 1.00 | -1.2 | -0.6 | -1.7 | 0.155 | 0.177 |
| Airship V7 | 2.00 | 0.0 | 0.0 | -0.0 | 0.178 | 0.179 |
| Airship V7 | 3.00 | 0.0 | 0.0 | 0.0 | 0.182 | 0.182 |
| Airship V7 | 6.00 | 0.0 | 0.0 | 0.0 | 0.190 | 0.190 |
| Airship A0S | 0.25 | -12.5 | -7.3 | -16.2 | 0.062 | 0.087 |
| Airship A0S | 0.50 | -4.4 | -2.6 | -8.3 | 0.082 | 0.087 |
| Airship A0S | 1.00 | -1.1 | -0.6 | -1.7 | 0.083 | 0.087 |
| Airship A0S | 2.00 | 0.0 | 0.0 | -0.0 | 0.087 | 0.087 |
| Airship A0S | 3.00 | 0.0 | 0.0 | 0.0 | 0.087 | 0.087 |
| Airship A0S | 6.00 | 0.0 | 0.0 | 0.0 | 0.087 | 0.087 |
| Volantex Ranger | 0.25 | -4.9 | -6.3 | -2.9 | 0.006 | 0.011 |
| Volantex Ranger | 0.50 | -1.8 | -2.3 | -1.1 | 0.009 | 0.011 |
| Volantex Ranger | 1.00 | -0.4 | -0.6 | -0.3 | 0.011 | 0.011 |
| Volantex Ranger | 2.00 | 0.0 | 0.0 | 0.0 | 0.011 | 0.011 |
| Volantex Ranger | 3.00 | 0.0 | 0.0 | 0.0 | 0.011 | 0.011 |
| Volantex Ranger | 6.00 | 0.0 | 0.0 | 0.0 | 0.011 | 0.011 |
| Navion | 0.25 | -6.5 | -4.5 | – | 38.0 | 37.7 |
| Navion | 0.50 | -2.5 | -1.8 | – | 37.8 | 37.7 |
| Navion | 1.00 | -0.6 | -0.4 | – | 37.6 | 37.6 |
| Navion | 2.00 | 0.0 | 0.0 | – | 37.5 | 37.5 |
| Navion | 3.00 | 0.0 | 0.0 | – | 37.3 | 37.3 |
| Navion | 6.00 | 0.0 | 0.0 | – | 36.9 | 36.9 |
<!-- /TABLE:ground_effect -->

<!-- OBS:ground_effect -->
- **Flying low costs less thrust on every airframe,** most on the two wing-in-ground-effect
  airframes, and the effect is gone by h/b ≈ 2.
- **On the wing-in-ground-effect airframes, re-trimming doubles the saving.** At a fixed angle of
  attack their drag change is about half the trimmed one: the extra lift lets the aircraft fly at a
  lower angle of attack, and drag falls further. The Ranger, at a low lift coefficient, gains
  nothing from re-trimming.
- **On the two wing-in-ground-effect airframes the saving grows under the policy,** and every
  policy tracks altitude better in ground effect. The Ranger gains least: at
  its 15 m/s cruise it flies at a low lift coefficient, so there is little induced drag to cut. The
  Navion policy does not hold the near-ground band, so it has no closed-loop row.
<!-- /OBS:ground_effect -->

![Thrust change and tracking accuracy against height over span](results/figures/fig_ge_energy.png)

The two ground-effect sources, where both apply (`results/ge_models/`):

<!-- TABLE:ge_models -->
| Airframe | ΔT OpenVSP [%] | ΔT P–H [%] | C<sub>L</sub> ratio OpenVSP | C<sub>L</sub> ratio P–H | C<sub>D</sub> ratio OpenVSP | C<sub>D</sub> ratio P–H | ΔC<sub>m</sub> OpenVSP |
|---|--:|--:|--:|--:|--:|--:|--:|
| Airship V7 | -13.4 | -13.3 | 1.049 | 1.038 | 0.951 | 0.942 | -0.029 |
| Airship A0S | -12.5 | -12.3 | 1.044 | 1.038 | 0.945 | 0.934 | -0.041 |
| Volantex Ranger | -4.9 | -4.4 | 1.041 | 1.029 | 0.951 | 0.939 | -0.054 |
| Navion | -6.5 | -5.4 | 1.065 | 1.046 | 0.963 | 0.970 | -0.020 |
<!-- /TABLE:ge_models -->

<!-- OBS:ge_models -->
- **Where both apply, the two sources agree:** the trimmed thrust saving differs by at most half a
  percentage point on the ground-effect airframes.
- **Only the sweep changes the pitching moment,** which a lift-and-drag correlation cannot
  represent. That is why the measured sweep is used wherever it exists, and the correlation only
  below its lowest height.
<!-- /OBS:ge_models -->

**Does training in ground effect matter?** Each PPO altitude policy is fine-tuned twice near the
ground (h/b 0.20–0.46), with and without ground effect, then flown with and without it.

<!-- TABLE:ge_training -->
| Airframe | Band | Fine-tuned | Flown with GE | Survival | Altitude error [m] |
|---|---|---|---|--:|--:|
| Airship V7 | near ground | with GE | on | 1.00 [1.00, 1.00] | 0.370 [0.360, 0.380] |
| Airship V7 | near ground | with GE | off | 1.00 [1.00, 1.00] | 0.687 [0.682, 0.691] |
| Airship V7 | near ground | without GE | on | 1.00 [1.00, 1.00] | 0.202 [0.200, 0.205] |
| Airship V7 | near ground | without GE | off | 1.00 [1.00, 1.00] | 0.254 [0.251, 0.257] |
| Airship V7 | 30 m | with GE | on | 1.00 [1.00, 1.00] | 0.723 [0.719, 0.728] |
| Airship V7 | 30 m | with GE | off | 1.00 [1.00, 1.00] | 0.723 [0.719, 0.728] |
| Airship V7 | 30 m | without GE | on | 1.00 [1.00, 1.00] | 0.282 [0.279, 0.285] |
| Airship V7 | 30 m | without GE | off | 1.00 [1.00, 1.00] | 0.282 [0.279, 0.285] |
| Volantex Ranger | near ground | with GE | on | 1.00 [1.00, 1.00] | 0.141 [0.141, 0.141] |
| Volantex Ranger | near ground | with GE | off | 1.00 [1.00, 1.00] | 0.128 [0.128, 0.128] |
| Volantex Ranger | near ground | without GE | on | 1.00 [1.00, 1.00] | 0.161 [0.161, 0.161] |
| Volantex Ranger | near ground | without GE | off | 1.00 [1.00, 1.00] | 0.148 [0.148, 0.148] |
| Volantex Ranger | 30 m | with GE | on | 1.00 [1.00, 1.00] | 0.124 [0.124, 0.124] |
| Volantex Ranger | 30 m | with GE | off | 1.00 [1.00, 1.00] | 0.124 [0.124, 0.124] |
| Volantex Ranger | 30 m | without GE | on | 1.00 [1.00, 1.00] | 0.144 [0.144, 0.144] |
| Volantex Ranger | 30 m | without GE | off | 1.00 [1.00, 1.00] | 0.144 [0.144, 0.144] |

| Airframe | Band | Effect | Value [m] |
|---|---|---|--:|
| Airship V7 | near ground | deploy penalty | -0.168 [-0.178, -0.157] |
| Airship V7 | near ground | interaction | 0.265 [0.253, 0.276] |
| Airship V7 | 30 m | deploy penalty | -0.441 [-0.447, -0.436] |
| Airship V7 | 30 m | interaction | 0 [-0.007, 0.008] |
| Volantex Ranger | near ground | deploy penalty | 0.020 [0.020, 0.020] |
| Volantex Ranger | near ground | interaction | -3.1×10<sup>-4</sup> [-8.0×10<sup>-4</sup>, 1.9×10<sup>-4</sup>] |
| Volantex Ranger | 30 m | deploy penalty | 0.020 [0.020, 0.020] |
| Volantex Ranger | 30 m | interaction | 0 [-2.8×10<sup>-4</sup>, 2.9×10<sup>-4</sup>] |
<!-- /TABLE:ge_training -->

<!-- OBS:ge_training -->
- **On the V7, fine-tuning in ground effect does not pay.** The child trained with it tracks worse
  wherever it is flown, near the ground and at 30 m alike: it learns to lean on the extra lift.
- **On the Ranger it helps a little in both conditions,** with no interaction: at its operating
  point ground effect is too small to change what the policy learns.
<!-- /OBS:ge_training -->

### Robustness

The PPO attitude executor flies the four maneuvers under one disturbance at a time, at three
severities in physical units ([docs/disturbance_model.md](docs/disturbance_model.md)). Metrics come
from the true state, never the disturbed observation. The second table is the same executor trained
through random 0–100 ms delays (`falcons train --latency-ms 100`).

<!-- TABLE:robustness -->
| Airframe | Disturbance | Severity | Bank error [deg] | Survival |
|---|---|---|--:|--:|
| Airship V7 | Nominal | – | 1.9 [1.4, 2.1] | 1.00 |
| Airship V7 | Sensor noise | mild | 1.9 [1.9, 2.0] | 1.00 |
| Airship V7 | Sensor noise | medium | 1.9 [1.9, 2.0] | 1.00 |
| Airship V7 | Sensor noise | severe | 2.0 [1.9, 2.0] | 1.00 |
| Airship V7 | Sensor delay | mild | 1.9 [1.4, 2.1] | 1.00 |
| Airship V7 | Sensor delay | medium | 1.9 [1.4, 2.1] | 1.00 |
| Airship V7 | Sensor delay | severe | 1.9 [1.4, 2.1] | 1.00 |
| Airship V7 | Action delay | mild | 1.9 [1.4, 2.1] | 1.00 |
| Airship V7 | Action delay | medium | 1.9 [1.4, 2.1] | 1.00 |
| Airship V7 | Action delay | severe | 1.9 [1.4, 2.1] | 1.00 |
| Airship V7 | Turbulence | mild | 1.9 [1.9, 2.0] | 1.00 |
| Airship V7 | Turbulence | medium | 2.0 [1.9, 2.0] | 1.00 |
| Airship V7 | Turbulence | severe | 2.1 [2.0, 2.1] | 0.94 |
| Volantex Ranger | Nominal | – | 1.4 [1.0, 1.8] | 1.00 |
| Volantex Ranger | Sensor noise | mild | 1.5 [1.3, 1.6] | 1.00 |
| Volantex Ranger | Sensor noise | medium | 1.5 [1.4, 1.6] | 1.00 |
| Volantex Ranger | Sensor noise | severe | 1.5 [1.4, 1.6] | 1.00 |
| Volantex Ranger | Sensor delay | mild | 1.4 [0.9, 1.8] | 1.00 |
| Volantex Ranger | Sensor delay | medium | 4.4 [2.0, 5.4] | 1.00 |
| Volantex Ranger | Sensor delay | severe | 9.4 [9.0, 9.6] | 1.00 |
| Volantex Ranger | Action delay | mild | 1.4 [1.0, 1.8] | 1.00 |
| Volantex Ranger | Action delay | medium | 6.3 [3.9, 6.7] | 1.00 |
| Volantex Ranger | Action delay | severe | 11.2 [10.9, 11.5] | 1.00 |
| Volantex Ranger | Turbulence | mild | 1.5 [1.3, 1.6] | 1.00 |
| Volantex Ranger | Turbulence | medium | 1.5 [1.4, 1.6] | 1.00 |
| Volantex Ranger | Turbulence | severe | 1.5 [1.4, 1.6] | 1.00 |
<!-- /TABLE:robustness -->

<!-- TABLE:robustness_dr -->
| Airframe | Disturbance | Severity | Bank error [deg] | Survival |
|---|---|---|--:|--:|
| Airship V7 | Nominal | – | 2.9 [2.0, 3.5] | 1.00 |
| Airship V7 | Sensor noise | mild | 2.9 [2.8, 2.9] | 1.00 |
| Airship V7 | Sensor noise | medium | 2.9 [2.8, 2.9] | 1.00 |
| Airship V7 | Sensor noise | severe | 2.9 [2.8, 3.0] | 1.00 |
| Airship V7 | Sensor delay | mild | 2.9 [2.1, 3.5] | 1.00 |
| Airship V7 | Sensor delay | medium | 2.9 [2.1, 3.5] | 1.00 |
| Airship V7 | Sensor delay | severe | 2.9 [2.1, 3.5] | 1.00 |
| Airship V7 | Action delay | mild | 2.9 [2.1, 3.5] | 1.00 |
| Airship V7 | Action delay | medium | 2.9 [2.1, 3.5] | 1.00 |
| Airship V7 | Action delay | severe | 3.0 [2.1, 3.5] | 1.00 |
| Airship V7 | Turbulence | mild | 2.9 [2.8, 3.0] | 1.00 |
| Airship V7 | Turbulence | medium | 3.1 [3.0, 3.2] | 1.00 |
| Airship V7 | Turbulence | severe | 3.3 [3.3, 3.4] | 0.99 |
| Volantex Ranger | Nominal | – | 2.6 [1.7, 3.2] | 1.00 |
| Volantex Ranger | Sensor noise | mild | 2.6 [2.5, 2.6] | 1.00 |
| Volantex Ranger | Sensor noise | medium | 2.6 [2.5, 2.6] | 1.00 |
| Volantex Ranger | Sensor noise | severe | 2.6 [2.5, 2.6] | 1.00 |
| Volantex Ranger | Sensor delay | mild | 2.6 [1.7, 3.2] | 1.00 |
| Volantex Ranger | Sensor delay | medium | 2.6 [1.7, 3.2] | 1.00 |
| Volantex Ranger | Sensor delay | severe | 2.6 [1.7, 3.2] | 1.00 |
| Volantex Ranger | Action delay | mild | 2.6 [1.7, 3.2] | 1.00 |
| Volantex Ranger | Action delay | medium | 2.6 [1.7, 3.2] | 1.00 |
| Volantex Ranger | Action delay | severe | 2.6 [1.7, 3.2] | 1.00 |
| Volantex Ranger | Turbulence | mild | 2.6 [2.5, 2.7] | 0.99 |
| Volantex Ranger | Turbulence | medium | 2.7 [2.6, 2.8] | 0.92 |
| Volantex Ranger | Turbulence | severe | 2.7 [2.6, 2.8] | 0.79 |
<!-- /TABLE:robustness_dr -->

<!-- OBS:robustness -->
- **Sensor noise costs nothing.**
- **The V7 executor ignores delay; the Ranger's tracking degrades with it** but never departs.
- **Training through random delays removes the Ranger's sensitivity,** at the cost of some nominal
  accuracy and of survival in severe turbulence on the Ranger.
- **Severe turbulence ends a few V7 runs.**
<!-- /OBS:robustness -->

![The PPO executor under four disturbance families](results/figures/fig_robustness_ppo.png)

### Training

PPO trains for 1.5 × 10⁸ environment steps, SAC and TD3 for 3 × 10⁷, five seeds per cell. PPO was
re-timed alone on the GPU; the SAC and TD3 times are those of the shipped runs, which shared the GPU
with the other training jobs (the TD3 attitude runs with up to eighteen), so they are upper bounds.

![Learning curves, IQM over five seeds](results/figures/fig_learning_curves.png)

<!-- TABLE:training -->
| Algorithm | Altitude task (median per run) | Attitude task (median per run) | Timed |
|---|--:|--:|--:|
| PPO | 5 min | 5 min | alone on the GPU |
| SAC | 101 min | 101 min | GPU shared by four lanes |
| TD3 | 43 min | 274 min | GPU shared by four lanes |
<!-- /TABLE:training -->

### Simulator throughput

Env-steps per second on one RTX 4090 (`falcons benchmark throughput`). The full sweep and the
NeuralPlane comparison are on the [project page](docs/index.html#throughput).

<!-- TABLE:throughput -->
| Backend | Parallel envs | Env-steps / s |
|---|--:|--:|
| NumPy, CPU | 1 | 1,761 |
| Warp, GPU | 64 | 99,236 |
| Warp, GPU | 256 | 393,466 |
| Warp, GPU | 1,024 | 1,551,247 |
| Warp, GPU | 4,096 | 5,968,541 |
| Warp, GPU | 16,384 | 20,366,929 |
| PyTorch, GPU | 64 | 11,190 |
| PyTorch, GPU | 256 | 43,732 |
| PyTorch, GPU | 1,024 | 168,526 |
| PyTorch, GPU | 4,096 | 653,323 |
| PyTorch, GPU | 16,384 | 2,586,676 |
<!-- /TABLE:throughput -->

### Cross-simulator validation

Zero-input free flight from identical initial conditions in FALCON-S and JSBSim
(`tools/jsbsim_validate`), which shares the derivative set but has no ground effect: with it off
the two should match, with it on at 1 m they should part.

<!-- TABLE:jsbsim -->
| Airframe | h₀ | Ground effect | r altitude | r pitch rate | r airspeed | RMSE altitude [m] | RMSE airspeed [m/s] |
|---|--:|---|--:|--:|--:|--:|--:|
| Airship V7 | 1 m | on | 0.9993 | 0.8653 | 0.8487 | 0.079 | 0.093 |
| Airship V7 | 1 m | off | 1.0000 | 0.9997 | 1.0000 | 0.001 | 0.001 |
| Airship V7 | 10 m | off | 1.0000 | 0.9997 | 1.0000 | 0.008 | 0.002 |
| Airship V7 | 100 m | off | 1.0000 | 0.9997 | 1.0000 | 0.008 | 0.002 |
| Volantex Ranger | 1 m | on | 1.0000 | 0.9319 | 0.9915 | 0.023 | 0.022 |
| Volantex Ranger | 1 m | off | 1.0000 | 0.9962 | 0.9999 | 5.2×10<sup>-4</sup> | 0.003 |
| Volantex Ranger | 10 m | off | 1.0000 | 0.9986 | 1.0000 | 0.003 | 0.004 |
| Volantex Ranger | 100 m | off | 1.0000 | 0.9987 | 1.0000 | 0.004 | 0.004 |
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
