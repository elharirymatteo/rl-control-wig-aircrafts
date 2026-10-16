# TMLR revision tracker

Branch `tmlr-revision` (worktree `../falcon-s-tmlr`), based on GitHub `main` (48973c2, which
already carries the sideslip-sign fix c6d61f7). The paper is `../falcon-s/main.tex` (untracked).

## Decisions (user)
- Baseline: GitHub `main` + fixes; polynomial aero + Phillips-Hunsaker GE; five airframes.
  `aero/openvsp-derivatives` stays for later work.
- GE drag: correction on induced drag only, k from a parabolic polar fit per airframe.
- MPPI: fix the vertical-speed term and horizon (not removed).
- LQR: measure and state the bank envelope (no gain scheduling).
- Statistics: 5 training seeds, IQM + bootstrap CIs, dropped-run counts, 100 realizations per
  robustness cell.

## Code changes (merged into tmlr-revision)
| item | commit | notes |
|---|---|---|
| GE on induced drag only; AR = b^2/S | ffa0e94 | A0S polynomial gives negative parasitic drag -> Oswald k (e = 0.82) |
| CPU plant aligned to Warp; single-step parity test | dc0a411 (rev/parity) | tolerances ~10x float32 round-off; Torch plant is longitudinal only |
| GE train x eval 2x2 study (`falcons benchmark ge-training`) | 452f93e (rev/gestudy) | survivor-only metrics, span-scaled bands |
| IQM / bootstrap CIs, counts, N_REAL robustness, maneuver termination bug | f766c7e (rev/stats) | `fly()` no longer scores a respawned aircraft |
| LQR derivation script, re-derived gains, CPU termination | 6619336 (rev/lqr) | old acquisition gains unstable on the corrected plant |
| JSBSim tool: induced-drag GE, GE switch, paper table | c46fc88 | results/jsbsim/paper_table.csv |
| Turbulence calibrated from the MIL-F-8785C relation | dc5f1da | old labels were ~3x too low |
| Figures at TMLR text width (user WIP) | ff5a2ea | base for the readability pass |
| MPPI climb-rate term, 2 s horizon, termination = env rules, DP5 plant | 5170e5f (rev/mppi) | docs/mppi_tuning.md: 11 configs x 2 airframes x 12 episodes |
| Runaway guard 2 km -> 20 km in fly()/fly_batch() | 5170e5f | the 2 km box scored a 5 deg constant-bank LQR probe as a departure |
| Figures redrawn at printed size, 2-D maneuver tracks, throughput + learning curves | rev/figures | in progress |

## Compute still to run (after the training queue)
1. `scripts/train_seeds.sh` -> runs/checkpoints (60 + 3 runs; ~40 h on the shared GPU).
2. Copy runs/checkpoints over checkpoints/, then: `falcons benchmark altitude` (~2.5 h),
   `maneuvers --refresh` (~30 min), `ground-effect` (trim + energy), `robustness` (N_REAL 100,
   ~50 min on a free GPU), `ge-training` (~40 min), `throughput` (6 min, IDLE GPU required),
   `scripts/lqr_envelope.py`, `tools/jsbsim_validate/paper_table.py` (done on current code).
3. Regenerate tests/golden (configs, warp_step.npz, tables) and the docs/ project-page data.

## Paper items
- Done: claims (high-fidelity, 13.4x, five airframes, tasks, videos), aero + GE equations,
  validity and limitations, integrator, Dryden filter order + MIL citations, parity, JSBSim
  (no GE term, r/RMSE/max), LQR design, appendices (airframes, aero, rewards, training, SAC study).
- Done since: MPPI description + tuning appendix, LQR design appendix.
- Pending numbers: altitude / maneuver / GE / robustness tables and text, throughput,
  wall-clock per run, learning-curve figure, LQR envelope (results ready: runs/results/lqr_envelope.csv),
  GE 2x2 section.
- Pending: figures readability pass; 12-page budget (main text currently ~0.6 page over).

## Open questions for the authors
- Phillips & Hunsaker validity range and whether the implemented coefficients drop their C_L
  dependence (full text not accessible here).
- A0S data: polynomial (negative parasitic drag, half lift slope) and propeller constants
  (momentum theory gives 131 N at cruise vs 60 N rated).
- Volantex J_xz sign opposite to the other four airframes.
