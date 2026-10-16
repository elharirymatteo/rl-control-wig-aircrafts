# MPPI altitude tuning

The complete tuning budget behind the MPPI row of the altitude table: every configuration that was
flown, with its result. Rerun with `scripts/mppi_tuning.py` (sweep) and
`scripts/mppi_tuning.py --gamma --horizon 200 --w-vz 1.0` (control-cost check); raw numbers in
`results/_tuning/mppi_altitude_grid.csv`.

**Budget:** 11 configurations x 2 airframes x 12 episodes = 264 episodes of 40 s, sampler seed 0,
1000 samples per step. Nothing else was tuned: noise standard deviations (0.10 elevator/throttle,
0.08 aileron/rudder), temperature (3.0), the cost weights other than the vertical-speed term, and the
attitude-task MPPI are unchanged.

**Protocol:** the altitude-acquisition protocol of `falcons.benchmark.altitude` on a reduced grid:
commanded altitudes 20, 60 and 100 m; the first two of the benchmark's own spawn-offset draws on each
side of the target (|offset| ~ U(12, 18) m); 40 s episodes instead of 60 s. Every configuration
already runs under the harmonized conditions of this revision:

- the plant is integrated like the RL environments (fixed-step Dormand-Prince, `solver_type = 1`;
  it was explicit Euler);
- an episode terminates exactly as an RL episode does (`check_termination_batch`: wing ground
  contact, |alpha| > 1.5 x stall angle, airspeed < 0.6 x base_vel). The previous check stopped only
  at 4 x the stall angle (2 x in the kernel times 2 x at the call site) and had no airspeed floor;
- the reference ramps at the RL environment's `ref_rate` and commands `base_vel`, the airspeed the
  RL environments and the LQR use (previously `TRIM_VA`, and the ramp rate was never set);
- the cost tracks the reference's climb rate with weight `w_vz` (`climb_rate_tracking_cost`), in
  place of the former sink-rate term |v_z| that, when enabled, penalized the commanded ramp itself;
- the importance-sampling control cost is lambda * gamma * u * eps / sigma^2 with sigma the noise
  STANDARD DEVIATION. The former term, lambda * (u + eps) * u / sqrt(sigma), equals it with
  gamma = sigma^1.5 (0.032 / 0.023); gamma = 0.03 keeps that magnitude.

## Sweep: horizon x vertical-speed weight (gamma = 0.03)

| airframe | horizon | w_vz | gamma | episodes | survival | acquired | settled RMSE (m) | overshoot (m) | settling (s) |
|---|---|---|---|---|---|---|---|---|---|
| Airship_V7 | 1 s | 0 | 0.03 | 12 | 1.00 | 1.00 | 0.01 | 0.02 | 7.3 |
| Volantex_Ranger | 1 s | 0 | 0.03 | 12 | 0.50 | 0.50 | 0.02 | 0.01 | 7.8 |
| Airship_V7 | 1 s | 1 | 0.03 | 12 | 1.00 | 0.92 | 0.57 | 0.40 | 9.2 |
| Volantex_Ranger | 1 s | 1 | 0.03 | 12 | 1.00 | 1.00 | 0.02 | 0.15 | 7.3 |
| Airship_V7 | 1 s | 3 | 0.03 | 12 | 1.00 | 0.83 | 1.37 | 0.07 | 13.6 |
| Volantex_Ranger | 1 s | 3 | 0.03 | 12 | 1.00 | 0.67 | 8.00 | 1.43 | 14.2 |
| Airship_V7 | 2 s | 0 | 0.03 | 12 | 1.00 | 1.00 | 0.05 | 0.17 | 7.4 |
| Volantex_Ranger | 2 s | 0 | 0.03 | 12 | 1.00 | 1.00 | 0.09 | 0.32 | 7.4 |
| Airship_V7 | 2 s | 1 | 0.03 | 12 | 1.00 | 1.00 | 0.05 | 0.19 | 7.4 |
| Volantex_Ranger | 2 s | 1 | 0.03 | 12 | 1.00 | 1.00 | 0.11 | 0.35 | 7.6 |
| Airship_V7 | 2 s | 3 | 0.03 | 12 | 1.00 | 1.00 | 0.10 | 0.23 | 10.8 |
| Volantex_Ranger | 2 s | 3 | 0.03 | 12 | 1.00 | 0.92 | 0.18 | 0.46 | 7.7 |
| Airship_V7 | 3 s | 0 | 0.03 | 12 | 1.00 | 0.92 | 0.28 | 0.77 | 11.1 |
| Volantex_Ranger | 3 s | 0 | 0.03 | 12 | 1.00 | 0.75 | 0.41 | 1.07 | 21.7 |
| Airship_V7 | 3 s | 1 | 0.03 | 12 | 1.00 | 1.00 | 0.31 | 0.86 | 14.0 |
| Volantex_Ranger | 3 s | 1 | 0.03 | 12 | 1.00 | 0.92 | 0.61 | 2.10 | 24.8 |
| Airship_V7 | 3 s | 3 | 0.03 | 12 | 1.00 | 0.92 | 0.26 | 0.48 | 13.3 |
| Volantex_Ranger | 3 s | 3 | 0.03 | 12 | 1.00 | 0.83 | 0.76 | 1.64 | 29.4 |

## Control-cost check at the chosen horizon and weight

The gamma = 0.03 row repeats the sweep's (2 s, w_vz = 1) configuration in a separate run, so the
pair also shows the run-to-run spread of a cell.

| airframe | horizon | w_vz | gamma | episodes | survival | acquired | settled RMSE (m) | overshoot (m) | settling (s) |
|---|---|---|---|---|---|---|---|---|---|
| Airship_V7 | 2 s | 1 | 0.00 | 12 | 1.00 | 1.00 | 0.06 | 0.24 | 7.4 |
| Volantex_Ranger | 2 s | 1 | 0.00 | 12 | 0.92 | 0.92 | 0.14 | 0.36 | 10.6 |
| Airship_V7 | 2 s | 1 | 0.03 | 12 | 1.00 | 1.00 | 0.06 | 0.20 | 7.4 |
| Volantex_Ranger | 2 s | 1 | 0.03 | 12 | 1.00 | 1.00 | 0.10 | 0.39 | 7.4 |

## Choice

**Horizon 2 s, w_vz = 1, gamma = 0.03, for both airframes** (`falcons/controllers/mppi/altitude.py`).

- A 1 s preview loses half the Volantex Ranger episodes without the vertical-speed term; the term
  rescues them (survival 0.50 -> 1.00) but costs the V7 acquisitions (1.00 -> 0.92).
- At 2 s both airframes survive and acquire every episode, with settled RMSE <= 0.11 m and
  overshoot <= 0.35 m; the vertical-speed weight is neutral there (w_vz 0 vs 1 within the
  episode-to-episode spread), and is kept for the robustness it showed at 1 s.
- A 3 s preview is worse on both airframes (more overshoot, slower settling), so the shortest
  horizon that works is also the best one.
- Dropping the control cost (gamma = 0) loses one Ranger episode (survival 0.92); gamma = 0.03, the
  magnitude of the former term, keeps all of them. Its compute cost is 9.9 ms per control step on an
  RTX 4090 (7.0 ms at 1 s, 13.1 ms at 3 s), shared with a training job.

The published table's MPPI row (V7: survival 0.93, acquisition 0.01, overshoot 41.5 m) came from the
1 s, position-only planner under the old conditions; the 1 s, w_vz = 0 row above is the same planner
under the harmonized conditions.
