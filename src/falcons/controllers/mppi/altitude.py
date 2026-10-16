"""MPPI on the altitude-acquisition scenario: the classical MPPI row of the altitude table.

falcons.benchmark.altitude runs this in a child process and bakes the reference constants
(TRAJ_CONST_ALT, TRAJ_ACQ_H0, TRAJ_REF_VA) into that child's environment before importing this
module: falcons.controllers.mppi.reference freezes them into warp constants at import time.

The planner settings below were chosen by the sweep recorded in docs/mppi_tuning.md.
"""
import numpy as np
import warp as wp

from falcons.aircraft.params import load_params
from falcons.controllers.lqr.reference import compute_reference as compute_reference_np
from falcons.controllers.mppi.control import MPPIAltitudeControl
from falcons.controllers.mppi.reference import compute_reference
from falcons.sim.warp.aircraft import Aircraft
from falcons.sim.warp.termination import check_termination_batch

SAMPLES = 1000                 # sampled control sequences per step
HORIZON = 200                  # preview length in steps of dt = 0.01 s (2 s)
VERTICAL_SPEED_WEIGHT = 1.0    # weight of |climb - reference climb rate| (docs/mppi_tuning.md)
TEMPERATURE = 3.0              # softmax temperature lambda
CONTROL_COST_GAMMA = 0.03      # information-theoretic control cost, ~ the historical magnitude
NOISE_STD = dict(elevator=0.10, aileron=0.08, rudder=0.08, throttle=0.10)   # normalized commands


class MPPIAltitude:
    """Adapter for running MPPI control scenarios."""

    def __init__(self, aircraft_name: str = "Airship_V7", init_altitude: float = None,
                 mppi_seed: int = 0, samples: int = SAMPLES, horizon: int = HORIZON,
                 vertical_speed_weight: float = VERTICAL_SPEED_WEIGHT,
                 temperature: float = TEMPERATURE, control_cost_gamma: float = CONTROL_COST_GAMMA,
                 max_steps: int = 10000):
        self.aircraft_name = aircraft_name
        # Height (m) to spawn at. None = spawn ON the reference at t=0, which is what makes the
        # comparison against the other controllers fair: the previous hardcoded 1 m spawn put the
        # aircraft 9 m below a ramp_ascent reference that starts at 10 m, so the run was dominated
        # by (and usually stalled during) the catch-up climb rather than measuring tracking.
        self.init_altitude = init_altitude
        self.mppi_seed = mppi_seed
        self.samples = samples
        self.horizon = horizon
        self.vertical_speed_weight = vertical_speed_weight
        self.temperature = temperature
        self.control_cost_gamma = control_cost_gamma
        self.max_steps = int(max_steps)     # the benchmark scores 6000 steps; fly no more than that

    def run(self, scenario: dict) -> dict:
        """
        Run MPPI scenario and return only data needed for metrics.
        
        Returns:
            dict with keys: 'time', 'positions', 'references', 'actions'
        """
        trajectory_type = scenario['trajectory_type']
        use_sensors = scenario['use_sensor_noise']
        
        # Load configuration from JSON file
        try:
            config_objects = load_params(self.aircraft_name)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return {
                'time': np.array([]),
                'positions': np.array([]),
                'references': np.array([]),
                'actions': np.array([])
            }
        
        # Extract configuration objects
        AP = config_objects['aero_params']
        VP = config_objects['vehicle_params']
        CL = config_objects['control_limits']
        EP = config_objects['environment_params']
        
        # Convert to WARP structs
        AP = AP.as_warp_struct()
        VP.WP = VP.WP.as_warp_struct()
        VP.J = VP.J.as_warp_struct()
        VP.PP = VP.PP.as_warp_struct()
        
        config = {'AP': AP, 'VP': VP, 'CL': CL, 'EP': EP}
        
        # Initialize the real simulated aircraft. Integrated exactly like the RL environments
        # (envs/altitude.py, envs/attitude.py): fixed-step Dormand-Prince stages with the forces
        # frozen over the step (solver_type 1). The Aircraft default is explicit Euler.
        aircraft = Aircraft(1, "cuda", config, save_history=False)
        aircraft.solver_type = 1

        # Termination identical to the RL environments' check_termination_batch: ground contact
        # of the wing, |alpha| > 1.5 x stall angle, or airspeed below 0.6 x the airframe's
        # default airspeed (base_vel). The historical check stopped only at 4 x stall (2 x in the
        # kernel times 2 x at the call) and had no airspeed floor.
        base_vel = float(config_objects["default_initial_state"].linear_vel[0])
        alpha_limit = float(1.5 * VP.stall_angle)            # stall_angle is in radians
        va_min = 0.6 * base_vel
        wing_cg_z = float(VP.WP.cg_offset_vector[2])
        term = wp.zeros(1, dtype=wp.int32, device="cuda")
        reason = wp.zeros(1, dtype=wp.int32, device="cuda")

        # MPPI specs. The preview horizon (iterations x dt) is the planner's main tuning knob on
        # the acquisition task: a 1 s preview cannot see far enough ahead to arrest a commanded
        # descent, so it dives to catch the reference ramp and overshoots. The vertical-speed term
        # tracks the reference's own climb rate; see docs/mppi_tuning.md for the sweep.
        mppi_specs = {
            "number_of_sampled_trajectories": int(self.samples),
            "number_of_iterations_per_sample": int(self.horizon),
            "noise_std_elevator": NOISE_STD["elevator"],
            "noise_std_aileron": NOISE_STD["aileron"],
            "noise_std_rudder": NOISE_STD["rudder"],
            "noise_std_throttle": NOISE_STD["throttle"],
            "temperature": float(self.temperature),
            "control_cost_gamma": float(self.control_cost_gamma),
            "vertical_speed_weight": float(self.vertical_speed_weight),
        }
        
        # Define trajectory targets
        targets = {
            'trajectory_type': trajectory_type,
        }
        
        # Disable sensors and wind for the MPPI sampled trajectories
        if hasattr(VP, 'sensor_system'):
            VP.sensor_system = None
        if hasattr(EP, 'constant_wind'):
            EP.constant_wind = (0.0, 0.0, 0.0)
        if hasattr(EP, 'turbulence') and EP.turbulence:
            EP.turbulence = {'enable': False}
        
        config_MPPI = {'AP': AP, 'VP': VP, 'CL': CL, 'EP': EP}
        
        # Initialize the MPPI controller
        MPPI_controller = MPPIAltitudeControl(mppi_specs, targets, config_MPPI, save_history=False)
        # Pin the sampling noise: the controller's RNG is otherwise seeded from entropy by the
        # warp aircraft base whenever nobody calls seed(). This removes one source of run-to-run
        # variation but NOT all of it -- the cost reduction in falcons/controllers/mppi/kernels.py
        # sums rollout costs with wp.atomic_add, whose ordering across threads is not
        # deterministic, so identical seeds still yield slightly different plans. MPPI rows must
        # therefore be reported over repetitions, not as a single rollout; `mppi_seed` picks the
        # realization (None restores the historical entropy seeding).
        if self.mppi_seed is not None:
            MPPI_controller.seed(int(self.mppi_seed))
        
        # Initial state and action — spawn on the reference (see __init__) unless overridden
        z0 = -self.init_altitude if self.init_altitude is not None \
            else float(compute_reference_np(0.0, trajectory_type)[2])
        # Spawn at the airframe's own trim airspeed. The historical hardcoded 30 m/s is
        # roughly the V7's trim but TWICE the Volantex Ranger's, so on that airframe the
        # run measured a deceleration transient from double trim, not the controller.
        init_state = {
            "position": np.array([0, 0, z0], dtype=float),
            "linear_vel": np.array(config_objects["default_initial_state"].linear_vel,
                                   dtype=float),
            "angular_vel": np.array([0, 0, 0], dtype=float),
            "orientation": np.array([0, 0, 0, 1], dtype=float) 
        }
        init_action = {
            "elevator": 0.0,
            "aileron": 0.0,
            "rudder": 0.0,
            "throttle_left": 0.60,   
            "throttle_right": 0.60,
            "elevator_dot": 0.0,
            "aileron_dot": 0.0,
            "rudder_dot": 0.0
        }
        
        # Reset the aircraft and the MPPI controller
        aircraft.reset(init_state, init_action)
        MPPI_controller.reset(init_state, init_action)
        
        # Pre-allocate arrays
        max_steps = self.max_steps
        times = np.zeros(max_steps)
        positions = np.zeros((max_steps, 3))
        references = np.zeros((max_steps, 3))
        actions = np.zeros((max_steps, 5))
        
        print(f"\nRunning MPPI: {scenario['name']}")
        
        # Run simulation
        for step in range(max_steps):
            # MPPI step to get the next action (advances internal simulation time)
            action = MPPI_controller.MPPI_step()
            
            # Compute the reference trajectory using MPPI's internal time
            # Returns vec3f(y_lateral, z_altitude, forward_velocity)
            ref_traj = compute_reference(MPPI_controller.simulation_time, MPPI_controller.trajectory_type_id)
            
            # Apply the action and get sensor measurements
            state, sensor_measurements = aircraft.step(action, use_sensors)
            
            # Use measured or true state
            if use_sensors:
                current_state = {
                    "position": sensor_measurements['gps_position'].numpy()[0],
                    "linear_vel": sensor_measurements['gps_velocity'].numpy()[0],
                    "angular_vel": sensor_measurements['gyroscope'].numpy()[0],
                    "orientation": sensor_measurements['attitude_sensor'].numpy()[0]
                }
            else:
                current_state = {
                    "position": state['position'].numpy()[0],
                    "linear_vel": state['linear_vel'].numpy()[0],
                    "angular_vel": state['angular_vel'].numpy()[0],
                    "orientation": state['orientation'].numpy()[0]
                }
            
            # Map actions to standard format: [throttle_left, throttle_right, elevator, aileron, rudder]
            mapped_actions = [
                action.get('throttle_left', 0),
                action.get('throttle_right', 0),
                action.get('elevator', 0),
                action.get('aileron', 0),
                action.get('rudder', 0)
            ]
            
            # Save data for metrics
            times[step] = MPPI_controller.simulation_time
            positions[step] = current_state['position']
            references[step] = [current_state['position'][0], ref_traj.x, ref_traj.y]
            actions[step] = mapped_actions
            
            # Update MPPI with new state and action from the actual aircraft
            new_init_state = current_state
            new_init_action = {
                "elevator": aircraft._actuator_states['elevator'].numpy()[0],
                "aileron": aircraft._actuator_states['aileron'].numpy()[0],
                "rudder": aircraft._actuator_states['rudder'].numpy()[0],
                "throttle_left": aircraft._actuator_states['throttle_left'].numpy()[0],
                "throttle_right": aircraft._actuator_states['throttle_right'].numpy()[0],
                "elevator_dot": aircraft._actuator_states['elevator_dot'].numpy()[0],
                "aileron_dot": aircraft._actuator_states['aileron_dot'].numpy()[0],
                "rudder_dot": aircraft._actuator_states['rudder_dot'].numpy()[0]
            }
            
            # Check termination with the RL environments' own kernel and thresholds
            wp.launch(check_termination_batch, dim=1, inputs=[
                aircraft._state["position"], aircraft._alpha, aircraft._Va,
                wing_cg_z, alpha_limit, va_min, term, reason], device="cuda")
            r = int(reason.numpy()[0])
            if r:
                print(f"Simulation ended at step {step}: "
                      f"{'crashed into the surface' if r == 1 else 'stall (alpha or airspeed)'}")
                actual_steps = step + 1
                break
            
            # Reset MPPI controller with updated state for next iteration
            MPPI_controller.reset(new_init_state, new_init_action)
        else:
            actual_steps = max_steps
        
        # Return same format as LQR
        return {
            'time': times[:actual_steps],
            'positions': positions[:actual_steps],
            'references': references[:actual_steps],
            'actions': actions[:actual_steps]
        }
