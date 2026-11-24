import numpy as np
import matplotlib.pyplot as plt

class DislocationKMC:
    """
    Independent 1D dislocation segments with constant forward/backward rates.

    Optional event recording observes the original algorithm without consuming
    additional random numbers. Segments have no mechanical connection.
    """
    def __init__(self, temp_k=300.0, stress_mpa=100.0, num_segments=100, disloc_density=1e9, seed=None):
        if not np.isfinite(temp_k) or temp_k <= 0:
            raise ValueError("temp_k must be finite and positive")
        if not np.isfinite(stress_mpa):
            raise ValueError("stress_mpa must be finite")
        if not isinstance(num_segments, (int, np.integer)) or isinstance(num_segments, bool) or num_segments <= 0:
            raise ValueError("num_segments must be a positive integer")
        if not np.isfinite(disloc_density) or disloc_density < 0:
            raise ValueError("disloc_density must be finite and nonnegative")
        self.seed = seed
        # RandomState preserves the original np.random.seed(seed) stream.
        self.rng = np.random if seed is None else np.random.RandomState(seed)
        self.events = []
        self._trace = None
        # Fundamental physical constants
        self.k_B_eV = 8.617333262e-5      # eV/K
        self.k_B_J = 1.380649e-23         # J/K
        self.eV_to_J = 1.602176634e-19    # J/eV

        # FCC Ni geometry; the hopping prefactor below is an uncalibrated input.
        self.a_lattice = 0.352e-9         # Lattice parameter (m)
                                          # Approximately in-line with lattice parameter in materials-project
        self.b = self.a_lattice / np.sqrt(2)  # Burgers vector magnitude (m) ~ 0.249 nm
        self.nu_0 = 1e12                  # Attempt frequency (Hz)
        
        # Simulation parameters
        self.T = temp_k                   # Temperature (K)
        self.tau = stress_mpa * 1e6       # Resolved shear stress (Pa)
        self.num_segments = num_segments  # Number of dislocation line segments
        self.rho = disloc_density         # Mobile dislocation density (m^-2)

        # Assumed effective barrier parameters inherited from the starting code.
        # No source or fit currently validates these values for Ni glide.
        # Stress enters only through E± = E0 ∓ tau*V*, not a force integration.
        self.E_barrier_0 = 0.35           # Unstressed energy barrier (eV)
        self.activation_volume = 20 * (self.b ** 3)  # Activation volume V* (m^3)

        # Dislocation state track: array of segment positions along glide direction x (m)
        self.positions = np.zeros(self.num_segments)
        self.time = 0.0

        # Data collection
        self.time_history = [0.0]
        self.mean_position_history = [0.0]

    def compute_barriers(self):
        """Original barrier law, including its forward-only floor."""
        work_eV = (self.tau * self.activation_volume) / self.eV_to_J
        return max(1e-4, self.E_barrier_0 - work_eV), self.E_barrier_0 + work_eV

    def compute_rates(self):
        """
        Calculates stress-dependent energy barriers and transition rates 
        for forward and backward dislocation hops.
        """
        E_forward, E_backward = self.compute_barriers()

        # Arrhenius transition rates (s^-1)
        rate_forward = self.nu_0 * np.exp(-E_forward / (self.k_B_eV * self.T))
        rate_backward = self.nu_0 * np.exp(-E_backward / (self.k_B_eV * self.T))

        return rate_forward, rate_backward

    def run_simulation(self, total_steps=10000, *, record_events=False):
        """
        Executes the Residence-Time / Bortz-Kalos-Lebowitz (BKL) kMC algorithm.
        """
        if not isinstance(total_steps, (int, np.integer)) or isinstance(total_steps, bool) or total_steps < 0:
            raise ValueError("total_steps must be a nonnegative integer")
        self.events = []
        self._trace = None
        self.positions = np.zeros(self.num_segments)
        self.time = 0.0
        self.time_history = [0.0]
        self.mean_position_history = [0.0]

        rate_fwd, rate_bwd = self.compute_rates()
        step_distance = self.b  # Hop distance equals Burgers vector magnitude
        if not np.all(np.isfinite([rate_fwd, rate_bwd])):
            raise ValueError("Rates must be finite; check temperature and stress")
        if record_events:
            forward_barrier, backward_barrier = self.compute_barriers()
            state = self.rng.get_state()
            self._trace = {
                "schema_version": 1,
                "model_version": "v0_independent_glide",
                "parameters": {
                    "temperature_k": float(self.T), "stress_mpa": float(self.tau / 1e6),
                    "num_segments": int(self.num_segments), "requested_events": int(total_steps),
                    "mobile_density_m2": float(self.rho), "lattice_parameter_m": float(self.a_lattice),
                    "burgers_vector_m": float(self.b), "hop_distance_m": float(step_distance),
                    "attempt_frequency_s1": float(self.nu_0), "barrier_zero_ev": float(self.E_barrier_0),
                    "activation_volume_m3": float(self.activation_volume), "k_B_ev_k": self.k_B_eV,
                    "barrier_forward_ev": float(forward_barrier), "barrier_backward_ev": float(backward_barrier),
                    "rate_forward_s1": float(rate_fwd), "rate_backward_s1": float(rate_bwd),
                    "analytic_velocity_m_s": float(step_distance * (rate_fwd - rate_bwd)),
                    "seed": None if self.seed is None else int(self.seed),
                    "forward_barrier_floor_active": bool(self.E_barrier_0 - self.tau * self.activation_volume / self.eV_to_J <= 1e-4),
                },
                "rng_state_before": {"algorithm": state[0], "keys": state[1].tolist(),
                                     "position": int(state[2]), "has_gauss": int(state[3]),
                                     "cached_gaussian": float(state[4])},
                "initial_positions_m": self.positions.tolist(),
                "events": self.events,
            }

        for _ in range(total_steps):
            # Array of individual event rates for all segments ( forward & backward )
            # Index 0 to N-1: Forward transitions; N to 2N-1: Backward transitions
            rates = np.hstack([
                np.full(self.num_segments, rate_fwd),
                np.full(self.num_segments, rate_bwd)
            ])
            
            total_rate = np.sum(rates)
            if total_rate <= 0:
                break

            # 1. Advance time via Poisson process: dt = -ln(u) / R_total
            time_before = self.time
            u1 = self.rng.uniform(1e-12, 1.0)
            dt = -np.log(u1) / total_rate
            self.time += dt

            # 2. Select discrete event based on cumulative rate distribution
            u2 = self.rng.uniform(0.0, 1.0)
            cum_rates = np.cumsum(rates) / total_rate
            event_idx = np.searchsorted(cum_rates, u2)
            seg_id = int(event_idx % self.num_segments)
            position_before = self.positions[seg_id]

            # 3. Update segment state
            if event_idx < self.num_segments:
                # Forward move
                seg_id = event_idx
                self.positions[seg_id] += step_distance
            else:
                # Backward move
                seg_id = event_idx - self.num_segments
                self.positions[seg_id] -= step_distance

            # Record trajectory
            self.time_history.append(self.time)
            self.mean_position_history.append(np.mean(self.positions))
            if record_events:
                self.events.append({
                    "event": len(self.events) + 1, "segment_id": int(seg_id),
                    "direction": 1 if event_idx < self.num_segments else -1,
                    "time_before_s": float(time_before), "time_s": float(self.time), "dt_s": float(dt),
                    "position_before_m": float(position_before), "position_after_m": float(self.positions[seg_id]),
                    "mean_position_m": float(self.mean_position_history[-1]),
                    "rate_forward_s1": float(rate_fwd), "rate_backward_s1": float(rate_bwd),
                    "total_rate_s1": float(total_rate),
                    "selected_rate_s1": float(rate_fwd if event_idx < self.num_segments else rate_bwd),
                    "u_time": float(u1), "u_event": float(u2),
                })

        result = self.calculate_macroscopic_properties()
        if self._trace is not None:
            self._trace["results"] = result.copy()
            self._trace["final_positions_m"] = self.positions.tolist()
        return result

    def recorded_trace(self):
        """Return the most recent recorded run in SI units; no interpolation."""
        if self._trace is None:
            raise ValueError("Run with record_events=True before exporting a trace")
        return self._trace

    def calculate_macroscopic_properties(self):
        """
        Derives average dislocation velocity (v) and plastic strain rate (dot_gamma) 
        using Orowan's equation: dot_gamma = rho * b * v
        """
        total_time = self.time_history[-1]
        total_displacement = self.mean_position_history[-1] - self.mean_position_history[0]

        # Dislocation velocity (m/s)
        velocity = total_displacement / total_time if total_time > 0 else 0.0

        # Orowan plastic strain rate (s^-1)
        strain_rate = self.rho * self.b * velocity

        return {
            "total_time_s": total_time,
            "total_displacement_m": total_displacement,
            "velocity_m_s": velocity,
            "strain_rate_s1": strain_rate
        }


def stress_sweep_experiment(temp_k=300.0, stresses_mpa=np.linspace(20, 200, 10)):
    """
    Sweeps through applied stress levels to evaluate stress dependence 
    of velocity and strain rate for model validation.
    """
    velocities = []
    strain_rates = []

    for stress in stresses_mpa:
        sim = DislocationKMC(temp_k=temp_k, stress_mpa=stress, num_segments=50)
        results = sim.run_simulation(total_steps=5000)
        velocities.append(results["velocity_m_s"])
        strain_rates.append(results["strain_rate_s1"])

    return stresses_mpa, np.array(velocities), np.array(strain_rates)


if __name__ == "__main__":
    # Single simulation run example
    kmc = DislocationKMC(temp_k=300.0, stress_mpa=10.0, num_segments=100)
    metrics = kmc.run_simulation(total_steps=10000)

    print("--- Simulation Metrics (Pure Ni) ---")
    print(f"Total Simulation Time : {metrics['total_time_s']:.4e} s")
    print(f"Net Dislocation Glide : {metrics['total_displacement_m'] * 1e9:.3f} nm")
    print(f"Dislocation Velocity  : {metrics['velocity_m_s']:.4e} m/s")
    print(f"Plastic Strain Rate   : {metrics['strain_rate_s1']:.4e} s^-1")
