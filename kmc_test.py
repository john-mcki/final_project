import numpy as np
import matplotlib.pyplot as plt

class DislocationKMC:
    """
    Kinetic Monte Carlo (kMC) model for simulating 1D/2D dislocation segment 
    motion in pure FCC Nickel under applied stress and temperature.
    """
    def __init__(self, temp_k=300.0, stress_mpa=100.0, num_segments=100, disloc_density=1e9):
        # Fundamental physical constants
        self.k_B_eV = 8.617333262e-5      # eV/K
        self.k_B_J = 1.380649e-23         # J/K
        self.eV_to_J = 1.602176634e-19    # J/eV

        # Material parameters for pure Nickel (FCC)
        self.a_lattice = 0.352e-9         # Lattice parameter (m)
                                          # Approximately in-line with lattice parameter in materials-project
        self.b = self.a_lattice / np.sqrt(2)  # Burgers vector magnitude (m) ~ 0.249 nm
        self.nu_0 = 1e12                  # Attempt frequency (Hz)
        
        # Simulation parameters
        self.T = temp_k                   # Temperature (K)
        self.tau = stress_mpa * 1e6       # Resolved shear stress (Pa)
        self.num_segments = num_segments  # Number of dislocation line segments
        self.rho = disloc_density         # Mobile dislocation density (m^-2)

        # Potential Energy Surface (PES) parameters for Pure Ni
        self.E_barrier_0 = 0.35           # Unstressed energy barrier (eV)
        self.activation_volume = 20 * (self.b ** 3)  # Activation volume V* (m^3)

        # Dislocation state track: array of segment positions along glide direction x (m)
        self.positions = np.zeros(self.num_segments)
        self.time = 0.0

        # Data collection
        self.time_history = [0.0]
        self.mean_position_history = [0.0]

    def compute_rates(self):
        """
        Calculates stress-dependent energy barriers and transition rates 
        for forward and backward dislocation hops.
        """
        # Mechanical work performed by applied stress (eV)
        work_eV = (self.tau * self.activation_volume) / self.eV_to_J

        # Energy barriers modified by stress
        E_forward = max(1e-4, self.E_barrier_0 - work_eV)
        E_backward = self.E_barrier_0 + work_eV

        # Arrhenius transition rates (s^-1)
        rate_forward = self.nu_0 * np.exp(-E_forward / (self.k_B_eV * self.T))
        rate_backward = self.nu_0 * np.exp(-E_backward / (self.k_B_eV * self.T))

        return rate_forward, rate_backward

    def run_simulation(self, total_steps=10000):
        """
        Executes the Residence-Time / Bortz-Kalos-Lebowitz (BKL) kMC algorithm.
        """
        self.positions = np.zeros(self.num_segments)
        self.time = 0.0
        self.time_history = [0.0]
        self.mean_position_history = [0.0]

        rate_fwd, rate_bwd = self.compute_rates()
        step_distance = self.b  # Hop distance equals Burgers vector magnitude

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
            u1 = np.random.uniform(1e-12, 1.0)
            dt = -np.log(u1) / total_rate
            self.time += dt

            # 2. Select discrete event based on cumulative rate distribution
            u2 = np.random.uniform(0.0, 1.0)
            cum_rates = np.cumsum(rates) / total_rate
            event_idx = np.searchsorted(cum_rates, u2)

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

        return self.calculate_macroscopic_properties()

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