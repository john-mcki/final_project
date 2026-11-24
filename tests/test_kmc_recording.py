"""Checks for event logging and the unchanged Version 0 stochastic process."""
import unittest
import numpy as np
from kmc_test import DislocationKMC
from kmc_visualize import verify_trace


class RecordingTests(unittest.TestCase):
    def test_recording_does_not_change_random_stream_or_results(self):
        for stress in (0, 10, -10, 100):
            plain = DislocationKMC(stress_mpa=stress, seed=454)
            logged = DislocationKMC(stress_mpa=stress, seed=454)
            self.assertEqual(plain.run_simulation(500), logged.run_simulation(500, record_events=True))
            np.testing.assert_array_equal(plain.positions, logged.positions)
            np.testing.assert_array_equal(plain.time_history, logged.time_history)
            self.assertEqual(plain.rng.random(), logged.rng.random())

    def test_every_event_reconstructs_the_original_state(self):
        sim = DislocationKMC(stress_mpa=0, num_segments=7, seed=43)
        sim.run_simulation(2000, record_events=True)
        self.assertTrue(verify_trace(sim.recorded_trace())["passed"])
        self.assertEqual(set(e["direction"] for e in sim.events), {-1, 1})

    def test_trace_rng_state_reproduces_a_later_run(self):
        sim = DislocationKMC(seed=43)
        sim.run_simulation(25)
        expected = sim.run_simulation(50, record_events=True)
        state = sim.recorded_trace()["rng_state_before"]
        replay = DislocationKMC(seed=0)
        replay.rng.set_state((state["algorithm"], np.array(state["keys"], dtype=np.uint32),
                              state["position"], state["has_gauss"], state["cached_gaussian"]))
        self.assertEqual(expected, replay.run_simulation(50))

    def test_stress_reversal_swaps_rates_below_floor(self):
        for stress in (0, 10, 100):
            positive = DislocationKMC(stress_mpa=stress).compute_rates()
            negative = DislocationKMC(stress_mpa=-stress).compute_rates()
            np.testing.assert_array_equal(positive, negative[::-1])

    def test_segment_count_changes_time_not_mean_velocity(self):
        values = []
        for count in (1, 10, 100):
            sim = DislocationKMC(num_segments=count, stress_mpa=10, seed=454)
            values.append(sim.run_simulation(500)["velocity_m_s"])
        np.testing.assert_allclose(values, values[0], rtol=1e-12)

    def test_zero_events_are_a_valid_record(self):
        sim = DislocationKMC(seed=0)
        sim.run_simulation(0, record_events=True)
        self.assertTrue(verify_trace(sim.recorded_trace())["passed"])
        self.assertEqual(sim.recorded_trace()["results"]["velocity_m_s"], 0)

    def test_reset_drops_previous_trace(self):
        sim = DislocationKMC(seed=0)
        sim.run_simulation(3, record_events=True)
        sim.run_simulation(1)
        self.assertEqual(sim.events, [])
        with self.assertRaises(ValueError): sim.recorded_trace()

    def test_original_forward_floor_is_reported_not_changed(self):
        sim = DislocationKMC(stress_mpa=200, seed=0)
        sim.run_simulation(1, record_events=True)
        self.assertTrue(sim.recorded_trace()["parameters"]["forward_barrier_floor_active"])
        self.assertEqual(sim.compute_barriers()[0], 1e-4)

    def test_invalid_inputs_fail_before_sampling(self):
        for kw in ({"temp_k": 0}, {"temp_k": float("nan")}, {"num_segments": 0},
                   {"num_segments": 1.5}, {"stress_mpa": float("inf")}, {"disloc_density": -1}):
            with self.assertRaises(ValueError): DislocationKMC(**kw)
        with self.assertRaises(ValueError): DislocationKMC().run_simulation(-1)


if __name__ == "__main__":
    unittest.main()
