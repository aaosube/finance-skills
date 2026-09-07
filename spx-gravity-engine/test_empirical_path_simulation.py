import unittest

import numpy as np

from empirical_path_simulation import BlockBootstrapConfig, empirical_first_hit


class EmpiricalPathSimulationTests(unittest.TestCase):
    def test_constant_positive_history_hits_upper_first(self):
        history = np.full(50, 0.01)
        out = empirical_first_hit(
            history,
            start_price=100.0,
            lower_barrier=95.0,
            upper_barrier=102.0,
            config=BlockBootstrapConfig(horizon_steps=5, n_paths=200, block_size=3, seed=1),
        )
        self.assertAlmostEqual(out["upper_first_frequency"], 1.0)
        self.assertAlmostEqual(out["lower_first_frequency"], 0.0)
        self.assertAlmostEqual(out["neither_frequency"], 0.0)
        self.assertEqual(out["status"], "EMPIRICAL_BLOCK_BOOTSTRAP_UNCALIBRATED")

    def test_frequencies_sum_to_one(self):
        history = np.array([0.004, -0.003, 0.002, -0.005, 0.006, -0.001] * 20)
        out = empirical_first_hit(
            history,
            start_price=100.0,
            lower_barrier=98.5,
            upper_barrier=101.5,
            config=BlockBootstrapConfig(horizon_steps=8, n_paths=500, block_size=4, seed=9),
        )
        total = out["upper_first_frequency"] + out["lower_first_frequency"] + out["neither_frequency"]
        self.assertAlmostEqual(total, 1.0, places=12)

    def test_invalid_barriers_fail_closed(self):
        with self.assertRaises(ValueError):
            empirical_first_hit(
                [0.001, -0.001, 0.002],
                start_price=100.0,
                lower_barrier=101.0,
                upper_barrier=102.0,
                config=BlockBootstrapConfig(horizon_steps=2, n_paths=10, block_size=2),
            )


if __name__ == "__main__":
    unittest.main()
