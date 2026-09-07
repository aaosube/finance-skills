import unittest

import numpy as np
import pandas as pd

from structural_mass import fit_structural_mass, rms_mass_from_z


class StructuralMassTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        n = 120
        base = rng.normal(size=n)
        self.history = pd.DataFrame(
            {
                "gex": 1.0 * base + rng.normal(scale=0.3, size=n),
                "dex": 0.7 * base + rng.normal(scale=0.5, size=n),
                "vex": -0.4 * base + rng.normal(scale=0.6, size=n),
                "chex": rng.normal(size=n),
                "oi": 0.2 * base + rng.normal(scale=0.8, size=n),
            },
            index=pd.date_range("2026-01-01", periods=n, freq="h"),
        )

    def test_fitted_mass_is_finite_and_outlier_is_larger(self):
        model = fit_structural_mass(self.history, min_rows=30)
        center = {c: float(self.history[c].median()) for c in self.history.columns}
        ordinary = model.score(center)
        outlier = dict(center)
        outlier["gex"] += 8.0
        outlier["dex"] += 6.0
        stressed = model.score(outlier)
        self.assertTrue(np.isfinite(ordinary["mass"]))
        self.assertTrue(np.isfinite(stressed["mass"]))
        self.assertGreater(stressed["mass"], ordinary["mass"])
        self.assertFalse(stressed["distance_included"])
        self.assertIsNone(stressed["probability"])

    def test_missing_history_feature_fails_closed(self):
        with self.assertRaises(ValueError):
            fit_structural_mass(self.history.drop(columns=["chex"]), min_rows=30)

    def test_insufficient_rows_fail_closed(self):
        with self.assertRaises(ValueError):
            fit_structural_mass(self.history.iloc[:10], min_rows=30)

    def test_rms_fallback_is_exact_and_not_probability(self):
        out = rms_mass_from_z([1.0, -1.0, 2.0, -2.0])
        self.assertAlmostEqual(out["mass"], np.sqrt(2.5))
        self.assertEqual(out["status"], "RMS_FALLBACK_EXPLORATORY")
        self.assertIsNone(out["probability"])


if __name__ == "__main__":
    unittest.main()
