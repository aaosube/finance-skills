import unittest

import pandas as pd

from regime_support import RegimeSupportConfig, audit_regime_support


class RegimeSupportTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "regime": ["CALM"] * 8 + ["VOLATILE"] * 2,
                "position": [0, 1, 1, 0, -1, 0, 1, 0, 1, 0],
            }
        )

    def test_without_thresholds_is_diagnostic_only(self):
        out = audit_regime_support(self.df, RegimeSupportConfig(regime_col="regime"))
        self.assertEqual(out["status"], "DIAGNOSTIC_ONLY")
        self.assertEqual(out["regimes"]["CALM"]["observations"], 8)
        self.assertEqual(out["regimes"]["VOLATILE"]["observations"], 2)

    def test_explicit_support_threshold_can_fail_small_regime(self):
        out = audit_regime_support(
            self.df,
            RegimeSupportConfig(regime_col="regime", min_observations=5),
        )
        self.assertEqual(out["status"], "FAIL")
        self.assertEqual(out["regimes"]["CALM"]["status"], "PASS")
        self.assertEqual(out["regimes"]["VOLATILE"]["status"], "FAIL")

    def test_active_support_is_separate_from_total_support(self):
        out = audit_regime_support(
            self.df,
            RegimeSupportConfig(
                regime_col="regime",
                min_observations=2,
                min_active_observations=2,
            ),
        )
        self.assertEqual(out["regimes"]["VOLATILE"]["active_observations"], 1)
        self.assertEqual(out["regimes"]["VOLATILE"]["status"], "FAIL")

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            RegimeSupportConfig(regime_col="regime", min_observations=0).validate()


if __name__ == "__main__":
    unittest.main()
