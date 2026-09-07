import unittest

import pandas as pd

from iv_surface_dynamics import SurfaceConfig, describe_surface_snapshot, surface_change_outcome


class IVSurfaceDynamicsTests(unittest.TestCase):
    def _snapshot(self, as_of: str, shift: float = 0.0) -> pd.DataFrame:
        rows = []
        for expiry, forward, base in [
            ("2026-09-18T20:00:00Z", 100.0, 0.20),
            ("2026-10-16T20:00:00Z", 101.0, 0.22),
        ]:
            for strike, bump in [(90, 0.04), (95, 0.02), (100, 0.0), (105, 0.01), (110, 0.03)]:
                rows.append(
                    {
                        "as_of": as_of,
                        "expiry": expiry,
                        "strike": strike,
                        "forward": forward,
                        "iv": base + bump + shift,
                    }
                )
        return pd.DataFrame(rows)

    def test_snapshot_state_is_descriptive(self):
        state = describe_surface_snapshot(self._snapshot("2026-09-07T14:30:00Z"))
        self.assertEqual(state["status"], "SURFACE_STATE")
        self.assertEqual(len(state["expiries"]), 2)
        self.assertIn("SKEW_AND_CURVATURE_ARE_DESCRIPTIVE_NOT_PROBABILITIES", state["quality_flags"])

    def test_later_snapshot_is_explicit_future_label(self):
        earlier = self._snapshot("2026-09-07T14:30:00Z", shift=0.0)
        later = self._snapshot("2026-09-07T15:00:00Z", shift=0.01)
        out = surface_change_outcome(earlier, later)
        self.assertEqual(out["status"], "FUTURE_LABEL_ONLY")
        self.assertEqual(out["matched_contract_points"], 10)
        self.assertAlmostEqual(out["median_matched_iv_change"], 0.01, places=10)
        self.assertIn("THIS_OUTPUT_IS_A_LABEL_NOT_A_DECISION_TIME_FEATURE", out["hard_guards"])

    def test_nonchronological_surface_pair_is_rejected(self):
        a = self._snapshot("2026-09-07T15:00:00Z")
        b = self._snapshot("2026-09-07T14:30:00Z")
        with self.assertRaises(ValueError):
            surface_change_outcome(a, b)

    def test_percent_form_iv_is_rejected(self):
        bad = self._snapshot("2026-09-07T14:30:00Z")
        bad.loc[0, "iv"] = 20.0
        with self.assertRaises(ValueError):
            describe_surface_snapshot(bad, SurfaceConfig())


if __name__ == "__main__":
    unittest.main()
