import unittest
from math import exp

import pandas as pd

from option_integrity import ParityInputs, audit_parity_frame, audit_put_call_parity


class PutCallParityIntegrityTests(unittest.TestCase):
    def test_forward_parity_exact_mid_passes_without_arbitrary_threshold(self):
        strike = 100.0
        t = 30 / 365
        r = 0.04
        forward = 102.0
        theo = exp(-r * t) * (forward - strike)
        result = audit_put_call_parity(
            ParityInputs(
                strike=strike,
                time_years=t,
                rate=r,
                forward=forward,
                call_mid=5.0 + theo,
                put_mid=5.0,
                call_bid=5.0 + theo - 0.10,
                call_ask=5.0 + theo + 0.10,
                put_bid=4.90,
                put_ask=5.10,
            )
        )
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["inside_executable_interval"])
        self.assertAlmostEqual(result["mid_residual"], 0.0, places=10)

    def test_executable_interval_violation_fails(self):
        result = audit_put_call_parity(
            ParityInputs(
                strike=100.0,
                time_years=0.25,
                rate=0.0,
                forward=120.0,
                call_mid=5.0,
                put_mid=4.0,
                call_bid=4.9,
                call_ask=5.1,
                put_bid=3.9,
                put_ask=4.1,
            )
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("PUT_CALL_PARITY_EXECUTABLE_INTERVAL_VIOLATION", result["quality_flags"])

    def test_spot_dividend_form_is_supported_and_flagged_as_model_input(self):
        result = audit_put_call_parity(
            ParityInputs(
                strike=100.0,
                time_years=0.5,
                rate=0.03,
                spot=101.0,
                dividend_yield=0.01,
                call_mid=8.0,
                put_mid=6.0,
            )
        )
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
        self.assertIn("DIVIDEND_YIELD_IS_MODEL_INPUT", result["quality_flags"])

    def test_american_style_uses_bounds_not_european_equality(self):
        result = audit_put_call_parity(
            ParityInputs(
                strike=100.0,
                time_years=0.5,
                rate=0.04,
                spot=105.0,
                dividend_yield=0.01,
                call_mid=10.0,
                put_mid=5.0,
                call_bid=9.9,
                call_ask=10.1,
                put_bid=4.9,
                put_ask=5.1,
                exercise_style="american",
            )
        )
        self.assertEqual(result["method"], "AMERICAN_PARITY_BOUNDS")
        self.assertIn("AMERICAN_STYLE_USES_BOUNDS_NOT_EUROPEAN_EQUALITY", result["quality_flags"])
        self.assertNotIn("mid_residual", result)
        self.assertEqual(result["status"], "PASS")

    def test_american_style_rejects_european_residual_threshold(self):
        with self.assertRaises(ValueError):
            audit_put_call_parity(
                ParityInputs(
                    strike=100.0,
                    time_years=0.5,
                    rate=0.04,
                    spot=105.0,
                    call_mid=10.0,
                    put_mid=5.0,
                    exercise_style="american",
                ),
                max_abs_mid_residual=0.05,
            )

    def test_frame_audit_does_not_pair_quotes_itself(self):
        df = pd.DataFrame(
            {
                "strike": [100.0],
                "time_years": [0.1],
                "rate": [0.0],
                "forward": [100.0],
                "call_mid": [3.0],
                "put_mid": [3.0],
            },
            index=["paired-row-1"],
        )
        out = audit_parity_frame(df, forward_col="forward", spot_col=None)
        self.assertEqual(out.loc["paired-row-1", "status"], "DIAGNOSTIC_ONLY")
        self.assertAlmostEqual(out.loc["paired-row-1", "mid_residual"], 0.0)


if __name__ == "__main__":
    unittest.main()
