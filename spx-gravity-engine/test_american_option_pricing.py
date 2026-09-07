import unittest
from math import erf, exp, log, sqrt

from american_option_pricing import CRRInputs, crr_price


def _norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _bsm_call(spot, strike, t, r, q, sigma):
    d1 = (log(spot / strike) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)
    return spot * exp(-q * t) * _norm_cdf(d1) - strike * exp(-r * t) * _norm_cdf(d2)


class CRRTests(unittest.TestCase):
    def test_european_tree_converges_near_black_scholes(self):
        inp = CRRInputs(
            spot=100.0,
            strike=100.0,
            time_years=0.5,
            rate=0.04,
            dividend_yield=0.01,
            volatility=0.20,
            option_type="call",
            exercise_style="european",
            steps=600,
        )
        out = crr_price(inp)
        bsm = _bsm_call(100.0, 100.0, 0.5, 0.04, 0.01, 0.20)
        self.assertAlmostEqual(out["price"], bsm, delta=0.03)
        self.assertTrue(-1.0 <= out["delta"] <= 1.0)
        self.assertGreaterEqual(out["gamma"], 0.0)

    def test_american_put_not_below_european_put(self):
        common = dict(
            spot=90.0,
            strike=100.0,
            time_years=1.0,
            rate=0.05,
            dividend_yield=0.0,
            volatility=0.25,
            option_type="put",
            steps=400,
        )
        european = crr_price(CRRInputs(**common, exercise_style="european"))
        american = crr_price(CRRInputs(**common, exercise_style="american"))
        self.assertGreaterEqual(american["price"] + 1e-10, european["price"])
        self.assertGreater(american["early_exercise_nodes"], 0)

    def test_no_dividend_american_call_is_near_european(self):
        common = dict(
            spot=100.0,
            strike=95.0,
            time_years=0.4,
            rate=0.03,
            dividend_yield=0.0,
            volatility=0.22,
            option_type="call",
            steps=500,
        )
        european = crr_price(CRRInputs(**common, exercise_style="european"))
        american = crr_price(CRRInputs(**common, exercise_style="american"))
        self.assertAlmostEqual(american["price"], european["price"], delta=1e-8)

    def test_invalid_style_fails_closed(self):
        with self.assertRaises(ValueError):
            crr_price(CRRInputs(
                spot=100,
                strike=100,
                time_years=0.1,
                rate=0.02,
                volatility=0.2,
                option_type="call",
                exercise_style="bermudan",
            ))


if __name__ == "__main__":
    unittest.main()
