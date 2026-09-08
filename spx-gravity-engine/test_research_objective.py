import unittest

from research_objective import (
    ResearchLossInputs,
    ResearchLossWeights,
    TradeCostInputs,
    research_loss,
    trade_cost,
)


class ResearchObjectiveTests(unittest.TestCase):
    def test_loss_is_auditable_weighted_sum(self):
        weights = ResearchLossWeights(1, 2, 3, 4, 5, 6, 7, 8)
        inputs = ResearchLossInputs(1.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
        out = research_loss(inputs, weights)
        expected = -1.5 + 0.2 + 0.6 + 1.2 + 2.0 + 3.0 + 4.2 + 5.6
        self.assertAlmostEqual(out["loss"], expected)
        self.assertEqual(sum(out["contributions"].values()), out["loss"])

    def test_penalties_must_be_normalized(self):
        weights = ResearchLossWeights(1, 1, 1, 1, 1, 1, 1, 1)
        bad = ResearchLossInputs(1, 1.01, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(ValueError):
            research_loss(bad, weights)

    def test_zero_weight_vector_is_rejected(self):
        weights = ResearchLossWeights(0, 0, 0, 0, 0, 0, 0, 0)
        inputs = ResearchLossInputs(1, 0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(ValueError):
            research_loss(inputs, weights)

    def test_trade_cost_includes_every_component_and_stress(self):
        out = trade_cost(TradeCostInputs(
            notional=100_000,
            spread_bps_round_trip=4,
            slippage_bps_round_trip=2,
            market_impact_bps_round_trip=1,
            borrow_bps_holding_period=0.5,
            other_bps_round_trip=0.5,
            fixed_round_trip_cash=2,
            option_fees_cash=3,
            stress_multiplier=1.5,
        ))
        self.assertEqual(out["variable_cost_bps_round_trip"], 8.0)
        self.assertEqual(out["base_cost_cash"], 85.0)
        self.assertEqual(out["stressed_cost_cash"], 127.5)

    def test_negative_or_subunit_stress_is_rejected(self):
        with self.assertRaises(ValueError):
            trade_cost(TradeCostInputs(notional=100, stress_multiplier=0.9))


if __name__ == "__main__":
    unittest.main()
