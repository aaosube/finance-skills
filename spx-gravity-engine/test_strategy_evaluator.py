import unittest

import pandas as pd

from strategy_evaluator import ExecutionConfig, apply_costs, deflated_sharpe_ratio, evaluate_strategy


class StrategyEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.idx = pd.date_range("2026-01-01", periods=12, freq="D")
        self.df = pd.DataFrame(
            {
                "position": [0, 1, 1, 0, -1, -1, 0, 1, 1, 0, 0, 1],
                "forward_return": [0.0, 0.01, -0.005, 0.0, -0.01, 0.004, 0.0, 0.006, 0.002, 0.0, 0.0, 0.003],
                "regime": ["CALM"] * 6 + ["VOLATILE"] * 6,
            },
            index=self.idx,
        )
        self.cfg = ExecutionConfig(
            regime_col="regime",
            periods_per_year=252,
            transaction_cost_bps=1.0,
            slippage_bps=1.0,
        )

    def test_costs_are_applied_to_turnover(self):
        out = apply_costs(self.df, self.cfg)
        self.assertAlmostEqual(out.iloc[1]["turnover"], 1.0)
        self.assertAlmostEqual(out.iloc[1]["execution_cost"], 0.0002)
        self.assertAlmostEqual(out.iloc[1]["net_strategy_return"], 0.0098)

    def test_evaluator_reports_regimes_and_entries(self):
        out = evaluate_strategy(self.df, self.cfg, trial_sharpes_annualized=[0.2, 0.4, 0.6])
        self.assertIn("CALM", out["regime_metrics"])
        self.assertIn("VOLATILE", out["regime_metrics"])
        self.assertGreater(out["metrics"]["entries"], 0)
        self.assertEqual(out["deflated_sharpe"]["n_trials"], 3)

    def test_dsr_rejects_empty_trial_set(self):
        with self.assertRaises(ValueError):
            deflated_sharpe_ratio(
                [0.01, -0.005, 0.004, 0.002],
                trial_sharpes_annualized=[],
                periods_per_year=252,
            )

    def test_positions_must_be_normalized(self):
        bad = self.df.copy()
        bad.loc[bad.index[0], "position"] = 2.0
        with self.assertRaises(ValueError):
            apply_costs(bad, self.cfg)


if __name__ == "__main__":
    unittest.main()
