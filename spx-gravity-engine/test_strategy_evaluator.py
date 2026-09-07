import unittest

import pandas as pd

from strategy_evaluator import (
    ExecutionConfig,
    apply_costs,
    deflated_sharpe_ratio,
    empirical_tail_risk,
    evaluate_strategy,
)


class StrategyEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.idx = pd.date_range("2026-01-01", periods=20, freq="D")
        decision = pd.date_range("2026-01-01 14:30:00Z", periods=20, freq="D")
        self.df = pd.DataFrame(
            {
                "position": [0,1,1,0,-1,-1,0,1,1,0,0,1,1,0,-1,-1,0,1,0,1],
                "forward_return": [0.0,0.01,-0.005,0.0,-0.01,0.004,0.0,0.006,0.002,0.0,0.0,0.003,-0.02,0.0,0.008,-0.004,0.0,0.005,0.0,-0.006],
                "regime": ["CALM"] * 10 + ["VOLATILE"] * 10,
                "spread_bps": [2.0] * 18 + [8.0, 2.0],
                "dollar_volume": [1_000_000.0] * 20,
                "trade_notional": [5_000.0] * 20,
                "decision_time": decision,
                "order_time": decision + pd.to_timedelta(100, unit="ms"),
                "fill_time": decision + pd.to_timedelta(350, unit="ms"),
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

    def test_evaluator_reports_regimes_entries_tail_and_compounded_metrics(self):
        out = evaluate_strategy(self.df, self.cfg, trial_sharpes_annualized=[0.2, 0.4, 0.6])
        self.assertIn("CALM", out["regime_metrics"])
        self.assertIn("VOLATILE", out["regime_metrics"])
        self.assertGreater(out["metrics"]["entries"], 0)
        self.assertEqual(out["deflated_sharpe"]["n_trials"], 3)
        self.assertEqual(out["tail_risk"]["method"], "EMPIRICAL_NO_NORMALITY_ASSUMPTION")
        self.assertIn("0.9500", out["tail_risk"]["levels"])
        self.assertIn("recovery_gain_required_after_max_drawdown", out["metrics"])
        self.assertIn("net_total_return_compounded", out["metrics"])
        self.assertIn("sortino_annualized", out["metrics"])
        self.assertIn("calmar_ratio", out["metrics"])
        self.assertEqual(out["metrics"]["minimum_acceptable_return_per_period"], 0.0)

    def test_empirical_expected_shortfall_is_at_least_var_in_loss_space(self):
        tail = empirical_tail_risk([0.01, -0.01, -0.02, 0.005, -0.05, 0.02], [0.90])
        level = tail["levels"]["0.9000"]
        self.assertGreaterEqual(level["empirical_expected_shortfall"], level["empirical_loss_var"])

    def test_liquidity_capacity_limits_are_explicit(self):
        cfg = ExecutionConfig(
            periods_per_year=252,
            spread_bps_col="spread_bps",
            dollar_volume_col="dollar_volume",
            trade_notional_col="trade_notional",
            max_spread_bps=5.0,
            max_participation_rate=0.01,
        )
        out = evaluate_strategy(self.df, cfg)
        self.assertEqual(out["liquidity_feasibility"]["status"], "FAIL")
        self.assertEqual(out["liquidity_feasibility"]["violations"]["spread"], 1)
        self.assertEqual(out["liquidity_feasibility"]["violations"]["participation"], 0)

    def test_execution_latency_is_measured_only_when_timestamps_are_configured(self):
        cfg = ExecutionConfig(
            periods_per_year=252,
            decision_time_col="decision_time",
            order_time_col="order_time",
            fill_time_col="fill_time",
            max_decision_to_order_ms=150,
            max_order_to_fill_ms=300,
        )
        out = evaluate_strategy(self.df, cfg)
        self.assertEqual(out["execution_latency"]["status"], "PASS")
        self.assertAlmostEqual(out["execution_latency"]["decision_to_order"]["median_ms"], 100.0)
        self.assertAlmostEqual(out["execution_latency"]["order_to_fill"]["median_ms"], 250.0)

    def test_execution_latency_chronology_violation_fails(self):
        bad = self.df.copy()
        entry_idx = bad.index[1]
        bad.loc[entry_idx, "fill_time"] = bad.loc[entry_idx, "order_time"] - pd.to_timedelta(1, unit="s")
        cfg = ExecutionConfig(
            periods_per_year=252,
            decision_time_col="decision_time",
            order_time_col="order_time",
            fill_time_col="fill_time",
        )
        out = evaluate_strategy(bad, cfg)
        self.assertEqual(out["execution_latency"]["status"], "FAIL")
        self.assertGreater(out["execution_latency"]["violations"]["timestamp_chronology"], 0)

    def test_stability_is_diagnostic_not_automatic_adaptation(self):
        cfg = ExecutionConfig(periods_per_year=252, stability_window=5)
        out = evaluate_strategy(self.df, cfg)
        self.assertEqual(out["stability_diagnostic"]["status"], "DIAGNOSTIC_ONLY")
        self.assertIn("delta", out["stability_diagnostic"])
        self.assertIn("No automatic retraining", out["stability_diagnostic"]["note"])

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

    def test_liquidity_threshold_requires_data_column(self):
        with self.assertRaises(ValueError):
            ExecutionConfig(max_spread_bps=5.0).validate()

    def test_latency_threshold_requires_all_timestamp_columns(self):
        with self.assertRaises(ValueError):
            ExecutionConfig(max_decision_to_order_ms=10.0).validate()


if __name__ == "__main__":
    unittest.main()
