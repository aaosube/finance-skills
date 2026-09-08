import unittest
import numpy as np
import pandas as pd
from strategy_evaluator import _max_drawdown, apply_costs, evaluate_strategy, ExecutionConfig


class AuditRegressions(unittest.TestCase):
    def test_initial_loss_counts_from_starting_capital(self):
        self.assertAlmostEqual(_max_drawdown(pd.Series([-.1])), -.1)
        self.assertAlmostEqual(_max_drawdown(pd.Series([-.1, -.1])), -.19)
        self.assertAlmostEqual(_max_drawdown(pd.Series([.1, -.1])), -.1)

    def test_execution_gaps_cannot_disappear(self):
        for invalid in [None, np.nan, np.inf]:
            data = pd.DataFrame({'position': [1, 1, 0], 'forward_return': [.01, invalid, .02]}, index=pd.date_range('2026-01-01', periods=3))
            with self.assertRaisesRegex(ValueError, 'gaps cannot be dropped'):
                apply_costs(data, ExecutionConfig())

    def test_nan_costs_cannot_produce_performance(self):
        for config in [ExecutionConfig(transaction_cost_bps=np.nan), ExecutionConfig(periods_per_year=np.inf)]:
            with self.assertRaises(ValueError):
                config.validate()

    def test_missing_liquidity_cannot_pass_limits(self):
        data = pd.DataFrame({'position': [1, 1, 0], 'forward_return': [.01, -.01, .02], 'spread': [1, None, 1]}, index=pd.date_range('2026-01-01', periods=3))
        report = evaluate_strategy(data, ExecutionConfig(spread_bps_col='spread', max_spread_bps=2))
        self.assertEqual(report['liquidity_feasibility']['status'], 'INSUFFICIENT_DATA')


if __name__ == '__main__':
    unittest.main()
