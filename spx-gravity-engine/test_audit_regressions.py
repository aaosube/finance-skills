import unittest
import numpy as np
import pandas as pd
from strategy_evaluator import _max_drawdown, apply_costs, evaluate_strategy, ExecutionConfig


class AuditRegressions(unittest.TestCase):
    def test_target_cannot_leak_into_feature_pool(self):
        from ai_research_loop import ResearchConfig
        with self.assertRaisesRegex(ValueError, 'target label'):
            ResearchConfig(target_col='y', feature_pool=('x', 'y')).validate()

    def test_outer_split_purges_future_labels(self):
        from ai_research_loop import ResearchConfig, chronological_split
        idx = pd.date_range('2026-01-01', periods=60, tz='UTC')
        data = pd.DataFrame({'x': range(60), 'y': [0,1]*30, 'end': idx + pd.Timedelta(days=3), 'session': idx.date.astype(str)}, index=idx)
        split = chronological_split(data, ResearchConfig(target_col='y', feature_pool=('x',), label_end_col='end', session_col='session'))
        self.assertTrue((split.train['end'] < split.calibration.index[0]).all())
        self.assertTrue((split.calibration['end'] < split.test.index[0]).all())

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
