import unittest
import pandas as pd
from shadow_research import ShadowPolicy, ShadowPolicyError, run_shadow_policy

class ShadowResearchTests(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2026-09-04 09:30", periods=6, freq="5min")
        self.frame = pd.DataFrame({"spot": [100, 101, 102, 101, 99, 98], "score": [0.2, 0.8, 0.9, 0.1, -0.8, -0.9], "exit_flag": [0, 0, 1, 0, 0, 1]}, index=idx)

    def test_structured_policy_replay(self):
        policy = ShadowPolicy(name="test", long_when={"feature": "score", "op": "ge", "value": 0.8}, short_when={"feature": "score", "op": "le", "value": -0.8}, exit_when={"feature": "exit_flag", "op": "eq", "value": 1})
        out = run_shadow_policy(self.frame, policy, cost_bps=1.0)
        self.assertEqual(out["summary"]["trade_count"], 2)
        self.assertIn("NO_BROKER_ORDERS", out["hard_guards"])

    def test_missing_feature_rejected(self):
        policy = ShadowPolicy(name="bad", long_when={"feature": "future_close", "op": "gt", "value": 1})
        with self.assertRaises(ShadowPolicyError): run_shadow_policy(self.frame, policy)

    def test_conflicting_signals_rejected(self):
        policy = ShadowPolicy(name="conflict", long_when={"feature": "score", "op": "ge", "value": 0}, short_when={"feature": "score", "op": "ge", "value": 0})
        with self.assertRaises(ShadowPolicyError): run_shadow_policy(self.frame, policy)

if __name__ == "__main__": unittest.main()
