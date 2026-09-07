import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ai_research_loop import ResearchConfig
from research_orchestrator import run_validated_ai_research


class ResearchOrchestratorTests(unittest.TestCase):
    def test_test_baseline_is_attached_after_frozen_research(self):
        idx = pd.date_range("2026-01-01", periods=60, freq="h")
        df = pd.DataFrame(
            {
                "x": np.linspace(-1, 1, 60),
                "target": np.where(np.arange(60) % 3 == 0, "L", "U"),
            },
            index=idx,
        )
        cfg = ResearchConfig(target_col="target", feature_pool=("x",), iterations=1)
        fake = {
            "winner": {"test_log_loss": 0.4, "test_brier": 0.3},
            "hard_guards": ["TEST_RESULTS_NEVER_FED_TO_LLM"],
        }
        with patch("research_orchestrator.run_ai_research", return_value=fake):
            out = run_validated_ai_research(df, cfg, client=object())
        self.assertEqual(out["test_baseline"]["benchmark"], "TRAIN_EMPIRICAL_PRIOR")
        self.assertIn("MODEL_MUST_BE_REPORTED_AGAINST_NAIVE_PRIOR", out["hard_guards"])
        self.assertTrue(out["test_baseline_comparison"]["beats_baseline_both"])


if __name__ == "__main__":
    unittest.main()
