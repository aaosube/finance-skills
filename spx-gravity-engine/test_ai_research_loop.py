import unittest

import numpy as np
import pandas as pd

from ai_research_loop import (
    HypothesisSpec,
    ResearchConfig,
    build_design_matrix,
    chronological_split,
    inner_walk_forward_score,
    validate_hypothesis,
)


class AIResearchLayerTests(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2026-01-01", periods=90, freq="h")
        rng = np.random.default_rng(7)
        x1 = rng.normal(size=90)
        x2 = rng.normal(size=90)
        self.df = pd.DataFrame(
            {
                "asia_range_z": x1,
                "gex_flip_distance": x2,
                "ngc_lower": rng.normal(size=90),
                "target": np.where(x1 + 0.2 * x2 > 0, "UPPER_FIRST", "LOWER_FIRST"),
            },
            index=idx,
        )
        self.cfg = ResearchConfig(
            target_col="target",
            feature_pool=("asia_range_z", "gex_flip_distance", "ngc_lower"),
            iterations=2,
            inner_min_train=20,
            inner_test_size=5,
            inner_gap=1,
        )

    def test_split_is_chronological(self):
        s = chronological_split(self.df, self.cfg)
        self.assertLess(s.train.index.max(), s.calibration.index.min())
        self.assertLess(s.calibration.index.max(), s.test.index.min())

    def test_unknown_feature_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_hypothesis(
                {
                    "name": "bad",
                    "features": ["future_close"],
                    "interactions": [],
                    "assumptions": ["future data would be causal"],
                    "rationale": "leakage",
                },
                self.cfg,
            )

    def test_quant_hypothesis_metadata_is_retained(self):
        spec = validate_hypothesis(
            {
                "name": "session-mechanism",
                "features": ["asia_range_z", "gex_flip_distance"],
                "interactions": [["asia_range_z", "gex_flip_distance"]],
                "mechanism": "Overnight displacement interacts with dealer geometry.",
                "assumptions": ["the dealer-state snapshot is causal at decision time"],
                "expected_regimes": ["volatile"],
                "failure_modes": ["macro discontinuity"],
                "rationale": "testable mechanism",
            },
            self.cfg,
        )
        self.assertEqual(spec.assumptions, ("the dealer-state snapshot is causal at decision time",))
        self.assertEqual(spec.expected_regimes, ("volatile",))
        self.assertEqual(spec.failure_modes, ("macro discontinuity",))

    def test_missing_assumption_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_hypothesis(
                {
                    "name": "assumption-free",
                    "features": ["asia_range_z"],
                    "interactions": [],
                    "mechanism": "claim",
                    "failure_modes": ["regime change"],
                    "rationale": "claim",
                },
                self.cfg,
            )

    def test_interactions_are_engine_owned(self):
        spec = HypothesisSpec(
            name="x",
            features=("asia_range_z", "ngc_lower"),
            interactions=(("asia_range_z", "ngc_lower"),),
            rationale="test",
        )
        x = build_design_matrix(self.df, spec)
        self.assertIn("INT__asia_range_z__X__ngc_lower", x.columns)
        np.testing.assert_allclose(
            x["INT__asia_range_z__X__ngc_lower"].values,
            (self.df["asia_range_z"] * self.df["ngc_lower"]).values,
        )

    def test_train_feedback_uses_inner_walk_forward(self):
        split = chronological_split(self.df, self.cfg)
        spec = HypothesisSpec(
            name="wf",
            features=("asia_range_z", "gex_flip_distance"),
            interactions=(),
            rationale="test",
        )
        loss, brier, folds = inner_walk_forward_score(spec, split.train, self.cfg)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.isfinite(brier))
        self.assertGreaterEqual(folds, 1)


if __name__ == "__main__":
    unittest.main()
