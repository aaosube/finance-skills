import unittest
import numpy as np
import pandas as pd

from ai_research_loop import (
    ResearchConfig,
    HypothesisSpec,
    chronological_split,
    validate_hypothesis,
    build_design_matrix,
)


class AIResearchLayerTests(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2026-01-01", periods=60, freq="h")
        rng = np.random.default_rng(7)
        self.df = pd.DataFrame(
            {
                "asia_range_z": rng.normal(size=60),
                "gex_flip_distance": rng.normal(size=60),
                "ngc_lower": rng.normal(size=60),
                "target": np.where(np.arange(60) % 3 == 0, "LOWER_FIRST", "UPPER_FIRST"),
            },
            index=idx,
        )
        self.cfg = ResearchConfig(
            target_col="target",
            feature_pool=("asia_range_z", "gex_flip_distance", "ngc_lower"),
            iterations=2,
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
                    "rationale": "leakage",
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


if __name__ == "__main__":
    unittest.main()
