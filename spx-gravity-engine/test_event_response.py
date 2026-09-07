import unittest

import numpy as np
import pandas as pd

from event_response import (
    EventResponseConfig,
    causal_event_absorption_state,
    completed_event_decay_diagnostic,
)


class EventResponseTests(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2026-09-04 12:00:00Z", periods=40, freq="5min")
        activity = np.ones(40, dtype=float)
        # Event at index 12. Post-event shock peaks immediately then decays.
        shock = [6.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.7, 1.5, 1.3, 1.2, 1.1, 1.05]
        activity[12:12 + len(shock)] = shock
        self.df = pd.DataFrame({"activity": activity}, index=idx)
        self.event_time = idx[12]
        self.cfg = EventResponseConfig(activity_col="activity", pre_bars=10, post_bars=12)

    def test_causal_state_uses_only_rows_up_to_as_of(self):
        as_of = self.df.index[16]
        out = causal_event_absorption_state(
            self.df,
            event_time=self.event_time,
            as_of=as_of,
            config=self.cfg,
        )
        self.assertEqual(out["status"], "CAUSAL_EVENT_ABSORPTION_DIAGNOSTIC")
        self.assertEqual(out["post_observations"], 5)
        self.assertGreater(out["peak_activity_ratio_so_far"], out["current_activity_ratio"])
        self.assertGreater(out["absorbed_fraction_of_peak_excess_so_far"], 0.0)
        self.assertIsNone(out["probability"])

    def test_completed_decay_estimates_positive_half_life(self):
        out = completed_event_decay_diagnostic(
            self.df,
            event_time=self.event_time,
            config=self.cfg,
        )
        self.assertEqual(out["status"], "EXPONENTIAL_EVENT_DECAY_DIAGNOSTIC")
        self.assertGreater(out["beta_per_minute"], 0.0)
        self.assertGreater(out["half_life_minutes"], 0.0)
        self.assertLessEqual(out["fit_r2"], 1.0)
        self.assertIn("available_at", out)
        self.assertIsNone(out["probability"])

    def test_pre_event_as_of_is_rejected(self):
        with self.assertRaises(ValueError):
            causal_event_absorption_state(
                self.df,
                event_time=self.event_time,
                as_of=self.event_time - pd.Timedelta(minutes=5),
                config=self.cfg,
            )

    def test_naive_timezone_is_rejected(self):
        naive = self.df.copy()
        naive.index = naive.index.tz_localize(None)
        with self.assertRaises(ValueError):
            causal_event_absorption_state(
                naive,
                event_time=self.event_time,
                as_of=self.event_time,
                config=self.cfg,
            )

    def test_nondecaying_series_does_not_invent_half_life(self):
        df = self.df.copy()
        df.loc[df.index >= self.event_time, "activity"] = np.linspace(2.0, 5.0, len(df.loc[df.index >= self.event_time]))
        out = completed_event_decay_diagnostic(df, event_time=self.event_time, config=self.cfg)
        self.assertIn(out["status"], {"NO_DECAYING_EXPONENTIAL_FIT", "NO_IDENTIFIABLE_POSITIVE_DECAY"})
        self.assertNotIn("half_life_minutes", out)


if __name__ == "__main__":
    unittest.main()
