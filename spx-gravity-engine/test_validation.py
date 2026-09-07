import unittest
from validation import empirical_prior_metrics, compare_to_prior, purged_walk_forward_splits

class ValidationTests(unittest.TestCase):
    def test_purged_walk_forward_has_gap_and_no_overlap(self):
        windows = purged_walk_forward_splits(100, min_train=40, test_size=10, gap=3, step=10)
        self.assertTrue(windows)
        for w in windows:
            self.assertEqual(w.test_start - w.train_end, 3)
            self.assertLessEqual(w.train_end, w.test_start)

    def test_empirical_prior_metrics(self):
        m = empirical_prior_metrics(["U", "U", "L", "U"], ["U", "L"])
        self.assertAlmostEqual(m["prior"]["U"], 0.75)
        c = compare_to_prior(model_log_loss=0.1, model_brier=0.1, prior_metrics=m)
        self.assertTrue(c["beats_baseline_both"])

    def test_unseen_test_class_rejected(self):
        with self.assertRaises(ValueError): empirical_prior_metrics(["U", "L"], ["N"])

if __name__ == "__main__": unittest.main()
