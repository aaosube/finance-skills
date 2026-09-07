import unittest

from asset_profiles import AssetClass, SessionPolicy, get_profile, required_modules


class AssetProfileTests(unittest.TestCase):
    def test_spx_is_index_cash_profile(self):
        p = get_profile("SPX")
        self.assertEqual(p.asset_class, AssetClass.INDEX)
        self.assertEqual(p.session_policy, SessionPolicy.CASH_RTH)
        self.assertFalse(p.supports_extended_hours_target_tape)
        self.assertIn("derivatives_state", required_modules(p))
        self.assertNotIn("borrow_state", required_modules(p))

    def test_spy_is_etf_extended_hours_profile(self):
        p = get_profile("SPY")
        self.assertEqual(p.asset_class, AssetClass.ETF)
        self.assertTrue(p.supports_extended_hours_target_tape)
        self.assertTrue(p.supports_finra_short_volume)
        self.assertEqual(p.reference_symbol, "SPX")

    def test_generic_stock_enables_stock_specific_modules(self):
        p = get_profile("TSLA", asset_class="STOCK", reference_symbol="SPX")
        mods = required_modules(p)
        self.assertEqual(p.asset_class, AssetClass.STOCK)
        self.assertIn("off_exchange_liquidity", mods)
        self.assertIn("lagged_short_volume", mods)
        self.assertIn("borrow_state", mods)
        self.assertIn("ftd_state", mods)
        self.assertIn("corporate_event_regime", mods)

    def test_unknown_index_requires_explicit_registration(self):
        with self.assertRaises(ValueError):
            get_profile("UNKNOWNINDEX", asset_class="INDEX")


if __name__ == "__main__":
    unittest.main()
