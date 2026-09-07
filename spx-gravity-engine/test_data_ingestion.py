import unittest
import pandas as pd

from data_ingestion import DataQualityError, normalize_market_frame, to_engine_rows


class DataIngestionTests(unittest.TestCase):
    def test_normalizes_common_option_fields_and_preserves_rows(self):
        raw = pd.DataFrame({
            "Date Time": ["2026-09-04 09:45", "2026-09-04 09:46"],
            "Ticker": ["SPX", "SPX"],
            "Strike Price": [6500, 6510],
            "Put/Call": ["C", "P"],
            "Open Interest": [100, 200],
            "Gamma": [0.001, 0.002],
            "Unused Vendor Field": [1, 2],
        })
        out, report = normalize_market_frame(
            raw,
            source="test_export",
            column_map={"Date Time": "timestamp", "Put/Call": "option_type"},
            required=("strike", "option_type", "open_interest", "gamma"),
            duplicate_key=("symbol", "strike", "option_type"),
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(list(out["option_type"]), ["call", "put"])
        self.assertEqual(report.output_rows, 2)
        self.assertIn("Unused Vendor Field", report.unmapped_columns)
        self.assertEqual(str(out["timestamp"].dt.tz), "UTC")

    def test_engine_adapter_uses_canonical_fields(self):
        raw = pd.DataFrame({
            "timestamp": ["2026-09-04 09:45"],
            "strike": [6500],
            "option_type": ["call"],
            "open_interest": [100],
            "gamma": [0.001],
            "iv": [0.20],
            "dte": [1],
        })
        out, _ = normalize_market_frame(
            raw,
            source="x",
            required=("strike", "option_type", "open_interest", "gamma"),
        )
        rows = to_engine_rows(out)
        self.assertEqual(rows[0]["type"], "call")
        self.assertEqual(rows[0]["openInterest"], 100)
        self.assertEqual(rows[0]["daysToExpiry"], 1)

    def test_ambiguous_mapping_fails_closed(self):
        raw = pd.DataFrame({
            "timestamp": ["2026-09-04 09:45"],
            "strike": [6500],
            "strike price": [6501],
        })
        with self.assertRaises(DataQualityError):
            normalize_market_frame(raw, source="x")

    def test_invalid_timestamp_fails_closed(self):
        raw = pd.DataFrame({"timestamp": ["not-a-time"], "strike": [6500]})
        with self.assertRaises(DataQualityError):
            normalize_market_frame(raw, source="x")


if __name__ == "__main__":
    unittest.main()
