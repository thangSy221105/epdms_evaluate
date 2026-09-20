import unittest

from scripts.build_epdms_xodr_ddc_vector import build_fail_closed_row, validate_identity


class XodrDdcFailClosedTests(unittest.TestCase):
    def test_ddc_is_null_when_xodr_contract_is_unresolved(self):
        row = build_fail_closed_row({"record_key": "k", "clip_id": "c", "missing_components": ["ddc"]})
        self.assertIsNone(row["ddc"])
        self.assertIsNone(row["ddc_proxy"])
        self.assertFalse(row["ddc_proxy_implemented"])
        self.assertFalse(row["ddc_proxy_enabled"])
        self.assertIn("ddc", row["missing_components"])

    def test_identity_contract_preserves_all_rows(self):
        rows = [{"record_key": str(i)} for i in range(4800)]
        result = validate_identity(rows, list(reversed(rows)))
        self.assertTrue(result["same_record_key_set"])
        self.assertTrue(result["records_unique"])
        self.assertTrue(result["baseline_unique"])

    def test_duplicate_identity_is_rejected(self):
        result = validate_identity([{"record_key": "same"}, {"record_key": "same"}], [{"record_key": "same"}])
        self.assertFalse(result["records_unique"])
        self.assertTrue(result["baseline_unique"])


if __name__ == "__main__":
    unittest.main()
