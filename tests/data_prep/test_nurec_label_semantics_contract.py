import unittest

from scripts.audit_nurec_label_semantics_contract import (
    CONTRACT_C,
    FRAME_GRID_STATUS,
    classify_missing_timestamp,
)


class NurecLabelSemanticsContractTests(unittest.TestCase):
    def test_sparse_rows_do_not_prove_empty(self):
        result = classify_missing_timestamp({"query_timestamp_us": "100"}, {}, {})
        self.assertEqual(result["contract_classification"], "UNRESOLVED_NO_AUTHORITATIVE_FRAME_GRID")
        self.assertTrue(result["unresolved"])
        self.assertFalse(result["confirmed_empty"])

    def test_exact_object_row_is_reconciliation_conflict(self):
        result = classify_missing_timestamp({"query_timestamp_us": "100"}, {100: 3}, {})
        self.assertTrue(result["object_empty_evidence_conflict"])
        self.assertFalse(result["unresolved"])

    def test_outside_object_range_is_not_outside_authoritative_grid(self):
        result = classify_missing_timestamp({"query_timestamp_us": "100"}, {200: 1}, {})
        self.assertFalse(result["outside_authoritative_frame_grid"])
        self.assertEqual(result["contract_classification"], "UNRESOLVED_NO_AUTHORITATIVE_FRAME_GRID")

    def test_contract_constants_are_fail_closed(self):
        self.assertEqual(CONTRACT_C, "CONTRACT_C_SPARSE_OBJECT_ROWS_NO_EMPTY_GUARANTEE")
        self.assertEqual(FRAME_GRID_STATUS, "NOT_FOUND_IN_RELEASED_LABEL_ARTIFACTS")


if __name__ == "__main__":
    unittest.main()
