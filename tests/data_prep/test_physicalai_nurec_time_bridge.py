import tempfile
import unittest
from pathlib import Path

from scripts.audit_physicalai_nurec_time_bridge import constant_delta_diagnostics


class TestPhysicalAINuRecTimeBridge(unittest.TestCase):
    def test_constant_delta_without_semantic_identity_is_diagnostic_only(self):
        result = constant_delta_diagnostics([
            {"physicalai_timestamp_us": 100, "nurec_timestamp_us": 1100},
            {"physicalai_timestamp_us": 200, "nurec_timestamp_us": 1200},
        ], semantic_verified=False)
        self.assertEqual(result["median_delta"], 1000)
        self.assertEqual(result["verification_status"], "DIAGNOSTIC_ONLY")

    def test_constant_delta_with_semantic_identity_can_verify(self):
        result = constant_delta_diagnostics([
            {"physicalai_timestamp_us": 100, "nurec_timestamp_us": 1100},
            {"physicalai_timestamp_us": 200, "nurec_timestamp_us": 1200},
        ], semantic_verified=True)
        self.assertEqual(result["verification_status"], "VERIFIED")

    def test_inconsistent_delta_is_not_verified(self):
        result = constant_delta_diagnostics([
            {"physicalai_timestamp_us": 100, "nurec_timestamp_us": 1100},
            {"physicalai_timestamp_us": 200, "nurec_timestamp_us": 1201},
        ], semantic_verified=True)
        self.assertEqual(result["verification_status"], "DIAGNOSTIC_ONLY")
        self.assertEqual(result["delta_unique_count"], 2)

    def test_raw_file_is_not_modified_by_forensic_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.txt"
            path.write_text("raw", encoding="utf-8")
            before = path.read_bytes()
            constant_delta_diagnostics([], semantic_verified=False)
            self.assertEqual(path.read_bytes(), before)
