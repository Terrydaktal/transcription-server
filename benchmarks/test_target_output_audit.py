"""The reader view must not disguise unfinished ASR as a complete transcript."""

import copy
import unittest

from audit_target_outputs import reader_view, router_audit
from test_target_generation_retry import row


class AuditTests(unittest.TestCase):
    def test_reader_marks_failure_and_preserves_success_and_source_rows(self):
        rows = [row("valid"), row("failed", True)]
        for item in rows:
            item["vad_rejected"] = True
        rows[1]["text"] = "UNRELIABLE DECODER OUTPUT"
        before = copy.deepcopy(rows)
        plain, timestamped, warnings = reader_view(rows, 16000)
        self.assertEqual(warnings, 1)
        self.assertIn("original", plain)
        self.assertIn("ASR did not finish", plain)
        self.assertNotIn("UNRELIABLE DECODER OUTPUT", plain)
        self.assertIn("[LOW-VAD / REVIEW]", timestamped)
        self.assertEqual(rows, before)

    def test_router_early_rejection_is_explicit_and_paired(self):
        base = row("same")
        routed = {**row("same", True), "route": "auto", "lora_active": True}
        result = router_audit([base], [routed], 10)
        self.assertEqual(result["completed_rows"], 1)
        self.assertEqual(result["router_generation_warning_rows"], 1)
        self.assertEqual(result["paired_base_generation_warning_rows"], 0)
        self.assertFalse(result["is_full_target_comparison"])
        self.assertIsNone(result["target_wer"])

    def test_router_unpaired_unknown_duplicate_or_unadapted_rows_fail(self):
        base = row("same")
        routed = {**row("same"), "route": "auto", "lora_active": True}
        changed = {**routed, "audio": "wrong.wav"}
        cases = [[changed], [routed, routed], [{**routed, "id": "wrong"}],
                 [{**routed, "lora_active": False}]]
        for candidates in cases:
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                router_audit([base], candidates, 10)


if __name__ == "__main__":
    unittest.main()
