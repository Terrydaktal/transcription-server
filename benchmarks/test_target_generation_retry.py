"""Offline regression coverage for target generation recovery, not ASR accuracy."""

import copy
import unittest

from retry_target_generation import INPUT_FIELDS, merge_retries, retry_manifest


def row(name, capped=False):
    result = {key: None for key in INPUT_FIELDS}
    result.update(id=name, audio=name + ".wav", start=0.0, end=1.0,
                  start_sample=0, end_sample=16000, coverage_origin="uncovered",
                  text="original", sampling_seed=123, beam=2,
                  generation=[{"new_tokens": 512 if capped else 12,
                               "ended_with_eos": not capped, "token_cap_hit": capped}])
    return result


class RecoveryTests(unittest.TestCase):
    def test_gate_uses_generation_not_text_or_reference(self):
        rows = [row("ordinary"), row("capped", True)]
        rows[0].update(text="suspicious words", reference="irrelevant reference")
        manifest = retry_manifest(rows)
        self.assertEqual([item["id"] for item in manifest], ["capped"])
        self.assertNotIn("text", manifest[0])
        self.assertNotIn("reference", manifest[0])
        self.assertNotIn("generation", manifest[0])

    def test_complete_retry_preserves_original_evidence_and_other_rows(self):
        rows = [row("ordinary"), row("capped", True)]
        candidate = row("capped")
        candidate.update(text="retry text", beam=1)
        saved = copy.deepcopy(rows)
        result = merge_retries(rows, [candidate])
        self.assertEqual(result[0], rows[0])
        self.assertEqual(result[1]["text"], "retry text")
        self.assertEqual(result[1]["generation_recovery"]["original_text"], "original")
        self.assertTrue(result[1]["generation_recovery"]["accepted"])
        self.assertEqual(rows, saved)

    def test_still_capped_retry_does_not_hide_original_failure(self):
        original = row("capped", True)
        candidate = row("capped", True)
        candidate["text"] = "different incomplete text"
        result = merge_retries([original], [candidate])[0]
        self.assertEqual(result["text"], "original")
        self.assertEqual(result["generation"], original["generation"])
        self.assertFalse(result["generation_recovery"]["accepted"])

    def test_missing_eos_audit_is_not_accepted(self):
        candidate = row("capped")
        del candidate["generation"][0]["ended_with_eos"]
        self.assertFalse(merge_retries([row("capped", True)], [candidate])[0]
                         ["generation_recovery"]["accepted"])

    def test_missing_extra_duplicate_changed_metadata_and_seed_are_rejected(self):
        original, candidate = row("capped", True), row("capped")
        cases = [[], [candidate, candidate], [row("ordinary")]]
        for field, value in (("audio", "wrong.wav"), ("end_sample", 15999),
                             ("sampling_seed", 124)):
            changed = copy.deepcopy(candidate)
            changed[field] = value
            cases.append([changed])
        for retries in cases:
            with self.subTest(retries=retries), self.assertRaises(ValueError):
                merge_retries([original], retries)

    def test_empty_gate_and_input_are_valid(self):
        self.assertEqual(retry_manifest([]), [])
        self.assertEqual(merge_retries([], []), [])
        ordinary = row("ordinary")
        self.assertEqual(merge_retries([ordinary], []), [ordinary])


if __name__ == "__main__":
    unittest.main()
