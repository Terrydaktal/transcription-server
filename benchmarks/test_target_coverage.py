"""CPU-only coverage, provenance, resume and completion-gate regressions."""

import hashlib
import importlib.util
import io
import json
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


HELPER_PATH = Path(__file__).with_name("complete_target_coverage.py")
SPEC = importlib.util.spec_from_file_location("target_coverage", HELPER_PATH)
coverage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coverage)
CONFIG = coverage.selected_configuration({"stage": "mega", "route": "base", "beam": 2, "kind": "raw"})


def interval_row(name, start, end, rate=16000, **metadata):
    return {"id": name, "start": start / rate, "end": end / rate,
            "duration_s": (end - start) / rate, **metadata}


def result_row(row, text=" raw text ", **metadata):
    return {**row, **CONFIG, "text": text, "sampling_seed": coverage.clip_seed(row["id"]),
            "degraded_probability": 0.1, "lora_active": False,
            "generation": [{"new_tokens": 8, "ended_with_eos": True, "token_cap_hit": False}],
            "infer_s": 0.25, "peak_allocated_gib": 1.0, **metadata}


class ComplementTests(unittest.TestCase):
    def test_empty_and_entirely_retained_waveforms(self):
        self.assertEqual(coverage.complement_intervals([], 0), [])
        self.assertEqual(coverage.complement_intervals([], 7), [(0, 7)])
        self.assertEqual(coverage.complement_intervals([(0, 7)], 7), [])
        coverage.validate_full_coverage([], 0)

    def test_touching_unsorted_intervals_and_nonzero_boundaries(self):
        self.assertEqual(coverage.complement_intervals([(9, 11), (4, 9), (2, 4)], 13),
                         [(0, 2), (11, 13)])
        coverage.validate_full_coverage([(7, 13), (0, 7)], 13)

    def test_invalid_intervals_overlap_duplicate_reverse_outside_and_boolean(self):
        for intervals in ([(0, 4), (3, 5)], [(0, 4), (0, 4)], [(4, 4)], [(5, 4)],
                          [(-1, 2)], [(0, 11)], [(False, 2)], [(0.5, 2)]):
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                coverage.complement_intervals(intervals, 10)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            coverage.validate_full_coverage([(1, 10)], 10)

    def test_existing_gaps_reused_and_sub_two_second_tails_recovered(self):
        rate = 16000
        retained = [interval_row("target-00000", 1, 10, rate),
                    interval_row("target-00001", 46 * rate + 10, 46 * rate + 20, rate)]
        old = [interval_row("target-gap-00000", 10, 45 * rate + 10, rate,
                            old_asr_text="not truth", reference="not truth")]
        frames = 46 * rate + 21
        rows = coverage.build_uncovered_rows(retained, old, frames, rate)
        self.assertEqual([(r["start_sample"], r["end_sample"]) for r in rows],
                         [(0, 1), (10, 45 * rate + 10), (45 * rate + 10, 46 * rate + 10),
                          (46 * rate + 20, frames)])
        self.assertEqual(rows[1]["id"], "target-gap-00000")
        self.assertEqual(rows[1]["reused_gap_id"], "target-gap-00000")
        self.assertEqual(rows[1]["start"], 10 / rate)
        for row in rows:
            self.assertTrue(row["vad_rejected"])
            self.assertTrue(row["review_required"])
            self.assertFalse(row["speech_is_verified"])
            self.assertNotIn("old_asr_text", row)
            self.assertNotIn("reference", row)

    def test_no_legacy_gaps_splits_at_exact_limit_without_dropping_last_sample(self):
        rows = coverage.build_uncovered_rows([], [], 21, 1, maximum_samples=10)
        self.assertEqual([(r["start_sample"], r["end_sample"]) for r in rows],
                         [(0, 10), (10, 20), (20, 21)])
        self.assertEqual(coverage.build_uncovered_rows([interval_row("all", 0, 21, 1)], [], 21, 1), [])

    def test_new_ids_do_not_collide_with_retained_or_reused_ids(self):
        retained = [interval_row("target-uncovered-00000", 3, 5, 1)]
        gaps = [interval_row("target-uncovered-00001", 6, 7, 1)]
        rows = coverage.build_uncovered_rows(retained, gaps, 9, 1)
        self.assertEqual(len({r["id"] for r in retained + rows}), len(retained + rows))

    def test_legacy_overlap_and_duplicate_ids_fail(self):
        retained = [interval_row("a", 3, 8, 1)]
        for gaps in ([interval_row("gap", 7, 10, 1)], [interval_row("a", 8, 10, 1)],
                     [interval_row("gap", 8, 10, 1), interval_row("gap", 0, 2, 1)]):
            with self.subTest(gaps=gaps), self.assertRaises(ValueError):
                coverage.build_uncovered_rows(retained, gaps, 12, 1)

    def test_samples_seconds_duration_and_id_validation(self):
        good = {**interval_row("a", 101, 201), "start_sample": 101, "end_sample": 201}
        self.assertEqual(coverage.row_interval(good, 500, 16000), (101, 201))
        for change in ({"end_sample": 202}, {"duration_s": 1}, {"start": 0.00003},
                       {"start": float("nan")}, {"end": float("inf")},
                       {"sample_rate": 8000}, {"start_sample": True}, {"id": "../escape"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                coverage.normalize_rows([{**good, **change}], 500, 16000)

    def test_randomized_conservation_against_per_sample_oracle(self):
        randomizer = random.Random(20261002)
        for _ in range(150):
            frames = randomizer.randint(1, 90)
            mask = [randomizer.choice((True, False)) for _ in range(frames)]
            retained = []
            cursor = 0
            while cursor < frames:
                if not mask[cursor]:
                    cursor += 1
                    continue
                start = cursor
                while cursor < frames and mask[cursor]:
                    cursor += 1
                retained.append(interval_row(f"r-{len(retained)}", start, cursor, 1))
            rows = coverage.build_uncovered_rows(retained, [], frames, 1, maximum_samples=7)
            counts = [0] * frames
            recovered = set()
            for row in retained + rows:
                start, end = coverage.row_interval(row, frames, 1)
                for sample in range(start, end):
                    counts[sample] += 1
                    if row in rows:
                        recovered.add(sample)
            self.assertEqual(counts, [1] * frames)
            self.assertEqual(recovered, {i for i, kept in enumerate(mask) if not kept})


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.info = {"frames": 12, "sample_rate": 16000, "path": "/source.wav", "sha256": "abc"}
        self.retained = [interval_row("kept", 3, 8, audio="/kept.wav", split="dev", corpus="target", rms=0.2)]
        self.gaps = coverage.build_uncovered_rows(self.retained, [], 12, 16000)
        for row in self.gaps:
            row["audio"] = f"/{row['id']}.wav"
        self.retained_results = [result_row(self.retained[0], text="  repeated  \nraw ", custom={"keep": [1, 2]})]
        self.gap_results = [result_row(row, text="  repeated  \nraw " if index == 0 else "")
                            for index, row in enumerate(self.gaps)]

    def merge(self, **changes):
        values = dict(retained_manifest=self.retained, uncovered_manifest=self.gaps,
                      retained_results=self.retained_results, uncovered_results=self.gap_results,
                      info=self.info, config=CONFIG)
        values.update(changes)
        return coverage.merge_results(**values)

    def test_merge_preserves_raw_text_metadata_blank_rows_and_absolute_time(self):
        merged = self.merge(uncovered_results=list(reversed(self.gap_results)))
        self.assertEqual([(r["start_sample"], r["end_sample"]) for r in merged], [(0, 3), (3, 8), (8, 12)])
        self.assertEqual([r["text"] for r in merged], ["  repeated  \nraw ", "  repeated  \nraw ", ""])
        self.assertEqual(merged[1]["custom"], {"keep": [1, 2]})
        self.assertEqual(merged[1]["generation"], self.retained_results[0]["generation"])
        self.assertFalse(merged[1]["vad_rejected"])
        self.assertTrue(merged[0]["vad_rejected"])
        self.assertEqual(merged[1]["start"], 3 / 16000)
        self.assertEqual(merged[0]["source_sha256"], "abc")
        self.assertNotIn("start_sample", self.retained_results[0])

    def test_missing_extra_duplicate_changed_id_interval_metadata_and_config_fail(self):
        mutations = [[], self.gap_results + [self.gap_results[0]],
                     [{**self.gap_results[0], "id": "other"}, self.gap_results[1]],
                     [{**self.gap_results[0], "end_sample": 2, "end": 2 / 16000,
                       "duration_s": 2 / 16000}, self.gap_results[1]],
                     [{**self.gap_results[0], "audio": "/wrong.wav"}, self.gap_results[1]],
                     [{**self.gap_results[0], "vad_rejected": False}, self.gap_results[1]],
                     [{**self.gap_results[0], "route": "auto"}, self.gap_results[1]]]
        for rows in mutations:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.merge(uncovered_results=rows)

    def test_complete_ids_without_full_sample_coverage_fail(self):
        gap = interval_row("incomplete-gap", 0, 2, audio="/gap.wav", split="dev", corpus="target")
        with self.assertRaisesRegex(ValueError, "Incomplete source coverage"):
            self.merge(uncovered_manifest=[gap], uncovered_results=[result_row(gap)])

    def test_overlapping_result_sets_and_shared_ids_fail(self):
        for gap in (interval_row("overlap", 0, 4), interval_row("kept", 8, 12)):
            with self.subTest(gap=gap), self.assertRaises(ValueError):
                self.merge(uncovered_manifest=[gap], uncovered_results=[result_row(gap)])

    def test_conflicting_source_provenance_is_not_silently_replaced(self):
        with self.assertRaisesRegex(ValueError, "Conflicting merge metadata"):
            self.merge(retained_results=[{**self.retained_results[0], "source_sha256": "wrong"}])

    def test_generation_audit_reports_eos_caps_and_missing_fields_without_dropping_rows(self):
        merged = self.merge()
        merged[0]["generation"] = [{"new_tokens": 512, "ended_with_eos": False, "token_cap_hit": False}]
        merged[1]["generation"] = []
        merged[2]["generation"] = [{"new_tokens": 0}, "malformed"]
        before = coverage.jsonl_bytes(merged)
        warnings = coverage.audit_generation(merged)
        self.assertEqual(len(warnings), 3)
        self.assertEqual({issue["code"] for issue in warnings[0]["issues"]}, {"token_cap_hit", "not_ended_with_eos"})
        self.assertEqual(warnings[1]["issues"][0]["code"], "missing_generation_audit")
        self.assertEqual(coverage.jsonl_bytes(merged), before)
        review = coverage.review_text(merged, warnings, 16000).decode()
        self.assertEqual(review.count("[LOW-VAD / REVIEW]"), 3)  # Header plus both uncovered rows.
        self.assertIn("[GENERATION / REVIEW]", review)
        self.assertIn("  repeated  \nraw ", review)
        self.assertIn(merged[2]["id"], review)
        self.assertEqual(coverage.sample_stamp(16001, 16000), "00:00:01.0000625")


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for directory in ("data", "results", "code"):
            (self.root / directory).mkdir()
        self.source_path = self.root / "data/Recording_1828.wav"
        self.pcm = struct.pack("<" + "h" * 21, *range(-10, 11))
        info = {"channels": 1, "sample_width": 2, "sample_rate": 16000}
        self.source_path.write_bytes(coverage.wav_bytes(self.pcm, info))
        self.target_clip = self.root / "data/kept.wav"
        self.target_clip.write_bytes(coverage.wav_bytes(self.pcm[4:20], info))
        self.target = interval_row("kept", 2, 10, audio=str(self.target_clip), split="dev", corpus="target")
        self.target_path = self.root / "data/target-manifest.jsonl"
        self.target_path.write_bytes(coverage.jsonl_bytes([self.target]))
        gap = interval_row("target-gap-00000", 10, 20, old_asr_text="never use this", old_asr_words=3)
        (self.root / "data/target-gap-manifest.jsonl").write_bytes(coverage.jsonl_bytes([gap]))
        (self.root / "results/extra-selection.json").write_bytes(coverage.json_bytes({"winner": CONFIG}))
        (self.root / "code/noisy_asr_search.py").write_text("raise RuntimeError('GPU helper must never run in tests')\n")
        shutil.copyfile(Path(__file__).with_name("render_target_transcript.py"),
                        self.root / "code/render_target_transcript.py")

    def prepare(self):
        return coverage.prepare_coverage(self.root)

    def save_run(self, name, manifest, rows=None, summary=True):
        rows = [result_row(row) for row in manifest] if rows is None else rows
        prefix = self.root / "results" / name
        prefix.with_suffix(".jsonl").write_bytes(coverage.jsonl_bytes(rows))
        prefix.with_suffix(".run.json").write_bytes(coverage.json_bytes(coverage.run_fingerprint(self.root, manifest, CONFIG)))
        if summary:
            prefix.with_suffix(".summary.json").write_bytes(coverage.json_bytes(
                {"name": name, "split": "dev", "count": len(rows), **CONFIG, "wer": None}))
        return rows

    def finish_parent(self, context, rows=None):
        saved = self.save_run(coverage.SELECTED_NAME, context["retained"], rows=rows)
        (self.root / "results/target-complete.json").write_bytes(coverage.json_bytes({"configuration": CONFIG}))
        return saved

    def test_prepare_is_cpu_only_sample_exact_and_idempotent(self):
        with patch.object(coverage.subprocess, "run", side_effect=AssertionError("No subprocess during prepare")), \
                redirect_stdout(io.StringIO()):
            coverage.main(["--root", str(self.root), "--prepare-only"])
        context = self.prepare()
        self.assertFalse((self.root / "results/target-complete.json").exists())
        self.assertEqual([(r["start_sample"], r["end_sample"]) for r in context["uncovered"]],
                         [(0, 2), (10, 20), (20, 21)])
        manifest_path = self.root / "data" / coverage.MANIFEST_NAME
        before = manifest_path.read_bytes()
        recovered = bytearray(len(self.pcm))
        for row in context["retained"] + context["uncovered"]:
            with wave.open(row["audio"], "rb") as clip:
                recovered[row["start_sample"] * 2:row["end_sample"] * 2] = clip.readframes(clip.getnframes())
        self.assertEqual(bytes(recovered), self.pcm)
        self.prepare()
        self.assertEqual(before, manifest_path.read_bytes())
        self.assertEqual(self.source_path.read_bytes(), coverage.wav_bytes(self.pcm, context["source"]))
        self.assertNotIn("old_asr", before.decode())

    def test_source_config_manifest_or_helper_changes_fail_without_overwriting_artifacts(self):
        for changed in ("source", "selection", "target", "helper"):
            with self.subTest(changed=changed):
                context = self.prepare()
                manifest_before = context["manifest"].read_bytes()
                paths = {"source": self.source_path, "selection": self.root / "results/extra-selection.json",
                         "target": self.target_path, "helper": self.root / "code/noisy_asr_search.py"}
                path = paths[changed]
                original = path.read_bytes()
                if changed == "source":
                    path.write_bytes(coverage.wav_bytes(self.pcm[:-2] + b"\x7f\x00", context["source"]))
                elif changed == "selection":
                    path.write_bytes(coverage.json_bytes({"winner": {**CONFIG, "threshold": 0.6}}))
                else:
                    path.write_bytes(original + b"\n")
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertEqual(context["manifest"].read_bytes(), manifest_before)
                path.write_bytes(original)

    def test_orphan_existing_artifacts_are_not_adopted(self):
        manifest = self.root / "data" / coverage.MANIFEST_NAME
        manifest.write_text("existing work\n")
        with self.assertRaisesRegex(ValueError, "lack.*fingerprint"):
            self.prepare()
        self.assertEqual(manifest.read_text(), "existing work\n")

    def test_corrupted_existing_manifest_or_clip_fails_without_repairing_it(self):
        context = self.prepare()
        path = Path(context["uncovered"][0]["audio"])
        original = path.read_bytes()
        path.write_bytes(original[:-1] + b"x")
        with self.assertRaisesRegex(ValueError, "Existing artifact differs"):
            self.prepare()
        self.assertEqual(path.read_bytes(), original[:-1] + b"x")
        path.write_bytes(original)
        context["manifest"].write_bytes(b"corrupt\n")
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            self.prepare()
        self.assertEqual(context["manifest"].read_bytes(), b"corrupt\n")

    def test_matching_preparation_resumes_missing_clip_and_manifest(self):
        context = self.prepare()
        context["manifest"].unlink()
        Path(context["uncovered"][0]["audio"]).unlink()
        self.prepare()
        self.assertTrue(context["manifest"].exists())
        self.assertTrue(Path(context["uncovered"][0]["audio"]).exists())

    def test_retained_clip_must_match_source_samples(self):
        original = self.target_clip.read_bytes()
        self.target_clip.write_bytes(original[:-2] + b"xx")
        with self.assertRaisesRegex(ValueError, "Retained clip samples"):
            self.prepare()
        self.assertFalse((self.root / "data" / coverage.STATE_NAME).exists())

    def test_completion_marker_without_full_parent_rows_cannot_start_gpu(self):
        context = self.prepare()
        self.finish_parent(context, rows=[])
        with patch.object(coverage.subprocess, "run") as run, self.assertRaisesRegex(ValueError, "Incomplete result set"):
            coverage.complete_uncovered(self.root, context)
        run.assert_not_called()

    def test_missing_marker_and_merge_only_incomplete_runs_never_start_gpu(self):
        context = self.prepare()
        with patch.object(coverage.subprocess, "run") as run, self.assertRaises(FileNotFoundError):
            coverage.complete_uncovered(self.root, context)
        run.assert_not_called()
        self.finish_parent(context)
        with patch.object(coverage.subprocess, "run") as run, self.assertRaisesRegex(ValueError, "incomplete"):
            coverage.complete_uncovered(self.root, context, merge_only=True)
        run.assert_not_called()

    def test_incomplete_uncovered_rows_with_stale_summary_resume_instead_of_skip(self):
        context = self.prepare()
        self.finish_parent(context)
        self.save_run(coverage.UNCOVERED_NAME, context["uncovered"], rows=[result_row(context["uncovered"][0])])
        summary = self.root / "results" / f"{coverage.UNCOVERED_NAME}.summary.json"
        value = coverage.read_json(summary)
        summary.write_bytes(coverage.json_bytes({**value, "count": len(context["uncovered"])}))

        def cpu_fake_asr(command, **kwargs):
            self.assertTrue((self.root / "results/target-complete.json").exists())
            self.assertEqual(command, coverage.asr_command(self.root, context))
            self.save_run(coverage.UNCOVERED_NAME, context["uncovered"])

        with patch.object(coverage.subprocess, "run", side_effect=cpu_fake_asr) as run:
            rows = coverage.complete_uncovered(self.root, context)
        self.assertEqual(len(rows), 3)
        run.assert_called_once()
        command = run.call_args.args[0]
        for option, expected in (("--limit", "0"), ("--corpus", "all"), ("--route", "base"),
                                 ("--beam", "2"), ("--filter", "raw"), ("--name", coverage.UNCOVERED_NAME)):
            self.assertEqual(command[command.index(option) + 1], expected)

    def test_complete_jsonl_without_summary_is_cpu_resumable(self):
        context = self.prepare()
        self.finish_parent(context)
        self.save_run(coverage.UNCOVERED_NAME, context["uncovered"], summary=False)
        with patch.object(coverage.subprocess, "run") as run:
            rows = coverage.complete_uncovered(self.root, context)
        run.assert_not_called()
        self.assertEqual(len(rows), 3)
        coverage.validate_summary(self.root, coverage.UNCOVERED_NAME, 3, CONFIG)

    def test_run_fingerprint_and_malformed_jsonl_fail_without_discarding_data(self):
        context = self.prepare()
        self.finish_parent(context)
        self.save_run(coverage.UNCOVERED_NAME, context["uncovered"])
        path = self.root / "results" / f"{coverage.UNCOVERED_NAME}.jsonl"
        original = path.read_bytes()
        path.write_bytes(original + b'{"partial":')
        with self.assertRaisesRegex(ValueError, "saved data was not discarded"):
            coverage.complete_uncovered(self.root, context)
        self.assertEqual(path.read_bytes(), original + b'{"partial":')
        path.write_bytes(original)
        sidecar = path.with_suffix(".run.json")
        sidecar.write_bytes(coverage.json_bytes({"wrong": True}))
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            coverage.complete_uncovered(self.root, context)

    def test_waits_until_parent_marker_before_any_asr_invocation(self):
        calls = []

        def release_parent(seconds):
            self.assertFalse((self.root / "results/target-complete.json").exists())
            self.assertEqual(calls, [])
            self.finish_parent(self.prepare())

        real_run = subprocess.run

        def only_cpu_subprocess(command, **kwargs):
            if "noisy_asr_search.py" in command[1]:
                self.assertTrue((self.root / "results/target-complete.json").exists())
                calls.append("ASR simulated")
                context = self.prepare()
                self.save_run(coverage.UNCOVERED_NAME, context["uncovered"])
            else:
                return real_run(command, **kwargs)

        with patch.object(coverage.time, "sleep", side_effect=release_parent) as sleep, \
                patch.object(coverage.subprocess, "run", side_effect=only_cpu_subprocess), redirect_stdout(io.StringIO()):
            coverage.main(["--root", str(self.root)])
        sleep.assert_called_once()
        self.assertEqual(calls, ["ASR simulated"])
        self.assertTrue((self.root / "results" / f"{coverage.FULL_NAME}.coverage.json").exists())

    def test_full_export_uses_real_cpu_renderer_and_preserves_primary_and_raw_rows(self):
        context = self.prepare()
        selected = self.finish_parent(context)
        gaps = [result_row(row, text=" same raw \n " if index < 2 else "")
                for index, row in enumerate(context["uncovered"])]
        gaps[0]["generation"] = [{"new_tokens": 512, "ended_with_eos": False, "token_cap_hit": True}]
        self.save_run(coverage.UNCOVERED_NAME, context["uncovered"], rows=gaps)
        primary_path = self.root / "results" / f"{coverage.SELECTED_NAME}.jsonl"
        primary_before = primary_path.read_bytes()
        with redirect_stdout(io.StringIO()):
            coverage.main(["--root", str(self.root), "--merge-only"])
        prefix = self.root / "results" / coverage.FULL_NAME
        merged = coverage.read_jsonl(prefix.with_suffix(".jsonl"))
        self.assertEqual([r["text"] for r in merged], [gaps[0]["text"], selected[0]["text"], gaps[1]["text"], ""])
        rendered = coverage.read_json(prefix.with_suffix(".json"))
        self.assertEqual(rendered["segments"], merged)
        self.assertFalse(rendered["verified_reference"])
        summary = coverage.read_json(prefix.with_suffix(".coverage.json"))
        self.assertEqual(summary["source_samples"], 21)
        self.assertEqual(summary["retained"]["samples"], 8)
        self.assertEqual(summary["uncovered"]["samples"], 13)
        self.assertAlmostEqual(summary["retained"]["proportion"] + summary["uncovered"]["proportion"], 1)
        self.assertEqual(summary["generation_warning_rows"], 1)
        self.assertIsNone(summary["verified_target_wer"])
        self.assertEqual(primary_path.read_bytes(), primary_before)
        for suffix in (".txt", "_timestamped.txt", "_review_timestamped.txt", ".warnings.json"):
            self.assertTrue((prefix.parent / (coverage.FULL_NAME + suffix)).exists())
        self.assertFalse((prefix.parent / (coverage.FULL_NAME + "_comparison.md")).exists())
        before = prefix.with_suffix(".jsonl").read_bytes()
        coverage.finalize(self.root, context, selected, gaps)
        self.assertEqual(prefix.with_suffix(".jsonl").read_bytes(), before)

    def test_all_retained_source_never_invokes_asr(self):
        self.target_clip.write_bytes(self.source_path.read_bytes())
        self.target = interval_row("kept", 0, 21, audio=str(self.target_clip), split="dev", corpus="target")
        self.target_path.write_bytes(coverage.jsonl_bytes([self.target]))
        (self.root / "data/target-gap-manifest.jsonl").write_bytes(b"")
        context = self.prepare()
        self.finish_parent(context)
        with patch.object(coverage.subprocess, "run") as run:
            self.assertEqual(coverage.complete_uncovered(self.root, context), [])
        run.assert_not_called()

    def test_concurrent_workflow_lock_refuses_second_owner(self):
        with coverage.coverage_lock(self.root):
            with self.assertRaisesRegex(ValueError, "holds the lock"):
                with coverage.coverage_lock(self.root):
                    self.fail("Second lock owner entered")

    def prepare_router_base(self, probabilities=(0.49, 0.5, 0.8, 0.0)):
        context = self.prepare()
        selected = self.finish_parent(context)
        uncovered = self.save_run(coverage.UNCOVERED_NAME, context["uncovered"])
        selected[0]["degraded_probability"] = probabilities[1]
        for row, probability in zip(uncovered, (probabilities[0], probabilities[2], probabilities[3])):
            row["degraded_probability"] = probability
        self.save_run(coverage.SELECTED_NAME, context["retained"], rows=selected)
        self.save_run(coverage.UNCOVERED_NAME, context["uncovered"], rows=uncovered)
        coverage.finalize(self.root, context, selected, uncovered)
        base = coverage.read_jsonl(self.root / "results" / f"{coverage.FULL_NAME}.jsonl")
        return context, base

    def test_router_selection_is_reference_free_and_includes_exact_threshold_only(self):
        context, base = self.prepare_router_base()
        rows = coverage.router_manifest_rows(base, context["source"])
        self.assertEqual([row["id"] for row in rows], [base[1]["id"], base[2]["id"]])
        for row in rows:
            self.assertNotIn("text", row)
            self.assertNotIn("reference", row)
            self.assertNotIn("generation", row)
            self.assertEqual(row["base_audio_sha256"], hashlib.sha256(Path(row["audio"]).read_bytes()).hexdigest())
        for changed in ({"degraded_probability": None}, {"degraded_probability": float("nan")},
                        {"degraded_probability": 1.1}, {"sampling_seed": 0}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                coverage.router_manifest_rows([{**base[0], **changed}] + base[1:], context["source"])

    def test_router_merge_validates_audio_interval_seed_and_keeps_low_rows_identical(self):
        context, base = self.prepare_router_base()
        manifest = coverage.router_manifest_rows(base, context["source"])
        config = {**CONFIG, "route": "auto"}
        routed = [result_row(row, text=" routed raw ", route="auto", lora_active=True,
                             degraded_probability=row["base_degraded_probability"]) for row in manifest]
        merged = coverage.merge_router_results(base, routed, manifest, context["source"], config)
        self.assertEqual(merged[0], base[0])
        self.assertEqual(merged[3], base[3])
        self.assertEqual(merged[1]["text"], " routed raw ")
        for changed in ({"sampling_seed": 1}, {"audio": "/wrong.wav"}, {"start_sample": 1},
                        {"base_audio_sha256": "wrong"}, {"lora_active": False}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                coverage.merge_router_results(base, [{**routed[0], **changed}, routed[1]], manifest,
                                              context["source"], config)

    def test_router_gpu_is_serial_after_primary_export_and_does_not_replace_primary(self):
        context, base = self.prepare_router_base()
        primary_path = self.root / "results" / f"{coverage.FULL_NAME}.jsonl"
        primary_before = primary_path.read_bytes()
        real_run = subprocess.run
        calls = []

        def fake_router_only(command, **kwargs):
            if "noisy_asr_search.py" not in command[1]:
                return real_run(command, **kwargs)
            self.assertTrue(primary_path.with_suffix(".coverage.json").exists())
            self.assertTrue(primary_path.with_suffix(".json").exists())
            self.assertTrue(primary_path.with_suffix(".txt").exists())
            self.assertEqual(command[command.index("--route") + 1], "auto")
            self.assertEqual(command[command.index("--name") + 1], coverage.ROUTED_NAME)
            manifest = coverage.read_jsonl(Path(command[command.index("--manifest") + 1]))
            config = {**CONFIG, "route": "auto"}
            rows = [result_row(row, text=" alternative raw ", route="auto", lora_active=True,
                               degraded_probability=row["base_degraded_probability"]) for row in manifest]
            prefix = self.root / "results" / coverage.ROUTED_NAME
            prefix.with_suffix(".jsonl").write_bytes(coverage.jsonl_bytes(rows))
            prefix.with_suffix(".run.json").write_bytes(coverage.json_bytes(coverage.run_fingerprint(self.root, manifest, config)))
            prefix.with_suffix(".summary.json").write_bytes(coverage.json_bytes(
                {"name": coverage.ROUTED_NAME, "count": len(rows), "split": "dev", **config}))
            calls.append("simulated router")

        with patch.object(coverage.subprocess, "run", side_effect=fake_router_only):
            summary = coverage.router_alternative(self.root, context)
            coverage.router_alternative(self.root, context, merge_only=True)
        self.assertEqual(calls, ["simulated router"])
        self.assertEqual(summary["routed_rows"], 2)
        self.assertEqual(summary["unchanged_rows"], 2)
        self.assertIsNone(summary["verified_target_wer"])
        self.assertEqual(primary_before, primary_path.read_bytes())
        prefix = self.root / "results" / coverage.ROUTER_NAME
        alternate = coverage.read_jsonl(prefix.with_suffix(".jsonl"))
        self.assertEqual(alternate[0], base[0])
        self.assertEqual(alternate[3], base[3])
        for suffix in (".json", ".txt", "_timestamped.txt", "_review_timestamped.txt", ".warnings.json"):
            self.assertTrue((prefix.parent / (coverage.ROUTER_NAME + suffix)).exists())

    def test_router_requires_primary_completion_and_merge_only_never_launches_gpu(self):
        context = self.prepare()
        self.finish_parent(context)
        with patch.object(coverage.subprocess, "run") as run, self.assertRaises(FileNotFoundError):
            coverage.router_alternative(self.root, context)
        run.assert_not_called()
        context, _ = self.prepare_router_base()
        with patch.object(coverage.subprocess, "run") as run, self.assertRaisesRegex(ValueError, "incomplete"):
            coverage.router_alternative(self.root, context, merge_only=True)
        run.assert_not_called()

    def test_router_empty_subset_never_launches_gpu(self):
        context, base = self.prepare_router_base(probabilities=(0.1, 0.2, 0.3, 0.4))
        real_run = subprocess.run

        def renderer_only(command, **kwargs):
            self.assertIn("render_target_transcript.py", command[1])
            return real_run(command, **kwargs)

        with patch.object(coverage.subprocess, "run", side_effect=renderer_only):
            summary = coverage.router_alternative(self.root, context)
        self.assertEqual(summary["routed_rows"], 0)
        self.assertEqual(coverage.read_jsonl(self.root / "results" / f"{coverage.ROUTER_NAME}.jsonl"), base)


if __name__ == "__main__":
    unittest.main()
