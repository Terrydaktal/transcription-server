"""Small, offline helper tests; no model loading, GPU work, or real VAD."""

import hashlib
import importlib.util
import io
import json
import math
import random
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def load_helper(filename):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


search = load_helper("noisy_asr_search.py")
target = load_helper("prepare_target_audio.py")
with patch.dict(sys.modules, {"noisy_asr_search": search}):
    extra = load_helper("extra_noisy_models.py")


def module_stub(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


def audio_stubs(audio, rate=16000):
    soundfile = module_stub(
        "soundfile",
        read=Mock(return_value=(audio, rate)),
        write=Mock(side_effect=lambda path, *args, **kwargs: Path(path).touch()),
    )
    signal = module_stub(
        "scipy.signal", resample_poly=Mock(side_effect=lambda audio, *args, **kwargs: audio.copy())
    )
    return {
        "soundfile": soundfile,
        "scipy": module_stub("scipy", signal=signal, __path__=[]),
        "scipy.signal": signal,
    }


class ManifestSplitTests(unittest.TestCase):
    def test_prepare_isolates_group_and_speaker_across_variants_and_reordering(self):
        vitw_rows = [
            {"name": f"vitw-{group}-{noise}", "index": group,
             "subset": f"noise-en-{noise}", "audio": {"bytes": b"fixture"}, "answer": "reference"}
            for group in range(4) for noise in ("quiet", "street")
        ]
        vitw_rows.append({"subset": "noise-fr-street"})
        krisp_rows = [
            {"id": f"{speaker}-{environment}", "speaker_id": speaker,
             "file_name": f"{speaker}-{environment}.wav", "environment": environment,
             "full_transcript": "reference", "recording_device": "phone"}
            for speaker in ("speaker-a", "speaker-b", "speaker-c", "speaker-d")
            for environment in ("office", "street")
        ]
        expected_groups = {row["name"]: ("vitw", str(row["index"])) for row in vitw_rows[:-1]}
        expected_groups.update({f"krisp-phone-{row['id']}": ("krisp", row["speaker_id"])
                                for row in krisp_rows})
        parquet = module_stub(
            "pyarrow.parquet",
            read_table=Mock(side_effect=lambda path: SimpleNamespace(to_pylist=lambda: vitw_rows)),
        )
        stubs = audio_stubs(np.array([[0.125, 0.25], [-0.25, 0.5]], dtype=np.float32))
        stubs.update({"pyarrow": module_stub("pyarrow", parquet=parquet, __path__=[]),
                      "pyarrow.parquet": parquet})
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, stubs), \
                patch.object(search, "emit"):
            root = Path(directory)
            shard = root / "data/vitw/data/fixture.parquet"
            shard.parent.mkdir(parents=True)
            shard.touch()
            metadata = root / "data/krisp/phone_calls/metadata.jsonl"
            metadata.parent.mkdir(parents=True)
            for row in krisp_rows:
                (metadata.parent / row["file_name"]).touch()
            metadata.write_text("".join(json.dumps(row) + "\n" for row in krisp_rows))
            search.prepare(root)
            manifest = search.read_jsonl(root / "data/manifest.jsonl")

            self.assertCountEqual([row["id"] for row in manifest], expected_groups)
            splits = {}
            for row in manifest:
                identity = expected_groups[row["id"]]
                self.assertEqual((row["corpus"], row["group"]), identity)
                splits.setdefault(identity, set()).add(row["split"])
            self.assertTrue(all(len(values) == 1 for values in splits.values()))
            for corpus in ("vitw", "krisp"):
                self.assertEqual({row["split"] for row in manifest if row["corpus"] == corpus},
                                 {"dev", "holdout"})

            # Neither source order nor scoring references may influence selection.
            vitw_rows.reverse()
            krisp_rows.reverse()
            for row in vitw_rows:
                row["answer"] = "changed reference"
            for row in krisp_rows:
                row["full_transcript"] = "changed reference"
            metadata.write_text("".join(json.dumps(row) + "\n" for row in krisp_rows))
            search.prepare(root)
            repeated = search.read_jsonl(root / "data/manifest.jsonl")
            self.assertEqual([(row["id"], row["split"]) for row in repeated],
                             [(row["id"], row["split"]) for row in manifest])

    def test_cli_filters_before_limit_and_zero_means_unlimited(self):
        rows = [
            {"id": "held-out", "split": "holdout", "corpus": "vitw"},
            {"id": "other-corpus", "split": "dev", "corpus": "krisp"},
            {"id": "first", "split": "dev", "corpus": "vitw"},
            {"id": "second", "split": "dev", "corpus": "vitw"},
            {"id": "third", "split": "dev", "corpus": "vitw"},
        ]
        for limit, expected in ((1, ["first"]), (0, ["first", "second", "third"]),
                                (10, ["first", "second", "third"])):
            with self.subTest(limit=limit), patch.object(search, "read_jsonl", return_value=rows), \
                    patch.object(search, "run_mega") as run, patch.object(
                        sys, "argv", ["noisy_asr_search.py", "mega", "--root", "/unused",
                                      "--split", "dev", "--corpus", "vitw", "--limit", str(limit)]):
                search.main()
                run.assert_called_once()
                self.assertEqual([row["id"] for row in run.call_args.args[2]], expected)


class ResumeResultsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.destination = self.root / "results/partial.jsonl"
        self.destination.parent.mkdir()
        self.sidecar = self.destination.with_suffix(".run.json")
        self.args = SimpleNamespace(root=self.root, stage="whisper", filter="raw", beam=2,
                                    route="auto", language="en", length_penalty=1.1,
                                    threshold=0.5, lora_scale=1.0)
        self.rows = []
        for index in range(3):
            audio = self.root / f"clip-{index}.wav"
            audio.write_bytes(f"audio-source-{index}".encode())
            self.rows.append({"id": f"clip-{index}", "audio": str(audio),
                              "reference": f"reference {index}", "split": "holdout"})
        self.metadata = self.root / "models/whisper-large-v3/.cache/huggingface/download/config.json.metadata"
        self.metadata.parent.mkdir(parents=True)
        self.metadata.write_text("commit-fixture\netag-fixture\n123.0\n")
        self.code = self.root / "benchmark-source.fixture"
        self.code.write_bytes(b"benchmark-code-v1")
        self.enterContext(patch.object(search, "__file__", str(self.code)))

    def save_results(self, results):
        self.destination.write_text("".join(json.dumps(row) + "\n" for row in results))

    def start_partial_run(self):
        self.assertEqual(search.resume_results(self.destination, self.rows, self.args), [])
        results = [{**self.rows[index], "text": f"prediction {index}", "stage": "whisper",
                    "filter": "raw", "beam": 2, "length_penalty": 1.1} for index in (2, 0)]
        self.save_results(results)
        return results

    def assert_resume_rejected(self, pattern, rows=None):
        paths = (self.destination, self.sidecar)
        before = {path: path.read_bytes() if path.exists() else None for path in paths}
        with self.assertRaisesRegex(ValueError, pattern):
            search.resume_results(self.destination, self.rows if rows is None else rows, self.args)
        self.assertEqual({path: path.read_bytes() if path.exists() else None for path in paths}, before)

    def test_new_run_records_model_code_configuration_and_all_audio_hashes(self):
        self.assertEqual(search.resume_results(self.destination, self.rows, self.args), [])
        self.assertFalse(self.destination.exists())
        fingerprint = json.loads(self.sidecar.read_text())
        self.assertEqual(fingerprint["stage"], "whisper")
        self.assertEqual(fingerprint["model"], str(self.root / "models/whisper-large-v3"))
        self.assertEqual(fingerprint["model_metadata"], {
            ".cache/huggingface/download/config.json.metadata": ["commit-fixture", "etag-fixture"]})
        self.assertEqual(fingerprint["code_sha256"], hashlib.sha256(b"benchmark-code-v1").hexdigest())
        self.assertEqual(fingerprint["configuration"], {
            "filter": "raw", "beam": 2, "route": "auto", "language": "en",
            "length_penalty": 1.1, "threshold": 0.5, "lora_scale": 1.0})
        self.assertEqual(fingerprint["audio_sha256"], {
            f"clip-{index}": hashlib.sha256(f"audio-source-{index}".encode()).hexdigest()
            for index in range(3)})

    def test_exact_partial_ids_are_accepted_independently_of_manifest_order(self):
        results = self.start_partial_run()
        before = (self.destination.read_bytes(), self.sidecar.read_bytes())
        for rows in (self.rows, list(reversed(self.rows))):
            with self.subTest(ids=[row["id"] for row in rows]):
                self.assertEqual(search.resume_results(self.destination, rows, self.args), results)
        self.assertEqual((self.destination.read_bytes(), self.sidecar.read_bytes()), before)

    def test_changed_stage_is_rejected(self):
        self.start_partial_run()
        for stage in ("mega", "vibe", "moss"):
            with self.subTest(stage=stage), patch.object(self.args, "stage", stage):
                self.assert_resume_rejected("changed")

    def test_changed_code_is_rejected(self):
        self.start_partial_run()
        self.code.write_bytes(b"benchmark-code-v2")
        self.assert_resume_rejected("changed")

    def test_every_changed_configuration_field_is_rejected(self):
        self.start_partial_run()
        for key, value in (("filter", "hp80"), ("beam", 4), ("route", "base"),
                           ("language", "auto"), ("length_penalty", 0.9),
                           ("threshold", 0.25), ("lora_scale", 0.5)):
            with self.subTest(field=key), patch.object(self.args, key, value):
                self.assert_resume_rejected("changed")

    def test_changed_model_commit_or_etag_is_rejected(self):
        self.start_partial_run()
        for metadata in ("other-commit\netag-fixture\n123.0\n",
                         "commit-fixture\nother-etag\n123.0\n"):
            with self.subTest(metadata=metadata):
                self.metadata.write_text(metadata)
                self.assert_resume_rejected("changed")

    def test_changed_audio_bytes_are_rejected_for_saved_and_unfinished_ids(self):
        self.start_partial_run()
        for index in (0, 1):
            with self.subTest(index=index):
                audio = Path(self.rows[index]["audio"])
                original = audio.read_bytes()
                audio.write_bytes(b"audio-source-X")
                self.assert_resume_rejected("changed")
                audio.write_bytes(original)

    def test_nonempty_results_without_sidecar_are_rejected(self):
        self.start_partial_run()
        self.sidecar.unlink()
        self.assert_resume_rejected("lack a reproducibility fingerprint")

    def test_saved_ids_must_be_a_unique_subset_of_unique_manifest_ids(self):
        results = self.start_partial_run()
        for rows, saved, message in (
            (self.rows + [self.rows[0]], results, "Duplicate audio IDs"),
            (self.rows, results + [results[0]], "Duplicate audio IDs"),
            (self.rows, [{**results[0], "id": "outside-cohort"}], "requested split/subset"),
        ):
            with self.subTest(message=message, manifest_count=len(rows), saved_count=len(saved)):
                self.save_results(saved)
                self.assert_resume_rejected(message, rows)

    def test_changed_saved_manifest_fields_are_rejected(self):
        results = self.start_partial_run()
        for key, value in (("audio", "/different.wav"), ("reference", "different reference"),
                           ("split", "dev")):
            with self.subTest(field=key):
                self.save_results([{**results[0], key: value}])
                self.assert_resume_rejected(f"Changed manifest field {key}")


class ClipSeedingTests(unittest.TestCase):
    def test_seed_contract_and_metadata_independence_with_fake_torch(self):
        manual_seed = Mock()
        with patch.dict(sys.modules, {"torch": module_stub("torch", manual_seed=manual_seed)}):
            seed = search.seed_clip({"id": "clip-a"})
            self.assertEqual(seed, 20261002 ^ 0x8F1690EC)
            manual_seed.assert_called_once_with(seed)
            manual_seed.reset_mock()
            self.assertEqual(search.seed_clip({"id": "clip-a", "reference": "changed reference",
                                               "audio": "/other.wav", "split": "holdout"}), seed)
            manual_seed.assert_called_once_with(seed)

    def test_per_id_random_streams_are_order_and_resume_independent(self):
        rng = random.Random(0)
        manual_seed = Mock(side_effect=rng.seed)
        ids = ["clip-a", "clip-b", "clip-c"]

        def streams(order):
            return {identity: (search.seed_clip({"id": identity}), [rng.random() for _ in range(4)])
                    for identity in order}

        with patch.dict(sys.modules, {"torch": module_stub("torch", manual_seed=manual_seed)}):
            original = streams(ids)
            self.assertEqual(len({seed for seed, values in original.values()}), len(ids))
            self.assertEqual(streams(list(reversed(ids))), original)
            self.assertEqual(streams(["clip-c", "clip-a"]),
                             {identity: original[identity] for identity in ("clip-c", "clip-a")})
            self.assertEqual(manual_seed.call_count, 8)


class TargetGroupingTests(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(target.group_speech([], 30, 4), [])

    def test_join_keeps_original_offsets_and_does_not_mutate_spans(self):
        spans = [{"start": 10, "end": 25}, {"start": 27, "end": 40},
                 {"start": 44, "end": 55}, {"start": 70, "end": 90}]
        original = [span.copy() for span in spans]
        groups = target.group_speech(spans, 50, 4)
        self.assertEqual(groups, [{"start": 10, "end": 55}, {"start": 70, "end": 90}])
        self.assertEqual(spans, original)
        groups[0]["start"] = -1
        self.assertEqual(spans, original)

    def test_cap_counts_pauses_and_gap_and_cap_boundaries_are_inclusive(self):
        for start, end, expected in (
            (20, 30, [{"start": 0, "end": 30}]),
            (21, 30, [{"start": 0, "end": 10}, {"start": 21, "end": 30}]),
            (20, 31, [{"start": 0, "end": 10}, {"start": 20, "end": 31}]),
        ):
            with self.subTest(start=start, end=end):
                self.assertEqual(target.group_speech(
                    [{"start": 0, "end": 10}, {"start": start, "end": end}], 30, 10), expected)

    def test_oversized_individual_span_is_preserved_not_truncated(self):
        # The cap bounds joins only; callers must bound individual VAD spans.
        spans = [{"start": 10, "end": 41}, {"start": 43, "end": 45}]
        self.assertEqual(target.group_speech(spans, 30, 4), spans)

    def test_bounded_spans_have_no_lost_or_duplicated_speech_samples(self):
        rng = random.Random(1828)
        for case in range(30):
            spans = []
            cursor = 0
            for _ in range(12):
                start = cursor + rng.randrange(9)
                cursor = start + rng.randrange(1, 16)
                spans.append({"start": start, "end": cursor})
            original = [span.copy() for span in spans]
            with self.subTest(case=case):
                groups = target.group_speech(spans, 30, 5)
                covered = [sample for group in groups for sample in range(group["start"], group["end"])]
                self.assertEqual(len(covered), len(set(covered)))
                self.assertTrue(all(group["end"] - group["start"] <= 30 for group in groups))
                for span in spans:
                    containing = [group for group in groups if group["start"] <= span["start"]
                                  and group["end"] >= span["end"]]
                    self.assertEqual(len(containing), 1)
                self.assertEqual(spans, original)

    def test_target_exports_original_slices_including_internal_silence(self):
        audio = np.zeros(50010, dtype=np.float32)
        audio[2:5] = [0.1, 0.2, 0.3]
        audio[8:11] = [0.4, 0.5, 0.6]
        audio[50000:50003] = [0.2, 0.4, 0.6]
        original = audio.copy()
        spans = [{"start": 2, "end": 5}, {"start": 8, "end": 11},
                 {"start": 50000, "end": 50003}]
        stubs = audio_stubs(audio)
        vad = module_stub("silero_vad", load_silero_vad=Mock(return_value=sentinel.vad),
                          get_speech_timestamps=Mock(return_value=spans))
        stubs.update({"silero_vad": vad,
                      "torch": module_stub("torch", set_num_threads=Mock(), from_numpy=Mock(
                          side_effect=lambda array: array))})
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, stubs), \
                patch.object(sys, "argv", ["prepare_target_audio.py", "--root", directory]), \
                redirect_stdout(io.StringIO()):
            target.main()
            writes = stubs["soundfile"].write.call_args_list
            manifest = search.read_jsonl(Path(directory) / "data/target-manifest.jsonl")
            self.assertEqual(len(writes), 2)
            self.assertEqual([row["id"] for row in manifest], ["target-00000", "target-00001"])
            exported_positions = []
            for call, row, (start, end) in zip(writes, manifest, ((2, 11), (50000, 50003))):
                np.testing.assert_array_equal(call.args[1], original[start:end])
                self.assertEqual(call.args[2], 16000)
                self.assertEqual(call.kwargs["subtype"], "PCM_16")
                self.assertEqual(row["audio"], str(call.args[0]))
                self.assertEqual((row["start"], row["end"]), (start / 16000, end / 16000))
                self.assertEqual(row["duration_s"], (end - start) / 16000)
                exported_positions.extend(range(start, end))
            np.testing.assert_array_equal(writes[0].args[1][3:6], np.zeros(3))
            np.testing.assert_array_equal(audio, original)
            self.assertEqual(len(exported_positions), len(set(exported_positions)))
            self.assertEqual(json.loads((Path(directory) / "data/target-vad.json").read_text())[
                "speech_spans"], spans)


class QuietChunksTests(unittest.TestCase):
    def assert_exact_partition(self, audio, rate=100):
        original = audio.copy()
        chunks = extra.quiet_chunks(audio, rate=rate)
        if not len(audio):
            self.assertEqual(chunks, [])
            return chunks
        self.assertTrue(chunks)
        cursor = 0
        for start, end in chunks:
            self.assertEqual(start, cursor)
            self.assertGreater(end, start)
            self.assertLessEqual(end, len(audio))
            self.assertLessEqual(end - start, 30 * rate)
            cursor = end
        self.assertEqual(cursor, len(audio))
        np.testing.assert_array_equal(np.concatenate([audio[start:end] for start, end in chunks]), original)
        np.testing.assert_array_equal(audio, original)
        return chunks

    def test_empty_short_exact_cap_and_long_boundaries(self):
        for count in (0, 1, 2999, 3000, 3001, 6000, 6001):
            with self.subTest(samples=count):
                audio = np.arange(count, dtype=np.float32) / max(count, 1)
                chunks = self.assert_exact_partition(audio)
                if count <= 3000:
                    self.assertEqual(chunks, [(0, count)] if count else [])

    def test_default_30_second_cap_at_16khz(self):
        for count in (480000, 480001):
            with self.subTest(samples=count):
                chunks = self.assert_exact_partition(np.arange(count, dtype=np.float32) / count, rate=16000)
                self.assertEqual(len(chunks), 1 if count == 480000 else 2)

    def test_quiet_cut_splits_silence_without_dropping_or_repeating_it(self):
        audio = np.ones(3100, dtype=np.float32)
        audio[2800:2810] = 0
        chunks = self.assert_exact_partition(audio)
        self.assertEqual(chunks, [(0, 2805), (2805, 3100)])
        np.testing.assert_array_equal(audio[chunks[0][1] - 5:chunks[0][1]], np.zeros(5))
        np.testing.assert_array_equal(audio[chunks[1][0]:chunks[1][0] + 5], np.zeros(5))

    def test_long_recording_preserves_internal_and_edge_gaps(self):
        audio = np.random.default_rng(1828).uniform(0.1, 0.6, 9503).astype(np.float32)
        for start, end in ((0, 50), (700, 1200), (2800, 2850), (5605, 5645), (9413, 9503)):
            audio[start:end] = 0
        self.assertGreater(len(self.assert_exact_partition(audio)), 2)


@unittest.skipIf(torch is None, "CPU PyTorch is needed for router tensor tests")
class RouterPredictionTests(unittest.TestCase):
    def make_router(self, sample_count, probabilities, threshold):
        waveform = torch.arange(sample_count, dtype=torch.float64, device="cpu").reshape(1, -1)
        logits = [torch.tensor([[math.log(1 - value), math.log(value)]],
                               dtype=torch.float64, device="cpu") for value in probabilities]
        return SimpleNamespace(
            sample_rate=2, threshold=threshold, _load_audio=Mock(return_value=waveform),
            mel_extractor=Mock(side_effect=lambda window: torch.stack((window, window + 1), dim=1)),
            model=Mock(side_effect=logits),
        )

    def test_all_long_audio_windows_are_weighted_by_actual_sample_count(self):
        router = self.make_router(10, [0.25, 0.75, 0.9], 0.6)
        degraded, probability = search.router_prediction(router, "fixture.wav", window_seconds=2)
        self.assertFalse(degraded)
        self.assertAlmostEqual(probability, 0.58, places=12)
        router._load_audio.assert_called_once_with("fixture.wav")
        windows = [call.args[0] for call in router.mel_extractor.call_args_list]
        self.assertEqual([window.shape[-1] for window in windows], [4, 4, 2])
        self.assertTrue(torch.equal(torch.cat(windows, dim=-1), router._load_audio.return_value))
        self.assertEqual(router.model.call_count, 3)
        for call, length in zip(router.model.call_args_list, (4, 4, 2)):
            self.assertEqual(tuple(call.args[0].shape), (1, length, 2))
            self.assertEqual(call.kwargs, {"mask": None})

    def test_short_and_exact_windows_and_threshold_equality(self):
        for samples, windows in ((1, 1), (4, 1), (8, 2)):
            with self.subTest(samples=samples):
                router = self.make_router(samples, [0.5] * windows, 0.5)
                self.assertEqual(search.router_prediction(router, "fixture.wav", 2), (True, 0.5))
                self.assertEqual(router.model.call_count, windows)

    def test_empty_audio_is_rejected_without_model_calls(self):
        router = self.make_router(0, [], 0.5)
        with self.assertRaisesRegex(ValueError, "empty audio"):
            search.router_prediction(router, "empty.wav", 2)
        router.mel_extractor.assert_not_called()
        router.model.assert_not_called()


@unittest.skipIf(torch is None, "CPU PyTorch is needed for real BF16 roundtrip tests")
class PristineAdapterTests(unittest.TestCase):
    def test_bf16_pristine_restore_is_exact_across_repeated_route_switches(self):
        class Switch:
            def __init__(self):
                self.active = False
                self.items = []

            def add_adapter(self, weight, delta):
                self.items.append({"weight": weight, "delta": delta})

        constructor = Mock(side_effect=AssertionError("must not load a real model"))
        fake_modules = {
            name: module_stub(name, __path__=[])
            for name in ("MegaASR", "MegaASR.model", "MegaASR.model.utils")
        }
        fake_modules.update({
            "MegaASR.model.megaASR": module_stub("MegaASR.model.megaASR", MegaASR=constructor),
            "MegaASR.model.utils.lora_switch": module_stub(
                "MegaASR.model.utils.lora_switch", LoRADeltaSwitch=Switch),
        })
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, fake_modules))
            stack.enter_context(patch.object(sys, "path", sys.path.copy()))
            stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
            for backend in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_cudnn_sdp"):
                stack.enter_context(patch.object(torch.backends.cuda, backend))
            # load_mega installs the switch helpers before the GPU availability guard.
            with self.assertRaisesRegex(RuntimeError, "ROCm GPU"):
                search.load_mega(SimpleNamespace(lora_scale=1.0), Path("/unused"))
            constructor.assert_not_called()

            base = torch.tensor([1.0, 0.5, -2.0, 2.0], dtype=torch.bfloat16, device="cpu")
            delta = torch.tensor([0.003, 0.25, 0.01, -0.02], device="cpu")
            naive_restore = base.clone().add_(delta).sub_(delta)
            self.assertFalse(torch.equal(naive_restore, base), "fixture must expose BF16 drift")
            weight = torch.nn.Parameter(base.clone())
            second = torch.nn.Parameter(torch.tensor([0.25], dtype=torch.bfloat16, device="cpu"))
            switch = Switch()
            switch.add_adapter(weight, delta)
            switch.add_adapter(second, torch.tensor([0.125], device="cpu"))
            self.assertEqual(len(switch.pristine), 2)
            saved_weight, saved_base = switch.pristine[id(weight)]
            self.assertIs(saved_weight, weight)
            self.assertNotEqual(saved_base.data_ptr(), weight.data_ptr())
            self.assertFalse(saved_base.requires_grad)
            self.assertEqual(switch.set_active(False), 0.0)

            expected_active = torch.tensor([1.0, 0.75, -1.9921875, 1.9765625],
                                           dtype=torch.bfloat16, device="cpu")
            for cycle in range(5):
                with self.subTest(cycle=cycle):
                    self.assertGreaterEqual(switch.set_active(True), 0.0)
                    self.assertTrue(switch.active)
                    self.assertTrue(torch.equal(weight, expected_active))
                    self.assertEqual(second.item(), 0.375)
                    self.assertEqual(switch.set_active(True), 0.0)
                    self.assertTrue(torch.equal(weight, expected_active))
                    self.assertGreaterEqual(switch.set_active(False), 0.0)
                    self.assertFalse(switch.active)
                    self.assertTrue(torch.equal(weight, base))
                    self.assertEqual(second.item(), 0.25)
                    self.assertTrue(torch.equal(saved_base, base))


if __name__ == "__main__":
    unittest.main()
