import io
import logging
import os
import re
import subprocess
import tempfile
import unittest
import wave
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from starlette.requests import Request
from uvicorn.logging import DefaultFormatter

from whisper_service import (
    ANSI_BREAKDOWN_VALUE,
    PARAKEET_PRESETS,
    FasterWhisperTranscriber,
    FriendlyLogSource,
    GraniteSpeechTranscriber,
    ParakeetTranscriber,
    app,
    app_lifespan,
    build_log_config,
    build_runtime_config,
    canonicalize_technical_text,
    colour_timing_fields,
    configure_shared_model_caches,
    default_model_for_backend,
    desktop_delivery_report,
    first_nemo_transcription_text,
    format_audio_size,
    format_client_preparation,
    format_timing_fields,
    format_transcription_device,
    load_whisper_hotwords,
    normalize_technical_text,
    parse_parakeet_channel_selector,
    pipewire_microphone_sources,
    prepare_granite_audio,
    preserve_failed_request,
    preserve_keyword_casing,
    protect_technical_terms,
    report_microphone_startup_status,
    resume_desktop_dictation_queue,
    restore_technical_terms,
    transcribe_raw,
    write_recovery_copy,
)


class SharedModelCacheTests(unittest.TestCase):
    def test_uses_xdg_cache_and_preserves_explicit_overrides(self):
        with patch.dict(
            os.environ,
            {
                "XDG_CACHE_HOME": "/shared/cache",
                "HF_HOME": "/custom/huggingface",
            },
            clear=True,
        ):
            configure_shared_model_caches()

            self.assertEqual(os.environ["HF_HOME"], "/custom/huggingface")
            self.assertEqual(os.environ["TORCH_HOME"], "/shared/cache/torch")


class MicrophoneStartupStatusTests(unittest.TestCase):
    @patch("whisper_service.shutil.which", return_value="/usr/bin/pactl")
    @patch("whisper_service.subprocess.run")
    def test_monitor_source_does_not_count_as_microphone(self, run, _which):
        run.return_value = SimpleNamespace(
            returncode=0,
            stdout=(
                "71\talsa_output.pci-0000_01_00.1.hdmi-stereo.monitor\t"
                "PipeWire\ts32le 2ch 48000Hz\n"
            ),
        )

        self.assertEqual(pipewire_microphone_sources(), ())

    @patch("whisper_service.shutil.which", return_value="/usr/bin/pactl")
    @patch("whisper_service.subprocess.run")
    def test_real_capture_source_is_reported(self, run, _which):
        run.return_value = SimpleNamespace(
            returncode=0,
            stdout=(
                "70\talsa_input.usb-Blue_Microphones_Yeti-00.analog-stereo\t"
                "PipeWire\ts16le 2ch 48000Hz\n"
            ),
        )

        self.assertEqual(
            pipewire_microphone_sources(),
            ("alsa_input.usb-Blue_Microphones_Yeti-00.analog-stereo",),
        )

    @patch("whisper_service.pipewire_microphone_sources", return_value=())
    @patch("whisper_service.print")
    def test_missing_microphone_logs_clear_warning(self, service_print, _sources):
        report_microphone_startup_status()

        message = service_print.call_args.args[0]
        self.assertIn("WARNING: No microphone", message)
        self.assertIn("Local '+' dictation", message)
        self.assertIn("uploaded audio transcription is unaffected", message)


class DesktopQueueStartupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = self.root / "state" / "assistant-desktop-dictation"
        self.queue = self.state / "queue"
        self.queue.mkdir(parents=True)
        self.runtime = self.root / "runtime" / "assistant-desktop-dictation"
        self.runtime.mkdir(parents=True)
        env = patch.dict(
            os.environ,
            {
                "XDG_STATE_HOME": str(self.root / "state"),
                "XDG_RUNTIME_DIR": str(self.root / "runtime"),
            },
        )
        env.start()
        self.addCleanup(env.stop)

    @patch("whisper_service.subprocess.run")
    def test_empty_queue_does_not_start_a_worker(self, run):
        resume_desktop_dictation_queue()
        run.assert_not_called()

    @patch("whisper_service.print")
    @patch("whisper_service.subprocess.run")
    def test_backlog_starts_or_wakes_worker_and_is_not_deleted(self, run, service_print):
        recording = self.queue / "recording-001.wav"
        recording.write_bytes(b"fixture")
        resume_desktop_dictation_queue()

        command = run.call_args.args[0]
        self.assertEqual(Path(command[0]).name, "queue-worker.sh")
        self.assertEqual(command[1:], ["--background"])
        self.assertTrue(run.call_args.kwargs["close_fds"])
        self.assertTrue(run.call_args.kwargs["start_new_session"])
        self.assertEqual(run.call_args.kwargs["timeout"], 2)
        self.assertIn("1 completed recording(s)", service_print.call_args.args[0])
        self.assertTrue(recording.exists())

    @patch("whisper_service.subprocess.run")
    def test_only_active_audio_is_not_a_completed_backlog(self, run):
        recording = self.queue / "recording-001.wav"
        recording.write_bytes(b"fixture")
        (self.runtime / "current-audio-path").write_text(str(recording))
        resume_desktop_dictation_queue()
        run.assert_not_called()
        self.assertTrue(recording.exists())

    @patch("whisper_service.print")
    @patch("whisper_service.subprocess.run", side_effect=OSError("cannot launch"))
    def test_failed_worker_launch_warns_without_failing_server_startup(self, _run, service_print):
        (self.queue / "recording-001.wav").write_bytes(b"fixture")
        resume_desktop_dictation_queue()
        self.assertIn("WARNING: Could not resume", service_print.call_args.args[0])

    @patch("whisper_service.print")
    @patch("whisper_service.subprocess.run", side_effect=subprocess.TimeoutExpired("worker", 2))
    def test_launch_timeout_warns_and_preserves_backlog(self, _run, service_print):
        recording = self.queue / "recording-001.wav"
        recording.write_bytes(b"fixture")
        resume_desktop_dictation_queue()
        self.assertTrue(recording.exists())
        self.assertIn("WARNING: Could not resume", service_print.call_args.args[0])


class DesktopQueueLifespanTests(unittest.IsolatedAsyncioTestCase):
    @patch("whisper_service.resume_desktop_dictation_queue")
    async def test_lifespan_resumes_queue_and_closes_model_even_on_error(self, resume):
        model = Mock()
        application = SimpleNamespace(state=SimpleNamespace(transcriber=model))
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            async with app_lifespan(application):
                resume.assert_called_once_with()
                model.close.assert_not_called()
                raise RuntimeError("fixture")
        model.close.assert_called_once_with()


class FasterWhisperBatchingTests(unittest.TestCase):
    def make_transcriber(self, duration_seconds: float | None = 61.0):
        transcriber = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
        transcriber.config = SimpleNamespace(
            device="cuda",
            long_audio_batch_size=2,
            batch_threshold_seconds=60.0,
            language="en",
            beam_size=4,
            patience=1.0,
            length_penalty=1.05,
            vad_filter=True,
            vad_threshold=0.3,
            vad_min_silence_ms=1000,
            vad_speech_pad_ms=400,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.3,
            compression_ratio_threshold=2.4,
            initial_prompt="",
            hotwords="SQLAlchemy, Pydantic",
            cpu_fallback_enabled=True,
        )
        transcriber.model = object()
        transcriber._batched_pipeline = None
        transcriber._gpu_model_needs_reload = False
        transcriber._audio_duration_seconds = Mock(return_value=duration_seconds)
        transcriber._get_primary_model = Mock(return_value=transcriber.model)
        transcriber._release_cuda_cache = Mock()
        transcriber._invalidate_primary_model = Mock()
        transcriber._get_cpu_fallback_model = Mock(return_value="cpu-model")
        return transcriber

    def test_transcription_options_pass_hotwords_to_both_whisper_paths(self):
        options = self.make_transcriber()._transcription_options()

        self.assertEqual(options["hotwords"], "SQLAlchemy, Pydantic")
        self.assertEqual(options["patience"], 1.0)
        self.assertEqual(options["length_penalty"], 1.05)

    def test_long_audio_defaults_to_accuracy_validated_batch_size_two(self):
        with patch.dict(os.environ, {}, clear=True):
            config = build_runtime_config()

        self.assertEqual(config.long_audio_batch_size, 2)

    def test_whisper_production_defaults_match_validated_run(self):
        with patch.dict(os.environ, {}, clear=True):
            config = build_runtime_config()

        self.assertEqual(config.backend, "faster-whisper")
        self.assertEqual(config.model_name, "large-v3")
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.compute_type, "float16")
        self.assertEqual(config.beam_size, 4)
        self.assertEqual(config.patience, 1.0)
        self.assertEqual(config.length_penalty, 1.0)
        self.assertTrue(config.vad_filter)
        self.assertEqual(config.vad_threshold, 0.30)
        self.assertEqual(config.vad_min_silence_ms, 1000)
        self.assertEqual(config.vad_speech_pad_ms, 400)
        self.assertEqual(config.log_prob_threshold, -1.0)
        self.assertEqual(config.no_speech_threshold, 0.3)
        self.assertEqual(config.compression_ratio_threshold, 2.4)
        self.assertEqual(config.batch_threshold_seconds, 60.0)
        self.assertEqual(config.long_audio_batch_size, 2)
        self.assertTrue(config.cpu_fallback_enabled)
        self.assertEqual(config.cpu_fallback_compute_type, "int8")
        self.assertTrue(config.variant_conversion_enabled)
        self.assertEqual(config.variant_source, "en_US")
        self.assertEqual(config.variant_target, "en_GB")

    def test_low_frequency_filter_defaults_are_enabled_and_configurable(self):
        with patch.dict(os.environ, {}, clear=True):
            config = build_runtime_config()

        self.assertTrue(config.low_frequency_filter_enabled)
        self.assertEqual(config.low_frequency_filter_cutoff_hz, 80.0)
        self.assertEqual(config.low_frequency_filter_ratio_threshold, 0.25)
        self.assertEqual(config.low_frequency_filter_absolute_dbfs, -35.0)
        self.assertEqual(config.low_frequency_filter_dc_offset_threshold, 0.005)

        with patch.dict(
            os.environ,
            {
                "WHISPER_LOW_FREQUENCY_FILTER": "false",
                "WHISPER_LOW_FREQUENCY_FILTER_CUTOFF_HZ": "100",
                "WHISPER_LOW_FREQUENCY_FILTER_RATIO_THRESHOLD": "0.4",
                "WHISPER_LOW_FREQUENCY_FILTER_ABSOLUTE_DBFS": "-30",
                "WHISPER_LOW_FREQUENCY_FILTER_DC_OFFSET_THRESHOLD": "0.01",
            },
            clear=True,
        ):
            configured = build_runtime_config()

        self.assertFalse(configured.low_frequency_filter_enabled)
        self.assertEqual(configured.low_frequency_filter_cutoff_hz, 100.0)
        self.assertEqual(configured.low_frequency_filter_ratio_threshold, 0.4)
        self.assertEqual(configured.low_frequency_filter_absolute_dbfs, -30.0)
        self.assertEqual(configured.low_frequency_filter_dc_offset_threshold, 0.01)

    def test_long_audio_uses_batched_gpu(self):
        transcriber = self.make_transcriber()
        transcriber._transcribe_batched = Mock(return_value="batched")
        transcriber._transcribe_with_model = Mock(return_value="ordinary")

        self.assertEqual(transcriber.transcribe_file("audio.wav"), "batched")
        self.assertEqual(format_transcription_device(transcriber), "GPU")
        transcriber._transcribe_batched.assert_called_once_with(transcriber.model, "audio.wav")
        transcriber._transcribe_with_model.assert_not_called()

    def test_audio_at_threshold_uses_batched_gpu(self):
        transcriber = self.make_transcriber(duration_seconds=60.0)
        transcriber._transcribe_batched = Mock(return_value="batched")
        transcriber._transcribe_with_model = Mock(return_value="ordinary")

        self.assertEqual(transcriber.transcribe_file("audio.wav"), "batched")
        transcriber._transcribe_batched.assert_called_once_with(transcriber.model, "audio.wav")
        transcriber._transcribe_with_model.assert_not_called()

    def test_audio_below_threshold_uses_ordinary_gpu(self):
        transcriber = self.make_transcriber(duration_seconds=59.99)
        transcriber._transcribe_batched = Mock(return_value="batched")
        transcriber._transcribe_with_model = Mock(return_value="ordinary")

        self.assertEqual(transcriber.transcribe_file("audio.wav"), "ordinary")
        transcriber._transcribe_batched.assert_not_called()
        transcriber._transcribe_with_model.assert_called_once_with(transcriber.model, "audio.wav")

    def test_batched_failure_retries_ordinary_gpu_without_unloading_model(self):
        transcriber = self.make_transcriber()
        transcriber._transcribe_batched = Mock(side_effect=RuntimeError("CUDA out of memory"))
        transcriber._transcribe_with_model = Mock(return_value="ordinary")

        self.assertEqual(transcriber.transcribe_file("audio.wav"), "ordinary")
        self.assertEqual(format_transcription_device(transcriber), "GPU")
        transcriber._release_cuda_cache.assert_called_once_with()
        transcriber._invalidate_primary_model.assert_not_called()
        transcriber._get_cpu_fallback_model.assert_not_called()

    def test_batched_and_ordinary_oom_then_retry_cpu(self):
        transcriber = self.make_transcriber()
        transcriber._transcribe_batched = Mock(side_effect=RuntimeError("CUDA out of memory"))
        transcriber._transcribe_with_model = Mock(
            side_effect=[RuntimeError("CUDA out of memory"), "cpu result"]
        )

        self.assertEqual(transcriber.transcribe_file("audio.wav"), "cpu result")
        self.assertEqual(format_transcription_device(transcriber), "CPU(fallback)")
        transcriber._invalidate_primary_model.assert_called_once()
        transcriber._get_cpu_fallback_model.assert_called_once_with()
        self.assertEqual(
            transcriber._transcribe_with_model.call_args_list[1].args,
            ("cpu-model", "audio.wav"),
        )

        transcriber._transcribe_batched = Mock(return_value="gpu result")
        self.assertEqual(transcriber.transcribe_file("next.wav"), "gpu result")
        self.assertEqual(format_transcription_device(transcriber), "GPU")

    def test_device_reporting_uses_the_resolved_model_device(self):
        for resolved_device, expected in (("cpu", "CPU"), ("cuda", "GPU")):
            with self.subTest(device=resolved_device):
                transcriber = self.make_transcriber(duration_seconds=1)
                transcriber.config.device = "auto"
                transcriber._get_primary_model.return_value = SimpleNamespace(
                    model=SimpleNamespace(device=resolved_device)
                )
                transcriber._transcribe_with_model = Mock(return_value="result")

                self.assertEqual(transcriber.transcribe_file("audio.wav"), "result")
                self.assertEqual(format_transcription_device(transcriber), expected)

    def test_duration_probe_reads_wav_container_metadata(self):
        transcriber = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "sample.wav"
            with wave.open(str(audio_path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16_000)
                output.writeframes(b"\0\0" * 24_000)

            self.assertAlmostEqual(
                transcriber._audio_duration_seconds(str(audio_path)),
                1.5,
                places=2,
            )

    def test_segment_metadata_is_retained_for_adaptive_consensus(self):
        transcriber = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
        segments = [
            SimpleNamespace(
                start=1.0,
                end=2.5,
                text=" Hello world.",
                avg_logprob=-0.25,
                compression_ratio=1.1,
                no_speech_prob=0.02,
            )
        ]

        self.assertEqual(transcriber._consume_segments(segments), "Hello world.")
        self.assertEqual(
            transcriber.last_segment_metadata,
            [
                {
                    "start": 1.0,
                    "end": 2.5,
                    "text": " Hello world.",
                    "avg_logprob": -0.25,
                    "compression_ratio": 1.1,
                    "no_speech_prob": 0.02,
                }
            ],
        )


class ParakeetBackendTests(unittest.TestCase):
    def test_parakeet_has_expected_default_model(self):
        self.assertEqual(
            default_model_for_backend("parakeet"),
            "nvidia/parakeet-tdt-0.6b-v2",
        )

    def test_parakeet_presets_expose_requested_modes(self):
        programming = PARAKEET_PRESETS["programming"]
        conversation = PARAKEET_PRESETS["rustly-pocket-conversation"]
        accuracy = PARAKEET_PRESETS["accuracy"]
        beam = PARAKEET_PRESETS["accuracy-beam-experimental"]

        self.assertEqual(programming.decoder_strategy, "greedy_batch")
        self.assertEqual(programming.max_symbols_per_step, 10)
        self.assertTrue(programming.loop_labels)
        self.assertTrue(programming.use_cuda_graph_decoder)
        self.assertIsNone(programming.vad_threshold)
        self.assertTrue(programming.phrase_boosting)
        self.assertEqual(programming.boosting_tree_alpha, 1.5)
        self.assertEqual(conversation.vad_threshold, 0.30)
        self.assertEqual(conversation.min_speech_duration_ms, 250)
        self.assertEqual(conversation.min_silence_duration_ms, 600)
        self.assertEqual(conversation.speech_pad_ms, 450)
        self.assertTrue(conversation.word_timestamps)
        self.assertTrue(conversation.word_confidence)
        self.assertFalse(conversation.phrase_boosting)
        self.assertEqual(accuracy.decoder_strategy, "greedy_batch")
        self.assertEqual(accuracy.beam_size, 1)
        self.assertTrue(accuracy.loop_labels)
        self.assertTrue(accuracy.use_cuda_graph_decoder)
        self.assertEqual(accuracy.max_segment_seconds, 120.0)
        self.assertEqual(accuracy.forced_split_overlap_seconds, 0.0)
        self.assertTrue(accuracy.preserve_internal_silence)
        self.assertEqual(accuracy.vad_threshold, 0.30)
        self.assertEqual(accuracy.min_silence_duration_ms, 700)
        self.assertEqual(accuracy.speech_pad_ms, 400)
        self.assertTrue(accuracy.word_timestamps)
        self.assertTrue(accuracy.word_confidence)
        self.assertTrue(accuracy.phrase_boosting)
        self.assertEqual(accuracy.boosting_tree_alpha, 0.5)
        self.assertEqual(beam.decoder_strategy, "malsd_batch")
        self.assertEqual(beam.beam_size, 5)
        self.assertTrue(beam.allow_cuda_graphs)
        self.assertFalse(beam.word_confidence)

    def test_accuracy_preset_builds_greedy_decoder_configuration(self):
        model = Mock()
        transcriber = ParakeetTranscriber.__new__(ParakeetTranscriber)
        transcriber.model = model
        transcriber.preset = PARAKEET_PRESETS["accuracy"]
        transcriber.torch = SimpleNamespace(inference_mode=nullcontext)

        with tempfile.NamedTemporaryFile() as phrases_file:
            transcriber.config = SimpleNamespace(
                parakeet_key_phrases_file=phrases_file.name,
            )
            transcriber._configure_decoding()

        decoding_config = model.change_decoding_strategy.call_args.args[0]
        self.assertEqual(decoding_config.strategy, "greedy_batch")
        self.assertEqual(decoding_config.greedy.max_symbols_per_step, 10)
        self.assertTrue(decoding_config.greedy.loop_labels)
        self.assertTrue(decoding_config.greedy.use_cuda_graph_decoder)
        self.assertEqual(
            decoding_config.greedy.boosting_tree.key_phrases_file,
            phrases_file.name,
        )
        self.assertEqual(decoding_config.greedy.boosting_tree_alpha, 0.5)
        self.assertTrue(decoding_config.confidence_cfg.preserve_frame_confidence)
        self.assertTrue(decoding_config.confidence_cfg.preserve_token_confidence)
        self.assertTrue(decoding_config.confidence_cfg.preserve_word_confidence)
        self.assertEqual(decoding_config.confidence_cfg.aggregation, "mean")
        self.assertEqual(decoding_config.confidence_cfg.method_cfg.name, "entropy")
        self.assertEqual(decoding_config.confidence_cfg.method_cfg.entropy_type, "tsallis")
        self.assertEqual(decoding_config.confidence_cfg.method_cfg.alpha, 0.33)
        self.assertEqual(decoding_config.confidence_cfg.method_cfg.entropy_norm, "exp")
        self.assertNotIn("beam", decoding_config)

    def test_experimental_beam_preset_keeps_beam_decoder(self):
        model = Mock()
        transcriber = ParakeetTranscriber.__new__(ParakeetTranscriber)
        transcriber.model = model
        transcriber.preset = PARAKEET_PRESETS["accuracy-beam-experimental"]

        with tempfile.NamedTemporaryFile() as phrases_file:
            transcriber.config = SimpleNamespace(
                parakeet_key_phrases_file=phrases_file.name,
            )
            transcriber._configure_decoding()

        decoding_config = model.change_decoding_strategy.call_args.args[0]
        self.assertEqual(decoding_config.strategy, "malsd_batch")
        self.assertEqual(decoding_config.beam.beam_size, 5)
        self.assertEqual(decoding_config.beam.max_symbols_per_step, 10)
        self.assertTrue(decoding_config.beam.allow_cuda_graphs)
        self.assertEqual(decoding_config.beam.boosting_tree_alpha, 0.5)

    def test_channel_selector_parser(self):
        self.assertEqual(parse_parakeet_channel_selector(None, 0), 0)
        self.assertEqual(parse_parakeet_channel_selector("1", 0), 1)
        self.assertEqual(parse_parakeet_channel_selector("average", 0), "average")
        self.assertIsNone(parse_parakeet_channel_selector("none", 0))

    def test_whisper_hotwords_combine_inline_and_file_entries(self):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as hotwords_file:
            hotwords_file.write("Pydantic\n# ignored\nSQLAlchemy\n")
            hotwords_file.flush()

            hotwords = load_whisper_hotwords(
                "SQLAlchemy, async/await",
                hotwords_file.name,
            )

        self.assertEqual(hotwords, "SQLAlchemy, async/await, Pydantic")

    def test_whisper_hotwords_can_be_loaded_from_environment_file(self):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as hotwords_file:
            hotwords_file.write("SQLAlchemy\nPydantic\n")
            hotwords_file.flush()
            with patch.dict(
                os.environ,
                {"WHISPER_HOTWORDS_FILE": hotwords_file.name},
                clear=True,
            ):
                config = build_runtime_config()

        self.assertEqual(config.hotwords, "SQLAlchemy, Pydantic")

    def test_parakeet_backend_can_be_selected_from_environment(self):
        with patch.dict(os.environ, {"WHISPER_BACKEND": "parakeet"}, clear=True):
            config = build_runtime_config()

        self.assertEqual(config.backend, "parakeet")
        self.assertEqual(config.model_name, "nvidia/parakeet-tdt-0.6b-v2")

    def test_accuracy_preset_uses_repository_phrase_file_by_default(self):
        with patch.dict(
            os.environ,
            {"WHISPER_BACKEND": "parakeet", "WHISPER_PRESET": "accuracy"},
            clear=True,
        ):
            config = build_runtime_config()

        phrase_file = Path(config.parakeet_key_phrases_file)
        self.assertEqual(phrase_file.name, "parakeet_programming_phrases.txt")
        self.assertTrue(phrase_file.is_file())

    def test_parakeet_transcribes_one_file_without_progress_output(self):
        model = Mock()
        model.transcribe.return_value = [SimpleNamespace(text="A Parakeet result.")]
        transcriber = ParakeetTranscriber.__new__(ParakeetTranscriber)
        transcriber.model = model
        transcriber.preset = PARAKEET_PRESETS["programming"]
        transcriber.config = SimpleNamespace(parakeet_channel_selector=0)
        transcriber.device = "cpu"
        transcriber.torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            inference_mode=nullcontext,
        )
        transcriber._trim_cuda_cache = Mock()
        transcriber._log_cuda_memory = Mock()

        self.assertEqual(
            transcriber.transcribe_file("audio.wav"),
            "A Parakeet result.",
        )
        model.transcribe.assert_called_once_with(
            ["audio.wav"],
            batch_size=1,
            return_hypotheses=False,
            timestamps=False,
            channel_selector=0,
            verbose=False,
        )
        transcriber._trim_cuda_cache.assert_not_called()

    def test_parakeet_releases_cuda_cache_after_oom(self):
        model = Mock()
        model.transcribe.side_effect = RuntimeError("CUDA out of memory")
        transcriber = ParakeetTranscriber.__new__(ParakeetTranscriber)
        transcriber.model = model
        transcriber.preset = PARAKEET_PRESETS["programming"]
        transcriber.config = SimpleNamespace(parakeet_channel_selector=0)
        transcriber.torch = SimpleNamespace(inference_mode=nullcontext)
        transcriber._trim_cuda_cache = Mock()
        transcriber._log_cuda_memory = Mock()

        with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
            transcriber.transcribe_file("audio.wav")

        transcriber._trim_cuda_cache.assert_called_once_with()

    def test_parakeet_vad_audio_loader_does_not_require_torchcodec(self):
        import struct

        import torch

        transcriber = ParakeetTranscriber.__new__(ParakeetTranscriber)
        transcriber.config = SimpleNamespace(parakeet_channel_selector=1)
        transcriber.preset = PARAKEET_PRESETS["accuracy"]
        transcriber.torch = torch

        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            with wave.open(audio_file.name, "wb") as output:
                output.setnchannels(2)
                output.setsampwidth(2)
                output.setframerate(16_000)
                output.writeframes(struct.pack("<hhhh", 1_000, 2_000, 3_000, 4_000))

            waveform = transcriber._read_audio_for_vad(audio_file.name)

        self.assertEqual(tuple(waveform.shape), (2,))
        torch.testing.assert_close(
            waveform,
            torch.tensor([2_000 / 32_768, 4_000 / 32_768]),
        )

    def test_accuracy_segments_up_to_two_minutes_without_overlap(self):
        transcriber = ParakeetTranscriber.__new__(ParakeetTranscriber)
        transcriber.preset = PARAKEET_PRESETS["accuracy"]
        sample_rate = transcriber.preset.sample_rate

        segments = transcriber._segment_vad_timestamps(
            [
                {"start": 0, "end": 130 * sample_rate},
            ],
            130 * sample_rate,
        )
        self.assertEqual(
            segments,
            [
                (0, 120 * sample_rate, False),
                (120 * sample_rate, 130 * sample_rate, False),
            ],
        )

    def test_accuracy_vad_preserves_internal_silence_between_speech_islands(self):
        transcriber = ParakeetTranscriber.__new__(ParakeetTranscriber)
        transcriber.preset = PARAKEET_PRESETS["accuracy"]

        self.assertEqual(
            transcriber._vad_inference_timestamps(
                [
                    {"start": 1_000, "end": 10_000},
                    {"start": 20_000, "end": 30_000},
                ]
            ),
            [{"start": 1_000, "end": 30_000}],
        )

    def test_natural_vad_boundaries_do_not_deduplicate_repeated_words(self):
        self.assertEqual(
            ParakeetTranscriber._merge_segment_transcripts(
                [("clean", False), ("clean", False)]
            ),
            "clean clean",
        )
        self.assertEqual(
            ParakeetTranscriber._merge_segment_transcripts(
                [("worktrees clean", False), ("clean shfmt", True)]
            ),
            "worktrees clean shfmt",
        )

    def test_programming_normalization_canonicalizes_known_terms(self):
        self.assertEqual(
            normalize_technical_text(
                "origin slash main commit D89F D01 uses copy q, ASROC, NCT 6683, "
                "expined key files, MODPRO, SSH FMT, shell check and S C two zero eight eight."
            ),
            "origin/main commit D89FD01 uses CopyQ, ASRock, NCT6683, xbindkeys files, "
            "modprobe, shfmt, ShellCheck and SC2088.",
        )

    def test_shared_canonicalizer_applies_keyword_casing(self):
        self.assertEqual(
            canonicalize_technical_text(
                "copy queue fish udev nvidia shell check and origin slash main"
            ),
            "CopyQ Fish UDEV NVIDIA ShellCheck and origin/main",
        )

    def test_variant_protection_round_trips_identifiers(self):
        text = "color behavior normalize src/color.rs --color D89FD01"
        protected, replacements = protect_technical_terms(text)

        self.assertNotIn("src/color.rs", protected)
        self.assertNotIn("--color", protected)
        self.assertNotIn("D89FD01", protected)
        self.assertEqual(restore_technical_terms(protected, replacements), text)

    def test_word_diagnostics_accept_nemo_timestep_field(self):
        transcriber = ParakeetTranscriber.__new__(ParakeetTranscriber)
        transcriber.preset = PARAKEET_PRESETS["accuracy"]
        hypothesis = SimpleNamespace(
            text="ASRock",
            timestep={"word": [{"word": "ASRock", "start": 0.1, "end": 0.5}]},
            word_confidence=[0.75],
        )

        with patch("assistant_transcription.backends.parakeet.print") as log:
            transcriber._log_word_diagnostics([hypothesis])

        self.assertIn(
            '"word":"ASRock","start":0.1,"end":0.5,"confidence":0.75',
            log.call_args.args[0],
        )

    def test_nemo_result_normalizer_handles_text_and_nested_batches(self):
        self.assertEqual(first_nemo_transcription_text("text"), "text")
        self.assertEqual(
            first_nemo_transcription_text(([SimpleNamespace(text="nested")], [])),
            "nested",
        )


class GraniteSpeechBackendTests(unittest.TestCase):
    def test_keyword_casing_uses_curated_canonical_forms(self):
        self.assertEqual(
            preserve_keyword_casing(
                "commit d89fd01 used fish firefox udev nvidia dkms and bootstrap.sh",
                ["D89FD01", "Fish", "Firefox", "UDEV", "NVIDIA", "DKMS", "bootstrap.sh"],
            ),
            "commit D89FD01 used Fish Firefox UDEV NVIDIA DKMS and bootstrap.sh",
        )

    def test_granite_has_expected_default_model(self):
        self.assertEqual(
            default_model_for_backend("granite-speech"),
            "ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0",
        )

    def test_granite_backend_can_be_selected_from_environment(self):
        with patch.dict(os.environ, {"WHISPER_BACKEND": "granite-speech"}, clear=True):
            config = build_runtime_config()

        self.assertEqual(config.backend, "granite-speech")
        self.assertEqual(
            config.model_name,
            "ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0",
        )
        self.assertEqual(config.granite_max_new_tokens, 512)
        self.assertEqual(config.granite_temperature, 0.0)
        self.assertEqual(config.granite_server_port, 9797)
        self.assertEqual(config.granite_context_size, 4096)
        self.assertTrue(config.parakeet_key_phrases_file.endswith("granite_programming_keywords.txt"))

    def test_granite_prompt_includes_keyword_file(self):
        transcriber = GraniteSpeechTranscriber.__new__(GraniteSpeechTranscriber)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as phrases_file:
            phrases_file.write("ASROCK\n# ignored\nSC2088\n")
            phrases_file.flush()
            transcriber.config = SimpleNamespace(
                granite_prompt="transcribe the speech with proper punctuation.",
                parakeet_key_phrases_file=phrases_file.name,
            )

            prompt = transcriber._build_prompt()

        self.assertEqual(
            prompt,
            "transcribe the speech with proper punctuation. "
            "Keywords: ASROCK, SC2088",
        )

    def test_granite_preprocessor_selects_channel_zero_and_writes_16khz_mono(self):
        import struct

        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            with wave.open(audio_file.name, "wb") as output:
                output.setnchannels(2)
                output.setsampwidth(2)
                output.setframerate(16_000)
                output.writeframes(struct.pack("<hhhh", 1_000, 3_000, 1_000, 3_000))

            with prepare_granite_audio(audio_file.name) as prepared_path:
                with wave.open(prepared_path, "rb") as prepared:
                    self.assertEqual(prepared.getnchannels(), 1)
                    self.assertEqual(prepared.getframerate(), 16_000)
                    self.assertEqual(prepared.getsampwidth(), 2)
                    samples = struct.unpack("<hh", prepared.readframes(2))

        self.assertEqual(samples, (1_000, 1_000))

    def test_granite_server_command_uses_q8_and_full_gpu_offload(self):
        transcriber = GraniteSpeechTranscriber.__new__(GraniteSpeechTranscriber)
        transcriber.model_name = "ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0"
        transcriber.config = SimpleNamespace(
            granite_server_host="127.0.0.1",
            granite_server_port=9797,
            granite_context_size=4096,
            device="cuda",
        )

        command = transcriber._server_command("/usr/bin/llama-server")

        self.assertEqual(command[0], "/usr/bin/llama-server")
        self.assertIn("ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0", command)
        self.assertEqual(command[command.index("--n-gpu-layers") + 1], "auto")
        self.assertEqual(command[command.index("--ctx-size") + 1], "4096")

    def test_granite_transcription_streams_audio_to_llama_server(self):
        response = Mock(is_error=False)
        response.json.return_value = {"text": "Granite Q8 result."}
        transcriber = GraniteSpeechTranscriber.__new__(GraniteSpeechTranscriber)
        transcriber.config = SimpleNamespace(
            granite_max_new_tokens=512,
            granite_temperature=0.0,
        )
        transcriber.model_name = "ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0"
        transcriber.base_url = "http://127.0.0.1:9797"
        transcriber.prompt = "transcribe the speech."
        transcriber.keywords = []
        transcriber.client = Mock()
        transcriber.client.post.return_value = response

        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            audio_file.write(b"audio")
            audio_file.flush()
            with patch(
                "assistant_transcription.backends.granite_speech.prepare_granite_audio",
                return_value=nullcontext(audio_file.name),
            ):
                result = transcriber.transcribe_file(audio_file.name)

        self.assertEqual(result, "Granite Q8 result.")
        post_kwargs = transcriber.client.post.call_args.kwargs
        self.assertEqual(post_kwargs["data"]["max_tokens"], "512")
        self.assertEqual(post_kwargs["data"]["temperature"], "0")
        self.assertEqual(post_kwargs["data"]["prompt"], "transcribe the speech.")


class TimingFormatTests(unittest.TestCase):
    def test_audio_size_is_shown_in_binary_units(self):
        self.assertEqual(
            format_audio_size(37_536),
            "audio_bytes=37536 (36.66 KiB, 0.036 MiB, 0.000035 GiB)",
        )

    def test_parenthesized_total_constituents_use_breakdown_colour(self):
        coloured = colour_timing_fields(
            "total_ms=694 (client_stop_to_upload_ms=92 + "
            "client_network_and_delivery_overhead_ms=6 + server_total_ms=531 + "
            "client_paste_ms=65)"
        )

        for value in ("92", "6", "531", "65"):
            self.assertIn(f"{ANSI_BREAKDOWN_VALUE}{value}", coloured)

    def test_transcription_timing_is_grouped_in_pipeline_order(self):
        timings = {
            "client_recording_duration_ms": 2789,
            "client_stop_finalize_ms": 62,
            "client_queue_wait_ms": 149,
            "client_file_read_ms": 0,
            "client_stop_to_upload_ms": 211,
            "transcriber_ready_ms": 0,
            "temp_file_open_ms": 0,
            "upload_body_read_ms": 0,
            "temp_file_close_ms": 0,
            "latest_recovery_copy_ms": 1,
            "server_transcribe_ms": 692,
            "postprocess_ms": 0,
            "temp_file_cleanup_ms": 0,
            "server_total_ms": 693,
            "audio_bytes": 88736,
            "failed_recovery_copy_ms": -1,
        }

        self.assertEqual(
            format_timing_fields("success", timings),
            "outcome=success server_total_ms=693 (transcriber_ready_ms=0 + "
            "temp_file_open_ms=0 + temp_file_close_ms=0 + latest_recovery_copy_ms=1 + "
            "server_transcribe_ms=692 + postprocess_ms=0 + temp_file_cleanup_ms=0)",
        )
        self.assertEqual(
            format_client_preparation(timings),
            "client_recording_duration_ms=2789 client_stop_to_upload_ms=211 "
            "[client_stop_finalize_ms=62 + client_queue_wait_ms=149 + "
            "client_file_read_ms=0]",
        )

    def test_failed_recovery_copy_only_appears_for_failed_event(self):
        timings = {
            "client_stop_finalize_ms": -1,
            "client_queue_wait_ms": -1,
            "client_file_read_ms": -1,
            "client_stop_to_upload_ms": -1,
            "transcriber_ready_ms": 0,
            "temp_file_open_ms": 0,
            "temp_file_close_ms": 0,
            "latest_recovery_copy_ms": 1,
            "server_transcribe_ms": -1,
            "postprocess_ms": -1,
            "temp_file_cleanup_ms": 0,
            "server_total_ms": 30,
            "failed_recovery_copy_ms": 4,
        }

        fields = format_timing_fields("failed", timings)

        self.assertTrue(fields.startswith("outcome=failed failed_recovery_copy_ms=4 "))

    def test_audio_preprocessing_is_in_server_timing_order(self):
        timings = {
            "transcriber_ready_ms": 0,
            "temp_file_open_ms": 0,
            "temp_file_close_ms": 0,
            "latest_recovery_copy_ms": 1,
            "audio_preprocess_ms": 12,
            "server_transcribe_ms": 692,
            "postprocess_ms": 0,
            "temp_file_cleanup_ms": 0,
            "server_total_ms": 705,
            "failed_recovery_copy_ms": -1,
        }

        self.assertEqual(
            format_timing_fields("success", timings),
            "outcome=success server_total_ms=705 (transcriber_ready_ms=0 + "
            "temp_file_open_ms=0 + temp_file_close_ms=0 + latest_recovery_copy_ms=1 + "
            "audio_preprocess_ms=12 + server_transcribe_ms=692 + postprocess_ms=0 + "
            "temp_file_cleanup_ms=0)",
        )


class TranscriptionLogTests(unittest.IsolatedAsyncioTestCase):
    async def render_request(
        self,
        *,
        device,
        fallback=False,
        fail=False,
        colour=False,
        body=b"audio",
        desktop=False,
        text="Test transcript.",
        delivery_payload=None,
    ):
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(
            {
                "type": "http",
                "headers": [(b"x-client-delivery-report", b"desktop")] if desktop else [],
                "client": ("127.0.0.1", 1234),
            },
            receive=receive,
        )
        transcriber = SimpleNamespace(
            last_inference_device=device,
            last_inference_used_cpu_fallback=fallback,
            transcribe_file=Mock(
                side_effect=RuntimeError("fixture failure") if fail else None,
                return_value=text,
            ),
        )
        prepared = SimpleNamespace(
            transcription_path="fixture.audio",
            analysis=None,
            filtered_analysis=None,
            filter_applied=False,
            filter_reason="disabled",
            error=None,
            as_dict=lambda: {},
        )
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.addFilter(FriendlyLogSource())
        formatter_config = build_log_config()["formatters"]["default"]
        handler.setFormatter(DefaultFormatter(
            fmt=formatter_config["fmt"],
            datefmt=formatter_config["datefmt"],
            use_colors=colour,
        ))
        logger = logging.Logger("assistant.whisper", level=logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(app.state, "runtime_config", build_runtime_config()),
            patch.object(app.state, "transcriber", transcriber),
            patch.object(app.state, "variant_post_processor", None),
            patch("whisper_service.ensure_transcriber_loaded"),
            patch("whisper_service.write_recovery_copy"),
            patch("whisper_service.preserve_failed_request"),
            patch("whisper_service.canonicalize_technical_text", side_effect=lambda text: text),
            patch("whisper_service.prepare_audio_for_transcription", return_value=nullcontext(prepared)),
            patch("assistant_transcription.log_output.service_logger", logger),
        ):
            response = await transcribe_raw(request)
            if delivery_payload is not None:
                await desktop_delivery_report(
                    SimpleNamespace(json=AsyncMock(return_value=delivery_payload))
                )
        return response, stream.getvalue(), transcriber

    async def test_result_device_and_blank_separator_survive_console_formatting(self):
        devices = (
            ("cuda", False, "GPU"),
            ("cpu", False, "CPU"),
            ("cpu", True, "CPU(fallback)"),
        )
        for device, fallback, label in devices:
            for colour in (False, True):
                with self.subTest(device=label, colour=colour):
                    response, output, _ = await self.render_request(
                        device=device, fallback=fallback, colour=colour
                    )
                    plain = re.sub(r"\x1b\[[0-9;]*m", "", output)
                    self.assertEqual(response["text"], "Test transcript.")
                    self.assertIn(f"outcome=success device={label}", plain)
                    self.assertIn("Server: Transcribed Raw", plain)
                    self.assertTrue(output.endswith("\n\n"), repr(output))
                    self.assertFalse(output.endswith("\n\n\n"))

    async def test_failed_request_reports_last_attempt_and_dividing_blank_line(self):
        response, output, _ = await self.render_request(device="cpu", fallback=True, fail=True)
        self.assertEqual(response.status_code, 500)
        self.assertIn("outcome=failed device=CPU(fallback)", output)
        self.assertTrue(output.endswith("\n\n"))

    async def test_rejected_upload_does_not_reuse_previous_inference_device(self):
        response, output, transcriber = await self.render_request(device="cuda", body=b"")
        self.assertEqual(response.status_code, 400)
        transcriber.transcribe_file.assert_not_called()
        self.assertIn("outcome=failed device=not-run", output)
        self.assertTrue(output.endswith("\n\n"))

    async def test_desktop_result_defers_blank_separator_until_delivery(self):
        for colour in (False, True):
            with self.subTest(colour=colour):
                response, output, _ = await self.render_request(
                    device="cuda", desktop=True, colour=colour,
                )
                plain = re.sub(r"\x1b\[[0-9;]*m", "", output)
                self.assertEqual(response["text"], "Test transcript.")
                self.assertIn("outcome=success device=GPU", plain)
                self.assertTrue(plain.endswith("\n"))
                self.assertNotIn("\n\n", plain)

    async def test_desktop_terminal_reports_close_event_after_delivery(self):
        details = (
            "total_ms=668 (client_stop_to_upload_ms=106 + "
            "client_network_and_delivery_overhead_ms=5 + "
            "server_total_ms=391 + client_paste_ms=166)"
        )
        for status in ("complete", "refused", "failed"):
            for colour in (False, True):
                with self.subTest(status=status, colour=colour):
                    _, output, _ = await self.render_request(
                        device="cuda", desktop=True, colour=colour,
                        delivery_payload={"status": status, "details": details},
                    )
                    plain = re.sub(r"\x1b\[[0-9;]*m", "", output)
                    self.assertIn(f"Client: Delivery {status}: {details}", plain)
                    self.assertLess(
                        plain.index("Server: Transcribed Raw"),
                        plain.index(f"Client: Delivery {status}"),
                    )
                    self.assertTrue(plain.endswith("\n\n"), repr(plain))
                    self.assertEqual(plain.count("\n\n"), 1)

    async def test_legacy_report_without_details_still_closes_desktop_event(self):
        _, output, _ = await self.render_request(
            device="cuda", desktop=True,
            delivery_payload={"event": "desktop-delivery-complete"},
        )
        self.assertTrue(output.endswith("Client: Delivery complete\n\n"))
        self.assertEqual(output.count("\n\n"), 1)

    async def test_desktop_empty_transcript_closes_without_delivery_report(self):
        for text in ("", " \t "):
            with self.subTest(text=text):
                response, output, _ = await self.render_request(
                    device="cuda", desktop=True, text=text,
                )
                self.assertEqual(response["text"], text)
                self.assertIn("outcome=success device=GPU", output)
                self.assertTrue(output.endswith("\n\n"))

    async def test_desktop_failed_transcription_closes_without_delivery_report(self):
        response, output, _ = await self.render_request(
            device="cuda", desktop=True, fail=True,
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn("outcome=failed device=GPU", output)
        self.assertTrue(output.endswith("\n\n"))

    async def test_desktop_rejected_upload_closes_without_delivery_report(self):
        response, output, transcriber = await self.render_request(
            device="cuda", desktop=True, body=b"",
        )
        self.assertEqual(response.status_code, 400)
        transcriber.transcribe_file.assert_not_called()
        self.assertTrue(output.endswith("\n\n"))


class RecoveryRetentionTests(unittest.TestCase):
    def setUp(self):
        self.original_config = app.state.runtime_config

    def tearDown(self):
        app.state.runtime_config = self.original_config

    def test_recovery_limits_have_documented_defaults_and_environment_overrides(self):
        with patch.dict(os.environ, {}, clear=True):
            defaults = build_runtime_config()
        with patch.dict(
            os.environ,
            {
                "WHISPER_RECOVERY_REQUEST_LIMIT": "3",
                "WHISPER_FAILED_RECOVERY_LIMIT": "7",
            },
            clear=True,
        ):
            overridden = build_runtime_config()

        self.assertEqual(defaults.recovery_request_limit, 20)
        self.assertEqual(defaults.failed_recovery_limit, 20)
        self.assertEqual(overridden.recovery_request_limit, 3)
        self.assertEqual(overridden.failed_recovery_limit, 7)

    def test_recent_requests_rotate_and_keep_five_recordings(self):
        with tempfile.TemporaryDirectory() as recovery_dir:
            app.state.runtime_config = SimpleNamespace(
                recovery_dir=recovery_dir,
                recovery_request_limit=5,
                failed_recovery_limit=20,
            )
            source = Path(recovery_dir) / "source.wav"

            for number in range(6):
                source.write_bytes(f"recording-{number}".encode())
                write_recovery_copy(source)

            directory = Path(recovery_dir)
            self.assertEqual((directory / "latest_request.wav").read_bytes(), b"recording-5")
            for slot, expected in enumerate(range(4, 0, -1), start=1):
                self.assertEqual(
                    (directory / f"recent_request_{slot}.wav").read_bytes(),
                    f"recording-{expected}".encode(),
                )
                self.assertTrue((directory / f"recent_request_{slot}.meta").is_file())
            self.assertFalse((directory / "recent_request_5.wav").exists())

    def test_failed_requests_are_capped_at_twenty_pairs(self):
        with tempfile.TemporaryDirectory() as recovery_dir:
            app.state.runtime_config = SimpleNamespace(
                recovery_dir=recovery_dir,
                recovery_request_limit=5,
                failed_recovery_limit=20,
            )
            source = Path(recovery_dir) / "source.wav"
            source.write_bytes(b"failed recording")

            for number in range(22):
                preserve_failed_request(source.as_posix(), RuntimeError(f"failure {number}"))

            directory = Path(recovery_dir)
            self.assertEqual(len(list(directory.glob("failed_request_*.wav"))), 20)
            self.assertEqual(len(list(directory.glob("failed_request_*.txt"))), 20)

    def test_zero_limits_disable_recovery_copies(self):
        with tempfile.TemporaryDirectory() as recovery_dir:
            app.state.runtime_config = SimpleNamespace(
                recovery_dir=recovery_dir,
                recovery_request_limit=0,
                failed_recovery_limit=0,
            )
            source = Path(recovery_dir) / "source.wav"
            source.write_bytes(b"private recording")

            self.assertIsNone(write_recovery_copy(source))
            preserve_failed_request(source.as_posix(), RuntimeError("failure"))

            self.assertFalse((Path(recovery_dir) / "latest_request.wav").exists())
            self.assertEqual(list(Path(recovery_dir).glob("failed_request_*.*")), [])


class FriendlyLogSourceTests(unittest.TestCase):
    def test_log_config_registers_source_filter_on_both_handlers(self):
        config = build_log_config()

        self.assertIn("friendly_source", config["filters"])
        self.assertEqual(config["handlers"]["default"]["filters"], ["friendly_source"])
        self.assertEqual(config["handlers"]["access"]["filters"], ["friendly_source"])

    def test_server_records_are_labelled_server(self):
        source_filter = FriendlyLogSource()
        logger_names = ("assistant.whisper", "uvicorn.access", "uvicorn.error")

        for logger_name in logger_names:
            with self.subTest(logger_name=logger_name):
                record = logging.LogRecord(
                    logger_name,
                    logging.INFO,
                    __file__,
                    1,
                    "message",
                    (),
                    None,
                )
                self.assertTrue(source_filter.filter(record))
                self.assertEqual(record.source, "Server")

    def test_delivery_report_is_labelled_client(self):
        record = logging.LogRecord(
            "assistant.whisper",
            logging.INFO,
            __file__,
            1,
            "Delivery complete: total_ms=653",
            (),
            None,
        )

        self.assertTrue(FriendlyLogSource().filter(record))
        self.assertEqual(record.source, "Client")

    def test_recording_preparation_is_labelled_client(self):
        record = logging.LogRecord(
            "assistant.whisper",
            logging.INFO,
            __file__,
            1,
            "Recording prepared: client_stop_to_upload_ms=92",
            (),
            None,
        )

        self.assertTrue(FriendlyLogSource().filter(record))
        self.assertEqual(record.source, "Client")

    def test_application_startup_complete_adds_blank_separator_line(self):
        record = logging.LogRecord(
            "uvicorn.error",
            logging.INFO,
            __file__,
            1,
            "Application startup complete.",
            (),
            None,
        )
        record.color_message = "Application startup complete."

        source_filter = FriendlyLogSource()
        self.assertTrue(source_filter.filter(record))
        self.assertEqual(record.source, "Server")
        self.assertEqual(record.getMessage(), "Application startup complete.\n")
        self.assertEqual(record.color_message, "Application startup complete.\n")
        # Idempotent if filtered more than once
        self.assertTrue(source_filter.filter(record))
        self.assertEqual(record.getMessage(), "Application startup complete.\n")


if __name__ == "__main__":
    unittest.main()
