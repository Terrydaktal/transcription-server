"""Tests for AsyncPriorityLock, GET /health, and POST /transcribe_segments."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from assistant_transcription import server
from assistant_transcription.backends.faster_whisper import FasterWhisperTranscriber
from assistant_transcription.config import build_runtime_config
from assistant_transcription.priority_lock import (
    PRIORITY_BATCH_SEGMENTS,
    PRIORITY_LIVE_DICTATION,
    AsyncPriorityLock,
)
from assistant_transcription.segments import (
    SegmentTranscriptionResult,
    transcribe_with_segments,
)


class _StubSegmentTranscriber:
    device = "cuda"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.last_inference_device = "cuda"
        self.last_inference_used_cpu_fallback = False

    def transcribe_file(self, audio_path: str) -> str:
        self.calls.append(("file", {"path": audio_path}))
        return "colorize api"

    def transcribe_segments(self, audio_path: str, options=None) -> SegmentTranscriptionResult:
        opts = dict(options or {})
        self.calls.append(("segments", opts))
        return SegmentTranscriptionResult(
            segments=[
                {"start": 0.0, "end": 1.5, "text": " colorize api"},
                {"start": 1.5, "end": 3.0, "text": " second phrase"},
            ],
            text="colorize api second phrase",
            language=opts.get("language") or "en",
            language_probability=0.98,
            timestamp_source="native",
        )


class _StubFlatOnlyTranscriber:
    device = "cuda"
    last_inference_device = "cuda"
    last_inference_used_cpu_fallback = False

    def transcribe_file(self, audio_path: str) -> str:
        return "plain flat text"


class FasterWhisperSegmentTests(unittest.TestCase):
    def test_batched_segments_enable_timestamp_tokens_without_changing_dictation(self) -> None:
        config = replace(
            build_runtime_config(), device="cuda", long_audio_batch_size=16,
            batch_threshold_seconds=60.0,
        )
        with mock.patch("faster_whisper.WhisperModel"), mock.patch(
            "faster_whisper.BatchedInferencePipeline"
        ) as pipeline:
            pipeline.return_value.transcribe.return_value = ([], None)
            transcriber = FasterWhisperTranscriber(config)
            with mock.patch.object(
                transcriber, "_audio_duration_seconds", return_value=360.0
            ):
                transcriber.transcribe_segments("audio.flac", {"batched": True})
                self.assertIs(
                    pipeline.return_value.transcribe.call_args.kwargs.get("without_timestamps"),
                    False,
                )
                transcriber.transcribe_file("audio.flac")
                self.assertEqual(pipeline.return_value.transcribe.call_count, 2)
                self.assertNotIn(
                    "without_timestamps", pipeline.return_value.transcribe.call_args.kwargs,
                )


class PriorityLockTests(unittest.TestCase):
    def test_live_dictation_preempts_queued_batch_waiters(self) -> None:
        async def run() -> list[str]:
            lock = AsyncPriorityLock()
            order: list[str] = []

            async with lock.hold(priority=PRIORITY_BATCH_SEGMENTS):
                order.append("holder-start")

                async def batch_waiter(name: str) -> None:
                    async with lock.hold(priority=PRIORITY_BATCH_SEGMENTS):
                        order.append(name)

                async def live_waiter(name: str) -> None:
                    async with lock.hold(priority=PRIORITY_LIVE_DICTATION):
                        order.append(name)

                t_b1 = asyncio.create_task(batch_waiter("batch-1"))
                t_b2 = asyncio.create_task(batch_waiter("batch-2"))
                await asyncio.sleep(0)
                t_live = asyncio.create_task(live_waiter("live-dictation"))
                await asyncio.sleep(0)
                order.append("holder-end")

            await asyncio.gather(t_b1, t_b2, t_live)
            return order

        order = asyncio.run(run())
        self.assertEqual(
            order,
            ["holder-start", "holder-end", "live-dictation", "batch-1", "batch-2"],
        )


class TranscribeSegmentsEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_config = getattr(server.app.state, "runtime_config", None)
        self._old_transcriber = getattr(server.app.state, "transcriber", None)
        self._old_variant = getattr(server.app.state, "variant_post_processor", None)
        self._old_lock = getattr(server.app.state, "gpu_lock", None)

        self.stub = _StubSegmentTranscriber()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.config = replace(
            build_runtime_config(),
            backend="faster-whisper",
            model_name="large-v3",
            recovery_dir=self.tmpdir.name,
        )
        server.app.state.runtime_config = self.config
        server.app.state.transcriber = self.stub
        server.app.state.variant_post_processor = None
        server.app.state.gpu_lock = AsyncPriorityLock()
        self.client = TestClient(server.app)

    def tearDown(self) -> None:
        server.app.state.runtime_config = self._old_config
        server.app.state.transcriber = self._old_transcriber
        server.app.state.variant_post_processor = self._old_variant
        server.app.state.gpu_lock = self._old_lock
        self.tmpdir.cleanup()

    def test_health_reports_endpoints_and_model(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["backend"], "faster-whisper")
        self.assertEqual(body["model"], "large-v3")
        self.assertIn("/transcribe_segments", body["endpoints"])
        self.assertIn("/transcribe_raw", body["endpoints"])
        self.assertEqual(body["active_transcriptions"], 0)
        self.assertEqual(body["gpu_waiters"], 0)
        self.assertFalse(body["gpu_busy"])

    def test_transcribe_segments_envelope_forwards_options_and_skips_recovery(self) -> None:
        options = {
            "language": "en",
            "beam_size": 3,
            "vad_filter": False,
            "clip_timestamps": [0.0, 2.0, 2.0, 4.0],
            "batched": True,
            "batch_size": 8,
        }
        hdr = json.dumps(options).encode("utf-8")
        envelope = struct.pack(">I", len(hdr)) + hdr + b"RIFFfakewavbytes"

        resp = self.client.post(
            "/transcribe_segments",
            content=envelope,
            headers={"content-type": server.SEGMENTS_ENVELOPE_MIME},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(len(body["segments"]), 2)
        self.assertEqual(body["segments"][0]["start"], 0.0)
        self.assertEqual(body["segments"][0]["end"], 1.5)
        self.assertEqual(body["segments"][0]["text"], " colorize api")
        self.assertEqual(body["timestamp_source"], "native")
        self.assertEqual(body["backend"], "faster-whisper")
        self.assertEqual(body["model"], "large-v3")

        # Verify decode options arrived intact and no recovery file was written
        self.assertEqual(self.stub.calls[0][0], "segments")
        self.assertEqual(self.stub.calls[0][1]["clip_timestamps"], [0.0, 2.0, 2.0, 4.0])
        self.assertEqual(self.stub.calls[0][1]["beam_size"], 3)
        self.assertEqual(list(Path(self.tmpdir.name).iterdir()), [])

    def test_transcribe_segments_raw_body_with_query_params(self) -> None:
        resp = self.client.post(
            "/transcribe_segments?language=fr&beam_size=2&vad_filter=false",
            content=b"OggSfakeopus",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.stub.calls[0][1]["language"], "fr")
        self.assertEqual(self.stub.calls[0][1]["beam_size"], 2)
        self.assertFalse(self.stub.calls[0][1]["vad_filter"])

    def test_transcribe_segments_empty_body_returns_400(self) -> None:
        resp = self.client.post("/transcribe_segments", content=b"")
        self.assertEqual(resp.status_code, 400)

    def test_non_timestamped_backend_falls_back_cleanly(self) -> None:
        stub = _StubFlatOnlyTranscriber()
        result = transcribe_with_segments(
            stub,
            "nonexistent.wav",
            options={"vad_chunks_fallback": False, "language": "en"},
        )
        self.assertEqual(result.timestamp_source, "single_segment")
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0]["text"], "plain flat text")


if __name__ == "__main__":
    unittest.main()
