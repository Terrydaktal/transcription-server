"""Segment-level transcription helpers across all transcription-server backends."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import tempfile
from typing import Any
import wave


@dataclass(frozen=True)
class SegmentTranscriptionResult:
    segments: list[dict[str, Any]]
    text: str
    language: str | None = None
    language_probability: float | None = None
    timestamp_source: str = "native"
    metadata: dict[str, Any] = field(default_factory=dict)


def probe_audio_duration_seconds(audio_path: str) -> float | None:
    try:
        import av

        with av.open(audio_path) as container:
            if container.duration is not None:
                return float(container.duration / av.time_base)
            for stream in container.streams.audio:
                if stream.duration is not None and stream.time_base is not None:
                    return float(stream.duration * stream.time_base)
    except Exception:
        pass
    try:
        with wave.open(audio_path, "rb") as wf:
            rate = wf.getframerate()
            if rate > 0:
                return float(wf.getnframes()) / float(rate)
    except Exception:
        pass
    return None


def _normalize_segments(raw_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
        except (TypeError, ValueError):
            continue
        text = str(seg.get("text", ""))
        item: dict[str, Any] = {
            "start": max(0.0, start),
            "end": max(start, end),
            "text": text,
        }
        for extra_key in ("avg_logprob", "compression_ratio", "no_speech_prob"):
            if extra_key in seg and seg[extra_key] is not None:
                try:
                    item[extra_key] = float(seg[extra_key])
                except (TypeError, ValueError):
                    pass
        normalized.append(item)
    return normalized


def _transcribe_via_vad_chunks(
    transcriber: Any,
    audio_path: str,
    options: dict[str, Any],
) -> SegmentTranscriptionResult | None:
    """Fallback for backends without native timestamps: slice with Silero VAD."""
    if options.get("vad_filter") is False:
        return None
    try:
        import av
        import numpy as np
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
    except Exception:
        return None

    sample_rate = 16000
    try:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        chunks: list[np.ndarray] = []
        with av.open(audio_path) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                return SegmentTranscriptionResult(
                    segments=[], text="", timestamp_source="vad_chunks"
                )
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    arr = resampled.to_ndarray()
                    chunks.append(arr.reshape(-1))
        if not chunks:
            return SegmentTranscriptionResult(
                segments=[], text="", timestamp_source="vad_chunks"
            )
        pcm = np.concatenate(chunks).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(pcm)
    except Exception:
        return None

    vad_threshold = float(options.get("vad_threshold", 0.5))
    min_silence_ms = int(options.get("vad_min_silence_ms", 1000))
    speech_pad_ms = int(options.get("vad_speech_pad_ms", 400))
    max_chunk_s = float(options.get("chunk_length", 30) or 30)

    vad_model = load_silero_vad()
    raw_ts = get_speech_timestamps(
        audio_tensor,
        vad_model,
        threshold=vad_threshold,
        sampling_rate=sample_rate,
        min_speech_duration_ms=150,
        max_speech_duration_s=max(5.0, max_chunk_s),
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )
    if not raw_ts:
        return SegmentTranscriptionResult(
            segments=[],
            text="",
            language=options.get("language"),
            language_probability=1.0 if options.get("language") else None,
            timestamp_source="vad_chunks",
        )

    segments: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for item in raw_ts:
        start_idx = max(0, int(item["start"]))
        end_idx = min(len(pcm), int(item["end"]))
        if end_idx <= start_idx:
            continue
        seg_pcm = np.clip(pcm[start_idx:end_idx], -1.0, 1.0)
        seg_i16 = (seg_pcm * 32767.0).astype("<i2").tobytes()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            seg_path = tmp.name
        try:
            with wave.open(seg_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(seg_i16)
            seg_text = str(transcriber.transcribe_file(seg_path) or "").strip()
        finally:
            if os.path.exists(seg_path):
                os.remove(seg_path)
        if seg_text:
            segments.append(
                {
                    "start": round(start_idx / sample_rate, 3),
                    "end": round(end_idx / sample_rate, 3),
                    "text": seg_text,
                }
            )
            text_parts.append(seg_text)

    return SegmentTranscriptionResult(
        segments=segments,
        text=" ".join(text_parts).strip(),
        language=options.get("language"),
        language_probability=1.0 if options.get("language") else None,
        timestamp_source="vad_chunks",
    )


def transcribe_with_segments(
    transcriber: Any,
    audio_path: str,
    options: dict[str, Any] | None = None,
) -> SegmentTranscriptionResult:
    """Transcribe an audio file and return timestamped segments for any backend."""
    opts = dict(options or {})
    if hasattr(transcriber, "transcribe_segments"):
        result = transcriber.transcribe_segments(audio_path, options=opts)
        if isinstance(result, SegmentTranscriptionResult):
            return SegmentTranscriptionResult(
                segments=_normalize_segments(result.segments),
                text=result.text,
                language=result.language,
                language_probability=result.language_probability,
                timestamp_source=result.timestamp_source,
                metadata=dict(result.metadata),
            )

    # Try VAD chunking for non-timestamped backends only if they don't already populate
    # last_segment_metadata on transcribe_file (e.g., granite-speech, canary-qwen).
    if opts.get("vad_chunks_fallback", True) and not hasattr(transcriber, "last_segment_metadata"):
        vad_result = _transcribe_via_vad_chunks(transcriber, audio_path, opts)
        if vad_result is not None:
            return vad_result

    text = str(transcriber.transcribe_file(audio_path) or "").strip()
    meta = getattr(transcriber, "last_segment_metadata", None)
    if isinstance(meta, list) and meta:
        return SegmentTranscriptionResult(
            segments=_normalize_segments(meta),
            text=text,
            language=opts.get("language"),
            language_probability=1.0 if opts.get("language") else None,
            timestamp_source="native",
        )

    duration = probe_audio_duration_seconds(audio_path)
    end_s = float(duration) if duration is not None and duration > 0 else 0.0
    segments = [{"start": 0.0, "end": end_s, "text": text}] if text else []
    return SegmentTranscriptionResult(
        segments=segments,
        text=text,
        language=opts.get("language"),
        language_probability=1.0 if opts.get("language") else None,
        timestamp_source="single_segment",
    )
