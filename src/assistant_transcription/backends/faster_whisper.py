import gc
from typing import Any

from ..config import RuntimeConfig
from ..log_output import service_log as print
from ..segments import SegmentTranscriptionResult


def _split_clip_intervals(
    intervals: list[tuple[float, float]], max_len: float
) -> list[tuple[float, float]]:
    if max_len <= 0:
        return intervals
    out: list[tuple[float, float]] = []
    for start, end in intervals:
        cur = float(start)
        end_f = float(end)
        if end_f <= cur:
            continue
        while end_f - cur > max_len:
            out.append((cur, cur + max_len))
            cur += max_len
        if end_f > cur:
            out.append((cur, end_f))
    return out


class FasterWhisperTranscriber:
    def __init__(self, config: RuntimeConfig) -> None:
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        self.config = config
        self._whisper_model_cls = WhisperModel
        self._batched_pipeline_cls = BatchedInferencePipeline
        self.model = None
        self._batched_pipeline = None
        self._cpu_fallback_model = None
        self._gpu_model_needs_reload = False
        self.last_inference_device: str | None = None
        self.last_inference_used_cpu_fallback = False
        self.last_segment_metadata: list[dict[str, object]] = []
        self.last_language: str | None = None
        self.last_language_probability: float | None = None
        self._load_primary_model()

    def transcribe_file(self, audio_path: str) -> str:
        return self._run_transcription(audio_path, options=None)

    def transcribe_segments(
        self,
        audio_path: str,
        options: dict[str, Any] | None = None,
    ) -> SegmentTranscriptionResult:
        text = self._run_transcription(
            audio_path, options={**(options or {}), "without_timestamps": False}
        )
        return SegmentTranscriptionResult(
            segments=list(self.last_segment_metadata),
            text=text,
            language=self.last_language,
            language_probability=self.last_language_probability,
            timestamp_source="native",
        )

    def _run_transcription(
        self,
        audio_path: str,
        options: dict[str, Any] | None,
    ) -> str:
        self.last_segment_metadata = []
        self.last_language = None
        self.last_language_probability = None
        self.last_inference_device = None
        self.last_inference_used_cpu_fallback = False
        opts = dict(options or {})
        self._active_options = opts
        try:
            primary_model = self._get_primary_model()
            duration_seconds = self._audio_duration_seconds(audio_path)

            batched_override = opts.get("batched")
            batch_size = int(opts.get("batch_size") or self.config.long_audio_batch_size)
            if batched_override is not None:
                use_batched = bool(batched_override) and batch_size > 0
            else:
                use_batched = (
                    self.config.device.startswith("cuda")
                    and batch_size > 1
                    and duration_seconds is not None
                    and duration_seconds >= self.config.batch_threshold_seconds
                )

            self.last_inference_device = getattr(
                getattr(primary_model, "model", None), "device", self.config.device
            )
            if use_batched:
                if batched_override is None and duration_seconds is not None:
                    print(
                        "Long audio detected: "
                        f"duration_seconds={duration_seconds:.2f} "
                        f"threshold_seconds={self.config.batch_threshold_seconds:g} "
                        f"batch_size={batch_size}; "
                        "using batched GPU transcription."
                    )
                try:
                    return self._transcribe_batched(primary_model, audio_path)
                except Exception as exc:
                    self._batched_pipeline = None
                    batched_error = str(exc)
                    print(
                        "WARNING: Batched GPU transcription failed; retrying the same audio "
                        f"with ordinary GPU transcription before CPU fallback: {batched_error}"
                    )
                # Leave the exception scope before clearing cache so its traceback cannot
                # retain temporary allocations needed by the ordinary GPU retry.
                self._release_cuda_cache()

            try:
                return self._transcribe_with_model(primary_model, audio_path)
            except Exception as exc:
                if self._should_reload_primary_model(exc):
                    self._invalidate_primary_model(str(exc))
                if not self._should_retry_on_cpu(exc):
                    raise
                print(
                    "CUDA OOM during faster-whisper transcription; retrying same audio on CPU "
                    "and scheduling a fresh GPU model load for the next GPU request."
                )
                cpu_model = self._get_cpu_fallback_model()
                self.last_inference_device = getattr(
                    getattr(cpu_model, "model", None), "device", "cpu"
                )
                self.last_inference_used_cpu_fallback = True
                return self._transcribe_with_model(cpu_model, audio_path)
        finally:
            self._active_options = {}

    def _transcription_options(
        self,
        overrides: dict[str, Any] | None = None,
        *,
        for_batched: bool = False,
    ) -> dict[str, Any]:
        ov = overrides or {}
        language = ov["language"] if "language" in ov else self.config.language
        if isinstance(language, str) and not language.strip():
            language = None

        vad_filter = (
            bool(ov["vad_filter"])
            if "vad_filter" in ov and ov["vad_filter"] is not None
            else self.config.vad_filter
        )
        vad_threshold = float(
            ov["vad_threshold"]
            if "vad_threshold" in ov and ov["vad_threshold"] is not None
            else self.config.vad_threshold
        )
        vad_min_silence_ms = int(
            ov["vad_min_silence_ms"]
            if "vad_min_silence_ms" in ov and ov["vad_min_silence_ms"] is not None
            else self.config.vad_min_silence_ms
        )
        vad_speech_pad_ms = int(
            ov["vad_speech_pad_ms"]
            if "vad_speech_pad_ms" in ov and ov["vad_speech_pad_ms"] is not None
            else self.config.vad_speech_pad_ms
        )

        initial_prompt = (
            ov["initial_prompt"] if "initial_prompt" in ov else self.config.initial_prompt
        )
        if isinstance(initial_prompt, str) and not initial_prompt.strip():
            initial_prompt = None

        hotwords = ov["hotwords"] if "hotwords" in ov else self.config.hotwords
        if isinstance(hotwords, str) and not hotwords.strip():
            hotwords = None

        opts: dict[str, Any] = {
            "word_timestamps": False,
            "language": language,
            "beam_size": int(
                ov["beam_size"]
                if "beam_size" in ov and ov["beam_size"] is not None
                else self.config.beam_size
            ),
            "patience": float(
                ov["patience"]
                if "patience" in ov and ov["patience"] is not None
                else self.config.patience
            ),
            "length_penalty": float(
                ov["length_penalty"]
                if "length_penalty" in ov and ov["length_penalty"] is not None
                else self.config.length_penalty
            ),
            "vad_filter": vad_filter,
            "vad_parameters": {
                "threshold": vad_threshold,
                "min_silence_duration_ms": vad_min_silence_ms,
                "speech_pad_ms": vad_speech_pad_ms,
            },
            "log_prob_threshold": (
                float(ov["log_prob_threshold"])
                if "log_prob_threshold" in ov and ov["log_prob_threshold"] is not None
                else self.config.log_prob_threshold
            ),
            "no_speech_threshold": (
                float(ov["no_speech_threshold"])
                if "no_speech_threshold" in ov and ov["no_speech_threshold"] is not None
                else self.config.no_speech_threshold
            ),
            "compression_ratio_threshold": (
                float(ov["compression_ratio_threshold"])
                if "compression_ratio_threshold" in ov
                and ov["compression_ratio_threshold"] is not None
                else self.config.compression_ratio_threshold
            ),
            "initial_prompt": initial_prompt,
            "hotwords": hotwords,
        }

        if ov.get("task"):
            opts["task"] = str(ov["task"])
        # The batched pipeline otherwise defaults to suppressing timestamp tokens.
        if for_batched and "without_timestamps" in ov:
            opts["without_timestamps"] = ov["without_timestamps"]
        chunk_len = ov.get("chunk_length")
        if chunk_len is not None:
            opts["chunk_length"] = int(chunk_len)
        max_chunk_len = float(chunk_len) if chunk_len is not None else 30.0

        if ov.get("hallucination_silence_threshold") is not None:
            opts["hallucination_silence_threshold"] = float(
                ov["hallucination_silence_threshold"]
            )

        if "clip_timestamps" in ov and ov["clip_timestamps"] is not None:
            raw_clips = ov["clip_timestamps"]
            if isinstance(raw_clips, list):
                if raw_clips and isinstance(raw_clips[0], dict):
                    intervals = [
                        (float(c["start"]), float(c["end"])) for c in raw_clips
                    ]
                else:
                    if len(raw_clips) % 2:
                        raise ValueError("clip_timestamps must contain start/end pairs")
                    intervals = [
                        (float(raw_clips[i]), float(raw_clips[i + 1]))
                        for i in range(0, len(raw_clips), 2)
                    ]
                intervals = _split_clip_intervals(intervals, max_chunk_len)
                if for_batched:
                    opts["clip_timestamps"] = [
                        {"start": s, "end": e} for s, e in intervals
                    ]
                    opts["vad_filter"] = False
                    opts["vad_parameters"] = None
                else:
                    flat: list[float] = []
                    for s, e in intervals:
                        flat.extend([s, e])
                    opts["clip_timestamps"] = flat
                    opts["vad_filter"] = False
                    opts["vad_parameters"] = None
            elif isinstance(raw_clips, str) and not for_batched:
                opts["clip_timestamps"] = raw_clips

        if not opts["vad_filter"]:
            opts["vad_parameters"] = None

        return opts

    def _transcribe_with_model(self, model, audio_path: str) -> str:
        opts = getattr(self, "_active_options", None) or {}
        segments, info = model.transcribe(
            audio_path, **self._transcription_options(opts, for_batched=False)
        )
        self._record_info(info)
        return self._consume_segments(segments)

    def _transcribe_batched(self, primary_model, audio_path: str) -> str:
        if self._batched_pipeline is None:
            self._batched_pipeline = self._batched_pipeline_cls(primary_model)
        opts = getattr(self, "_active_options", None) or {}
        effective_batch_size = int(
            opts.get("batch_size") or self.config.long_audio_batch_size
        )
        segments, info = self._batched_pipeline.transcribe(
            audio_path,
            batch_size=effective_batch_size,
            **self._transcription_options(opts, for_batched=True),
        )
        self._record_info(info)
        return self._consume_segments(segments)

    def _record_info(self, info) -> None:
        language = getattr(info, "language", None)
        lang_prob = getattr(info, "language_probability", None)
        self.last_language = str(language) if isinstance(language, str) else None
        try:
            self.last_language_probability = (
                float(lang_prob) if lang_prob is not None else None
            )
        except (TypeError, ValueError):
            self.last_language_probability = None

    def _consume_segments(self, segments) -> str:
        text_parts: list[str] = []
        metadata: list[dict[str, object]] = []
        for segment in segments:
            text = str(segment.text)
            text_parts.append(text)
            metadata.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": text,
                    "avg_logprob": float(segment.avg_logprob),
                    "compression_ratio": float(segment.compression_ratio),
                    "no_speech_prob": float(segment.no_speech_prob),
                }
            )
        self.last_segment_metadata = metadata
        return "".join(text_parts).strip()

    def _audio_duration_seconds(self, audio_path: str) -> float | None:
        try:
            import av

            with av.open(audio_path) as container:
                if container.duration is not None:
                    return float(container.duration / av.time_base)
                for stream in container.streams.audio:
                    if stream.duration is not None and stream.time_base is not None:
                        return float(stream.duration * stream.time_base)
        except Exception as exc:
            print(
                "WARNING: Could not inspect audio duration; using ordinary transcription: "
                f"{exc}"
            )
        return None

    def _release_cuda_cache(self) -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

    def _should_retry_on_cpu(self, exc: Exception) -> bool:
        if not self.config.cpu_fallback_enabled:
            return False
        if not self.config.device.startswith("cuda"):
            return False
        message = " ".join(str(arg) for arg in getattr(exc, "args", ()) if arg)
        message = message or str(exc)
        lowered = message.lower()
        return "out of memory" in lowered or "cuda" in lowered and "memory" in lowered

    def _get_cpu_fallback_model(self):
        if self._cpu_fallback_model is not None:
            return self._cpu_fallback_model

        print(
            f"Loading faster-whisper CPU fallback model '{self.config.model_name}' "
            f"with compute type '{self.config.cpu_fallback_compute_type}' and "
            f"cpu_threads={self.config.cpu_fallback_cpu_threads}..."
        )
        self._cpu_fallback_model = self._whisper_model_cls(
            self.config.model_name,
            device="cpu",
            compute_type=self.config.cpu_fallback_compute_type,
            cpu_threads=self.config.cpu_fallback_cpu_threads,
        )
        print("faster-whisper CPU fallback model loaded successfully.")
        return self._cpu_fallback_model

    def _load_primary_model(self) -> None:
        print(
            f"Loading faster-whisper model '{self.config.model_name}' on device '{self.config.device}' "
            f"with compute type '{self.config.compute_type}'..."
        )
        self.model = self._whisper_model_cls(
            self.config.model_name,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )
        self._batched_pipeline = None
        self._gpu_model_needs_reload = False
        print("faster-whisper model loaded successfully.")

    def _get_primary_model(self):
        if self.model is None or self._gpu_model_needs_reload:
            print("Reloading faster-whisper primary model before GPU transcription attempt...")
            self._load_primary_model()
        return self.model

    def _invalidate_primary_model(self, reason: str) -> None:
        self.model = None
        self._batched_pipeline = None
        self._gpu_model_needs_reload = True
        print(f"Marked faster-whisper primary model for reload: {reason}")

    def _should_reload_primary_model(self, exc: Exception) -> bool:
        if not self.config.device.startswith("cuda"):
            return False
        message = " ".join(str(arg) for arg in getattr(exc, "args", ()) if arg)
        message = message or str(exc)
        lowered = message.lower()
        reload_markers = (
            "out of memory",
            "cudaerrorinvaliddevice",
            "invalid device ordinal",
            "cuda error",
            "cublas",
            "cudnn",
        )
        return any(marker in lowered for marker in reload_markers)
