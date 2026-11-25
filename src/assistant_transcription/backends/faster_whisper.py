import gc

from ..config import RuntimeConfig
from ..log_output import service_log as print


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
        self._load_primary_model()

    def transcribe_file(self, audio_path: str) -> str:
        self.last_segment_metadata = []
        self.last_inference_device = None
        self.last_inference_used_cpu_fallback = False
        primary_model = self._get_primary_model()
        duration_seconds = self._audio_duration_seconds(audio_path)
        use_batched = (
            self.config.device.startswith("cuda")
            and self.config.long_audio_batch_size > 1
            and duration_seconds is not None
            and duration_seconds >= self.config.batch_threshold_seconds
        )

        self.last_inference_device = getattr(
            getattr(primary_model, "model", None), "device", self.config.device
        )
        if use_batched:
            print(
                "Long audio detected: "
                f"duration_seconds={duration_seconds:.2f} "
                f"threshold_seconds={self.config.batch_threshold_seconds:g} "
                f"batch_size={self.config.long_audio_batch_size}; "
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

    def _transcription_options(self) -> dict:
        return {
            "word_timestamps": False,
            "language": self.config.language,
            "beam_size": self.config.beam_size,
            "patience": self.config.patience,
            "length_penalty": self.config.length_penalty,
            "vad_filter": self.config.vad_filter,
            "vad_parameters": {
                "threshold": self.config.vad_threshold,
                "min_silence_duration_ms": self.config.vad_min_silence_ms,
                "speech_pad_ms": self.config.vad_speech_pad_ms,
            },
            "log_prob_threshold": self.config.log_prob_threshold,
            "no_speech_threshold": self.config.no_speech_threshold,
            "compression_ratio_threshold": self.config.compression_ratio_threshold,
            "initial_prompt": self.config.initial_prompt or None,
            "hotwords": self.config.hotwords or None,
        }

    def _transcribe_with_model(self, model, audio_path: str) -> str:
        segments, _info = model.transcribe(audio_path, **self._transcription_options())
        return self._consume_segments(segments)

    def _transcribe_batched(self, primary_model, audio_path: str) -> str:
        if self._batched_pipeline is None:
            self._batched_pipeline = self._batched_pipeline_cls(primary_model)
        segments, _info = self._batched_pipeline.transcribe(
            audio_path,
            batch_size=self.config.long_audio_batch_size,
            **self._transcription_options(),
        )
        return self._consume_segments(segments)

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
