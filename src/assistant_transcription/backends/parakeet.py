import json
import os
import re
import tempfile
import wave
from pathlib import Path

from ..config import PARAKEET_PRESETS, RuntimeConfig
from ..log_output import service_log as print


def first_nemo_transcription_text(result) -> str:
    """Normalize NeMo ASR's text, hypothesis, and nested batch return shapes."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if hasattr(result, "text"):
        return str(result.text).strip()
    if isinstance(result, dict):
        for key in ("text", "transcript", "transcription"):
            if key in result:
                return first_nemo_transcription_text(result[key])
        return ""
    if isinstance(result, (list, tuple)):
        if not result:
            return ""
        return first_nemo_transcription_text(result[0])
    return str(result).strip()


def first_nemo_hypothesis(result):
    if result is None:
        return None
    if hasattr(result, "text"):
        return result
    if isinstance(result, (list, tuple)):
        for item in result:
            hypothesis = first_nemo_hypothesis(item)
            if hypothesis is not None:
                return hypothesis
    return None


class ParakeetTranscriber:
    def __init__(self, config: RuntimeConfig) -> None:
        try:
            import torch
            from nemo.collections.asr.models import ASRModel
        except ImportError as exc:
            raise RuntimeError(
                "Parakeet requires NeMo ASR. Install it with "
                "`uv sync --extra parakeet` or `uv pip install 'nemo-toolkit[asr]'`."
            ) from exc

        self.config = config
        self.model_name = config.model_name
        self.device = config.device
        self.preset = PARAKEET_PRESETS[config.parakeet_preset]
        self.torch = torch
        self.vad_model = None
        requested_device = torch.device(config.device)
        print(
            f"Loading Parakeet TDT model '{self.model_name}' via NeMo ASR on device "
            f"'{requested_device}'..."
        )
        print(
            f"torch.cuda.is_available()={torch.cuda.is_available()} "
            f"torch.cuda.device_count()={torch.cuda.device_count()}"
        )
        print(
            f"Parakeet CUDA stack: torch={torch.__version__} "
            f"cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()}"
        )
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Parakeet was requested on CUDA but torch.cuda.is_available() is false."
            )

        compute_dtype = self._select_compute_dtype(requested_device)
        # Convert on CPU so CUDA only receives the final low-precision parameter copy.
        self.model = ASRModel.from_pretrained(
            model_name=self.model_name,
            map_location=torch.device("cpu"),
        )
        self.model = self.model.to(dtype=compute_dtype)
        self.model = self.model.to(requested_device)
        self.model.eval()
        self._configure_decoding()
        self._trim_cuda_cache()
        first_param = next(self.model.parameters(), None)
        param_device = str(first_param.device) if first_param is not None else "unknown"
        print(f"Parakeet first parameter device: {param_device}")
        print(
            "Parakeet preset: "
            f"{self.preset.name} sample_rate={self.preset.sample_rate} "
            f"channels={self.preset.channels} compute_dtype={compute_dtype} "
            f"batch_size={self.preset.batch_size} "
            f"channel_selector={config.parakeet_channel_selector}"
        )
        self._log_cuda_memory("after model load")
        print("Parakeet TDT model loaded successfully.")

    def transcribe_file(self, audio_path: str) -> str:
        try:
            if self.preset.vad_threshold is not None:
                text = self._transcribe_with_vad(audio_path)
            else:
                text = self._transcribe_one_file(audio_path)
            return text
        except Exception as exc:
            if "out of memory" in str(exc).lower():
                self._trim_cuda_cache()
            raise
        finally:
            self._log_cuda_memory("after request")

    def _trim_cuda_cache(self) -> None:
        if self.device.startswith("cuda") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _log_cuda_memory(self, stage: str) -> None:
        if not self.device.startswith("cuda") or not self.torch.cuda.is_available():
            return
        device = self.torch.device(self.device)
        mib = 1024 * 1024
        allocated = self.torch.cuda.memory_allocated(device) / mib
        reserved = self.torch.cuda.memory_reserved(device) / mib
        free, total = self.torch.cuda.mem_get_info(device)
        print(
            f"Parakeet CUDA memory {stage}: allocated_mib={allocated:.1f} "
            f"reserved_mib={reserved:.1f} driver_free_mib={free / mib:.1f} "
            f"driver_total_mib={total / mib:.1f}"
        )

    def _select_compute_dtype(self, requested_device):
        if requested_device.type != "cuda":
            return self.torch.float32
        if self.preset.compute_dtype == "bf16" and self.torch.cuda.is_bf16_supported():
            return self.torch.bfloat16
        return self.torch.float16

    def _configure_decoding(self) -> None:
        from omegaconf import OmegaConf

        if not hasattr(self.model, "change_decoding_strategy"):
            raise RuntimeError("The selected Parakeet model does not expose NeMo decoding configuration.")

        boosting_tree = {
            "context_score": self.preset.context_score,
            "depth_scaling": self.preset.depth_scaling,
        }
        key_phrases_file = self.config.parakeet_key_phrases_file.strip()
        if self.preset.phrase_boosting and key_phrases_file:
            key_phrases_path = Path(key_phrases_file).expanduser()
            if not key_phrases_path.is_file():
                raise FileNotFoundError(f"Parakeet key phrases file does not exist: {key_phrases_path}")
            boosting_tree["key_phrases_file"] = str(key_phrases_path)
            print(
                f"Parakeet phrase boosting enabled: file='{key_phrases_path}' "
                f"alpha={self.preset.boosting_tree_alpha}"
            )
        elif self.preset.phrase_boosting:
            print(
                "Parakeet phrase boosting preset is enabled, but no key phrases file was supplied; "
                "decoding will run without a boosting tree."
            )
        elif key_phrases_file:
            print(
                f"Parakeet preset '{self.preset.name}' disables phrase boosting; "
                f"ignoring key phrases file '{key_phrases_file}'."
            )

        decoder_options = {
            "max_symbols_per_step": self.preset.max_symbols_per_step,
            "ngram_lm_model": None,
            "ngram_lm_alpha": 0.0,
            "boosting_tree": boosting_tree,
            "boosting_tree_alpha": self.preset.boosting_tree_alpha,
        }
        if self.preset.decoder_strategy == "greedy_batch":
            decoder_section = "greedy"
            decoder_options.update(
                {
                    "loop_labels": self.preset.loop_labels,
                    "use_cuda_graph_decoder": self.preset.use_cuda_graph_decoder,
                }
            )
        elif self.preset.decoder_strategy == "malsd_batch":
            decoder_section = "beam"
            decoder_options.update(
                {
                    "beam_size": self.preset.beam_size,
                    "allow_cuda_graphs": self.preset.allow_cuda_graphs,
                }
            )
        else:
            raise ValueError(
                f"Unsupported Parakeet decoder strategy: {self.preset.decoder_strategy}"
            )

        confidence_config = {
            "preserve_frame_confidence": self.preset.word_confidence,
            "preserve_token_confidence": self.preset.word_confidence,
            "preserve_word_confidence": self.preset.word_confidence,
            "exclude_blank": True,
            "aggregation": "mean",
            "method_cfg": {
                "name": "entropy",
                "entropy_type": "tsallis",
                "alpha": 0.33,
                "entropy_norm": "exp",
            },
        }
        decoding_config = OmegaConf.create(
            {
                "strategy": self.preset.decoder_strategy,
                "compute_timestamps": self.preset.word_timestamps,
                "rnnt_timestamp_type": "word",
                "confidence_cfg": confidence_config,
                decoder_section: decoder_options,
            }
        )
        self.model.change_decoding_strategy(decoding_config, verbose=False)
        graph_setting = (
            self.preset.use_cuda_graph_decoder
            if decoder_section == "greedy"
            else self.preset.allow_cuda_graphs
        )
        print(
            "Parakeet decoding configured: "
            f"strategy={self.preset.decoder_strategy} "
            f"beam_size={self.preset.beam_size} "
            f"max_symbols_per_step={self.preset.max_symbols_per_step} "
            f"cuda_graphs={graph_setting}"
        )

    def _transcribe_one_file(self, audio_path: str) -> str:
        with self.torch.inference_mode():
            result = self.model.transcribe(
                [audio_path],
                batch_size=self.preset.batch_size,
                return_hypotheses=self.preset.word_timestamps or self.preset.word_confidence,
                timestamps=self.preset.word_timestamps,
                channel_selector=self.config.parakeet_channel_selector,
                verbose=False,
            )
        self._log_word_diagnostics(result)
        return first_nemo_transcription_text(result)

    def _log_word_diagnostics(self, result) -> None:
        if not self.preset.word_timestamps and not self.preset.word_confidence:
            return
        hypothesis = first_nemo_hypothesis(result)
        if hypothesis is None:
            print("Parakeet word diagnostics unavailable: no hypothesis returned.")
            return

        timestamp_data = (
            getattr(hypothesis, "timestamp", None)
            or getattr(hypothesis, "timestep", None)
            or {}
        )
        words = timestamp_data.get("word", []) if isinstance(timestamp_data, dict) else []
        confidences = list(getattr(hypothesis, "word_confidence", None) or [])
        diagnostics = []
        for index, word_data in enumerate(words):
            if isinstance(word_data, dict):
                entry = {
                    key: word_data[key]
                    for key in ("word", "start", "end")
                    if key in word_data
                }
            else:
                entry = {"word": str(word_data)}
            if index < len(confidences):
                confidence = confidences[index]
                if hasattr(confidence, "item"):
                    confidence = confidence.item()
                entry["confidence"] = round(float(confidence), 4)
            diagnostics.append(entry)
        print(
            "Parakeet word diagnostics: "
            + json.dumps(diagnostics, ensure_ascii=True, separators=(",", ":"))
        )

    def _get_vad_model(self):
        if self.vad_model is None:
            from silero_vad import load_silero_vad

            print(f"Loading Silero VAD model for Parakeet '{self.preset.name}' preset...")
            self.vad_model = load_silero_vad(onnx=False)
        return self.vad_model

    def _read_audio_for_vad(self, audio_path: str):
        import soundfile
        import torchaudio

        samples, sample_rate = soundfile.read(
            audio_path,
            dtype="float32",
            always_2d=True,
        )
        waveform = self.torch.from_numpy(samples.T)
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            channel_selector = self.config.parakeet_channel_selector
            if isinstance(channel_selector, int):
                if channel_selector >= waveform.shape[0]:
                    raise ValueError(
                        f"Cannot select channel {channel_selector} from audio with "
                        f"{waveform.shape[0]} channels."
                    )
                waveform = waveform[channel_selector : channel_selector + 1]
            else:
                waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.preset.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                self.preset.sample_rate,
            )
        return waveform.squeeze(0)

    def _save_vad_segment(self, path: str, audio) -> None:
        pcm = (
            audio.detach()
            .cpu()
            .clamp(-1.0, 1.0)
            .mul(32767.0)
            .to(self.torch.int16)
            .numpy()
            .tobytes()
        )
        with wave.open(path, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(self.preset.sample_rate)
            output.writeframes(pcm)

    def _transcribe_with_vad(self, audio_path: str) -> str:
        from silero_vad import get_speech_timestamps

        audio = self._read_audio_for_vad(audio_path)
        timestamps = get_speech_timestamps(
            audio,
            self._get_vad_model(),
            threshold=self.preset.vad_threshold,
            sampling_rate=self.preset.sample_rate,
            min_speech_duration_ms=self.preset.min_speech_duration_ms,
            max_speech_duration_s=float("inf"),
            min_silence_duration_ms=self.preset.min_silence_duration_ms,
            speech_pad_ms=self.preset.speech_pad_ms,
        )
        if not timestamps:
            print("Parakeet VAD found no speech in the request.")
            return ""

        segmentation_timestamps = self._vad_inference_timestamps(timestamps)
        segment_ranges = self._segment_vad_timestamps(
            segmentation_timestamps,
            audio.shape[-1],
        )
        transcripts = []
        for start, end, overlaps_previous in segment_ranges:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as segment_file:
                segment_path = segment_file.name
            try:
                self._save_vad_segment(segment_path, audio[start:end])
                text = self._transcribe_one_file(segment_path)
                if text:
                    transcripts.append((text, overlaps_previous))
            finally:
                if os.path.exists(segment_path):
                    os.remove(segment_path)

        print(
            f"Parakeet VAD segmented audio: natural_segments={len(timestamps)} "
            f"inference_segments={len(segment_ranges)} "
            f"max_segment_seconds={self.preset.max_segment_seconds:g} "
            f"preserve_internal_silence={self.preset.preserve_internal_silence} "
            "natural_overlap_seconds=0 "
            f"forced_split_overlap_seconds={self.preset.forced_split_overlap_seconds:g}"
        )
        return self._merge_segment_transcripts(transcripts)

    def _vad_inference_timestamps(
        self,
        timestamps: list[dict[str, int]],
    ) -> list[dict[str, int]]:
        if not timestamps or not self.preset.preserve_internal_silence:
            return timestamps
        return [
            {
                "start": int(timestamps[0]["start"]),
                "end": int(timestamps[-1]["end"]),
            }
        ]

    def _segment_vad_timestamps(
        self,
        timestamps: list[dict[str, int]],
        audio_length_samples: int,
    ) -> list[tuple[int, int, bool]]:
        max_samples = max(1, round(self.preset.max_segment_seconds * self.preset.sample_rate))
        overlap_samples = min(
            max_samples - 1,
            max(
                0,
                round(
                    self.preset.forced_split_overlap_seconds
                    * self.preset.sample_rate
                ),
            ),
        )
        segments = []
        for timestamp in timestamps:
            natural_start = max(0, int(timestamp["start"]))
            natural_end = min(audio_length_samples, int(timestamp["end"]))
            if natural_end <= natural_start:
                continue

            start = natural_start
            overlaps_previous = False
            while start < natural_end:
                end = min(start + max_samples, natural_end)
                segments.append((start, end, overlaps_previous))
                if end >= natural_end:
                    break
                start = end - overlap_samples
                overlaps_previous = overlap_samples > 0
        return segments

    @classmethod
    def _merge_segment_transcripts(
        cls,
        transcripts: list[tuple[str, bool]],
    ) -> str:
        merged = ""
        for transcript, overlaps_previous in transcripts:
            next_text = transcript.strip()
            if not next_text:
                continue
            if not merged:
                merged = next_text
            elif overlaps_previous:
                merged = cls._merge_overlapping_transcripts([merged, next_text])
            else:
                merged = f"{merged} {next_text}"
        return merged

    @staticmethod
    def _merge_overlapping_transcripts(transcripts: list[str]) -> str:
        if not transcripts:
            return ""
        merged = transcripts[0].strip()
        for transcript in transcripts[1:]:
            next_text = transcript.strip()
            if not next_text:
                continue
            left_words = merged.split()
            right_words = next_text.split()
            overlap = 0
            for size in range(min(12, len(left_words), len(right_words)), 0, -1):
                left = [re.sub(r"\W+", "", word).casefold() for word in left_words[-size:]]
                right = [re.sub(r"\W+", "", word).casefold() for word in right_words[:size]]
                if left == right and all(left):
                    overlap = size
                    break
            merged = f"{merged} {' '.join(right_words[overlap:])}".strip()
        return merged
