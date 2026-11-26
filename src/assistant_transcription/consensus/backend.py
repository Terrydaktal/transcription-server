import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..config import DEFAULT_FASTER_WHISPER_MODEL, DEFAULT_PARAKEET_MODEL, RuntimeConfig
from ..log_output import service_log as print
from ..paths import REPO_ROOT, SERVICE_DIR


class ConsensusTranscriber:
    """Run three isolated recognizers sequentially and majority-vote their output."""

    def __init__(self, config: RuntimeConfig) -> None:
        if not config.device.startswith("cuda"):
            raise RuntimeError("The validated consensus backend requires a CUDA device.")

        from .merge import merge_transcripts

        self.config = config
        self.merge_transcripts = merge_transcripts
        cohere_project = SERVICE_DIR / "workers" / "cohere"
        self.cohere_worker_path = cohere_project / "transcribe.py"
        configured_python = os.environ.get("ASSISTANT_CONSENSUS_COHERE_PYTHON")
        self.cohere_python = Path(configured_python) if configured_python else (
            cohere_project / ".venv" / "bin" / "python"
        )
        if not configured_python and not self.cohere_python.is_file():
            legacy_python = REPO_ROOT / "consensus_cohere" / ".venv" / "bin" / "python"
            if legacy_python.is_file():
                self.cohere_python = legacy_python
        if not self.cohere_python.is_file():
            raise RuntimeError(
                "The consensus Cohere environment is missing. Run "
                "`UV_CACHE_DIR=/data/.cache/uv uv sync --project workers/cohere`."
            )
        print(
            "Consensus backend ready: fixed sequential voters are faster-whisper "
            "large-v3, Cohere Transcribe BF16, and Parakeet TDT 0.6B v2."
        )

    def transcribe_file(self, audio_path: str) -> str:
        transcripts = [
            self._run_project_worker("faster-whisper", audio_path),
            self._run_cohere_worker(audio_path),
            self._run_project_worker("parakeet", audio_path),
        ]
        started = time.perf_counter()
        text = self.merge_transcripts(transcripts)
        print(
            "Consensus ROVER merge complete: "
            f"elapsed_ms={round((time.perf_counter() - started) * 1000)} "
            f"output_words={len(text.split())}"
        )
        return text

    def _run_project_worker(self, backend: str, audio_path: str) -> str:
        payload = self._run_project_worker_payload(backend, audio_path)
        text = payload.get("text")
        if not isinstance(text, str):
            raise RuntimeError(f"Consensus voter '{backend}' returned no transcript text.")
        return text

    def _run_project_worker_payload(
        self,
        backend: str,
        audio_path: str,
    ) -> dict[str, object]:
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONUNBUFFERED": "1",
                "WHISPER_BACKEND": backend,
                "WHISPER_DEVICE": "cuda",
                "WHISPER_LANGUAGE": "en",
                "WHISPER_VARIANT_CONVERSION": "false",
            }
        )
        if backend == "faster-whisper":
            environment.update(
                {
                    "WHISPER_MODEL": DEFAULT_FASTER_WHISPER_MODEL,
                    "WHISPER_COMPUTE_TYPE": "float16",
                    "WHISPER_BEAM_SIZE": "4",
                    "WHISPER_PATIENCE": "1.0",
                    "WHISPER_LENGTH_PENALTY": "1.0",
                    "WHISPER_VAD_FILTER": "true",
                    "WHISPER_VAD_THRESHOLD": "0.30",
                    "WHISPER_VAD_MIN_SILENCE_MS": "1000",
                    "WHISPER_VAD_SPEECH_PAD_MS": "400",
                    "WHISPER_LOG_PROB_THRESHOLD": "-1.0",
                    "WHISPER_NO_SPEECH_THRESHOLD": "0.3",
                    "WHISPER_COMPRESSION_RATIO_THRESHOLD": "2.4",
                    "WHISPER_INITIAL_PROMPT": "",
                    "WHISPER_HOTWORDS": self.config.hotwords,
                    "WHISPER_HOTWORDS_FILE": "",
                    "WHISPER_BATCH_THRESHOLD_SECONDS": "60",
                    "WHISPER_LONG_AUDIO_BATCH_SIZE": str(
                        self.config.long_audio_batch_size
                    ),
                    "WHISPER_CPU_FALLBACK": str(
                        self.config.cpu_fallback_enabled
                    ).lower(),
                    "WHISPER_CPU_FALLBACK_COMPUTE_TYPE": (
                        self.config.cpu_fallback_compute_type
                    ),
                    "WHISPER_CPU_FALLBACK_CPU_THREADS": str(
                        self.config.cpu_fallback_cpu_threads
                    ),
                }
            )
        else:
            environment.update(
                {
                    "WHISPER_MODEL": DEFAULT_PARAKEET_MODEL,
                    "WHISPER_PRESET": "accuracy",
                    "WHISPER_KEY_PHRASES_FILE": "",
                    "WHISPER_CHANNEL_SELECTOR": "0",
                }
            )
        return self._run_worker_payload(
            backend,
            [
                sys.executable,
                "-m",
                "assistant_transcription.consensus.worker",
                "--backend",
                backend,
                "--input",
                audio_path,
            ],
            environment,
        )

    def _run_cohere_worker(self, audio_path: str) -> str:
        return self._run_worker(
            "cohere",
            [
                str(self.cohere_python),
                str(self.cohere_worker_path),
                "--input",
                audio_path,
            ],
            os.environ.copy(),
        )

    def _run_cohere_regions_worker(
        self,
        audio_path: str,
        regions,
    ) -> list[str]:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as region_file:
            region_path = Path(region_file.name)
        region_path.write_text(
            json.dumps(
                [
                    {
                        "start": region.start,
                        "end": region.end,
                    }
                    for region in regions
                ]
            ),
            encoding="utf-8",
        )
        try:
            payload = self._run_worker_payload(
                "cohere-regions",
                [
                    str(self.cohere_python),
                    str(self.cohere_worker_path),
                    "--input",
                    audio_path,
                    "--regions-file",
                    str(region_path),
                ],
                os.environ.copy(),
            )
        finally:
            region_path.unlink(missing_ok=True)

        results = payload.get("regions")
        if not isinstance(results, list) or len(results) != len(regions):
            raise RuntimeError("Cohere returned an invalid adaptive-region result.")
        texts = [result.get("text") if isinstance(result, dict) else None for result in results]
        if any(not isinstance(text, str) for text in texts):
            raise RuntimeError("Cohere omitted an adaptive-region transcript.")
        return texts

    @staticmethod
    def _run_worker(
        name: str,
        command: list[str],
        environment: dict[str, str],
    ) -> str:
        payload = ConsensusTranscriber._run_worker_payload(name, command, environment)
        text = payload.get("text")
        if not isinstance(text, str):
            raise RuntimeError(f"Consensus voter '{name}' returned no transcript text.")
        return text

    @staticmethod
    def _run_worker_payload(
        name: str,
        command: list[str],
        environment: dict[str, str],
    ) -> dict[str, object]:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as result_file:
            result_path = Path(result_file.name)
        full_command = [*command, "--output", str(result_path)]
        started = time.perf_counter()
        print(f"Consensus voter started: backend={name}")
        try:
            subprocess.run(full_command, check=True, env=environment)
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Consensus voter '{name}' failed: {exc}") from exc
        finally:
            result_path.unlink(missing_ok=True)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Consensus voter '{name}' returned an invalid result.")
        text = payload.get("text", "")
        if not isinstance(text, str):
            raise RuntimeError(f"Consensus voter '{name}' returned an invalid result.")
        print(
            "Consensus voter complete: "
            f"backend={name} elapsed_ms={round((time.perf_counter() - started) * 1000)} "
            f"words={len(text.split())}"
        )
        return payload


class AdaptiveConsensusTranscriber(ConsensusTranscriber):
    """Adjudicate only Whisper/Parakeet disagreement regions with Cohere."""

    def __init__(self, config: RuntimeConfig) -> None:
        super().__init__(config)
        from .merge import (
            find_disagreement_regions,
            merge_adaptive_transcripts,
        )

        self.find_disagreement_regions = find_disagreement_regions
        self.merge_adaptive_transcripts = merge_adaptive_transcripts
        self.context_seconds = 1.0
        print(
            "Adaptive consensus enabled: Cohere receives only Whisper/Parakeet "
            f"disagreement regions with context_seconds={self.context_seconds:g}."
        )

    def transcribe_file(self, audio_path: str) -> str:
        self.last_segment_metadata: list[dict[str, object]] = []
        whisper_payload = self._run_project_worker_payload(
            "faster-whisper",
            audio_path,
        )
        whisper_text = whisper_payload.get("text")
        segments = whisper_payload.get("segments")
        if not isinstance(whisper_text, str) or not isinstance(segments, list):
            raise RuntimeError("Whisper returned no adaptive-consensus metadata.")

        parakeet_text = self._run_project_worker("parakeet", audio_path)
        try:
            regions = self.find_disagreement_regions(
                whisper_text,
                parakeet_text,
                segments,
                context_seconds=self.context_seconds,
            )
        except (KeyError, TypeError, ValueError) as exc:
            print(
                "WARNING: Adaptive region detection failed; running full Cohere voter: "
                f"{exc}"
            )
            cohere_text = self._run_cohere_worker(audio_path)
            return self.merge_transcripts(
                [whisper_text, cohere_text, parakeet_text]
            )

        if not regions:
            print("Adaptive consensus skipped Cohere: first two voters agree.")
            self.last_segment_metadata = segments
            return whisper_text

        covered_seconds = sum(region.end - region.start for region in regions)
        duration_seconds = max(
            (float(segment.get("end", 0.0)) for segment in segments),
            default=0.0,
        )
        coverage = covered_seconds / duration_seconds if duration_seconds else 1.0
        print(
            "Adaptive consensus regions selected: "
            f"regions={len(regions)} covered_seconds={covered_seconds:.2f} "
            f"audio_coverage={coverage:.2%}"
        )
        adjudications = self._run_cohere_regions_worker(audio_path, regions)
        started = time.perf_counter()
        text = self.merge_adaptive_transcripts(
            whisper_text,
            parakeet_text,
            regions,
            adjudications,
        )
        print(
            "Adaptive consensus merge complete: "
            f"elapsed_ms={round((time.perf_counter() - started) * 1000)} "
            f"output_words={len(text.split())}"
        )
        return text
