import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_FASTER_WHISPER_MODEL = "large-v3"
DEFAULT_CANARY_MODEL = "nvidia/canary-qwen-2.5b"
DEFAULT_PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v2"
DEFAULT_GRANITE_SPEECH_MODEL = "ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0"
DEFAULT_CONSENSUS_MODEL = "large-v3+cohere-transcribe-03-2026+parakeet-tdt-0.6b-v2"
DEFAULT_ADAPTIVE_CONSENSUS_MODEL = f"{DEFAULT_CONSENSUS_MODEL}-adaptive"
DEFAULT_GRANITE_PROMPT = "transcribe the speech with proper punctuation and capitalization."
DEFAULT_PARAKEET_PHRASES_FILE = Path(__file__).with_name("parakeet_programming_phrases.txt")
DEFAULT_GRANITE_KEYWORDS_FILE = Path(__file__).with_name("granite_programming_keywords.txt")


@dataclass(frozen=True)
class ParakeetPreset:
    name: str
    sample_rate: int
    channels: int
    channel_selector: int | str | None
    input_encoding: str
    compute_dtype: str
    decoder_strategy: str
    beam_size: int
    max_symbols_per_step: int
    loop_labels: bool
    use_cuda_graph_decoder: bool
    allow_cuda_graphs: bool
    batch_size: int
    max_segment_seconds: float
    forced_split_overlap_seconds: float
    preserve_internal_silence: bool
    vad_threshold: float | None
    min_speech_duration_ms: int
    min_silence_duration_ms: int
    speech_pad_ms: int
    word_timestamps: bool
    word_confidence: bool
    phrase_boosting: bool
    context_score: float
    depth_scaling: float
    boosting_tree_alpha: float


PARAKEET_PRESETS = {
    "programming": ParakeetPreset(
        name="programming",
        sample_rate=16_000,
        channels=1,
        channel_selector=0,
        input_encoding="lossless",
        compute_dtype="bf16",
        decoder_strategy="greedy_batch",
        beam_size=1,
        max_symbols_per_step=10,
        loop_labels=True,
        use_cuda_graph_decoder=True,
        allow_cuda_graphs=False,
        batch_size=1,
        max_segment_seconds=25.0,
        forced_split_overlap_seconds=0.0,
        preserve_internal_silence=True,
        vad_threshold=None,
        min_speech_duration_ms=150,
        min_silence_duration_ms=800,
        speech_pad_ms=250,
        word_timestamps=False,
        word_confidence=False,
        phrase_boosting=True,
        context_score=1.0,
        depth_scaling=2.0,
        boosting_tree_alpha=1.5,
    ),
    "rustly-pocket-conversation": ParakeetPreset(
        name="rustly-pocket-conversation",
        sample_rate=16_000,
        channels=1,
        channel_selector=0,
        input_encoding="lossless",
        compute_dtype="bf16",
        decoder_strategy="greedy_batch",
        beam_size=1,
        max_symbols_per_step=10,
        loop_labels=True,
        use_cuda_graph_decoder=True,
        allow_cuda_graphs=False,
        batch_size=1,
        max_segment_seconds=25.0,
        forced_split_overlap_seconds=2.0,
        preserve_internal_silence=False,
        vad_threshold=0.30,
        min_speech_duration_ms=250,
        min_silence_duration_ms=600,
        speech_pad_ms=450,
        word_timestamps=True,
        word_confidence=True,
        phrase_boosting=False,
        context_score=1.0,
        depth_scaling=2.0,
        boosting_tree_alpha=0.5,
    ),
    "accuracy": ParakeetPreset(
        name="accuracy",
        sample_rate=16_000,
        channels=1,
        channel_selector=0,
        input_encoding="lossless",
        compute_dtype="bf16",
        decoder_strategy="greedy_batch",
        beam_size=1,
        max_symbols_per_step=10,
        loop_labels=True,
        use_cuda_graph_decoder=True,
        allow_cuda_graphs=False,
        batch_size=1,
        max_segment_seconds=120.0,
        forced_split_overlap_seconds=0.0,
        preserve_internal_silence=True,
        vad_threshold=0.30,
        min_speech_duration_ms=150,
        min_silence_duration_ms=700,
        speech_pad_ms=400,
        word_timestamps=True,
        word_confidence=True,
        phrase_boosting=True,
        context_score=1.0,
        depth_scaling=2.0,
        boosting_tree_alpha=0.5,
    ),
    "accuracy-beam-experimental": ParakeetPreset(
        name="accuracy-beam-experimental",
        sample_rate=16_000,
        channels=1,
        channel_selector=0,
        input_encoding="lossless",
        compute_dtype="bf16",
        decoder_strategy="malsd_batch",
        beam_size=5,
        max_symbols_per_step=10,
        loop_labels=False,
        use_cuda_graph_decoder=False,
        allow_cuda_graphs=True,
        batch_size=1,
        max_segment_seconds=120.0,
        forced_split_overlap_seconds=0.0,
        preserve_internal_silence=True,
        vad_threshold=0.30,
        min_speech_duration_ms=150,
        min_silence_duration_ms=700,
        speech_pad_ms=400,
        word_timestamps=True,
        word_confidence=False,
        phrase_boosting=True,
        context_score=1.0,
        depth_scaling=2.0,
        boosting_tree_alpha=0.5,
    ),
}


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str
    model_name: str
    parakeet_preset: str
    parakeet_key_phrases_file: str
    parakeet_channel_selector: int | str | None
    granite_prompt: str
    granite_max_new_tokens: int
    granite_temperature: float
    granite_server_binary: str
    granite_server_host: str
    granite_server_port: int
    granite_server_startup_timeout_seconds: float
    granite_request_timeout_seconds: float
    granite_context_size: int
    device: str
    compute_type: str
    language: str
    beam_size: int
    patience: float
    length_penalty: float
    vad_filter: bool
    vad_threshold: float
    vad_min_silence_ms: int
    vad_speech_pad_ms: int
    log_prob_threshold: float
    no_speech_threshold: float
    compression_ratio_threshold: float
    low_frequency_filter_enabled: bool
    low_frequency_filter_cutoff_hz: float
    low_frequency_filter_ratio_threshold: float
    low_frequency_filter_absolute_dbfs: float
    low_frequency_filter_dc_offset_threshold: float
    initial_prompt: str
    hotwords: str
    batch_threshold_seconds: float
    long_audio_batch_size: int
    cpu_fallback_enabled: bool
    cpu_fallback_compute_type: str
    cpu_fallback_cpu_threads: int
    recovery_dir: str
    recovery_request_limit: int
    failed_recovery_limit: int
    variant_conversion_enabled: bool
    variant_source: str
    variant_target: str
    host: str
    port: int
    http_access_log: bool


def default_model_for_backend(backend: str) -> str:
    if backend == "canary-qwen":
        return DEFAULT_CANARY_MODEL
    if backend == "parakeet":
        return DEFAULT_PARAKEET_MODEL
    if backend == "granite-speech":
        return DEFAULT_GRANITE_SPEECH_MODEL
    if backend == "consensus":
        return DEFAULT_CONSENSUS_MODEL
    if backend == "adaptive-consensus":
        return DEFAULT_ADAPTIVE_CONSENSUS_MODEL
    return DEFAULT_FASTER_WHISPER_MODEL


def parse_parakeet_channel_selector(value: str | None, default: int | str | None):
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"none", "all"}:
        return None
    if normalized == "average":
        return "average"
    try:
        selector = int(normalized)
    except ValueError as exc:
        raise ValueError(
            "Parakeet channel selector must be '0', '1', 'average', or 'none'."
        ) from exc
    if selector < 0:
        raise ValueError("Parakeet channel selector must be non-negative.")
    return selector


def load_whisper_hotwords(value: str, file_name: str) -> str:
    entries = [entry.strip() for entry in re.split(r"[,\n]", value) if entry.strip()]
    if file_name.strip():
        path = Path(file_name).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Whisper hotwords file does not exist: {path}")
        entries.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )

    unique_entries: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        key = entry.casefold()
        if key not in seen:
            seen.add(key)
            unique_entries.append(entry)
    return ", ".join(unique_entries)


def build_runtime_config(args: argparse.Namespace | None = None) -> RuntimeConfig:
    backend = (args.backend if args else None) or os.environ.get("WHISPER_BACKEND", "faster-whisper")
    model_name = (args.model if args else None) or os.environ.get(
        "WHISPER_MODEL", default_model_for_backend(backend)
    )
    default_preset = "programming" if backend == "parakeet" else ""
    parakeet_preset = (args.preset if args else None) or os.environ.get(
        "WHISPER_PRESET", default_preset
    )
    if backend == "parakeet" and parakeet_preset not in PARAKEET_PRESETS:
        valid_presets = ", ".join(PARAKEET_PRESETS)
        raise ValueError(f"Unsupported Parakeet preset '{parakeet_preset}'. Choose from: {valid_presets}")
    preset_defaults = PARAKEET_PRESETS.get(parakeet_preset)
    if args is not None and args.key_phrases_file is not None:
        key_phrases_file = args.key_phrases_file
    elif "WHISPER_KEY_PHRASES_FILE" in os.environ:
        key_phrases_file = os.environ["WHISPER_KEY_PHRASES_FILE"]
    elif parakeet_preset in {"accuracy", "accuracy-beam-experimental"}:
        key_phrases_file = str(DEFAULT_PARAKEET_PHRASES_FILE)
    elif backend == "granite-speech":
        key_phrases_file = str(DEFAULT_GRANITE_KEYWORDS_FILE)
    else:
        key_phrases_file = ""
    channel_selector_value = (args.channel_selector if args else None) or os.environ.get(
        "WHISPER_CHANNEL_SELECTOR"
    )
    parakeet_channel_selector = parse_parakeet_channel_selector(
        channel_selector_value,
        preset_defaults.channel_selector if preset_defaults else None,
    )
    granite_prompt = (args.granite_prompt if args else None) or os.environ.get(
        "WHISPER_GRANITE_PROMPT", DEFAULT_GRANITE_PROMPT
    )
    granite_max_new_tokens_value = (
        args.granite_max_new_tokens
        if args is not None and args.granite_max_new_tokens is not None
        else os.environ.get("WHISPER_GRANITE_MAX_NEW_TOKENS", "512")
    )
    granite_temperature_value = (
        args.granite_temperature
        if args is not None and args.granite_temperature is not None
        else os.environ.get("WHISPER_GRANITE_TEMPERATURE", "0")
    )
    granite_server_binary = (args.granite_server_binary if args else None) or os.environ.get(
        "WHISPER_GRANITE_SERVER_BINARY", "llama-server"
    )
    granite_server_host = (args.granite_server_host if args else None) or os.environ.get(
        "WHISPER_GRANITE_SERVER_HOST", "127.0.0.1"
    )
    granite_server_port_value = (
        args.granite_server_port
        if args is not None and args.granite_server_port is not None
        else os.environ.get("WHISPER_GRANITE_SERVER_PORT", "9797")
    )
    granite_server_startup_timeout_value = (
        args.granite_server_startup_timeout
        if args is not None and args.granite_server_startup_timeout is not None
        else os.environ.get("WHISPER_GRANITE_SERVER_STARTUP_TIMEOUT", "300")
    )
    granite_request_timeout_value = (
        args.granite_request_timeout
        if args is not None and args.granite_request_timeout is not None
        else os.environ.get("WHISPER_GRANITE_REQUEST_TIMEOUT", "900")
    )
    granite_context_size_value = (
        args.granite_context_size
        if args is not None and args.granite_context_size is not None
        else os.environ.get("WHISPER_GRANITE_CONTEXT_SIZE", "4096")
    )
    device = (args.device if args else None) or os.environ.get("WHISPER_DEVICE", "cuda")
    compute_type = (args.compute_type if args else None) or os.environ.get(
        "WHISPER_COMPUTE_TYPE", "float16"
    )
    language = (args.language if args else None) or os.environ.get("WHISPER_LANGUAGE", "en")
    beam_size_value = (args.beam_size if args else None) or os.environ.get("WHISPER_BEAM_SIZE", "4")
    patience_value = (
        args.patience
        if args is not None and args.patience is not None
        else os.environ.get("WHISPER_PATIENCE", "1.0")
    )
    length_penalty_value = (
        args.length_penalty
        if args is not None and args.length_penalty is not None
        else os.environ.get("WHISPER_LENGTH_PENALTY", "1.0")
    )
    vad_filter = (
        (args.vad_filter if args else None)
        if args is not None and args.vad_filter is not None
        else os.environ.get("WHISPER_VAD_FILTER", "true").strip().lower() in {"1", "true", "yes", "on"}
    )
    vad_threshold_value = (args.vad_threshold if args else None) or os.environ.get("WHISPER_VAD_THRESHOLD", "0.30")
    vad_min_silence_ms_value = (args.vad_min_silence_ms if args else None) or os.environ.get(
        "WHISPER_VAD_MIN_SILENCE_MS", "1000"
    )
    vad_speech_pad_ms_value = (args.vad_speech_pad_ms if args else None) or os.environ.get(
        "WHISPER_VAD_SPEECH_PAD_MS", "400"
    )
    log_prob_threshold_value = (args.log_prob_threshold if args else None) or os.environ.get(
        "WHISPER_LOG_PROB_THRESHOLD", "-1.0"
    )
    no_speech_threshold_value = (args.no_speech_threshold if args else None) or os.environ.get(
        "WHISPER_NO_SPEECH_THRESHOLD", "0.3"
    )
    compression_ratio_threshold_value = (
        (args.compression_ratio_threshold if args else None)
        or os.environ.get("WHISPER_COMPRESSION_RATIO_THRESHOLD", "2.4")
    )
    low_frequency_filter_enabled = (
        (args.low_frequency_filter if args else None)
        if args is not None and args.low_frequency_filter is not None
        else os.environ.get("WHISPER_LOW_FREQUENCY_FILTER", "true").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    low_frequency_filter_cutoff_value = (
        args.low_frequency_filter_cutoff_hz
        if args is not None and args.low_frequency_filter_cutoff_hz is not None
        else os.environ.get("WHISPER_LOW_FREQUENCY_FILTER_CUTOFF_HZ", "80")
    )
    low_frequency_filter_ratio_value = (
        args.low_frequency_filter_ratio_threshold
        if args is not None and args.low_frequency_filter_ratio_threshold is not None
        else os.environ.get("WHISPER_LOW_FREQUENCY_FILTER_RATIO_THRESHOLD", "0.25")
    )
    low_frequency_filter_absolute_value = (
        args.low_frequency_filter_absolute_dbfs
        if args is not None and args.low_frequency_filter_absolute_dbfs is not None
        else os.environ.get("WHISPER_LOW_FREQUENCY_FILTER_ABSOLUTE_DBFS", "-35")
    )
    low_frequency_filter_dc_value = (
        args.low_frequency_filter_dc_offset_threshold
        if args is not None and args.low_frequency_filter_dc_offset_threshold is not None
        else os.environ.get("WHISPER_LOW_FREQUENCY_FILTER_DC_OFFSET_THRESHOLD", "0.005")
    )
    initial_prompt = (args.initial_prompt if args else None) or os.environ.get("WHISPER_INITIAL_PROMPT", "")
    hotwords_value = (getattr(args, "hotwords", None) if args else None) or os.environ.get(
        "WHISPER_HOTWORDS", ""
    )
    hotwords_file = (getattr(args, "hotwords_file", None) if args else None) or os.environ.get(
        "WHISPER_HOTWORDS_FILE", ""
    )
    hotwords = load_whisper_hotwords(hotwords_value, hotwords_file)
    batch_threshold_seconds_value = (
        args.batch_threshold_seconds
        if args is not None and args.batch_threshold_seconds is not None
        else os.environ.get("WHISPER_BATCH_THRESHOLD_SECONDS", "60")
    )
    long_audio_batch_size_value = (
        args.long_audio_batch_size
        if args is not None and args.long_audio_batch_size is not None
        else os.environ.get("WHISPER_LONG_AUDIO_BATCH_SIZE", "2")
    )
    cpu_fallback_enabled = (
        (args.cpu_fallback if args else None)
        if args is not None and args.cpu_fallback is not None
        else os.environ.get("WHISPER_CPU_FALLBACK", "true").strip().lower() in {"1", "true", "yes", "on"}
    )
    cpu_fallback_compute_type = (
        (args.cpu_fallback_compute_type if args else None)
        or os.environ.get("WHISPER_CPU_FALLBACK_COMPUTE_TYPE", "int8")
    )
    cpu_fallback_cpu_threads_value = (
        (args.cpu_fallback_cpu_threads if args else None)
        or os.environ.get("WHISPER_CPU_FALLBACK_CPU_THREADS")
        or str(os.cpu_count() or 4)
    )
    recovery_dir = (
        (args.recovery_dir if args else None)
        or os.environ.get("WHISPER_RECOVERY_DIR", ".transcription_recovery")
    )
    recovery_request_limit_value = (
        args.recovery_request_limit
        if args is not None and args.recovery_request_limit is not None
        else os.environ.get("WHISPER_RECOVERY_REQUEST_LIMIT", "20")
    )
    failed_recovery_limit_value = (
        args.failed_recovery_limit
        if args is not None and args.failed_recovery_limit is not None
        else os.environ.get("WHISPER_FAILED_RECOVERY_LIMIT", "20")
    )
    variant_conversion_enabled = (
        (args.variant_conversion if args else None)
        if args is not None and args.variant_conversion is not None
        else os.environ.get("WHISPER_VARIANT_CONVERSION", "true").strip().lower() in {"1", "true", "yes", "on"}
    )
    variant_source = (args.variant_source if args else None) or os.environ.get("WHISPER_VARIANT_SOURCE", "en_US")
    variant_target = (args.variant_target if args else None) or os.environ.get("WHISPER_VARIANT_TARGET", "en_GB")
    host = (args.host if args else None) or os.environ.get("WHISPER_HOST", "0.0.0.0")
    port_value = (args.port if args else None) or os.environ.get("WHISPER_PORT", "5001")
    http_access_log = (
        args.http_access_log
        if args is not None and args.http_access_log is not None
        else os.environ.get("WHISPER_HTTP_ACCESS_LOG", "false").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    return RuntimeConfig(
        backend=backend,
        model_name=model_name,
        parakeet_preset=parakeet_preset,
        parakeet_key_phrases_file=key_phrases_file,
        parakeet_channel_selector=parakeet_channel_selector,
        granite_prompt=granite_prompt,
        granite_max_new_tokens=max(1, int(granite_max_new_tokens_value)),
        granite_temperature=float(granite_temperature_value),
        granite_server_binary=granite_server_binary,
        granite_server_host=granite_server_host,
        granite_server_port=int(granite_server_port_value),
        granite_server_startup_timeout_seconds=max(
            1.0, float(granite_server_startup_timeout_value)
        ),
        granite_request_timeout_seconds=max(1.0, float(granite_request_timeout_value)),
        granite_context_size=max(512, int(granite_context_size_value)),
        device=device,
        compute_type=compute_type,
        language=language,
        beam_size=int(beam_size_value),
        patience=max(1.0, float(patience_value)),
        length_penalty=float(length_penalty_value),
        vad_filter=bool(vad_filter),
        vad_threshold=float(vad_threshold_value),
        vad_min_silence_ms=int(vad_min_silence_ms_value),
        vad_speech_pad_ms=int(vad_speech_pad_ms_value),
        log_prob_threshold=float(log_prob_threshold_value),
        no_speech_threshold=float(no_speech_threshold_value),
        compression_ratio_threshold=float(compression_ratio_threshold_value),
        low_frequency_filter_enabled=bool(low_frequency_filter_enabled),
        low_frequency_filter_cutoff_hz=max(20.0, float(low_frequency_filter_cutoff_value)),
        low_frequency_filter_ratio_threshold=min(
            1.0, max(0.0, float(low_frequency_filter_ratio_value))
        ),
        low_frequency_filter_absolute_dbfs=float(low_frequency_filter_absolute_value),
        low_frequency_filter_dc_offset_threshold=max(0.0, float(low_frequency_filter_dc_value)),
        initial_prompt=initial_prompt,
        hotwords=hotwords,
        batch_threshold_seconds=max(0.0, float(batch_threshold_seconds_value)),
        long_audio_batch_size=max(1, int(long_audio_batch_size_value)),
        cpu_fallback_enabled=bool(cpu_fallback_enabled),
        cpu_fallback_compute_type=cpu_fallback_compute_type,
        cpu_fallback_cpu_threads=max(1, int(cpu_fallback_cpu_threads_value)),
        recovery_dir=recovery_dir,
        recovery_request_limit=max(0, int(recovery_request_limit_value)),
        failed_recovery_limit=max(0, int(failed_recovery_limit_value)),
        variant_conversion_enabled=bool(variant_conversion_enabled),
        variant_source=variant_source,
        variant_target=variant_target,
        host=host,
        port=int(port_value),
        http_access_log=bool(http_access_log),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local transcription service.")
    parser.add_argument(
        "--backend",
        choices=[
            "faster-whisper",
            "canary-qwen",
            "parakeet",
            "granite-speech",
            "consensus",
            "adaptive-consensus",
        ],
        default=None,
        help=(
            "Transcription backend. Consensus votes faster-whisper, Cohere, and "
            "Parakeet over the full recording; adaptive-consensus sends only "
            "Whisper/Parakeet disagreement regions to Cohere. Defaults to "
            "faster-whisper."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name to load. Defaults to 'large-v3' for faster-whisper or "
            f"'{DEFAULT_CANARY_MODEL}' for canary-qwen or "
            f"'{DEFAULT_PARAKEET_MODEL}' for parakeet or "
            f"'{DEFAULT_GRANITE_SPEECH_MODEL}' for granite-speech."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=list(PARAKEET_PRESETS),
        default=None,
        help="Parakeet preset. Defaults to programming when --backend parakeet is selected.",
    )
    parser.add_argument(
        "--key-phrases-file",
        default=None,
        help=(
            "One phrase per line for Parakeet phrase boosting or Granite Speech "
            "prompt-based keyword biasing."
        ),
    )
    parser.add_argument(
        "--no-key-phrases",
        dest="key_phrases_file",
        action="store_const",
        const="",
        help="Disable the default phrase file for an unboosted A/B test.",
    )
    parser.add_argument(
        "--channel-selector",
        default=None,
        help="Parakeet channel: 0, 1, average, or none. Presets default to channel 0.",
    )
    parser.add_argument(
        "--granite-prompt",
        default=None,
        help=(
            "Granite Speech transcription instruction. Defaults to punctuated and "
            "capitalized transcription."
        ),
    )
    parser.add_argument(
        "--granite-max-new-tokens",
        type=int,
        default=None,
        help="Maximum Granite Speech output tokens per recording. Defaults to 512.",
    )
    parser.add_argument(
        "--granite-temperature",
        type=float,
        default=None,
        help="Granite Speech generation temperature. Defaults to deterministic 0.",
    )
    parser.add_argument(
        "--granite-server-binary",
        default=None,
        help="llama-server executable for Granite Speech Q8. Defaults to llama-server.",
    )
    parser.add_argument(
        "--granite-server-host",
        default=None,
        help="Private Granite Speech llama-server host. Defaults to 127.0.0.1.",
    )
    parser.add_argument(
        "--granite-server-port",
        type=int,
        default=None,
        help="Private Granite Speech llama-server port. Defaults to 9797.",
    )
    parser.add_argument(
        "--granite-server-startup-timeout",
        type=float,
        default=None,
        help="Seconds to wait for the Q8 model to load. Defaults to 300.",
    )
    parser.add_argument(
        "--granite-request-timeout",
        type=float,
        default=None,
        help="Seconds allowed for one Granite Speech transcription. Defaults to 900.",
    )
    parser.add_argument(
        "--granite-context-size",
        type=int,
        default=None,
        help="Granite Speech llama.cpp context size. Defaults to its trained 4096 tokens.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="Compute type for faster-whisper, for example 'float16' or 'int8'.",
    )
    parser.add_argument("--language", default=None, help="Language code for faster-whisper. Defaults to en.")
    parser.add_argument("--beam-size", type=int, default=None, help="Beam size for faster-whisper decoding.")
    parser.add_argument(
        "--patience",
        type=float,
        default=None,
        help="Beam-search patience for faster-whisper. Defaults to 1.0.",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=None,
        help="Beam-search length penalty for faster-whisper. Defaults to 1.0.",
    )
    parser.add_argument(
        "--vad-filter",
        dest="vad_filter",
        action="store_true",
        default=None,
        help="Enable faster-whisper VAD filtering.",
    )
    parser.add_argument(
        "--no-vad-filter",
        dest="vad_filter",
        action="store_false",
        help="Disable faster-whisper VAD filtering.",
    )
    parser.add_argument("--vad-threshold", type=float, default=None, help="VAD threshold.")
    parser.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=None,
        help="Minimum silence duration in milliseconds for VAD segmentation.",
    )
    parser.add_argument(
        "--vad-speech-pad-ms",
        type=int,
        default=None,
        help="Speech padding in milliseconds to preserve edges after VAD.",
    )
    parser.add_argument(
        "--log-prob-threshold",
        type=float,
        default=None,
        help="Low-confidence rejection threshold for faster-whisper.",
    )
    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        default=None,
        help="No-speech threshold for faster-whisper.",
    )
    parser.add_argument(
        "--compression-ratio-threshold",
        type=float,
        default=None,
        help="Compression ratio threshold to guard against degenerate transcripts.",
    )
    parser.add_argument(
        "--low-frequency-filter",
        dest="low_frequency_filter",
        action="store_true",
        default=None,
        help=(
            "Conditionally high-pass filter recordings with strong low-frequency "
            "contamination before transcription. Enabled by default."
        ),
    )
    parser.add_argument(
        "--no-low-frequency-filter",
        dest="low_frequency_filter",
        action="store_false",
        help="Disable conditional low-frequency audio filtering.",
    )
    parser.add_argument(
        "--low-frequency-filter-cutoff-hz",
        type=float,
        default=None,
        help="High-pass cutoff used by the conditional audio filter. Defaults to 80 Hz.",
    )
    parser.add_argument(
        "--low-frequency-filter-ratio-threshold",
        type=float,
        default=None,
        help=(
            "Minimum active-frame sub-60-Hz energy ratio that triggers filtering. "
            "Defaults to 0.25."
        ),
    )
    parser.add_argument(
        "--low-frequency-filter-absolute-dbfs",
        type=float,
        default=None,
        help=(
            "Minimum active-frame low-frequency level that triggers filtering. "
            "Defaults to -35 dBFS."
        ),
    )
    parser.add_argument(
        "--low-frequency-filter-dc-offset-threshold",
        type=float,
        default=None,
        help="Absolute DC offset threshold that contributes to filtering. Defaults to 0.005.",
    )
    parser.add_argument(
        "--initial-prompt",
        default=None,
        help="Optional initial prompt for names, jargon, acronyms, and spelling hints.",
    )
    parser.add_argument(
        "--hotwords",
        default=None,
        help=(
            "Comma-separated faster-whisper names and jargon to bias without a style "
            "instruction. Disabled by default."
        ),
    )
    parser.add_argument(
        "--hotwords-file",
        default=None,
        help=(
            "Text file containing one faster-whisper hotword or phrase per line; entries "
            "are combined with --hotwords."
        ),
    )
    parser.add_argument(
        "--batch-threshold-seconds",
        type=float,
        default=None,
        help="Use batched GPU transcription at or above this audio duration. Defaults to 60 seconds.",
    )
    parser.add_argument(
        "--long-audio-batch-size",
        type=int,
        default=None,
        help="Batch size for long-audio GPU transcription. Defaults to 2; 1 disables batching.",
    )
    parser.add_argument(
        "--cpu-fallback",
        dest="cpu_fallback",
        action="store_true",
        default=None,
        help="Retry faster-whisper requests on CPU if CUDA transcription hits an out-of-memory error.",
    )
    parser.add_argument(
        "--no-cpu-fallback",
        dest="cpu_fallback",
        action="store_false",
        help="Disable CPU retry after CUDA out-of-memory errors.",
    )
    parser.add_argument(
        "--cpu-fallback-compute-type",
        default=None,
        help="Compute type for the lazy-loaded CPU fallback model. Defaults to int8.",
    )
    parser.add_argument(
        "--cpu-fallback-cpu-threads",
        type=int,
        default=None,
        help="CPU thread count for the lazy-loaded CPU fallback model. Defaults to os.cpu_count().",
    )
    parser.add_argument(
        "--recovery-dir",
        default=None,
        help="Directory for latest and failed request audio recovery files.",
    )
    parser.add_argument(
        "--recovery-request-limit",
        type=int,
        default=None,
        help="Number of recent request recordings to retain. Defaults to 20; 0 disables it.",
    )
    parser.add_argument(
        "--failed-recovery-limit",
        type=int,
        default=None,
        help="Number of failed request recordings to retain. Defaults to 20; 0 disables it.",
    )
    parser.add_argument(
        "--variant-conversion",
        dest="variant_conversion",
        action="store_true",
        default=None,
        help="Enable deterministic post-processing to convert transcript spelling variants.",
    )
    parser.add_argument(
        "--no-variant-conversion",
        dest="variant_conversion",
        action="store_false",
        help="Disable deterministic English variant post-processing.",
    )
    parser.add_argument(
        "--variant-source",
        default=None,
        help="Source English variant code for post-processing, e.g. en_US.",
    )
    parser.add_argument(
        "--variant-target",
        default=None,
        help="Target English variant code for post-processing, e.g. en_GB.",
    )
    parser.add_argument("--host", default=None, help="Bind host. Defaults to 0.0.0.0.")
    parser.add_argument("--port", type=int, default=None, help="Bind port. Defaults to 5001.")
    parser.add_argument(
        "--http-access-log",
        dest="http_access_log",
        action="store_true",
        default=None,
        help="Show one generic HTTP completion line per request. Disabled by default.",
    )
    parser.add_argument(
        "--no-http-access-log",
        dest="http_access_log",
        action="store_false",
        help="Hide generic HTTP completion lines.",
    )
    return parser.parse_args()
