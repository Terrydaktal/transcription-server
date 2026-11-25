import os
import re
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from logging.config import dictConfig
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import log_output
# Explicit re-exports support existing imports through the root whisper_service module.
from .backends import (
    AdaptiveConsensusTranscriber as AdaptiveConsensusTranscriber,
    CanaryQwenTranscriber as CanaryQwenTranscriber,
    ConsensusTranscriber as ConsensusTranscriber,
    FasterWhisperTranscriber as FasterWhisperTranscriber,
    GraniteSpeechTranscriber as GraniteSpeechTranscriber,
    ParakeetTranscriber as ParakeetTranscriber,
    build_transcriber,
)
from .backends.granite_speech import prepare_granite_audio as prepare_granite_audio
from .backends.parakeet import first_nemo_transcription_text as first_nemo_transcription_text
from .config import (
    PARAKEET_PRESETS as PARAKEET_PRESETS,
    RuntimeConfig,
    build_runtime_config,
    default_model_for_backend as default_model_for_backend,
    load_whisper_hotwords as load_whisper_hotwords,
    parse_args,
    parse_parakeet_channel_selector as parse_parakeet_channel_selector,
)
from .environment import configure_shared_model_caches as configure_shared_model_caches
from .log_output import (
    ANSI_BREAKDOWN_VALUE as ANSI_BREAKDOWN_VALUE,
    FriendlyLogSource as FriendlyLogSource,
    build_log_config,
    colour_timing_fields as colour_timing_fields,
    service_log as print,
)
from .paths import REPO_ROOT
from .postprocessing import (
    EnglishVariantPostProcessor,
    canonicalize_technical_text,
    normalize_technical_text as normalize_technical_text,
    preserve_keyword_casing as preserve_keyword_casing,
    protect_technical_terms as protect_technical_terms,
    restore_technical_terms as restore_technical_terms,
)
from .preprocessing import prepare_audio_for_transcription


@asynccontextmanager
async def app_lifespan(application: FastAPI):
    try:
        yield
    finally:
        transcriber = getattr(application.state, "transcriber", None)
        if transcriber is not None and hasattr(transcriber, "close"):
            transcriber.close()


app = FastAPI(title="Local Transcription Server", lifespan=app_lifespan)
app.state.runtime_config = None
app.state.transcriber = None
app.state.variant_post_processor = None


def recovery_dir_path() -> Path:
    config = app.state.runtime_config or build_runtime_config()
    path = Path(config.recovery_dir).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _move_recovery_pair(directory: Path, source_stem: str, target_stem: str) -> None:
    for suffix in (".wav", ".meta"):
        source = directory / f"{source_stem}{suffix}"
        target = directory / f"{target_stem}{suffix}"
        if source.exists():
            source.replace(target)
        else:
            target.unlink(missing_ok=True)


def _remove_excess_recent_recovery_files(directory: Path, limit: int) -> None:
    for path in directory.glob("recent_request_*.*"):
        match = re.fullmatch(r"recent_request_(\d+)\.(?:wav|meta)", path.name)
        if match and int(match.group(1)) >= limit:
            path.unlink(missing_ok=True)


def write_recovery_copy(audio_path: str | Path) -> Path | None:
    config = app.state.runtime_config or build_runtime_config()
    limit = config.recovery_request_limit
    if limit == 0:
        return None

    tmp_audio_path: Path | None = None
    tmp_meta_path: Path | None = None
    try:
        directory = recovery_dir_path()
        latest_path = directory / "latest_request.wav"
        tmp_audio_path = directory / "latest_request.wav.tmp"
        tmp_meta_path = directory / "latest_request.meta.tmp"
        shutil.copyfile(audio_path, tmp_audio_path)
        tmp_meta_path.write_text(
            f"received_at_unix={time.time()}\nbytes={Path(audio_path).stat().st_size}\n",
            encoding="utf-8",
        )

        _remove_excess_recent_recovery_files(directory, limit)
        for slot in range(limit - 1, 0, -1):
            source_stem = "latest_request" if slot == 1 else f"recent_request_{slot - 1}"
            _move_recovery_pair(directory, source_stem, f"recent_request_{slot}")

        tmp_audio_path.replace(latest_path)
        tmp_meta_path.replace(directory / "latest_request.meta")
        return latest_path
    except Exception as recovery_error:
        print(f"Failed to write latest transcription recovery audio: {recovery_error}")
        return None
    finally:
        if tmp_audio_path is not None:
            tmp_audio_path.unlink(missing_ok=True)
        if tmp_meta_path is not None:
            tmp_meta_path.unlink(missing_ok=True)


def _prune_failed_recovery_pairs(directory: Path, limit: int) -> None:
    stems = sorted(
        {
            path.stem
            for path in directory.glob("failed_request_*.*")
            if path.suffix in {".wav", ".txt"}
        },
        reverse=True,
    )
    for stem in stems[limit:]:
        for suffix in (".wav", ".txt"):
            (directory / f"{stem}{suffix}").unlink(missing_ok=True)


def preserve_failed_request(tmp_path: str, reason: Exception) -> None:
    config = app.state.runtime_config or build_runtime_config()
    limit = config.failed_recovery_limit
    if limit == 0:
        return

    try:
        directory = recovery_dir_path()
        timestamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000_000:09d}"
        failed_path = directory / f"failed_request_{timestamp}.wav"
        shutil.copy2(tmp_path, failed_path)
        (directory / f"failed_request_{timestamp}.txt").write_text(
            f"{type(reason).__name__}: {reason}\n",
            encoding="utf-8",
        )
        _prune_failed_recovery_pairs(directory, limit)
        print(f"Preserved failed transcription audio for recovery: {failed_path}")
    except Exception as preserve_error:
        print(f"Failed to preserve failed transcription audio: {preserve_error}")


def configure_app(config: RuntimeConfig) -> None:
    current_config = getattr(app.state, "runtime_config", None)
    if current_config == config and getattr(app.state, "transcriber", None) is not None:
        return
    current_transcriber = getattr(app.state, "transcriber", None)
    if current_transcriber is not None and hasattr(current_transcriber, "close"):
        current_transcriber.close()
    app.state.runtime_config = config
    app.state.transcriber = build_transcriber(config)
    app.state.variant_post_processor = (
        EnglishVariantPostProcessor(config.variant_source, config.variant_target)
        if config.variant_conversion_enabled
        else None
    )


def ensure_transcriber_loaded() -> None:
    if getattr(app.state, "transcriber", None) is None:
        configure_app(build_runtime_config())


def format_timing_fields(
    outcome: str, timings: dict[str, int], device: str | None = None
) -> str:
    server_components = [
        "transcriber_ready_ms",
        "temp_file_open_ms",
        "temp_file_close_ms",
        "latest_recovery_copy_ms",
        "server_transcribe_ms",
        "postprocess_ms",
        "temp_file_cleanup_ms",
    ]
    if "audio_preprocess_ms" in timings:
        server_components.insert(4, "audio_preprocess_ms")
    server_breakdown = format_timing_breakdown(
        timings,
        "server_total_ms",
        tuple(server_components),
        "server_overhead_ms",
    )
    outcome_line = f"outcome={outcome}"
    if device is not None:
        outcome_line += f" device={device}"
    if timings["failed_recovery_copy_ms"] >= 0:
        outcome_line += f" failed_recovery_copy_ms={timings['failed_recovery_copy_ms']}"
    return " ".join(
        (
            outcome_line,
            f"server_total_ms={timings['server_total_ms']}{server_breakdown}",
        )
    )


def format_client_preparation(timings: dict[str, int]) -> str:
    client_components = (
        "client_stop_finalize_ms",
        "client_queue_wait_ms",
        "client_file_read_ms",
    )
    client_breakdown = format_timing_breakdown(
        timings,
        "client_stop_to_upload_ms",
        client_components,
        "client_pre_upload_overhead_ms",
        opening="[",
        closing="]",
    )
    return (
        f"client_recording_duration_ms={timings['client_recording_duration_ms']} "
        f"client_stop_to_upload_ms={timings['client_stop_to_upload_ms']}"
        f"{client_breakdown}"
    )


def format_audio_size(audio_bytes: int) -> str:
    return (
        f"audio_bytes={audio_bytes} "
        f"({audio_bytes / 1024:.2f} KiB, "
        f"{audio_bytes / (1024**2):.3f} MiB, "
        f"{audio_bytes / (1024**3):.6f} GiB)"
    )


def format_timing_breakdown(
    timings: dict[str, int],
    aggregate_key: str,
    component_keys: tuple[str, ...],
    overhead_key: str,
    opening: str = "(",
    closing: str = ")",
) -> str:
    values = [timings.get(key, -1) for key in component_keys]
    aggregate = timings[aggregate_key]
    if aggregate < 0 or any(value < 0 for value in values):
        return ""

    overhead = aggregate - sum(values)
    if overhead < 0:
        return ""

    components = [f"{key}={timings[key]}" for key in component_keys]
    if overhead:
        components.append(f"{overhead_key}={overhead}")
    return f" {opening}{' + '.join(components)}{closing}"


def client_timing_value(request: Request, header_name: str) -> int:
    try:
        value = int(request.headers.get(header_name, "-1"))
    except (TypeError, ValueError):
        return -1
    return value if value >= 0 else -1


def format_transcription_device(transcriber) -> str:
    device = getattr(
        transcriber, "last_inference_device", getattr(transcriber, "device", None)
    )
    if not isinstance(device, str):
        return "UNKNOWN"
    if device == "cpu":
        if getattr(transcriber, "last_inference_used_cpu_fallback", False):
            return "CPU(fallback)"
        return "CPU"
    if device.startswith(("cuda", "mps", "xpu", "vulkan")):
        return "GPU"
    return "UNKNOWN"


@app.post("/transcribe_raw")
async def transcribe_raw(request: Request):
    request_started = time.perf_counter()
    tmp_path: str | None = None
    preprocessing_result = None
    transcription_event: str | None = None
    inference_attempted = False
    outcome = "failed"
    timings = {
        "client_recording_duration_ms": client_timing_value(
            request, "x-client-recording-duration-ms"
        ),
        "client_stop_finalize_ms": client_timing_value(
            request, "x-client-stop-finalize-ms"
        ),
        "client_queue_wait_ms": client_timing_value(request, "x-client-queue-wait-ms"),
        "client_file_read_ms": client_timing_value(request, "x-client-file-read-ms"),
        "client_stop_to_upload_ms": client_timing_value(
            request, "x-client-stop-to-upload-ms"
        ),
        "transcriber_ready_ms": -1,
        "temp_file_open_ms": -1,
        "upload_body_read_ms": -1,
        "temp_file_close_ms": -1,
        "latest_recovery_copy_ms": -1,
        "audio_preprocess_ms": -1,
        "server_transcribe_ms": -1,
        "postprocess_ms": -1,
        "temp_file_cleanup_ms": -1,
        "server_total_ms": -1,
        "audio_bytes": 0,
        "failed_recovery_copy_ms": -1,
    }
    try:
        transcriber_started = time.perf_counter()
        ensure_transcriber_loaded()
        timings["transcriber_ready_ms"] = round(
            (time.perf_counter() - transcriber_started) * 1000
        )

        temp_file_open_started = time.perf_counter()
        temp_file_close_started: float | None = None
        # Clients send WAV, FLAC, M4A, or MP3; let PyAV probe the actual bytes.
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
            tmp_path = tmp.name
            timings["temp_file_open_ms"] = round(
                (time.perf_counter() - temp_file_open_started) * 1000
            )
            body_read_started = time.perf_counter()
            async for chunk in request.stream():
                if not chunk:
                    continue
                tmp.write(chunk)
                timings["audio_bytes"] += len(chunk)
            timings["upload_body_read_ms"] = round(
                (time.perf_counter() - body_read_started) * 1000
            )
            client_host = request.client.host if request.client is not None else "unknown"
            print(f"Recording prepared: {format_client_preparation(timings)}")
            print(
                "Audio received: "
                f"client={client_host} "
                f"client_recording_duration_ms={timings['client_recording_duration_ms']} "
                f"{format_audio_size(timings['audio_bytes'])} "
                f"upload_body_read_ms={timings['upload_body_read_ms']}"
            )
            temp_file_close_started = time.perf_counter()

        if temp_file_close_started is not None:
            timings["temp_file_close_ms"] = round(
                (time.perf_counter() - temp_file_close_started) * 1000
            )
        if timings["audio_bytes"] == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Empty request body.", "timings": timings},
            )

        recovery_started = time.perf_counter()
        write_recovery_copy(tmp_path)
        timings["latest_recovery_copy_ms"] = round(
            (time.perf_counter() - recovery_started) * 1000
        )

        preprocessing_started = time.perf_counter()
        with prepare_audio_for_transcription(
            tmp_path,
            enabled=app.state.runtime_config.low_frequency_filter_enabled,
            cutoff_hz=app.state.runtime_config.low_frequency_filter_cutoff_hz,
            ratio_threshold=app.state.runtime_config.low_frequency_filter_ratio_threshold,
            absolute_dbfs_threshold=app.state.runtime_config.low_frequency_filter_absolute_dbfs,
            dc_offset_threshold=app.state.runtime_config.low_frequency_filter_dc_offset_threshold,
        ) as preprocessing_result:
            timings["audio_preprocess_ms"] = round(
                (time.perf_counter() - preprocessing_started) * 1000
            )
            analysis = (
                preprocessing_result.analysis.as_dict()
                if preprocessing_result.analysis
                else {}
            )
            filtered_analysis = (
                preprocessing_result.filtered_analysis.as_dict()
                if preprocessing_result.filtered_analysis
                else {}
            )
            print(
                "Audio preprocessing: "
                f"enabled={app.state.runtime_config.low_frequency_filter_enabled} "
                f"filter_applied={preprocessing_result.filter_applied} "
                f"reason={preprocessing_result.filter_reason} "
                f"low_frequency_ratio={analysis.get('low_frequency_ratio', 'n/a')} "
                f"low_frequency_dbfs={analysis.get('low_frequency_dbfs', 'n/a')} "
                f"filtered_low_frequency_ratio={filtered_analysis.get('low_frequency_ratio', 'n/a')} "
                f"dc_offset={analysis.get('dc_offset', 'n/a')} "
                f"error={preprocessing_result.error or 'none'} "
                f"audio_preprocess_ms={timings['audio_preprocess_ms']}"
            )

            transcribe_started = time.perf_counter()
            inference_attempted = True
            original_text = app.state.transcriber.transcribe_file(
                preprocessing_result.transcription_path
            )
            timings["server_transcribe_ms"] = round(
                (time.perf_counter() - transcribe_started) * 1000
            )

        postprocess_started = time.perf_counter()
        canonicalized_text = canonicalize_technical_text(original_text)
        converted_text = canonicalized_text
        if app.state.variant_post_processor is not None:
            converted_text = app.state.variant_post_processor.convert_text(converted_text)
        timings["postprocess_ms"] = round((time.perf_counter() - postprocess_started) * 1000)
        transformations = []
        if canonicalized_text != original_text:
            transformations.append("technical canonicalization")
        if converted_text != canonicalized_text:
            transformations.append(
                f"{app.state.runtime_config.variant_source}->{app.state.runtime_config.variant_target}"
            )
        if transformations:
            transcription_event = (
                f"Transcribed Raw ({app.state.runtime_config.backend}, "
                f"{', '.join(transformations)}): "
                f"'{converted_text}'"
            )
        else:
            transcription_event = f"Transcribed Raw ({app.state.runtime_config.backend}): '{converted_text}'"
        outcome = "success"
        return {
            "text": converted_text,
            "original_text": original_text,
            "backend": app.state.runtime_config.backend,
            "model": app.state.runtime_config.model_name,
            "variant_conversion_enabled": app.state.runtime_config.variant_conversion_enabled,
            "variant_source": app.state.runtime_config.variant_source,
            "variant_target": app.state.runtime_config.variant_target,
            "audio_preprocessing": preprocessing_result.as_dict(),
            "timings": timings,
        }
    except Exception as exc:
        if tmp_path is not None:
            failed_recovery_started = time.perf_counter()
            preserve_failed_request(tmp_path, exc)
            timings["failed_recovery_copy_ms"] = round(
                (time.perf_counter() - failed_recovery_started) * 1000
            )
        log_output.service_logger.exception("Transcription request failed")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"{type(exc).__name__}: {exc}",
                "backend": getattr(app.state.runtime_config, "backend", ""),
                "model": getattr(app.state.runtime_config, "model_name", ""),
                "audio_preprocessing": (
                    preprocessing_result.as_dict() if preprocessing_result else None
                ),
                "timings": timings,
            },
        )
    finally:
        cleanup_started = time.perf_counter()
        if tmp_path is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)
        timings["temp_file_cleanup_ms"] = round(
            (time.perf_counter() - cleanup_started) * 1000
        )
        timings["server_total_ms"] = round((time.perf_counter() - request_started) * 1000)
        device = (
            format_transcription_device(app.state.transcriber)
            if inference_attempted
            else "not-run"
        )
        timing_fields = format_timing_fields(outcome, timings, device=device)
        if transcription_event is not None:
            print(f"{transcription_event} | {timing_fields}\n")
        else:
            print(f"Transcription timing | {timing_fields}\n")


@app.post("/client-event", include_in_schema=False)
@app.post("/desktop-delivery-report")
async def desktop_delivery_report(request: Request):
    payload = await request.json()
    status = str(payload.get("status", "")).strip()
    if not status:
        legacy_event = str(payload.get("event", "")).strip()
        status = legacy_event.removeprefix("desktop-delivery-") or "unknown"
    details = str(payload.get("details", "")).strip()
    label = {
        "complete": "Delivery complete",
        "refused": "Delivery refused",
    }.get(status, f"Delivery report received [{status}]")
    if details:
        print(f"{label}: {details}")
    else:
        print(label)
    return {"ok": True}


def main() -> int:
    import uvicorn

    log_config = build_log_config()
    dictConfig(log_config)

    runtime_config = build_runtime_config(parse_args())
    configure_app(runtime_config)
    uvicorn.run(
        app,
        host=runtime_config.host,
        port=runtime_config.port,
        log_config=log_config,
        access_log=runtime_config.http_access_log,
    )
    return 0
