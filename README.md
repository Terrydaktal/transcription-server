# Transcription Server (`~/Dev/transcription-server`)

Local GPU speech-to-text HTTP server exposing `POST /transcribe_raw` (`audio/wav`, `audio/flac`, or raw audio → `{"text": "...", "backend": "...", "model": "..."}`) for all dictation clients (`~/Dev/linux-dictation`, `~/Dev/voice-keyboard`, and `~/Dev/assistant`).

## Project Structure

```text
transcription-server/
├── pyproject.toml, uv.lock              # uv environment and CUDA/ASR extras
├── whisper_service.py                   # Compatibility launcher (delegates to assistant_transcription.server)
├── transcription_consensus.py           # Symlink to consensus/merge.py for benchmark imports
├── src/assistant_transcription/
│   ├── server.py                        # FastAPI /transcribe_raw server, startup lock & desktop queue wakeup
│   ├── config.py                        # CLI flags, env vars, defaults & presets
│   ├── paths.py                         # Repo/service root resolution
│   ├── preprocessing.py                 # Conditional 80 Hz high-pass / DC cleanup
│   ├── postprocessing.py                # British English variant conversion & normalization
│   ├── backends/                        # faster-whisper, Parakeet, Canary, Granite Speech
│   └── consensus/                       # Multi-model voting & adaptive consensus
├── workers/cohere/                      # Isolated uv project for Cohere Transcribe worker
├── benchmarks/                          # ASR & audio preprocessing evaluation scripts
├── docs/                                # ASR experiment catalogue (md/csv/json) & noisy-ASR notes
└── tests/
    ├── test_audio_preprocessing.py
    ├── test_transcription_consensus.py
    ├── test_whisper_service.py
    └── test_whisper_startup.py
```

## Setup & Running

1. Install dependencies with `uv`:
   ```sh
   UV_CACHE_DIR=/data/.cache/uv uv sync
   ```
   Optional extras: `--extra parakeet`, `--extra canary`, `--extra granite-speech`, `--extra consensus` (and `UV_CACHE_DIR=/data/.cache/uv uv sync --project workers/cohere` for Cohere consensus).
2. Start the server (listens on `0.0.0.0:5001` by default):
   ```sh
   UV_CACHE_DIR=/data/.cache/uv uv run python whisper_service.py
   ```
   When the server comes up, it automatically wakes `~/Dev/linux-dictation/whisper/queue-worker.sh` if completed desktop recordings are waiting in the queue.

## Running Tests

```sh
UV_CACHE_DIR=/data/.cache/uv uv run python -m unittest discover -s tests -v
```
