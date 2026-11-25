# Transcription Server (`~/Dev/transcription-server`)

Local GPU speech-to-text HTTP service exposing `POST /transcribe_raw` (`audio/wav`, `audio/flac`, or raw audio → `{"text": "...", "backend": "...", "model": "..."}` plus optional preprocessing and variant-conversion metadata) for all dictation clients ([`linux-dictation`](https://github.com/Terrydaktal/linux-dictation), [`voice-keyboard`](https://github.com/Terrydaktal/voice-keyboard), [`floating-dictation`](https://github.com/Terrydaktal/floating-dictation), and `~/Dev/assistant`).

The [transcription experiment catalogue](docs/asr-experiments.md) records the models, settings, WER, punctuation, runtime, and memory results recovered from project history. It includes [CSV](docs/asr-experiments.csv) and [JSON evidence](docs/asr-experiments.json), and separates measured runs from failures and untested suggestions.

---

## Project Structure

```text
transcription-server/
├── pyproject.toml, uv.lock              # uv environment and CUDA/ASR extras
├── whisper_service.py                   # Compatibility launcher (delegates to assistant_transcription.server)
├── transcription_consensus.py           # Symlink to src/assistant_transcription/consensus/merge.py for benchmarks
├── src/assistant_transcription/
│   ├── server.py                        # FastAPI /transcribe_raw server, startup lock & desktop queue wakeup
│   ├── config.py                        # CLI flags, env vars, defaults & backend presets
│   ├── paths.py                         # Repo/service root resolution
│   ├── environment.py                   # Cache/device environment setup
│   ├── log_output.py                    # Structured timing & device log formatting
│   ├── preprocessing.py                 # Conditional 80 Hz high-pass / DC-offset source cleanup
│   ├── postprocessing.py                # Technical canonicalizer & deterministic US→UK variant conversion
│   ├── granite_programming_keywords.txt # Curated prompt keywords for Granite Speech
│   ├── parakeet_programming_phrases.txt # Uppercase NeMo phrase-boosting vocabulary for Parakeet
│   ├── backends/                        # faster-whisper, Parakeet, Canary-Qwen, Granite Speech
│   └── consensus/                       # Isolated multi-model worker orchestration & majority-vote merge
├── workers/cohere/                      # Isolated uv project and lockfile for Cohere Transcribe
├── benchmarks/                          # ASR and audio preprocessing evaluation scripts
├── docs/                                # Benchmark results, consensus study & noisy-ASR notes
└── tests/
    ├── test_audio_preprocessing.py
    ├── test_transcription_consensus.py
    ├── test_whisper_service.py
    └── test_whisper_startup.py
```

---

## Inputs, Outputs & Pipeline Behavior

- **Input:** `POST /transcribe_raw` with binary audio payload (WAV, FLAC, or ffmpeg-decodable audio). Optional `X-Client-Recording-Duration-Ms` header for end-to-end timing logs.
- **Output:** JSON `{"text": "...", "backend": "...", "model": "..."}` plus preprocessing/filter decision metadata and variant-conversion metadata when enabled.
- **Conditional source cleanup (`preprocessing.py`):** Enabled by default. Measures active-frame sub-60 Hz energy and DC offset before transcription. If low-frequency energy is sufficiently strong and either its ratio or the DC-offset trigger is exceeded, creates a temporary 80 Hz high-pass WAV via system `ffmpeg`; otherwise the original upload is used unchanged. Original audio is always retained for recovery, filtering fails open (reporting any `ffmpeg` error in logs and response), and `audio_preprocess_ms`, `filter_applied`, trigger reason, low-frequency ratio/level, and DC offset are logged for every request.
  - Disable with `--no-low-frequency-filter` or `WHISPER_LOW_FREQUENCY_FILTER=false`.
  - Tune thresholds with `--low-frequency-filter-cutoff-hz` (`WHISPER_LOW_FREQUENCY_FILTER_CUTOFF_HZ`), `--low-frequency-filter-ratio-threshold` (`WHISPER_LOW_FREQUENCY_FILTER_RATIO_THRESHOLD`), `--low-frequency-filter-absolute-dbfs` (`WHISPER_LOW_FREQUENCY_FILTER_ABSOLUTE_DBFS`), and `--low-frequency-filter-dc-offset-threshold` (`WHISPER_LOW_FREQUENCY_FILTER_DC_OFFSET_THRESHOLD`).
- **Technical canonicalizer & British English post-processing (`postprocessing.py`):** Every backend passes through the technical canonicalizer after raw ASR. Canonical filenames, paths, hashes, CLI flags, and curated technical terms are masked while optional `en_US -> en_GB` conversion runs (via Trelis English Variant Converter), then restored verbatim so spelling conversion never mutates code identifiers.
  - Disable variant conversion with `--no-variant-conversion` or `WHISPER_VARIANT_CONVERSION=false`.
  - Override locales with `--variant-source` / `--variant-target` (`WHISPER_VARIANT_SOURCE` / `WHISPER_VARIANT_TARGET`).
- **Diagnostic retention (`.transcription_recovery`):** Keeps the 20 most recent request recordings (`latest_request.wav` + 19 numbered history slots) and up to 20 timestamped failed request/error pairs. Configure with `--recovery-request-limit` (`WHISPER_RECOVERY_REQUEST_LIMIT`) and `--failed-recovery-limit` (`WHISPER_FAILED_RECOVERY_LIMIT`); `0` disables that category.
- **Desktop queue wakeup & startup lock:** On startup, the server reserves a per-user, per-port lock before loading any model — launching a second instance on the same port exits with a warning without touching the running server or queued recordings (locks release automatically on exit or crash). Once bound, startup automatically wakes `~/Dev/linux-dictation/whisper/queue-worker.sh` if completed desktop recordings are waiting.
- **Logging:** Generic HTTP access logs are hidden by default (`--http-access-log` or `WHISPER_HTTP_ACCESS_LOG=true` enables them). Each result/failure timing block ends with a blank line and labels Whisper execution with `device=GPU`, `device=CPU`, `device=CPU(fallback)`, or `device=not-run`.

---

## Setup & Running Backends

The project shares `/data/.cache/uv` with hard-linked packages. Hugging Face and Torch models use `${XDG_CACHE_HOME:-~/.cache}/huggingface` and `${XDG_CACHE_HOME:-~/.cache}/torch` unless `HF_HOME` / `TORCH_HOME` are explicitly set. The `uv` environment pins PyTorch and Torchaudio 2.11 to CUDA 12.8, cuDNN 9.19, and cuBLAS 12.8 so NeMo and `faster-whisper` do not load conflicting CUDA sublibraries.

Listens on `http://0.0.0.0:5001` by default. Override `--backend`, `--model`, `--device`, `--compute-type`, `--host`, and `--port` via CLI flags or `WHISPER_BACKEND`, `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`, `WHISPER_HOST`, and `WHISPER_PORT`.

### 1. Default: `faster-whisper` (`large-v3`)
```sh
cd ~/Dev/transcription-server
UV_CACHE_DIR=/data/.cache/uv uv sync
UV_CACHE_DIR=/data/.cache/uv uv run python whisper_service.py
# or: uv run assistant-transcription
```
- **Long-audio batching:** Recordings ≥ 60 s automatically use `faster-whisper`’s batched GPU pipeline at accuracy-validated batch size `2`. Batch size `4` (`--long-audio-batch-size 4` / `WHISPER_LONG_AUDIO_BATCH_SIZE=4`) reduces long-file runtime ~24% (~4,924 MiB peak on RTX 5060) at a slight accuracy trade-off. Configure threshold with `--batch-threshold-seconds` (`WHISPER_BATCH_THRESHOLD_SECONDS`); batch size `1` disables batching. Failed batched runs retry with standard GPU transcription before falling back to cached CPU on CUDA OOM.
- **Hotwords & beam controls:** Optional `--hotwords` / `--hotwords-file` (`WHISPER_HOTWORDS` / `WHISPER_HOTWORDS_FILE`, disabled by default — keep narrowly scoped because broader lists hurt technical WER) and `--patience` / `--length-penalty` (`WHISPER_PATIENCE` / `WHISPER_LENGTH_PENALTY`, validated defaults `1.0`).

### 2. Parakeet TDT 0.6B v2
```sh
cd ~/Dev/transcription-server
UV_CACHE_DIR=/data/.cache/uv uv sync --extra parakeet
UV_CACHE_DIR=/data/.cache/uv uv run assistant-transcription \
  --backend parakeet \
  --preset programming \
  --model nvidia/parakeet-tdt-0.6b-v2 \
  --device cuda
```
Presets (`--preset` / `WHISPER_PRESET`):
- `programming` *(default)*: `greedy_batch` for push-to-talk; optional phrase boosting (`--key-phrases-file`, alpha `1.5`, context score `1.0`, depth scaling `2.0`).
- `accuracy`: `greedy_batch`, label looping, CUDA graph decoding, `parakeet_programming_phrases.txt` (alpha `0.5`, override with `--key-phrases-file` or disable with `--no-key-phrases`), Silero VAD (`0.30` threshold, `150 ms` min speech, `700 ms` min silence, `400 ms` padding) trimming once from first onset to last offset and sending up to 120 s per uninterrupted chunk with no forced overlap.
- `accuracy-beam-experimental`: same segmentation with beam-5 `malsd_batch` + CUDA graphs for A/B testing.
- `rustly-pocket-conversation`: Silero VAD (`250 ms` speech, `600 ms` silence, `450 ms` padding, 25 s segments with 2 s overlap), phrase boosting off.
- All presets use 16 kHz mono lossless input via SoundFile, BF16 on supported CUDA (FP16 fallback), and channel `0` unless `--channel-selector 1` (`WHISPER_CHANNEL_SELECTOR`) is set.

### 3. Canary-Qwen 2.5B
```sh
cd ~/Dev/transcription-server
UV_CACHE_DIR=/data/.cache/uv uv sync --extra canary
UV_CACHE_DIR=/data/.cache/uv uv run assistant-transcription \
  --backend canary-qwen \
  --model nvidia/canary-qwen-2.5b
```

### 4. Granite Speech 4.1 2B (GGUF via `llama-server`)
```sh
cd ~/Dev/transcription-server
sudo pacman -S --needed llama-cpp
UV_CACHE_DIR=/data/.cache/uv uv sync --extra granite-speech
UV_CACHE_DIR=/data/.cache/uv uv run assistant-transcription \
  --backend granite-speech \
  --device cuda
```
Defaults to `ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0` (~1.96 GB), starting a private managed `llama-server` on `127.0.0.1:9797` with 4096-token context, temperature `0`, 512-token output guard, and `granite_programming_keywords.txt` appended to IBM’s prompt. Configurable via `--key-phrases-file`, `--no-key-phrases`, `--granite-prompt`, `--granite-max-new-tokens`, `--granite-temperature`, `--granite-server-binary`, `--granite-server-host`, `--granite-server-port`, `--granite-server-startup-timeout`, `--granite-request-timeout`, and `--granite-context-size`. Note: with `llama.cpp` `b10221`, Q8 GGUF returns unpunctuated text and deterministic post-processing provides canonical technical casing.

### 5. Multi-Model Consensus (`consensus` & `adaptive-consensus`)
```sh
cd ~/Dev/transcription-server
UV_CACHE_DIR=/data/.cache/uv uv sync --extra consensus
UV_CACHE_DIR=/data/.cache/uv uv sync --project workers/cohere
UV_CACHE_DIR=/data/.cache/uv uv run assistant-transcription --backend consensus --device cuda
```
- Runs three isolated voters sequentially (each exits before the next starts so VRAM is never additive — measured peak 4,380 MiB on RTX 5060): `faster-whisper` large-v3 (beam 4, batch 2 for ≥ 60 s), Cohere Transcribe BF16 (gated Hugging Face checkpoint — accept terms and run `hf auth login` first), and unboosted Parakeet TDT 0.6B v2 `accuracy` preset (120 s segments).
- Deterministic word & punctuation majority voting with Whisper as tie-breaking anchor reduced WER from `5.1518%` to `4.4964%`, raised punctuation micro-F1 from `0.7872` to `0.8254`, and sentence-boundary F1 from `0.8191` to `0.8662` across 41,500 reference words (see [`docs/asr-consensus-benchmark.md`](docs/asr-consensus-benchmark.md)).
- `--backend adaptive-consensus` runs Whisper + Parakeet first and sends only merged disagreement timestamp ranges to Cohere.

---

## Running Tests

```sh
cd ~/Dev/transcription-server
UV_CACHE_DIR=/data/.cache/uv uv run python -m unittest discover -s tests -v
```
