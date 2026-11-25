# Transcription Experiment Catalogue

Compiled **2026-09-12** from the original conversation, archived tool outputs, and the retained [consensus report](asr-consensus-benchmark.md). This is a historical reconstruction, **not a new benchmark run**.

There are **128 entries**: 86 scored/aggregate/verification entries, 30 unscored or failed attempts, and 12 proposed or configured options without a recovered benchmark. Repeats, derived totals and broken runs are labelled; they are not 128 independent successful experiments.

- [CSV: all entries and parameters](asr-experiments.csv)
- [JSON: exact available values, profiles, evidence and provenance](asr-experiments.json)
- [Corpora and measurement rules](#corpora-and-measurement-rules)
- [Parameter profiles](#parameter-profiles)
- [Original model comparison](#original-model-comparison)
- [Whisper parameter sweep](#whisper-parameter-sweep)
- [Long-recording comparisons](#long-recording-comparisons)
- [Cohere parameter controls](#cohere-parameter-controls)
- [Consensus and production checks](#consensus-and-production-checks)
- [Short technical dictation](#short-technical-dictation)
- [Failures and unscored backends](#failures-and-unscored-backends)
- [Microphone preprocessing](#microphone-preprocessing)
- [Discussed-but-unmeasured options](#discussed-but-unmeasured-options)
- [Memory and timing](#memory-and-timing)
- [Evidence register](#evidence-register)

## Aggregate Results

Same four recordings, **41,500 reference words**. WER is pooled errors / pooled words, not the mean of episode percentages. Higher punctuation and sentence F1 are better. NR means **not recovered/not measured**, never zero.

| ID | Configuration | S / D / I | Errors | WER % | Punct. F1 | Sentence F1 | Seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A-W | Whisper beam 4 / batch 2 | 711 / 937 / 490 | 2138 | 5.1518 | 0.7872 | 0.8191 | 351.42 | [E055](#e055) |
| A-B4 | Whisper beam 4 / batch 4 | 714 / 936 / 491 | 2141 | 5.1590 | NR | NR | 267.67 | [E029](#e029), [E018](#e018), [E022](#e022), [E030](#e030), [E035](#e035), [E041](#e041), [E042](#e042) |
| A-C | Cohere official greedy | 832 / 966 / 523 | 2321 | 5.5928 | NR | NR | 148.43 | [E009](#e009), [E010](#e010), [E011](#e011), [E012](#e012), [E005](#e005), [E006](#e006), [E037](#e037), [E038](#e038), [E033](#e033), [E036](#e036), [E039](#e039), [E043](#e043) |
| A-P | Parakeet unboosted greedy | 983 / 777 / 1823 | 3583 | 8.6337 | NR | NR | 97.24 | [E004](#e004), [E005](#e005), [E006](#e006), [E037](#e037), [E033](#e033), [E036](#e036), [E039](#e039), [E043](#e043) |
| A-R | Full three-model consensus | 650 / 782 / 434 | 1866 | 4.4964 | 0.8254 | 0.8662 | NR | [E055](#e055), [E056](#e056) |

The batch-4, Cohere and Parakeet aggregate WERs and runtime sums were calculated here from their recorded constituent runs. The Whisper and full-consensus pooled punctuation scores were retained in the original report. Runtime sums exclude model loading; **Cohere's sum is generation only**, excluding CPU feature extraction. Full-consensus total runtime across all four recordings was not retained.

The evidence supports full consensus for the best **combined WER/punctuation result across this corpus**, with a substantial latency cost. Whisper remains the strongest single-model aggregate here. Cohere actually beats Whisper on both metrics on the Ruff recording alone; rankings are not universal.

## Corpora And Measurement Rules

| Code | Recording | Duration | Reference words | Reference / role |
| --- | --- | ---: | ---: | --- |
| TECH32 | Personal Git/configuration dictation | 32.7 s | None verified | Personal Git/configuration passage; qualitative only |
| R332 | [Talk Python #332: Robust Python excerpt](https://talkpython.fm/episodes/show/332/robust-python) | 601.88 s | 2060 | Publisher WebVTT; development/control recording |
| S344 | [Talk Python #344: SQLAlchemy 2.0](https://talkpython.fm/episodes/show/344/sqlalchemy-2.0) | 3979 s | 13739 | Publisher WebVTT; development/control recording |
| R400 | [Talk Python #400: Ruff](https://talkpython.fm/episodes/transcript/400/ruff-the-fast-rust-based-python-linter) | 3825 s | 13316 | Publisher WebVTT; development/control recording |
| R487 | [Talk Python #487: Rust Extensions (held out)](https://talkpython.fm/episodes/show/487/building-rust-extensions-for-python) | 3744 s | 12385 | Publisher transcript; held out during consensus development |
| MIC | Personal microphone/gain/towel recordings | Varies | None verified | Historical microphone, gain and towel comparisons |

The actual saved R332 excerpt was **601.880 seconds**, rather than the approximate 599 seconds used in parts of the chat. Other episode durations are rounded to seconds.

- These are **reference-relative WERs**, not human-adjudicated true error rates. The podcast references were publisher-provided/AssemblyAI-derived and contain suspected mistakes. No corrected verbatim reference was recovered.
- WER scoring ignored case and punctuation. S/D/I are substitutions, deletions and insertions. Where only WER and reference length survived, the integer error count is reconstructed and flagged in JSON.
- Punctuation micro-F1 covers the retained scorer's comma, period, question and exclamation classes; sentence-boundary F1 is separate. Neither is a human readability rating. Output punctuation counts alone are not accuracy scores.
- Raw ASR and post-processed results are kept separate. Generic podcast tests have no repository canonicalization/UK conversion unless the row says otherwise; keyword contamination is explicitly marked.
- Entries with missing F1 do not acquire one from an unrelated run. Punctuation requested in a prompt does not establish that the output was well punctuated.
- Time is the archived **transcription timer** unless a section specifies generation-only, processor-plus-generation, cold pipeline or preprocessing. No table is a guaranteed stop-button-to-clipboard latency.
- This is an evidence inventory, not proof every possible configuration was tried. Recommendations, theoretical WER estimates and external leaderboard numbers are not local measurements.

## Parameter Profiles

A row inherits its profile, with changes named in its configuration/notes. Fields that were not retained are not silently filled with present-day library defaults.

### W4

**Model:** `large-v3`

**Runtime:** faster-whisper / CTranslate2

CUDA float16, English, beam_size=4, patience=1.0, length_penalty=1.0; Silero VAD on, threshold=0.30, min_silence_duration_ms=1000, speech_pad_ms=400; log_prob_threshold=-1.0, no_speech_threshold=0.3, compression_ratio_threshold=2.4; no initial prompt or hotwords, word_timestamps=false. BatchedInferencePipeline batch_size=2 at duration >=60 s, ordinary sequential WhisperModel below 60 s.

Beam 4 and batch 2 are different controls. Voter/benchmark raw results exclude technical normalization and US-to-UK conversion. Other library defaults are not reconstructed as explicit settings.

### W5

**Model:** `large-v3`

**Runtime:** faster-whisper / CTranslate2

Same as W4, but beam_size=5 (earlier production configuration).

For the 32.70 s passage this is the ordinary sequential path, not batched.

### P120

**Model:** `nvidia/parakeet-tdt-0.6b-v2`

**Runtime:** NeMo

CUDA BF16 (FP16 fallback), 16 kHz mono channel 0, batch_size=1, greedy_batch, max_symbols_per_step=10, loop_labels=true, CUDA graph decoder=true; VAD threshold=0.30, minimum speech=150 ms, silence=700 ms, padding=400 ms; trim outer edges only, preserve internal pauses, max segment=120 s, overlap=0. No phrase file for generic benchmarks; word timestamps enabled.

Word confidence was initially enabled; the final consensus worker disables it because of a NeMo metadata failure. Keep the model/cache resident; free unused cache for OOM recovery, not after every successful request.

### P25

**Model:** `nvidia/parakeet-tdt-0.6b-v2`

**Runtime:** NeMo

Earlier greedy accuracy preset: P120 decoder and VAD values, but max segment=25 s, forced overlap=2 s; programming phrase file, alpha=0.5, context_score=1.0, depth_scaling=2.0; timestamps/confidence on.

The short technical test exposed duplicate/corrupted words in the overlap merger. Not the corrected P120 benchmark.

### P-BEAM-OLD

**Model:** `nvidia/parakeet-tdt-0.6b-v2`

**Runtime:** NeMo

Earlier accuracy preset: CUDA BF16, malsd_batch beam=5, batch=1, max_symbols_per_step=10, CUDA graphs off; VAD 0.30 / 150 ms speech / 1000 ms silence / 400 ms pad; 25 s segments, 2 s overlap; no active phrase file, configured alpha=1.5 therefore inactive; timestamps/confidence off; empty_cache after requests.

Superseded; setting alpha without loading phrases does not activate boosting.

### GQ

**Model:** `ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0`

**Runtime:** llama.cpp llama-server (CUDA in benchmark)

Q8_0 language weights, FP16 speech projector, context=4096, one slot, automatic GPU layer offload; channel 0 mono 16 kHz PCM16; temperature=0, max_tokens=512; prompt='transcribe the speech with proper punctuation and capitalization.'; seven non-overlapping 90 s chunks on R332; no keywords.

Earlier live configuration used Vulkan and 2048 tokens without explicit temperature. Those settings are not the GQ scored benchmark. Prompt must be sent per chunk.

### GB

**Model:** `ibm-granite/granite-speech-4.1-2b`

**Runtime:** Official Transformers

CUDA BF16, eval/inference mode, do_sample=false, num_beams=1, max_new_tokens=512; 16 kHz mono, same seven 90 s chunks and punctuation-only prompt as GQ; no keywords.

Native-runtime control, not Q8.

### Q90

**Model:** `Qwen/Qwen3-ASR-1.7B-hf`

**Runtime:** Official Transformers

CUDA BF16, eval/inference mode, 16 kHz mono, English forced, no context/glossary, deterministic generation, max_new_tokens=512, independent 90 s chunks.

Q60 changes only chunk duration to 60 s.

### Q60

**Model:** `Qwen/Qwen3-ASR-1.7B-hf`

**Runtime:** Official Transformers

Q90 with 60 s chunks.

No post-processing for raw WER.

### CW

**Model:** `nyralabs/CrisperWhisper2.0_large`

**Runtime:** CTranslate2 CrisperWhisper inference

30 s chunks, stride=26 s, max_new_tokens=256, word_timestamps=false; verbatim/intended mode and continuation/chunked_lcs strategy as specified per row. Continuation: context_words=12, drop_words=2, temperature fallback on, speculative decoding off, hallucination mitigation on.

Continuation dropped large portions of this sample. Verbatim mode is additionally mismatched to a cleaned publisher reference. Not all low-level runtime defaults were retained.

### C35-FIXED

**Model:** `CohereLabs/cohere-transcribe-03-2026`

**Runtime:** Official Transformers, manually split input

CUDA BF16, English, punctuation on, greedy, max_new_tokens=256, 16 kHz mono, VAD; independent fixed 35 s blocks, one at a time (18 blocks for R332).

Initial memory workaround. It is NOT the later official-boundary control.

### C-OFFICIAL

**Model:** `CohereLabs/cohere-transcribe-03-2026`

**Runtime:** Transformers 5.14.1 / PyTorch 2.11 CUDA 12.8 (final voter)

BF16, English, punctuation=true, greedy deterministic generation, max_new_tokens=256; 16 kHz mono, edge-only Silero VAD; complete waveform through official CPU processor, max_audio_clip_s=35 and backward low-energy search window=5 s; GPU microbatch=1, preserve attention_mask, decoder_input_ids, audio_chunk_index; ordered reassembly.

The 5 s search window is not duplicated overlapping audio. R332 yields 19 official chunks. Recorded generation-only timings exclude CPU processor time.

### ROVER

**Model:** `Whisper large-v3 + Cohere Transcribe + Parakeet TDT v2`

**Runtime:** Isolated sequential voters, deterministic ROVER merge

W4 + C-OFFICIAL + unboosted P120; strict three-way word/punctuation majority, Whisper tie-breaking anchor; no reference, LLM, glossary, or confidence score used to merge.

Model loads are repeated in separate processes; cold pipeline duration is not comparable to one resident model's warm duration.

### ADAPTIVE

**Model:** `Whisper large-v3 + Parakeet TDT v2 + selective Cohere`

**Runtime:** adaptive-consensus

Full W4 and P120 passes, merge timestamped disagreement regions, run one C-OFFICIAL worker on those regions only, three-way voting inside regions and Whisper elsewhere.

Not the proposed inverse cascade of full Cohere with selective Whisper fallback.

### CANARY

**Model:** `nvidia/canary-qwen-2.5b`

**Runtime:** NeMo

Initial CPU-loaded path, then explicit CUDA placement; complete settings and comparable quality benchmark not retained.

CPU transcription worked slowly; subsequent CUDA transfer ran out of memory. Not evidence of a clean under-5-GiB run.

### HIGGS

**Model:** `bosonai/higgs-audio-v3-stt Q8_0 conversion`

**Runtime:** CrispASR CUDA / sm_120

Whole R332 recording, punctuation request; architecture-specific Q8 path.

Repetition failure and 5716 MiB sampled driver memory. No completed WER/punctuation score recovered.

### FULLSTOP

**Model:** `Whisper large-v3 + FullStop punctuation restorer`

**Runtime:** Separate post-ASR punctuation model

Restore punctuation on the earlier 65-error Whisper transcript.

Exact FullStop checkpoint and extra processing duration not recovered; do not assume lexical WER is a new ASR run.

### MIC-HP80

**Model:** `large-v3 + conditional audio cleanup`

**Runtime:** audio_preprocessing.py / ffmpeg

At least 5 active speech frames; sub-60 Hz energy >= -35 dBFS AND (low-frequency ratio >=0.25 OR absolute DC offset >=0.005); if triggered, mono WAV with a two-pole 80 Hz high-pass. Preserve original and fail open.

Source-signal measurements and transcript equality checks, not reference-scored WER tests.

### W-SEQ

**Model:** `large-v3`

**Runtime:** faster-whisper WhisperModel (not BatchedInferencePipeline)

W4 audio/decoder values, but ordinary sequential decoding for the entire long file. The two official-style controls also set beam=5 and no_speech_threshold=0.6; external VAD on/off is specified per row.

Sequential context, temperature fallback and timestamp-window behavior differ from batching. The controls did not establish that sequential decoding improves this corpus.


### Exact Prompts And Vocabulary

Whisper W14/W15 used this **initial prompt**, not a separate post-processing model:

```text
Produce a verbatim English transcript with professional capitalization and punctuation. Use commas, periods, and question marks where natural.
```

| Trial | Hotwords |
| --- | --- |
| W18, broad list | Python, TypeScript, async/await, Perforce, COBOL, mypy, API, SQLAlchemy, MongoDB, MongoEngine, Michael Feathers |
| W19, focused list | async/await, MongoDB, MongoEngine |

Granite raw control:

```text
can you transcribe the speech into a written format?
```

Granite punctuation control:

```text
transcribe the speech with proper punctuation and capitalization.
```

The combined programming prompt appended `Keywords:` and the 23 canonical entries:

```text
origin/main, D89FD01, CopyQ, Fish, Firefox, KDE, launcher, UDEV, NVIDIA, environment, ASRock, NCT6683, DKMS, xbindkeys, modprobe, worktree, worktrees, shfmt, staged diff, ShellCheck, SC2088, tilde, bootstrap.sh
```

Parakeet boosting used its separate uppercase phrase file, with context score 1.0 and depth scaling 2.0. It did not use Granite's prompt mechanism. The corrected short test used alpha 0.5; the generic unboosted controls disabled the phrase file.

### Present Production Default

The source still uses **Whisper beam 4**, not beam 2. **Batch 2** applies at 60 seconds or longer; shorter inputs use ordinary single-recording sequential inference. The beam-2 trial is W03, not the selected default.

The raw benchmark profile is W4. The normal service additionally enables conditional source cleanup and US-to-UK post-processing, and uses shared technical canonicalization. Those later stages must not be credited with the raw benchmark scores. Current settings are defined in [whisper_service.py](../whisper_service.py); the original fixed voter settings are in the [consensus report](asr-consensus-benchmark.md#fixed-configuration).

## Original Model Comparison

**R332, 2,060 words.** The early Whisper baseline is beam 5 (65 errors). The later W00 baseline is beam 4 (62 errors). Do not mix those baselines when calculating improvements.

| ID | Configuration | S / D / I | Errors | WER % | Punct. F1 | Sentence F1 | Seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R332-W5 | Whisper original, beam 5 | 17 / 31 / 17 | 65 | 3.1553 | NR | NR | 17.90 | [E001](#e001), [E002](#e002), [E003](#e003), [E004](#e004) |
| R332-PB | Parakeet, phrase boosting alpha 0.5 | 30 / 23 / 35 | 88 | 4.2718 | NR | NR | 5.29 | [E001](#e001), [E002](#e002), [E003](#e003) |
| R332-P0 | Parakeet, no phrase file | 30 / 23 / 35 | 88 | 4.2718 | 0.8087 | 0.8817 | 5.00 | [E004](#e004), [E005](#e005), [E006](#e006) |
| R332-GQ-K | Granite Q8, punctuation + unrelated keywords | 289 / 50 / 27 | 366 | 17.7670 | NR | NR | 20.29 | [E002](#e002) |
| R332-GQ | Granite Q8, punctuation only | 32 / 26 / 38 | 96 | 4.6602 | 0.4743 | 0.5041 | 19.45 | [E004](#e004), [E005](#e005), [E006](#e006) |
| R332-GB | Granite native BF16, punctuation only | 36 / 70 / 22 | 128 | 6.2136 | 0.6240 | NR | 31.65 | [E004](#e004), [E007](#e007), [E006](#e006), [E073](#e073) |
| R332-Q90 | Qwen3-ASR, 90 s chunks | 38 / 24 / 67 | 129 | 6.2621 | NR | NR | 36.02 | [E072](#e072), [E074](#e074) |
| R332-Q60 | Qwen3-ASR, 60 s chunks | 34 / 23 / 53 | 110 | 5.3398 | 0.6932 | 0.7664 | 33.52 | [E008](#e008), [E005](#e005), [E006](#e006), [E075](#e075) |
| R332-CWV-C | CrisperWhisper verbatim, continuation | 62 / 641 / 62 | 765 | 37.1359 | NR | NR | 28.97 | [E008](#e008) |
| R332-CWI-C | CrisperWhisper intended, continuation | 23 / 816 / 4 | 843 | 40.9223 | NR | NR | 23.08 | [E008](#e008) |
| R332-CWV-L | CrisperWhisper verbatim, chunked LCS | 63 / 24 / 98 | 185 | 8.9806 | NR | NR | 26.64 | [E008](#e008) |
| R332-CWI-L | CrisperWhisper intended, chunked LCS | 31 / 58 / 11 | 100 | 4.8544 | 0.7993 | 0.8660 | 24.98 | [E008](#e008), [E005](#e005) |
| R332-C-FIX1 | Cohere, fixed 35 s blocks, first run | 29 / 33 / 19 | 81 | 3.9320 | NR | NR | 9.09 | [E009](#e009), [E076](#e076) |
| R332-C-FIX2 | Cohere, fixed 35 s blocks, repeat | 29 / 33 / 19 | 81 | 3.9320 | NR | NR | 6.40 | [E009](#e009), [E010](#e010) |
| R332-C-OFF | Cohere, official boundaries, microbatch 1 | 26 / 26 / 21 | 73 | 3.5437 | 0.8191 | 0.8696 | 7.05 | [E009](#e009), [E010](#e010), [E011](#e011), [E012](#e012), [E005](#e005), [E006](#e006) |

- **R332-W5:** Original 65-error baseline. Punctuation score retained separately for the saved output and later beam-5 repeat; see W05.
- **R332-PB:** Corrected segmentation. Raw text was identical to the unboosted control; no independent numeric punctuation F1 retained here.
- **R332-P0:** Phrase removal left raw text and 88 errors unchanged.
- **R332-GQ-K:** Seven 90 s chunks; keyword prompt induced origin/origin repetition and unrelated terms. A measured broken-pipeline result, not ordinary Granite accuracy.
- **R332-GQ:** No keywords; raw/final WER both 96 errors. Inconsistent punctuation despite the request.
- **R332-GB:** Punctuation F1 recovered only to four decimal places. Native control did not close the lexical gap.
- **R332-Q90:** First load is not a repeat warm-load measurement.
- **R332-CWV-C:** Large omissions; do not interpret as a normal successful transcription.
- **R332-CWI-C:** Large omissions; do not interpret as a normal successful transcription.
- **R332-CWV-L:** Verbatim output penalized against a cleaned publisher reference.
- **R332-C-FIX1:** Initial 18-block workaround; not official boundary placement.
- **R332-C-FIX2:** Same 81-error transcript; often rounded to 3.93% / 6.40 s in chat.
- **R332-C-OFF:** 19 official chunks, 73 errors (3.54%), superseding the 81-error fixed-block result. CPU processing approximately 2.86 s is additional; about 9.91 s generation plus processing.

For R332-C-OFF, 7.05 s is **generation only**, with approximately 2.86 s CPU processing additional. The 6.40 s result often cited in the chat is the earlier fixed-block repeat, not official-boundary processing. Crisper continuation and Granite keyword-loop scores are retained as failure diagnostics, not fair measures of a functioning pipeline.

### Post-Processing Controls

| ID | Configuration | S / D / I | Errors | WER % | Punct. F1 | Sentence F1 | Seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R332-W5-FINAL | Original Whisper after shared normalization/variant stage | 22 / 31 / 17 | 70 | 3.3981 | 0.8288 | 0.8503 | NR | [E001](#e001), [E002](#e002), [E003](#e003), [E004](#e004), [E051](#e051) |
| R332-PB-FINAL | Boosted Parakeet after normalization/variant stage | 34 / 23 / 35 | 92 | 4.4660 | NR | NR | NR | [E001](#e001), [E002](#e002), [E003](#e003) |
| R332-FULLSTOP | Whisper plus FullStop punctuation restoration | NR / NR / NR | 65 | 3.1553 | 0.6036 | 0.6690 | NR | [E031](#e031), [E025](#e025) |

- **R332-W5-FINAL:** 70 errors versus raw 65; reference uses spoken spellings. This is not evidence normalization always damages real technical identifiers.
- **R332-PB-FINAL:** 92 errors versus raw 88. Other recovered primary controls kept the same raw/final WER.
- **R332-FULLSTOP:** 65 lexical errors retained, but punctuation F1 fell to 0.6036. Extra restoration time not retained.

The original post-processed Whisper punctuation score was recovered from `faster-whisper.json`, whose WER was 70 errors; it is not silently assigned to the original raw 65-error result. The later raw beam-5 repeat W05 independently retained the same numeric punctuation F1.

### Punctuation Counts Where Available

These are **output counts, not correctness scores**. They provide a limited punctuation observation for runs with no preserved F1.

| Trial | Periods | Commas | Questions | Exclamations |
| --- | ---: | ---: | ---: | ---: |
| R332-W5 | 146 | 107 | 19 | 0 |
| R332-P0 | 145 | 141 | 16 | 0 |
| R332-GQ | 60 | 49 | 9 | 0 |
| R332-GB | 101 | 96 | 12 | 0 |
| R332-Q90 | 122 | 155 | 19 | 0 |
| R332-C-FIX1 | 147 | 116 | 17 | 0 |
| R332-C-FIX2 | 147 | 116 | 17 | 0 |

## Whisper Parameter Sweep

**R332.** Every row inherits W4, changing only the named values. W16-W22 explicitly use batch 4, not batch 2. Runtime is model transcription, including consumption of faster-whisper's lazy segment generator.

| ID | Configuration | S / D / I | Errors | WER % | Punct. F1 | Sentence F1 | Seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| W00 | Baseline: beam 4, batch 2 | 18 / 27 / 17 | 62 | 3.0097 | 0.8387 | 0.8529 | 17.22 | [E013](#e013), [E014](#e014), [E015](#e015), [E016](#e016), [E017](#e017), [E018](#e018), [E019](#e019), [E020](#e020), [E021](#e021), [E022](#e022), [E023](#e023), [E024](#e024), [E025](#e025), [E026](#e026), [E027](#e027), [E028](#e028), [E006](#e006) |
| W01 | Batch 3 | 18 / 27 / 17 | 62 | 3.0097 | 0.8387 | 0.8529 | 14.66 | [E029](#e029), [E018](#e018), [E030](#e030) |
| W02 | Batch 4 | 18 / 27 / 17 | 62 | 3.0097 | 0.8387 | 0.8529 | 13.21 | [E029](#e029), [E018](#e018), [E022](#e022), [E030](#e030) |
| W03 | Beam 2, batch 2 | NR / NR / NR | 67 | 3.2524 | 0.8566 | 0.8772 | 16.61 | [E020](#e020), [E030](#e030) |
| W04 | Beam 3, batch 2 | NR / NR / NR | 62 | 3.0097 | 0.8372 | 0.8529 | 16.57 | [E020](#e020), [E030](#e030) |
| W05 | Beam 5, batch 2 | 17 / 31 / 17 | 65 | 3.1553 | 0.8288 | 0.8503 | 17.57 | [E013](#e013), [E018](#e018), [E030](#e030) |
| W06 | Beam 8, batch 2 | 18 / 39 / 17 | 74 | 3.5922 | 0.8117 | 0.8399 | 19.28 | [E013](#e013), [E018](#e018), [E030](#e030) |
| W07 | No-speech threshold 0.6 | 18 / 27 / 17 | 62 | 3.0097 | 0.8387 | 0.8529 | 17.29 | [E013](#e013), [E018](#e018), [E030](#e030) |
| W08 | VAD minimum silence 500 ms | 18 / 27 / 17 | 62 | 3.0097 | 0.8387 | 0.8529 | 17.31 | [E029](#e029), [E030](#e030) |
| W09 | VAD threshold 0.10 | 23 / 29 / 17 | 69 | 3.3495 | 0.8307 | 0.8452 | 16.63 | [E029](#e029), [E030](#e030) |
| W10 | VAD threshold 0.20 | 27 / 32 / 17 | 76 | 3.6893 | 0.8401 | 0.8943 | 17.47 | [E029](#e029), [E030](#e030) |
| W11 | VAD threshold 0.20, padding 600 ms | 25 / 31 / 17 | 73 | 3.5437 | 0.8486 | 0.9036 | 17.77 | [E029](#e029), [E030](#e030) |
| W12 | VAD padding 600 ms | 18 / 29 / 30 | 77 | 3.7379 | 0.8131 | 0.8389 | 17.00 | [E029](#e029), [E030](#e030) |
| W13 | VAD padding 800 ms | 20 / 26 / 32 | 78 | 3.7864 | 0.8037 | 0.8261 | 16.89 | [E029](#e029), [E030](#e030) |
| W14 | Verbatim/punctuation style initial prompt | 13 / 403 / 11 | 427 | 20.7282 | 0.7181 | 0.7692 | NR | [E013](#e013), [E018](#e018) |
| W15 | Same style prompt, beam 5 | 14 / 438 / 13 | 465 | 22.5728 | 0.7102 | 0.7548 | NR | [E013](#e013) |
| W16 | Batch 4, no VAD | NR / NR / NR | 88 | 4.2718 | 0.7941 | 0.8822 | 47.69 | [E021](#e021), [E030](#e030) |
| W17 | Batch 4, timestamp-token decoding | NR / NR / NR | 104 | 5.0485 | 0.7140 | 0.7450 | 18.32 | [E022](#e022), [E030](#e030) |
| W18 | Batch 4, broad 11-term hotwords | 30 / 111 / 7 | 148 | 7.1845 | 0.7853 | 0.8272 | 21.38 | [E023](#e023), [E030](#e030) |
| W19 | Batch 4, focused 3-term hotwords | NR / NR / NR | 74 | 3.5922 | 0.8479 | 0.8776 | 19.46 | [E024](#e024), [E030](#e030) |
| W20 | Batch 4, length penalty 1.05 | 18 / 27 / 17 | 62 | 3.0097 | 0.8387 | 0.8529 | 14.61 | [E026](#e026), [E030](#e030) |
| W21 | Batch 4, length penalty 1.10 | NR / NR / NR | 62 | 3.0097 | 0.8387 | 0.8529 | 13.16 | [E027](#e027), [E030](#e030) |
| W22 | Batch 4, patience 1.2 | NR / NR / NR | 62 | 3.0097 | 0.8372 | 0.8529 | 16.39 | [E028](#e028), [E030](#e030) |

- **W01:** Same 62 word errors as baseline.
- **W02:** Same 62 word errors on this excerpt; three more errors over ALL4.
- **W03:** Better punctuation here, worse WER.
- **W04:** Same WER, slightly worse punctuation.
- **W05:** Repeat of earlier beam-5 configuration; timing differs from the original run.
- **W14:** 403 deletions; inappropriate initial prompt caused major omissions.
- **W15:** 438 deletions; major omissions.
- **W17:** without_timestamps=false, not word_timestamps=true.
- **W18:** 111 deletions; list in profile notes below.

No tested single-setting change improved both target metrics sufficiently to replace W4 across the validated corpus. Beam 2 improved punctuation on this excerpt but added five word errors. Batch 4 was faster, not strictly accuracy-equivalent. The style prompt and hotword trials show why an ASR prompt must not be treated as a safe instruction to improve prose.

## Long-Recording Comparisons

Cohere seconds below are **generation only**; for S344, the separate 4.53 s CPU processor brings processing time to 51.80 s. Full-consensus rows are merges of retained outputs; complete request timings were not recovered.

### S344

| ID | Configuration | S / D / I | Errors | WER % | Punct. F1 | Sentence F1 | Seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| S344-W | Whisper beam 4, batch 2 | 280 / 279 / 180 | 739 | 5.3788 | 0.8109 | 0.8374 | 114.00 | [E032](#e032), [E033](#e033), [E034](#e034), [E035](#e035), [E036](#e036) |
| S344-C | Cohere official boundaries, greedy | 364 / 289 / 254 | 907 | 6.6016 | 0.7781 | 0.8303 | 47.27 | [E037](#e037), [E038](#e038), [E033](#e033), [E036](#e036) |
| S344-P | Parakeet unboosted, greedy | 458 / 260 / 734 | 1452 | 10.5685 | 0.7267 | 0.7643 | 31.25 | [E037](#e037), [E033](#e033), [E036](#e036) |
| S344-R | Full three-model consensus | 278 / 260 / 194 | 732 | 5.3279 | 0.8115 | 0.8463 | NR | [E033](#e033), [E034](#e034) |

- **S344-C:** 47.27 s generation plus 4.53 s processor = 51.80 s processing quoted earlier.
- **S344-R:** Score from merged saved voter outputs; complete request duration was not retained.

### R400

| ID | Configuration | S / D / I | Errors | WER % | Punct. F1 | Sentence F1 | Seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R400-W | Whisper beam 4, batch 2 | 263 / 418 / 128 | 809 | 6.0754 | 0.7779 | 0.8178 | 104.23 | [E032](#e032), [E039](#e039), [E040](#e040), [E041](#e041), [E042](#e042), [E036](#e036) |
| R400-C | Cohere official boundaries, greedy | 268 / 403 / 127 | 798 | 5.9928 | 0.7864 | 0.8568 | 47.93 | [E039](#e039), [E036](#e036) |
| R400-P | Parakeet unboosted, greedy | 294 / 358 / 527 | 1179 | 8.8540 | 0.7565 | 0.8065 | 31.78 | [E039](#e039), [E036](#e036) |
| R400-R | Full three-model consensus | 227 / 309 / 111 | 647 | 4.8588 | 0.8200 | 0.8706 | NR | [E039](#e039) |

- **R400-C:** Cohere beats Whisper on both scores on this individual recording; it does not beat the four-corpus Whisper aggregate.
- **R400-R:** Score from merged saved voter outputs; complete request duration was not retained.

### R487

| ID | Configuration | S / D / I | Errors | WER % | Punct. F1 | Sentence F1 | Seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R487-W | Whisper beam 4, batch 2 | 150 / 213 / 165 | 528 | 4.2632 | 0.7585 | 0.7853 | 115.97 | [E043](#e043), [E041](#e041), [E036](#e036) |
| R487-C | Cohere official boundaries, greedy | 174 / 248 / 121 | 543 | 4.3843 | 0.8129 | 0.8635 | 46.18 | [E043](#e043), [E036](#e036) |
| R487-P | Parakeet unboosted, greedy | 201 / 136 / 527 | 864 | 6.9762 | 0.7825 | 0.8195 | 29.21 | [E043](#e043), [E036](#e036) |
| R487-R | Full three-model consensus | 125 / 186 / 115 | 426 | 3.4396 | 0.8423 | 0.8834 | NR | [E043](#e043) |

- **R487-R:** Score from merged saved voter outputs; complete request duration was not retained.

### Sequential And Batch-Size Controls

Corpus is included in each row ID. R400 sequential controls were measured, not merely recommended:

| ID | Configuration | S / D / I | Errors | WER % | Punct. F1 | Sentence F1 | Seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R400-SEQ | Sequential, existing settings | 315 / 163 / 450 | 928 | 6.9691 | 0.4353 | 0.4523 | 242.54 | [E040](#e040), [E044](#e044) |
| R400-SEQ-NOVAD | Sequential, no VAD, no-speech 0.6, beam 5 | 327 / 349 / 372 | 1048 | 7.8702 | 0.7493 | 0.7712 | 272.25 | [E040](#e040), [E044](#e044) |
| R400-SEQ-VAD | Sequential, VAD, no-speech 0.6, beam 5 | 361 / 185 / 433 | 979 | 7.3521 | 0.7185 | 0.7425 | 327.40 | [E040](#e040), [E044](#e044) |
| S344-B4 | Whisper beam 4, batch 4 | 280 / 278 / 180 | 738 | 5.3716 | 0.8110 | 0.8374 | 88.12 | [E035](#e035) |
| R400-B3 | Whisper beam 4, batch 3 | 266 / 420 / 129 | 815 | 6.1205 | 0.7782 | 0.8172 | 96.80 | [E042](#e042) |
| R400-B4 | Whisper beam 4, batch 4 | 266 / 418 / 129 | 813 | 6.1054 | 0.7784 | 0.8174 | 85.29 | [E041](#e041), [E042](#e042) |
| R487-B4 | Whisper beam 4, batch 4 | 150 / 213 / 165 | 528 | 4.2632 | 0.7587 | 0.7858 | 81.04 | [E041](#e041) |

- **R400-SEQ:** Ordinary sequential WhisperModel; no long-file batching. The latter two controls also restore sequential temperature/context heuristics, so they are not a one-variable VAD comparison.
- **R400-SEQ-NOVAD:** Ordinary sequential WhisperModel; no long-file batching. The latter two controls also restore sequential temperature/context heuristics, so they are not a one-variable VAD comparison.
- **R400-SEQ-VAD:** Ordinary sequential WhisperModel; no long-file batching. The latter two controls also restore sequential temperature/context heuristics, so they are not a one-variable VAD comparison.
- **R487-B4:** Same word-error counts as batch 2; punctuation changes slightly.

The measured Ruff results contradict the later generic advice to disable batching for maximum accuracy. Existing sequential settings were slower and less accurate; the no-VAD/0.6/beam-5 bundle was worse still. This does not prove sequential decoding always loses, only that the proposed change failed this control.

Across all four recordings, batch 4 reduced the sum of transcription timers from **351.42 s to 267.67 s (23.83% less time)**, with **three additional errors**. The per-recording changes were 0, -1, +4 and 0 errors respectively.

## Cohere Parameter Controls

**S344, 13,739 words.** Times here include **processor plus generation**, unlike the generation-only model table above. C0 is the same underlying baseline as S344-C, not another independent experiment.

| ID | Configuration | S / D / I | Errors | WER % | Punct. F1 | Sentence F1 | Seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C0 | Greedy baseline | 364 / 289 / 254 | 907 | 6.6016 | 0.7781 | 0.8303 | 51.80 | [E038](#e038), [E033](#e033), [E045](#e045) |
| C1 | No VAD | 364 / 289 / 254 | 907 | 6.6016 | NR | NR | 51.53 | [E045](#e045) |
| C2 | Feature normalization on | 364 / 289 / 254 | 907 | 6.6016 | NR | NR | 51.62 | [E045](#e045) |
| C3 | Punctuation disabled | 482 / 173 / 537 | 1192 | 8.6760 | NR | NR | 47.83 | [E046](#e046) |
| C4 | Beam 2, length penalty 1.0 | 340 / 295 / 221 | 856 | 6.2304 | NR | NR | 63.00 | [E047](#e047) |
| C5 | Beam 4, length penalty 1.0 | 326 / 297 / 218 | 841 | 6.1213 | NR | NR | 79.58 | [E048](#e048) |
| C6 | Beam 4, length penalty 1.05 | 326 / 296 / 218 | 840 | 6.1140 | NR | NR | 79.62 | [E049](#e049) |
| C7 | Beam 4, length penalty 1.10 | 326 / 296 / 218 | 840 | 6.1140 | NR | NR | 79.87 | [E050](#e050) |

- **C0:** Same run as S344-C, but total processing (not generation-only).
- **C1:** Same non-empty transcript; one extra empty final chunk; VAD only trimmed outer edges.
- **C2:** All 256-token runs reached EOS; no output-cap truncation.
- **C3:** Punctuation intentionally off; no independently retained numeric punctuation F1.
- **C4:** All 256-token runs reached EOS; no output-cap truncation.
- **C5:** All 256-token runs reached EOS; no output-cap truncation.
- **C6:** All 256-token runs reached EOS; no output-cap truncation.
- **C7:** All 256-token runs reached EOS; no output-cap truncation.

All retained 256-token runs reached EOS without cap hits. Turning VAD off changed no non-empty transcript, confirming that the tested VAD path did not remove all internal pauses. Numerical punctuation F1 was not retained for the beam/normalization controls: requesting punctuation alone is not a measured F1 result.

## Consensus And Production Checks

Default corpus **R332**; V07 is S344. Voting pilots reused transcripts and have no retained full-pipeline timer unless shown. V06/V07 and V08/V09/V13 are **cold pipeline durations**, including worker/model startup.

| ID | Configuration | S / D / I | Errors | WER % | Punct. F1 | Sentence F1 | Seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V01 | W4 + Cohere + Parakeet | 20 / 27 / 14 | 61 | 2.9612 | 0.8688 | 0.9036 | NR | [E014](#e014), [E051](#e051), [E017](#e017), [E019](#e019) |
| V02 | Whisper counted twice + Cohere + Parakeet | 18 / 27 / 17 | 62 | 3.0097 | 0.8387 | 0.8529 | NR | [E014](#e014) |
| V03 | Whisper beams 4 + 5 + 8 | 17 / 31 / 17 | 65 | 3.1553 | 0.8288 | 0.8503 | NR | [E014](#e014), [E017](#e017) |
| V04 | Whisper beams 4 + 5 + 8 + Cohere + Parakeet | 17 / 28 / 17 | 62 | 3.0097 | 0.8309 | 0.8503 | NR | [E014](#e014), [E017](#e017) |
| V05 | Whisper + Cohere + Parakeet + Qwen60 + Crisper intended LCS | 19 / 23 / 13 | 55 | 2.6699 | 0.8663 | 0.9024 | NR | [E014](#e014) |
| V06 | Adaptive disagreement-only Cohere | 21 / 25 / 14 | 60 | 2.9126 | 0.8541 | 0.8854 | 45.86 | [E019](#e019), [E030](#e030) |
| V07 | Adaptive disagreement-only Cohere (SQLAlchemy) | 278 / 282 / 173 | 733 | 5.3352 | 0.8152 | 0.8443 | 189.06 | [E034](#e034) |
| V08 | Initial production consensus | 20 / 25 / 17 | 62 | 3.0097 | 0.8587 | 0.8943 | 57.75 | [E051](#e051), [E030](#e030) |
| V09 | Production alignment retry | 20 / 25 / 17 | 62 | 3.0097 | 0.8587 | 0.8943 | 49.79 | [E052](#e052), [E030](#e030) |
| V10 | Initial production Parakeet voter | 37 / 22 / 93 | 152 | 7.3786 | 0.7601 | 0.8205 | NR | [E053](#e053) |
| V11 | Corrected production Parakeet accuracy voter | 30 / 23 / 35 | 88 | 4.2718 | 0.8087 | 0.8817 | NR | [E054](#e054) |
| V12 | Corrected production consensus | 20 / 27 / 14 | 61 | 2.9612 | 0.8688 | 0.9036 | NR | [E015](#e015) |
| V13 | Final isolated production consensus | 20 / 27 / 14 | 61 | 2.9612 | 0.8688 | 0.9036 | 78.64 | [E016](#e016), [E030](#e030) |

- **V02:** Four votes: W4, W4, Cohere, Parakeet. No improvement over W4 baseline.
- **V03:** Three different beam outputs of the same recognizer; not three independent model families.
- **V05:** Five-voter pilot on one excerpt; no four-corpus validation, complete duration, or pipeline memory measurement.
- **V08:** Before voter settings were fully aligned; not the final validated pipeline.
- **V09:** Still 62 rather than final 61 errors.
- **V10:** Wrong production preset relative to the benchmark; retained to explain the 152-error outlier.
- **V11:** Reproduced the unboosted 88-error benchmark.
- **V12:** Reproduced the 61-error benchmark; final timed repeat is V13.
- **V13:** Includes isolated worker startup and model loading. 78.64 s is not warm inference time.

The five-voter pilot's 55-error result was not validated on all four recordings or against the whole-process 5 GiB constraint. It is not evidence to replace the validated three-voter configuration. The wrong-preset production attempts remain in the catalogue so they cannot later be mistaken for the final Parakeet or consensus result.

## Short Technical Dictation

**TECH32, normally 32.70 s. No verified reference or numeric WER/F1.** These are qualitative readability/identifier checks. Original model-loading times are in the CSV/JSON (Whisper 4.11 s; Parakeet 7.31 s; Granite 1.28 s).

| ID | Trial | Profile | Seconds | Outcome / limitation | Evidence |
| --- | --- | --- | ---: | --- | --- |
| T01 | Whisper original beam-5 passage | W5 | 1.49 | Most legible sentence punctuation; identifiers still in spoken form; UK conversion on, before shared canonicalization. | [E058](#e058) |
| T02 | Parakeet greedy, 25 s split + 2 s overlap | P25 | 0.77 | Duplicated/corrupted boundary: staged diff chip / HFMT and staged diff checks passed. Irregular capitalization and sentence punctuation. | [E058](#e058) |
| T03 | Granite Q8, combined punctuation + 23 keywords | GQ | 0.84 | No sentence punctuation; canonical identifiers are partly deterministic post-processing. Single 32.70 s request, not seven chunks. | [E058](#e058) |
| T04 | Whisper after shared canonical normalization | W5 | 1.45 | Readable punctuation retained; canonical names fixed. Same audio, no true reference WER. | [E059](#e059) |
| T05 | Parakeet single-pass 120 s cap, alpha 0.5 | P120 | 0.74 | No forced split for this 32.70 s request; overlap duplication removed; still worktree as clean. Long comma-joined sentences. | [E059](#e059) |
| T06 | Granite Q8 repeat, raw + normalized output | GQ | 0.72 | Still unpunctuated; raw identifiers are less canonical than the normalized output. | [E060](#e060) |
| T07 | Original Parakeet malsd_batch beam 5 | P-BEAM-OLD | NR | Technical passage misrecognized ASRock, xbindkeys and SC2088; qualitative comparison only. Exact duration for this earlier user recording was not independently retained. | [E061](#e061) |
| T08 | Earlier boosted-greedy Parakeet user retry | P25 | NR | User reported ASROCKT6683 and MODPROBESICIAN. No saved reference score or runtime; exact effective boosting alpha for that specific retry is not established. | [CHAT](#chat) |
| T09 | Granite HTTP raw-prompt control | GQ | NR | Prompt: can you transcribe the speech into a written format? No sentence punctuation on this passage; no numeric punctuation score or separate duration retained. | [E062](#e062) |
| T10 | Granite HTTP punctuation-only control | GQ | NR | Prompt: transcribe the speech with proper punctuation and capitalization. No sentence punctuation on this passage; no numeric punctuation score or separate duration retained. | [E062](#e062) |
| T11 | Granite HTTP combined punctuation/keywords control | GQ | NR | Punctuation prompt plus canonical programming keywords. No sentence punctuation on this passage; no numeric punctuation score or separate duration retained. | [E062](#e062) |
| T12 | Granite direct llama-cli control | GQ | NR | Bypassed the HTTP wrapper; raw/punctuation/combined testing was described with llama.cpp b10221. No sentence punctuation on this passage; no numeric punctuation score or separate duration retained. | [E062](#e062) |

Identifier spelling in service output can come from deterministic canonicalization, not the recognizer. The early Parakeet overlap defect was a pipeline assembly failure, not solely an acoustic recognition error.

## Failures And Unscored Backends

**No valid WER/F1 for these entries.** A request returning an error is not a 100% WER measurement. Recovery/retry copies of the same failed recording are not new independent test clips.

| ID | Trial | Profile | Seconds | Outcome / limitation | Evidence |
| --- | --- | --- | ---: | --- | --- |
| F01 | Canary-Qwen initial CPU-loaded model | CANARY | NR | CPU transcription succeeded but was slow and used many CPU cores. No fair reference/audio-runtime benchmark. | [CHAT](#chat) |
| F02 | Canary-Qwen explicit CUDA transfer | CANARY | NR | CUDA OOM during transfer on a 7.56 GiB GPU with other workloads. The log's process memory at failure is not a successful model peak. | [CHAT](#chat) |
| F03 | Canary dependency/output handling attempts | CANARY | NR | Missing NeMo, then tensor truth-value handling failure; fixed before successful CPU transcripts. No WER can be assigned. | [CHAT](#chat) |
| F04 | Parakeet CUDA sublibrary mismatch | P-BEAM-OLD | NR | CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH; requests preserved/retried. Runtime/dependency failure, not a recognition-quality score. | [CHAT](#chat) |
| F05 | Granite Q8, whole ten-minute request | GQ | NR | 6111 input prompt tokens exceeded fixed 4096 context; no valid completed transcript. | [E063](#e063) |
| F06 | Granite Q8, larger context request | GQ | NR | Runtime capped context back to 4096; did not solve whole-file input failure. | [E063](#e063) |
| F07 | Cohere gated checkpoint download | C-OFFICIAL | NR | 401 then 403 before terms/account access took effect; later official-weight runs succeeded. | [E064](#e064) |
| F08 | Cohere full GPU batch, all chunks at once | C-OFFICIAL | NR | All-chunk batch OOM on 8 GiB; replaced first by fixed-block sequential processing, then official CPU chunks with microbatch 1. | [E065](#e065) |
| F09 | Higgs Audio v3 STT Q8_0 | HIGGS | 186.00 | Whole-file repetition failure; sampled process VRAM 5716 MiB, above 5120 MiB limit; no recovered WER or punctuation F1. | [E066](#e066) |
| F10 | Whisper CPU int8 fallback | W4 | NR | Operational GPU OOM recovery was exercised; no matched CPU-vs-GPU WER/punctuation result retained. Device/compute override: CPU/int8, not the CUDA/FP16 W4 profile. | [CHAT](#chat) |

## Microphone Preprocessing

**No reference-scored WER or punctuation result was demonstrated.** Seconds in this section measure source analysis/filtering only, not ASR.

| ID | Trial | Profile | Seconds | Outcome / limitation | Evidence |
| --- | --- | --- | ---: | --- | --- |
| M01 | Bad historical ~12 s clip, unfiltered vs HP80 | MIC-HP80 | NR | Sub-60 Hz power 77.5% -> 4.3%; DC offset about 0.015. Whisper text unchanged despite signal cleanup; no verified reference WER. | [E067](#e067) |
| M02 | Clean historical ~32.7 s clip, filtering bypassed | MIC-HP80 | 0.055 | Normal clean input did not trigger filtering; about 55 ms analysis/validation. Historical speech sub-60 Hz ratio reported as 1.97%. | [E068](#e068) |
| M03 | Clean ~8 s clip, filtering bypassed | MIC-HP80 | 0.050 | About 50 ms analysis/validation; no reference-scored WER. | [E068](#e068) |
| M04 | Bad ~12 s clip, automatic filter enabled | MIC-HP80 | 0.070 | About 70 ms including detection, filtering and validation, NOT total transcription time. | [E068](#e068) |
| M05 | baseline.wav gain sweep | MIC-HP80 | NR | Preferred eighth gain increment at 29.34-31.82 s: peak -8.7 dBFS, speech RMS -23 dBFS. Seventh: -11.7/-24.6; ninth: -6.4/-20.8. Selection based on headroom, not WER. | [E069](#e069) |
| M06 | baseline2.wav, folded towel mount | MIC-HP80 | NR | Speech peak -11.41 dBFS, whole-file peak -8.61 dBFS; no speech clipping. Speech sub-60 Hz share 3.85% versus 14.9% without towel; different takes/gain prevent a causal WER claim. | [E070](#e070) |
| M07 | baseline/baseline2 automatic-filter on/off checks | MIC-HP80 | NR | Neither recording triggered the filter; each recording produced the same transcript with the option on/off. This does NOT mean the two different recordings have identical words. | [E067](#e067) |
| M08 | Historical recovery-file audit | MIC-HP80 | NR | Only three unique historical recordings were available; other copies were retries. No matched towel/no-towel reference corpus; retaining 20 recordings enables future evaluation. | [E071](#e071) |

The high-pass improved measured low-frequency contamination in one bad source, but its transcript stayed identical. The towel comparison changes the take and gain as well as the mount, so it cannot establish a causal WER improvement. Ratio measurements from whole-file spectra and the active-frame detector are different quantities and should not be interchanged.

## Discussed-But-Unmeasured Options

These entries are kept separate specifically to avoid converting chat suggestions or available flags into claimed experiments.

| ID | Trial | Profile | Seconds | Outcome / limitation | Evidence |
| --- | --- | --- | ---: | --- | --- |
| N01 | ARK-ASR-3B | NONE | NR | Mentioned/screened as a larger-GPU candidate; no local transcript/score recovered. | [CHAT](#chat) |
| N02 | MOSS Transcribe Preview 2B | NONE | NR | Discussed and screened; no local benchmark result recovered. | [CHAT](#chat) |
| N03 | Hojo-ASR-V1 | NONE | NR | Suggested candidate; no completed local quality benchmark recovered. | [CHAT](#chat) |
| N04 | Native OpenAI Whisper PyTorch control | NONE | NR | Proposed, but the recorded sequential controls use faster-whisper/CTranslate2, not OpenAI's PyTorch implementation. | [CHAT](#chat) |
| N05 | Parakeet programming / rustly-pocket presets | P120 | NR | Available configurations, not independent scored rows in the recovered benchmark. Programming: VAD off; pocket: VAD 250 ms speech / 600 ms silence / 450 ms pad, 25 s segments and 2 s forced overlap. Do not assign P120 scores to them. | [CHAT](#chat) |
| N06 | Parakeet accuracy-beam-experimental | P120 | NR | Available later malsd_batch beam-5 / CUDA-graphs-on preset with corrected 120 s segmentation; no controlled score recovered. Distinct from the early graphs-off P-BEAM-OLD passage. | [CHAT](#chat) |
| N07 | Boost alpha/context/phrase-list size sweeps | P120 | NR | Alpha 0.5 and removal of the phrase file are measured above; the proposed complete 0/0.5/1/1.5 greedy-and-beam matrix has no complete retained results. | [CHAT](#chat) |
| N08 | Further Whisper sweeps | W4 | NR | No distinct recovered scores for no-speech 0.45, beam 10/16, a systematic context-on/off pair, repetition penalties, length penalties below 1, or hallucination-silence thresholds. Do not confuse suggestions with W00-W22. | [CHAT](#chat) |
| N09 | Further Cohere controls | C-OFFICIAL | NR | No recovered scores for backward boundary search 8/10 s, channel-by-channel audit, prefix-trie boosting, N-best rescoring or domain LoRA. EOS audit did not justify token-cap increases. | [CHAT](#chat) |
| N10 | Cohere-first selective Whisper cascade | NONE | NR | Proposed only in the supplied discussion; implemented ADAPTIVE instead runs full Whisper/Parakeet and selective Cohere. | [CHAT](#chat) |
| N11 | LLM transcript/punctuation rewriting, fine-tuning, external LM | NONE | NR | Llama 3.1-style rewriting, Whisper-CD, LoRA and n-gram LM were suggestions, not measured alternatives. FullStop and deterministic canonicalization were actually tested and are listed separately. | [CHAT](#chat) |
| N12 | Generic gain boost, band-pass, denoising or compression | MIC-HP80 | NR | Discussed as possibilities; no representative-corpus WER win was demonstrated. Only the specific gain/mount/high-pass experiments above have retained local evidence. | [CHAT](#chat) |

## Memory And Timing

### VRAM Evidence

| Configuration | Peak MiB | GiB (MiB / 1024) | Measurement |
| --- | ---: | ---: | --- |
| Whisper W4, batch 2 | 4380.00 | 4.277 | Sampled driver process memory |
| Whisper W4, batch 4 | 4924.00 | 4.809 | Sampled driver process memory |
| Cohere official microbatch 1 | 4348.00 | 4.246 | Sampled driver process memory |
| Parakeet unboosted accuracy | 2082.00 | 2.033 | Sampled driver process memory |
| Three isolated voters (not resident together) | 4380.00 | 4.277 | Sampled driver process memory |
| Higgs Q8 failed run | 5716.00 | 5.582 | Sampled driver process memory |
| Granite native BF16, punctuation only | 4665.38 | 4.556 | PyTorch allocated only |
| Qwen3-ASR, 90 s chunks | 4575.21 | 4.468 | PyTorch allocated only |
| Qwen3-ASR, 60 s chunks | 4354.19 | 4.252 | PyTorch allocated only |
| Cohere, fixed 35 s blocks, first run | 4059.60 | 3.964 | PyTorch allocated only |
| Cohere, fixed 35 s blocks, repeat | 4059.60 | 3.964 | PyTorch allocated only |
| Cohere, official boundaries, microbatch 1 | 4058.97 | 3.964 | PyTorch allocated only |

Driver measurements sampled `nvidia-smi` every 50 ms for the worker and descendants. The 5 GiB limit is **5,120 MiB**. The isolated three-voter result peaked at 4,380 MiB; Higgs failed the limit. A sampled peak is not a universal worst-case guarantee for all inputs.

PyTorch allocation omits driver/context and non-PyTorch memory, so it cannot independently certify the whole-process cap. Some earlier prose called 4,059 MiB approximately "4.06 GiB"; the correct binary conversion is approximately **3.964 GiB**, while the later measured Cohere driver peak is **4.246 GiB**. The same distinction applies to Granite and Qwen.

### Cold Load Versus Recognition

A single `load_seconds + transcribe_seconds` sum does not include every startup or request stage. For Cohere it can also omit CPU processor time.

| Trial | Model load s | Recorded operation s | Additional CPU processor s | Operation scope |
| --- | ---: | ---: | ---: | --- |
| R332-W5 | 1.97 | 17.90 | NR | transcription |
| R332-PB | 6.85 | 5.29 | NR | transcription |
| R332-P0 | 6.88 | 5.00 | NR | transcription |
| R332-GQ-K | 1.03 | 20.29 | NR | transcription |
| R332-GQ | 1.06 | 19.45 | NR | transcription |
| R332-GB | 3.06 | 31.65 | NR | transcription |
| R332-Q90 | 38.69 | 36.02 | NR | transcription |
| R332-Q60 | 3.20 | 33.52 | NR | transcription |
| R332-CWV-C | 6.75 | 28.97 | NR | transcription |
| R332-CWI-C | 6.75 | 23.08 | NR | transcription |
| R332-CWV-L | 1.79 | 26.64 | NR | transcription |
| R332-CWI-L | 1.79 | 24.98 | NR | transcription |
| R332-C-FIX1 | 3.58 | 9.09 | NR | transcription |
| R332-C-FIX2 | 3.33 | 6.40 | NR | transcription |
| R332-C-OFF | 1.29 | 7.05 | 2.86 | generation |
| W00 | 1.65 | 17.22 | NR | transcription |
| S344-W | 3.21 | 114.00 | NR | transcription |
| S344-C | 1.32 | 47.27 | 4.53 | generation |
| S344-P | 7.17 | 31.25 | NR | transcription |
| R400-W | 1.88 | 104.23 | NR | transcription |
| R400-C | 2.35 | 47.93 | NR | generation |
| R400-P | 7.50 | 31.78 | NR | transcription |
| R487-W | 2.30 | 115.97 | NR | transcription |
| R487-C | 1.20 | 46.18 | NR | generation |
| R487-P | 6.11 | 29.21 | NR | transcription |
| C0 | 1.32 | 51.80 | NR | processor_plus_generation |
| T01 | 4.11 | 1.49 | NR | transcription |
| T02 | 7.31 | 0.77 | NR | transcription |
| T03 | 1.28 | 0.84 | NR | transcription |

Network upload, desktop/phone file preparation, queue wait, UK conversion, clipboard insertion and TTS are not included in model-only runtime. The 50-70 ms automatic-filter measurements are an additional preprocessing cost when enabled. No new live latency measurements were made while compiling this document.

## Evidence Register

Priority: **numeric archived tool output**, then the retained report/source, then final chat summaries, then supplied user logs. Proposals and externally quoted WERs are not treated as local test results.

The original benchmark directory `/tmp/assistant-asr-benchmark` was absent at reconstruction time. The JSON preserves recovered scalar evidence, historical artifact basenames and exact available numeric precision; it does **not** claim to restore missing audio, complete transcript files, model snapshots, environment snapshots or all scorer source code.

Local archive used:

```text
/home/lewis/.codex/sessions/2026/06/18/rollout-2026-06-18T03-46-17-019ed89f-4824-7ad3-a9cb-57d6ac5b9c90.jsonl
```

Evidence IDs identify an exact archived response by UTC timestamp and tool call ID where available. Some summaries combined values from earlier records; multiple IDs retain that provenance. The source archive is local and is not copied into the repository. The companion JSON stores selected numeric records, not the full private conversation.

| ID | UTC timestamp | Evidence type | Tool call ID |
| --- | --- | --- | --- |
| <a id="e001"></a>E001 | 2026-08-05T11:38:01.251Z | archived_tool_output | call_ZMkcHKogwPymW1DVuKiKsXaF |
| <a id="e002"></a>E002 | 2026-08-05T11:39:49.363Z | archived_tool_output | call_bOlQ6xJSDOPKxOvgDl0yAAb2 |
| <a id="e003"></a>E003 | 2026-08-05T11:41:12.059Z | archived_tool_output | call_zUXCDgBCVAg6nOMaHHUKi4Ro |
| <a id="e004"></a>E004 | 2026-08-05T12:20:34.615Z | archived_tool_output | call_6S0FOpa0tSPxv90x279SlY0T |
| <a id="e005"></a>E005 | 2026-08-12T17:40:25.796Z | archived_tool_output | call_pyyj06PMSLbE3lVSzTNfyPxs |
| <a id="e006"></a>E006 | 2026-08-12T21:32:14.789Z | archived_tool_output | call_qHLSqnW4KClOzouxsSs61jsz |
| <a id="e007"></a>E007 | 2026-08-12T21:32:05.363Z | archived_tool_output | call_hxZVbNM6pQqhDJpNZpQtk2Qp |
| <a id="e008"></a>E008 | 2026-08-12T17:17:22.624Z | archived_tool_output | Not a tool result |
| <a id="e009"></a>E009 | 2026-08-05T14:44:28.699Z | archived_tool_output | call_MB1wpVn2Bq0qp7C7nfzqLdfE |
| <a id="e010"></a>E010 | 2026-08-05T14:45:13.753Z | archived_tool_output | call_dEgWf1vEk6IrBCC7U75MHRJ5 |
| <a id="e011"></a>E011 | 2026-08-05T14:46:45.856Z | archived_tool_output | call_7GJxxrzPhv179N32j0JbJWn0 |
| <a id="e012"></a>E012 | 2026-08-07T06:20:46.726Z | archived_tool_output | call_qx1j1vrUB3yVPoVoOInY5EkW |
| <a id="e013"></a>E013 | 2026-08-12T17:36:02.350Z | archived_tool_output | call_8zRF0wfHxZFUxp5tZn9aDbkx |
| <a id="e014"></a>E014 | 2026-08-12T17:41:40.574Z | archived_tool_output | call_HPSc3DZ88jihvLhkw6cQNgxA |
| <a id="e015"></a>E015 | 2026-08-12T18:25:28.552Z | archived_tool_output | call_s4h3nCbNs6tpVPEplakhQyyO |
| <a id="e016"></a>E016 | 2026-08-12T18:47:04.235Z | archived_tool_output | call_REXkLGUWv7SlMwMTK9VXDRDF |
| <a id="e017"></a>E017 | 2026-08-12T20:31:42.738Z | archived_tool_output | call_NC8oV7cl4gHPvtEGTmrRu9jg |
| <a id="e018"></a>E018 | 2026-08-12T20:33:08.951Z | archived_tool_output | call_9ymg16u5wItEJWjPJFWsA9an |
| <a id="e019"></a>E019 | 2026-08-12T20:41:03.066Z | archived_tool_output | call_wGs8hNArwhBCpFghbKTBHKT2 |
| <a id="e020"></a>E020 | 2026-08-12T20:52:56.581Z | archived_tool_output | call_IYfQ0sPpENi79tR2azP9mxvd |
| <a id="e021"></a>E021 | 2026-08-12T20:59:01.712Z | archived_tool_output | call_LgUe0S6dPGkkMLNt5GWuVeFx |
| <a id="e022"></a>E022 | 2026-08-12T21:01:27.311Z | archived_tool_output | call_6tbBhWQr8dk0dXQrxSqLuxWo |
| <a id="e023"></a>E023 | 2026-08-12T21:04:00.261Z | archived_tool_output | call_qWn9ivuyR8dLoH0Q72bVQS0s |
| <a id="e024"></a>E024 | 2026-08-12T21:04:43.521Z | archived_tool_output | call_UTVm1dda6JjOesS4deVewv8L |
| <a id="e025"></a>E025 | 2026-08-12T21:05:17.682Z | archived_tool_output | call_dFaoCfMef6BJe6FX7Pbkceml |
| <a id="e026"></a>E026 | 2026-08-12T21:06:55.730Z | archived_tool_output | call_VF4fD9M9NtSfsDa2CLQFYaHX |
| <a id="e027"></a>E027 | 2026-08-12T21:07:32.236Z | archived_tool_output | call_1wTwnuI4ppFCE2vQw20PIaxd |
| <a id="e028"></a>E028 | 2026-08-12T21:08:12.248Z | archived_tool_output | call_9eos29lkmYCZ3YC6fFgVWjFs |
| <a id="e029"></a>E029 | 2026-08-12T17:39:33.566Z | archived_tool_output | call_MgG6yCqrq39a0q0UdcnXQM7r |
| <a id="e030"></a>E030 | 2026-08-12T21:30:48.141Z | archived_tool_output | call_5z1yqTj9tAz0mXrVn3048oZh |
| <a id="e031"></a>E031 | 2026-08-12T17:22:24.792Z | archived_tool_output | call_7dyO9TkFAGSKvtDvwBMjLlY6 |
| <a id="e032"></a>E032 | 2026-08-12T17:19:42.463Z | archived_tool_output | call_FItYCJb8fZZR6O50Sa2TaHxD |
| <a id="e033"></a>E033 | 2026-08-12T17:42:12.408Z | archived_tool_output | call_IPS2Sr7w9i1toyJghfwCETkd |
| <a id="e034"></a>E034 | 2026-08-12T20:50:57.296Z | archived_tool_output | call_StoZBDrvChnCXhWI79ryMaCd |
| <a id="e035"></a>E035 | 2026-08-12T21:10:27.756Z | archived_tool_output | call_rJKOeTSXxizdDyOl1pBt7Y1C |
| <a id="e036"></a>E036 | 2026-08-12T21:31:52.915Z | archived_tool_output | call_N6RRiGU9qDS3oFGy0z3ubuRN |
| <a id="e037"></a>E037 | 2026-08-07T06:38:35.500Z | archived_tool_output | call_NJS9pVFBEfgERTJJMbWmLXHD |
| <a id="e038"></a>E038 | 2026-08-07T07:21:10.669Z | archived_tool_output | call_cWBr7IK9B5xHtnIQPYj5Eq2B |
| <a id="e039"></a>E039 | 2026-08-12T17:46:41.432Z | archived_tool_output | call_qy4VRTVxmIZpfEYKhg0xjFVJ |
| <a id="e040"></a>E040 | 2026-08-12T20:31:21.546Z | archived_tool_output | call_F0O7cQqHfnz1FmFAXgVC0bqG |
| <a id="e041"></a>E041 | 2026-08-12T21:16:29.469Z | archived_tool_output | call_PRlCHXQNXygVmeCS3p916ylH |
| <a id="e042"></a>E042 | 2026-08-12T21:18:32.335Z | archived_tool_output | call_01Z05ZOS3oVqP3crNFFkOewL |
| <a id="e043"></a>E043 | 2026-08-12T17:55:04.848Z | archived_tool_output | call_QxhquxNIvbamPzV7nvck6cw6 |
| <a id="e044"></a>E044 | 2026-08-12T21:31:37.077Z | archived_tool_output | call_aagBuk1lfQrfBKKn5QXiqINR |
| <a id="e045"></a>E045 | 2026-08-07T07:23:59.866Z | archived_tool_output | call_FYTKhekUgPbFqqwNWJCaouqp |
| <a id="e046"></a>E046 | 2026-08-07T07:25:30.587Z | archived_tool_output | call_GbvSOQlF7f1FtMG5d2jZlr8E |
| <a id="e047"></a>E047 | 2026-08-07T07:27:11.617Z | archived_tool_output | call_RLgsAhys7m4HVIFhvCCSQnm7 |
| <a id="e048"></a>E048 | 2026-08-07T07:29:24.286Z | archived_tool_output | call_TMgOHd1jxZBS2QcdiRdUqHAD |
| <a id="e049"></a>E049 | 2026-08-07T07:31:42.551Z | archived_tool_output | call_wiltW8FtDzr6NBnb3fnzUNKv |
| <a id="e050"></a>E050 | 2026-08-07T07:33:41.218Z | archived_tool_output | call_lzOgdN8mvg2tTxePVY8qFWpy |
| <a id="e051"></a>E051 | 2026-08-12T18:16:56.011Z | archived_tool_output | call_GPlYq3mCZtpCLS4VjMx9cuDN |
| <a id="e052"></a>E052 | 2026-08-12T18:20:33.999Z | archived_tool_output | call_buRtC5a54Y90SFJZfExwkxoJ |
| <a id="e053"></a>E053 | 2026-08-12T18:22:16.610Z | archived_tool_output | call_xRajCDVsZhOFfotqSNm1Wg2J |
| <a id="e054"></a>E054 | 2026-08-12T18:25:06.784Z | archived_tool_output | call_RJpISJsnhrqu7mu1XdKo71MQ |
| <a id="e055"></a>E055 | 2026-08-12T17:55:51.132Z | archived_tool_output | call_tSFaUjnxMEDoAgtOORJ6n25P |
| <a id="e056"></a>E056 | 2026-08-12T20:30:39.833Z | archived_tool_output | call_aOpnLfbf96bi7ERCMvKpOurv |
| <a id="chat"></a>CHAT | Supplied conversation | user_supplied_history | Not a tool result |
| <a id="e058"></a>E058 | 2026-08-05T10:41:01.820Z | assistant_final_answer | Not a tool result |
| <a id="e059"></a>E059 | 2026-08-05T11:20:08.815Z | assistant_final_answer | Not a tool result |
| <a id="e060"></a>E060 | 2026-08-05T11:19:01.149Z | assistant_final_answer | Not a tool result |
| <a id="e061"></a>E061 | 2026-08-05T09:15:33.828Z | assistant_final_answer | Not a tool result |
| <a id="e062"></a>E062 | 2026-08-05T11:15:09.704Z | assistant_final_answer | Not a tool result |
| <a id="e063"></a>E063 | 2026-08-05T11:43:00.347Z | assistant_final_answer | Not a tool result |
| <a id="e064"></a>E064 | 2026-08-05T13:48:48.882Z | assistant_final_answer | Not a tool result |
| <a id="e065"></a>E065 | 2026-08-05T14:47:00.533Z | assistant_final_answer | Not a tool result |
| <a id="e066"></a>E066 | 2026-08-12T18:54:14.575Z | assistant_final_answer | Not a tool result |
| <a id="e067"></a>E067 | 2026-08-14T16:48:04.688Z | assistant_final_answer | Not a tool result |
| <a id="e068"></a>E068 | 2026-08-14T16:57:42.349Z | assistant_final_answer | Not a tool result |
| <a id="e069"></a>E069 | 2026-08-14T16:16:03.418Z | assistant_final_answer | Not a tool result |
| <a id="e070"></a>E070 | 2026-08-14T16:22:48.962Z | assistant_final_answer | Not a tool result |
| <a id="e071"></a>E071 | 2026-08-14T15:54:02.590Z | assistant_final_answer | Not a tool result |
| <a id="e072"></a>E072 | 2026-08-05T12:25:01.411Z | archived_tool_output | call_j0nWrEmoBnGhfZ4RWZOYyQ4G |
| <a id="e073"></a>E073 | 2026-08-05T12:20:18.259Z | archived_tool_output | call_nM1pF6OEuTza3LSZMWFo9lmO |
| <a id="e074"></a>E074 | 2026-08-05T12:24:35.870Z | archived_tool_output | call_7mY3vktYn34kOabFN2HQuvAA |
| <a id="e075"></a>E075 | 2026-08-05T12:26:22.725Z | archived_tool_output | call_kg4R60dRMJEdQ71C5wWsEHHh |
| <a id="e076"></a>E076 | 2026-08-05T14:44:23.156Z | archived_tool_output | call_rtvIkvwRMfINig7OgAV1upx7 |

### Maintaining The Catalogue

For future tests, preserve the recording hash, exact reference and normalizer, model revision, dependency versions, effective runtime options, raw transcript, separate post-processed transcript, per-stage timers, and driver-level peak memory. Store benchmark artifacts durably rather than only under `/tmp`. Append a new run ID for a changed configuration instead of overwriting an old result.

The CSV is a flat view of the JSON experiment entries. Empty numeric cells are unknown. It includes corpus, inherited parameters, overrides, raw/final stage, S/D/I, WER, punctuation, sentence F1, timing scope, memory scope and evidence IDs. Preserve those distinctions when sorting or comparing results.
