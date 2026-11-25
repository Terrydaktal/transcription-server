"""Reproducible noisy-ASR comparison; never uses references as model prompts."""

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


FILTERS = {
    "raw": None,
    "hp80": "highpass=f=80",
    "hp120": "highpass=f=120",
    "denoise6": "highpass=f=80,afftdn=nr=6:nf=-35:tn=1",
    "denoise12": "highpass=f=80,afftdn=nr=12:nf=-35:tn=1",
    "level": "highpass=f=80,dynaudnorm=f=150:g=15:p=0.95:m=10",
    "presence": "highpass=f=80,equalizer=f=1800:t=q:w=0.6:g=5,alimiter=limit=0.95:level=0",
    "dfn3": None,
    "dfn3lim12": None,
    "peak_norm": None,
    "rms_norm": None,
    "rms_hp80": None,
}


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare(root):
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly

    rows = []
    for parquet in sorted((root / "data/vitw/data").glob("*.parquet")):
        for row in pq.read_table(parquet).to_pylist():
            if "-en-" not in row["subset"]:
                continue
            name = row["name"]
            path = root / "data/clips" / f"{name}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=True)
            if rate != 16000:
                divisor = math.gcd(rate, 16000)
                audio = resample_poly(audio, 16000 // divisor, rate // divisor, axis=0)
                rate = 16000
            sf.write(path, audio.mean(axis=1), rate, subtype="PCM_16")
            group = str(row["index"])
            holdout = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 3 == 0
            rows.append({
                "id": name, "group": group, "corpus": "vitw", "split": "holdout" if holdout else "dev",
                "category": row["subset"], "audio": str(path), "reference": row["answer"],
                "duration_s": len(audio) / rate,
                "peak": float(np.abs(audio).max()), "rms": float(np.sqrt(np.mean(audio ** 2))),
            })
    metadata = root / "data/krisp/phone_calls/metadata.jsonl"
    if metadata.exists():
        for row in read_jsonl(metadata):
            source = metadata.parent / row["file_name"]
            if not source.exists():
                continue
            name = f"krisp-phone-{row['id']}"
            path = root / "data/clips" / f"{name}.wav"
            audio, rate = sf.read(source, dtype="float32", always_2d=True)
            divisor = math.gcd(rate, 16000)
            audio = resample_poly(audio.mean(axis=1), 16000 // divisor, rate // divisor)
            sf.write(path, audio, 16000, subtype="PCM_16")
            group = row["speaker_id"]
            holdout = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 3 == 0
            rows.append({"id": name, "group": group, "corpus": "krisp",
                         "split": "holdout" if holdout else "dev",
                         "category": f"krisp-{row['environment']}", "audio": str(path),
                         "reference": row["full_transcript"], "duration_s": len(audio) / 16000,
                         "device": row["recording_device"], "peak": float(np.abs(audio).max()),
                         "rms": float(np.sqrt(np.mean(audio ** 2)))})
    # Interleave categories and use an audio-ID-only hash, not errors or references.
    categories = sorted({row["category"] for row in rows})
    buckets = {key: sorted((row for row in rows if row["category"] == key),
                            key=lambda row: hashlib.sha256(row["id"].encode()).digest()) for key in categories}
    ordered = [bucket[i] for i in range(max(map(len, buckets.values())))
               for bucket in buckets.values() if i < len(bucket)]
    manifest = root / "data/manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in ordered))
    emit({"manifest": str(manifest), "count": len(rows), "categories": categories,
          "dev": sum(row["split"] == "dev" for row in rows),
          "holdout": sum(row["split"] == "holdout" for row in rows)})


def filtered_path(root, row, kind):
    if kind == "raw":
        return Path(row["audio"])
    destination = root / "data/filtered" / kind / f"{row['id']}.wav"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        if kind.startswith("dfn3"):
            raise FileNotFoundError(f"Run enhance_audio.py first: {destination}")
        if kind in ("peak_norm", "rms_norm", "rms_hp80"):
            import numpy as np
            import soundfile as sf

            source = filtered_path(root, row, "hp80") if kind == "rms_hp80" else row["audio"]
            audio, rate = sf.read(source, dtype="float32")
            peak = float(np.abs(audio).max())
            rms = float(np.sqrt(np.mean(audio ** 2)))
            gain = min(100.0, 0.89 / max(peak, 1e-12))
            if kind != "peak_norm":
                gain = min(gain, 0.1 / max(rms, 1e-12))
            gain = max(1.0, gain) if peak < 0.89 else 1.0
            sf.write(destination, audio * gain, rate, subtype="PCM_16")
        else:
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", row["audio"],
                            "-af", FILTERS[kind], "-ar", "16000", "-ac", "1",
                            "-c:a", "pcm_s16le", str(destination)], check=True)
    return destination


def score(rows):
    from jiwer import process_words
    from whisper_normalizer.english import EnglishTextNormalizer

    normalizer = EnglishTextNormalizer()
    totals = {key: 0 for key in ("substitutions", "deletions", "insertions", "reference_words")}
    for row in rows:
        if "reference" not in row:
            continue
        result = process_words(normalizer(row["reference"]), normalizer(row["text"]))
        row["wer"] = result.wer
        for key in ("substitutions", "deletions", "insertions"):
            totals[key] += getattr(result, key)
        totals["reference_words"] += result.hits + result.substitutions + result.deletions
    totals["wer"] = (sum(totals[key] for key in ("substitutions", "deletions", "insertions"))
                     / totals["reference_words"] if totals["reference_words"] else None)
    return totals


def router_prediction(router, path, window_seconds=30):
    import torch

    waveform = router._load_audio(path)
    limit = int(window_seconds * router.sample_rate)
    probabilities = []
    for window in waveform.split(limit, dim=-1):
        if not window.shape[-1]:
            continue
        mel = router.mel_extractor(window).squeeze(0).transpose(0, 1).unsqueeze(0)
        probability = torch.softmax(router.model(mel, mask=None), dim=-1)[0, 1].item()
        probabilities.append((probability, window.shape[-1]))
    if not probabilities:
        raise ValueError("Cannot route empty audio")
    probability = sum(value * length for value, length in probabilities) / sum(
        length for _, length in probabilities)
    return probability >= router.threshold, probability


def resume_results(destination, rows, args):
    previous = read_jsonl(destination) if destination.exists() else []
    root = getattr(args, "root", destination.parent.parent)
    folders = {"mega": "Mega-ASR", "whisper": "whisper-large-v3", "vibe": "VibeVoice-ASR",
               "moss": "MOSS-Transcribe-preview-2B"}
    model_root = root / "models" / folders[args.stage]
    model_metadata = {str(path.relative_to(model_root)): path.read_text().splitlines()[:2]
                      for path in sorted(model_root.glob("**/.cache/huggingface/download/*.metadata"))}
    fingerprint = {"schema": 2, "stage": args.stage, "model": str(model_root),
                   "model_metadata": model_metadata,
                   "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "configuration": {key: getattr(args, key) for key in
                                     ("filter", "beam", "route", "language", "length_penalty", "threshold", "lora_scale")},
                   "seeding": "20261002 XOR SHA256(audio_id), per clip",
                   "audio_sha256": {row["id"]: hashlib.sha256(Path(row["audio"]).read_bytes()).hexdigest()
                                    for row in rows}}
    sidecar = destination.with_suffix(".run.json")
    if sidecar.exists():
        if json.loads(sidecar.read_text()) != fingerprint:
            raise ValueError("Model, audio, code or configuration changed; use a new --name")
    elif previous:
        raise ValueError("Saved results lack a reproducibility fingerprint; use a new --name")
    else:
        sidecar.write_text(json.dumps(fingerprint, indent=2))
    expected = {row["id"]: row for row in rows}
    if len(expected) != len(rows) or len({row["id"] for row in previous}) != len(previous):
        raise ValueError("Duplicate audio IDs in manifest or saved results")
    for row in previous:
        if row["id"] not in expected:
            raise ValueError("Saved results do not belong to the requested split/subset")
        for key in ("audio", "reference", "split"):
            if row.get(key) != expected[row["id"]].get(key):
                raise ValueError(f"Changed manifest field {key}: {row['id']}")
        for key in ("filter", "beam", "route", "language", "length_penalty", "threshold", "lora_scale"):
            if key in row and row[key] != getattr(args, key):
                raise ValueError(f"Changed {key}; use a new --name instead of mixing configurations")
    return previous


def seed_clip(row):
    import torch

    seed = 20261002 ^ int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16)
    torch.manual_seed(seed)
    return seed


def load_mega(args, root):
    import torch
    # Avoid a ROCm fused-attention hang on longer clips; this affects only
    # this benchmark process, not the machine's other model services.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    sys.path.insert(0, str(root / "Mega-ASR/src"))
    from MegaASR.model.megaASR import MegaASR
    from MegaASR.model.utils.lora_switch import LoRADeltaSwitch

    # Restore pristine weights on route changes; BF16 add/subtract is not
    # reversible and otherwise makes results depend on earlier clip order.
    original_add_adapter = LoRADeltaSwitch.add_adapter

    def add_adapter(switch, *positional, **keywords):
        original_add_adapter(switch, *positional, **keywords)
        switch.pristine = {id(item["weight"]): (item["weight"], item["weight"].detach().clone())
                           for item in switch.items}

    @torch.no_grad()
    def set_active(switch, active):
        if switch.active == active:
            return 0.0
        begin = time.perf_counter()
        for weight, pristine in switch.pristine.values():
            weight.copy_(pristine)
        if active:
            for item in switch.items:
                item["weight"].add_(item["delta"].to(item["weight"].device) * args.lora_scale)
        switch.active = active
        return time.perf_counter() - begin

    LoRADeltaSwitch.add_adapter = add_adapter
    LoRADeltaSwitch.set_active = set_active

    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("This benchmark requires the ai machine's ROCm GPU")
    model_root = root / "models/Mega-ASR"
    started = time.perf_counter()
    mega = MegaASR(model_path=model_root / "Qwen3-ASR-1.7B",
                   lora_dir=model_root / "mega-asr-merged",
                   router_checkpoint=model_root / "audio_quality_router/best_acc_model.safetensors",
                   device_map="cuda:0", quality_device="cuda", dtype=torch.bfloat16,
                   max_inference_batch_size=1, max_new_tokens=512,
                   routing_enabled=True, quality_threshold=args.threshold,
                   attn_implementation="sdpa")
    generate = mega.asr.model.model.generate
    generation = []

    def measured_generate(*positional, **keywords):
        keywords.update(do_sample=False, num_beams=args.beam, length_penalty=args.length_penalty)
        result = generate(*positional, **keywords)
        sequences = result.sequences
        input_length = keywords["input_ids"].shape[1]
        new = sequences[:, input_length:]
        eos = mega.asr.model.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        generation.append({"new_tokens": new.shape[1],
                           "ended_with_eos": int(new[0, -1]) in eos,
                           "token_cap_hit": new.shape[1] >= keywords["max_new_tokens"]})
        return result

    mega.asr.model.model.generate = measured_generate
    mega.asr.model.model.eval()
    emit({"model_loaded_s": time.perf_counter() - started, "gpu": torch.cuda.get_device_name(0),
          "torch": torch.__version__, "hip": torch.version.hip,
          "adapter_switch": "pristine_bf16_reset", "attention_backend": "sdpa_math"})
    return mega, generation


def run_mega(args, root, rows):
    import torch

    mega, generation = load_mega(args, root)
    destination = root / "results" / f"{args.name}.jsonl"
    previous = resume_results(destination, rows, args)
    done = {row["id"] for row in previous}
    for row in rows:
        if row["id"] in done:
            continue
        path = filtered_path(root, row, args.filter)
        sampling_seed = seed_clip(row)
        begin = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        generation.clear()
        with torch.inference_mode():
            # The router has a finite positional buffer. Classify all windows
            # instead of truncating audio or extending its trained positions.
            degraded, probability = router_prediction(mega.router, path)
            active = degraded if args.route == "auto" else args.route == "mega"
            mega._set_lora(active)
            text = mega.asr.infer(path, language="English" if args.language == "en" else None)
        torch.cuda.synchronize()
        result = {**row, "text": " ".join(text) if isinstance(text, list) else text,
                  "route": args.route, "lora_active": active, "degraded_probability": probability,
                  "filter": args.filter, "beam": args.beam, "length_penalty": args.length_penalty,
                  "threshold": args.threshold, "lora_scale": args.lora_scale,
                  "language": args.language, "generation": generation.copy(),
                  "stage": args.stage, "sampling_seed": sampling_seed,
                  "infer_s": time.perf_counter() - begin,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
        with destination.open("a") as output:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
        previous.append(result)
        emit({"run": args.name, "progress": f"{len(previous)}/{len(rows)}", "id": row["id"],
              "infer_s": round(result["infer_s"], 3), "lora": active})
    summary = {"name": args.name, "split": args.split, "count": len(previous),
               "route": args.route, "filter": args.filter, "beam": args.beam,
               "language": args.language,
               "length_penalty": args.length_penalty, "threshold": args.threshold,
               "lora_scale": args.lora_scale,
               **score(previous), "infer_s": sum(row["infer_s"] for row in previous),
               "peak_allocated_gib": max(row["peak_allocated_gib"] for row in previous)}
    (root / "results" / f"{args.name}.summary.json").write_text(json.dumps(summary, indent=2))
    emit(summary)


def run_whisper(args, root, rows):
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model_path = root / "models/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    destination = root / "results" / f"{args.name}.jsonl"
    previous = resume_results(destination, rows, args)
    done = {row["id"] for row in previous}
    for row in rows:
        if row["id"] in done:
            continue
        path = filtered_path(root, row, args.filter)
        sampling_seed = seed_clip(row)
        begin = time.perf_counter()
        audio, rate = sf.read(path, dtype="float32")
        features = processor(audio, sampling_rate=rate, return_tensors="pt", truncation=False,
                             padding="longest", return_attention_mask=True)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            predicted = model.generate(features.input_features.to("cuda", torch.bfloat16),
                                       attention_mask=features.attention_mask.to("cuda"),
                                       language="en", task="transcribe", num_beams=args.beam,
                                       length_penalty=args.length_penalty,
                                       do_sample=False, max_length=448, return_timestamps=True,
                                       condition_on_prev_tokens=True,
                                       temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                                       compression_ratio_threshold=2.4, logprob_threshold=-1.0,
                                       no_speech_threshold=0.6)
        text = processor.batch_decode(predicted, skip_special_tokens=True)[0].strip()
        torch.cuda.synchronize()
        result = {**row, "text": text, "beam": args.beam, "filter": args.filter,
                  "stage": args.stage, "length_penalty": args.length_penalty, "sampling_seed": sampling_seed,
                  "infer_s": time.perf_counter() - begin,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
        with destination.open("a") as output:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
        previous.append(result)
        emit({"run": args.name, "progress": f"{len(previous)}/{len(rows)}", "id": row["id"],
              "infer_s": round(result["infer_s"], 3)})
    summary = {"name": args.name, "split": args.split, "count": len(previous),
               "model": "whisper-large-v3", "runtime": "transformers-bf16-rocm",
               "filter": args.filter, "beam": args.beam, "length_penalty": args.length_penalty,
               **score(previous),
               "infer_s": sum(row["infer_s"] for row in previous),
               "peak_allocated_gib": max(row["peak_allocated_gib"] for row in previous)}
    (root / "results" / f"{args.name}.summary.json").write_text(json.dumps(summary, indent=2))
    emit(summary)


def run_vibe(args, root, rows):
    import torch

    sys.path.insert(0, str(root / "VibeVoice"))
    sys.path.insert(0, str(root / "VibeVoice/demo"))
    from vibevoice_asr_inference_from_file import VibeVoiceASRBatchInference

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    backend = VibeVoiceASRBatchInference(str(root / "models/VibeVoice-ASR"))
    generate = backend.model.generate
    generation = []

    def measured_generate(*positional, **keywords):
        keywords["length_penalty"] = args.length_penalty
        output = generate(*positional, **keywords)
        new = output[:, keywords["input_ids"].shape[1]:]
        generation.append({"new_tokens": new.shape[1],
                           "ended_with_eos": int(new[0, -1]) == backend.processor.tokenizer.eos_token_id,
                           "token_cap_hit": new.shape[1] >= keywords["max_new_tokens"]})
        return output

    backend.model.generate = measured_generate
    destination = root / "results" / f"{args.name}.jsonl"
    previous = resume_results(destination, rows, args)
    done = {row["id"] for row in previous}
    for row in rows:
        if row["id"] in done:
            continue
        path = filtered_path(root, row, args.filter)
        sampling_seed = seed_clip(row)
        begin = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        generation.clear()
        with torch.inference_mode():
            response = backend.transcribe_batch([str(path)],
                                                max_new_tokens=2048, do_sample=False,
                                                num_beams=args.beam)[0]
        torch.cuda.synchronize()
        result = {**row, "text": " ".join(part["text"] for part in response["segments"]),
                  "raw_response": response["raw_text"], "segments": response["segments"],
                  "filter": args.filter, "beam": args.beam, "generation": generation.copy(),
                  "stage": args.stage, "length_penalty": args.length_penalty, "sampling_seed": sampling_seed,
                  "infer_s": time.perf_counter() - begin,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
        with destination.open("a") as output:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
        previous.append(result)
        emit({"run": args.name, "progress": f"{len(previous)}/{len(rows)}", "id": row["id"],
              "infer_s": round(result["infer_s"], 3)})
    summary = {"name": args.name, "split": args.split, "count": len(previous),
               "model": "microsoft/VibeVoice-ASR", "runtime": "transformers-bf16-rocm",
               "filter": args.filter, "beam": args.beam, "length_penalty": args.length_penalty,
               **score(previous),
               "infer_s": sum(row["infer_s"] for row in previous),
               "peak_allocated_gib": max(row["peak_allocated_gib"] for row in previous)}
    (root / "results" / f"{args.name}.summary.json").write_text(json.dumps(summary, indent=2))
    emit(summary)


def run_moss(args, root, rows):
    import soundfile as sf
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    model_path = str(root / "models/MOSS-Transcribe-preview-2B")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor_class = get_class_from_dynamic_module("processing_Moss.MossProcessor", model_path)
    mel_class = get_class_from_dynamic_module("processing_Moss.MelConfig", model_path)
    processor = processor_class(tokenizer, config=mel_class(
        mel_sr=16000, mel_dim=128, mel_n_fft=400, mel_hop_length=160), enable_time_marker=False)
    processor.load_template(str(Path(model_path) / "chat_template_default.py"))
    destination = root / "results" / f"{args.name}.jsonl"
    previous = resume_results(destination, rows, args)
    done = {row["id"] for row in previous}
    for row in rows:
        if row["id"] in done:
            continue
        path = filtered_path(root, row, args.filter)
        sampling_seed = seed_clip(row)
        begin = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        audio, _ = sf.read(path, dtype="float32")
        inputs = processor(audio=audio, return_tensors="pt").to("cuda")
        inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                                        num_beams=args.beam, length_penalty=args.length_penalty, use_cache=True,
                                        eos_token_id=[processor.end_token_id])
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        torch.cuda.synchronize()
        result = {**row, "text": text, "filter": args.filter, "beam": args.beam,
                  "stage": args.stage, "length_penalty": args.length_penalty, "sampling_seed": sampling_seed,
                  "generation": [{"new_tokens": generated.shape[1],
                                  "ended_with_eos": int(generated[0, -1]) == processor.end_token_id,
                                  "token_cap_hit": generated.shape[1] >= 512}],
                  "infer_s": time.perf_counter() - begin,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
        with destination.open("a") as output:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
        previous.append(result)
        emit({"run": args.name, "progress": f"{len(previous)}/{len(rows)}", "id": row["id"],
              "infer_s": round(result["infer_s"], 3)})
    summary = {"name": args.name, "split": args.split, "count": len(previous),
               "model": "OpenMOSS-Team/MOSS-Transcribe-preview-2B", "runtime": "transformers-bf16-rocm",
               "filter": args.filter, "beam": args.beam, "length_penalty": args.length_penalty,
               **score(previous),
               "infer_s": sum(row["infer_s"] for row in previous),
               "peak_allocated_gib": max(row["peak_allocated_gib"] for row in previous)}
    (root / "results" / f"{args.name}.summary.json").write_text(json.dumps(summary, indent=2))
    emit(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "mega", "whisper", "vibe", "moss"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--name", default="mega-auto-raw")
    parser.add_argument("--split", choices=["dev", "holdout"], default="dev")
    parser.add_argument("--corpus", choices=["vitw", "krisp", "berst", "all"], default="vitw")
    parser.add_argument("--limit", type=int, default=36)
    parser.add_argument("--route", choices=["auto", "mega", "base"], default="auto")
    parser.add_argument("--filter", choices=FILTERS, default="raw")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--language", choices=["en", "auto"], default="en")
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.root)
    else:
        rows = [row for row in read_jsonl(args.manifest or args.root / "data/manifest.jsonl")
                if row["split"] == args.split]
        if args.corpus != "all":
            rows = [row for row in rows if row["corpus"] == args.corpus]
        if args.limit:
            rows = rows[:args.limit]
        {"mega": run_mega, "whisper": run_whisper, "vibe": run_vibe, "moss": run_moss}[
            args.stage](args, args.root, rows)


if __name__ == "__main__":
    main()
