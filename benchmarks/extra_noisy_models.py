"""Evaluate larger ARK/Hojo checkpoints without truncating long benchmark clips."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from noisy_asr_search import emit, filtered_path, read_jsonl, score, seed_clip


def quiet_chunks(audio, rate=16000, maximum_seconds=30):
    import numpy as np

    limit = int(maximum_seconds * rate)
    frame = int(0.1 * rate)
    start = 0
    slices = []
    while len(audio) - start > limit:
        candidates = list(range(start + limit - 5 * rate, start + limit - frame, frame))
        cut = min(candidates, key=lambda index: float(np.mean(audio[index:index + frame] ** 2)))
        end = cut + frame // 2
        slices.append((start, end))
        start = end
    if start < len(audio):
        slices.append((start, len(audio)))
    return slices


def main():
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["ark", "hojo"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", choices=["dev", "holdout"], default="dev")
    parser.add_argument("--corpus", default="all")
    parser.add_argument("--limit", type=int, default=36)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--filter", default="raw")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    args = parser.parse_args()
    root = args.root
    rows = [row for row in read_jsonl(args.manifest or root / "data/manifest.jsonl")
            if row["split"] == args.split and (args.corpus == "all" or row["corpus"] == args.corpus)]
    if args.limit:
        rows = rows[:args.limit]
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    model_root = root / "models" / ("ARK-ASR-3B" if args.stage == "ark" else "Hojo-ASR-V1")
    config = {"stage": args.stage, "beam": args.beam, "filter": args.filter,
              "repetition_penalty": args.repetition_penalty, "model": str(model_root),
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "audio_sha256": {row["id"]: hashlib.sha256(Path(row["audio"]).read_bytes()).hexdigest() for row in rows}}
    destination = root / "results" / f"{args.name}.jsonl"
    previous = read_jsonl(destination) if destination.exists() else []
    sidecar = destination.with_suffix(".run.json")
    if sidecar.exists() and json.loads(sidecar.read_text()) != config:
        raise ValueError("Configuration changed; use a new name")
    if previous and not sidecar.exists():
        raise ValueError("No saved configuration; use a new name")
    sidecar.write_text(json.dumps(config, indent=2))
    expected = {row["id"]: row for row in rows}
    if len(expected) != len(rows) or len({row["id"] for row in previous}) != len(previous):
        raise ValueError("Duplicate audio IDs")
    for row in previous:
        if row["id"] not in expected or any(row.get(key) != expected[row["id"]].get(key)
                                            for key in ("reference", "audio", "split")):
            raise ValueError("Saved predictions belong to a different cohort")
    done = {row["id"] for row in previous}

    if args.stage == "ark":
        processor = AutoProcessor.from_pretrained(model_root, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_root, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_root, trust_remote_code=True,
                                                    torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        forbidden = set(tokenizer.all_special_ids) - {tokenizer.eos_token_id}
        forbidden.update(value for key, value in tokenizer.get_added_vocab().items()
                         if key.startswith("<") and key.endswith(">") and value != tokenizer.eos_token_id)

        def infer(audio):
            conversation = [{"role": "user", "content": [{"type": "audio", "array": audio},
                             {"type": "text", "text": "Please transcribe this audio."}]}]
            inputs = processor.apply_chat_template(conversation, add_generation_prompt=True,
                        return_tensors="pt", sampling_rate=16000, audio_padding="longest",
                        text_kwargs={"padding": "longest"}, audio_max_length=30 * 16000).to("cuda")
            inputs["audios"] = inputs["audios"].to(torch.bfloat16)
            ids = model.generate(**inputs, do_sample=False, num_beams=args.beam, max_new_tokens=512,
                                 repetition_penalty=args.repetition_penalty, pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=tokenizer.eos_token_id, bad_words_ids=[[value] for value in sorted(forbidden)])
            new = ids[:, inputs.input_ids.shape[1]:]
            audits.append({"new_tokens": new.shape[1], "token_cap_hit": new.shape[1] >= 512,
                           "ended_with_eos": int(new[0, -1]) == tokenizer.eos_token_id})
            return tokenizer.batch_decode(new, skip_special_tokens=True)[0].strip()
    else:
        sys.path.insert(0, str(root / "Hojo-ASR"))
        sys.path.append(str(root / "hojo-deps"))
        from hojo_asr import HOJO_ASR
        from hojo_asr import dataset
        # TorchAudio 2.10 requires TorchCodec. Read the same lossless PCM through
        # SoundFile rather than installing a CUDA-only decoding dependency.
        def load_pcm(source):
            waveform, rate = sf.read(source, dtype="float32", always_2d=True)
            return torch.from_numpy(waveform.T.copy()), rate
        dataset.torchaudio.load = load_pcm
        model = HOJO_ASR.load_model(str(model_root), device="cuda:0").to(torch.bfloat16).eval()
        model.config.generate.num_beams = args.beam
        model.config.generate.max_new_tokens = 512
        model.config.generate.do_sample = False
        model.config.generate.repetition_penalty = args.repetition_penalty
        model.autocast_context = lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16)
        original = model.decoder_model.generate
        def generate(*positional, **keywords):
            ids = original(*positional, **keywords)
            audits.append({"new_tokens": ids.shape[1], "token_cap_hit": ids.shape[1] >= keywords["max_new_tokens"],
                           "ended_with_eos": int(ids[0, -1]) == model.tokenizer.eos_token_id})
            return ids
        model.decoder_model.generate = generate
        def infer(audio):
            outputs = model.run_infer([np.asarray(audio, dtype=np.float32)], batch_size=1)
            if len(outputs) != 1:
                raise ValueError("Hojo dropped or duplicated a chunk")
            return outputs[0]["text"]

    for row in rows:
        if row["id"] in done:
            continue
        path = filtered_path(root, row, args.filter)
        seed = seed_clip(row)
        begin = time.perf_counter()
        audio, rate = sf.read(path, dtype="float32")
        if rate != 16000 or audio.ndim != 1:
            raise ValueError("Expected mono 16 kHz PCM")
        chunks = quiet_chunks(audio)
        audits = []
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            text = " ".join(infer(audio[start:end]) for start, end in chunks)
        torch.cuda.synchronize()
        result = {**row, "text": text, "stage": args.stage, "beam": args.beam, "filter": args.filter,
                  "repetition_penalty": args.repetition_penalty, "sampling_seed": seed,
                  "chunk_samples": chunks, "generation": audits,
                  "infer_s": time.perf_counter() - begin,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
        with destination.open("a") as output:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
        previous.append(result)
        emit({"run": args.name, "progress": f"{len(previous)}/{len(rows)}", "id": row["id"]})
    summary = {"name": args.name, "split": args.split, "count": len(previous), "stage": args.stage,
               "beam": args.beam, "filter": args.filter, "repetition_penalty": args.repetition_penalty,
               **score(previous), "infer_s": sum(row["infer_s"] for row in previous),
               "peak_allocated_gib": max(row["peak_allocated_gib"] for row in previous)}
    (root / "results" / f"{args.name}.summary.json").write_text(json.dumps(summary, indent=2))
    emit(summary)


if __name__ == "__main__":
    main()
