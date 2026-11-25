"""Preserve target pauses and create non-overlapping, silence-bounded ASR clips."""

import argparse
import json
from pathlib import Path


def group_speech(spans, maximum_samples, gap_samples):
    groups = []
    for span in spans:
        start, end = span["start"], span["end"]
        if groups and start - groups[-1]["end"] <= gap_samples and end - groups[-1]["start"] <= maximum_samples:
            groups[-1]["end"] = end
        else:
            groups.append({"start": start, "end": end})
    return groups


def main():
    import numpy as np
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    torch.set_num_threads(1)
    audio, rate = sf.read(root / "data/Recording_1828.wav", dtype="float32")
    if rate != 16000 or audio.ndim != 1:
        raise ValueError("Expected mono 16 kHz PCM")
    model = load_silero_vad()
    spans = get_speech_timestamps(torch.from_numpy(audio), model, sampling_rate=rate,
                                  threshold=0.25, min_speech_duration_ms=150,
                                  min_silence_duration_ms=700, speech_pad_ms=400,
                                  max_speech_duration_s=45)
    groups = group_speech(spans, 45 * rate, 3 * rate)
    directory = root / "data/target-clips"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, group in enumerate(groups):
        path = directory / f"target-{index:05d}.wav"
        chunk = audio[group["start"]:group["end"]]
        sf.write(path, chunk, rate, subtype="PCM_16")
        manifest.append({"id": path.stem, "audio": str(path), "split": "dev", "corpus": "target",
                         "start": group["start"] / rate,
                         "end": group["end"] / rate, "duration_s": len(chunk) / rate,
                         "rms": float(np.sqrt(np.mean(chunk ** 2))),
                         "peak": float(np.abs(chunk).max())})
    (root / "data/target-manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest))
    sample_indices = sorted(set(np.linspace(0, len(manifest) - 1, 18).round().astype(int)))
    (root / "data/target-sample-manifest.jsonl").write_text(
        "".join(json.dumps(manifest[index]) + "\n" for index in sample_indices))
    (root / "data/target-vad.json").write_text(json.dumps(
        {"threshold": 0.25, "padding_ms": 400, "min_silence_ms": 700,
         "max_chunk_s": 45, "max_join_gap_s": 3, "speech_spans": spans,
         "original_duration_s": len(audio) / rate}, indent=2))
    print(json.dumps({"target_clips": len(manifest), "original_duration_s": len(audio) / rate,
                      "retained_duration_s": sum(row["duration_s"] for row in manifest)}), flush=True)


if __name__ == "__main__":
    main()
