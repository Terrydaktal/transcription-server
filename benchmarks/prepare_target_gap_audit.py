"""Preserve and sample VAD-rejected target regions for a false-negative audit."""

import argparse
import json
from pathlib import Path


def main():
    import numpy as np
    import soundfile as sf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    groups = [json.loads(line) for line in (root / "data/target-manifest.jsonl").read_text().splitlines()]
    old = json.loads((root / "data/Recording_1828_whisper_large-v3.json").read_text())["segments"]
    directory = root / "data/target-gap-clips"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    with sf.SoundFile(root / "data/Recording_1828.wav") as source:
        rate = source.samplerate
        duration = len(source) / rate
        boundaries = [(0.0, groups[0]["start"]),
                      *((a["end"], b["start"]) for a, b in zip(groups, groups[1:])),
                      (groups[-1]["end"], duration)]
        for start, end in boundaries:
            if end - start < 2:
                continue
            for offset in np.arange(start, end, 45):
                limit = min(float(offset) + 45, end)
                if limit - offset < 2:
                    continue
                begin_sample, end_sample = round(float(offset) * rate), round(limit * rate)
                source.seek(begin_sample)
                audio = source.read(end_sample - begin_sample, dtype="float32")
                name = f"target-gap-{len(rows):05d}"
                path = directory / f"{name}.wav"
                sf.write(path, audio, rate, subtype="PCM_16")
                prior = " ".join(segment["text"].strip() for segment in old
                                 if segment["end"] > offset and segment["start"] < limit)
                rows.append({"id": name, "audio": str(path), "corpus": "target", "split": "dev",
                             "start": float(offset), "end": limit, "duration_s": len(audio) / rate,
                             "rms": float(np.sqrt(np.mean(audio ** 2))),
                             "peak": float(np.abs(audio).max()), "old_asr_words": len(prior.split()),
                             "old_asr_text": prior, "reference_is_verified": False})
    regular = {int(index) for index in np.linspace(0, len(rows) - 1, 10).round()}
    disputed = sorted(range(len(rows)), key=lambda i: rows[i]["old_asr_words"], reverse=True)[:10]
    selected = sorted(regular | set(disputed))
    (root / "data/target-gap-manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "data/target-gap-sample-manifest.jsonl").write_text(
        "".join(json.dumps(rows[index]) + "\n" for index in selected))
    print(json.dumps({"gaps": len(rows), "samples": len(selected),
                      "rejected_audio_s": sum(row["duration_s"] for row in rows),
                      "note": "Old ASR used only to locate disputed regions, never as ground truth or prompts"}), flush=True)


if __name__ == "__main__":
    main()
