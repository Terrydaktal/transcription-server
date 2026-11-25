"""Create speaker-disjoint pocket/bag/covered-microphone benchmark subsets."""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path


def main():
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    rows = []
    directory = root / "data/berst-clips"
    directory.mkdir(parents=True, exist_ok=True)
    for parquet in sorted((root / "data/berst/data").glob("*.parquet")):
        for batch in pq.ParquetFile(parquet).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                position = row["phone_position"].lower()
                if row["shout_level"] != "no-shout" or not any(
                    word in position for word in ("pocket", "bag", "covering the mic")):
                    continue
                group = row["user_id"]
                content_hash = hashlib.sha256(row["audio"]["bytes"]).hexdigest()
                name = "berst-" + content_hash[:20]
                path = directory / f"{name}.wav"
                audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=True)
                divisor = math.gcd(rate, 16000)
                audio = resample_poly(audio.mean(axis=1), 16000 // divisor, rate // divisor)
                sf.write(path, audio, 16000, subtype="PCM_16")
                holdout = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 3 == 0
                rows.append({"id": name, "group": group, "corpus": "berst",
                             "split": "holdout" if holdout else "dev", "category": position,
                             "audio": str(path), "reference": row["script"],
                             "duration_s": len(audio) / 16000, "device": row["phone_model"],
                             "peak": float(np.abs(audio).max()),
                             "rms": float(np.sqrt(np.mean(audio ** 2))),
                             "original_split": parquet.stem.split("-")[0]})
    # Identical sound bytes must not count as multiple independent observations.
    unique = {}
    conflicts = {}
    for row in rows:
        if row["id"] in unique and unique[row["id"]]["reference"] != row["reference"]:
            conflicts.setdefault(row["id"], [unique[row["id"]]["reference"]]).append(row["reference"])
        unique[row["id"]] = row
    for identifier in conflicts:
        del unique[identifier]
    (root / "data/berst-conflicting-references.json").write_text(json.dumps(conflicts, indent=2))
    categories = sorted({row["category"] for row in unique.values()})
    buckets = [[row for row in sorted(unique.values(), key=lambda row: row["id"])
                if row["category"] == category] for category in categories]
    ordered = [bucket[i] for i in range(max(map(len, buckets))) for bucket in buckets if i < len(bucket)]
    (root / "data/berst-manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ordered))
    print(json.dumps({"clips": len(ordered), "excluded_conflicting_references": len(conflicts),
                      "dev": sum(row["split"] == "dev" for row in ordered),
                      "holdout": sum(row["split"] == "holdout" for row in ordered),
                      "categories": categories}), flush=True)


if __name__ == "__main__":
    main()
