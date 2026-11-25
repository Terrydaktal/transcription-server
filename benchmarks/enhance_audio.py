"""Cache two DeepFilterNet3 variants without altering the source audio."""

import argparse
import json
import math
from pathlib import Path

import soundfile as sf
import torch
from df.enhance import enhance, init_df
from scipy.signal import resample_poly


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--corpus", default="all")
    parser.add_argument("--limit", type=int, default=36)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, state, _ = init_df(model_base_dir="DeepFilterNet3", log_level="ERROR")
    rate = state.sr()
    rows = [json.loads(line) for line in (args.manifest or args.root / "data/manifest.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row["split"] == args.split and
            (args.corpus == "all" or row["corpus"] == args.corpus)]
    if args.limit:
        rows = rows[:args.limit]
    for row in rows:
        audio, input_rate = sf.read(row["audio"], dtype="float32")
        divisor = math.gcd(rate, input_rate)
        audio = resample_poly(audio, rate // divisor, input_rate // divisor)
        for kind, limit in (("dfn3", None), ("dfn3lim12", 12)):
            destination = args.root / "data/filtered" / kind / f"{row['id']}.wav"
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            result = enhance(model, state, torch.from_numpy(audio).unsqueeze(0), pad=True, atten_lim_db=limit)
            result = resample_poly(result.squeeze(0).numpy(), 16000 // divisor, rate // divisor)
            if abs(len(result) - round(row["duration_s"] * 16000)) > 2:
                raise ValueError(f"Enhancement changed recording length: {row['id']}")
            sf.write(destination, result, 16000, subtype="PCM_16")
        print(json.dumps({"enhanced": row["id"]}), flush=True)


if __name__ == "__main__":
    main()
