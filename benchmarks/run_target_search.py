"""Transcribe the private target only after development and holdout jobs finish."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    for marker in ("all-benchmarks-complete.json",):
        while not (root / "results" / marker).exists():
            time.sleep(5)
    winner = json.loads((root / "results/extra-selection.json").read_text())["winner"]

    def run(name, config, manifest):
        extra = config["stage"] in ("ark", "hojo")
        program = "extra_noisy_models.py" if extra else "noisy_asr_search.py"
        command = [sys.executable, str(root / "code" / program), config["stage"],
                   "--root", str(root), "--manifest", str(root / "data" / manifest),
                   "--corpus", "all", "--limit", "0", "--name", name]
        options = [("--beam", "beam", 2), ("--filter", "kind", "raw")]
        options += [("--repetition-penalty", "repetition_penalty", 1.0)] if extra else [
                                     ("--route", "route", "base"), ("--language", "language", "en"),
                                     ("--length-penalty", "length_penalty", 1.0),
                                     ("--threshold", "threshold", 0.5), ("--lora-scale", "lora_scale", 1.0)]
        for option, key, default in options:
            command.extend([option, str(config.get(key, default))])
        print(json.dumps({"starting": name, "configuration": config}), flush=True)
        if not (root / "results" / f"{name}.summary.json").exists():
            with (root / "results" / f"{name}.log").open("a") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, str(root / "code/render_target_transcript.py"),
                        "--input", str(root / "results" / f"{name}.jsonl"),
                        "--prefix", str(root / "results" / name), "--baseline",
                        str(root / "data/Recording_1828_whisper_large-v3.json")], check=True)

    run("target-selected-samples", winner, "target-sample-manifest.jsonl")
    run("target-selected-gap-audit", winner, "target-gap-sample-manifest.jsonl")
    run("target-mega-router-samples", {"stage": "mega", "route": "auto"}, "target-sample-manifest.jsonl")
    run("target-vibe-samples", {"stage": "vibe", "beam": 1}, "target-sample-manifest.jsonl")
    run("target-whisper-samples", {"stage": "whisper"}, "target-sample-manifest.jsonl")
    (root / "results/target-samples-complete.json").write_text(json.dumps({"configuration": winner}))
    run("Recording_1828_selected_20261002", winner, "target-manifest.jsonl")
    (root / "results/target-complete.json").write_text(json.dumps({"configuration": winner}))


if __name__ == "__main__":
    main()
