"""Run the frozen third recognizer only where two holdout transcripts disagree."""

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
    while not (root / "results/extra-search-complete.json").exists():
        time.sleep(5)
    for corpus in ("all", "berst"):
        name = f"vibe-auto-b1-raw-en-{corpus}-holdout-disagreements"
        manifest = root / "results" / f"noisy-consensus-holdout-gated.{corpus}.vibe-disagreements.jsonl"
        command = [sys.executable, str(root / "code/noisy_asr_search.py"), "vibe", "--root", str(root),
                   "--manifest", str(manifest), "--name", name, "--split", "holdout",
                   "--corpus", "all", "--limit", "0", "--beam", "1", "--filter", "raw"]
        print(json.dumps({"starting": name}), flush=True)
        if not (root / "results" / f"{name}.summary.json").exists():
            with (root / "results" / f"{name}.log").open("a") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, str(root / "code/evaluate_noisy_consensus.py"),
                    "--root", str(root), "--split", "holdout"], check=True)
    (root / "results/consensus-holdout-complete.json").write_text(json.dumps({"complete": True}))


if __name__ == "__main__":
    main()
