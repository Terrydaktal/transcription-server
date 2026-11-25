"""Test larger checkpoints on development data before freezing their holdout settings."""

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
    while not (root / "results/holdout-results.json").exists():
        time.sleep(5)

    def run(stage, beam, corpus, split="dev", kind="raw", repetition_penalty=1.0):
        name = f"{stage}-auto-b{beam}-{kind}-en-{corpus}-{split}"
        if repetition_penalty != 1.0:
            name += f"-rp{repetition_penalty:g}"
        command = [sys.executable, str(root / "code/extra_noisy_models.py"), stage,
                   "--root", str(root), "--name", name, "--beam", str(beam),
                   "--filter", kind, "--split", split, "--corpus", corpus,
                   "--repetition-penalty", str(repetition_penalty), "--limit",
                   str(0 if split == "holdout" else 84 if corpus == "berst" else 36)]
        if corpus == "berst":
            command.extend(["--manifest", str(root / "data/berst-manifest.jsonl")])
        print(json.dumps({"starting": name}), flush=True)
        if not (root / "results" / f"{name}.summary.json").exists():
            with (root / "results" / f"{name}.log").open("a") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((root / "results" / f"{name}.summary.json").read_text())
        print(json.dumps(summary), flush=True)
        return summary

    original = json.loads((root / "results/final-selection.json").read_text())["winner"]
    candidates = [original]
    for stage in ("ark", "hojo"):
        settings = [(1, 1.0), (2, 1.0), (4, 1.0)]
        if stage == "hojo":
            settings.append((4, 2.0))
        for beam, penalty in settings:
            broad = run(stage, beam, "all", repetition_penalty=penalty)
            pocket = run(stage, beam, "berst", repetition_penalty=penalty)
            candidates.append({"stage": stage, "beam": beam, "kind": "raw", "language": "en",
                               "repetition_penalty": penalty, "summary": broad, "pocket_summary": pocket,
                               "selection_wer": (broad["wer"] + pocket["wer"]) / 2})
    winner = min(candidates, key=lambda row: row["selection_wer"])
    if winner["stage"] in ("ark", "hojo"):
        for kind in ("hp80", "hp120", "presence", "dfn3", "dfn3lim12", "rms_norm"):
            broad = run(winner["stage"], winner["beam"], "all", kind=kind,
                        repetition_penalty=winner["repetition_penalty"])
            pocket = run(winner["stage"], winner["beam"], "berst", kind=kind,
                         repetition_penalty=winner["repetition_penalty"])
            candidates.append({**winner, "kind": kind, "summary": broad, "pocket_summary": pocket,
                               "selection_wer": (broad["wer"] + pocket["wer"]) / 2})
    winner = min(candidates, key=lambda row: row["selection_wer"])
    selection = {"winner": winner, "runs": candidates, "selection_data": "development only"}
    (root / "results/extra-selection.json").write_text(json.dumps(selection, indent=2))
    if winner["stage"] in ("ark", "hojo"):
        for corpus in ("all", "berst"):
            run(winner["stage"], winner["beam"], corpus, split="holdout", kind=winner["kind"],
                repetition_penalty=winner["repetition_penalty"])
    (root / "results/extra-search-complete.json").write_text(json.dumps(selection))


if __name__ == "__main__":
    main()
