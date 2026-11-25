"""Run a bounded, resumable search on development clips, then frozen holdout."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=["dev", "extended", "tune", "holdout"], default="dev")
    args = parser.parse_args()
    root = args.root
    program = root / "code/noisy_asr_search.py"

    def run(stage, route="auto", beam=1, kind="raw", language="en", split="dev", limit=36, corpus="all",
            length_penalty=1.0, threshold=0.5, lora_scale=1.0):
        name = f"{stage}-{route}-b{beam}-{kind}-{language}-{corpus}-{split}"
        if (length_penalty, threshold, lora_scale) != (1.0, 0.5, 1.0):
            name += f"-lp{length_penalty:g}-t{threshold:g}-a{lora_scale:g}"
        command = [sys.executable, str(program), stage, "--root", str(root), "--name", name,
                   "--corpus", corpus, "--split", split, "--limit", str(limit),
                   "--route", route, "--beam", str(beam), "--filter", kind, "--language", language]
        command.extend(["--length-penalty", str(length_penalty), "--threshold", str(threshold),
                        "--lora-scale", str(lora_scale)])
        if corpus == "berst":
            command.extend(["--manifest", str(root / "data/berst-manifest.jsonl")])
        if not (root / "results" / f"{name}.summary.json").exists():
            print(json.dumps({"starting": name}), flush=True)
            with (root / "results" / f"{name}.log").open("a") as output:
                subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((root / "results" / f"{name}.summary.json").read_text())
        print(json.dumps(summary), flush=True)
        return {"stage": stage, "route": route, "beam": beam, "kind": kind,
                "language": language, "corpus": corpus, "length_penalty": length_penalty,
                "threshold": threshold, "lora_scale": lora_scale, "summary": summary}

    if args.phase == "dev":
        results = []
        for stage, route, beam, language in [
            ("whisper", "auto", 2, "en"), ("whisper", "auto", 5, "en"),
            ("whisper", "auto", 8, "en"),
            ("mega", "base", 1, "en"), ("mega", "base", 2, "en"),
            ("mega", "base", 4, "en"),
            ("mega", "auto", 1, "en"), ("mega", "auto", 2, "en"),
            ("mega", "auto", 4, "en"),
            ("mega", "auto", 1, "auto"), ("mega", "base", 1, "auto"),
        ]:
            results.append(run(stage, route, beam, language=language))
        winner = min(results, key=lambda result: result["summary"]["wer"])
        for kind in ("hp80", "hp120", "denoise6", "denoise12", "level", "presence", "dfn3", "dfn3lim12"):
            results.append(run(winner["stage"], winner["route"], winner["beam"], kind,
                               winner["language"]))
        selected = min(results, key=lambda result: result["summary"]["wer"])
        (root / "results/development-selection.json").write_text(json.dumps(
            {"winner": selected, "runs": results}, indent=2))
        print(json.dumps({"selected": selected}), flush=True)
    elif args.phase == "extended":
        while not (root / "results/development-selection.json").exists():
            time.sleep(5)
        results = []
        candidates = [("whisper", "auto", 2), ("whisper", "auto", 5),
                      ("mega", "base", 1), ("mega", "base", 2), ("mega", "base", 4),
                      ("mega", "auto", 1), ("mega", "auto", 2),
                      ("mega", "mega", 1), ("mega", "mega", 2),
                      ("moss", "auto", 1), ("vibe", "auto", 1)]
        for stage, route, beam in candidates:
            broad = run(stage, route, beam)
            pocket = run(stage, route, beam, limit=84, corpus="berst")
            broad["pocket_summary"] = pocket["summary"]
            broad["selection_wer"] = (broad["summary"]["wer"] + pocket["summary"]["wer"]) / 2
            results.append(broad)
        winner = min(results, key=lambda result: result["selection_wer"])
        for kind in ("hp80", "hp120", "denoise6", "denoise12", "level", "presence", "dfn3", "dfn3lim12"):
            broad = run(winner["stage"], winner["route"], winner["beam"], kind)
            pocket = run(winner["stage"], winner["route"], winner["beam"], kind,
                         limit=84, corpus="berst")
            broad["pocket_summary"] = pocket["summary"]
            broad["selection_wer"] = (broad["summary"]["wer"] + pocket["summary"]["wer"]) / 2
            results.append(broad)
        selected = min(results, key=lambda result: result["selection_wer"])
        (root / "results/extended-selection.json").write_text(json.dumps(
            {"winner": selected, "runs": results}, indent=2))
        print(json.dumps({"selected": selected}), flush=True)
    elif args.phase == "tune":
        while not (root / "results/extended-selection.json").exists():
            time.sleep(5)
        selection = json.loads((root / "results/extended-selection.json").read_text())
        winner = selection["winner"]
        experiments = [dict(length_penalty=0.9), dict(length_penalty=1.1), dict(beam=4),
                       dict(kind="peak_norm"), dict(kind="rms_norm"), dict(kind="rms_hp80")]
        if winner["stage"] == "mega":
            experiments.extend(dict(route="mega", lora_scale=value) for value in (0.25, 0.5, 0.75))
            experiments.extend(dict(route="auto", threshold=value) for value in (0.05, 0.1, 0.25, 0.75))
        results = [winner]
        for experiment in experiments:
            config = {key: winner[key] for key in ("stage", "route", "beam", "kind", "language")}
            config.update(experiment)
            broad = run(**config)
            pocket = run(**config, limit=84, corpus="berst")
            broad["pocket_summary"] = pocket["summary"]
            broad["selection_wer"] = (broad["summary"]["wer"] + pocket["summary"]["wer"]) / 2
            results.append(broad)
        selected = min(results, key=lambda result: result["selection_wer"])
        (root / "results/final-selection.json").write_text(json.dumps(
            {"winner": selected, "runs": results}, indent=2))
        print(json.dumps({"selected": selected}), flush=True)
    else:
        while not (root / "results/final-selection.json").exists():
            time.sleep(5)
        selection = json.loads((root / "results/final-selection.json").read_text())
        winner = selection["winner"]
        candidates = {(winner["stage"], winner["route"], winner["beam"], winner["kind"], winner["language"]),
                      ("whisper", "auto", 2, "raw", "en"), ("whisper", "auto", 5, "raw", "en"),
                      ("mega", "auto", 1, "raw", "en"), ("mega", "base", 1, "raw", "en")}
        results = [run(*candidate, split="holdout", limit=0, corpus=corpus,
                       **({key: winner.get(key, default) for key, default in
                           (("length_penalty", 1.0), ("threshold", 0.5), ("lora_scale", 1.0))}
                          if candidate == (winner["stage"], winner["route"], winner["beam"], winner["kind"], winner["language"])
                          else {}))
                   for candidate in sorted(candidates) for corpus in ("all", "berst")]
        (root / "results/holdout-results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
