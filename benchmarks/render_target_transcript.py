"""Render timestamped target ASR results without treating an older ASR as truth."""

import argparse
import json
from pathlib import Path


def stamp(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    rows = sorted([json.loads(line) for line in args.input.read_text().splitlines() if line],
                  key=lambda row: row["start"])
    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    args.prefix.with_suffix(".txt").write_text("\n\n".join(row["text"].strip() for row in rows if row["text"].strip()) + "\n")
    args.prefix.with_name(args.prefix.name + "_timestamped.txt").write_text("\n\n".join(
        f"[{stamp(row['start'])} - {stamp(row['end'])}] {row['text'].strip()}"
        for row in rows if row["text"].strip()) + "\n")
    args.prefix.with_suffix(".json").write_text(json.dumps(
        {"verified_reference": False, "source_results": str(args.input), "segments": rows},
        indent=2, ensure_ascii=False))
    if args.baseline:
        old = json.loads(args.baseline.read_text())["segments"]
        comparison = ["# Target transcript comparison", "",
                      "Neither column is verified ground truth. Listen to disputed spans before asserting accuracy.", ""]
        for row in rows:
            previous = " ".join(segment["text"].strip() for segment in old
                                if segment["end"] > row["start"] and segment["start"] < row["end"])
            comparison.extend([f"## {stamp(row['start'])} - {stamp(row['end'])}", "",
                               f"Old Whisper: {previous}", "", f"New: {row['text']}", ""])
        args.prefix.with_name(args.prefix.name + "_comparison.md").write_text("\n".join(comparison))
    print(json.dumps({"segments": len(rows), "nonempty": sum(bool(row["text"].strip()) for row in rows),
                      "transcript": str(args.prefix.with_suffix(".txt"))}), flush=True)


if __name__ == "__main__":
    main()
