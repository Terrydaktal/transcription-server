"""Score saved predictions by corpus and bootstrap paired, group-level WER deltas."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate audio IDs: {path}")
    return rows


def counts(rows):
    from jiwer import process_words
    from whisper_normalizer.english import EnglishTextNormalizer

    normalize = EnglishTextNormalizer()
    result = dict(substitutions=0, deletions=0, insertions=0, reference_words=0)
    for row in rows:
        aligned = process_words(normalize(row["reference"]), normalize(row["text"]))
        for key in ("substitutions", "deletions", "insertions"):
            result[key] += getattr(aligned, key)
        result["reference_words"] += aligned.hits + aligned.substitutions + aligned.deletions
    result["errors"] = sum(result[key] for key in ("substitutions", "deletions", "insertions"))
    result["wer"] = result["errors"] / result["reference_words"] if result["reference_words"] else None
    return result


def paired_bootstrap(baseline, candidate, iterations=5000):
    import numpy as np

    by_id = {row["id"]: row for row in candidate}
    if set(by_id) != {row["id"] for row in baseline}:
        raise ValueError("Paired comparison needs identical audio IDs")
    groups = defaultdict(lambda: [[], []])
    for row in baseline:
        other = by_id[row["id"]]
        for key in ("reference", "audio", "split", "group", "corpus", "category"):
            if row.get(key) != other.get(key):
                raise ValueError(f"Mismatched {key}: {row['id']}")
        group = (row["corpus"], row["group"])
        groups[group][0].append(row)
        groups[group][1].append(other)
    totals = []
    for old, new in groups.values():
        a, b = counts(old), counts(new)
        totals.append((a["errors"], b["errors"], a["reference_words"]))
    values = np.asarray(totals, dtype=np.float64)
    rng = np.random.default_rng(20261002)
    draws = values[rng.integers(0, len(values), size=(iterations, len(values)))].sum(axis=1)
    deltas = (draws[:, 1] - draws[:, 0]) / draws[:, 2]
    return {"groups": len(groups), "iterations": iterations,
            "candidate_minus_baseline_wer": float((values[:, 1].sum() - values[:, 0].sum()) / values[:, 2].sum()),
            "ci95": np.quantile(deltas, [0.025, 0.975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-prefix", default="whisper-auto-b2-raw-en")
    parser.add_argument("--candidate-prefix", default="mega-base-b2-raw-en")
    args = parser.parse_args()
    root = args.root
    summaries = []
    for path in sorted((root / "results").glob("*.summary.json")):
        summary = json.loads(path.read_text())
        data = path.with_name(path.name.replace(".summary.json", ".jsonl"))
        if not data.exists():
            continue
        rows = read_rows(data)
        if not rows or "reference" not in rows[0]:
            continue
        for dimension in ("all", "corpus", "category"):
            buckets = defaultdict(list)
            for row in rows:
                buckets["all" if dimension == "all" else row[dimension]].append(row)
            for label, subset in sorted(buckets.items()):
                summaries.append({"name": summary["name"], "split": summary["split"],
                                  "dimension": dimension, "label": label, "clips": len(subset),
                                  **counts(subset), "infer_s": sum(row["infer_s"] for row in subset),
                                  "audio_s": sum(row["duration_s"] for row in subset),
                                  "peak_allocated_gib": max(row["peak_allocated_gib"] for row in subset),
                                  "cap_hits": sum(any(part.get("token_cap_hit", False)
                                                      for part in row.get("generation", [])) for row in subset),
                                  "empty_outputs": sum(not row["text"].strip() for row in subset)})
    comparisons = {}
    for corpus in ("all", "berst"):
        old = root / "results" / f"{args.baseline_prefix}-{corpus}-holdout.jsonl"
        new = root / "results" / f"{args.candidate_prefix}-{corpus}-holdout.jsonl"
        markers = [path.with_name(path.name.replace(".jsonl", ".summary.json")) for path in (old, new)]
        if not old.exists() or not new.exists() or not all(path.exists() for path in markers):
            continue
        baseline, candidate = read_rows(old), read_rows(new)
        for label in sorted({row["corpus"] for row in baseline}):
            a = [row for row in baseline if row["corpus"] == label]
            b = [row for row in candidate if row["corpus"] == label]
            comparisons[label] = paired_bootstrap(a, b)
    report = {"normalizer": "whisper_normalizer.english.EnglishTextNormalizer + jiwer",
              "bootstrap": "paired resampling of speaker/source groups; 95% percentile interval",
              "summaries": summaries, "comparisons": comparisons}
    (root / "results/all-scores.json").write_text(json.dumps(report, indent=2))
    with (root / "results/all-scores.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps({"summaries": len(summaries), "comparisons": comparisons}), flush=True)


if __name__ == "__main__":
    main()
