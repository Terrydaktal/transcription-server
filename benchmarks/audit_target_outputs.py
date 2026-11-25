"""Publish a conservative reader view and audit the rejected router experiment."""

import argparse
import json
from pathlib import Path

import complete_target_coverage as coverage


def reader_view(rows, rate):
    warned = {row["id"] for row in coverage.audit_generation(rows)}
    parts = ["Qwen ASR transcript - no verified target reference.",
             "Unfinished decoder outputs are marked, not asserted as speech.", ""]
    timestamped = parts + ["Quiet-region ASR remains unverified; see the full review transcript.", ""]
    for row in rows:
        start = coverage.sample_stamp(row["start_sample"], rate)
        end = coverage.sample_stamp(row["end_sample"], rate)
        text = row["text"].strip()
        if row["id"] in warned:
            text = f"[ASR did not finish at {start} - {end}; review the original audio.]"
        if text:
            parts.extend([text, ""])
        flags = " [LOW-VAD / REVIEW]" if row["vad_rejected"] else ""
        timestamped.extend([f"[{start} - {end}]{flags}", text, ""])
    return "\n".join(parts) + "\n", "\n".join(timestamped) + "\n", len(warned)


def router_audit(primary, routed, requested):
    by_id = {row["id"]: row for row in primary}
    if len(by_id) != len(primary) or len({row["id"] for row in routed}) != len(routed):
        raise ValueError("Duplicate IDs in router evidence")
    paired = []
    for row in routed:
        if row["id"] not in by_id:
            raise ValueError(f"Unknown router result: {row['id']}")
        base = by_id[row["id"]]
        for key in ("audio", "start", "end", "sampling_seed"):
            if row[key] != base[key]:
                raise ValueError(f"Unpaired router evidence: {row['id']} {key}")
        if row.get("route") != "auto" or row.get("lora_active") is not True:
            raise ValueError("Expected only adapter-selected router rows")
        paired.append(base)
    return {"outcome": "rejected_early", "completed_rows": len(routed),
            "requested_rows": requested, "is_full_target_comparison": False,
            "router_generation_warning_rows": len(coverage.audit_generation(routed)),
            "paired_base_generation_warning_rows": len(coverage.audit_generation(paired)),
            "reason": "Much higher capped/unfinished generation rate and repetition loops on paired target audio; not promoted.",
            "target_wer": None, "verified_reference": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    context = coverage.prepare_coverage(root)
    coverage.validate_parent(root, context)
    prefix = root / "results" / (coverage.FULL_NAME + "_recovered")
    evidence = coverage.read_json(prefix.with_suffix(".coverage.json"))
    if (not evidence.get("complete") or evidence["merged_results_sha256"] !=
            coverage.sha256_file(prefix.with_suffix(".jsonl"))):
        raise ValueError("Reader view requires validated, completed source evidence")
    rows = coverage.read_jsonl(prefix.with_suffix(".jsonl"))
    coverage.validate_full_coverage([coverage.row_interval(row, context["source"]["frames"], 16000)
                                     for row in rows], context["source"]["frames"])
    plain, timestamped, warnings = reader_view(rows, 16000)
    output = root / "results/Recording_1828_recommended_20261002"
    coverage.publish(output.with_suffix(".txt"), plain.encode("utf-8"))
    coverage.publish(output.with_name(output.name + "_timestamped.txt"), timestamped.encode("utf-8"))
    coverage.publish(output.with_suffix(".audit.json"), coverage.json_bytes(
        {"source_results": str(prefix.with_suffix(".jsonl")),
         "source_results_sha256": evidence["merged_results_sha256"],
         "source_samples": context["source"]["frames"], "display_warning_rows": warnings,
         "text_source": "Native ASR API, with completed greedy retries only on capped rows",
         "reader_rule": "Replace unfinished decode text with a review marker; raw evidence remains unchanged",
         "target_wer": None, "verified_reference": False}))
    original = coverage.read_jsonl(root / "results" / (coverage.FULL_NAME + ".jsonl"))
    routed_path = root / "results" / (coverage.ROUTED_NAME + ".jsonl")
    routed = coverage.read_jsonl(routed_path)
    requested = len(coverage.read_jsonl(root / "data/target-router-manifest.jsonl"))
    audit = {**router_audit(original, routed, requested),
             "preserved_results": str(routed_path), "results_sha256": coverage.sha256_file(routed_path)}
    coverage.publish(root / "results/target-router-rejected.json", coverage.json_bytes(audit))
    print(json.dumps({"recommended": str(output.with_suffix(".txt")),
                      "warning_markers": warnings, "router_audit": audit}), flush=True)


if __name__ == "__main__":
    main()
