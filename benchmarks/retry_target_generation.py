"""Retry capped target generations without replacing or concealing raw evidence."""

import argparse
import json
from pathlib import Path

import complete_target_coverage as coverage


INPUT_FIELDS = (
    "id", "audio", "split", "corpus", "start", "end", "duration_s",
    "start_sample", "end_sample", "sample_rate", "source_audio", "source_sha256",
    "vad_rejected", "review_required", "coverage_origin", "reference_is_verified",
    "speech_is_verified",
)


def retry_manifest(rows):
    warned = {row["id"] for row in coverage.audit_generation(rows)}
    return [{key: row[key] for key in INPUT_FIELDS} for row in rows if row["id"] in warned]


def merge_retries(rows, retries):
    by_id = {row["id"]: row for row in retries}
    if len(by_id) != len(retries):
        raise ValueError("Duplicate retry IDs")
    expected = {row["id"] for row in coverage.audit_generation(rows)}
    if set(by_id) != expected:
        raise ValueError("Retry IDs must exactly match the generation-warning rows")
    merged = []
    for row in rows:
        candidate = by_id.get(row["id"])
        if candidate is None:
            merged.append(row)
            continue
        for key in INPUT_FIELDS:
            if candidate.get(key) != row.get(key):
                raise ValueError(f"Changed retry input {row['id']}: {key}")
        if candidate.get("sampling_seed") != row.get("sampling_seed"):
            raise ValueError(f"Changed retry seed: {row['id']}")
        accepted = not coverage.audit_generation([candidate])
        recovery = {"trigger": "generation audit warning", "accepted": accepted,
                    "original_text": row["text"], "original_generation": row.get("generation"),
                    "retry_text": candidate["text"], "retry_generation": candidate.get("generation"),
                    "not_an_accuracy_verification": True}
        merged.append({**row, **(candidate if accepted else {}), "generation_recovery": recovery})
    return merged


def run(root, basename):
    context = coverage.prepare_coverage(root)
    coverage.validate_parent(root, context)
    prefix = root / "results" / basename
    evidence = coverage.read_json(prefix.with_suffix(".coverage.json"))
    digest = coverage.sha256_file(prefix.with_suffix(".jsonl"))
    if (not evidence.get("complete") or evidence.get("merged_results_sha256") != digest or
            evidence.get("preparation_fingerprint") != context["fingerprint"]):
        raise ValueError("Requires a completed, fingerprint-matched full-source transcript")
    rows = coverage.read_jsonl(prefix.with_suffix(".jsonl"))
    coverage.validate_full_coverage(
        [coverage.row_interval(row, context["source"]["frames"], 16000) for row in rows],
        context["source"]["frames"])
    manifest_rows = retry_manifest(rows)
    name = basename + "_greedy_retries"
    manifest = root / "data" / (name + ".jsonl")
    state = {"source_prediction_sha256": digest,
             "retry_helper_sha256": coverage.sha256_file(Path(__file__)),
             "preparation_fingerprint": context["fingerprint"],
             "configuration": {**context["configuration"], "beam": 1, "route": "base"}}
    coverage.publish(manifest.with_suffix(".state.json"), coverage.json_bytes(state))
    coverage.publish(manifest, coverage.jsonl_bytes(manifest_rows))
    retry_context = {**context, "configuration": state["configuration"], "manifest": manifest}
    retries = coverage.complete_run(root, retry_context, name, manifest_rows, False)
    merged = merge_retries(rows, retries)
    output = root / "results" / (basename + "_recovered")
    warnings = coverage.publish_transcript(root, output, merged, 16000)
    result = {**evidence, "source_prediction_sha256": digest,
              "merged_results_sha256": coverage.sha256_file(output.with_suffix(".jsonl")),
              "generation_warning_rows": len(warnings),
              "warnings_file": str(output.with_suffix(".warnings.json")),
              "retry_rows": len(retries),
              "accepted_retries": sum(row.get("generation_recovery", {}).get("accepted", False)
                                      for row in merged),
              "retry_configuration": state["configuration"],
              "note": "Greedy retries accepted only on complete generation. EOS does not verify speech or improve target WER."}
    coverage.publish(output.with_suffix(".coverage.json"), coverage.json_bytes(result))
    print(json.dumps({"output": str(output), "retry_rows": len(retries),
                      "accepted": result["accepted_retries"], "remaining_warnings": len(warnings)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--basename", choices=[coverage.FULL_NAME, coverage.ROUTER_NAME], required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    with coverage.coverage_lock(root):
        run(root, args.basename)


if __name__ == "__main__":
    main()
