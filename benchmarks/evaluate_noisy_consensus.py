"""CPU-only dev comparison and frozen holdout/target consensus from cached text.

Dev compares anchored triples and five voters. Holdout/explicit inputs use only
the frozen base Qwen + Whisper + Vibe triple, never rank or reselect. The lexical
agreement gate and Vibe manifests are reference-free. References are optional
for explicit inputs and are used only for validation and subsequent scoring.
No audio, models, downloads, or GPU jobs are accessed or started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from itertools import combinations
from pathlib import Path


VOTERS = (
    "mega-base-b2-raw-en",
    "mega-mega-b2-raw-en",
    "whisper-auto-b2-raw-en",
    "vibe-auto-b1-raw-en",
    "moss-auto-b1-raw-en",
)
ANCHOR = VOTERS[0]
FROZEN_VOTERS = (ANCHOR, VOTERS[2], VOTERS[3])
FROZEN_PARAMETERS = {
    ANCHOR: {"beam": 2, "filter": "raw", "route": "base", "language": "en",
             "length_penalty": 1.0, "threshold": 0.5, "lora_scale": 1.0},
    VOTERS[2]: {"beam": 2, "filter": "raw", "language": "en", "length_penalty": 1.0},
    VOTERS[3]: {"beam": 1, "filter": "raw", "length_penalty": 1.0},
}
MANIFEST_FIELDS = ("id", "audio", "split", "corpus", "category", "group", "duration_s",
                   "device", "peak", "rms")
COHORT_COUNTS = {"all": 36, "berst": 84}
OUTPUT_NAME = "noisy-consensus-dev"
CAVEATS = [
    "Base Qwen and Mega are correlated Qwen-derived voters, not independent "
    "recognizers. Including both can double-count shared errors; the five-voter "
    "candidate also contains this pair.",
    "The unchanged merge helper uses anchor-relative word alignment and "
    "plurality voting, not calibrated confidence or a guaranteed strict "
    "majority. Noisy alignments and whole-gap insertion voting can lose words.",
    "The base Qwen transcript is first, preserving anchor word surfaces on "
    "tied word votes. The helper separately votes punctuation, rerenders "
    "whitespace, and tokenizes ASCII words; it does not preserve the raw "
    "anchor transcript byte-for-byte or use the scoring normalizer to merge.",
    "This is a dev-only candidate comparison. The best dev candidate is not "
    "a production choice or evidence of holdout improvement.",
]


def load_cohort(root: Path, cohort: str) -> tuple[dict, list[dict]]:
    tables = {}
    inputs = []
    for voter in VOTERS:
        path = root / "results" / f"{voter}-{cohort}-dev.jsonl"
        content = path.read_bytes()
        rows = {}
        for line_number, line in enumerate(content.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            for field in ("id", "audio", "reference", "text"):
                if not isinstance(row.get(field), str):
                    raise ValueError(f"{path}:{line_number}: {field} must be a string")
            if not row["id"] or not row["audio"]:
                raise ValueError(f"{path}:{line_number}: empty audio ID or path")
            if row.get("split") != "dev":
                raise ValueError(f"{path}:{line_number}: only explicit dev rows are allowed")
            if cohort == "berst" and row.get("corpus") != "berst":
                raise ValueError(f"{path}:{line_number}: expected BERSt corpus")
            if cohort == "all" and row.get("corpus") not in {"vitw", "krisp"}:
                raise ValueError(f"{path}:{line_number}: expected broad vitw/krisp corpus")
            if row["id"] in rows:
                raise ValueError(f"{path}:{line_number}: duplicate audio ID {row['id']!r}")
            rows[row["id"]] = row
        if len(rows) != COHORT_COUNTS[cohort]:
            raise ValueError(
                f"{path}: expected {COHORT_COUNTS[cohort]} clips, found {len(rows)}"
            )
        tables[voter] = rows
        inputs.append({"path": str(path), "sha256": hashlib.sha256(content).hexdigest()})

    anchor_rows = tables[ANCHOR]
    for voter, rows in tables.items():
        if rows.keys() != anchor_rows.keys():
            missing = sorted(anchor_rows.keys() - rows.keys())
            extra = sorted(rows.keys() - anchor_rows.keys())
            raise ValueError(f"{cohort}/{voter}: mismatched audio IDs; {missing=}, {extra=}")
        for audio_id, anchor_row in anchor_rows.items():
            for field in ("audio", "reference", "split", "corpus", "category"):
                if rows[audio_id].get(field) != anchor_row.get(field):
                    raise ValueError(f"{cohort}/{voter}/{audio_id}: mismatched {field}")
    return tables, inputs


def load_helpers():
    # Support both benchmarks/ in the repo and a flat ROOT/code deployment.
    directory = Path(__file__).resolve().parent
    for location in (directory.parent, directory):
        if str(location) not in sys.path:
            sys.path.insert(0, str(location))
    import noisy_asr_search
    import transcription_consensus

    helpers = [
        {"path": str(Path(module.__file__).resolve()),
         "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
        for module in (noisy_asr_search, transcription_consensus)
    ]
    return transcription_consensus.merge_transcripts, noisy_asr_search.score, helpers


def cohort_score(rows: list[dict], score) -> dict:
    metrics = score(rows)
    if metrics["wer"] is None or not math.isfinite(metrics["wer"]):
        raise ValueError("Scoring requires a finite corpus WER and nonempty reference words")
    return {"count": len(rows), **metrics}


def selection_wer(result: dict) -> float:
    return (result["all"]["wer"] + result["berst"]["wer"]) / 2


def atomic_write(path: Path, content: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_table(path: Path, split: str | None, *, manifest: bool = False) -> tuple[dict, dict]:
    path = path.expanduser().resolve()
    content = path.read_bytes()
    rows = {}
    seen = set()
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        for field in ("id", "audio") + (() if manifest else ("text",)):
            if not isinstance(row.get(field), str) or (field != "text" and not row[field]):
                raise ValueError(f"{path}:{line_number}: invalid {field}")
        if row["id"] in seen:
            raise ValueError(f"{path}:{line_number}: duplicate audio ID {row['id']!r}")
        seen.add(row["id"])
        if manifest and row.get("split") != split:
            continue
        if not manifest and split is not None and row.get("split", "target") != split:
            raise ValueError(f"{path}:{line_number}: expected {split} split")
        if not manifest and split is not None:
            row["split"] = split
        rows[row["id"]] = row
    return rows, {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}


def validate_alignment(anchor_rows: dict, rows: dict, label: str, *, references: bool) -> None:
    if rows.keys() != anchor_rows.keys():
        missing = sorted(anchor_rows.keys() - rows.keys())
        extra = sorted(rows.keys() - anchor_rows.keys())
        raise ValueError(f"{label}: mismatched audio IDs; {missing=}, {extra=}")
    for audio_id, anchor_row in anchor_rows.items():
        for field in ("audio", "split", "corpus", "category"):
            if rows[audio_id].get(field) != anchor_row.get(field):
                raise ValueError(f"{label}/{audio_id}: mismatched {field}")
        if references:
            reference_values = [row["reference"] for row in (anchor_row, rows[audio_id])
                                if "reference" in row]
            if any(not isinstance(value, str) for value in reference_values):
                raise ValueError(f"{label}/{audio_id}: reference must be a string when supplied")
            if len(reference_values) == 2 and reference_values[0] != reference_values[1]:
                raise ValueError(f"{label}/{audio_id}: mismatched reference")


def validate_parameters(rows: dict, voter: str, *, strict: bool = False) -> None:
    for audio_id, row in rows.items():
        required = {"beam", "filter"} | ({"route", "language"} if voter == ANCHOR else set())
        for field, expected in FROZEN_PARAMETERS[voter].items():
            if (strict and field in required and field not in row) or (
                field in row and row[field] != expected
            ):
                raise ValueError(f"{voter}/{audio_id}: frozen {field} must be {expected!r}")


def check_evidence(evidence: list[dict]) -> None:
    for item in evidence:
        if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Input/helper changed during evaluation: {item['path']}")


def jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)


def frozen_recipe() -> dict:
    return {
        "voters": list(FROZEN_VOTERS), "anchor": ANCHOR, "parameters": FROZEN_PARAMETERS,
        "freeze_source": "Selected on development data and frozen before holdout; hardcoded, never reselected from holdout scores",
        "gate": "Exact equality of parse_transcript word.normalized sequences for Qwen and Whisper",
        "agreement_fallback": "merge_transcripts([qwen_text, whisper_text, ''])",
        "disagreement_merge": "merge_transcripts([qwen_text, whisper_text, vibe_text])",
        "scorer": "benchmarks.noisy_asr_search.score",
        "lexical_proof": "Identical Qwen/Whisper sequences align equally: two votes retain each anchor "
                         "word and two empty votes suppress every insertion, for any third transcript.",
        "limitation": "Lexical-path equivalence only; punctuation/surface text can differ from a full "
                      "three-real-voter merge. No separate scoring-normalizer equivalence is claimed.",
    }


def load_frozen_pair(root: Path, split: str, *, references: bool) -> tuple[dict, list[dict]]:
    cohorts = {}
    inputs = []
    for cohort in COHORT_COUNTS:
        tables = {}
        for voter in FROZEN_VOTERS[:2]:
            path = root / "results" / f"{voter}-{cohort}-{split}.jsonl"
            tables[voter], evidence = read_table(path, split)
            inputs.append(evidence)
            validate_parameters(tables[voter], voter, strict=True)
        anchor_rows = tables[ANCHOR]
        if not anchor_rows:
            raise ValueError(f"{cohort}: empty frozen baseline cache")
        validate_alignment(anchor_rows, tables[FROZEN_VOTERS[1]], cohort, references=references)
        if split == "holdout":
            manifest_path = root / "data" / ("berst-manifest.jsonl" if cohort == "berst" else "manifest.jsonl")
            expected, evidence = read_table(manifest_path, split, manifest=True)
            inputs.append(evidence)
            validate_alignment(expected, anchor_rows, f"{cohort}/manifest", references=references)
            for voter in FROZEN_VOTERS[:2]:
                completion_path = root / "results" / f"{voter}-{cohort}-{split}.summary.json"
                content = completion_path.read_bytes()
                completion = json.loads(content)
                if not isinstance(completion, dict) or any(
                    completion.get(field) != value
                    for field, value in (("name", f"{voter}-{cohort}-{split}"),
                                         ("split", split), ("count", len(expected)))
                ):
                    raise ValueError(f"{completion_path}: frozen baseline run is not complete")
                inputs.append({"path": str(completion_path), "sha256": hashlib.sha256(content).hexdigest()})
        elif len(anchor_rows) != COHORT_COUNTS[cohort]:
            raise ValueError(f"{cohort}: expected {COHORT_COUNTS[cohort]} dev clips")
        expected_corpora = {"berst"} if cohort == "berst" else {"vitw", "krisp"}
        if any(row.get("corpus") not in expected_corpora for row in anchor_rows.values()):
            raise ValueError(f"{cohort}: mismatched corpus")
        cohorts[cohort] = tables
    if cohorts["all"][ANCHOR].keys() & cohorts["berst"][ANCHOR].keys():
        raise ValueError("Broad and BERSt cohorts must have disjoint audio IDs")
    return cohorts, inputs


def optional_score(rows: list[dict], score) -> dict:
    if not rows or not all("reference" in row for row in rows):
        return {"count": len(rows), "wer": None, "scored": False,
                "reason": "No complete references; no partial-cohort WER reported"}
    if any(not isinstance(row["reference"], str) for row in rows):
        raise ValueError("Reference must be a string when supplied")
    return {**cohort_score(rows, score), "scored": True}


def evaluate_frozen(
    split: str, *, root: Path | None = None, paths: tuple | None = None,
    output_prefix: Path | None = None, prepare_vibe: bool = False,
) -> dict:
    if split not in {"dev", "holdout", "target"}:
        raise ValueError(f"Unsupported split: {split}")
    if (root is None) == (paths is None):
        raise ValueError("Provide either --root or explicit Qwen/Whisper JSONL inputs, not both")
    if root is not None:
        if split == "target":
            raise ValueError("Target mode requires explicit JSONL inputs")
        root = root.expanduser().resolve()
        cohorts, inputs = load_frozen_pair(root, split, references=not prepare_vibe)
        output_prefix = output_prefix or root / "results" / f"noisy-consensus-{split}-gated"
    else:
        if output_prefix is None:
            raise ValueError("Explicit inputs require --output-prefix")
        if not paths or paths[0] is None or paths[1] is None:
            raise ValueError("Explicit inputs require Qwen and Whisper JSONL files")
        provided = [Path(path).expanduser().resolve() for path in paths if path is not None]
        if len(set(provided)) != len(provided):
            raise ValueError("Each explicit voter must have a distinct input file")
        tables = {}
        inputs = []
        for voter, path in zip(FROZEN_VOTERS[:2], paths[:2]):
            tables[voter], evidence = read_table(path, split)
            inputs.append(evidence)
            validate_parameters(tables[voter], voter)
        if not tables[ANCHOR]:
            raise ValueError("Explicit Qwen/Whisper inputs must not be empty")
        validate_alignment(tables[ANCHOR], tables[FROZEN_VOTERS[1]], "explicit inputs",
                           references=not prepare_vibe)
        cohorts = {"explicit": tables}

    output_prefix = output_prefix.expanduser().resolve()
    merge_transcripts, score, helpers = load_helpers()
    from transcription_consensus import parse_transcript

    def lexical_sequence(text):
        return [word.normalized for word in parse_transcript(text)]

    gate_rows = []
    manifests = {}
    gates = {}
    fallbacks = {}
    for cohort, tables in cohorts.items():
        disagreements = []
        counterexamples = []
        fallback_rows = {}
        for audio_id, anchor_row in tables[ANCHOR].items():
            qwen_text = anchor_row["text"]
            whisper_text = tables[FROZEN_VOTERS[1]][audio_id]["text"]
            qwen_sequence = lexical_sequence(qwen_text)
            whisper_sequence = lexical_sequence(whisper_text)
            agrees = qwen_sequence == whisper_sequence
            fallback = merge_transcripts([qwen_text, whisper_text, ""]) if agrees else None
            safe = agrees and lexical_sequence(fallback) == qwen_sequence
            if agrees and not safe:
                counterexamples.append({"id": audio_id, "qwen_text": qwen_text,
                                        "whisper_text": whisper_text, "dummy_vibe_text": "",
                                        "fallback": fallback, "expected_lexical_sequence": qwen_sequence})
            gate_rows.append({"id": audio_id, "audio": anchor_row["audio"], "cohort": cohort,
                              "split": split, "qwen_text": qwen_text, "whisper_text": whisper_text,
                              "qwen_lexical_sequence": qwen_sequence,
                              "whisper_lexical_sequence": whisper_sequence,
                              "lexical_agreement": agrees, "skip_vibe": safe})
            if safe:
                fallback_rows[audio_id] = fallback
            else:
                manifest_row = {field: anchor_row[field] for field in MANIFEST_FIELDS if field in anchor_row}
                manifest_row["split"] = split
                disagreements.append(manifest_row)
        fallbacks[cohort] = fallback_rows
        manifests[cohort] = disagreements
        gates[cohort] = {"count": len(tables[ANCHOR]), "vibe_skipped": len(fallback_rows),
                         "vibe_required": len(disagreements), "lexical_counterexamples": counterexamples}

    predictions = []
    candidate = {"name": "consensus-" + "+".join(FROZEN_VOTERS), "voters": list(FROZEN_VOTERS),
                 "anchor": ANCHOR, "frozen": True, "contains_correlated_qwen_pair": False}
    baselines = []
    if not prepare_vibe:
        for cohort, tables in cohorts.items():
            needed = {row["id"] for row in manifests[cohort]}
            vibe_path = paths[2] if paths is not None and len(paths) > 2 else None
            if root is not None:
                full_path = root / "results" / f"{FROZEN_VOTERS[2]}-{cohort}-{split}.jsonl"
                sparse_path = root / "results" / f"{FROZEN_VOTERS[2]}-{cohort}-{split}-disagreements.jsonl"
                vibe_path = full_path if full_path.exists() else sparse_path if sparse_path.exists() else None
            vibe_rows = {}
            if vibe_path is not None:
                vibe_rows, evidence = read_table(vibe_path, split)
                inputs.append(evidence)
                if vibe_rows.keys() != tables[ANCHOR].keys() and vibe_rows.keys() != needed:
                    raise ValueError(f"{cohort}: Vibe IDs must match the full cohort or exact disagreement set")
                validate_parameters(vibe_rows, FROZEN_VOTERS[2], strict=root is not None)
                validate_alignment({key: tables[ANCHOR][key] for key in vibe_rows}, vibe_rows,
                                   f"{cohort}/Vibe", references=True)
                validate_alignment({key: tables[FROZEN_VOTERS[1]][key] for key in vibe_rows}, vibe_rows,
                                   f"{cohort}/Whisper-Vibe", references=True)
            elif needed:
                raise ValueError(f"{cohort}: need Vibe cache for {len(needed)} disagreements; use --prepare-vibe first")
            rows = []
            for audio_id, anchor_row in tables[ANCHOR].items():
                skipped = audio_id in fallbacks[cohort]
                if skipped:
                    text = fallbacks[cohort][audio_id]
                    if audio_id in vibe_rows:
                        full_text = merge_transcripts([anchor_row["text"],
                                                      tables[FROZEN_VOTERS[1]][audio_id]["text"],
                                                      vibe_rows[audio_id]["text"]])
                        if lexical_sequence(full_text) != lexical_sequence(text):
                            raise ValueError(f"{cohort}/{audio_id}: lexical gate counterexample with real Vibe")
                else:
                    text = merge_transcripts([anchor_row["text"],
                                              tables[FROZEN_VOTERS[1]][audio_id]["text"],
                                              vibe_rows[audio_id]["text"]])
                row = {field: anchor_row[field] for field in MANIFEST_FIELDS + ("reference",) if field in anchor_row}
                row.update({"split": split, "cohort": cohort, "candidate": candidate["name"],
                            "voters": list(FROZEN_VOTERS), "anchor": ANCHOR,
                            "vibe_skipped": skipped, "text": text})
                rows.append(row)
            candidate[cohort] = optional_score(rows, score)
            predictions.extend(rows)
        if set(cohorts) == {"all", "berst"} and all(candidate[key]["wer"] is not None for key in cohorts):
            candidate["mean_wer"] = selection_wer(candidate)
        for voter in FROZEN_VOTERS[:2]:
            baseline = {"name": voter, "voters": [voter]}
            for cohort, tables in cohorts.items():
                baseline[cohort] = optional_score([dict(row) for row in tables[voter].values()], score)
            baselines.append(baseline)

    artifact_paths = {"summary": Path(f"{output_prefix}.summary.json"),
                      "freeze": Path(f"{output_prefix}.frozen.json"),
                      "gate": Path(f"{output_prefix}.gate.jsonl")}
    manifest_paths = {cohort: Path(f"{output_prefix}.{cohort}.vibe-disagreements.jsonl") for cohort in cohorts}
    if not prepare_vibe:
        artifact_paths["predictions"] = Path(f"{output_prefix}.predictions.jsonl")
    input_paths = {Path(item["path"]) for item in inputs + helpers}
    if input_paths & (set(artifact_paths.values()) | set(manifest_paths.values())):
        raise ValueError("Output artifacts must not overwrite input/helper files")
    recipe = frozen_recipe()
    if artifact_paths["freeze"].exists():
        previous = json.loads(artifact_paths["freeze"].read_text())
        frozen_keys = ("voters", "anchor", "parameters", "gate", "agreement_fallback",
                       "disagreement_merge", "scorer")
        if not isinstance(previous, dict) or any(previous.get(key) != recipe[key] for key in frozen_keys):
            raise ValueError("Output prefix already has a different frozen recipe; use a new output prefix")
    check_evidence(inputs + helpers)
    summary = {
        "name": output_prefix.name, "split": split, "mode": "prepare-vibe" if prepare_vibe else "frozen-gated",
        "frozen_recipe": recipe, "gate": gates, "inputs": inputs, "helpers": helpers,
        "candidate": None if prepare_vibe else candidate, "baselines": baselines,
        "ranking_performed": False, "reference_free_gate": True,
        "caveats": CAVEATS[:3] + [recipe["limitation"],
                                   "Frozen voter identity and gate never depend on holdout/target scores."],
        "artifacts": {**{key: str(path) for key, path in artifact_paths.items()},
                      "vibe_disagreements": {key: str(path) for key, path in manifest_paths.items()},
                      "prediction_count": len(predictions)},
    }
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(artifact_paths["freeze"], json.dumps(recipe, indent=2) + "\n")
    atomic_write(artifact_paths["gate"], jsonl(gate_rows))
    for cohort, path in manifest_paths.items():
        atomic_write(path, jsonl(manifests[cohort]))
    if not prepare_vibe:
        predictions_text = jsonl(predictions)
        summary["artifacts"]["predictions_sha256"] = hashlib.sha256(predictions_text.encode("utf-8")).hexdigest()
        atomic_write(artifact_paths["predictions"], predictions_text)
    atomic_write(artifact_paths["summary"], json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return summary


def evaluate(root: Path, split: str = "dev", *, output_prefix: Path | None = None,
             prepare_vibe: bool = False) -> dict:
    if split != "dev" or prepare_vibe:
        return evaluate_frozen(split, root=root, output_prefix=output_prefix, prepare_vibe=prepare_vibe)
    root = root.expanduser().resolve()
    completion_path = root / "results" / f"{VOTERS[3]}-berst-dev.summary.json"
    completion_content = completion_path.read_bytes()
    completion = json.loads(completion_content)
    if not isinstance(completion, dict) or any(
        completion.get(field) != expected
        for field, expected in (
            ("name", f"{VOTERS[3]}-berst-dev"), ("split", "dev"), ("count", 84),
        )
    ):
        raise ValueError(f"{completion_path}: raw Vibe BERSt dev run is not complete")

    cohorts = {}
    inputs = [{"path": str(completion_path),
               "sha256": hashlib.sha256(completion_content).hexdigest()}]
    for cohort in COHORT_COUNTS:
        cohorts[cohort], cohort_inputs = load_cohort(root, cohort)
        inputs.extend(cohort_inputs)
    if cohorts["all"][ANCHOR].keys() & cohorts["berst"][ANCHOR].keys():
        raise ValueError("Broad and BERSt cohorts must have disjoint audio IDs")
    merge_transcripts, score, helpers = load_helpers()

    baselines = []
    for voter in VOTERS:
        result = {"name": voter, "voters": [voter]}
        for cohort, tables in cohorts.items():
            result[cohort] = cohort_score([dict(row) for row in tables[voter].values()], score)
        result["selection_wer"] = selection_wer(result)
        baselines.append(result)
    anchor_baseline = baselines[0]

    candidates = []
    predictions = []
    voter_sets = [(ANCHOR, *pair) for pair in combinations(VOTERS[1:], 2)] + [VOTERS]
    for voters in voter_sets:
        name = "consensus-" + "+".join(voters)
        result = {"name": name, "voters": list(voters), "anchor": ANCHOR,
                  "contains_correlated_qwen_pair": VOTERS[1] in voters}
        for cohort, tables in cohorts.items():
            rows = []
            for audio_id, anchor_row in tables[ANCHOR].items():
                # Only cached hypothesis strings reach the merge helper.
                text = merge_transcripts([tables[voter][audio_id]["text"] for voter in voters])
                row = {field: anchor_row[field]
                       for field in ("id", "audio", "reference", "split", "corpus")}
                if "category" in anchor_row:
                    row["category"] = anchor_row["category"]
                row.update({"candidate": name, "cohort": cohort, "anchor": ANCHOR,
                            "voters": list(voters), "text": text})
                rows.append(row)
            result[cohort] = cohort_score(rows, score)
            predictions.extend(rows)
        result["selection_wer"] = selection_wer(result)
        result["delta_vs_anchor"] = result["selection_wer"] - anchor_baseline["selection_wer"]
        candidates.append(result)
    candidates.sort(key=lambda result: result["selection_wer"])
    baselines.sort(key=lambda result: result["selection_wer"])

    # Refuse to publish a comparison if another process changed its evidence.
    check_evidence(inputs + helpers)

    prefix = (output_prefix or root / "results" / OUTPUT_NAME).expanduser().resolve()
    predictions_path = Path(f"{prefix}.predictions.jsonl")
    summary_path = Path(f"{prefix}.summary.json")
    predictions_text = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in predictions
    )
    summary = {
        "name": OUTPUT_NAME, "split": "dev", "evaluation": "CPU-only offline cached hypotheses",
        "anchor": ANCHOR, "available_voters": list(VOTERS),
        "cohort_counts": COHORT_COUNTS,
        "selection_objective": "arithmetic mean of corpus WERs: (broad/all WER + BERSt WER) / 2",
        "scorer": "benchmarks.noisy_asr_search.score",
        "normalizer": "whisper_normalizer.english.EnglishTextNormalizer",
        "merge_helper": "transcription_consensus.merge_transcripts",
        "anchor_tie_policy": "base Qwen first; unchanged helper preserves tied anchor word surfaces",
        "caveats": CAVEATS, "inputs": inputs, "helpers": helpers,
        "baselines": baselines, "candidates": candidates,
        "best_dev_candidate": candidates[0], "best_dev_single_voter": baselines[0],
        "artifacts": {"summary": str(summary_path), "predictions": str(predictions_path),
                      "prediction_count": len(predictions),
                      "predictions_sha256": hashlib.sha256(predictions_text.encode("utf-8")).hexdigest()},
    }
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if {predictions_path, summary_path} & {Path(item["path"]) for item in inputs + helpers}:
        raise ValueError("Output artifacts must not overwrite input/helper files")
    atomic_write(predictions_path, predictions_text)
    atomic_write(summary_path, json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Search root containing cached results/ and holdout data manifests")
    parser.add_argument("--split", choices=("dev", "holdout", "target"),
                        help="Default dev with --root, target with explicit inputs; holdout never reselects")
    parser.add_argument("--qwen-jsonl", "--qwen-input", type=Path, help="Explicit base Qwen b2 raw predictions")
    parser.add_argument("--whisper-jsonl", "--whisper-input", type=Path, help="Explicit Whisper b2 raw predictions")
    parser.add_argument("--vibe-jsonl", "--vibe-input", type=Path, help="Explicit Vibe b1 raw predictions: full or disagreement-only")
    parser.add_argument("--output-prefix", "--outputprefix", type=Path, help="Artifact path prefix; required with explicit inputs")
    parser.add_argument("--prepare-vibe", action="store_true",
                        help="Only generate reference-free gate audit/frozen recipe/Vibe disagreement manifests; no scores")
    args = parser.parse_args()
    explicit = any(path is not None for path in (args.qwen_jsonl, args.whisper_jsonl, args.vibe_jsonl))
    split = args.split or ("target" if explicit else "dev")
    if explicit and (args.root is not None or args.qwen_jsonl is None or args.whisper_jsonl is None):
        parser.error("Explicit mode requires Qwen and Whisper JSONL inputs, without --root")
    if explicit and not args.prepare_vibe and args.vibe_jsonl is None:
        parser.error("Provide all three JSONL inputs, or use --prepare-vibe with Qwen/Whisper only")
    if args.prepare_vibe and args.vibe_jsonl is not None:
        parser.error("--prepare-vibe reads Qwen/Whisper only; omit --vibe-jsonl")
    if not explicit and args.root is None:
        parser.error("Provide --root or explicit JSONL inputs")
    try:
        if explicit:
            summary = evaluate_frozen(split, paths=(args.qwen_jsonl, args.whisper_jsonl, args.vibe_jsonl),
                                      output_prefix=args.output_prefix, prepare_vibe=args.prepare_vibe)
        else:
            summary = evaluate(args.root, split, output_prefix=args.output_prefix, prepare_vibe=args.prepare_vibe)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    print(json.dumps({key: summary[key] for key in (
        "artifacts", "candidates", "candidate", "baselines", "gate", "frozen_recipe", "caveats"
    ) if key in summary}, indent=2))


if __name__ == "__main__":
    main()
