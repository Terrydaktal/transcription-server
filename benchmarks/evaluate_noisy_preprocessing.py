"""CPU-only audio-gated preprocessing search, freeze, manifests, and evaluation.

Search reads only the named legacy dev prediction caches and original WAVs.
Gate writes reference-free raw/filtered subsets using the frozen audio rule.
Evaluate combines raw predictions with filtered predictions for triggered IDs
only. No command loads models, creates filtered audio, or starts GPU work.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path


FILTERS = ("hp80", "hp120", "denoise6", "dfn3lim12")
COHORTS = {"all": 36, "berst": 84}
RATIOS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7)
RMS_DBFS = (-40, -35, -30)
BANDS = (80, 120)
COUNTS = ("substitutions", "deletions", "insertions", "reference_words")
DECODER = {"stage": "mega", "route": "base", "beam": 2, "language": "en", "length_penalty": 1.0}
FEATURE_CONFIG = {
    "method": "scipy.signal.welch", "sample_rate": 16000, "window": "hann",
    "segment_seconds": 1, "overlap_fraction": 0.5, "detrend": False,
    "scaling": "density", "average": "mean", "bands_hz": list(BANDS),
    "low_band": "0 <= frequency < cutoff; includes DC",
    "power": "sum(one-sided PSD bins) * sample_rate / nperseg",
    "ratio": "low-band power / total power; zero for silent audio",
    "rms_dbfs": "10 * log10(low-band power); silent bands never trigger",
}
MANIFEST_FIELDS = ("id", "audio", "split", "corpus", "category", "group", "duration_s", "device")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(content):
    return hashlib.sha256(content).hexdigest()


def source(path):
    return {"path": str(path.resolve()), "sha256": sha(path.read_bytes())}


def write_new(path, value, *, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = ("".join(json.dumps(row, allow_nan=False) + "\n" for row in value) if jsonl else
               json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def seal(value):
    return {**value, "artifact_sha256": sha(canonical(value))}


def read_sealed(path):
    value = json.loads(path.read_text())
    expected = value.pop("artifact_sha256")
    if sha(canonical(value)) != expected:
        raise ValueError(f"Changed artifact: {path}")
    return {**value, "artifact_sha256": expected}


def read_table(path):
    content = path.read_bytes()
    rows = {}
    for line in content.decode().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in rows:
            raise ValueError(f"Missing or duplicate ID in {path}")
        rows[row["id"]] = row
    return rows, {"path": str(path.resolve()), "sha256": sha(content), "count": len(rows)}


def load_scorer():
    path = Path(__file__).resolve().with_name("noisy_asr_search.py")
    spec = importlib.util.spec_from_file_location("preprocessing_benchmark_scorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score, {**source(path), "function_sha256": sha(inspect.getsource(module.score).encode())}


def validate_prediction(row, kind, *, legacy=False):
    expected = {**DECODER, "filter": kind}
    for key, value in expected.items():
        actual = row.get(key, value if legacy and key in ("stage", "length_penalty") else None)
        if actual != value:
            raise ValueError(f"{row['id']}: expected {key}={value!r}, got {actual!r}")
    if row.get("lora_active") is not False or not isinstance(row.get("text"), str):
        raise ValueError(f"{row['id']}: expected inactive LoRA and a transcript string")


def audio_features(path):
    import numpy as np
    import soundfile as sf
    from scipy.signal import welch

    content = path.read_bytes()
    audio, rate = sf.read(io.BytesIO(content), dtype="float64")
    if rate != FEATURE_CONFIG["sample_rate"] or audio.ndim != 1 or not len(audio):
        raise ValueError(f"Expected nonempty mono 16 kHz original WAV: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Nonfinite audio: {path}")
    segment = min(rate, len(audio))
    frequencies, density = welch(audio, fs=rate, window="hann", nperseg=segment,
                                 noverlap=segment // 2, detrend=False, scaling="density", average="mean")
    total_power = float(density.sum() * rate / segment)
    bands = {}
    for cutoff in BANDS:
        power = float(density[frequencies < cutoff].sum() * rate / segment)
        bands[str(cutoff)] = {"lf_power_ratio": power / total_power if total_power else 0.0,
                             "low_band_rms_dbfs": 10 * math.log10(power) if power > 0 else None}
    return {"audio": str(path), "audio_sha256": sha(content), "sample_rate": rate,
            "sample_count": len(audio), "total_power": total_power, "bands": bands}


def triggered(features, rule):
    if rule["filter"] == "raw":
        return False
    band = features["bands"][str(rule["cutoff_hz"])]
    rms = band["low_band_rms_dbfs"]
    return (band["lf_power_ratio"] >= rule["ratio_threshold"] and rms is not None
            and rms >= rule["low_band_rms_dbfs_threshold"])


def errors(metrics):
    return sum(metrics[key] for key in COUNTS[:3])


def metrics_sum(metrics):
    result = {key: sum(row[key] for row in metrics) for key in COUNTS}
    result.update(count=len(metrics), errors=errors(result))
    result["wer"] = result["errors"] / result["reference_words"] if result["reference_words"] else None
    return result


def objective(result):
    return sum((Fraction(result[cohort]["errors"], result[cohort]["reference_words"])
                for cohort in COHORTS), Fraction()) / len(COHORTS)


def policy_result(tables, counts, features, rule):
    result = {"rule": rule, "triggered_clip_ids": {}, "clip_changes": []}
    for cohort in COHORTS:
        selected = []
        ids = []
        for identity in tables[cohort]["raw"]:
            use_filter = triggered(features[identity], rule)
            kind = rule["filter"] if use_filter else "raw"
            selected.append(counts[cohort][kind][identity])
            if use_filter:
                ids.append(identity)
                change = errors(selected[-1]) - errors(counts[cohort]["raw"][identity])
                result["clip_changes"].append({"id": identity, "cohort": cohort, "error_delta": change,
                                               "raw_errors": errors(counts[cohort]["raw"][identity]),
                                               "selected_errors": errors(selected[-1]),
                                               "text_changed": tables[cohort][kind][identity]["text"] !=
                                               tables[cohort]["raw"][identity]["text"]})
        result[cohort] = metrics_sum(selected)
        result["triggered_clip_ids"][cohort] = ids
    result["arithmetic_mean_wer"] = float(objective(result))
    result["exact_objective"] = str(objective(result))
    result["triggered_count"] = sum(map(len, result["triggered_clip_ids"].values()))
    result["combined_errors"] = sum(result[cohort]["errors"] for cohort in COHORTS)
    return result


def search(args):
    scorer, scorer_source = load_scorer()
    tables, counts, features, sources = {}, {}, {}, []
    for cohort, expected_count in COHORTS.items():
        tables[cohort], counts[cohort] = {}, {}
        for kind in ("raw", *FILTERS):
            path = args.root / "results" / f"mega-base-b2-{kind}-en-{cohort}-dev.jsonl"
            rows, provenance = read_table(path)
            if len(rows) != expected_count:
                raise ValueError(f"{path}: expected {expected_count} clips, found {len(rows)}")
            for row in rows.values():
                validate_prediction(row, kind, legacy=True)
                allowed = {"vitw", "krisp"} if cohort == "all" else {"berst"}
                if row.get("split") != "dev" or row.get("corpus") not in allowed:
                    raise ValueError(f"Non-dev or wrong-corpus input: {path}/{row['id']}")
                if not isinstance(row.get("reference"), str) or not Path(row["audio"]).is_absolute():
                    raise ValueError(f"Invalid reference or original audio path: {path}/{row['id']}")
            if kind != "raw":
                anchor = tables[cohort]["raw"]
                if rows.keys() != anchor.keys():
                    raise ValueError(f"Mismatched IDs: {path}")
                for identity, row in rows.items():
                    for field in ("audio", "reference", "split", "corpus", "category"):
                        if row.get(field) != anchor[identity].get(field):
                            raise ValueError(f"Changed {field}: {path}/{identity}")
            tables[cohort][kind] = rows
            counts[cohort][kind] = {identity: scorer([{"reference": row["reference"], "text": row["text"]}])
                                    for identity, row in rows.items()}
            if any(metrics["reference_words"] <= 0 for metrics in counts[cohort][kind].values()):
                raise ValueError(f"Empty normalized reference: {path}")
            sidecar = path.with_suffix(".run.json")
            provenance.update(cohort=cohort, filter=kind,
                              all_rows_seeded=all("sampling_seed" in row for row in rows.values()),
                              run_sidecar=source(sidecar) if sidecar.exists() else None)
            sources.append(provenance)
        for identity, row in tables[cohort]["raw"].items():
            if identity in features:
                raise ValueError(f"Audio ID shared between cohorts: {identity}")
            features[identity] = {"id": identity, "cohort": cohort, **audio_features(Path(row["audio"]))}

    rules = [{"filter": "raw"}] + [
        {"filter": kind, "cutoff_hz": band, "ratio_threshold": ratio, "low_band_rms_dbfs_threshold": rms}
        for kind in FILTERS for band in BANDS for ratio in RATIOS for rms in RMS_DBFS
    ]
    candidates = [policy_result(tables, counts, features, rule) for rule in rules]
    winner_index = min(range(len(candidates)), key=lambda index: (
        objective(candidates[index]), candidates[index]["rule"]["filter"] != "raw",
        candidates[index]["triggered_count"], index))
    baseline, winner = candidates[0], candidates[winner_index]
    for result in (baseline, winner):
        for cohort in COHORTS:
            selected_ids = set(result["triggered_clip_ids"][cohort])
            selected = [tables[cohort][result["rule"]["filter"] if identity in selected_ids else "raw"][identity]
                        for identity in tables[cohort]["raw"]]
            direct = scorer([{"reference": row["reference"], "text": row["text"]} for row in selected])
            if any(direct[key] != result[cohort][key] for key in COUNTS):
                raise ValueError("Additive counts differ from the identical full-cohort scorer")
    configuration = {
        "decoder": DECODER, "features": FEATURE_CONFIG, "selected_rule": winner["rule"],
        "search_grid": {"filters": list(FILTERS), "ratios": list(RATIOS), "rms_dbfs": list(RMS_DBFS)},
        "objective": "arithmetic mean of mixed (all) and BERSt corpus WER, not pooled WER",
        "tie_break": "raw null first; otherwise fewer triggers; otherwise declared grid order",
        "cohort_counts": COHORTS, "scorer_function_sha256": scorer_source["function_sha256"],
    }
    unconditional = {kind: {cohort: metrics_sum(list(counts[cohort][kind].values())) for cohort in COHORTS}
                     for kind in FILTERS}
    for result in unconditional.values():
        result["arithmetic_mean_wer"] = float(objective(result))
    payload = seal({
        "schema": 1, "frozen": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration, "configuration_sha256": sha(canonical(configuration)),
        "script": source(Path(__file__)), "scorer": scorer_source,
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("numpy", "scipy", "soundfile", "jiwer", "whisper-normalizer")},
        "prediction_sources": sources, "original_audio_features": features,
        "heldout_no_look": {"selection_split": "dev", "holdout_used_for_selection": False,
                            "holdout_predictions_read": [], "holdout_audio_read": [],
                            "holdout_evaluation_pending": True},
        "caveat": "Legacy cached dev predictions are not rerun or relabeled as seeded/new-protocol results. "
                  "Missing old generation/audio provenance cannot be reconstructed. Current original WAVs "
                  "and exact cached prediction bytes are hashed. Dev gain is not holdout evidence.",
        "baseline": baseline, "unconditional": unconditional, "winner": winner,
        "strict_dev_improvement": objective(winner) < objective(baseline), "candidates": candidates,
    })
    write_new(args.output, payload)
    changes = winner["clip_changes"]
    print(json.dumps({"output": str(args.output), "configuration_sha256": payload["configuration_sha256"],
                      "rule": winner["rule"], "strict_dev_improvement": payload["strict_dev_improvement"],
                      "baseline_mean_wer": baseline["arithmetic_mean_wer"],
                      "selected_mean_wer": winner["arithmetic_mean_wer"],
                      "baseline_errors": baseline["combined_errors"], "selected_errors": winner["combined_errors"],
                      "triggered_clips": winner["triggered_count"],
                      "improved_clips": sum(row["error_delta"] < 0 for row in changes),
                      "worsened_clips": sum(row["error_delta"] > 0 for row in changes),
                      "cohorts": {key: winner[key] for key in COHORTS}}, indent=2))


def load_frozen(path):
    frozen = read_sealed(path)
    config = frozen["configuration"]
    if (not frozen["frozen"] or sha(canonical(config)) != frozen["configuration_sha256"]
            or frozen["script"]["sha256"] != sha(Path(__file__).read_bytes())
            or config["features"] != FEATURE_CONFIG or config["decoder"] != DECODER):
        raise ValueError("Changed frozen rule, script, features, or decoder")
    return frozen


def gate(args):
    frozen = load_frozen(args.frozen)
    rows, manifest_source = read_table(args.manifest)
    selected = [row for row in rows.values() if row.get("split") == args.split and
                (args.corpus == "all" or row.get("corpus") == args.corpus)]
    if not selected:
        raise ValueError("Empty gated cohort")
    rule = frozen["configuration"]["selected_rule"]
    raw, filtered, decisions = [], [], []
    for row in selected:
        features = audio_features(Path(row["audio"]))
        use_filter = triggered(features, rule)
        entry = {key: row[key] for key in MANIFEST_FIELDS if key in row}
        entry["selected_preprocessing"] = rule["filter"] if use_filter else "raw"
        (filtered if use_filter else raw).append(entry)
        decisions.append({**entry, "features": features})
    raw_path = Path(str(args.output_prefix) + ".raw.jsonl")
    filtered_path = Path(str(args.output_prefix) + ".filtered.jsonl")
    gate_path = Path(str(args.output_prefix) + ".gate.json")
    if any(path.exists() for path in (raw_path, filtered_path, gate_path)):
        raise FileExistsError("Gated outputs already exist; choose a new output prefix")
    write_new(raw_path, raw, jsonl=True)
    write_new(filtered_path, filtered, jsonl=True)
    write_new(gate_path, seal({"schema": 1, "frozen_source": source(args.frozen),
                             "configuration_sha256": frozen["configuration_sha256"], "rule": rule,
                             "manifest_source": manifest_source, "split": args.split,
                             "decisions": decisions, "raw_manifest": source(raw_path),
                             "filtered_manifest": source(filtered_path),
                             "references_or_predictions_used_for_gate": False}))
    print(json.dumps({"gate": str(gate_path), "raw_clips": len(raw), "triggered_clips": len(filtered),
                      "filtered_manifest": str(filtered_path), "filter": rule["filter"],
                      "next": "Run base Qwen beam 2 with the selected --filter on the filtered manifest only, "
                              "using a fresh --name and --limit 0. Reuse full-cohort raw predictions. "
                              "If zero clips trigger, do not start a filtered inference job."}, indent=2))


def evaluate(args):
    frozen = load_frozen(args.frozen)
    gating = read_sealed(args.gate)
    if (gating["configuration_sha256"] != frozen["configuration_sha256"]
            or gating["frozen_source"]["sha256"] != source(args.frozen)["sha256"]):
        raise ValueError("Gate belongs to a different frozen selection")
    scorer, scorer_source = load_scorer()
    if scorer_source["function_sha256"] != frozen["scorer"]["function_sha256"]:
        raise ValueError("Scoring function changed")
    raw, raw_source = read_table(args.raw)
    expected = {row["id"]: row for row in gating["decisions"]}
    triggered_ids = {identity for identity, row in expected.items() if row["selected_preprocessing"] != "raw"}
    filtered, filtered_source = read_table(args.filtered) if args.filtered else ({}, None)
    if raw.keys() != expected.keys() or filtered.keys() != triggered_ids:
        raise ValueError("Require exact full-cohort raw IDs and triggered-only filtered IDs")
    provenance = []
    for path, rows, kind in ((args.raw, raw, "raw"), (args.filtered, filtered, gating["rule"]["filter"])):
        if path is None:
            continue
        sidecar = path.with_suffix(".run.json")
        run = json.loads(sidecar.read_text())
        if run["stage"] != "mega" or any(run["configuration"].get(key) != value for key, value in
                                           {**{key: DECODER[key] for key in DECODER if key != "stage"},
                                            "filter": kind}.items()):
            raise ValueError("Wrong model/decoder configuration in inference sidecar")
        for identity, row in rows.items():
            validate_prediction(row, kind)
            if run["audio_sha256"].get(identity) != expected[identity]["features"]["audio_sha256"]:
                raise ValueError(f"Inference/gate audio hash mismatch: {identity}")
        provenance.append((run, source(sidecar)))
    if len(provenance) == 2:
        for key in ("stage", "model", "model_metadata", "code_sha256", "seeding"):
            if provenance[0][0].get(key) != provenance[1][0].get(key):
                raise ValueError(f"Raw and filtered generation provenance differs: {key}")
    merged = []
    for identity, row in raw.items():
        decision = expected[identity]
        if (row.get("audio") != decision["audio"] or row.get("split") != decision["split"]
                or sha(Path(row["audio"]).read_bytes()) != decision["features"]["audio_sha256"]):
            raise ValueError(f"Changed evaluation cohort/audio: {identity}")
        chosen = filtered[identity] if identity in triggered_ids else row
        if chosen.get("audio") != row["audio"] or chosen.get("split") != row["split"]:
            raise ValueError(f"Changed filtered cohort: {identity}")
        if "reference" in chosen and chosen.get("reference") != row.get("reference"):
            raise ValueError(f"Changed filtered reference: {identity}")
        merged.append({**row, "text": chosen["text"], "selected_preprocessing": decision["selected_preprocessing"]})
    baseline = scorer([dict(row) for row in raw.values()])
    metrics = scorer(merged)
    merged_path = args.output.with_suffix(".jsonl")
    if args.output.exists() or merged_path.exists():
        raise FileExistsError("Evaluation outputs already exist")
    write_new(merged_path, merged, jsonl=True)
    write_new(args.output, seal({"schema": 1, "retuned": False, "configuration_sha256": frozen["configuration_sha256"],
                                "gate_source": source(args.gate), "raw_source": raw_source,
                                "filtered_source": filtered_source, "scorer": scorer_source,
                                "inference_sidecars": [item[1] for item in provenance],
                                "count": len(merged), "triggered_count": len(triggered_ids),
                                "baseline": baseline, "selected": metrics, "merged_predictions": source(merged_path)}))
    print(json.dumps({"output": str(args.output), "count": len(merged), "triggered_clips": len(triggered_ids),
                      "baseline": baseline, "selected": metrics, "retuned": False}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    search_parser = commands.add_parser("search", help="Search and freeze from the exact 36/84 dev caches only")
    search_parser.add_argument("--root", type=Path, required=True)
    search_parser.add_argument("--output", type=Path)
    gate_parser = commands.add_parser("gate", help="Create reference-free subsets without looking at predictions")
    gate_parser.add_argument("--frozen", type=Path, required=True)
    gate_parser.add_argument("--manifest", type=Path, required=True)
    gate_parser.add_argument("--split", choices=("dev", "holdout"), default="holdout")
    gate_parser.add_argument("--corpus", default="all")
    gate_parser.add_argument("--output-prefix", type=Path, required=True)
    evaluate_parser = commands.add_parser("evaluate", help="Score a frozen policy; never search or retune")
    for option in ("frozen", "gate", "raw", "output"):
        evaluate_parser.add_argument(f"--{option}", type=Path, required=True)
    evaluate_parser.add_argument("--filtered", type=Path)
    args = parser.parse_args()
    if args.command == "search":
        args.root = args.root.resolve()
        args.output = args.output or args.root / "results/noisy-preprocessing-dev-frozen.json"
    {"search": search, "gate": gate, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
