"""Recover every uncovered target sample, then merge raw ASR with review flags.

Run with the benchmark runtime's Python and --root pointing at the benchmark
workspace (data/, code/, models/, results/). --prepare-only never starts ASR;
--merge-only requires already complete results and never starts ASR either.
"""

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from contextlib import contextmanager
from pathlib import Path


SELECTED_NAME = "Recording_1828_selected_20261002"
UNCOVERED_NAME = "Recording_1828_uncovered_20261002"
FULL_NAME = "Recording_1828_fullsource_20261002"
ROUTED_NAME = "Recording_1828_router_subset_20261002"
ROUTER_NAME = "Recording_1828_router_20261002"
MANIFEST_NAME = "target-uncovered-manifest.jsonl"
STATE_NAME = "target-uncovered-manifest.coverage.json"
CONFIG_DEFAULTS = {
    "beam": 2, "route": "base", "filter": "raw", "language": "en",
    "length_penalty": 1.0, "threshold": 0.5, "lora_scale": 1.0,
}


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def jsonl_bytes(rows):
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                   for row in rows).encode("utf-8")


def read_jsonl(path):
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f"Invalid JSONL at {path}:{number}; saved data was not discarded") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{number}")
            rows.append(row)
    return rows


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as output:
            temporary = Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish(path, content):
    """Never silently replace an existing, different transcript or clip."""
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing artifact differs: {path}; no data was overwritten")
    else:
        atomic_write(path, content)


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def selected_configuration(winner):
    if not isinstance(winner, dict) or winner.get("stage") != "mega":
        raise ValueError("Expected the selected Mega-ASR configuration")
    config = {"stage": "mega", **CONFIG_DEFAULTS}
    config.update({key: winner[key] for key in CONFIG_DEFAULTS if key in winner})
    config["filter"] = winner.get("kind", config["filter"])
    if "kind" in winner and "filter" in winner and winner["kind"] != winner["filter"]:
        raise ValueError("Conflicting selected filter/kind")
    if (type(config["beam"]) is not int or config["beam"] != 2 or
            config["route"] != "base" or config["filter"] != "raw"):
        raise ValueError("Expected Mega-ASR Qwen base, beam 2, raw; refusing a different selection")
    if config["language"] not in ("en", "auto"):
        raise ValueError("Invalid selected language")
    for key in ("length_penalty", "threshold", "lora_scale"):
        finite_number(config[key], key)
    return config


def validate_intervals(intervals, frames):
    integer(frames, "source frames")
    ordered = sorted(intervals)
    previous_end = 0
    for start, end in ordered:
        integer(start, "start_sample")
        integer(end, "end_sample")
        if start >= end or end > frames:
            raise ValueError(f"Invalid source interval [{start}, {end}) for {frames} frames")
        if start < previous_end:
            raise ValueError(f"Overlapping source intervals at sample {start}")
        previous_end = end
    return ordered


def complement_intervals(intervals, frames):
    missing = []
    cursor = 0
    for start, end in validate_intervals(intervals, frames):
        if start > cursor:
            missing.append((cursor, start))
        cursor = end
    if cursor < frames:
        missing.append((cursor, frames))
    return missing


def validate_full_coverage(intervals, frames):
    missing = complement_intervals(intervals, frames)
    if missing:
        raise ValueError(f"Incomplete source coverage: missing sample intervals {missing[:5]}")


def row_interval(row, frames, rate):
    integer(rate, "sample rate", 1)
    samples = []
    has_samples = "start_sample" in row or "end_sample" in row
    has_seconds = "start" in row or "end" in row
    if not has_samples and not has_seconds:
        raise ValueError(f"Missing source interval: {row.get('id')}")
    for key in ("start", "end"):
        sample_key = f"{key}_sample"
        if has_samples:
            samples.append(integer(row.get(sample_key), sample_key))
        if has_seconds:
            seconds = finite_number(row.get(key), key)
            scaled = seconds * rate
            if not math.isfinite(scaled) or abs(scaled - round(scaled)) > 1e-5:
                raise ValueError(f"Timestamp is not sample-aligned: {row.get('id')} {key}")
            sample = round(scaled)
            if has_samples and samples[-1] != sample:
                raise ValueError(f"Timestamp/sample boundary mismatch: {row.get('id')} {key}")
            if not has_samples:
                samples.append(sample)
    start, end = samples
    validate_intervals([(start, end)], frames)
    if "duration_s" in row:
        duration = finite_number(row["duration_s"], "duration_s")
        if abs(duration * rate - (end - start)) > 1e-5:
            raise ValueError(f"Duration/sample boundary mismatch: {row.get('id')}")
    if "sample_rate" in row and row["sample_rate"] != rate:
        raise ValueError(f"Sample rate mismatch: {row.get('id')}")
    return start, end


def normalize_rows(rows, frames, rate):
    normalized = []
    ids = set()
    for row in rows:
        name = row.get("id")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"Invalid audio ID: {name!r}")
        if name in ids:
            raise ValueError(f"Duplicate audio ID: {name}")
        ids.add(name)
        start, end = row_interval(row, frames, rate)
        normalized.append({**row, "start_sample": start, "end_sample": end})
    validate_intervals([(row["start_sample"], row["end_sample"]) for row in normalized], frames)
    return sorted(normalized, key=lambda row: row["start_sample"])


def build_uncovered_rows(retained, existing_gaps, frames, rate, maximum_samples=None):
    """Reuse gap IDs/boundaries, not their transcripts, and include one-sample gaps."""
    maximum_samples = maximum_samples if maximum_samples is not None else 45 * rate
    integer(maximum_samples, "maximum chunk samples", 1)
    retained = normalize_rows(retained, frames, rate)
    gaps = normalize_rows(existing_gaps, frames, rate)
    ids = {row["id"] for row in retained}
    for row in gaps:
        if row["id"] in ids:
            raise ValueError(f"Duplicate audio ID across manifests: {row['id']}")
        ids.add(row["id"])
        if row["end_sample"] - row["start_sample"] > maximum_samples:
            raise ValueError(f"Existing gap exceeds the chunk limit: {row['id']}")
    occupied = [(row["start_sample"], row["end_sample"]) for row in retained + gaps]
    missing = complement_intervals(occupied, frames)
    intervals = [(row["id"], row["start_sample"], row["end_sample"], True) for row in gaps]
    index = 0
    for start, end in missing:
        for offset in range(start, end, maximum_samples):
            while f"target-uncovered-{index:05d}" in ids:
                index += 1
            name = f"target-uncovered-{index:05d}"
            ids.add(name)
            index += 1
            intervals.append((name, offset, min(offset + maximum_samples, end), False))
    rows = []
    for name, start, end, reused in sorted(intervals, key=lambda item: item[1]):
        row = {"id": name, "start_sample": start, "end_sample": end,
               "sample_rate": rate, "start": start / rate, "end": end / rate,
               "duration_s": (end - start) / rate, "split": "dev", "corpus": "target",
               "vad_rejected": True, "review_required": True,
               "reference_is_verified": False, "speech_is_verified": False}
        if reused:
            row["reused_gap_id"] = name
        rows.append(row)
    validate_full_coverage([(row["start_sample"], row["end_sample"]) for row in retained + rows], frames)
    return rows


def source_info(path):
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getframerate() != 16000 or source.getcomptype() != "NONE":
            raise ValueError("Expected uncompressed mono 16 kHz WAV")
        info = {"path": str(path), "frames": source.getnframes(), "sample_rate": source.getframerate(),
                "channels": source.getnchannels(), "sample_width": source.getsampwidth()}
    integer(info["frames"], "source frames", 1)
    info["sha256"] = sha256_file(path)
    return info


def source_pcm(source, row, info):
    start, end = row_interval(row, info["frames"], info["sample_rate"])
    source.setpos(start)
    pcm = source.readframes(end - start)
    if len(pcm) != (end - start) * info["sample_width"]:
        raise ValueError(f"Truncated source audio at {row['id']}")
    return pcm


def wav_bytes(pcm, info):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(info["channels"])
        output.setsampwidth(info["sample_width"])
        output.setframerate(info["sample_rate"])
        output.writeframes(pcm)
    return buffer.getvalue()


def validate_retained_clips(rows, info):
    with wave.open(info["path"], "rb") as source:
        for row in rows:
            if row.get("split") != "dev" or row.get("corpus") != "target":
                raise ValueError(f"Retained row would be filtered out by the ASR helper: {row['id']}")
            expected = source_pcm(source, row, info)
            with wave.open(row["audio"], "rb") as clip:
                if (clip.getnchannels(), clip.getframerate(), clip.getsampwidth(), clip.getnframes()) != (
                        1, info["sample_rate"], info["sample_width"], row["end_sample"] - row["start_sample"]):
                    raise ValueError(f"Retained clip format/frames differ from source: {row['id']}")
                if clip.readframes(clip.getnframes()) != expected:
                    raise ValueError(f"Retained clip samples differ from source: {row['id']}")


def model_metadata(root):
    directory = root / "models/Mega-ASR"
    return {str(path.relative_to(directory)): path.read_text(encoding="utf-8").splitlines()[:2]
            for path in sorted(directory.glob("**/.cache/huggingface/download/*.metadata"))}


def run_fingerprint(root, rows, config):
    return {"schema": 2, "stage": config["stage"], "model": str(root / "models/Mega-ASR"),
            "model_metadata": model_metadata(root),
            "code_sha256": sha256_file(root / "code/noisy_asr_search.py"),
            "configuration": {key: config[key] for key in CONFIG_DEFAULTS},
            "seeding": "20261002 XOR SHA256(audio_id), per clip",
            "audio_sha256": {row["id"]: sha256_file(Path(row["audio"])) for row in rows}}


def prepare_coverage(root):
    config = selected_configuration(read_json(root / "results/extra-selection.json")["winner"])
    info = source_info(root / "data/Recording_1828.wav")
    target_path = root / "data/target-manifest.jsonl"
    gap_path = root / "data/target-gap-manifest.jsonl"
    retained = normalize_rows(read_jsonl(target_path), info["frames"], info["sample_rate"])
    gaps = read_jsonl(gap_path) if gap_path.exists() else []
    rows = build_uncovered_rows(retained, gaps, info["frames"], info["sample_rate"])
    directory = root / "data/target-uncovered-clips"
    for row in rows:
        row.update(audio=str(directory / f"{row['id']}.wav"), source_audio=info["path"],
                   source_sha256=info["sha256"])
    content = jsonl_bytes(rows)
    fingerprint = {"schema": 1, "source": info, "configuration": config,
                   "target_manifest_sha256": sha256_file(target_path),
                   "existing_gap_manifest_sha256": sha256_file(gap_path) if gap_path.exists() else None,
                   "selection_sha256": sha256_file(root / "results/extra-selection.json"),
                   "helper_sha256": sha256_file(root / "code/noisy_asr_search.py"),
                   "renderer_sha256": sha256_file(root / "code/render_target_transcript.py"),
                   "model_metadata": model_metadata(root), "maximum_chunk_samples": 45 * info["sample_rate"]}
    manifest = root / "data" / MANIFEST_NAME
    state_path = root / "data" / STATE_NAME
    plan = {"fingerprint": fingerprint, "manifest_sha256": hashlib.sha256(content).hexdigest()}
    if state_path.exists():
        state = read_json(state_path)
        if any(state.get(key) != value for key, value in plan.items()):
            raise ValueError("Coverage source, manifests, code or configuration changed; refusing to reuse artifacts")
    else:
        owned_results = [path for name in (UNCOVERED_NAME, FULL_NAME)
                         for path in (root / "results").glob(f"{name}*")]
        if manifest.exists() or (directory.exists() and any(directory.iterdir())) or owned_results:
            raise ValueError("Existing coverage artifacts lack the exact source/config fingerprint")
        state = {**plan, "prepared": False, "audio_sha256": {}}
    if manifest.exists() and manifest.read_bytes() != content:
        raise ValueError("Existing uncovered manifest differs; no manifest was overwritten")
    validate_retained_clips(retained, info)
    if not state_path.exists():
        atomic_write(state_path, json_bytes(state))
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with wave.open(info["path"], "rb") as source:
        for row in rows:
            audio = wav_bytes(source_pcm(source, row, info), info)
            publish(Path(row["audio"]), audio)
            hashes[row["id"]] = hashlib.sha256(audio).hexdigest()
    if state.get("audio_sha256") and state["audio_sha256"] != hashes:
        raise ValueError("Saved uncovered clip hashes differ from the exact source")
    if sha256_file(Path(info["path"])) != info["sha256"]:
        raise ValueError("Source changed during coverage preparation")
    publish(manifest, content)
    atomic_write(state_path, json_bytes({**plan, "prepared": True, "audio_sha256": hashes}))
    return {"source": info, "configuration": config, "retained": retained,
            "uncovered": rows, "manifest": manifest, "fingerprint": fingerprint}


def validate_results(rows, manifest, frames, rate, config, complete=True):
    expected = {row["id"]: row for row in normalize_rows(manifest, frames, rate)}
    actual = normalize_rows(rows, frames, rate)
    for row in actual:
        name = row["id"]
        if name not in expected:
            raise ValueError(f"Unexpected result ID: {name}")
        wanted = expected[name]
        if row_interval(row, frames, rate) != row_interval(wanted, frames, rate):
            raise ValueError(f"Result source interval changed: {name}")
        for key, value in wanted.items():
            if key not in ("start", "end", "start_sample", "end_sample") and row.get(key) != value:
                raise ValueError(f"Result manifest metadata changed: {name} {key}")
        for key, value in config.items():
            if row.get(key) != value or (key == "beam" and type(row.get(key)) is not int):
                raise ValueError(f"Result configuration changed: {name} {key}")
        if not isinstance(row.get("text"), str):
            raise ValueError(f"Missing raw text: {name}")
    if complete and len(actual) != len(expected):
        missing = sorted(set(expected) - {row["id"] for row in actual})
        raise ValueError(f"Incomplete result set: {len(actual)}/{len(expected)}; missing {missing[:5]}")
    return actual


def read_run(root, name, manifest, context, complete):
    prefix = root / "results" / name
    rows = read_jsonl(prefix.with_suffix(".jsonl")) if prefix.with_suffix(".jsonl").exists() else []
    info = context["source"]
    rows = validate_results(rows, manifest, info["frames"], info["sample_rate"],
                            context["configuration"], complete=complete)
    sidecar = prefix.with_suffix(".run.json")
    if sidecar.exists():
        if read_json(sidecar) != run_fingerprint(root, manifest, context["configuration"]):
            raise ValueError(f"Saved ASR run fingerprint changed: {name}")
    elif rows or prefix.with_suffix(".summary.json").exists():
        raise ValueError(f"Saved ASR run lacks a reproducibility fingerprint: {name}")
    elif complete and manifest:
        raise ValueError(f"Missing complete ASR run: {name}")
    return rows


def validate_summary(root, name, count, config):
    summary = read_json(root / "results" / f"{name}.summary.json")
    if summary.get("name") != name or type(summary.get("count")) is not int or summary["count"] != count:
        raise ValueError(f"Incomplete or inconsistent ASR summary: {name}")
    if summary.get("split") != "dev":
        raise ValueError(f"ASR summary split changed: {name}")
    for key in CONFIG_DEFAULTS:
        if summary.get(key) != config[key]:
            raise ValueError(f"ASR summary configuration changed: {name} {key}")
    return summary


def validate_parent(root, context):
    marker = read_json(root / "results/target-complete.json")
    if selected_configuration(marker.get("configuration")) != context["configuration"]:
        raise ValueError("Parent completion marker uses a different configuration")
    rows = read_run(root, SELECTED_NAME, context["retained"], context, complete=True)
    validate_summary(root, SELECTED_NAME, len(rows), context["configuration"])
    return rows


def asr_command(root, context, name=UNCOVERED_NAME):
    config = context["configuration"]
    command = [sys.executable, str(root / "code/noisy_asr_search.py"), config["stage"],
               "--root", str(root), "--manifest", str(context["manifest"]),
               "--corpus", "all", "--limit", "0", "--split", "dev", "--name", name]
    for key in CONFIG_DEFAULTS:
        command.extend(["--" + key.replace("_", "-"), str(config[key])])
    return command


def complete_uncovered(root, context, merge_only=False):
    # Recheck the gate here as well as in main; this function must be safe alone.
    validate_parent(root, context)
    return complete_run(root, context, UNCOVERED_NAME, context["uncovered"], merge_only)


def complete_run(root, context, name, manifest, merge_only):
    rows = read_run(root, name, manifest, context, complete=False)
    prefix = root / "results" / name
    if len(rows) != len(manifest):
        if merge_only:
            raise ValueError(f"{name} results are incomplete; --merge-only never starts ASR")
        with prefix.with_suffix(".log").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"starting": name, "configuration": context["configuration"]}) + "\n")
            log.flush()
            subprocess.run(asr_command(root, context, name), stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = read_run(root, name, manifest, context, complete=True)
        validate_summary(root, name, len(rows), context["configuration"])
    else:
        if not manifest:
            publish(prefix.with_suffix(".jsonl"), b"")
            publish(prefix.with_suffix(".run.json"), json_bytes(run_fingerprint(root, manifest, context["configuration"])))
        if not prefix.with_suffix(".summary.json").exists():
            summary = {"name": name, "split": "dev", "count": len(rows),
                       **context["configuration"], "wer": None, "verified_reference": False,
                       "completion_basis": "all manifest IDs and exact intervals validated on CPU"}
            atomic_write(prefix.with_suffix(".summary.json"), json_bytes(summary))
        validate_summary(root, name, len(rows), context["configuration"])
        with prefix.with_suffix(".log").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"validated_complete": len(rows), "gpu_started": False}) + "\n")
    return rows


def merge_results(retained_manifest, uncovered_manifest, retained_results, uncovered_results, info, config):
    frames, rate = info["frames"], info["sample_rate"]
    retained = validate_results(retained_results, retained_manifest, frames, rate, config)
    uncovered = validate_results(uncovered_results, uncovered_manifest, frames, rate, config)
    normalized = normalize_rows(retained + uncovered, frames, rate)
    validate_full_coverage([(row["start_sample"], row["end_sample"]) for row in normalized], frames)
    uncovered_ids = {row["id"] for row in uncovered}
    merged = []
    for row in normalized:
        rejected = row["id"] in uncovered_ids
        metadata = {"source_audio": info["path"], "source_sha256": info["sha256"], "sample_rate": rate,
                    "vad_rejected": rejected, "review_required": rejected,
                    "coverage_origin": "uncovered" if rejected else "retained",
                    "reference_is_verified": False, "speech_is_verified": False}
        for key, value in metadata.items():
            if key in row and row[key] != value:
                raise ValueError(f"Conflicting merge metadata: {row['id']} {key}")
        merged.append({**row, **metadata})
    return merged


def audit_generation(rows):
    warnings = []
    for row in rows:
        events = row.get("generation")
        issues = []
        if not isinstance(events, list) or not events:
            issues.append({"code": "missing_generation_audit"})
        else:
            for index, event in enumerate(events):
                if not isinstance(event, dict):
                    issues.append({"code": "invalid_generation_audit", "event": index})
                    continue
                tokens = event.get("new_tokens")
                if type(tokens) is not int or tokens <= 0:
                    issues.append({"code": "invalid_new_tokens", "event": index})
                if type(event.get("ended_with_eos")) is not bool:
                    issues.append({"code": "missing_eos_audit", "event": index})
                elif not event["ended_with_eos"]:
                    issues.append({"code": "not_ended_with_eos", "event": index})
                if type(event.get("token_cap_hit")) is not bool:
                    issues.append({"code": "missing_token_cap_audit", "event": index})
                if event.get("token_cap_hit") is True or (type(tokens) is int and tokens >= 512):
                    issues.append({"code": "token_cap_hit", "event": index})
        if issues:
            warnings.append({"id": row["id"], "start": row["start"], "end": row["end"],
                             "start_sample": row["start_sample"], "end_sample": row["end_sample"],
                             "coverage_origin": row["coverage_origin"], "issues": issues})
    return warnings


def sample_stamp(sample, rate):
    seconds, remainder = divmod(sample, rate)
    return (f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}."
            f"{remainder * 10_000_000 // rate:07d}")


def review_text(rows, warnings, rate):
    warning_ids = {row["id"] for row in warnings}
    lines = ["Full-source ASR review transcript",
             "No verified target reference or WER. Audio coverage does not verify speech.",
             "[LOW-VAD / REVIEW] marks VAD-rejected audio, not confirmed speech.",
             "Raw text is preserved; empty ASR rows remain visible. Boundaries are [start_sample, end_sample).", ""]
    for row in rows:
        flags = " [LOW-VAD / REVIEW]" if row["vad_rejected"] else ""
        if row["id"] in warning_ids:
            flags += " [GENERATION / REVIEW]"
        lines.append(f"[{sample_stamp(row['start_sample'], rate)} - {sample_stamp(row['end_sample'], rate)}]"
                     f" [{row['id']}; samples {row['start_sample']}:{row['end_sample']})]{flags}\n{row['text']}\n")
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_and_publish(root, prefix):
    # Render into our own temporary directory so conflicting existing outputs
    # are detected without allowing the renderer to overwrite them.
    name = prefix.name
    with tempfile.TemporaryDirectory(dir=prefix.parent, prefix=f".{name}-render-") as directory:
        temporary_prefix = Path(directory) / name
        with prefix.with_suffix(".render.log").open("a", encoding="utf-8") as log:
            subprocess.run([sys.executable, str(root / "code/render_target_transcript.py"),
                            "--input", str(prefix.with_suffix(".jsonl")), "--prefix", str(temporary_prefix)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        for suffix in (".json", ".txt", "_timestamped.txt"):
            publish(prefix.parent / (name + suffix),
                    (temporary_prefix.parent / (name + suffix)).read_bytes())


def publish_transcript(root, prefix, rows, rate):
    publish(prefix.with_suffix(".jsonl"), jsonl_bytes(rows))
    warnings = audit_generation(rows)
    publish(prefix.with_suffix(".warnings.json"),
            json_bytes({"note": "Review warnings only; no rows or text removed/revised.",
                        "warning_rows": len(warnings), "warnings": warnings}))
    publish(prefix.with_name(prefix.name + "_review_timestamped.txt"), review_text(rows, warnings, rate))
    render_and_publish(root, prefix)
    return warnings


def finalize(root, context, retained_results, uncovered_results):
    info, config = context["source"], context["configuration"]
    merged = merge_results(context["retained"], context["uncovered"], retained_results,
                           uncovered_results, info, config)
    prefix = root / "results" / FULL_NAME
    warnings = publish_transcript(root, prefix, merged, info["sample_rate"])
    warning_path = prefix.with_suffix(".warnings.json")
    retained_samples = sum(row["end_sample"] - row["start_sample"] for row in context["retained"])
    uncovered_samples = info["frames"] - retained_samples
    summary = {"complete": True, "source": info, "configuration": config,
               "coverage_basis": "exact WAV frames; integer half-open source sample intervals",
               "source_samples": info["frames"], "covered_samples": info["frames"],
               "missing_samples": 0, "overlap_samples": 0, "segments": len(merged),
               "verified_reference": False, "verified_target_wer": None,
               "note": "No verified target reference or WER. Uncovered audio is VAD-rejected, not verified speech.",
               "retained": {"rows": len(context["retained"]), "samples": retained_samples,
                            "seconds": retained_samples / info["sample_rate"],
                            "proportion": retained_samples / info["frames"]},
               "uncovered": {"rows": len(context["uncovered"]), "samples": uncovered_samples,
                             "seconds": uncovered_samples / info["sample_rate"],
                             "proportion": uncovered_samples / info["frames"]},
               "generation_warning_rows": len(warnings), "warnings_file": str(warning_path),
               "input_results": {name: sha256_file(root / "results" / f"{name}.jsonl")
                                 for name in (SELECTED_NAME, UNCOVERED_NAME)},
               "merged_results_sha256": sha256_file(prefix.with_suffix(".jsonl")),
               "preparation_fingerprint": context["fingerprint"]}
    publish(prefix.with_suffix(".coverage.json"), json_bytes(summary))
    return summary


def clip_seed(name):
    return 20261002 ^ int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def router_manifest_rows(base_rows, info, threshold=0.5):
    rows = normalize_rows(base_rows, info["frames"], info["sample_rate"])
    validate_full_coverage([(row["start_sample"], row["end_sample"]) for row in rows], info["frames"])
    selected = []
    keys = ("id", "audio", "split", "corpus", "start", "end", "duration_s", "start_sample",
            "end_sample", "sample_rate", "source_audio", "source_sha256", "vad_rejected",
            "review_required", "coverage_origin", "reference_is_verified", "speech_is_verified", "sampling_seed")
    for row in rows:
        probability = finite_number(row.get("degraded_probability"), "degraded_probability")
        if not 0 <= probability <= 1:
            raise ValueError(f"Invalid degraded probability: {row['id']}")
        if type(row.get("sampling_seed")) is not int or row["sampling_seed"] != clip_seed(row["id"]):
            raise ValueError(f"Primary sampling seed differs from the helper's ID seed: {row['id']}")
        if probability >= threshold:
            selected.append({**{key: row[key] for key in keys},
                             "base_degraded_probability": probability,
                             "base_audio_sha256": sha256_file(Path(row["audio"]))})
    return selected


def merge_router_results(base_rows, routed_rows, manifest, info, config):
    routed = validate_results(routed_rows, manifest, info["frames"], info["sample_rate"], config)
    replacements = {row["id"]: row for row in routed}
    base = normalize_rows(base_rows, info["frames"], info["sample_rate"])
    if not set(replacements) <= {row["id"] for row in base}:
        raise ValueError("Routed IDs do not belong to the primary full-source result set")
    merged = []
    for row in base:
        alternate = replacements.get(row["id"])
        if alternate is None:
            merged.append(row)
            continue
        if (alternate["sampling_seed"] != row["sampling_seed"] or alternate["audio"] != row["audio"] or
                row_interval(alternate, info["frames"], info["sample_rate"]) !=
                row_interval(row, info["frames"], info["sample_rate"]) or
                alternate["base_audio_sha256"] != sha256_file(Path(row["audio"]))):
            raise ValueError(f"Routed interval/seed/audio identity changed: {row['id']}")
        probability = finite_number(alternate.get("degraded_probability"), "routed degraded_probability")
        if (abs(probability - row["degraded_probability"]) > 1e-9 or alternate.get("lora_active") is not True):
            raise ValueError(f"Routed probability/adapter activation changed: {row['id']}")
        merged.append({**row, **alternate})
    validate_full_coverage([(row["start_sample"], row["end_sample"]) for row in merged], info["frames"])
    return merged


def router_alternative(root, context, merge_only=False):
    """Only callable after the primary renderer and exact-coverage export finish."""
    primary = root / "results" / FULL_NAME
    summary = read_json(primary.with_suffix(".coverage.json"))
    base_hash = sha256_file(primary.with_suffix(".jsonl"))
    if (summary.get("complete") is not True or summary.get("merged_results_sha256") != base_hash or
            summary.get("preparation_fingerprint") != context["fingerprint"]):
        raise ValueError("Primary full-source transcript is not completely validated/rendered")
    if context["configuration"]["threshold"] != 0.5:
        raise ValueError("Router alternative requires the primary reference-free threshold 0.5")
    validate_parent(root, context)
    base = read_jsonl(primary.with_suffix(".jsonl"))
    manifest_rows = router_manifest_rows(base, context["source"])
    config = {**context["configuration"], "route": "auto"}
    manifest = root / "data/target-router-manifest.jsonl"
    content = jsonl_bytes(manifest_rows)
    state_path = root / "data/target-router-manifest.coverage.json"
    state = {"preparation_fingerprint": context["fingerprint"], "base_results_sha256": base_hash,
             "configuration": config, "manifest_sha256": hashlib.sha256(content).hexdigest()}
    if state_path.exists():
        if read_json(state_path) != state:
            raise ValueError("Router alternative source/config/base result fingerprint changed")
    else:
        if manifest.exists() or any(path for name in (ROUTED_NAME, ROUTER_NAME)
                                    for path in (root / "results").glob(f"{name}*")):
            raise ValueError("Existing router artifacts lack their source/config/base result fingerprint")
        atomic_write(state_path, json_bytes(state))
    publish(manifest, content)
    routed_context = {**context, "configuration": config, "manifest": manifest}
    routed = complete_run(root, routed_context, ROUTED_NAME, manifest_rows, merge_only)
    merged = merge_router_results(base, routed, manifest_rows, context["source"], config)
    prefix = root / "results" / ROUTER_NAME
    warnings = publish_transcript(root, prefix, merged, context["source"]["sample_rate"])
    alternate_summary = {**summary, "configuration": {"primary": context["configuration"], "routed": config},
                         "primary_results": str(primary.with_suffix(".jsonl")),
                         "primary_results_sha256": base_hash, "router_threshold": 0.5,
                         "routed_rows": len(routed), "unchanged_rows": len(base) - len(routed),
                         "router_subset_results_sha256": sha256_file(root / "results" / f"{ROUTED_NAME}.jsonl"),
                         "merged_results_sha256": sha256_file(prefix.with_suffix(".jsonl")),
                         "generation_warning_rows": len(warnings), "warnings_file": str(prefix.with_suffix(".warnings.json")),
                         "note": "Separate reference-free router alternative; not verified speech or target WER. Primary unchanged."}
    publish(prefix.with_suffix(".coverage.json"), json_bytes(alternate_summary))
    return alternate_summary


@contextmanager
def coverage_lock(root):
    with (root / "results/.target-coverage.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another full-source coverage workflow holds the lock") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="Prepare and validate exact coverage on CPU, then exit")
    mode.add_argument("--merge-only", action="store_true", help="Require both complete runs; never launch ASR")
    parser.add_argument("--router-alternative", action="store_true",
                        help="After the primary full-source export, serially route only probability >= 0.5 rows")
    parser.add_argument("--poll-seconds", type=float, default=5, help="Parent completion poll interval (default: 5)")
    args = parser.parse_args(argv)
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("--poll-seconds must be finite and positive")
    root = args.root.resolve()
    with coverage_lock(root):
        context = prepare_coverage(root)
        print(json.dumps({"prepared": str(context["manifest"]), "uncovered_rows": len(context["uncovered"]),
                          "source_frames": context["source"]["frames"], "gpu_started": False}), flush=True)
        if args.prepare_only:
            return
        marker = root / "results/target-complete.json"
        if args.merge_only and not marker.exists():
            raise ValueError("Missing target-complete.json; --merge-only never starts ASR")
        if not marker.exists():
            print(json.dumps({"waiting_for": str(marker), "gpu_started": False}), flush=True)
        while not marker.exists():
            time.sleep(args.poll_seconds)
        # Inputs may have changed while the parent was running. Never rebind
        # existing artifacts to a different source/configuration after waiting.
        context = prepare_coverage(root)
        retained_results = validate_parent(root, context)
        uncovered_results = complete_uncovered(root, context, merge_only=args.merge_only)
        context = prepare_coverage(root)
        retained_results = validate_parent(root, context)
        summary = finalize(root, context, retained_results, uncovered_results)
        if args.router_alternative:
            alternative = router_alternative(root, context, merge_only=args.merge_only)
            print(json.dumps({"router_alternative": str(root / "results" / f"{ROUTER_NAME}.coverage.json"),
                              "routed_rows": alternative["routed_rows"]}), flush=True)
        print(json.dumps({"complete": True, "segments": summary["segments"],
                          "coverage": str(root / "results" / f"{FULL_NAME}.coverage.json"),
                          "generation_warning_rows": summary["generation_warning_rows"]}), flush=True)


if __name__ == "__main__":
    main()
