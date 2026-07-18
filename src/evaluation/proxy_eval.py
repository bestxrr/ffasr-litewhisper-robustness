from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from src.augmentation.effects import butter_filter, convolve_rir, frame_dropout, mix_at_snr, soft_clip
from src.models.lite_whisper import load_model_and_processor
from src.models.lite_whisper import discover_lora_targets
from src.models.lora import attach_lora, load_lora, trainable_parameter_report
from src.utils.config import config_hash, load_yaml
from src.utils.disk import assert_within_budget, project_usage
from src.utils.metrics import edit_stats


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalizer():
    try:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

        return EnglishTextNormalizer({})
    except Exception:
        import re

        def norm(s: str) -> str:
            s = re.sub(r"[^\w\s]", " ", (s or "").lower())
            return re.sub(r"\s+", " ", s).strip()

        return norm


def colored_noise(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.normal(size=n).astype(np.float32)
    if kind == "white":
        return white
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    if kind == "pink":
        spec = spec / np.sqrt(freqs)
    elif kind == "brown":
        spec = spec / freqs
    elif kind == "babble_like":
        mod = 0.5 + 0.5 * np.sin(np.linspace(0, 12 * np.pi, n))
        return (white * mod).astype(np.float32)
    out = np.fft.irfft(spec, n=n).astype(np.float32)
    return out / (np.max(np.abs(out)) + 1e-6)


def procedural_rir(sr: int, rt60: float, distance: float, rng: np.random.Generator) -> np.ndarray:
    n = max(int(sr * min(rt60, 1.5)), 32)
    t = np.arange(n, dtype=np.float32) / sr
    decay = np.exp(-6.91 * t / max(rt60, 0.05))
    rir = rng.normal(size=n).astype(np.float32) * decay
    direct = min(n - 1, max(0, int(distance / 343.0 * sr)))
    rir[direct] += 1.0
    return rir / (np.max(np.abs(rir)) + 1e-6)


def apply_proxy_degradation(row: dict) -> tuple[np.ndarray, int, list[dict]]:
    audio, sr = sf.read(row["audio_filepath"], dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio[:, 0]
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    logs: list[dict] = []
    condition = row["condition"]
    if condition == "dry":
        return audio.astype(np.float32), sr, [{"name": "dry", "params": {}}]
    rng = np.random.default_rng(int(row["sample_seed"]))
    if row.get("rt60") is not None:
        rir = procedural_rir(sr, float(row["rt60"]), float(row["source_distance"]), rng)
        audio, log = convolve_rir(audio, rir)
        logs.append(log.to_dict())
    noise = colored_noise(row.get("noise_type") or "white", len(audio), rng)
    audio, log = mix_at_snr(audio, noise, float(row["snr_db"]))
    logs.append(log.to_dict())
    effects = set(row.get("effects", []))
    if "lowpass" in effects:
        audio, log = butter_filter(audio, sr, "lowpass", float(rng.uniform(2800, 7200)))
        logs.append(log.to_dict())
    elif "highpass" in effects:
        audio, log = butter_filter(audio, sr, "highpass", float(rng.uniform(50, 220)))
        logs.append(log.to_dict())
    elif "bandpass" in effects:
        lo = float(rng.uniform(80, 250))
        hi = float(rng.uniform(3000, 7600))
        audio, log = butter_filter(audio, sr, "bandpass", (lo, hi))
        logs.append(log.to_dict())
    if "soft_clip" in effects:
        audio, log = soft_clip(audio, float(rng.uniform(1.0, 7.0)))
        logs.append(log.to_dict())
    if condition == "low" and rng.random() < 0.08:
        start_s = float(rng.uniform(0, max(0.01, len(audio) / sr - 0.1)))
        audio, log = frame_dropout(audio, sr, start_s, float(rng.uniform(20, 80)))
        logs.append(log.to_dict())
    peak = float(np.max(np.abs(audio)) + 1e-9)
    if peak > 0.99:
        audio = audio / peak * 0.98
    return audio.astype(np.float32), sr, logs


def transcribe(model, processor, audio: np.ndarray, sr: int, cfg: dict) -> str:
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    device = cfg["model"].get("device", "cuda")
    dtype = torch.float16 if cfg["model"].get("dtype") == "float16" else torch.float32
    input_features = inputs.input_features.to(device=device, dtype=dtype)
    gen_kwargs = {
        "max_new_tokens": int(cfg["decoding"]["max_new_tokens"]),
        "num_beams": int(cfg["decoding"]["num_beams"]),
    }
    optional_decode_args = {
        "repetition_penalty": float,
        "no_repeat_ngram_size": int,
        "length_penalty": float,
        "early_stopping": bool,
        "min_new_tokens": int,
    }
    for name, caster in optional_decode_args.items():
        if name in cfg["decoding"] and cfg["decoding"][name] is not None:
            gen_kwargs[name] = caster(cfg["decoding"][name])
    try:
        predicted_ids = model.generate(
            input_features,
            language=cfg["decoding"].get("language", "en"),
            task=cfg["decoding"].get("task", "transcribe"),
            **gen_kwargs,
        )
    except (TypeError, ValueError):
        predicted_ids = model.generate(input_features, **gen_kwargs)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def maybe_load_adapter(model, cfg: dict) -> dict:
    adapter_cfg = cfg.get("adapter")
    if not adapter_cfg:
        return {"adapter_loaded": False}
    for p in model.parameters():
        p.requires_grad = False
    lora_cfg = adapter_cfg["lora"]
    targets = discover_lora_targets(model, lora_cfg["target_name_regex"])
    attached = attach_lora(
        model,
        targets,
        rank=int(lora_cfg["rank"]),
        alpha=float(lora_cfg["alpha"]),
        dropout=float(lora_cfg.get("dropout", 0.0)),
    )
    if not attached:
        raise RuntimeError("Adapter config matched no LoRA targets.")
    load_lora(model, adapter_cfg["path"])
    return {
        "adapter_loaded": True,
        "adapter_path": adapter_cfg["path"],
        "lora_targets": attached,
        "parameter_report": trainable_parameter_report(model),
    }


def summarize(rows: list[dict], baseline_dry_wer: float | None, length_ratio_threshold: float) -> dict:
    by_cond = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    cond_summary = {}
    for cond, items in by_cond.items():
        ref_words = sum(int(r["reference_length"]) for r in items)
        s = sum(int(r["substitutions"]) for r in items)
        d = sum(int(r["deletions"]) for r in items)
        i = sum(int(r["insertions"]) for r in items)
        wer = 0.0 if ref_words == 0 else 100.0 * (s + d + i) / ref_words
        cond_summary[cond] = {
            "n": len(items),
            "wer": wer,
            "substitutions": s,
            "deletions": d,
            "insertions": i,
            "reference_words": ref_words,
            "empty_outputs": sum(1 for r in items if not r["hypothesis_norm"]),
            "wer_gt_100": sum(1 for r in items if float(r["wer"]) > 100.0),
        }
    ordered = [c for c in ["dry", "high", "mid", "low"] if c in cond_summary]
    avg = sum(cond_summary[c]["wer"] for c in ordered) / max(len(ordered), 1)
    catastrophic = 0
    for row in rows:
        ref_len = max(1, int(row["reference_length"]))
        ratio = float(row["hypothesis_length"]) / ref_len
        if (not row["hypothesis_norm"]) or float(row["wer"]) > 100.0 or ratio > length_ratio_threshold:
            catastrophic += 1
    catastrophic_rate = 100.0 * catastrophic / max(len(rows), 1)
    dry_wer = cond_summary.get("dry", {}).get("wer", math.nan)
    if baseline_dry_wer is None:
        baseline_dry_wer = dry_wer
    promotion_score = avg + 2.0 * max(0.0, dry_wer - baseline_dry_wer - 0.5) + 0.5 * catastrophic_rate
    return {
        "average_wer": avg,
        "condition_summary": cond_summary,
        "catastrophic_output_rate": catastrophic_rate,
        "promotion_score": promotion_score,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    assert_within_budget(".", float(cfg["limits"].get("max_project_gb", 48)))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_cfg = cfg.get("adapter")
    print(json.dumps({
        "hypothesis": cfg.get("hypothesis", "baseline proxy evaluation before training"),
        "starting_checkpoint": cfg["model"]["id"],
        "changed_variable": cfg.get("changed_variable", "adapter checkpoint" if adapter_cfg else "none"),
        "control_run": cfg.get("control_run", "official public baseline row"),
        "maximum_updates": 0,
        "maximum_wall_time": "evaluation only",
        "expected_vram": "<8 GB",
        "expected_disk_usage": "<0.2 GB reports",
        "promotion_criterion": cfg.get("promotion_criterion", "not applicable; establishes control"),
    }, indent=2))
    start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model, processor = load_model_and_processor(cfg)
    device = cfg["model"].get("device", "cuda")
    dtype = torch.float16 if cfg["model"].get("dtype") == "float16" else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    adapter_report = maybe_load_adapter(model, cfg)
    model.eval()
    norm = normalizer()
    manifest = read_jsonl(cfg["manifest"])
    max_samples = cfg["limits"].get("max_samples")
    if max_samples:
        manifest = manifest[: int(max_samples)]
    rows = []
    with torch.no_grad(), open(out_dir / "errors.csv", "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "proxy_id", "condition", "reference", "hypothesis", "reference_norm", "hypothesis_norm",
            "wer", "substitutions", "deletions", "insertions", "reference_length", "hypothesis_length",
            "snr_db", "rt60", "room_volume", "source_distance", "noise_type", "rir_source",
            "duration", "effects", "effect_logs",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(manifest, 1):
            audio, sr, effect_logs = apply_proxy_degradation(row)
            hyp = transcribe(model, processor, audio, sr, cfg)
            ref_norm = norm(row["reference"])
            hyp_norm = norm(hyp)
            stats = edit_stats(ref_norm, hyp_norm)
            out = {
                "proxy_id": row["proxy_id"],
                "condition": row["condition"],
                "reference": row["reference"],
                "hypothesis": hyp,
                "reference_norm": ref_norm,
                "hypothesis_norm": hyp_norm,
                "wer": 100.0 * stats.wer,
                "substitutions": stats.substitutions,
                "deletions": stats.deletions,
                "insertions": stats.insertions,
                "reference_length": stats.ref_words,
                "hypothesis_length": len(hyp_norm.split()),
                "snr_db": row.get("snr_db"),
                "rt60": row.get("rt60"),
                "room_volume": row.get("room_volume"),
                "source_distance": row.get("source_distance"),
                "noise_type": row.get("noise_type"),
                "rir_source": row.get("rir_source"),
                "duration": row.get("duration"),
                "effects": json.dumps(row.get("effects", [])),
                "effect_logs": json.dumps(effect_logs),
            }
            writer.writerow(out)
            rows.append(out)
            if idx % 25 == 0:
                print(f"evaluated {idx}/{len(manifest)}")
    baseline_dry = cfg["promotion"].get("baseline_dry_wer")
    summary = summarize(
        rows,
        baseline_dry_wer=None if baseline_dry is None else float(baseline_dry),
        length_ratio_threshold=float(cfg["promotion"]["catastrophic_length_ratio_threshold"]),
    )
    summary.update({
        "run_id": cfg["run_id"],
        "config_hash": config_hash(cfg),
        "adapter": adapter_report,
        "wall_time_s": time.time() - start,
        "num_samples": len(rows),
        "disk_gb": project_usage("."),
    })
    if torch.cuda.is_available():
        summary["peak_vram_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        summary["peak_vram_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024**3)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(out_dir / "condition_summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "n", "wer", "substitutions", "deletions", "insertions", "reference_words", "empty_outputs", "wer_gt_100"])
        writer.writeheader()
        for cond, vals in summary["condition_summary"].items():
            writer.writerow({"condition": cond, **vals})
    print(json.dumps(summary, indent=2)[:8000])


if __name__ == "__main__":
    main()
