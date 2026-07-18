from __future__ import annotations

import json
import time
from pathlib import Path

import librosa
import soundfile as sf
import torch

from src.models.lite_whisper import discover_lora_targets, load_model_and_processor
from src.models.lora import attach_lora, load_lora
from src.training.sft import make_labels
from src.utils.config import config_hash, load_yaml
from src.utils.disk import assert_within_budget, project_usage


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_audio(path: str | Path) -> tuple:
    audio, sr = sf.read(path, dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio[:, 0]
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    return audio, sr


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
    load_lora(model, adapter_cfg["path"])
    return {"adapter_loaded": True, "adapter_path": adapter_cfg["path"], "lora_targets": attached}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    assert_within_budget(".", float(cfg["limits"].get("max_project_gb", 48)))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "hypothesis": cfg.get("hypothesis", "fixed clean CE evaluation"),
        "starting_checkpoint": cfg["model"]["id"],
        "changed_variable": cfg.get("changed_variable", "adapter"),
        "control_run": cfg.get("control_run", "base model"),
        "maximum_updates": 0,
        "maximum_wall_time": "loss evaluation only",
        "expected_vram": "<8 GB",
        "expected_disk_usage": "<0.05 GB report",
        "promotion_criterion": cfg.get("promotion_criterion", "loss should decrease versus control"),
    }, indent=2))
    start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model, processor = load_model_and_processor(cfg)
    device = cfg["model"].get("device", "cuda")
    dtype = torch.float16 if cfg["model"].get("dtype") == "float16" else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    adapter_report = maybe_load_adapter(model, cfg)
    rows = read_jsonl(cfg["manifest"])
    max_samples = cfg["limits"].get("max_samples")
    if max_samples:
        rows = rows[: int(max_samples)]
    label_cfg = cfg.get("labels", {})
    losses = []
    durations = []
    with torch.no_grad():
        for row in rows:
            audio, sr = read_audio(row["audio_filepath"])
            labels = make_labels(processor, row.get("text") or row.get("reference"), device, label_cfg)
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
            input_features = inputs.input_features.to(device=device, dtype=dtype)
            with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16):
                out = model(input_features=input_features, labels=labels)
            losses.append(float(out.loss.detach().cpu()))
            durations.append(float(row.get("duration") or len(audio) / sr))
    summary = {
        "run_id": cfg["run_id"],
        "config_hash": config_hash(cfg),
        "adapter": adapter_report,
        "num_samples": len(rows),
        "mean_loss": sum(losses) / max(len(losses), 1),
        "min_loss": min(losses) if losses else None,
        "max_loss": max(losses) if losses else None,
        "audio_hours": sum(durations) / 3600.0,
        "wall_time_s": time.time() - start,
        "disk_gb": project_usage("."),
    }
    if torch.cuda.is_available():
        summary["peak_vram_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        summary["peak_vram_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024**3)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)[:8000])


if __name__ == "__main__":
    main()
