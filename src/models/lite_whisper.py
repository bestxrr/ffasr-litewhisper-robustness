from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.models.lora import iter_linear_targets


def load_model_and_processor(config: dict[str, Any]):
    from transformers import AutoModel, AutoProcessor

    model_cfg = config["model"]
    dtype = torch.float16 if model_cfg.get("dtype") == "float16" else torch.float32
    model = AutoModel.from_pretrained(
        model_cfg["id"],
        revision=model_cfg.get("revision"),
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    processor = AutoProcessor.from_pretrained(model_cfg.get("processor_id", "openai/whisper-large-v3"))
    return model, processor


def discover_lora_targets(model: nn.Module, patterns: list[str]) -> list[str]:
    regexes = [re.compile(p) for p in patterns]
    names = []
    for target in iter_linear_targets(model):
        if any(r.search(target.name) for r in regexes):
            names.append(target.name)
    return names


def write_module_report(model: nn.Module, path: str | Path, limit: int = 400) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (name, module) in enumerate(model.named_modules()):
        if i >= limit:
            rows.append(f"... truncated after {limit} modules")
            break
        params = sum(p.numel() for p in module.parameters(recurse=False))
        rows.append(f"{name or '<root>'}\t{module.__class__.__name__}\tparams={params}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def make_synthetic_inputs(processor, transcript: str, seconds: float, device: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    sr = 16000
    t = torch.linspace(0, seconds, int(sr * seconds))
    audio = (0.03 * torch.sin(2 * torch.pi * 440 * t)).numpy()
    features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device=device, dtype=dtype)
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    labels = tok(text=transcript, return_tensors="pt").input_ids.to(device)
    return {"input_features": features, "labels": labels}
