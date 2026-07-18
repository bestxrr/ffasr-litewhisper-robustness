from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import load_file, save_file
from torch import nn


@dataclass(frozen=True)
class LoraTarget:
    name: str
    in_features: int
    out_features: int


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)
        self.dropout = nn.Dropout(dropout)
        device = base.weight.device
        # Keep trainable adapter weights in FP32 even when the frozen base model is
        # FP16. GradScaler refuses to unscale FP16 gradients, and FP32 adapters are
        # the standard stable mixed-precision path for PEFT.
        dtype = torch.float32
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        dropped = self.dropout(x).to(torch.float32)
        delta = torch.nn.functional.linear(
            torch.nn.functional.linear(dropped, self.lora_A), self.lora_B
        ).to(out.dtype)
        return out + delta * self.scaling


def iter_linear_targets(model: nn.Module) -> list[LoraTarget]:
    out: list[LoraTarget] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            out.append(LoraTarget(name, module.in_features, module.out_features))
    return out


def _parent_and_attr(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def attach_lora(model: nn.Module, target_names: Iterable[str], rank: int, alpha: float, dropout: float) -> list[str]:
    attached: list[str] = []
    for name in list(target_names):
        parent, attr = _parent_and_attr(model, name)
        module = getattr(parent, attr)
        if not isinstance(module, nn.Linear):
            continue
        setattr(parent, attr, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        attached.append(name)
    return attached


def trainable_parameter_report(model: nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if ".lora_A" in k or ".lora_B" in k or k in trainable
    }


def save_lora(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(lora_state_dict(model), str(path))


def load_lora(model: nn.Module, path: str | Path) -> None:
    tensors = load_file(str(path))
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    bad_missing = [m for m in missing if ".lora_A" in m or ".lora_B" in m]
    if bad_missing or unexpected:
        raise RuntimeError(f"LoRA load mismatch missing={bad_missing[:5]} unexpected={unexpected[:5]}")


def merge_lora_inplace(model: nn.Module) -> list[str]:
    merged: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        parent, attr = _parent_and_attr(model, name)
        base = module.base
        delta = (module.lora_B @ module.lora_A).to(device=base.weight.device, dtype=base.weight.dtype)
        with torch.no_grad():
            base.weight.add_(delta * module.scaling)
        setattr(parent, attr, base)
        merged.append(name)
    return merged
