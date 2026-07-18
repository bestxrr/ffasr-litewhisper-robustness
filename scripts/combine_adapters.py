#!/usr/bin/env python3
"""Combine compatible safetensors adapter checkpoints with scalar weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_weighted_path(value: str) -> tuple[float, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("expected WEIGHT:PATH")
    weight_s, path_s = value.split(":", 1)
    return float(weight_s), Path(path_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, type=parse_weighted_path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    loaded = [(weight, load_file(str(path))) for weight, path in args.input]
    keyset = set(loaded[0][1])
    for _, tensors in loaded[1:]:
        if set(tensors) != keyset:
            missing = sorted(keyset - set(tensors))[:10]
            extra = sorted(set(tensors) - keyset)[:10]
            raise SystemExit(f"incompatible adapter keys; missing={missing}, extra={extra}")

    combined: dict[str, torch.Tensor] = {}
    for key in sorted(keyset):
        ref = loaded[0][1][key]
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for weight, tensors in loaded:
            tensor = tensors[key]
            if tensor.shape != ref.shape:
                raise SystemExit(f"shape mismatch for {key}: {tensor.shape} vs {ref.shape}")
            acc += weight * tensor.float()
        combined[key] = acc.to(ref.dtype)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(combined, str(args.output))
    print(f"wrote {args.output} with {len(combined)} tensors")


if __name__ == "__main__":
    main()
