from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import safe_open


LORA_RE = re.compile(r"model\.encoder\.layers\.(\d+)\.(.+)\.lora_([AB])$")


def read_lora(path: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if ".lora_" in key:
                tensors[key] = handle.get_tensor(key).float()
    return tensors


def module_prefix(key: str) -> str:
    return key.rsplit(".lora_", 1)[0]


def checkpoint_rows(path: Path, alpha: float, rank: int) -> list[dict]:
    tensors = read_lora(path)
    modules = sorted({module_prefix(key) for key in tensors})
    rows = []
    for module in modules:
        a = tensors.get(f"{module}.lora_A")
        b = tensors.get(f"{module}.lora_B")
        if a is None or b is None:
            continue
        match = LORA_RE.match(f"{module}.lora_A")
        layer = int(match.group(1)) if match else -1
        submodule = match.group(2) if match else module
        update = (b @ a) * (alpha / rank)
        rows.append(
            {
                "checkpoint": path.name,
                "module": module,
                "layer": layer,
                "submodule": submodule,
                "lora_a_fro": float(torch.linalg.vector_norm(a)),
                "lora_b_fro": float(torch.linalg.vector_norm(b)),
                "update_fro": float(torch.linalg.vector_norm(update)),
                "update_mean_abs": float(update.abs().mean()),
                "update_max_abs": float(update.abs().max()),
            }
        )
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["checkpoint"], int(row["layer"]))].append(row)
    out = []
    for (checkpoint, layer), items in sorted(buckets.items()):
        out.append(
            {
                "checkpoint": checkpoint,
                "layer": layer,
                "modules": len(items),
                "sum_update_fro": sum(float(item["update_fro"]) for item in items),
                "mean_update_fro": sum(float(item["update_fro"]) for item in items) / len(items),
                "max_update_fro": max(float(item["update_fro"]) for item in items),
                "sum_lora_a_fro": sum(float(item["lora_a_fro"]) for item in items),
                "sum_lora_b_fro": sum(float(item["lora_b_fro"]) for item in items),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--rank", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    rows: list[dict] = []
    for checkpoint in args.checkpoints:
        rows.extend(checkpoint_rows(Path(checkpoint), alpha=args.alpha, rank=args.rank))
    layer_rows = aggregate(rows)
    write_csv(out_dir / "adapter_module_norms.csv", rows)
    write_csv(out_dir / "adapter_layer_norms.csv", layer_rows)

    top_layers = sorted(layer_rows, key=lambda row: float(row["sum_update_fro"]), reverse=True)[:10]
    (out_dir / "adapter_norms_summary.json").write_text(
        json.dumps(
            {
                "num_checkpoints": len(args.checkpoints),
                "num_module_rows": len(rows),
                "alpha": args.alpha,
                "rank": args.rank,
                "top_layers_by_sum_update_fro": top_layers,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
