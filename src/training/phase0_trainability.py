from __future__ import annotations

import json
import traceback
from pathlib import Path

import torch

from src.models.lite_whisper import (
    discover_lora_targets,
    load_model_and_processor,
    make_synthetic_inputs,
    write_module_report,
)
from src.models.lora import (
    attach_lora,
    load_lora,
    lora_state_dict,
    merge_lora_inplace,
    save_lora,
    trainable_parameter_report,
)
from src.utils.config import config_hash, load_yaml
from src.utils.disk import assert_within_budget, project_usage
from src.utils.env import configure_project_environment


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    root = Path.cwd()
    configure_project_environment(root)
    cfg = load_yaml(args.config)
    assert_within_budget(root, float(cfg.get("resources", {}).get("max_project_gb", 48)))
    run_dir = root / "artifacts" / "runs" / cfg["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "run_id": cfg["run_id"],
        "config_hash": config_hash(cfg),
        "disk_gb_start": project_usage(root),
        "checks": {},
    }
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model, processor = load_model_and_processor(cfg)
        device = cfg["model"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if cfg["model"].get("dtype") == "float16" else torch.float32
        model = model.to(device=device, dtype=dtype)
        model.train()
        write_module_report(model, run_dir / "module_tree.txt")
        target_names = discover_lora_targets(model, cfg["phase0"]["lora"]["target_name_regex"])
        (run_dir / "lora_targets.json").write_text(json.dumps(target_names, indent=2), encoding="utf-8")
        report["checks"]["lora_targets_found"] = len(target_names)
        for p in model.parameters():
            p.requires_grad = False
        attached = attach_lora(
            model,
            target_names,
            rank=int(cfg["phase0"]["lora"]["rank"]),
            alpha=float(cfg["phase0"]["lora"]["alpha"]),
            dropout=float(cfg["phase0"]["lora"]["dropout"]),
        )
        report["checks"]["lora_attached"] = attached
        report["parameter_report"] = trainable_parameter_report(model)
        inputs = make_synthetic_inputs(
            processor,
            cfg["phase0"]["synthetic_transcript"],
            float(cfg["phase0"]["smoke_seconds"]),
            device=device,
            dtype=dtype,
        )
        out = model(**inputs)
        loss = getattr(out, "loss", None)
        if loss is None:
            raise RuntimeError("Model forward did not return .loss with labels; HF training path may be inference-only.")
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss: {loss}")
        loss.backward()
        grad_nonzero = {
            name: bool(param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum().item() > 0)
            for name, param in model.named_parameters()
            if ".lora_" in name
        }
        report["checks"]["finite_loss"] = float(loss.detach().float().cpu())
        report["checks"]["nonzero_lora_gradients"] = grad_nonzero
        if not any(grad_nonzero.values()):
            raise RuntimeError("No nonzero LoRA gradients observed.")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.endswith(".lora_B"):
                    param.add_(torch.full_like(param, 1e-5))
        adapter_path = run_dir / "adapter_smoke.safetensors"
        save_lora(model, adapter_path)
        with torch.no_grad():
            logits_before_reload = model(**inputs).logits.detach().float().cpu()
            generated_a = model.generate(inputs["input_features"], max_new_tokens=8)
            generated_b = model.generate(inputs["input_features"], max_new_tokens=8)
        report["checks"]["deterministic_generation"] = torch.equal(generated_a, generated_b)
        del model
        torch.cuda.empty_cache()

        reloaded, _processor2 = load_model_and_processor(cfg)
        reloaded = reloaded.to(device=device, dtype=dtype)
        for p in reloaded.parameters():
            p.requires_grad = False
        attached2 = attach_lora(
            reloaded,
            target_names,
            rank=int(cfg["phase0"]["lora"]["rank"]),
            alpha=float(cfg["phase0"]["lora"]["alpha"]),
            dropout=float(cfg["phase0"]["lora"]["dropout"]),
        )
        if attached2 != attached:
            raise RuntimeError("LoRA target set changed across reload.")
        load_lora(reloaded, adapter_path)
        reloaded.eval()
        with torch.no_grad():
            logits_after_reload = reloaded(**inputs).logits.detach().float().cpu()
        if not torch.isfinite(logits_before_reload).all() or not torch.isfinite(logits_after_reload).all():
            raise RuntimeError("Non-finite logits during save/reload equivalence check.")
        reload_max_abs = (logits_before_reload - logits_after_reload).abs().max().item()
        report["checks"]["save_reload_logits_max_abs"] = reload_max_abs
        reload_tol = float(cfg["phase0"].get("tolerances", {}).get("reload_logits_max_abs", 1e-3))
        if not torch.isfinite(torch.tensor(reload_max_abs)) or reload_max_abs > reload_tol:
            raise RuntimeError(f"Reload logits mismatch: max_abs={reload_max_abs}")

        with torch.no_grad():
            logits_before_merge = reloaded(**inputs).logits.detach().float().cpu()
        merged = merge_lora_inplace(reloaded)
        report["checks"]["merged_lora_modules"] = merged
        with torch.no_grad():
            logits_after_merge = reloaded(**inputs).logits.detach().float().cpu()
        if not torch.isfinite(logits_before_merge).all() or not torch.isfinite(logits_after_merge).all():
            raise RuntimeError("Non-finite logits during merge/export equivalence check.")
        merge_max_abs = (logits_before_merge - logits_after_merge).abs().max().item()
        report["checks"]["merge_logits_max_abs"] = merge_max_abs
        merge_tol = float(cfg["phase0"].get("tolerances", {}).get("merge_logits_max_abs", 5e-2))
        if not torch.isfinite(torch.tensor(merge_max_abs)) or merge_max_abs > merge_tol:
            raise RuntimeError(f"Merged export logits mismatch: max_abs={merge_max_abs}")
        export_dir = run_dir / "merged_export"
        reloaded.save_pretrained(export_dir, safe_serialization=True, max_shard_size="2GB")
        processor.save_pretrained(export_dir)
        report["checks"]["export_dir"] = str(export_dir)
        report["status"] = "pass"
    except Exception as exc:
        report["status"] = "fail"
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        debug_log = root / "docs" / "DEBUG_LOG.md"
        with open(debug_log, "a", encoding="utf-8") as f:
            f.write(f"\n## {cfg['run_id']} failure\n\n")
            f.write(f"Symptom: `{exc!r}`\n\n")
            f.write("Root cause: pending investigation.\n\n")
            f.write("Fix: not applied yet.\n\n")
            f.write("Regression test: Phase-0 trainability script.\n")
    finally:
        if torch.cuda.is_available():
            report["peak_vram_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        report["disk_gb_end"] = project_usage(root)
        (run_dir / "phase0_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2)[:8000])
        if report.get("status") != "pass":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
