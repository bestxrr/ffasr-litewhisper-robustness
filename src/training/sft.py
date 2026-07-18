from __future__ import annotations

import csv
import json
import random
import re
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from src.augmentation.effects import butter_filter, convolve_rir, frame_dropout, mix_at_snr, soft_clip
from src.evaluation.proxy_eval import colored_noise, procedural_rir
from src.models.lite_whisper import discover_lora_targets, load_model_and_processor
from src.models.lora import attach_lora, save_lora, trainable_parameter_report
from src.utils.config import config_hash, load_yaml
from src.utils.disk import assert_within_budget, project_usage


def gradient_health(model: torch.nn.Module, limit: int = 20) -> dict:
    bad = []
    max_abs = 0.0
    max_name = ""
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        grad = param.grad.detach()
        finite = torch.isfinite(grad)
        if not bool(finite.all()):
            bad.append({
                "name": name,
                "shape": list(grad.shape),
                "nan": int(torch.isnan(grad).sum().detach().cpu()),
                "inf": int(torch.isinf(grad).sum().detach().cpu()),
            })
            if len(bad) >= limit:
                break
        finite_grad = grad[finite]
        if finite_grad.numel():
            local = float(finite_grad.abs().max().detach().cpu())
            if local > max_abs:
                max_abs = local
                max_name = name
    return {"bad_gradients": bad, "max_abs_finite_grad": max_abs, "max_abs_finite_grad_name": max_name}


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def choose_condition(rng: random.Random, dist: dict[str, float]) -> str:
    r = rng.random()
    acc = 0.0
    for key, prob in dist.items():
        acc += float(prob)
        if r <= acc:
            return key
    return list(dist)[-1]


def augmentation_cfg_for_update(cfg: dict, update: int) -> dict:
    aug = cfg["augmentation"]
    stages = aug.get("curriculum")
    if not stages:
        return aug
    selected = stages[-1]
    for stage in stages:
        until = stage.get("until_update")
        if until is None or update < int(until):
            selected = stage
            break
    merged = dict(aug)
    merged.update({k: v for k, v in selected.items() if k != "until_update"})
    return merged


def read_audio(row: dict) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(row["audio_filepath"], dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio[:, 0]
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    return audio.astype(np.float32), sr


def augment_audio(audio: np.ndarray, sr: int, cfg: dict, step_seed: int) -> tuple[np.ndarray, int, str]:
    rng_py = random.Random(step_seed)
    rng = np.random.default_rng(step_seed)
    condition = choose_condition(rng_py, cfg["condition_distribution"])
    if condition == "dry":
        return audio.astype(np.float32), sr, condition
    snr = rng_py.uniform(*cfg["snr_db"][condition])
    rt60 = rng_py.uniform(0.2, 1.1)
    distance = rng_py.uniform(0.8, 6.0)
    rir = procedural_rir(sr, rt60, distance, rng)
    audio, _ = convolve_rir(audio, rir)
    noise = colored_noise(rng_py.choice(["white", "pink", "brown", "babble_like"]), len(audio), rng)
    audio, _ = mix_at_snr(audio, noise, snr)
    effects_cfg = cfg.get("effects", {})
    spectral_prob = effects_cfg.get("spectral_probability", {"mid": 0.35, "low": 0.35})
    if rng_py.random() < float(spectral_prob.get(condition, 0.0)):
        spectral_types = effects_cfg.get("spectral_types", {})
        choices = spectral_types.get(condition, ["lowpass"])
        kind = rng_py.choice(choices)
        ranges = effects_cfg.get("spectral_ranges", {})
        if kind == "lowpass":
            lo, hi = ranges.get("lowpass", [3200, 7400])
            audio, _ = butter_filter(audio, sr, "lowpass", rng_py.uniform(float(lo), float(hi)))
        elif kind == "highpass":
            lo, hi = ranges.get("highpass", [50, 220])
            audio, _ = butter_filter(audio, sr, "highpass", rng_py.uniform(float(lo), float(hi)))
        elif kind == "bandpass":
            lo_range = ranges.get("bandpass_low", [80, 250])
            hi_range = ranges.get("bandpass_high", [3000, 7600])
            lo = rng_py.uniform(float(lo_range[0]), float(lo_range[1]))
            hi = rng_py.uniform(float(hi_range[0]), float(hi_range[1]))
            audio, _ = butter_filter(audio, sr, "bandpass", (lo, hi))
    clipping_prob = effects_cfg.get("clipping_probability", {"low": 0.15})
    if rng_py.random() < float(clipping_prob.get(condition, 0.0)):
        drive = effects_cfg.get("clipping_drive_db", [1.0, 6.0])
        audio, _ = soft_clip(audio, rng_py.uniform(float(drive[0]), float(drive[1])))
    dropout_prob = effects_cfg.get("dropout_probability", {})
    if rng_py.random() < float(dropout_prob.get(condition, 0.0)):
        max_ms = float(effects_cfg.get("dropout_max_ms", 80.0))
        start_s = rng_py.uniform(0, max(0.01, len(audio) / sr - 0.1))
        audio, _ = frame_dropout(audio, sr, start_s, rng_py.uniform(20.0, max_ms))
    peak = float(np.max(np.abs(audio)) + 1e-9)
    if peak > 0.99:
        audio = audio / peak * 0.98
    return audio.astype(np.float32), sr, condition


def load_and_augment(row: dict, cfg: dict, step_seed: int) -> tuple[np.ndarray, int, str]:
    audio, sr = read_audio(row)
    return augment_audio(audio, sr, cfg["augmentation"], step_seed)


def make_labels(processor, text: str, device: str, label_cfg: dict) -> torch.Tensor:
    tok = processor.tokenizer
    if hasattr(tok, "set_prefix_tokens"):
        tok.set_prefix_tokens(
            language=label_cfg.get("language", "en"),
            task=label_cfg.get("task", "transcribe"),
            predict_timestamps=bool(label_cfg.get("predict_timestamps", False)),
        )
    encoded = tok(text=text, return_tensors="pt", padding=True)
    labels = encoded.input_ids
    if getattr(tok, "pad_token_id", None) is not None:
        labels = labels.masked_fill(encoded.attention_mask.ne(1), -100)
    return labels.to(device=device)


def _valid_frame_count(max_frames: int, audio_num_samples: int, sr: int) -> int:
    # Whisper's encoder produces 50 frames/s after the convolutional frontend.
    return max(1, min(max_frames, int(np.ceil(audio_num_samples / sr * 50.0))))


def pooled_encoder_state(encoder_last_hidden_state: torch.Tensor, audio_num_samples: int, sr: int) -> torch.Tensor:
    valid_frames = _valid_frame_count(encoder_last_hidden_state.shape[1], audio_num_samples, sr)
    return encoder_last_hidden_state[:, :valid_frames, :].mean(dim=1)


def encoder_valid_frames(encoder_last_hidden_state: torch.Tensor, audio_num_samples: int, sr: int) -> torch.Tensor:
    valid_frames = _valid_frame_count(encoder_last_hidden_state.shape[1], audio_num_samples, sr)
    return encoder_last_hidden_state[:, :valid_frames, :]


def frame_distillation_loss(degraded_frames: torch.Tensor, clean_frames: torch.Tensor) -> torch.Tensor:
    """Per-frame cosine distance between the degraded and (detached, teacher) clean encoder
    representations. Pilot Q pooled the frames into a single mean vector before comparing, which
    averages away exactly the per-frame acoustic detail that degradation destroys; matching every
    frame instead pushes the degraded encoder output toward the clean one where the information
    was actually lost. Frames are index-aligned and truncated to the shorter valid-frame count."""
    t = min(degraded_frames.shape[1], clean_frames.shape[1])
    if t < 1:
        return degraded_frames.new_zeros(())
    deg = degraded_frames[:, :t, :].float()
    clean = clean_frames[:, :t, :].detach().float()
    cos = F.cosine_similarity(F.normalize(deg, dim=-1), F.normalize(clean, dim=-1), dim=-1)
    return (1.0 - cos).mean()


def loss_for_audio(
    model,
    processor,
    audio: np.ndarray,
    sr: int,
    labels: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    want: tuple[str, ...] = (),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device=device, dtype=dtype)
    need_encoder = ("pool" in want) or ("frames" in want)
    out = model(
        input_features=input_features,
        labels=labels,
        output_hidden_states=need_encoder,
        return_dict=True,
    )
    extras: dict[str, torch.Tensor] = {}
    if "pool" in want:
        extras["pool"] = pooled_encoder_state(out.encoder_last_hidden_state, len(audio), sr)
    if "frames" in want:
        extras["frames"] = encoder_valid_frames(out.encoder_last_hidden_state, len(audio), sr)
    if "logits" in want:
        extras["logits"] = out.logits
    return out.loss, extras


def eos_suppression_penalty(logits: torch.Tensor, labels: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Mean predicted probability of the EOS token at every non-final, non-padding label
    position. Penalizes the model for assigning early-stop mass to positions where the
    reference continues, which is the failure mode behind rising low-condition deletions."""
    non_final_mask = (labels != -100) & (labels != eos_token_id)
    if not bool(non_final_mask.any()):
        return logits.new_zeros(())
    probs = torch.softmax(logits.float(), dim=-1)
    eos_probs = probs[..., eos_token_id]
    return eos_probs[non_final_mask].mean()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    assert_within_budget(".", float(cfg["resources"].get("max_project_gb", 48)))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "hypothesis": cfg["hypothesis"],
        "starting_checkpoint": cfg["starting_checkpoint"],
        "changed_variable": cfg["changed_variable"],
        "control_run": cfg["control_run"],
        "maximum_updates": cfg["max_updates"],
        "maximum_wall_time": f"{cfg['max_wall_time_minutes']} minutes",
        "expected_vram": cfg["resources"]["expected_vram"],
        "expected_disk_usage": cfg["resources"]["expected_disk_usage"],
        "promotion_criterion": cfg["promotion_criterion"],
    }, indent=2))
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    model, processor = load_model_and_processor(cfg)
    device = cfg["model"].get("device", "cuda")
    dtype = torch.float16 if cfg["model"].get("dtype") == "float16" else torch.float32
    model = model.to(device=device, dtype=dtype)
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    for p in model.parameters():
        p.requires_grad = False
    targets = discover_lora_targets(model, cfg["lora"]["target_name_regex"])
    attached = attach_lora(
        model,
        targets,
        rank=int(cfg["lora"]["rank"]),
        alpha=float(cfg["lora"]["alpha"]),
        dropout=float(cfg["lora"]["dropout"]),
    )
    if not attached:
        raise RuntimeError("No LoRA targets attached.")
    extra_trainable = []
    trainable_regex = cfg.get("trainable_parameter_regex", [])
    if trainable_regex:
        patterns = [re.compile(p) for p in trainable_regex]
        for name, param in model.named_parameters():
            if any(p.search(name) for p in patterns):
                param.data = param.data.to(torch.float32)
                param.requires_grad = True
                extra_trainable.append(name)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    amp_cfg = cfg.get("amp", {})
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device == "cuda"),
        init_scale=float(amp_cfg.get("init_scale", 128.0)),
        growth_interval=int(amp_cfg.get("growth_interval", 2000)),
    )
    rows = read_jsonl(cfg["train_manifest"])
    order = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(order)
    accum = int(cfg["grad_accumulation"])
    clean_ce_weight = float(cfg.get("loss", {}).get("clean_ce_weight", 0.0))
    normalize_clean_anchor = bool(cfg.get("loss", {}).get("normalize_clean_anchor", False))
    condition_ce_weights = {str(k): float(v) for k, v in cfg.get("loss", {}).get("condition_ce_weights", {}).items()}
    eos_suppress_weight = float(cfg.get("loss", {}).get("eos_suppress_weight", 0.0))
    eos_suppress_conditions = cfg.get("loss", {}).get("eos_suppress_conditions")
    eos_suppress_conditions = set(eos_suppress_conditions) if eos_suppress_conditions else None
    eos_token_id = processor.tokenizer.eos_token_id
    feature_distill_weight = float(cfg.get("loss", {}).get("feature_distill_weight", 0.0))
    feature_distill_conditions = cfg.get("loss", {}).get("feature_distill_conditions")
    feature_distill_conditions = set(feature_distill_conditions) if feature_distill_conditions else None
    label_cfg = cfg.get("labels", {})
    checkpoint_steps = {int(s) for s in cfg.get("checkpoint_steps", [])}
    max_updates = int(cfg["max_updates"])
    max_wall = float(cfg["max_wall_time_minutes"]) * 60.0
    log_path = out_dir / "train_log.csv"
    step_idx = 0
    update = 0
    running = 0.0
    running_deg = 0.0
    running_clean = 0.0
    running_consistency = 0.0
    running_eos_penalty = 0.0
    running_feature_distill = 0.0
    consistency_weight = float(cfg.get("loss", {}).get("consistency_weight", 0.0))
    with open(log_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "update",
                "loss",
                "degraded_ce",
                "clean_ce",
                "consistency",
                "eos_penalty",
                "feature_distill",
                "grad_norm",
                "condition",
                "condition_weight",
                "amp_scale",
                "elapsed_s",
            ],
        )
        writer.writeheader()
        while update < max_updates:
            if time.time() - start > max_wall:
                print("wall clock stop")
                break
            opt.zero_grad(set_to_none=True)
            last_condition = ""
            for _ in range(accum):
                row = rows[order[step_idx % len(order)]]
                step_idx += 1
                clean_audio, sr = read_audio(row)
                aug_cfg = augmentation_cfg_for_update(cfg, update)
                audio, sr, condition = augment_audio(clean_audio, sr, aug_cfg, seed + step_idx)
                last_condition = condition
                labels = make_labels(processor, row["text"], device, label_cfg)
                condition_weight = float(condition_ce_weights.get(condition, 1.0))
                with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16):
                    apply_eos_suppress = eos_suppress_weight > 0 and (
                        eos_suppress_conditions is None or condition in eos_suppress_conditions
                    )
                    apply_feature_distill = feature_distill_weight > 0 and (
                        feature_distill_conditions is None or condition in feature_distill_conditions
                    )
                    deg_want: list[str] = []
                    if consistency_weight > 0:
                        deg_want.append("pool")
                    if apply_feature_distill:
                        deg_want.append("frames")
                    if apply_eos_suppress:
                        deg_want.append("logits")
                    degraded_loss, degraded_extras = loss_for_audio(
                        model, processor, audio, sr, labels, device, dtype, want=tuple(deg_want),
                    )
                    clean_loss = torch.zeros_like(degraded_loss)
                    consistency_loss = torch.zeros_like(degraded_loss)
                    eos_penalty = torch.zeros_like(degraded_loss)
                    feature_distill_loss = torch.zeros_like(degraded_loss)
                    if apply_eos_suppress:
                        eos_penalty = eos_suppression_penalty(degraded_extras["logits"], labels, eos_token_id)
                    loss = condition_weight * degraded_loss
                    if clean_ce_weight > 0 or consistency_weight > 0 or apply_feature_distill:
                        clean_want: list[str] = []
                        if consistency_weight > 0:
                            clean_want.append("pool")
                        if apply_feature_distill:
                            clean_want.append("frames")
                        clean_loss, clean_extras = loss_for_audio(
                            model, processor, clean_audio, sr, labels, device, dtype, want=tuple(clean_want),
                        )
                        if consistency_weight > 0:
                            consistency_loss = 1.0 - F.cosine_similarity(
                                F.normalize(degraded_extras["pool"].float(), dim=-1),
                                F.normalize(clean_extras["pool"].detach().float(), dim=-1),
                                dim=-1,
                            ).mean()
                        if apply_feature_distill:
                            feature_distill_loss = frame_distillation_loss(
                                degraded_extras["frames"], clean_extras["frames"]
                            )
                        if clean_ce_weight > 0:
                            loss = condition_weight * degraded_loss + clean_ce_weight * clean_loss
                            if normalize_clean_anchor:
                                loss = loss / (condition_weight + clean_ce_weight)
                    if consistency_weight > 0:
                        loss = loss + consistency_weight * consistency_loss
                    if apply_eos_suppress:
                        loss = loss + eos_suppress_weight * eos_penalty
                    if apply_feature_distill:
                        loss = loss + feature_distill_weight * feature_distill_loss
                    loss = loss / accum
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at update {update}: {loss}")
                scaler.scale(loss).backward()
                running += float(loss.detach().cpu()) * accum
                running_deg += float(degraded_loss.detach().cpu())
                running_clean += float(clean_loss.detach().cpu()) if clean_ce_weight > 0 else 0.0
                running_consistency += float(consistency_loss.detach().cpu()) if consistency_weight > 0 else 0.0
                running_eos_penalty += float(eos_penalty.detach().cpu()) if apply_eos_suppress else 0.0
                running_feature_distill += float(feature_distill_loss.detach().cpu()) if apply_feature_distill else 0.0
            scaler.unscale_(opt)
            health = gradient_health(model)
            if health["bad_gradients"]:
                (out_dir / f"bad_gradients_update_{update + 1}.json").write_text(
                    json.dumps(health, indent=2),
                    encoding="utf-8",
                )
                raise RuntimeError(f"Non-finite gradients before clipping at update {update + 1}: {health['bad_gradients'][:3]}")
            grad_norm = torch.nn.utils.clip_grad_norm_(params, float(cfg["gradient_clip_norm"]))
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite grad norm at update {update + 1}: {grad_norm}; health={health}")
            scaler.step(opt)
            scaler.update()
            update += 1
            mean_loss = running / accum
            mean_deg = running_deg / accum
            mean_clean = running_clean / accum if clean_ce_weight > 0 else 0.0
            mean_consistency = running_consistency / accum if consistency_weight > 0 else 0.0
            mean_eos_penalty = running_eos_penalty / accum if eos_suppress_weight > 0 else 0.0
            mean_feature_distill = running_feature_distill / accum if feature_distill_weight > 0 else 0.0
            running = 0.0
            running_deg = 0.0
            running_clean = 0.0
            running_consistency = 0.0
            running_eos_penalty = 0.0
            running_feature_distill = 0.0
            writer.writerow({
                "update": update,
                "loss": mean_loss,
                "degraded_ce": mean_deg,
                "clean_ce": mean_clean,
                "consistency": mean_consistency,
                "eos_penalty": mean_eos_penalty,
                "feature_distill": mean_feature_distill,
                "grad_norm": float(grad_norm.detach().cpu()),
                "condition": last_condition,
                "condition_weight": float(condition_ce_weights.get(last_condition, 1.0)),
                "amp_scale": float(scaler.get_scale()),
                "elapsed_s": time.time() - start,
            })
            f.flush()
            if update in checkpoint_steps:
                save_lora(model, out_dir / f"adapter_update_{update}.safetensors")
            if update % 10 == 0:
                print(
                    f"update {update}/{max_updates} loss={mean_loss:.4f} "
                    f"deg_ce={mean_deg:.4f} clean_ce={mean_clean:.4f} "
                    f"cons={mean_consistency:.4f} grad_norm={float(grad_norm):.3f}"
                )
    adapter_path = out_dir / "adapter.safetensors"
    save_lora(model, adapter_path)
    report = {
        "run_id": cfg["run_id"],
        "config_hash": config_hash(cfg),
        "updates": update,
        "micro_batches": step_idx,
        "wall_time_s": time.time() - start,
        "adapter_path": str(adapter_path),
        "checkpoint_steps": sorted(checkpoint_steps),
        "lora_targets": attached,
        "extra_trainable_parameters": extra_trainable,
        "parameter_report": trainable_parameter_report(model),
        "disk_gb": project_usage("."),
    }
    if torch.cuda.is_available():
        report["peak_vram_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        report["peak_vram_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024**3)
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)[:8000])


if __name__ == "__main__":
    main()
