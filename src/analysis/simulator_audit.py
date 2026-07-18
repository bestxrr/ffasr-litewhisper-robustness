from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from src.utils.config import load_yaml


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def q(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    vals = sorted(vals)
    return {
        "n": len(vals),
        "min": vals[0],
        "p25": vals[int(0.25 * (len(vals) - 1))],
        "median": median(vals),
        "p75": vals[int(0.75 * (len(vals) - 1))],
        "max": vals[-1],
    }


def summarize_metadata(rows: list[dict]) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cond[str(row.get("condition"))].append(row)
    effect_counts = Counter()
    noise_counts = Counter()
    rir_counts = Counter()
    compound = 0
    for row in rows:
        effects = row.get("effects") or []
        if isinstance(effects, str):
            try:
                effects = json.loads(effects)
            except json.JSONDecodeError:
                effects = [effects]
        effect_counts.update(effects)
        if len(effects) > 1:
            compound += 1
        if row.get("noise_type"):
            noise_counts[str(row.get("noise_type"))] += 1
        if row.get("rir_source"):
            rir_counts[str(row.get("rir_source"))] += 1
    return {
        "count": len(rows),
        "conditions": {k: len(v) for k, v in sorted(by_cond.items())},
        "effects": dict(effect_counts),
        "compound_effect_count": compound,
        "compound_effect_rate": compound / max(len(rows), 1),
        "noise_types": dict(noise_counts),
        "rir_sources": dict(rir_counts),
        "snr_by_condition": {
            cond: q([float(r["snr_db"]) for r in items if r.get("snr_db") is not None])
            for cond, items in sorted(by_cond.items())
        },
        "rt60_by_condition": {
            cond: q([float(r["rt60"]) for r in items if r.get("rt60") is not None])
            for cond, items in sorted(by_cond.items())
        },
        "distance_by_condition": {
            cond: q([float(r["source_distance"]) for r in items if r.get("source_distance") is not None])
            for cond, items in sorted(by_cond.items())
        },
    }


def sample_training_augmentation(train_cfg: dict, samples: int) -> list[dict]:
    seed = int(train_cfg["seed"])
    rng_order = random.Random(seed)
    rows = []
    max_updates = int(train_cfg.get("max_updates", samples))
    accum = int(train_cfg.get("grad_accumulation", 1))
    for n in range(samples):
        update = min(max_updates - 1, n // max(accum, 1))
        step_seed = seed + n + 1
        rng = random.Random(step_seed)
        aug = augmentation_cfg_for_update(train_cfg, update)
        condition = choose_condition(rng, aug["condition_distribution"])
        out = {
            "condition": condition,
            "effects": ["dry"] if condition == "dry" else ["additive_noise", "rir"],
            "snr_db": None,
            "rt60": None,
            "source_distance": None,
            "noise_type": None,
            "rir_source": None,
        }
        if condition != "dry":
            out["snr_db"] = rng.uniform(*aug["snr_db"][condition])
            out["rt60"] = rng.uniform(0.2, 1.1)
            out["source_distance"] = rng.uniform(0.8, 6.0)
            out["noise_type"] = rng.choice(["white", "pink", "brown", "babble_like"])
            out["rir_source"] = "procedural_exponential"
            effects_cfg = aug.get("effects", {})
            spectral_prob = effects_cfg.get("spectral_probability", {"mid": 0.35, "low": 0.35})
            if rng.random() < float(spectral_prob.get(condition, 0.0)):
                spectral_types = effects_cfg.get("spectral_types", {})
                out["effects"].append(rng.choice(spectral_types.get(condition, ["lowpass"])))
            clipping_prob = effects_cfg.get("clipping_probability", {"low": 0.15})
            if rng.random() < float(clipping_prob.get(condition, 0.0)):
                out["effects"].append("soft_clip")
            dropout_prob = effects_cfg.get("dropout_probability", {})
            if rng.random() < float(dropout_prob.get(condition, 0.0)):
                out["effects"].append("frame_dropout")
        rows.append(out)
        # Keep the same object alive for deterministic parity if future sampling uses it.
        rng_order.random()
    return rows


def summarize_errors_by_effect(errors: list[dict]) -> dict:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in errors:
        try:
            effects = json.loads(row.get("effects") or "[]")
        except json.JSONDecodeError:
            effects = []
        for effect in effects or ["none"]:
            buckets[str(effect)].append(row)
        buckets["compound" if len(effects) > 1 else "single_or_dry"].append(row)
    out = {}
    for effect, items in sorted(buckets.items()):
        ref = sum(int(r["reference_length"]) for r in items)
        s = sum(int(r["substitutions"]) for r in items)
        d = sum(int(r["deletions"]) for r in items)
        i = sum(int(r["insertions"]) for r in items)
        out[effect] = {
            "n": len(items),
            "wer": 0.0 if ref == 0 else 100.0 * (s + d + i) / ref,
            "substitutions": s,
            "deletions": d,
            "insertions": i,
            "reference_words": ref,
            "sdi_mix": {
                "substitution_rate": 0.0 if s + d + i == 0 else s / (s + d + i),
                "deletion_rate": 0.0 if s + d + i == 0 else d / (s + d + i),
                "insertion_rate": 0.0 if s + d + i == 0 else i / (s + d + i),
            },
        }
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--proxy-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--errors-csv", action="append", default=[])
    parser.add_argument("--train-samples", type=int, default=2000)
    args = parser.parse_args()

    train_cfg = load_yaml(args.train_config)
    proxy_rows = read_jsonl(args.proxy_manifest)
    train_rows = sample_training_augmentation(train_cfg, args.train_samples)
    proxy_summary = summarize_metadata(proxy_rows)
    train_summary = summarize_metadata(train_rows)
    proxy_effects = set(proxy_summary["effects"])
    train_effects = set(train_summary["effects"])
    report = {
        "train_config": args.train_config,
        "proxy_manifest": args.proxy_manifest,
        "train_samples": args.train_samples,
        "train_sampled_metadata": train_summary,
        "proxy_metadata": proxy_summary,
        "missing_in_train": sorted(proxy_effects - train_effects),
        "train_only": sorted(train_effects - proxy_effects),
        "error_slices": {},
    }
    for path in args.errors_csv:
        report["error_slices"][path] = summarize_errors_by_effect(read_csv(path))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "simulator_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    rows = []
    for source, summary in [("train_sampled", train_summary), ("proxy", proxy_summary)]:
        total = max(summary["count"], 1)
        for effect, count in sorted(summary["effects"].items()):
            rows.append({"source": source, "effect": effect, "count": count, "rate": count / total})
    write_csv(out_dir / "effect_distribution.csv", rows, ["source", "effect", "count", "rate"])

    with open(out_dir / "README.md", "w", encoding="utf-8") as f:
        f.write("# Simulator Audit\n\n")
        f.write(f"- Train config: `{args.train_config}`\n")
        f.write(f"- Proxy manifest: `{args.proxy_manifest}`\n")
        f.write(f"- Missing effects in train: {', '.join(report['missing_in_train']) or 'none'}\n")
        f.write(f"- Train-only effects: {', '.join(report['train_only']) or 'none'}\n\n")
        f.write("See `simulator_audit.json` and `effect_distribution.csv` for distributions and error slices.\n")
    print(json.dumps({
        "output_dir": str(out_dir),
        "missing_in_train": report["missing_in_train"],
        "train_only": report["train_only"],
        "train_effects": train_summary["effects"],
        "proxy_effects": proxy_summary["effects"],
    }, indent=2))


if __name__ == "__main__":
    main()
