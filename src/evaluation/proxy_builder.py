from __future__ import annotations

import json
import random
from pathlib import Path

from src.utils.config import config_hash, load_yaml
from src.utils.disk import assert_within_budget


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _uniform(rng: random.Random, lo_hi: list[float]) -> float:
    return rng.uniform(float(lo_hi[0]), float(lo_hi[1]))


def make_effect_params(cfg: dict, condition: str, idx: int, clean: dict, seed: int) -> dict:
    rng = random.Random(seed)
    params = {
        "condition": condition,
        "condition_index": idx,
        "sample_seed": seed,
        "effects": [],
        "snr_db": None,
        "rt60": None,
        "room_volume": None,
        "source_distance": None,
        "noise_type": None,
        "rir_source": None,
    }
    if condition == "dry":
        params["effects"] = ["dry"]
        return params
    params["snr_db"] = _uniform(rng, cfg["snr_db"][condition])
    params["noise_type"] = rng.choice(["white", "pink", "brown", "babble_like"])
    params["effects"].append("additive_noise")
    if condition in set(cfg["rir"]["enabled_for"]):
        params["rt60"] = _uniform(rng, cfg["rir"]["rt60_range"])
        params["room_volume"] = _uniform(rng, cfg["rir"]["room_volume_m3_range"])
        params["source_distance"] = _uniform(rng, cfg["rir"]["distance_m_range"])
        params["rir_source"] = "procedural_exponential"
        params["effects"].append("rir")
    if rng.random() < float(cfg["effects"]["spectral_probability"][condition]):
        params["effects"].append(rng.choice(["lowpass", "highpass", "bandpass"]))
    if rng.random() < float(cfg["effects"]["clipping_probability"][condition]):
        params["effects"].append("soft_clip")
    return params


def build_proxy(cfg: dict) -> tuple[list[dict], list[dict]]:
    source_rows = read_jsonl(cfg["source_manifest"])
    needed = int(cfg["counts"]["full_per_condition"]) * len(cfg["conditions"])
    if len(source_rows) < needed:
        raise RuntimeError(
            f"Need at least {needed} clean utterances for disjoint full proxy; found {len(source_rows)}"
        )
    rng = random.Random(int(cfg["seed"]))
    shuffled = list(source_rows)
    rng.shuffle(shuffled)
    full_rows: list[dict] = []
    cursor = 0
    for condition in cfg["conditions"]:
        for idx in range(int(cfg["counts"]["full_per_condition"])):
            clean = shuffled[cursor]
            cursor += 1
            seed = rng.randrange(1, 2**31 - 1)
            params = make_effect_params(cfg, condition, idx, clean, seed)
            row = {
                "proxy_id": f"{condition}_{idx:04d}",
                "condition": condition,
                "reference": clean["text"],
                "clean_sample_id": clean["sample_id"],
                "speaker_id": clean.get("speaker_id", ""),
                "audio_filepath": clean["audio_filepath"],
                "duration": clean["duration"],
                "sample_rate": clean["sample_rate"],
                **params,
            }
            full_rows.append(row)
    mini_rows = []
    for condition in cfg["conditions"]:
        cond_rows = [r for r in full_rows if r["condition"] == condition]
        mini_rows.extend(cond_rows[: int(cfg["counts"]["mini_per_condition"])])
    return full_rows, mini_rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    assert_within_budget(".", 48.0)
    cfg = load_yaml(args.config)
    full_rows, mini_rows = build_proxy(cfg)
    write_jsonl(cfg["full_proxy_manifest"], full_rows)
    write_jsonl(cfg["mini_proxy_manifest"], mini_rows)
    summary = {
        "config": args.config,
        "config_hash": config_hash(cfg),
        "full_manifest": cfg["full_proxy_manifest"],
        "mini_manifest": cfg["mini_proxy_manifest"],
        "full_count": len(full_rows),
        "mini_count": len(mini_rows),
        "conditions": {c: sum(1 for r in full_rows if r["condition"] == c) for c in cfg["conditions"]},
    }
    out = Path(cfg["full_proxy_manifest"]).parent / "proxy_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
