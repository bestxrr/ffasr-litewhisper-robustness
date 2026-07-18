from __future__ import annotations

import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

from src.utils.config import load_yaml
from src.utils.disk import assert_within_budget
from src.utils.env import configure_project_environment


def iter_rows_from_parquet(path: str | Path):
    table = pq.read_table(path)
    for row in table.to_pylist():
        yield row


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    root = Path.cwd()
    configure_project_environment(root)
    assert_within_budget(root, 48.0)
    cfg = load_yaml(args.config)
    out_root = root / cfg["output"]["root"]
    audio_dir = out_root / cfg["output"]["audio_dir"]
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / cfg["output"]["manifest"]
    max_items = int(cfg["max_items"])
    n = 0
    with open(manifest_path, "w", encoding="utf-8") as f:
        for remote_file in cfg["files"]:
            local = hf_hub_download(
                cfg["repo_id"],
                remote_file,
                repo_type=cfg.get("repo_type", "dataset"),
            )
            for row in iter_rows_from_parquet(local):
                audio = row["audio"]
                audio_bytes = audio.get("bytes")
                if audio_bytes is None:
                    continue
                samples, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                if getattr(samples, "ndim", 1) > 1:
                    samples = samples[:, 0]
                dur = float(len(samples)) / float(sr)
                if dur < cfg["selection"]["min_seconds"] or dur > cfg["selection"]["max_seconds"]:
                    continue
                sample_id = str(row.get("id") or f"sample_{n:06d}")
                wav_path = audio_dir / f"{sample_id}.wav"
                sf.write(wav_path, samples, sr)
                out = {
                    "sample_id": sample_id,
                    "audio_filepath": str(wav_path.relative_to(root)),
                    "text": row.get("text") or "",
                    "speaker_id": str(row.get("speaker_id") or ""),
                    "chapter_id": str(row.get("chapter_id") or ""),
                    "duration": dur,
                    "sample_rate": int(sr),
                    "source_dataset": cfg["repo_id"],
                    "source_file": remote_file,
                }
                f.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")
                n += 1
                if n >= max_items:
                    break
            if n >= max_items:
                break
    print(json.dumps({"manifest": str(manifest_path), "items": n}, indent=2))


if __name__ == "__main__":
    main()
