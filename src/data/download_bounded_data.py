from __future__ import annotations

import json
import io
from pathlib import Path

import soundfile as sf

from src.utils.config import load_yaml
from src.utils.disk import assert_within_budget
from src.utils.env import configure_project_environment


def _audio_array(example: dict):
    audio = example.get("audio")
    if isinstance(audio, dict):
        if "array" in audio:
            return audio["array"], int(audio["sampling_rate"])
        if audio.get("bytes") is not None:
            return sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        path = audio.get("path") or example.get("file")
        if path:
            return sf.read(path, dtype="float32")
    path = example.get("file")
    if path:
        return sf.read(path, dtype="float32")
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        data = samples.data
        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        if data.ndim > 1:
            data = data[0]
        return data, int(samples.sample_rate)
    raise ValueError(f"Unsupported audio field type: {type(audio)!r}")


def main() -> None:
    import argparse
    from datasets import Audio, load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    root = Path.cwd()
    configure_project_environment(root)
    cfg = load_yaml(args.config)
    assert_within_budget(root, 48.0)
    out_root = root / cfg["output"]["root"]
    audio_dir = out_root / cfg["output"]["audio_dir"]
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / cfg["output"]["manifest"]
    ds_cfg = cfg["dataset"]
    ds = load_dataset(
        ds_cfg["hf_id"],
        ds_cfg.get("name"),
        split=ds_cfg["split"],
        streaming=bool(ds_cfg.get("streaming", True)),
        trust_remote_code=bool(ds_cfg.get("trust_remote_code", False)),
    )
    try:
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception:
        pass
    n = 0
    with open(manifest_path, "w", encoding="utf-8") as f:
        for ex in ds:
            audio, sr = _audio_array(ex)
            dur = len(audio) / sr
            if dur < cfg["selection"]["min_seconds"] or dur > cfg["selection"]["max_seconds"]:
                continue
            sample_id = str(ex.get("id") or ex.get("file") or f"sample_{n:06d}").replace("/", "_")
            wav_path = audio_dir / f"{sample_id}.wav"
            sf.write(wav_path, audio, sr)
            row = {
                "sample_id": sample_id,
                "audio_filepath": str(wav_path.relative_to(root)),
                "text": ex.get("text") or ex.get("sentence") or "",
                "speaker_id": str(ex.get("speaker_id") or ex.get("speaker") or ""),
                "duration": dur,
                "sample_rate": sr,
                "source_dataset": ds_cfg["hf_id"],
                "source_config": ds_cfg.get("name"),
                "source_split": ds_cfg["split"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if n >= int(ds_cfg["max_items"]):
                break
    print(json.dumps({"manifest": str(manifest_path), "items": n}, indent=2))


if __name__ == "__main__":
    main()
