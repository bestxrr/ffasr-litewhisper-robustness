from __future__ import annotations

import os
from pathlib import Path


def configure_project_environment(root: str | Path) -> None:
    root = Path(root).resolve()
    os.environ.setdefault("HF_HOME", str(root / ".cache" / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(root / ".cache" / "torch"))
    os.environ.setdefault("WANDB_DIR", str(root / "artifacts" / "wandb"))
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    for key in ("HF_HOME", "TORCH_HOME", "WANDB_DIR"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def credential_presence() -> dict[str, bool]:
    return {
        key: bool(os.environ.get(key))
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY", "CONTAINER_API_KEY")
    }
