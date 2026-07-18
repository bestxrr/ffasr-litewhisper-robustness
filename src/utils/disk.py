from __future__ import annotations

import os
from pathlib import Path


def bytes_to_gb(n: int) -> float:
    return n / (1024**3)


def dir_size_bytes(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not Path(base, d).is_symlink()]
        for name in files:
            p = Path(base, name)
            try:
                if not p.is_symlink():
                    total += p.stat().st_size
            except FileNotFoundError:
                pass
    return total


def project_usage(workspace: str | Path = ".") -> dict[str, float]:
    root = Path(workspace)
    keys = [".cache", "external", "artifacts", "configs", "src", "scripts", "tests", "docs"]
    out = {k: bytes_to_gb(dir_size_bytes(root / k)) for k in keys}
    out["total"] = bytes_to_gb(dir_size_bytes(root))
    return out


def assert_within_budget(workspace: str | Path = ".", max_gb: float = 48.0) -> None:
    total = project_usage(workspace)["total"]
    if total > max_gb:
        raise SystemExit(f"Project disk usage {total:.2f} GB exceeds budget {max_gb:.2f} GB")
