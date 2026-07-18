from __future__ import annotations

import csv
import json
from pathlib import Path

from src.evaluation.proxy_eval import summarize
from src.utils.config import config_hash, load_yaml
from src.utils.disk import project_usage


def read_rows(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_condition_summary(path: Path, summary: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "n",
                "wer",
                "substitutions",
                "deletions",
                "insertions",
                "reference_words",
                "empty_outputs",
                "wer_gt_100",
            ],
        )
        writer.writeheader()
        for cond, vals in summary["condition_summary"].items():
            writer.writerow({"condition": cond, **vals})


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--errors-csv")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(cfg["output_dir"])
    errors_csv = Path(args.errors_csv) if args.errors_csv else out_dir / "errors.csv"
    rows = read_rows(errors_csv)
    baseline_dry = cfg["promotion"].get("baseline_dry_wer")
    summary = summarize(
        rows,
        baseline_dry_wer=None if baseline_dry is None else float(baseline_dry),
        length_ratio_threshold=float(cfg["promotion"]["catastrophic_length_ratio_threshold"]),
    )
    summary.update(
        {
            "run_id": cfg["run_id"],
            "config_hash": config_hash(cfg),
            "adapter": {"adapter_loaded": bool(cfg.get("adapter"))},
            "wall_time_s": None,
            "num_samples": len(rows),
            "disk_gb": project_usage("."),
            "summarized_from_errors_csv": str(errors_csv),
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_condition_summary(out_dir / "condition_summary.csv", summary)
    print(json.dumps(summary, indent=2)[:8000])


if __name__ == "__main__":
    main()
