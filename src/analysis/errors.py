from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.utils.metrics import align_words


def read_rows(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(value) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def token_pairs(ref: str, hyp: str) -> Counter[tuple[str, str]]:
    pairs = Counter()
    for op in align_words(ref, hyp):
        if op.kind == "substitution" and op.ref is not None and op.hyp is not None:
            pairs[(op.ref, op.hyp)] += 1
    return pairs


def bin_value(value: float | None, edges: list[float], label: str) -> str:
    if value is None:
        return f"{label}:unknown"
    prev = "-inf"
    for edge in edges:
        if value <= edge:
            return f"{label}:{prev}..{edge}"
        prev = str(edge)
    return f"{label}:{prev}..inf"


def summarize_slice(rows: list[dict]) -> dict:
    ref_words = sum(int(r["reference_length"]) for r in rows)
    s = sum(int(r["substitutions"]) for r in rows)
    d = sum(int(r["deletions"]) for r in rows)
    i = sum(int(r["insertions"]) for r in rows)
    return {
        "n": len(rows),
        "wer": 0.0 if ref_words == 0 else 100.0 * (s + d + i) / ref_words,
        "substitutions": s,
        "deletions": d,
        "insertions": i,
        "reference_words": ref_words,
        "empty_outputs": sum(1 for r in rows if not r["hypothesis_norm"]),
        "wer_gt_100": sum(1 for r in rows if float(r["wer"]) > 100.0),
    }


def length_bucket(duration: float | None) -> str:
    if duration is None:
        return "duration:unknown"
    if duration < 4:
        return "duration:short"
    if duration < 10:
        return "duration:medium"
    return "duration:long"


def analyze(input_csv: str | Path, output_dir: str | Path) -> None:
    rows = read_rows(input_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slices: dict[str, dict[str, dict]] = defaultdict(dict)
    for key in ["condition", "noise_type", "rir_source"]:
        buckets = defaultdict(list)
        for row in rows:
            buckets[row.get(key) or "unknown"].append(row)
        for name, items in buckets.items():
            slices[key][name] = summarize_slice(items)

    numeric_specs = {
        "snr_db": [-3, 0, 6, 8, 12, 14, 20, 25],
        "rt60": [0.2, 0.5, 0.8, 1.1, 1.3],
        "source_distance": [1.0, 2.0, 4.0, 6.0, 8.0],
    }
    for key, edges in numeric_specs.items():
        buckets = defaultdict(list)
        for row in rows:
            buckets[bin_value(safe_float(row.get(key)), edges, key)].append(row)
        for name, items in buckets.items():
            slices[key][name] = summarize_slice(items)

    duration_buckets = defaultdict(list)
    for row in rows:
        duration_buckets[length_bucket(safe_float(row.get("duration")))].append(row)
    for name, items in duration_buckets.items():
        slices["duration"][name] = summarize_slice(items)

    effect_buckets = defaultdict(list)
    for row in rows:
        try:
            effects = json.loads(row.get("effects") or "[]")
        except json.JSONDecodeError:
            effects = []
        effect_buckets["compound" if len(effects) > 1 else "single_or_dry"].append(row)
        for effect in effects or ["dry"]:
            effect_buckets[str(effect)].append(row)
    for name, items in effect_buckets.items():
        slices["effects"][name] = summarize_slice(items)

    global_summary = summarize_slice(rows)
    (out_dir / "slice_metrics.json").write_text(
        json.dumps({"global": global_summary, "slices": slices}, indent=2),
        encoding="utf-8",
    )

    worst = sorted(rows, key=lambda r: float(r["wer"]), reverse=True)[:100]
    with open(out_dir / "worst_100.md", "w", encoding="utf-8") as f:
        f.write("# Worst 100\n\n")
        for row in worst:
            f.write(
                f"## {row['proxy_id']} | {row['condition']} | WER {float(row['wer']):.1f}\n\n"
                f"- ref: {row['reference_norm']}\n"
                f"- hyp: {row['hypothesis_norm']}\n"
                f"- S/D/I: {row['substitutions']}/{row['deletions']}/{row['insertions']}\n\n"
            )

    pairs = Counter()
    for row in rows:
        pairs.update(token_pairs(row["reference_norm"], row["hypothesis_norm"]))
    with open(out_dir / "confusion_pairs.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["reference_token", "hypothesis_token", "count"])
        writer.writeheader()
        for (ref, hyp), count in pairs.most_common(500):
            writer.writerow({"reference_token": ref, "hypothesis_token": hyp, "count": count})

    with open(out_dir / "condition_summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["condition", "n", "wer", "substitutions", "deletions", "insertions", "reference_words", "empty_outputs", "wer_gt_100"],
        )
        writer.writeheader()
        for condition, vals in sorted(slices["condition"].items()):
            writer.writerow({"condition": condition, **vals})

    (out_dir / "regressions.md").write_text(
        "# Regressions\n\nProvide a control and candidate errors.csv to compute paired regressions.\n",
        encoding="utf-8",
    )
    (out_dir / "report.html").write_text(
        "<html><body><h1>Error report</h1><p>See slice_metrics.json, condition_summary.csv, confusion_pairs.csv, and worst_100.md.</p></body></html>\n",
        encoding="utf-8",
    )


def paired_regressions(control_csv: str | Path, candidate_csv: str | Path, output_dir: str | Path) -> None:
    control = {r["proxy_id"]: r for r in read_rows(control_csv)}
    candidate = {r["proxy_id"]: r for r in read_rows(candidate_csv)}
    ids = sorted(set(control) & set(candidate))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for proxy_id in ids:
        c = control[proxy_id]
        n = candidate[proxy_id]
        delta = float(n["wer"]) - float(c["wer"])
        rows.append({
            "proxy_id": proxy_id,
            "condition": n["condition"],
            "control_wer": float(c["wer"]),
            "candidate_wer": float(n["wer"]),
            "delta_wer": delta,
            "control_s": int(c["substitutions"]),
            "control_d": int(c["deletions"]),
            "control_i": int(c["insertions"]),
            "candidate_s": int(n["substitutions"]),
            "candidate_d": int(n["deletions"]),
            "candidate_i": int(n["insertions"]),
            "delta_s": int(n["substitutions"]) - int(c["substitutions"]),
            "delta_d": int(n["deletions"]) - int(c["deletions"]),
            "delta_i": int(n["insertions"]) - int(c["insertions"]),
            "reference": n["reference"],
            "control_hypothesis": c["hypothesis"],
            "candidate_hypothesis": n["hypothesis"],
        })
    with open(out_dir / "paired_regressions.csv", "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "proxy_id", "condition", "control_wer", "candidate_wer", "delta_wer",
            "control_s", "control_d", "control_i", "candidate_s", "candidate_d", "candidate_i",
            "delta_s", "delta_d", "delta_i", "reference", "control_hypothesis", "candidate_hypothesis",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["delta_wer"], reverse=True))
    by_cond = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    summary = {
        "paired_count": len(rows),
        "mean_delta_wer": sum(r["delta_wer"] for r in rows) / max(len(rows), 1),
        "improved": sum(1 for r in rows if r["delta_wer"] < 0),
        "regressed": sum(1 for r in rows if r["delta_wer"] > 0),
        "unchanged": sum(1 for r in rows if r["delta_wer"] == 0),
        "condition": {},
    }
    for cond, items in by_cond.items():
        summary["condition"][cond] = {
            "n": len(items),
            "mean_delta_wer": sum(r["delta_wer"] for r in items) / max(len(items), 1),
            "improved": sum(1 for r in items if r["delta_wer"] < 0),
            "regressed": sum(1 for r in items if r["delta_wer"] > 0),
            "delta_s": sum(r["delta_s"] for r in items),
            "delta_d": sum(r["delta_d"] for r in items),
            "delta_i": sum(r["delta_i"] for r in items),
        }
    (out_dir / "paired_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(out_dir / "paired_worst_50.md", "w", encoding="utf-8") as f:
        f.write("# Worst Paired Regressions\n\n")
        for row in sorted(rows, key=lambda r: r["delta_wer"], reverse=True)[:50]:
            f.write(
                f"## {row['proxy_id']} | {row['condition']} | delta {row['delta_wer']:.1f}\n\n"
                f"- ref: {row['reference']}\n"
                f"- control: {row['control_hypothesis']}\n"
                f"- candidate: {row['candidate_hypothesis']}\n\n"
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--control")
    parser.add_argument("--candidate")
    args = parser.parse_args()
    analyze(args.input, args.output_dir)
    if args.control and args.candidate:
        paired_regressions(args.control, args.candidate, args.output_dir)


if __name__ == "__main__":
    main()
