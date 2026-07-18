from __future__ import annotations

import csv
import html
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any


SPACE_URL = "https://treble-technologies-ffasr.hf.space"


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _read_sse(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = resp.read().decode("utf-8")
    for block in payload.split("\n\n"):
        if block.startswith("event: complete"):
            data_line = next((line[5:].strip() for line in block.splitlines() if line.startswith("data:")), "")
            return json.loads(data_line)
    raise RuntimeError(f"No complete event in response: {payload[:500]}")


def fetch_startup(space_url: str = SPACE_URL) -> list[Any]:
    started = _post_json(f"{space_url}/gradio_api/call/_on_startup", {"data": []})
    event_id = started["event_id"]
    time.sleep(0.5)
    return _read_sse(f"{space_url}/gradio_api/call/_on_startup/{event_id}")


def clean_model_cell(cell: str) -> str:
    text = re.sub(r"<[^>]+>", "", str(cell))
    return html.unescape(text).strip()


def extract_leaderboard(startup_payload: list[Any]) -> tuple[list[str], list[list[Any]]]:
    table_update = startup_payload[0]
    value = table_update["value"]
    return value["headers"], value["data"]


def write_leaderboard_snapshot(out_dir: str | Path, space_url: str = SPACE_URL) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = fetch_startup(space_url)
    headers, rows = extract_leaderboard(payload)
    normalized_rows = []
    for row in rows:
        r = list(row)
        r[0] = clean_model_cell(r[0])
        normalized_rows.append(r)
    csv_path = out / "leaderboard_latest.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(normalized_rows)
    raw_path = out / "leaderboard_startup_payload.json"
    raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = summarize_thresholds(headers, normalized_rows)
    (out / "leaderboard_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def summarize_thresholds(headers: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    avg_idx = headers.index("Avg WER (%)")
    model_idx = headers.index("Model")
    sorted_rows = sorted(rows, key=lambda r: float(r[avg_idx]))
    def row_at(rank: int) -> dict[str, Any] | None:
        if len(sorted_rows) < rank:
            return None
        r = sorted_rows[rank - 1]
        return {"rank": rank, "model": r[model_idx], "avg_wer": float(r[avg_idx])}
    return {
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_entries": len(sorted_rows),
        "top5_threshold": row_at(5),
        "top7_threshold": row_at(7),
        "top_entries": [
            {"rank": i + 1, "model": r[model_idx], "avg_wer": float(r[avg_idx])}
            for i, r in enumerate(sorted_rows[:10])
        ],
    }
