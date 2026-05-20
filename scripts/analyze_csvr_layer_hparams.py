#!/usr/bin/env python
"""
Summarize Step9R retrieved-doc CSV layer and hyperparameter sweeps.

This is an offline helper for the paper's hyperparameter/ablation section. It
parses existing sweep JSON files and writes long-form CSV tables that can be
used directly for heatmaps or appendix tables.

Recommended:
  python scripts/analyze_csvr_layer_hparams.py

Outputs:
  results/analysis/paper_supplements/csvr_layer_sweep.csv
  results/analysis/paper_supplements/csvr_best_by_model.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "analysis" / "paper_supplements"

DEFAULT_SWEEPS = [
    PROJECT_ROOT / "results" / "csv_retrieved" / "gemma2b_retrieved_csv_sweep.json",
    PROJECT_ROOT / "results" / "csv_retrieved" / "qwen3_4b_retrieved_csv_sweep.json",
    PROJECT_ROOT / "results" / "csv_retrieved_qwen3_4b_cls22_hparam_grid" / "lr0p001_ema0p999" / "qwen3_4b_retrieved_csv_sweep.json",
]

KEY_RE = re.compile(r"inject_(?P<inject>\d+)_cls_(?P<cls>final|-?\d+)")


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_float_name(text: str) -> float | None:
    try:
        return float(text.replace("p", ".").replace("m", "-"))
    except Exception:
        return None


def infer_source_meta(path: Path) -> tuple[str, str, float | None, float | None]:
    name = path.name
    model = "unknown"
    if "gemma2b" in name or "gemma2b" in str(path):
        model = "gemma2b"
    elif "qwen3_4b" in name or "qwen3_4b" in str(path):
        model = "qwen3_4b"
    elif "gemma9b" in name or "gemma9b" in str(path):
        model = "gemma9b"

    source = path.parent.name
    lr = None
    ema = None
    m = re.match(r"lr(?P<lr>[^_]+)_ema(?P<ema>.+)", source)
    if m:
        lr = parse_float_name(m.group("lr"))
        ema = parse_float_name(m.group("ema"))
    return model, source, lr, ema


def parse_sweep(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a dict sweep")

    model, source, lr_from_dir, ema_from_dir = infer_source_meta(path)
    rows = []
    for key, entry in payload.items():
        if not isinstance(entry, dict):
            continue
        match = KEY_RE.search(key)
        if not match:
            continue
        cls_raw = match.group("cls")
        cls_layer = -1 if cls_raw == "final" else int(cls_raw)

        rows.append({
            "model": entry.get("base_model_key", model),
            "run_name": entry.get("run_name", entry.get("model_name", model)),
            "source": source,
            "sweep_file": str(path.relative_to(PROJECT_ROOT)),
            "config": key,
            "inject_layer": int(match.group("inject")),
            "cls_layer": cls_layer,
            "best_auroc": float(entry.get("best_auroc", 0.0)),
            "best_accuracy": float(entry.get("best_accuracy", 0.0)),
            "best_avg_margin": float(entry.get("best_avg_margin", 0.0)),
            "best_epoch": int(entry.get("best_epoch", -1)),
            "lr": entry.get("lr", lr_from_dir),
            "ema_decay": entry.get("ema_decay", ema_from_dir),
            "checkpoint": entry.get("checkpoint", ""),
        })
    return rows


def best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["model"]), str(row["source"])), []).append(row)

    out = []
    for (model, source), group in sorted(grouped.items()):
        best = max(group, key=lambda r: (float(r["best_auroc"]), float(r["best_accuracy"])))
        out.append({
            "model": model,
            "source": source,
            "config": best["config"],
            "inject_layer": best["inject_layer"],
            "cls_layer": best["cls_layer"],
            "best_auroc": best["best_auroc"],
            "best_accuracy": best["best_accuracy"],
            "best_epoch": best["best_epoch"],
            "lr": best["lr"],
            "ema_decay": best["ema_decay"],
            "sweep_file": best["sweep_file"],
            "checkpoint": best["checkpoint"],
        })
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweeps", nargs="*", default=[str(p) for p in DEFAULT_SWEEPS])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for raw in args.sweeps:
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            print(f"[skip] missing sweep: {path}")
            continue
        parsed = parse_sweep(path)
        print(f"Loaded {len(parsed)} configs from {path}")
        rows.extend(parsed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows, OUT_DIR / "csvr_layer_sweep.csv")
    write_csv(best_rows(rows), OUT_DIR / "csvr_best_by_model.csv")
    print(f"Wrote {len(rows)} layer-sweep rows -> {OUT_DIR / 'csvr_layer_sweep.csv'}")
    print(f"Wrote best rows -> {OUT_DIR / 'csvr_best_by_model.csv'}")


if __name__ == "__main__":
    main()
