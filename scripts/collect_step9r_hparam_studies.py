#!/usr/bin/env python
"""
Collect Step9R hyperparameter-study results into compact CSV tables.

Expected input directory layout is produced by:
  scripts/run_step9r_hparam_studies.sh

Outputs:
  results/analysis/hparam_studies_retrieved/classification_layer_c_sweep.csv
  results/analysis/hparam_studies_retrieved/lambda_sweep.csv
  results/analysis/hparam_studies_retrieved/hparam_studies_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_DIR = PROJECT_ROOT / "results" / "hparam_studies_retrieved"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "analysis" / "hparam_studies_retrieved"

MODEL_RUN_NAMES = {
    "gemma2b": "gemma2b_retrieved",
    "qwen3_4b": "qwen3_4b_retrieved",
}


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def cls_label(x: int) -> str:
    return "final" if int(x) == -1 else str(int(x))


def parse_lam_from_dir(path: Path) -> float | None:
    for part in reversed(path.parts):
        if part.startswith("lam"):
            raw = part[len("lam"):]
            try:
                return float(raw.replace("m", "-").replace("p", "."))
            except ValueError:
                return None
    return None


def row_from_result(
    *,
    model: str,
    study: str,
    result: dict[str, Any],
    source: Path,
    lam: float | None,
) -> dict[str, Any]:
    return {
        "study": study,
        "model": model,
        "lambda": lam,
        "str_layer": int(result["str_layer"]),
        "cls_layer": int(result["cls_layer"]),
        "cls_layer_label": cls_label(int(result["cls_layer"])),
        "best_auroc": float(result["best_auroc"]),
        "best_accuracy": float(result["best_accuracy"]),
        "best_avg_margin": float(result.get("best_avg_margin", 0.0)),
        "best_epoch": int(result.get("best_epoch", -1)),
        "final_auroc": float(result.get("final_auroc", result["best_auroc"])),
        "final_accuracy": float(result.get("final_accuracy", result["best_accuracy"])),
        "checkpoint": result.get("checkpoint"),
        "source_file": relpath(source),
    }


def collect_c_sweep(base_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_RUN_NAMES:
        for sweep_path in sorted(base_dir.glob(f"{model}_c_sweep_fixed_k*/{MODEL_RUN_NAMES[model]}_csv_sweep.json")):
            sweep = load_json(sweep_path)
            if not isinstance(sweep, dict):
                continue
            for result in sweep.values():
                rows.append(row_from_result(
                    model=model,
                    study="classification_layer_c",
                    result=result,
                    source=sweep_path,
                    lam=5.0,
                ))

    rows.sort(key=lambda r: (
        str(r["model"]),
        int(r["str_layer"]),
        10_000 if int(r["cls_layer"]) == -1 else int(r["cls_layer"]),
    ))
    return rows


def collect_lambda_sweep(base_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, run_name in MODEL_RUN_NAMES.items():
        pattern = f"{model}_lambda_sweep_fixed_k*_c*/lam*/{run_name}_inject_*_cls_*_csv_result.json"
        for result_path in sorted(base_dir.glob(pattern)):
            result = load_json(result_path)
            if not isinstance(result, dict):
                continue
            rows.append(row_from_result(
                model=model,
                study="steering_strength_lambda",
                result=result,
                source=result_path,
                lam=parse_lam_from_dir(result_path.parent),
            ))

    rows.sort(key=lambda r: (
        str(r["model"]),
        float(r["lambda"]) if r["lambda"] is not None else -1.0,
    ))
    return rows


def best_rows(rows: list[dict[str, Any]], study: str) -> list[dict[str, Any]]:
    out = []
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    for model, model_rows in sorted(by_model.items()):
        best = max(model_rows, key=lambda r: (float(r["best_auroc"]), float(r["best_accuracy"])))
        out.append({
            "study": study,
            "model": model,
            "lambda": best["lambda"],
            "str_layer": best["str_layer"],
            "cls_layer": best["cls_layer"],
            "cls_layer_label": best["cls_layer_label"],
            "best_auroc": best["best_auroc"],
            "best_accuracy": best["best_accuracy"],
            "best_epoch": best["best_epoch"],
            "source_file": best["source_file"],
        })
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Step9R hparam-study results")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    if not base_dir.is_absolute():
        base_dir = PROJECT_ROOT / base_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    c_rows = collect_c_sweep(base_dir)
    lambda_rows = collect_lambda_sweep(base_dir)

    c_path = out_dir / "classification_layer_c_sweep.csv"
    lambda_path = out_dir / "lambda_sweep.csv"
    write_csv(c_rows, c_path)
    write_csv(lambda_rows, lambda_path)

    summary = {
        "base_dir": relpath(base_dir),
        "classification_layer_c": {
            "n_rows": len(c_rows),
            "best_by_model": best_rows(c_rows, "classification_layer_c") if c_rows else [],
            "csv": relpath(c_path),
        },
        "steering_strength_lambda": {
            "n_rows": len(lambda_rows),
            "best_by_model": best_rows(lambda_rows, "steering_strength_lambda") if lambda_rows else [],
            "csv": relpath(lambda_path),
        },
    }
    summary_path = out_dir / "hparam_studies_summary.json"
    save_json(summary, summary_path)

    print(f"Wrote c sweep rows:      {len(c_rows)} -> {c_path}")
    print(f"Wrote lambda sweep rows: {len(lambda_rows)} -> {lambda_path}")
    print(f"Wrote summary -> {summary_path}")
    if c_rows:
        print("\nBest c by model:")
        for row in summary["classification_layer_c"]["best_by_model"]:
            print(
                f"  {row['model']}: k={row['str_layer']} c={row['cls_layer_label']} "
                f"AUROC={row['best_auroc']:.4f} Acc={row['best_accuracy']:.4f}"
            )
    if lambda_rows:
        print("\nBest lambda by model:")
        for row in summary["steering_strength_lambda"]["best_by_model"]:
            print(
                f"  {row['model']}: lambda={row['lambda']} "
                f"k={row['str_layer']} c={row['cls_layer_label']} "
                f"AUROC={row['best_auroc']:.4f} Acc={row['best_accuracy']:.4f}"
            )


if __name__ == "__main__":
    main()
