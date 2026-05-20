#!/usr/bin/env python
"""Collect completed experiment results into one CSV table.

The collector is intentionally read-only for existing result artifacts. It
normalizes the result formats used in this repo into one wide CSV:

  results/all_finished_experiments.csv

Each row is one completed unit, such as a Step11/12 pipeline evaluation, one CSV
probe layer configuration, one probing layer/dataset result, or one norm/logit
lens layer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_OUT = RESULTS_DIR / "all_finished_experiments.csv"

COMMON_COLUMNS = [
    "result_family",
    "result_source",
    "model",
    "dataset",
    "method",
    "split",
    "layer",
    "str_layer",
    "cls_layer",
    "alpha",
    "tau",
    "contrast_layer",
    "temperature",
    "top_p",
    "max_new_tokens",
    "n",
    "em",
    "f1",
    "accuracy",
    "auroc",
    "best_accuracy",
    "best_auroc",
    "final_accuracy",
    "final_auroc",
    "avg_margin",
    "best_avg_margin",
    "final_avg_margin",
    "best_epoch",
    "empty_rate",
    "unique_rate",
    "top1_prediction_rate",
    "avg_prediction_words",
    "repeat_long_rate",
    "elapsed_sec",
    "note",
    "source_file",
]


def load_json(path: Path) -> Any | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return {"_load_error": repr(exc)}


def scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def add_scalar_fields(row: dict[str, Any], data: dict[str, Any], prefix: str = "") -> None:
    for key, value in data.items():
        out_key = f"{prefix}{key}" if prefix else str(key)
        if scalar(value):
            row[out_key] = value


def infer_model_from_name(path: Path) -> str:
    name = path.name
    for model in ("qwen3_4b", "gemma9b", "gemma2b", "llama8b"):
        if name.startswith(model):
            return model
    return ""


def layer_index(layer_name: str) -> str:
    m = re.search(r"layer_(-?\d+|final)$", layer_name)
    if m:
        return m.group(1)
    m = re.search(r"layer_(-?\d+|final)", layer_name)
    return m.group(1) if m else ""


def prediction_stats(output_path: Path) -> dict[str, Any]:
    data = load_json(output_path)
    if not isinstance(data, dict) or not isinstance(data.get("outputs"), list):
        return {}

    preds = [
        str(item.get("generation", {}).get("prediction", "")).strip()
        for item in data["outputs"]
        if isinstance(item, dict)
    ]
    if not preds:
        return {}

    n = len(preds)
    counts = Counter(preds)
    lens = [len(p.split()) for p in preds]

    repeat_long = 0
    for pred in preds:
        toks = pred.lower().split()
        if len(toks) >= 8 and max(Counter(toks).values(), default=0) >= 5:
            repeat_long += 1
            continue
        grams = [" ".join(toks[i : i + 3]) for i in range(max(0, len(toks) - 2))]
        if grams and max(Counter(grams).values()) >= 3:
            repeat_long += 1

    return {
        "empty_rate": sum(not p for p in preds) / n,
        "unique_rate": len(counts) / n,
        "top1_prediction_rate": counts.most_common(1)[0][1] / n,
        "avg_prediction_words": sum(lens) / n,
        "repeat_long_rate": repeat_long / n,
    }


def collect_pipeline(rows: list[dict[str, Any]]) -> None:
    for path in sorted((RESULTS_DIR / "pipeline").glob("eval_*.json")):
        data = load_json(path)
        if not isinstance(data, dict) or "_load_error" in data:
            rows.append({
                "result_family": "pipeline_eval",
                "source_file": str(path.relative_to(PROJECT_ROOT)),
                "note": data.get("_load_error") if isinstance(data, dict) else "load failed",
            })
            continue

        meta = data.get("meta", {}) if isinstance(data.get("meta"), dict) else {}
        metrics = data.get("metrics", {}) if isinstance(data.get("metrics"), dict) else {}
        overall = metrics.get("overall", {}) if isinstance(metrics.get("overall"), dict) else {}

        row = {
            "result_family": "pipeline_eval",
            "result_source": "step11_step12",
            "source_file": str(path.relative_to(PROJECT_ROOT)),
            "model": meta.get("model", ""),
            "dataset": meta.get("dataset", ""),
            "method": meta.get("method", ""),
            "alpha": meta.get("alpha", ""),
            "tau": meta.get("tau", ""),
            "contrast_layer": meta.get("contrast_layer", ""),
            "temperature": meta.get("temperature", ""),
            "top_p": meta.get("top_p", ""),
            "max_new_tokens": meta.get("max_new_tokens", ""),
            "n": overall.get("n", meta.get("num_samples", "")),
            "em": overall.get("em", ""),
            "f1": overall.get("f1", ""),
            "elapsed_sec": meta.get("elapsed_sec", ""),
        }

        input_file = data.get("input_file")
        if isinstance(input_file, str):
            input_path = Path(input_file)
            if not input_path.is_absolute():
                input_path = PROJECT_ROOT / input_path
            if input_path.exists():
                row.update(prediction_stats(input_path))

        empty = row.get("empty_rate")
        if isinstance(empty, float):
            if empty >= 0.25:
                row["note"] = "bad_empty_collapse"
            elif empty >= 0.10:
                row["note"] = "warn_empty"
            else:
                row["note"] = "ok"
        rows.append(row)


def collect_csv_results(rows: list[dict[str, Any]]) -> None:
    roots = [
        RESULTS_DIR / "csv",
        RESULTS_DIR / "csv_retrieved",
        RESULTS_DIR / "csv_retrieved_tune_qwen_i4_c30",
        RESULTS_DIR / "csv_retrieved_qwen3_4b_cls22_hparam_grid",
        RESULTS_DIR / "csv_retrieved_qwen3_4b_best_for_pipeline",
    ]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*_csv_result.json")):
            data = load_json(path)
            row = {
                "result_family": "csv_result",
                "result_source": path.parent.relative_to(RESULTS_DIR).as_posix(),
                "source_file": str(path.relative_to(PROJECT_ROOT)),
                "model": infer_model_from_name(path),
                "dataset_variant": "retrieved" if "csv_retrieved" in path.parts else "original",
            }
            if isinstance(data, dict):
                add_scalar_fields(row, data)
                row["str_layer"] = data.get("str_layer", row.get("str_layer", ""))
                row["cls_layer"] = data.get("cls_layer", row.get("cls_layer", ""))
                if "best_accuracy" in data:
                    row["accuracy"] = data.get("best_accuracy")
                if "best_auroc" in data:
                    row["auroc"] = data.get("best_auroc")
            rows.append(row)

        for path in sorted(root.rglob("*_csv_sweep.json")):
            data = load_json(path)
            if not isinstance(data, dict):
                continue
            model = infer_model_from_name(path)
            for key, entry in sorted(data.items()):
                if not isinstance(entry, dict):
                    continue
                row = {
                    "result_family": "csv_sweep",
                    "result_source": path.parent.relative_to(RESULTS_DIR).as_posix(),
                    "source_file": str(path.relative_to(PROJECT_ROOT)),
                    "model": model,
                    "method": key,
                    "dataset_variant": "retrieved" if "csv_retrieved" in path.parts else "original",
                }
                add_scalar_fields(row, entry)
                row["str_layer"] = entry.get("str_layer", row.get("str_layer", ""))
                row["cls_layer"] = entry.get("cls_layer", row.get("cls_layer", ""))
                if "best_accuracy" in entry:
                    row["accuracy"] = entry.get("best_accuracy")
                if "best_auroc" in entry:
                    row["auroc"] = entry.get("best_auroc")
                rows.append(row)


def collect_probe(rows: list[dict[str, Any]]) -> None:
    for root_name in ("probe", "probe_retrieved"):
        root = RESULTS_DIR / root_name
        if not root.exists():
            continue
        for path in sorted(root.glob("*.json")):
            data = load_json(path)
            if not isinstance(data, dict):
                continue
            model = infer_model_from_name(path)
            method = ""
            m = re.search(r"_probe_results_([^./]+)\.json$", path.name)
            if m:
                method = m.group(1)
            elif path.name == "dpr_baseline.json":
                method = "dpr_baseline"

            for layer_key, layer_data in sorted(data.items()):
                if not isinstance(layer_data, dict):
                    continue
                if not layer_key.startswith("layer_") and path.name != "dpr_baseline.json":
                    continue
                if all(scalar(v) for v in layer_data.values()):
                    row = {
                        "result_family": "probe",
                        "result_source": root_name,
                        "source_file": str(path.relative_to(PROJECT_ROOT)),
                        "model": model,
                        "method": method,
                        "layer": layer_index(layer_key),
                        "dataset_variant": "retrieved" if root_name == "probe_retrieved" else "original",
                    }
                    add_scalar_fields(row, layer_data)
                    rows.append(row)
                    continue
                for dataset, metrics in sorted(layer_data.items()):
                    if not isinstance(metrics, dict):
                        continue
                    row = {
                        "result_family": "probe",
                        "result_source": root_name,
                        "source_file": str(path.relative_to(PROJECT_ROOT)),
                        "model": model,
                        "dataset": dataset,
                        "method": method,
                        "layer": layer_index(layer_key),
                        "dataset_variant": "retrieved" if root_name == "probe_retrieved" else "original",
                    }
                    add_scalar_fields(row, metrics)
                    rows.append(row)


def collect_csv_validations(rows: list[dict[str, Any]]) -> None:
    root = RESULTS_DIR / "pipeline"
    if not root.exists():
        return
    for path in sorted(root.glob("*_retrieved_csv_eval_validation.json")):
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        computed = data.get("computed", {})
        if not isinstance(computed, dict):
            continue
        for dataset, metrics in sorted(computed.items()):
            if not isinstance(metrics, dict):
                continue
            row = {
                "result_family": "csv_validation",
                "result_source": "pipeline_csv_validation",
                "source_file": str(path.relative_to(PROJECT_ROOT)),
                "model": data.get("model_name", infer_model_from_name(path)),
                "dataset": dataset,
                "method": data.get("best_key", ""),
                "dataset_variant": "retrieved",
                "str_layer": data.get("str_layer", ""),
                "cls_layer": data.get("cls_layer", ""),
                "best_auroc": data.get("saved_best_auroc", ""),
                "best_accuracy": data.get("saved_best_accuracy", ""),
                "note": "passed" if data.get("passed") is True else "",
            }
            add_scalar_fields(row, metrics)
            if "accuracy" in metrics:
                row["accuracy"] = metrics.get("accuracy")
            if "auroc" in metrics:
                row["auroc"] = metrics.get("auroc")
            if "n_prompts" in metrics:
                row["n"] = metrics.get("n_prompts")
            rows.append(row)


def collect_pipeline_classifier_scores(rows: list[dict[str, Any]]) -> None:
    root = RESULTS_DIR / "pipeline_classifier_scores"
    if not root.exists():
        return
    for path in sorted(root.glob("*_test_retrieval_csv_score_summary.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for data in reader:
                row = {
                    "result_family": "pipeline_classifier_score",
                    "result_source": "pipeline_classifier_scores",
                    "source_file": str(path.relative_to(PROJECT_ROOT)),
                    "model": data.get("model", infer_model_from_name(path)),
                    "dataset": data.get("dataset", ""),
                    "method": data.get("best_key", ""),
                    "dataset_variant": "test_retrieval",
                    "str_layer": data.get("str_layer", ""),
                    "cls_layer": data.get("cls_layer", ""),
                    "tau": data.get("tau", ""),
                    "n": data.get("n", ""),
                    "best_auroc": data.get("saved_best_auroc", ""),
                    "best_accuracy": data.get("saved_best_accuracy", ""),
                }
                row.update(data)
                rows.append(row)


def collect_norms(rows: list[dict[str, Any]]) -> None:
    for root_name in ("norms", "norms_retrieved"):
        root = RESULTS_DIR / root_name
        if not root.exists():
            continue
        for path in sorted(root.glob("*_norm_stats.json")):
            data = load_json(path)
            if not isinstance(data, dict):
                continue
            for layer_key, metrics in sorted(data.items()):
                if not isinstance(metrics, dict):
                    continue
                row = {
                    "result_family": "norm_stats",
                    "result_source": root_name,
                    "source_file": str(path.relative_to(PROJECT_ROOT)),
                    "model": infer_model_from_name(path),
                    "layer": layer_index(layer_key),
                    "dataset_variant": "retrieved" if root_name == "norms_retrieved" else "original",
                }
                add_scalar_fields(row, metrics)
                rows.append(row)


def collect_logit_lens(rows: list[dict[str, Any]]) -> None:
    root = RESULTS_DIR / "logit_lens"
    if not root.exists():
        return
    for path in sorted(root.glob("*_logit_lens.json")):
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        model = data.get("model_name") or infer_model_from_name(path)
        per_layer = data.get("per_layer")
        if isinstance(per_layer, dict):
            iterable = per_layer.items()
        elif isinstance(per_layer, list):
            iterable = ((str(i), v) for i, v in enumerate(per_layer))
        else:
            iterable = []
        for layer_key, metrics in iterable:
            if not isinstance(metrics, dict):
                continue
            row = {
                "result_family": "logit_lens",
                "result_source": "logit_lens",
                "source_file": str(path.relative_to(PROJECT_ROOT)),
                "model": model,
                "layer": layer_index(str(layer_key)) or str(layer_key),
            }
            add_scalar_fields(row, metrics)
            rows.append(row)

        summary_keys = [
            "answer_signal_emerge_layer",
            "delta_gold_peak_layer",
            "delta_gold_peak_value",
        ]
        if any(k in data for k in summary_keys):
            row = {
                "result_family": "logit_lens_summary",
                "result_source": "logit_lens",
                "source_file": str(path.relative_to(PROJECT_ROOT)),
                "model": model,
            }
            for key in summary_keys:
                row[key] = data.get(key, "")
            rows.append(row)


def clean_value(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    seen = set(COMMON_COLUMNS)
    extra = sorted({key for row in rows for key in row.keys() if key not in seen})
    columns = COMMON_COLUMNS + extra
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean_value(row.get(key, "")) for key in columns})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    collect_pipeline(rows)
    collect_csv_results(rows)
    collect_probe(rows)
    collect_csv_validations(rows)
    collect_pipeline_classifier_scores(rows)
    collect_norms(rows)
    collect_logit_lens(rows)
    write_csv(rows, args.out)

    counts = Counter(row.get("result_family", "") for row in rows)
    print(f"Wrote {len(rows)} rows to {args.out}")
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
