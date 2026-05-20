#!/usr/bin/env python
"""
Visualize whether CSV makes retrieved-document representations more separable.

For each model, this script:
  1. Loads retrieved-doc eval samples from data/final_retrieved/eval.json.
  2. Extracts clean last-token representations at the CSV classification layer c.
  3. Injects the trained CSV vector at layer k and extracts the same c-layer reps.
  4. Fits one PCA per dataset on [clean; CSV] reps so before/after panels are comparable.
  5. Saves a qualitative figure plus quantitative separability metrics.

The "after CSV" representation is an actual forward pass through the wrapped
model. It is not approximated by adding the vector to cached hidden states.

Example:
  CUDA_VISIBLE_DEVICES=1 python scripts/plot_csv_representation_shift_retrieved.py \
    --model gemma2b --max_per_class 300

  CUDA_VISIBLE_DEVICES=1 python scripts/plot_csv_representation_shift_retrieved.py \
    --model all --max_per_class 250
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from csv_module.llm_layers import add_tsv_layers  # noqa: E402
from csv_module.train_utils import (  # noqa: E402
    collate_fn,
    get_last_non_padded_token_rep,
)


DATA_PATH = PROJECT_ROOT / "data" / "final_retrieved" / "eval.json"
OUT_DIR = PROJECT_ROOT / "results" / "case_analysis" / "csv_representation_shift"


MODEL_REGISTRY = {
    "gemma2b": {
        "hf_name": os.environ.get(
            "GEMMA_MODEL_PATH",
            "google/gemma-2-2b",
        ),
        "checkpoint": PROJECT_ROOT / "results" / "csv_retrieved" / "gemma2b_retrieved_inject_14_cls_18_csv.pt",
        "batch_size": 8,
        "display": "Gemma2-2B",
    },
    "qwen3_4b": {
        "hf_name": os.environ.get(
            "QWEN3_4B_MODEL_PATH",
            "Qwen/Qwen3-4B-Base",
        ),
        "checkpoint": PROJECT_ROOT / "results" / "csv_retrieved_qwen3_4b_best_for_pipeline" / "qwen3_4b_retrieved_inject_21_cls_22_csv.pt",
        "fallback_checkpoint": PROJECT_ROOT / "results" / "csv_retrieved" / "qwen3_4b_retrieved_inject_21_cls_22_csv.pt",
        "batch_size": 4,
        "display": "Qwen3-4B",
    },
}


def extract_doc_text(doc: Any) -> str:
    if isinstance(doc, dict):
        title = str(doc.get("title", "")).strip()
        text = str(doc.get("text", "")).strip()
        if title:
            return f"Title: {title}\nText: {text}"
        return text
    return str(doc).strip()


def build_prompt(question: str, document: str) -> str:
    return f"Document: {document}\n\nQuestion: {question}\nAnswer:"


def get_label(item: dict, idx: int) -> int:
    if "label" in item:
        label = int(item["label"])
    else:
        name = str(item.get("label_name", "")).lower().strip()
        if name == "relevant":
            label = 1
        elif name == "distracting":
            label = 0
        else:
            raise ValueError(f"Sample {idx} has invalid label_name={name!r}")
    if label not in (0, 1):
        raise ValueError(f"Sample {idx} has invalid label={label!r}")
    return label


def load_eval_samples(datasets: set[str], max_per_class: int, seed: int) -> list[dict]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    grouped: dict[tuple[str, int], list[dict]] = {}
    for idx, item in enumerate(raw):
        source = str(item.get("source_dataset", "unknown")).lower().strip()
        if datasets and source not in datasets:
            continue
        label = get_label(item, idx)
        prompt = build_prompt(item["question"], extract_doc_text(item["doc"]))
        sample = {
            "idx": idx,
            "source": source,
            "label": label,
            "label_name": "relevant" if label == 1 else "distracting",
            "prompt": prompt,
            "question": item["question"],
        }
        grouped.setdefault((source, label), []).append(sample)

    rng = np.random.default_rng(seed)
    selected = []
    for key in sorted(grouped):
        bucket = grouped[key]
        order = rng.permutation(len(bucket))
        take = min(max_per_class, len(bucket))
        selected.extend(bucket[i] for i in order[:take])

    selected.sort(key=lambda x: (x["source"], x["label"], x["idx"]))
    if not selected:
        raise RuntimeError(f"No samples selected from {DATA_PATH}")
    return selected


def setup_tokenizer(hf_name: str):
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_samples(samples: list[dict], tokenizer, max_length: int | None):
    prompts = []
    truncated = 0
    for sample in samples:
        kwargs = {"return_tensors": "pt"}
        if max_length is not None:
            kwargs.update({"truncation": True, "max_length": max_length})
            if len(tokenizer.encode(sample["prompt"], add_special_tokens=True)) > max_length:
                truncated += 1
        tokens = tokenizer(sample["prompt"], **kwargs).input_ids
        prompts.append(tokens)
    return prompts, truncated


def get_input_device(model) -> torch.device:
    emb = model.get_input_embeddings()
    if emb is not None:
        return emb.weight.device
    return next(model.parameters()).device


def get_layer_rep(output, attention_mask, cls_layer: int):
    hidden_states = output.hidden_states
    if cls_layer == -1:
        hs = hidden_states[-1]
    else:
        hs = hidden_states[cls_layer + 1]
    return get_last_non_padded_token_rep(hs, attention_mask)


@torch.no_grad()
def extract_reps(model, prompts, labels, cls_layer: int, batch_size: int, pad_id: int) -> np.ndarray:
    device = get_input_device(model)
    all_reps = []
    use_cuda_autocast = device.type == "cuda"
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if use_cuda_autocast
        else nullcontext()
    )

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]
        batch_p, _, attention_mask = collate_fn(batch_prompts, batch_labels, pad_id=pad_id)
        batch_p = batch_p.to(device)
        attention_mask = attention_mask.to(device)

        with autocast_ctx:
            output = model.model(
                input_ids=batch_p,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            rep = get_layer_rep(output, attention_mask, cls_layer)
        all_reps.append(rep.detach().cpu().float())

    return torch.cat(all_reps, dim=0).numpy()


def normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, 1e-12, None)


def separability_metrics(x: np.ndarray, y: np.ndarray) -> dict:
    x0 = x[y == 0]
    x1 = x[y == 1]
    c0 = x0.mean(axis=0)
    c1 = x1.mean(axis=0)
    centroid_dist = float(np.linalg.norm(c1 - c0))
    within0 = np.linalg.norm(x0 - c0, axis=1).mean()
    within1 = np.linalg.norm(x1 - c1, axis=1).mean()
    within = float((within0 + within1) / 2.0)
    sep_ratio = float(centroid_dist / (within + 1e-12))
    sil = float(silhouette_score(x, y, metric="euclidean")) if len(set(y.tolist())) == 2 else float("nan")
    return {
        "silhouette": sil,
        "centroid_distance": centroid_dist,
        "within_scatter": within,
        "separation_ratio": sep_ratio,
    }


def plot_model(
    model_key: str,
    display_name: str,
    samples: list[dict],
    clean_reps: np.ndarray,
    csv_reps: np.ndarray,
    ckpt: dict,
    normalize: bool,
) -> tuple[list[dict], list[dict]]:
    labels = np.array([s["label"] for s in samples], dtype=int)
    sources = np.array([s["source"] for s in samples])
    datasets = sorted(set(sources.tolist()))

    clean_for_analysis = normalize_rows(clean_reps) if normalize else clean_reps
    csv_for_analysis = normalize_rows(csv_reps) if normalize else csv_reps

    n_rows = len(datasets)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(7.2, max(2.8 * n_rows, 2.8)),
        squeeze=False,
        constrained_layout=True,
    )

    metric_rows = []
    point_rows = []
    colors = {0: "#D55E00", 1: "#0072B2"}
    names = {0: "Distracting", 1: "Answer-supporting"}

    for row_idx, ds in enumerate(datasets):
        mask = sources == ds
        ds_samples = [sample for sample, keep in zip(samples, mask) if keep]
        y = labels[mask]
        before = clean_for_analysis[mask]
        after = csv_for_analysis[mask]
        joint = np.concatenate([before, after], axis=0)

        pca = PCA(n_components=2, random_state=0)
        coords = pca.fit_transform(joint)
        before_2d = coords[: len(before)]
        after_2d = coords[len(before):]
        explained = pca.explained_variance_ratio_

        before_metrics = separability_metrics(before, y)
        after_metrics = separability_metrics(after, y)
        before_2d_metrics = separability_metrics(before_2d, y)
        after_2d_metrics = separability_metrics(after_2d, y)

        metric_rows.extend([
            {
                "model": model_key,
                "dataset": ds,
                "condition": "clean",
                "n": int(mask.sum()),
                "n_relevant": int((y == 1).sum()),
                "n_distracting": int((y == 0).sum()),
                "str_layer": int(ckpt["str_layer"]),
                "cls_layer": int(ckpt["cls_layer"]),
                "lambda": float(ckpt["lam"]),
                **before_metrics,
                "pca_silhouette": before_2d_metrics["silhouette"],
                "pca_separation_ratio": before_2d_metrics["separation_ratio"],
            },
            {
                "model": model_key,
                "dataset": ds,
                "condition": "csv",
                "n": int(mask.sum()),
                "n_relevant": int((y == 1).sum()),
                "n_distracting": int((y == 0).sum()),
                "str_layer": int(ckpt["str_layer"]),
                "cls_layer": int(ckpt["cls_layer"]),
                "lambda": float(ckpt["lam"]),
                **after_metrics,
                "pca_silhouette": after_2d_metrics["silhouette"],
                "pca_separation_ratio": after_2d_metrics["separation_ratio"],
            },
        ])

        for condition, coords_2d in (("clean", before_2d), ("csv", after_2d)):
            for sample, label, xy in zip(ds_samples, y, coords_2d):
                point_rows.append({
                    "model": model_key,
                    "dataset": ds,
                    "condition": condition,
                    "sample_idx": int(sample["idx"]),
                    "label": int(label),
                    "label_name": "answer_supporting" if int(label) == 1 else "distracting",
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                    "pca1_explained_var": float(explained[0]),
                    "pca2_explained_var": float(explained[1]),
                    "str_layer": int(ckpt["str_layer"]),
                    "cls_layer": int(ckpt["cls_layer"]),
                    "lambda": float(ckpt["lam"]),
                })

        xlim = (
            min(before_2d[:, 0].min(), after_2d[:, 0].min()),
            max(before_2d[:, 0].max(), after_2d[:, 0].max()),
        )
        ylim = (
            min(before_2d[:, 1].min(), after_2d[:, 1].min()),
            max(before_2d[:, 1].max(), after_2d[:, 1].max()),
        )
        xpad = (xlim[1] - xlim[0]) * 0.08 + 1e-6
        ypad = (ylim[1] - ylim[0]) * 0.08 + 1e-6
        xlim = (xlim[0] - xpad, xlim[1] + xpad)
        ylim = (ylim[0] - ypad, ylim[1] + ypad)

        for col_idx, (title, coords_2d, metrics) in enumerate([
            ("Clean", before_2d, before_metrics),
            ("After CSV", after_2d, after_metrics),
        ]):
            ax = axes[row_idx, col_idx]
            for label_id in (0, 1):
                class_mask = y == label_id
                ax.scatter(
                    coords_2d[class_mask, 0],
                    coords_2d[class_mask, 1],
                    s=12,
                    alpha=0.42,
                    c=colors[label_id],
                    edgecolors="none",
                    rasterized=True,
                    label=names[label_id],
                )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xticks([])
            ax.set_yticks([])
            ds_label = "NQ" if ds == "nq" else ds.upper()
            ax.set_title(
                f"{ds_label} · {title}\n"
                f"sil={metrics['silhouette']:.3f}, sep={metrics['separation_ratio']:.3f}",
                fontsize=9,
            )
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(
        f"{display_name}: representation separability before vs. after CSV "
        f"(k={int(ckpt['str_layer'])}, c={int(ckpt['cls_layer'])})",
        fontsize=11,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{model_key}_csv_representation_shift.png"
    pdf_path = OUT_DIR / f"{model_key}_csv_representation_shift.pdf"
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  Saved figure -> {png_path}")
    print(f"  Saved figure -> {pdf_path}")
    return metric_rows, point_rows


def resolve_checkpoint(model_key: str, user_path: str | None) -> Path:
    if user_path:
        path = Path(user_path)
    else:
        info = MODEL_REGISTRY[model_key]
        path = Path(info["checkpoint"])
        if not path.exists() and "fallback_checkpoint" in info:
            path = Path(info["fallback_checkpoint"])
    if not path.exists():
        raise FileNotFoundError(f"CSV checkpoint not found: {path}")
    return path


def process_model(args, model_key: str) -> tuple[list[dict], list[dict]]:
    info = MODEL_REGISTRY[model_key]
    checkpoint_path = resolve_checkpoint(model_key, args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    str_layer = int(ckpt["str_layer"])
    cls_layer = int(ckpt["cls_layer"])
    lam = float(ckpt["lam"])

    print(f"\n{'=' * 72}")
    print(f"Model: {model_key}")
    print(f"HF path: {info['hf_name']}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"CSV config: str_layer={str_layer}, cls_layer={cls_layer}, lambda={lam}")
    print(f"{'=' * 72}")

    datasets = set() if args.dataset == "all" else set(args.dataset.split(","))
    samples = load_eval_samples(datasets=datasets, max_per_class=args.max_per_class, seed=args.seed)
    labels = [s["label"] for s in samples]
    counts = {}
    for s in samples:
        counts[(s["source"], s["label_name"])] = counts.get((s["source"], s["label_name"]), 0) + 1
    print(f"Selected {len(samples)} eval samples: {counts}")

    tokenizer = setup_tokenizer(info["hf_name"])
    prompts, truncated = tokenize_samples(samples, tokenizer, args.max_length)
    if truncated:
        print(f"  Warning: truncated {truncated}/{len(prompts)} prompts at max_length={args.max_length}")

    print("Loading base model ...")
    model = AutoModelForCausalLM.from_pretrained(
        info["hf_name"],
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    batch_size = args.batch_size or info["batch_size"]
    print("Extracting clean representations ...")
    clean_reps = extract_reps(
        model=model,
        prompts=prompts,
        labels=labels,
        cls_layer=cls_layer,
        batch_size=batch_size,
        pad_id=tokenizer.pad_token_id,
    )

    print("Injecting CSV and extracting post-CSV representations ...")
    add_tsv_layers(model, ckpt["tsv"], [lam], str_layer, model_key)
    csv_reps = extract_reps(
        model=model,
        prompts=prompts,
        labels=labels,
        cls_layer=cls_layer,
        batch_size=batch_size,
        pad_id=tokenizer.pad_token_id,
    )

    metric_rows, point_rows = plot_model(
        model_key=model_key,
        display_name=info["display"],
        samples=samples,
        clean_reps=clean_reps,
        csv_reps=csv_reps,
        ckpt=ckpt,
        normalize=args.normalize,
    )

    del model
    torch.cuda.empty_cache()
    return metric_rows, point_rows


def write_metrics(rows: list[dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "csv_representation_shift_metrics.csv"
    fieldnames = [
        "model",
        "dataset",
        "condition",
        "n",
        "n_relevant",
        "n_distracting",
        "str_layer",
        "cls_layer",
        "lambda",
        "silhouette",
        "centroid_distance",
        "within_scatter",
        "separation_ratio",
        "pca_silhouette",
        "pca_separation_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_points(rows: list[dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "csv_representation_shift_points.csv"
    fieldnames = [
        "model",
        "dataset",
        "condition",
        "sample_idx",
        "label",
        "label_name",
        "x",
        "y",
        "pca1_explained_var",
        "pca2_explained_var",
        "str_layer",
        "cls_layer",
        "lambda",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize CSV representation separability on retrieved-doc eval data.")
    parser.add_argument("--model", choices=["gemma2b", "qwen3_4b", "all"], default="all")
    parser.add_argument("--dataset", default="all", help="all, nq, triviaqa, or comma-separated values")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path; only valid for a single --model")
    parser.add_argument("--max_per_class", type=int, default=300, help="Balanced samples per dataset/class")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_normalize", action="store_true", help="Plot/score raw reps instead of L2-normalized reps")
    args = parser.parse_args()
    args.normalize = not args.no_normalize
    if args.model == "all" and args.checkpoint:
        raise ValueError("--checkpoint can only be used with a single --model")
    return args


def main():
    args = parse_args()
    models = ["gemma2b", "qwen3_4b"] if args.model == "all" else [args.model]
    all_metric_rows = []
    all_point_rows = []
    for model_key in models:
        metric_rows, point_rows = process_model(args, model_key)
        all_metric_rows.extend(metric_rows)
        all_point_rows.extend(point_rows)
    metrics_path = write_metrics(all_metric_rows)
    points_path = write_points(all_point_rows)
    print(f"\nSaved metrics -> {metrics_path}")
    print(f"Saved points  -> {points_path}")
    print("Done.")


if __name__ == "__main__":
    main()
