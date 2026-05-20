#!/usr/bin/env python
"""
Plot original retrieved-document representations before CSV intervention.

This script makes one figure for each model x dataset pair:
  Gemma2-2B / Qwen3-4B x NQ / TriviaQA = 4 figures by default.

It intentionally does not run the CSV-wrapped model. It reads cached clean
hidden states from data/hidden_states_retrieved and visualizes only the
original representations at the classification layer used by the best CSV
checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HIDDEN_DIR = PROJECT_ROOT / "data" / "hidden_states_retrieved"
OUT_DIR = PROJECT_ROOT / "results" / "case_analysis" / "original_representation_overlap"


MODEL_INFO = {
    "gemma2b": {
        "display": "Gemma2-2B",
        "cls_layer": 18,
    },
    "qwen3_4b": {
        "display": "Qwen3-4B",
        "cls_layer": 22,
    },
}

DATASET_LABELS = {
    "nq": "NQ",
    "triviaqa": "TriviaQA",
}

COLORS = {
    0: "#D8AAA8",  # distracting
    1: "#6EA9C9",  # answer-supporting
}
BACKGROUND = "#FFF7FD"


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, 1e-12, None)


def labels_to_ids(labels: list[str]) -> np.ndarray:
    ids = []
    for label in labels:
        name = str(label).lower().strip()
        if name == "relevant":
            ids.append(1)
        elif name == "distracting":
            ids.append(0)
        else:
            raise ValueError(f"Unknown label: {label!r}")
    return np.asarray(ids, dtype=int)


def balanced_indices(y: np.ndarray, max_per_class: int | None, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        if max_per_class is not None and len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        selected.append(idx)
    out = np.concatenate(selected)
    return out[np.argsort(out)]


def cache_layer_for_cls(cls_layer: int, num_hidden_entries: int) -> int:
    if cls_layer == -1:
        return num_hidden_entries - 1
    layer_idx = cls_layer + 1
    if layer_idx < 0 or layer_idx >= num_hidden_entries:
        raise ValueError(
            f"cls_layer={cls_layer} maps to cached hidden_states[{layer_idx}], "
            f"but file has {num_hidden_entries} hidden-state entries."
        )
    return layer_idx


def ellipse_params(points: np.ndarray, n_std: float) -> tuple[float, float, float, float, float] | None:
    if len(points) < 3:
        return None
    cov = np.cov(points, rowvar=False)
    if not np.isfinite(cov).all():
        return None
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = np.clip(vals[order], 0.0, None)
    vecs = vecs[:, order]
    if np.all(vals <= 1e-12):
        return None
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    center = points.mean(axis=0)
    width, height = 2.0 * n_std * np.sqrt(vals)
    return float(center[0]), float(center[1]), float(width), float(height), float(angle)


def separation_metrics(coords: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x0 = coords[y == 0]
    x1 = coords[y == 1]
    c0 = x0.mean(axis=0)
    c1 = x1.mean(axis=0)
    centroid_distance = float(np.linalg.norm(c1 - c0))
    within0 = np.linalg.norm(x0 - c0, axis=1).mean()
    within1 = np.linalg.norm(x1 - c1, axis=1).mean()
    within_scatter = float((within0 + within1) / 2.0)
    silhouette = float(silhouette_score(coords, y)) if len(set(y.tolist())) == 2 else float("nan")
    return {
        "silhouette": silhouette,
        "centroid_distance": centroid_distance,
        "within_scatter": within_scatter,
        "separation_ratio": float(centroid_distance / (within_scatter + 1e-12)),
    }


def plot_one(
    coords: np.ndarray,
    y: np.ndarray,
    out_stem: Path,
    formats: list[str],
    title: str | None,
    point_size: float,
    point_alpha: float,
    ellipse_std: float,
) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.05))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    for label_id in (0, 1):
        pts = coords[y == label_id]
        if len(pts) == 0:
            continue
        params = ellipse_params(pts, ellipse_std)
        if params is not None:
            cx, cy, width, height, angle = params
            ax.add_patch(
                Ellipse(
                    (cx, cy),
                    width=width,
                    height=height,
                    angle=angle,
                    facecolor=COLORS[label_id],
                    edgecolor=COLORS[label_id],
                    linewidth=3.0,
                    alpha=0.20,
                    zorder=1,
                )
            )
            ax.add_patch(
                Ellipse(
                    (cx, cy),
                    width=width,
                    height=height,
                    angle=angle,
                    facecolor="none",
                    edgecolor=COLORS[label_id],
                    linewidth=3.0,
                    alpha=0.62,
                    zorder=2,
                )
            )

    for label_id in (0, 1):
        pts = coords[y == label_id]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=point_size,
            c=COLORS[label_id],
            alpha=point_alpha,
            edgecolors="none",
            zorder=3,
        )

    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    xpad = (xmax - xmin) * 0.10 + 1e-6
    ypad = (ymax - ymin) * 0.10 + 1e-6
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, pad=8)

    fig.tight_layout(pad=0.05)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_path = out_stem.with_suffix(f".{fmt}")
        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if fmt.lower() == "png":
            save_kwargs["dpi"] = 300
        fig.savefig(out_path, **save_kwargs)
        print(f"  Saved -> {out_path}")
    plt.close(fig)


def process_model(args: argparse.Namespace, model_key: str) -> list[dict[str, object]]:
    info = MODEL_INFO[model_key]
    hs_path = HIDDEN_DIR / model_key / f"{args.split}_hidden_states.pt"
    if not hs_path.exists():
        raise FileNotFoundError(f"Hidden-state file not found: {hs_path}")

    print(f"\nProcessing {model_key}: {hs_path}")
    data = torch.load(hs_path, map_location="cpu", weights_only=False)
    hidden_states = data["hidden_states"]
    labels = labels_to_ids(data["labels"])
    sources = np.asarray(data["dataset_source"])
    layer_idx = cache_layer_for_cls(int(info["cls_layer"]), hidden_states.shape[1])

    metrics_rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        ds_mask = sources == dataset
        if not ds_mask.any():
            print(f"  [skip] no samples for dataset={dataset}")
            continue

        ds_positions = np.flatnonzero(ds_mask)
        ds_labels = labels[ds_mask]
        chosen_rel = balanced_indices(ds_labels, args.max_per_class, args.seed)
        chosen = ds_positions[chosen_rel]
        y = labels[chosen]

        reps = hidden_states[chosen, layer_idx, :].float().numpy()
        if args.normalize:
            reps = normalize_rows(reps)
        coords = PCA(n_components=2, random_state=args.seed).fit_transform(reps)
        metrics = separation_metrics(coords, y)

        ds_label = DATASET_LABELS.get(dataset, dataset.upper())
        title = f"{info['display']} / {ds_label}" if args.with_title else None
        stem = OUT_DIR / f"{model_key}_{dataset}_original_representation_overlap"
        plot_one(
            coords=coords,
            y=y,
            out_stem=stem,
            formats=args.formats,
            title=title,
            point_size=args.point_size,
            point_alpha=args.point_alpha,
            ellipse_std=args.ellipse_std,
        )

        row = {
            "model": model_key,
            "model_display": info["display"],
            "dataset": dataset,
            "split": args.split,
            "cls_layer": int(info["cls_layer"]),
            "cached_hidden_state_index": int(layer_idx),
            "n": int(len(y)),
            "n_distracting": int((y == 0).sum()),
            "n_relevant": int((y == 1).sum()),
            **metrics,
        }
        metrics_rows.append(row)
        print(
            "  "
            f"{model_key}/{dataset}: n={row['n']} "
            f"sil={metrics['silhouette']:.3f} sep={metrics['separation_ratio']:.3f}"
        )

    del data, hidden_states
    gc.collect()
    return metrics_rows


def write_metrics(rows: list[dict[str, object]]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "original_representation_overlap_metrics.csv"
    fieldnames = [
        "model",
        "model_display",
        "dataset",
        "split",
        "cls_layer",
        "cached_hidden_state_index",
        "n",
        "n_distracting",
        "n_relevant",
        "silhouette",
        "centroid_distance",
        "within_scatter",
        "separation_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot clean/original retrieved-doc representation overlap."
    )
    parser.add_argument("--models", default="gemma2b,qwen3_4b")
    parser.add_argument("--datasets", default="nq,triviaqa")
    parser.add_argument("--split", default="eval", choices=["train", "eval"])
    parser.add_argument("--max_per_class", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--formats", default="png,svg,pdf")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--with_title", action="store_true")
    parser.add_argument("--point_size", type=float, default=62.0)
    parser.add_argument("--point_alpha", type=float, default=0.58)
    parser.add_argument("--ellipse_std", type=float, default=1.65)
    args = parser.parse_args()
    args.models = parse_csv(args.models)
    args.datasets = parse_csv(args.datasets)
    args.formats = parse_csv(args.formats)
    args.normalize = not args.no_normalize

    unknown_models = [m for m in args.models if m not in MODEL_INFO]
    if unknown_models:
        raise ValueError(f"Unknown model(s): {unknown_models}. Available: {sorted(MODEL_INFO)}")
    return args


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for model_key in args.models:
        all_rows.extend(process_model(args, model_key))
    metrics_path = write_metrics(all_rows)
    print(f"\nSaved metrics -> {metrics_path}")
    print("Done.")


if __name__ == "__main__":
    main()
