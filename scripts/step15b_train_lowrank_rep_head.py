#!/usr/bin/env python
"""
Step 15B: Low-rank representation-head baseline on frozen hidden states.

This is a lightweight ReFT/LoRA-style baseline for utility classification.
It does not modify the backbone LLM. Instead, it uses already extracted hidden
states and trains a small low-rank residual adapter plus a classifier:

    z = h + (alpha / rank) * B(A h)
    logits = W z + b

where h is a selected frozen representation, typically the final-token
representation from the final layer. This is not a full LoRA or full LoReFT
implementation; it is the controlled "small low-rank head on final
representation" baseline.

Example:
  python scripts/step15b_train_lowrank_rep_head.py \
    --model gemma2b \
    --setting retrieved \
    --layer -1 \
    --rank 8 \
    --alpha 16 \
    --epochs 30

Run both gold-vs-distractor and retrieved-document settings:
  python scripts/step15b_train_lowrank_rep_head.py --model all --setting all --layer -1
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HIDDEN_DIRS = {
    "gold": PROJECT_ROOT / "data" / "hidden_states",
    "retrieved": PROJECT_ROOT / "data" / "hidden_states_retrieved",
}
RESULTS_DIRS = {
    "gold": PROJECT_ROOT / "results" / "lowrank_rep_head_gold",
    "retrieved": PROJECT_ROOT / "results" / "lowrank_rep_head_retrieved",
}
MODEL_KEYS = ["gemma2b", "gemma9b", "qwen3_4b"]
SETTINGS = ["gold", "retrieved"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metric_block(labels: np.ndarray, probs: np.ndarray) -> dict[str, float | int]:
    preds = (probs >= 0.5).astype(np.int64)
    return {
        "n": int(labels.shape[0]),
        "auroc": float(roc_auc_score(labels, probs)),
        "accuracy": float(accuracy_score(labels, preds)),
        "positive_rate": float(labels.mean()),
        "predicted_positive_rate": float(preds.mean()),
        "p_relevant_mean": float(probs.mean()),
    }


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    sources: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    out = {"combined": metric_block(labels, probs)}
    for src in sorted(set(sources.tolist())):
        mask = sources == src
        if mask.sum() == 0:
            continue
        out[src] = metric_block(labels[mask], probs[mask])
    return out


class LowRankRepHead(nn.Module):
    def __init__(self, dim: int, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.classifier = nn.Linear(dim, 2)

        # LoRA-style initialization: the residual adapter starts as a no-op.
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.down(self.dropout(x))) * self.scale
        z = x + delta
        return self.classifier(self.dropout(z))


def layer_label(layer_arg: int, resolved_layer: int, num_layers_in_file: int) -> str:
    if layer_arg == -1 or resolved_layer == num_layers_in_file - 1:
        return "final"
    return f"layer_{resolved_layer}"


def resolve_layer(layer_arg: int, num_layers_in_file: int) -> int:
    if layer_arg < 0:
        resolved = num_layers_in_file + layer_arg
    else:
        resolved = layer_arg
    if resolved < 0 or resolved >= num_layers_in_file:
        raise ValueError(
            f"Invalid layer {layer_arg}; hidden state file has {num_layers_in_file} layers"
        )
    return resolved


def load_split(
    model_key: str,
    setting: str,
    split: str,
    layer_arg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    path = HIDDEN_DIRS[setting] / model_key / f"{split}_hidden_states.pt"
    if not path.exists():
        prep = (
            "step6_extract_hidden_states_retrieved.py"
            if setting == "retrieved"
            else "step6_extract_hidden_states.py"
        )
        raise FileNotFoundError(f"Missing {path}. Run {prep} first.")
    data = torch.load(path, map_location="cpu", weights_only=False)
    hidden = data["hidden_states"]
    if isinstance(hidden, torch.Tensor):
        hidden_np = hidden.float().numpy()
    else:
        hidden_np = np.asarray(hidden, dtype=np.float32)

    resolved_layer = resolve_layer(layer_arg, hidden_np.shape[1])
    x = hidden_np[:, resolved_layer, :].astype(np.float32)
    labels = np.array([1 if x == "relevant" else 0 for x in data["labels"]], dtype=np.int64)
    sources = np.array([str(x).lower() for x in data["dataset_source"]])
    return x, labels, sources, resolved_layer, int(hidden_np.shape[1])


def standardize_train_eval(
    train_x: np.ndarray,
    eval_x: np.ndarray,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    if not enabled:
        return train_x, eval_x, {"enabled": False}, {}
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return (
        ((train_x - mean) / std).astype(np.float32),
        ((eval_x - mean) / std).astype(np.float32),
        {
            "enabled": True,
            "mean_shape": list(mean.shape),
            "std_min": float(std.min()),
            "std_max": float(std.max()),
        },
        {
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        },
    )


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        probs.append(p)
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def write_scores(
    path: Path,
    labels: np.ndarray,
    probs: np.ndarray,
    sources: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["idx", "source_dataset", "label", "p_relevant", "pred"],
        )
        writer.writeheader()
        for idx, (label, prob, src) in enumerate(zip(labels, probs, sources)):
            writer.writerow(
                {
                    "idx": idx,
                    "source_dataset": src,
                    "label": int(label),
                    "p_relevant": float(prob),
                    "pred": int(prob >= 0.5),
                }
            )


def train_model(model_key: str, setting: str, args: argparse.Namespace) -> dict[str, Any] | None:
    try:
        train_x, train_y, train_sources, resolved_layer, train_num_layers = load_split(
            model_key, setting, "train", args.layer
        )
        eval_x, eval_y, eval_sources, eval_resolved_layer, eval_num_layers = load_split(
            model_key, setting, "eval", args.layer
        )
    except FileNotFoundError as exc:
        print(f"[skip] {model_key}: {exc}")
        return None

    if eval_resolved_layer != resolved_layer:
        raise RuntimeError("Train/eval resolved to different layers")
    if eval_num_layers != train_num_layers:
        raise RuntimeError("Train/eval hidden-state files have different layer counts")

    if args.limit_train is not None:
        train_x = train_x[: args.limit_train]
        train_y = train_y[: args.limit_train]
        train_sources = train_sources[: args.limit_train]
    if args.limit_eval is not None:
        eval_x = eval_x[: args.limit_eval]
        eval_y = eval_y[: args.limit_eval]
        eval_sources = eval_sources[: args.limit_eval]

    train_x, eval_x, scaler_info, scaler_state = standardize_train_eval(
        train_x, eval_x, enabled=not args.no_standardize
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = int(train_x.shape[1])
    layer_name = layer_label(args.layer, resolved_layer, train_num_layers)
    run_name = args.run_name or f"{model_key}_{setting}_{layer_name}_rank{args.rank}_alpha{str(args.alpha).replace('.', 'p')}"
    results_dir = args.results_dir or RESULTS_DIRS[setting]
    out_dir = results_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Step15B: low-rank representation head ===")
    print(f"model:       {model_key}")
    print(f"setting:     {setting}")
    print(f"layer:       arg={args.layer}, resolved={resolved_layer}")
    print(f"train/eval:  {len(train_x)} / {len(eval_x)}")
    print(f"dim/rank:    {dim} / {args.rank}")
    print(f"output dir:  {out_dir}")

    train_ds = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.long),
    )
    eval_ds = TensorDataset(
        torch.tensor(eval_x, dtype=torch.float32),
        torch.tensor(eval_y, dtype=torch.long),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size or args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = LowRankRepHead(
        dim=dim,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    best_epoch = -1
    best_auroc = -1.0
    best_metrics: dict[str, dict[str, float | int]] | None = None
    best_probs: np.ndarray | None = None
    best_labels: np.ndarray | None = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            bsz = int(y.shape[0])
            total_loss += float(loss.detach().cpu()) * bsz
            seen += bsz

        labels, probs = predict(model, eval_loader, device)
        metrics = compute_metrics(labels, probs, eval_sources)
        auroc = float(metrics["combined"]["auroc"])
        acc = float(metrics["combined"]["accuracy"])
        print(
            f"epoch {epoch:02d}  "
            f"train_loss={total_loss / max(1, seen):.4f}  "
            f"AUROC={auroc:.4f}  Acc={acc:.4f}"
        )

        if auroc > best_auroc:
            best_epoch = epoch
            best_auroc = auroc
            best_metrics = metrics
            best_probs = probs.copy()
            best_labels = labels.copy()
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_key": model_key,
                    "layer_arg": args.layer,
                    "resolved_layer": resolved_layer,
                    "rank": args.rank,
                    "alpha": args.alpha,
                    "dropout": args.dropout,
                    "scaler_info": scaler_info,
                    "scaler_state": scaler_state,
                },
                out_dir / "best_head.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop at epoch {epoch} (patience={args.patience})")
                break

    assert best_metrics is not None
    assert best_probs is not None
    assert best_labels is not None

    scores_path = out_dir / "eval_scores.csv"
    write_scores(scores_path, best_labels, best_probs, eval_sources)

    result = {
        "meta": {
            "script": Path(__file__).name,
            "model": model_key,
            "setting": setting,
            "hidden_dir": str(HIDDEN_DIRS[setting] / model_key),
            "layer_arg": args.layer,
            "resolved_layer": resolved_layer,
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size or args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "threshold": 0.5,
            "standardize": scaler_info,
            "note": (
                "Low-rank residual adapter on frozen hidden states; not a full "
                "LoRA/LoReFT backbone finetune."
            ),
        },
        "best_epoch": best_epoch,
        "metrics": best_metrics,
        "scores_path": str(scores_path),
        "checkpoint_path": str(out_dir / "best_head.pt"),
    }

    result_path = out_dir / "results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved results: {result_path}")
    print(f"Saved scores:  {scores_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a low-rank representation-head retrieved-doc baseline."
    )
    parser.add_argument("--model", type=str, default="all", choices=MODEL_KEYS + ["all"])
    parser.add_argument(
        "--setting",
        type=str,
        default="retrieved",
        choices=SETTINGS + ["all"],
        help="gold = gold-vs-distractor hidden states; retrieved = pipeline-matched retrieved-doc hidden states.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Hidden-state index. -1 = final hidden representation.",
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_eval", type=int, default=None)
    parser.add_argument("--no_standardize", action="store_true")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=None,
        help="Override output directory. By default this is setting-specific.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Only use with --model != all unless you intentionally want shared output dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    models = MODEL_KEYS if args.model == "all" else [args.model]
    settings = SETTINGS if args.setting == "all" else [args.setting]
    all_results = {}
    for setting in settings:
        for model_key in models:
            result = train_model(model_key, setting, args)
            if result is not None:
                all_results[f"{setting}/{model_key}"] = result

    if len(all_results) > 1:
        if args.results_dir is not None:
            summary_dir = args.results_dir
        elif len(settings) == 1:
            summary_dir = RESULTS_DIRS[settings[0]]
        else:
            summary_dir = PROJECT_ROOT / "results" / "lowrank_rep_head"
        summary_path = summary_dir / "summary_latest.json"
        summary_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved multi-model summary: {summary_path}")


if __name__ == "__main__":
    main()
