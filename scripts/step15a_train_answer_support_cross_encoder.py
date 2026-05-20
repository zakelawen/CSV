#!/usr/bin/env python
"""
Step 15A: Cross-encoder baseline for utility classification.

This script trains a BERT-style cross-encoder sequence classifier on
either:

  - retrieved-document data: data/final_retrieved/{train,eval}.json
  - gold-vs-distractor data: data/final/{train,eval}.json

  input A: Question: ...
           [Candidate answer: ...]
  input B: Title: ...
           Document: ...

  label: 1 = relevant / answer-supporting
         0 = distracting

The default answer_mode="none" is the fair retrieved-document utility
baseline: it uses only question + document at inference time. The
gold-answer modes are useful as oracle diagnostics for an answer-support
classifier, but should not be reported as a non-oracle pipeline baseline
because they use gold answers at evaluation time.

Related methods:
  - monoBERT-style passage re-ranking: query/document cross-encoder.
  - answer-aware reranking / answer verification: query+answer/document
    cross-encoder.

Retrieved-doc example:
  python scripts/step15a_train_answer_support_cross_encoder.py \
    --model_name cross-encoder/ms-marco-MiniLM-L6-v2 \
    --setting retrieved \
    --answer_mode none \
    --batch_size 16 \
    --epochs 3

Gold-vs-distractor example:
  python scripts/step15a_train_answer_support_cross_encoder.py \
    --model_name cross-encoder/ms-marco-MiniLM-L6-v2 \
    --setting gold \
    --answer_mode none \
    --batch_size 16 \
    --epochs 3
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRS = {
    "retrieved": PROJECT_ROOT / "data" / "final_retrieved",
    "gold": PROJECT_ROOT / "data" / "final",
}
RESULTS_DIRS = {
    "retrieved": PROJECT_ROOT / "results" / "cross_encoder_retrieved",
    "gold": PROJECT_ROOT / "results" / "cross_encoder_gold",
}

SETTINGS = ["retrieved", "gold"]
ANSWER_MODES = ["none", "gold_first", "gold_all_joined"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_doc_text(doc: Any) -> tuple[str, str]:
    if isinstance(doc, str):
        return "", doc
    if isinstance(doc, dict):
        title = str(doc.get("title") or "").strip()
        text = str(doc.get("text") or doc.get("contents") or "").strip()
        return title, text
    return "", str(doc)


def _try_parse_collection_string(s: str) -> Any | None:
    t = s.strip()
    if not t:
        return None
    if not (
        (t.startswith("[") and t.endswith("]"))
        or (t.startswith("(") and t.endswith(")"))
        or (t.startswith("{") and t.endswith("}"))
    ):
        return None
    try:
        return json.loads(t)
    except Exception:
        pass
    try:
        return ast.literal_eval(t)
    except Exception:
        return None


def flatten_answers(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list) or isinstance(x, tuple):
        out: list[str] = []
        for item in x:
            out.extend(flatten_answers(item))
        return out
    if isinstance(x, dict):
        out: list[str] = []
        for key in ("text", "answer", "answers", "aliases"):
            if key in x:
                out.extend(flatten_answers(x[key]))
        return out
    if isinstance(x, str):
        parsed = _try_parse_collection_string(x)
        if parsed is not None:
            return flatten_answers(parsed)
        return [x]
    return [str(x)]


def answer_for_item(item: dict[str, Any], answer_mode: str) -> str:
    if answer_mode == "none":
        return ""

    answers = [a.strip() for a in flatten_answers(item.get("answers")) if str(a).strip()]
    if not answers:
        return ""

    if answer_mode == "gold_first":
        return answers[0]
    if answer_mode == "gold_all_joined":
        # Keep this short because it sits in sequence A and competes with
        # question tokens. Ten aliases is already generous for TriviaQA.
        return "; ".join(answers[:10])

    raise ValueError(f"Unknown answer_mode: {answer_mode}")


def source_name(item: dict[str, Any]) -> str:
    return str(item.get("source_dataset") or item.get("dataset") or "unknown").lower()


@dataclass
class CrossEncoderExample:
    idx: int
    text_a: str
    text_b: str
    label: int
    doc_role: str
    source_dataset: str
    question: str
    answer_text: str
    doc_title: str


def make_example(
    item_idx: int,
    item: dict[str, Any],
    doc: Any,
    label: int,
    doc_role: str,
    answer_mode: str,
) -> CrossEncoderExample:
    if label not in (0, 1):
        raise ValueError(f"Invalid label at item {item_idx}: {label!r}")

    question = str(item["question"]).strip()
    answer_text = answer_for_item(item, answer_mode)
    doc_title, doc_text = extract_doc_text(doc)

    if answer_text:
        text_a = f"Question: {question}\nCandidate answer: {answer_text}"
    else:
        text_a = f"Question: {question}"

    if doc_title:
        text_b = f"Title: {doc_title}\nDocument: {doc_text}"
    else:
        text_b = f"Document: {doc_text}"

    return CrossEncoderExample(
        idx=item_idx,
        text_a=text_a,
        text_b=text_b,
        label=label,
        doc_role=doc_role,
        source_dataset=source_name(item),
        question=question,
        answer_text=answer_text,
        doc_title=doc_title,
    )


def build_examples(
    data: list[dict[str, Any]],
    answer_mode: str,
    setting: str,
) -> list[CrossEncoderExample]:
    examples: list[CrossEncoderExample] = []
    for idx, item in enumerate(data):
        if setting == "retrieved":
            label = int(item.get("label", 1 if item.get("label_name") == "relevant" else 0))
            examples.append(
                make_example(
                    item_idx=idx,
                    item=item,
                    doc=item["doc"],
                    label=label,
                    doc_role=str(item.get("label_name") or label),
                    answer_mode=answer_mode,
                )
            )
        elif setting == "gold":
            examples.append(
                make_example(
                    item_idx=idx * 2,
                    item=item,
                    doc=item["relevant_doc"],
                    label=1,
                    doc_role="relevant_doc",
                    answer_mode=answer_mode,
                )
            )
            examples.append(
                make_example(
                    item_idx=idx * 2 + 1,
                    item=item,
                    doc=item["distracting_doc"],
                    label=0,
                    doc_role="distracting_doc",
                    answer_mode=answer_mode,
                )
            )
        else:
            raise ValueError(f"Unknown setting: {setting}")
    return examples


class CrossEncoderDataset(Dataset):
    def __init__(self, examples: list[CrossEncoderExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> CrossEncoderExample:
        return self.examples[idx]


class Collator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[CrossEncoderExample]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [x.text_a for x in batch],
            [x.text_b for x in batch],
            padding=True,
            truncation="only_second",
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([x.label for x in batch], dtype=torch.long)
        encoded["idx"] = torch.tensor([x.idx for x in batch], dtype=torch.long)
        return encoded


def model_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1 or logits.shape[-1] == 1:
        return torch.sigmoid(logits.reshape(-1))
    return torch.softmax(logits, dim=-1)[:, 1]


@torch.no_grad()
def predict(
    model,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []

    for batch in loader:
        indices = batch.pop("idx").numpy()
        labels = batch["labels"].numpy()
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        probs = model_probs_from_logits(output.logits).detach().cpu().numpy()

        all_indices.append(indices)
        all_labels.append(labels)
        all_probs.append(probs)

    return (
        np.concatenate(all_indices),
        np.concatenate(all_labels),
        np.concatenate(all_probs),
    )


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
    examples: list[CrossEncoderExample],
    labels: np.ndarray,
    probs: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    sources = np.array([x.source_dataset for x in examples])
    out: dict[str, dict[str, float | int]] = {
        "combined": metric_block(labels, probs)
    }
    for src in sorted(set(sources.tolist())):
        mask = sources == src
        if mask.sum() == 0:
            continue
        out[src] = metric_block(labels[mask], probs[mask])
    return out


def sanitize_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s.strip())
    return s.strip("_") or "model"


def write_scores(
    path: Path,
    examples: list[CrossEncoderExample],
    labels: np.ndarray,
    probs: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "source_dataset",
                "label",
                "doc_role",
                "p_relevant",
                "pred",
                "question",
                "candidate_answer",
                "doc_title",
            ],
        )
        writer.writeheader()
        for ex, label, prob in zip(examples, labels, probs):
            writer.writerow(
                {
                    "idx": ex.idx,
                    "source_dataset": ex.source_dataset,
                    "label": int(label),
                    "doc_role": ex.doc_role,
                    "p_relevant": float(prob),
                    "pred": int(prob >= 0.5),
                    "question": ex.question,
                    "candidate_answer": ex.answer_text,
                    "doc_title": ex.doc_title,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a cross-encoder retrieved-document utility baseline."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L6-v2",
        help="HF encoder/cross-encoder checkpoint.",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="retrieved",
        choices=SETTINGS,
        help="retrieved = data/final_retrieved doc/label; gold = data/final relevant_doc vs distracting_doc.",
    )
    parser.add_argument(
        "--answer_mode",
        type=str,
        default="none",
        choices=ANSWER_MODES,
        help=(
            "none = fair question-document classifier; gold_* = oracle "
            "answer-support diagnostic using gold answers."
        ),
    )
    parser.add_argument("--train_path", type=Path, default=None)
    parser.add_argument("--eval_path", type=Path, default=None)
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=None,
        help="Override output directory. Default is setting-specific.",
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_eval", type=int, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Do not try to download from Hugging Face; require local cache/path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    eval_batch_size = args.eval_batch_size or args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.fp16 and device.type == "cuda")

    train_path = args.train_path or (DATA_DIRS[args.setting] / "train.json")
    eval_path = args.eval_path or (DATA_DIRS[args.setting] / "eval.json")
    results_dir = args.results_dir or RESULTS_DIRS[args.setting]

    train_data = load_json(train_path)
    eval_data = load_json(eval_path)
    if args.limit_train is not None:
        train_data = train_data[: args.limit_train]
    if args.limit_eval is not None:
        eval_data = eval_data[: args.limit_eval]

    train_examples = build_examples(train_data, args.answer_mode, args.setting)
    eval_examples = build_examples(eval_data, args.answer_mode, args.setting)

    print("=== Step15A: cross-encoder utility baseline ===")
    print(f"model_name:  {args.model_name}")
    print(f"setting:     {args.setting}")
    print(f"answer_mode: {args.answer_mode}")
    print(f"train:       {train_path}  n={len(train_examples)}")
    print(f"eval:        {eval_path}  n={len(eval_examples)}")
    if args.answer_mode.startswith("gold"):
        print("NOTE: gold_* answer modes are oracle diagnostics because they use gold answers.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            use_fast=True,
            local_files_only=args.local_files_only,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
            ignore_mismatched_sizes=True,
            local_files_only=args.local_files_only,
        )
    except OSError as exc:
        raise SystemExit(
            "\nCould not load the cross-encoder checkpoint.\n"
            f"  model_name={args.model_name}\n"
            "This usually means the model is not cached locally and the server "
            "cannot reach Hugging Face / hf-mirror.\n"
            "Fix options:\n"
            "  1. Pass a local HF snapshot path with --model_name /path/to/model.\n"
            "  2. Download the checkpoint first on a machine with network access.\n"
            "  3. If you only want local cache, rerun with --local_files_only.\n"
            f"\nOriginal error:\n{exc}"
        ) from exc
    if tokenizer.pad_token is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    collator = Collator(tokenizer=tokenizer, max_length=args.max_length)
    train_loader = DataLoader(
        CrossEncoderDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        CrossEncoderDataset(eval_examples),
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = int(math.ceil(args.warmup_ratio * total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_name = args.run_name
    if run_name is None:
        run_name = f"{sanitize_name(args.model_name)}_{args.setting}_{args.answer_mode}"
    out_dir = results_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    best_epoch = -1
    best_metrics: dict[str, dict[str, float | int]] | None = None
    best_probs: np.ndarray | None = None
    best_labels: np.ndarray | None = None
    best_auroc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for batch in train_loader:
            batch.pop("idx")
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(**batch)
                loss = output.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bsz = int(batch["labels"].shape[0])
            running_loss += float(loss.detach().cpu()) * bsz
            seen += bsz

        _, eval_labels, eval_probs = predict(model, eval_loader, device)
        metrics = compute_metrics(eval_examples, eval_labels, eval_probs)
        combined_auroc = float(metrics["combined"]["auroc"])
        print(
            f"epoch {epoch:02d}  "
            f"train_loss={running_loss / max(1, seen):.4f}  "
            f"AUROC={combined_auroc:.4f}  "
            f"Acc={float(metrics['combined']['accuracy']):.4f}"
        )

        if combined_auroc > best_auroc:
            best_epoch = epoch
            best_auroc = combined_auroc
            best_metrics = metrics
            best_probs = eval_probs.copy()
            best_labels = eval_labels.copy()
            if args.save_model:
                model.save_pretrained(out_dir / "best_model")
                tokenizer.save_pretrained(out_dir / "best_model")

    assert best_metrics is not None
    assert best_probs is not None
    assert best_labels is not None

    scores_path = out_dir / "eval_scores.csv"
    write_scores(scores_path, eval_examples, best_labels, best_probs)

    result = {
        "meta": {
            "script": Path(__file__).name,
            "model_name": args.model_name,
            "setting": args.setting,
            "answer_mode": args.answer_mode,
            "oracle_answer_mode": args.answer_mode.startswith("gold"),
            "train_path": str(train_path),
            "eval_path": str(eval_path),
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "eval_batch_size": eval_batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
            "local_files_only": args.local_files_only,
            "threshold": 0.5,
        },
        "best_epoch": best_epoch,
        "metrics": best_metrics,
        "scores_path": str(scores_path),
    }

    result_path = out_dir / "results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved results: {result_path}")
    print(f"Saved scores:  {scores_path}")


if __name__ == "__main__":
    main()
