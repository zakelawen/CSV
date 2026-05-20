"""
Step 8 Baseline: CRAG Retrieval Evaluator

Uses the T5-based retrieval evaluator from CRAG (Yan et al., 2024) to classify
retrieved documents as relevant vs. distracting. This serves as a baseline
to compare against the linear probe approach.

The evaluator takes "question [SEP] passage" as input and outputs a scalar score.
Score > threshold → relevant, otherwise → distracting.

Modes:
  retrieved:
    Input:  data/final_retrieved/{train,eval}.json
    Output: results/probe_retrieved/crag_baseline_results.json

  gold_distractor:
    Input:  data/final/{train,eval}.json
    Output: results/probe/crag_gold_distractor_baseline_results.json

Usage:
    python scripts/step8_crag_baseline.py --mode retrieved
    python scripts/step8_crag_baseline.py --mode gold_distractor
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import T5Tokenizer, T5ForSequenceClassification
from sklearn.metrics import roc_auc_score, accuracy_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def extract_doc_text(doc):
    """Extract text from repo document schemas."""
    if isinstance(doc, dict):
        title = str(doc.get("title", "")).strip()
        text = str(doc.get("text", "")).strip()
        if title and text:
            return f"Title: {title}\nText: {text}"
        return text or title
    return str(doc).strip()


def load_retrieved_data(split: str):
    """Load Step4b retrieved-doc data: one retrieved doc per item."""
    path = PROJECT_ROOT / "data" / "final_retrieved" / f"{split}.json"
    with open(path, "r") as f:
        items = json.load(f)

    questions = [item["question"] for item in items]
    passages = [extract_doc_text(item["doc"]) for item in items]
    labels = np.array([item["label"] for item in items])  # 0=distracting, 1=relevant
    sources = [item["source_dataset"] for item in items]
    return questions, passages, labels, sources


def load_gold_distractor_data(split: str):
    """Load old Step4 data and expand each item into gold/distractor pairs."""
    path = PROJECT_ROOT / "data" / "final" / f"{split}.json"
    with open(path, "r") as f:
        items = json.load(f)

    questions = []
    passages = []
    labels = []
    sources = []

    for item in items:
        question = item["question"]
        source = item.get("source_dataset", "unknown")

        questions.append(question)
        passages.append(extract_doc_text(item["relevant_doc"]))
        labels.append(1)
        sources.append(source)

        questions.append(question)
        passages.append(extract_doc_text(item["distracting_doc"]))
        labels.append(0)
        sources.append(source)

    return questions, passages, np.array(labels), sources


def load_data(split: str, mode: str):
    if mode == "retrieved":
        return load_retrieved_data(split)
    if mode == "gold_distractor":
        return load_gold_distractor_data(split)
    raise ValueError(f"Unknown mode: {mode}")


def score_all(questions, passages, tokenizer, model, device, batch_size=32):
    """Score all question-passage pairs with the CRAG evaluator."""
    model.eval()
    all_scores = []

    for i in tqdm(range(0, len(questions), batch_size), desc="Scoring"):
        batch_q = questions[i:i + batch_size]
        batch_p = passages[i:i + batch_size]
        inputs_text = [q + " [SEP] " + p for q, p in zip(batch_q, batch_p)]

        inputs = tokenizer(
            inputs_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = model(
                inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
            )
            scores = outputs["logits"].squeeze(-1).cpu().numpy()

        all_scores.append(scores)

    return np.concatenate(all_scores)


def predict_from_scores(scores, threshold=0.0, direction="greater"):
    """Convert scalar evaluator scores to 0/1 predictions."""
    if direction == "greater":
        return (scores > threshold).astype(int)
    if direction == "less":
        return (scores < threshold).astype(int)
    raise ValueError(f"Unknown direction: {direction}")


def orient_scores(scores, direction="greater"):
    """Orient scores so higher means more relevant for AUROC reporting."""
    if direction == "greater":
        return scores
    if direction == "less":
        return -scores
    raise ValueError(f"Unknown direction: {direction}")


def evaluate(scores, labels, sources, threshold=0.0, direction="greater"):
    """Compute AUROC, accuracy, avg_margin on eval set, per-dataset and combined."""
    sources = np.array(sources)
    datasets = sorted(set(sources))

    results = {}

    for subset_name in datasets + ["combined"]:
        if subset_name == "combined":
            mask = np.ones(len(labels), dtype=bool)
        else:
            mask = sources == subset_name

        s = scores[mask]
        l = labels[mask]

        preds = predict_from_scores(s, threshold, direction)
        auroc = roc_auc_score(l, orient_scores(s, direction))
        acc = accuracy_score(l, preds)
        margin = float(np.abs(s).mean())

        results[subset_name] = {
            "auroc": float(auroc),
            "accuracy": float(acc),
            "avg_margin": float(margin),
            "n_samples": int(mask.sum()),
            "n_relevant": int(l.sum()),
            "n_distracting": int((1 - l).sum()),
        }

    return results


def find_best_threshold(scores, labels):
    """Search train split for threshold and score direction that maximize accuracy."""
    best_acc = 0.0
    best_t = 0.0
    best_direction = "greater"
    for t in np.arange(scores.min() - 0.1, scores.max() + 0.1, 0.01):
        for direction in ("greater", "less"):
            preds = predict_from_scores(scores, t, direction)
            acc = accuracy_score(labels, preds)
            if acc > best_acc:
                best_acc = acc
                best_t = t
                best_direction = direction
    return float(best_t), float(best_acc), best_direction


def parse_args():
    parser = argparse.ArgumentParser(description="CRAG retrieval evaluator baseline")
    parser.add_argument(
        "--mode",
        choices=["retrieved", "gold_distractor"],
        default="retrieved",
        help="retrieved = data/final_retrieved; gold_distractor = data/final expanded to rel/dis pairs",
    )
    parser.add_argument(
        "--evaluator_path",
        default=os.environ.get(
            "CRAG_EVALUATOR_PATH",
            str(PROJECT_ROOT / "CRAG" / "model"),
        ),
        help="Local CRAG T5 sequence-classification evaluator path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed threshold. Default: search best threshold on train split.",
    )
    parser.add_argument(
        "--direction",
        choices=["greater", "less"],
        default=None,
        help="Fixed score direction. Default: search greater/less on train split.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def output_path_for_mode(mode: str) -> Path:
    if mode == "retrieved":
        return PROJECT_ROOT / "results" / "probe_retrieved" / "crag_baseline_results.json"
    if mode == "gold_distractor":
        return PROJECT_ROOT / "results" / "probe" / "crag_gold_distractor_baseline_results.json"
    raise ValueError(f"Unknown mode: {mode}")


def main():
    args = parse_args()
    out_path = output_path_for_mode(args.mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading CRAG evaluator from {args.evaluator_path} ...")
    tokenizer = T5Tokenizer.from_pretrained(args.evaluator_path)
    model = T5ForSequenceClassification.from_pretrained(args.evaluator_path, num_labels=1)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  Device: {device}")

    # Load data
    print("Loading data ...")
    train_q, train_p, train_labels, train_sources = load_data("train", args.mode)
    eval_q, eval_p, eval_labels, eval_sources = load_data("eval", args.mode)
    print(f"  Mode: {args.mode}")
    print(f"  Train: {len(train_q)}, Eval: {len(eval_q)}")

    # Score
    print("Scoring train set ...")
    train_scores = score_all(train_q, train_p, tokenizer, model, device, args.batch_size)
    print("Scoring eval set ...")
    eval_scores = score_all(eval_q, eval_p, tokenizer, model, device, args.batch_size)

    # Determine threshold
    if args.threshold is not None:
        threshold = args.threshold
        direction = args.direction or "greater"
        print(f"Using fixed threshold: {threshold}")
        print(f"Using fixed direction: {direction}")
    else:
        threshold, train_acc, direction = find_best_threshold(train_scores, train_labels)
        print(
            f"Best threshold from train set: {threshold:.4f} "
            f"(direction={direction}, train acc: {train_acc:.4f})"
        )

    # Evaluate on eval set
    print("\n=== Eval Results ===")
    eval_results = evaluate(eval_scores, eval_labels, eval_sources, threshold, direction)

    for subset, metrics in eval_results.items():
        print(f"  {subset:12s}: AUROC={metrics['auroc']:.4f}  "
              f"Acc={metrics['accuracy']:.4f}  "
              f"Margin={metrics['avg_margin']:.4f}  "
              f"(n={metrics['n_samples']})")

    # Also evaluate on train set for reference
    train_results = evaluate(train_scores, train_labels, train_sources, threshold, direction)

    # Save
    output = {
        "method": "crag_retrieval_evaluator",
        "mode": args.mode,
        "evaluator_path": args.evaluator_path,
        "threshold": threshold,
        "direction": direction,
        "eval": eval_results,
        "train": train_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results → {out_path}")


if __name__ == "__main__":
    main()
