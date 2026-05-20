"""
Step 8b: DPR Similarity Baseline for retrieved-document relevance.

For each retrieved (question, doc, label) sample, use a DPR retrieval score as
the predicted score for binary classification.

By default this script uses the stored doc["score"], which is the DPR/FAISS
score saved by scripts/step2_retrieve.py. Pass --score_source embedding to
recompute a cosine score with the DPR encoders instead.

This is the retrieved-doc counterpart of scripts/step8b_dpr_baseline.py.
The original script evaluates paired gold/distractor data in data/final;
this script evaluates flat retrieved-document data in data/final_retrieved.

Usage:
    python scripts/step8b_dpr_baseline_retrieved.py
    python scripts/step8b_dpr_baseline_retrieved.py --score_source retriever_score
    python scripts/step8b_dpr_baseline_retrieved.py --score_source embedding --batch_size 64
    python scripts/step8b_dpr_baseline_retrieved.py --device cuda:1

Output:
    results/probe_retrieved/dpr_baseline.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "final_retrieved" / "train.json"
EVAL_PATH = PROJECT_ROOT / "data" / "final_retrieved" / "eval.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "probe_retrieved"

Q_ENCODER = os.environ.get(
    "DPR_Q_ENCODER",
    "facebook/dpr-question_encoder-single-nq-base",
)
C_ENCODER = os.environ.get(
    "DPR_C_ENCODER",
    "facebook/dpr-ctx_encoder-single-nq-base",
)


def extract_doc_text(doc) -> str:
    if isinstance(doc, dict):
        title = str(doc.get("title", "")).strip()
        text = str(doc.get("text", "")).strip()
        assert text, f"Document dict has empty 'text' field: {doc.keys()}"
        if title:
            return f"Title: {title}\nText: {text}"
        return text
    return str(doc).strip()


def get_source_dataset(sample: dict) -> str:
    for key in ["source_dataset", "dataset_source", "source", "dataset"]:
        if key in sample:
            return str(sample[key]).lower().strip()
    return "unknown"


def get_label(sample: dict) -> int:
    if "label" in sample:
        return int(sample["label"])
    label_name = str(sample.get("label_name", "")).lower().strip()
    if label_name == "relevant":
        return 1
    if label_name == "distracting":
        return 0
    raise ValueError(f"Cannot parse label from sample keys: {sample.keys()}")


def get_retriever_score(sample: dict) -> float:
    doc = sample.get("doc")
    if not isinstance(doc, dict) or "score" not in doc:
        raise ValueError("score_source=retriever_score requires sample['doc']['score']")
    return float(doc["score"])


def embed_texts(texts, tokenizer, encoder, max_length, batch_size, device, desc):
    import torch

    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch = tokenizer(
                texts[i: i + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            out.append(encoder(**batch).pooler_output.cpu())
    return torch.cat(out, dim=0)


def process_split(split_path: Path,
                  q_tok,
                  q_enc,
                  c_tok,
                  c_enc,
                  q_max_len,
                  c_max_len,
                  batch_size,
                  device,
                  score_source):
    with open(split_path, "r") as f:
        samples = json.load(f)

    questions = [str(s["question"]) for s in samples]
    docs = [extract_doc_text(s["doc"]) for s in samples]
    labels = np.array([get_label(s) for s in samples], dtype=np.int64)
    sources = np.array([get_source_dataset(s) for s in samples])

    if score_source == "retriever_score":
        scores = np.array([get_retriever_score(s) for s in samples], dtype=np.float32)

    elif score_source == "embedding":
        import torch.nn.functional as F

        print(f"  Encoding {len(samples)} questions ({split_path.name}) ...")
        q_emb = embed_texts(
            questions, q_tok, q_enc, q_max_len, batch_size, device, "    questions"
        )

        print(f"  Encoding {len(samples)} retrieved docs ...")
        doc_emb = embed_texts(
            docs, c_tok, c_enc, c_max_len, batch_size, device, "    docs"
        )

        q_n = F.normalize(q_emb, dim=1)
        doc_n = F.normalize(doc_emb, dim=1)
        scores = (q_n * doc_n).sum(dim=1).numpy()

    else:
        raise ValueError(f"Unknown score_source: {score_source}")

    print("  Score diagnostics:")
    print(f"    score source: {score_source}")
    print(f"    label=1 score mean: {scores[labels == 1].mean():.4f}")
    print(f"    label=0 score mean: {scores[labels == 0].mean():.4f}")

    return scores, labels, sources


def find_best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    s_sorted = scores[order]
    y_sorted = labels[order]

    n = len(scores)
    n_pos = int(labels.sum())

    cum_pos = np.cumsum(y_sorted == 1)
    cum_neg = np.cumsum(y_sorted == 0)

    best_acc = n_pos / n
    best_t = float(s_sorted[0]) - 1e-6

    for i in range(n):
        tn = cum_neg[i]
        tp = n_pos - cum_pos[i]
        acc = (tn + tp) / n
        if acc > best_acc:
            best_acc = acc
            if i + 1 < n:
                best_t = float((s_sorted[i] + s_sorted[i + 1]) / 2.0)
            else:
                best_t = float(s_sorted[i]) + 1e-6

    return best_t


def evaluate_flat(scores: np.ndarray,
                  labels: np.ndarray,
                  sources: np.ndarray,
                  threshold: float) -> dict:
    out = {}

    for subset in sorted(set(sources.tolist())):
        mask = sources == subset
        if not mask.any():
            continue

        s = scores[mask]
        y = labels[mask]
        pred = (s >= threshold).astype(int)
        out[subset] = {
            "auroc": float(roc_auc_score(y, s)),
            "accuracy": float(accuracy_score(y, pred)),
            "score_mean_label_1": float(s[y == 1].mean()),
            "score_mean_label_0": float(s[y == 0].mean()),
            "n_samples": int(mask.sum()),
            "n_label_1": int(y.sum()),
            "n_label_0": int((y == 0).sum()),
        }

    pred = (scores >= threshold).astype(int)
    out["combined"] = {
        "auroc": float(roc_auc_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, pred)),
        "score_mean_label_1": float(scores[labels == 1].mean()),
        "score_mean_label_0": float(scores[labels == 0].mean()),
        "n_samples": int(len(scores)),
        "n_label_1": int(labels.sum()),
        "n_label_0": int((labels == 0).sum()),
    }
    return out


def print_results(title: str, metrics: dict) -> None:
    print(f"\n=== {title} ===")
    for subset, m in metrics.items():
        print(
            f"  {subset:10s}: "
            f"AUROC={m['auroc']:.4f}  "
            f"Acc={m['accuracy']:.4f}  "
            f"mean_pos={m['score_mean_label_1']:.4f}  "
            f"mean_neg={m['score_mean_label_0']:.4f}  "
            f"N={m['n_samples']} "
            f"(pos={m['n_label_1']}, neg={m['n_label_0']})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Step 8b: DPR similarity baseline for retrieved docs"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--q_max_len", type=int, default=128)
    parser.add_argument("--c_max_len", type=int, default=512)
    parser.add_argument(
        "--score_source",
        type=str,
        default="retriever_score",
        choices=["retriever_score", "embedding"],
        help="retriever_score uses stored doc['score']; embedding recomputes DPR cosine.",
    )
    args = parser.parse_args()

    if args.score_source == "embedding":
        import torch
        from transformers import (
            DPRContextEncoder,
            DPRContextEncoderTokenizerFast,
            DPRQuestionEncoder,
            DPRQuestionEncoderTokenizerFast,
        )

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device or "cpu"

    print(f"Device: {device}")
    print(f"Question encoder: {Q_ENCODER}")
    print(f"Context encoder:  {C_ENCODER}")
    print(f"Question max length: {args.q_max_len}")
    print(f"Context max length:  {args.c_max_len}")
    print(f"Batch size: {args.batch_size}")
    print(f"Score source: {args.score_source}")

    q_tok = q_enc = c_tok = c_enc = None
    if args.score_source == "embedding":
        print("\nLoading DPR encoders ...")
        q_tok = DPRQuestionEncoderTokenizerFast.from_pretrained(Q_ENCODER)
        q_enc = DPRQuestionEncoder.from_pretrained(Q_ENCODER).to(device).eval()

        c_tok = DPRContextEncoderTokenizerFast.from_pretrained(C_ENCODER)
        c_enc = DPRContextEncoder.from_pretrained(C_ENCODER).to(device).eval()

    print(f"\n=== Train ({TRAIN_PATH.name}) ===")
    tr_scores, tr_labels, tr_src = process_split(
        TRAIN_PATH, q_tok, q_enc, c_tok, c_enc,
        args.q_max_len, args.c_max_len, args.batch_size, device,
        args.score_source,
    )

    print(f"\n=== Eval ({EVAL_PATH.name}) ===")
    ev_scores, ev_labels, ev_src = process_split(
        EVAL_PATH, q_tok, q_enc, c_tok, c_enc,
        args.q_max_len, args.c_max_len, args.batch_size, device,
        args.score_source,
    )

    print("\n=== Calibrating threshold on train only ===")
    threshold = find_best_threshold(tr_scores, tr_labels)
    print(f"  Best train threshold: {threshold:.6f}")

    eval_flat = evaluate_flat(ev_scores, ev_labels, ev_src, threshold)
    train_flat = evaluate_flat(tr_scores, tr_labels, tr_src, threshold)

    print_results("Eval flat classification results", eval_flat)
    print_results("Train flat classification results (sanity)", train_flat)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "dpr_baseline.json"
    payload = {
        "encoder_q": Q_ENCODER,
        "encoder_c": C_ENCODER,
        "q_max_len": args.q_max_len,
        "c_max_len": args.c_max_len,
        "batch_size": args.batch_size,
        "score_source": args.score_source,
        "threshold_selected_on_train": float(threshold),
        "eval": {
            "flat": eval_flat,
        },
        "train_sanity": {
            "flat": train_flat,
        },
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
