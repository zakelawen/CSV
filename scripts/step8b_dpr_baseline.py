"""
Step 8b: DPR baseline for gold-vs-distractor data.

For each (question, doc, label) sample, compute:
    score = DPR_q_encoder(question)  -  DPR_ctx_encoder(doc)
and use this raw similarity as the predicted score for binary classification.

Important data note:
    data/final/{train,eval}.json has:
      - relevant_doc:    gold evidence from DPR positive contexts, with text/title
      - distracting_doc: DPR/FAISS top-1 retrieved document, with text/title/score

    Because relevant_doc does not have a stored DPR/FAISS retrieval score, this
    baseline intentionally recomputes a DPR encoder score for BOTH classes. Do
    not mix relevant_doc recomputed score with distracting_doc["score"]; that
    would compare two different score scales.

    The default score_source is "dot": raw DPR encoder inner product, closer to
    the FAISS/DPR retrieval score used by scripts/step2_retrieve.py and by the
    retrieved-doc counterpart scripts/step8b_dpr_baseline_retrieved.py.
    Pass --score_source cosine to reproduce the older cosine-normalized result.

Why this baseline matters:
    Step 8 trains LR / MLP / NearestCentroid probes on LLM hidden states.
    If those probes simply re-encode DPR's retrieval similarity, then strong
    Step 8 numbers don't actually demonstrate that LLM hidden states contain
    document-quality information beyond what DPR already provides.
    A weak DPR-similarity baseline + strong LLM probe = real signal.

This script does NOT depend on any LLM. It only uses the DPR encoders.

Encoder choice:
    facebook/dpr-question_encoder-single-nq-base
    facebook/dpr-ctx_encoder-single-nq-base

Threshold:
    Sweep all unique scores on TRAIN to find the threshold that maximizes
    train accuracy. Apply the same threshold to EVAL. AUROC is threshold-free.

Additional paired metric:
    Since each question has exactly one relevant_doc and one distracting_doc,
    we also report paired_acc:
        paired_acc = mean(score(relevant_doc) > score(distracting_doc))

Usage:
    python scripts/step8b_dpr_baseline.py
    python scripts/step8b_dpr_baseline.py --batch_size 64
    python scripts/step8b_dpr_baseline.py --device cuda:1
    python scripts/step8b_dpr_baseline.py --score_source cosine

Output:
    results/probe/dpr_baseline.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizerFast,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "final" / "train.json"
EVAL_PATH = PROJECT_ROOT / "data" / "final" / "eval.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "probe"

# Default encoders. Override via env vars if you have local snapshots.
Q_ENCODER = os.environ.get(
    "DPR_Q_ENCODER",
    "facebook/dpr-question_encoder-single-nq-base",
)
C_ENCODER = os.environ.get(
    "DPR_C_ENCODER",
    "facebook/dpr-ctx_encoder-single-nq-base",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_doc_text(doc) -> str:
    """
    Robust document text extraction.

    Matches the style used in Step6/Step9:
      - if dict: use title + text
      - if string or other object: convert to string
    """
    if isinstance(doc, dict):
        title = str(doc.get("title", "")).strip()
        text = str(doc.get("text", "")).strip()
        assert text, f"Document dict has empty 'text' field: {doc.keys()}"

        if title:
            return f"Title: {title}\nText: {text}"
        return text

    return str(doc).strip()


def get_source_dataset(sample: dict) -> str:
    """
    Robust source extraction.

    Expected field is source_dataset. Fallbacks are included only to avoid
    crashing if older intermediate files used a slightly different field name.
    """
    for key in ["source_dataset", "dataset_source", "source", "dataset"]:
        if key in sample:
            return str(sample[key]).lower().strip()

    return "unknown"


@torch.no_grad()
def embed_texts(texts, tokenizer, encoder, max_length, batch_size, device, desc):
    out = []

    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = tokenizer(
            texts[i: i + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        emb = encoder(**batch).pooler_output  # [B, 768]
        out.append(emb.cpu())

    return torch.cat(out, dim=0)


def paired_metrics(rel_scores: np.ndarray,
                   dis_scores: np.ndarray,
                   sources_per_pair: np.ndarray) -> dict:
    """
    Compute paired metrics.

    Each original QA item gives:
      rel_score = score(question, relevant_doc)
      dis_score = score(question, distracting_doc)

    paired_acc answers:
      how often DPR assigns higher similarity to the relevant doc than the
      distracting doc for the same question?
    """
    out = {}

    unique_sources = sorted(set(sources_per_pair.tolist()))
    for subset in unique_sources:
        mask = sources_per_pair == subset
        if not mask.any():
            continue

        rel = rel_scores[mask]
        dis = dis_scores[mask]
        margin = rel - dis

        out[subset] = {
            "paired_acc_rel_gt_dis": float((rel > dis).mean()),
            "paired_tie_rate": float((rel == dis).mean()),
            "paired_margin_mean_rel_minus_dis": float(margin.mean()),
            "paired_margin_median_rel_minus_dis": float(np.median(margin)),
            "rel_score_mean": float(rel.mean()),
            "dis_score_mean": float(dis.mean()),
            "n_pairs": int(mask.sum()),
        }

    # Combined
    margin = rel_scores - dis_scores
    out["combined"] = {
        "paired_acc_rel_gt_dis": float((rel_scores > dis_scores).mean()),
        "paired_tie_rate": float((rel_scores == dis_scores).mean()),
        "paired_margin_mean_rel_minus_dis": float(margin.mean()),
        "paired_margin_median_rel_minus_dis": float(np.median(margin)),
        "rel_score_mean": float(rel_scores.mean()),
        "dis_score_mean": float(dis_scores.mean()),
        "n_pairs": int(len(rel_scores)),
    }

    return out


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
    """
    Returns:
        scores: np.ndarray [2N]
            First N are relevant scores, next N are distracting scores.
        labels: np.ndarray [2N]
            First N are 1, next N are 0.
        sources_flat: np.ndarray [2N]
            Source for each flattened score.
        rel_scores: np.ndarray [N]
        dis_scores: np.ndarray [N]
        sources_per_pair: np.ndarray [N]
    """
    with open(split_path, "r") as f:
        samples = json.load(f)

    questions = [s["question"] for s in samples]
    rel_docs = [extract_doc_text(s["relevant_doc"]) for s in samples]
    dis_docs = [extract_doc_text(s["distracting_doc"]) for s in samples]
    sources_per_pair = np.array([get_source_dataset(s) for s in samples])

    print(f"  Encoding {len(samples)} questions ({split_path.name}) ...")
    q_emb = embed_texts(
        questions,
        q_tok,
        q_enc,
        q_max_len,
        batch_size,
        device,
        desc="    questions",
    )

    print(f"  Encoding {len(samples)} relevant docs ...")
    rel_emb = embed_texts(
        rel_docs,
        c_tok,
        c_enc,
        c_max_len,
        batch_size,
        device,
        desc="    relevant docs",
    )

    print(f"  Encoding {len(samples)} distracting docs ...")
    dis_emb = embed_texts(
        dis_docs,
        c_tok,
        c_enc,
        c_max_len,
        batch_size,
        device,
        desc="    distract docs",
    )

    if score_source == "dot":
        rel_scores = (q_emb * rel_emb).sum(dim=1).numpy()
        dis_scores = (q_emb * dis_emb).sum(dim=1).numpy()
    elif score_source == "cosine":
        q_n = F.normalize(q_emb, dim=1)
        rel_n = F.normalize(rel_emb, dim=1)
        dis_n = F.normalize(dis_emb, dim=1)
        rel_scores = (q_n * rel_n).sum(dim=1).numpy()
        dis_scores = (q_n * dis_n).sum(dim=1).numpy()
    else:
        raise ValueError(f"Unknown score_source: {score_source}")

    scores = np.concatenate([rel_scores, dis_scores])
    labels = np.concatenate([
        np.ones(len(samples), dtype=np.int64),
        np.zeros(len(samples), dtype=np.int64),
    ])
    sources_flat = np.concatenate([sources_per_pair, sources_per_pair])

    print("  Score diagnostics:")
    print(f"    score source: {score_source}")
    print(f"    rel score mean: {rel_scores.mean():.4f}")
    print(f"    dis score mean: {dis_scores.mean():.4f}")
    print(f"    paired acc rel>dis: {(rel_scores > dis_scores).mean():.4f}")
    print(f"    paired margin rel-dis: {(rel_scores - dis_scores).mean():.4f}")

    return scores, labels, sources_flat, rel_scores, dis_scores, sources_per_pair


def find_best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Find the threshold that maximizes accuracy on (scores, labels).

    threshold t means:
        predict 1 iff score >= t

    This threshold is selected on TRAIN only and then applied to EVAL.
    """
    order = np.argsort(scores)
    s_sorted = scores[order]
    y_sorted = labels[order]

    n = len(scores)
    n_pos = int(labels.sum())

    cum_pos = np.cumsum(y_sorted == 1)
    cum_neg = np.cumsum(y_sorted == 0)

    best_acc = -1.0
    best_t = float(s_sorted[0]) - 1e-6

    # Case: threshold below min score, predict all 1
    acc_all_pos = n_pos / n
    if acc_all_pos > best_acc:
        best_acc = acc_all_pos
        best_t = float(s_sorted[0]) - 1e-6

    # Cases: threshold just above s_sorted[i]
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
    """
    Evaluate flattened binary classification:
      relevant docs = positive class
      distracting docs = negative class
    """
    out = {}

    unique_sources = sorted(set(sources.tolist()))

    for subset in unique_sources:
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
        }

    # Combined
    pred = (scores >= threshold).astype(int)
    out["combined"] = {
        "auroc": float(roc_auc_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, pred)),
        "score_mean_label_1": float(scores[labels == 1].mean()),
        "score_mean_label_0": float(scores[labels == 0].mean()),
        "n_samples": int(len(scores)),
    }

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 8b: DPR similarity baseline"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="DPR encoder batch size. Default: 32",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda / cuda:0 / cuda:1 / cpu. Default: auto",
    )
    parser.add_argument(
        "--q_max_len",
        type=int,
        default=256,
        help="Question max token length. Default: 256, matching Step2 retrieval.",
    )
    parser.add_argument(
        "--c_max_len",
        type=int,
        default=512,
        help="Context max token length. Default: 512, DPR encoder upper bound",
    )
    parser.add_argument(
        "--score_source",
        type=str,
        default="dot",
        choices=["dot", "cosine"],
        help="dot = raw DPR inner product, closer to FAISS score; cosine = older normalized baseline.",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Question encoder: {Q_ENCODER}")
    print(f"Context encoder:  {C_ENCODER}")
    print(f"Question max length: {args.q_max_len}")
    print(f"Context max length:  {args.c_max_len}")
    print(f"Batch size: {args.batch_size}")
    print(f"Score source: DPR encoder {args.score_source} recomputed for both relevant_doc and distracting_doc")

    # Load encoders
    print("\nLoading DPR encoders ...")
    q_tok = DPRQuestionEncoderTokenizerFast.from_pretrained(Q_ENCODER)
    q_enc = DPRQuestionEncoder.from_pretrained(Q_ENCODER).to(device).eval()

    c_tok = DPRContextEncoderTokenizerFast.from_pretrained(C_ENCODER)
    c_enc = DPRContextEncoder.from_pretrained(C_ENCODER).to(device).eval()

    # Encode train
    print(f"\n=== Train ({TRAIN_PATH.name}) ===")
    (
        tr_scores,
        tr_labels,
        tr_src,
        tr_rel_scores,
        tr_dis_scores,
        tr_sources_per_pair,
    ) = process_split(
        TRAIN_PATH,
        q_tok,
        q_enc,
        c_tok,
        c_enc,
        args.q_max_len,
        args.c_max_len,
        args.batch_size,
        device,
        args.score_source,
    )

    # Encode eval
    print(f"\n=== Eval ({EVAL_PATH.name}) ===")
    (
        ev_scores,
        ev_labels,
        ev_src,
        ev_rel_scores,
        ev_dis_scores,
        ev_sources_per_pair,
    ) = process_split(
        EVAL_PATH,
        q_tok,
        q_enc,
        c_tok,
        c_enc,
        args.q_max_len,
        args.c_max_len,
        args.batch_size,
        device,
        args.score_source,
    )

    # Find threshold on TRAIN
    print("\n=== Calibrating threshold on train only ===")
    threshold = find_best_threshold(tr_scores, tr_labels)
    print(f"  Best train threshold: {threshold:.6f}")

    # Evaluate EVAL
    print("\n=== Eval flat classification results ===")
    eval_flat = evaluate_flat(ev_scores, ev_labels, ev_src, threshold)
    for subset, m in eval_flat.items():
        print(
            f"  {subset:10s}: "
            f"AUROC={m['auroc']:.4f}  "
            f"Acc={m['accuracy']:.4f}  "
            f"mean_pos={m['score_mean_label_1']:.4f}  "
            f"mean_neg={m['score_mean_label_0']:.4f}  "
            f"N={m['n_samples']}"
        )

    print("\n=== Eval paired results ===")
    eval_paired = paired_metrics(ev_rel_scores, ev_dis_scores, ev_sources_per_pair)
    for subset, m in eval_paired.items():
        print(
            f"  {subset:10s}: "
            f"paired_acc={m['paired_acc_rel_gt_dis']:.4f}  "
            f"margin_mean={m['paired_margin_mean_rel_minus_dis']:.4f}  "
            f"rel_mean={m['rel_score_mean']:.4f}  "
            f"dis_mean={m['dis_score_mean']:.4f}  "
            f"N={m['n_pairs']}"
        )

    # Train sanity
    print("\n=== Train flat classification results (sanity) ===")
    train_flat = evaluate_flat(tr_scores, tr_labels, tr_src, threshold)
    for subset, m in train_flat.items():
        print(
            f"  {subset:10s}: "
            f"AUROC={m['auroc']:.4f}  "
            f"Acc={m['accuracy']:.4f}  "
            f"mean_pos={m['score_mean_label_1']:.4f}  "
            f"mean_neg={m['score_mean_label_0']:.4f}  "
            f"N={m['n_samples']}"
        )

    print("\n=== Train paired results (sanity) ===")
    train_paired = paired_metrics(tr_rel_scores, tr_dis_scores, tr_sources_per_pair)
    for subset, m in train_paired.items():
        print(
            f"  {subset:10s}: "
            f"paired_acc={m['paired_acc_rel_gt_dis']:.4f}  "
            f"margin_mean={m['paired_margin_mean_rel_minus_dis']:.4f}  "
            f"rel_mean={m['rel_score_mean']:.4f}  "
            f"dis_mean={m['dis_score_mean']:.4f}  "
            f"N={m['n_pairs']}"
        )

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "dpr_baseline.json"

    payload = {
        "dataset_variant": "gold_distractor",
        "score_source": f"dpr_encoder_{args.score_source}_recomputed",
        "positive_doc": "relevant_doc",
        "negative_doc": "distracting_doc",
        "score_note": (
            "relevant_doc has no stored DPR/FAISS retrieval score, so scores "
            "are recomputed with the DPR question/context encoders for both classes. "
            "Default dot score is closer to the FAISS/DPR retrieval score than cosine."
        ),
        "encoder_q": Q_ENCODER,
        "encoder_c": C_ENCODER,
        "q_max_len": args.q_max_len,
        "c_max_len": args.c_max_len,
        "batch_size": args.batch_size,
        "threshold_selected_on_train": float(threshold),
        "eval": {
            "flat": eval_flat,
            "paired": eval_paired,
        },
        "train_sanity": {
            "flat": train_flat,
            "paired": train_paired,
        },
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
