"""
Step 9b: Re-evaluate CSV best checkpoint per dataset subset (NQ / TriviaQA).

Step 9 only reports combined AUROC/Acc (averaged over NQ + TriviaQA).
For the second-stage summary table we need per-subset numbers comparable to
Step 8's LR/MLP/Centroid probes. This script:

  1. Locates the best CSV checkpoint per model (highest best_auroc in the sweep).
  2. Loads the model + injects the saved CSV vector into the saved str_layer.
  3. Loads the saved centroids.
  4. Runs the same evaluation pipeline as Step 9, but splits eval into
     {nq, triviaqa, combined} and reports each separately.
  5. Saves results to results/csv/{model}_csv_per_subset.json.

Usage:
    # Re-eval a single model (uses the best ckpt found in {model}_csv_sweep.json)
    python scripts/step9b_reeval_per_subset.py --model gemma2b
    python scripts/step9b_reeval_per_subset.py --model gemma9b
    python scripts/step9b_reeval_per_subset.py --model qwen3_4b

    # Use a specific checkpoint instead of the sweep best
    python scripts/step9b_reeval_per_subset.py --model gemma2b \
        --ckpt results/csv/gemma2b_inject_2_cls_14_csv.pt

    # Run all models that have a sweep file
    python scripts/step9b_reeval_per_subset.py --model all

Output:
    results/csv/{model}_csv_per_subset.json
    {
      "model_name": "gemma2b",
      "checkpoint": "results/csv/gemma2b_inject_2_cls_14_csv.pt",
      "str_layer": 2,
      "cls_layer": 14,
      "lam": 5.0,
      "cos_temp": 0.1,
      "results": {
        "combined": {"auroc": ..., "accuracy": ..., "avg_margin": ..., "n_samples": ...},
        "nq":       {"auroc": ..., "accuracy": ..., "avg_margin": ..., "n_samples": ...},
        "triviaqa": {"auroc": ..., "accuracy": ..., "avg_margin": ..., "n_samples": ...}
      }
    }

Notes:
  - The saved best ckpt's reported best_auroc is on combined eval. The
    per-subset numbers from this script's `combined` output should match
    that value to within float rounding.
  - This script does not retrain anything. It only forwards eval data once.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make scripts/ importable so we can reuse step9 helpers without copy/paste.
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from step9_train_csv import (  # noqa: E402
    DATA_DIR,
    RESULTS_DIR,
    extract_doc_text,
    get_layer_rep,
    setup_tokenizer,
)
sys.path.insert(0, str(PROJECT_ROOT))
from csv_module.llm_layers import add_tsv_layers  # noqa: E402
from csv_module.train_utils import collate_fn  # noqa: E402


# ---------------------------------------------------------------------------
# Per-subset eval data loader
# ---------------------------------------------------------------------------

def load_eval_with_sources(tokenizer):
    """
    Like step9.load_and_tokenize but also returns per-prompt source labels.

    For each sample with source_dataset = "nq" / "triviaqa", emit:
        - prompt(relevant_doc),   label=1, source=src
        - prompt(distract_doc),   label=0, source=src

    Returns:
        prompts: list[Tensor[1, L_i]]
        labels:  list[int]
        sources: list[str]   (parallel to prompts)
    """
    eval_path = DATA_DIR / "eval.json"
    with open(eval_path, "r") as f:
        data = json.load(f)

    prompts: list = []
    labels: list = []
    sources: list = []

    for item in data:
        q = item["question"]
        # Use same source key that step8b uses, with light fallbacks
        src = None
        for k in ("source_dataset", "dataset_source", "source", "dataset"):
            if k in item:
                src = str(item[k]).lower().strip()
                break
        if src is None:
            src = "unknown"

        rel_text = extract_doc_text(item["relevant_doc"])
        rel_prompt = f"Document: {rel_text}\n\nQuestion: {q}\nAnswer:"
        prompts.append(tokenizer(rel_prompt, return_tensors="pt").input_ids)
        labels.append(1)
        sources.append(src)

        dis_text = extract_doc_text(item["distracting_doc"])
        dis_prompt = f"Document: {dis_text}\n\nQuestion: {q}\nAnswer:"
        prompts.append(tokenizer(dis_prompt, return_tensors="pt").input_ids)
        labels.append(0)
        sources.append(src)

    print(f"  Loaded eval: {len(data)} samples → {len(prompts)} prompts")
    print(f"  Sources: {dict((s, sources.count(s)) for s in set(sources))}")
    return prompts, labels, sources


# ---------------------------------------------------------------------------
# Eval forward — same logic as step9.evaluate, but returns per-prompt scores
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_scores_all(model, centroids, prompts, labels, device,
                       batch_size, cls_layer, cos_temp, pad_id):
    """
    Run forward on all eval prompts, return per-prompt:
        scores  P(relevant) ∈ [0, 1]
        preds   argmax (0 or 1)
        margins |logit_1 - logit_0|

    All shapes: [N_eval]
    """
    model.eval()
    all_scores = []
    all_preds = []
    all_labels = []
    all_margins = []

    n = len(prompts)
    for s in range(0, n, batch_size):
        bp_list = prompts[s: s + batch_size]
        bl_list = labels[s: s + batch_size]

        bp, _, mask = collate_fn(bp_list, bl_list, pad_id=pad_id)
        bp = bp.to(device)
        mask = mask.to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model.model(
                input_ids=bp,
                attention_mask=mask,
                output_hidden_states=True,
            )
            last_token_rep = get_layer_rep(output, mask, cls_layer)

        last_token_rep = F.normalize(last_token_rep.float(), p=2, dim=-1)
        c = F.normalize(centroids, p=2, dim=-1)
        sims = torch.matmul(last_token_rep, c.T)        # [B, 2]
        logits = sims / cos_temp
        probs = torch.softmax(logits, dim=-1)           # [B, 2]

        all_scores.append(probs[:, 1].cpu())
        all_preds.append(torch.argmax(probs, dim=-1).cpu())
        all_labels.append(torch.tensor(bl_list))
        all_margins.append(torch.abs(logits[:, 1] - logits[:, 0]).cpu())

    return (
        torch.cat(all_scores).numpy(),
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_margins).numpy(),
    )


def metrics_for_subset(scores, preds, labels, margins, mask) -> dict:
    """
    Compute AUROC / Accuracy / avg_margin / n_samples on a subset mask.
    """
    s = scores[mask]
    p = preds[mask]
    y = labels[mask]
    m = margins[mask]

    n = int(mask.sum())
    if n == 0:
        return {"auroc": None, "accuracy": None, "avg_margin": None,
                "n_samples": 0}

    # Guard: AUROC needs both classes
    has_both_classes = (y == 0).any() and (y == 1).any()
    return {
        "auroc": float(roc_auc_score(y, s)) if has_both_classes else None,
        "accuracy": float(accuracy_score(y, p)),
        "avg_margin": float(m.mean()),
        "n_samples": n,
    }


# ---------------------------------------------------------------------------
# Best-checkpoint resolution
# ---------------------------------------------------------------------------

def find_best_checkpoint(model_name: str) -> tuple[Path, dict]:
    """
    Read {model}_csv_sweep.json, pick the entry with max best_auroc, return
    (checkpoint_path, sweep_entry).
    """
    sweep_path = RESULTS_DIR / f"{model_name}_csv_sweep.json"
    if not sweep_path.exists():
        raise FileNotFoundError(
            f"Sweep file not found: {sweep_path}. Run Step 9 first."
        )
    with open(sweep_path, "r") as f:
        sweep = json.load(f)

    if not sweep:
        raise ValueError(f"Sweep file is empty: {sweep_path}")

    best_key = max(sweep, key=lambda k: sweep[k]["best_auroc"])
    entry = sweep[best_key]
    ckpt_path = Path(entry["checkpoint"])

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint pointed by sweep does not exist: {ckpt_path}"
        )

    print(f"  Best config in sweep: {best_key}")
    print(f"    best_auroc (saved):   {entry['best_auroc']:.4f}")
    print(f"    best_accuracy(saved): {entry['best_accuracy']:.4f}")
    print(f"    checkpoint:           {ckpt_path}")

    return ckpt_path, entry


# ---------------------------------------------------------------------------
# Main per-model logic
# ---------------------------------------------------------------------------

def reeval_one_model(model_name: str, ckpt_path: Path | None,
                     batch_size: int, device: str):
    print(f"\n=== Re-evaluating CSV per subset for {model_name} ===")

    if ckpt_path is None:
        ckpt_path, _ = find_best_checkpoint(model_name)
    else:
        print(f"  Using user-specified checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    str_layer = int(ckpt["str_layer"])
    cls_layer = int(ckpt["cls_layer"])
    lam = float(ckpt["lam"])
    cos_temp = float(ckpt["cos_temp"])
    hf_name = ckpt["hf_name"]
    tsv = ckpt["tsv"]                # [1, 1, D] fp32 cpu
    centroids = ckpt["centroids"]    # [2, D] fp32 cpu

    print(f"  str_layer={str_layer}  cls_layer={cls_layer}  "
          f"lam={lam}  cos_temp={cos_temp}")
    print(f"  hf_name: {hf_name}")

    # --- Load tokenizer + model exactly as Step 9 does ---
    print("  Loading tokenizer …")
    tokenizer = setup_tokenizer(hf_name)
    pad_id = tokenizer.pad_token_id

    print("  Loading model …")
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    # --- Inject saved CSV ---
    # Make tsv a non-trainable Parameter (we're not training, just forwarding)
    tsv_param = nn.Parameter(tsv.float(), requires_grad=False)
    add_tsv_layers(model, tsv_param, [lam], str_layer, model_name)

    # Move centroids to whatever device evaluate expects (we'll keep on CPU
    # and let metrics_for_subset run on numpy; the score computation in
    # compute_scores_all needs centroids on the same device as last_token_rep).
    centroids_dev = centroids.to(device)

    # --- Load eval data with source labels ---
    print("  Loading eval data …")
    prompts, labels, sources = load_eval_with_sources(tokenizer)

    # --- Forward all of eval, collect per-prompt scores ---
    print("  Forwarding eval …")
    scores, preds, labels_arr, margins = compute_scores_all(
        model, centroids_dev,
        prompts, labels, device,
        batch_size=batch_size,
        cls_layer=cls_layer,
        cos_temp=cos_temp,
        pad_id=pad_id,
    )

    sources_arr = np.array(sources)

    # --- Per-subset metrics ---
    out = {}
    out["combined"] = metrics_for_subset(
        scores, preds, labels_arr, margins,
        mask=np.ones_like(labels_arr, dtype=bool),
    )
    for sub in sorted(set(sources)):
        out[sub] = metrics_for_subset(
            scores, preds, labels_arr, margins,
            mask=(sources_arr == sub),
        )

    print("\n  Per-subset results:")
    for sub, m in out.items():
        if m["auroc"] is None:
            print(f"    {sub:10s}: N={m['n_samples']}  (AUROC unavailable)")
            continue
        print(f"    {sub:10s}: AUROC={m['auroc']:.4f}  "
              f"Acc={m['accuracy']:.4f}  "
              f"Margin={m['avg_margin']:.4f}  "
              f"N={m['n_samples']}")

    # Sanity: this script's combined AUROC should match the saved best_auroc
    saved_best = float(ckpt["best_auroc"])
    diff = abs(out["combined"]["auroc"] - saved_best)
    if diff > 1e-3:
        print(f"\n  ⚠ combined AUROC ({out['combined']['auroc']:.4f}) "
              f"differs from saved best_auroc ({saved_best:.4f}) by {diff:.4f}")
    else:
        print(f"\n  ✓ combined AUROC matches saved best_auroc "
              f"(diff={diff:.5f})")

    # --- Save ---
    save_path = RESULTS_DIR / f"{model_name}_csv_per_subset.json"
    payload = {
        "model_name": model_name,
        "checkpoint": str(ckpt_path),
        "str_layer": str_layer,
        "cls_layer": cls_layer,
        "lam": lam,
        "cos_temp": cos_temp,
        "saved_best_auroc": saved_best,
        "saved_best_accuracy": float(ckpt["best_accuracy"]),
        "saved_best_avg_margin": float(ckpt["best_avg_margin"]),
        "results": out,
    }
    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved → {save_path}")

    # Free GPU
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 9b: re-evaluate CSV best checkpoint per dataset subset"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="gemma2b / gemma9b / qwen3_4b / all",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help=(
            "Specific checkpoint path to re-eval. If omitted, the best ckpt "
            "from {model}_csv_sweep.json is used."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Eval forward batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. cuda:1). Default: auto",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.model == "all":
        if args.ckpt is not None:
            print("Cannot use --ckpt with --model all. Run per model.")
            sys.exit(1)
        for m in ["gemma2b", "gemma9b", "qwen3_4b"]:
            try:
                reeval_one_model(m, None, args.batch_size, device)
            except FileNotFoundError as e:
                print(f"\n[skip {m}] {e}")
    else:
        ckpt = Path(args.ckpt) if args.ckpt else None
        reeval_one_model(args.model, ckpt, args.batch_size, device)

    print("\nDone.")


if __name__ == "__main__":
    main()