#!/usr/bin/env python
"""
Step 14: CSV + non-prototype head ablation.

This script runs ablation (c):

    CSV = yes, Prototype = no

It loads an already trained CSV checkpoint, injects the CSV vector into the
base model, extracts the last-token representation at the CSV checkpoint's
classification layer, and trains an ordinary LR/MLP classifier on top of those
CSV-modified representations.

Default setting is retrieved-document classification, matching the gate used in
the pipeline:

    data/final_retrieved/train.json
    data/final_retrieved/eval.json

The output is directly comparable to:
  (a) results/probe_retrieved/*_probe_results_lr.json
  (b) results/probe_retrieved/*_probe_results_centroid.json
  Ours results/pipeline/*_retrieved_csv_eval_validation.json

Example:
  python scripts/step14_csv_nonprototype_ablation.py \
    --model gemma2b \
    --variant retrieved \
    --classifier lr
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from tqdm import tqdm

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from csv_module.llm_layers import add_tsv_layers  # noqa: E402
from csv_module.train_utils import collate_fn, get_last_non_padded_token_rep  # noqa: E402
from step11_contrastive_decoding import (  # noqa: E402
    MODEL_REGISTRY,
    extract_doc_text,
    load_base_model_and_tokenizer,
)


warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="Stochastic Optimizer:.*")
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")


DATA_DIRS = {
    "retrieved": PROJECT_ROOT / "data" / "final_retrieved",
    "gold": PROJECT_ROOT / "data" / "final",
}

DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "ablation_csv_nonprototype"
CLASSIFIERS = ["lr", "mlp"]


RUN_NAMES = {
    "retrieved": {
        "gemma2b": "gemma2b_retrieved",
        "gemma9b": "gemma9b_retrieved",
        "qwen3_4b": "qwen3_4b_retrieved",
    },
    "gold": {
        "gemma2b": "gemma2b",
        "gemma9b": "gemma9b",
        "qwen3_4b": "qwen3_4b",
    },
}


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_dataset_name(x: Any) -> str:
    s = str(x).lower().strip()
    aliases = {
        "naturalquestions": "nq",
        "natural_questions": "nq",
        "tqa": "triviaqa",
        "trivia": "triviaqa",
        "trivia_qa": "triviaqa",
    }
    return aliases.get(s, s)


def default_csv_dir(model: str, variant: str) -> Path:
    if variant == "gold":
        return PROJECT_ROOT / "results" / "csv"

    env_dir = os.environ.get("CSV_RETRIEVED_DIR")
    if env_dir:
        return Path(env_dir)

    # The final Qwen3 retrieved-doc CSV used by the pipeline is curated here.
    qwen_pipeline_dir = PROJECT_ROOT / "results" / "csv_retrieved_qwen3_4b_best_for_pipeline"
    if model == "qwen3_4b" and qwen_pipeline_dir.exists():
        return qwen_pipeline_dir

    return PROJECT_ROOT / "results" / "csv_retrieved"


def resolve_best_csv_checkpoint(
    model: str,
    variant: str,
    csv_dir: Path | None,
    checkpoint: Path | None,
) -> tuple[Path, dict[str, Any], str]:
    if checkpoint is not None:
        ckpt_path = checkpoint
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Cannot find checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        key = f"explicit_{ckpt_path.stem}"
        entry = {
            "checkpoint": str(ckpt_path),
            "str_layer": int(ckpt["str_layer"]),
            "cls_layer": int(ckpt["cls_layer"]),
            "best_auroc": ckpt.get("best_auroc"),
            "best_accuracy": ckpt.get("best_accuracy"),
        }
        return ckpt_path, entry, key

    run_name = RUN_NAMES[variant][model]
    resolved_csv_dir = csv_dir if csv_dir is not None else default_csv_dir(model, variant)
    sweep_path = resolved_csv_dir / f"{run_name}_csv_sweep.json"
    if not sweep_path.exists():
        raise FileNotFoundError(
            f"Cannot find CSV sweep: {sweep_path}\n"
            "Pass --csv_dir or --checkpoint explicitly."
        )

    sweep = load_json(sweep_path)
    if not isinstance(sweep, dict) or not sweep:
        raise ValueError(f"Bad sweep file: {sweep_path}")

    best_key, best_entry = max(
        sweep.items(),
        key=lambda kv: float(kv[1].get("best_auroc", -1.0)),
    )
    ckpt_path = Path(best_entry["checkpoint"])
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    if not ckpt_path.exists():
        fallback = resolved_csv_dir / ckpt_path.name
        if fallback.exists():
            ckpt_path = fallback
        else:
            raise FileNotFoundError(
                f"Best checkpoint from sweep does not exist: {ckpt_path}"
            )

    return ckpt_path, best_entry, best_key


def prompt_from_doc(question: str, doc: Any) -> str:
    doc_text = extract_doc_text(doc)
    return f"Document: {doc_text}\n\nQuestion: {question}\nAnswer:"


def label_from_retrieved_item(item: dict[str, Any], idx: int) -> tuple[int, str]:
    if "label" in item:
        label = int(item["label"])
    else:
        label_name = str(item.get("label_name", "")).lower().strip()
        if label_name == "relevant":
            label = 1
        elif label_name == "distracting":
            label = 0
        else:
            raise ValueError(f"Item {idx} has invalid label_name={label_name!r}")
    if label not in (0, 1):
        raise ValueError(f"Item {idx} has invalid label={label!r}")
    return label, "relevant" if label == 1 else "distracting"


def load_split_as_prompts(
    variant: str,
    split: str,
    tokenizer,
    limit: int | None,
) -> dict[str, Any]:
    path = DATA_DIRS[variant] / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")

    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")

    prompts: list[torch.Tensor] = []
    labels: list[int] = []
    label_names: list[str] = []
    sources: list[str] = []
    questions: list[str] = []

    if variant == "retrieved":
        for idx, item in enumerate(data):
            q = item["question"]
            label, label_name = label_from_retrieved_item(item, idx)
            source = normalize_dataset_name(item.get("source_dataset", "unknown"))
            prompts.append(tokenizer(prompt_from_doc(q, item["doc"]), return_tensors="pt").input_ids)
            labels.append(label)
            label_names.append(label_name)
            sources.append(source)
            questions.append(q)
            if limit is not None and len(prompts) >= limit:
                break
    elif variant == "gold":
        for idx, item in enumerate(data):
            q = item["question"]
            source = normalize_dataset_name(item.get("source_dataset", "unknown"))

            prompts.append(tokenizer(prompt_from_doc(q, item["relevant_doc"]), return_tensors="pt").input_ids)
            labels.append(1)
            label_names.append("relevant")
            sources.append(source)
            questions.append(q)

            prompts.append(tokenizer(prompt_from_doc(q, item["distracting_doc"]), return_tensors="pt").input_ids)
            labels.append(0)
            label_names.append("distracting")
            sources.append(source)
            questions.append(q)

            if limit is not None and len(prompts) >= limit:
                prompts = prompts[:limit]
                labels = labels[:limit]
                label_names = label_names[:limit]
                sources = sources[:limit]
                questions = questions[:limit]
                break
    else:
        raise ValueError(f"Unknown variant: {variant}")

    print(
        f"  Loaded {variant}/{split}: {len(prompts)} prompts "
        f"labels={dict(Counter(label_names))} sources={dict(Counter(sources))}"
    )
    return {
        "path": str(path),
        "prompts": prompts,
        "labels": np.array(labels, dtype=np.int64),
        "label_names": label_names,
        "sources": np.array(sources),
        "questions": questions,
    }


def summarize_split_without_tokenizer(variant: str, split: str, limit: int | None) -> dict[str, Any]:
    path = DATA_DIRS[variant] / f"{split}.json"
    data = load_json(path)
    label_names: list[str] = []
    sources: list[str] = []

    if variant == "retrieved":
        for idx, item in enumerate(data):
            _, label_name = label_from_retrieved_item(item, idx)
            label_names.append(label_name)
            sources.append(normalize_dataset_name(item.get("source_dataset", "unknown")))
            if limit is not None and len(label_names) >= limit:
                break
    else:
        for item in data:
            source = normalize_dataset_name(item.get("source_dataset", "unknown"))
            label_names.extend(["relevant", "distracting"])
            sources.extend([source, source])
            if limit is not None and len(label_names) >= limit:
                label_names = label_names[:limit]
                sources = sources[:limit]
                break

    return {
        "path": str(path),
        "n_prompts": len(label_names),
        "label_counts": dict(Counter(label_names)),
        "source_counts": dict(Counter(sources)),
    }


def get_layer_rep(output, attention_mask: torch.Tensor, cls_layer: int) -> torch.Tensor:
    hidden_states = output.hidden_states
    if cls_layer == -1:
        hs = hidden_states[-1]
    else:
        hs = hidden_states[cls_layer + 1]
    return get_last_non_padded_token_rep(hs, attention_mask.to(hs.device))


@torch.no_grad()
def extract_csv_reps(
    model,
    prompts: list[torch.Tensor],
    labels: np.ndarray,
    batch_size: int,
    pad_id: int,
    cls_layer: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    all_reps = []
    label_list = labels.tolist()

    for start in tqdm(range(0, len(prompts), batch_size), desc="extract CSV reps"):
        batch_prompts = prompts[start:start + batch_size]
        batch_labels = label_list[start:start + batch_size]
        batch_ids, _, attention_mask = collate_fn(batch_prompts, batch_labels, pad_id=pad_id)
        batch_ids = batch_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model.model(
                input_ids=batch_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            reps = get_layer_rep(output, attention_mask, cls_layer)

        all_reps.append(reps.cpu().float())

    return torch.cat(all_reps, dim=0).numpy()


def fit_eval_nonprototype_head(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
    classifier: str,
    pca_dim: int,
    seed: int,
) -> dict[str, float]:
    n_components = min(pca_dim, train_x.shape[1], train_x.shape[0])
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_x)
    eval_pca = pca.transform(eval_x)

    train_norm = normalize(train_pca, norm="l2")
    eval_norm = normalize(eval_pca, norm="l2")

    if classifier == "lr":
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        clf.fit(train_norm, train_y)
        scores = clf.predict_proba(eval_norm)[:, 1]
        preds = clf.predict(eval_norm)
        margin = np.abs(clf.decision_function(eval_norm)).mean()
    elif classifier == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
        )
        clf.fit(train_norm, train_y)
        scores = clf.predict_proba(eval_norm)[:, 1]
        preds = clf.predict(eval_norm)
        eps = 1e-8
        logit = np.log((scores + eps) / (1.0 - scores + eps))
        margin = np.abs(logit).mean()
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    return {
        "auroc": float(roc_auc_score(eval_y, scores)),
        "accuracy": float(accuracy_score(eval_y, preds)),
        "avg_margin": float(margin),
        "pca_dim": int(n_components),
        "n_train": int(len(train_y)),
        "n_eval": int(len(eval_y)),
    }


def run_classifier_by_subset(
    train_reps: np.ndarray,
    eval_reps: np.ndarray,
    train_labels: np.ndarray,
    eval_labels: np.ndarray,
    train_sources: np.ndarray,
    eval_sources: np.ndarray,
    classifier: str,
    pca_dim: int,
    seed: int,
) -> dict[str, Any]:
    subsets = sorted(set(train_sources.tolist()) | set(eval_sources.tolist()))
    subsets.append("combined")

    results: dict[str, Any] = {}
    for subset in subsets:
        if subset == "combined":
            train_mask = np.ones(len(train_labels), dtype=bool)
            eval_mask = np.ones(len(eval_labels), dtype=bool)
        else:
            train_mask = train_sources == subset
            eval_mask = eval_sources == subset

        if train_mask.sum() == 0 or eval_mask.sum() == 0:
            continue
        if len(np.unique(train_labels[train_mask])) < 2 or len(np.unique(eval_labels[eval_mask])) < 2:
            print(f"  [skip] {subset}: needs both classes in train/eval")
            continue

        metrics = fit_eval_nonprototype_head(
            train_x=train_reps[train_mask],
            train_y=train_labels[train_mask],
            eval_x=eval_reps[eval_mask],
            eval_y=eval_labels[eval_mask],
            classifier=classifier,
            pca_dim=pca_dim,
            seed=seed,
        )
        results[subset] = metrics
        print(
            f"  {classifier:3s} {subset:9s}: "
            f"AUROC={metrics['auroc']:.4f}  "
            f"Acc={metrics['accuracy']:.4f}  "
            f"N_eval={metrics['n_eval']}"
        )

    return results


def inject_checkpoint_csv(model, ckpt: dict[str, Any], model_name: str) -> None:
    tsv_param = nn.Parameter(ckpt["tsv"].float(), requires_grad=False)
    lam = [float(ckpt["lam"])]
    add_tsv_layers(model, tsv_param, lam, int(ckpt["str_layer"]), model_name)


def process_model(args, model: str) -> list[Path]:
    ckpt_path, sweep_entry, best_key = resolve_best_csv_checkpoint(
        model=model,
        variant=args.variant,
        csv_dir=Path(args.csv_dir) if args.csv_dir else None,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    str_layer = int(ckpt["str_layer"])
    cls_layer = int(args.cls_layer) if args.cls_layer is not None else int(ckpt["cls_layer"])

    print("\n" + "=" * 80)
    print(f"CSV non-prototype ablation: model={model}, variant={args.variant}")
    print("=" * 80)
    print(f"  checkpoint: {ckpt_path}")
    print(f"  best key:   {best_key}")
    print(f"  inject L:   {str_layer}")
    print(f"  cls L:      {cls_layer}")
    print(f"  csv AUROC:  {sweep_entry.get('best_auroc')}")
    print(f"  csv Acc:    {sweep_entry.get('best_accuracy')}")

    if args.dry_run:
        train_summary = summarize_split_without_tokenizer(args.variant, "train", args.limit)
        eval_summary = summarize_split_without_tokenizer(args.variant, "eval", args.limit)
        print("  Dry run data summary:")
        print(f"    train: {train_summary}")
        print(f"    eval:  {eval_summary}")
        return []

    gen_model, tokenizer, hf_name = load_base_model_and_tokenizer(model)
    inject_checkpoint_csv(gen_model, ckpt, model)

    train_data = load_split_as_prompts(args.variant, "train", tokenizer, args.limit)
    eval_data = load_split_as_prompts(args.variant, "eval", tokenizer, args.limit)

    print("\nExtracting CSV-modified train representations ...")
    train_reps = extract_csv_reps(
        model=gen_model,
        prompts=train_data["prompts"],
        labels=train_data["labels"],
        batch_size=args.batch_size,
        pad_id=tokenizer.pad_token_id,
        cls_layer=cls_layer,
    )
    print(f"  train reps: {train_reps.shape}")

    print("\nExtracting CSV-modified eval representations ...")
    eval_reps = extract_csv_reps(
        model=gen_model,
        prompts=eval_data["prompts"],
        labels=eval_data["labels"],
        batch_size=args.batch_size,
        pad_id=tokenizer.pad_token_id,
        cls_layer=cls_layer,
    )
    print(f"  eval reps:  {eval_reps.shape}")

    classifiers = CLASSIFIERS if args.classifier == "all" else [args.classifier]
    out_paths: list[Path] = []
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    for classifier in classifiers:
        print(f"\nTraining non-prototype head: {classifier}")
        results = run_classifier_by_subset(
            train_reps=train_reps,
            eval_reps=eval_reps,
            train_labels=train_data["labels"],
            eval_labels=eval_data["labels"],
            train_sources=train_data["sources"],
            eval_sources=eval_data["sources"],
            classifier=classifier,
            pca_dim=args.pca_dim,
            seed=args.seed,
        )

        payload = {
            "meta": {
                "ablation": "csv_nonprototype_head",
                "csv": True,
                "prototype": False,
                "classifier": classifier,
                "variant": args.variant,
                "model": model,
                "hf_name": hf_name,
                "csv_checkpoint": str(ckpt_path),
                "csv_best_key": best_key,
                "csv_sweep_entry": sweep_entry,
                "str_layer": str_layer,
                "cls_layer": cls_layer,
                "pca_dim": int(args.pca_dim),
                "seed": int(args.seed),
                "train_source": train_data["path"],
                "eval_source": eval_data["path"],
                "train_label_counts": dict(Counter(train_data["label_names"])),
                "eval_label_counts": dict(Counter(eval_data["label_names"])),
                "train_source_counts": dict(Counter(train_data["sources"].tolist())),
                "eval_source_counts": dict(Counter(eval_data["sources"].tolist())),
            },
            "results": results,
        }
        out_path = out_dir / f"{model}_{args.variant}_csv_nonprototype_{classifier}.json"
        save_json(payload, out_path)
        out_paths.append(out_path)
        print(f"  Saved -> {out_path}")

    del gen_model
    torch.cuda.empty_cache()
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CSV + non-prototype head ablation")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()) + ["all"])
    parser.add_argument("--variant", type=str, default="retrieved",
                        choices=["retrieved", "gold"])
    parser.add_argument("--classifier", type=str, default="lr",
                        choices=CLASSIFIERS + ["all"])
    parser.add_argument("--csv_dir", type=str, default=None,
                        help="Directory containing {run_name}_csv_sweep.json.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Explicit CSV checkpoint .pt. Overrides --csv_dir.")
    parser.add_argument("--cls_layer", type=int, default=None,
                        help="Classification layer override. Default: checkpoint cls_layer.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pca_dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="Debug: use the first N prompts from train/eval.")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--dry_run", action="store_true",
                        help="Resolve checkpoint and summarize data without loading the model.")
    args = parser.parse_args()

    if args.checkpoint and args.model == "all":
        raise ValueError("--checkpoint can only be used with one --model, not --model all")

    models = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    all_outputs: list[Path] = []
    for model in models:
        all_outputs.extend(process_model(args, model))

    print("\nDone. Outputs:")
    for path in all_outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
