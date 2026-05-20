#!/usr/bin/env python
"""
Step 10R: Retrieved-document logit lens analysis.

This is the pipeline-matched version of Step10. Instead of analyzing
gold_ctx / distracting_ctx from data/final/eval.json, it analyzes retrieved
top-1 documents from data/final_retrieved/eval.json:

  - no_ctx
  - retrieved_ctx with label=1 (relevant)
  - retrieved_ctx with label=0 (irrelevant)

The main question is whether the layer selected by gold-doc Step10 remains a
good layer when the context is a real retrieved document.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import step10_logit_lens as base  # noqa: E402


EVAL_PATH = PROJECT_ROOT / "data" / "final_retrieved" / "eval.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "logit_lens_retrieved"

LOCAL_GEMMA2B_PATH = "google/gemma-2-2b"
LOCAL_GEMMA9B_PATH = "google/gemma-2-9b"
LOCAL_QWEN3_4B_PATH = "Qwen/Qwen3-4B-Base"

MODEL_REGISTRY = {
    "gemma2b": {
        "hf_name": os.environ.get("GEMMA_MODEL_PATH", LOCAL_GEMMA2B_PATH),
    },
    "gemma9b": {
        "hf_name": os.environ.get("GEMMA9B_MODEL_PATH", LOCAL_GEMMA9B_PATH),
    },
    "qwen3_4b": {
        "hf_name": os.environ.get("QWEN3_4B_MODEL_PATH", LOCAL_QWEN3_4B_PATH),
    },
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_label(item: dict) -> int:
    if "label" in item:
        label = int(item["label"])
    else:
        name = str(item.get("label_name", "")).strip().lower()
        if name == "relevant":
            label = 1
        elif name == "distracting":
            label = 0
        else:
            raise ValueError(f"Cannot parse label from item keys={list(item.keys())}")
    if label not in (0, 1):
        raise ValueError(f"Invalid label={label!r}")
    return label


def valid_answers(item: dict) -> list[str]:
    return [
        a for a in (item.get("answers") or [])
        if isinstance(a, str) and a.strip()
    ][:base.MAX_CANDIDATE_ANSWERS]


def per_subset_layer_mean(arr_2d: np.ndarray, mask: np.ndarray) -> list[float | None]:
    sub = arr_2d[mask]
    if len(sub) == 0:
        return [None] * arr_2d.shape[1]
    with np.errstate(all="ignore"):
        m = np.nanmean(sub, axis=0)
    return [float(x) if np.isfinite(x) else None for x in m]


def per_sample_diff_mean(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    mask: np.ndarray,
) -> list[float | None]:
    diff = arr_a - arr_b
    sub = diff[mask]
    if len(sub) == 0:
        return [None] * diff.shape[1]
    with np.errstate(all="ignore"):
        m = np.nanmean(sub, axis=0)
    return [float(x) if np.isfinite(x) else None for x in m]


def argmax_layer(values: list[float | None]) -> tuple[int, float]:
    arr = np.array([x if x is not None else -np.inf for x in values])
    idx = int(np.argmax(arr))
    return idx, float(arr[idx])


def first_above(values: list[float | None], threshold: float) -> int:
    arr = np.array([x if x is not None else -np.inf for x in values])
    above = arr > threshold
    return int(np.argmax(above)) if above.any() else -1


def top_layers(delta: list[float | None], other: list[float | None], k: int = 8) -> list[dict]:
    pairs = [
        (i, v)
        for i, v in enumerate(delta)
        if v is not None and np.isfinite(v)
    ]
    pairs.sort(key=lambda x: x[1], reverse=True)
    out = []
    for i, v in pairs[:k]:
        other_v = other[i] if i < len(other) else None
        out.append({
            "layer": int(i),
            "delta_relevant_minus_no": float(v),
            "delta_irrelevant_minus_no": None if other_v is None else float(other_v),
        })
    return out


def process_model(
    model_key: str,
    dataset_filter: str,
    max_length: int,
    max_samples: int | None,
    requested_device: str | None,
) -> Path:
    print(f"\n=== Step 10R Retrieved Logit Lens: {model_key} ===")
    hf_name = MODEL_REGISTRY[model_key]["hf_name"]
    print(f"  hf_name: {hf_name}")
    print(f"  eval:    {EVAL_PATH}")

    print("  Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"    pad_token = {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")

    print("  Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=torch.float16,
        device_map="auto" if requested_device is None else None,
        low_cpu_mem_usage=True,
    )
    if requested_device is not None:
        model = model.to(requested_device)
    model.eval()

    model_device = next(model.parameters()).device
    print(f"  model input device: {model_device}")

    final_norm = base.get_final_norm(model)
    lm_head = base.get_lm_head(model)
    softcap = getattr(model.config, "final_logit_softcapping", None)
    if softcap is not None:
        softcap = float(softcap)
        print(f"  final_logit_softcapping = {softcap}")
    else:
        print("  final_logit_softcapping = None")

    base.verify_final_layer_projection(
        model=model,
        tokenizer=tokenizer,
        device=model_device,
        softcap=softcap,
    )

    with torch.no_grad():
        dummy_ids = torch.tensor(
            [[tokenizer.encode("hello", add_special_tokens=True)[0]]],
            dtype=torch.long,
            device=model_device,
        )
        dummy_out = model.model(
            input_ids=dummy_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        num_layers = len(dummy_out.hidden_states)
    print(f"  num_layers (incl. embedding) = {num_layers}")

    samples = load_json(EVAL_PATH)
    if dataset_filter != "all":
        samples = [
            s for s in samples
            if str(s.get("source_dataset", "")).lower().strip() == dataset_filter
        ]
        print(f"  Filtered dataset={dataset_filter}: {len(samples)} samples")
    if max_samples is not None and max_samples < len(samples):
        samples = samples[:max_samples]
        print(f"  Using {max_samples} eval samples (truncated)")
    else:
        print(f"  Using all {len(samples)} eval samples")

    n_samples = len(samples)
    signal = {
        "no_ctx": np.full((n_samples, num_layers), np.nan, dtype=np.float64),
        "retrieved_ctx": np.full((n_samples, num_layers), np.nan, dtype=np.float64),
    }
    sources: list[str] = []
    labels: list[int] = []
    canonical_answer_idx: list[int] = []
    skipped = 0

    for s_idx, sample in enumerate(tqdm(samples, desc="  retrieved samples")):
        question = sample["question"]
        answers = valid_answers(sample)
        label = get_label(sample)
        source = str(sample.get("source_dataset", "unknown")).lower().strip()
        labels.append(label)
        sources.append(source)

        if not answers:
            skipped += 1
            canonical_answer_idx.append(-1)
            continue

        no_prompt = base.build_prompt(question, None)
        retrieved_doc = base.extract_doc_text(sample["doc"])
        retrieved_prompt = base.build_prompt(question, retrieved_doc)

        if len(answers) == 1:
            canon_ans = answers[0]
            canon_idx = 0
            no_per_layer, _ = base.compute_per_layer_answer_logp(
                model=model,
                final_norm=final_norm,
                lm_head=lm_head,
                tokenizer=tokenizer,
                prompt_text=no_prompt,
                answer_text=canon_ans,
                device=model_device,
                max_length=max_length,
                softcap=softcap,
            )
        else:
            best_score = -np.inf
            canon_ans = None
            canon_idx = -1
            no_per_layer = None
            for a_idx, ans in enumerate(answers):
                pl, _ = base.compute_per_layer_answer_logp(
                    model=model,
                    final_norm=final_norm,
                    lm_head=lm_head,
                    tokenizer=tokenizer,
                    prompt_text=no_prompt,
                    answer_text=ans,
                    device=model_device,
                    max_length=max_length,
                    softcap=softcap,
                )
                if pl is None:
                    continue
                score = float(np.mean(pl))
                if score > best_score:
                    best_score = score
                    canon_ans = ans
                    canon_idx = a_idx
                    no_per_layer = pl

        canonical_answer_idx.append(canon_idx)
        if canon_ans is None or no_per_layer is None:
            skipped += 1
            continue

        signal["no_ctx"][s_idx, :] = no_per_layer

        retrieved_pl, _ = base.compute_per_layer_answer_logp(
            model=model,
            final_norm=final_norm,
            lm_head=lm_head,
            tokenizer=tokenizer,
            prompt_text=retrieved_prompt,
            answer_text=canon_ans,
            device=model_device,
            max_length=max_length,
            softcap=softcap,
        )
        if retrieved_pl is not None:
            signal["retrieved_ctx"][s_idx, :] = retrieved_pl

    if skipped > 0:
        print(f"  Skipped {skipped} samples with no usable answer")

    labels_arr = np.array(labels, dtype=np.int64)
    sources_arr = np.array(sources)
    masks = {
        "combined": np.ones(n_samples, dtype=bool),
        "relevant": labels_arr == 1,
        "irrelevant": labels_arr == 0,
        "nq": sources_arr == "nq",
        "triviaqa": sources_arr == "triviaqa",
        "nq_relevant": (sources_arr == "nq") & (labels_arr == 1),
        "nq_irrelevant": (sources_arr == "nq") & (labels_arr == 0),
        "triviaqa_relevant": (sources_arr == "triviaqa") & (labels_arr == 1),
        "triviaqa_irrelevant": (sources_arr == "triviaqa") & (labels_arr == 0),
    }

    per_layer = {}
    for cond, arr in signal.items():
        per_layer[cond] = {
            f"mean_logP_{name}": per_subset_layer_mean(arr, mask)
            for name, mask in masks.items()
        }

    delta_relevant_no = per_sample_diff_mean(
        signal["retrieved_ctx"], signal["no_ctx"], masks["relevant"]
    )
    delta_irrelevant_no = per_sample_diff_mean(
        signal["retrieved_ctx"], signal["no_ctx"], masks["irrelevant"]
    )
    delta_by_subset = {
        name: per_sample_diff_mean(signal["retrieved_ctx"], signal["no_ctx"], mask)
        for name, mask in masks.items()
        if name not in {"combined", "relevant", "irrelevant"}
    }

    peak_layer, peak_value = argmax_layer(delta_relevant_no)
    irrelevant_peak_layer, irrelevant_peak_value = argmax_layer(delta_irrelevant_no)
    emerge_layer = first_above(
        per_layer["retrieved_ctx"]["mean_logP_relevant"],
        threshold=-5.0,
    )

    n_with_multiple = sum(1 for s in samples if len(valid_answers(s)) > 1)
    n_picked_nonzero = sum(1 for i in canonical_answer_idx if i > 0)

    payload = {
        "model_name": model_key,
        "hf_name": hf_name,
        "data_path": str(EVAL_PATH.relative_to(PROJECT_ROOT)),
        "dataset_filter": dataset_filter,
        "num_layers": num_layers,
        "num_samples": {
            name: int(mask.sum())
            for name, mask in masks.items()
        },
        "canonical_answer_stats": {
            "n_samples_with_multiple_answers": int(n_with_multiple),
            "n_picked_non_first_answer": int(n_picked_nonzero),
            "max_candidate_answers": int(base.MAX_CANDIDATE_ANSWERS),
        },
        "per_layer": per_layer,
        "delta_retrieved_relevant_minus_no": delta_relevant_no,
        "delta_retrieved_irrelevant_minus_no": delta_irrelevant_no,
        "delta_retrieved_minus_no_by_subset": delta_by_subset,
        "retrieved_relevant_peak_layer": peak_layer,
        "retrieved_relevant_peak_value": peak_value,
        "retrieved_irrelevant_peak_layer": irrelevant_peak_layer,
        "retrieved_irrelevant_peak_value": irrelevant_peak_value,
        "retrieved_relevant_answer_signal_emerge_layer": emerge_layer,
        "top_relevant_layers": top_layers(delta_relevant_no, delta_irrelevant_no),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_suffix = "" if dataset_filter == "all" else f"_{dataset_filter}"
    limit_suffix = f"_limit{max_samples}" if max_samples is not None else ""
    suffix = f"{dataset_suffix}{limit_suffix}"
    out_path = RESULTS_DIR / f"{model_key}_retrieved_logit_lens{suffix}.json"
    save_json(payload, out_path)

    print(f"\n  Layers: {num_layers}")
    print("  Final-layer mean logP:")
    for subset in ["relevant", "irrelevant"]:
        v = per_layer["retrieved_ctx"][f"mean_logP_{subset}"][-1]
        n = int(masks[subset].sum())
        if v is None:
            print(f"    retrieved_{subset:10s}: n={n} logP=None")
        else:
            print(f"    retrieved_{subset:10s}: n={n} logP={v:.4f}  P~{np.exp(v):.4f}")
    print(f"  DeltaP_relevant-no   peak: layer {peak_layer}, value = {peak_value:+.4f}")
    print(
        f"  DeltaP_irrelevant-no peak: layer {irrelevant_peak_layer}, "
        f"value = {irrelevant_peak_value:+.4f}"
    )
    print("  Top relevant layers:")
    for item in payload["top_relevant_layers"]:
        print(
            f"    layer {item['layer']:2d}: "
            f"rel-no={item['delta_relevant_minus_no']:+.4f}  "
            f"irrel-no={item['delta_irrelevant_minus_no']:+.4f}"
        )
    print(f"  Retrieved relevant answer signal emerge layer (logP > -5): {emerge_layer}")
    print(f"  Saved -> {out_path}")

    del model
    torch.cuda.empty_cache()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 10R: retrieved-doc logit lens analysis")
    parser.add_argument(
        "--model",
        required=True,
        choices=["gemma2b", "gemma9b", "qwen3_4b", "all"],
    )
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", "nq", "triviaqa"],
        help="Filter data/final_retrieved/eval.json by source_dataset.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional explicit device, e.g. cuda:0. If omitted, use device_map='auto'.",
    )
    args = parser.parse_args()

    print(f"Requested device: {args.device}")
    models = ["gemma2b", "gemma9b", "qwen3_4b"] if args.model == "all" else [args.model]
    for model_key in models:
        try:
            process_model(
                model_key=model_key,
                dataset_filter=args.dataset,
                max_length=args.max_length,
                max_samples=args.max_samples,
                requested_device=args.device,
            )
        except Exception as exc:
            print(f"\n[{model_key}] FAILED: {exc}")
            if args.model != "all":
                raise
    print("\nDone.")


if __name__ == "__main__":
    main()
