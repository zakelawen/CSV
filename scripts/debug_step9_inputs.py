#!/usr/bin/env python
"""
Debug Step9 training inputs.

Purpose:
  Inspect exactly what Step9 uses as training/eval inputs:
    - raw JSON sample
    - extracted document text
    - exact prompt string
    - token ids
    - decoded tokenized input
    - labels
    - right-padded batch from collate_fn
    - attention_mask
    - last non-pad token position

This script imports functions from scripts/step9_train_csv.py and
csv_module/train_utils.py, so prompt construction and padding match Step9.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import step9_train_csv as step9
from csv_module.train_utils import collate_fn


def patch_model_registry_for_wrappers(model_key: str):
    """
    Match the wrapper behavior of:
      scripts/step9_train_csv_qwen3_4b.py
      scripts/step9_train_csv_gemma9b.py

    This lets the debug script inspect the same model/tokenizer choices
    as your Step9 wrapper scripts.
    """
    if model_key == "qwen3_4b":
        step9.MODEL_REGISTRY = {
            "qwen3_4b": {
                "hf_name": os.environ.get(
                    "QWEN3_4B_MODEL_PATH",
                    "Qwen/Qwen3-4B-Base",
                )
            }
        }

    elif model_key == "gemma9b":
        step9.MODEL_REGISTRY = {
            "gemma9b": {
                "hf_name": os.environ.get(
                    "GEMMA9B_MODEL_PATH",
                    "google/gemma-2-9b",
                )
            }
        }

    elif model_key == "gemma2b":
        # Use base step9 registry if present; otherwise define explicitly.
        if "gemma2b" not in step9.MODEL_REGISTRY:
            step9.MODEL_REGISTRY["gemma2b"] = {
                "hf_name": os.environ.get(
                    "GEMMA_MODEL_PATH",
                    "google/gemma-2-2b",
                )
            }

    else:
        raise ValueError(f"Unknown model_key: {model_key}")


def label_name(y: int) -> str:
    return "relevant" if y == 1 else "distracting"


def short(s: str, n: int = 500) -> str:
    s = str(s)
    if len(s) <= n:
        return s
    return s[:n] + f"\n... <truncated, total chars={len(s)}>"


def get_prompt_from_raw_item(item: dict, which: str) -> str:
    """
    Reconstruct the same prompt string as step9.load_and_tokenize().
    This is only for display and verification.

    The actual tokenized prompt used below still comes from:
      step9.load_and_tokenize(split, tokenizer)
    """
    q = item["question"]

    if which == "relevant":
        doc_text = step9.extract_doc_text(item["relevant_doc"])
    elif which == "distracting":
        doc_text = step9.extract_doc_text(item["distracting_doc"])
    else:
        raise ValueError(which)

    return f"Document: {doc_text}\n\nQuestion: {q}\nAnswer:"


def inspect_one_prompt(tokenizer, raw_data, prompts, labels, item_idx: int, which: str):
    """
    item_idx indexes original data/final/{split}.json.
    which is relevant or distracting.

    In Step9:
      prompt_idx = 2 * item_idx     -> relevant, label 1
      prompt_idx = 2 * item_idx + 1 -> distracting, label 0
    """
    assert which in ["relevant", "distracting"]

    prompt_idx = 2 * item_idx if which == "relevant" else 2 * item_idx + 1
    item = raw_data[item_idx]

    token_ids = prompts[prompt_idx]
    y = labels[prompt_idx]

    reconstructed_prompt = get_prompt_from_raw_item(item, which)
    reencoded = tokenizer(reconstructed_prompt, return_tensors="pt").input_ids

    same_tokens = torch.equal(token_ids.cpu(), reencoded.cpu())

    print("=" * 100)
    print(f"ORIGINAL ITEM INDEX: {item_idx}")
    print(f"EXPANDED PROMPT INDEX: {prompt_idx}")
    print(f"TYPE: {which}")
    print(f"LABEL: {y} ({label_name(y)})")
    print(f"Tokenization matches Step9 load_and_tokenize(): {same_tokens}")

    if not same_tokens:
        print("WARNING: reconstructed prompt tokenization differs from Step9 token_ids.")

    print("\nRaw item keys:")
    print(list(item.keys()))

    print("\nQuestion:")
    print(item["question"])

    print("\nAnswers:")
    print(item.get("answers", "<no answers field>"))

    doc_key = "relevant_doc" if which == "relevant" else "distracting_doc"
    doc_obj = item[doc_key]

    print(f"\nRaw {doc_key} type:", type(doc_obj))
    if isinstance(doc_obj, dict):
        print(f"Raw {doc_key} keys:", list(doc_obj.keys()))

    print(f"\nExtracted document text used in prompt ({which}):")
    print(short(step9.extract_doc_text(doc_obj), 1000))

    print("\nExact prompt string used by Step9:")
    print("-" * 100)
    print(short(reconstructed_prompt, 2000))
    print("-" * 100)

    # Check whether extra fields accidentally entered prompt
    if isinstance(doc_obj, dict):
        for forbidden_key in ["score", "passage_id", "annotation", "confidence", "label"]:
            if forbidden_key in doc_obj:
                val = str(doc_obj[forbidden_key])
                appears = val and (val in reconstructed_prompt)
                print(f"Field leakage check: doc['{forbidden_key}'] value appears in prompt? {appears}")

    ids = token_ids.squeeze(0).tolist()

    print("\nToken info:")
    print("  tensor shape:", tuple(token_ids.shape))
    print("  num tokens:", len(ids))
    print("  first 30 token ids:", ids[:30])
    print("  last 30 token ids:", ids[-30:])

    print("\nDecoded tokenized input, skip_special_tokens=False:")
    print("-" * 100)
    print(short(tokenizer.decode(ids, skip_special_tokens=False), 2000))
    print("-" * 100)

    print("\nDecoded tokenized input, skip_special_tokens=True:")
    print("-" * 100)
    print(short(tokenizer.decode(ids, skip_special_tokens=True), 2000))
    print("-" * 100)

    return prompt_idx


def inspect_batch(tokenizer, prompts, labels, prompt_indices, pad_id: int):
    """
    Use the exact collate_fn from csv_module.train_utils.
    This shows the actual tensors that go into model.model().
    """
    batch_prompts = [prompts[i] for i in prompt_indices]
    batch_labels = [labels[i] for i in prompt_indices]

    batch_p, batch_l, attention_mask = collate_fn(
        batch_prompts,
        batch_labels,
        pad_id=pad_id,
    )

    print("\n" + "#" * 100)
    print("BATCH INSPECTION - exact output of collate_fn()")
    print("#" * 100)
    print("prompt_indices:", prompt_indices)
    print("labels:", batch_labels)
    print("labels tensor:", batch_l.tolist())
    print("input_ids shape:", tuple(batch_p.shape))
    print("attention_mask shape:", tuple(attention_mask.shape))
    print("pad_id:", pad_id)
    print("pad token:", tokenizer.pad_token)

    lengths = attention_mask.sum(dim=1).long().tolist()
    print("real lengths from attention_mask:", lengths)

    for row_idx, original_prompt_idx in enumerate(prompt_indices):
        seq = batch_p[row_idx].tolist()
        mask = attention_mask[row_idx].tolist()
        length = lengths[row_idx]

        last_real_pos = length - 1
        last_real_token_id = seq[last_real_pos]
        last_real_token = tokenizer.decode([last_real_token_id], skip_special_tokens=False)

        pad_after = seq[length:]
        all_pad_after = all(x == pad_id for x in pad_after)

        print("\n" + "-" * 100)
        print(f"Batch row {row_idx}, original prompt_idx={original_prompt_idx}")
        print(f"label={batch_labels[row_idx]} ({label_name(batch_labels[row_idx])})")
        print(f"length={length}, last_real_pos={last_real_pos}")
        print(f"last_real_token_id={last_real_token_id}, decoded={repr(last_real_token)}")
        print(f"all tokens after length are pad_id? {all_pad_after}")
        print(f"num pad tokens after real sequence: {len(pad_after)}")
        print("attention_mask first 80:", mask[:80])
        print("attention_mask last 80:", mask[-80:])
        print("input_ids first 40:", seq[:40])
        print("input_ids last 40:", seq[-40:])

        decoded_real = tokenizer.decode(seq[:length], skip_special_tokens=False)
        print("\nDecoded real sequence only:")
        print(short(decoded_real, 1200))


def build_prompt_indices(args, raw_data):
    """
    Build prompt indices to inspect.

    If item_indices are given:
      inspect relevant and distracting for each item.

    If prompt_indices are given:
      inspect those expanded prompt indices directly.

    If simulate_first_train_batch is enabled:
      mimic Step9's per-epoch permutation logic with a seed.
      Note: Step9 currently uses torch.randperm(num_train) without a fixed
      seed, so exact live-run order is random unless you set the same seed
      inside Step9. This simulation is for inspecting what a shuffled batch
      looks like.
    """
    n_items = len(raw_data)
    n_prompts = 2 * n_items

    if args.prompt_indices:
        out = []
        for x in args.prompt_indices.split(","):
            x = x.strip()
            if x:
                idx = int(x)
                assert 0 <= idx < n_prompts, f"prompt_idx {idx} out of range 0..{n_prompts-1}"
                out.append(idx)
        return out

    if args.simulate_first_train_batch:
        g = torch.Generator()
        g.manual_seed(args.seed)
        perm = torch.randperm(n_prompts, generator=g).tolist()
        return perm[: args.batch_size]

    # Default: original item indices
    item_indices = []
    for x in args.item_indices.split(","):
        x = x.strip()
        if x:
            idx = int(x)
            assert 0 <= idx < n_items, f"item_idx {idx} out of range 0..{n_items-1}"
            item_indices.append(idx)

    out = []
    for item_idx in item_indices:
        out.append(2 * item_idx)      # relevant
        out.append(2 * item_idx + 1)  # distracting

    return out


def main():
    parser = argparse.ArgumentParser(description="Inspect exact Step9 training inputs.")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3_4b",
        choices=["qwen3_4b", "gemma2b", "gemma9b"],
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "eval"],
    )
    parser.add_argument(
        "--item_indices",
        type=str,
        default="0,1",
        help="Original sample indices in data/final/{split}.json. "
             "Each item expands to relevant + distracting. Default: 0,1",
    )
    parser.add_argument(
        "--prompt_indices",
        type=str,
        default=None,
        help="Expanded prompt indices directly. Overrides --item_indices. "
             "Example: 0,1,2,3",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for collate inspection.",
    )
    parser.add_argument(
        "--simulate_first_train_batch",
        action="store_true",
        help="Simulate a shuffled first training batch using torch.randperm with --seed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used only for --simulate_first_train_batch.",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Optional path to save a compact JSON summary.",
    )
    args = parser.parse_args()

    patch_model_registry_for_wrappers(args.model)

    hf_name = step9.MODEL_REGISTRY[args.model]["hf_name"]

    print("=" * 100)
    print("DEBUG STEP9 INPUTS")
    print("=" * 100)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("model:", args.model)
    print("hf_name:", hf_name)
    print("split:", args.split)

    print("\nSetting up tokenizer using step9.setup_tokenizer() ...")
    tokenizer = step9.setup_tokenizer(hf_name)
    pad_id = tokenizer.pad_token_id

    print("tokenizer class:", type(tokenizer).__name__)
    print("pad_token:", repr(tokenizer.pad_token))
    print("pad_token_id:", pad_id)
    print("eos_token:", repr(tokenizer.eos_token))
    print("eos_token_id:", tokenizer.eos_token_id)
    print("bos_token:", repr(tokenizer.bos_token))
    print("bos_token_id:", tokenizer.bos_token_id)

    raw_path = step9.DATA_DIR / f"{args.split}.json"
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print("\nLoading prompts using step9.load_and_tokenize() ...")
    prompts, labels = step9.load_and_tokenize(args.split, tokenizer)

    print("\nDataset summary:")
    print("raw json path:", raw_path)
    print("num raw samples:", len(raw_data))
    print("num expanded prompts:", len(prompts))
    print("label counts:", {0: labels.count(0), 1: labels.count(1)})

    assert len(prompts) == 2 * len(raw_data), "Expected 2 prompts per raw sample."
    assert len(labels) == len(prompts), "labels/prompts length mismatch."

    prompt_indices = build_prompt_indices(args, raw_data)
    print("\nPrompt indices selected for inspection:", prompt_indices)

    compact_records = []

    # Inspect individual prompts
    for prompt_idx in prompt_indices:
        item_idx = prompt_idx // 2
        which = "relevant" if prompt_idx % 2 == 0 else "distracting"

        inspect_one_prompt(
            tokenizer=tokenizer,
            raw_data=raw_data,
            prompts=prompts,
            labels=labels,
            item_idx=item_idx,
            which=which,
        )

        compact_records.append({
            "split": args.split,
            "item_idx": item_idx,
            "prompt_idx": prompt_idx,
            "which": which,
            "label": labels[prompt_idx],
            "num_tokens": int(prompts[prompt_idx].size(1)),
            "question": raw_data[item_idx]["question"],
            "doc_keys": list(raw_data[item_idx][f"{which}_doc"].keys())
                        if isinstance(raw_data[item_idx].get(f"{which}_doc"), dict)
                        else None,
        })

    # Inspect collated batch. If too many selected, use first batch_size.
    batch_indices = prompt_indices[: args.batch_size]
    inspect_batch(
        tokenizer=tokenizer,
        prompts=prompts,
        labels=labels,
        prompt_indices=batch_indices,
        pad_id=pad_id,
    )

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": args.model,
            "hf_name": hf_name,
            "split": args.split,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": pad_id,
            "num_raw_samples": len(raw_data),
            "num_expanded_prompts": len(prompts),
            "label_counts": {str(k): labels.count(k) for k in [0, 1]},
            "records": compact_records,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved compact summary -> {out_path}")


if __name__ == "__main__":
    main()