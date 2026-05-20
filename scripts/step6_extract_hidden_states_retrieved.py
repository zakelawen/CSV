#!/usr/bin/env python
"""
Step 6R: Extract hidden states for retrieved-doc classification.

This is the retrieved-doc version of Step6. It reads:
  data/final_retrieved/{train,eval}.json
where each item has one retrieved top-1 document:
  doc + label / label_name

It saves to:
  data/hidden_states_retrieved/{model}/train_hidden_states.pt
  data/hidden_states_retrieved/{model}/eval_hidden_states.pt

Each .pt file contains:
  {
    "hidden_states": Tensor[N, L, D],
    "labels":        List[str],      # "relevant" / "distracting"
    "label_ids":     List[int],      # 1 / 0
    "dataset_source": List[str],     # "nq" / "triviaqa"
    "questions":     List[str],
    "answers":       List[Any],
  }

This file intentionally writes to hidden_states_retrieved so the old
 gold-vs-distractor hidden states in data/hidden_states are not overwritten.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_REGISTRY = {
    "gemma2b": {
        "hf_name": os.environ.get("GEMMA_MODEL_PATH", "google/gemma-2-2b"),
        "save_key": "gemma2b",
    },
    "gemma9b": {
        "hf_name": os.environ.get("GEMMA9B_MODEL_PATH", "google/gemma-2-9b"),
        "save_key": "gemma9b",
    },
    "qwen3_4b": {
        "hf_name": os.environ.get("QWEN3_4B_MODEL_PATH", "Qwen/Qwen3-4B-Base"),
        "save_key": "qwen3_4b",
    },
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FINAL_RETRIEVED_DIR = DATA_DIR / "final_retrieved"
HIDDEN_DIR = DATA_DIR / "hidden_states_retrieved"


def extract_doc_text(doc: Any) -> str:
    """Use only title/text, never score/passage_id/annotation fields."""
    if isinstance(doc, dict):
        title = str(doc.get("title", "")).strip()
        text = str(doc.get("text", "")).strip()
        if title:
            return f"Title: {title}\nText: {text}"
        return text
    return str(doc).strip()


def build_prompt(question: str, document: str) -> str:
    return f"Document: {document}\n\nQuestion: {question}\nAnswer:"


def get_label(item: dict, idx: int) -> tuple[int, str]:
    if "label" in item:
        label = int(item["label"])
    else:
        name = str(item.get("label_name", "")).strip().lower()
        if name == "relevant":
            label = 1
        elif name == "distracting":
            label = 0
        else:
            raise ValueError(f"Item {idx} has invalid label_name={name!r}")
    if label not in (0, 1):
        raise ValueError(f"Item {idx} has invalid label={label!r}; expected 0/1")
    return label, "relevant" if label == 1 else "distracting"


def load_split(split: str) -> list[dict]:
    path = FINAL_RETRIEVED_DIR / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Run scripts/step4b_build_retrieved_dataset.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {path}")
    return data


def prepare_inputs(data: list[dict]) -> tuple[list[str], list[str], list[int], list[str], list[str], list[Any]]:
    prompts = []
    labels = []
    label_ids = []
    dataset_sources = []
    questions = []
    answers = []

    counts = {"relevant": 0, "distracting": 0}
    source_counts = {}

    for idx, item in enumerate(data):
        q = item["question"]
        if "doc" not in item:
            raise KeyError(f"Item {idx} has no 'doc' field. Keys: {list(item.keys())}")
        label_id, label_name = get_label(item, idx)
        source = str(item.get("source_dataset", "unknown")).lower().strip()

        prompts.append(build_prompt(q, extract_doc_text(item["doc"])))
        labels.append(label_name)
        label_ids.append(label_id)
        dataset_sources.append(source)
        questions.append(q)
        answers.append(item.get("answers", []))

        counts[label_name] = counts.get(label_name, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1

    print(f"  labels: relevant={counts.get('relevant', 0)} distracting={counts.get('distracting', 0)}")
    print(f"  sources: {source_counts}")
    return prompts, labels, label_ids, dataset_sources, questions, answers


def check_truncation(tokenizer, prompts: list[str], max_length: int) -> int:
    lengths = []
    truncated_count = 0
    for p in prompts:
        token_ids = tokenizer.encode(p, add_special_tokens=True)
        lengths.append(len(token_ids))
        if len(token_ids) > max_length:
            truncated_count += 1

    lengths_tensor = torch.tensor(lengths, dtype=torch.float)
    print(f"\n  Token length statistics ({len(prompts)} prompts):")
    print(
        f"    min={int(lengths_tensor.min())}, max={int(lengths_tensor.max())}, "
        f"mean={lengths_tensor.mean():.1f}, median={lengths_tensor.median():.1f}"
    )
    print(
        f"    Prompts exceeding max_length ({max_length}): {truncated_count} / {len(prompts)} "
        f"({100*truncated_count/len(prompts):.1f}%)"
    )

    if truncated_count > 0:
        pct = truncated_count / len(prompts)
        if pct > 0.05:
            raise RuntimeError(
                f"{truncated_count} prompts ({pct*100:.1f}%) exceed max_length={max_length}. "
                f"Increase --max_length or inspect final_retrieved. Max observed length: "
                f"{int(lengths_tensor.max())}."
            )
        print("    ⚠ WARNING: a small number of prompts will be truncated.")
    return truncated_count


@torch.no_grad()
def extract_hidden_states(model, tokenizer, prompts: list[str], batch_size: int, max_length: int) -> torch.Tensor:
    device = next(model.parameters()).device
    all_hidden = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="Extracting"):
        batch_texts = prompts[start:start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        reps = [layer_h[:, -1, :] for layer_h in out.hidden_states]
        stacked = torch.stack(reps, dim=1)  # [B, L, D]
        all_hidden.append(stacked.cpu().float())

    return torch.cat(all_hidden, dim=0)


@torch.no_grad()
def run_left_padding_sanity_check(model, tokenizer, max_length: int) -> None:
    print("\n  Running left-padding sanity check …")
    device = next(model.parameters()).device
    short_prompt = "Document: The sky is blue.\n\nQuestion: What color is the sky?\nAnswer:"
    long_prompt = (
        "Document: The Great Wall of China is a series of fortifications that were built "
        "across historical northern borders.\n\nQuestion: What is the Great Wall of China?\nAnswer:"
    )

    def forward_single(text: str) -> torch.Tensor:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        return torch.cat([h[:, -1, :].cpu().float() for h in out.hidden_states], dim=0)

    def forward_batch(texts: list[str]) -> list[torch.Tensor]:
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        per_sample = []
        for i in range(len(texts)):
            per_sample.append(torch.stack([h[i, -1, :].cpu().float() for h in out.hidden_states], dim=0))
        return per_sample

    single_short = forward_single(short_prompt)
    single_long = forward_single(long_prompt)
    batch_short, batch_long = forward_batch([short_prompt, long_prompt])

    cos_short = torch.nn.functional.cosine_similarity(single_short, batch_short, dim=1)
    cos_long = torch.nn.functional.cosine_similarity(single_long, batch_long, dim=1)
    print(f"    Short prompt: min cosine similarity = {cos_short.min().item():.6f} (across {len(cos_short)} layers)")
    print(f"    Long prompt:  min cosine similarity = {cos_long.min().item():.6f} (across {len(cos_long)} layers)")
    if cos_short.min().item() < 0.9999 or cos_long.min().item() < 0.9999:
        raise RuntimeError("Left-padding sanity check failed.")
    print("    ✓ Left-padding sanity check passed.")


def process_model(model_key: str, batch_size: int, max_length: int, overwrite: bool) -> None:
    info = MODEL_REGISTRY[model_key]
    hf_name = info["hf_name"]
    save_key = info["save_key"]
    save_dir = HIDDEN_DIR / save_key
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Processing retrieved-doc hidden states: {model_key} ({hf_name})")
    print(f"Output dir: {save_dir}")
    print(f"{'='*70}")

    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        print(f"  Added dedicated [PAD] token (id={tokenizer.pad_token_id})")
    else:
        print(f"  tokenizer.pad_token already set: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
    tokenizer.padding_side = "left"

    print("Loading model …")
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  num_transformer_layers={num_layers}, hidden_dim={hidden_dim}")
    print(f"  output.hidden_states entries={num_layers + 1} (0=embedding, final={num_layers})")

    run_left_padding_sanity_check(model, tokenizer, max_length)

    for split in ("train", "eval"):
        out_path = save_dir / f"{split}_hidden_states.pt"
        if out_path.exists() and not overwrite:
            print(f"\n  [skip] {out_path} already exists. Use --overwrite to regenerate.")
            continue

        data = load_split(split)
        prompts, labels, label_ids, sources, questions, answers = prepare_inputs(data)
        print(f"  Split={split}: {len(data)} samples → {len(prompts)} forward passes")
        check_truncation(tokenizer, prompts, max_length)

        hidden_states = extract_hidden_states(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            batch_size=batch_size,
            max_length=max_length,
        )
        print(f"  Hidden states shape: {tuple(hidden_states.shape)}")
        assert hidden_states.shape == (len(prompts), num_layers + 1, hidden_dim)

        payload = {
            "hidden_states": hidden_states,
            "labels": labels,
            "label_ids": label_ids,
            "dataset_source": sources,
            "questions": questions,
            "answers": answers,
            "dataset_variant": "retrieved",
            "source_json": str(FINAL_RETRIEVED_DIR / f"{split}.json"),
            "model_key": model_key,
            "hf_name": hf_name,
            "max_length": max_length,
        }
        torch.save(payload, out_path)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  Saved → {out_path} ({size_mb:.0f} MB)")

    del model
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6R: Extract retrieved-doc hidden states")
    parser.add_argument("--model", type=str, default="all", choices=list(MODEL_REGISTRY.keys()) + ["all"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    models = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    for m in models:
        process_model(m, batch_size=args.batch_size, max_length=args.max_length, overwrite=args.overwrite)
    print("\nDone.")


if __name__ == "__main__":
    main()
