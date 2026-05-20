# """
# Step 6: Extract Hidden States from LLMs

# For each sample in train.json / eval.json, construct two inputs:
#   - (query, relevant_doc)   -> label "relevant"
#   - (query, distracting_doc) -> label "distracting"

# Run a forward pass through the base LLM with output_hidden_states=True,
# extract the last token's hidden state at every layer, and save to disk.

# Usage:
#     python scripts/step6_extract_hidden_states.py --model gemma2b
#     python scripts/step6_extract_hidden_states.py --model llama8b
#     python scripts/step6_extract_hidden_states.py --model all

# Output (per model):
#     data/hidden_states/{model_key}/train_hidden_states.pt
#     data/hidden_states/{model_key}/eval_hidden_states.pt

# Each .pt file contains:
#     {
#         "hidden_states": Tensor[N, L, D],   # N=samples, L=num_layers+1, D=hidden_dim
#         "labels":        List[str],          # "relevant" / "distracting"
#         "dataset_source": List[str],         # "nq" / "triviaqa"
#         "questions":     List[str],          # original query text
#     }

# Layer indexing convention (important for downstream steps):
#     output.hidden_states returns num_layers+1 tensors:
#       index 0         -> embedding layer output
#       index 1 .. N    -> transformer layer 1 .. N output
#       index -1 (= N)  -> final transformer layer output

#     TSV code uses two different indexing schemes:
#       - add_tsv_layers() uses str_layer based on model.layers (0-based transformer layers)
#         -> str_layer=9 means the 10th transformer layer
#         -> corresponds to hidden_states[10] in output.hidden_states
#       - train_model()/test_model() uses layer_number=-1 for the final layer

#     We save ALL layers (embedding + transformer) to keep maximum flexibility.
#     Downstream code should be aware of this +1 offset.
# """

# import os
# import json
# import argparse
# from pathlib import Path

# import torch
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM


# # ---------------------------------------------------------------------------
# # Model registry
# # ---------------------------------------------------------------------------

# MODEL_REGISTRY = {
#     "gemma2b": {
#         "hf_name": "google/gemma-2-2b",
#         "save_key": "gemma2b",
#     },
#     "llama8b": {
#         "hf_name": "meta-llama/Meta-Llama-3.1-8B",
#         "save_key": "llama8b",
#     },
# }


# # ---------------------------------------------------------------------------
# # Paths
# # ---------------------------------------------------------------------------

# PROJECT_ROOT = Path(__file__).resolve().parent.parent  # rag-probe/
# DATA_DIR = PROJECT_ROOT / "data"
# FINAL_DIR = DATA_DIR / "final"
# HIDDEN_DIR = DATA_DIR / "hidden_states"


# # ---------------------------------------------------------------------------
# # Prompt template
# # ---------------------------------------------------------------------------

# def extract_doc_text(doc) -> str:
#     """
#     Extract document text from either a dict or a plain string.

#     train.json stores docs as dicts with different keys:
#       relevant_doc:    {'text': ..., 'title': ...}
#       distracting_doc: {'text': ..., 'title': ..., 'score': ..., 'passage_id': ...}

#     We only use 'title' and 'text' to build the prompt, ensuring both
#     document types produce the same format (no label shortcut from extra keys).
#     """
#     if isinstance(doc, dict):
#         title = str(doc.get("title", "")).strip()
#         text = str(doc.get("text", "")).strip()
#         if title:
#             return f"Title: {title}\nText: {text}"
#         return text
#     return str(doc).strip()


# def build_prompt(question: str, document: str) -> str:
#     """
#     Construct the RAG-style prompt fed to the base LLM.
#     Base models have no chat template, so we use a simple format.
#     """
#     return (
#         f"Document: {document}\n\n"
#         f"Question: {question}\n"
#         f"Answer:"
#     )


# # ---------------------------------------------------------------------------
# # Data loading
# # ---------------------------------------------------------------------------

# def load_split(split: str) -> list[dict]:
#     """Load train.json or eval.json and return the list of records."""
#     path = FINAL_DIR / f"{split}.json"
#     if not path.exists():
#         raise FileNotFoundError(f"Cannot find {path}. Run step1-5 first.")
#     with open(path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     print(f"Loaded {len(data)} samples from {path}")
#     return data


# def prepare_inputs(data: list[dict]) -> tuple[list[str], list[str], list[str], list[str]]:
#     """
#     Expand each sample into two entries (relevant + distracting).

#     Returns:
#         prompts:         list of prompt strings  (length = 2 * len(data))
#         labels:          "relevant" / "distracting"
#         dataset_sources: "nq" / "triviaqa"
#         questions:       original question text
#     """
#     prompts = []
#     labels = []
#     dataset_sources = []
#     questions = []

#     for item in data:
#         q = item["question"]
#         source = item.get("source_dataset", "unknown")

#         # --- relevant doc ---
#         prompts.append(build_prompt(q, extract_doc_text(item["relevant_doc"])))
#         labels.append("relevant")
#         dataset_sources.append(source)
#         questions.append(q)

#         # --- distracting doc ---
#         prompts.append(build_prompt(q, extract_doc_text(item["distracting_doc"])))
#         labels.append("distracting")
#         dataset_sources.append(source)
#         questions.append(q)

#     return prompts, labels, dataset_sources, questions


# # ---------------------------------------------------------------------------
# # Truncation check
# # ---------------------------------------------------------------------------

# def check_truncation(tokenizer, prompts: list[str], max_length: int) -> int:
#     """
#     Check how many prompts exceed max_length tokens.
#     Returns the count of truncated prompts.
#     Prints statistics and raises an error if too many are truncated.
#     """
#     lengths = []
#     truncated_count = 0

#     for p in prompts:
#         token_ids = tokenizer.encode(p, add_special_tokens=True)
#         lengths.append(len(token_ids))
#         if len(token_ids) > max_length:
#             truncated_count += 1

#     lengths_tensor = torch.tensor(lengths, dtype=torch.float)
#     print(f"\n  Token length statistics ({len(prompts)} prompts):")
#     print(f"    min={int(lengths_tensor.min())}, "
#           f"max={int(lengths_tensor.max())}, "
#           f"mean={lengths_tensor.mean():.1f}, "
#           f"median={lengths_tensor.median():.1f}")
#     print(f"    Prompts exceeding max_length ({max_length}): "
#           f"{truncated_count} / {len(prompts)} "
#           f"({100*truncated_count/len(prompts):.1f}%)")

#     if truncated_count > 0:
#         pct = truncated_count / len(prompts)
#         if pct > 0.05:
#             raise RuntimeError(
#                 f"{truncated_count} prompts ({pct*100:.1f}%) exceed max_length={max_length}. "
#                 f"This is too many - increase --max_length or check your data. "
#                 f"Max observed length: {int(lengths_tensor.max())} tokens."
#             )
#         else:
#             print(f"    WARNING WARNING: {truncated_count} prompts will be truncated. "
#                   f"These samples' last-token hidden states may not be meaningful. "
#                   f"Consider increasing --max_length.")

#     return truncated_count


# # ---------------------------------------------------------------------------
# # Hidden-state extraction
# # ---------------------------------------------------------------------------

# @torch.no_grad()
# def extract_hidden_states(
#     model,
#     tokenizer,
#     prompts: list[str],
#     batch_size: int = 4,
#     max_length: int = 1024,
# ) -> torch.Tensor:
#     """
#     Run forward passes and collect the last-token hidden state at every layer.

#     We use LEFT padding so that the last token of every sequence in the batch
#     is always at position seq_len-1, regardless of sequence length.
#     This simplifies extraction: just take hidden_states[:, -1, :].

#     The attention_mask is passed to the model so that real tokens do NOT
#     attend to left-padding tokens. HuggingFace causal LMs handle this
#     correctly when attention_mask is provided.

#     We call model.model() (the inner transformer, e.g. Gemma2Model or
#     LlamaModel) instead of model() (Gemma2ForCausalLM) to skip the
#     lm_head projection. This avoids allocating a huge
#     [batch x seq_len x vocab_size] logits tensor, saving significant VRAM.

#     Returns:
#         Tensor of shape [N, L, D]
#         where N = len(prompts), L = num_layers + 1 (embedding + transformer layers),
#         D = hidden_size.
#     """
#     device = next(model.parameters()).device
#     all_hidden = []

#     for start in tqdm(range(0, len(prompts), batch_size), desc="Extracting"):
#         batch_texts = prompts[start : start + batch_size]

#         encodings = tokenizer(
#             batch_texts,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=max_length,
#         )
#         input_ids = encodings["input_ids"].to(device)
#         attention_mask = encodings["attention_mask"].to(device)

#         with torch.amp.autocast("cuda", dtype=torch.float16):
#             # Use model.model (inner transformer) to skip lm_head logits computation.
#             # This returns BaseModelOutputWithPast with .hidden_states but no .logits.
#             outputs = model.model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 output_hidden_states=True,
#             )

#         # outputs.hidden_states: tuple of (num_layers+1) tensors,
#         # each [B, seq_len, D].
#         #
#         # With left padding, real tokens are right-aligned:
#         #   [PAD, PAD, ..., tok1, tok2, ..., tokN]
#         # So the last token is always at position -1 for all sequences.
#         hidden_states = outputs.hidden_states

#         # Gather last-token hidden state for every layer -> [B, L, D]
#         layer_reps = []
#         for layer_h in hidden_states:
#             # layer_h: [B, seq_len, D]
#             # With left padding, last position is always the last real token.
#             selected = layer_h[:, -1, :]  # [B, D]
#             layer_reps.append(selected)

#         stacked = torch.stack(layer_reps, dim=1)  # [B, L, D]
#         all_hidden.append(stacked.cpu().float())   # store as fp32 for downstream

#     return torch.cat(all_hidden, dim=0)  # [N, L, D]


# # ---------------------------------------------------------------------------
# # Left-padding sanity check
# # ---------------------------------------------------------------------------

# @torch.no_grad()
# def _run_padding_sanity_check(model, tokenizer, max_length: int):
#     """
#     Verify that left-padding produces the same last-token hidden state as
#     no-padding (batch_size=1).

#     We create two prompts of different lengths, run them individually
#     (no padding needed) and as a batch (left-padding applied to the shorter
#     one), then compare the last-token hidden states at every layer.

#     If cosine similarity < 0.9999 at any layer, something is wrong with
#     the model's handling of left-padding + attention_mask.
#     """
#     print("\n  Running left-padding sanity check ...")

#     device = next(model.parameters()).device

#     short_prompt = "Document: The sky is blue.\n\nQuestion: What color is the sky?\nAnswer:"
#     long_prompt = (
#         "Document: The Great Wall of China is a series of fortifications "
#         "that were built across the historical northern borders of ancient "
#         "Chinese states and Imperial China.\n\n"
#         "Question: What is the Great Wall of China?\nAnswer:"
#     )

#     def _forward_single(text):
#         enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
#         ids = enc["input_ids"].to(device)
#         mask = enc["attention_mask"].to(device)
#         with torch.amp.autocast("cuda", dtype=torch.float16):
#             out = model.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
#         # batch_size=1, no padding -> last token is at position -1
#         reps = [h[:, -1, :].cpu().float() for h in out.hidden_states]
#         return torch.cat(reps, dim=0)  # [L, D]

#     def _forward_batch(texts):
#         enc = tokenizer(texts, return_tensors="pt", padding=True,
#                         truncation=True, max_length=max_length)
#         ids = enc["input_ids"].to(device)
#         mask = enc["attention_mask"].to(device)
#         with torch.amp.autocast("cuda", dtype=torch.float16):
#             out = model.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
#         # left padding -> last token at position -1 for all sequences
#         reps_per_sample = []
#         for sample_idx in range(len(texts)):
#             reps = [h[sample_idx, -1, :].cpu().float() for h in out.hidden_states]
#             reps_per_sample.append(torch.stack(reps, dim=0))  # [L, D]
#         return reps_per_sample

#     # Single forward passes (ground truth, no padding)
#     single_short = _forward_single(short_prompt)  # [L, D]
#     single_long = _forward_single(long_prompt)    # [L, D]

#     # Batched forward pass (left-padding applied to short_prompt)
#     batched = _forward_batch([short_prompt, long_prompt])
#     batch_short = batched[0]  # [L, D]
#     batch_long = batched[1]   # [L, D]

#     # Compare via cosine similarity at each layer
#     cos_short = torch.nn.functional.cosine_similarity(single_short, batch_short, dim=1)  # [L]
#     cos_long = torch.nn.functional.cosine_similarity(single_long, batch_long, dim=1)     # [L]

#     min_cos_short = cos_short.min().item()
#     min_cos_long = cos_long.min().item()

#     print(f"    Short prompt: min cosine similarity = {min_cos_short:.6f} "
#           f"(across {len(cos_short)} layers)")
#     print(f"    Long prompt:  min cosine similarity = {min_cos_long:.6f} "
#           f"(across {len(cos_long)} layers)")

#     threshold = 0.9999
#     if min_cos_short < threshold or min_cos_long < threshold:
#         raise RuntimeError(
#             f"Left-padding sanity check FAILED.\n"
#             f"  Short prompt min cosine: {min_cos_short:.6f}\n"
#             f"  Long prompt min cosine:  {min_cos_long:.6f}\n"
#             f"  Threshold: {threshold}\n"
#             f"This model may not correctly handle left-padding with attention_mask. "
#             f"Consider switching to right-padding with explicit last-token indexing."
#         )

#     print("    OK Left-padding sanity check passed.")


# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------

# def process_model(model_key: str, batch_size: int, max_length: int):
#     """Load one model, extract hidden states for train + eval, save to disk."""

#     info = MODEL_REGISTRY[model_key]
#     hf_name = info["hf_name"]
#     save_key = info["save_key"]
#     save_dir = HIDDEN_DIR / save_key
#     save_dir.mkdir(parents=True, exist_ok=True)

#     print(f"\n{'='*60}")
#     print(f"Processing model: {model_key}  ({hf_name})")
#     print(f"{'='*60}")

#     # ---- Load tokenizer ----
#     print("Loading tokenizer ...")
#     tokenizer = AutoTokenizer.from_pretrained(hf_name)

#     # --- Pad token setup ---
#     # Base models often have no pad token. We add a dedicated pad token
#     # rather than reusing eos_token, because with left padding the pad
#     # tokens appear on the LEFT side of the sequence and real tokens CAN
#     # attend to them via the causal mask (they are at earlier positions).
#     # Using eos_token as pad would inject semantically meaningful tokens
#     # into those positions, potentially contaminating hidden states.
#     #
#     # Adding a new token with a random embedding is safe because:
#     # 1. attention_mask=0 for pad positions -> model ignores them in attention
#     # 2. We never read hidden states at pad positions
#     if tokenizer.pad_token is None:
#         tokenizer.add_special_tokens({"pad_token": "[PAD]"})
#         print(f"  Added dedicated [PAD] token (id={tokenizer.pad_token_id})")

#     # Left padding: all sequences right-aligned, last token always at position -1.
#     tokenizer.padding_side = "left"

#     # ---- Load model ----
#     print("Loading model ...")
#     model = AutoModelForCausalLM.from_pretrained(
#         hf_name,
#         dtype=torch.float16,
#         device_map="auto",
#         low_cpu_mem_usage=True,
#     )

#     # Resize embeddings to accommodate the new [PAD] token.
#     # The new token gets a randomly initialized embedding, which is fine
#     # because it's masked out in attention and we never read its hidden state.
#     model.resize_token_embeddings(len(tokenizer))
#     model.eval()

#     # Print model info for reference
#     num_layers = model.config.num_hidden_layers
#     hidden_dim = model.config.hidden_size
#     print(f"  num_transformer_layers={num_layers}, hidden_dim={hidden_dim}")
#     print(f"  output.hidden_states will have {num_layers + 1} entries "
#           f"(index 0=embedding, 1..{num_layers}=transformer layers)")

#     # ---- Left-padding sanity check ----
#     # Verify that left-padding + attention_mask produces identical hidden
#     # states to no-padding (batch_size=1).  This catches any model-specific
#     # issues with Flash Attention or causal mask handling.
#     _run_padding_sanity_check(model, tokenizer, max_length)

#     # ---- Process each split ----
#     for split in ("train", "eval"):
#         out_path = save_dir / f"{split}_hidden_states.pt"
#         if out_path.exists():
#             print(f"\n  [skip] {out_path} already exists.")
#             continue

#         data = load_split(split)
#         prompts, labels, sources, questions = prepare_inputs(data)
#         print(f"  Split={split}: {len(data)} samples -> {len(prompts)} forward passes")

#         # Check for truncation before running expensive forward passes
#         check_truncation(tokenizer, prompts, max_length)

#         hidden_states = extract_hidden_states(
#             model, tokenizer, prompts,
#             batch_size=batch_size,
#             max_length=max_length,
#         )
#         print(f"  Hidden states shape: {hidden_states.shape}")
#         # Expected: [2*len(data), num_layers+1, hidden_dim]

#         assert hidden_states.shape[0] == len(prompts), \
#             f"Sample count mismatch: {hidden_states.shape[0]} vs {len(prompts)}"
#         assert hidden_states.shape[1] == num_layers + 1, \
#             f"Layer count mismatch: {hidden_states.shape[1]} vs {num_layers + 1}"
#         assert hidden_states.shape[2] == hidden_dim, \
#             f"Hidden dim mismatch: {hidden_states.shape[2]} vs {hidden_dim}"

#         payload = {
#             "hidden_states": hidden_states,
#             "labels": labels,
#             "dataset_source": sources,
#             "questions": questions,
#         }
#         torch.save(payload, out_path)
#         size_mb = out_path.stat().st_size / (1024 * 1024)
#         print(f"  Saved -> {out_path}  ({size_mb:.0f} MB)")

#     # ---- Free GPU memory before next model ----
#     del model
#     torch.cuda.empty_cache()


# def main():
#     parser = argparse.ArgumentParser(description="Step 6: Extract hidden states")
#     parser.add_argument(
#         "--model",
#         type=str,
#         default="all",
#         choices=list(MODEL_REGISTRY.keys()) + ["all"],
#         help="Which model to process (default: all)",
#     )
#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=4,
#         help=(
#             "Batch size for forward pass. "
#             "Gemma-2B (~4GB VRAM) can use 16-32 on a 24GB GPU. "
#             "LLaMA-8B (~16GB VRAM) should use 2-4 on a 24GB GPU. "
#             "Reduce if OOM."
#         ),
#     )
#     parser.add_argument(
#         "--max_length",
#         type=int,
#         default=1024,
#         help=(
#             "Max token length. Prompts exceeding this will cause an error "
#             "if >5%% are affected, or a warning otherwise."
#         ),
#     )
#     args = parser.parse_args()

#     models = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

#     for m in models:
#         process_model(m, batch_size=args.batch_size, max_length=args.max_length)

#     print("\nDone.")


# if __name__ == "__main__":
#     main()

"""
Step 6: Extract Hidden States from LLMs

For each sample in train.json / eval.json, construct two inputs:
  - (query, relevant_doc)   -> label "relevant"
  - (query, distracting_doc) -> label "distracting"

Run a forward pass through the base LLM with output_hidden_states=True,
extract the last token's hidden state at every layer, and save to disk.

Usage:
    python scripts/step6_extract_hidden_states.py --model gemma2b
    python scripts/step6_extract_hidden_states.py --model gemma9b
    python scripts/step6_extract_hidden_states.py --model qwen3_4b
    python scripts/step6_extract_hidden_states.py --model all

Output (per model):
    data/hidden_states/{model_key}/train_hidden_states.pt
    data/hidden_states/{model_key}/eval_hidden_states.pt

Each .pt file contains:
    {
        "hidden_states": Tensor[N, L, D],   # N=samples, L=num_layers+1, D=hidden_dim
        "labels":        List[str],          # "relevant" / "distracting"
        "dataset_source": List[str],         # "nq" / "triviaqa"
        "questions":     List[str],          # original query text
    }

Layer indexing convention (important for downstream steps):
    output.hidden_states returns num_layers+1 tensors:
      index 0         -> embedding layer output
      index 1 .. N    -> transformer layer 1 .. N output
      index -1 (= N)  -> final transformer layer output

    TSV code uses two different indexing schemes:
      - add_tsv_layers() uses str_layer based on model.layers (0-based transformer layers)
        -> str_layer=9 means the 10th transformer layer
        -> corresponds to hidden_states[10] in output.hidden_states
      - train_model()/test_model() uses layer_number=-1 for the final layer

    We save ALL layers (embedding + transformer) to keep maximum flexibility.
    Downstream code should be aware of this +1 offset.
"""

import os
import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "gemma2b": {
        "hf_name": os.environ.get(
            "GEMMA_MODEL_PATH",
            "google/gemma-2-2b",
        ),
        "save_key": "gemma2b",
    },
    "gemma9b": {
        "hf_name": os.environ.get(
            "GEMMA9B_MODEL_PATH",
            "google/gemma-2-9b",
        ),
        "save_key": "gemma9b",
    },
    "qwen3_4b": {
        "hf_name": os.environ.get(
            "QWEN3_4B_MODEL_PATH",
            "Qwen/Qwen3-4B-Base",
        ),
        "save_key": "qwen3_4b",
    },
}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # rag-probe/
DATA_DIR = PROJECT_ROOT / "data"
FINAL_DIR = DATA_DIR / "final"
HIDDEN_DIR = DATA_DIR / "hidden_states"


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

def extract_doc_text(doc) -> str:
    """
    Extract document text from either a dict or a plain string.

    train.json stores docs as dicts with different keys:
      relevant_doc:    {'text': ..., 'title': ...}
      distracting_doc: {'text': ..., 'title': ..., 'score': ..., 'passage_id': ...}

    We only use 'title' and 'text' to build the prompt, ensuring both
    document types produce the same format (no label shortcut from extra keys).
    """
    if isinstance(doc, dict):
        title = str(doc.get("title", "")).strip()
        text = str(doc.get("text", "")).strip()
        if title:
            return f"Title: {title}\nText: {text}"
        return text
    return str(doc).strip()


def build_prompt(question: str, document: str) -> str:
    """
    Construct the RAG-style prompt fed to the base LLM.
    Base models have no chat template, so we use a simple format.
    """
    return (
        f"Document: {document}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(split: str) -> list[dict]:
    """Load train.json or eval.json and return the list of records."""
    path = FINAL_DIR / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}. Run step1-5 first.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {path}")
    return data


def prepare_inputs(data: list[dict]) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Expand each sample into two entries (relevant + distracting).

    Returns:
        prompts:         list of prompt strings  (length = 2 * len(data))
        labels:          "relevant" / "distracting"
        dataset_sources: "nq" / "triviaqa"
        questions:       original question text
    """
    prompts = []
    labels = []
    dataset_sources = []
    questions = []

    for item in data:
        q = item["question"]
        source = item.get("source_dataset", "unknown")

        # --- relevant doc ---
        prompts.append(build_prompt(q, extract_doc_text(item["relevant_doc"])))
        labels.append("relevant")
        dataset_sources.append(source)
        questions.append(q)

        # --- distracting doc ---
        prompts.append(build_prompt(q, extract_doc_text(item["distracting_doc"])))
        labels.append("distracting")
        dataset_sources.append(source)
        questions.append(q)

    return prompts, labels, dataset_sources, questions


# ---------------------------------------------------------------------------
# Truncation check
# ---------------------------------------------------------------------------

def check_truncation(tokenizer, prompts: list[str], max_length: int) -> int:
    """
    Check how many prompts exceed max_length tokens.
    Returns the count of truncated prompts.
    Prints statistics and raises an error if too many are truncated.
    """
    lengths = []
    truncated_count = 0

    for p in prompts:
        token_ids = tokenizer.encode(p, add_special_tokens=True)
        lengths.append(len(token_ids))
        if len(token_ids) > max_length:
            truncated_count += 1

    lengths_tensor = torch.tensor(lengths, dtype=torch.float)
    print(f"\n  Token length statistics ({len(prompts)} prompts):")
    print(f"    min={int(lengths_tensor.min())}, "
          f"max={int(lengths_tensor.max())}, "
          f"mean={lengths_tensor.mean():.1f}, "
          f"median={lengths_tensor.median():.1f}")
    print(f"    Prompts exceeding max_length ({max_length}): "
          f"{truncated_count} / {len(prompts)} "
          f"({100*truncated_count/len(prompts):.1f}%)")

    if truncated_count > 0:
        pct = truncated_count / len(prompts)
        if pct > 0.05:
            raise RuntimeError(
                f"{truncated_count} prompts ({pct*100:.1f}%) exceed max_length={max_length}. "
                f"This is too many - increase --max_length or check your data. "
                f"Max observed length: {int(lengths_tensor.max())} tokens."
            )
        else:
            print(f"    WARNING WARNING: {truncated_count} prompts will be truncated. "
                  f"These samples' last-token hidden states may not be meaningful. "
                  f"Consider increasing --max_length.")

    return truncated_count


# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_hidden_states(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int = 4,
    max_length: int = 1024,
) -> torch.Tensor:
    """
    Run forward passes and collect the last-token hidden state at every layer.

    We use LEFT padding so that the last token of every sequence in the batch
    is always at position seq_len-1, regardless of sequence length.
    This simplifies extraction: just take hidden_states[:, -1, :].

    The attention_mask is passed to the model so that real tokens do NOT
    attend to left-padding tokens. HuggingFace causal LMs handle this
    correctly when attention_mask is provided.

    We call model.model() (the inner transformer, e.g. Gemma2Model or
    LlamaModel) instead of model() (Gemma2ForCausalLM) to skip the
    lm_head projection. This avoids allocating a huge
    [batch x seq_len x vocab_size] logits tensor, saving significant VRAM.

    Returns:
        Tensor of shape [N, L, D]
        where N = len(prompts), L = num_layers + 1 (embedding + transformer layers),
        D = hidden_size.
    """
    device = next(model.parameters()).device
    all_hidden = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="Extracting"):
        batch_texts = prompts[start : start + batch_size]

        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            # Use model.model (inner transformer) to skip lm_head logits computation.
            # This returns BaseModelOutputWithPast with .hidden_states but no .logits.
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # outputs.hidden_states: tuple of (num_layers+1) tensors,
        # each [B, seq_len, D].
        #
        # With left padding, real tokens are right-aligned:
        #   [PAD, PAD, ..., tok1, tok2, ..., tokN]
        # So the last token is always at position -1 for all sequences.
        hidden_states = outputs.hidden_states

        # Gather last-token hidden state for every layer -> [B, L, D]
        layer_reps = []
        for layer_h in hidden_states:
            # layer_h: [B, seq_len, D]
            # With left padding, last position is always the last real token.
            selected = layer_h[:, -1, :]  # [B, D]
            layer_reps.append(selected)

        stacked = torch.stack(layer_reps, dim=1)  # [B, L, D]
        all_hidden.append(stacked.cpu().float())   # store as fp32 for downstream

    return torch.cat(all_hidden, dim=0)  # [N, L, D]


# ---------------------------------------------------------------------------
# Left-padding sanity check
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_padding_sanity_check(model, tokenizer, max_length: int):
    """
    Verify that left-padding produces the same last-token hidden state as
    no-padding (batch_size=1).

    We create two prompts of different lengths, run them individually
    (no padding needed) and as a batch (left-padding applied to the shorter
    one), then compare the last-token hidden states at every layer.

    If cosine similarity < 0.9999 at any layer, something is wrong with
    the model's handling of left-padding + attention_mask.
    """
    print("\n  Running left-padding sanity check ...")

    device = next(model.parameters()).device

    short_prompt = "Document: The sky is blue.\n\nQuestion: What color is the sky?\nAnswer:"
    long_prompt = (
        "Document: The Great Wall of China is a series of fortifications "
        "that were built across the historical northern borders of ancient "
        "Chinese states and Imperial China.\n\n"
        "Question: What is the Great Wall of China?\nAnswer:"
    )

    def _forward_single(text):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        # batch_size=1, no padding -> last token is at position -1
        reps = [h[:, -1, :].cpu().float() for h in out.hidden_states]
        return torch.cat(reps, dim=0)  # [L, D]

    def _forward_batch(texts):
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        # left padding -> last token at position -1 for all sequences
        reps_per_sample = []
        for sample_idx in range(len(texts)):
            reps = [h[sample_idx, -1, :].cpu().float() for h in out.hidden_states]
            reps_per_sample.append(torch.stack(reps, dim=0))  # [L, D]
        return reps_per_sample

    # Single forward passes (ground truth, no padding)
    single_short = _forward_single(short_prompt)  # [L, D]
    single_long = _forward_single(long_prompt)    # [L, D]

    # Batched forward pass (left-padding applied to short_prompt)
    batched = _forward_batch([short_prompt, long_prompt])
    batch_short = batched[0]  # [L, D]
    batch_long = batched[1]   # [L, D]

    # Compare via cosine similarity at each layer
    cos_short = torch.nn.functional.cosine_similarity(single_short, batch_short, dim=1)  # [L]
    cos_long = torch.nn.functional.cosine_similarity(single_long, batch_long, dim=1)     # [L]

    min_cos_short = cos_short.min().item()
    min_cos_long = cos_long.min().item()

    print(f"    Short prompt: min cosine similarity = {min_cos_short:.6f} "
          f"(across {len(cos_short)} layers)")
    print(f"    Long prompt:  min cosine similarity = {min_cos_long:.6f} "
          f"(across {len(cos_long)} layers)")

    threshold = 0.9999
    if min_cos_short < threshold or min_cos_long < threshold:
        raise RuntimeError(
            f"Left-padding sanity check FAILED.\n"
            f"  Short prompt min cosine: {min_cos_short:.6f}\n"
            f"  Long prompt min cosine:  {min_cos_long:.6f}\n"
            f"  Threshold: {threshold}\n"
            f"This model may not correctly handle left-padding with attention_mask. "
            f"Consider switching to right-padding with explicit last-token indexing."
        )

    print("    OK Left-padding sanity check passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_model(model_key: str, batch_size: int, max_length: int):
    """Load one model, extract hidden states for train + eval, save to disk."""

    info = MODEL_REGISTRY[model_key]
    hf_name = info["hf_name"]
    save_key = info["save_key"]
    save_dir = HIDDEN_DIR / save_key
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing model: {model_key}  ({hf_name})")
    print(f"{'='*60}")

    # ---- Load tokenizer ----
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    # --- Pad token setup ---
    # Base models often have no pad token. We add a dedicated pad token
    # rather than reusing eos_token, because with left padding the pad
    # tokens appear on the LEFT side of the sequence and real tokens CAN
    # attend to them via the causal mask (they are at earlier positions).
    # Using eos_token as pad would inject semantically meaningful tokens
    # into those positions, potentially contaminating hidden states.
    #
    # Adding a new token with a random embedding is safe because:
    # 1. attention_mask=0 for pad positions -> model ignores them in attention
    # 2. We never read hidden states at pad positions
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        print(f"  Added dedicated [PAD] token (id={tokenizer.pad_token_id})")

    # Left padding: all sequences right-aligned, last token always at position -1.
    tokenizer.padding_side = "left"

    # ---- Load model ----
    print("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # Resize embeddings to accommodate the new [PAD] token.
    # The new token gets a randomly initialized embedding, which is fine
    # because it's masked out in attention and we never read its hidden state.
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    # Print model info for reference
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  num_transformer_layers={num_layers}, hidden_dim={hidden_dim}")
    print(f"  output.hidden_states will have {num_layers + 1} entries "
          f"(index 0=embedding, 1..{num_layers}=transformer layers)")

    # ---- Left-padding sanity check ----
    # Verify that left-padding + attention_mask produces identical hidden
    # states to no-padding (batch_size=1).  This catches any model-specific
    # issues with Flash Attention or causal mask handling.
    _run_padding_sanity_check(model, tokenizer, max_length)

    # ---- Process each split ----
    for split in ("train", "eval"):
        out_path = save_dir / f"{split}_hidden_states.pt"
        if out_path.exists():
            print(f"\n  [skip] {out_path} already exists.")
            continue

        data = load_split(split)
        prompts, labels, sources, questions = prepare_inputs(data)
        print(f"  Split={split}: {len(data)} samples -> {len(prompts)} forward passes")

        # Check for truncation before running expensive forward passes
        check_truncation(tokenizer, prompts, max_length)

        hidden_states = extract_hidden_states(
            model, tokenizer, prompts,
            batch_size=batch_size,
            max_length=max_length,
        )
        print(f"  Hidden states shape: {hidden_states.shape}")
        # Expected: [2*len(data), num_layers+1, hidden_dim]

        assert hidden_states.shape[0] == len(prompts), \
            f"Sample count mismatch: {hidden_states.shape[0]} vs {len(prompts)}"
        assert hidden_states.shape[1] == num_layers + 1, \
            f"Layer count mismatch: {hidden_states.shape[1]} vs {num_layers + 1}"
        assert hidden_states.shape[2] == hidden_dim, \
            f"Hidden dim mismatch: {hidden_states.shape[2]} vs {hidden_dim}"

        payload = {
            "hidden_states": hidden_states,
            "labels": labels,
            "dataset_source": sources,
            "questions": questions,
        }
        torch.save(payload, out_path)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  Saved -> {out_path}  ({size_mb:.0f} MB)")

    # ---- Free GPU memory before next model ----
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Step 6: Extract hidden states")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        help="Which model to process (default: all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help=(
            "Batch size for forward pass. Tune by GPU memory: "
            "Gemma-2-2B (~4GB) ~16-32 on a 24GB GPU; "
            "Qwen3-4B (~8GB) ~4-8 on a 24GB GPU; "
            "Gemma-2-9B (~18GB) ~1-2 on a 24GB GPU. "
            "Reduce if OOM."
        ),
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help=(
            "Max token length. Prompts exceeding this will cause an error "
            "if >5%% are affected, or a warning otherwise."
        ),
    )
    args = parser.parse_args()

    models = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

    for m in models:
        process_model(m, batch_size=args.batch_size, max_length=args.max_length)

    print("\nDone.")


if __name__ == "__main__":
    main()