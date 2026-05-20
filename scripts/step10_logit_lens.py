"""
Step 10: Logit Lens Analysis.

Purpose
-------
Verify and characterize the assumption that drives Step 11's contrastive
Decoding (CD):

    "Given a relevant document, the LLM's probability of the correct
     answer should be higher than without any document, and lower with a
     distracting document."

Concretely, this script answers three questions per model:

  Q1 (feasibility):
      Is P(answer | gold_ctx) systematically higher than
      P(answer | no_ctx)? If yes, CD is meaningful;
      otherwise CD just amplifies noise.

  Q2 (which layer):
      At which transformer layer is the gap
      ΔP = logP(answer | gold) - logP(answer | no_ctx)
      maximized? This determines where CD should be applied.
      If the gap peaks at the final layer, standard CD; if it peaks
      mid-network, DoLA-style CD using that layer's logits is preferable.

  Q3 (asymmetry):
      Does a distracting document actually depress the answer signal?
      ΔP_dis = logP(answer | dis) - logP(answer | no_ctx) should be
      ≤ 0 (and ideally clearly negative).

Method (logit lens, teacher forcing)
=====================================
For each sample s and each context condition c ∈ {no, gold, dis}:

  prompt_c = build_prompt(question, doc_for_c)
  full_ids = prompt_c_ids + answer_ids       # teacher forcing
  output   = model(full_ids, output_hidden_states=True)
  hs[ℓ]    = output.hidden_states[ℓ]         # [1, T, D]   ℓ=0..L

For each layer ℓ:
  if ℓ is the final hidden state:
      norm_hs = hs[ℓ]                        # already final-normalized by HF model
  else:
      norm_hs = final_layernorm(hs[ℓ])       # critical for intermediate logit lens

  logits[ℓ]  = lm_head(norm_hs)              # [1, T, V]
  logits[ℓ]  = apply_final_logit_softcapping_if_needed(logits[ℓ])
  per_token_logP[ℓ, t] = log_softmax(logits[ℓ, prompt_len + t - 1])[answer_ids[t]]
  per_sample_logP[ℓ] = mean_t( per_token_logP[ℓ, t] )

The classic "next-token" formulation: at position prompt_len + t - 1, we
predict answer_ids[t]. This is teacher forcing, identical to how
generation likelihoods are computed.

Important fix: final layer must NOT be final-normed twice
---------------------------------------------------------
Hugging Face decoder models such as Gemma2 / LLaMA / Qwen usually return:

    hidden_states[-1] == outputs.last_hidden_state

and this final hidden state is already after model.model.norm.
Therefore:

    lm_head(hidden_states[-1])

matches the model's real final logits before model-specific final logit
processing.

For intermediate layers, however, hidden_states[0:-1] are not final-normalized,
so we still apply final_norm before lm_head for logit-lens projection.

Note on Gemma2 final_logit_softcapping
--------------------------------------
Gemma2 applies tanh-based logit softcapping (cap=30.0) AFTER the LM head
in normal forward:
    logits = cap * tanh(lm_head(hidden) / cap)

We apply softcapping in logit lens too, because:
  (a) Gemma2's raw lm_head logits can be pathologically large for certain
      tokens, so without softcap the predicted argmax may differ wildly
      from the model's actual output.
  (b) Gemma2 was trained assuming this transform exists at the output,
      so the "what would this layer predict if generation stopped here"
      interpretation should use the same transform.
  (c) softcap is monotonic, but it compresses runaway logits, which can
      change argmax when extreme logits dominate.

For models without final_logit_softcapping, e.g. Qwen3 / LLaMA / Mistral,
this is a no-op.

Span aggregation
================
"Answer signal" per sample, per condition, per layer:

  signal[s, c, ℓ] = mean over answer tokens of logP(answer_t | ...)

We average logP, not P, because P fluctuates over many orders of magnitude
across layers; mean logP is more numerically stable and matches how CD is
computed.

For reporting we convert:

  P_mean[c, ℓ] = exp(mean_s(signal[s, c, ℓ]))

Multiple answers per sample
---------------------------
Each NQ/TriviaQA sample can have multiple gold answers. We pick a single
"canonical answer" per sample by:

  1. Compute mean-over-layers logP under no_ctx for each candidate answer.
  2. Pick the answer with the highest mean-layers logP under no_ctx.
  3. Use this canonical answer for all three conditions:
     no_ctx, gold_ctx, dis_ctx.

Why select using no_ctx?
  - We want the baseline to be fair. Using gold_ctx to select the answer
    would bias toward whichever phrasing was promoted by the gold document.
  - Using the same canonical answer across all three conditions makes
    within-sample ΔP an apples-to-apples comparison.

Outputs
-------
results/logit_lens/{model}_logit_lens.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = PROJECT_ROOT / "data" / "final" / "eval.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "logit_lens"

# Cap candidate answers per sample to bound forward count.
# eval.json can contain samples with many alias answers.
MAX_CANDIDATE_ANSWERS = 10


MODEL_REGISTRY = {
    "gemma2b": {
        "hf_name": os.environ.get(
            "GEMMA_MODEL_PATH",
            "google/gemma-2-2b",
        ),
    },
    "gemma9b": {
        "hf_name": os.environ.get(
            "GEMMA9B_MODEL_PATH",
            "google/gemma-2-9b",
        ),
    },
    "qwen3_4b": {
        "hf_name": os.environ.get(
            "QWEN3_4B_MODEL_PATH",
            "Qwen/Qwen3-4B-Base",
        ),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_doc_text(doc) -> str:
    if isinstance(doc, dict):
        title = str(doc.get("title", "")).strip()
        text = str(doc.get("text", "")).strip()
        if title:
            return f"Title: {title}\nText: {text}"
        return text
    return str(doc).strip()


def build_prompt(question: str, doc_text: str | None) -> str:
    """
    Mirror Step 6/9 prompt format.
    doc_text=None means no_ctx, i.e., only question.
    """
    if doc_text is None:
        return f"Question: {question}\nAnswer:"
    return f"Document: {doc_text}\n\nQuestion: {question}\nAnswer:"


def get_final_norm(model):
    """
    Return the final RMSNorm/LayerNorm that sits between the last decoder
    layer and the LM head.

    Gemma2: model.model.norm
    LLaMA:  model.model.norm
    Qwen3:  model.model.norm
    """
    inner = model.model
    if hasattr(inner, "norm"):
        return inner.norm
    raise RuntimeError(
        f"Cannot locate final norm on {type(inner).__name__}. "
        "Inspect model.model.children() to find the right attribute."
    )


def get_lm_head(model):
    """Return the LM head module. Both tied and untied embeddings work."""
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise RuntimeError("model has no lm_head attribute")


def apply_final_logit_softcapping(logits: torch.Tensor, softcap: float | None):
    """
    Apply model-specific final logit softcapping if the model config defines it.

    Gemma2 uses:
        logits = cap * tanh(logits / cap)

    Qwen3 / LLaMA usually do not define final_logit_softcapping, so this is a no-op.
    """
    if softcap is None:
        return logits
    return softcap * torch.tanh(logits / softcap)


def project_hidden_to_logits(
    model,
    final_norm,
    lm_head,
    hs: torch.Tensor,
    layer_idx: int,
    num_hidden_states: int,
    softcap: float | None,
):
    """
    Project one hidden-state tensor to vocabulary logits for logit lens.

    Critical detail:
      - hidden_states[-1] is already final-normalized in HF decoder models.
        Do NOT apply final_norm again to the final layer.
      - intermediate hidden_states are not final-normalized, so we apply
        final_norm before lm_head.
    """
    if layer_idx == num_hidden_states - 1:
        # Final hidden state already matches model(...).last_hidden_state.
        hs_for_lm = hs
    else:
        # Intermediate logit lens projection uses the final norm before lm_head.
        hs_for_lm = final_norm(hs)

    logits = lm_head(hs_for_lm)
    logits = apply_final_logit_softcapping(logits, softcap)
    return logits


@torch.no_grad()
def verify_final_layer_projection(model, tokenizer, device, softcap: float | None):
    """
    Quick sanity check:
    lm_head(hidden_states[-1]) + softcap should match model(...).logits.

    This catches the double-final-norm bug.
    """
    model.eval()
    text = "Question: What is the capital of France?\nAnswer:"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    base_out = model.model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
    )
    full_out = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
    )

    last_hs = base_out.hidden_states[-1]
    logits_from_hs = model.lm_head(last_hs)
    logits_from_hs = apply_final_logit_softcapping(logits_from_hs, softcap)

    real_logits = full_out.logits

    max_abs_diff = (logits_from_hs - real_logits).abs().max().item()
    argmax_match = (logits_from_hs.argmax(dim=-1) == real_logits.argmax(dim=-1)).all().item()

    print("  Final-layer projection sanity check:")
    print(f"    max abs diff: {max_abs_diff:.8f}")
    print(f"    argmax match: {argmax_match}")

    if max_abs_diff > 1e-2 or not argmax_match:
        print(
            "  [Warning] Final-layer projection does not exactly match model logits. "
            "This may be due to architecture-specific output processing. "
            "Inspect the model forward if results look suspicious."
        )


# ---------------------------------------------------------------------------
# Per-sample logit lens forward
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_per_layer_answer_logp(
    model,
    final_norm,
    lm_head,
    tokenizer,
    prompt_text: str,
    answer_text: str,
    device,
    max_length: int,
    softcap: float | None = None,
):
    """
    For one (prompt, answer) pair, run a single forward pass with
    output_hidden_states=True, then for each layer ℓ compute the mean
    log-probability assigned to the answer tokens under teacher forcing.

    Args:
        softcap:
            If not None, apply Gemma2-style logit softcapping:
                logits = cap * tanh(logits / cap)
            after lm_head at every layer.

    Returns:
        per_layer_mean_logP: list[float], length L+1
        n_answer_tokens: int

        Returns (None, 0) if answer cannot be tokenized or prompt truncation
        makes the sample unusable.
    """
    # Tokenize prompt and answer separately so we know answer-token boundaries.
    # The leading space matters for BPE/SentencePiece tokenizers.
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    answer_ids = tokenizer.encode(" " + answer_text, add_special_tokens=False)

    if len(answer_ids) == 0:
        return None, 0

    # Defensive truncation: trim from prompt head if total too long.
    # Keep answer intact.
    full_ids = prompt_ids + answer_ids
    if len(full_ids) > max_length:
        overflow = len(full_ids) - max_length
        prompt_ids = prompt_ids[overflow:]
        full_ids = prompt_ids + answer_ids
        if len(prompt_ids) == 0:
            return None, 0

    prompt_len = len(prompt_ids)
    n_ans = len(answer_ids)

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    output = model.model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=False,
    )
    hs_tuple = output.hidden_states  # tuple of (L+1) tensors, each [1, T, D]

    num_hidden_states = len(hs_tuple)
    per_layer_mean = []

    # Positions where we predict answer_ids[t]:
    #   model predicts token at position p+1 from logits at position p.
    #   answer token t sits at position prompt_len + t.
    #   therefore read logits at position prompt_len + t - 1.
    pred_positions = torch.tensor(
        [prompt_len + t - 1 for t in range(n_ans)],
        dtype=torch.long,
        device=device,
    )
    answer_ids_t = torch.tensor(answer_ids, dtype=torch.long, device=device)

    for layer_idx in range(num_hidden_states):
        hs = hs_tuple[layer_idx]  # [1, T, D]

        logits = project_hidden_to_logits(
            model=model,
            final_norm=final_norm,
            lm_head=lm_head,
            hs=hs,
            layer_idx=layer_idx,
            num_hidden_states=num_hidden_states,
            softcap=softcap,
        )  # [1, T, V]

        ans_logits = logits[0, pred_positions, :]  # [n_ans, V]
        log_probs = F.log_softmax(ans_logits.float(), dim=-1)  # [n_ans, V]
        per_token_logp = log_probs[range(n_ans), answer_ids_t]  # [n_ans]

        per_layer_mean.append(float(per_token_logp.mean().item()))

    return per_layer_mean, n_ans


# ---------------------------------------------------------------------------
# Main per-model logic
# ---------------------------------------------------------------------------

def process_model(
    model_key: str,
    max_length: int,
    max_samples: int | None,
    requested_device: str | None,
):
    print(f"\n=== Step 10 Logit Lens: {model_key} ===")
    hf_name = MODEL_REGISTRY[model_key]["hf_name"]
    print(f"  hf_name: {hf_name}")

    # --- Load tokenizer ---
    print("  Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"    pad_token = {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")

    # --- Load model ---
    print("  Loading model …")
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=torch.float16,
        device_map="auto" if requested_device is None else None,
        low_cpu_mem_usage=True,
    )

    if requested_device is not None:
        model = model.to(requested_device)

    model.eval()

    # Use the actual first parameter device. This is safer with device_map="auto".
    model_device = next(model.parameters()).device
    print(f"  model input device: {model_device}")

    final_norm = get_final_norm(model)
    lm_head = get_lm_head(model)

    # Read final_logit_softcapping from model config.
    # Gemma2 / VaultGemma / T5Gemma2: typically 30.0
    # Qwen3 / LLaMA / Mistral: not set or None → no softcap applied.
    softcap = getattr(model.config, "final_logit_softcapping", None)
    if softcap is not None:
        softcap = float(softcap)
        print(f"  final_logit_softcapping = {softcap}  (will be applied after lm_head)")
    else:
        print("  final_logit_softcapping = None  (no softcap)")

    # Sanity check: final hidden state projection should match model logits.
    verify_final_layer_projection(
        model=model,
        tokenizer=tokenizer,
        device=model_device,
        softcap=softcap,
    )

    # Probe number of layers via a tiny forward.
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
        L_plus_1 = len(dummy_out.hidden_states)
    print(f"  num_layers (incl. embedding) = {L_plus_1}")

    # --- Load eval data ---
    with open(EVAL_PATH, "r") as f:
        samples = json.load(f)

    if max_samples is not None and max_samples < len(samples):
        samples = samples[:max_samples]
        print(f"  Using {max_samples} eval samples (truncated)")
    else:
        print(f"  Using all {len(samples)} eval samples")

    # --- Forward each sample × each condition ---
    conditions = ["no_ctx", "gold_ctx", "dis_ctx"]
    n_samples = len(samples)

    # signal[c][s, ℓ]
    signal = {
        c: np.full((n_samples, L_plus_1), np.nan, dtype=np.float64)
        for c in conditions
    }
    sources = []

    skipped = 0
    canonical_answer_idx = []

    for s_idx, sample in enumerate(tqdm(samples, desc="  samples")):
        question = sample["question"]

        # Filter valid answers and cap to bound forward count.
        candidate_answers = [
            a for a in (sample.get("answers") or [])
            if isinstance(a, str) and a.strip()
        ][:MAX_CANDIDATE_ANSWERS]

        if not candidate_answers:
            skipped += 1
            sources.append("unknown")
            canonical_answer_idx.append(-1)
            continue

        sources.append(str(sample.get("source_dataset", "unknown")).lower())

        rel_doc = extract_doc_text(sample["relevant_doc"])
        dis_doc = extract_doc_text(sample["distracting_doc"])

        no_prompt = build_prompt(question, None)
        gold_prompt = build_prompt(question, rel_doc)
        dis_prompt = build_prompt(question, dis_doc)

        # ----------------------------------------------------------
        # Pick canonical answer = answer with highest mean-layer logP
        # under no_ctx.
        # ----------------------------------------------------------
        if len(candidate_answers) == 1:
            canon_ans = candidate_answers[0]
            canon_idx = 0
            no_per_layer, _ = compute_per_layer_answer_logp(
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

            for a_idx, ans in enumerate(candidate_answers):
                pl, _ = compute_per_layer_answer_logp(
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

                # Score = mean over layers of mean-token logP.
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

        # ----------------------------------------------------------
        # Now run gold_ctx and dis_ctx with the same canonical answer.
        # ----------------------------------------------------------
        gold_pl, _ = compute_per_layer_answer_logp(
            model=model,
            final_norm=final_norm,
            lm_head=lm_head,
            tokenizer=tokenizer,
            prompt_text=gold_prompt,
            answer_text=canon_ans,
            device=model_device,
            max_length=max_length,
            softcap=softcap,
        )
        if gold_pl is not None:
            signal["gold_ctx"][s_idx, :] = gold_pl

        dis_pl, _ = compute_per_layer_answer_logp(
            model=model,
            final_norm=final_norm,
            lm_head=lm_head,
            tokenizer=tokenizer,
            prompt_text=dis_prompt,
            answer_text=canon_ans,
            device=model_device,
            max_length=max_length,
            softcap=softcap,
        )
        if dis_pl is not None:
            signal["dis_ctx"][s_idx, :] = dis_pl

    if skipped > 0:
        print(f"  Skipped {skipped} samples with no usable answer")

    # Stats on canonical answer selection.
    n_with_multiple = sum(
        1 for s in samples
        if len([
            a for a in (s.get("answers") or [])
            if isinstance(a, str) and a.strip()
        ]) > 1
    )
    n_picked_nonzero = sum(1 for i in canonical_answer_idx if i > 0)
    print(f"  Samples with multiple answers: {n_with_multiple} / {len(samples)}")
    print(f"  Of those, picked a non-first answer: {n_picked_nonzero}")

    sources_arr = np.array(sources)

    # --- Aggregate per layer ---
    def per_subset_layer_mean(arr_2d, mask):
        """arr_2d: [N, L+1], mask: [N] bool. Mean over masked rows, ignoring NaN."""
        sub = arr_2d[mask]
        if len(sub) == 0:
            return [None] * arr_2d.shape[1]
        with np.errstate(all="ignore"):
            m = np.nanmean(sub, axis=0)
        return [float(x) if np.isfinite(x) else None for x in m]

    masks = {
        "combined": np.ones(n_samples, dtype=bool),
        "nq": sources_arr == "nq",
        "triviaqa": sources_arr == "triviaqa",
    }

    per_layer_results = {}
    for cond in conditions:
        per_layer_results[cond] = {
            "mean_logP_combined": per_subset_layer_mean(
                signal[cond], masks["combined"]
            ),
            "mean_logP_nq": per_subset_layer_mean(
                signal[cond], masks["nq"]
            ),
            "mean_logP_triviaqa": per_subset_layer_mean(
                signal[cond], masks["triviaqa"]
            ),
        }

    # ΔP curves per layer. Compute mean over samples of within-sample diff:
    #   signal[gold, s, ℓ] - signal[no, s, ℓ]
    def per_sample_diff_mean(arr_a, arr_b, mask):
        diff = arr_a - arr_b
        sub = diff[mask]
        if len(sub) == 0:
            return [None] * diff.shape[1]
        with np.errstate(all="ignore"):
            m = np.nanmean(sub, axis=0)
        return [float(x) if np.isfinite(x) else None for x in m]

    delta_gold_no = per_sample_diff_mean(
        signal["gold_ctx"], signal["no_ctx"], masks["combined"]
    )
    delta_dis_no = per_sample_diff_mean(
        signal["dis_ctx"], signal["no_ctx"], masks["combined"]
    )

    # Find peak layer for delta_gold_no.
    delta_arr = np.array([x if x is not None else -np.inf for x in delta_gold_no])
    peak_layer = int(np.argmax(delta_arr))
    peak_value = float(delta_arr[peak_layer])

    # Find first layer where mean_logP(gold) > weak threshold (-5).
    gold_logp_combined = np.array([
        x if x is not None else -np.inf
        for x in per_layer_results["gold_ctx"]["mean_logP_combined"]
    ])
    above_thresh = gold_logp_combined > -5.0
    emerge_layer = int(np.argmax(above_thresh)) if above_thresh.any() else -1

    # --- Save ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{model_key}_logit_lens.json"

    payload = {
        "model_name": model_key,
        "hf_name": hf_name,
        "num_layers": L_plus_1,
        "num_samples": {
            "combined": int(masks["combined"].sum()),
            "nq": int(masks["nq"].sum()),
            "triviaqa": int(masks["triviaqa"].sum()),
        },
        "canonical_answer_stats": {
            "n_samples_with_multiple_answers": int(n_with_multiple),
            "n_picked_non_first_answer": int(n_picked_nonzero),
        },
        "per_layer": per_layer_results,
        "delta_gold_minus_no": delta_gold_no,
        "delta_dis_minus_no": delta_dis_no,
        "answer_signal_emerge_layer": emerge_layer,
        "delta_gold_peak_layer": peak_layer,
        "delta_gold_peak_value": peak_value,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n  Layers: {L_plus_1}")
    print("  Final-layer mean logP:")
    for cond in conditions:
        v = per_layer_results[cond]["mean_logP_combined"][-1]
        if v is None:
            print(f"    {cond:9s}: logP=None")
        else:
            print(f"    {cond:9s}: logP={v:.4f}  P≈{np.exp(v):.4f}")
    print(f"  ΔP_gold-no  peak: layer {peak_layer}, value = {peak_value:+.4f}")
    print(f"  ΔP_dis-no   final-layer: {delta_dis_no[-1]:+.4f}")
    print(f"  Answer signal emerge layer (logP > -5): {emerge_layer}")
    print(f"  Saved → {out_path}")

    # Free GPU.
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 10: Logit Lens analysis")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gemma2b", "gemma9b", "qwen3_4b", "all"],
        help="gemma2b / gemma9b / qwen3_4b / all",
    )
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="If set, truncate eval set to first N samples for testing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Optional explicit device, e.g. cuda:0. "
            "If omitted, use device_map='auto'."
        ),
    )
    args = parser.parse_args()

    print(f"Requested device: {args.device}")

    if args.model == "all":
        for m in ["gemma2b", "gemma9b", "qwen3_4b"]:
            try:
                process_model(
                    model_key=m,
                    max_length=args.max_length,
                    max_samples=args.max_samples,
                    requested_device=args.device,
                )
            except Exception as e:
                print(f"\n[{m}] FAILED: {e}")
    else:
        process_model(
            model_key=args.model,
            max_length=args.max_length,
            max_samples=args.max_samples,
            requested_device=args.device,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
