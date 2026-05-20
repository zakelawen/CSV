#!/usr/bin/env python
"""
Step 13: Gold-evidence RAG diagnostic.

This script asks a deliberately simple question:

  If the model is given the controlled gold evidence passage from
  data/final/{train,eval}.json:relevant_doc, how often does it still fail to
  answer correctly?

The output format is compatible with scripts/step12_evaluate_pipeline.py, so
the same EM/F1 evaluator can be reused.

Example
-------
python scripts/step13_gold_evidence_rag.py \
  --model gemma2b \
  --dataset all \
  --split eval \
  --max_new_tokens 20 \
  --eval_after_generate
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from step11_contrastive_decoding import (
    DATASETS,
    MODEL_REGISTRY,
    PIPELINE_DIR,
    PROJECT_ROOT,
    build_prompt,
    extract_doc_text,
    format_float_for_name,
    generate_standard,
    load_base_model_and_tokenizer,
    load_json,
    save_json,
)
from step12_evaluate_pipeline import (
    evaluate_pipeline_file,
    get_answers,
    normalize_answer,
    save_json as save_eval_json,
    summarize_group,
    write_global_summary_from_eval_files,
)


DATA_DIR = PROJECT_ROOT / "data" / "final"
DEFAULT_METHOD_NAME = "gold_evidence_rag"


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


def sanitize_method_name(name: str) -> str:
    name = name.strip()
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError(
            "--method_name must contain only letters, numbers, and underscores."
        )
    return name


def load_gold_samples(split: str, dataset: str, limit: int | None) -> tuple[Path, list[dict[str, Any]]]:
    path = DATA_DIR / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")

    raw = load_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list in {path}")

    samples: list[dict[str, Any]] = []
    for orig_idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue

        src = normalize_dataset_name(item.get("source_dataset", "unknown"))
        if dataset != "all" and src != dataset:
            continue

        if "relevant_doc" not in item:
            raise KeyError(f"Item {orig_idx} in {path} has no relevant_doc field")

        row = dict(item)
        row["_orig_idx"] = orig_idx
        row["source_dataset"] = src
        samples.append(row)

        if limit is not None and len(samples) >= limit:
            break

    print(f"  Loaded {split}/{dataset}: {len(samples)} gold-evidence samples")
    return path, samples


def answer_hits_in_doc(answers: list[str], doc_text: str) -> list[str]:
    norm_doc = normalize_answer(doc_text)
    hits: list[str] = []
    seen: set[str] = set()
    for answer in answers:
        norm_answer = normalize_answer(answer)
        if not norm_answer:
            continue
        if norm_answer in norm_doc and answer not in seen:
            hits.append(answer)
            seen.add(answer)
    return hits


def summarize_gold_doc_answer_hits(samples: list[dict[str, Any]]) -> dict[str, Any]:
    hit_count = 0
    multi_alias_count = 0
    for item in samples:
        answers = get_answers(item)
        doc_text = extract_doc_text(item["relevant_doc"])
        hits = answer_hits_in_doc(answers, doc_text)
        if hits:
            hit_count += 1
        if len(answers) > 1:
            multi_alias_count += 1

    n = len(samples)
    return {
        "n": n,
        "gold_doc_answer_hits": hit_count,
        "gold_doc_answer_hit_rate": hit_count / max(1, n),
        "multi_alias_samples": multi_alias_count,
    }


def run_gold_generation_for_dataset(
    model,
    tokenizer,
    dataset: str,
    split: str,
    input_path: Path,
    samples: list[dict[str, Any]],
    args,
) -> Path:
    print(f"\n=== Gold-evidence generation: dataset={dataset}, split={split} ===")

    outputs = []
    start_time = time.time()
    hit_count = 0

    for local_idx, item in enumerate(tqdm(samples, desc=f"gold generate {dataset}")):
        question = item["question"]
        answers = get_answers(item)
        gold_doc = item["relevant_doc"]
        doc_text = extract_doc_text(gold_doc)
        prompt = build_prompt(question, doc_text)
        matched_answers = answer_hits_in_doc(answers, doc_text)
        if matched_answers:
            hit_count += 1

        prediction = generate_standard(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            max_input_length=args.max_input_length,
            temperature=args.temperature,
            top_p=args.top_p,
            stop_on_newline=args.stop_on_newline,
        )

        outputs.append({
            "idx": local_idx,
            "orig_idx": item.get("_orig_idx"),
            "source_dataset": dataset,
            "question": question,
            "answers": item.get("answers", []),
            "gold_doc": gold_doc,
            # Kept for compatibility with Step11-style consumers. This is not
            # a retrieved DPR top-1 document; see meta.context_source.
            "retrieved_doc": gold_doc,
            "csv": None,
            "diagnostics": {
                "gold_doc_answer_hit": bool(matched_answers),
                "matched_gold_answers": matched_answers,
                "num_gold_aliases": len(answers),
            },
            "generation": {
                "method": args.method_name,
                "actual_mode": "gold_evidence_rag",
                "alpha_requested": 0.0,
                "alpha_used": 0.0,
                "max_new_tokens": int(args.max_new_tokens),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "prediction": prediction,
                "details": {
                    "context_source": f"data/final/{split}.json:relevant_doc",
                    "input_file": str(input_path),
                    "decoding": "greedy",
                },
            },
        })

    elapsed = time.time() - start_time
    n = len(outputs)
    hit_rate = hit_count / max(1, n)

    result_obj = {
        "meta": {
            "model": args.model,
            "dataset": dataset,
            "method": args.method_name,
            "split": split,
            "alpha": 0.0,
            "tau": None,
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_new_tokens": int(args.max_new_tokens),
            "max_input_length": args.max_input_length,
            "limit": args.limit,
            "num_samples": n,
            "needs_csv": False,
            "csv_source": None,
            "context_source": "gold_evidence_relevant_doc",
            "input_file": str(input_path),
            "gold_doc_answer_hits": hit_count,
            "gold_doc_answer_hit_rate": hit_rate,
            "trusted_docs": None,
            "trusted_rate": None,
            "elapsed_sec": elapsed,
        },
        "outputs": outputs,
    }

    temp_name = format_float_for_name(args.temperature)
    limit_suffix = f"_limit{args.limit}" if args.limit is not None else ""
    out_path = (
        PIPELINE_DIR
        / f"{args.model}_{dataset}_{args.method_name}_{split}_temp{temp_name}{limit_suffix}.json"
    )
    save_json(result_obj, out_path)

    print(f"  Saved generation outputs -> {out_path}")
    print(f"  gold-doc answer string hits: {hit_count}/{n} ({hit_rate:.2%})")
    print(f"  elapsed: {elapsed:.1f}s")
    return out_path


def evaluate_generated_file(path: Path) -> None:
    result = evaluate_pipeline_file(path)
    payload = load_json(path)
    outputs = payload.get("outputs", []) if isinstance(payload, dict) else []

    for eval_row, output_row in zip(result["per_sample"], outputs):
        diagnostics = output_row.get("diagnostics", {}) if isinstance(output_row, dict) else {}
        if isinstance(diagnostics, dict):
            eval_row["gold_doc_answer_hit"] = diagnostics.get("gold_doc_answer_hit")
            eval_row["matched_gold_answers"] = diagnostics.get("matched_gold_answers")

    by_gold_hit = {}
    for val in [True, False]:
        group = [x for x in result["per_sample"] if x.get("gold_doc_answer_hit") is val]
        by_gold_hit[str(val).lower()] = summarize_group(group)
    result["metrics"]["by_gold_doc_answer_hit"] = by_gold_hit

    out_path = PIPELINE_DIR / f"eval_{path.stem}.json"
    save_eval_json(result, out_path)

    overall = result["metrics"]["overall"]
    print(f"  Eval N={overall['n']}  EM={overall['em']:.4f}  F1={overall['f1']:.4f}")
    hit_group = by_gold_hit["true"]
    if hit_group["n"]:
        print(
            f"  Gold-doc-hit subset N={hit_group['n']}  "
            f"EM={hit_group['em']:.4f}  F1={hit_group['f1']:.4f}"
        )
    print(f"  Saved eval -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold-evidence RAG diagnostic")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS + ["all"])
    parser.add_argument("--split", type=str, default="eval", choices=["train", "eval"])
    parser.add_argument("--method_name", type=str, default=DEFAULT_METHOD_NAME)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--stop_on_newline", action="store_true", default=True)
    parser.add_argument("--no_stop_on_newline", action="store_false", dest="stop_on_newline")

    parser.add_argument("--dry_run", action="store_true",
                        help="Only load data and report answer-in-gold-doc coverage.")
    parser.add_argument("--eval_after_generate", action="store_true",
                        help="Run Step12 evaluation immediately after generation.")
    args = parser.parse_args()

    args.method_name = sanitize_method_name(args.method_name)
    if args.max_input_length is not None and args.max_input_length <= 0:
        args.max_input_length = None

    if args.temperature is not None and args.temperature > 0:
        raise ValueError("Use greedy decoding for this diagnostic: --temperature 0.0")

    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("Step 13: Gold-evidence RAG diagnostic")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model:        {args.model}")
    print(f"Dataset:      {args.dataset}")
    print(f"Split:        {args.split}")
    print(f"Method name:  {args.method_name}")
    print(f"max_new:      {args.max_new_tokens}")
    print("decoding:     greedy (temperature=0.0)")

    datasets_to_run = DATASETS if args.dataset == "all" else [args.dataset]

    samples_by_dataset: dict[str, tuple[Path, list[dict[str, Any]]]] = {}
    for dataset in datasets_to_run:
        input_path, samples = load_gold_samples(
            split=args.split,
            dataset=dataset,
            limit=args.limit,
        )
        samples_by_dataset[dataset] = (input_path, samples)
        diag = summarize_gold_doc_answer_hits(samples)
        print(
            f"  {dataset}: answer string appears in gold doc for "
            f"{diag['gold_doc_answer_hits']}/{diag['n']} "
            f"({diag['gold_doc_answer_hit_rate']:.2%}); "
            f"multi-alias samples={diag['multi_alias_samples']}"
        )

    if args.dry_run:
        print("\nDry run complete; no model was loaded.")
        return

    gen_model, gen_tokenizer, _ = load_base_model_and_tokenizer(args.model)

    output_paths: list[Path] = []
    for dataset, (input_path, samples) in samples_by_dataset.items():
        out_path = run_gold_generation_for_dataset(
            model=gen_model,
            tokenizer=gen_tokenizer,
            dataset=dataset,
            split=args.split,
            input_path=input_path,
            samples=samples,
            args=args,
        )
        output_paths.append(out_path)

        if args.eval_after_generate:
            evaluate_generated_file(out_path)

    if args.eval_after_generate:
        write_global_summary_from_eval_files(PIPELINE_DIR)

    print("\n" + "=" * 80)
    print("Step13 done.")
    print("Outputs:")
    for p in output_paths:
        print(f"  {p}")
    if not args.eval_after_generate:
        print("Evaluate with:")
        for p in output_paths:
            print(f"  python scripts/step12_evaluate_pipeline.py --input {p}")
    print("=" * 80)


if __name__ == "__main__":
    main()
