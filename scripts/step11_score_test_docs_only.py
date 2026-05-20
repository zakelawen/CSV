#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import step11_contrastive_decoding as s11  # noqa: E402


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def summarize_scores(scores: np.ndarray, preds: np.ndarray, tau: float):
    return {
        "n": int(len(scores)),
        "tau": float(tau),
        "p_relevant_mean": float(scores.mean()),
        "p_relevant_std": float(scores.std()),
        "p_relevant_min": float(scores.min()),
        "p_relevant_p10": float(np.quantile(scores, 0.10)),
        "p_relevant_p25": float(np.quantile(scores, 0.25)),
        "p_relevant_p50": float(np.quantile(scores, 0.50)),
        "p_relevant_p75": float(np.quantile(scores, 0.75)),
        "p_relevant_p90": float(np.quantile(scores, 0.90)),
        "p_relevant_max": float(scores.max()),
        "pred_relevant_count": int((preds == 1).sum()),
        "pred_distracting_count": int((preds == 0).sum()),
        "pred_relevant_rate": float((preds == 1).mean()),
        "trusted_count_at_tau": int((scores >= tau).sum()),
        "trusted_rate_at_tau": float((scores >= tau).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma2b", choices=["gemma2b", "gemma9b", "qwen3_4b"])
    parser.add_argument("--dataset", default="all", choices=["nq", "triviaqa", "all"])
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_validation", action="store_true")
    parser.add_argument("--validation_tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    datasets = ["nq", "triviaqa"] if args.dataset == "all" else [args.dataset]

    print(f"\n=== Score final test retrieved docs only ===")
    print(f"model={args.model}")
    print(f"datasets={datasets}")
    print(f"tau={args.tau}")
    print(f"batch_size={args.batch_size}")

    ckpt_info = s11.resolve_best_csv_checkpoint(args.model)

    model, tokenizer, _ = s11.load_base_model_and_tokenizer(args.model)
    s11.check_checkpoint_metadata(args.model, ckpt_info, model.config.hidden_size)
    s11.inject_csv_into_model(model, ckpt_info)

    if not args.skip_validation:
        s11.validate_csv_full_eval(
            model=model,
            tokenizer=tokenizer,
            ckpt_info=ckpt_info,
            batch_size=args.batch_size,
            tolerance=args.validation_tolerance,
        )

    out_dir = PROJECT_ROOT / "results" / "pipeline_classifier_scores"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summary_rows = []

    ckpt = ckpt_info.ckpt

    for dataset in datasets:
        test_path = PROJECT_ROOT / "data" / "final" / f"test_retrieval_{dataset}.json"
        if not test_path.exists():
            raise FileNotFoundError(f"Missing test file: {test_path}")

        samples = load_json(test_path)
        if args.limit is not None:
            samples = samples[: args.limit]

        print(f"\n=== Scoring {dataset}: {len(samples)} test retrieved docs ===")

        prompts = s11.prepare_doc_scoring_prompts(tokenizer, samples)

        scored = s11.score_prompts_with_csv(
            model=model,
            prompts=prompts,
            batch_size=args.batch_size,
            pad_id=tokenizer.pad_token_id,
            cls_layer=int(ckpt["cls_layer"]),
            cos_temp=float(ckpt["cos_temp"]),
            centroids=ckpt["centroids"],
        )

        outputs = []
        for i, item in enumerate(samples):
            doc = item.get("retrieved_doc", {})
            outputs.append({
                "index": i,
                "question": item.get("question", ""),
                "answers": item.get("answers", []),
                "retrieved_doc": {
                    "title": doc.get("title", "") if isinstance(doc, dict) else "",
                    "text": doc.get("text", "") if isinstance(doc, dict) else str(doc),
                    "score": doc.get("score", None) if isinstance(doc, dict) else None,
                    "passage_id": doc.get("passage_id", None) if isinstance(doc, dict) else None,
                },
                "csv": {
                    "p_relevant": float(scored.scores[i]),
                    "pred_label": int(scored.preds[i]),
                    "pred_label_name": "relevant" if int(scored.preds[i]) == 1 else "distracting",
                    "trusted_at_tau": bool(scored.scores[i] >= args.tau),
                    "margin": float(scored.margins[i]),
                    "logits": [float(x) for x in scored.raw_logits[i].tolist()],
                },
            })

        summary = summarize_scores(scored.scores, scored.preds, args.tau)
        summary.update({
            "model": args.model,
            "dataset": dataset,
            "best_key": ckpt_info.best_key,
            "checkpoint": str(ckpt_info.ckpt_path),
            "str_layer": int(ckpt["str_layer"]),
            "cls_layer": int(ckpt["cls_layer"]),
            "lam": float(ckpt["lam"]),
            "cos_temp": float(ckpt["cos_temp"]),
            "saved_best_auroc": float(ckpt_info.sweep_entry.get("best_auroc", ckpt["best_auroc"])),
            "saved_best_accuracy": float(ckpt_info.sweep_entry.get("best_accuracy", ckpt["best_accuracy"])),
        })

        out_json = out_dir / f"{args.model}_{dataset}_test_retrieval_csv_scores_tau{str(args.tau).replace('.', 'p')}.json"
        save_json({
            "summary": summary,
            "outputs": outputs,
        }, out_json)

        print(f"Saved scores -> {out_json}")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

        all_summary_rows.append(summary)

    summary_csv = out_dir / f"{args.model}_test_retrieval_csv_score_summary.csv"
    fieldnames = sorted(set().union(*(row.keys() for row in all_summary_rows)))
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_summary_rows)

    print(f"\nSaved summary CSV -> {summary_csv}")


if __name__ == "__main__":
    main()