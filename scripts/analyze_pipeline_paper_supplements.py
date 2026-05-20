#!/usr/bin/env python
"""
Build supplementary experiment tables for the paper from existing pipeline files.

This script is offline: it does not load an LLM or regenerate answers. It reads
Step12 eval_*.json files and computes:

  1. Paired bootstrap confidence intervals for baseline vs CSV-gated methods.
  2. Gate-policy ablations:
       all_trust, all_reject, random_rate_matched, dpr_rate_matched,
       csv_offline, csv_actual, oracle.
  3. Gate mechanism diagnostics:
       on CSV-trusted examples, baseline - no_doc;
       on CSV-rejected examples, no_doc - baseline.

Recommended:
  python scripts/analyze_pipeline_paper_supplements.py

Outputs:
  results/analysis/paper_supplements/bootstrap_ci.csv
  results/analysis/paper_supplements/gate_policy_ablation.csv
  results/analysis/paper_supplements/gate_mechanism.csv
  results/analysis/paper_supplements/summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "results" / "analysis" / "paper_supplements"

DEFAULT_METHODS = [
    "vanilla_rag",
    "cad_fixed",
    "acd",
    "context_ucd",
    "dola",
]

GATED_METHOD = {
    "vanilla_rag": "csv_gated_vanilla_rag",
    "cad_fixed": "csv_gated_cad",
    "acd": "csv_gated_acd",
    "context_ucd": "csv_gated_context_ucd",
    "dola": "csv_gated_dola",
}

DEFAULT_PIPELINE_DIRS = {
    "gemma2b": PROJECT_ROOT / "results" / "pipeline",
    "qwen3_4b": PROJECT_ROOT / "autodl_5090_2" / "results" / "pipeline",
}


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt_float_name(x: float) -> str:
    return f"{x:g}".replace("-", "m").replace(".", "p")


def find_eval_file(
    pipeline_dir: Path,
    model: str,
    dataset: str,
    method: str,
    tau: float | None = None,
) -> Path:
    tau_part = f"_tau{fmt_float_name(tau)}_" if tau is not None else "_tau*_"
    pattern = f"eval_{model}_{dataset}_{method}_alpha*{tau_part}temp0p0.json"
    matches = sorted(pipeline_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matched {pipeline_dir / pattern}")
    if len(matches) > 1:
        # Prefer the canonical fixed-alpha main result when multiple alphas exist.
        alpha0p5 = [p for p in matches if "_alpha0p5_" in p.name]
        alpha1p0 = [p for p in matches if "_alpha1p0_" in p.name]
        if method.startswith("csv_gated") and alpha0p5:
            return alpha0p5[0]
        if method == "cad_fixed" and alpha0p5:
            return alpha0p5[0]
        if alpha1p0:
            return alpha1p0[0]
    return matches[0]


def load_eval_rows(path: Path) -> list[dict[str, Any]]:
    obj = load_json(path)
    rows = obj.get("per_sample")
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain per_sample")
    return rows


def metric(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(r[key]) for r in rows) / max(1, len(rows))


def align(*row_lists: list[dict[str, Any]]) -> list[tuple[dict[str, Any], ...]]:
    lengths = {len(x) for x in row_lists}
    if len(lengths) != 1:
        raise ValueError(f"Mismatched row counts: {sorted(lengths)}")
    out = []
    for items in zip(*row_lists):
        idxs = {int(x["idx"]) for x in items}
        if len(idxs) != 1:
            raise ValueError(f"Rows are not aligned by idx: {idxs}")
        out.append(items)
    return out


def mean_from_values(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def paired_bootstrap_delta(
    baseline_vals: list[float],
    gated_vals: list[float],
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    if len(baseline_vals) != len(gated_vals):
        raise ValueError("Bootstrap arrays must have equal length")
    deltas = np.asarray(gated_vals, dtype=np.float64) - np.asarray(baseline_vals, dtype=np.float64)
    observed = float(deltas.mean())
    rng = np.random.default_rng(seed)
    n = len(deltas)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = deltas[idx].mean(axis=1)
    lo = float(np.quantile(samples, 0.025))
    hi = float(np.quantile(samples, 0.975))
    return observed, lo, hi


def combine_by_mask(
    base_rows: list[dict[str, Any]],
    no_doc_rows: list[dict[str, Any]],
    trust_mask: list[bool],
) -> list[dict[str, Any]]:
    return [base if trust else no_doc for base, no_doc, trust in zip(base_rows, no_doc_rows, trust_mask)]


def rate_matched_topk_mask(scores: list[float], k: int, higher_is_better: bool = True) -> list[bool]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=higher_is_better)
    keep = set(order[:k])
    return [i in keep for i in range(len(scores))]


def random_rate_matched_metrics(
    base_rows: list[dict[str, Any]],
    no_doc_rows: list[dict[str, Any]],
    k: int,
    n_random: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(base_rows)
    base_em = np.asarray([float(r["em"]) for r in base_rows], dtype=np.float64)
    base_f1 = np.asarray([float(r["f1"]) for r in base_rows], dtype=np.float64)
    no_em = np.asarray([float(r["em"]) for r in no_doc_rows], dtype=np.float64)
    no_f1 = np.asarray([float(r["f1"]) for r in no_doc_rows], dtype=np.float64)
    ems: list[float] = []
    f1s: list[float] = []
    for _ in range(n_random):
        mask = np.zeros(n, dtype=bool)
        if k > 0:
            mask[rng.choice(n, size=k, replace=False)] = True
        ems.append(float(np.where(mask, base_em, no_em).mean()))
        f1s.append(float(np.where(mask, base_f1, no_f1).mean()))
    return {
        "em": statistics.mean(ems),
        "f1": statistics.mean(f1s),
        "em_std": statistics.pstdev(ems),
        "f1_std": statistics.pstdev(f1s),
    }


def load_retrieval_scores(dataset: str) -> list[float]:
    path = PROJECT_ROOT / "data" / "final" / f"test_retrieval_{dataset}.json"
    data = load_json(path)
    return [float(x.get("retrieved_doc", {}).get("score", 0.0)) for x in data]


def summarize_policy(
    model: str,
    dataset: str,
    method: str,
    tau: float,
    policy: str,
    rows: list[dict[str, Any]],
    trusted_rate: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "model": model,
        "dataset": dataset,
        "baseline_method": method,
        "tau": tau,
        "policy": policy,
        "n": len(rows),
        "trusted_rate": trusted_rate,
        "em": metric(rows, "em"),
        "f1": metric(rows, "f1"),
    }
    if extra:
        row.update(extra)
    return row


def run_one(
    model: str,
    dataset: str,
    method: str,
    tau: float,
    pipeline_dir: Path,
    n_boot: int,
    n_random: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    no_doc_path = find_eval_file(pipeline_dir, model, dataset, "no_doc", tau=0.5)
    base_path = find_eval_file(pipeline_dir, model, dataset, method, tau=0.5)
    gated_method = GATED_METHOD[method]
    gated_path = find_eval_file(pipeline_dir, model, dataset, gated_method, tau=tau)

    no_doc_rows = load_eval_rows(no_doc_path)
    base_rows = load_eval_rows(base_path)
    gated_rows = load_eval_rows(gated_path)
    align(no_doc_rows, base_rows, gated_rows)

    p_scores = [float(r["p_relevant"]) for r in gated_rows]
    csv_mask = [s >= tau for s in p_scores]
    k = sum(csv_mask)
    trusted_rate = k / max(1, len(csv_mask))

    csv_offline_rows = combine_by_mask(base_rows, no_doc_rows, csv_mask)
    retrieval_scores = load_retrieval_scores(dataset)
    if len(retrieval_scores) != len(base_rows):
        raise ValueError(f"{dataset} retrieval-score count does not match eval rows")
    dpr_mask = rate_matched_topk_mask(retrieval_scores, k, higher_is_better=True)
    dpr_rows = combine_by_mask(base_rows, no_doc_rows, dpr_mask)

    oracle_rows = []
    oracle_mask = []
    for base, no_doc in zip(base_rows, no_doc_rows):
        base_key = (float(base["f1"]), float(base["em"]))
        no_doc_key = (float(no_doc["f1"]), float(no_doc["em"]))
        use_base = base_key >= no_doc_key
        oracle_mask.append(use_base)
        oracle_rows.append(base if use_base else no_doc)

    random_metrics = random_rate_matched_metrics(
        base_rows, no_doc_rows, k=k, n_random=n_random, seed=seed
    )

    policy_rows = [
        summarize_policy(model, dataset, method, tau, "all_trust_baseline", base_rows, 1.0),
        summarize_policy(model, dataset, method, tau, "all_reject_no_doc", no_doc_rows, 0.0),
        summarize_policy(
            model, dataset, method, tau, "random_rate_matched", [],
            trusted_rate,
            {
                "n": len(base_rows),
                "em": random_metrics["em"],
                "f1": random_metrics["f1"],
                "random_em_std": random_metrics["em_std"],
                "random_f1_std": random_metrics["f1_std"],
            },
        ),
        summarize_policy(model, dataset, method, tau, "dpr_score_rate_matched", dpr_rows, trusted_rate),
        summarize_policy(model, dataset, method, tau, "csv_offline_recombine", csv_offline_rows, trusted_rate),
        summarize_policy(model, dataset, method, tau, "csv_actual_generated", gated_rows, trusted_rate),
        summarize_policy(model, dataset, method, tau, "oracle_base_vs_no_doc", oracle_rows, sum(oracle_mask) / len(oracle_mask)),
    ]

    boot_rows = []
    for compare_name, compare_rows in [
        ("csv_actual_vs_baseline", gated_rows),
        ("csv_offline_vs_baseline", csv_offline_rows),
        ("dpr_gate_vs_baseline", dpr_rows),
        ("csv_actual_vs_no_doc", gated_rows),
    ]:
        ref_rows = no_doc_rows if compare_name.endswith("_vs_no_doc") else base_rows
        for key in ["em", "f1"]:
            delta, lo, hi = paired_bootstrap_delta(
                [float(r[key]) for r in ref_rows],
                [float(r[key]) for r in compare_rows],
                n_boot=n_boot,
                seed=seed,
            )
            boot_rows.append({
                "model": model,
                "dataset": dataset,
                "baseline_method": method,
                "tau": tau,
                "comparison": compare_name,
                "metric": key,
                "delta": delta,
                "ci_low": lo,
                "ci_high": hi,
                "n_boot": n_boot,
                "n": len(ref_rows),
            })

    trusted_base = [b for b, keep in zip(base_rows, csv_mask) if keep]
    trusted_no = [n for n, keep in zip(no_doc_rows, csv_mask) if keep]
    rejected_base = [b for b, keep in zip(base_rows, csv_mask) if not keep]
    rejected_no = [n for n, keep in zip(no_doc_rows, csv_mask) if not keep]

    mechanism_rows = []
    for split, left_name, left_rows, right_name, right_rows in [
        ("trusted", "baseline", trusted_base, "no_doc", trusted_no),
        ("rejected", "no_doc", rejected_no, "baseline", rejected_base),
    ]:
        row = {
            "model": model,
            "dataset": dataset,
            "baseline_method": method,
            "tau": tau,
            "split": split,
            "n": len(left_rows),
            "left": left_name,
            "right": right_name,
            "left_em": metric(left_rows, "em"),
            "right_em": metric(right_rows, "em"),
            "left_minus_right_em": metric(left_rows, "em") - metric(right_rows, "em"),
            "left_f1": metric(left_rows, "f1"),
            "right_f1": metric(right_rows, "f1"),
            "left_minus_right_f1": metric(left_rows, "f1") - metric(right_rows, "f1"),
        }
        mechanism_rows.append(row)

    return policy_rows, boot_rows, mechanism_rows


def write_summary_md(policy_rows: list[dict[str, Any]], boot_rows: list[dict[str, Any]]) -> None:
    path = ANALYSIS_DIR / "summary.md"
    lines = [
        "# Paper Supplement Experiment Summary",
        "",
        "This file is generated by `scripts/analyze_pipeline_paper_supplements.py`.",
        "",
        "## Gate Policy Ablation",
        "",
        "| model | dataset | method | tau | policy | trusted | EM | F1 |",
        "|---|---|---|---:|---|---:|---:|---:|",
    ]
    for r in policy_rows:
        lines.append(
            f"| {r['model']} | {r['dataset']} | {r['baseline_method']} | {float(r['tau']):.1f} "
            f"| {r['policy']} | {float(r['trusted_rate'] or 0):.3f} "
            f"| {float(r['em']):.4f} | {float(r['f1']):.4f} |"
        )
    lines.extend([
        "",
        "## Bootstrap CI",
        "",
        "| model | dataset | method | tau | comparison | metric | delta | 95% CI |",
        "|---|---|---|---:|---|---|---:|---:|",
    ])
    for r in boot_rows:
        lines.append(
            f"| {r['model']} | {r['dataset']} | {r['baseline_method']} | {float(r['tau']):.1f} "
            f"| {r['comparison']} | {r['metric']} | {float(r['delta']):+.4f} "
            f"| [{float(r['ci_low']):+.4f}, {float(r['ci_high']):+.4f}] |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=["gemma2b", "qwen3_4b"])
    parser.add_argument("--datasets", nargs="*", default=["nq", "triviaqa"])
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--taus", type=float, nargs="*", default=[0.5])
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--n_random", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_policy: list[dict[str, Any]] = []
    all_boot: list[dict[str, Any]] = []
    all_mechanism: list[dict[str, Any]] = []

    for model in args.models:
        pipeline_dir = DEFAULT_PIPELINE_DIRS.get(model, PROJECT_ROOT / "results" / "pipeline")
        for dataset in args.datasets:
            for method in args.methods:
                for tau in args.taus:
                    try:
                        policy, boot, mechanism = run_one(
                            model=model,
                            dataset=dataset,
                            method=method,
                            tau=tau,
                            pipeline_dir=pipeline_dir,
                            n_boot=args.n_boot,
                            n_random=args.n_random,
                            seed=args.seed,
                        )
                    except FileNotFoundError as exc:
                        print(f"[skip] {model}/{dataset}/{method}/tau={tau}: {exc}")
                        continue
                    all_policy.extend(policy)
                    all_boot.extend(boot)
                    all_mechanism.extend(mechanism)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(all_policy, ANALYSIS_DIR / "gate_policy_ablation.csv")
    write_csv(all_boot, ANALYSIS_DIR / "bootstrap_ci.csv")
    write_csv(all_mechanism, ANALYSIS_DIR / "gate_mechanism.csv")
    write_summary_md(all_policy, all_boot)

    print(f"Wrote {len(all_policy)} policy rows")
    print(f"Wrote {len(all_boot)} bootstrap rows")
    print(f"Wrote {len(all_mechanism)} mechanism rows")
    print(f"Output dir: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
