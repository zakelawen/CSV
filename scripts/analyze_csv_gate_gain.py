#!/usr/bin/env python
"""
Offline analysis for CSV-gated CAD gains.

This script does not run any model. It recombines existing Step12 eval files:

  if p_relevant >= tau:
      use cad_fixed(alpha) metrics
  else:
      use no_doc metrics

That is exactly the decision made by csv_gated_cad, so it lets us sweep tau
without regenerating answers.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = PROJECT_ROOT / "results" / "pipeline"
ANALYSIS_DIR = PROJECT_ROOT / "results" / "analysis"


def format_float_for_name(x: float) -> str:
    return f"{x:g}".replace("-", "m").replace(".", "p")


def format_float_name_candidates(x: float) -> list[str]:
    names = [
        format_float_for_name(x),
        f"{x:.1f}".replace("-", "m").replace(".", "p"),
    ]
    out = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


def find_eval_file(model: str, dataset: str, method: str, alpha: float, tau: float = 0.5) -> Path:
    tau_name = format_float_for_name(tau)
    for alpha_name in format_float_name_candidates(alpha):
        path = (
            PIPELINE_DIR
            / f"eval_{model}_{dataset}_{method}_alpha{alpha_name}_tau{tau_name}_temp0p0.json"
        )
        if path.exists():
            return path
    alpha_names = "|".join(format_float_name_candidates(alpha))
    raise FileNotFoundError(
        f"No eval file found for {model}/{dataset}/{method} alpha in [{alpha_names}] tau={tau:g}"
    )


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_eval(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    obj = load_json(path)
    rows = obj.get("per_sample")
    if not isinstance(rows, list):
        raise ValueError(f"{path} has no per_sample list")
    return rows


def mean_metric(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(r[key]) for r in rows) / len(rows)


def align_by_idx(*row_lists: list[dict]) -> list[tuple[dict, ...]]:
    if not row_lists:
        return []
    lengths = {len(x) for x in row_lists}
    if len(lengths) != 1:
        raise ValueError(f"Eval files have mismatched lengths: {sorted(lengths)}")

    out = []
    for items in zip(*row_lists):
        idxs = {int(x["idx"]) for x in items}
        if len(idxs) != 1:
            raise ValueError(f"Eval files are not aligned by idx: {idxs}")
        out.append(items)
    return out


def discover_alphas(model: str, dataset: str) -> list[float]:
    prefix = f"eval_{model}_{dataset}_cad_fixed_alpha"
    suffix = "_tau0p5_temp0p0.json"
    alphas = []
    for path in PIPELINE_DIR.glob(f"{prefix}*{suffix}"):
        name = path.name
        alpha_name = name[len(prefix): -len(suffix)]
        try:
            alphas.append(float(alpha_name.replace("p", ".").replace("m", "-")))
        except ValueError:
            continue
    return sorted(set(alphas))


def choose_score_file(model: str, dataset: str, preferred_alpha: float | None) -> Path:
    if preferred_alpha is not None:
        alpha_name = format_float_for_name(preferred_alpha)
        path = (
            PIPELINE_DIR
            / f"eval_{model}_{dataset}_csv_gated_cad_alpha{alpha_name}_tau0p5_temp0p0.json"
        )
        if path.exists():
            return path

    candidates = sorted(
        PIPELINE_DIR.glob(f"eval_{model}_{dataset}_csv_gated_cad_alpha*_tau0p5_temp0p0.json")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No csv_gated_cad eval file found for {model}/{dataset} under {PIPELINE_DIR}"
        )
    return candidates[0]


def summarize_subset(rows: list[dict], mask: list[bool]) -> tuple[int, float, float]:
    selected = [r for r, keep in zip(rows, mask) if keep]
    if not selected:
        return 0, 0.0, 0.0
    return len(selected), mean_metric(selected, "em"), mean_metric(selected, "f1")


def run(args: argparse.Namespace) -> Path:
    alphas = args.alphas
    if not alphas:
        alphas = discover_alphas(args.model, args.dataset)
    if not alphas:
        raise FileNotFoundError(f"No cad_fixed eval files found for {args.model}/{args.dataset}")

    no_doc_path = find_eval_file(args.model, args.dataset, "no_doc", 1.0)
    no_doc_rows = load_eval(no_doc_path)
    no_doc_em = mean_metric(no_doc_rows, "em")
    no_doc_f1 = mean_metric(no_doc_rows, "f1")

    score_path = choose_score_file(args.model, args.dataset, alphas[0])
    score_rows = load_eval(score_path)
    aligned_scores = align_by_idx(no_doc_rows, score_rows)
    scores = [float(score["p_relevant"]) for _, score in aligned_scores]

    out_rows = []
    for alpha in alphas:
        cad_path = find_eval_file(args.model, args.dataset, "cad_fixed", alpha)
        cad_rows = load_eval(cad_path)
        aligned = align_by_idx(no_doc_rows, cad_rows, score_rows)

        cad_em = mean_metric(cad_rows, "em")
        cad_f1 = mean_metric(cad_rows, "f1")

        for tau in args.taus:
            trusted_mask = [s >= tau for s in scores]
            combined = []
            for no_doc, cad, score_row in aligned:
                score = float(score_row["p_relevant"])
                combined.append(cad if score >= tau else no_doc)

            gated_em = mean_metric(combined, "em")
            gated_f1 = mean_metric(combined, "f1")

            n_trusted, trusted_cad_em, trusted_cad_f1 = summarize_subset(cad_rows, trusted_mask)
            _, trusted_no_doc_em, trusted_no_doc_f1 = summarize_subset(no_doc_rows, trusted_mask)
            untrusted_mask = [not x for x in trusted_mask]
            n_untrusted, untrusted_cad_em, untrusted_cad_f1 = summarize_subset(cad_rows, untrusted_mask)
            _, untrusted_no_doc_em, untrusted_no_doc_f1 = summarize_subset(no_doc_rows, untrusted_mask)

            out_rows.append({
                "model": args.model,
                "dataset": args.dataset,
                "alpha": f"{alpha:g}",
                "tau": f"{tau:g}",
                "n": len(combined),
                "trusted_n": n_trusted,
                "trusted_rate": n_trusted / max(1, len(combined)),
                "cad_fixed_em": cad_em,
                "cad_fixed_f1": cad_f1,
                "no_doc_em": no_doc_em,
                "no_doc_f1": no_doc_f1,
                "offline_gated_em": gated_em,
                "offline_gated_f1": gated_f1,
                "gain_vs_cad_em": gated_em - cad_em,
                "gain_vs_cad_f1": gated_f1 - cad_f1,
                "gain_vs_no_doc_em": gated_em - no_doc_em,
                "gain_vs_no_doc_f1": gated_f1 - no_doc_f1,
                "trusted_cad_em": trusted_cad_em,
                "trusted_cad_f1": trusted_cad_f1,
                "trusted_no_doc_em": trusted_no_doc_em,
                "trusted_no_doc_f1": trusted_no_doc_f1,
                "trusted_cad_minus_no_doc_em": trusted_cad_em - trusted_no_doc_em,
                "trusted_cad_minus_no_doc_f1": trusted_cad_f1 - trusted_no_doc_f1,
                "untrusted_n": n_untrusted,
                "untrusted_cad_em": untrusted_cad_em,
                "untrusted_cad_f1": untrusted_cad_f1,
                "untrusted_no_doc_em": untrusted_no_doc_em,
                "untrusted_no_doc_f1": untrusted_no_doc_f1,
                "untrusted_no_doc_minus_cad_em": untrusted_no_doc_em - untrusted_cad_em,
                "untrusted_no_doc_minus_cad_f1": untrusted_no_doc_f1 - untrusted_cad_f1,
                "score_file": str(score_path.relative_to(PROJECT_ROOT)),
            })

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / f"{args.model}_{args.dataset}_csv_gate_tau_sweep.csv"
    fieldnames = list(out_rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    best_em = max(out_rows, key=lambda r: (r["offline_gated_em"], r["offline_gated_f1"]))
    best_f1 = max(out_rows, key=lambda r: (r["offline_gated_f1"], r["offline_gated_em"]))

    print(f"Wrote {len(out_rows)} rows -> {out_path}")
    print("\nBest by EM:")
    print(
        f"  alpha={best_em['alpha']} tau={best_em['tau']} "
        f"EM={best_em['offline_gated_em']:.4f} "
        f"F1={best_em['offline_gated_f1']:.4f} "
        f"gain_vs_cad=+{best_em['gain_vs_cad_em']:.4f} EM "
        f"+{best_em['gain_vs_cad_f1']:.4f} F1 "
        f"trusted={best_em['trusted_rate']:.2%}"
    )
    print("Best by F1:")
    print(
        f"  alpha={best_f1['alpha']} tau={best_f1['tau']} "
        f"EM={best_f1['offline_gated_em']:.4f} "
        f"F1={best_f1['offline_gated_f1']:.4f} "
        f"gain_vs_cad=+{best_f1['gain_vs_cad_em']:.4f} EM "
        f"+{best_f1['gain_vs_cad_f1']:.4f} F1 "
        f"trusted={best_f1['trusted_rate']:.2%}"
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline tau sweep for CSV-gated CAD")
    parser.add_argument("--model", default="gemma2b")
    parser.add_argument("--dataset", default="nq")
    parser.add_argument("--alphas", type=float, nargs="*", default=None,
                        help="CAD alphas to analyze. Default: discover available cad_fixed eval files.")
    parser.add_argument("--taus", type=float, nargs="*", default=[i / 10 for i in range(1, 10)],
                        help="Tau thresholds to sweep. Default: 0.1 ... 0.9")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
