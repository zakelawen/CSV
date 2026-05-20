"""
Step 10b: Plot Logit Lens results.

Produces three figures per model:
  1. {model}_logp_by_layer.png
     Three curves (no_ctx / gold_ctx / dis_ctx), each showing
     mean logP across layers. Tells you "at which layer does each
     condition's answer probability mature".

  2. {model}_delta_by_layer.png
     Two curves: DeltaP_gold-no, DeltaP_dis-no.
     The peak of DeltaP_gold-no is the recommended Step 11 contrastive
     decoding layer. DeltaP_dis-no should be <= 0.

  3. {model}_logp_by_subset.png
     Same as (1) but split into NQ vs TriviaQA panels - shows
     whether the gap behavior is dataset-dependent.

Usage:
    python scripts/step10b_plot_logit_lens.py --model gemma2b
    python scripts/step10b_plot_logit_lens.py --model all
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "logit_lens"

MODEL_KEYS = ["gemma2b", "gemma9b", "qwen3_4b"]

COLORS = {
    "no_ctx":   "#7f7f7f",  # gray
    "gold_ctx": "#2ca02c",  # green
    "dis_ctx":  "#d62728",  # red
}
LABELS = {
    "no_ctx":   "no_ctx (no document)",
    "gold_ctx": "gold_ctx (relevant doc)",
    "dis_ctx":  "dis_ctx (distracting doc)",
}


def to_array(seq):
    """Convert list with possible None to np.array with NaN."""
    return np.array([np.nan if x is None else x for x in seq], dtype=np.float64)


def plot_logp_by_layer(payload, model_key, out_path):
    L = payload["num_layers"]
    layers = np.arange(L)
    fig, ax = plt.subplots(figsize=(11, 5))

    for cond in ["no_ctx", "gold_ctx", "dis_ctx"]:
        y = to_array(payload["per_layer"][cond]["mean_logP_combined"])
        ax.plot(layers, y, "-", linewidth=1.6, marker="o", markersize=3,
                color=COLORS[cond], label=LABELS[cond])

    emerge = payload.get("answer_signal_emerge_layer", -1)
    if emerge >= 0:
        ax.axvline(emerge, color="#2ca02c", linestyle=":", linewidth=0.8,
                   alpha=0.6, label=f"emerge layer = {emerge}")

    ax.set_xlabel("Layer (0=embedding, 1..N=transformer)")
    ax.set_ylabel("Mean logP(answer span)")
    ax.set_title(f"{model_key} - Logit Lens: answer logP across layers (combined)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_path}")


def plot_delta_by_layer(payload, model_key, out_path):
    L = payload["num_layers"]
    layers = np.arange(L)
    fig, ax = plt.subplots(figsize=(11, 5))

    delta_gold = to_array(payload["delta_gold_minus_no"])
    delta_dis  = to_array(payload["delta_dis_minus_no"])

    ax.plot(layers, delta_gold, "-", linewidth=1.8, marker="o", markersize=3,
            color="#2ca02c", label="DeltaP_gold - no_ctx")
    ax.plot(layers, delta_dis,  "-", linewidth=1.8, marker="o", markersize=3,
            color="#d62728", label="DeltaP_dis - no_ctx")

    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)

    peak_layer = payload.get("delta_gold_peak_layer", -1)
    peak_val = payload.get("delta_gold_peak_value", 0.0)
    if peak_layer >= 0:
        ax.axvline(peak_layer, color="#2ca02c", linestyle=":", linewidth=1.0)
        ax.annotate(
            f"peak DeltaP_gold = {peak_val:+.3f}\nat layer {peak_layer}",
            xy=(peak_layer, peak_val),
            xytext=(8, -8), textcoords="offset points",
            fontsize=9, color="#2ca02c",
        )

    ax.set_xlabel("Layer (0=embedding, 1..N=transformer)")
    ax.set_ylabel("Mean Delta logP")
    ax.set_title(f"{model_key} - DeltalogP relative to no_ctx, per layer (combined)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_path}")


def plot_per_subset(payload, model_key, out_path):
    L = payload["num_layers"]
    layers = np.arange(L)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for ax, sub in zip(axes, ["nq", "triviaqa"]):
        for cond in ["no_ctx", "gold_ctx", "dis_ctx"]:
            y = to_array(payload["per_layer"][cond][f"mean_logP_{sub}"])
            ax.plot(layers, y, "-", linewidth=1.4, marker="o", markersize=2.5,
                    color=COLORS[cond], label=LABELS[cond])
        n = payload["num_samples"].get(sub, 0)
        ax.set_xlabel("Layer")
        ax.set_title(f"{sub} (N={n})")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean logP(answer span)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{model_key} - Logit Lens by subset")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_path}")


def process_model(model_key):
    json_path = RESULTS_DIR / f"{model_key}_logit_lens.json"
    if not json_path.exists():
        print(f"  [skip] {json_path} not found. Run step10_logit_lens first.")
        return

    print(f"\n=== Plotting {model_key} ===")
    with open(json_path) as f:
        payload = json.load(f)

    plot_logp_by_layer(payload, model_key,
                       RESULTS_DIR / f"{model_key}_logp_by_layer.png")
    plot_delta_by_layer(payload, model_key,
                        RESULTS_DIR / f"{model_key}_delta_by_layer.png")
    plot_per_subset(payload, model_key,
                    RESULTS_DIR / f"{model_key}_logp_by_subset.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all")
    args = parser.parse_args()

    if args.model == "all":
        for m in MODEL_KEYS:
            process_model(m)
    else:
        process_model(args.model)


if __name__ == "__main__":
    main()