# """
# Step 6bR: Verify Retrieved-Doc Embedding Norm Distributions

# Check whether the L2 norms of last-token hidden states are concentrated
# at each layer. This validates the von Mises-Fisher (vMF) modeling assumption
# used in CSV training (Step 9).

# Reference: TSV paper Appendix G - LLaMA-3.1-8B norms ~140, Qwen-2.5-7B ~300-330.

# Usage:
#     python scripts/step6b_verify_norms_retrieved.py --model gemma2b
#     python scripts/step6b_verify_norms.py --model llama8b
#     python scripts/step6b_verify_norms_retrieved.py --model all

# Output:
#     results/norms_retrieved/{model_key}_norm_distributions.png
#     results/norms_retrieved/{model_key}_norm_stats.json
# """

# import json
# import argparse
# from pathlib import Path

# import torch
# import numpy as np
# import matplotlib.pyplot as plt


# # ---------------------------------------------------------------------------
# # Paths
# # ---------------------------------------------------------------------------

# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# HIDDEN_DIR = PROJECT_ROOT / "data" / "hidden_states_retrieved"
# RESULTS_DIR = PROJECT_ROOT / "results" / "norms_retrieved"

# MODEL_KEYS = ["gemma2b", "llama8b"]


# # ---------------------------------------------------------------------------
# # Main logic
# # ---------------------------------------------------------------------------

# def process_model(model_key: str):
#     RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#     # Load train hidden states (larger sample for more reliable statistics)
#     hs_path = HIDDEN_DIR / model_key / "train_hidden_states.pt"
#     if not hs_path.exists():
#         print(f"  [skip] {hs_path} not found. Run step6_extract_hidden_states_retrieved.py first.")
#         return

#     print(f"\nProcessing {model_key} ...")
#     data = torch.load(hs_path, map_location="cpu", weights_only=False)
#     hidden_states = data["hidden_states"]  # [N, L, D]
#     N, L, D = hidden_states.shape
#     print(f"  Shape: N={N}, L={L} layers, D={D}")

#     # Compute L2 norms: [N, L]
#     norms = torch.norm(hidden_states, p=2, dim=2)  # [N, L]

#     # Select layers to visualize: embedding, ~25%, ~50%, ~75%, final
#     num_transformer_layers = L - 1  # L includes embedding layer at index 0
#     layer_indices = sorted(set([
#         0,                                    # embedding layer
#         1,                                    # first transformer layer
#         num_transformer_layers // 4,          # ~25%
#         num_transformer_layers // 2,          # ~50%
#         3 * num_transformer_layers // 4,      # ~75%
#         num_transformer_layers,               # final transformer layer (= L-1)
#     ]))

#     # --- Plot histograms ---
#     fig, axes = plt.subplots(2, 3, figsize=(15, 8))
#     axes = axes.flatten()

#     for idx, (ax, layer_idx) in enumerate(zip(axes, layer_indices)):
#         layer_norms = norms[:, layer_idx].numpy()
#         ax.hist(layer_norms, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
#         ax.set_title(f"Layer {layer_idx}" + (" (emb)" if layer_idx == 0 else
#                      f" (final)" if layer_idx == num_transformer_layers else ""))
#         ax.set_xlabel("L2 Norm")
#         ax.set_ylabel("Count")

#         mean_val = layer_norms.mean()
#         std_val = layer_norms.std()
#         cv = std_val / mean_val if mean_val > 0 else float("inf")
#         ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5)
#         ax.text(0.95, 0.95, f"mu={mean_val:.1f}\nsigma={std_val:.1f}\nCV={cv:.3f}",
#                 transform=ax.transAxes, ha="right", va="top", fontsize=9,
#                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

#     # Hide unused axes
#     for idx in range(len(layer_indices), len(axes)):
#         axes[idx].set_visible(False)

#     fig.suptitle(f"{model_key} - Retrieved-Doc L2 Norm Distributions per Layer", fontsize=14)
#     plt.tight_layout()

#     fig_path = RESULTS_DIR / f"{model_key}_norm_distributions.png"
#     fig.savefig(fig_path, dpi=150)
#     plt.close(fig)
#     print(f"  Saved figure -> {fig_path}")

#     # --- Compute stats for ALL layers and save ---
#     stats = {}
#     for layer_idx in range(L):
#         layer_norms = norms[:, layer_idx].numpy()
#         stats[f"layer_{layer_idx}"] = {
#             "mean": float(layer_norms.mean()),
#             "std": float(layer_norms.std()),
#             "min": float(layer_norms.min()),
#             "max": float(layer_norms.max()),
#             "cv": float(layer_norms.std() / layer_norms.mean()) if layer_norms.mean() > 0 else None,
#         }

#     stats_path = RESULTS_DIR / f"{model_key}_norm_stats.json"
#     with open(stats_path, "w") as f:
#         json.dump(stats, f, indent=2)
#     print(f"  Saved stats -> {stats_path}")

#     # --- Summary: check ALL layers ---
#     warning_layers = []
#     for layer_idx in range(L):
#         cv = stats[f"layer_{layer_idx}"]["cv"]
#         if cv is not None and cv >= 0.1:
#             warning_layers.append((layer_idx, cv))

#     final_stats = stats[f"layer_{num_transformer_layers}"]
#     print(f"  Final layer (layer {num_transformer_layers}): "
#           f"mean={final_stats['mean']:.1f}, std={final_stats['std']:.1f}, CV={final_stats['cv']:.4f}")

#     if warning_layers:
#         print(f"  WARNING {len(warning_layers)} layer(s) have CV >= 0.1 (norm not concentrated):")
#         for li, cv in warning_layers:
#             print(f"      layer {li}: CV={cv:.4f}")
#         print(f"    If CSV is trained on these layers, consider explicit L2 normalization.")
#     else:
#         print(f"  OK All {L} layers have CV < 0.1 - vMF modeling is appropriate at every layer.")


# def main():
#     parser = argparse.ArgumentParser(description="Step 6bR: Verify retrieved-doc embedding norm distributions")
#     parser.add_argument("--model", type=str, default="all",
#                         choices=MODEL_KEYS + ["all"])
#     args = parser.parse_args()

#     models = MODEL_KEYS if args.model == "all" else [args.model]
#     for m in models:
#         process_model(m)

#     print("\nDone.")


# if __name__ == "__main__":
#     main()

"""
Step 6bR: Verify Retrieved-Doc Embedding Norm Distributions

Check whether the L2 norms of last-token hidden states are concentrated
at each layer. This validates the von Mises-Fisher (vMF) modeling assumption
used in CSV training (Step 9).

Reference: TSV paper Appendix G - LLaMA-3.1-8B norms ~140, Qwen-2.5-7B ~300-330.

Usage:
    python scripts/step6b_verify_norms_retrieved.py --model gemma2b
    python scripts/step6b_verify_norms_retrieved.py --model gemma9b
    python scripts/step6b_verify_norms_retrieved.py --model qwen3_4b
    python scripts/step6b_verify_norms_retrieved.py --model all

Output:
    results/norms_retrieved/{model_key}_norm_distributions.png
    results/norms_retrieved/{model_key}_norm_stats.json
"""

import json
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HIDDEN_DIR = PROJECT_ROOT / "data" / "hidden_states_retrieved"
RESULTS_DIR = PROJECT_ROOT / "results" / "norms_retrieved"

MODEL_KEYS = ["gemma2b", "gemma9b", "qwen3_4b"]


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def process_model(model_key: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load train hidden states (larger sample for more reliable statistics)
    hs_path = HIDDEN_DIR / model_key / "train_hidden_states.pt"
    if not hs_path.exists():
        print(f"  [skip] {hs_path} not found. Run step6_extract_hidden_states_retrieved.py first.")
        return

    print(f"\nProcessing {model_key} ...")
    data = torch.load(hs_path, map_location="cpu", weights_only=False)
    hidden_states = data["hidden_states"]  # [N, L, D]
    N, L, D = hidden_states.shape
    print(f"  Shape: N={N}, L={L} layers, D={D}")

    # Compute L2 norms: [N, L]
    norms = torch.norm(hidden_states, p=2, dim=2)  # [N, L]

    # Select layers to visualize: embedding, ~25%, ~50%, ~75%, final
    num_transformer_layers = L - 1  # L includes embedding layer at index 0
    layer_indices = sorted(set([
        0,                                    # embedding layer
        1,                                    # first transformer layer
        num_transformer_layers // 4,          # ~25%
        num_transformer_layers // 2,          # ~50%
        3 * num_transformer_layers // 4,      # ~75%
        num_transformer_layers,               # final transformer layer (= L-1)
    ]))

    # --- Plot histograms ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, (ax, layer_idx) in enumerate(zip(axes, layer_indices)):
        layer_norms = norms[:, layer_idx].numpy()
        ax.hist(layer_norms, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_title(f"Layer {layer_idx}" + (" (emb)" if layer_idx == 0 else
                     f" (final)" if layer_idx == num_transformer_layers else ""))
        ax.set_xlabel("L2 Norm")
        ax.set_ylabel("Count")

        mean_val = layer_norms.mean()
        std_val = layer_norms.std()
        cv = std_val / mean_val if mean_val > 0 else float("inf")
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5)
        ax.text(0.95, 0.95, f"mu={mean_val:.1f}\nsigma={std_val:.1f}\nCV={cv:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Hide unused axes
    for idx in range(len(layer_indices), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"{model_key} - Retrieved-Doc L2 Norm Distributions per Layer", fontsize=14)
    plt.tight_layout()

    fig_path = RESULTS_DIR / f"{model_key}_norm_distributions.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved figure -> {fig_path}")

    # --- Compute stats for ALL layers and save ---
    stats = {}
    for layer_idx in range(L):
        layer_norms = norms[:, layer_idx].numpy()
        stats[f"layer_{layer_idx}"] = {
            "mean": float(layer_norms.mean()),
            "std": float(layer_norms.std()),
            "min": float(layer_norms.min()),
            "max": float(layer_norms.max()),
            "cv": float(layer_norms.std() / layer_norms.mean()) if layer_norms.mean() > 0 else None,
        }

    stats_path = RESULTS_DIR / f"{model_key}_norm_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats -> {stats_path}")

    # --- Summary: check ALL layers ---
    warning_layers = []
    for layer_idx in range(L):
        cv = stats[f"layer_{layer_idx}"]["cv"]
        if cv is not None and cv >= 0.1:
            warning_layers.append((layer_idx, cv))

    final_stats = stats[f"layer_{num_transformer_layers}"]
    print(f"  Final layer (layer {num_transformer_layers}): "
          f"mean={final_stats['mean']:.1f}, std={final_stats['std']:.1f}, CV={final_stats['cv']:.4f}")

    if warning_layers:
        print(f"  WARNING {len(warning_layers)} layer(s) have CV >= 0.1 (norm not concentrated):")
        for li, cv in warning_layers:
            print(f"      layer {li}: CV={cv:.4f}")
        print(f"    If CSV is trained on these layers, consider explicit L2 normalization.")
    else:
        print(f"  OK All {L} layers have CV < 0.1 - vMF modeling is appropriate at every layer.")


def main():
    parser = argparse.ArgumentParser(description="Step 6bR: Verify retrieved-doc embedding norm distributions")
    parser.add_argument("--model", type=str, default="all",
                        choices=MODEL_KEYS + ["all"])
    args = parser.parse_args()

    models = MODEL_KEYS if args.model == "all" else [args.model]
    for m in models:
        process_model(m)

    print("\nDone.")


if __name__ == "__main__":
    main()