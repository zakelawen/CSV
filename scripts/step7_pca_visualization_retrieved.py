# """
# Step 7R: PCA Visualization for Retrieved-Doc Hidden States

# For each model × dataset, perform PCA to 2D on the last-token hidden states
# at EVERY layer, colored by relevant / distracting.
# Reproduces the style of Yeh & Li Figure 6 (layer-wise evolution).

# This is QUALITATIVE analysis only — layer selection is done by Step 8 / Step 9.

# Usage:
#     python scripts/step7_pca_visualization_retrieved.py --model gemma2b
#     python scripts/step7_pca_visualization.py --model llama8b
#     python scripts/step7_pca_visualization_retrieved.py --model all

# Output:
#     results/pca_retrieved/{model_key}_pca_{dataset}.png  (one figure per model × dataset)
# """

# import argparse
# import math
# from pathlib import Path

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA


# # ---------------------------------------------------------------------------
# # Paths
# # ---------------------------------------------------------------------------

# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# HIDDEN_DIR = PROJECT_ROOT / "data" / "hidden_states_retrieved"
# RESULTS_DIR = PROJECT_ROOT / "results" / "pca_retrieved"

# MODEL_KEYS = ["gemma2b", "llama8b"]


# # ---------------------------------------------------------------------------
# # Main logic
# # ---------------------------------------------------------------------------

# def process_model(model_key: str):
#     RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#     hs_path = HIDDEN_DIR / model_key / "train_hidden_states.pt"
#     if not hs_path.exists():
#         print(f"  [skip] {hs_path} not found. Run step6_extract_hidden_states_retrieved.py first.")
#         return

#     print(f"\nProcessing {model_key} …")
#     data = torch.load(hs_path, map_location="cpu", weights_only=False)
#     hidden_states = data["hidden_states"]  # [N, L, D]
#     labels = data["labels"]                # list of "relevant" / "distracting"
#     sources = data["dataset_source"]       # list of "nq" / "triviaqa" / "unknown"
#     N, L, D = hidden_states.shape

#     is_relevant = np.array([l == "relevant" for l in labels])
#     source_arr = np.array(sources)

#     datasets = sorted(set(sources))
#     # Skip embedding layer (index 0), only show transformer layers 1..L-1
#     layer_indices = list(range(1, L))
#     n_layers = len(layer_indices)

#     for ds in datasets:
#         ds_mask = source_arr == ds
#         n_ds = ds_mask.sum()

#         # Grid layout: aim for ~6 columns
#         n_cols = min(6, n_layers)
#         n_rows = math.ceil(n_layers / n_cols)

#         fig, axes = plt.subplots(n_rows, n_cols,
#                                  figsize=(3.5 * n_cols, 3 * n_rows))
#         axes = np.array(axes).flatten()

#         for idx, layer_idx in enumerate(layer_indices):
#             ax = axes[idx]
#             hs = hidden_states[ds_mask, layer_idx, :].numpy()
#             rel_mask = is_relevant[ds_mask]

#             pca = PCA(n_components=2)
#             coords = pca.fit_transform(hs)

#             ax.scatter(coords[~rel_mask, 0], coords[~rel_mask, 1],
#                        c="#E57373", alpha=0.3, s=5, rasterized=True)
#             ax.scatter(coords[rel_mask, 0], coords[rel_mask, 1],
#                        c="#4CAF50", alpha=0.3, s=5, rasterized=True)

#             ax.set_title(f"L{layer_idx}", fontsize=9)
#             ax.tick_params(labelsize=6)
#             ax.set_xticks([])
#             ax.set_yticks([])

#         # Hide unused axes
#         for idx in range(n_layers, len(axes)):
#             axes[idx].set_visible(False)

#         # Manual legend
#         from matplotlib.lines import Line2D
#         legend_elements = [
#             Line2D([0], [0], marker="o", color="w", markerfacecolor="#4CAF50",
#                    markersize=6, label="Relevant"),
#             Line2D([0], [0], marker="o", color="w", markerfacecolor="#E57373",
#                    markersize=6, label="Distracting"),
#         ]
#         fig.legend(handles=legend_elements, loc="upper right", fontsize=9,
#                    framealpha=0.9)

#         ds_label = ds.upper() if ds != "unknown" else "All Data"
#         fig.suptitle(f"{model_key} — Retrieved-doc {ds_label} — PCA All Layers (n={n_ds})",
#                      fontsize=13)
#         plt.tight_layout(rect=[0, 0, 0.95, 0.95])

#         fig_path = RESULTS_DIR / f"{model_key}_pca_{ds}.png"
#         fig.savefig(fig_path, dpi=150)
#         plt.close(fig)
#         print(f"  Saved → {fig_path}")


# def main():
#     parser = argparse.ArgumentParser(description="Step 7R: PCA visualization for retrieved-doc hidden states")
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
Step 7R: PCA Visualization for Retrieved-Doc Hidden States

For each model × dataset, perform PCA to 2D on the last-token hidden states
at EVERY layer, colored by relevant / distracting.
Reproduces the style of Yeh & Li Figure 6 (layer-wise evolution).

This is QUALITATIVE analysis only — layer selection is done by Step 8 / Step 9.

Usage:
    python scripts/step7_pca_visualization_retrieved.py --model gemma2b
    python scripts/step7_pca_visualization_retrieved.py --model gemma9b
    python scripts/step7_pca_visualization_retrieved.py --model qwen3_4b
    python scripts/step7_pca_visualization_retrieved.py --model all

Output:
    results/pca_retrieved/{model_key}_pca_{dataset}.png  (one figure per model × dataset)
"""

import argparse
import math
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HIDDEN_DIR = PROJECT_ROOT / "data" / "hidden_states_retrieved"
RESULTS_DIR = PROJECT_ROOT / "results" / "pca_retrieved"

MODEL_KEYS = ["gemma2b", "gemma9b", "qwen3_4b"]


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def process_model(model_key: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    hs_path = HIDDEN_DIR / model_key / "train_hidden_states.pt"
    if not hs_path.exists():
        print(f"  [skip] {hs_path} not found. Run step6_extract_hidden_states_retrieved.py first.")
        return

    print(f"\nProcessing {model_key} …")
    data = torch.load(hs_path, map_location="cpu", weights_only=False)
    hidden_states = data["hidden_states"]  # [N, L, D]
    labels = data["labels"]                # list of "relevant" / "distracting"
    sources = data["dataset_source"]       # list of "nq" / "triviaqa" / "unknown"
    N, L, D = hidden_states.shape

    is_relevant = np.array([l == "relevant" for l in labels])
    source_arr = np.array(sources)

    datasets = sorted(set(sources))
    # Skip embedding layer (index 0), only show transformer layers 1..L-1
    layer_indices = list(range(1, L))
    n_layers = len(layer_indices)

    for ds in datasets:
        ds_mask = source_arr == ds
        n_ds = ds_mask.sum()

        # Grid layout: aim for ~6 columns
        n_cols = min(6, n_layers)
        n_rows = math.ceil(n_layers / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(3.5 * n_cols, 3 * n_rows))
        axes = np.array(axes).flatten()

        for idx, layer_idx in enumerate(layer_indices):
            ax = axes[idx]
            hs = hidden_states[ds_mask, layer_idx, :].numpy()
            rel_mask = is_relevant[ds_mask]

            pca = PCA(n_components=2)
            coords = pca.fit_transform(hs)

            ax.scatter(coords[~rel_mask, 0], coords[~rel_mask, 1],
                       c="#E57373", alpha=0.3, s=5, rasterized=True)
            ax.scatter(coords[rel_mask, 0], coords[rel_mask, 1],
                       c="#4CAF50", alpha=0.3, s=5, rasterized=True)

            ax.set_title(f"L{layer_idx}", fontsize=9)
            ax.tick_params(labelsize=6)
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused axes
        for idx in range(n_layers, len(axes)):
            axes[idx].set_visible(False)

        # Manual legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#4CAF50",
                   markersize=6, label="Relevant"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#E57373",
                   markersize=6, label="Distracting"),
        ]
        fig.legend(handles=legend_elements, loc="upper right", fontsize=9,
                   framealpha=0.9)

        ds_label = ds.upper() if ds != "unknown" else "All Data"
        fig.suptitle(f"{model_key} — Retrieved-doc {ds_label} — PCA All Layers (n={n_ds})",
                     fontsize=13)
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])

        fig_path = RESULTS_DIR / f"{model_key}_pca_{ds}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Saved → {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Step 7R: PCA visualization for retrieved-doc hidden states")
    parser.add_argument("--model", type=str, default="all",
                        choices=MODEL_KEYS + ["all"])
    args = parser.parse_args()

    models = MODEL_KEYS if args.model == "all" else [args.model]
    for m in models:
        process_model(m)

    print("\nDone.")


if __name__ == "__main__":
    main()