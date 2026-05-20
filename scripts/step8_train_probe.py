# """
# Step 8: Train Linear Probes at Every Layer

# For each layer, train a linear classifier on the hidden states to distinguish
# relevant vs. distracting documents.

# Pipeline per layer:
#   1. Extract layer's hidden states from train/eval .pt files
#   2. PCA → 16 dimensions
#   3. L2 normalize
#   4. Train logistic regression (single run, full training set)

# Metrics: AUROC, Accuracy, Avg Margin.
# Reports: per-dataset (NQ, TriviaQA) and combined.

# Usage:
#     python scripts/step8_train_probe.py --model gemma2b
#     python scripts/step8_train_probe.py --model llama8b
#     python scripts/step8_train_probe.py --model all

# Output:
#     results/probe/{model_key}_probe_results.json
#     results/probe/{model_key}_auroc_by_layer.png
# """

# import json
# import argparse
# import warnings
# from pathlib import Path

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score, accuracy_score
# from sklearn.preprocessing import normalize

# # Suppress PCA explained_variance_ratio_ warning on constant-variance layers
# warnings.filterwarnings("ignore", category=RuntimeWarning,
#                         message="invalid value encountered in divide")


# # ---------------------------------------------------------------------------
# # Paths
# # ---------------------------------------------------------------------------

# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# HIDDEN_DIR = PROJECT_ROOT / "data" / "hidden_states"
# RESULTS_DIR = PROJECT_ROOT / "results" / "probe"

# MODEL_KEYS = ["gemma2b", "llama8b"]

# PCA_DIM = 16


# # ---------------------------------------------------------------------------
# # Probe training for one layer + one data subset
# # ---------------------------------------------------------------------------

# def train_probe_single(
#     train_hs: np.ndarray,      # [N_train, D]
#     train_labels: np.ndarray,  # [N_train] binary
#     eval_hs: np.ndarray,       # [N_eval, D]
#     eval_labels: np.ndarray,   # [N_eval] binary
#     pca_dim: int = PCA_DIM,
# ) -> dict:
#     """
#     Train a single logistic regression probe using the full training set.
#     """
#     # PCA fit on train, transform both
#     n_components = min(pca_dim, train_hs.shape[1], train_hs.shape[0])
#     pca = PCA(n_components=n_components)
#     train_pca = pca.fit_transform(train_hs)
#     eval_pca = pca.transform(eval_hs)

#     # L2 normalize
#     train_norm = normalize(train_pca, norm="l2")
#     eval_norm = normalize(eval_pca, norm="l2")

#     clf = LogisticRegression(
#         max_iter=1000,
#         solver="lbfgs",
#         C=1.0,
#     )
#     clf.fit(train_norm, train_labels)

#     # Evaluate on eval set
#     probs = clf.predict_proba(eval_norm)[:, 1]
#     preds = clf.predict(eval_norm)

#     auroc = roc_auc_score(eval_labels, probs)
#     acc = accuracy_score(eval_labels, preds)

#     # Avg margin: mean of |decision_function(x)| on eval set.
#     # Higher = more confident predictions. Relative measure across layers.
#     margin = np.abs(clf.decision_function(eval_norm)).mean()

#     return {
#         "auroc": float(auroc),
#         "accuracy": float(acc),
#         "avg_margin": float(margin),
#     }


# # ---------------------------------------------------------------------------
# # Main logic
# # ---------------------------------------------------------------------------

# def process_model(model_key: str):
#     RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#     train_path = HIDDEN_DIR / model_key / "train_hidden_states.pt"
#     eval_path = HIDDEN_DIR / model_key / "eval_hidden_states.pt"
#     if not train_path.exists() or not eval_path.exists():
#         print(f"  [skip] Hidden states not found for {model_key}. Run step6 first.")
#         return

#     print(f"\nProcessing {model_key} …")

#     train_data = torch.load(train_path, map_location="cpu", weights_only=False)
#     eval_data = torch.load(eval_path, map_location="cpu", weights_only=False)

#     train_hs = train_data["hidden_states"].numpy()   # [N_train, L, D]
#     eval_hs = eval_data["hidden_states"].numpy()     # [N_eval, L, D]

#     train_labels_str = train_data["labels"]
#     eval_labels_str = eval_data["labels"]
#     train_labels = np.array([1 if l == "relevant" else 0 for l in train_labels_str])
#     eval_labels = np.array([1 if l == "relevant" else 0 for l in eval_labels_str])

#     train_sources = np.array(train_data["dataset_source"])
#     eval_sources = np.array(eval_data["dataset_source"])

#     N_train, L, D = train_hs.shape
#     num_transformer_layers = L - 1
#     print(f"  Train: {N_train}, Eval: {eval_hs.shape[0]}, "
#           f"Layers: {L} (0=emb, 1..{num_transformer_layers}=transformer)")

#     datasets = sorted(set(train_data["dataset_source"]))
#     subsets = {ds: {
#         "train_mask": train_sources == ds,
#         "eval_mask": eval_sources == ds,
#     } for ds in datasets}
#     subsets["combined"] = {
#         "train_mask": np.ones(N_train, dtype=bool),
#         "eval_mask": np.ones(eval_hs.shape[0], dtype=bool),
#     }

#     # Results: layer → subset → metrics
#     all_results = {}

#     for layer_idx in range(L):
#         layer_key = f"layer_{layer_idx}"
#         all_results[layer_key] = {}

#         tr_hs_layer = train_hs[:, layer_idx, :]  # [N_train, D]
#         ev_hs_layer = eval_hs[:, layer_idx, :]   # [N_eval, D]

#         for subset_name, masks in subsets.items():
#             tr_mask = masks["train_mask"]
#             ev_mask = masks["eval_mask"]

#             metrics = train_probe_single(
#                 tr_hs_layer[tr_mask], train_labels[tr_mask],
#                 ev_hs_layer[ev_mask], eval_labels[ev_mask],
#             )
#             all_results[layer_key][subset_name] = metrics

#         # Progress
#         combined = all_results[layer_key]["combined"]
#         print(f"  Layer {layer_idx:2d}: AUROC={combined['auroc']:.4f}  "
#               f"Acc={combined['accuracy']:.4f}")

#     # --- Save results ---
#     results_path = RESULTS_DIR / f"{model_key}_probe_results.json"
#     with open(results_path, "w") as f:
#         json.dump(all_results, f, indent=2)
#     print(f"\n  Saved results → {results_path}")

#     # --- Find best layer ---
#     best_layer = None
#     best_auroc = -1
#     for layer_key, subsets_data in all_results.items():
#         auroc = subsets_data["combined"]["auroc"]
#         if auroc > best_auroc:
#             best_auroc = auroc
#             best_layer = layer_key
#     print(f"  Best layer (combined): {best_layer} with AUROC={best_auroc:.4f}")

#     # --- Plot AUROC by layer ---
#     fig, ax = plt.subplots(figsize=(12, 5))

#     for subset_name in list(datasets) + ["combined"]:
#         aurocs = [all_results[f"layer_{i}"][subset_name]["auroc"] for i in range(L)]

#         line_style = "-" if subset_name == "combined" else "--"
#         line_width = 2.0 if subset_name == "combined" else 1.2
#         ax.plot(range(L), aurocs, line_style, linewidth=line_width, label=subset_name)

#     ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="chance")
#     ax.set_xlabel("Layer Index (0=embedding, 1..N=transformer layers)")
#     ax.set_ylabel("AUROC")
#     ax.set_title(f"{model_key} — Linear Probe AUROC by Layer")
#     ax.legend(fontsize=9)
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()

#     fig_path = RESULTS_DIR / f"{model_key}_auroc_by_layer.png"
#     fig.savefig(fig_path, dpi=150)
#     plt.close(fig)
#     print(f"  Saved figure → {fig_path}")


# def main():
#     parser = argparse.ArgumentParser(description="Step 8: Train linear probes")
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
Step 8: Train Probes at Every Layer (LR / MLP / NearestCentroid)

For each layer, train a probe on the hidden states to distinguish
relevant vs. distracting documents. Three classifier families are run:

  - lr:        Logistic Regression (linear decision boundary)
  - mlp:       MLPClassifier with hidden_layer_sizes=(128,)
               (single hidden layer, ReLU + L2)
  - centroid:  NearestCentroid in L2-normalized PCA space
               (= cosine nearest-centroid; matches CSV evaluation rule)
  - mass_mean: mass-mean probe direction in L2-normalized PCA space
               (= project onto mean(relevant) - mean(distracting))

Pipeline per layer:
  1. Extract layer's hidden states from train/eval .pt files
  2. PCA → 16 dimensions (fit on train, transform both)
  3. L2 normalize
  4. Fit each classifier on full training set, evaluate on eval set

Metrics: AUROC, Accuracy, Avg Margin.
Reports: per-dataset (NQ, TriviaQA) and combined.

Usage:
    python scripts/step8_train_probe.py --model gemma2b
    python scripts/step8_train_probe.py --model gemma9b
    python scripts/step8_train_probe.py --model qwen3_4b
    python scripts/step8_train_probe.py --model all

    # Run a single classifier only
    python scripts/step8_train_probe.py --model gemma2b --classifier mlp

Output (per model):
    results/probe/{model_key}_probe_results_lr.json
    results/probe/{model_key}_probe_results_mlp.json
    results/probe/{model_key}_probe_results_centroid.json
    results/probe/{model_key}_probe_results_mass_mean.json
    results/probe/{model_key}_auroc_by_layer.png   # all classifiers in one figure

MLP hidden-size choice:
    hidden_layer_sizes=(128,) was selected by gemma2b layer-12 grid search
    over {(32,), (64,), (128,), (32,16), (64,32)} with 3 random seeds.
    (128,) had the highest mean AUROC (0.6976) and lowest std (0.0019).
"""

import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize

# Suppress PCA explained_variance_ratio_ warning on constant-variance layers
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="invalid value encountered in divide")
# MLPClassifier may complain about convergence on some layers; that's expected
# (early-stopping handles it). Don't spam the log.
warnings.filterwarnings("ignore", message="Stochastic Optimizer:.*")
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HIDDEN_DIR = PROJECT_ROOT / "data" / "hidden_states"
RESULTS_DIR = PROJECT_ROOT / "results" / "probe"

# Active main-experiment models (LLaMA-8B archived).
MODEL_KEYS = ["gemma2b", "gemma9b", "qwen3_4b"]

CLASSIFIERS = ["lr", "mlp", "centroid", "mass_mean"]

PCA_DIM = 16


# ---------------------------------------------------------------------------
# Probe training for one layer + one data subset + one classifier
# ---------------------------------------------------------------------------

def train_probe_single(
    train_hs: np.ndarray,      # [N_train, D]
    train_labels: np.ndarray,  # [N_train] binary
    eval_hs: np.ndarray,       # [N_eval, D]
    eval_labels: np.ndarray,   # [N_eval] binary
    classifier: str = "lr",
    pca_dim: int = PCA_DIM,
) -> dict:
    """
    Train a single probe in (PCA → L2-normalize) space using the full
    training set, evaluate on the eval set.

    Args:
        classifier: one of "lr", "mlp", "centroid", "mass_mean".

    Returns:
        dict with keys "auroc", "accuracy", "avg_margin".
    """
    # PCA fit on train, transform both
    n_components = min(pca_dim, train_hs.shape[1], train_hs.shape[0])
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_hs)
    eval_pca = pca.transform(eval_hs)

    # L2 normalize → cosine geometry
    train_norm = normalize(train_pca, norm="l2")
    eval_norm = normalize(eval_pca, norm="l2")

    if classifier == "lr":
        clf = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
        )
        clf.fit(train_norm, train_labels)
        probs = clf.predict_proba(eval_norm)[:, 1]
        preds = clf.predict(eval_norm)
        # Mean of |decision_function| — a relative confidence measure.
        margin = float(np.abs(clf.decision_function(eval_norm)).mean())

    elif classifier == "mlp":
        # Single hidden layer, 128 units. Chosen by layer-12 grid search
        # on gemma2b: highest mean AUROC and lowest std among
        # {(32,), (64,), (128,), (32,16), (64,32)}.
        clf = MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        clf.fit(train_norm, train_labels)
        probs = clf.predict_proba(eval_norm)[:, 1]
        preds = clf.predict(eval_norm)
        # MLP has no decision_function. Use logit gap as a confidence
        # surrogate, comparable to LR's margin in spirit.
        eps = 1e-8
        logit = np.log((probs + eps) / (1.0 - probs + eps))
        margin = float(np.abs(logit).mean())

    elif classifier == "centroid":
        # Nearest centroid in L2-normalized space. Compute centroids directly
        # instead of using sklearn.neighbors.NearestCentroid: sklearn's fit
        # path divides by per-feature variance and raises when a layer/subset
        # is completely constant, while raw nearest-centroid remains defined.
        if not np.array_equal(np.unique(train_labels), np.array([0, 1])):
            raise ValueError(
                f"Centroid probe requires both labels, got {np.unique(train_labels)}"
            )
        c0 = train_norm[train_labels == 0].mean(axis=0)
        c1 = train_norm[train_labels == 1].mean(axis=0)

        d0 = np.linalg.norm(eval_norm - c0, axis=1)
        d1 = np.linalg.norm(eval_norm - c1, axis=1)
        preds = (d1 < d0).astype(int)

        # Build a continuous score:
        # signed distance gap (d_to_class0 - d_to_class1). Larger = farther
        # from "distracting" centroid, closer to "relevant" → higher score
        # for the positive class. Suitable for AUROC.
        probs = d0 - d1  # raw score for AUROC
        margin = float(np.abs(d0 - d1).mean())

    elif classifier == "mass_mean":
        # Mass-mean probing: use the difference between class means as a
        # linear direction. This is intentionally bias-free; accuracy uses
        # the natural zero threshold, while AUROC uses the raw projection.
        if not np.array_equal(np.unique(train_labels), np.array([0, 1])):
            raise ValueError(
                f"Mass-mean probe requires both labels, got {np.unique(train_labels)}"
            )
        c0 = train_norm[train_labels == 0].mean(axis=0)
        c1 = train_norm[train_labels == 1].mean(axis=0)
        direction = c1 - c0

        probs = eval_norm @ direction
        preds = (probs >= 0.0).astype(int)
        margin = float(np.abs(probs).mean())

    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    auroc = roc_auc_score(eval_labels, probs)
    acc = accuracy_score(eval_labels, preds)

    return {
        "auroc": float(auroc),
        "accuracy": float(acc),
        "avg_margin": float(margin),
    }


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def process_model(model_key: str, classifiers_to_run: list[str]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_path = HIDDEN_DIR / model_key / "train_hidden_states.pt"
    eval_path = HIDDEN_DIR / model_key / "eval_hidden_states.pt"
    if not train_path.exists() or not eval_path.exists():
        print(f"  [skip] Hidden states not found for {model_key}. Run step6 first.")
        return

    print(f"\nProcessing {model_key} …")

    train_data = torch.load(train_path, map_location="cpu", weights_only=False)
    eval_data = torch.load(eval_path, map_location="cpu", weights_only=False)

    train_hs = train_data["hidden_states"].numpy()   # [N_train, L, D]
    eval_hs = eval_data["hidden_states"].numpy()     # [N_eval, L, D]

    train_labels_str = train_data["labels"]
    eval_labels_str = eval_data["labels"]
    train_labels = np.array([1 if l == "relevant" else 0 for l in train_labels_str])
    eval_labels = np.array([1 if l == "relevant" else 0 for l in eval_labels_str])

    train_sources = np.array(train_data["dataset_source"])
    eval_sources = np.array(eval_data["dataset_source"])

    N_train, L, D = train_hs.shape
    num_transformer_layers = L - 1
    print(f"  Train: {N_train}, Eval: {eval_hs.shape[0]}, "
          f"Layers: {L} (0=emb, 1..{num_transformer_layers}=transformer)")
    print(f"  Classifiers: {classifiers_to_run}")

    datasets = sorted(set(train_data["dataset_source"]))
    subsets = {ds: {
        "train_mask": train_sources == ds,
        "eval_mask": eval_sources == ds,
    } for ds in datasets}
    subsets["combined"] = {
        "train_mask": np.ones(N_train, dtype=bool),
        "eval_mask": np.ones(eval_hs.shape[0], dtype=bool),
    }

    # results: classifier → layer → subset → metrics
    all_results: dict[str, dict] = {clf: {} for clf in classifiers_to_run}

    for layer_idx in range(L):
        layer_key = f"layer_{layer_idx}"
        for clf in classifiers_to_run:
            all_results[clf][layer_key] = {}

        tr_hs_layer = train_hs[:, layer_idx, :]  # [N_train, D]
        ev_hs_layer = eval_hs[:, layer_idx, :]   # [N_eval, D]

        for subset_name, masks in subsets.items():
            tr_mask = masks["train_mask"]
            ev_mask = masks["eval_mask"]

            for clf in classifiers_to_run:
                metrics = train_probe_single(
                    tr_hs_layer[tr_mask], train_labels[tr_mask],
                    ev_hs_layer[ev_mask], eval_labels[ev_mask],
                    classifier=clf,
                )
                all_results[clf][layer_key][subset_name] = metrics

        # One-line per-layer comparison across classifiers (combined subset)
        line = f"  Layer {layer_idx:2d}:"
        for clf in classifiers_to_run:
            a = all_results[clf][layer_key]["combined"]["auroc"]
            line += f"  {clf}={a:.4f}"
        print(line)

    # --- Save results, one JSON per classifier ---
    for clf in classifiers_to_run:
        results_path = RESULTS_DIR / f"{model_key}_probe_results_{clf}.json"
        with open(results_path, "w") as f:
            json.dump(all_results[clf], f, indent=2)
        print(f"  Saved → {results_path}")

    # --- Find best layer per classifier on combined subset ---
    print("  Best layer (combined) per classifier:")
    for clf in classifiers_to_run:
        best_layer = None
        best_auroc = -1.0
        for layer_key, subsets_data in all_results[clf].items():
            auroc = subsets_data["combined"]["auroc"]
            if auroc > best_auroc:
                best_auroc = auroc
                best_layer = layer_key
        print(f"    {clf:9s}: {best_layer}  AUROC={best_auroc:.4f}")

    # --- Plot: combined-subset AUROC across layers, one line per classifier ---
    fig, ax = plt.subplots(figsize=(12, 5))
    color_map = {
        "lr": "#1f77b4",
        "mlp": "#d62728",
        "centroid": "#2ca02c",
        "mass_mean": "#9467bd",
    }
    for clf in classifiers_to_run:
        aurocs = [all_results[clf][f"layer_{i}"]["combined"]["auroc"]
                  for i in range(L)]
        ax.plot(range(L), aurocs, "-", linewidth=1.6,
                color=color_map.get(clf, None),
                label=clf, marker="o", markersize=3)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="chance")
    ax.set_xlabel("Layer Index (0=embedding, 1..N=transformer layers)")
    ax.set_ylabel("AUROC (combined)")
    ax.set_title(f"{model_key} — Probe AUROC by Layer (combined subset)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = RESULTS_DIR / f"{model_key}_auroc_by_layer.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved figure → {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Step 8: Train probes (LR / MLP / Centroid / Mass-Mean)")
    parser.add_argument("--model", type=str, default="all",
                        choices=MODEL_KEYS + ["all"])
    parser.add_argument("--classifier", type=str, default="all",
                        choices=CLASSIFIERS + ["all"],
                        help="Which classifier to run. Default: all classifiers.")
    args = parser.parse_args()

    models = MODEL_KEYS if args.model == "all" else [args.model]
    classifiers_to_run = CLASSIFIERS if args.classifier == "all" else [args.classifier]

    for m in models:
        process_model(m, classifiers_to_run=classifiers_to_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
