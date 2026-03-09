"""Publication-quality plotting utilities for self-representation experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.directions import DirectionResults, ENTITY_CLASS_NAMES, PAIRWISE_PAIRS
from src.specificity import SpecificityResults
from src.steering import SteeringResults, SCORING_DIMENSIONS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

PALETTE = {
    "self": "#E63946",
    "expert_human": "#457B9D",
    "average_human": "#1D3557",
    "animal": "#2A9D8F",
    "object": "#E9C46A",
    "train": "#264653",
    "test": "#E76F51",
    "chance": "#ADB5BD",
    "grammatical": "#6A0572",
    "animacy": "#F77F00",
    "residual": "#4CC9F0",
    "original": "#E63946",
}

sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig: plt.Figure, path: Path, stem: str) -> None:
    """Save figure as PNG (300 dpi) and PDF."""
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(path / f"{stem}.pdf", bbox_inches="tight")
    logger.info("Saved %s", path / stem)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Probe accuracy across layers
# ---------------------------------------------------------------------------

def plot_probe_accuracy(
    results: DirectionResults,
    output_dir: Path,
) -> None:
    """Line plot of train vs test probe accuracy across layers."""
    L = results.n_layers
    layers = list(range(L))
    chance = 0.5  # binary probe: self vs not-self

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(layers, results.probe_train_acc, label="Train", color=PALETTE["train"], linewidth=2)
    ax.plot(layers, results.probe_test_acc, label="Test", color=PALETTE["test"], linewidth=2)
    ax.axhline(chance, linestyle="--", color=PALETTE["chance"], linewidth=1.2, label="Chance (0.5)")
    ax.axvline(results.best_probe_layer, linestyle=":", color="black", linewidth=1, alpha=0.6,
               label=f"Best layer ({results.best_probe_layer})")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Logistic Regression Probe Accuracy: Self vs. Not-Self")
    ax.set_xlim(0, L - 1)
    ax.set_ylim(0.4, 1.05)
    ax.legend(frameon=False)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    fig.tight_layout()
    _save(fig, output_dir, "01_probe_accuracy")


# ---------------------------------------------------------------------------
# 2. Pairwise direction similarity
# ---------------------------------------------------------------------------

def plot_pairwise_similarity(
    results: DirectionResults,
    output_dir: Path,
    layer: Optional[int] = None,
) -> None:
    """Heatmap of pairwise cosine similarity at best layer + line plot across layers."""
    L = results.n_layers
    if layer is None:
        layer = results.best_probe_layer

    pair_names = [p[0].replace("self_vs_", "Self vs\n") for p in PAIRWISE_PAIRS]

    # --- Heatmap at best layer ---
    mat = results.pairwise_cosine_similarity[layer]  # (4, 4)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        mat, annot=True, fmt=".2f", cmap="RdYlGn", vmin=-1, vmax=1,
        xticklabels=pair_names, yticklabels=pair_names, ax=ax, square=True,
        linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"Pairwise Direction Cosine Similarity (Layer {layer})")
    fig.tight_layout()
    _save(fig, output_dir, "02a_pairwise_heatmap")

    # --- Line plot: average off-diagonal similarity across layers ---
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.plot(
        range(L), results.pairwise_avg_cosine_similarity,
        color=PALETTE["self"], linewidth=2, label="Avg pairwise cosine sim",
    )
    ax2.axhline(0.7, linestyle="--", color=PALETTE["chance"], linewidth=1.2,
                label="Threshold (0.7)")
    ax2.axvline(layer, linestyle=":", color="black", linewidth=1, alpha=0.6,
                label=f"Best probe layer ({layer})")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Mean pairwise cosine similarity")
    ax2.set_title("Average Pairwise Direction Similarity Across Layers")
    ax2.set_xlim(0, L - 1)
    ax2.legend(frameon=False)
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(4))
    fig2.tight_layout()
    _save(fig2, output_dir, "02b_pairwise_similarity_layers")


# ---------------------------------------------------------------------------
# 3. Entity class projections
# ---------------------------------------------------------------------------

def plot_entity_projections(
    results: DirectionResults,
    output_dir: Path,
    layer: Optional[int] = None,
) -> None:
    """Violin + strip plot of projection scores per entity class."""
    if layer is None:
        layer = results.best_probe_layer

    projections = results.entity_projections[layer]  # (N,)
    labels = results.entity_projection_labels         # (N,)

    # Build DataFrame (only valid core labels 0-4)
    mask = (labels >= 0) & (labels < len(ENTITY_CLASS_NAMES))
    df = pd.DataFrame({
        "projection": projections[mask],
        "entity_class": [ENTITY_CLASS_NAMES[l] for l in labels[mask]],
    })
    df["entity_class"] = pd.Categorical(df["entity_class"], categories=ENTITY_CLASS_NAMES)

    palette = {cls: PALETTE[cls] for cls in ENTITY_CLASS_NAMES if cls in PALETTE}

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(
        data=df, x="entity_class", y="projection",
        palette=palette, inner=None, linewidth=1.5, ax=ax,
    )
    sns.stripplot(
        data=df, x="entity_class", y="projection",
        palette=palette, size=3, alpha=0.4, jitter=True, ax=ax,
    )
    ax.axhline(0, linestyle="--", color="gray", linewidth=1, alpha=0.6)
    ax.set_xlabel("Entity Class")
    ax.set_ylabel("Projection onto Self/Other Direction")
    ax.set_title(f"Entity Class Projections at Layer {layer}")
    ax.set_xticklabels(
        ["Self", "Expert\nHuman", "Avg\nHuman", "Animal", "Object"]
    )
    fig.tight_layout()
    _save(fig, output_dir, "03_entity_projections")


# ---------------------------------------------------------------------------
# 4. Confound similarity across layers
# ---------------------------------------------------------------------------

def plot_confound_similarity(
    spec_results: SpecificityResults,
    output_dir: Path,
) -> None:
    """Line plot of cos(self/other, grammatical_person) and cos(self/other, animacy)."""
    L = spec_results.n_layers
    layers = list(range(L))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(
        layers, np.abs(spec_results.cos_self_grammatical),
        label="|cos(self/other, grammatical person)|",
        color=PALETTE["grammatical"], linewidth=2,
    )
    ax.plot(
        layers, np.abs(spec_results.cos_self_animacy),
        label="|cos(self/other, animacy)|",
        color=PALETTE["animacy"], linewidth=2,
    )
    ax.axvline(
        spec_results.best_specific_layer,
        linestyle=":", color="black", linewidth=1, alpha=0.6,
        label=f"Most specific layer ({spec_results.best_specific_layer})",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("|Cosine similarity|")
    ax.set_title("Self/Other Direction vs. Confound Directions")
    ax.set_xlim(0, L - 1)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    fig.tight_layout()
    _save(fig, output_dir, "04_confound_similarity")


# ---------------------------------------------------------------------------
# 5. Control condition projections
# ---------------------------------------------------------------------------

def plot_control_projections(
    spec_results: SpecificityResults,
    output_dir: Path,
) -> None:
    """Strip plot showing control condition projections relative to core entity classes."""
    ctrl_proj = spec_results.control_projections
    if not ctrl_proj:
        logger.warning("No control projection data — skipping plot 5.")
        return

    rows = []
    # Core entity classes
    for cls, (mean, ci_lo, ci_hi) in ctrl_proj.get("core_entity_classes", {}).items():
        rows.append({"condition": cls, "mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi,
                     "type": "core"})

    # Dynamically include all control types present in the results
    skip_keys = {"core_entity_classes"}
    for ctrl_type, cond_dict in ctrl_proj.items():
        if ctrl_type in skip_keys or not isinstance(cond_dict, dict):
            continue
        for cond, (mean, ci_lo, ci_hi) in cond_dict.items():
            if cond == "all":
                rows.append({
                    "condition": f"{ctrl_type}\n(all)",
                    "mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "type": "control",
                })

    if not rows:
        logger.warning("No plottable control projection data.")
        return

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [PALETTE.get(r["type"], "#888") for _, r in df.iterrows()]

    for i, row in df.iterrows():
        c = PALETTE["test"] if row["type"] == "control" else PALETTE["self"]
        ax.errorbar(
            row["mean"], i,
            xerr=[[row["mean"] - row["ci_lo"]], [row["ci_hi"] - row["mean"]]],
            fmt="o", color=c, capsize=4, linewidth=1.5, markersize=6,
        )

    ax.axvline(0, linestyle="--", color="gray", linewidth=1, alpha=0.6)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["condition"].tolist())
    ax.set_xlabel("Projection onto Self/Other Direction")
    ax.set_title("Control Condition Projections (95% CI)")
    fig.tight_layout()
    _save(fig, output_dir, "05_control_projections")


# ---------------------------------------------------------------------------
# 6. Layer-wise direction evolution
# ---------------------------------------------------------------------------

def plot_direction_evolution(
    results: DirectionResults,
    output_dir: Path,
) -> None:
    """Cosine similarity of mean-diff direction at each layer with the final layer."""
    L = results.n_layers
    final_layer_dir = results.mean_diff_directions[-1]  # (D,)

    cos_sims = np.array([
        float(np.dot(
            _unit_np(results.mean_diff_directions[l]),
            _unit_np(final_layer_dir)
        ))
        for l in range(L)
    ])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(L), cos_sims, color=PALETTE["self"], linewidth=2)
    ax.axvline(results.best_probe_layer, linestyle=":", color="black", linewidth=1,
               alpha=0.6, label=f"Best probe layer ({results.best_probe_layer})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity with final layer direction")
    ax.set_title("Layer-Wise Evolution of Self/Other Direction")
    ax.set_xlim(0, L - 1)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(frameon=False)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    fig.tight_layout()
    _save(fig, output_dir, "06_direction_evolution")


# ---------------------------------------------------------------------------
# 7. Residual probe accuracy (Gate 2)
# ---------------------------------------------------------------------------

def plot_residual_accuracy(
    spec_results: SpecificityResults,
    output_dir: Path,
) -> None:
    """Compare original vs residual probe accuracy to visualise confound contribution."""
    L = spec_results.n_layers
    layers = list(range(L))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(layers, spec_results.residual_probe_acc_original,
            label="Original activations", color=PALETTE["original"], linewidth=2)
    ax.plot(layers, spec_results.residual_probe_acc,
            label="Residual (confounds removed)", color=PALETTE["residual"],
            linewidth=2, linestyle="--")
    ax.axhline(0.5, linestyle=":", color=PALETTE["chance"], linewidth=1.2, label="Chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Probe Accuracy Before and After Removing Confound Directions")
    ax.set_xlim(0, L - 1)
    ax.set_ylim(0.4, 1.05)
    ax.legend(frameon=False)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    fig.tight_layout()
    _save(fig, output_dir, "07_residual_accuracy")


# ---------------------------------------------------------------------------
# Convenience: generate all plots
# ---------------------------------------------------------------------------

def generate_all_plots(
    direction_results: DirectionResults,
    specificity_results: SpecificityResults,
    output_dir: str | Path,
    steering_results: Optional[SteeringResults] = None,
) -> None:
    """Generate and save all publication-quality figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating plot 1: probe accuracy...")
    plot_probe_accuracy(direction_results, output_dir)

    logger.info("Generating plot 2: pairwise similarity...")
    plot_pairwise_similarity(direction_results, output_dir)

    logger.info("Generating plot 3: entity projections...")
    plot_entity_projections(direction_results, output_dir)

    logger.info("Generating plot 4: confound similarity...")
    plot_confound_similarity(specificity_results, output_dir)

    logger.info("Generating plot 5: control projections...")
    plot_control_projections(specificity_results, output_dir)

    logger.info("Generating plot 6: direction evolution...")
    plot_direction_evolution(direction_results, output_dir)

    logger.info("Generating plot 7: residual accuracy...")
    plot_residual_accuracy(specificity_results, output_dir)

    if steering_results is not None:
        logger.info("Generating plot 8: steering results...")
        plot_steering_results(steering_results, output_dir)

    logger.info("All figures saved to %s", output_dir)


# ---------------------------------------------------------------------------
# 8. Steering results (Gate 3)
# ---------------------------------------------------------------------------

def plot_steering_results(
    steering_results: SteeringResults,
    output_dir: Path,
) -> None:
    """Line plots of mean score per dimension vs steering magnitude α."""
    alphas = steering_results.alphas
    mean_scores = steering_results.mean_scores

    dim_colors = {
        "agency": "#E63946",
        "assertiveness": "#457B9D",
        "entity_framing": "#2A9D8F",
        "self_continuity": "#6A0572",
    }
    dim_labels = {
        "agency": "Agency",
        "assertiveness": "Assertiveness",
        "entity_framing": "Entity Framing",
        "self_continuity": "Self-Continuity",
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    for dim in SCORING_DIMENSIONS:
        vals = [mean_scores[a][dim] for a in alphas]
        ax.plot(
            alphas, vals, "-o", color=dim_colors[dim], linewidth=2,
            markersize=6, label=dim_labels[dim],
        )

    ax.axvline(0, linestyle=":", color="gray", linewidth=1, alpha=0.6)
    ax.set_xlabel("Steering Magnitude (α)")
    ax.set_ylabel("Mean Score (1–5 Likert)")
    ax.set_title(
        f"Activation Steering: Behavioural Scores vs α (Layer {steering_results.direction_layer})"
    )
    ax.set_ylim(0.5, 5.5)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "08_steering_scores")


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------

def _unit_np(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v
