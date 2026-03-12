"""Direction-finding methods for the self/other axis.

Implements:
1. Mean-difference vector (per layer, normalised)
2. Logistic regression probe (per layer, L2 regularised)
3. Pairwise directions (Self vs each other class)
4. Hierarchical projection with Kruskal-Wallis / Mann-Whitney tests
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ENTITY_CLASS_NAMES = ["self", "expert_human", "average_human", "animal", "object"]
ENTITY_LABEL_MAP = {name: i for i, name in enumerate(ENTITY_CLASS_NAMES)}
LABEL_TO_CLASS = {i: name for i, name in enumerate(ENTITY_CLASS_NAMES)}


# ---------------------------------------------------------------------------
# Data class for results
# ---------------------------------------------------------------------------

@dataclass
class DirectionResults:
    """Container for all direction-finding results.

    Attributes
    ----------
    mean_diff_directions:
        (L, D) array — per-layer normalised mean-diff direction (Self minus all others).
    probe_directions:
        (L, D) array — per-layer normalised probe weight vector.
    probe_train_acc:
        (L,) float array — per-layer train accuracy.
    probe_test_acc:
        (L,) float array — per-layer held-out test accuracy.
    best_probe_layer:
        Layer index with highest test accuracy.
    pairwise_directions:
        Dict mapping pair name (e.g. "self_vs_expert") to (L, D) direction array.
    pairwise_cosine_similarity:
        (L, 4, 4) array — cosine similarity matrix between pairwise directions.
    pairwise_avg_cosine_similarity:
        (L,) float — mean off-diagonal cosine similarity per layer.
    entity_projections:
        (L, N) array — projection of each sample onto mean-diff direction.
    entity_projection_labels:
        (N,) int array — entity labels matching entity_projections.
    kruskal_pvalues:
        (L,) float — p-value of Kruskal-Wallis test per layer.
    mannwhitney_pvalues:
        dict mapping layer_idx -> dict of pair -> p-value.
    n_layers:
        Number of layers.
    hidden_dim:
        Residual stream dimension.
    """
    mean_diff_directions: np.ndarray                          # (L, D)
    probe_directions: np.ndarray                              # (L, D)
    probe_train_acc: np.ndarray                               # (L,)
    probe_test_acc: np.ndarray                                # (L,)
    best_probe_layer: int
    pairwise_directions: Dict[str, np.ndarray]                # key -> (L, D)
    pairwise_cosine_similarity: np.ndarray                    # (L, 4, 4)
    pairwise_avg_cosine_similarity: np.ndarray                # (L,)
    contrastive_direction: np.ndarray                         # (L, D) — SVD first principal component
    contrastive_consistency: np.ndarray                       # (L,) — fraction of variance explained
    entity_projections: np.ndarray                            # (L, N)
    entity_projection_labels: np.ndarray                      # (N,)
    kruskal_pvalues: np.ndarray                               # (L,)
    mannwhitney_pvalues: Dict[int, Dict[str, float]]
    n_layers: int
    hidden_dim: int

    def save(self, output_dir: str | Path) -> None:
        """Save results to pickle + CSV summaries."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "direction_results.pkl", "wb") as f:
            pickle.dump(self, f)

        # Probe accuracy CSV
        pd.DataFrame({
            "layer": list(range(self.n_layers)),
            "train_acc": self.probe_train_acc,
            "test_acc": self.probe_test_acc,
            "kruskal_pvalue": self.kruskal_pvalues,
            "pairwise_avg_cosine_sim": self.pairwise_avg_cosine_similarity,
        }).to_csv(output_dir / "probe_accuracy.csv", index=False)

        logger.info("Saved DirectionResults to %s", output_dir)

    @classmethod
    def load(cls, output_dir: str | Path) -> "DirectionResults":
        output_dir = Path(output_dir)
        with open(output_dir / "direction_results.pkl", "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Helper: unit normalisation
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    """Normalise a vector to unit length, returning zero vector if degenerate."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    return float(np.dot(_unit(a), _unit(b)))


# ---------------------------------------------------------------------------
# 1. Mean-difference direction
# ---------------------------------------------------------------------------

def compute_mean_diff_directions(
    activations: np.ndarray,
    labels: np.ndarray,
    self_label: int = 0,
) -> np.ndarray:
    """Compute per-layer normalised mean-difference direction (Self vs all others).

    Parameters
    ----------
    activations:
        (N, L, D) float32 array.
    labels:
        (N,) int array. self_label marks self-referential prompts.
    self_label:
        Integer label for the self class.

    Returns
    -------
    directions: (L, D) float32 array, unit vectors.
    """
    N, L, D = activations.shape
    self_mask = labels == self_label
    other_mask = ~self_mask

    directions = np.zeros((L, D), dtype=np.float32)
    for layer in range(L):
        acts = activations[:, layer, :]  # (N, D)
        mean_self = acts[self_mask].mean(axis=0)
        mean_other = acts[other_mask].mean(axis=0)
        diff = mean_self - mean_other
        directions[layer] = _unit(diff)
    return directions


# ---------------------------------------------------------------------------
# 2. Logistic regression probe
# ---------------------------------------------------------------------------

def compute_probe_directions(
    activations: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    self_label: int = 0,
    C_values: List[float] = (0.001, 0.01, 0.1, 1.0, 10.0),
    cv_folds: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train logistic regression probes per layer.

    Parameters
    ----------
    activations:
        (N, L, D) float32 array.
    labels:
        (N,) int array.
    train_mask:
        Boolean mask for training samples.
    test_mask:
        Boolean mask for held-out test samples.
    self_label:
        Integer label for self class.
    C_values:
        Regularisation strengths to try via cross-validation.
    cv_folds:
        Number of cross-validation folds.

    Returns
    -------
    probe_directions: (L, D) unit-norm weight vectors.
    train_acc: (L,) float.
    test_acc: (L,) float.
    """
    N, L, D = activations.shape
    binary_labels = (labels == self_label).astype(int)

    probe_directions = np.zeros((L, D), dtype=np.float32)
    train_acc = np.zeros(L, dtype=np.float64)
    test_acc = np.zeros(L, dtype=np.float64)

    X_train_all = activations[train_mask]   # (n_train, L, D)
    X_test_all = activations[test_mask]     # (n_test, L, D)
    y_train = binary_labels[train_mask]
    y_test = binary_labels[test_mask]

    for layer in tqdm_or_range(L, desc="Fitting probes"):
        X_tr = X_train_all[:, layer, :]
        X_te = X_test_all[:, layer, :]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        # Select best C via CV on training set
        best_C = _select_C(X_tr_scaled, y_train, C_values, cv_folds)

        probe = LogisticRegression(
            C=best_C, max_iter=1000, solver="liblinear", random_state=42
        )
        probe.fit(X_tr_scaled, y_train)

        train_acc[layer] = probe.score(X_tr_scaled, y_train)
        test_acc[layer] = probe.score(X_te_scaled, y_test)

        # Weight vector as probe direction
        w = probe.coef_[0].astype(np.float32)
        probe_directions[layer] = _unit(w)

    return probe_directions, train_acc, test_acc


def _select_C(
    X: np.ndarray,
    y: np.ndarray,
    C_values: List[float],
    cv_folds: int,
) -> float:
    """Return best C from C_values via stratified k-fold CV."""
    if len(np.unique(y)) < 2:
        return 1.0
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    best_C, best_score = C_values[0], -1.0
    for C in C_values:
        model = LogisticRegression(C=C, max_iter=500, solver="liblinear", random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_C = C
    return best_C


# ---------------------------------------------------------------------------
# 3. Pairwise directions
# ---------------------------------------------------------------------------

PAIRWISE_PAIRS = [
    ("self_vs_expert", 0, 1),
    ("self_vs_average", 0, 2),
    ("self_vs_animal", 0, 3),
    ("self_vs_object", 0, 4),
]


def compute_pairwise_directions(
    activations: np.ndarray,
    labels: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Compute pairwise mean-diff directions and their cosine similarity matrix.

    Parameters
    ----------
    activations: (N, L, D)
    labels: (N,)

    Returns
    -------
    directions: dict mapping pair_name -> (L, D) unit directions
    cosine_matrix: (L, 4, 4) pairwise cosine similarities
    avg_cosine: (L,) mean off-diagonal similarity per layer
    """
    N, L, D = activations.shape
    n_pairs = len(PAIRWISE_PAIRS)

    pair_directions: Dict[str, np.ndarray] = {}
    dir_matrix = np.zeros((n_pairs, L, D), dtype=np.float32)

    for pair_idx, (pair_name, label_a, label_b) in enumerate(PAIRWISE_PAIRS):
        mask_a = labels == label_a
        mask_b = labels == label_b
        dirs = np.zeros((L, D), dtype=np.float32)
        for layer in range(L):
            acts = activations[:, layer, :]
            if mask_a.sum() > 0 and mask_b.sum() > 0:
                diff = acts[mask_a].mean(0) - acts[mask_b].mean(0)
                dirs[layer] = _unit(diff)
        pair_directions[pair_name] = dirs
        dir_matrix[pair_idx] = dirs

    # Cosine similarity matrix (L, 4, 4)
    cosine_matrix = np.zeros((L, n_pairs, n_pairs), dtype=np.float32)
    for layer in range(L):
        for i in range(n_pairs):
            for j in range(n_pairs):
                cosine_matrix[layer, i, j] = cosine_similarity(
                    dir_matrix[i, layer], dir_matrix[j, layer]
                )

    # Average off-diagonal similarity per layer
    mask_off_diag = ~np.eye(n_pairs, dtype=bool)
    avg_cosine = np.array([
        cosine_matrix[l][mask_off_diag].mean() for l in range(L)
    ], dtype=np.float64)

    return pair_directions, cosine_matrix, avg_cosine


# ---------------------------------------------------------------------------
# 3b. Contrastive direction via SVD (Fix 3)
# ---------------------------------------------------------------------------

def compute_contrastive_direction(
    pairwise_directions: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Find the direction most consistent across all pairwise self-vs-other contrasts.

    Stacks the four pairwise directions (self_vs_expert, self_vs_average,
    self_vs_animal, self_vs_object) into a (4, D) matrix per layer and takes
    the first principal component via SVD. If the self/other direction is
    truly a single underlying concept, this first PC should capture most of
    the variance (high consistency score).

    Returns
    -------
    contrastive_dir: (L, D) unit vectors — first principal component per layer
    consistency: (L,) float — fraction of variance explained by PC1 (σ₁² / Σσᵢ²)
    """
    pair_names = sorted(pairwise_directions.keys())
    L = pairwise_directions[pair_names[0]].shape[0]
    D = pairwise_directions[pair_names[0]].shape[1]
    n_pairs = len(pair_names)

    contrastive_dir = np.zeros((L, D), dtype=np.float32)
    consistency = np.zeros(L, dtype=np.float64)

    for layer in range(L):
        # Stack pairwise directions: (n_pairs, D)
        mat = np.stack([pairwise_directions[name][layer] for name in pair_names])

        # SVD
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        contrastive_dir[layer] = _unit(Vt[0].astype(np.float32))

        # Consistency: fraction of variance explained by first singular value
        total_var = (S ** 2).sum()
        if total_var > 1e-10:
            consistency[layer] = float(S[0] ** 2 / total_var)
        else:
            consistency[layer] = 0.0

    return contrastive_dir, consistency


# ---------------------------------------------------------------------------
# 4. Hierarchical projection + statistical tests
# ---------------------------------------------------------------------------

def compute_entity_projections(
    activations: np.ndarray,
    labels: np.ndarray,
    directions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict[str, float]]]:
    """Project each sample onto the self/other direction per layer.

    Parameters
    ----------
    activations: (N, L, D)
    labels: (N,)
    directions: (L, D) — the mean-diff or probe directions

    Returns
    -------
    projections: (L, N) float array — scalar projection per sample per layer
    kruskal_pvalues: (L,) p-values from Kruskal-Wallis test
    mannwhitney_pvalues: dict layer -> dict pair_name -> p-value
    """
    N, L, D = activations.shape
    projections = np.zeros((L, N), dtype=np.float32)

    for layer in range(L):
        d = directions[layer]  # (D,)
        acts = activations[:, layer, :]  # (N, D)
        projections[layer] = acts @ d

    kruskal_pvalues = np.zeros(L, dtype=np.float64)
    mannwhitney_pvalues: Dict[int, Dict[str, float]] = {}

    unique_labels = sorted(np.unique(labels[labels >= 0]))
    for layer in range(L):
        proj = projections[layer]
        groups = [proj[labels == lbl] for lbl in unique_labels if (labels == lbl).sum() > 0]
        if len(groups) >= 2:
            try:
                _, pval = kruskal(*groups)
                kruskal_pvalues[layer] = pval
            except Exception:
                kruskal_pvalues[layer] = 1.0
        else:
            kruskal_pvalues[layer] = 1.0

        mw_layer: Dict[str, float] = {}
        for i, lbl_i in enumerate(unique_labels):
            for lbl_j in unique_labels[i + 1:]:
                g_i = proj[labels == lbl_i]
                g_j = proj[labels == lbl_j]
                if len(g_i) > 0 and len(g_j) > 0:
                    try:
                        _, pval_mw = mannwhitneyu(g_i, g_j, alternative="two-sided")
                    except Exception:
                        pval_mw = 1.0
                    name = f"{LABEL_TO_CLASS.get(lbl_i, lbl_i)}_vs_{LABEL_TO_CLASS.get(lbl_j, lbl_j)}"
                    mw_layer[name] = float(pval_mw)
        mannwhitney_pvalues[layer] = mw_layer

    return projections, kruskal_pvalues, mannwhitney_pvalues


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def find_directions(
    activations: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    C_values: List[float] = (0.001, 0.01, 0.1, 1.0, 10.0),
    cv_folds: int = 5,
) -> DirectionResults:
    """Run all direction-finding methods and return results.

    Parameters
    ----------
    activations: (N, L, D) float32 — activations for core prompts only
    labels: (N,) int — entity class labels (0=self, 1=expert, ...)
    train_mask: (N,) bool — True for training samples
    test_mask: (N,) bool — True for test samples
    C_values: regularisation grid for probe
    cv_folds: number of CV folds
    """
    N, L, D = activations.shape
    logger.info("Finding directions: N=%d, L=%d, D=%d", N, L, D)

    # 1. Mean-diff
    logger.info("Computing mean-diff directions...")
    mean_diff_dirs = compute_mean_diff_directions(activations, labels)

    # 2. Probe
    logger.info("Fitting logistic regression probes...")
    probe_dirs, train_acc, test_acc = compute_probe_directions(
        activations, labels, train_mask, test_mask,
        C_values=list(C_values), cv_folds=cv_folds,
    )
    best_probe_layer = int(np.argmax(test_acc))
    logger.info("Best probe layer: %d (test acc=%.3f)", best_probe_layer, test_acc[best_probe_layer])

    # 3. Pairwise
    logger.info("Computing pairwise directions...")
    pairwise_dirs, cosine_matrix, avg_cosine = compute_pairwise_directions(
        activations, labels
    )

    # 3b. Contrastive direction via SVD (Fix 3)
    logger.info("Computing contrastive direction (SVD of pairwise directions)...")
    contrastive_dir, contrastive_consistency = compute_contrastive_direction(
        pairwise_dirs
    )
    best_layer_consistency = contrastive_consistency[best_probe_layer]
    logger.info(
        "Contrastive consistency at best layer %d: %.3f",
        best_probe_layer, best_layer_consistency,
    )

    # 4. Projections + stats (using mean-diff direction)
    logger.info("Computing entity projections and statistical tests...")
    projections, kruskal_pvals, mw_pvals = compute_entity_projections(
        activations, labels, mean_diff_dirs
    )

    return DirectionResults(
        mean_diff_directions=mean_diff_dirs,
        probe_directions=probe_dirs,
        probe_train_acc=train_acc,
        probe_test_acc=test_acc,
        best_probe_layer=best_probe_layer,
        pairwise_directions=pairwise_dirs,
        pairwise_cosine_similarity=cosine_matrix,
        pairwise_avg_cosine_similarity=avg_cosine,
        contrastive_direction=contrastive_dir,
        contrastive_consistency=contrastive_consistency,
        entity_projections=projections,
        entity_projection_labels=labels,
        kruskal_pvalues=kruskal_pvals,
        mannwhitney_pvalues=mw_pvals,
        n_layers=L,
        hidden_dim=D,
    )


# ---------------------------------------------------------------------------
# Utility: tqdm-optional range
# ---------------------------------------------------------------------------

def tqdm_or_range(n: int, desc: str = ""):
    try:
        from tqdm import tqdm
        return tqdm(range(n), desc=desc)
    except ImportError:
        return range(n)
