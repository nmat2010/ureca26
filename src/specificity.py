"""Gate 2: Testing whether the self/other direction is reducible to known confounds.

Computes:
1. Confound directions (grammatical person, animacy) from control prompts
2. Cosine similarity of self/other direction with confound directions per layer
3. Control condition clustering analysis
4. Residual analysis: classification accuracy after projecting out confounds
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class SpecificityResults:
    """All Gate-2 specificity analysis results.

    Attributes
    ----------
    grammatical_person_directions:
        (L, D) — first-person minus third-person mean-diff direction.
    animacy_directions:
        (L, D) — animate minus inanimate mean-diff direction.
    cos_self_grammatical:
        (L,) cosine similarity between self/other direction and grammatical-person direction.
    cos_self_animacy:
        (L,) cosine similarity between self/other direction and animacy direction.
    best_specific_layer:
        Layer where self/other direction has LOWEST mean cosine with confounds.
    control_projections:
        dict mapping control_type -> dict mapping condition -> (mean, ci_low, ci_high).
    residual_probe_acc:
        (L,) accuracy of probe trained on confound-residual activations.
    residual_probe_acc_original:
        (L,) accuracy of probe on original activations (for comparison).
    n_layers:
        Number of layers.
    """
    grammatical_person_directions: np.ndarray          # (L, D)
    animacy_directions: np.ndarray                     # (L, D)
    cos_self_grammatical: np.ndarray                   # (L,)
    cos_self_animacy: np.ndarray                       # (L,)
    best_specific_layer: int
    control_projections: Dict[str, Dict[str, Tuple[float, float, float]]]
    residual_probe_acc: np.ndarray                     # (L,)
    residual_probe_acc_original: np.ndarray            # (L,)
    n_layers: int

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "specificity_results.pkl", "wb") as f:
            pickle.dump(self, f)

        pd.DataFrame({
            "layer": list(range(self.n_layers)),
            "cos_self_grammatical": self.cos_self_grammatical,
            "cos_self_animacy": self.cos_self_animacy,
            "residual_probe_acc": self.residual_probe_acc,
            "original_probe_acc": self.residual_probe_acc_original,
        }).to_csv(output_dir / "specificity_summary.csv", index=False)
        logger.info("Saved SpecificityResults to %s", output_dir)

    @classmethod
    def load(cls, output_dir: str | Path) -> "SpecificityResults":
        with open(Path(output_dir) / "specificity_results.pkl", "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit(a), _unit(b)))


def _bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via bootstrap."""
    if rng is None:
        rng = np.random.default_rng(42)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean, ci_low, ci_high


def _project_out(
    activations: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Remove the component of activations along direction.

    Parameters
    ----------
    activations: (N, D) float32
    direction: (D,) unit vector

    Returns
    -------
    residual: (N, D) float32 with direction projected out
    """
    d = _unit(direction).astype(np.float32)
    projection = (activations @ d)[:, None] * d[None, :]
    return activations - projection


# ---------------------------------------------------------------------------
# 1. Confound direction computation
# ---------------------------------------------------------------------------

def compute_grammatical_person_direction(
    first_person_acts: np.ndarray,
    third_person_acts: np.ndarray,
) -> np.ndarray:
    """Compute per-layer grammatical person direction.

    Parameters
    ----------
    first_person_acts: (N1, L, D) — first-person prompts (self class)
    third_person_acts: (N2, L, D) — third-person prompts (third-person self control)

    Returns
    -------
    directions: (L, D) unit vectors
    """
    L = first_person_acts.shape[1]
    D = first_person_acts.shape[2]
    directions = np.zeros((L, D), dtype=np.float32)
    for layer in range(L):
        fp = first_person_acts[:, layer, :].mean(0)
        tp = third_person_acts[:, layer, :].mean(0)
        directions[layer] = _unit(fp - tp)
    return directions


def compute_animacy_direction(
    animate_acts: np.ndarray,
    inanimate_acts: np.ndarray,
) -> np.ndarray:
    """Compute per-layer animacy direction.

    Parameters
    ----------
    animate_acts: (N1, L, D) — animate entity prompts (human + animal classes)
    inanimate_acts: (N2, L, D) — inanimate entity prompts (object class)

    Returns
    -------
    directions: (L, D) unit vectors
    """
    L = animate_acts.shape[1]
    D = animate_acts.shape[2]
    directions = np.zeros((L, D), dtype=np.float32)
    for layer in range(L):
        an = animate_acts[:, layer, :].mean(0)
        inan = inanimate_acts[:, layer, :].mean(0)
        directions[layer] = _unit(an - inan)
    return directions


# ---------------------------------------------------------------------------
# 2. Cosine similarity analysis
# ---------------------------------------------------------------------------

def compute_confound_cosines(
    self_other_dirs: np.ndarray,
    grammatical_dirs: np.ndarray,
    animacy_dirs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-layer cosine similarities between self/other and confound directions.

    Returns
    -------
    cos_grammatical: (L,)
    cos_animacy: (L,)
    """
    L = self_other_dirs.shape[0]
    cos_gram = np.array([
        _cosine(self_other_dirs[l], grammatical_dirs[l]) for l in range(L)
    ])
    cos_anim = np.array([
        _cosine(self_other_dirs[l], animacy_dirs[l]) for l in range(L)
    ])
    return cos_gram, cos_anim


# ---------------------------------------------------------------------------
# 3. Control condition clustering
# ---------------------------------------------------------------------------

def compute_control_projections(
    core_activations: np.ndarray,
    core_labels: np.ndarray,
    control_activations_by_type: Dict[str, np.ndarray],
    control_labels_by_type: Dict[str, np.ndarray],
    self_other_dirs: np.ndarray,
    best_layer: int,
    n_bootstrap: int = 1000,
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """Project control prompts onto the self/other direction and compare to entity classes.

    For each control type and condition, return (mean, ci_low, ci_high) of projections.

    Parameters
    ----------
    core_activations: (N, L, D)
    core_labels: (N,)
    control_activations_by_type: dict control_type -> (M, L, D)
    control_labels_by_type: dict control_type -> (M,) labels
    self_other_dirs: (L, D)
    best_layer: layer to use for projection
    n_bootstrap: bootstrap samples for CIs
    """
    rng = np.random.default_rng(42)
    d = self_other_dirs[best_layer]  # (D,)
    results: Dict[str, Dict[str, Tuple[float, float, float]]] = {}

    # Project core classes at best_layer
    core_acts_layer = core_activations[:, best_layer, :]  # (N, D)
    core_proj = core_acts_layer @ _unit(d)

    # Core entity class projections
    core_results: Dict[str, Tuple[float, float, float]] = {}
    for label, name in [(0, "self"), (1, "expert"), (2, "average"), (3, "animal"), (4, "object")]:
        mask = core_labels == label
        if mask.sum() > 0:
            core_results[name] = _bootstrap_ci(core_proj[mask], n_bootstrap, rng=rng)
    results["core_entity_classes"] = core_results

    # Control conditions
    for ctrl_type, ctrl_acts in control_activations_by_type.items():
        ctrl_labels = control_labels_by_type.get(ctrl_type, np.array([]))
        ctrl_acts_layer = ctrl_acts[:, best_layer, :]
        ctrl_proj = ctrl_acts_layer @ _unit(d)

        ctrl_results: Dict[str, Tuple[float, float, float]] = {}
        unique_labels = np.unique(ctrl_labels)
        for lbl in unique_labels:
            mask = ctrl_labels == lbl
            if mask.sum() > 0:
                label_name = str(lbl)
                ctrl_results[label_name] = _bootstrap_ci(
                    ctrl_proj[mask], n_bootstrap, rng=rng
                )
        # Also report all-control aggregate
        ctrl_results["all"] = _bootstrap_ci(ctrl_proj, n_bootstrap, rng=rng)
        results[ctrl_type] = ctrl_results

    return results


# ---------------------------------------------------------------------------
# 4. Residual analysis
# ---------------------------------------------------------------------------

def compute_residual_probe_accuracy(
    activations: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    grammatical_dirs: np.ndarray,
    animacy_dirs: np.ndarray,
    cv_folds: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Measure probe accuracy before and after removing confound directions.

    At each layer:
    - Project out grammatical-person direction
    - Project out animacy direction
    - Fit logistic regression (Self vs not-Self) on residuals
    - Report test accuracy

    Returns
    -------
    residual_acc: (L,) test accuracy on residual activations
    original_acc: (L,) test accuracy on original activations
    """
    N, L, D = activations.shape
    binary_labels = (labels == 0).astype(int)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    residual_acc = np.zeros(L, dtype=np.float64)
    original_acc = np.zeros(L, dtype=np.float64)

    X_train_all = activations[train_mask]
    X_test_all = activations[test_mask]
    y_train = binary_labels[train_mask]
    y_test = binary_labels[test_mask]

    for layer in range(L):
        # Original
        X_tr = X_train_all[:, layer, :]
        X_te = X_test_all[:, layer, :]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        probe_orig = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
        probe_orig.fit(X_tr_s, y_train)
        original_acc[layer] = probe_orig.score(X_te_s, y_test)

        # Residual: project out both confound directions
        gram_dir = grammatical_dirs[layer]
        anim_dir = animacy_dirs[layer]

        X_tr_res = _project_out(_project_out(X_tr, gram_dir), anim_dir)
        X_te_res = _project_out(_project_out(X_te, gram_dir), anim_dir)

        scaler_res = StandardScaler()
        X_tr_res_s = scaler_res.fit_transform(X_tr_res)
        X_te_res_s = scaler_res.transform(X_te_res)

        probe_res = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
        probe_res.fit(X_tr_res_s, y_train)
        residual_acc[layer] = probe_res.score(X_te_res_s, y_test)

    return residual_acc, original_acc


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def test_specificity(
    core_activations: np.ndarray,
    core_labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    self_other_dirs: np.ndarray,
    grammatical_person_acts_first: np.ndarray,
    grammatical_person_acts_third: np.ndarray,
    control_activations_by_type: Dict[str, np.ndarray],
    control_labels_by_type: Dict[str, np.ndarray],
    best_probe_layer: int,
    n_bootstrap: int = 1000,
    cv_folds: int = 5,
) -> SpecificityResults:
    """Run all Gate-2 analyses.

    Parameters
    ----------
    core_activations: (N, L, D) — core prompt activations
    core_labels: (N,) — entity class labels
    train_mask, test_mask: (N,) bool masks
    self_other_dirs: (L, D) — mean-diff direction from Gate 1
    grammatical_person_acts_first: (N1, L, D) — self class activations (first-person)
    grammatical_person_acts_third: (N2, L, D) — third-person self control activations
    control_activations_by_type: dict type -> (M, L, D)
    control_labels_by_type: dict type -> (M,)
    best_probe_layer: layer with best probe accuracy (from Gate 1)
    n_bootstrap: bootstrap samples
    cv_folds: CV folds for residual probe
    """
    L = core_activations.shape[1]
    logger.info("Running Gate-2 specificity analysis...")

    # 1. Confound directions
    logger.info("Computing grammatical person direction...")
    gram_dirs = compute_grammatical_person_direction(
        grammatical_person_acts_first, grammatical_person_acts_third
    )

    # Animacy: animate = self + expert + average + animal; inanimate = object
    animate_mask = np.isin(core_labels, [0, 1, 2, 3])
    inanimate_mask = core_labels == 4
    logger.info("Computing animacy direction...")
    anim_dirs = compute_animacy_direction(
        core_activations[animate_mask], core_activations[inanimate_mask]
    )

    # 2. Cosine similarities
    logger.info("Computing confound cosine similarities...")
    cos_gram, cos_anim = compute_confound_cosines(self_other_dirs, gram_dirs, anim_dirs)

    # Best specific layer: lowest mean absolute cosine with confounds
    mean_cos = (np.abs(cos_gram) + np.abs(cos_anim)) / 2.0
    best_specific_layer = int(np.argmin(mean_cos))
    logger.info(
        "Best specific layer: %d (mean |cos|=%.3f)",
        best_specific_layer, mean_cos[best_specific_layer],
    )

    # 3. Control projections
    logger.info("Computing control condition projections...")
    ctrl_proj = compute_control_projections(
        core_activations, core_labels,
        control_activations_by_type, control_labels_by_type,
        self_other_dirs, best_probe_layer, n_bootstrap,
    )

    # 4. Residual analysis
    logger.info("Computing residual probe accuracy...")
    residual_acc, original_acc = compute_residual_probe_accuracy(
        core_activations, core_labels, train_mask, test_mask,
        gram_dirs, anim_dirs, cv_folds,
    )

    return SpecificityResults(
        grammatical_person_directions=gram_dirs,
        animacy_directions=anim_dirs,
        cos_self_grammatical=cos_gram,
        cos_self_animacy=cos_anim,
        best_specific_layer=best_specific_layer,
        control_projections=ctrl_proj,
        residual_probe_acc=residual_acc,
        residual_probe_acc_original=original_acc,
        n_layers=L,
    )
