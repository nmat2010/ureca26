"""Script 04: Gate 2 — test whether self/other direction is reducible to confounds.

Requires:
    - data/activations/*_core.h5 (from script 02)
    - data/activations/*_ctrl_*.h5 (from script 02)
    - data/directions/*/final/direction_results.pkl (from script 03)

Usage:
    python scripts/04_test_specificity.py --config configs/experiment.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.directions import DirectionResults
from src.extraction import load_activations
from src.specificity import test_specificity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gate 2: specificity / confound analysis.")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--model_name", help="Override model name.")
    p.add_argument("--token_position", choices=["final", "entity"], default="final")
    p.add_argument("--prompts_file", help="Override prompts JSON path.")
    return p.parse_args()


def _load_ctrl(
    acts_dir: str,
    model_slug: str,
    ctrl_type: str,
    token_position: str,
) -> tuple:
    """Load control activations and return (activations, labels, metadata)."""
    path = Path(acts_dir) / f"{model_slug}_ctrl_{ctrl_type}.h5"
    if not path.exists():
        logger.warning("Control activations not found: %s", path)
        return None, None, None
    return load_activations(path, token_position)


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    spec_cfg = cfg.get("specificity", {})
    paths_cfg = cfg.get("paths", {})
    dir_cfg = cfg.get("directions", {})

    model_name = args.model_name or model_cfg.get("primary", "")
    model_slug = model_name.replace("/", "_")
    acts_dir = paths_cfg.get("activations_dir", "data/activations")
    dirs_dir = paths_cfg.get("directions_dir", "data/directions")
    prompts_file = args.prompts_file or paths_cfg.get("prompts_file", "data/prompts.json")
    n_bootstrap = spec_cfg.get("n_bootstrap", 1000)
    cv_folds = dir_cfg.get("probe_cv_folds", 5)

    # Check prerequisites
    core_acts_file = Path(acts_dir) / f"{model_slug}_core.h5"
    dirs_file = Path(dirs_dir) / model_slug / args.token_position / "direction_results.pkl"

    for required in (core_acts_file, dirs_file, Path(prompts_file)):
        if not required.exists():
            logger.error("Required file not found: %s", required)
            sys.exit(1)

    # Load core activations + labels
    logger.info("Loading core activations...")
    core_acts, core_labels, core_meta = load_activations(core_acts_file, args.token_position)
    logger.info("Core activations shape: %s", core_acts.shape)

    # Build train/test masks
    splits = [m.get("split", "train") for m in core_meta]
    train_mask = np.array([s == "train" for s in splits])
    test_mask = np.array([s == "test" for s in splits])

    # Load direction results
    logger.info("Loading direction results from %s", dirs_file.parent)
    dir_results = DirectionResults.load(dirs_file.parent)

    # Grammatical-person control: first-person = self class; third-person = ctrl_grammatical
    first_person_acts = core_acts[core_labels == 0]  # self class
    ctrl_gram_acts, _, _ = _load_ctrl(
        acts_dir, model_slug, "grammatical_person", args.token_position
    )
    if ctrl_gram_acts is None:
        logger.error("Grammatical person control activations missing — cannot run Gate 2.")
        sys.exit(1)
    third_person_acts = ctrl_gram_acts

    # Control activations dict
    ctrl_acts_dict = {}
    ctrl_labels_dict = {}
    for ctrl_type in ("grammatical_person", "role_play", "animacy", "identity_decoupled"):
        acts, labels, _ = _load_ctrl(acts_dir, model_slug, ctrl_type, args.token_position)
        if acts is not None:
            ctrl_acts_dict[ctrl_type] = acts
            ctrl_labels_dict[ctrl_type] = labels if labels is not None else np.array([])

    # Split animacy control into animate / inanimate for direction computation.
    # Labels: 1 = animate, 0 = inanimate (set in dataset.py).
    animacy_acts_animate = None
    animacy_acts_inanimate = None
    if "animacy" in ctrl_acts_dict:
        anim_labels = ctrl_labels_dict["animacy"]
        anim_acts = ctrl_acts_dict["animacy"]
        anim_mask = anim_labels == 1
        inanim_mask = anim_labels == 0
        if anim_mask.sum() > 0 and inanim_mask.sum() > 0:
            animacy_acts_animate = anim_acts[anim_mask]
            animacy_acts_inanimate = anim_acts[inanim_mask]
            logger.info(
                "Animacy controls: %d animate, %d inanimate",
                anim_mask.sum(), inanim_mask.sum(),
            )
        else:
            logger.warning(
                "Animacy control labels don't contain both animate (1) and "
                "inanimate (0). Found labels: %s. Falling back to core splitting.",
                np.unique(anim_labels),
            )

    # Run specificity
    spec_results = test_specificity(
        core_activations=core_acts,
        core_labels=core_labels,
        train_mask=train_mask,
        test_mask=test_mask,
        self_other_dirs=dir_results.mean_diff_directions,
        grammatical_person_acts_first=first_person_acts,
        grammatical_person_acts_third=third_person_acts,
        control_activations_by_type=ctrl_acts_dict,
        control_labels_by_type=ctrl_labels_dict,
        best_probe_layer=dir_results.best_probe_layer,
        n_bootstrap=n_bootstrap,
        cv_folds=cv_folds,
        animacy_acts_animate=animacy_acts_animate,
        animacy_acts_inanimate=animacy_acts_inanimate,
    )

    # Save
    out_dir = Path(dirs_dir) / model_slug / args.token_position
    spec_results.save(out_dir)

    logger.info("Best specific layer (lowest confound similarity): %d",
                spec_results.best_specific_layer)
    logger.info(
        "At best probe layer %d — cos(self/other, grammatical): %.3f | cos(self/other, animacy): %.3f",
        dir_results.best_probe_layer,
        spec_results.cos_self_grammatical[dir_results.best_probe_layer],
        spec_results.cos_self_animacy[dir_results.best_probe_layer],
    )
    logger.info(
        "Residual probe test acc at best probe layer: %.3f (original: %.3f)",
        spec_results.residual_probe_acc[dir_results.best_probe_layer],
        spec_results.residual_probe_acc_original[dir_results.best_probe_layer],
    )
    logger.info("Specificity results saved to %s", out_dir)


if __name__ == "__main__":
    main()
