"""Script 03: Compute self/other directions and probes.

Requires:
    - data/activations/*_core.h5 (from script 02)

Usage:
    python scripts/03_find_directions.py --config configs/experiment.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.directions import find_directions
from src.extraction import load_activations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find self/other directions and probes.")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--model_name", help="Override model name (for file lookup).")
    p.add_argument("--token_position", choices=["final", "entity"], default="final",
                   help="Which token position to analyse.")
    p.add_argument("--activations_file", help="Override path to core activations HDF5.")
    p.add_argument("--prompts_file", help="Override prompts JSON path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    dir_cfg = cfg.get("directions", {})
    paths_cfg = cfg.get("paths", {})

    model_name = args.model_name or model_cfg.get("primary", "")
    model_slug = model_name.replace("/", "_")
    acts_dir = paths_cfg.get("activations_dir", "data/activations")
    dirs_dir = paths_cfg.get("directions_dir", "data/directions")
    prompts_file = args.prompts_file or paths_cfg.get("prompts_file", "data/prompts.json")

    acts_file = args.activations_file or f"{acts_dir}/{model_slug}_core.h5"

    # Check prerequisite
    if not Path(acts_file).exists():
        logger.error(
            "Activations file not found: %s\nRun scripts/02_extract_activations.py first.",
            acts_file,
        )
        sys.exit(1)

    if not Path(prompts_file).exists():
        logger.error("Prompts file not found: %s", prompts_file)
        sys.exit(1)

    # Load activations
    logger.info("Loading activations from %s (position=%s)...", acts_file, args.token_position)
    activations, labels, metadata = load_activations(acts_file, args.token_position)
    logger.info("Activations shape: %s", activations.shape)

    # Build train/test masks from metadata
    splits = [m.get("split", "train") for m in metadata]
    train_mask = np.array([s == "train" for s in splits])
    test_mask = np.array([s == "test" for s in splits])

    logger.info(
        "Train: %d | Test: %d | Total: %d",
        train_mask.sum(), test_mask.sum(), len(train_mask),
    )

    # Run direction finding
    C_values = dir_cfg.get("probe_C_values", [0.001, 0.01, 0.1, 1.0, 10.0])
    cv_folds = dir_cfg.get("probe_cv_folds", 5)

    results = find_directions(
        activations=activations,
        labels=labels,
        train_mask=train_mask,
        test_mask=test_mask,
        C_values=C_values,
        cv_folds=cv_folds,
    )

    # Save
    out_dir = Path(dirs_dir) / model_slug / args.token_position
    results.save(out_dir)

    logger.info("Best probe layer: %d (test acc=%.3f)",
                results.best_probe_layer, results.probe_test_acc[results.best_probe_layer])
    logger.info(
        "Pairwise avg cosine sim at best layer: %.3f",
        results.pairwise_avg_cosine_similarity[results.best_probe_layer],
    )
    logger.info("Direction results saved to %s", out_dir)


if __name__ == "__main__":
    main()
