"""Script 05: Generate all publication-quality figures.

Requires:
    - data/directions/*/final/direction_results.pkl (from script 03)
    - data/directions/*/final/specificity_results.pkl (from script 04)

Usage:
    python scripts/05_visualize.py --config configs/experiment.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.directions import DirectionResults
from src.specificity import SpecificityResults
from src.steering import SteeringResults
from src.visualization import generate_all_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate all figures.")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--model_name", help="Override model name.")
    p.add_argument("--token_position", choices=["final", "entity"], default="final")
    p.add_argument("--output_dir", help="Override figures output directory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    paths_cfg = cfg.get("paths", {})

    model_name = args.model_name or model_cfg.get("primary", "")
    model_slug = model_name.replace("/", "_")
    dirs_dir = paths_cfg.get("directions_dir", "data/directions")
    figs_dir = args.output_dir or paths_cfg.get("figures_dir", "figures")

    results_dir = Path(dirs_dir) / model_slug / args.token_position
    dir_pkl = results_dir / "direction_results.pkl"
    spec_pkl = results_dir / "specificity_results.pkl"

    for required in (dir_pkl, spec_pkl):
        if not required.exists():
            logger.error("Required file not found: %s", required)
            sys.exit(1)

    logger.info("Loading direction results...")
    dir_results = DirectionResults.load(results_dir)

    logger.info("Loading specificity results...")
    spec_results = SpecificityResults.load(results_dir)

    # Optionally load steering results (from script 06)
    steering_json = results_dir / "steering_results.json"
    steer_results = None
    if steering_json.exists():
        logger.info("Loading steering results...")
        steer_results = SteeringResults.load(results_dir)

    out_dir = Path(figs_dir) / model_slug / args.token_position
    logger.info("Generating all figures to %s", out_dir)
    generate_all_plots(dir_results, spec_results, out_dir, steering_results=steer_results)

    logger.info("Done.")


if __name__ == "__main__":
    main()
