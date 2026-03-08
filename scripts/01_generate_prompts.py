"""Script 01: Generate and save all prompts.

Usage:
    python scripts/01_generate_prompts.py --config configs/experiment.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import generate_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate all prompts for self-repr experiments.")
    p.add_argument("--config", default="configs/experiment.yaml", help="Path to YAML config.")
    p.add_argument("--num_scenarios", type=int, help="Override num_scenarios.")
    p.add_argument("--num_control_scenarios", type=int, help="Override num_control_scenarios.")
    p.add_argument("--random_seed", type=int, help="Override random_seed.")
    p.add_argument("--output", help="Override output path for prompts JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Config values with CLI overrides
    ds_cfg = cfg.get("dataset", {})
    num_scenarios = args.num_scenarios or ds_cfg.get("num_scenarios", 200)
    train_size = ds_cfg.get("train_size", 150)
    num_control = args.num_control_scenarios or ds_cfg.get("num_control_scenarios", 50)
    seed = args.random_seed or cfg.get("random_seed", 42)
    prompts_path = args.output or cfg.get("paths", {}).get("prompts_file", "data/prompts.json")

    logger.info("Generating dataset: %d core scenarios, %d control scenarios, seed=%d",
                num_scenarios, num_control, seed)

    dataset = generate_dataset(
        num_scenarios=num_scenarios,
        train_size=train_size,
        num_control_scenarios=num_control,
        random_seed=seed,
    )

    logger.info(
        "Dataset generated: %d total prompts (%d core, %d control)",
        len(dataset),
        len(dataset.get_core_prompts()),
        len(dataset) - len(dataset.get_core_prompts()),
    )
    logger.info(
        "Train split: %d | Test split: %d",
        len(dataset.get_train_split()),
        len(dataset.get_test_split()),
    )

    dataset.save(prompts_path)
    logger.info("Prompts saved to %s", prompts_path)

    # Print a few examples
    core = dataset.get_core_prompts()
    logger.info("\n--- Example prompts ---")
    for p in core[:5]:
        logger.info("[%s | %s | %s] %s", p.domain, p.entity_class, p.split, p.text)

    # Save config snapshot alongside data
    out_dir = Path(prompts_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(cfg, f)
    logger.info("Config snapshot saved to %s/config_snapshot.yaml", out_dir)


if __name__ == "__main__":
    main()
