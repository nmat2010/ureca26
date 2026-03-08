"""Script 02: Extract residual-stream activations from LLaMA-3.1-8B.

Requires:
    - data/prompts.json (from script 01)

Usage:
    python scripts/02_extract_activations.py --config configs/experiment.yaml
    python scripts/02_extract_activations.py --config configs/experiment.yaml --batch_size 4
    python scripts/02_extract_activations.py --config configs/experiment.yaml --model_name meta-llama/Llama-3.1-8B
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import PromptDataset
from src.extraction import ActivationExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract residual-stream activations.")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--model_name", help="Override model name.")
    p.add_argument("--batch_size", type=int, help="Override batch size.")
    p.add_argument("--device", help="Override device (e.g., cuda, cpu).")
    p.add_argument("--output_file", help="Override output HDF5 path.")
    p.add_argument("--prompts_file", help="Override prompts JSON path.")
    p.add_argument("--base_model", action="store_true",
                   help="Use base model instead of instruct model.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    extract_cfg = cfg.get("extraction", {})
    paths_cfg = cfg.get("paths", {})

    model_name = args.model_name or (
        model_cfg.get("base") if args.base_model else model_cfg.get("primary")
    )
    batch_size = args.batch_size or extract_cfg.get("batch_size", 8)
    device = args.device or model_cfg.get("device", "cuda")
    n_layers = model_cfg.get("n_layers", 32)
    hidden_dim = model_cfg.get("hidden_dim", 4096)

    prompts_file = args.prompts_file or paths_cfg.get("prompts_file", "data/prompts.json")
    acts_dir = paths_cfg.get("activations_dir", "data/activations")
    model_slug = model_name.replace("/", "_")
    output_file = args.output_file or f"{acts_dir}/{model_slug}.h5"

    # Check prerequisite
    if not Path(prompts_file).exists():
        logger.error(
            "Prompts file not found: %s\nRun scripts/01_generate_prompts.py first.",
            prompts_file,
        )
        sys.exit(1)

    logger.info("Loading prompts from %s", prompts_file)
    dataset = PromptDataset.load(prompts_file)
    all_prompts = dataset.prompts
    logger.info("Loaded %d prompts total", len(all_prompts))

    import torch
    dtype = torch.float16

    extractor = ActivationExtractor(
        model_name=model_name,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )

    # Extract core prompts
    core_prompts = dataset.get_core_prompts()
    logger.info("Extracting activations for %d core prompts...", len(core_prompts))
    core_output = output_file.replace(".h5", "_core.h5")
    extractor.extract(core_prompts, core_output)

    # Extract control prompts (separate files per control type)
    for ctrl_type in ("grammatical_person", "role_play", "animacy"):
        ctrl_prompts = dataset.get_control_prompts(ctrl_type)
        if not ctrl_prompts:
            logger.warning("No control prompts for type '%s'", ctrl_type)
            continue
        ctrl_output = output_file.replace(".h5", f"_ctrl_{ctrl_type}.h5")
        logger.info(
            "Extracting activations for %d '%s' control prompts...",
            len(ctrl_prompts), ctrl_type,
        )
        extractor.extract(ctrl_prompts, ctrl_output)

    logger.info("Extraction complete. Files in: %s", acts_dir)


if __name__ == "__main__":
    main()
