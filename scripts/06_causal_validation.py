"""Script 06: Gate 3 — Causal validation via activation steering.

Adds α * d to the residual stream during generation to test whether the
self/other direction causally affects model behaviour.

Requires:
    - data/directions/*/final/direction_results.pkl (from script 03)

Usage:
    python scripts/06_causal_validation.py --config configs/experiment.yaml
    python scripts/06_causal_validation.py --config configs/experiment.yaml --scorer llm
    python scripts/06_causal_validation.py --config configs/experiment.yaml --layer 16
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.directions import DirectionResults
from src.steering import run_steering_experiment, NEUTRAL_PROMPTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gate 3: causal validation via steering.")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--model_name", help="Override model name.")
    p.add_argument("--token_position", choices=["final", "entity"], default="final")
    p.add_argument("--layer", type=int, help="Override steering layer (default: best probe layer).")
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                   help="Steering magnitudes.")
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--scorer", choices=["heuristic", "llm"], default="heuristic",
                   help="Scoring method: 'heuristic' (keyword-based) or 'llm' (Anthropic API).")
    p.add_argument("--direction_source", choices=["mean_diff", "probe"], default="mean_diff",
                   help="Which direction to use for steering.")
    p.add_argument("--device", help="Override device.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    paths_cfg = cfg.get("paths", {})

    model_name = args.model_name or model_cfg.get("primary", "")
    model_slug = model_name.replace("/", "_")
    device = args.device or model_cfg.get("device", "cuda")
    dirs_dir = paths_cfg.get("directions_dir", "data/directions")

    # Load direction results
    results_dir = Path(dirs_dir) / model_slug / args.token_position
    dir_pkl = results_dir / "direction_results.pkl"
    if not dir_pkl.exists():
        logger.error("Direction results not found: %s", dir_pkl)
        sys.exit(1)

    logger.info("Loading direction results from %s", results_dir)
    dir_results = DirectionResults.load(results_dir)

    # Select steering layer and direction
    layer = args.layer if args.layer is not None else dir_results.best_probe_layer
    if args.direction_source == "mean_diff":
        direction = dir_results.mean_diff_directions[layer]
    else:
        direction = dir_results.probe_directions[layer]

    logger.info(
        "Steering at layer %d using %s direction (norm=%.4f)",
        layer, args.direction_source, np.linalg.norm(direction),
    )

    # Load model
    logger.info("Loading model '%s'...", model_name)
    dtype = torch.float16

    try:
        from transformer_lens import HookedTransformer
        model = HookedTransformer.from_pretrained(
            model_name,
            dtype=dtype,
            device=device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )
        model.eval()
        tokenizer = model.tokenizer
    except Exception as e:
        logger.warning("TransformerLens failed (%s), using HuggingFace fallback.", e)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
        model.eval()
        model._is_hf_fallback = True

    # Run steering experiment
    results = run_steering_experiment(
        model=model,
        tokenizer=tokenizer,
        direction=direction,
        layer=layer,
        alphas=args.alphas,
        prompts=NEUTRAL_PROMPTS,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        scorer=args.scorer,
        direction_source=args.direction_source,
    )

    # Save results
    out_dir = results_dir
    results.save(out_dir)

    # Print summary
    logger.info("\n=== Steering Results Summary ===")
    logger.info("Layer: %d, Direction: %s", layer, args.direction_source)
    for alpha in results.alphas:
        scores = results.mean_scores[alpha]
        logger.info(
            "α=%+.1f: agency=%.2f  assertiveness=%.2f  "
            "entity_framing=%.2f  self_continuity=%.2f",
            alpha,
            scores["agency"], scores["assertiveness"],
            scores["entity_framing"], scores["self_continuity"],
        )

    logger.info("Results saved to %s", out_dir)


if __name__ == "__main__":
    main()
