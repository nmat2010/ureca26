# self-repr

Mechanistic interpretability research investigating whether LLaMA-3.1-8B-Instruct encodes a structured self/other direction in its residual stream.

**Gate 1 (Existence):** Is there a linear direction separating self-referential from other-referential activations?
**Gate 2 (Specificity):** Is that direction reducible to grammatical person, animacy, or instruction-tuning artifacts?

---

## AWS EC2 Setup

**Recommended instance:** `g5.xlarge` (1× A10G 24 GB, ~$1.006/hr) or `g5.2xlarge`

**AMI:** AWS Deep Learning AMI (Ubuntu 22.04) with PyTorch pre-installed

**Storage:** 100 GB EBS gp3

### Launch steps

1. Launch a `g5.xlarge` instance with the Deep Learning AMI, 100 GB gp3 root volume.
2. Open inbound SSH (port 22) in the security group.
3. SSH in:
   ```bash
   ssh -i your-key.pem ubuntu@<instance-public-ip>
   ```
4. Activate the PyTorch environment (pre-installed on the AMI):
   ```bash
   source activate pytorch
   ```
5. Clone this repo:
   ```bash
   git clone <your-repo-url> self-repr && cd self-repr
   ```
6. Install dependencies:
   ```bash
   pip install -e .
   ```
7. Log in to HuggingFace (required for LLaMA — you must accept Meta's license first at huggingface.co/meta-llama):
   ```bash
   huggingface-cli login
   ```

---

## Install (local or EC2)

```bash
pip install -e .
# or with dev dependencies:
pip install -e ".[dev]"
```

---

## Quick start

Run scripts in order:

```bash
python scripts/01_generate_prompts.py --config configs/experiment.yaml
python scripts/02_extract_activations.py --config configs/experiment.yaml
python scripts/03_find_directions.py --config configs/experiment.yaml
python scripts/04_test_specificity.py --config configs/experiment.yaml
python scripts/05_visualize.py --config configs/experiment.yaml
```

All outputs go into `data/` and `figures/`. Each script is independently re-runnable.

### Common overrides

```bash
# Use smaller batch size if OOM
python scripts/02_extract_activations.py --batch_size 4

# Use entity token position instead of final token
python scripts/03_find_directions.py --token_position entity
python scripts/04_test_specificity.py --token_position entity
python scripts/05_visualize.py --token_position entity

# Extract base model for comparison
python scripts/02_extract_activations.py --base_model
```

---

## Expected runtime on g5.xlarge (A10G 24 GB)

| Script | Description | Estimated time |
|--------|-------------|----------------|
| 01 | Generate prompts | < 1 minute |
| 02 | Extract activations (instruct + controls) | 20–40 minutes |
| 03 | Fit probes + compute directions | 5–15 minutes |
| 04 | Specificity / Gate 2 analysis | 5–10 minutes |
| 05 | Generate figures | < 2 minutes |

---

## Notes

- **LLaMA-3.1-8B** requires accepting Meta's license at [huggingface.co/meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) before downloading.
- Model weights are ~16 GB in float16; on a 24 GB A10G this leaves ~8 GB for activations. Default batch size is 8; reduce to 4 if you see OOM errors.
- TransformerLens is used by default; the code falls back to plain HuggingFace `transformers` if TransformerLens can't load the model.
- All activations are saved as compressed HDF5 (not pickle) for portability and memory efficiency.
