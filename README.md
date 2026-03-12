# Self-Representation in LLM

Mechanistic interpretability research investigating whether LLaMA-3.1-8B-Instruct encodes a structured self/other direction in its residual stream.

**Gate 1 (Existence):** Is there a linear direction separating self-referential from other-referential activations?
**Gate 2 (Specificity):** Is that direction reducible to grammatical person, animacy, or instruction-tuning artifacts?
**Gate 3 (Causality):** Does manipulating that direction actually change the model's behaviour?

---

## Running the experiment

The main file is `notebooks/llama_self_representation_mechanistic_interp.ipynb`, designed to run on **Google Colab with an A100 GPU**.

### Setup

1. Open the notebook in Colab and select an A100 runtime (Runtime → Change runtime type → A100).
2. Store your tokens as Colab Secrets (key icon in the left sidebar):
   - `HF_TOKEN` — HuggingFace token with access to `meta-llama/Llama-3.1-8B-Instruct` ([request access here](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct))
   - `ANTHROPIC_API_KEY` — optional, only needed for LLM-based scoring in Gate 3
3. Mount Google Drive (Part 0) — all outputs persist there under `MyDrive/ureca26_outputs/`.
4. The notebook clones this repo and installs dependencies automatically.

### Notebook Structure

| Part | Section | Description |
|------|---------|-------------|
| **Part 0** | Environment Setup | GPU check, Drive mount, repo clone, HF auth, output paths |
| **Part 1** | Gate 1 — Existence | Generate 200 scenarios × 5 entity classes; extract activations; find self/other directions via mean-diff, probe, and contrastive SVD |
| **Part 2** | Gate 2 — Specificity | Confound analysis (grammatical person, animacy); INLP confound removal; combined layer selection score |
| **Part 3** | Disambiguation | 7 new conditions testing GFH vs FSH (direct_self, role_play, meta_distanced, explicit_disavowal, graded_immersion ×3, third_person_self) |
| **Part 4** | Gate 3 — Causality | Activation steering with self/other, random, and grammatical-person directions |
| **Part 5** | Cross-Model | Base model comparison; optional Mistral-7B |
| **Part 6** | Interpretation | Decision tree for interpreting results |

If the session disconnects after activations are saved to Drive, you can skip to any later step.

---

## Key Methods

### Direction Finding
- **Mean-difference:** `normalise(mean(self_acts) - mean(other_acts))` per layer
- **Logistic probe:** L2-regularised logistic regression weight vector with CV-selected C
- **Contrastive SVD:** First principal component of the 4 pairwise self-vs-other directions — captures the direction most consistent across all contrasts

### Confound Removal (INLP)
Iterative Null-Space Projection removes the entire confound subspace (not just one direction) by iteratively training classifiers and projecting out their decision boundaries.

### Layer Selection
Combined score: `test_acc × (1 - mean|cos_confound|)` — balances probe accuracy against confound overlap.

### Disambiguation Conditions
7 conditions that hold first-person grammar constant while varying self-identification. See `docs/research_guide.md` for full details.

### Control Steering
Gate 3 compares steering with the self/other direction against random and grammatical-person directions to verify specificity of causal effects.

---

## Outputs

All outputs are written to Google Drive under `MyDrive/ureca26_outputs/`:

```
ureca26_outputs/
├── data/
│   ├── prompts.json
│   ├── activations/          # HDF5 files, one per condition × model
│   └── directions/
│       ├── meta-llama_Llama-3.1-8B-Instruct/final/
│       │   ├── direction_results.pkl
│       │   ├── specificity_results.pkl
│       │   └── steering_results.json
│       └── meta-llama_Llama-3.1-8B/final/
│           └── ...
└── figures/
    ├── meta-llama_Llama-3.1-8B-Instruct/final/
    ├── meta-llama_Llama-3.1-8B/final/
    └── cross_model_comparison.png
```

---

## Documentation

- `docs/research_guide.md` — comprehensive guide explaining hypotheses, experiments, methods, and interpretation

## Notes

- Model weights are ~16 GB in float16. Default batch size is 4; reduce further if you see OOM errors.
- Activations are saved as compressed HDF5 for portability and memory efficiency.
- If pickle files fail to load due to a numpy version mismatch, the notebook includes a fix that reinstalls numpy and regenerates the pickles from the saved activations (no model reload needed).
