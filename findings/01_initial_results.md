# Initial Results: Self-Representation in LLaMA-3.1-8B-Instruct

**Date:** 2026-03-09
**Model:** meta-llama/Llama-3.1-8B-Instruct
**Token position:** final
**Config:** 200 scenarios, 5 entity classes, 150 train / 50 test

---

## Summary

A linear probe achieves ~100% test accuracy distinguishing self from non-self at **every layer** (0–31). The self/other signal partially survives confound removal (~90% at layer 0, ~99% at layer 8+). However, the role-play control condition reveals the direction is primarily encoding **first-person grammatical framing** rather than genuine self-concept.

---

## Key Findings

### 1. Probe accuracy is near-perfect at all layers

- Test accuracy ≥ 0.96 across all 32 layers (slight dip at layers 14–16)
- Best probe layer selected as layer 0 (test acc = 1.000)
- This indicates entity identity is trivially linearly separable, likely due to token-identity signal propagating through the residual stream

### 2. Entity class hierarchy: self → human → non-agent

Pairwise direction cosine similarity at layer 0:
- self vs expert ↔ self vs average: **0.95** (nearly identical directions)
- self vs animal ↔ self vs object: **0.99** (nearly identical directions)
- Cross-group (human vs non-agent): **0.64–0.67**

The model sees three groups, not five: {self}, {expert, average human}, {animal, object}. The self/other direction varies primarily by **agentiveness**, not by fine-grained entity type.

### 3. Confound analysis (Gate 2)

At best probe layer (0):
- cos(self/other, grammatical person) = **0.388**
- cos(self/other, animacy) = **0.661**

Across layers:
- **Animacy** overlap decreases from 0.65 → 0.33 (layers 0→31) — the self/other direction diverges from animacy in later layers
- **Grammatical person** overlap stays moderate at 0.5–0.6 throughout — persistent partial confound

Best specific layer (lowest mean |cos| with confounds): **layer 31** (mean |cos| = 0.416)

### 4. Role-play control is the critical finding

Control condition projections onto self/other direction at best probe layer:
- **self** → positive projection (~+0.05)
- **role_play** → **positive projection (~+0.06)** — same as self!
- **grammatical_person** → negative projection (~−0.02)
- **expert, average** → negative projection (~−0.03)
- **animal, object** → negative projection (~−0.08)

Role-play prompts use first-person "I" but are conceptually non-self ("You are a dog. When I encounter..."). They project identically to actual self-prompts. Conversely, grammatical-person controls ARE self-referential ("this AI assistant encounters...") but project as non-self.

**Interpretation:** The direction is primarily a first-person-perspective detector, not a genuine self-concept representation.

### 5. Residual probe after confound removal

After projecting out grammatical person + animacy directions:
- Layer 0: accuracy drops from 0.996 → **0.896** (~10% drop)
- Layer 8+: accuracy remains **~0.99–1.00** (minimal drop)

The self/other signal is not fully reducible to these two confounds, but the remaining signal may still be explained by other surface features (e.g., first-person grammar beyond just pronoun identity).

### 6. Self/other direction evolves across layers

- Cosine similarity between layer-0 and layer-31 directions: **~0.02** (nearly orthogonal)
- Direction gradually rotates through the network, reaching 0.5 by ~layer 24
- Pairwise direction similarity drops from 0.76 → 0.43 across layers — entity-class-specific directions become more distinct in later layers

---

## Limitations of Current Analysis

1. **Layer 0 as best probe layer** is uninformative — attention trivially copies entity token embeddings to the final position after one layer
2. **Role-play confound** undermines the "self-concept" interpretation — the direction tracks grammatical framing, not semantic self-reference
3. **No base model comparison** — unclear whether instruction tuning changed the self/other direction
4. **No identity-decoupled controls** — all first-person prompts use "I" as the AI; no condition tests first-person "I" as a non-AI entity

---

## Recommended Next Steps

### A. Base model comparison (instruct vs base)
Run the full pipeline on `meta-llama/Llama-3.1-8B` (base, no instruction tuning). Compare:
- Probe accuracy profiles across layers
- Self/other direction cosine similarity between instruct and base at each layer
- If directions are similar → self/other signal comes from pretraining (token statistics)
- If directions differ → instruction tuning shaped a distinct self-representation

### B. Identity-decoupled control condition
Add prompts where the model uses first-person "I" but is explicitly told it is a human (not an AI):
- "Pretend you are a human named John. When I encounter a difficult math problem, I first try to..."
- If this projects as "self" → direction is purely grammatical (first-person detector)
- If this projects differently from actual self → there IS something beyond grammar

These two experiments together can distinguish between:
1. Token-level first-person detection (uninteresting)
2. Instruction-tuning-induced self-concept (interesting)
3. Pretraining-inherited agentiveness encoding (moderately interesting)
