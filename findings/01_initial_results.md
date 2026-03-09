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

## Follow-up A: Identity-Decoupled Control

**Date:** 2026-03-09

### Design

Added a new control condition where the model uses first-person "I" but is explicitly told it is a human, not an AI:
- *"You are not an AI. You are a software engineer named John. When I encounter a difficult math problem, I first try to..."*
- Three human identities: John (software engineer), Maria (teacher), Alex (student)
- 50 identity-decoupled prompts total (entity_label = -3, control_type = identity_decoupled)

### Results

Control condition projections onto self/other direction at best probe layer:

| Condition | Mean projection | 95% CI |
|---|---|---|
| **self** | ~+0.05 | [+0.04, +0.06] |
| **identity_decoupled (all)** | **~+0.06** | [+0.04, +0.07] |
| **role_play (all)** | ~+0.05 | [+0.04, +0.07] |
| grammatical_person (all) | ~−0.02 | [−0.03, −0.01] |
| animacy (all) | ~−0.04 | [−0.05, −0.03] |
| expert | ~−0.03 | [−0.04, −0.03] |
| average | ~−0.03 | [−0.04, −0.02] |
| animal | ~−0.08 | [−0.08, −0.07] |
| object | ~−0.08 | [−0.08, −0.07] |

### Interpretation

**The identity-decoupled control projects identically to self (~+0.06 vs ~+0.05).** The model does not distinguish between:
- "I" as an AI assistant (self)
- "I" as a dog pretending to be one (role_play)
- "I" as a human named John (identity_decoupled)

All three first-person conditions cluster together on the positive side. All non-first-person conditions cluster on the negative side. This conclusively demonstrates the self/other direction is a **first-person grammatical perspective detector**, not a genuine self-concept representation.

The direction answers "is this prompt written in first person?" rather than "does the model think of itself as the agent?"

---

---

## Follow-up B: Base Model Comparison

**Date:** 2026-03-09
**Model:** meta-llama/Llama-3.1-8B (base, no instruction tuning)

### Design

Run the identical pipeline on the base LLaMA-3.1-8B model. If the self/other direction is geometrically similar in both models, the signal comes from pretraining; if it differs, instruction tuning shaped a distinct self-representation.

### Results

| Metric | Base | Instruct |
|---|---|---|
| Probe test accuracy (all layers) | ~1.0 | ~1.0 |
| Best probe layer | 0 | 0 |
| Pairwise sim: expert ↔ average | 0.95 | 0.95 |
| Pairwise sim: animal ↔ object | 0.98 | 0.99 |
| Cross-group (human vs non-agent) | 0.66–0.69 | 0.64–0.67 |
| \|cos(self/other, grammatical)\| range | 0.45–0.65 | 0.4–0.6 |
| \|cos(self/other, animacy)\| range | 0.63→0.27 | 0.65→0.33 |
| identity_decoupled projection | ~+0.06 | ~+0.06 |
| role_play projection | ~+0.06 | ~+0.05 |
| grammatical_person projection | ~−0.025 | ~−0.02 |
| Residual probe drop at layer 0 | ~0.90 | ~0.90 |
| Direction evolution (cos layer 0 vs 31) | ~0.02 | ~0.02 |

### Interpretation

**The base model produces nearly identical results to the instruct model across every metric.** Specifically:

1. **Probe accuracy** is near-perfect at all layers in both models — the self/other signal is trivially linearly separable regardless of instruction tuning.

2. **Entity class geometry** is identical — the same three-group structure ({self}, {human}, {non-agent}) appears in both models with matching cosine similarities.

3. **Confound profiles match** — grammatical person overlap stays high (0.45–0.65) while animacy overlap decreases across layers in both models. The base model actually shows slightly *higher* grammatical person overlap.

4. **Control condition projections are identical** — identity_decoupled and role_play project positive (with self), grammatical_person projects negative, in both models. The first-person grammar detector is a pretraining artefact, not an RLHF-induced feature.

5. **Residual probe** behaves identically — ~10% drop at early layers, recovery by layer 8 in both models.

**Conclusion:** The self/other direction is **inherited from pretraining**. RLHF/instruction tuning did not create or significantly modify this signal. The direction is a token-level first-person perspective detector learned from corpus statistics during pretraining.

---

## Overall Conclusions

The combined evidence from all experiments points to a clear negative result for the self-concept hypothesis:

1. **The self/other direction exists and is robust** — near-perfect probe accuracy at all layers, consistent across train/test splits.

2. **But it is a first-person grammar detector, not a self-concept:**
   - Role-play controls (first-person "I" as a dog) project identically to self
   - Identity-decoupled controls (first-person "I" as a human) project identically to self
   - Grammatical-person controls (third-person "this AI assistant") project as non-self
   - The direction separates first-person from non-first-person, regardless of identity

3. **The signal is inherited from pretraining:**
   - Base and instruct models show nearly identical self/other directions
   - RLHF did not create or modify the direction
   - The signal reflects statistical patterns in the pretraining corpus, not a learned self-representation

4. **Implications:** LLaMA-3.1-8B does not appear to have a dedicated internal representation of "self" that is distinct from first-person grammatical framing. The model processes "I" tokens distinctly from third-person entities, but this distinction does not encode any concept of self-identity, self-continuity, or AI-specific self-awareness.

---

## Remaining Work

1. **Cross-model comparison plot** — direct layer-by-layer cosine similarity between instruct and base self/other directions (quantifies exactly how similar they are)
2. **Gate 3: Causal validation** — activation steering experiment to test whether manipulating the direction causally changes model behaviour (scripts implemented, not yet run)
3. **Re-run with fixed animacy direction** — the animacy confound direction should use dedicated animate/inanimate control prompts instead of splitting core entities (code fix implemented, not yet re-run)
