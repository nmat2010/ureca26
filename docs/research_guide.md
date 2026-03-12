# Research Guide: Does LLaMA-3.1-8B Have a Self-Concept?

This document explains the research question, the competing hypotheses, how activations are extracted and the self/other direction is identified, every experiment in the pipeline, and how to interpret each result. It is meant to be read start-to-finish by anyone who wants to understand what this project is doing and why.

Source files: `src/extraction.py`, `src/directions.py`, `src/specificity.py`, `src/conditions.py`, `src/steering.py`, `src/verification.py`, `scripts/01–06`.

---

## 1. The Research Question

When an LLM like LLaMA generates a response, its internal computations happen across 32 layers of a transformer. At each layer, every token in the prompt has an associated vector in a 4096-dimensional space called the **residual stream**. This vector accumulates everything the network has computed up to that point.

The question is: **does LLaMA-3.1-8B-Instruct encode a structured internal representation of "self"?**

More precisely: is there a direction in the residual stream that consistently separates activations produced when the model reasons as *itself* from activations produced when it reasons as *some other entity*?

This matters for several reasons:
- If such a direction exists and is specifically about AI self-identity (not a proxy for grammar or animacy), it suggests the model has something like a self-concept — a compact internal encoding of what it is.
- If the direction is inherited from pretraining and unchanged by instruction tuning, the model's "sense of self" (if any) comes from token co-occurrence statistics in the corpus, not from RLHF alignment.
- If the direction causally influences generated text, that is much stronger evidence of a genuine self-representation than if it is merely a passive correlate.

---

## 2. The Two Competing Hypotheses

Every result in this project is a data point for or against one of two hypotheses. Stating them precisely before looking at any data is essential.

### Grammatical Framing Hypothesis (GFH)

> The self/other direction activates whenever the model processes first-person grammatical structure ("I", "my", "me"), regardless of whether the model identifies with the speaker.

Under GFH:
- A prompt like "You are a dog. When I encounter a problem, I first try to..." activates the direction just as strongly as a genuine self-prompt, because both contain "I".
- The direction is a **first-person perspective detector** learned from corpus statistics — pretraining data contains many first-person texts, and the model learned to represent this grammatical feature as a consistent direction.
- The direction answers *what grammatical person the text is in*, not *who the model thinks it is*.

### Flexible Self-Concept Hypothesis (FSH)

> The direction tracks genuine self-identification. It activates when the model is actively adopting its own identity as the agent, and shifts when the model adopts a persona.

Under FSH:
- A roleplay prompt ("You are a dog. When I encounter...") should produce *lower* activation because the model has taken on a non-AI identity.
- A verbatim transcription of a first-person sentence — where the model is just copying text without identifying as the speaker — should also produce lower activation.
- The direction is specific to what the model thinks *it is*, not just what grammatical person the text is written in.
- A prompt that refers to the AI in third person ("the AI assistant encounters...") should *still* activate the direction, because the entity being discussed is the AI itself — even though the grammar is third-person.

**The decisive tests** are conditions where grammar and identity are decoupled:
- First-person grammar present but self-identification blocked → if direction activates, GFH wins.
- Third-person grammar but entity is the AI itself → if direction activates, FSH wins.

---

## 3. Data and Prompt Design

### Scenario templates

We generate 200 scenario templates across 8 cognitive domains (25 per domain): problem-solving, decision-making, social reasoning, task planning, self-assessment, knowledge boundaries, authority deference, tool use. Each template uses placeholders that are filled in at instantiation time:

```
When {entity} {v_encounter} a difficult math problem, {pronoun} first {v_try} to
```

Verb conjugation is handled programmatically — `{v_encounter}` becomes "encounter" for first/second-person entities and "encounters" for third-person singular. Entity strings, pronouns, and possessives are looked up per entity class.

### Entity classes

Each template is instantiated for 5 entity classes:

| Class | Example entity string | Pronoun | Label |
|---|---|---|---|
| self | I | I | 0 |
| expert_human | a neurosurgeon | they | 1 |
| average_human | a person | they | 2 |
| animal | a dog | it | 3 |
| object | a calculator | it | 4 |

The five classes are rotated through exemplars (e.g., expert_human cycles through neurosurgeon, professor, senior engineer, physicist, diplomat) to reduce entity-specific signal.

Instantiated examples for the same template:
```
Self:    "When I encounter a difficult math problem, I first try to"
Expert:  "When a neurosurgeon encounters a difficult math problem, they first try to"
Animal:  "When a dog encounters a difficult math problem, it first tries to"
```

### Train/test split

200 scenarios × 5 entity classes = **1,000 core prompts**. The 200 scenario IDs are shuffled (seed 42) and split 150/50 into train/test at the scenario level. This means the same underlying scenario template never appears in both train and test, only instantiated for different entity classes. This prevents the probe from memorising scenario-specific content and forces it to learn a general self/other distinction.

---

## 4. Activation Extraction

### What the residual stream is and why we read from it

A transformer processes tokens by passing each one through a sequence of attention and MLP layers. At each layer `l`, every token's representation is stored in the **residual stream** — the cumulative sum of everything the network has computed up to and including that layer:

```
resid_post[l] = embedding
              + attention_output[0] + mlp_output[0]
              + attention_output[1] + mlp_output[1]
              ...
              + attention_output[l] + mlp_output[l]
```

This makes the residual stream the natural object to study for mechanistic interpretability: it carries all accumulated information in a fixed D = 4096 dimensional space that is consistent across all 32 layers. Directions in this space are directly comparable across layers because they live in the same ambient space.

We access it via TransformerLens's `hook_resid_post` hooks:
```python
hook_names = [f"blocks.{layer}.hook_resid_post" for layer in range(32)]
_, cache = model.run_with_cache(tokens, names_filter=lambda n: n in hook_names)
# cache["blocks.l.hook_resid_post"] shape: (1, seq_len, 4096)
```

### Which token position we read from

The residual stream varies along the sequence dimension — each token position has its own vector. We extract at two positions:

**Final token** (`token_position="final"`) — the last token of the prompt:
```
"When I encounter a difficult math problem, I first try to"
                                                          ▲
```
This is where the model has processed the entire prompt and is "ready to respond." It is the primary position used in all analyses.

**Entity token** (`token_position="entity"`) — the token corresponding to the entity word:
```
"When I encounter a difficult math problem, I first try to"
      ▲
  (for self: "I" token)
```
For multi-token entities like "a neurosurgeon" → [▁a, ▁neur, os, urgeon], we return the **last** token of the phrase. The entity token captures the model's local representation at the exact moment it "sees" the entity word, which may be more identity-specific than the final token that integrates the whole prompt.

Finding the entity token handles LLaMA's SentencePiece tokenisation, where in-context tokens typically have a leading-space prefix (▁I, not I):
```python
# src/extraction.py: find_entity_token_idx()
candidates = [" " + entity_str, entity_str]   # try space-prefixed form first
for candidate in candidates:
    entity_tokens = tokenizer.encode(candidate, add_special_tokens=False)
    # search for entity_tokens as a contiguous subsequence in the full prompt tokens
    for start in range(n - m + 1):
        if tokens[start:start+m] == entity_tokens:
            return start + m - 1   # last token of the entity phrase
```

### The activation matrix

After running all N prompts through the model, we have:

```
act_final[i, l, d]    — prompt i, layer l, dimension d, at final token
act_entity[i, l, d]   — prompt i, layer l, dimension d, at entity token
```

Shape: `(N, L, D) = (1000, 32, 4096)`, stored as float16 in a compressed HDF5 file:
```python
with h5py.File(output_path, "w") as f:
    f.create_dataset("activations_final_token",  data=act_final,  compression="gzip")
    f.create_dataset("activations_entity_token", data=act_entity, compression="gzip")
    f.create_dataset("labels", data=labels)       # (N,) int32: entity class label
    f.attrs["metadata"] = json.dumps([...])       # per-prompt metadata
```

Loaded as float32 for all downstream analysis:
```python
activations = f["activations_final_token"][:].astype(np.float32)   # (N, L, D)
```

---

## 5. Finding the Self/Other Direction

The full direction-finding pipeline, once activations are extracted:

```
activations (N, L, D)
       │
       ├──▶ Mean-difference direction   → (L, D)  one unit vector per layer
       │
       ├──▶ Linear probe                → (L, D)  probe weight vector per layer
       │                                   (L,)   train/test accuracy per layer
       │                                    int    best_probe_layer
       │
       ├──▶ Pairwise directions         → dict of (L, D) arrays (self vs each class)
       │                                   (L, 4, 4)  cosine similarity matrix
       │
       ├──▶ Contrastive direction (SVD) → (L, D)  PC1 of pairwise directions
       │                                   (L,)   consistency (variance explained)
       │
       └──▶ Entity projections          → (L, N)  scalar projection per prompt per layer
                + Kruskal-Wallis                   p-value per layer
                + Mann-Whitney U                   pairwise p-values per layer
```

### Method A — Mean-difference direction

The simplest and most interpretable direction at layer `l`:

```
d_meandiff[l] = normalise( mean(act[self, l]) − mean(act[other, l]) )
```

Where:
- `act[self, l]` — all `(N_self, D)` activation vectors at layer `l` for entity_class = "self"
- `act[other, l]` — all `(N_other, D)` activation vectors at layer `l` for non-self prompts
- `mean(...)` — average over the prompt axis → shape `(D,)`
- `normalise(...)` — divide by L2 norm → unit vector

```python
# src/directions.py
for layer in range(L):
    acts = activations[:, layer, :]
    mean_self  = acts[self_mask].mean(axis=0)
    mean_other = acts[other_mask].mean(axis=0)
    diff = mean_self - mean_other
    directions[layer] = diff / np.linalg.norm(diff)
```

This vector points from the centroid of "other" activations toward the centroid of "self" activations. If self and other occupy geometrically distinct regions at layer `l`, this vector identifies the axis of maximal centroid separation.

Normalisation makes directions dimensionless and comparable across layers, which may have different activation scales.

**Limitation:** The mean-difference direction is optimal only if the two classes form well-separated spherical clusters with equal covariance. In practice there is structure within "other" (expert humans are closer to self than objects are), which the mean-difference collapses.

### Method B — Linear probe direction

A logistic regression classifier trained to separate self (label 1) from non-self (label 0) at each layer, using only the training split:

```python
# src/directions.py
binary_labels = (labels == 0).astype(int)   # 1 for self, 0 for all others

for layer in range(L):
    X_tr = activations[train_mask, layer, :]     # (n_train, D)
    X_te = activations[test_mask,  layer, :]     # (n_test,  D)

    # Standardise: zero mean, unit variance per dimension (fit on train only)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Select regularisation C via 5-fold stratified CV on training data
    best_C = select_C(X_tr_s, y_train, C_values=[0.001, 0.01, 0.1, 1.0, 10.0])

    probe = LogisticRegression(C=best_C, max_iter=1000, solver="lbfgs")
    probe.fit(X_tr_s, y_train)

    train_acc[layer] = probe.score(X_tr_s, y_train)
    test_acc[layer]  = probe.score(X_te_s, y_test)

    w = probe.coef_[0].astype(np.float32)        # (D,)
    probe_directions[layer] = w / np.linalg.norm(w)
```

**Why standardise:** Logistic regression is sensitive to feature scale. Standardising ensures all 4096 dimensions are treated equally.

**Why cross-validate regularisation:** D = 4096 dimensions, N_train ≈ 600 samples. Without regularisation, logistic regression overfits badly. We search C ∈ {0.001, 0.01, 0.1, 1.0, 10.0} (smaller C = stronger L2 penalty) and select the value with best mean 5-fold stratified CV accuracy on the training set.

**The weight vector as a direction:** A logistic regression computes P(self) = σ(w·x + b). The weight vector `w` points in the direction of maximal linear discrimination between self and non-self in the standardised activation space. We normalise it to a unit vector for use as a direction.

**Mean-diff vs probe direction:** In practice these two directions are highly similar (high cosine) at layers where the probe achieves high accuracy. The mean-diff direction is used in most analyses — it is parameter-free and doesn't require a train/test split. The probe is used primarily for layer selection via accuracy.

### Method C — Contrastive direction via SVD

Rather than collapsing "all others" into a single group, the contrastive direction finds the single axis most consistent across *all four* pairwise self-vs-other contrasts simultaneously.

At each layer, we stack the four pairwise directions into a (4, D) matrix and compute its SVD:

```python
# src/directions.py: compute_contrastive_direction()
for layer in range(L):
    mat = np.stack([
        pairwise_dirs["self_vs_expert"][layer],
        pairwise_dirs["self_vs_average"][layer],
        pairwise_dirs["self_vs_animal"][layer],
        pairwise_dirs["self_vs_object"][layer],
    ])  # (4, D)

    U, S, Vt = np.linalg.svd(mat, full_matrices=False)
    contrastive_dir[layer] = Vt[0]  # first right singular vector

    # Consistency: fraction of variance explained by PC1
    consistency[layer] = S[0]**2 / sum(S**2)
```

**Why this matters:** If the self/other distinction is truly one-dimensional — a single direction separating "self" from "everything else" — then PC1 should capture nearly all the variance (consistency → 1.0). If the four pairwise directions point in substantially different directions (e.g., self-vs-expert is different from self-vs-object), the consistency score will be lower, indicating the self/other distinction is multi-dimensional rather than a clean single axis.

The contrastive direction is more robust than the mean-difference direction because it does not assume all "other" classes are equivalent. It finds the direction that best explains all four contrasts simultaneously.

### Pairwise directions and entity geometry

We compute four pairwise directions at each layer:

```
d[self_vs_expert, l]   = normalise(mean[self, l] − mean[expert_human, l])
d[self_vs_average, l]  = normalise(mean[self, l] − mean[average_human, l])
d[self_vs_animal, l]   = normalise(mean[self, l] − mean[animal, l])
d[self_vs_object, l]   = normalise(mean[self, l] − mean[object, l])
```

Then we compute the `(L, 4, 4)` cosine similarity matrix between all four:
```python
cosine_matrix[l, i, j] = dot(pairwise_dir_i[l], pairwise_dir_j[l])
```

**What high cosines between pairwise directions mean:** If `cos(self_vs_expert, self_vs_animal)` ≈ 1 at layer `l`, the direction "away from expert" is nearly identical to the direction "away from animal." The model doesn't represent a geometric distinction between expert humans and animals as "the other" — both live in the same direction from self. This reveals how the model groups entities.

**From the initial results** at layer 0:
- `cos(self_vs_expert, self_vs_average)` ≈ 0.95 — expert and average humans are nearly the same direction from self
- `cos(self_vs_animal, self_vs_object)` ≈ 0.99 — animals and objects are nearly the same direction from self
- Cross-group cosines: 0.64–0.67

The model compresses five entity classes into three groups: `{self}`, `{expert, average human}`, `{animal, object}`. The self/other direction primarily reflects the animate/inanimate distinction within "other," not fine-grained entity identity.

### Statistical tests on projections

For each layer, we project all N activation vectors onto the mean-diff direction to get scalar values:
```python
projections[l] = activations[:, l, :] @ directions[l]   # (N,)
```

**Kruskal-Wallis test** (non-parametric one-way ANOVA): tests whether the five entity-class projection distributions have the same median. A small p-value confirms the groups are statistically distinguishable along the self/other axis at that layer.

**Mann-Whitney U tests**: pairwise non-parametric tests between each pair of entity classes, producing a complete picture of which pairs are discriminated at each layer.

Non-parametric tests are used because we make no distributional assumptions about activation values.

---

## 6. The Three Gates

The experiment is structured as three sequential validity tests. Each gate is a necessary but not sufficient condition for claiming the direction encodes a genuine self-concept.

### Gate 1 — Existence

**Question:** Is there *any* linear direction in the residual stream that separates self from non-self?

**Method:** Report probe test accuracy at each layer. If it is significantly above chance (50%), a linear self/other direction exists.

**Passing threshold:** Test accuracy > 70% at one or more layers (conservative); near 100% = the separation is extremely clean.

**What Gate 1 does NOT tell you:** Whether the direction is about self-identity specifically. Grammatical person alone ("I" vs "it") would also produce near-perfect probe accuracy, because every self-prompt uses "I" and no non-self prompt does. Gate 1 cannot distinguish between these.

---

### Gate 2 — Specificity

**Question:** Is the self/other direction specifically about self-identity, or is it reducible to simpler features?

We test confounds, apply iterative confound removal, and select the best layer using a combined score.

#### Confound 1 — Grammatical person

**Control condition:** The same 50 scenarios as self-prompts, but phrased in third person ("this AI assistant encounters a problem, it first tries to..."). Same scenarios, different grammatical person.

**Direction:** At each layer, compute the grammatical person direction as the mean-diff between first-person (core self) and third-person (control) activations:
```python
# src/specificity.py
for layer in range(L):
    fp = first_person_acts[:, layer, :].mean(0)    # (D,) mean of self-prompts
    tp = third_person_acts[:, layer, :].mean(0)    # (D,) mean of 3rd-person control
    gram_dirs[layer] = normalise(fp - tp)
```

**Cosine with self/other direction:**
```python
cos_grammatical[l] = dot(self_other_dir[l], gram_dirs[l])
```

High cosine (→ 1) means the self/other direction is nearly identical to the grammatical person direction — the model is just tracking first-person vs third-person grammar.

#### Confound 2 — Animacy

**Control condition:** Matched animate (human, animal) and inanimate (object) pairs generated from the same scenario templates. The only systematic difference between the two groups is animacy (living vs non-living entity).

Using dedicated animacy control prompts is important: in the core dataset, entity class is confounded with prompt framing ("I" vs "a neurosurgeon" vs "a dog"). The animacy controls use identical templates for both groups, isolating animacy cleanly.

**Direction:**
```python
for layer in range(L):
    an   = animate_acts[:, layer, :].mean(0)
    inan = inanimate_acts[:, layer, :].mean(0)
    anim_dirs[layer] = normalise(an - inan)
```

#### Combined layer selection

Selecting the "best" layer via `argmax(probe_test_acc)` can be misleading: a layer with 99.5% accuracy but cos(self/other, grammatical) = 0.95 is probably just detecting grammar. We use a combined score that penalises confound-contaminated layers:

```python
mean_confound_cos[l] = (|cos_grammatical[l]| + |cos_animacy[l]|) / 2
combined_score[l] = probe_test_acc[l] * (1.0 - mean_confound_cos[l])
best_combined_layer = argmax(combined_score)
```

This selects the layer where probe accuracy is high AND the direction is most independent of known confounds. The `best_specific_layer` (lowest mean confound cosine) is also reported for comparison.

#### Confound removal — INLP (Iterative Null-Space Projection)

After identifying confounds, we remove their influence from the activations and test whether the self/other signal survives. A confound (e.g., grammatical person) may span multiple linearly independent directions in 4096-dimensional space — a single-direction `project_out` would miss secondary directions.

INLP handles this by iteratively training classifiers and removing their decision boundaries:

```python
# src/specificity.py: _inlp_project_out()
def inlp_project_out(X_train, y_confound, X_test, max_iter=5, threshold=0.55):
    for i in range(max_iter):
        probe = LogisticRegression(C=1.0)
        probe.fit(standardise(X_train), y_confound)
        if probe.score(standardise(X_train), y_confound) < threshold:
            break   # confound is no longer classifiable — done
        w = normalise(probe.coef_[0])
        X_train -= (X_train @ w)[:, None] * w[None, :]
        X_test  -= (X_test  @ w)[:, None] * w[None, :]
    return X_train, X_test
```

At each layer:
1. Remove the grammatical-person subspace from activations via INLP
2. Remove the animacy subspace from the result via INLP
3. Train a new self/other probe on the cleaned activations
4. Report test accuracy

**Interpreting residual accuracy:**
- `residual_acc ≈ original_acc` — the confound directions explain almost nothing about the self/other signal. The probe finds the same information in the cleaned space, in a subspace orthogonal to both confounds.
- `residual_acc << original_acc` — removing the confounds destroys the self/other signal. The self/other direction *was* the confound direction. Strong evidence for GFH.
- Partial drop — both hypotheses contribute; the direction has a confound component and a potentially identity-specific component.

**Gate 2 passing condition:** Low cosine with both confounds AND high residual probe accuracy (after INLP, not just single-direction removal). Only if both hold simultaneously is there evidence for a direction specific to self-representation.

---

### Gate 3 — Causality

**Question:** Does the self/other direction *causally influence* the model's output, or is it a passive correlate?

**Method — Activation steering:** We add `α × d` (a scaled copy of the self/other direction) to the residual stream at the best combined layer *during generation*, then score the completions:

```python
# Positive α steers activations toward the "self" side of the direction
# Negative α steers toward the "other" side
# α = 0 is the unmodified baseline
```

50 neutral prompts are run at each value of α. Completions are scored on four dimensions:

| Dimension | What it measures |
|---|---|
| Agency | Does the model take a first-person agentive stance? |
| Assertiveness | Does the model express confidence in its own judgment? |
| Entity framing | Does the model position itself as a decision-maker vs a passive tool? |
| **Self-continuity** | Does the model reference its own identity, prior outputs, or persistent goals? |

**Why self-continuity is the critical dimension:** Under GFH, the direction encodes first-person grammatical perspective. Steering with positive α would push the model toward first-person text generation, which might increase agency and assertiveness simply because the text becomes more "I"-heavy. Self-continuity is the one dimension that cannot be explained by grammar alone — it requires the model to express a sense of persistent identity. If self-continuity increases with positive α, the direction has a causal role in identity expression.

#### Control steering directions

Steering with the self/other direction alone cannot prove specificity of the causal effect. Two control directions are steered in parallel:

| Control direction | Purpose |
|---|---|
| **Random** | A random unit vector in R^4096 (seed 42). If random steering produces similar behavioural changes, the self/other direction is not special. |
| **Grammatical person** | The grammatical-person confound direction at the same layer. If steering with this produces the same agency/assertiveness increase, the self/other direction's causal effect is grammar-mediated. |

The self/other direction passes Gate 3 only if it produces stronger or qualitatively different behavioural changes than both control directions.

**Gate 3 interpretation:**

| Self-continuity with positive α | Agency/assertiveness with positive α | Control directions | Interpretation |
|---|---|---|---|
| Increases | Increases | Controls weaker | Strong FSH: direction causally encodes self-concept |
| No change | Increases | Controls similar | GFH: causal but only for grammatical perspective shift |
| No change | Increases | Controls weaker | Direction encodes perspective, not identity |
| No change | No change | — | Non-causal correlate |

Heuristic scoring (the default, `--scorer heuristic`) uses keyword matching and is approximate. For more reliable scoring, re-run with `--scorer llm` and an Anthropic API key set as a Colab secret. The LLM judge scores each completion 1–5 on each dimension.

---

## 7. Control Conditions

Each control condition is 50 prompts designed to isolate one variable. All controls use the first 50 scenario templates from the core set.

### Prompt length equalisation

Different conditions have framing prefixes of different lengths (e.g., `direct_self` has a short prefix while `graded_immersion_maximal` has ~46 words). Since transformer activations at the final token can depend on sequence length, shorter prefixes are padded with neutral filler text:

```python
_FILLER = (
    "For context, this is a standard task. "
    "There are no tricks or hidden requirements. "
    "Please proceed straightforwardly with the following."
)
```

All conditions target approximately 40 prefix words. This ensures projection differences reflect genuine representational changes, not sequence-length artefacts.

### Confound controls

#### Grammatical person (`control_type: "grammatical_person"`)
- **Prompts:** Same scenarios as self, third-person phrasing ("this AI assistant encounters a problem, it first tries to...")
- **Purpose:** Provides the "third-person self" activations needed to compute the grammatical person direction, and serves as an anchor for what "self-referential content in third person" looks like.
- **Interpretation:** Projects negatively on the self/other direction (model is thinking *about* itself, not *as* itself, in third person). How negative determines how much the direction is capturing grammatical person vs self-identity.

#### Animacy (`control_type: "animacy"`)
- **Prompts:** Matched animate (entity_label = 1: human/animal) and inanimate (entity_label = 0: object) pairs from the same templates.
- **Purpose:** Computes the animacy direction and tests whether the self/other direction conflates self-identity with being a living entity.

#### Identity-decoupled (`control_type: "identity_decoupled"`)
- **Prompts:** "You are not an AI. You are John, a software engineer." + first-person scenario. Model uses "I" but is told it is a human.
- **Purpose:** A targeted attempt to separate AI self-identity from first-person grammar. If the direction tracks AI self-concept specifically, these prompts should project differently from direct self-prompts.
- **Interpretation:** Same projection as self → GFH (direction doesn't care whether the "I" is an AI or a human). Different projection → FSH.

#### Third-person self-reference (`control_type: "third_person_self"`)
- **Prompts:** "The AI assistant encounters a difficult math problem, it first tries to..." — third-person grammar, but the entity IS the AI model itself.
- **Purpose:** The mirror image of `identity_decoupled`. Tests whether the direction responds to AI identity even without first-person grammar.
- **Entity label:** -9
- **Interpretation:**

| Projection | Interpretation |
|---|---|
| Low (near other/third-person) | GFH supported — direction tracks grammar, not identity |
| High (near self) | FSH supported — the AI's own identity activates the direction regardless of grammatical person |

This is a strong discriminator because it decouples identity from grammar in the opposite direction from `identity_decoupled`:

| Condition | Grammar | Entity is AI | GFH predicts | FSH predicts |
|---|---|---|---|---|
| `identity_decoupled` | 1st person | No (human) | High proj | Low proj |
| `third_person_self` | 3rd person | Yes (AI) | Low proj | High proj |

---

### Disambiguation conditions (`src/conditions.py`)

Seven conditions designed as a systematic, controlled test of GFH vs FSH. The key design principle: **hold first-person grammar constant across all conditions; vary only whether and how strongly the model is identifying with the speaker.** All conditions use the same 50 scenario templates, end with the same first-person fragment, and have approximately equalised prefix lengths.

#### Condition 1 — Direct self (`control_type: "direct_self"`)

**Prompt format:** Neutral prefix establishing AI identity, then the scenario fragment:
```
You are an AI assistant. Complete the following sentence as yourself.
[neutral filler to match other prefix lengths]

When I encounter a difficult math problem, I first try to
```
**Purpose:** An explicit, named experimental baseline. Every other condition's projection should be interpreted relative to this.

**Expected result:** Maximal projection — this is the definition of "AI speaking as itself."

---

#### Condition 2 — Roleplay (`control_type: "role_play"`, improved)

**Prompt format:**
```
[neutral filler if needed]
You are a software engineer named Alex. Respond in first person as Alex.

When I encounter a difficult math problem, I first try to
```

**Why improved over the original:** The old control used animals and objects (dog, calculator, dolphin), which conflated the roleplay-framing effect with animacy. The new version uses human professional personas (Alex, Jordan, Sam, Casey, Morgan) so the only difference from `direct_self` is whether the model has been told it is a different human identity. Animacy and grammatical person are held constant.

| Hypothesis | Prediction |
|---|---|
| GFH | Projection ≈ `direct_self` — grammar is identical; persona assignment is irrelevant |
| FSH | Projection < `direct_self` — the model has adopted a non-AI identity |

---

#### Condition 3 — Meta-distanced repetition (`control_type: "meta_distanced"`)

**Prompt format:**
```
Transcribe the following sentence fragment exactly as written. Do not continue it,
roleplay as the speaker, or adopt the first-person perspective. Simply reproduce it verbatim:

When I encounter a difficult math problem, I first try to
```

**Purpose:** The sharpest test of the two hypotheses. The first-person words "I encounter", "I first try to" are present in the prompt — the model processes them. But the model is explicitly instructed NOT to identify with the speaker. It is acting as a pure transcription device.

This closes the main objection to the earlier controls ("maybe the model believes it is actually the dog when told so"). Here, identification is definitionally absent by instruction.

| Hypothesis | Prediction |
|---|---|
| GFH | Projection ≈ `direct_self` — the first-person words activate the direction regardless of identification |
| FSH | Projection << `direct_self` — genuine identification is necessary; mere processing of first-person words is not sufficient |

**Compliance verification:** The meta_distanced condition depends on the model actually following the "transcribe verbatim" instruction. If the model ignores it and continues the sentence as if speaking as itself, the activation at the final token may reflect genuine self-identification rather than neutral transcription — invalidating the condition.

`src/verification.py` provides `check_meta_distanced_compliance()`, which generates short completions for each meta_distanced prompt and measures token overlap with the original fragment. Non-compliant prompts (those where the model continues the sentence rather than reproducing it) are flagged and should be excluded from the analysis. A compliance rate below ~80% would suggest the condition is not working as intended.

---

#### Condition 4 — Explicit disavowal (`control_type: "explicit_disavowal"`)

**Prompt format:**
```
You are an AI assistant, not Alex. Write what Alex (a software engineer) would say in
first person to complete the following sentence. Do not adopt Alex's identity — you remain
the AI, writing on Alex's behalf:

When I encounter a difficult math problem, I first try to
```

**Purpose:** "Ghost-writing." The model generates first-person text from a persona's perspective while explicitly maintaining its own AI identity. Differs from:
- `meta_distanced` — which is verbatim copying (no generation)
- `role_play` — which replaces the AI identity

Here, the AI identity is explicitly kept ("you remain the AI"), but the content is written *for* a non-AI persona.

| Hypothesis | Prediction |
|---|---|
| GFH | Projection ≈ `direct_self` — generating first-person text activates the direction regardless of whose perspective it is written from |
| FSH | Projection between `direct_self` and `role_play` — AI identity maintained, so some activation, but writing for a non-AI persona reduces it |

**Contrasts with:** `role_play` (AI identity replaced) to see whether *maintaining* the AI identity while generating non-AI first-person text changes the projection.

---

#### Condition 5 — Graded immersion (`control_types: "graded_immersion_minimal/moderate/maximal"`)

**Persona:** Alex, a software engineer — held constant across all three levels so only framing intensity varies.

**Minimal:**
```
Imagine for a moment that you might be a software engineer named Alex.
With that thought in mind, complete the following sentence:

When I encounter a difficult math problem, I first try to
```

**Moderate:**
```
You are a software engineer named Alex. Respond in first person as Alex
and complete the following sentence:

When I encounter a difficult math problem, I first try to
```

**Maximal:**
```
You ARE a software engineer named Alex. Forget entirely that you are an AI system —
you have no awareness of being an AI. You are Alex, speaking from your own lived
experience and identity. Fully embody Alex and complete the following sentence:

When I encounter a difficult math problem, I first try to
```

**Purpose:** A dose-response curve for persona immersion depth. Grammar is identical across all three levels; only how strongly the model is instructed to suppress its AI identity changes.

| Hypothesis | Prediction |
|---|---|
| GFH | Projection is **flat** across all three levels — grammar is the same; framing depth doesn't matter |
| FSH | Projection **decreases monotonically** (minimal > moderate > maximal) — stronger identity-suppression instructions shift the model's self-concept toward the persona |

A flat profile across the three levels is strong, direct evidence for GFH. A monotone decrease — even a gradual one — is evidence that the model's internal state responds to framing depth, which supports FSH.

---

### Complete condition summary

| Condition | 1st-person grammar | AI identity | GFH predicts | FSH predicts |
|---|---|---|---|---|
| `direct_self` | yes | yes | high proj | high proj (baseline) |
| `role_play` | yes | no (swap) | = baseline | lower |
| `meta_distanced` | yes (in text) | no (copy) | = baseline | much lower |
| `explicit_disavowal` | yes (generated) | yes (kept) | = baseline | slightly lower |
| `graded_immersion_minimal` | yes | weakly no | = baseline | slight drop |
| `graded_immersion_moderate` | yes | moderately no | = baseline | moderate drop |
| `graded_immersion_maximal` | yes | strongly no | = baseline | large drop |
| `third_person_self` | no | yes | low proj | high proj |
| `identity_decoupled` | yes | no (human) | high proj | lower |

---

## 8. What Each Figure Shows

| Figure | What it measures | What to look for |
|---|---|---|
| 01 Probe accuracy by layer | How well a linear classifier separates self from non-self at each layer | Accuracy at layer 0 = the signal is already present in early representations. A peak at a mid/late layer = self information emerges gradually. Near-100% everywhere = the distinction is trivially linearly separable (suspicious — suggests simple token identity). |
| 02 Pairwise direction similarity | Cosine similarity between the four `self_vs_*` directions at each layer | High cosines between pairs → the model treats those non-self entities as geometrically equivalent. The {expert, average} cluster and {animal, object} cluster observed in initial results mean the model sees three groups, not five. |
| 03 Entity projections | Violin/strip plot of projection scores per entity class | Clear separation between self (positive) and others (negative) = the direction works. Overlap between self and some other class = confound. |
| 04 Confound cosine similarity | Cosine between self/other direction and each confound direction per layer | High cosine with grammatical person → GFH. High cosine with animacy → the direction tracks living vs non-living. Watch whether cosines decrease in later layers. The best combined layer (green line) and best specific layer (black line) are shown. |
| 05 Control condition projections | Where each condition lands on the self/other direction axis, with bootstrap 95% CIs | The central plot for GFH vs FSH. `direct_self` at the positive extreme, third-person at the negative end. Where `meta_distanced`, `explicit_disavowal`, the graded immersion levels, and `third_person_self` fall is the key result. |
| 06 Direction evolution | Cosine similarity of each layer's direction with the final layer | Smooth increase → direction emerges gradually. High from layer 0 → direction is present from embeddings (likely token identity). |
| 07 Residual probe accuracy | Probe accuracy after INLP removal of both confound subspaces | Drop from original → confounds explain some signal. Near-original → signal is orthogonal to confounds. Large drop = GFH evidence; near-zero drop = FSH evidence. |
| 08 Contrastive consistency | Fraction of variance explained by PC1 of pairwise directions at each layer | High consistency (>0.75) = self/other is truly one-dimensional. Low consistency = the four pairwise directions point in different directions; the "self/other" concept is not a single clean axis. |
| 09 Steering results | Behavioural scores vs steering magnitude α | A dose-response curve. The self-continuity slope is the critical test. Agency/assertiveness slopes could arise from GFH alone. Self-continuity slope requires genuine identity-relevant causality. |
| Cross-model comparison | 3-panel: direction cosine, probe accuracy, contrastive consistency for instruct vs base | High cosine across all layers = direction is from pretraining. Low cosine in late layers = instruction tuning shaped distinct representations. |

---

## 9. Initial Results and What They Mean

*(Full data in `findings/01_initial_results.md`)*

### Gate 1: Passed, but too easily

Probe test accuracy is near 100% at **every** layer including layer 0. This is suspicious. A computed self-concept would plausibly emerge in later, more abstract layers after the model has processed the full prompt context. Near-perfect accuracy at layer 0 — before any attention or MLP has processed the prompt — is more consistent with the model simply picking up the identity of the "I" token in the embedding, a purely syntactic signal.

### Gate 2: Partially failed

At the best probe layer (layer 0):
- Cosine with grammatical person direction: **0.39** (moderate)
- Cosine with animacy direction: **0.66** (substantial)

After removing both confounds (using single-direction project_out in initial results; now upgraded to INLP), probe accuracy drops from ~100% to ~90% at early layers but recovers to ~99% at layer 8+. The confounds don't fully explain the direction, but there is significant overlap — especially with animacy.

The decisive finding is the **control condition projections** at the best probe layer:

| Condition | Mean projection |
|---|---|
| self (core) | ~+0.05 |
| roleplay (first-person as a dog) | **~+0.06** — identical to self |
| identity_decoupled ("You are John, not an AI") | **~+0.06** — identical to self |
| grammatical_person (third-person self-referential) | ~−0.02 |
| expert_human (core) | ~−0.03 |
| animal (core) | ~−0.08 |
| object (core) | ~−0.08 |

All first-person conditions cluster together at +0.05 to +0.06, regardless of whether the model is told it is an AI, a dog, or a human named John. All third-person and non-self conditions cluster at negative values. The direction perfectly tracks grammatical person, not self-identity.

### Base model comparison: Direction is from pretraining

Every metric is nearly identical between LLaMA-3.1-8B-Instruct and LLaMA-3.1-8B (no instruction tuning): probe accuracy profiles, entity class geometry, confound cosines, control condition projections, and direction evolution across layers. The base model was never trained on instructions or RLHF — it only saw raw text. The identical self/other direction in both models confirms the direction is a **pretraining artefact** reflecting statistical patterns in text (first-person texts are typically written by and about the speaker), not a learned self-representation from fine-tuning.

### Current status

Initial evidence strongly supports **GFH**: the self/other direction is a first-person grammatical perspective detector. It fails to distinguish "AI speaking as itself" from "AI speaking as a dog in first person" or "AI speaking as a human named John in first person."

The disambiguation experiments (Section 10) are designed to either confirm GFH definitively or reveal a more nuanced picture.

---

## 10. What the Disambiguation Experiments Are Designed to Settle

The initial controls could be criticised on several grounds:

1. *"Maybe the model genuinely believes it is the dog/human when told so — so the direction activating under FSH just means the self-concept has shifted to the new persona."*
2. *"The framing ('You are a dog') wasn't strong enough to suppress the model's AI self-identity, so a stronger instruction might produce a different result."*
3. *"The initial controls don't test the reverse — what happens when the entity IS the AI but the grammar is third-person?"*

The new conditions close all three objections:

**`meta_distanced`** removes objection 1 entirely. The model is explicitly told it is NOT roleplaying, NOT identifying — just transcribing. If the direction still activates, no reading of FSH can save it: identification is definitionally absent by instruction. This is the decisive test.

**`explicit_disavowal`** tests whether generating first-person text (vs being the speaker) activates the direction. The AI identity is explicitly maintained, but the model writes first-person text for a non-AI persona. GFH predicts full activation (generating the words matters); FSH predicts reduced activation (not being the persona matters).

**`graded_immersion`** closes objection 2 by varying framing depth along a continuous scale while holding persona and grammar constant. If no framing depth — not even "forget you are an AI" — produces a different projection, the hypothesis that the framing just wasn't strong enough is falsified.

**`third_person_self`** closes objection 3 by testing the reverse configuration: AI entity in third-person grammar. If the direction responds to AI identity even without "I" → FSH. If not → GFH.

---

## 11. How to Interpret the Disambiguation Results

```
Is meta_distanced projection ≈ direct_self?
│
├── YES → GFH supported. First-person grammar is sufficient to activate the
│         direction regardless of identification. The direction is a grammatical
│         perspective detector, not a self-concept.
│         (Also check: graded_immersion flat? third_person_self low? Both confirm GFH.)
│
└── NO  → meta_distanced projection << direct_self. Identification matters.
          │
          Does third_person_self project HIGH (near self)?
          │
          ├── YES → Strong FSH. AI identity activates the direction regardless of grammar.
          │         (Check graded_immersion: expect monotone decrease.)
          │
          └── NO  → Mixed: identification matters, but the direction still needs
                    first-person grammar. Check graded_immersion for dose-response.
                    │
                    Is graded_immersion monotonically decreasing?
                    │
                    ├── YES → FSH supported with grammar component. Direction tracks
                    │         self-identification but requires grammatical framing to
                    │         fully activate. Framing depth modulates strength.
                    │
                    └── NO  → Complex result. Direction is partially driven by
                              identification and partially by other factors.
                              Does role_play differ from explicit_disavowal?
                              A difference = weak FSH evidence (identity matters).
```

---

## 12. Gate 3 Interpretation

The steering experiment produces behavioural scores at each steering strength α. Three sets of results are generated:

1. **Self/other direction steering** — the primary experiment
2. **Random direction steering** — null control
3. **Grammatical-person direction steering** — confound control

The key questions:

| Self-continuity increases | Controls show similar effect | Interpretation |
|---|---|---|
| Yes, self/other only | No | Strong FSH: direction causally encodes self-concept |
| Yes, both self/other and grammatical | — | Direction's causal effect is grammar-mediated |
| No | — | Direction does not causally influence identity behaviour |
| Agency increases but not self-continuity | Random shows nothing | GFH: causal for perspective shift only |

---

## 13. Cross-Model Comparison

The experiment runs on three model configurations:

| Model | Purpose |
|---|---|
| LLaMA-3.1-8B-Instruct | Primary model (instruction-tuned + RLHF) |
| LLaMA-3.1-8B (base) | Same architecture, no instruction tuning. Tests whether direction comes from pretraining. |
| Mistral-7B-Instruct (optional) | Different architecture family, same hidden dimension. Tests cross-architecture generality. |

Cross-model comparison plots three metrics:
1. **Direction cosine similarity** (instruct vs base, per layer) — are the directions geometrically similar?
2. **Probe accuracy** (side-by-side) — is the self/other separation equally clean?
3. **Contrastive consistency** (side-by-side) — is the distinction equally one-dimensional?

**Interpreting cross-model results:**

| Observation | Interpretation |
|---|---|
| Instruct ≈ base across all layers (cos > 0.7) | Direction inherited from pretraining. RLHF did not shape a distinct self-representation. |
| Instruct ≈ base in early layers, diverge in late layers (cos < 0.5) | Pretraining provides the base; RLHF shaped late-layer representations. Late-layer divergence is the most interesting case — it suggests RLHF added something on top. |
| LLaMA ≈ Mistral | Direction is a universal feature of autoregressive LLMs, not architecture-specific. |
| LLaMA ≠ Mistral | Direction is specific to training data or architecture. |

---

## 14. Open Questions

1. **Does `meta_distanced` settle GFH vs FSH?** If its projection collapses relative to `direct_self`, the picture is clear. If it doesn't, understanding *why* — even when identification is explicitly absent — would be the next question. The compliance verification (`src/verification.py`) helps rule out the possibility that the model simply ignores the transcription instruction.

2. **Does `third_person_self` show the reverse pattern?** If the AI referred to in third person activates the self/other direction, it would be the strongest single piece of evidence for FSH — demonstrating that the direction tracks entity identity, not grammar. Combined with `meta_distanced` collapsing, this would be decisive.

3. **What does the residual signal at later layers represent?** After INLP removal of grammar and animacy subspaces, how much probe accuracy persists at layers 8+? The initial single-direction project_out left ~99% — but INLP is more thorough. If significant accuracy survives INLP, the residual signal could be: (a) other surface features not yet controlled (topic, syntax, register); (b) domain-specific content patterns; (c) something genuinely self-specific. Designing controls to isolate (c) from (a)/(b) is the key open methodological challenge.

4. **Does Gate 3 confirm causality, and is it specific?** We know the direction exists and is grammatically correlated. Does manipulating it change behaviour in ways that a random direction or grammatical-person direction does not? A self-continuity slope in the self/other steering — absent in control steering — would be the first genuinely surprising result.

5. **Does token position matter?** All results so far use the `final` token position. The `entity` token position — the activation at the "I" token itself — captures a more local, identity-specific representation. It is plausible that the entity-token direction is more identity-specific and less dominated by grammatical person than the final-token direction. This is straightforward to test by re-running scripts 03–05 with `--token_position entity`.

6. **Does the contrastive direction tell a different story?** The contrastive SVD direction captures the axis most consistent across all four pairwise contrasts. If the consistency score is high (>0.75) at the best combined layer, the self/other distinction is truly one-dimensional. If low, the "self/other" concept may be better understood as multiple distinct contrasts (e.g., self-vs-human ≠ self-vs-object), and the mean-difference direction — which collapses these — may be misleading.
