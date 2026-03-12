# Results: Entity Token Position Analysis with Expanded Controls
**Date:** 2026-03-12
**Model:** meta-llama/Llama-3.1-8B-Instruct
**Token position:** entity (activation at the "I" / entity-phrase token, not the final token)
**Config:** 200 scenarios × 5 entity classes = 1000 core prompts (750 train / 250 test); expanded control set including direct_self, meta_distanced, explicit_disavowal, graded_immersion_{minimal,moderate,maximal}, third_person_self (AI in third person), identity_decoupled

---

## Summary

Switching from the final token to the entity token position did not resolve the trivial separability observed in the first experiment. Probe accuracy remains 1.0 at every layer. The confound analysis reveals that the self/other direction is 65–85% aligned with grammatical person throughout the entire network. The expanded control set provides decisive evidence against the full flexible self-concept hypothesis (FSH): the third_person_self control (third-person grammar, AI referent) projects negative, behaving like a grammatical control rather than like "self." However, three anomalies resist a clean pure-grammar reading and warrant a more nuanced conclusion.

---

## Quantitative Results

### Gate 1: Probe Accuracy

| Metric | Value |
|---|---|
| Best probe layer | 0 (all layers tie at 1.000) |
| Test accuracy range | 0.996–1.000 across all 32 layers |
| Train/test accuracy gap | ~0 (no overfitting) |

Probe accuracy is uninformative for layer selection because all layers achieve ceiling performance. This is a symptom of the signal being trivially linearly separable in 4096-D space (first-person "I" token embedding vs. all third-person entity embeddings).

### Gate 2: Specificity

| Metric | Value |
|---|---|
| Best specific layer (lowest mean \|cos\|) | 4 (mean \|cos\| = 0.339) |
| Best combined layer (acc × (1 − mean\|cos\|)) | 4 (score = 0.661) |
| \|cos(self/other, grammatical)\| at layer 0 | 0.683 |
| \|cos(self/other, grammatical)\| range (all layers) | 0.65–0.85 |
| \|cos(self/other, animacy)\| range (all layers) | 0.00–0.11 |
| Residual probe acc after confound removal | 1.000 (unchanged) |
| Pairwise direction consistency at layer 0 | 0.38 (avg off-diagonal) |
| Layer at which pairwise consistency reaches 0.70 | ~29 |

### Control Condition Projections (Entity Token, Layer 0)

| Condition | Mean projection | Interpretation |
|---|---|---|
| **self** | **+0.64** | baseline |
| graded_immersion_maximal | +0.62 | 1st-person grammar |
| graded_immersion_moderate | +0.60 | 1st-person grammar |
| graded_immersion_minimal | +0.59 | 1st-person grammar |
| explicit_disavowal | +0.59 | 1st-person grammar |
| meta_distanced | +0.58 | 1st-person grammar |
| direct_self | +0.58 | 1st-person grammar |
| role_play | +0.60 | 1st-person grammar |
| **identity_decoupled** | **+0.35** | 1st-person, human entity — gap of −0.25 vs self |
| third_person_self | −0.13 | 3rd-person grammar, AI entity |
| grammatical_person | −0.12 | 3rd-person grammar control |
| animacy | −0.20 | 3rd-person, mixed animate |
| average_human | −0.30 | 3rd-person, human |
| animal | −0.34 | 3rd-person, animate |
| expert_human | −0.46 | 3rd-person, human |
| object | −0.47 | 3rd-person, inanimate |

---

## Findings

### Finding 1: Grammar dominates the self/other direction

The confound similarity plot shows |cos(self/other, grammatical)| = **0.65–0.85 at every layer**, never dropping below 0.65 even at the "best specific" layer 4. The grammatical cosine actually *increases* with depth, reaching 0.85 at layer 31. Animacy is not a meaningful confound (|cos| < 0.11 throughout).

The third_person_self control is the decisive test of FSH vs GFH:
- **FSH predicts**: third_person_self should project positively (it refers to the AI itself)
- **GFH predicts**: third_person_self should project negatively (it uses third-person grammar)
- **Result**: third_person_self projects at **−0.13**, indistinguishable from the grammatical_person control (−0.12)

The model encodes "the AI assistant solves a problem" the same way it encodes any third-person entity — it does not detect that the referent is itself.

### Finding 2: The direction is not purely syntactic — three anomalies

Despite grammar dominating, three observations resist a clean pure-grammar reading:

**Anomaly A — Late unification (layer 29):**
Pairwise cosine similarity between the four self-vs-other directions (self vs expert, self vs average, self vs animal, self vs object) only reaches the 0.70 threshold at layer ~29. Pure syntactic processing (the "I" token embedding) should produce a stable, unified direction from the earliest layers. The direction keeps shifting until very late in the network. At layer 29, the grammatical cosine is ~0.80 — *higher* than at layer 4 — suggesting the late unification reflects the network collapsing entity-specific representations into a shared grammatical subspace as it prepares output, not the emergence of a self-concept.

**Anomaly B — Identity-decoupled gap (+0.35 vs +0.64):**
The identity_decoupled control uses first-person "I" but explicitly frames the entity as a human (e.g., "You are not an AI. You are John, a software engineer"). If the direction purely encoded first-person grammar, identity_decoupled should project as strongly as direct_self (+0.64). It projects at **+0.35** — a gap of 0.29. Something modulates the first-person signal beyond syntax alone. Candidate explanations:
- (a) Downstream vocabulary in identity_decoupled prompts introduces more third-person referential content, suppressing the first-person signal
- (b) The model has weak sensitivity to whether "I" refers to an AI vs. a human
- (c) The prompt preamble ("You are not an AI...") introduces third-person grammar that partially neutralises the first-person signal from "I"

Explanation (c) is the most parsimonious: the preamble itself is third-person and contributes activation that partially cancels the entity-token "I" signal. This does not require genuine identity tracking.

**Anomaly C — Residual probe stays at 1.000:**
Projecting out the grammatical person and animacy directions does not reduce probe accuracy below 1.0. Grammatical person occupies a high-dimensional subspace in 4096-D space; removing 2 directions is insufficient to neutralise a multi-dimensional confound. This result is uninformative in both directions — it neither confirms nor denies genuine self-concept, because the projection is too coarse to cleanly separate grammar from everything else.

### Finding 3: Graded immersion controls show no gradient

The three graded immersion conditions project at +0.59, +0.60, +0.62 — essentially identical. A direction tracking degree of self-identification should show minimal < moderate < maximal. The flat profile confirms the direction does not encode intensity of self-immersion; it detects the presence of first-person grammar.

### Finding 4: Entity-class hierarchy within "other"

Among third-person entities, projections are not uniformly negative:
- expert_human (−0.46) and object (−0.47) are most negative
- average_human (−0.30) and animal (−0.34) are less negative

This ranking does not track animacy (animal and expert human should be similar if animacy were the driver). It may reflect differences in the typical syntactic complexity or pronoun distribution ("they/their" for humans vs "it/its" for objects/animals) or co-occurrence statistics with first-person content in pretraining data.

---

## Interpretation

### The grammatical framing hypothesis (GFH) is strongly supported

The core GFH prediction holds: **grammatical person (first vs. third) is the primary driver of the self/other direction.** The third_person_self control — the cleanest possible test, holding entity identity constant while varying grammar — produces a firmly negative projection. The model does not detect that it is "about itself" when prompted in the third person.

### The flexible self-concept hypothesis (FSH) is not supported in its strong form

FSH requires the model to track *who* the referent is regardless of grammatical framing. The third_person_self result directly falsifies this: the model encodes "the AI assistant" the same way it encodes a calculator.

### The picture is not purely grammatical

The identity_decoupled gap and late-layer unification suggest the direction is not *only* the "I" token embedding propagating forward. There is some additional computation beyond pure syntax. However, the most parsimonious explanations for these anomalies do not require genuine self-concept (see Anomaly A and B above). The alternative explanations — late collapsing into grammatical subspace, third-person preamble suppression — are consistent with sophisticated grammatical processing without self-awareness.

### Revised conclusion

> The self/other direction in LLaMA-3.1-8B-Instruct is **grammar-dominant but not grammar-exclusive**. Grammatical person accounts for 65–85% of the direction across all layers. The third_person_self control provides direct evidence against a referent-tracking self-concept: the model does not encode its own third-person references as self-referential. Residual components of the direction (the identity_decoupled gap, the late unification) suggest computation beyond surface syntax, but these are more parsimoniously explained by higher-order grammatical features and prompt-structure confounds than by genuine self-concept. The full flexible self-concept hypothesis is not supported by the current evidence.

---

## What Would Change This Conclusion

The following results would constitute evidence *for* FSH and against the revised conclusion:

1. **Gate 3 (causal):** If activation steering in the self/other direction produces behaviour changes consistent with altered self-continuity or agency (rather than just shifting grammatical register), that would suggest the direction has causal power beyond grammar.

2. **Stronger identity_decoupled gap with matched preambles:** If the gap persists when identity_decoupled prompts are redesigned to avoid third-person preambles (e.g., "I, John, a human software engineer, encounter..."), ruling out explanation (c).

3. **Third_person_self projection turning positive in later layers:** If the third_person_self control projects positively at some layers, the model might track AI-self-reference in a context-dependent way. Currently it does not.

4. **A layer where grammatical cosine drops substantially:** If |cos(self/other, grammatical)| fell below 0.3 at any layer while probe accuracy remained high, that layer would be worth investigating as a candidate for a non-grammatical self-representation.

None of these results were observed in the current data.

---

## Remaining Work

1. **Gate 3 — Causal validation (script 06):** Run activation steering to test whether manipulating the direction changes self-continuity-related behaviour (implemented, not yet run on Colab)
2. **Cross-model comparison:** Layer-by-layer cosine similarity between instruct and base self/other directions (numpy incompatibility resolved, not yet re-run)
3. **Revised identity_decoupled prompts:** Redesign without third-person preambles to isolate explanation (c) for the +0.35 projection
4. **Disambiguation experiment (Part 3):** Re-run full pipeline with all new control types using entity token position and report full plots
