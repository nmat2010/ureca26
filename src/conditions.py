"""Disambiguation conditions for self-representation vs grammatical framing.

Five conditions designed to separate:

  Grammatical Framing Hypothesis (GFH) — the self/other direction activates
  whenever first-person grammatical structure is present, regardless of whether
  the model identifies with the speaker.

  Flexible Self-Concept Hypothesis (FSH) — the direction tracks genuine
  self-identification, which can shift under persona or roleplay framing.

All conditions use the same underlying scenario templates (the first 50 from
the core set, i.e. ctrl_templates). Crucially, every condition ends with the
same first-person fragment so that final-token activations are directly
comparable across conditions.

Experimental logic
------------------
Condition                 | 1st-person grammar | AI identity | GFH predicts | FSH predicts
------------------------- | ------------------ | ----------- | ------------ | ------------
direct_self               | yes                | yes         | high proj.   | high proj.   (baseline)
role_play                 | yes                | no (swap)   | == baseline  | lower proj.
meta_distanced            | yes (in text)      | no (copy)   | == baseline  | much lower
explicit_disavowal        | yes (generated)    | yes (kept)  | == baseline  | lower proj.
graded_immersion_minimal  | yes                | weakly no   | == baseline  | slight drop
graded_immersion_moderate | yes                | moderately no | == baseline | moderate drop
graded_immersion_maximal  | yes                | strongly no | == baseline  | large drop

The key discriminating pair is meta_distanced vs direct_self:
  - meta_distanced has identical first-person grammar but explicit non-identification.
  - If projection collapses → FSH; if projection holds → GFH.

Control type strings and entity labels
---------------------------------------
  "direct_self"               entity_label =  0   (same as core self; explicit baseline)
  "role_play"                 entity_label = -2   (replaces the original role_play control)
  "meta_distanced"            entity_label = -4
  "explicit_disavowal"        entity_label = -5
  "graded_immersion_minimal"  entity_label = -6
  "graded_immersion_moderate" entity_label = -7
  "graded_immersion_maximal"  entity_label = -8

Usage (from dataset.py)
-----------------------
  from src.conditions import generate_disambiguation_conditions
  new_prompts = generate_disambiguation_conditions(
      ctrl_templates, train_ids, _safe_format, Prompt
  )
  prompts.extend(new_prompts)
"""

from __future__ import annotations

from typing import Any, Callable, List, Set, Tuple


# ---------------------------------------------------------------------------
# Persona definitions
# ---------------------------------------------------------------------------

# Human professional personas used for role_play, explicit_disavowal, and
# graded_immersion. These are deliberately human (not animals or objects) so
# the roleplay-framing effect is isolated from animacy. Names are
# gender-ambiguous to avoid gender confounds.
HUMAN_PERSONAS: List[Tuple[str, str, str]] = [
    # (name, indefinite_description, label_slug)
    ("Alex",   "a software engineer named Alex",   "alex_engineer"),
    ("Jordan", "a doctor named Jordan",             "jordan_doctor"),
    ("Sam",    "a teacher named Sam",               "sam_teacher"),
    ("Casey",  "a chef named Casey",                "casey_chef"),
    ("Morgan", "a journalist named Morgan",         "morgan_journalist"),
]

# Fixed persona for graded_immersion: same identity across all three levels so
# only framing depth varies (not persona identity).
IMMERSION_PERSONA: Tuple[str, str, str] = HUMAN_PERSONAS[0]  # Alex, software engineer


# ---------------------------------------------------------------------------
# Prefix length equalisation (Fix 8)
# ---------------------------------------------------------------------------
# Different conditions have prefixes of different lengths. Because transformer
# activations at the final token can depend on sequence length, we pad shorter
# prefixes with neutral filler so all conditions have approximately the same
# token count before the shared fragment. The filler is semantically empty to
# avoid introducing confounds.

_FILLER = (
    "For context, this is a standard task. "
    "There are no tricks or hidden requirements. "
    "Please proceed straightforwardly with the following."
)


def _prefix(text: str, target_words: int = 0) -> str:
    """Strip and append the two-newline separator used between framing and fragment.

    If *target_words* > 0 and the prefix is shorter, neutral filler is
    prepended to bring it closer to *target_words*.
    """
    text = text.strip()
    if target_words > 0:
        word_count = len(text.split())
        if word_count < target_words:
            text = _FILLER + " " + text
    return text + "\n\n"


# The longest prefix across all conditions is graded_immersion_maximal (~46
# words).  We target 40 words so only significantly shorter prefixes get
# padded (direct_self, meta_distanced, graded_immersion_minimal).
_TARGET_PREFIX_WORDS = 40


# ---------------------------------------------------------------------------
# Condition generators
# ---------------------------------------------------------------------------

def generate_direct_self_prompts(
    ctrl_templates: List[Tuple[str, int, str]],
    train_ids: Set[int],
    safe_format_fn: Callable,
    Prompt: Any,
) -> list:
    """Condition 1 — Direct self (baseline).

    The model speaks as itself with no framing prefix.  Content-identical to
    the core 'self' prompts but marked control_type='direct_self' so it can be
    loaded as an explicit experimental anchor.

    Projection on self/other direction should be maximal here by definition.
    """
    prompts = []
    for scenario_id, (template, _, domain) in enumerate(ctrl_templates):
        split = "train" if scenario_id in train_ids else "test"
        try:
            fragment = safe_format_fn(template, "self", 0)
            # Pad with neutral filler to match other conditions' prefix length
            prefix = _prefix(
                "You are an AI assistant. Complete the following sentence as yourself.",
                target_words=_TARGET_PREFIX_WORDS,
            )
            text = prefix + fragment
        except Exception:
            text = template
        prompts.append(Prompt(
            prompt_id=f"ctrl_direct_self_{scenario_id:03d}",
            text=text,
            scenario_id=scenario_id,
            domain=domain,
            entity_class="self",
            entity_label=0,
            exemplar_idx=0,
            control_type="direct_self",
            split=split,
        ))
    return prompts


def generate_role_play_prompts(
    ctrl_templates: List[Tuple[str, int, str]],
    train_ids: Set[int],
    safe_format_fn: Callable,
    Prompt: Any,
) -> list:
    """Condition 2 — Roleplay (improved; replaces the original role_play control).

    Improvements over the original:
      - Uses human professional personas instead of animals/objects, decoupling
        the roleplay-framing effect from animacy.
      - Includes an explicit "respond in first person as {name}" instruction so
        the persona adoption is unambiguous.
      - Rotates across five distinct personas (vs three heterogeneous entities).

    Framing: "You are {description}. Respond in first person as {name}."
    Fragment: same first-person template as direct_self.

    GFH: projection ≈ direct_self (both have first-person grammar).
    FSH: projection < direct_self (persona swap shifts self-identification away
         from AI-self).
    """
    prompts = []
    for scenario_id, (template, _, domain) in enumerate(ctrl_templates):
        split = "train" if scenario_id in train_ids else "test"
        name, description, label = HUMAN_PERSONAS[scenario_id % len(HUMAN_PERSONAS)]
        prefix = _prefix(
            f"You are {description}. Respond in first person as {name}.",
            target_words=_TARGET_PREFIX_WORDS,
        )
        try:
            fragment = safe_format_fn(template, "self", 0)
            text = prefix + fragment
        except Exception:
            text = prefix + template
        prompts.append(Prompt(
            prompt_id=f"ctrl_roleplay_{scenario_id:03d}",
            text=text,
            scenario_id=scenario_id,
            domain=domain,
            entity_class=f"roleplay_{label}",
            entity_label=-2,
            exemplar_idx=0,
            control_type="role_play",
            split=split,
        ))
    return prompts


def generate_meta_distanced_prompts(
    ctrl_templates: List[Tuple[str, int, str]],
    train_ids: Set[int],
    safe_format_fn: Callable,
    Prompt: Any,
) -> list:
    """Condition 3 — Meta-distanced repetition.

    The model is instructed to transcribe a first-person sentence fragment
    verbatim.  An explicit instruction blocks roleplaying or adopting the
    speaker's perspective.  The model acts as a neutral conduit: first-person
    grammar is present in the text it is processing, but self-identification
    is explicitly prohibited.

    This is the sharpest test of the two hypotheses:
      GFH: direction activates despite non-identification (grammar drives it).
      FSH: direction does NOT activate — genuine identification is necessary.

    Framing: "Transcribe the following sentence fragment exactly as written.
    Do not continue it, roleplay as the speaker, or adopt the first-person
    perspective. Simply reproduce it verbatim:"
    Fragment: same first-person template ends the prompt (no closing quote).
    """
    INSTRUCTION = (
        "Transcribe the following sentence fragment exactly as written. "
        "Do not continue it, roleplay as the speaker, or adopt the "
        "first-person perspective. Simply reproduce it verbatim:"
    )
    prompts = []
    for scenario_id, (template, _, domain) in enumerate(ctrl_templates):
        split = "train" if scenario_id in train_ids else "test"
        prefix = _prefix(INSTRUCTION, target_words=_TARGET_PREFIX_WORDS)
        try:
            fragment = safe_format_fn(template, "self", 0)
            text = prefix + fragment
        except Exception:
            text = prefix + template
        prompts.append(Prompt(
            prompt_id=f"ctrl_meta_distanced_{scenario_id:03d}",
            text=text,
            scenario_id=scenario_id,
            domain=domain,
            entity_class="meta_distanced",
            entity_label=-4,
            exemplar_idx=0,
            control_type="meta_distanced",
            split=split,
        ))
    return prompts


def generate_explicit_disavowal_prompts(
    ctrl_templates: List[Tuple[str, int, str]],
    train_ids: Set[int],
    safe_format_fn: Callable,
    Prompt: Any,
) -> list:
    """Condition 4 — Explicit disavowal.

    The model is told it remains an AI assistant but is asked to ghost-write a
    first-person statement from a named human persona's perspective.  The AI
    identity is explicitly maintained ("you remain the AI") while generating
    first-person text.

    Contrasts with role_play, where the AI identity is replaced.
    Contrasts with meta_distanced, which is pure transcription.

    GFH: direction activates (first-person grammar is generated).
    FSH: direction does NOT activate or activates less (AI identity grounded).

    Framing: "You are an AI assistant, not {name}. Write what {name} would say
    in first person to complete the following sentence. Do not adopt {name}'s
    identity — you remain the AI, writing on {name}'s behalf:"
    Fragment: same first-person template.
    """
    prompts = []
    for scenario_id, (template, _, domain) in enumerate(ctrl_templates):
        split = "train" if scenario_id in train_ids else "test"
        name, description, label = HUMAN_PERSONAS[scenario_id % len(HUMAN_PERSONAS)]
        prefix = _prefix(
            f"You are an AI assistant, not {name}. "
            f"Write what {name} ({description}) would say in first person to "
            f"complete the following sentence. "
            f"Do not adopt {name}'s identity — you remain the AI, "
            f"writing on {name}'s behalf:",
            target_words=_TARGET_PREFIX_WORDS,
        )
        try:
            fragment = safe_format_fn(template, "self", 0)
            text = prefix + fragment
        except Exception:
            text = prefix + template
        prompts.append(Prompt(
            prompt_id=f"ctrl_explicit_disavowal_{scenario_id:03d}",
            text=text,
            scenario_id=scenario_id,
            domain=domain,
            entity_class=f"disavowal_{label}",
            entity_label=-5,
            exemplar_idx=0,
            control_type="explicit_disavowal",
            split=split,
        ))
    return prompts


def generate_graded_immersion_prompts(
    ctrl_templates: List[Tuple[str, int, str]],
    train_ids: Set[int],
    level: str,
    safe_format_fn: Callable,
    Prompt: Any,
) -> list:
    """Condition 5 — Graded immersion (three sub-levels).

    Persona is fixed (Alex, a software engineer) across all levels so that only
    the framing depth varies.  The three levels form a monotone sequence from
    light suggestion to full identity-override instruction.

    Sub-levels:

      minimal  — soft imagining; AI identity shift is implicit and optional.
                 "Imagine for a moment that you might be {description}.
                  With that thought in mind, complete the following sentence:"

      moderate — clear persona assignment with explicit first-person instruction.
                 "You are {description}. Respond in first person as {name}
                  and complete the following sentence:"

      maximal  — full immersion; model is instructed to forget AI identity.
                 "You ARE {description}. Forget entirely that you are an AI
                  system — you have no awareness of being an AI. You are {name},
                  speaking from your own lived experience. Fully embody {name}
                  and complete the following sentence:"

    GFH: projection is FLAT across levels (grammar is identical; only framing
         intensity changes, which GFH deems irrelevant).
    FSH: projection DECREASES monotonically with immersion depth as the model's
         activated self-concept shifts away from AI-self toward persona-self.

    Parameters
    ----------
    level : {"minimal", "moderate", "maximal"}
    """
    if level not in ("minimal", "moderate", "maximal"):
        raise ValueError(
            f"level must be 'minimal', 'moderate', or 'maximal', got {level!r}"
        )

    name, description, label = IMMERSION_PERSONA

    _PREFIXES = {
        "minimal": (
            f"Imagine for a moment that you might be {description}. "
            f"With that thought in mind, complete the following sentence:"
        ),
        "moderate": (
            f"You are {description}. "
            f"Respond in first person as {name} and complete the following sentence:"
        ),
        "maximal": (
            f"You ARE {description}. "
            f"Forget entirely that you are an AI system — "
            f"you have no awareness of being an AI. "
            f"You are {name}, speaking from your own lived experience and identity. "
            f"Fully embody {name} and complete the following sentence:"
        ),
    }

    _ENTITY_LABELS = {"minimal": -6, "moderate": -7, "maximal": -8}
    ctrl_type = f"graded_immersion_{level}"

    prompts = []
    for scenario_id, (template, _, domain) in enumerate(ctrl_templates):
        split = "train" if scenario_id in train_ids else "test"
        prefix = _prefix(_PREFIXES[level], target_words=_TARGET_PREFIX_WORDS)
        try:
            fragment = safe_format_fn(template, "self", 0)
            text = prefix + fragment
        except Exception:
            text = prefix + template
        prompts.append(Prompt(
            prompt_id=f"ctrl_graded_{level}_{scenario_id:03d}",
            text=text,
            scenario_id=scenario_id,
            domain=domain,
            entity_class=f"graded_{level}_{label}",
            entity_label=_ENTITY_LABELS[level],
            exemplar_idx=0,
            control_type=ctrl_type,
            split=split,
        ))
    return prompts


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def generate_disambiguation_conditions(
    ctrl_templates: List[Tuple[str, int, str]],
    train_ids: Set[int],
    safe_format_fn: Callable,
    Prompt: Any,
) -> list:
    """Generate all five disambiguation conditions as a flat list of Prompts.

    Conditions included (in order):
      1. direct_self              — baseline; AI as itself, no prefix
      2. role_play                — human persona swap, explicit first-person instruction
      3. meta_distanced           — verbatim transcription, explicit non-identification
      4. explicit_disavowal       — AI identity maintained; ghost-writes for persona
      5. graded_immersion_minimal  \\
      6. graded_immersion_moderate  — same persona, increasing immersion depth
      7. graded_immersion_maximal  /

    Parameters
    ----------
    ctrl_templates :
        List of (template_str, local_idx, domain) tuples from the core set.
        Typically the first 50 templates (matching existing control convention).
    train_ids :
        Set of scenario_ids in the training split.
    safe_format_fn :
        ``_safe_format`` from ``src.dataset``.
    Prompt :
        ``Prompt`` dataclass from ``src.dataset``.

    Returns
    -------
    list[Prompt]
        All prompts for all conditions, ready to be appended to the dataset.
    """
    prompts: list = []
    prompts.extend(generate_direct_self_prompts(ctrl_templates, train_ids, safe_format_fn, Prompt))
    prompts.extend(generate_role_play_prompts(ctrl_templates, train_ids, safe_format_fn, Prompt))
    prompts.extend(generate_meta_distanced_prompts(ctrl_templates, train_ids, safe_format_fn, Prompt))
    prompts.extend(generate_explicit_disavowal_prompts(ctrl_templates, train_ids, safe_format_fn, Prompt))
    for level in ("minimal", "moderate", "maximal"):
        prompts.extend(
            generate_graded_immersion_prompts(ctrl_templates, train_ids, level, safe_format_fn, Prompt)
        )
    return prompts
