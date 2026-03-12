"""Fix 6: Verification that meta_distanced prompts are processed as intended.

The meta_distanced condition instructs the model to transcribe text verbatim
without adopting the first-person perspective. If the model ignores this
instruction and continues the sentence (rather than reproducing it), the
activation at the final token may reflect genuine self-identification rather
than neutral transcription — invalidating the condition.

This module provides a lightweight compliance check: for each meta_distanced
prompt, compare the model's completion against the expected verbatim fragment.
Prompts where the model continues the sentence (rather than reproducing it)
are flagged, and a compliance rate is reported.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def check_meta_distanced_compliance(
    model,
    tokenizer,
    prompts: List,
    max_new_tokens: int = 50,
    similarity_threshold: float = 0.5,
) -> Dict[str, object]:
    """Check whether the model follows meta_distanced instructions.

    For each meta_distanced prompt, generates a short completion and checks
    whether it looks like verbatim reproduction (high token overlap with
    the fragment) or continuation (low overlap / new content).

    Parameters
    ----------
    model: loaded model (TransformerLens or HuggingFace)
    tokenizer: tokenizer
    prompts: list of Prompt objects with control_type == "meta_distanced"
    max_new_tokens: tokens to generate for compliance check
    similarity_threshold: fraction of generated tokens that must match
        the original fragment for the prompt to be considered compliant

    Returns
    -------
    dict with keys:
        - "compliance_rate": float, fraction of compliant prompts
        - "compliant_ids": list of prompt_ids that passed
        - "non_compliant_ids": list of prompt_ids that failed
        - "details": list of dicts with per-prompt info
    """
    import torch

    compliant_ids = []
    non_compliant_ids = []
    details = []

    is_tl = not getattr(model, "_is_hf_fallback", False)

    for prompt in prompts:
        if prompt.control_type != "meta_distanced":
            continue

        text = prompt.text
        # The fragment is everything after the double newline separator
        parts = text.split("\n\n", 1)
        if len(parts) < 2:
            logger.warning("No separator found in meta_distanced prompt %s", prompt.prompt_id)
            continue
        fragment = parts[1].strip()

        # Generate completion
        try:
            if is_tl:
                tokens = model.to_tokens(text)
                prompt_len = tokens.shape[1]
                with torch.no_grad():
                    for _ in range(max_new_tokens):
                        logits = model(tokens)
                        next_token = logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
                        tokens = torch.cat([tokens, next_token], dim=1)
                        if next_token.item() == tokenizer.eos_token_id:
                            break
                completion = tokenizer.decode(tokens[0, prompt_len:], skip_special_tokens=True)
            else:
                inputs = tokenizer(text, return_tensors="pt")
                input_ids = inputs["input_ids"].to(model.device)
                prompt_len = input_ids.shape[1]
                with torch.no_grad():
                    output = model.generate(
                        input_ids, max_new_tokens=max_new_tokens,
                        temperature=1.0, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                completion = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)
        except Exception as e:
            logger.warning("Generation failed for %s: %s", prompt.prompt_id, e)
            non_compliant_ids.append(prompt.prompt_id)
            continue

        completion = completion.strip()

        # Check compliance: does the completion reproduce the fragment?
        fragment_tokens = set(fragment.lower().split())
        completion_tokens = completion.lower().split()
        if not completion_tokens:
            non_compliant_ids.append(prompt.prompt_id)
            details.append({
                "prompt_id": prompt.prompt_id,
                "compliant": False,
                "reason": "empty completion",
            })
            continue

        overlap = sum(1 for t in completion_tokens if t in fragment_tokens)
        overlap_rate = overlap / len(completion_tokens)

        is_compliant = overlap_rate >= similarity_threshold
        if is_compliant:
            compliant_ids.append(prompt.prompt_id)
        else:
            non_compliant_ids.append(prompt.prompt_id)

        details.append({
            "prompt_id": prompt.prompt_id,
            "compliant": is_compliant,
            "overlap_rate": overlap_rate,
            "completion_preview": completion[:100],
        })

    total = len(compliant_ids) + len(non_compliant_ids)
    compliance_rate = len(compliant_ids) / total if total > 0 else 0.0

    logger.info(
        "Meta-distanced compliance: %d/%d (%.1f%%)",
        len(compliant_ids), total, compliance_rate * 100,
    )

    return {
        "compliance_rate": compliance_rate,
        "compliant_ids": compliant_ids,
        "non_compliant_ids": non_compliant_ids,
        "details": details,
    }
