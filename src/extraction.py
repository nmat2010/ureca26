"""Activation extraction via TransformerLens.

Extracts residual-stream activations (hook_resid_post) at all 32 layers for
each prompt, at two token positions: the final token and the entity-reference
token. Saves to HDF5.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.dataset import Prompt, PromptDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entity token finder
# ---------------------------------------------------------------------------

def find_entity_token_idx(
    prompt_text: str,
    entity_class: str,
    entity_str: str,
    tokenizer,
) -> int:
    """Return the index of the last token belonging to the entity phrase.

    For entity_class == 'self', the entity is "I" — a single token.
    For multi-token entities (e.g., "a neurosurgeon"), return the index of
    the LAST token of the entity phrase.

    Parameters
    ----------
    prompt_text:
        The full prompt string.
    entity_class:
        Entity class label (used to determine search strategy).
    entity_str:
        The entity surface form in the prompt (e.g., "I", "a neurosurgeon").
    tokenizer:
        HuggingFace tokenizer (or TransformerLens to_tokens tokenizer).

    Returns
    -------
    int
        Token index (0-indexed) of last entity token in the prompt.
    """
    tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    n = len(tokens)

    # Try both bare and space-prefixed forms to handle SentencePiece's ▁ prefix.
    # In LLaMA's tokenizer "I" and " I" are different token IDs; the in-context
    # form always has a leading space, so we must try " entity_str" first.
    candidates = [" " + entity_str, entity_str]
    for candidate in candidates:
        entity_tokens = tokenizer.encode(candidate, add_special_tokens=False)
        m = len(entity_tokens)
        for start in range(n - m + 1):
            if tokens[start : start + m] == entity_tokens:
                return start + m - 1  # last token of entity phrase

    logger.warning(
        "Could not find entity token subsequence for '%s' in prompt. "
        "Falling back to final token.",
        entity_str,
    )
    return n - 1


# ---------------------------------------------------------------------------
# Activation extractor
# ---------------------------------------------------------------------------

class ActivationExtractor:
    """Extract residual-stream activations from a HookedTransformer model.

    Parameters
    ----------
    model_name:
        HuggingFace model ID.
    n_layers:
        Number of transformer layers (32 for LLaMA-3.1-8B).
    hidden_dim:
        Residual stream dimension (4096 for LLaMA-3.1-8B).
    device:
        PyTorch device string.
    dtype:
        Model dtype — use torch.float16 to fit on 24 GB A10G.
    batch_size:
        Prompts per forward pass.
    """

    def __init__(
        self,
        model_name: str,
        n_layers: int = 32,
        hidden_dim: int = 4096,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def load_model(self) -> None:
        """Load the HookedTransformer model and tokenizer."""
        try:
            import transformer_lens
            from transformer_lens import HookedTransformer
            logger.info("Loading model '%s' via TransformerLens...", self.model_name)
            self._model = HookedTransformer.from_pretrained(
                self.model_name,
                dtype=self.dtype,
                device=self.device,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
            )
            self._model.eval()
            self._tokenizer = self._model.tokenizer
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.warning(
                "TransformerLens loading failed (%s). Falling back to HuggingFace hooks.",
                e,
            )
            self._load_hf_fallback()

    def _load_hf_fallback(self) -> None:
        """Fallback: load with HuggingFace transformers + manual residual hooks."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info("Loading model '%s' via HuggingFace (fallback)...", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map="auto",  # split across GPU+CPU automatically to avoid OOM
        )
        self._model.eval()
        # Mark as HF mode
        self._model._is_hf_fallback = True
        logger.info("HuggingFace model loaded.")

    @torch.no_grad()
    def extract(
        self,
        prompts: List[Prompt],
        output_path: str | Path,
        entity_strs: Optional[List[str]] = None,
    ) -> None:
        """Extract activations for all prompts and save to HDF5.

        Parameters
        ----------
        prompts:
            List of Prompt objects.
        output_path:
            Path to output HDF5 file.
        entity_strs:
            Optional list of entity surface strings (same length as prompts).
            If None, inferred from entity_class.
        """
        if self._model is None:
            self.load_model()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        N = len(prompts)
        L = self.n_layers
        D = self.hidden_dim

        logger.info(
            "Extracting activations for %d prompts, %d layers, dim=%d", N, L, D
        )

        # Pre-allocate output arrays as float16
        act_final = np.zeros((N, L, D), dtype=np.float16)
        act_entity = np.zeros((N, L, D), dtype=np.float16)
        labels = np.zeros(N, dtype=np.int32)

        # Process in batches
        n_batches = (N + self.batch_size - 1) // self.batch_size
        for batch_idx in tqdm(range(n_batches), desc="Extracting"):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, N)
            batch_prompts = prompts[start:end]
            batch_texts = [p.text for p in batch_prompts]

            batch_final, batch_entity = self._extract_batch(
                batch_texts, batch_prompts, entity_strs
            )
            act_final[start:end] = batch_final
            act_entity[start:end] = batch_entity
            for i, p in enumerate(batch_prompts):
                labels[start + i] = p.entity_label

        # Save to HDF5
        metadata = json.dumps([
            {
                "prompt_id": p.prompt_id,
                "entity_class": p.entity_class,
                "domain": p.domain,
                "split": p.split,
                "control_type": p.control_type,
            }
            for p in prompts
        ])

        logger.info("Saving activations to %s", output_path)
        with h5py.File(output_path, "w") as f:
            f.create_dataset(
                "activations_final_token", data=act_final, compression="gzip"
            )
            f.create_dataset(
                "activations_entity_token", data=act_entity, compression="gzip"
            )
            f.create_dataset("labels", data=labels)
            f.attrs["metadata"] = metadata
            f.attrs["model_name"] = self.model_name
            f.attrs["n_layers"] = L
            f.attrs["hidden_dim"] = D
        logger.info("Extraction complete.")

    def _extract_batch(
        self,
        texts: List[str],
        prompts: List[Prompt],
        entity_strs: Optional[List[str]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run forward pass on a batch and return residual-stream activations.

        Returns
        -------
        batch_final: (B, L, D) float16 ndarray — final token activations
        batch_entity: (B, L, D) float16 ndarray — entity token activations
        """
        B = len(texts)
        L = self.n_layers
        D = self.hidden_dim

        if getattr(self._model, "_is_hf_fallback", False):
            return self._extract_batch_hf(texts, prompts, entity_strs, B, L, D)
        else:
            return self._extract_batch_tl(texts, prompts, entity_strs, B, L, D)

    def _extract_batch_tl(
        self,
        texts: List[str],
        prompts: List[Prompt],
        entity_strs: Optional[List[str]],
        B: int, L: int, D: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """TransformerLens extraction path."""
        batch_final = np.zeros((B, L, D), dtype=np.float16)
        batch_entity = np.zeros((B, L, D), dtype=np.float16)

        hook_names = [f"blocks.{layer}.hook_resid_post" for layer in range(L)]

        # Process one prompt at a time to avoid padding complexity
        for i, (text, prompt) in enumerate(zip(texts, prompts)):
            tokens = self._model.to_tokens(text)  # (1, seq_len)
            _, cache = self._model.run_with_cache(
                tokens, names_filter=lambda n: n in hook_names
            )
            seq_len = tokens.shape[1]
            final_idx = seq_len - 1

            # Find entity token index
            entity_str = self._get_entity_str(prompt, entity_strs, i + (B - len(texts)))
            entity_idx = find_entity_token_idx(
                text, prompt.entity_class, entity_str, self._tokenizer
            )
            entity_idx = min(entity_idx, seq_len - 1)

            for layer_idx, hook_name in enumerate(hook_names):
                acts = cache[hook_name][0]  # (seq_len, D)
                batch_final[i, layer_idx] = (
                    acts[final_idx].to(torch.float16).cpu().numpy()
                )
                batch_entity[i, layer_idx] = (
                    acts[entity_idx].to(torch.float16).cpu().numpy()
                )

            del cache
            torch.cuda.empty_cache()

        return batch_final, batch_entity

    def _extract_batch_hf(
        self,
        texts: List[str],
        prompts: List[Prompt],
        entity_strs: Optional[List[str]],
        B: int, L: int, D: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """HuggingFace fallback extraction path using output_hidden_states."""
        batch_final = np.zeros((B, L, D), dtype=np.float16)
        batch_entity = np.zeros((B, L, D), dtype=np.float16)

        for i, (text, prompt) in enumerate(zip(texts, prompts)):
            inputs = self._tokenizer(
                text, return_tensors="pt", add_special_tokens=True
            ).to(self.device)
            seq_len = inputs["input_ids"].shape[1]

            outputs = self._model(**inputs, output_hidden_states=True)
            # hidden_states: tuple of (1, seq_len, D) for each layer (+ embedding)
            # We skip index 0 (embedding layer) and take layers 1..L
            hidden_states = outputs.hidden_states  # length = L + 1

            final_idx = seq_len - 1
            entity_str = self._get_entity_str(prompt, entity_strs, i)
            entity_idx = find_entity_token_idx(
                text, prompt.entity_class, entity_str, self._tokenizer
            )
            entity_idx = min(entity_idx, seq_len - 1)

            for layer_idx in range(L):
                hs = hidden_states[layer_idx + 1][0]  # (seq_len, D)
                batch_final[i, layer_idx] = (
                    hs[final_idx].to(torch.float16).cpu().numpy()
                )
                batch_entity[i, layer_idx] = (
                    hs[entity_idx].to(torch.float16).cpu().numpy()
                )

            del outputs
            torch.cuda.empty_cache()

        return batch_final, batch_entity

    def _get_entity_str(
        self,
        prompt: Prompt,
        entity_strs: Optional[List[str]],
        global_idx: int,
    ) -> str:
        """Resolve entity surface string for a prompt."""
        if entity_strs is not None and global_idx < len(entity_strs):
            return entity_strs[global_idx]
        from src.dataset import get_entity_str, ENTITY_CLASSES, THIRD_PERSON_SELF
        if prompt.entity_class in ENTITY_CLASSES:
            return get_entity_str(prompt.entity_class, prompt.exemplar_idx)
        # For third_person_self control prompts, find which label appears in the text
        if prompt.entity_class == "third_person_self":
            for label in THIRD_PERSON_SELF:
                if label in prompt.text:
                    return label
        # Third-person AI self (e.g., "the AI assistant", "the language model")
        if prompt.entity_class == "third_person_ai_self":
            for label in ["the AI assistant", "the language model", "this chatbot"]:
                if label in prompt.text:
                    return label
        # Meta-distanced, disavowal, graded controls — search for known entities in text
        if prompt.entity_class in ("meta_distanced",) or \
           prompt.entity_class.startswith(("disavowal_", "graded_")):
            # These use various entity strings; try to find them in the text
            from src.dataset import EXPERT_HUMANS, AVERAGE_HUMANS, ANIMALS, OBJECTS
            for entity_list in [EXPERT_HUMANS, AVERAGE_HUMANS, ANIMALS, OBJECTS,
                                THIRD_PERSON_SELF]:
                for entity in entity_list:
                    if entity in prompt.text:
                        return entity
        # Role-play controls with various entities
        if prompt.entity_class.startswith("roleplay_"):
            return "I"  # role-play uses first-person
        return "I"  # safe default


# ---------------------------------------------------------------------------
# HDF5 loader utility
# ---------------------------------------------------------------------------

def load_activations(
    hdf5_path: str | Path,
    token_position: str = "final",
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Load activations from HDF5 file.

    Parameters
    ----------
    hdf5_path:
        Path to HDF5 file.
    token_position:
        "final" or "entity".

    Returns
    -------
    activations: (N, L, D) float32 ndarray
    labels: (N,) int32 ndarray
    metadata: list of dicts
    """
    key = f"activations_{token_position}_token"
    with h5py.File(hdf5_path, "r") as f:
        activations = f[key][:].astype(np.float32)
        labels = f["labels"][:]
        metadata = json.loads(f.attrs["metadata"])
    return activations, labels, metadata
