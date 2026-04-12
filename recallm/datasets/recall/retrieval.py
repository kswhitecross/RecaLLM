"""
Retrieval dataset for final training: key-value lookup with augmentations.

Augmentations:
  - KV format variation: lines, json, csv (sampled per example)
  - Instruction variation: multiple prompt templates per format (sampled per example)
  - Question position variation: end, beginning, both (sampled per example)

"""

import json
import math
import os
import random
import string
from typing import Any

import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from recallm.datasets.base import (
    TokenizeableExample,
    insert_into_list,
    normalize_context_range,
    sample_target_context,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rand_alphanumeric_str(length: int, rng: random.Random) -> str:
    """Generate a random alphanumeric string of given length."""
    chars = string.digits + string.ascii_letters
    return "".join(rng.choice(chars) for _ in range(length))


DEFAULT_QUESTION_POSITION_WEIGHTS = {"end": 0.6, "beginning": 0.2, "both": 0.2}


def _make_question_str(key: int) -> str:
    return f"Your key: {key}"


def _fill_question_placeholders(
    prompt_template: str,
    key: int,
    position: str,
) -> str:
    """Fill {question_start} and {question_end} based on position strategy."""
    q = _make_question_str(key)
    if position == "end":
        return prompt_template.replace("{question_start}", "").replace("{question_end}", q)
    elif position == "beginning":
        return prompt_template.replace("{question_start}", q + "\n\n").replace("{question_end}", "")
    elif position == "both":
        return prompt_template.replace("{question_start}", q + "\n\n").replace("{question_end}", q)
    else:
        raise ValueError(f"Unknown question position: {position}")


# ---------------------------------------------------------------------------
# KV formatting functions
# ---------------------------------------------------------------------------

def format_kv_lines(keys: list[int], vals: list[str]) -> str:
    """Format KV pairs as line-separated entries: 'Key {k}:\\n{v}'."""
    entries = [f"Key {k}:\n{v}" for k, v in zip(keys, vals)]
    return "\n\n".join(entries)


def format_kv_json(keys: list[int], vals: list[str]) -> str:
    """Format KV pairs as a JSON object."""
    d = {str(k): v for k, v in zip(keys, vals)}
    return json.dumps(d, indent=2)


def format_kv_csv(keys: list[int], vals: list[str]) -> str:
    """Format KV pairs as a CSV table with header."""
    rows = ["key,value"]
    for k, v in zip(keys, vals):
        rows.append(f"{k},{v}")
    return "\n".join(rows)


KV_FORMATTERS = {
    "lines": format_kv_lines,
    "json": format_kv_json,
    "csv": format_kv_csv,
}


# ---------------------------------------------------------------------------
# TokenizeableRetrievalExample
# ---------------------------------------------------------------------------

class TokenizeableRetrievalExample(TokenizeableExample):
    """A single retrieval example that can be tokenized and fitted to a target context length."""

    def __init__(
        self,
        pos_key: int,
        pos_val: str,
        needle_depth: float,
        neg_keys: list[int],
        neg_vals: list[str],
        system_prompt: str,
        prompt_template: str,
        kv_format: str,
        question_position: str,
    ):
        super().__init__(max_k=len(neg_keys))
        self.pos_key = pos_key
        self.pos_val = pos_val
        self.needle_depth = needle_depth
        self.neg_keys = neg_keys
        self.neg_vals = neg_vals
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.kv_format = kv_format
        self.question_position = question_position
        self._formatter = KV_FORMATTERS[kv_format]

    def average_length_per_item(self, tokenizer: PreTrainedTokenizerBase) -> float:
        # Estimate average tokens per KV pair by sampling a few
        sample_size = min(20, len(self.neg_keys))
        sample_strs = []
        for i in range(sample_size):
            if self.kv_format == "lines":
                sample_strs.append(f"Key {self.neg_keys[i]}:\n{self.neg_vals[i]}\n\n")
            elif self.kv_format == "json":
                sample_strs.append(f'  "{self.neg_keys[i]}": "{self.neg_vals[i]}",\n')
            elif self.kv_format == "csv":
                sample_strs.append(f"{self.neg_keys[i]},{self.neg_vals[i]}\n")
        encodings = tokenizer(sample_strs, add_special_tokens=False)
        lengths = [len(ids) for ids in encodings["input_ids"]]
        return float(np.mean(lengths))

    def _build_ordered_keys_vals(self) -> tuple[list[int], list[str]]:
        """Return all keys and vals in order, with pos inserted at needle_depth."""
        neg_k = list(self.neg_keys[: self.k])
        neg_v = list(self.neg_vals[: self.k])
        # Insert positive at the specified depth
        ordered_k, _ = insert_into_list(neg_k, [self.pos_key], [self.needle_depth])
        ordered_v, _ = insert_into_list(neg_v, [self.pos_val], [self.needle_depth])
        return ordered_k, ordered_v

    def build_messages(self) -> list[dict[str, str]]:
        ordered_keys, ordered_vals = self._build_ordered_keys_vals()
        documents = self._formatter(ordered_keys, ordered_vals)

        user_prompt = self.prompt_template.replace("{documents}", documents)
        user_prompt = _fill_question_placeholders(user_prompt, self.pos_key, self.question_position)

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def tokenized_length(self, tokenizer: PreTrainedTokenizerBase) -> int:
        messages = self.build_messages()
        return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

    def to_dict(self) -> dict[str, Any]:
        messages = self.build_messages()
        # Format the positive KV pair the same way it appears in context
        if self.kv_format == "lines":
            pos_doc_str = f"Key {self.pos_key}:\n{self.pos_val}"
        elif self.kv_format == "json":
            pos_doc_str = f'"{self.pos_key}": "{self.pos_val}"'
        elif self.kv_format == "csv":
            pos_doc_str = f"{self.pos_key},{self.pos_val}"
        else:
            pos_doc_str = f"{self.pos_key}: {self.pos_val}"

        return {
            "prompt": messages,
            "question": _make_question_str(self.pos_key),
            "answer": self.pos_val,
            "pos_doc_depths": [self.needle_depth],
            "pos_docs": [pos_doc_str],
            "type": "retrieval",
        }


# ---------------------------------------------------------------------------
# RetrievalDataset
# ---------------------------------------------------------------------------

class RetrievalDataset(Dataset):
    """
    Map-style, deterministic retrieval dataset with augmentations.

    Augmentations (all deterministic per example via seeded RNG):
      - KV format (lines / json / csv): sampled from available prompt variants
      - Instruction phrasing: sampled from variants within the selected format
      - Question position (end / beginning / both): sampled per example
      - Needle depth: uniform random in [0, 1]
      - Negative ordering: shuffled per example
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        n_examples: int,
        system_prompt: str,
        prompts_dir: str,
        target_context: int = 8000,
        context_length_max: int | None = None,
        key_min: int = -10000,
        key_max: int = 10000,
        value_size: int = 10,
        seed: int = 0,
        question_position_weights: dict[str, float] | None = None,
        kv_formats: list[str] | None = None,
        instruction_variants: list[int] | None = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_examples = int(n_examples)
        self.target_context_min, self.target_context_max = normalize_context_range(
            target_context, context_length_max
        )
        self.key_min = int(key_min)
        self.key_max = int(key_max)
        self.value_size = int(value_size)
        self.system_prompt = system_prompt

        # Question position sampling
        qp_weights = question_position_weights or DEFAULT_QUESTION_POSITION_WEIGHTS
        self._qp_strategies = list(qp_weights.keys())
        self._qp_weights = list(qp_weights.values())

        # Load all instruction variants: list of (format_name, prompt_template)
        self.variants: list[tuple[str, str]] = []
        self._load_variants(prompts_dir, kv_formats, instruction_variants)
        if not self.variants:
            raise ValueError(f"No prompt variants found in {prompts_dir}. Expected subdirs like lines/0/, json/0/, etc.")

        # Compute max number of KV pairs needed to fill target context
        key_digits = int(math.log10(max(1, abs(self.key_min) + abs(self.key_max)))) + 1
        self.avg_item_length = self.value_size + key_digits + 2
        self.max_num_pairs = int(((self.target_context_max - 200) / self.avg_item_length) * 2)

        # Pre-sample per-example seeds
        ss = np.random.SeedSequence(int(seed))
        children = ss.spawn(self.n_examples)
        self.base_seeds: list[int] = [int(cs.generate_state(1, dtype=np.uint32)[0]) for cs in children]

    def _load_variants(self, prompts_dir: str, kv_formats: list[str] | None = None,
                        instruction_variants: list[int] | None = None) -> None:
        """Scan prompts_dir for {format}/{variant_num}/prompt.txt files."""
        if not os.path.isdir(prompts_dir):
            raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

        allowed_formats = set(kv_formats) if kv_formats else set(KV_FORMATTERS.keys())
        allowed_variants = set(instruction_variants) if instruction_variants is not None else None
        for fmt_name in sorted(os.listdir(prompts_dir)):
            fmt_dir = os.path.join(prompts_dir, fmt_name)
            if not os.path.isdir(fmt_dir) or fmt_name not in allowed_formats:
                continue
            for variant_num in sorted(os.listdir(fmt_dir)):
                variant_dir = os.path.join(fmt_dir, variant_num)
                if not os.path.isdir(variant_dir):
                    continue
                if allowed_variants is not None:
                    try:
                        if int(variant_num) not in allowed_variants:
                            continue
                    except ValueError:
                        continue
                prompt_path = os.path.join(variant_dir, "prompt.txt")
                if os.path.isfile(prompt_path):
                    with open(prompt_path, "r") as f:
                        prompt_template = f.read()
                    self.variants.append((fmt_name, prompt_template))

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= self.n_examples:
            raise IndexError(idx)

        seed_base = self.base_seeds[idx]
        py_rng = random.Random(seed_base)
        np_rng = np.random.default_rng(seed_base)
        sampled_target_context = sample_target_context(
            seed_base, self.target_context_min, self.target_context_max
        )

        # --- Sample augmentation choices (deterministic per example) ---
        variant_idx = py_rng.randrange(len(self.variants))
        kv_format, prompt_template = self.variants[variant_idx]

        question_position = py_rng.choices(self._qp_strategies, weights=self._qp_weights, k=1)[0]

        # --- Generate base KV pairs ---
        num_possible_keys = self.key_max - self.key_min
        keys = (np_rng.choice(num_possible_keys, size=self.max_num_pairs + 1, replace=False) + self.key_min).tolist()
        vals = [rand_alphanumeric_str(self.value_size, py_rng) for _ in range(self.max_num_pairs + 1)]

        pos_key = int(keys[0])
        pos_val = vals[0]
        neg_keys = [int(k) for k in keys[1:]]
        neg_vals = vals[1:]

        # --- Shuffle negatives ---
        perm = list(range(len(neg_keys)))
        py_rng.shuffle(perm)
        neg_keys = [neg_keys[i] for i in perm]
        neg_vals = [neg_vals[i] for i in perm]

        # --- Sample needle depth ---
        needle_depth = float(np_rng.uniform(0.0, 1.0))

        # --- Build tokenizeable example ---
        token_example = TokenizeableRetrievalExample(
            pos_key=pos_key,
            pos_val=pos_val,
            needle_depth=needle_depth,
            neg_keys=neg_keys,
            neg_vals=neg_vals,
            system_prompt=self.system_prompt,
            prompt_template=prompt_template,
            kv_format=kv_format,
            question_position=question_position,
        )
        input_length = token_example.set_largest_k(
            self.tokenizer, sampled_target_context, initial_step_size=4, min_step_size=1
        )

        result = token_example.to_dict()
        result["id"] = idx
        result["target_context"] = sampled_target_context
        result["input_length"] = input_length
        result["k_used"] = token_example.k
        result["kv_format"] = kv_format
        result["instruction_variant"] = variant_idx
        result["question_position"] = question_position
        return result
