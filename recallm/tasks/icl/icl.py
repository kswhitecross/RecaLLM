"""
ICL prompt-building dataset: wraps a raw ICL dataset (Banking77 or MASSIVE)
with label formatting and instruction variants.

Augmentations (all deterministic per example via seeded RNG):
  - Label format (numeric / text): sampled from available prompt variants
  - Instruction phrasing: sampled from variants within the selected label format
  - Numeric label permutation: random mapping per example
  - Demo format (multiline / pipe / arrow / paren): sampled per example

The test text always appears at the end of the prompt (after demonstrations).
Templates use varied test markers for diversity.
"""

import json
import os
import random
from typing import Any

import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from recallm.tasks.base import (
    TokenizeableExample,
    normalize_context_range,
    sample_target_context,
)
from recallm.tasks.icl import ICLExample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Conservative lower bound on tokens per demo ("{text}\nlabel: {label}\n\n").
# Used to compute a safe upper bound on how many demos to generate.
_AVG_DEMO_TOKEN_LEN = 15

# Demo format variations and their sampling weights.
# "multiline" is the standard evaluation format (highest weight).
# Single-line formats encourage contiguous recall across the text+label boundary.
DEMO_FORMAT_NAMES = ["multiline", "pipe", "arrow", "paren"]
DEMO_FORMAT_WEIGHTS = [0.80, 0.05, 0.05, 0.10]


def _format_demo(text: str, label: str, demo_format: str = "multiline") -> str:
    """Format a single demonstration as it appears in the prompt.

    Args:
        text: The demonstration text (e.g., "i want to post on facebook").
        label: The display label (e.g., "social_post" or "7").
        demo_format: One of "multiline", "pipe", "arrow", "paren".
    """
    if demo_format == "multiline":
        return f"{text}\nlabel: {label}"
    elif demo_format == "pipe":
        return f"{text} | label: {label}"
    elif demo_format == "arrow":
        return f"{text} -> label: {label}"
    elif demo_format == "paren":
        return f"{text} (label: {label})"
    else:
        raise ValueError(f"Unknown demo_format: {demo_format!r}")


# ---------------------------------------------------------------------------
# TokenizeableICLExample
# ---------------------------------------------------------------------------


class TokenizeableICLExample(TokenizeableExample):
    """A single ICL example that can be tokenized and fitted to a target context length."""

    def __init__(
        self,
        example: ICLExample,
        display_labels: list[str],
        test_display_label: str,
        system_prompt: str,
        prompt_template: str,
        demo_format: str = "multiline",
    ):
        super().__init__(max_k=len(example.demo_texts))
        self.example = example
        self.display_labels = display_labels
        self.test_display_label = test_display_label
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.demo_format = demo_format

    def average_length_per_item(self, tokenizer: PreTrainedTokenizerBase) -> float:
        # Estimate average tokens per demonstration by sampling a few
        sample_size = min(20, len(self.example.demo_texts))
        sample_strs = [
            _format_demo(self.example.demo_texts[i], self.display_labels[i], self.demo_format) + "\n\n"
            for i in range(sample_size)
        ]
        encodings = tokenizer(sample_strs, add_special_tokens=False)
        lengths = [len(ids) for ids in encodings["input_ids"]]
        return float(np.mean(lengths))

    def _build_demonstrations(self) -> str:
        """Format the first k demonstrations as text."""
        demos = [
            _format_demo(self.example.demo_texts[i], self.display_labels[i], self.demo_format)
            for i in range(self.k)
        ]
        return "\n\n".join(demos)

    def build_messages(self) -> list[dict[str, str]]:
        demonstrations = self._build_demonstrations()
        user_prompt = self.prompt_template.replace("{demonstrations}", demonstrations)
        user_prompt = user_prompt.replace("{question_end}", self.example.test_text)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def tokenized_length(self, tokenizer: PreTrainedTokenizerBase) -> int:
        messages = self.build_messages()
        return len(
            tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        )

    def to_dict(self) -> dict[str, Any]:
        messages = self.build_messages()

        # pos_docs = rendered positive-class demos (same label as test query)
        # from the first k included demonstrations.
        # Needed for gold_doc_overlap ablation (Option B in design doc §7.3).
        pos_docs = []
        pos_doc_labels = []  # label strings for label-presence bonus in reward
        for i in range(self.k):
            if self.example.demo_labels[i] == self.example.test_label:
                pos_docs.append(
                    _format_demo(
                        self.example.demo_texts[i], self.display_labels[i],
                        self.demo_format,
                    )
                )
                pos_doc_labels.append(self.display_labels[i])

        return {
            "prompt": messages,
            "question": self.example.test_text,
            "answer": self.test_display_label,
            "pos_doc_depths": [],
            "pos_docs": pos_docs,
            "type": self.example.dataset_type,
            "settings": json.dumps({
                "pos_doc_labels": pos_doc_labels,
                "demo_format": self.demo_format,
            }),
        }


# ---------------------------------------------------------------------------
# ICLPromptDataset
# ---------------------------------------------------------------------------


class ICLPromptDataset(Dataset):
    """
    Prompt-building wrapper around a raw ICL dataset (Banking77 or MASSIVE).

    Augmentations (all deterministic per example via seeded RNG):
      - Label format (numeric / text): sampled from available prompt variants
      - Demo format (multiline / pipe / arrow / paren): sampled per example
      - Instruction phrasing: sampled from variants within the selected format
      - For numeric format: random label-to-number permutation per example

    Test text always appears at the end of the prompt (after demonstrations).
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        system_prompt: str,
        prompts_dir: str,
        dataset_task_description: str,
        target_context: int = 8000,
        context_length_max: int | None = None,
        seed: int = 0,
        label_formats: list[str] | None = None,
        instruction_variants: list[int] | None = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.target_context_min, self.target_context_max = normalize_context_range(
            target_context, context_length_max
        )
        self.system_prompt = system_prompt
        self.dataset_task_description = dataset_task_description

        # Load all instruction variants: list of (label_format, prompt_template)
        self.variants: list[tuple[str, str]] = []
        self._load_variants(prompts_dir, label_formats, instruction_variants)
        if not self.variants:
            raise ValueError(
                f"No prompt variants found in {prompts_dir}. "
                "Expected subdirs like numeric/0/, text/0/, etc."
            )

        # Pre-compute per-example seeds
        ss = np.random.SeedSequence(int(seed))
        children = ss.spawn(len(self.dataset))
        self.base_seeds: list[int] = [
            int(cs.generate_state(1, dtype=np.uint32)[0]) for cs in children
        ]

    def _load_variants(
        self,
        prompts_dir: str,
        label_formats: list[str] | None = None,
        instruction_variants: list[int] | None = None,
    ) -> None:
        """Scan prompts_dir for {label_format}/{variant_num}/prompt.txt files."""
        if not os.path.isdir(prompts_dir):
            raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

        allowed_formats = set(label_formats) if label_formats else {"numeric", "text"}
        allowed_variants = (
            set(instruction_variants) if instruction_variants is not None else None
        )

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
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example: ICLExample = self.dataset[idx]

        seed_base = self.base_seeds[idx]
        py_rng = random.Random(seed_base)
        sampled_target_context = sample_target_context(
            seed_base, self.target_context_min, self.target_context_max
        )

        # --- Sample augmentation choices (deterministic per example) ---
        variant_idx = py_rng.randrange(len(self.variants))
        demo_format = py_rng.choices(DEMO_FORMAT_NAMES, weights=DEMO_FORMAT_WEIGHTS, k=1)[0]
        label_format, prompt_template = self.variants[variant_idx]

        # --- Build display labels based on label format ---
        if label_format == "numeric":
            # Random label-to-number permutation
            perm = list(range(example.num_labels))
            py_rng.shuffle(perm)
            display_labels = [str(perm[label]) for label in example.demo_labels]
            test_display_label = str(perm[example.test_label])
        else:
            # Text: use actual label names
            display_labels = [
                example.label_names[label] for label in example.demo_labels
            ]
            test_display_label = example.label_names[example.test_label]

        # --- Fill dataset_task_description in template ---
        filled_template = prompt_template.replace(
            "{dataset_task_description}", self.dataset_task_description
        )

        # --- Build tokenizeable example ---
        token_example = TokenizeableICLExample(
            example=example,
            display_labels=display_labels,
            test_display_label=test_display_label,
            system_prompt=self.system_prompt,
            prompt_template=filled_template,
            demo_format=demo_format,
        )
        input_length = token_example.set_largest_k(
            self.tokenizer, sampled_target_context, initial_step_size=10, min_step_size=1
        )

        result = token_example.to_dict()
        result["id"] = idx
        result["target_context"] = sampled_target_context
        result["input_length"] = input_length
        result["k_used"] = token_example.k
        result["label_format"] = label_format
        result["instruction_variant"] = variant_idx
        result["num_labels"] = example.num_labels
        result["num_demos"] = token_example.k
        result["demo_format"] = demo_format
        return result
