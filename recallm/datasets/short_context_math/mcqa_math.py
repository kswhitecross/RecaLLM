"""
MCQA math dataset for final training: multiple-choice math problems with no context.

Short-context (~275 tokens) multiple-choice math problems that teach the model
recall is counterproductive when there is nothing to recall from.

Augmentations:
  - Instruction variation: multiple prompt templates (sampled per example)
  - Choice ordering: shuffled per example (deterministic via seed)
  - Choice formatting: different presentation styles per variant
    (e.g., "A) ...", "A. ...", "[A] ...", "- A: ...")

Source: stellaathena/math_mcqa (HuggingFace, train: 7,498 + test: 5,000).

Train/eval split: HF train split for training, HF test split for eval.
"""

import os
import random
from typing import Any, Callable

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


# Choice formatting functions — one per variant index.
# Each takes (letter, choice_text) and returns a formatted line.
CHOICE_FORMATS: list[Callable[[str, str], str]] = [
    lambda letter, text: f"{letter}) {text}",      # variant 0: A) choice
    lambda letter, text: f"{letter}. {text}",       # variant 1: A. choice
    lambda letter, text: f"[{letter}] {text}",      # variant 2: [A] choice
    lambda letter, text: f"- {letter}: {text}",     # variant 3: - A: choice
]


def _filter_example(example: dict, min_level: int, max_level: int) -> bool:
    """Filter MCQA examples by level and content quality."""
    level_str = example["level"][-1]
    if not level_str.isdigit():
        return False
    level = int(level_str)
    if level < min_level or level > max_level:
        return False
    if "[asy]" in example["problem"]:
        return False
    if any("Error" in choice for choice in example["choices"]):
        return False
    return True


class MCQAMathDataset(Dataset):
    """
    Map-style, deterministic MCQA math dataset with instruction and choice format augmentation.

    Augmentations (all deterministic per example via seeded RNG):
      - Instruction phrasing: sampled from prompt variants
      - Choice ordering: letters A-D shuffled per example
      - Choice formatting: presentation style tied to variant index
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        n_examples: int,
        system_prompt: str,
        prompts_dir: str,
        seed: int = 0,
        instruction_variants: list[int] | None = None,
        min_level: int = 2,
        max_level: int = 4,
        split: str = "train",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_examples = int(n_examples)
        self.system_prompt = system_prompt

        # Load prompt variants (stores (prompt_template, variant_index) pairs)
        self.variants: list[tuple[str, int]] = []
        self._load_variants(prompts_dir, instruction_variants)
        if not self.variants:
            raise ValueError(f"No prompt variants found in {prompts_dir}. Expected subdirs like 0/, 1/, etc.")

        # Load and filter HF dataset
        hf_split = "test" if split in ("validation", "eval", "test") else "train"
        raw = load_dataset("stellaathena/math_mcqa")[hf_split]
        self.hf_dataset = raw.filter(
            lambda ex: _filter_example(ex, min_level, max_level)
        )

        # Sample n_examples from filtered dataset
        sample_rng = np.random.default_rng(int(seed))
        if self.n_examples > len(self.hf_dataset):
            raise ValueError(
                f"Requested {self.n_examples} examples but filtered {hf_split} split only has "
                f"{len(self.hf_dataset)} (min_level={min_level}, max_level={max_level})."
            )
        self.example_indices: list[int] = sample_rng.choice(
            len(self.hf_dataset), size=self.n_examples, replace=False
        ).tolist()

        # Pre-sample per-example seeds for deterministic augmentation
        ss = np.random.SeedSequence(int(seed))
        children = ss.spawn(self.n_examples)
        self.base_seeds: list[int] = [int(cs.generate_state(1, dtype=np.uint32)[0]) for cs in children]

    def _load_variants(self, prompts_dir: str, instruction_variants: list[int] | None = None) -> None:
        """Scan prompts_dir for {variant_num}/prompt.txt files.
        Stores (prompt_template, variant_index) tuples so we can look up
        the matching choice format by variant index.
        """
        if not os.path.isdir(prompts_dir):
            raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

        allowed_variants = set(instruction_variants) if instruction_variants is not None else None
        for entry in sorted(os.listdir(prompts_dir)):
            variant_dir = os.path.join(prompts_dir, entry)
            if not os.path.isdir(variant_dir):
                continue
            try:
                variant_num = int(entry)
            except ValueError:
                continue
            if allowed_variants is not None and variant_num not in allowed_variants:
                continue
            prompt_path = os.path.join(variant_dir, "prompt.txt")
            if os.path.isfile(prompt_path):
                with open(prompt_path, "r") as f:
                    self.variants.append((f.read(), variant_num))

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= self.n_examples:
            raise IndexError(idx)

        seed_base = self.base_seeds[idx]
        py_rng = random.Random(seed_base)

        # --- Sample instruction variant ---
        variant_list_idx = py_rng.randrange(len(self.variants))
        prompt_template, variant_num = self.variants[variant_list_idx]

        # Select choice format for this variant
        choice_fmt = CHOICE_FORMATS[variant_num % len(CHOICE_FORMATS)]

        # --- Fetch HF example ---
        hf_idx = self.example_indices[idx]
        example = self.hf_dataset[hf_idx]

        # --- Shuffle choice ordering (deterministic per example) ---
        letters = ["A", "B", "C", "D"]
        py_rng.shuffle(letters)

        choices = example["choices"]
        answer_idx = example["answer"]  # 0-3 index of correct answer
        math_ans = choices[answer_idx]  # text of correct choice
        answer_letter = letters[answer_idx]  # letter assigned to correct choice

        # Sort by letter for display
        ordered_pairs = sorted(zip(letters, choices), key=lambda x: x[0])
        choices_str = "\n".join(choice_fmt(letter, choice) for letter, choice in ordered_pairs)

        # --- Build prompt ---
        user_content = (
            prompt_template
            .replace("{question}", example["problem"])
            .replace("{choices}", choices_str)
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        return {
            "prompt": messages,
            "question": example["problem"],
            "answer": answer_letter,
            "math_answer": math_ans,
            "pos_docs": [],
            "type": "mcqa_math",
            "id": str(idx),
            "instruction_variant": variant_list_idx,
        }
