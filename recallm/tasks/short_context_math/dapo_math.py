"""
DAPO math dataset for final training: open-ended math problems with no context.

Short-context (~275 tokens) math problems that teach the model recall is
counterproductive when there is nothing to recall from.

Augmentations:
  - Instruction variation: multiple prompt templates (sampled per example)

Source: ftajwar/deduplicated_dapo_dataset (HuggingFace, train split only, 17,398 examples).
~19% of examples contain Chinese math problems and are filtered out, leaving ~14,116.

Train/eval split: Fixed-seed shuffle with last 200 examples held out as eval pool.
"""

import os
import random
import re
from typing import Any

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


# Fixed seed for the train/eval partition — never change this.
_PARTITION_SEED = 0
_EVAL_HOLDOUT_SIZE = 200


class DAPOMathDataset(Dataset):
    """
    Map-style, deterministic DAPO math dataset with instruction augmentation.

    Augmentations (all deterministic per example via seeded RNG):
      - Instruction phrasing: sampled from prompt variants
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        n_examples: int,
        system_prompt: str,
        prompts_dir: str,
        seed: int = 0,
        instruction_variants: list[int] | None = None,
        split: str = "train",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_examples = int(n_examples)
        self.system_prompt = system_prompt

        # Load prompt variants
        self.variants: list[str] = []
        self._load_variants(prompts_dir, instruction_variants)
        if not self.variants:
            raise ValueError(f"No prompt variants found in {prompts_dir}. Expected subdirs like 0/, 1/, etc.")

        # Load HF dataset, filter out Chinese-language examples, then partition
        hf_dataset = load_dataset("ftajwar/deduplicated_dapo_dataset")["train"]
        hf_dataset = hf_dataset.filter(lambda ex: not _CJK_RE.search(ex["prompt"]))
        partition_rng = np.random.default_rng(_PARTITION_SEED)
        all_indices = partition_rng.permutation(len(hf_dataset)).tolist()

        if split == "train":
            pool_indices = all_indices[:-_EVAL_HOLDOUT_SIZE]
        elif split in ("validation", "eval", "test"):
            pool_indices = all_indices[-_EVAL_HOLDOUT_SIZE:]
        else:
            raise ValueError(f"Unknown split: {split!r}. Expected 'train' or 'validation'.")

        self.hf_dataset = hf_dataset

        # Sample n_examples from the pool
        sample_rng = np.random.default_rng(int(seed))
        if self.n_examples > len(pool_indices):
            raise ValueError(
                f"Requested {self.n_examples} examples but {split} pool only has {len(pool_indices)}."
            )
        sampled = sample_rng.choice(len(pool_indices), size=self.n_examples, replace=False)
        self.example_indices: list[int] = [pool_indices[i] for i in sampled]

        # Pre-sample per-example seeds for deterministic augmentation
        ss = np.random.SeedSequence(int(seed))
        children = ss.spawn(self.n_examples)
        self.base_seeds: list[int] = [int(cs.generate_state(1, dtype=np.uint32)[0]) for cs in children]

    def _load_variants(self, prompts_dir: str, instruction_variants: list[int] | None = None) -> None:
        """Scan prompts_dir for {variant_num}/prompt.txt files."""
        if not os.path.isdir(prompts_dir):
            raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

        allowed_variants = set(instruction_variants) if instruction_variants is not None else None
        for entry in sorted(os.listdir(prompts_dir)):
            variant_dir = os.path.join(prompts_dir, entry)
            if not os.path.isdir(variant_dir):
                continue
            if allowed_variants is not None:
                try:
                    if int(entry) not in allowed_variants:
                        continue
                except ValueError:
                    continue
            prompt_path = os.path.join(variant_dir, "prompt.txt")
            if os.path.isfile(prompt_path):
                with open(prompt_path, "r") as f:
                    self.variants.append(f.read())

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= self.n_examples:
            raise IndexError(idx)

        seed_base = self.base_seeds[idx]
        py_rng = random.Random(seed_base)

        # --- Sample instruction variant ---
        variant_idx = py_rng.randrange(len(self.variants))
        prompt_template = self.variants[variant_idx]

        # --- Fetch HF example ---
        hf_idx = self.example_indices[idx]
        example = self.hf_dataset[hf_idx]

        # --- Build prompt ---
        user_content = prompt_template.replace("{question}", example["prompt"])
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        return {
            "prompt": messages,
            "question": example["prompt"],
            "answer": example["answer"],
            "pos_docs": [],
            "type": "dapo_math",
            "id": str(example.get("id", hf_idx)),
            "instruction_variant": variant_idx,
        }
