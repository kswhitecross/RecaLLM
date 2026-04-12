"""
Threshold Filter dataset for final training: synthetic catalog filtering aggregation task.

Teaches the model that recall is counterproductive for aggregation tasks.
A catalog of items with numeric attributes; the model must count items
satisfying a multi-condition filter or identify extremes among filtered items.

Augmentations:
  - Difficulty levels: easy/medium/hard (controls attributes, conditions, precision)
  - Instruction variation: multiple prompt templates (sampled per example)
  - Question position variation: end, beginning, both (sampled per example)

Difficulty presets:
  Easy   — 2 attributes, 1 condition, round numbers
  Medium — 3 attributes, 2 conditions (AND), integers
  Hard   — 4-5 attributes, 3 conditions (AND), 1 decimal, count or extremum
"""

import os
import random
import string
from typing import Any

import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from recallm.datasets.base import (
    TokenizeableExample,
    normalize_context_range,
    sample_target_context,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_QUESTION_POSITION_WEIGHTS = {"end": 0.6, "beginning": 0.2, "both": 0.2}
DEFAULT_DIFFICULTY_WEIGHTS = {"easy": 0.45, "medium": 0.45, "hard": 0.10}

# Attribute pool: (name, min_val, max_val, prefix, suffix)
# decimal_places is set per-difficulty, not per-attribute
ATTRIBUTE_POOL = [
    ("price", 10, 500, "$", ""),
    ("rating", 1.0, 5.0, "", "/5"),
    ("weight", 0.1, 50.0, "", "kg"),
    ("stock", 0, 1000, "", " units"),
    ("score", 0, 100, "", "%"),
    ("quantity", 1, 500, "", ""),
    ("discount", 0, 50, "", "%"),
    ("reviews", 0, 5000, "", ""),
    ("capacity", 1, 200, "", "L"),
    ("power", 10, 2000, "", "W"),
]

COMPARATORS = [">", "<", ">=", "<="]

# Item name components
_GREEK_PREFIXES = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi",
    "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
]

DIFFICULTY_LEVELS = {
    "easy": {
        "n_attributes_range": (2, 2),
        "n_conditions_range": (1, 1),
        "decimal_places": 0,
        "rounding_step": 1,  # Integer values (was 10, which collapsed small-range attrs like rating to all zeros)
        "answer_type": "count",
    },
    "medium": {
        "n_attributes_range": (3, 3),
        "n_conditions_range": (2, 2),
        "decimal_places": 0,
        "rounding_step": 1,  # Integer precision
        "answer_type": "count",
    },
    "hard": {
        "n_attributes_range": (4, 5),
        "n_conditions_range": (3, 3),
        "decimal_places": 1,
        "rounding_step": None,  # Use decimal_places directly
        "answer_type_weights": {"count": 0.7, "extremum": 0.3},
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_unique_item_names(n: int, rng: random.Random) -> list[str]:
    """Generate n unique item names."""
    names = set()
    result = []
    while len(result) < n:
        prefix = rng.choice(_GREEK_PREFIXES)
        suffix = rng.randint(1, 99)
        name = f"{prefix}-{suffix}"
        if name not in names:
            names.add(name)
            result.append(name)
    return result


def _format_value(val: float, attr_spec: tuple, decimal_places: int, rounding_step: int | None) -> str:
    """Format an attribute value with its prefix/suffix."""
    _, _, _, prefix, suffix = attr_spec
    if rounding_step is not None and rounding_step > 1:
        val = round(val / rounding_step) * rounding_step
    if decimal_places == 0:
        return f"{prefix}{int(round(val))}{suffix}"
    return f"{prefix}{val:.{decimal_places}f}{suffix}"


def _raw_value(val: float, decimal_places: int, rounding_step: int | None) -> float:
    """Get the numeric value after rounding (for comparison)."""
    if rounding_step is not None and rounding_step > 1:
        val = round(val / rounding_step) * rounding_step
    if decimal_places == 0:
        return float(int(round(val)))
    return round(val, decimal_places)


def _format_condition(attr_name: str, comparator: str, threshold: float,
                      attr_spec: tuple, decimal_places: int, rounding_step: int | None) -> str:
    """Format a condition for the question string."""
    threshold_str = _format_value(threshold, attr_spec, decimal_places, rounding_step)
    return f"{attr_name} {comparator} {threshold_str}"


def _evaluate_condition(val: float, comparator: str, threshold: float) -> bool:
    if comparator == ">":
        return val > threshold
    elif comparator == "<":
        return val < threshold
    elif comparator == ">=":
        return val >= threshold
    elif comparator == "<=":
        return val <= threshold
    raise ValueError(f"Unknown comparator: {comparator}")


def _fill_question_placeholders(prompt_template: str, question: str, position: str) -> str:
    if position == "end":
        return prompt_template.replace("{question_start}", "").replace("{question_end}", question + "\n")
    elif position == "beginning":
        return prompt_template.replace("{question_start}", question + "\n\n").replace("{question_end}", "")
    elif position == "both":
        return prompt_template.replace("{question_start}", question + "\n\n").replace("{question_end}", question + "\n")
    raise ValueError(f"Unknown question position: {position}")


# ---------------------------------------------------------------------------
# TokenizeableThresholdFilterExample
# ---------------------------------------------------------------------------

class TokenizeableThresholdFilterExample(TokenizeableExample):
    """A single threshold filter example that can be tokenized and fitted to a target context length."""

    def __init__(
        self,
        item_names: list[str],
        attr_specs: list[tuple],  # Selected attribute specs
        n_conditions: int,
        decimal_places: int,
        rounding_step: int | None,
        answer_type: str,  # "count" or "extremum"
        system_prompt: str,
        prompt_template: str,
        question_position: str,
        np_rng: np.random.Generator,
        py_rng: random.Random,
    ):
        super().__init__(max_k=len(item_names))
        self.item_names = item_names
        self.attr_specs = attr_specs
        self.n_conditions = n_conditions
        self.decimal_places = decimal_places
        self.rounding_step = rounding_step
        self.answer_type = answer_type
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.question_position = question_position
        self.np_rng = np_rng
        self.py_rng = py_rng

        # Generated after set_largest_k + finalize
        self._items_str: str | None = None
        self._question_str: str | None = None
        self.answer: str | None = None
        self._item_values: list[list[float]] | None = None  # [item_idx][attr_idx]
        self._conditions: list[tuple] | None = None  # [(attr_idx, comparator, threshold), ...]

    def average_length_per_item(self, tokenizer: PreTrainedTokenizerBase) -> float:
        # Estimate tokens per item line
        sample_size = min(20, len(self.item_names))
        sample_strs = []
        for i in range(sample_size):
            parts = []
            for spec in self.attr_specs:
                name, min_v, max_v, prefix, suffix = spec
                mid = (min_v + max_v) / 2
                parts.append(f"{name} {_format_value(mid, spec, self.decimal_places, self.rounding_step)}")
            line = f"Product {self.item_names[i]}: {', '.join(parts)}\n"
            sample_strs.append(line)
        encodings = tokenizer(sample_strs, add_special_tokens=False)
        lengths = [len(ids) for ids in encodings["input_ids"]]
        return float(np.mean(lengths))

    def finalize(self):
        """After set_largest_k, generate items, conditions, and compute answer."""
        n_items = self.k
        if n_items == 0:
            self.answer = "0"
            self._items_str = ""
            self._question_str = "How many products match the criteria?"
            return

        n_attrs = len(self.attr_specs)

        # Generate item values
        self._item_values = []
        for _ in range(n_items):
            vals = []
            for spec in self.attr_specs:
                _, min_v, max_v, _, _ = spec
                raw = float(self.np_rng.uniform(min_v, max_v))
                vals.append(_raw_value(raw, self.decimal_places, self.rounding_step))
            self._item_values.append(vals)

        # Generate conditions with degeneracy prevention
        self._conditions = self._generate_non_degenerate_conditions(n_items)

        # Compute matching items
        matching_indices = self._get_matching_indices()
        n_matching = len(matching_indices)

        # Compute answer
        if self.answer_type == "extremum" and n_matching > 0:
            # Pick an attribute to find extremum of (different from condition attributes)
            cond_attr_idxs = {c[0] for c in self._conditions}
            available = [i for i in range(n_attrs) if i not in cond_attr_idxs]
            if not available:
                available = list(range(n_attrs))
            extremum_attr_idx = self.py_rng.choice(available)
            extremum_attr_name = self.attr_specs[extremum_attr_idx][0]

            # Find item with highest value of that attribute among matching
            best_idx = max(matching_indices, key=lambda i: self._item_values[i][extremum_attr_idx])
            self.answer = self.item_names[best_idx]

            condition_strs = [
                _format_condition(self.attr_specs[ai][0], comp, thresh, self.attr_specs[ai],
                                  self.decimal_places, self.rounding_step)
                for ai, comp, thresh in self._conditions
            ]
            self._question_str = (
                f"Which product has the highest {extremum_attr_name} among those where "
                f"{' AND '.join(condition_strs)}?"
            )
        else:
            self.answer = str(n_matching)
            condition_strs = [
                _format_condition(self.attr_specs[ai][0], comp, thresh, self.attr_specs[ai],
                                  self.decimal_places, self.rounding_step)
                for ai, comp, thresh in self._conditions
            ]
            self._question_str = f"How many products have {' AND '.join(condition_strs)}?"

        # Build items string
        lines = []
        for i in range(n_items):
            parts = []
            for j, spec in enumerate(self.attr_specs):
                name = spec[0]
                val_str = _format_value(self._item_values[i][j], spec, self.decimal_places, self.rounding_step)
                parts.append(f"{name} {val_str}")
            lines.append(f"Product {self.item_names[i]}: {', '.join(parts)}")
        self._items_str = "\n".join(lines)

    def _generate_non_degenerate_conditions(self, n_items: int) -> list[tuple]:
        """Generate conditions ensuring 5-95% of items match. Retry up to 10 times."""
        n_attrs = len(self.attr_specs)

        for attempt in range(10):
            conditions = []
            used_attrs = set()
            for _ in range(self.n_conditions):
                # Pick attribute (prefer unused)
                available = [i for i in range(n_attrs) if i not in used_attrs]
                if not available:
                    available = list(range(n_attrs))
                attr_idx = self.py_rng.choice(available)
                used_attrs.add(attr_idx)

                comparator = self.py_rng.choice(COMPARATORS)

                # Pick threshold: use a percentile of actual values to control match rate
                attr_vals = [self._item_values[i][attr_idx] for i in range(n_items)]
                # Target ~30-70% of items matching this single condition
                percentile = self.np_rng.uniform(30, 70)
                threshold = float(np.percentile(attr_vals, percentile))
                threshold = _raw_value(threshold, self.decimal_places, self.rounding_step)

                conditions.append((attr_idx, comparator, threshold))

            # Check match fraction
            matching = self._count_matching(conditions, n_items)
            frac = matching / n_items if n_items > 0 else 0
            if 0.05 <= frac <= 0.95:
                return conditions

        # Fallback: return last attempt (might be slightly degenerate)
        return conditions

    def _count_matching(self, conditions: list[tuple], n_items: int) -> int:
        count = 0
        for i in range(n_items):
            if all(
                _evaluate_condition(self._item_values[i][ai], comp, thresh)
                for ai, comp, thresh in conditions
            ):
                count += 1
        return count

    def _get_matching_indices(self) -> list[int]:
        matching = []
        for i in range(self.k):
            if all(
                _evaluate_condition(self._item_values[i][ai], comp, thresh)
                for ai, comp, thresh in self._conditions
            ):
                matching.append(i)
        return matching

    def build_messages(self) -> list[dict[str, str]]:
        if self._items_str is None:
            # For tokenized_length estimation during set_largest_k, use placeholder lines
            lines = []
            for i in range(self.k):
                parts = []
                for spec in self.attr_specs:
                    name, min_v, max_v, prefix, suffix = spec
                    mid = (min_v + max_v) / 2
                    parts.append(f"{name} {_format_value(mid, spec, self.decimal_places, self.rounding_step)}")
                lines.append(f"Product {self.item_names[i]}: {', '.join(parts)}")
            items_str = "\n".join(lines)
            question_str = "How many products have price > $100?"
        else:
            items_str = self._items_str
            question_str = self._question_str

        user_prompt = self.prompt_template.replace("{items}", items_str)
        user_prompt = _fill_question_placeholders(user_prompt, question_str, self.question_position)

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def tokenized_length(self, tokenizer: PreTrainedTokenizerBase) -> int:
        messages = self.build_messages()
        return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

    def to_dict(self) -> dict[str, Any]:
        messages = self.build_messages()
        return {
            "prompt": messages,
            "question": self._question_str,
            "answer": self.answer,
            "pos_docs": [],
            "type": "threshold_filter",
        }


# ---------------------------------------------------------------------------
# ThresholdFilterDataset
# ---------------------------------------------------------------------------

class ThresholdFilterDataset(Dataset):
    """
    Map-style, deterministic threshold filter dataset with augmentations.

    Augmentations (all deterministic per example via seeded RNG):
      - Difficulty level: easy / medium / hard (uniform)
      - Instruction phrasing: sampled from prompt variants
      - Question position (end / beginning / both): sampled per example
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        n_examples: int,
        system_prompt: str,
        prompts_dir: str,
        target_context: int = 8000,
        context_length_max: int | None = None,
        seed: int = 0,
        question_position_weights: dict[str, float] | None = None,
        difficulty_weights: dict[str, float] | None = None,
        instruction_variants: list[int] | None = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_examples = int(n_examples)
        self.target_context_min, self.target_context_max = normalize_context_range(
            target_context, context_length_max
        )
        self.system_prompt = system_prompt

        # Question position sampling
        qp_weights = question_position_weights or DEFAULT_QUESTION_POSITION_WEIGHTS
        self._qp_strategies = list(qp_weights.keys())
        self._qp_weights = list(qp_weights.values())

        # Difficulty sampling
        diff_weights = difficulty_weights or DEFAULT_DIFFICULTY_WEIGHTS
        self._diff_levels = list(diff_weights.keys())
        self._diff_weights = list(diff_weights.values())

        # Load instruction variants
        self.variants: list[str] = []
        self._load_variants(prompts_dir, instruction_variants)
        if not self.variants:
            raise ValueError(f"No prompt variants found in {prompts_dir}.")

        # Estimate max items needed (worst case: 2 attrs, ~15 tokens per line)
        self.max_items = int(((self.target_context_max - 200) / 15) * 2)

        # Pre-sample per-example seeds
        ss = np.random.SeedSequence(int(seed))
        children = ss.spawn(self.n_examples)
        self.base_seeds: list[int] = [int(cs.generate_state(1, dtype=np.uint32)[0]) for cs in children]

    def _load_variants(self, prompts_dir: str, instruction_variants: list[int] | None = None) -> None:
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
        np_rng = np.random.default_rng(seed_base)
        sampled_target_context = sample_target_context(
            seed_base, self.target_context_min, self.target_context_max
        )

        # --- Sample augmentation choices ---
        variant_idx = py_rng.randrange(len(self.variants))
        prompt_template = self.variants[variant_idx]
        question_position = py_rng.choices(self._qp_strategies, weights=self._qp_weights, k=1)[0]

        # --- Sample difficulty ---
        difficulty = py_rng.choices(self._diff_levels, weights=self._diff_weights, k=1)[0]
        preset = DIFFICULTY_LEVELS[difficulty]

        n_attributes = py_rng.randint(*preset["n_attributes_range"])
        n_conditions = py_rng.randint(*preset["n_conditions_range"])
        decimal_places = preset["decimal_places"]
        rounding_step = preset.get("rounding_step")

        # Answer type
        if "answer_type_weights" in preset:
            types = list(preset["answer_type_weights"].keys())
            weights = list(preset["answer_type_weights"].values())
            answer_type = py_rng.choices(types, weights=weights, k=1)[0]
        else:
            answer_type = preset["answer_type"]

        # --- Sample attributes from pool ---
        pool = list(ATTRIBUTE_POOL)
        py_rng.shuffle(pool)
        attr_specs = pool[:n_attributes]

        # --- Pre-generate unique item names ---
        item_names = _generate_unique_item_names(self.max_items, py_rng)

        # --- Build tokenizeable example ---
        token_example = TokenizeableThresholdFilterExample(
            item_names=item_names,
            attr_specs=attr_specs,
            n_conditions=n_conditions,
            decimal_places=decimal_places,
            rounding_step=rounding_step,
            answer_type=answer_type,
            system_prompt=self.system_prompt,
            prompt_template=prompt_template,
            question_position=question_position,
            np_rng=np_rng,
            py_rng=py_rng,
        )

        input_length = token_example.set_largest_k(
            self.tokenizer, sampled_target_context, initial_step_size=10, min_step_size=1
        )

        # Finalize: generate actual item values, conditions, and compute answer
        token_example.finalize()

        result = token_example.to_dict()
        result["id"] = idx
        result["target_context"] = sampled_target_context
        result["input_length"] = input_length
        result["k_used"] = token_example.k
        result["instruction_variant"] = variant_idx
        result["question_position"] = question_position
        result["difficulty"] = difficulty
        result["n_attributes"] = n_attributes
        result["n_conditions"] = n_conditions
        result["answer_type"] = answer_type
        return result
