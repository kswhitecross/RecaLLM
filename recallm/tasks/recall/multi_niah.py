"""
Multi-NIAH dataset: needle-in-a-haystack with variable structure.

Key features:
  - K_total, V, K_query are sampled per-example (not fixed per dataset)
  - Value type (numbers/words/uuids) varies per example
  - Multiple instruction template variants
  - Question position variation (end/beginning/both)

Augmentations (all deterministic per example via seeded RNG):
  - NIAH structure: K_total (1-10), V (1-5), K_query (1-min(K_total,6)), K_total*V <= 16
  - Value type: numbers, words, uuids
  - Instruction variant
  - Question position: end, beginning, both
  - Needle depths: uniform [0, 1]
  - Essay ordering: shuffled
"""

import json
import os
import random
import re
import uuid
from typing import Any

import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

import wonderwords.random_word
from nltk.tokenize import sent_tokenize

from recallm.tasks.base import (
    TokenizeableExample,
    insert_into_list,
    normalize_context_range,
    sample_target_context,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Pre-load adjective/noun lists (NOT the Cartesian product — that's 6M+ entries)
_NOUNS = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
_ADJS = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")


def _enumerate_valid_configs(
    k_total_range: tuple[int, int] = (1, 10),
    v_range: tuple[int, int] = (1, 5),
    max_total_needles: int = 16,
    k_query_max: int = 6,
) -> list[tuple[int, int, int]]:
    """Enumerate all valid (K_total, V, K_query) triples."""
    configs = []
    for k_total in range(k_total_range[0], k_total_range[1] + 1):
        max_v = min(v_range[1], max_total_needles // k_total)
        for v in range(v_range[0], max_v + 1):
            for k_query in range(1, min(k_total, k_query_max) + 1):
                configs.append((k_total, v, k_query))
    return configs


def _generate_random_words(
    k: int, rng: random.Random, exclude: set[str] | None = None,
) -> list[str]:
    """Sample k unique adj-noun pairs on the fly (avoids 6M+ Cartesian product)."""
    exclude = exclude or set()
    result: list[str] = []
    while len(result) < k:
        word = f"{rng.choice(_ADJS)}-{rng.choice(_NOUNS)}"
        if word not in exclude:
            result.append(word)
    return result


def _generate_random_numbers(k: int, rng: random.Random, num_digits: int = 7) -> list[str]:
    lo = 10 ** (num_digits - 1)
    hi = 10 ** num_digits - 1
    return [str(rng.randint(lo, hi)) for _ in range(k)]


def _generate_random_uuids(k: int, rng: random.Random) -> list[str]:
    return [str(uuid.UUID(int=rng.getrandbits(128), version=4)) for _ in range(k)]


def _type_needle_v_str(value_type: str, singular: bool) -> str:
    """Return the display string for value type (e.g. 'number' or 'numbers')."""
    if singular:
        return value_type[:-1]  # "numbers" -> "number", "words" -> "word", "uuids" -> "uuid"
    return value_type


def _generate_needles(
    keys: list[str],
    v_per_key: int,
    value_type: str,
    needle_template: str,
    type_needle_v_display: str,
    rng: random.Random,
) -> tuple[dict[str, list[str]], list[str]]:
    """
    Generate key-value pairs and format as needle strings.

    Returns:
        key_value_map: dict mapping key -> list of values
        needles: list of formatted needle strings (len = len(keys) * v_per_key)
    """
    exclude = set(keys)
    key_value_map: dict[str, list[str]] = {}
    for key in keys:
        if value_type == "words":
            vals = _generate_random_words(v_per_key, rng, exclude)
            exclude.update(vals)
        elif value_type == "numbers":
            vals = _generate_random_numbers(v_per_key, rng)
        elif value_type == "uuids":
            vals = _generate_random_uuids(v_per_key, rng)
        else:
            raise ValueError(f"Unknown value_type: {value_type}")
        key_value_map[key] = vals

    needles = []
    for key, vals in key_value_map.items():
        for val in vals:
            needles.append(needle_template.format(
                key=key, value=val, type_needle_v=type_needle_v_display,
            ))
    return key_value_map, needles


def _fill_niah_question_placeholders(
    prompt_template: str,
    question_str: str,
    position: str,
) -> str:
    """Fill {question_start} and {question_end} based on position strategy."""
    if position == "end":
        return prompt_template.replace("{question_start}", "").replace("{question_end}", question_str)
    elif position == "beginning":
        return prompt_template.replace("{question_start}", question_str + "\n\n").replace("{question_end}", "")
    elif position == "both":
        return prompt_template.replace("{question_start}", question_str + "\n\n").replace("{question_end}", question_str)
    else:
        raise ValueError(f"Unknown question position: {position}")


def _load_essays(essays_path: str) -> list[list[str]]:
    """Load Paul Graham essays and sentence-tokenize them."""
    essays = []
    with open(essays_path, "r") as f:
        for line in f:
            text = json.loads(line.strip())["text"]
            text = re.sub(r"\s+", " ", text)
            essays.append(sent_tokenize(text))
    return essays


# ---------------------------------------------------------------------------
# TokenizeableNIAHExample
# ---------------------------------------------------------------------------

class TokenizeableNIAHExample(TokenizeableExample):
    """A single NIAH example that can be tokenized and fitted to a target context length."""

    def __init__(
        self,
        needles: list[str],
        needle_depths: list[float],
        haystack_sentences: list[str],
        system_prompt: str,
        prompt_template: str,
        question_position: str,
        question_str: str,
        # Metadata passed through for to_dict()
        pos_needles: list[str],
        pos_depths: list[float],
        answer_str: str,
        neg_answer_str: str,
    ):
        super().__init__(max_k=len(haystack_sentences))
        self.needles = needles
        self.needle_depths = needle_depths
        self.haystack_sentences = haystack_sentences
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.question_position = question_position
        self.question_str = question_str
        self.pos_needles = pos_needles
        self.pos_depths = pos_depths
        self.answer_str = answer_str
        self.neg_answer_str = neg_answer_str

    def average_length_per_item(self, tokenizer: PreTrainedTokenizerBase) -> float:
        sample_size = min(20, len(self.haystack_sentences))
        sample_strs = self.haystack_sentences[:sample_size]
        encodings = tokenizer(sample_strs, add_special_tokens=False)
        lengths = [len(ids) + 1 for ids in encodings["input_ids"]]  # +1 for join space
        return float(np.mean(lengths))

    def build_messages(self) -> list[dict[str, str]]:
        haystack = list(self.haystack_sentences[: self.k])
        ordered, _ = insert_into_list(haystack, list(self.needles), list(self.needle_depths))
        context = " ".join(ordered)

        user_prompt = self.prompt_template.replace("{context}", context)
        user_prompt = _fill_niah_question_placeholders(user_prompt, self.question_str, self.question_position)

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
            "question": self.question_str,
            "answer": self.answer_str,
            "neg_answer": self.neg_answer_str,
            "pos_docs": self.pos_needles,
            "pos_doc_depths": self.pos_depths,
            "type": "multi_niah",
        }


# ---------------------------------------------------------------------------
# MultiNIAHDataset
# ---------------------------------------------------------------------------

class MultiNIAHDataset(Dataset):
    """
    Map-style, deterministic NIAH dataset with variable K/V/Q structure per example.

    Per-example augmentations (all deterministic via seeded RNG):
      - NIAH config (K_total, V, K_query): sampled uniformly from valid space
      - Value type (numbers / words / uuids)
      - Instruction variant
      - Question position (end / beginning / both)
      - Needle depths: uniform [0, 1]
      - Essay ordering: shuffled
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
        instruction_variants: list[int] | None = None,
        value_types: list[str] | None = None,
        essays_path: str | None = None,
        k_total_range: tuple[int, int] = (1, 10),
        v_range: tuple[int, int] = (1, 5),
        max_total_needles: int = 16,
        k_query_max: int = 6,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_examples = int(n_examples)
        self.target_context_min, self.target_context_max = normalize_context_range(
            target_context, context_length_max
        )
        self.system_prompt = system_prompt
        self.value_types = value_types or ["numbers", "uuids"]

        # Question position sampling
        if question_position_weights is None:
            raise ValueError("question_position_weights must be provided")
        qp_weights = question_position_weights
        self._qp_strategies = list(qp_weights.keys())
        self._qp_weights = list(qp_weights.values())

        # Pre-enumerate valid NIAH configs
        self.valid_configs = _enumerate_valid_configs(
            k_total_range, v_range, max_total_needles, k_query_max,
        )

        # Load prompt variants: list of (prompt, prompt_singular, needle, needle_singular, question, question_singular)
        self.variants: list[tuple[str, str, str, str, str, str]] = []
        self._load_variants(prompts_dir, instruction_variants)
        if not self.variants:
            raise ValueError(f"No prompt variants found in {prompts_dir}")

        # Load essays
        if essays_path is None:
            essays_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "data", "niah", "essays.jsonl")
            )
        self.essays = _load_essays(essays_path)

        # Lazy-computed stats (deferred past DataLoader fork)
        self._avg_sentence_length: float | None = None
        self._max_num_sentences: int | None = None

        # Pre-sample per-example seeds
        ss = np.random.SeedSequence(int(seed))
        children = ss.spawn(self.n_examples)
        self.base_seeds: list[int] = [int(cs.generate_state(1, dtype=np.uint32)[0]) for cs in children]

    def _load_variants(self, prompts_dir: str, instruction_variants: list[int] | None = None) -> None:
        """Load prompt variant files from prompts_dir/{variant_num}/."""
        if not os.path.isdir(prompts_dir):
            raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

        allowed = set(instruction_variants) if instruction_variants is not None else None
        for variant_num in sorted(os.listdir(prompts_dir)):
            variant_dir = os.path.join(prompts_dir, variant_num)
            if not os.path.isdir(variant_dir):
                continue
            if allowed is not None:
                try:
                    if int(variant_num) not in allowed:
                        continue
                except ValueError:
                    continue

            files = {}
            for fname in [
                "prompt.txt", "prompt_singular.txt",
                "needle.txt", "needle_singular.txt",
                "question.txt", "question_singular.txt",
            ]:
                fpath = os.path.join(variant_dir, fname)
                if not os.path.isfile(fpath):
                    break
                with open(fpath, "r") as f:
                    files[fname] = f.read()
            else:
                # All 6 files found
                self.variants.append((
                    files["prompt.txt"],
                    files["prompt_singular.txt"],
                    files["needle.txt"],
                    files["needle_singular.txt"],
                    files["question.txt"],
                    files["question_singular.txt"],
                ))

    def _ensure_stats_computed(self) -> None:
        """Lazily compute avg sentence length (deferred past DataLoader fork)."""
        if self._avg_sentence_length is not None:
            return
        sample = []
        for essay in self.essays[:5]:
            sample.extend(essay[:100])
        encodings = self.tokenizer(sample, add_special_tokens=False)
        lengths = [len(ids) + 1 for ids in encodings["input_ids"]]
        self._avg_sentence_length = float(np.mean(lengths))
        # 2x safety margin for max sentences needed
        self._max_num_sentences = int(self.target_context_max * 2.0 / self._avg_sentence_length)

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= self.n_examples:
            raise IndexError(idx)

        self._ensure_stats_computed()

        seed_base = self.base_seeds[idx]
        py_rng = random.Random(seed_base)
        np_rng = np.random.default_rng(seed_base)
        sampled_target_context = sample_target_context(
            seed_base, self.target_context_min, self.target_context_max
        )

        # --- 1. Sample NIAH configuration ---
        config_idx = py_rng.randrange(len(self.valid_configs))
        k_total, v, k_query = self.valid_configs[config_idx]

        # --- 2. Sample value type ---
        value_type = py_rng.choice(self.value_types)

        # --- 3. Sample prompt variant ---
        variant_idx = py_rng.randrange(len(self.variants))
        (prompt_tpl, prompt_singular_tpl,
         needle_tpl, needle_singular_tpl,
         question_tpl, question_singular_tpl) = self.variants[variant_idx]

        # --- 4. Sample question position ---
        question_position = py_rng.choices(self._qp_strategies, weights=self._qp_weights, k=1)[0]

        # --- 5. Select singular/plural templates ---
        # Needle singular/plural depends on V (values per key):
        #   V=1 -> "The special magic number for X is: Y."
        #   V>1 -> "One of the special magic numbers for X is: Y."
        # Prompt/question singular/plural depends on total answer count:
        #   K_query*V == 1 -> "What is the special magic number..."
        #   K_query*V > 1  -> "What are all the special magic numbers..."
        # Plural prompt when V>1 OR K_query>1.
        singular_needle = (v == 1)
        singular_question = (k_query * v == 1)

        active_prompt = prompt_singular_tpl if singular_question else prompt_tpl
        active_needle = needle_singular_tpl if singular_needle else needle_tpl
        active_question = question_singular_tpl if singular_question else question_tpl
        # Display strings: needle and prompt may differ when V=1 but K_query > 1
        type_display_needle = _type_needle_v_str(value_type, singular_needle)
        type_display_prompt = _type_needle_v_str(value_type, singular_question)

        # --- 6. Generate keys ---
        keys = _generate_random_words(k_total, py_rng)

        # --- 7. Generate needles ---
        key_value_map, all_needles = _generate_needles(
            keys, v, value_type, active_needle, type_display_needle, py_rng,
        )

        # --- 8. Split pos/neg ---
        pos_keys = keys[:k_query]
        neg_keys = keys[k_query:]

        pos_needles = []
        for key in pos_keys:
            for val in key_value_map[key]:
                pos_needles.append(active_needle.format(
                    key=key, value=val, type_needle_v=type_display_needle,
                ))
        neg_needles = []
        for key in neg_keys:
            for val in key_value_map[key]:
                neg_needles.append(active_needle.format(
                    key=key, value=val, type_needle_v=type_display_needle,
                ))

        answer_values = [val for key in pos_keys for val in key_value_map[key]]
        neg_answer_values = [val for key in neg_keys for val in key_value_map[key]]
        answer_str = " ".join(answer_values)
        neg_answer_str = " ".join(neg_answer_values)

        # --- 9. Build question string and fill type_needle_v in prompt ---
        query_str = ", ".join(pos_keys)
        question_str = active_question.format(type_needle_v=type_display_prompt, query=query_str)
        # Fill {type_needle_v} in prompt template but leave {context}, {question_start}, {question_end}
        prompt_filled = active_prompt.replace("{type_needle_v}", type_display_prompt)

        # --- 10. Build haystack (shuffled essay sentences) ---
        essay_inds = list(range(len(self.essays)))
        py_rng.shuffle(essay_inds)
        haystack: list[str] = []
        for ei in essay_inds:
            haystack.extend(self.essays[ei])
            if len(haystack) >= self._max_num_sentences:
                break

        # --- 11. Sample needle depths ---
        needle_depths = np_rng.uniform(0.0, 1.0, size=len(all_needles)).tolist()
        pos_depths = needle_depths[: k_query * v]

        # --- 12. Create tokenizeable example ---
        token_example = TokenizeableNIAHExample(
            needles=all_needles,
            needle_depths=needle_depths,
            haystack_sentences=haystack,
            system_prompt=self.system_prompt,
            prompt_template=prompt_filled,
            question_position=question_position,
            question_str=question_str,
            pos_needles=pos_needles,
            pos_depths=pos_depths,
            answer_str=answer_str,
            neg_answer_str=neg_answer_str,
        )

        # --- 13. Fit to target context ---
        input_length = token_example.set_largest_k(
            self.tokenizer, sampled_target_context, initial_step_size=16, min_step_size=1,
        )

        # --- 14. Build result ---
        result = token_example.to_dict()
        result["id"] = idx
        result["target_context"] = sampled_target_context
        result["input_length"] = input_length
        result["k_used"] = token_example.k
        result["instruction_variant"] = variant_idx
        result["question_position"] = question_position
        result["value_type"] = value_type
        result["k_total"] = k_total
        result["v_per_key"] = v
        result["k_query"] = k_query
        result["settings"] = f"q{k_query}_k{k_total}_v{v}"
        return result
