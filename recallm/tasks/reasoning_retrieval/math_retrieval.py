"""
Math retrieval dataset for final training: solve math → look up key → retrieve value.

Unifies the old retrieval_math (easy, text-line KV) and math_retrieval (hard, JSON)
into a single dataset with format variation and variable difficulty.

Augmentations:
  - KV format variation: lines, json, csv (sampled per example)
  - Instruction variation: multiple prompt templates per format (sampled per example)
  - Question position variation: end, beginning, both (sampled per example)
  - Difficulty variation: n_terms sampled from [n_terms_min, n_terms_max] per example

"""

import json
import math
import os
import random
import string
from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from recallm.tasks.base import (
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


def get_key_range(ans: int, dict_size: int) -> tuple[int, int]:
    """Return (key_min, key_max) centered around the answer."""
    return ans - dict_size, ans + dict_size


def get_factors(val: int) -> list[int]:
    """Return sorted list of positive factors of |val|."""
    if val == 0:
        return [1]
    val = abs(val)
    factors = []
    for i in range(1, int(val**0.5) + 1):
        if val % i == 0:
            factors.append(i)
            if val // i != i:
                factors.append(val // i)
    return sorted(factors)


DEFAULT_QUESTION_POSITION_WEIGHTS = {"end": 0.6, "beginning": 0.2, "both": 0.2}


# ---------------------------------------------------------------------------
# Math problem generators
# ---------------------------------------------------------------------------

@dataclass
class MathProblem:
    """Result of sampling a math-retrieval problem."""
    problem_str: str       # LaTeX-formatted problem statement
    answer: int            # The correct integer key
    retrieval_val: str     # The value to retrieve (random alphanumeric)
    neg_keys: list[int]    # Negative keys (distractors)
    neg_vals: list[str]    # Corresponding negative values


class SimpleLinearEq:
    """
    Generate problems of the form a(x+b) + c(x+d) + ... = e*x + f.
    Solution is always a unique integer.
    """

    def __init__(self, n_terms: int, dict_val_size: int, coeff_min: int = -10, coeff_max: int = 10):
        self.n_terms = int(n_terms)
        self.dict_val_size = int(dict_val_size)
        self.coeff_min = int(coeff_min)
        self.coeff_max = int(coeff_max)

    def _rand_coeff(self, np_rng, nonzero: bool = False) -> int:
        if nonzero:
            ans = 0
            while ans == 0:
                ans = int(np_rng.integers(self.coeff_min, self.coeff_max))
            return ans
        return int(np_rng.integers(self.coeff_min, self.coeff_max))

    def _rand_ans(self, np_rng) -> int:
        return int(np_rng.integers(self.coeff_min * 10, self.coeff_max * 10))

    def _build_eq(self, py_rng: random.Random, np_rng) -> tuple:
        """Build a solvable linear equation. Returns (sympy.Eq, answer)."""
        while True:
            ans = self._rand_ans(np_rng)
            x = sp.Symbol("x")
            lhs_terms = []

            for _ in range(self.n_terms):
                a = self._rand_coeff(np_rng)
                b = self._rand_coeff(np_rng)
                if py_rng.choice([True, False]):
                    term = sp.Mul(a, sp.Add(x, b, evaluate=False), evaluate=False)
                else:
                    term = sp.Add(sp.Mul(a, x, evaluate=False), b, evaluate=False)
                lhs_terms.append(term)

            lhs = sp.Add(*lhs_terms, evaluate=False)
            rhs = sp.simplify(lhs)

            x_diff = self._rand_coeff(np_rng, nonzero=True)
            rhs = rhs - x_diff * x + x_diff * ans
            eq = sp.Eq(lhs, rhs)

            try:
                solutions = sp.solve(eq, x)
            except ZeroDivisionError:
                continue

            if len(solutions) == 1 and solutions[0] == ans:
                return eq, ans

    def sample(self, dict_size: int, py_rng: random.Random, np_rng) -> MathProblem:
        eq, ans = self._build_eq(py_rng, np_rng)
        retrieval_val = rand_alphanumeric_str(self.dict_val_size, py_rng)
        problem_str = f"Solve for $x$ in $${sp.latex(eq)}$$"

        key_min, key_max = get_key_range(ans, dict_size)
        raw = np_rng.choice(key_max - key_min, size=dict_size * 2, replace=False) + key_min
        neg_keys = [int(k) for k in raw.tolist() if int(k) != ans][:dict_size - 1]
        neg_vals = [rand_alphanumeric_str(self.dict_val_size, py_rng) for _ in range(len(neg_keys))]

        return MathProblem(
            problem_str=problem_str,
            answer=ans,
            retrieval_val=retrieval_val,
            neg_keys=neg_keys,
            neg_vals=neg_vals,
        )


class SimpleExpr:
    """
    Generate simple arithmetic expressions with +, -, *, / that evaluate to an integer.
    """

    def __init__(self, n_numbers: int, dict_val_size: int):
        self.n_numbers = int(n_numbers)
        self.dict_val_size = int(dict_val_size)
        self.ops = ["+", "-", "*", "/"]
        self.val_min = 0
        self.val_max = 100

    def _rand_num(self, np_rng) -> int:
        return int(np_rng.integers(self.val_min, self.val_max))

    def sample(self, dict_size: int, py_rng: random.Random, np_rng) -> MathProblem:
        numbers = [self._rand_num(np_rng)]
        operators = []
        current_term = numbers[0]
        terms = []

        for _ in range(1, self.n_numbers):
            op = py_rng.choice(self.ops)
            if op == "/":
                factors = get_factors(abs(current_term))
                next_num = int(py_rng.choice(factors))
                current_term //= next_num
            elif op == "*":
                next_num = self._rand_num(np_rng)
                current_term *= next_num
            else:
                next_num = self._rand_num(np_rng)
                terms.append(current_term)
                current_term = next_num if op == "+" else -next_num
            operators.append(op)
            numbers.append(next_num)

        terms.append(current_term)

        expr_str = f"{numbers[0]} " + " ".join(
            f"{op} {num}" for op, num in zip(operators, numbers[1:])
        )
        ans = int(sum(terms))
        expr = sp.parse_expr(f"x == {expr_str}", evaluate=False)
        problem_str = f"Solve for $x$: $${sp.latex(expr)}$$"
        retrieval_val = rand_alphanumeric_str(self.dict_val_size, py_rng)

        key_min, key_max = get_key_range(ans, dict_size)
        raw = np_rng.choice(key_max - key_min, size=dict_size * 2, replace=False) + key_min
        neg_keys = [int(k) for k in raw.tolist() if int(k) != ans][:dict_size - 1]
        neg_vals = [rand_alphanumeric_str(self.dict_val_size, py_rng) for _ in range(len(neg_keys))]

        return MathProblem(
            problem_str=problem_str,
            answer=ans,
            retrieval_val=retrieval_val,
            neg_keys=neg_keys,
            neg_vals=neg_vals,
        )


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
# Question positioning
# ---------------------------------------------------------------------------

def _fill_question_placeholders(
    prompt_template: str,
    problem_str: str,
    position: str,
) -> str:
    """Fill {question_start} and {question_end} based on position strategy."""
    q = f"Math problem:\n{problem_str}"
    if position == "end":
        return prompt_template.replace("{question_start}", "").replace("{question_end}", q)
    elif position == "beginning":
        return prompt_template.replace("{question_start}", q + "\n\n").replace("{question_end}", "")
    elif position == "both":
        return prompt_template.replace("{question_start}", q + "\n\n").replace("{question_end}", q)
    else:
        raise ValueError(f"Unknown question position: {position}")


# ---------------------------------------------------------------------------
# TokenizeableMathRetrievalExample
# ---------------------------------------------------------------------------

class TokenizeableMathRetrievalExample(TokenizeableExample):
    """A single math-retrieval example that can be tokenized and fitted to a target context length."""

    def __init__(
        self,
        problem: MathProblem,
        needle_depth: float,
        system_prompt: str,
        prompt_template: str,
        kv_format: str,
        question_position: str,
    ):
        super().__init__(max_k=len(problem.neg_keys))
        self.problem = problem
        self.needle_depth = needle_depth
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.kv_format = kv_format
        self.question_position = question_position
        self._formatter = KV_FORMATTERS[kv_format]

    def average_length_per_item(self, tokenizer: PreTrainedTokenizerBase) -> float:
        sample_size = min(20, len(self.problem.neg_keys))
        sample_strs = []
        for i in range(sample_size):
            k, v = self.problem.neg_keys[i], self.problem.neg_vals[i]
            if self.kv_format == "lines":
                sample_strs.append(f"Key {k}:\n{v}\n\n")
            elif self.kv_format == "json":
                sample_strs.append(f'  "{k}": "{v}",\n')
            elif self.kv_format == "csv":
                sample_strs.append(f"{k},{v}\n")
        encodings = tokenizer(sample_strs, add_special_tokens=False)
        lengths = [len(ids) for ids in encodings["input_ids"]]
        return float(np.mean(lengths))

    def _build_ordered_keys_vals(self) -> tuple[list[int], list[str]]:
        """Return all keys and vals in order, with positive inserted at needle_depth."""
        neg_k = list(self.problem.neg_keys[:self.k])
        neg_v = list(self.problem.neg_vals[:self.k])
        ordered_k, _ = insert_into_list(neg_k, [self.problem.answer], [self.needle_depth])
        ordered_v, _ = insert_into_list(neg_v, [self.problem.retrieval_val], [self.needle_depth])
        return ordered_k, ordered_v

    def build_messages(self) -> list[dict[str, str]]:
        ordered_keys, ordered_vals = self._build_ordered_keys_vals()
        documents = self._formatter(ordered_keys, ordered_vals)

        user_prompt = self.prompt_template.replace("{documents}", documents)
        user_prompt = _fill_question_placeholders(user_prompt, self.problem.problem_str, self.question_position)

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
            pos_doc_str = f"Key {self.problem.answer}:\n{self.problem.retrieval_val}"
        elif self.kv_format == "json":
            pos_doc_str = f'"{self.problem.answer}": "{self.problem.retrieval_val}"'
        elif self.kv_format == "csv":
            pos_doc_str = f"{self.problem.answer},{self.problem.retrieval_val}"
        else:
            pos_doc_str = f"{self.problem.answer}: {self.problem.retrieval_val}"

        return {
            "prompt": messages,
            "question": self.problem.problem_str,
            "answer": self.problem.retrieval_val,
            "math_answer": str(self.problem.answer),
            "pos_doc_depths": [self.needle_depth],
            "pos_docs": [pos_doc_str],
            "type": "math_retrieval",
        }


# ---------------------------------------------------------------------------
# MathRetrievalDataset
# ---------------------------------------------------------------------------

class MathRetrievalDataset(Dataset):
    """
    Map-style, deterministic math-retrieval dataset with augmentations.

    Augmentations (all deterministic per example via seeded RNG):
      - KV format (lines / json / csv): sampled from available prompt variants
      - Instruction phrasing: sampled from variants within the selected format
      - Question position (end / beginning / both): sampled per example
      - Needle depth: uniform random in [0, 1]
      - Negative ordering: shuffled per example
      - Difficulty (n_terms): sampled uniformly from [n_terms_min, n_terms_max]
      - Problem type: SimpleLinearEq (even idx) vs SimpleExpr (odd idx)
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
        kv_formats: list[str] | None = None,
        instruction_variants: list[int] | None = None,
        n_terms_min: int = 2,
        n_terms_max: int = 6,
        coeff_min: int = -10,
        coeff_max: int = 10,
        dict_val_size: int = 10,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_examples = int(n_examples)
        self.target_context_min, self.target_context_max = normalize_context_range(
            target_context, context_length_max
        )
        self.n_terms_min = int(n_terms_min)
        self.n_terms_max = int(n_terms_max)
        self.coeff_min = int(coeff_min)
        self.coeff_max = int(coeff_max)
        self.dict_val_size = int(dict_val_size)
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

        # Estimate max number of KV pairs needed to fill target context
        self.avg_item_length = self.dict_val_size + 5
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

        # --- Sample difficulty ---
        n_terms = py_rng.randint(self.n_terms_min, self.n_terms_max)

        # --- Generate math problem ---
        if idx % 2 == 0:
            generator = SimpleExpr(n_numbers=n_terms, dict_val_size=self.dict_val_size)
        else:
            generator = SimpleLinearEq(
                n_terms=n_terms,
                dict_val_size=self.dict_val_size,
                coeff_min=self.coeff_min,
                coeff_max=self.coeff_max,
            )

        problem = generator.sample(self.max_num_pairs, py_rng=py_rng, np_rng=np_rng)

        # --- Shuffle negatives ---
        perm = list(range(len(problem.neg_keys)))
        py_rng.shuffle(perm)
        problem = MathProblem(
            problem_str=problem.problem_str,
            answer=problem.answer,
            retrieval_val=problem.retrieval_val,
            neg_keys=[problem.neg_keys[i] for i in perm],
            neg_vals=[problem.neg_vals[i] for i in perm],
        )

        # --- Sample needle depth ---
        needle_depth = float(np_rng.uniform(0.0, 1.0))

        # --- Build tokenizeable example ---
        token_example = TokenizeableMathRetrievalExample(
            problem=problem,
            needle_depth=needle_depth,
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
        result["n_terms"] = n_terms
        result["problem_type"] = "simple_expr" if idx % 2 == 0 else "linear_eq"
        return result
