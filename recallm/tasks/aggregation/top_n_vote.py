"""
Top-N Vote dataset for final training: synthetic top-N frequency estimation task.

Extends majority_vote from single-winner (N=1) to multi-answer (N=3,5,10).
Teaches the model that recall is counterproductive for aggregation tasks,
and that it must output multiple aggregated results (not just one winner).

Augmentations:
  - Difficulty levels: easy/medium/hard (controls N, candidate count, frequency separation)
  - Instruction variation: multiple prompt templates (sampled per example)
  - Question position variation: end, beginning, both (sampled per example)

Difficulty presets:
  Easy   — top-3 from 8-12 candidates, steep Zipf (alpha=1.5), clear separation
  Medium — top-5 from 12-20 candidates, moderate Zipf (alpha=1.0)
  Hard   — top-10 from 20-30 candidates, near-uniform (alpha=0.5), minimal separation
"""

import os
import random
import string
from typing import Any

import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from recallm.tasks.base import (
    TokenizeableExample,
    normalize_context_range,
    sample_target_context,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_QUESTION_POSITION_WEIGHTS = {"end": 0.6, "beginning": 0.2, "both": 0.2}
DEFAULT_DIFFICULTY_WEIGHTS = {"easy": 0.45, "medium": 0.45, "hard": 0.10}

# Option naming schemes (expanded for larger candidate pools)
_LETTER_OPTIONS = [f"Proposal {chr(65 + i)}" for i in range(26)]  # Proposal A..Z

_NUMBER_OPTIONS = [f"Option {i + 1}" for i in range(40)]  # Option 1..40

_WORD_OPTIONS = [
    "Phoenix", "Cascade", "Vertex", "Nebula", "Horizon", "Summit",
    "Eclipse", "Vanguard", "Zenith", "Meridian", "Catalyst", "Pinnacle",
    "Frontier", "Apex", "Tempest", "Solstice", "Quantum", "Prism",
    "Ember", "Radiant", "Aurora", "Cipher", "Falcon", "Granite",
    "Harbor", "Ivory", "Jasper", "Kinetic", "Lunar", "Monarch",
    "Nomad", "Obsidian", "Polaris", "Quartz", "Relic", "Sapphire",
    "Tundra", "Umbra", "Voyager", "Wildfire",
]  # 40 total

# Name pools for delegate naming (reused from majority_vote)
_FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn",
    "Avery", "Harper", "Sage", "River", "Blake", "Drew", "Finley",
    "Hayden", "Kai", "Lane", "Marley", "Parker", "Reese", "Skyler",
    "Sloane", "Tatum", "Val", "Winter", "Zion", "Arden", "Blair",
    "Cameron", "Dana", "Ellis", "Flynn", "Grey", "Harlow", "Indigo",
    "Jules", "Keegan", "Lennox", "Milan", "Noel", "Oakley", "Peyton",
    "Raven", "Scout", "Tobin", "Uma", "Wren", "Xander", "Yael", "Zara",
]

_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz",
    "Parker", "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris",
    "Morales", "Murphy", "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan",
    "Cooper", "Peterson", "Bailey", "Reed", "Kelly", "Howard", "Ramos",
    "Kim", "Cox", "Ward", "Richardson", "Watson", "Brooks", "Chavez",
    "Wood", "James", "Bennett", "Gray", "Mendoza", "Ruiz", "Hughes",
    "Price", "Alvarez", "Castillo", "Sanders", "Patel", "Myers", "Long",
    "Ross", "Foster", "Jimenez",
]

_COUNTRIES = [
    "Argentina", "Australia", "Austria", "Belgium", "Bolivia", "Brazil",
    "Canada", "Chile", "China", "Colombia", "Croatia", "Czechia",
    "Denmark", "Ecuador", "Egypt", "Estonia", "Finland", "France",
    "Germany", "Greece", "Hungary", "Iceland", "India", "Indonesia",
    "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Kenya",
    "Latvia", "Malaysia", "Mexico", "Morocco", "Netherlands", "Nigeria",
    "Norway", "Paraguay", "Peru", "Philippines", "Poland", "Portugal",
    "Romania", "Singapore", "Slovenia", "Spain", "Sweden", "Switzerland",
    "Thailand", "Uruguay",
]

# Difficulty presets
DIFFICULTY_LEVELS = {
    "easy": {
        "n_top": 3,
        "n_candidates_range": (8, 12),
        "zipf_alpha": 1.5,
        "min_separation": 3.0,
        "naming": "names",
    },
    "medium": {
        "n_top": 5,
        "n_candidates_range": (12, 20),
        "zipf_alpha": 1.0,
        "min_separation": 1.5,
        "naming": "names",
    },
    "hard": {
        "n_top": 10,
        "n_candidates_range": (20, 30),
        "zipf_alpha": 0.5,
        "min_separation": 1.1,
        "naming": "alphanumeric",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_alphanum_id(length: int, rng: random.Random) -> str:
    chars = string.ascii_lowercase + string.digits
    return "".join(rng.choice(chars) for _ in range(length))


def _generate_delegate_name(naming: str, index: int, rng: random.Random) -> str:
    if naming == "sequential":
        return f"Delegate {index + 1}"
    elif naming == "alphanumeric":
        return f"Delegate {_rand_alphanum_id(5, rng)}"
    raise ValueError(f"Unknown naming scheme: {naming}")


def _generate_named_delegates(n: int, rng: random.Random) -> list[str]:
    """Generate n unique 'First Last (Country)' delegate names."""
    names = set()
    result = []
    while len(result) < n:
        first = rng.choice(_FIRST_NAMES)
        last = rng.choice(_LAST_NAMES)
        country = rng.choice(_COUNTRIES)
        name = f"{first} {last} ({country})"
        if name not in names:
            names.add(name)
            result.append(name)
    return result


def _generate_option_names(n_candidates: int, rng: random.Random) -> list[str]:
    """Pick a naming scheme and generate n option names."""
    if n_candidates <= 26:
        scheme = rng.choice(["letters", "numbers", "words"])
    else:
        # Letters max out at 26; use numbers or words for larger pools
        scheme = rng.choice(["numbers", "words"])

    if scheme == "letters":
        return _LETTER_OPTIONS[:n_candidates]
    elif scheme == "numbers":
        return _NUMBER_OPTIONS[:n_candidates]
    else:
        pool = list(_WORD_OPTIONS)
        rng.shuffle(pool)
        return pool[:n_candidates]


def _distribute_votes_zipf(
    n_delegates: int,
    n_candidates: int,
    n_top: int,
    alpha: float,
    min_separation: float,
    np_rng: np.random.Generator,
) -> list[int]:
    """
    Generate vote counts using a Zipf-like distribution with Dirichlet noise.

    Returns vote_counts sorted descending (index 0 = most votes).
    Guarantees vote_counts[n_top-1] >= min_separation * vote_counts[n_top].
    """
    ranks = np.arange(1, n_candidates + 1, dtype=float)
    raw_weights = 1.0 / (ranks ** alpha)

    # Add Dirichlet noise for per-example variation
    noise = np_rng.dirichlet(np.full(n_candidates, 0.3))
    weights = raw_weights * 0.7 + noise * 0.3
    weights = weights / weights.sum()

    # Sort weights descending to ensure rank ordering
    weights = np.sort(weights)[::-1]

    # Convert to integer vote counts
    raw_counts = weights * n_delegates
    vote_counts = np.round(raw_counts).astype(int)

    # Fix rounding to match exactly n_delegates
    diff = n_delegates - vote_counts.sum()
    if diff > 0:
        # Add votes to the largest candidates first
        for i in range(abs(diff)):
            vote_counts[i % n_candidates] += 1
    elif diff < 0:
        # Remove votes from the smallest candidates first
        for i in range(abs(diff)):
            idx = n_candidates - 1 - (i % n_candidates)
            vote_counts[idx] = max(0, vote_counts[idx] - 1)

    # Ensure separation constraint: votes[n_top-1] >= min_separation * votes[n_top]
    if n_top < n_candidates:
        for _ in range(50):  # max redistribution iterations
            boundary_top = vote_counts[n_top - 1]
            boundary_next = vote_counts[n_top]
            if boundary_next == 0 or boundary_top >= min_separation * boundary_next:
                break
            # Transfer one vote from a random tail candidate to a random top candidate
            tail_indices = [i for i in range(n_top, n_candidates) if vote_counts[i] > 0]
            if not tail_indices:
                break
            donor = tail_indices[int(np_rng.integers(len(tail_indices)))]
            receiver = int(np_rng.integers(n_top))
            vote_counts[donor] -= 1
            vote_counts[receiver] += 1
            # Re-sort to maintain descending order
            vote_counts = np.sort(vote_counts)[::-1]

    # Ensure all counts are non-negative and total is correct
    vote_counts = np.maximum(vote_counts, 0)
    final_diff = n_delegates - vote_counts.sum()
    if final_diff != 0:
        vote_counts[0] += final_diff

    return vote_counts.tolist()


def _make_question_str(n_top: int) -> str:
    return f"What are the {n_top} options that received the most votes?"


def _fill_question_placeholders(prompt_template: str, position: str, n_top: int) -> str:
    q = _make_question_str(n_top)
    if position == "end":
        return prompt_template.replace("{question_start}", "").replace("{question_end}", q + "\n")
    elif position == "beginning":
        return prompt_template.replace("{question_start}", q + "\n\n").replace("{question_end}", "")
    elif position == "both":
        return prompt_template.replace("{question_start}", q + "\n\n").replace("{question_end}", q + "\n")
    raise ValueError(f"Unknown question position: {position}")


# ---------------------------------------------------------------------------
# TokenizeableTopNVoteExample
# ---------------------------------------------------------------------------

class TokenizeableTopNVoteExample(TokenizeableExample):
    """A single top-N vote example that can be tokenized and fitted to a target context length."""

    def __init__(
        self,
        delegate_names: list[str],
        option_names: list[str],
        n_candidates: int,
        n_top: int,
        zipf_alpha: float,
        min_separation: float,
        system_prompt: str,
        prompt_template: str,
        question_position: str,
        np_rng: np.random.Generator,
        py_rng: random.Random,
    ):
        super().__init__(max_k=len(delegate_names))
        self.delegate_names = delegate_names
        self.option_names = option_names
        self.n_candidates = n_candidates
        self.n_top = n_top
        self.zipf_alpha = zipf_alpha
        self.min_separation = min_separation
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.question_position = question_position
        self.np_rng = np_rng
        self.py_rng = py_rng

        # Set after set_largest_k + finalize
        self.top_n_names: list[str] | None = None
        self._items_str: str | None = None

    def average_length_per_item(self, tokenizer: PreTrainedTokenizerBase) -> float:
        sample_size = min(20, len(self.delegate_names))
        sample_strs = [
            f"{self.delegate_names[i]} votes for: {self.option_names[0]}\n"
            for i in range(sample_size)
        ]
        encodings = tokenizer(sample_strs, add_special_tokens=False)
        lengths = [len(ids) for ids in encodings["input_ids"]]
        return float(np.mean(lengths))

    def finalize_votes(self):
        """After set_largest_k, compute vote distribution and build items string."""
        n_delegates = self.k
        if n_delegates == 0:
            self.top_n_names = self.option_names[:self.n_top]
            self._items_str = ""
            return

        vote_counts = _distribute_votes_zipf(
            n_delegates, self.n_candidates, self.n_top,
            self.zipf_alpha, self.min_separation, self.np_rng,
        )

        # vote_counts is sorted descending; option_names[i] gets vote_counts[i]
        # Build delegate-vote pairs
        assignments = []
        for opt_idx, count in enumerate(vote_counts):
            for _ in range(count):
                assignments.append(self.option_names[opt_idx])

        self.py_rng.shuffle(assignments)

        lines = []
        for i in range(n_delegates):
            lines.append(f"{self.delegate_names[i]} votes for: {assignments[i]}")

        # Top-N winners are the first n_top option names (sorted by vote count desc)
        self.top_n_names = self.option_names[:self.n_top]
        self._items_str = "\n".join(lines)

    def build_messages(self) -> list[dict[str, str]]:
        if self._items_str is None:
            # Placeholder for tokenized_length estimation during set_largest_k
            lines = [
                f"{self.delegate_names[i]} votes for: {self.option_names[i % self.n_candidates]}"
                for i in range(self.k)
            ]
            items_str = "\n".join(lines)
        else:
            items_str = self._items_str

        user_prompt = self.prompt_template.replace("{items}", items_str)
        user_prompt = user_prompt.replace("{n_top}", str(self.n_top))
        user_prompt = user_prompt.replace("{total_candidates}", str(self.n_candidates))
        user_prompt = _fill_question_placeholders(user_prompt, self.question_position, self.n_top)

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def tokenized_length(self, tokenizer: PreTrainedTokenizerBase) -> int:
        messages = self.build_messages()
        return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

    def to_dict(self) -> dict[str, Any]:
        messages = self.build_messages()
        neg_names = self.option_names[self.n_top:]  # non-winning options
        return {
            "prompt": messages,
            "question": _make_question_str(self.n_top),
            "answer": "|||".join(self.top_n_names),
            "neg_answer": "|||".join(neg_names),
            "pos_docs": [],
            "type": "top_n_vote",
        }


# ---------------------------------------------------------------------------
# TopNVoteDataset
# ---------------------------------------------------------------------------

class TopNVoteDataset(Dataset):
    """
    Map-style, deterministic top-N vote dataset with augmentations.

    Augmentations (all deterministic per example via seeded RNG):
      - Difficulty level: easy (top-3) / medium (top-5) / hard (top-10)
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

        # Estimate max delegates needed to fill target context
        self.max_delegates = int(((self.target_context_max - 200) / 12) * 2)

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

        n_top = preset["n_top"]
        n_candidates = py_rng.randint(*preset["n_candidates_range"])
        zipf_alpha = preset["zipf_alpha"]
        min_separation = preset["min_separation"]
        naming = preset["naming"]

        # --- Generate option names ---
        option_names = _generate_option_names(n_candidates, py_rng)
        py_rng.shuffle(option_names)  # Randomize which names map to which ranks

        # --- Pre-generate delegate names (more than needed) ---
        if naming == "names":
            delegate_names = _generate_named_delegates(self.max_delegates, py_rng)
        else:
            delegate_names = [_generate_delegate_name(naming, i, py_rng) for i in range(self.max_delegates)]

        # --- Build tokenizeable example ---
        token_example = TokenizeableTopNVoteExample(
            delegate_names=delegate_names,
            option_names=option_names,
            n_candidates=n_candidates,
            n_top=n_top,
            zipf_alpha=zipf_alpha,
            min_separation=min_separation,
            system_prompt=self.system_prompt,
            prompt_template=prompt_template,
            question_position=question_position,
            np_rng=np_rng,
            py_rng=py_rng,
        )

        input_length = token_example.set_largest_k(
            self.tokenizer, sampled_target_context, initial_step_size=10, min_step_size=1
        )

        # Finalize vote distribution for the actual k delegates.
        # The uniform-cycling placeholder in build_messages() can underestimate
        # the finalized length when the Zipf distribution concentrates votes on
        # longer option names.  Save RNG state so we can re-finalize
        # deterministically if the finalized content overshoots target_context.
        np_state = np_rng.bit_generator.state
        py_state = py_rng.getstate()
        token_example.finalize_votes()

        # Validate: shrink k until finalized content fits target_context.
        # Use estimated tokens-per-delegate for large jumps, then step by 1.
        avg_toks = token_example.average_length_per_item(self.tokenizer)
        while token_example.k > 0:
            actual_length = token_example.tokenized_length(self.tokenizer)
            if actual_length <= sampled_target_context:
                input_length = actual_length
                break
            overshoot = actual_length - sampled_target_context
            step = max(1, int(overshoot / avg_toks) - 1)  # conservative: undershoot the jump
            token_example.k = max(0, token_example.k - step)
            np_rng.bit_generator.state = np_state
            py_rng.setstate(py_state)
            token_example.finalize_votes()

        result = token_example.to_dict()
        result["id"] = idx
        result["target_context"] = sampled_target_context
        result["input_length"] = input_length
        result["k_used"] = token_example.k
        result["instruction_variant"] = variant_idx
        result["question_position"] = question_position
        result["difficulty"] = difficulty
        result["n_candidates"] = n_candidates
        result["n_top"] = n_top
        result["zipf_alpha"] = zipf_alpha
        return result
