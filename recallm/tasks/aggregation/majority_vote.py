"""
Majority Vote dataset for final training: synthetic vote-counting aggregation task.

Teaches the model that recall is counterproductive for aggregation tasks.
Delegates vote for one of K options; the model must determine the winner.
Recalling individual votes one-by-one would exhaust the token budget.

Augmentations:
  - Difficulty levels: easy/medium/hard (controls K, margin, naming)
  - Instruction variation: multiple prompt templates (sampled per example)
  - Question position variation: end, beginning, both (sampled per example)

Difficulty presets:
  Easy   — 2 candidates, 60-70% winner margin, sequential naming
  Medium — 3-5 candidates, 40-55% winner margin, random first names
  Hard   — 6-10 candidates, 30-40% winner margin, random alphanumeric IDs
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

# Option naming schemes
_LETTER_OPTIONS = [f"Proposal {chr(65 + i)}" for i in range(26)]  # Proposal A..Z
_NUMBER_OPTIONS = [f"Option {i + 1}" for i in range(20)]  # Option 1..20
_WORD_OPTIONS = [
    "Phoenix", "Cascade", "Vertex", "Nebula", "Horizon", "Summit",
    "Eclipse", "Vanguard", "Zenith", "Meridian", "Catalyst", "Pinnacle",
    "Frontier", "Apex", "Tempest", "Solstice", "Quantum", "Prism",
    "Ember", "Radiant",
]

# Name pools for medium difficulty delegate naming (first × last × country = 50 × 100 × 50 = 250K combos)
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
        "n_candidates_range": (2, 2),
        "winner_share_range": (0.60, 0.70),
        "naming": "sequential",
    },
    "medium": {
        "n_candidates_range": (3, 5),
        "winner_share_range": (0.40, 0.55),
        "naming": "names",
    },
    "hard": {
        "n_candidates_range": (6, 10),
        "winner_share_range": (0.30, 0.40),
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
    """Generate n unique 'First Last (Country)' delegate names via incremental sampling."""
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
    scheme = rng.choice(["letters", "numbers", "words"])
    if scheme == "letters":
        return _LETTER_OPTIONS[:n_candidates]
    elif scheme == "numbers":
        return _NUMBER_OPTIONS[:n_candidates]
    else:
        pool = list(_WORD_OPTIONS)
        rng.shuffle(pool)
        return pool[:n_candidates]


def _distribute_votes(n_delegates: int, n_candidates: int, winner_share: float,
                      np_rng: np.random.Generator) -> list[int]:
    """
    Return vote counts per candidate such that candidate 0 is the winner.
    """
    winner_votes = max(1, round(n_delegates * winner_share))

    remaining = n_delegates - winner_votes
    if n_candidates <= 1:
        return [n_delegates]

    # Distribute remaining votes among losers via Dirichlet
    n_losers = n_candidates - 1
    if remaining <= 0:
        loser_votes = [0] * n_losers
    else:
        # Dirichlet with alpha=1 gives uniform random splits
        shares = np_rng.dirichlet(np.ones(n_losers))
        raw = shares * remaining
        loser_votes = [int(round(v)) for v in raw]
        # Fix rounding to match exactly
        diff = remaining - sum(loser_votes)
        for i in range(abs(diff)):
            loser_votes[i % n_losers] += 1 if diff > 0 else -1

    # Ensure winner actually wins
    max_loser = max(loser_votes) if loser_votes else 0
    if max_loser >= winner_votes:
        # Steal votes from largest loser to ensure winner wins
        deficit = max_loser - winner_votes + 1
        largest_idx = loser_votes.index(max_loser)
        loser_votes[largest_idx] -= deficit
        winner_votes += deficit

    return [winner_votes] + loser_votes


def _make_question_str() -> str:
    return "Which option received the most votes?"


def _fill_question_placeholders(prompt_template: str, position: str) -> str:
    q = _make_question_str()
    if position == "end":
        return prompt_template.replace("{question_start}", "").replace("{question_end}", q + "\n")
    elif position == "beginning":
        return prompt_template.replace("{question_start}", q + "\n\n").replace("{question_end}", "")
    elif position == "both":
        return prompt_template.replace("{question_start}", q + "\n\n").replace("{question_end}", q + "\n")
    raise ValueError(f"Unknown question position: {position}")


# ---------------------------------------------------------------------------
# TokenizeableMajorityVoteExample
# ---------------------------------------------------------------------------

class TokenizeableMajorityVoteExample(TokenizeableExample):
    """A single majority vote example that can be tokenized and fitted to a target context length."""

    def __init__(
        self,
        delegate_names: list[str],
        option_names: list[str],
        n_candidates: int,
        winner_share: float,
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
        self.winner_share = winner_share
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.question_position = question_position
        self.np_rng = np_rng
        self.py_rng = py_rng

        # These get set after set_largest_k + finalize
        self.winner_name: str | None = None
        self._items_str: str | None = None

    def average_length_per_item(self, tokenizer: PreTrainedTokenizerBase) -> float:
        # Estimate tokens per delegate line by sampling a few
        sample_size = min(20, len(self.delegate_names))
        sample_strs = [
            f"{self.delegate_names[i]} votes for: {self.option_names[0]}\n"
            for i in range(sample_size)
        ]
        encodings = tokenizer(sample_strs, add_special_tokens=False)
        lengths = [len(ids) for ids in encodings["input_ids"]]
        return float(np.mean(lengths))

    def finalize_votes(self):
        """After set_largest_k, compute the actual vote distribution and build items string."""
        n_delegates = self.k
        if n_delegates == 0:
            self.winner_name = self.option_names[0]
            self._items_str = ""
            return

        vote_counts = _distribute_votes(n_delegates, self.n_candidates, self.winner_share, self.np_rng)

        # Build delegate-vote pairs: expand counts to per-delegate assignments
        assignments = []
        for opt_idx, count in enumerate(vote_counts):
            for _ in range(count):
                assignments.append(self.option_names[opt_idx])

        # Shuffle the assignments
        self.py_rng.shuffle(assignments)

        # Build lines
        lines = []
        for i in range(n_delegates):
            lines.append(f"{self.delegate_names[i]} votes for: {assignments[i]}")

        self.winner_name = self.option_names[0]  # candidate 0 is always the winner
        self._items_str = "\n".join(lines)

    def build_messages(self) -> list[dict[str, str]]:
        if self._items_str is None:
            # Distribution-aware placeholder for tokenized_length estimation
            # during set_largest_k.  Approximate the actual vote distribution:
            # winner (option_names[0]) gets ~winner_share of lines, losers
            # share the rest uniformly.  This is much closer to the finalized
            # length than the old uniform-cycling placeholder.
            n_winner = max(1, round(self.k * self.winner_share))
            n_remaining = self.k - n_winner
            loser_names = self.option_names[1:self.n_candidates]
            lines = [
                f"{self.delegate_names[i]} votes for: {self.option_names[0]}"
                for i in range(n_winner)
            ]
            if loser_names and n_remaining > 0:
                lines += [
                    f"{self.delegate_names[n_winner + i]} votes for: {loser_names[i % len(loser_names)]}"
                    for i in range(n_remaining)
                ]
            items_str = "\n".join(lines)
        else:
            items_str = self._items_str

        user_prompt = self.prompt_template.replace("{items}", items_str)
        user_prompt = _fill_question_placeholders(user_prompt, self.question_position)

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
            "question": _make_question_str(),
            "answer": self.winner_name,
            "pos_docs": [],
            "type": "majority_vote",
        }


# ---------------------------------------------------------------------------
# MajorityVoteDataset
# ---------------------------------------------------------------------------

class MajorityVoteDataset(Dataset):
    """
    Map-style, deterministic majority vote dataset with augmentations.

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

        # Estimate max delegates needed to fill target context
        # ~12 tokens per line "Delegate xxxxx votes for: Proposal A\n"
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

        n_candidates = py_rng.randint(*preset["n_candidates_range"])
        winner_share = py_rng.uniform(*preset["winner_share_range"])
        naming = preset["naming"]

        # --- Generate option names ---
        option_names = _generate_option_names(n_candidates, py_rng)
        py_rng.shuffle(option_names)  # Randomize which name maps to the winner (always index 0 in vote_counts)

        # --- Pre-generate delegate names (more than needed) ---
        if naming == "names":
            delegate_names = _generate_named_delegates(self.max_delegates, py_rng)
        else:
            delegate_names = [_generate_delegate_name(naming, i, py_rng) for i in range(self.max_delegates)]

        # --- Build tokenizeable example ---
        token_example = TokenizeableMajorityVoteExample(
            delegate_names=delegate_names,
            option_names=option_names,
            n_candidates=n_candidates,
            winner_share=winner_share,
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
        # The placeholder in build_messages() approximates the vote distribution
        # but cannot match it exactly (Dirichlet noise, name-length skew).
        # Save RNG state so we can re-finalize deterministically if the
        # finalized content overshoots target_context.
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
        result["winner_share"] = round(winner_share, 3)
        return result
