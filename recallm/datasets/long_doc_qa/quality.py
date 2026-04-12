"""
QuALITY long-document QA dataset for final training: single-document reading
comprehension with 4-way multiple-choice questions.

Teaches the model that some tasks involve one continuous narrative with no
pre-segmented passages.  The model must decide what's relevant within a single
long document — fundamentally different from multi-document RAG.

Augmentations (all deterministic per example via seeded RNG):
  - Instruction variation: multiple prompt templates (sampled per example)
  - Question position: end / beginning / both (sampled per example)
  - Choice ordering: shuffled per example (A-D random permutation)
  - Choice formatting: different presentation styles per variant

Source: QuALITY (nyu-mll/quality on GitHub), htmlstripped train/dev splits.
  - 300 articles, ~2,523 questions in train (~8.4 per article)
  - Articles: 2.6K–8.8K tokens (avg ~5K), Project Gutenberg + Open American National Corpus
  - ~50% of questions are annotated "hard"

Train/eval split: train split for training, dev split for eval.
"""

import json
import os
import random
import urllib.request
from typing import Any, Callable

import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GITHUB_BASE = "https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1"
_URLS = {
    "train": f"{_GITHUB_BASE}/QuALITY.v1.0.1.htmlstripped.train",
    "dev": f"{_GITHUB_BASE}/QuALITY.v1.0.1.htmlstripped.dev",
}

# Cache directory: alongside this file
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")

DEFAULT_QUESTION_POSITION_WEIGHTS = {"end": 0.6, "beginning": 0.2, "both": 0.2}

# Choice formatting functions — one per variant index.
CHOICE_FORMATS: list[Callable[[str, str], str]] = [
    lambda letter, text: f"{letter}) {text}",      # variant 0: A) choice
    lambda letter, text: f"{letter}. {text}",       # variant 1: A. choice
    lambda letter, text: f"[{letter}] {text}",      # variant 2: [A] choice
    lambda letter, text: f"- {letter}: {text}",     # variant 3: - A: choice
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_if_needed(split: str) -> str:
    """Download the QuALITY htmlstripped JSONL for *split* and return the local path."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    filename = os.path.basename(_URLS[split])
    local_path = os.path.join(_CACHE_DIR, filename)
    if not os.path.isfile(local_path):
        print(f"Downloading QuALITY {split} split from GitHub...")
        urllib.request.urlretrieve(_URLS[split], local_path)
        print(f"  Saved to {local_path}")
    return local_path


def _load_quality(split: str) -> list[dict]:
    """Load and parse the QuALITY JSONL for *split*."""
    path = _download_if_needed(split)
    articles = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                articles.append(json.loads(line))
    return articles


def _flatten_examples(articles: list[dict]) -> list[dict]:
    """Flatten article-question pairs into individual examples.

    Each returned dict has: article, article_id, title, question, options,
    gold_label (0-indexed int), difficult (bool).
    """
    examples = []
    for article_obj in articles:
        article_text = article_obj["article"]
        article_id = article_obj["article_id"]
        title = article_obj.get("title", "")
        for q_obj in article_obj["questions"]:
            examples.append({
                "article": article_text,
                "article_id": article_id,
                "title": title,
                "question_text": q_obj["question"],
                "options": q_obj["options"],
                "gold_label": q_obj["gold_label"] - 1,  # convert 1-indexed → 0-indexed
                "difficult": bool(q_obj.get("difficult", 0)),
            })
    return examples


def _fill_question_placeholders(
    prompt_template: str,
    question_str: str,
    position: str,
) -> str:
    """Fill {question_start} and {question_end} based on position strategy."""
    if position == "end":
        return prompt_template.replace("{question_start}", "").replace(
            "{question_end}", question_str
        )
    elif position == "beginning":
        return prompt_template.replace(
            "{question_start}", question_str + "\n\n"
        ).replace("{question_end}", "")
    elif position == "both":
        return prompt_template.replace(
            "{question_start}", question_str + "\n\n"
        ).replace("{question_end}", question_str)
    else:
        raise ValueError(f"Unknown question position: {position}")


def _estimate_token_length(
    article: str,
    question_text: str,
    options: list[str],
    system_prompt: str,
    prompt_template: str,
    tokenizer: PreTrainedTokenizerBase,
) -> int:
    """Compute the token length of a fully-rendered example (using 'end' position
    as the representative case, since that's the most common)."""
    # Build a representative MC question string
    letters = ["A", "B", "C", "D"]
    choice_fmt = CHOICE_FORMATS[0]
    choices_str = "\n".join(choice_fmt(l, o) for l, o in zip(letters, options))
    question_str = f"Question: {question_text}\n\n{choices_str}"

    user_content = prompt_template.replace("{article}", article)
    user_content = _fill_question_placeholders(user_content, question_str, "end")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))


def _sample_diverse(
    examples: list[dict],
    n_examples: int,
    rng: np.random.Generator,
    article_lengths: dict[str, int],
) -> list[dict]:
    """Sample *n_examples* from *examples* while maximising article diversity.

    Strategy:
    1. Group examples by article_id.
    2. Round-robin across articles: each pass takes 1 unsampled question per article.
    3. Within each pass, articles are shuffled with probability proportional to
       their token length (longer articles appear earlier → more likely to be
       included before n_examples is reached).
    4. Stop once n_examples is reached.
    """
    from collections import defaultdict

    by_article: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_article[ex["article_id"]].append(ex)

    # Shuffle questions within each article (deterministic)
    for aid in by_article:
        rng.shuffle(by_article[aid])

    # Build article ordering weights (proportional to token length)
    article_ids = list(by_article.keys())
    weights = np.array([article_lengths.get(aid, 1.0) for aid in article_ids], dtype=np.float64)
    weights /= weights.sum()

    # Track how many questions we've taken from each article
    taken: dict[str, int] = {aid: 0 for aid in article_ids}
    selected: list[dict] = []

    while len(selected) < n_examples:
        # Determine which articles still have unsampled questions
        available = [aid for aid in article_ids if taken[aid] < len(by_article[aid])]
        if not available:
            break  # exhausted all questions

        # Weighted shuffle: sample a permutation with weights
        avail_weights = np.array([article_lengths.get(aid, 1.0) for aid in available], dtype=np.float64)
        avail_weights /= avail_weights.sum()
        order = rng.choice(len(available), size=len(available), replace=False, p=avail_weights)

        for idx in order:
            if len(selected) >= n_examples:
                break
            aid = available[idx]
            ex = by_article[aid][taken[aid]]
            selected.append(ex)
            taken[aid] += 1

    return selected


# ---------------------------------------------------------------------------
# QualityDataset
# ---------------------------------------------------------------------------

class QualityDataset(Dataset):
    """
    Map-style, deterministic QuALITY dataset with augmentations.

    Augmentations (all deterministic per example via seeded RNG):
      - Instruction phrasing: sampled from prompt variants
      - Question position (end / beginning / both): sampled per example
      - Choice ordering: A-D shuffled per example
      - Choice formatting: presentation style tied to variant index
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        n_examples: int,
        system_prompt: str,
        prompts_dir: str,
        target_context: int = 10000,
        min_context: int | None = None,
        seed: int = 0,
        question_position_weights: dict[str, float] | None = None,
        instruction_variants: list[int] | None = None,
        split: str = "train",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_examples = int(n_examples)
        self.target_context = int(target_context)
        self.min_context = int(min_context) if min_context is not None else None
        self.system_prompt = system_prompt

        # Question position sampling
        qp_weights = question_position_weights or DEFAULT_QUESTION_POSITION_WEIGHTS
        self._qp_strategies = list(qp_weights.keys())
        self._qp_weights = list(qp_weights.values())

        # Load prompt variants: list of (prompt_template, variant_index)
        self.variants: list[tuple[str, int]] = []
        self._load_variants(prompts_dir, instruction_variants)
        if not self.variants:
            raise ValueError(f"No prompt variants found in {prompts_dir}. Expected subdirs like 0/, 1/, etc.")

        # Load and flatten QuALITY data
        quality_split = "dev" if split in ("validation", "eval", "test") else "train"
        articles = _load_quality(quality_split)
        all_examples = _flatten_examples(articles)
        print(f"QuALITY {quality_split}: {len(articles)} articles, {len(all_examples)} questions")

        # Pre-compute article token lengths (for filtering and weighted sampling)
        # Use the longest prompt template for conservative length estimation
        longest_template = max((t for t, _ in self.variants), key=len)
        article_lengths: dict[str, int] = {}
        for article_obj in articles:
            aid = article_obj["article_id"]
            if aid not in article_lengths:
                # Just tokenize the article alone for a rough length
                article_lengths[aid] = len(
                    tokenizer.encode(article_obj["article"], add_special_tokens=False)
                )

        # Filter by context length
        valid_examples = []
        for ex in all_examples:
            length = _estimate_token_length(
                ex["article"], ex["question_text"], ex["options"],
                system_prompt, longest_template, tokenizer,
            )
            if length <= target_context and (self.min_context is None or length > self.min_context):
                ex["_estimated_length"] = length
                valid_examples.append(ex)

        n_filtered = len(all_examples) - len(valid_examples)
        if n_filtered > 0:
            lo = self.min_context or 0
            print(f"  Filtered out {n_filtered} examples outside bracket ({lo}, {target_context}]")
        print(f"  {len(valid_examples)} examples remaining after filtering")

        if len(valid_examples) < n_examples:
            print(
                f"  WARNING: Only {len(valid_examples)} examples fit in target_context={target_context}, "
                f"using all of them instead of the requested {n_examples}."
            )
            n_examples = len(valid_examples)
            self.n_examples = n_examples

        # Sample with article-diversity-maximising strategy
        sample_rng = np.random.default_rng(int(seed))
        self.examples = _sample_diverse(valid_examples, n_examples, sample_rng, article_lengths)

        # Report article diversity
        unique_articles = len(set(ex["article_id"] for ex in self.examples))
        print(f"  Sampled {n_examples} examples from {unique_articles} unique articles")

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

        # --- Sample augmentation choices (deterministic per example) ---
        variant_list_idx = py_rng.randrange(len(self.variants))
        prompt_template, variant_num = self.variants[variant_list_idx]

        question_position = py_rng.choices(
            self._qp_strategies, weights=self._qp_weights, k=1
        )[0]

        # Select choice format for this variant
        choice_fmt = CHOICE_FORMATS[variant_num % len(CHOICE_FORMATS)]

        # --- Fetch example ---
        example = self.examples[idx]

        # --- Shuffle choice ordering (deterministic per example) ---
        letters = ["A", "B", "C", "D"]
        py_rng.shuffle(letters)

        options = example["options"]
        gold_label = example["gold_label"]  # 0-indexed
        answer_letter = letters[gold_label]

        # Sort by letter for display
        ordered_pairs = sorted(zip(letters, options), key=lambda x: x[0])
        choices_str = "\n".join(choice_fmt(letter, choice) for letter, choice in ordered_pairs)

        # --- Build question string ---
        question_str = f"Question: {example['question_text']}\n\n{choices_str}"

        # --- Build prompt ---
        user_content = prompt_template.replace("{article}", example["article"])
        user_content = _fill_question_placeholders(user_content, question_str, question_position)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Compute actual token length
        input_length = len(
            self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        )

        return {
            "prompt": messages,
            "question": example["question_text"],
            "answer": answer_letter,
            "pos_docs": [],
            "type": "quality",
            "id": str(idx),
            "instruction_variant": variant_list_idx,
            "question_position": question_position,
            "target_context": self.target_context,
            "input_length": input_length,
            "article_id": example["article_id"],
            "difficult": example["difficult"],
        }
