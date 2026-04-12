"""
Layer 2: Prompt building and context fitting for reranking.

Wraps MSMARCOv2Dataset (Layer 1), applies augmentations (document ID format,
question position, instruction variant), and uses TokenizeableExample to
binary-search on the number of passages that fit the target context length.
"""

import os
import random
from typing import Any

import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from recallm.datasets.base import (
    TokenizeableExample,
    normalize_context_range,
    sample_target_context,
)
from recallm.datasets.reranking import RerankingExample

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_QUESTION_POSITION_WEIGHTS = {"end": 0.6, "beginning": 0.2, "both": 0.2}

# Document ID format augmentations
DOC_FORMATS = ["bracket_id", "sequential", "random_id"]


def _format_passage(text: str, assigned_id: str, doc_format: str) -> str:
    """Render a single passage in the given document format."""
    if doc_format == "bracket_id":
        return f"[ID: {assigned_id}] Document: {text}"
    elif doc_format == "sequential":
        return f"Passage {assigned_id}: {text}"
    elif doc_format == "random_id":
        return f"[Doc {assigned_id}] {text}"
    else:
        raise ValueError(f"Unknown doc_format: {doc_format}")


def _assign_ids(n: int, doc_format: str, rng: random.Random) -> list[str]:
    """Assign IDs to passages based on the format."""
    if doc_format in ("bracket_id", "sequential"):
        return [str(i) for i in range(n)]
    elif doc_format == "random_id":
        ids = rng.sample(range(1000, 10000), n)
        return [str(i) for i in ids]
    else:
        raise ValueError(f"Unknown doc_format: {doc_format}")


def _fill_question_placeholders(
    prompt_template: str,
    query_str: str,
    position: str,
) -> str:
    """Fill {question_start} and {question_end} based on position strategy."""
    if position == "end":
        return prompt_template.replace("{question_start}", "").replace(
            "{question_end}", query_str
        )
    elif position == "beginning":
        return prompt_template.replace(
            "{question_start}", query_str + "\n\n"
        ).replace("{question_end}", "")
    elif position == "both":
        return prompt_template.replace(
            "{question_start}", query_str + "\n\n"
        ).replace("{question_end}", query_str)
    else:
        raise ValueError(f"Unknown question position: {position}")


# ---------------------------------------------------------------------------
# TokenizeableExample subclass
# ---------------------------------------------------------------------------

class _RerankingTokenizeableExample(TokenizeableExample):
    """Context-fitting wrapper for a reranking example.

    Binary-searches on the number of passages (k) to fit target context.
    """

    def __init__(
        self,
        example: RerankingExample,
        system_prompt: str,
        prompt_template: str,
        doc_format: str,
        question_position: str,
        assigned_ids: list[str],
    ):
        # max_k = total passages available
        super().__init__(max_k=len(example.passages))
        self.example = example
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.doc_format = doc_format
        self.question_position = question_position
        self.assigned_ids = assigned_ids

    def average_length_per_item(self, tokenizer: PreTrainedTokenizerBase) -> float:
        """Estimate average tokens per passage."""
        sample_size = min(20, len(self.example.passages))
        total = 0
        for i in range(sample_size):
            p = self.example.passages[i]
            rendered = _format_passage(p["text"], self.assigned_ids[i], self.doc_format)
            total += len(tokenizer.encode(rendered, add_special_tokens=False))
        return total / sample_size if sample_size > 0 else 100

    def _render_prompt(self, tokenizer: PreTrainedTokenizerBase) -> tuple[list[dict], int]:
        """Render the full prompt for current k, return (messages, token_length)."""
        passages = self.example.passages[:self.k]
        ids = self.assigned_ids[:self.k]

        # Render passages
        rendered_passages = [
            _format_passage(p["text"], aid, self.doc_format)
            for p, aid in zip(passages, ids)
        ]
        documents_str = "\n\n".join(rendered_passages)

        # Build query string
        query_str = f"\nQuery: {self.example.query}"

        # Fill template
        user_content = self.prompt_template.replace("{documents}", documents_str)
        user_content = user_content.replace("{query}", self.example.query)
        user_content = _fill_question_placeholders(user_content, query_str, self.question_position)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        length = len(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        ))
        return messages, length

    def tokenized_length(self, tokenizer: PreTrainedTokenizerBase) -> int:
        _, length = self._render_prompt(tokenizer)
        return length

    def to_dict(self, tokenizer: PreTrainedTokenizerBase) -> dict:
        """Build the final example dict after k is set."""
        passages = self.example.passages[:self.k]
        ids = self.assigned_ids[:self.k]

        messages, input_length = self._render_prompt(tokenizer)

        # Build gold ranking: sort by grade descending, break ties by position
        indexed_passages = [(i, p, aid) for i, (p, aid) in enumerate(zip(passages, ids))]
        indexed_passages.sort(key=lambda x: (-x[1]["grade"], x[0]))
        gold_ranking = " > ".join(aid for _, _, aid in indexed_passages)

        # Relevance grades mapping (assigned ID → grade)
        relevance_grades = {aid: p["grade"] for p, aid in zip(passages, ids)}

        # pos_docs: rendered text of relevant passages (grade >= 1)
        pos_docs = []
        for p, aid in zip(passages, ids):
            if p["grade"] >= 1:
                pos_docs.append(_format_passage(p["text"], aid, self.doc_format))

        n_relevant = sum(1 for p in passages if p["grade"] >= 1)

        # Build pos_doc_labels: format-specific identifiers for grade >= 1 passages
        pos_doc_labels = []
        for p, aid in zip(passages, ids):
            if p["grade"] >= 1:
                if self.doc_format == "bracket_id":
                    pos_doc_labels.append(f"[ID: {aid}]")
                elif self.doc_format == "sequential":
                    pos_doc_labels.append(f"Passage {aid}:")
                elif self.doc_format == "random_id":
                    pos_doc_labels.append(f"[Doc {aid}]")

        import json as _json
        settings = _json.dumps({
            "pos_doc_labels": pos_doc_labels,
            "n_total": self.k,
            "doc_format": self.doc_format,
        })

        return {
            "prompt": messages,
            "question": self.example.query,
            "answer": gold_ranking,
            "pos_docs": pos_docs,
            "relevance_grades": relevance_grades,
            "type": "msmarco_v2",
            "id": self.example.query_id,
            "instruction_variant": -1,  # filled by caller
            "question_position": self.question_position,
            "doc_format": self.doc_format,
            "n_passages": self.k,
            "n_relevant": n_relevant,
            "neg_source": self.example.neg_source,
            "target_context": -1,  # filled by caller
            "input_length": input_length,
            "settings": settings,
        }


# ---------------------------------------------------------------------------
# RerankingPromptDataset
# ---------------------------------------------------------------------------

class RerankingPromptDataset(Dataset):
    """
    Layer 2: Prompt building with context fitting for reranking.

    Wraps a Layer 1 dataset (MSMARCOv2Dataset), applies augmentations,
    and binary-searches on passage count to fit target context length.
    """

    def __init__(
        self,
        raw_dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        system_prompt: str,
        prompts_dir: str,
        target_context: int = 10000,
        context_length_max: int | None = None,
        seed: int = 0,
        question_position_weights: dict[str, float] | None = None,
        doc_format_weights: dict[str, float] | None = None,
        instruction_variants: list[int] | None = None,
    ):
        super().__init__()
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.target_context_min, self.target_context_max = normalize_context_range(
            target_context, context_length_max
        )

        # Question position sampling
        qp_weights = question_position_weights or DEFAULT_QUESTION_POSITION_WEIGHTS
        self._qp_strategies = list(qp_weights.keys())
        self._qp_weights = list(qp_weights.values())

        # Doc format sampling
        if doc_format_weights:
            self._doc_formats = list(doc_format_weights.keys())
            self._doc_format_weights = list(doc_format_weights.values())
        else:
            self._doc_formats = DOC_FORMATS
            self._doc_format_weights = [1.0] * len(DOC_FORMATS)

        # Load prompt variants
        self.variants: list[tuple[str, int]] = []
        self._load_variants(prompts_dir, instruction_variants)
        if not self.variants:
            raise ValueError(f"No prompt variants found in {prompts_dir}")

        # Per-example seeding
        ss = np.random.SeedSequence(int(seed) + 999)  # offset from Layer 1 seed
        children = ss.spawn(len(raw_dataset))
        self.base_seeds = [int(cs.generate_state(1, dtype=np.uint32)[0]) for cs in children]

    def _load_variants(self, prompts_dir: str, instruction_variants: list[int] | None = None) -> None:
        """Scan prompts_dir for {variant_num}/prompt.txt files."""
        if not os.path.isdir(prompts_dir):
            raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

        allowed = set(instruction_variants) if instruction_variants is not None else None
        for entry in sorted(os.listdir(prompts_dir)):
            variant_dir = os.path.join(prompts_dir, entry)
            if not os.path.isdir(variant_dir):
                continue
            try:
                variant_num = int(entry)
            except ValueError:
                continue
            if allowed is not None and variant_num not in allowed:
                continue
            prompt_path = os.path.join(variant_dir, "prompt.txt")
            if os.path.isfile(prompt_path):
                with open(prompt_path, "r") as f:
                    self.variants.append((f.read(), variant_num))

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self.raw_dataset):
            raise IndexError(idx)

        seed_base = self.base_seeds[idx]
        py_rng = random.Random(seed_base)
        sampled_target_context = sample_target_context(
            seed_base, self.target_context_min, self.target_context_max
        )

        # Get raw example from Layer 1
        example: RerankingExample = self.raw_dataset[idx]

        # Sample augmentation choices
        variant_idx = py_rng.randrange(len(self.variants))
        prompt_template, variant_num = self.variants[variant_idx]

        question_position = py_rng.choices(
            self._qp_strategies, weights=self._qp_weights, k=1
        )[0]

        doc_format = py_rng.choices(
            self._doc_formats, weights=self._doc_format_weights, k=1
        )[0]

        # Assign IDs to passages
        assigned_ids = _assign_ids(len(example.passages), doc_format, py_rng)

        # Build tokenizeable example and fit context
        tok_example = _RerankingTokenizeableExample(
            example=example,
            system_prompt=self.system_prompt,
            prompt_template=prompt_template,
            doc_format=doc_format,
            question_position=question_position,
            assigned_ids=assigned_ids,
        )

        tok_example.set_largest_k(self.tokenizer, sampled_target_context)

        if tok_example.k == 0:
            return None  # Degenerate: no passages fit in context

        result = tok_example.to_dict(self.tokenizer)
        result["instruction_variant"] = variant_idx
        result["target_context"] = sampled_target_context

        return result
