"""
Layer 2 prompt building and context fitting for qa_kilt.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from recallm.tasks.base import (
    TokenizeableExample,
    insert_into_list,
    normalize_context_range,
    sample_target_context,
)
from recallm.tasks.qa_kilt import QADocument, QAExample

DOC_FORMATS = ["document_title", "title_only", "id_only", "both", "seq_1"]
DEFAULT_DOC_FORMAT_WEIGHTS = {
    "document_title": 0.30,
    "title_only": 0.20,
    "id_only": 0.15,
    "both": 0.20,
    "seq_1": 0.15,
}

DEFAULT_QUESTION_POSITION_WEIGHTS = {"end": 0.70, "beginning": 0.15, "both": 0.15}


def _format_document(doc: QADocument, assigned_id: str | None, doc_format: str) -> str:
    if doc_format == "document_title":
        return f"Document (Title: {doc.title}): {doc.text}"
    if doc_format == "title_only":
        return f"{doc.title}: {doc.text}"
    if doc_format == "id_only":
        return f"[{assigned_id}] {doc.text}"
    if doc_format == "both":
        return f"[{assigned_id}] (Title: {doc.title}): {doc.text}"
    if doc_format == "seq_1":
        return f"Passage {assigned_id}: {doc.text}"
    raise ValueError(f"Unknown doc_format: {doc_format}")


def _assign_ids(n: int, doc_format: str, rng: random.Random) -> list[str | None]:
    if doc_format in ("document_title", "title_only"):
        return [None] * n
    if doc_format in ("id_only", "both"):
        ids = rng.sample(range(1000, 10000), min(n, 9000))
        return [str(i) for i in ids]
    if doc_format == "seq_1":
        return [str(i + 1) for i in range(n)]
    raise ValueError(f"Unknown doc_format: {doc_format}")


def _fill_question_placeholders(prompt_template: str, query_str: str, position: str) -> str:
    if position == "end":
        return prompt_template.replace("{question_start}", "").replace("{question_end}", query_str)
    if position == "beginning":
        return prompt_template.replace("{question_start}", query_str + "\n\n").replace("{question_end}", "")
    if position == "both":
        return prompt_template.replace("{question_start}", query_str + "\n\n").replace("{question_end}", query_str)
    raise ValueError(f"Unknown question position: {position}")


class _QATokenizeableExample(TokenizeableExample):
    def __init__(
        self,
        example: QAExample,
        system_prompt: str,
        prompt_template: str,
        doc_format: str,
        question_position: str,
        gold_depths: list[float],
        assigned_ids_all: list[str | None],
    ):
        super().__init__(max_k=len(example.neg_docs))
        self.example = example
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.doc_format = doc_format
        self.question_position = question_position
        self.gold_depths = gold_depths
        self.assigned_ids_all = assigned_ids_all

    def _get_ordered_docs(self) -> tuple[list[QADocument], list[str | None], list[int]]:
        neg_docs = list(self.example.neg_docs[: self.k])
        ordered_docs, gold_indices = insert_into_list(
            neg_docs, list(self.example.gold_docs), self.gold_depths
        )
        ordered_ids = self.assigned_ids_all[: len(ordered_docs)]
        return ordered_docs, ordered_ids, gold_indices

    def average_length_per_item(self, tokenizer: PreTrainedTokenizerBase) -> float:
        sample_size = min(20, len(self.example.neg_docs))
        if sample_size == 0:
            return 100.0
        total = 0
        for idx in range(sample_size):
            doc = self.example.neg_docs[idx]
            total += len(tokenizer.encode(_format_document(doc, str(idx), self.doc_format), add_special_tokens=False))
        return total / sample_size

    def _render_prompt(self, tokenizer: PreTrainedTokenizerBase) -> tuple[list[dict], int]:
        ordered_docs, ordered_ids, _ = self._get_ordered_docs()
        rendered_docs = [
            _format_document(doc, assigned_id, self.doc_format)
            for doc, assigned_id in zip(ordered_docs, ordered_ids)
        ]
        documents_str = "\n\n".join(rendered_docs)
        query_str = f"\nQuestion: {self.example.question}"
        user_content = self.prompt_template.replace("{documents}", documents_str)
        user_content = _fill_question_placeholders(user_content, query_str, self.question_position)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        input_length = len(
            tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        )
        return messages, input_length

    def tokenized_length(self, tokenizer: PreTrainedTokenizerBase) -> int:
        _, tokenized_length = self._render_prompt(tokenizer)
        return tokenized_length

    def to_dict(self, tokenizer: PreTrainedTokenizerBase) -> dict:
        ordered_docs, ordered_ids, gold_indices = self._get_ordered_docs()
        messages, input_length = self._render_prompt(tokenizer)
        n_total = len(ordered_docs)
        pos_docs = []
        pos_doc_depths = []
        for gold_index in gold_indices:
            pos_docs.append(_format_document(ordered_docs[gold_index], ordered_ids[gold_index], self.doc_format))
            pos_doc_depths.append(gold_index / n_total if n_total else 0.0)
        settings = json.dumps({
            "doc_format": self.doc_format,
            "n_gold": len(self.example.gold_docs),
            "n_neg": self.k,
            "n_total": n_total,
            "pos_doc_depths": pos_doc_depths,
        })
        return {
            "prompt": messages,
            "question": self.example.question,
            "answer": "|||".join(self.example.answers) if self.example.answers else "",
            "pos_docs": pos_docs,
            "type": self.example.dataset_type,
            "id": self.example.question_id,
            "question_id": self.example.question_id,
            "instruction_variant": -1,
            "question_position": self.question_position,
            "doc_format": self.doc_format,
            "corpus_type": self.example.corpus_type,
            "n_passages": n_total,
            "target_context": -1,
            "input_length": input_length,
            "settings": settings,
        }


class QAPromptDataset(Dataset):
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

        qp_weights = question_position_weights or DEFAULT_QUESTION_POSITION_WEIGHTS
        self._qp_strategies = list(qp_weights.keys())
        self._qp_weights = list(qp_weights.values())

        if doc_format_weights is None:
            doc_format_weights = DEFAULT_DOC_FORMAT_WEIGHTS
        self._doc_formats = list(doc_format_weights.keys())
        self._doc_format_weights = list(doc_format_weights.values())

        self.variants: list[tuple[str, int]] = []
        self._load_variants(prompts_dir, instruction_variants)
        if not self.variants:
            raise ValueError(f"No prompt variants found in {prompts_dir}")

        ss = np.random.SeedSequence(int(seed) + 777)
        children = ss.spawn(len(raw_dataset))
        self.base_seeds = [int(cs.generate_state(1, dtype=np.uint32)[0]) for cs in children]

    def _load_variants(self, prompts_dir: str, instruction_variants: list[int] | None) -> None:
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
                with open(prompt_path, "r") as handle:
                    self.variants.append((handle.read(), variant_num))

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        if idx < 0 or idx >= len(self.raw_dataset):
            raise IndexError(idx)

        py_rng = random.Random(self.base_seeds[idx])
        sampled_target_context = sample_target_context(
            self.base_seeds[idx], self.target_context_min, self.target_context_max
        )
        example = self.raw_dataset[idx]
        if example is None:
            return None

        prompt_template, variant_num = self.variants[py_rng.randrange(len(self.variants))]
        question_position = py_rng.choices(self._qp_strategies, weights=self._qp_weights, k=1)[0]
        doc_format = py_rng.choices(self._doc_formats, weights=self._doc_format_weights, k=1)[0]
        gold_depths = [py_rng.random() for _ in example.gold_docs]
        assigned_ids = _assign_ids(len(example.gold_docs) + len(example.neg_docs), doc_format, py_rng)

        tok_example = _QATokenizeableExample(
            example=example,
            system_prompt=self.system_prompt,
            prompt_template=prompt_template,
            doc_format=doc_format,
            question_position=question_position,
            gold_depths=gold_depths,
            assigned_ids_all=assigned_ids,
        )
        tok_example.set_largest_k(self.tokenizer, sampled_target_context)
        result = tok_example.to_dict(self.tokenizer)
        result["instruction_variant"] = variant_num
        result["target_context"] = sampled_target_context
        return result
