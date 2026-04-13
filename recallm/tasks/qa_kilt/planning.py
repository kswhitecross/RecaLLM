"""
Shared planning utilities for qa_kilt.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


QA_DOC_BUDGET_SAFETY_FACTOR = 1.6
QA_INITIAL_OVERSAMPLE_FACTOR = 1.25
QA_ESTIMATE_SAMPLE_SIZE = 256


@dataclass(frozen=True)
class PlannedQAExample:
    question_id: str
    question: str
    answers: list[str]
    dataset_type: str
    corpus_type: str
    gold_doc_ids: list[int]
    neg_doc_ids: list[int]


def render_budget_document(title: str, text: str) -> str:
    return f"Document (Title: {title}): {text}"


def compute_doc_budget(max_context: int, avg_doc_tokens: float) -> int:
    if avg_doc_tokens <= 0:
        raise ValueError(f"Average document token length must be positive, got {avg_doc_tokens}")
    return int(math.ceil(float(max_context) / float(avg_doc_tokens) * QA_DOC_BUDGET_SAFETY_FACTOR))


def initial_retrieval_k(n_docs: int) -> int:
    if n_docs <= 0:
        return 0
    return int(math.ceil(float(n_docs) * QA_DOC_BUDGET_SAFETY_FACTOR))


def deterministic_doc_sample_indices(total_docs: int, *, seed: int, sample_size: int = QA_ESTIMATE_SAMPLE_SIZE) -> list[int]:
    if total_docs <= 0:
        return []
    sample_size = min(int(sample_size), int(total_docs))
    if sample_size <= 0:
        return []
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xA11CE]))
    if sample_size == total_docs:
        return list(range(total_docs))
    return sorted(int(i) for i in rng.choice(total_docs, size=sample_size, replace=False).tolist())


def plan_seed(base_seed: int, plan_index: int, *, salt: int) -> int:
    return int(
        np.random.SeedSequence([int(base_seed), int(plan_index), int(salt)])
        .generate_state(1, dtype=np.uint32)[0]
    )
