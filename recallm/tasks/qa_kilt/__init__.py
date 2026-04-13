"""
KILT-only QA dataset bridge dataclasses and defaults.
"""

import os
from dataclasses import dataclass

DEFAULT_KILT_SOURCE_DIR = os.environ.get("RECALLM_KILT_SOURCE_DIR")
DEFAULT_KILT_WINDOW_DIR = os.environ.get("RECALLM_KILT_WINDOW_DIR")
DEFAULT_INDEX_DIR = os.environ.get("RECALLM_INDEX_DIR")
DEFAULT_DENSE_INDEX_DIR = os.environ.get("RECALLM_DENSE_INDEX_DIR")
DEFAULT_DENSE_MODEL_NAME = "Alibaba-NLP/gte-modernbert-base"
DEFAULT_DENSE_NPROBE = 64
DEFAULT_QA_BM25_BACKEND = "numba"
DEFAULT_QA_BM25_N_THREADS = -1
DEFAULT_FLAT_INDEX_THRESHOLD = 1_000_000
DEFAULT_WINDOW_SEED = 0


def resolve_bm25_n_threads(requested_threads: int) -> int:
    requested = int(requested_threads)
    if requested == 0:
        target = 1
    elif requested < 0:
        target = os.cpu_count() or 1
    else:
        target = requested
    try:
        from numba import config as numba_config
    except Exception:
        return max(1, target)
    max_threads = int(getattr(numba_config, "NUMBA_NUM_THREADS", target))
    return max(1, min(target, max_threads))


@dataclass
class QADocument:
    """A single context document."""

    doc_id: str
    title: str
    text: str
    bm25_score: float = 0.0


@dataclass
class QAExample:
    """Bridge between Layer 1 loading and Layer 2 prompt building."""

    question_id: str
    question: str
    answers: list[str]
    gold_docs: list[QADocument]
    neg_docs: list[QADocument]
    dataset_type: str
    corpus_type: str
