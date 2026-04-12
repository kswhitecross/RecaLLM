"""
Reward utility functions for RecaLLM GRPO training.

Consolidated from reward/reward_utils.py and reward/thinking_rewards.py.
"""

import re
import string
import unicodedata
from collections import Counter
from typing import Optional


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

ANSWER_RE = re.compile(r'.*Answer:\s*(\S.*)\s*$', re.DOTALL)


def extract_boxed_answer(text: str) -> Optional[str]:
    r"""Return the contents of the first \boxed{...} block, handling nested braces."""
    marker = r"\boxed"
    start = text.find(marker)
    if start == -1:
        return None

    i = start + len(marker)
    # skip optional whitespace before the opening brace
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != "{":
        return None

    i += 1
    depth = 1
    content_start = i
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        return None

    return text[content_start:i - 1]


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Official HotpotQA normalization function."""
    def lower(text):
        return text.lower()
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    return white_space_fix(remove_articles(remove_punc(lower(s))))


_QAMPARI_CITATION_RE = re.compile(r"\[\d+\]")
_QAMPARI_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
_QAMPARI_WHITESPACE_RE = re.compile(r"\s+")
_QAMPARI_SEPARATOR_RE = re.compile(r"[/_-]+")
_QAMPARI_PUNCT_TABLE = str.maketrans({ch: " " for ch in string.punctuation})


def normalize_qampari_answer_text(text: str) -> str:
    """Aggressive normalization for QAMPARI answer-coverage matching."""
    if not isinstance(text, str):
        return ""
    text = _QAMPARI_CITATION_RE.sub(" ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold().replace("&", " and ")
    text = _QAMPARI_SEPARATOR_RE.sub(" ", text)
    text = text.translate(_QAMPARI_PUNCT_TABLE)
    text = _QAMPARI_ARTICLE_RE.sub(" ", text)
    return _QAMPARI_WHITESPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Answer matching
# ---------------------------------------------------------------------------

def qampari_variant_matches_answer(
    variant: str,
    normalized_answer: str,
    normalized_answer_tokens: list[str] | None = None,
) -> bool:
    """One-directional normalized containment with a short-variant token guard."""
    variant_norm = normalize_qampari_answer_text(variant)
    if not variant_norm or not normalized_answer:
        return False

    variant_tokens = variant_norm.split()
    if len(variant_tokens) == 1 and len(variant_tokens[0]) <= 4:
        if normalized_answer_tokens is None:
            normalized_answer_tokens = normalized_answer.split()
        return variant_tokens[0] in normalized_answer_tokens

    return variant_norm in normalized_answer


def compute_qampari_answer_coverage_stats(
    model_answer: str,
    answer_variants: list[list[str]],
    available_answer_indices: list[int],
) -> dict:
    """Compute QAMPARI answer-coverage stats from parsed variants + availability."""
    normalized_answer = normalize_qampari_answer_text(model_answer)
    normalized_answer_tokens = normalized_answer.split()

    covered_answers = 0
    available_answers = 0
    seen_indices = set()
    for answer_idx in available_answer_indices:
        if answer_idx in seen_indices or answer_idx < 0 or answer_idx >= len(answer_variants):
            continue
        seen_indices.add(answer_idx)
        variants = answer_variants[answer_idx]
        if not variants:
            continue
        available_answers += 1
        if any(
            qampari_variant_matches_answer(variant, normalized_answer, normalized_answer_tokens)
            for variant in variants
        ):
            covered_answers += 1

    denom = min(5, available_answers)
    coverage = min(covered_answers, 5) / denom if denom > 0 else 0.0
    return {
        "qampari_answer_coverage": coverage,
        "qampari_covered_answers": covered_answers,
        "qampari_available_answers": available_answers,
    }


# ---------------------------------------------------------------------------
# Token-level metrics
# ---------------------------------------------------------------------------

def f1_toks(pred_toks: list[str], gold_toks: list[str]) -> float:
    """Computes the F1 score between two lists of tokens."""
    gold_counter = Counter(gold_toks)
    pred_counter = Counter(pred_toks)
    num_gold = sum(gold_counter.values())
    num_pred = sum(pred_counter.values())
    intersection = sum((gold_counter & pred_counter).values())
    return 2 * intersection / (num_gold + num_pred) if num_gold + num_pred > 0 else 0.0


def two_way_subEM(pred: str, gold: str) -> float:
    """
    Computes the two-way substring Exact Match (subEM) between two strings.
    Returns 1.0 if either string is a substring of the other after normalization, else 0.0.
    """
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    if pred_norm in gold_norm or gold_norm in pred_norm:
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Span overlap metrics (character-level, for recall spans vs gold docs)
# ---------------------------------------------------------------------------

def copy_overlap_f1(a: str, b: str) -> float:
    """F1 based on containment or longest edge overlap (suffix-prefix) between two strings."""
    if not a or not b:
        return 0.0

    # containment
    if a in b:
        inter = len(a)
    elif b in a:
        inter = len(b)
    else:
        lim = min(len(a), len(b))
        left = right = 0
        # suffix(a) == prefix(b)
        for k in range(lim, 0, -1):
            if a[-k:] == b[:k]:
                left = k
                break
        # suffix(b) == prefix(a)
        for k in range(lim, 0, -1):
            if b[-k:] == a[:k]:
                right = k
                break
        inter = max(left, right)

    return 2 * inter / (len(a) + len(b)) if (len(a) + len(b)) else 0.0


def copy_overlap_coverage(copied_str: str, source_str: str) -> float:
    """
    Returns the coverage of source_str by copied_str based on longest edge overlap.
    Coverage is computed as the length of the overlap divided by the length of the source_str.
    """
    if not copied_str or not source_str:
        return 0.0

    # containment
    if copied_str in source_str:
        inter = len(copied_str)
    elif source_str in copied_str:
        inter = len(source_str)
    else:
        lim = min(len(copied_str), len(source_str))
        left = right = 0
        # suffix(a) == prefix(b)
        for k in range(lim, 0, -1):
            if copied_str[-k:] == source_str[:k]:
                left = k
                break
        # suffix(b) == prefix(a)
        for k in range(lim, 0, -1):
            if source_str[-k:] == copied_str[:k]:
                right = k
                break
        inter = max(left, right)

    return inter / len(source_str) if len(source_str) else 1.0


# ---------------------------------------------------------------------------
# Recall span extraction
# ---------------------------------------------------------------------------

def get_recall_spans(content: str, recall_start: str = "<recall>", recall_end: str = "</recall>") -> list[str]:
    """Extract recall spans from the given string."""
    pattern = re.escape(recall_start) + r"(.*?)" + re.escape(recall_end)
    spans = re.findall(pattern, content, re.DOTALL)
    return spans
