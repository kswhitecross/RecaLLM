"""
Shared helpers for qa_kilt.
"""

from __future__ import annotations

import bisect
import hashlib
import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from recallm.tasks.qa_kilt import QADocument

WORD_RE = re.compile(r"\S+")
TOKEN_RE = re.compile(r"\w+")
_CONTROL_PREFIXES = ("Section::::", "BULLET::::", "NUMBERED::::")


def normalize_title(title: str) -> str:
    if not isinstance(title, str):
        return ""
    return " ".join(title.replace("_", " ").split()).strip().casefold()


def normalize_paragraph(paragraph: str) -> str:
    if not isinstance(paragraph, str):
        return ""
    text = paragraph.strip()
    for prefix in _CONTROL_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def paragraphs_to_article_text(paragraphs: list[str]) -> tuple[str, list[int], list[int]]:
    normalized = []
    paragraph_start_words = []
    paragraph_end_words = []
    word_cursor = 0
    for paragraph in paragraphs:
        normalized_paragraph = normalize_paragraph(paragraph)
        normalized.append(normalized_paragraph)
        paragraph_start_words.append(word_cursor)
        word_cursor += count_words(normalized_paragraph)
        paragraph_end_words.append(word_cursor)
    return "\n".join(normalized), paragraph_start_words, paragraph_end_words


def get_word_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in WORD_RE.finditer(text or "")]


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def deterministic_window_length(
    global_seed: int,
    wikipedia_id: str,
    start_word: int,
    min_words: int = 80,
    max_words: int = 100,
) -> int:
    payload = f"{int(global_seed)}::{wikipedia_id}::{start_word}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return min_words + (value % (max_words - min_words + 1))


def build_window_ranges(
    total_words: int,
    global_seed: int,
    wikipedia_id: str,
    min_words: int = 80,
    max_words: int = 100,
) -> list[tuple[int, int]]:
    ranges = []
    start_word = 0
    while start_word < total_words:
        window_size = deterministic_window_length(
            global_seed=global_seed,
            wikipedia_id=wikipedia_id,
            start_word=start_word,
            min_words=min_words,
            max_words=max_words,
        )
        end_word = min(start_word + window_size, total_words)
        ranges.append((start_word, end_word))
        start_word = end_word
    return ranges


def word_index_to_paragraph(
    paragraph_start_words: list[int],
    paragraph_end_words: list[int],
    word_index: int,
    *,
    use_previous_word: bool,
) -> int:
    if not paragraph_end_words:
        return -1
    target = max(word_index - 1, 0) if use_previous_word else max(word_index, 0)
    idx = bisect.bisect_right(paragraph_end_words, target)
    return min(idx, len(paragraph_end_words) - 1)


def normalized_word_counter(text: str) -> Counter:
    return Counter(TOKEN_RE.findall((text or "").casefold()))


def word_f1(a: str, b: str) -> float:
    a_counts = normalized_word_counter(a)
    b_counts = normalized_word_counter(b)
    if not a_counts or not b_counts:
        return 0.0
    overlap = sum((a_counts & b_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(b_counts.values())
    recall = overlap / sum(a_counts.values())
    return (2.0 * precision * recall) / (precision + recall)


def answer_in_text(answer: str, text: str) -> bool:
    if not isinstance(answer, str):
        return False
    answer = answer.strip()
    if len(answer) < 4:
        return False
    return answer.casefold() in (text or "").casefold()


def answer_in_document(answer: str, title: str, text: str) -> bool:
    return answer_in_text(answer, f"{title or ''}\n{text or ''}")


def dedup_documents(documents: list["QADocument"]) -> list["QADocument"]:
    deduped = []
    seen_doc_ids = set()
    seen_content = set()
    for document in documents:
        doc_id = str(document.doc_id)
        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        content_key = (normalize_title(document.title), (document.text or "").strip())
        if content_key in seen_content:
            continue
        seen_content.add(content_key)
        deduped.append(document)
    return deduped


def collect_answers_and_provenance(output_items: list[dict]) -> tuple[list[str], list[dict]]:
    answers = []
    provenance = []
    seen_answers = set()
    seen_provenance = set()
    for item in output_items or []:
        answer = item.get("answer")
        if isinstance(answer, str):
            cleaned = answer.strip()
            if cleaned and cleaned not in seen_answers:
                seen_answers.add(cleaned)
                answers.append(cleaned)
        for prov in item.get("provenance", []):
            wiki_id = str(prov.get("wikipedia_id", ""))
            start_pid = int(prov.get("start_paragraph_id", 0))
            end_pid = int(prov.get("end_paragraph_id", start_pid))
            key = (wiki_id, start_pid, end_pid)
            if not wiki_id or key in seen_provenance:
                continue
            seen_provenance.add(key)
            provenance.append({
                "wikipedia_id": wiki_id,
                "start_paragraph_id": start_pid,
                "end_paragraph_id": end_pid,
                "title": prov.get("title", ""),
            })
    return answers, provenance
