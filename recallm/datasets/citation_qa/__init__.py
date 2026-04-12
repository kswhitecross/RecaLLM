from dataclasses import dataclass, field


@dataclass
class QampariExample:
    """Raw QAMPARI example with question, answers, gold passages, and distractors."""

    question: str
    qid: str
    answers: list[str]                          # entity names (from answer_text)
    answer_aliases: list[list[str]]             # aliases per answer
    gold_passages: list[dict]                   # [{title, text, pid}] — deduplicated proof passages
    distractor_passages: list[dict]             # [{title, text, pid}] — hard negatives (included + BM25S)
    answer_to_gold_pids: dict[int, list[str]]   # answer_idx → list of proof pids
