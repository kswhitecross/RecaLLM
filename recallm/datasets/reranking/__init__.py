from dataclasses import dataclass, field


@dataclass
class RerankingExample:
    """Bridge dataclass between Layer 1 (raw data) and Layer 2 (prompt building).

    Holds a single query with its sampled passage set, ready for prompt rendering.
    """

    query: str
    query_id: str
    passages: list[dict] = field(default_factory=list)
    # Each passage dict: {"pid": str, "text": str, "grade": int}
    # grade: 0=irrelevant, 1=related, 2=highly relevant, 3=perfectly relevant
    n_relevant: int = 0  # count of passages with grade >= 1
    neg_source: str = ""  # which negative source was used (judged/bm25/random)
