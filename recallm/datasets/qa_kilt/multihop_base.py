"""
Shared base class for KILT-window multi-hop QA datasets.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import os
import random

import bm25s
import Stemmer
from torch.utils.data import Dataset
from tqdm import tqdm

from recallm.datasets.qa_kilt import (
    DEFAULT_DENSE_NPROBE,
    DEFAULT_INDEX_DIR,
    DEFAULT_QA_BM25_BACKEND,
    DEFAULT_QA_BM25_N_THREADS,
    QADocument,
    QAExample,
    resolve_bm25_n_threads,
)
from recallm.datasets.qa_kilt.corpus import KILTWindowCorpus
from recallm.datasets.qa_kilt.dense import (
    corpus_index_dir,
    get_dense_searcher,
    paragraph_corpus_name,
    resolve_dense_root,
    validate_index_metadata,
)
from recallm.datasets.qa_kilt.planning import (
    PlannedQAExample,
    QA_INITIAL_OVERSAMPLE_FACTOR,
    compute_doc_budget,
    deterministic_doc_sample_indices,
    initial_retrieval_k,
    plan_seed,
    render_budget_document,
)
from recallm.datasets.qa_kilt.utils import answer_in_document, answer_in_text, dedup_documents


_PARAGRAPH_BM25_RETRIEVER_CACHE: dict[tuple[str, str], bm25s.BM25] = {}


@dataclass(frozen=True)
class _MultiHopCandidateSpec:
    plan_index: int
    source_idx: int
    corpus_type: str
    neg_type: str


class BaseKILTMultiHopDataset(Dataset):
    DATASET_TYPE = ""
    INDEX_NAME = ""

    def __init__(
        self,
        *,
        kilt_corpus: KILTWindowCorpus,
        tokenizer,
        max_context: int,
        n_examples: int = 2000,
        seed: int = 0,
        pool_size: int | None = None,
        corpus_type_weights: dict[str, float] | None = None,
        neg_type_weights: dict[str, float] | None = None,
        bm25_index_dir: str | None = None,
        dense_index_dir: str | None = None,
        dense_nprobe: int = DEFAULT_DENSE_NPROBE,
        bm25_backend: str = DEFAULT_QA_BM25_BACKEND,
        bm25_n_threads: int = DEFAULT_QA_BM25_N_THREADS,
        filter_ids: list[int] | None = None,
        strict_corpus_type: bool = True,
        split: str = "train",
        num_proc: int = 8,
    ):
        super().__init__()
        self.kilt_corpus = kilt_corpus
        self.n_examples = int(n_examples)
        self.seed = int(seed)
        self.split = split
        self.max_context = int(max_context)
        self.strict_corpus_type = bool(strict_corpus_type)
        self.num_proc = max(1, int(num_proc))
        self._tokenizer = tokenizer
        self._legacy_pool_size = pool_size
        self._filter_ids = filter_ids

        ct_weights = corpus_type_weights or {"paragraph": 0.5, "chunk": 0.5}
        self._corpus_types = list(ct_weights.keys())
        self._corpus_weights = list(ct_weights.values())

        nt_weights = neg_type_weights or {"dense": 0.45, "bm25": 0.30, "mixed": 0.15, "random": 0.10}
        self._neg_types = list(nt_weights.keys())
        self._neg_weights = list(nt_weights.values())

        self._bm25_index_dir = bm25_index_dir or DEFAULT_INDEX_DIR
        self._dense_index_dir = resolve_dense_root(dense_index_dir)
        self._dense_nprobe = int(dense_nprobe)
        self._bm25_backend = str(bm25_backend)
        self._bm25_n_threads = resolve_bm25_n_threads(bm25_n_threads)
        self._paragraph_dense_corpus_name = paragraph_corpus_name(self.DATASET_TYPE)
        self._paragraph_dense_searcher = None
        self._paragraph_dense_cache: dict[str, list[int]] = {}
        self._paragraph_bm25_cache: dict[str, tuple[list[int], list[float]]] = {}
        self._document_row_cache: dict[int, dict] = {}
        self._dense_source_paths: dict[str, str] = {}
        self._paragraph_example_cache: dict[int, dict | None] = {}
        self._chunk_example_cache: dict[int, dict | None] = {}
        self._planning_skip_counts: Counter[str] = Counter()
        self._planning_skip_logs = 0
        self._ratio_is_preplanned = True

        self._questions = None
        self._documents = None
        self._load_source_data(split)
        if self._filter_ids is not None:
            self._questions = self._questions.select(self._filter_ids)
            print(f"    Filtered to {len(self._questions):,} questions")
        print(f"    Questions: {len(self._questions):,}, Documents: {len(self._documents):,}")

        self._stemmer = Stemmer.Stemmer("english")
        self._retriever = self._load_or_build_bm25()
        self._validate_dense_requirements()
        self._doc_budgets = self._estimate_doc_budgets()
        self._planned_examples = self._plan_examples()

    def _load_source_data(self, split: str) -> None:
        raise NotImplementedError

    def _question_pos_ids(self, question_row: dict) -> list[int]:
        return list(set(question_row["pos_doc_ids"]))

    @classmethod
    def release_shared_planning_resources(cls) -> None:
        _PARAGRAPH_BM25_RETRIEVER_CACHE.clear()

    def _paragraph_dense_corpus_dir(self) -> str:
        return corpus_index_dir(self._dense_index_dir, self._paragraph_dense_corpus_name)

    def _load_or_build_bm25(self) -> bm25s.BM25:
        index_name = os.path.join(self._bm25_index_dir, self.INDEX_NAME)
        cache_key = (index_name, self._bm25_backend)
        cached = _PARAGRAPH_BM25_RETRIEVER_CACHE.get(cache_key)
        if cached is not None:
            print(f"    Reusing paragraph BM25 index from {index_name}")
            return cached
        if os.path.exists(index_name):
            print(f"    Loading paragraph BM25 index from {index_name}")
            retriever = bm25s.BM25.load(
                index_name,
                override_params={"backend": self._bm25_backend},
            )
            _PARAGRAPH_BM25_RETRIEVER_CACHE[cache_key] = retriever
            return retriever

        print(f"    Building paragraph BM25 index at {index_name}")
        corpus = self._documents.map(
            lambda row: {"search_text": f"{row['title']}\n{row['text']}"},
            remove_columns=self._documents.column_names,
            num_proc=self.num_proc,
            desc=f"Preparing {self.DATASET_TYPE} paragraph search text",
        )["search_text"]
        tokenized = bm25s.tokenize(corpus, stopwords="en", stemmer=self._stemmer)
        retriever = bm25s.BM25(backend=self._bm25_backend)
        retriever.index(tokenized, show_progress=True)
        retriever.save(index_name)
        print(f"    Saved paragraph BM25 index to {index_name}")
        _PARAGRAPH_BM25_RETRIEVER_CACHE[cache_key] = retriever
        return retriever

    def _uses_dense_negatives(self) -> bool:
        return any(
            weight > 0.0
            for neg_type, weight in zip(self._neg_types, self._neg_weights)
            if neg_type in {"dense", "mixed"}
        )

    def _validate_dense_requirements(self) -> None:
        if not self._uses_dense_negatives():
            return
        if any(
            weight > 0.0
            for corpus_type, weight in zip(self._corpus_types, self._corpus_weights)
            if corpus_type == "chunk"
        ):
            self.kilt_corpus.require_dense_index()
        if any(
            weight > 0.0
            for corpus_type, weight in zip(self._corpus_types, self._corpus_weights)
            if corpus_type == "paragraph"
        ):
            self._require_paragraph_dense_index()

    def _require_paragraph_dense_index(self) -> None:
        searcher = self._get_paragraph_dense_searcher()
        validate_index_metadata(
            searcher.metadata,
            expected_corpus_name=self._paragraph_dense_corpus_name,
            expected_row_count=len(self._documents),
            expected_source_paths=self._dense_source_paths,
        )

    def _get_paragraph_dense_searcher(self):
        if self._paragraph_dense_searcher is None:
            self._paragraph_dense_searcher = get_dense_searcher(
                self._paragraph_dense_corpus_dir(),
                nprobe=self._dense_nprobe,
            )
        return self._paragraph_dense_searcher

    def _get_document_row(self, doc_id: int) -> dict:
        cached = self._document_row_cache.get(doc_id)
        if cached is not None:
            return cached
        row = self._documents[doc_id]
        self._document_row_cache[doc_id] = row
        return row

    def _get_document_rows_batch(self, doc_ids: list[int]) -> dict:
        all_cached = all(did in self._document_row_cache for did in doc_ids)
        if all_cached:
            result: dict[str, list] = {}
            for did in doc_ids:
                row = self._document_row_cache[did]
                for key, value in row.items():
                    result.setdefault(key, []).append(value)
            return result
        batch = self._documents[doc_ids]
        col_names = list(batch.keys())
        for i, did in enumerate(doc_ids):
            if did not in self._document_row_cache:
                self._document_row_cache[did] = {col: batch[col][i] for col in col_names}
        return batch

    @staticmethod
    def _split_mixed_targets(n_docs: int) -> dict[str, int]:
        counts = {"dense": n_docs // 3, "bm25": n_docs // 3, "random": n_docs // 3}
        for source_name in ("dense", "bm25", "random")[: n_docs % 3]:
            counts[source_name] += 1
        return counts

    def _target_source_counts(self, n_neg: int, neg_type: str) -> dict[str, int]:
        if neg_type == "dense":
            return {"dense": n_neg, "bm25": 0, "random": 0}
        if neg_type == "bm25":
            return {"dense": 0, "bm25": n_neg, "random": 0}
        if neg_type == "mixed":
            return self._split_mixed_targets(n_neg)
        return {"dense": 0, "bm25": 0, "random": n_neg}

    def _estimate_doc_budget(self, dataset, *, label: str, seed_salt: int) -> int:
        indices = deterministic_doc_sample_indices(
            len(dataset),
            seed=plan_seed(self.seed, 0, salt=seed_salt),
        )
        if not indices:
            raise RuntimeError(f"{self.DATASET_TYPE}: cannot estimate {label} doc budget from an empty dataset")
        batch = dataset[indices]
        total_tokens = 0
        for title, text in zip(batch["title"], batch["text"]):
            rendered = render_budget_document(str(title), str(text))
            total_tokens += len(self._tokenizer.encode(rendered, add_special_tokens=False))
        avg_doc_tokens = total_tokens / len(indices)
        budget = compute_doc_budget(self.max_context, avg_doc_tokens)
        print(
            f"    {self.DATASET_TYPE}: {label} avg doc length={avg_doc_tokens:.1f} tokens "
            f"-> doc budget={budget} for max_context={self.max_context}"
        )
        return budget

    def _estimate_doc_budgets(self) -> dict[str, int]:
        budgets: dict[str, int] = {}
        active_corpus_types = {
            corpus_type
            for corpus_type, weight in zip(self._corpus_types, self._corpus_weights)
            if weight > 0.0
        }
        if "paragraph" in active_corpus_types:
            budgets["paragraph"] = self._estimate_doc_budget(
                self._documents,
                label="paragraph",
                seed_salt=101,
            )
        if "chunk" in active_corpus_types:
            budgets["chunk"] = self._estimate_doc_budget(
                self.kilt_corpus.window_dataset,
                label="chunk",
                seed_salt=211,
            )
        return budgets

    def _question_id(self, question_row: dict) -> str:
        return str(question_row.get("question_id", question_row.get("id")))

    @staticmethod
    def _question_text(question_or_row) -> str:
        if isinstance(question_or_row, dict):
            return str(question_or_row["question"])
        return str(question_or_row)

    def _record_planning_skip(self, reason: str, source_idx: int, details: str | None = None) -> None:
        self._planning_skip_counts[reason] += 1
        if self._planning_skip_logs < 10:
            message = f"    {self.DATASET_TYPE}: skipping source example {source_idx} ({reason})"
            if details:
                message += f": {details}"
            print(message)
            self._planning_skip_logs += 1

    def _materialize_paragraph_base(self, source_idx: int) -> dict | None:
        cached = self._paragraph_example_cache.get(source_idx)
        if source_idx in self._paragraph_example_cache:
            return cached

        question_row = self._questions[source_idx]
        pos_ids = self._question_pos_ids(question_row)
        if not pos_ids:
            self._record_planning_skip("missing_pos_doc_ids", source_idx)
            self._paragraph_example_cache[source_idx] = None
            return None
        positives = self._get_document_rows_batch(pos_ids)
        gold_docs = dedup_documents([
            QADocument(doc_id=str(doc_id), title=title, text=text)
            for doc_id, title, text in zip(positives["id"], positives["title"], positives["text"])
        ])
        if not gold_docs:
            self._record_planning_skip("empty_paragraph_gold_docs", source_idx)
            self._paragraph_example_cache[source_idx] = None
            return None
        base = {
            "question_id": self._question_id(question_row),
            "question": question_row["question"],
            "answers": [question_row["answer"]],
            "gold_doc_ids": [int(doc.doc_id) for doc in gold_docs],
        }
        self._paragraph_example_cache[source_idx] = base
        return base

    def _materialize_chunk_base(self, source_idx: int) -> dict | None:
        cached = self._chunk_example_cache.get(source_idx)
        if source_idx in self._chunk_example_cache:
            return cached

        question_row = self._questions[source_idx]
        pos_ids = self._question_pos_ids(question_row)
        if not pos_ids:
            self._record_planning_skip("missing_pos_doc_ids", source_idx)
            self._chunk_example_cache[source_idx] = None
            return None
        positives = self._get_document_rows_batch(pos_ids)
        gold_docs = []
        needed_titles = set()
        for title in positives["title"]:
            title = str(title).strip()
            if title:
                needed_titles.add(title)
        if needed_titles:
            self.kilt_corpus.ensure_article_cache(titles=needed_titles)
        for title, text in zip(positives["title"], positives["text"]):
            document = self.kilt_corpus.best_window_by_overlap(title=title, passage_text=text, min_f1=0.5)
            if document is None:
                self._record_planning_skip("chunk_gold_mapping_failed", source_idx, f"title={title!r}")
                self._chunk_example_cache[source_idx] = None
                return None
            gold_docs.append(document)
        gold_docs = dedup_documents(gold_docs)
        if not gold_docs:
            self._record_planning_skip("empty_chunk_gold_docs", source_idx)
            self._chunk_example_cache[source_idx] = None
            return None

        answer = str(question_row["answer"])
        if len(answer) >= 4 and not any(answer_in_text(answer, doc.text) for doc in gold_docs):
            self._record_planning_skip("chunk_gold_missing_answer", source_idx)
            self._chunk_example_cache[source_idx] = None
            return None

        base = {
            "question_id": self._question_id(question_row),
            "question": question_row["question"],
            "answers": [answer],
            "gold_doc_ids": [int(doc.doc_id) for doc in gold_docs],
        }
        self._chunk_example_cache[source_idx] = base
        return base

    def _base_for_spec(self, spec: _MultiHopCandidateSpec) -> dict | None:
        if spec.corpus_type == "paragraph":
            return self._materialize_paragraph_base(spec.source_idx)
        if spec.corpus_type == "chunk":
            return self._materialize_chunk_base(spec.source_idx)
        raise ValueError(f"Unknown corpus_type: {spec.corpus_type}")

    def _batch_paragraph_dense_search(self, query_depths: dict[str, int]) -> None:
        if not query_depths:
            return
        max_k = min(len(self._documents), max(query_depths.values()))
        needed = [query for query in sorted(query_depths) if len(self._paragraph_dense_cache.get(query, [])) < max_k]
        if not needed:
            return
        self._require_paragraph_dense_index()
        print(
            f"    {self.DATASET_TYPE}: batch paragraph dense search for {len(needed):,} "
            f"queries (k={max_k})"
        )
        results = self._get_paragraph_dense_searcher().batch_search_texts(needed, max_k)
        for query, row_ids in zip(needed, results):
            self._paragraph_dense_cache[query] = [int(row_id) for row_id in row_ids]

    def _batch_paragraph_bm25_search(self, query_depths: dict[str, int], *, chunk_size: int = 64) -> None:
        if not query_depths:
            return
        max_k = min(len(self._documents), max(query_depths.values()))
        needed = [
            query for query in sorted(query_depths)
            if query not in self._paragraph_bm25_cache or len(self._paragraph_bm25_cache[query][0]) < max_k
        ]
        if not needed:
            return
        print(
            f"    {self.DATASET_TYPE}: batch paragraph BM25 search for {len(needed):,} "
            f"queries (k={max_k}, chunk_size={chunk_size})"
        )
        for start in tqdm(
            range(0, len(needed), chunk_size),
            desc=f"{self.DATASET_TYPE} paragraph BM25",
            total=math.ceil(len(needed) / chunk_size),
        ):
            chunk = needed[start:start + chunk_size]
            query_tokens = bm25s.tokenize(
                chunk,
                stopwords="en",
                stemmer=self._stemmer,
                show_progress=False,
            )
            row_ids_batch, scores_batch = self._retriever.retrieve(
                query_tokens,
                k=max_k,
                n_threads=self._bm25_n_threads,
                show_progress=False,
            )
            for query, row_ids, scores in zip(chunk, row_ids_batch, scores_batch):
                self._paragraph_bm25_cache[query] = (
                    [int(row_id) for row_id in row_ids],
                    [float(score) for score in scores],
                )

    def _paragraph_dense_candidates(self, question: str | dict, search_k: int) -> list[int]:
        search_k = min(int(search_k), len(self._documents))
        if search_k <= 0:
            return []
        question_text = self._question_text(question)
        cached = self._paragraph_dense_cache.get(question_text)
        if cached is not None and len(cached) >= search_k:
            return cached[:search_k]
        self._require_paragraph_dense_index()
        cached = self._get_paragraph_dense_searcher().search_text(question_text, search_k)
        cached = [int(row_id) for row_id in cached]
        self._paragraph_dense_cache[question_text] = cached
        return cached

    def _paragraph_bm25_candidates(self, question: str | dict, search_k: int) -> tuple[list[int], list[float]]:
        search_k = min(int(search_k), len(self._documents))
        if search_k <= 0:
            return [], []
        question_text = self._question_text(question)
        cached = self._paragraph_bm25_cache.get(question_text)
        if cached is not None and len(cached[0]) >= search_k:
            return cached[0][:search_k], cached[1][:search_k]
        query_tokens = bm25s.tokenize(
            question_text,
            stopwords="en",
            stemmer=self._stemmer,
            show_progress=False,
        )
        row_ids, scores = self._retriever.retrieve(
            query_tokens,
            k=search_k,
            n_threads=self._bm25_n_threads,
            show_progress=False,
        )
        cached_row_ids = [int(row_id) for row_id in row_ids[0]]
        cached_scores = [float(score) for score in scores[0]]
        self._paragraph_bm25_cache[question_text] = (cached_row_ids, cached_scores)
        return cached_row_ids, cached_scores

    def _precompute_candidate_retrieval(self, candidate_specs: list[_MultiHopCandidateSpec]) -> None:
        paragraph_dense_queries: dict[str, int] = {}
        paragraph_bm25_queries: dict[str, int] = {}
        chunk_dense_queries: dict[str, int] = {}
        chunk_bm25_queries: dict[str, int] = {}

        for spec in candidate_specs:
            base = self._base_for_spec(spec)
            if base is None:
                continue
            doc_budget = self._doc_budgets[spec.corpus_type]
            n_neg = max(0, doc_budget - len(base["gold_doc_ids"]))
            source_targets = self._target_source_counts(n_neg, spec.neg_type)
            if spec.corpus_type == "paragraph":
                if source_targets["dense"] > 0:
                    paragraph_dense_queries[base["question"]] = max(
                        paragraph_dense_queries.get(base["question"], 0),
                        initial_retrieval_k(source_targets["dense"]),
                    )
                if source_targets["bm25"] > 0:
                    paragraph_bm25_queries[base["question"]] = max(
                        paragraph_bm25_queries.get(base["question"], 0),
                        initial_retrieval_k(source_targets["bm25"]),
                    )
            else:
                if source_targets["dense"] > 0:
                    chunk_dense_queries[base["question"]] = max(
                        chunk_dense_queries.get(base["question"], 0),
                        initial_retrieval_k(source_targets["dense"]),
                    )
                if source_targets["bm25"] > 0:
                    chunk_bm25_queries[base["question"]] = max(
                        chunk_bm25_queries.get(base["question"], 0),
                        initial_retrieval_k(source_targets["bm25"]),
                    )

        self._batch_paragraph_dense_search(paragraph_dense_queries)
        self._batch_paragraph_bm25_search(paragraph_bm25_queries)
        if chunk_dense_queries:
            max_k = max(chunk_dense_queries.values())
            print(
                f"    {self.DATASET_TYPE}: batch chunk dense search for {len(chunk_dense_queries):,} "
                f"queries (k={max_k})"
            )
            self.kilt_corpus.batch_dense_search(sorted(chunk_dense_queries), max_k)
        if chunk_bm25_queries:
            max_k = max(chunk_bm25_queries.values())
            print(
                f"    {self.DATASET_TYPE}: batch chunk BM25 search for {len(chunk_bm25_queries):,} "
                f"queries (k={max_k})"
            )
            self.kilt_corpus.batch_bm25_search(sorted(chunk_bm25_queries), max_k)

    def _select_paragraph_retrieved_ids(
        self,
        *,
        question: str,
        n_docs: int,
        exclude_ids: set[int],
        answers: list[str],
        source_name: str,
    ) -> list[int]:
        if n_docs <= 0:
            return []
        total_docs = len(self._documents)
        search_k = min(total_docs, initial_retrieval_k(n_docs))
        while True:
            if source_name == "dense":
                candidate_row_ids = self._paragraph_dense_candidates(question, search_k)
                candidate_scores = None
            else:
                candidate_row_ids, candidate_scores = self._paragraph_bm25_candidates(question, search_k)
            negatives: list[int] = []
            used_ids = set(int(doc_id) for doc_id in exclude_ids)
            for idx, row_id in enumerate(candidate_row_ids):
                row_id = int(row_id)
                if row_id in used_ids:
                    continue
                row = self._get_document_row(row_id)
                if any(answer_in_document(answer, row["title"], row["text"]) for answer in answers):
                    continue
                negatives.append(row_id)
                used_ids.add(row_id)
                if len(negatives) >= n_docs:
                    return negatives
            if search_k >= total_docs or len(candidate_row_ids) >= total_docs:
                return negatives
            next_k = min(total_docs, search_k * 2)
            print(
                f"    {self.DATASET_TYPE}: expanding paragraph {source_name} search "
                f"for question={question!r} from k={search_k} to k={next_k}"
            )
            search_k = next_k
            if source_name == "bm25" and candidate_scores is not None:
                del candidate_scores

    def _select_paragraph_random_ids(
        self,
        n_docs: int,
        exclude_ids: set[int],
        answers: list[str],
        rng: random.Random,
    ) -> list[int]:
        negatives = []
        used_ids = set(int(doc_id) for doc_id in exclude_ids)
        attempts = 0
        total_docs = len(self._documents)
        max_attempts = max(total_docs, n_docs)
        while len(negatives) < n_docs and attempts < max_attempts:
            attempts += 1
            row_id = rng.randrange(total_docs)
            if row_id in used_ids:
                continue
            row = self._get_document_row(row_id)
            if any(answer_in_document(answer, row["title"], row["text"]) for answer in answers):
                continue
            negatives.append(row_id)
            used_ids.add(row_id)
        return negatives

    def _select_paragraph_negative_ids(
        self,
        question: str,
        n_docs: int,
        exclude_ids: set[int],
        answers: list[str],
        neg_type: str,
        rng: random.Random,
    ) -> list[int]:
        if n_docs <= 0:
            return []
        if neg_type == "dense":
            return self._select_paragraph_retrieved_ids(
                question=question,
                n_docs=n_docs,
                exclude_ids=exclude_ids,
                answers=answers,
                source_name="dense",
            )
        if neg_type == "bm25":
            return self._select_paragraph_retrieved_ids(
                question=question,
                n_docs=n_docs,
                exclude_ids=exclude_ids,
                answers=answers,
                source_name="bm25",
            )
        if neg_type == "mixed":
            counts = self._split_mixed_targets(n_docs)
            negatives = self._select_paragraph_retrieved_ids(
                question=question,
                n_docs=counts["dense"],
                exclude_ids=exclude_ids,
                answers=answers,
                source_name="dense",
            )
            used_ids = exclude_ids | set(int(doc_id) for doc_id in negatives)
            negatives.extend(
                self._select_paragraph_retrieved_ids(
                    question=question,
                    n_docs=counts["bm25"],
                    exclude_ids=used_ids,
                    answers=answers,
                    source_name="bm25",
                )
            )
            used_ids = exclude_ids | set(int(doc_id) for doc_id in negatives)
            negatives.extend(self._select_paragraph_random_ids(counts["random"], used_ids, answers, rng))
            while len(negatives) < n_docs:
                used_ids = exclude_ids | set(int(doc_id) for doc_id in negatives)
                for fallback in (
                    lambda: self._select_paragraph_retrieved_ids(
                        question=question,
                        n_docs=1,
                        exclude_ids=used_ids,
                        answers=answers,
                        source_name="dense",
                    ),
                    lambda: self._select_paragraph_retrieved_ids(
                        question=question,
                        n_docs=1,
                        exclude_ids=used_ids,
                        answers=answers,
                        source_name="bm25",
                    ),
                    lambda: self._select_paragraph_random_ids(1, used_ids, answers, rng),
                ):
                    extra = fallback()
                    if extra:
                        negatives.extend(extra)
                        break
                else:
                    break
            return negatives[:n_docs]
        return self._select_paragraph_random_ids(n_docs, exclude_ids, answers, rng)

    def _paragraph_dense_negatives(
        self,
        question_row: dict,
        pos_set: set[int],
        n_neg: int,
        exclude_ids: set[int] | None = None,
    ) -> list[QADocument]:
        row_ids = self._select_paragraph_retrieved_ids(
            question=question_row["question"],
            n_docs=n_neg,
            exclude_ids=pos_set | (exclude_ids or set()),
            answers=[question_row["answer"]],
            source_name="dense",
        )
        return [
            QADocument(doc_id=str(row_id), title=self._get_document_row(row_id)["title"], text=self._get_document_row(row_id)["text"])
            for row_id in row_ids
        ]

    def _paragraph_bm25_negatives(self, question_row: dict, pos_set: set[int], n_neg: int) -> list[QADocument]:
        row_ids = self._select_paragraph_retrieved_ids(
            question=question_row["question"],
            n_docs=n_neg,
            exclude_ids=pos_set,
            answers=[question_row["answer"]],
            source_name="bm25",
        )
        return [
            QADocument(doc_id=str(row_id), title=self._get_document_row(row_id)["title"], text=self._get_document_row(row_id)["text"])
            for row_id in row_ids
        ]

    def _paragraph_random_negatives(
        self,
        question_row: dict,
        pos_set: set[int],
        n_neg: int,
        rng: random.Random,
        exclude_ids: set[int] | None = None,
    ) -> list[QADocument]:
        row_ids = self._select_paragraph_random_ids(
            n_neg,
            pos_set | (exclude_ids or set()),
            [question_row["answer"]],
            rng,
        )
        return [
            QADocument(doc_id=str(row_id), title=self._get_document_row(row_id)["title"], text=self._get_document_row(row_id)["text"])
            for row_id in row_ids
        ]

    def _paragraph_mixed_negatives(
        self,
        question_row: dict,
        pos_set: set[int],
        n_neg: int,
        rng: random.Random,
    ) -> list[QADocument]:
        row_ids = self._select_paragraph_negative_ids(
            question_row["question"],
            n_neg,
            pos_set,
            [question_row["answer"]],
            "mixed",
            rng,
        )
        return [
            QADocument(doc_id=str(row_id), title=self._get_document_row(row_id)["title"], text=self._get_document_row(row_id)["text"])
            for row_id in row_ids
        ]

    def _paragraph_negatives(
        self,
        question_row: dict,
        pos_ids: list[int],
        n_neg: int,
        neg_type: str,
        rng: random.Random,
    ) -> list[QADocument]:
        pos_set = set(pos_ids)
        if neg_type == "dense":
            return self._paragraph_dense_negatives(question_row, pos_set, n_neg)
        if neg_type == "bm25":
            return self._paragraph_bm25_negatives(question_row, pos_set, n_neg)
        if neg_type == "mixed":
            return self._paragraph_mixed_negatives(question_row, pos_set, n_neg, rng)
        return self._paragraph_random_negatives(question_row, pos_set, n_neg, rng)

    def _select_chunk_retrieved_ids(
        self,
        *,
        question: str,
        n_docs: int,
        exclude_ids: set[str],
        answers: list[str],
        source_name: str,
    ) -> list[int]:
        if n_docs <= 0:
            return []
        total_windows = len(self.kilt_corpus.window_dataset)
        search_k = min(total_windows, initial_retrieval_k(n_docs))
        while True:
            if source_name == "dense":
                candidate_row_ids = self.kilt_corpus.dense_search(question, search_k)
            else:
                candidate_row_ids, _ = self.kilt_corpus.bm25_search(question, search_k)
            negatives: list[int] = []
            used_ids = set(str(doc_id) for doc_id in exclude_ids)
            for row_id in candidate_row_ids:
                row = self.kilt_corpus._get_window_row(int(row_id))
                doc_id = str(row["window_id"])
                if doc_id in used_ids:
                    continue
                if any(answer_in_document(answer, row["title"], row["text"]) for answer in answers):
                    continue
                negatives.append(int(row["window_id"]))
                used_ids.add(doc_id)
                if len(negatives) >= n_docs:
                    return negatives
            if search_k >= total_windows or len(candidate_row_ids) >= total_windows:
                return negatives
            next_k = min(total_windows, search_k * 2)
            print(
                f"    {self.DATASET_TYPE}: expanding chunk {source_name} search "
                f"for question={question!r} from k={search_k} to k={next_k}"
            )
            search_k = next_k

    def _select_chunk_random_ids(
        self,
        n_docs: int,
        exclude_ids: set[str],
        answers: list[str],
        rng: random.Random,
    ) -> list[int]:
        negatives = []
        used_ids = set(str(doc_id) for doc_id in exclude_ids)
        attempts = 0
        total_windows = len(self.kilt_corpus.window_dataset)
        max_attempts = max(total_windows, n_docs)
        while len(negatives) < n_docs and attempts < max_attempts:
            attempts += 1
            row_id = rng.randrange(total_windows)
            row = self.kilt_corpus._get_window_row(row_id)
            doc_id = str(row["window_id"])
            if doc_id in used_ids:
                continue
            if any(answer_in_document(answer, row["title"], row["text"]) for answer in answers):
                continue
            negatives.append(int(row["window_id"]))
            used_ids.add(doc_id)
        return negatives

    def _select_chunk_negative_ids(
        self,
        question: str,
        n_docs: int,
        exclude_ids: set[str],
        answers: list[str],
        neg_type: str,
        rng: random.Random,
    ) -> list[int]:
        if n_docs <= 0:
            return []
        if neg_type == "dense":
            return self._select_chunk_retrieved_ids(
                question=question,
                n_docs=n_docs,
                exclude_ids=exclude_ids,
                answers=answers,
                source_name="dense",
            )
        if neg_type == "bm25":
            return self._select_chunk_retrieved_ids(
                question=question,
                n_docs=n_docs,
                exclude_ids=exclude_ids,
                answers=answers,
                source_name="bm25",
            )
        if neg_type == "mixed":
            counts = self._split_mixed_targets(n_docs)
            negatives = self._select_chunk_retrieved_ids(
                question=question,
                n_docs=counts["dense"],
                exclude_ids=exclude_ids,
                answers=answers,
                source_name="dense",
            )
            used_ids = exclude_ids | {str(doc_id) for doc_id in negatives}
            negatives.extend(
                self._select_chunk_retrieved_ids(
                    question=question,
                    n_docs=counts["bm25"],
                    exclude_ids=used_ids,
                    answers=answers,
                    source_name="bm25",
                )
            )
            used_ids = exclude_ids | {str(doc_id) for doc_id in negatives}
            negatives.extend(self._select_chunk_random_ids(counts["random"], used_ids, answers, rng))
            while len(negatives) < n_docs:
                used_ids = exclude_ids | {str(doc_id) for doc_id in negatives}
                for fallback in (
                    lambda: self._select_chunk_retrieved_ids(
                        question=question,
                        n_docs=1,
                        exclude_ids=used_ids,
                        answers=answers,
                        source_name="dense",
                    ),
                    lambda: self._select_chunk_retrieved_ids(
                        question=question,
                        n_docs=1,
                        exclude_ids=used_ids,
                        answers=answers,
                        source_name="bm25",
                    ),
                    lambda: self._select_chunk_random_ids(1, used_ids, answers, rng),
                ):
                    extra = fallback()
                    if extra:
                        negatives.extend(extra)
                        break
                else:
                    break
            return negatives[:n_docs]
        return self._select_chunk_random_ids(n_docs, exclude_ids, answers, rng)

    def _chunk_negatives(
        self,
        question: str,
        n_neg: int,
        exclude_ids: set[str],
        answers: list[str],
        neg_type: str,
        rng: random.Random,
    ) -> list[QADocument]:
        row_ids = self._select_chunk_negative_ids(question, n_neg, exclude_ids, answers, neg_type, rng)
        return [
            self.kilt_corpus._document_from_row(self.kilt_corpus._get_window_row(int(row_id)))
            for row_id in row_ids
        ]

    def _plan_candidate(self, spec: _MultiHopCandidateSpec) -> PlannedQAExample | None:
        base = self._base_for_spec(spec)
        if base is None:
            return None
        doc_budget = self._doc_budgets[spec.corpus_type]
        n_neg = max(0, doc_budget - len(base["gold_doc_ids"]))
        rng = random.Random(plan_seed(self.seed, spec.plan_index, salt=73))
        if spec.corpus_type == "paragraph":
            neg_doc_ids = self._select_paragraph_negative_ids(
                base["question"],
                n_neg,
                set(int(doc_id) for doc_id in base["gold_doc_ids"]),
                base["answers"],
                spec.neg_type,
                rng,
            )
        else:
            neg_doc_ids = self._select_chunk_negative_ids(
                base["question"],
                n_neg,
                {str(doc_id) for doc_id in base["gold_doc_ids"]},
                base["answers"],
                spec.neg_type,
                rng,
            )
        return PlannedQAExample(
            question_id=base["question_id"],
            question=base["question"],
            answers=list(base["answers"]),
            dataset_type=self.DATASET_TYPE,
            corpus_type=spec.corpus_type,
            gold_doc_ids=list(base["gold_doc_ids"]),
            neg_doc_ids=neg_doc_ids,
        )

    def _target_corpus_counts(self) -> dict[str, int]:
        active = [
            (corpus_type, weight)
            for corpus_type, weight in zip(self._corpus_types, self._corpus_weights)
            if weight > 0.0
        ]
        if not active:
            raise RuntimeError(f"{self.DATASET_TYPE}: no active corpus types configured")
        total_weight = sum(weight for _, weight in active)
        counts: dict[str, int] = {}
        assigned = 0
        for corpus_type, weight in active[:-1]:
            count = int(round(self.n_examples * (weight / total_weight)))
            counts[corpus_type] = count
            assigned += count
        last_corpus_type = active[-1][0]
        counts[last_corpus_type] = self.n_examples - assigned
        for corpus_type, _ in active:
            counts.setdefault(corpus_type, 0)
        return counts

    def _build_candidate_specs(
        self,
        *,
        missing_counts: dict[str, int],
        source_pointer: int,
        first_round: bool,
    ) -> tuple[list[_MultiHopCandidateSpec], int]:
        plan_index_base = source_pointer
        counts = {}
        for corpus_type, missing in missing_counts.items():
            if missing <= 0:
                continue
            if first_round and len(missing_counts) > 1:
                counts[corpus_type] = max(missing, int(math.ceil(missing * QA_INITIAL_OVERSAMPLE_FACTOR)))
            else:
                counts[corpus_type] = missing
        specs: list[_MultiHopCandidateSpec] = []
        remaining = True
        while remaining:
            remaining = False
            for corpus_type in self._corpus_types:
                if counts.get(corpus_type, 0) <= 0:
                    continue
                remaining = True
                plan_index = plan_index_base + len(specs)
                rng = random.Random(plan_seed(self.seed, plan_index, salt=17))
                neg_type = rng.choices(self._neg_types, weights=self._neg_weights, k=1)[0]
                question_idx = source_pointer % len(self._questions)
                specs.append(
                    _MultiHopCandidateSpec(
                        plan_index=plan_index,
                        source_idx=question_idx,
                        corpus_type=corpus_type,
                        neg_type=neg_type,
                    )
                )
                counts[corpus_type] -= 1
                source_pointer += 1
        return specs, source_pointer

    def _plan_examples(self) -> list[PlannedQAExample]:
        target_counts = self._target_corpus_counts()
        planned_examples: list[PlannedQAExample] = []
        planned_by_corpus = Counter()
        source_pointer = 0
        first_round = True
        no_progress_questions = 0
        n_questions = len(self._questions)
        while len(planned_examples) < self.n_examples:
            missing_counts = {
                corpus_type: target - planned_by_corpus[corpus_type]
                for corpus_type, target in target_counts.items()
                if planned_by_corpus[corpus_type] < target
            }
            if not missing_counts:
                break
            candidate_specs, source_pointer = self._build_candidate_specs(
                missing_counts=missing_counts,
                source_pointer=source_pointer,
                first_round=first_round,
            )
            if not candidate_specs:
                break
            first_round = False
            self._precompute_candidate_retrieval(candidate_specs)
            before = len(planned_examples)
            for spec in candidate_specs:
                if planned_by_corpus[spec.corpus_type] >= target_counts[spec.corpus_type]:
                    continue
                example = self._plan_candidate(spec)
                if example is None:
                    continue
                planned_examples.append(example)
                planned_by_corpus[spec.corpus_type] += 1
            n_candidates = len(candidate_specs)
            if len(planned_examples) == before:
                no_progress_questions += n_candidates
                if no_progress_questions >= n_questions:
                    raise RuntimeError(
                        f"{self.DATASET_TYPE}: exhausted all {n_questions} questions "
                        f"without progress while filling {dict(missing_counts)}"
                    )
            else:
                no_progress_questions = 0

        if len(planned_examples) != self.n_examples:
            raise RuntimeError(
                f"{self.DATASET_TYPE}: planned {len(planned_examples)} examples "
                f"(requested {self.n_examples})"
            )
        parts = ", ".join(f"{corpus_type}={planned_by_corpus[corpus_type]}" for corpus_type in sorted(target_counts))
        print(f"    {self.DATASET_TYPE}: planned exact corpus mix -> {parts}")
        if self._planning_skip_counts:
            skipped = ", ".join(
                f"{reason}={count:,}" for reason, count in sorted(self._planning_skip_counts.items())
            )
            print(f"    {self.DATASET_TYPE}: planning skips -> {skipped}")
        return planned_examples

    def release_planning_resources(self) -> None:
        self._retriever = None
        self._paragraph_dense_searcher = None
        self._paragraph_dense_cache.clear()
        self._paragraph_bm25_cache.clear()
        self._document_row_cache.clear()
        self._paragraph_example_cache.clear()
        self._chunk_example_cache.clear()

    def __len__(self) -> int:
        return len(self._planned_examples)

    def __getitem__(self, idx: int) -> QAExample | None:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        plan = self._planned_examples[idx]
        if plan.corpus_type == "paragraph":
            gold_docs = [
                QADocument(
                    doc_id=str(doc_id),
                    title=self._get_document_row(int(doc_id))["title"],
                    text=self._get_document_row(int(doc_id))["text"],
                )
                for doc_id in plan.gold_doc_ids
            ]
            neg_docs = [
                QADocument(
                    doc_id=str(doc_id),
                    title=self._get_document_row(int(doc_id))["title"],
                    text=self._get_document_row(int(doc_id))["text"],
                )
                for doc_id in plan.neg_doc_ids
            ]
        else:
            gold_docs = [
                self.kilt_corpus._document_from_row(self.kilt_corpus._get_window_row(int(doc_id)))
                for doc_id in plan.gold_doc_ids
            ]
            neg_docs = [
                self.kilt_corpus._document_from_row(self.kilt_corpus._get_window_row(int(doc_id)))
                for doc_id in plan.neg_doc_ids
            ]
        return QAExample(
            question_id=plan.question_id,
            question=plan.question,
            answers=list(plan.answers),
            gold_docs=gold_docs,
            neg_docs=neg_docs,
            dataset_type=self.DATASET_TYPE,
            corpus_type=plan.corpus_type,
        )
