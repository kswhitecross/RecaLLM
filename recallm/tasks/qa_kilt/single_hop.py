"""
Single-hop KILT-backed QA datasets.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from recallm.tasks.qa_kilt import QAExample
from recallm.tasks.qa_kilt.corpus import KILTWindowCorpus
from recallm.tasks.qa_kilt.planning import (
    PlannedQAExample,
    compute_doc_budget,
    deterministic_doc_sample_indices,
    initial_retrieval_k,
    plan_seed,
    render_budget_document,
)
from recallm.tasks.qa_kilt.utils import (
    answer_in_document,
    collect_answers_and_provenance,
    dedup_documents,
)


@dataclass(frozen=True)
class _SingleHopCandidateSpec:
    plan_index: int
    source_idx: int
    neg_type: str


class BaseKILTSingleHopDataset(Dataset):
    DATASET_TYPE = ""
    TASK_NAME = ""

    def __init__(
        self,
        *,
        kilt_corpus: KILTWindowCorpus,
        tokenizer,
        max_context: int,
        n_examples: int = 2000,
        seed: int = 0,
        pool_size: int | None = None,
        neg_type_weights: dict[str, float] | None = None,
        filter_ids: list[int] | None = None,
        split: str = "train",
    ):
        super().__init__()
        self.kilt_corpus = kilt_corpus
        self.n_examples = int(n_examples)
        self.seed = int(seed)
        self.split = split
        self.max_context = int(max_context)
        self._tokenizer = tokenizer
        self._legacy_pool_size = pool_size

        nt_weights = neg_type_weights or {"dense": 0.45, "bm25": 0.30, "mixed": 0.15, "random": 0.10}
        self._neg_types = list(nt_weights.keys())
        self._neg_weights = list(nt_weights.values())

        self._source_examples = self._load_examples(split)
        if filter_ids is not None:
            self._source_examples = self._source_examples.select(filter_ids)
            print(f"  {self.DATASET_TYPE}: Filtered to {len(self._source_examples):,} source examples")

        self._example_cache: dict[int, dict | None] = {}
        self._materialization_skip_counts: Counter[str] = Counter()
        self._materialization_skip_logs = 0
        self._planning_skip_counts: Counter[str] = Counter()

        self._valid_source_indices, needed_wikipedia_ids = self._collect_needed_wikipedia_ids()
        if not self._valid_source_indices:
            raise RuntimeError(f"{self.DATASET_TYPE}: no usable source examples remain after provenance filtering")
        self.kilt_corpus.ensure_article_cache(wikipedia_ids=needed_wikipedia_ids)
        if self._uses_dense_negatives():
            self.kilt_corpus.require_dense_index()

        self.doc_budget = self._estimate_chunk_doc_budget()
        self._planned_examples = self._plan_examples()

    def _uses_dense_negatives(self) -> bool:
        return any(
            weight > 0.0
            for neg_type, weight in zip(self._neg_types, self._neg_weights)
            if neg_type in {"dense", "mixed"}
        )

    @staticmethod
    def _split_mixed_targets(n_docs: int) -> dict[str, int]:
        counts = {"dense": n_docs // 3, "bm25": n_docs // 3, "random": n_docs // 3}
        for source_name in ("dense", "bm25", "random")[: n_docs % 3]:
            counts[source_name] += 1
        return counts

    def _load_examples(self, split: str):
        raise NotImplementedError

    def _extract_question_text(self, row: dict, idx: int) -> str:
        return row["input"]

    def _extract_answers_and_provenance(self, row: dict, idx: int) -> tuple[list[str], list[dict]]:
        return collect_answers_and_provenance(row.get("output", []))

    def _collect_needed_wikipedia_ids(self) -> tuple[list[int], set[str]]:
        needed_ids = set()
        valid_source_indices = []
        skipped = Counter()
        print(f"  {self.DATASET_TYPE}: Collecting provenance article ids from split={self.split}")
        for idx in tqdm(range(len(self._source_examples)), desc=f"Collecting {self.DATASET_TYPE} provenance"):
            row = self._source_examples[idx]
            answers, provenance = self._extract_answers_and_provenance(row, idx)
            if not answers:
                skipped["missing_answers"] += 1
                continue
            if not provenance:
                skipped["missing_provenance"] += 1
                continue
            valid_source_indices.append(idx)
            for prov in provenance:
                needed_ids.add(str(prov["wikipedia_id"]))
        print(
            f"    {self.DATASET_TYPE}: Kept {len(valid_source_indices):,}/{len(self._source_examples):,} "
            f"source examples; need {len(needed_ids):,} distinct KILT articles"
        )
        if skipped:
            skipped_str = ", ".join(f"{reason}={count:,}" for reason, count in sorted(skipped.items()))
            print(f"    {self.DATASET_TYPE}: Skipped unusable source examples: {skipped_str}")
        return valid_source_indices, needed_ids

    def _record_materialization_skip(self, reason: str, source_idx: int, details: str | None = None) -> None:
        self._materialization_skip_counts[reason] += 1
        if self._materialization_skip_logs < 10:
            message = f"  {self.DATASET_TYPE}: skipping source example {source_idx} ({reason})"
            if details:
                message += f": {details}"
            print(message)
            self._materialization_skip_logs += 1

    def _estimate_chunk_doc_budget(self) -> int:
        indices = deterministic_doc_sample_indices(
            len(self.kilt_corpus.window_dataset),
            seed=self.seed,
        )
        if not indices:
            raise RuntimeError(f"{self.DATASET_TYPE}: cannot estimate chunk doc budget from an empty KILT window corpus")
        batch = self.kilt_corpus.window_dataset[indices]
        total_tokens = 0
        for title, text in zip(batch["title"], batch["text"]):
            rendered = render_budget_document(str(title), str(text))
            total_tokens += len(self._tokenizer.encode(rendered, add_special_tokens=False))
        avg_doc_tokens = total_tokens / len(indices)
        budget = compute_doc_budget(self.max_context, avg_doc_tokens)
        print(
            f"  {self.DATASET_TYPE}: chunk avg doc length={avg_doc_tokens:.1f} tokens "
            f"-> doc budget={budget} for max_context={self.max_context}"
        )
        return budget

    def _materialize_source_example(self, source_idx: int) -> dict | None:
        if source_idx in self._example_cache:
            return self._example_cache[source_idx]

        row = self._source_examples[source_idx]
        question = self._extract_question_text(row, source_idx)
        answers, provenance = self._extract_answers_and_provenance(row, source_idx)
        if not answers:
            self._record_materialization_skip("missing_answers", source_idx)
            self._example_cache[source_idx] = None
            return None
        if not provenance:
            self._record_materialization_skip("missing_provenance", source_idx)
            self._example_cache[source_idx] = None
            return None

        gold_doc_ids = []
        for prov in provenance:
            try:
                document = self.kilt_corpus.map_provenance_to_document(
                    wikipedia_id=prov["wikipedia_id"],
                    start_paragraph_id=int(prov["start_paragraph_id"]),
                    end_paragraph_id=int(prov.get("end_paragraph_id", prov["start_paragraph_id"])),
                    answers=answers,
                )
            except (IndexError, KeyError, RuntimeError) as exc:
                self._record_materialization_skip("provenance_mapping_failed", source_idx, str(exc))
                self._example_cache[source_idx] = None
                return None
            gold_doc_ids.append(int(document.doc_id))
        if not gold_doc_ids:
            self._record_materialization_skip("empty_gold_docs", source_idx)
            self._example_cache[source_idx] = None
            return None
        gold_docs = dedup_documents([
            self.kilt_corpus._document_from_row(self.kilt_corpus._get_window_row(doc_id))
            for doc_id in gold_doc_ids
        ])
        example = {
            "question_id": str(row.get("id", source_idx)),
            "question": question,
            "answers": answers,
            "gold_doc_ids": [int(doc.doc_id) for doc in gold_docs],
        }
        self._example_cache[source_idx] = example
        return example

    def _plan_examples(self) -> list[PlannedQAExample]:
        planned: list[PlannedQAExample] = []
        next_source_pointer = 0
        while len(planned) < self.n_examples:
            missing = self.n_examples - len(planned)
            candidate_specs = self._build_candidate_specs(next_source_pointer, missing)
            if not candidate_specs:
                break
            next_source_pointer += len(candidate_specs)
            self._precompute_candidate_retrieval(candidate_specs)
            before = len(planned)
            for spec in candidate_specs:
                example = self._plan_candidate(spec)
                if example is not None:
                    planned.append(example)
                if len(planned) >= self.n_examples:
                    break
            if len(planned) == before:
                break
        if len(planned) < self.n_examples:
            raise RuntimeError(
                f"{self.DATASET_TYPE}: only planned {len(planned)} examples "
                f"(requested {self.n_examples})"
            )
        return planned

    def _build_candidate_specs(self, start_pointer: int, count: int) -> list[_SingleHopCandidateSpec]:
        specs = []
        for offset in range(count):
            plan_index = start_pointer + offset
            source_idx = self._valid_source_indices[plan_index % len(self._valid_source_indices)]
            rng = random.Random(plan_seed(self.seed, plan_index, salt=17))
            neg_type = rng.choices(self._neg_types, weights=self._neg_weights, k=1)[0]
            specs.append(_SingleHopCandidateSpec(plan_index=plan_index, source_idx=source_idx, neg_type=neg_type))
        return specs

    def _precompute_candidate_retrieval(self, candidate_specs: list[_SingleHopCandidateSpec]) -> None:
        dense_queries: dict[str, int] = {}
        bm25_queries: dict[str, int] = {}
        for spec in candidate_specs:
            base = self._materialize_source_example(spec.source_idx)
            if base is None:
                continue
            n_neg = max(0, self.doc_budget - len(base["gold_doc_ids"]))
            source_targets = self._target_source_counts(n_neg, spec.neg_type)
            if source_targets["dense"] > 0:
                dense_queries[base["question"]] = max(
                    dense_queries.get(base["question"], 0),
                    initial_retrieval_k(source_targets["dense"]),
                )
            if source_targets["bm25"] > 0:
                bm25_queries[base["question"]] = max(
                    bm25_queries.get(base["question"], 0),
                    initial_retrieval_k(source_targets["bm25"]),
                )
        if dense_queries:
            max_k = max(dense_queries.values())
            print(
                f"  {self.DATASET_TYPE}: batch dense search for {len(dense_queries):,} "
                f"queries (k={max_k})"
            )
            self.kilt_corpus.batch_dense_search(sorted(dense_queries.keys()), max_k)
        if bm25_queries:
            max_k = max(bm25_queries.values())
            print(
                f"  {self.DATASET_TYPE}: batch BM25 search for {len(bm25_queries):,} "
                f"queries (k={max_k})"
            )
            self.kilt_corpus.batch_bm25_search(sorted(bm25_queries.keys()), max_k)

    @staticmethod
    def _target_source_counts(n_neg: int, neg_type: str) -> dict[str, int]:
        if neg_type == "dense":
            return {"dense": n_neg, "bm25": 0, "random": 0}
        if neg_type == "bm25":
            return {"dense": 0, "bm25": n_neg, "random": 0}
        if neg_type == "mixed":
            return BaseKILTSingleHopDataset._split_mixed_targets(n_neg)
        return {"dense": 0, "bm25": 0, "random": n_neg}

    def _plan_candidate(self, spec: _SingleHopCandidateSpec) -> PlannedQAExample | None:
        base = self._materialize_source_example(spec.source_idx)
        if base is None:
            self._planning_skip_counts["source_materialization_failed"] += 1
            return None
        exclude_ids = {str(doc_id) for doc_id in base["gold_doc_ids"]}
        n_neg = max(0, self.doc_budget - len(base["gold_doc_ids"]))
        rng = random.Random(plan_seed(self.seed, spec.plan_index, salt=29))
        neg_doc_ids = self._select_chunk_negative_ids(
            base["question"],
            n_neg,
            exclude_ids,
            base["answers"],
            spec.neg_type,
            rng,
        )
        return PlannedQAExample(
            question_id=base["question_id"],
            question=base["question"],
            answers=list(base["answers"]),
            dataset_type=self.DATASET_TYPE,
            corpus_type="chunk",
            gold_doc_ids=list(base["gold_doc_ids"]),
            neg_doc_ids=neg_doc_ids,
        )

    def _select_chunk_negative_ids(
        self,
        question: str,
        n_neg: int,
        exclude_ids: set[str],
        answers: list[str],
        neg_type: str,
        rng: random.Random,
    ) -> list[int]:
        if n_neg <= 0:
            return []
        if neg_type == "dense":
            return self._select_chunk_dense_ids(question, n_neg, exclude_ids, answers)
        if neg_type == "bm25":
            return self._select_chunk_bm25_ids(question, n_neg, exclude_ids, answers)
        if neg_type == "mixed":
            counts = self._split_mixed_targets(n_neg)
            negatives = self._select_chunk_dense_ids(question, counts["dense"], exclude_ids, answers)
            used_ids = exclude_ids | {str(doc_id) for doc_id in negatives}
            negatives.extend(self._select_chunk_bm25_ids(question, counts["bm25"], used_ids, answers))
            used_ids = exclude_ids | {str(doc_id) for doc_id in negatives}
            negatives.extend(self._select_chunk_random_ids(counts["random"], used_ids, answers, rng))
            while len(negatives) < n_neg:
                used_ids = exclude_ids | {str(doc_id) for doc_id in negatives}
                for fallback in (
                    lambda: self._select_chunk_dense_ids(question, 1, used_ids, answers),
                    lambda: self._select_chunk_bm25_ids(question, 1, used_ids, answers),
                    lambda: self._select_chunk_random_ids(1, used_ids, answers, rng),
                ):
                    extra = fallback()
                    if extra:
                        negatives.extend(extra)
                        break
                else:
                    break
            return negatives[:n_neg]
        return self._select_chunk_random_ids(n_neg, exclude_ids, answers, rng)

    def _select_chunk_dense_ids(
        self,
        question: str,
        n_docs: int,
        exclude_ids: set[str],
        answers: list[str],
    ) -> list[int]:
        return self._select_chunk_retrieved_ids(
            question=question,
            n_docs=n_docs,
            exclude_ids=exclude_ids,
            answers=answers,
            source_name="dense",
        )

    def _select_chunk_bm25_ids(
        self,
        question: str,
        n_docs: int,
        exclude_ids: set[str],
        answers: list[str],
    ) -> list[int]:
        return self._select_chunk_retrieved_ids(
            question=question,
            n_docs=n_docs,
            exclude_ids=exclude_ids,
            answers=answers,
            source_name="bm25",
        )

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
                f"  {self.DATASET_TYPE}: expanding chunk {source_name} search "
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

    def release_planning_resources(self) -> None:
        self._source_examples = None
        self._example_cache.clear()

    def __len__(self) -> int:
        return len(self._planned_examples)

    def __getitem__(self, idx: int) -> QAExample | None:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        plan = self._planned_examples[idx]
        gold_docs = [
            self.kilt_corpus._document_from_row(self.kilt_corpus._get_window_row(doc_id))
            for doc_id in plan.gold_doc_ids
        ]
        neg_docs = [
            self.kilt_corpus._document_from_row(self.kilt_corpus._get_window_row(doc_id))
            for doc_id in plan.neg_doc_ids
        ]
        return QAExample(
            question_id=plan.question_id,
            question=plan.question,
            answers=list(plan.answers),
            gold_docs=gold_docs,
            neg_docs=neg_docs,
            dataset_type=self.DATASET_TYPE,
            corpus_type="chunk",
        )


class NQLayer1Dataset(BaseKILTSingleHopDataset):
    DATASET_TYPE = "nq"
    TASK_NAME = "nq"

    def _load_examples(self, split: str):
        print(f"  {self.DATASET_TYPE}: Loading KILT task split={split}")
        return load_dataset("facebook/kilt_tasks", self.TASK_NAME, split=split)
