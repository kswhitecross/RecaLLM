"""
Layer 1: Raw MSMARCO V2 reranking data loading.

Loads precomputed index files (qrels, queries, BM25 negatives) from prepare_data.py
and fetches passage text via the ir_datasets docstore (fast byte-offset seeking).

Requires: prepare_data.py must have been run first (at minimum with --skip_bm25)
to ensure the corpus is extracted and index files exist.

Sampling design:
- Each example uses ONE negative source (judged, bm25, or random), selected by
  weighted coin flip. This controls the fraction of EXAMPLES using each source.
- Positive ratio is sampled uniformly from a configurable range (default 1-10%).
  Positives are subsampled from the query's relevant passages.
- Pool size controls how many total passages Layer 1 provides. Layer 2 binary-
  searches to fit the target context length.
"""

import json
import os
import random

import numpy as np
from torch.utils.data import Dataset

from recallm.datasets.reranking import RerankingExample

DEFAULT_DATA_DIR = os.environ.get("RECALLM_MSMARCO_DIR")
DEFAULT_IR_DATASETS_HOME = os.environ.get("IR_DATASETS_HOME")

DEFAULT_NEG_SOURCE_WEIGHTS = {"judged": 0.4, "bm25": 0.3, "random": 0.3}


class MSMARCOv2Dataset(Dataset):
    """Layer 1: Raw data loading for MSMARCO V2 reranking.

    Loads index files and ir_datasets docstore. Each __getitem__ call
    selects a query, samples positives (controlled by pos_ratio_range),
    picks ALL negatives from ONE source (controlled by neg_source_weights),
    and returns a RerankingExample.
    """

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        ir_datasets_home: str = DEFAULT_IR_DATASETS_HOME,
        n_examples: int = 2000,
        seed: int = 0,
        split: str = "train",
        pool_size: int = 500,
        neg_source_weights: dict[str, float] | None = None,
        pos_ratio_range: tuple[float, float] = (0.01, 0.10),
    ):
        super().__init__()
        self.n_examples = n_examples
        self.pool_size = pool_size
        self.pos_ratio_range = pos_ratio_range

        # Negative source weights
        weights = neg_source_weights or DEFAULT_NEG_SOURCE_WEIGHTS
        self._neg_sources = list(weights.keys())
        self._neg_weights = list(weights.values())

        # ---- Load index files ----
        print(f"MSMARCOv2Dataset: loading index files from {data_dir}")
        with open(os.path.join(data_dir, "qrels.json"), "r") as f:
            all_qrels: dict[str, dict[str, int]] = json.load(f)
        with open(os.path.join(data_dir, "queries.json"), "r") as f:
            all_queries: dict[str, str] = json.load(f)
        with open(os.path.join(data_dir, "split.json"), "r") as f:
            split_data: dict[str, list[str]] = json.load(f)

        # BM25 negatives are optional (may not exist if --skip_bm25 was used)
        bm25_path = os.path.join(data_dir, "bm25_negatives.json")
        if os.path.exists(bm25_path):
            with open(bm25_path, "r") as f:
                all_bm25_negatives: dict[str, list[str]] = json.load(f)
            print(f"  Loaded bm25_negatives.json ({len(all_bm25_negatives)} queries)")
        else:
            print(f"  WARNING: {bm25_path} not found. Using judged + random negatives only.")
            all_bm25_negatives = {}

        # ---- Filter to split ----
        split_qids = set(split_data[split])
        self.qids = sorted(qid for qid in split_qids if qid in all_qrels)
        self.queries = {qid: all_queries[qid] for qid in self.qids}
        self.qrels = {qid: all_qrels[qid] for qid in self.qids}
        self.bm25_negatives = {qid: all_bm25_negatives.get(qid, []) for qid in self.qids}

        # ---- Pre-compute per-query pools ----
        self.relevant_pids: dict[str, list[str]] = {}
        self.judged_neg_pids: dict[str, list[str]] = {}
        for qid in self.qids:
            judgments = self.qrels[qid]
            self.relevant_pids[qid] = [pid for pid, g in judgments.items() if g >= 1]
            self.judged_neg_pids[qid] = [pid for pid, g in judgments.items() if g == 0]

        # Cross-query random negative pool (deduplicated grade-0 PIDs)
        all_neg_set = set()
        for qid in self.qids:
            all_neg_set.update(self.judged_neg_pids[qid])
        self._all_neg_pids = sorted(all_neg_set)

        # ---- Open ir_datasets docstore ----
        os.environ["IR_DATASETS_HOME"] = ir_datasets_home
        import ir_datasets

        print(f"  Opening ir_datasets docstore for '{ir_datasets.__name__}' corpus...")
        corpus = ir_datasets.load("msmarco-passage-v2")
        self._docstore = corpus.docs_store()

        if not self._docstore.built():
            raise RuntimeError(
                f"Corpus is not extracted! The docstore at\n"
                f"  {self._docstore.base_path}\n"
                f"has no _built sentinel. Run prepare_data.py first:\n"
                f"  python -m recallm.datasets.reranking.prepare_data --skip_bm25"
            )
        print(f"  Docstore ready (extracted at {self._docstore.base_path})")

        # ---- Deterministic per-example seeding ----
        ss = np.random.SeedSequence(int(seed))
        children = ss.spawn(self.n_examples)
        self.base_seeds = [int(cs.generate_state(1, dtype=np.uint32)[0]) for cs in children]

        # ---- Summary ----
        avg_rel = np.mean([len(v) for v in self.relevant_pids.values()])
        avg_judged_neg = np.mean([len(v) for v in self.judged_neg_pids.values()])
        avg_bm25_neg = np.mean([len(v) for v in self.bm25_negatives.values()])
        print(f"  MSMARCOv2Dataset ({split}): {len(self.qids)} queries, {n_examples} examples")
        print(f"    Avg relevant/query: {avg_rel:.1f}")
        print(f"    Avg judged neg/query: {avg_judged_neg:.1f}")
        print(f"    Avg BM25 neg/query: {avg_bm25_neg:.1f}")
        print(f"    Cross-query neg pool: {len(self._all_neg_pids):,}")
        print(f"    Pool size: {pool_size}")
        print(f"    Positive ratio range: {pos_ratio_range[0]:.0%}-{pos_ratio_range[1]:.0%}")
        print(f"    Neg source weights: {dict(zip(self._neg_sources, self._neg_weights))}")

    def __len__(self) -> int:
        return self.n_examples

    def _fetch_many_texts(self, pids: list[str]) -> dict[str, str]:
        """Batch look up passage texts via ir_datasets docstore.

        Uses byte-offset seeking in the extracted flat files — fast.
        """
        result = {}
        for doc in self._docstore.get_many_iter(pids):
            result[doc.doc_id] = doc.text
        return result

    def __getitem__(self, idx: int) -> RerankingExample:
        if idx < 0 or idx >= self.n_examples:
            raise IndexError(idx)

        seed_base = self.base_seeds[idx]
        py_rng = random.Random(seed_base)

        # Select query via round-robin
        qid = self.qids[idx % len(self.qids)]
        query_text = self.queries[qid]

        # Sample positive ratio for this example
        pos_ratio = py_rng.uniform(self.pos_ratio_range[0], self.pos_ratio_range[1])
        n_pos_target = max(1, round(self.pool_size * pos_ratio))
        n_neg_target = self.pool_size - n_pos_target

        # Sample positives (subsample from available)
        available_pos = list(self.relevant_pids[qid])
        py_rng.shuffle(available_pos)
        pos_sample = available_pos[:n_pos_target]
        # If fewer positives than target, give the rest to negatives
        n_neg_target += (n_pos_target - len(pos_sample))

        # Select negative source for this example (weighted coin flip)
        neg_source = py_rng.choices(self._neg_sources, weights=self._neg_weights, k=1)[0]

        # Sample ALL negatives from the chosen source
        used_pids = set(pos_sample)
        neg_sample = self._sample_negatives(qid, neg_source, n_neg_target, used_pids, py_rng)

        # Combine all passage IDs and assign grades
        all_pids = pos_sample + neg_sample
        all_grades = {}
        for pid in pos_sample:
            all_grades[pid] = self.qrels[qid][pid]
        for pid in neg_sample:
            all_grades[pid] = 0

        # Fetch passage texts in batch
        texts = self._fetch_many_texts(all_pids)

        # Build passage list and shuffle
        passages = [
            {"pid": pid, "text": texts[pid], "grade": all_grades[pid]}
            for pid in all_pids
        ]
        py_rng.shuffle(passages)

        n_relevant = len(pos_sample)

        return RerankingExample(
            query=query_text,
            query_id=qid,
            passages=passages,
            n_relevant=n_relevant,
            neg_source=neg_source,
        )

    def _sample_negatives(
        self,
        qid: str,
        source: str,
        n_needed: int,
        used_pids: set[str],
        py_rng: random.Random,
    ) -> list[str]:
        """Sample n_needed negatives from a single source. Falls back to random pool."""
        sampled = []

        if source == "judged":
            candidates = [p for p in self.judged_neg_pids[qid] if p not in used_pids]
            py_rng.shuffle(candidates)
            sampled = candidates[:n_needed]
        elif source == "bm25":
            candidates = [p for p in self.bm25_negatives[qid] if p not in used_pids]
            py_rng.shuffle(candidates)
            sampled = candidates[:n_needed]
        elif source == "random":
            pass  # go straight to random pool below
        else:
            raise ValueError(f"Unknown negative source: {source}")

        # Fill shortfall from random pool
        shortfall = n_needed - len(sampled)
        if shortfall > 0:
            used_pids = used_pids | set(sampled)
            candidates = [p for p in self._all_neg_pids if p not in used_pids]
            py_rng.shuffle(candidates)
            sampled.extend(candidates[:shortfall])

        return sampled
