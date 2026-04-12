"""
QAMPARI Citation QA dataset: multi-answer entity questions with inline
passage citations backed by recall spans.

Two-layer pattern:
  - Layer 1 (QampariDataset): Loads raw QAMPARI data from JSONL.gz files
    (DPR retriever format from qampari_with_contexts.zip), builds
    QampariExample objects with gold passages and distractors,
    augments with BM25S hard negatives when needed.
  - Layer 2 (QampariPromptDataset): Wraps Layer 1, applies augmentations
    (instruction variant, question position), does context fitting via
    TokenizeableExample, and exports the final training dict.

Document format:
    Document [{ID}](Title: {title}): {text}

Uses ~100-word DPR Wikipedia chunks (psgs_w100, 2018-12-20 snapshot)
with BM25/DPR retriever for distractor selection.

Source: QAMPARI (Amouyal et al., 2022), DPR retriever format.
Data: https://aggreg-qa.s3.amazonaws.com/qampari_with_contexts.zip
"""

import gzip
import json
import os
import pickle
import random
import warnings
from typing import Any

import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from recallm.datasets.base import (
    TokenizeableExample,
    normalize_context_range,
    sample_target_context,
)
from recallm.datasets.citation_qa import QampariExample


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Citation QA places the question BEFORE documents,
# so we default to 60% beginning to match the ALCE citation format.
DEFAULT_QUESTION_POSITION_WEIGHTS = {"beginning": 0.6, "end": 0.2, "both": 0.2}


def _format_document(doc_id: int, title: str, text: str) -> str:
    """Format a single passage as it appears in the prompt (ALCE citation format)."""
    return f"Document [{doc_id}](Title: {title}): {text}"


def _fill_question_placeholders(
    prompt_template: str,
    question_str: str,
    position: str,
) -> str:
    """Fill {question_start} and {question_end} based on position strategy."""
    if position == "end":
        return prompt_template.replace("{question_start}", "").replace(
            "{question_end}", question_str
        )
    elif position == "beginning":
        return prompt_template.replace(
            "{question_start}", question_str + "\n\n"
        ).replace("{question_end}", "")
    elif position == "both":
        return prompt_template.replace(
            "{question_start}", question_str + "\n\n"
        ).replace("{question_end}", question_str)
    else:
        raise ValueError(f"Unknown question position: {position}")


def _passage_contains_answer(passage: dict, all_answer_strings: list[str]) -> bool:
    """Check if passage title+text contains any answer string (case-insensitive).

    Checks both title and text because the rendered document format includes
    the title: ``Document [{ID}](Title: {title}): {text}``.
    """
    combined = (passage.get("title", "") + " " + passage.get("text", "")).lower()
    return any(ans.lower() in combined for ans in all_answer_strings)


def _dedup_answer_variants(answer: str, aliases: list[str]) -> list[str]:
    """Canonical answer first, then aliases, preserving order and dropping empties."""
    variants = [answer, *(aliases or [])]
    deduped = []
    seen = set()
    for variant in variants:
        cleaned = str(variant).strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(cleaned)
    return deduped


# ---------------------------------------------------------------------------
# Layer 1: QampariDataset
# ---------------------------------------------------------------------------


class QampariDataset(Dataset):
    """
    Raw QAMPARI data loader.

    Loads from JSONL.gz files (DPR retriever format), builds QampariExample
    objects with gold passages and distractors. Augments with BM25S hard
    negatives when included distractors are insufficient.
    """

    def __init__(
        self,
        data_dir: str,
        n_examples: int,
        seed: int = 0,
        split: str = "train",
        min_distractors: int = 80,
    ):
        super().__init__()
        self.n_examples = int(n_examples)
        self.min_distractors = min_distractors

        jsonl_split = "dev" if split in ("validation", "eval", "test") else "train"
        cache_path = os.path.join(data_dir, f"cached_{jsonl_split}_md{min_distractors}.pkl")

        # Try loading from cache first
        if os.path.isfile(cache_path):
            print(f"Loading cached QAMPARI {jsonl_split} from {cache_path}...")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            self._all_passages = cached["all_passages"]
            self._all_passages_list = cached["all_passages_list"]
            self._pid_to_corpus_idx = cached["pid_to_corpus_idx"]
            self._examples = cached["examples"]
            print(f"  Loaded {len(self._examples)} examples, {len(self._all_passages_list)} passages from cache")
        else:
            self._build_from_raw(data_dir, jsonl_split, cache_path)

        # Sample n_examples deterministically
        rng = np.random.default_rng(int(seed))
        if len(self._examples) <= n_examples:
            self._examples = list(self._examples)
        else:
            indices = rng.choice(len(self._examples), size=n_examples, replace=False)
            self._examples = [self._examples[i] for i in sorted(indices)]

        print(f"  Sampled {len(self._examples)} examples (requested {n_examples})")

    def _build_from_raw(self, data_dir: str, jsonl_split: str, cache_path: str) -> None:
        """Load raw JSONL, build passage corpus + examples, save cache."""
        import bm25s as _bm25s
        import Stemmer as _Stemmer
        self._bm25s = _bm25s

        # Decompress .jsonl.gz → .jsonl on first use (faster reads later)
        jsonl_path = os.path.join(data_dir, f"full_{jsonl_split}_data.jsonl")
        gz_path = jsonl_path + ".gz"
        if not os.path.isfile(jsonl_path):
            if not os.path.isfile(gz_path):
                raise FileNotFoundError(
                    f"QAMPARI data not found at {jsonl_path} or {gz_path}. "
                    "Run download_qampari.py first."
                )
            print(f"Extracting {gz_path} → {jsonl_path}...")
            with gzip.open(gz_path, "rb") as f_in, open(jsonl_path, "wb") as f_out:
                while chunk := f_in.read(1 << 20):
                    f_out.write(chunk)

        print(f"Loading QAMPARI {jsonl_split} from {jsonl_path}...")
        with open(jsonl_path, "r") as f:
            raw_dataset = [json.loads(line) for line in f]
        print(f"QAMPARI {jsonl_split}: {len(raw_dataset)} examples loaded")

        # Build passage corpus
        self._all_passages: dict[str, dict] = {}
        self._all_passages_list: list[dict] = []
        self._pid_to_corpus_idx: dict[str, int] = {}
        self._build_passage_corpus(raw_dataset)

        # Build or load BM25S index
        self._stemmer = _Stemmer.Stemmer("english")
        index_dir = os.path.join(data_dir, f"bm25_index_{jsonl_split}")
        self._retriever = self._get_or_build_index(index_dir)

        # Build examples with BM25S augmentation
        self._examples = self._build_all_examples(raw_dataset)
        print(f"  Built {len(self._examples)} QampariExample objects")

        # Save cache for fast loading next time
        print(f"  Saving cache to {cache_path}...")
        with open(cache_path, "wb") as f:
            pickle.dump({
                "all_passages": self._all_passages,
                "all_passages_list": self._all_passages_list,
                "pid_to_corpus_idx": self._pid_to_corpus_idx,
                "examples": self._examples,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Cache saved ({os.path.getsize(cache_path) / 1e6:.0f} MB)")

    def _build_passage_corpus(self, raw_dataset) -> None:
        """Collect all unique passages from all examples for BM25S indexing.

        Actual JSONL schema:
          - positive_ctxs: [{text, title, score, chunk_id, pid}] — gold passages
          - ctxs: [{text, title, id, score}] — retrieved passages (distractors)
        Note: positive_ctxs use 'pid' as identifier, ctxs use 'id'.
        """
        for raw_ex in raw_dataset:
            for ctx in raw_ex["positive_ctxs"]:
                pid = ctx["pid"]
                if pid not in self._all_passages:
                    self._all_passages[pid] = {
                        "title": ctx["title"],
                        "text": ctx["text"],
                        "pid": pid,
                    }
            for ctx in raw_ex.get("ctxs", []):
                pid = ctx["id"]  # ctxs use 'id' not 'pid'
                if pid not in self._all_passages:
                    self._all_passages[pid] = {
                        "title": ctx["title"],
                        "text": ctx["text"],
                        "pid": pid,
                    }

        self._all_passages_list = list(self._all_passages.values())
        self._pid_to_corpus_idx = {
            p["pid"]: i for i, p in enumerate(self._all_passages_list)
        }
        print(f"  Passage corpus: {len(self._all_passages_list)} unique passages")

    def _get_or_build_index(self, index_dir: str):
        """Load BM25S index from cache or build from corpus."""
        bm25s = self._bm25s
        if os.path.isdir(index_dir):
            print(f"  Loading BM25S index from {index_dir}")
            return bm25s.BM25.load(index_dir)

        print(f"  Building BM25S index over {len(self._all_passages_list)} passages...")
        corpus_texts = [p["text"] for p in self._all_passages_list]
        corpus_tokens = bm25s.tokenize(corpus_texts, stemmer=self._stemmer)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        os.makedirs(index_dir, exist_ok=True)
        retriever.save(index_dir)
        print(f"  Saved BM25S index to {index_dir}")
        return retriever

    def _build_all_examples(self, raw_dataset) -> list[QampariExample]:
        """Convert all raw examples to QampariExample objects.

        Two-pass approach: first build all examples from their ctxs, then
        batch-retrieve BM25S augmentation for any that need more distractors.
        """
        # Pass 1: build examples without BM25S augmentation
        examples = []
        for raw_ex in raw_dataset:
            ex = self._build_one_example(raw_ex)
            if ex is not None:
                examples.append(ex)

        # Pass 2: batch BM25S augmentation for examples that need it
        needs_augmentation = [
            (i, ex) for i, ex in enumerate(examples)
            if len(ex.distractor_passages) < self.min_distractors
        ]
        if needs_augmentation:
            print(f"  {len(needs_augmentation)}/{len(examples)} examples need BM25S augmentation")
            queries = [ex.question for _, ex in needs_augmentation]
            max_k = min(
                self.min_distractors + 200, len(self._all_passages_list)
            )
            query_tokens = self._bm25s.tokenize(queries, stemmer=self._stemmer)
            results, _scores = self._retriever.retrieve(query_tokens, k=max_k)

            for batch_idx, (ex_idx, ex) in enumerate(needs_augmentation):
                needed = self.min_distractors - len(ex.distractor_passages)
                exclude_pids = {p["pid"] for p in ex.gold_passages} | {
                    p["pid"] for p in ex.distractor_passages
                }
                # Text-hash set catches gold passages stored under a different
                # ID (e.g. DPR id vs QAMPARI pid) that exclude_pids would miss.
                gold_texts = {p["text"].strip().lower() for p in ex.gold_passages}
                all_answer_strings = list(ex.answers)
                for aliases in ex.answer_aliases:
                    all_answer_strings.extend(aliases)

                additional = []
                for corpus_idx in results[batch_idx]:
                    if len(additional) >= needed:
                        break
                    passage = self._all_passages_list[int(corpus_idx)]
                    if passage["pid"] in exclude_pids:
                        continue
                    if passage["text"].strip().lower() in gold_texts:
                        continue
                    if _passage_contains_answer(passage, all_answer_strings):
                        continue
                    additional.append(passage)
                    exclude_pids.add(passage["pid"])

                ex.distractor_passages.extend(additional)

        return examples

    @staticmethod
    def _build_one_example(raw_ex: dict) -> QampariExample | None:
        """Build a single QampariExample from a raw JSONL record (no BM25S).

        Actual JSONL schema (from qampari_with_contexts.zip DPR retriever):
          - id: str — question ID (e.g., "806__wikidata_simple__dev")
          - question: str
          - answers: list[str] — answer entity names
          - ans_mappings: dict[str, list[str]] — answer → aliases
          - positive_ctxs: [{text, title, score, chunk_id, pid}] — gold passages
            pid format: "{qid}__{answer_idx}__{proof_idx}"
          - ctxs: [{text, title, id, score}] — retrieved passages (distractors)
        """
        question = raw_ex.get("question", "")
        qid = raw_ex.get("id", "")

        # Build pid → passage lookup from positive_ctxs
        pid_to_passage: dict[str, dict] = {}
        for ctx in raw_ex["positive_ctxs"]:
            pid = ctx["pid"]
            pid_to_passage[pid] = {
                "title": ctx["title"],
                "text": ctx["text"],
                "pid": pid,
            }

        # Build answers, aliases, and answer_to_gold_pids
        answers: list[str] = raw_ex.get("answers", [])
        ans_mappings: dict[str, list[str]] = raw_ex.get("ans_mappings", {})
        answer_aliases: list[list[str]] = [
            ans_mappings.get(ans, []) for ans in answers
        ]

        # Parse answer_to_gold_pids from positive_ctxs pid format:
        # pid = "{qid}__{answer_idx}__{proof_idx}"
        answer_to_gold_pids: dict[int, list[str]] = {}
        gold_pid_set: set[str] = set()
        qid_prefix = qid + "__"
        for ctx in raw_ex["positive_ctxs"]:
            pid = ctx["pid"]
            if pid.startswith(qid_prefix):
                suffix = pid[len(qid_prefix):]
                parts = suffix.split("__")
                if len(parts) >= 2:
                    ans_idx = int(parts[0])
                    answer_to_gold_pids.setdefault(ans_idx, []).append(pid)
                    gold_pid_set.add(pid)

        # Skip examples with no answers or no gold passages
        if not answers or not gold_pid_set:
            return None

        # Deduplicated gold passages
        gold_passages = [pid_to_passage[pid] for pid in gold_pid_set if pid in pid_to_passage]

        # Collect all answer strings for filtering
        all_answer_strings = list(answers)
        for aliases in answer_aliases:
            all_answer_strings.extend(aliases)

        # Distractors from ctxs (retrieved passages) — filter answer-containing ones
        # Note: ctxs use 'id' field, not 'pid'
        gold_chunk_ids = {ctx["chunk_id"] for ctx in raw_ex["positive_ctxs"]}
        distractor_pids_seen: set[str] = set()
        distractor_passages: list[dict] = []
        for ctx in raw_ex.get("ctxs", []):
            pid = ctx["id"]
            if pid in distractor_pids_seen or pid in gold_chunk_ids:
                continue
            passage = {"title": ctx["title"], "text": ctx["text"], "pid": pid}
            if _passage_contains_answer(passage, all_answer_strings):
                continue
            distractor_passages.append(passage)
            distractor_pids_seen.add(pid)

        return QampariExample(
            question=question,
            qid=qid,
            answers=answers,
            answer_aliases=answer_aliases,
            gold_passages=gold_passages,
            distractor_passages=distractor_passages,
            answer_to_gold_pids=answer_to_gold_pids,
        )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> QampariExample:
        return self._examples[idx]


# ---------------------------------------------------------------------------
# Layer 2: TokenizeableQampariExample + QampariPromptDataset
# ---------------------------------------------------------------------------


class TokenizeableQampariExample(TokenizeableExample):
    """A single QAMPARI example that can be tokenized and context-fitted."""

    def __init__(
        self,
        example: QampariExample,
        system_prompt: str,
        prompt_template: str,
        question_position: str,
        py_rng: random.Random,
    ):
        # Items = passages (gold + distractors)
        n_gold = len(example.gold_passages)
        n_distractor = len(example.distractor_passages)
        max_k = n_gold + n_distractor
        super().__init__(max_k=max_k)

        self.example = example
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.question_position = question_position

        # Build passage pool: gold passages + distractors
        self._all_passages = list(example.gold_passages) + list(example.distractor_passages)
        py_rng.shuffle(self._all_passages)

        self._gold_pids = {p["pid"] for p in example.gold_passages}
        self._n_gold = n_gold

        # Per-passage random sort keys for unbiased ordering of any subset.
        # Unlike sorting by shuffled index (which biases gold docs toward the
        # end when k < total), these keys produce a uniform permutation of
        # whatever subset is selected.
        self._passage_sort_keys = {p["pid"]: py_rng.random() for p in self._all_passages}

        # Assigned IDs and rendered docs will be computed per k value
        self._assigned_passages: list[dict] = []
        self._gold_pid_to_doc_id: dict[str, int] = {}

    def _select_passages(self) -> list[dict]:
        """Select passages for current k, ensuring all gold are included."""
        if self.k >= len(self._all_passages):
            return list(self._all_passages)

        if self.k < self._n_gold:
            # Can't fit all gold passages — take as many as possible
            warnings.warn(
                f"k={self.k} < n_gold={self._n_gold}; some gold passages will be excluded"
            )
            # Prioritize gold passages
            gold = [p for p in self._all_passages if p["pid"] in self._gold_pids]
            return gold[:self.k]

        # Ensure all gold are included, fill rest with distractors
        gold = [p for p in self._all_passages if p["pid"] in self._gold_pids]
        non_gold = [p for p in self._all_passages if p["pid"] not in self._gold_pids]
        remaining_slots = self.k - len(gold)
        selected = gold + non_gold[:remaining_slots]

        # Sort by per-passage random keys (assigned at init) so gold docs
        # are uniformly distributed regardless of which subset is selected.
        selected.sort(key=lambda p: self._passage_sort_keys[p["pid"]])
        return selected

    def _build_documents_str(self) -> str:
        """Render selected passages as the documents block."""
        self._assigned_passages = self._select_passages()
        self._gold_pid_to_doc_id = {}

        doc_strs = []
        for i, passage in enumerate(self._assigned_passages):
            doc_id = i + 1  # 1-indexed
            doc_str = _format_document(doc_id, passage["title"], passage["text"])
            doc_strs.append(doc_str)
            if passage["pid"] in self._gold_pids:
                self._gold_pid_to_doc_id[passage["pid"]] = doc_id

        return "\n\n".join(doc_strs)

    def average_length_per_item(self, tokenizer: PreTrainedTokenizerBase) -> float:
        sample_size = min(20, len(self._all_passages))
        sample_strs = [
            _format_document(i + 1, p["title"], p["text"]) + "\n\n"
            for i, p in enumerate(self._all_passages[:sample_size])
        ]
        encodings = tokenizer(sample_strs, add_special_tokens=False)
        lengths = [len(ids) for ids in encodings["input_ids"]]
        return float(np.mean(lengths))

    def build_messages(self) -> list[dict[str, str]]:
        documents_str = self._build_documents_str()
        question_str = f"Question: {self.example.question}"

        user_content = self.prompt_template.replace("{documents}", documents_str)
        user_content = _fill_question_placeholders(
            user_content, question_str, self.question_position
        )

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def tokenized_length(self, tokenizer: PreTrainedTokenizerBase) -> int:
        messages = self.build_messages()
        return len(
            tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        )

    def to_dict(self) -> dict[str, Any]:
        messages = self.build_messages()

        # pos_docs: rendered gold passage strings (must appear verbatim in prompt)
        pos_docs = []
        pos_doc_depths = []
        n_total = len(self._assigned_passages)
        for pid, doc_id in self._gold_pid_to_doc_id.items():
            passage = self._all_passages_dict_by_pid[pid]
            rendered = _format_document(doc_id, passage["title"], passage["text"])
            pos_docs.append(rendered)
            # Depth: position of this doc in the document list (0 = start, 1 = end)
            pos_doc_depths.append((doc_id - 1) / max(n_total - 1, 1))

        # Answer: JSON list (not comma-separated, because answers can contain commas)
        answer = json.dumps(self.example.answers)

        # Gold doc metadata for reward function
        gold_doc_ids = sorted(self._gold_pid_to_doc_id.values())

        # Build answer_to_doc_ids by text containment (more reliable than
        # PID-based QAMPARI annotations, which can map proofs to the wrong
        # answer index).
        answer_to_doc_ids: dict[str, list[int]] = {}
        answer_variants: list[list[str]] = []
        for ans_idx, answer_str in enumerate(self.example.answers):
            # Collect aliases for this answer
            variants = [answer_str]
            if ans_idx < len(self.example.answer_aliases):
                variants.extend(self.example.answer_aliases[ans_idx])
            deduped_variants = _dedup_answer_variants(answer_str, variants[1:])
            answer_variants.append(deduped_variants)

            matching_doc_ids = []
            for passage in self._assigned_passages:
                if passage["pid"] not in self._gold_pids:
                    continue
                doc_id = self._gold_pid_to_doc_id.get(passage["pid"])
                if doc_id is None:
                    continue
                combined = (passage.get("title", "") + " " + passage.get("text", "")).lower()
                if any(v.lower() in combined for v in deduped_variants):
                    matching_doc_ids.append(doc_id)

            if matching_doc_ids:
                answer_to_doc_ids[str(ans_idx)] = matching_doc_ids

        settings = json.dumps({
            "answer_to_doc_ids": answer_to_doc_ids,
            "answer_variants": answer_variants,
            "gold_doc_ids": gold_doc_ids,
            "num_answers": len(self.example.answers),
            "num_gold_passages": len(self.example.gold_passages),
            "num_total_passages": n_total,
            "pos_doc_labels": [f"[{doc_id}]" for doc_id in gold_doc_ids],
        })

        return {
            "prompt": messages,
            "question": self.example.question,
            "answer": answer,
            "pos_docs": pos_docs,
            "pos_doc_depths": pos_doc_depths,
            "type": "qampari",
            "settings": settings,
        }

    @property
    def _all_passages_dict_by_pid(self) -> dict[str, dict]:
        """Lazy lookup of pid → passage dict from the full pool."""
        if not hasattr(self, "_pid_lookup"):
            self._pid_lookup = {p["pid"]: p for p in self._all_passages}
        return self._pid_lookup


# ---------------------------------------------------------------------------
# QampariPromptDataset
# ---------------------------------------------------------------------------


class QampariPromptDataset(Dataset):
    """
    Prompt-building wrapper around QampariDataset.

    Augmentations (all deterministic per example via seeded RNG):
      - Instruction phrasing: sampled from prompt variants
      - Question position: end / beginning / both (sampled per example)
      - Passage ordering: shuffled per example
    """

    def __init__(
        self,
        dataset: QampariDataset,
        tokenizer: PreTrainedTokenizerBase,
        system_prompt: str,
        prompts_dir: str,
        target_context: int = 8000,
        context_length_max: int | None = None,
        seed: int = 0,
        question_position_weights: dict[str, float] | None = None,
        instruction_variants: list[int] | None = None,
        max_gold_ratio: float = 0.7,
    ):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.target_context_min, self.target_context_max = normalize_context_range(
            target_context, context_length_max
        )
        self.system_prompt = system_prompt
        self.max_gold_ratio = max_gold_ratio

        # Question position sampling
        qp_weights = question_position_weights or DEFAULT_QUESTION_POSITION_WEIGHTS
        self._qp_strategies = list(qp_weights.keys())
        self._qp_weights = list(qp_weights.values())

        # Load prompt variants
        self.variants: list[tuple[str, int]] = []
        self._load_variants(prompts_dir, instruction_variants)
        if not self.variants:
            raise ValueError(
                f"No prompt variants found in {prompts_dir}. "
                "Expected subdirs like 0/, 1/, etc."
            )

        # Pre-compute per-example seeds
        ss = np.random.SeedSequence(int(seed))
        children = ss.spawn(len(self.dataset))
        self.base_seeds: list[int] = [
            int(cs.generate_state(1, dtype=np.uint32)[0]) for cs in children
        ]

    def _load_variants(
        self, prompts_dir: str, instruction_variants: list[int] | None = None
    ) -> None:
        """Scan prompts_dir for {variant_num}/prompt.txt files."""
        if not os.path.isdir(prompts_dir):
            raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

        allowed_variants = (
            set(instruction_variants) if instruction_variants is not None else None
        )
        for entry in sorted(os.listdir(prompts_dir)):
            variant_dir = os.path.join(prompts_dir, entry)
            if not os.path.isdir(variant_dir):
                continue
            try:
                variant_num = int(entry)
            except ValueError:
                continue
            if allowed_variants is not None and variant_num not in allowed_variants:
                continue
            prompt_path = os.path.join(variant_dir, "prompt.txt")
            if os.path.isfile(prompt_path):
                with open(prompt_path, "r") as f:
                    self.variants.append((f.read(), variant_num))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example: QampariExample = self.dataset[idx]

        seed_base = self.base_seeds[idx]
        py_rng = random.Random(seed_base)
        sampled_target_context = sample_target_context(
            seed_base, self.target_context_min, self.target_context_max
        )

        # --- Sample augmentation choices (deterministic per example) ---
        variant_list_idx = py_rng.randrange(len(self.variants))
        prompt_template, variant_num = self.variants[variant_list_idx]

        question_position = py_rng.choices(
            self._qp_strategies, weights=self._qp_weights, k=1
        )[0]

        # --- Build tokenizeable example ---
        token_example = TokenizeableQampariExample(
            example=example,
            system_prompt=self.system_prompt,
            prompt_template=prompt_template,
            question_position=question_position,
            py_rng=py_rng,
        )
        input_length = token_example.set_largest_k(
            self.tokenizer, sampled_target_context, initial_step_size=5, min_step_size=1
        )

        # Skip examples where gold passages dominate the context —
        # trivial precision provides no useful GRPO learning signal.
        n_gold_in_context = min(token_example._n_gold, token_example.k)
        gold_ratio = n_gold_in_context / token_example.k if token_example.k > 0 else 1.0
        if gold_ratio > self.max_gold_ratio:
            return None

        result = token_example.to_dict()
        result["id"] = idx
        result["target_context"] = sampled_target_context
        result["input_length"] = input_length
        result["k_used"] = token_example.k
        result["instruction_variant"] = variant_list_idx
        result["question_position"] = question_position
        result["num_gold_passages"] = len(example.gold_passages)
        result["num_answers"] = len(example.answers)
        return result
