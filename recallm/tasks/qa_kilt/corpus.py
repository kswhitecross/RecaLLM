"""
KILT source preparation, window construction, and chunk retrieval.
"""

from __future__ import annotations

import json
import math
import os
import random
import sqlite3
from typing import Iterable

import bm25s
import requests
import Stemmer
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from recallm.tasks.qa_kilt import (
    DEFAULT_DENSE_NPROBE,
    DEFAULT_INDEX_DIR,
    DEFAULT_KILT_SOURCE_DIR,
    DEFAULT_KILT_WINDOW_DIR,
    DEFAULT_QA_BM25_BACKEND,
    DEFAULT_QA_BM25_N_THREADS,
    DEFAULT_WINDOW_SEED,
    QADocument,
    resolve_bm25_n_threads,
)
from recallm.tasks.qa_kilt.dense import (
    corpus_index_dir,
    dense_search_k,
    get_dense_searcher,
    resolve_dense_root,
    validate_index_metadata,
)
from recallm.tasks.qa_kilt.utils import (
    answer_in_document,
    answer_in_text,
    build_window_ranges,
    count_words,
    get_word_spans,
    normalize_title,
    paragraphs_to_article_text,
    word_f1,
    word_index_to_paragraph,
)

_RAW_SOURCE_DIRNAME = "hf_2019_08_01"
_RAW_JSON_FILENAME = "kilt_knowledgesource.json"
_ARTICLE_DATASET_DIRNAME = "articles"
_WINDOW_DATASET_DIRNAME = "windows"
_METADATA_FILENAME = "metadata.json"
_ARTICLE_METADATA_DB_FILENAME = "article_metadata.sqlite"
_RAW_KILT_URL = "http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json"


def _extract_source_article(article: dict) -> tuple[str, str, list[str]]:
    wikipedia_id = (
        article.get("wikipedia_id")
        or article.get("id")
        or article.get("wiki_id")
        or article.get("wikipediaId")
    )
    if wikipedia_id is None:
        raise KeyError(f"Article is missing wikipedia_id-like field: {article.keys()}")
    title = article.get("wikipedia_title") or article.get("title") or ""
    paragraphs = article.get("text") or article.get("paragraphs") or []
    if isinstance(paragraphs, str):
        paragraphs = paragraphs.splitlines()
    if not isinstance(paragraphs, list):
        raise TypeError(f"Unexpected paragraph field type: {type(paragraphs)}")
    return str(wikipedia_id), str(title), [str(paragraph) for paragraph in paragraphs]


def _prepare_articles_batch(batch: dict, *, window_seed: int) -> dict:
    rows = {
        "wikipedia_id": [],
        "title": [],
        "normalized_text": [],
        "paragraph_start_words": [],
        "paragraph_end_words": [],
        "total_words": [],
        "window_start_words": [],
        "window_end_words": [],
        "n_windows": [],
    }
    batch_size = len(next(iter(batch.values()))) if batch else 0
    for idx in range(batch_size):
        article = {key: values[idx] for key, values in batch.items()}
        wikipedia_id, title, paragraphs = _extract_source_article(article)
        normalized_text, paragraph_start_words, paragraph_end_words = paragraphs_to_article_text(paragraphs)
        total_words = count_words(normalized_text)
        window_ranges = build_window_ranges(
            total_words=total_words,
            global_seed=window_seed,
            wikipedia_id=wikipedia_id,
        )
        rows["wikipedia_id"].append(wikipedia_id)
        rows["title"].append(title)
        rows["normalized_text"].append(normalized_text)
        rows["paragraph_start_words"].append(paragraph_start_words)
        rows["paragraph_end_words"].append(paragraph_end_words)
        rows["total_words"].append(total_words)
        rows["window_start_words"].append([start for start, _ in window_ranges])
        rows["window_end_words"].append([end for _, end in window_ranges])
        rows["n_windows"].append(len(window_ranges))
    return rows


def _expand_windows_batch(batch: dict) -> dict:
    rows = {
        "window_id": [],
        "wikipedia_id": [],
        "title": [],
        "text": [],
        "start_word": [],
        "end_word": [],
        "start_char": [],
        "end_char": [],
        "start_paragraph_id": [],
        "end_paragraph_id": [],
        "is_tail_window": [],
    }
    batch_size = len(batch["wikipedia_id"])
    for idx in range(batch_size):
        normalized_text = batch["normalized_text"][idx]
        word_spans = get_word_spans(normalized_text)
        if not word_spans:
            continue
        paragraph_start_words = batch["paragraph_start_words"][idx]
        paragraph_end_words = batch["paragraph_end_words"][idx]
        window_start_words = batch["window_start_words"][idx]
        window_end_words = batch["window_end_words"][idx]
        window_row_start = batch["window_row_start"][idx]
        for local_idx, (start_word, end_word) in enumerate(zip(window_start_words, window_end_words)):
            start_char = word_spans[start_word][0]
            end_char = word_spans[end_word - 1][1]
            rows["window_id"].append(str(window_row_start + local_idx))
            rows["wikipedia_id"].append(batch["wikipedia_id"][idx])
            rows["title"].append(batch["title"][idx])
            rows["text"].append(normalized_text[start_char:end_char])
            rows["start_word"].append(start_word)
            rows["end_word"].append(end_word)
            rows["start_char"].append(start_char)
            rows["end_char"].append(end_char)
            rows["start_paragraph_id"].append(
                word_index_to_paragraph(
                    paragraph_start_words,
                    paragraph_end_words,
                    start_word,
                    use_previous_word=False,
                )
            )
            rows["end_paragraph_id"].append(
                word_index_to_paragraph(
                    paragraph_start_words,
                    paragraph_end_words,
                    end_word,
                    use_previous_word=True,
                )
            )
            rows["is_tail_window"].append(local_idx == len(window_start_words) - 1)
    return rows


class KILTWindowCorpus:
    def __init__(
        self,
        *,
        source_dir: str = DEFAULT_KILT_SOURCE_DIR,
        window_dir: str = DEFAULT_KILT_WINDOW_DIR,
        bm25_index_dir: str = DEFAULT_INDEX_DIR,
        dense_index_dir: str | None = None,
        dense_nprobe: int = DEFAULT_DENSE_NPROBE,
        bm25_backend: str = DEFAULT_QA_BM25_BACKEND,
        bm25_n_threads: int = DEFAULT_QA_BM25_N_THREADS,
        num_proc: int = 8,
        window_seed: int = DEFAULT_WINDOW_SEED,
        rebuild_articles: bool = False,
        rebuild_windows: bool = False,
        rebuild_bm25: bool = False,
        ensure_bm25: bool = True,
    ):
        self.source_dir = source_dir
        self.window_dir = window_dir
        self.bm25_index_dir = bm25_index_dir
        self.dense_index_dir = resolve_dense_root(dense_index_dir)
        self.dense_nprobe = int(dense_nprobe)
        self.bm25_backend = str(bm25_backend)
        self.bm25_n_threads = resolve_bm25_n_threads(bm25_n_threads)
        self.num_proc = max(1, int(num_proc))
        self.window_seed = int(window_seed)
        self.ensure_bm25 = bool(ensure_bm25)

        self.raw_source_path = os.path.join(self.source_dir, _RAW_SOURCE_DIRNAME)
        self.raw_json_path = os.path.join(self.source_dir, _RAW_JSON_FILENAME)
        self.article_dataset_path = os.path.join(self.window_dir, _ARTICLE_DATASET_DIRNAME)
        self.window_dataset_path = os.path.join(self.window_dir, _WINDOW_DATASET_DIRNAME)
        self.metadata_path = os.path.join(self.window_dir, _METADATA_FILENAME)
        self.article_metadata_db_path = os.path.join(self.window_dir, _ARTICLE_METADATA_DB_FILENAME)
        self.bm25_path = os.path.join(self.bm25_index_dir, "KILTWindowCorpus")

        self._article_dataset = None
        self._window_dataset = None
        self._retriever = None
        self._dense_searcher = None
        self._article_db_conn = None
        self._stemmer = Stemmer.Stemmer("english")
        self._title_cache: dict[str, list[dict] | None] = {}
        self._article_cache: dict[str, dict | None] = {}
        self._dense_cache: dict[str, list[int]] = {}
        self._bm25_cache: dict[str, tuple[list[int], list[float]]] = {}
        self._window_row_cache: dict[int, dict] = {}

        self._ensure_artifacts(
            rebuild_articles=rebuild_articles,
            rebuild_windows=rebuild_windows,
            rebuild_bm25=rebuild_bm25,
        )

    @property
    def article_dataset(self):
        if self._article_dataset is None:
            print(f"Loading prepared KILT article dataset from {self.article_dataset_path}")
            self._article_dataset = load_from_disk(self.article_dataset_path)
        return self._article_dataset

    @property
    def window_dataset(self):
        if self._window_dataset is None:
            print(f"Loading prepared KILT window dataset from {self.window_dataset_path}")
            self._window_dataset = load_from_disk(self.window_dataset_path)
        return self._window_dataset

    @property
    def retriever(self):
        if self._retriever is None:
            if not os.path.exists(self.bm25_path):
                raise FileNotFoundError(
                    f"KILT window BM25 index not found at {self.bm25_path}. "
                    "Build it first or construct KILTWindowCorpus with ensure_bm25=True."
                )
            print(f"Loading KILT window BM25 index from {self.bm25_path}")
            self._retriever = bm25s.BM25.load(
                self.bm25_path,
                override_params={"backend": self.bm25_backend},
            )
        return self._retriever

    @property
    def dense_corpus_name(self) -> str:
        return "kilt_windows"

    @property
    def dense_corpus_dir(self) -> str:
        return corpus_index_dir(self.dense_index_dir, self.dense_corpus_name)

    @property
    def article_db(self) -> sqlite3.Connection:
        if self._article_db_conn is None:
            self._article_db_conn = sqlite3.connect(self.article_metadata_db_path)
        return self._article_db_conn

    def _write_metadata(self) -> None:
        os.makedirs(self.window_dir, exist_ok=True)
        with open(self.metadata_path, "w") as handle:
            json.dump(
                {
                    "window_seed": self.window_seed,
                    "min_words": 80,
                    "max_words": 100,
                },
                handle,
                indent=2,
            )

    def _close_article_db(self) -> None:
        if self._article_db_conn is not None:
            self._article_db_conn.close()
            self._article_db_conn = None

    @staticmethod
    def _sqlite_article_row_to_cache_row(row: tuple) -> dict:
        return {
            "wikipedia_id": str(row[0]),
            "title": row[1],
            "paragraph_start_words": json.loads(row[2]),
            "paragraph_end_words": json.loads(row[3]),
            "window_row_start": int(row[4]),
            "n_windows": int(row[5]),
        }

    def _rebuild_article_metadata_db(self) -> None:
        if os.path.exists(self.article_metadata_db_path):
            os.remove(self.article_metadata_db_path)
        self._close_article_db()
        conn = self.article_db
        conn.execute(
            """
            CREATE TABLE articles (
                wikipedia_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                normalized_title TEXT NOT NULL,
                paragraph_start_words TEXT NOT NULL,
                paragraph_end_words TEXT NOT NULL,
                window_row_start INTEGER NOT NULL,
                n_windows INTEGER NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX idx_articles_normalized_title ON articles(normalized_title)")
        article_dataset = self.article_dataset
        total_batches = math.ceil(len(article_dataset) / 10_000)
        print(f"Building persistent article metadata cache at {self.article_metadata_db_path}")
        for batch, _, _ in tqdm(
            self._dataset_batches(article_dataset),
            total=total_batches,
            desc="Caching KILT article metadata to sqlite",
        ):
            rows = []
            batch_len = len(batch["wikipedia_id"])
            for idx in range(batch_len):
                title = str(batch["title"][idx])
                rows.append(
                    (
                        str(batch["wikipedia_id"][idx]),
                        title,
                        normalize_title(title),
                        json.dumps(batch["paragraph_start_words"][idx]),
                        json.dumps(batch["paragraph_end_words"][idx]),
                        int(batch["window_row_start"][idx]),
                        int(batch["n_windows"][idx]),
                    )
                )
            conn.executemany(
                """
                INSERT OR REPLACE INTO articles (
                    wikipedia_id,
                    title,
                    normalized_title,
                    paragraph_start_words,
                    paragraph_end_words,
                    window_row_start,
                    n_windows
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        conn.commit()

    def _ensure_article_metadata_db(self, *, rebuild: bool = False) -> None:
        if rebuild or not os.path.exists(self.article_metadata_db_path):
            self._rebuild_article_metadata_db()

    def _ensure_artifacts(self, *, rebuild_articles: bool, rebuild_windows: bool, rebuild_bm25: bool) -> None:
        os.makedirs(self.source_dir, exist_ok=True)
        os.makedirs(self.window_dir, exist_ok=True)
        os.makedirs(self.bm25_index_dir, exist_ok=True)

        if rebuild_articles:
            rebuild_windows = True
            rebuild_bm25 = True
        if rebuild_windows:
            rebuild_bm25 = True

        if rebuild_articles or not os.path.isdir(self.article_dataset_path):
            self._build_article_dataset()
        self._ensure_article_metadata_db(rebuild=rebuild_articles)
        if rebuild_windows or not os.path.isdir(self.window_dataset_path):
            self._build_window_dataset()
        if self.ensure_bm25 and (rebuild_bm25 or not os.path.exists(self.bm25_path)):
            self._build_bm25_index()
        self._write_metadata()

    def _load_raw_source(self):
        if os.path.isdir(self.raw_source_path):
            print(f"Loading cached KILT source dataset from {self.raw_source_path}")
            return load_from_disk(self.raw_source_path)
        if os.path.exists(self.raw_json_path):
            print(f"Loading raw KILT JSON knowledge source from {self.raw_json_path}")
            return self._load_raw_json_source()

        print("Downloading KILT source dataset with datasets.load_dataset(...)")
        try:
            source_dataset = load_dataset(
                "facebook/kilt_wikipedia",
                "2019-08-01",
                split="full",
            )
        except RuntimeError as exc:
            if "Dataset scripts are no longer supported" not in str(exc):
                raise
            print("datasets rejected the scripted kilt_wikipedia loader.")
            print("Falling back to direct raw KILT knowledge source download.")
            self._download_raw_kilt_json()
            return self._load_raw_json_source()

        print(f"Saving raw KILT source dataset to {self.raw_source_path}")
        source_dataset.save_to_disk(self.raw_source_path)
        return source_dataset

    def _download_raw_kilt_json(self) -> None:
        os.makedirs(self.source_dir, exist_ok=True)
        if os.path.exists(self.raw_json_path):
            size_gb = os.path.getsize(self.raw_json_path) / (1024 ** 3)
            print(f"Raw KILT knowledge source already exists: {self.raw_json_path} ({size_gb:.1f} GB)")
            return

        print(f"Downloading raw KILT knowledge source from {_RAW_KILT_URL}")
        response = requests.get(_RAW_KILT_URL, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with (
            open(self.raw_json_path, "wb") as handle,
            tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc="Downloading raw KILT knowledge source",
            ) as progress,
        ):
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))
        size_gb = os.path.getsize(self.raw_json_path) / (1024 ** 3)
        print(f"Downloaded raw KILT knowledge source to {self.raw_json_path} ({size_gb:.1f} GB)")

    def _load_raw_json_source(self):
        print("Loading raw KILT JSON via datasets json backend")
        return load_dataset("json", data_files=self.raw_json_path, split="train")

    def _build_article_dataset(self) -> None:
        source_dataset = self._load_raw_source()
        print("Preparing normalized KILT article dataset")
        article_dataset = source_dataset.map(
            _prepare_articles_batch,
            batched=True,
            fn_kwargs={"window_seed": self.window_seed},
            num_proc=self.num_proc,
            remove_columns=source_dataset.column_names,
            desc="Normalizing KILT articles",
        )
        counts = article_dataset["n_windows"]
        window_row_start = []
        running_total = 0
        for count in tqdm(counts, desc="Assigning article window offsets"):
            window_row_start.append(running_total)
            running_total += int(count)
        article_dataset = article_dataset.add_column("window_row_start", window_row_start)
        print(
            f"Prepared {len(article_dataset):,} KILT articles and {running_total:,} windows"
        )
        print(f"Saving normalized KILT articles to {self.article_dataset_path}")
        article_dataset.save_to_disk(self.article_dataset_path)
        self._article_dataset = article_dataset

    def _build_window_dataset(self) -> None:
        article_dataset = self.article_dataset
        print("Expanding normalized KILT articles into rolling windows")
        window_dataset = article_dataset.map(
            _expand_windows_batch,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=article_dataset.column_names,
            desc="Building KILT windows",
        )
        print(f"Saving {len(window_dataset):,} KILT windows to {self.window_dataset_path}")
        window_dataset.save_to_disk(self.window_dataset_path)
        self._window_dataset = window_dataset

    def _build_bm25_index(self) -> None:
        window_dataset = self.window_dataset
        print(f"Building BM25 index over {len(window_dataset):,} KILT windows")
        search_text = [
            f"{title}\n{text}"
            for title, text in tqdm(
                zip(window_dataset["title"], window_dataset["text"]),
                total=len(window_dataset),
                desc="Preparing BM25 corpus strings",
            )
        ]
        print("Tokenizing KILT windows for BM25")
        tokenized = bm25s.tokenize(search_text, stopwords="en", stemmer=self._stemmer)
        print("Indexing KILT windows with BM25")
        retriever = bm25s.BM25(backend=self.bm25_backend)
        retriever.index(tokenized, show_progress=True)
        retriever.save(self.bm25_path)
        print(f"Saved KILT BM25 index to {self.bm25_path}")
        self._retriever = retriever

    def _dataset_batches(self, dataset, batch_size: int = 10_000) -> Iterable[dict]:
        total = len(dataset)
        n_batches = math.ceil(total / batch_size)
        for start in range(0, total, batch_size):
            yield dataset[start:start + batch_size], start, n_batches

    def ensure_article_cache(
        self,
        *,
        wikipedia_ids: set[str] | None = None,
        titles: set[str] | None = None,
    ) -> None:
        wanted_ids = {str(wikipedia_id) for wikipedia_id in (wikipedia_ids or set())}
        wanted_titles = {normalize_title(title) for title in (titles or set()) if normalize_title(title)}
        missing_ids = wanted_ids - set(self._article_cache.keys())
        missing_titles = wanted_titles - set(self._title_cache.keys())
        if not missing_ids and not missing_titles:
            return

        print(
            f"Loading cached article metadata for {len(missing_ids):,} wikipedia ids "
            f"and {len(missing_titles):,} titles"
        )
        found_ids = set()
        found_titles = set()
        conn = self.article_db
        for wikipedia_id in tqdm(sorted(missing_ids), desc="Article id cache lookup", disable=not missing_ids):
            row = conn.execute(
                """
                SELECT wikipedia_id, title, paragraph_start_words, paragraph_end_words, window_row_start, n_windows
                FROM articles
                WHERE wikipedia_id = ?
                """,
                (wikipedia_id,),
            ).fetchone()
            if row is None:
                self._article_cache[wikipedia_id] = None
                continue
            self._article_cache[wikipedia_id] = self._sqlite_article_row_to_cache_row(row)
            found_ids.add(wikipedia_id)
        for title_key in tqdm(sorted(missing_titles), desc="Article title cache lookup", disable=not missing_titles):
            rows = conn.execute(
                """
                SELECT wikipedia_id, title, paragraph_start_words, paragraph_end_words, window_row_start, n_windows
                FROM articles
                WHERE normalized_title = ?
                ORDER BY wikipedia_id
                """,
                (title_key,),
            ).fetchall()
            if not rows:
                self._title_cache[title_key] = None
                continue
            self._title_cache[title_key] = [
                self._sqlite_article_row_to_cache_row(row)
                for row in rows
            ]
            found_titles.add(title_key)

        for wikipedia_id in missing_ids - found_ids:
            self._article_cache[wikipedia_id] = None
        for title_key in missing_titles - found_titles:
            self._title_cache[title_key] = None

        print(
            f"Article cache update complete: {len(found_ids):,}/{len(missing_ids):,} ids, "
            f"{len(found_titles):,}/{len(missing_titles):,} titles found"
        )

    def _document_from_row(self, row: dict, *, bm25_score: float = 0.0) -> QADocument:
        return QADocument(
            doc_id=str(row["window_id"]),
            title=row["title"],
            text=row["text"],
            bm25_score=float(bm25_score),
        )

    def require_dense_index(self) -> None:
        searcher = self._get_dense_searcher()
        validate_index_metadata(
            searcher.metadata,
            expected_corpus_name=self.dense_corpus_name,
            expected_row_count=len(self.window_dataset),
            expected_source_paths={
                "qa_kilt_source_dir": os.path.abspath(self.source_dir),
                "qa_kilt_window_dir": os.path.abspath(self.window_dir),
            },
        )

    def _get_dense_searcher(self):
        if self._dense_searcher is None:
            self._dense_searcher = get_dense_searcher(
                self.dense_corpus_dir,
                nprobe=self.dense_nprobe,
            )
        return self._dense_searcher

    def dense_search(self, query: str, k: int) -> list[int]:
        k = min(int(k), len(self.window_dataset))
        if k <= 0:
            return []
        cached = self._dense_cache.get(query)
        if cached is not None and len(cached) >= k:
            return cached[:k]
        self.require_dense_index()
        cached = self._get_dense_searcher().search_text(query, k)
        self._dense_cache[query] = cached
        return cached

    # ------------------------------------------------------------------
    # Batch retrieval helpers (for precompute_retrieval)
    # ------------------------------------------------------------------

    def _get_window_row(self, row_id: int) -> dict:
        """Read-through cache for individual window dataset rows."""
        cached = self._window_row_cache.get(row_id)
        if cached is not None:
            return cached
        row = self.window_dataset[row_id]
        self._window_row_cache[row_id] = row
        return row

    def batch_dense_search(self, queries: list[str], k: int) -> None:
        """Encode all *queries* in one GPU batch, search FAISS once, populate ``_dense_cache``."""
        k = min(int(k), len(self.window_dataset))
        if k <= 0 or not queries:
            return
        # Skip queries already cached at sufficient depth.
        needed = [q for q in queries if q not in self._dense_cache or len(self._dense_cache[q]) < k]
        if not needed:
            return
        self.require_dense_index()
        print(f"  KILTWindowCorpus: batch dense search for {len(needed)} queries (k={k})")
        results = self._get_dense_searcher().batch_search_texts(needed, k)
        for query, row_ids in zip(needed, results):
            self._dense_cache[query] = row_ids

    def batch_bm25_search(self, queries: list[str], k: int, *, chunk_size: int = 64) -> None:
        """Batch-tokenize and batch-retrieve BM25 results, populate ``_bm25_cache``.

        Processes queries in chunks of *chunk_size* to avoid massive memory
        spikes from scoring all queries against the full corpus at once.
        """
        k = min(int(k), len(self.window_dataset))
        if k <= 0 or not queries:
            return
        needed = [q for q in queries if q not in self._bm25_cache or len(self._bm25_cache[q][0]) < k]
        if not needed:
            return
        print(f"  KILTWindowCorpus: batch BM25 search for {len(needed)} queries (k={k})")
        for start in tqdm(range(0, len(needed), chunk_size), desc="BM25 batch search", total=math.ceil(len(needed) / chunk_size)):
            chunk = needed[start:start + chunk_size]
            query_tokens = bm25s.tokenize(
                chunk,
                stopwords="en",
                stemmer=self._stemmer,
                show_progress=False,
            )
            row_ids_batch, scores_batch = self.retriever.retrieve(
                query_tokens,
                k=k,
                n_threads=self.bm25_n_threads,
                show_progress=False,
            )
            for query, rids, scs in zip(chunk, row_ids_batch, scores_batch):
                self._bm25_cache[query] = ([int(r) for r in rids], [float(s) for s in scs])

    def bm25_search(self, query: str, k: int) -> tuple[list[int], list[float]]:
        k = min(int(k), len(self.window_dataset))
        if k <= 0:
            return [], []
        cached = self._bm25_cache.get(query)
        if cached is not None and len(cached[0]) >= k:
            return cached[0][:k], cached[1][:k]
        query_tokens = bm25s.tokenize(
            query,
            stopwords="en",
            stemmer=self._stemmer,
            show_progress=False,
        )
        row_ids, scores = self.retriever.retrieve(
            query_tokens,
            k=k,
            n_threads=self.bm25_n_threads,
            show_progress=False,
        )
        cached_row_ids = [int(r) for r in row_ids[0]]
        cached_scores = [float(s) for s in scores[0]]
        self._bm25_cache[query] = (cached_row_ids, cached_scores)
        return cached_row_ids, cached_scores

    def prefetch_window_rows(self, row_ids: set[int] | list[int], *, chunk_size: int = 50_000) -> None:
        """Batch-fetch window dataset rows into ``_window_row_cache`` via Arrow reads."""
        needed = sorted(set(int(r) for r in row_ids) - set(self._window_row_cache.keys()))
        if not needed:
            return
        print(f"  KILTWindowCorpus: prefetching {len(needed)} window rows")
        for start in tqdm(range(0, len(needed), chunk_size), desc="Prefetching window rows", total=math.ceil(len(needed) / chunk_size)):
            chunk = needed[start:start + chunk_size]
            batch = self.window_dataset[chunk]
            col_names = list(batch.keys())
            for i, row_id in enumerate(chunk):
                self._window_row_cache[row_id] = {col: batch[col][i] for col in col_names}

    def release_planning_resources(self) -> None:
        self._retriever = None
        self._dense_searcher = None
        self._article_dataset = None
        self._window_dataset = None
        self._dense_cache.clear()
        self._bm25_cache.clear()
        self._window_row_cache.clear()
        self._article_cache.clear()
        self._title_cache.clear()
        self._close_article_db()

    # ------------------------------------------------------------------

    def _candidate_window_rows(self, article_row: dict) -> Iterable[tuple[int, dict]]:
        start = int(article_row["window_row_start"])
        count = int(article_row["n_windows"])
        for local_idx in range(count):
            yield local_idx, self._get_window_row(start + local_idx)

    def get_article_metadata(self, wikipedia_id: str) -> dict:
        wikipedia_id = str(wikipedia_id)
        self.ensure_article_cache(wikipedia_ids={wikipedia_id})
        article_row = self._article_cache.get(wikipedia_id)
        if article_row is None:
            raise KeyError(f"Missing KILT article metadata for wikipedia_id={wikipedia_id}")
        return article_row

    def map_provenance_to_document(
        self,
        *,
        wikipedia_id: str,
        start_paragraph_id: int,
        end_paragraph_id: int | None = None,
        answers: list[str] | None = None,
    ) -> QADocument:
        article_row = self.get_article_metadata(wikipedia_id)
        paragraph_start_words = article_row["paragraph_start_words"]
        paragraph_end_words = article_row["paragraph_end_words"]
        if end_paragraph_id is None:
            end_paragraph_id = start_paragraph_id
        if start_paragraph_id < 0 or end_paragraph_id >= len(paragraph_end_words):
            raise IndexError(
                f"Paragraph span [{start_paragraph_id}, {end_paragraph_id}] is out of range "
                f"for wikipedia_id={wikipedia_id} with {len(paragraph_end_words)} paragraphs"
            )
        target_start_word = paragraph_start_words[start_paragraph_id]
        target_end_word = paragraph_end_words[end_paragraph_id]
        best_local_idx = None
        best_overlap = -1
        best_answer_bonus = -1
        for local_idx, row in self._candidate_window_rows(article_row):
            overlap = max(0, min(target_end_word, row["end_word"]) - max(target_start_word, row["start_word"]))
            if overlap <= 0:
                continue
            answer_bonus = int(any(answer_in_text(answer, row["text"]) for answer in (answers or [])))
            if (
                overlap > best_overlap
                or (overlap == best_overlap and answer_bonus > best_answer_bonus)
                or (
                    overlap == best_overlap
                    and answer_bonus == best_answer_bonus
                    and best_local_idx is not None
                    and local_idx < best_local_idx
                )
            ):
                best_local_idx = local_idx
                best_overlap = overlap
                best_answer_bonus = answer_bonus
        if best_local_idx is None:
            raise RuntimeError(
                f"No KILT window overlaps provenance span [{start_paragraph_id}, {end_paragraph_id}] "
                f"for wikipedia_id={wikipedia_id}"
            )
        return self._document_from_row(self._get_window_row(int(article_row["window_row_start"]) + best_local_idx))

    def best_window_by_overlap(
        self,
        *,
        title: str,
        passage_text: str,
        min_f1: float = 0.5,
    ) -> QADocument | None:
        title_key = normalize_title(title)
        if not title_key:
            return None
        self.ensure_article_cache(titles={title_key})
        candidate_articles = self._title_cache.get(title_key) or []
        best_row = None
        best_score = 0.0
        for article_row in candidate_articles:
            for _, window_row in self._candidate_window_rows(article_row):
                score = word_f1(passage_text, window_row["text"])
                if score > best_score:
                    best_row = window_row
                    best_score = score
        if best_row is None or best_score < min_f1:
            return None
        return self._document_from_row(best_row)

    def random_negatives(
        self,
        n_docs: int,
        rng: random.Random,
        exclude_ids: set[str],
        answers: list[str],
    ) -> list[QADocument]:
        negatives = []
        used_ids = set(str(doc_id) for doc_id in exclude_ids)
        attempts = 0
        max_attempts = max(1_000, n_docs * 200)
        total_windows = len(self.window_dataset)
        while len(negatives) < n_docs and attempts < max_attempts:
            attempts += 1
            row_id = rng.randrange(total_windows)
            row = self._get_window_row(row_id)
            doc_id = str(row["window_id"])
            if doc_id in used_ids:
                continue
            if any(answer_in_document(answer, row["title"], row["text"]) for answer in answers):
                continue
            negatives.append(self._document_from_row(row))
            used_ids.add(doc_id)
        return negatives

    def bm25_negatives(
        self,
        query: str,
        n_docs: int,
        exclude_ids: set[str],
        answers: list[str],
    ) -> list[QADocument]:
        if n_docs <= 0:
            return []
        used_ids = set(str(doc_id) for doc_id in exclude_ids)
        negatives = []
        row_ids, scores = self.bm25_search(query, dense_search_k(n_docs, len(self.window_dataset)))
        for row_id, score in zip(row_ids, scores):
            row = self._get_window_row(int(row_id))
            doc_id = str(row["window_id"])
            if doc_id in used_ids:
                continue
            if any(answer_in_document(answer, row["title"], row["text"]) for answer in answers):
                continue
            negatives.append(self._document_from_row(row, bm25_score=float(score)))
            used_ids.add(doc_id)
            if len(negatives) >= n_docs:
                break
        return negatives

    def dense_negatives(
        self,
        query: str,
        n_docs: int,
        exclude_ids: set[str],
        answers: list[str],
    ) -> list[QADocument]:
        if n_docs <= 0:
            return []
        used_ids = {str(doc_id) for doc_id in exclude_ids}
        negatives = []
        for row_id in self.dense_search(query, dense_search_k(n_docs, len(self.window_dataset))):
            row = self._get_window_row(int(row_id))
            doc_id = str(row["window_id"])
            if doc_id in used_ids:
                continue
            if any(answer_in_document(answer, row["title"], row["text"]) for answer in answers):
                continue
            negatives.append(self._document_from_row(row))
            used_ids.add(doc_id)
            if len(negatives) >= n_docs:
                break
        return negatives
