"""
Dense embedding and FAISS helpers for qa_kilt.
"""

from __future__ import annotations

import importlib
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from recallm.tasks.qa_kilt import (
    DEFAULT_DENSE_INDEX_DIR,
    DEFAULT_DENSE_MODEL_NAME,
    DEFAULT_DENSE_NPROBE,
    DEFAULT_FLAT_INDEX_THRESHOLD,
    DEFAULT_KILT_SOURCE_DIR,
    DEFAULT_KILT_WINDOW_DIR,
)
from recallm.tasks.qa_kilt.planning import initial_retrieval_k

EMBEDDINGS_META_FILENAME = "embeddings_meta.json"
INDEX_FILENAME = "index.faiss"
INDEX_METADATA_FILENAME = "metadata.json"
ROW_IDS_FILENAME = "row_ids.npy"
EMBEDDING_FILENAME_TEMPLATE = "embeddings_rank{rank}.npy"
ROW_ID_FILENAME_TEMPLATE = "row_ids_rank{rank}.npy"
DEFAULT_DOC_MAX_LENGTH = 1024
DEFAULT_QUERY_MAX_LENGTH = 128
DEFAULT_EMBED_BATCH_SIZE = 2048
DENSE_CORPUS_NAMES = (
    "kilt_windows",
    "hotpotqa_paragraphs",
    "musique_paragraphs",
    "2wikimultihopqa_paragraphs",
)

_ENCODER_CACHE: dict[tuple[str, str], "DenseTextEncoder"] = {}
_SEARCHER_CACHE: dict[tuple[str, int, str | None], "DenseFaissSearcher"] = {}


@dataclass(frozen=True)
class DenseIndexMetadata:
    corpus_name: str
    model_name: str
    embedding_dim: int
    row_count: int
    metric: str
    index_type: str
    source_paths: dict[str, str]
    nlist: int | None = None
    nprobe: int | None = None


@dataclass(frozen=True)
class DenseEmbeddingsMetadata:
    corpus_name: str
    model_name: str
    embedding_dim: int
    row_count: int
    world_size: int
    doc_max_length: int
    source_paths: dict[str, str]


@dataclass(frozen=True)
class DenseCorpus:
    name: str
    dataset: Any
    source_paths: dict[str, str]
    row_id_field: str

    def get_row_id(self, row: dict) -> int:
        return int(row[self.row_id_field])

    def get_text(self, row: dict) -> str:
        return f"{row['title']}\n{row['text']}"


def paragraph_corpus_name(dataset_type: str) -> str:
    return f"{dataset_type}_paragraphs"


def require_faiss():
    try:
        return importlib.import_module("faiss")
    except ImportError as exc:
        raise RuntimeError(
            "FAISS is required for qa_kilt dense retrieval. "
            "Install any FAISS build that imports as `faiss`, such as "
            "`faiss-gpu-cu12` on x86 dense-index envs or `faiss-cpu` for CPU-only fallback."
        ) from exc


def round_up_to_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (int(value - 1).bit_length())


def choose_index_type(row_count: int) -> str:
    if int(row_count) < DEFAULT_FLAT_INDEX_THRESHOLD:
        return "flat_ip"
    return "ivf_flat"


def compute_ivf_nlist(row_count: int) -> int:
    base = int(math.sqrt(max(1, int(row_count))))
    clamped = min(65536, max(4096, base))
    return round_up_to_power_of_2(clamped)


def compute_training_sample_size(row_count: int, nlist: int) -> int:
    return min(int(row_count), max(200_000, 100 * int(nlist)))


def dense_search_k(n_docs: int, corpus_size: int) -> int:
    if n_docs <= 0 or corpus_size <= 0:
        return 0
    return min(int(corpus_size), initial_retrieval_k(int(n_docs)))


def embeddings_meta_path(corpus_dir: str) -> str:
    return os.path.join(corpus_dir, EMBEDDINGS_META_FILENAME)


def index_meta_path(corpus_dir: str) -> str:
    return os.path.join(corpus_dir, INDEX_METADATA_FILENAME)


def index_path(corpus_dir: str) -> str:
    return os.path.join(corpus_dir, INDEX_FILENAME)


def row_ids_path(corpus_dir: str) -> str:
    return os.path.join(corpus_dir, ROW_IDS_FILENAME)


def shard_embeddings_path(corpus_dir: str, rank: int) -> str:
    return os.path.join(corpus_dir, EMBEDDING_FILENAME_TEMPLATE.format(rank=int(rank)))


def shard_row_ids_path(corpus_dir: str, rank: int) -> str:
    return os.path.join(corpus_dir, ROW_ID_FILENAME_TEMPLATE.format(rank=int(rank)))


def list_embedding_shards(corpus_dir: str) -> list[tuple[int, str, str]]:
    shards = []
    for filename in sorted(os.listdir(corpus_dir)):
        if not filename.startswith("embeddings_rank") or not filename.endswith(".npy"):
            continue
        rank_str = filename[len("embeddings_rank"):-len(".npy")]
        rank = int(rank_str)
        emb_path = os.path.join(corpus_dir, filename)
        row_path = shard_row_ids_path(corpus_dir, rank)
        if not os.path.exists(row_path):
            raise FileNotFoundError(f"Missing row-id shard for dense embeddings rank {rank}: {row_path}")
        shards.append((rank, emb_path, row_path))
    return shards


def save_json(path: str, payload: dict) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: str) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def validate_index_metadata(
    metadata: DenseIndexMetadata,
    *,
    expected_corpus_name: str,
    expected_model_name: str = DEFAULT_DENSE_MODEL_NAME,
    expected_row_count: int | None = None,
    expected_source_paths: dict[str, str] | None = None,
) -> None:
    if metadata.corpus_name != expected_corpus_name:
        raise RuntimeError(
            f"Dense index corpus mismatch: expected {expected_corpus_name}, "
            f"found {metadata.corpus_name}"
        )
    if metadata.model_name != expected_model_name:
        raise RuntimeError(
            f"Dense index model mismatch for {expected_corpus_name}: "
            f"expected {expected_model_name}, found {metadata.model_name}"
        )
    if expected_row_count is not None and int(metadata.row_count) != int(expected_row_count):
        raise RuntimeError(
            f"Dense index row-count mismatch for {expected_corpus_name}: "
            f"expected {expected_row_count}, found {metadata.row_count}"
        )
    for key, expected_value in (expected_source_paths or {}).items():
        actual_value = metadata.source_paths.get(key)
        if os.path.abspath(str(actual_value)) != os.path.abspath(str(expected_value)):
            raise RuntimeError(
                f"Dense index source-path mismatch for {expected_corpus_name} key={key}: "
                f"expected {expected_value}, found {actual_value}"
            )


def default_dense_root() -> str:
    return os.path.join(DEFAULT_DENSE_INDEX_DIR, DEFAULT_DENSE_MODEL_NAME.replace("/", "__"))


def resolve_dense_root(path: str | None) -> str:
    return os.path.abspath(path or default_dense_root())


def corpus_index_dir(dense_root: str, corpus_name: str) -> str:
    return os.path.join(resolve_dense_root(dense_root), corpus_name)


def _default_runtime_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class DenseTextEncoder:
    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device or _default_runtime_device()
        hf_token = os.environ.get("HF_ACCESS_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token)
        model_kwargs = {}
        if self.device.startswith("cuda"):
            model_kwargs["dtype"] = torch.bfloat16
        self.model = AutoModel.from_pretrained(self.model_name, token=hf_token, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = int(self.model.config.hidden_size)

    def encode_texts(
        self,
        texts: list[str],
        *,
        max_length: int,
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        outputs = []
        autocast_enabled = self.device.startswith("cuda")
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            with torch.inference_mode():
                if autocast_enabled:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        hidden = self.model(**tokens).last_hidden_state[:, 0, :]
                else:
                    hidden = self.model(**tokens).last_hidden_state[:, 0, :]
            hidden = torch.nn.functional.normalize(hidden.float(), dim=-1)
            outputs.append(hidden.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(outputs, axis=0)


def get_shared_dense_text_encoder(model_name: str, device: str | None = None) -> DenseTextEncoder:
    resolved_device = device or _default_runtime_device()
    cache_key = (model_name, resolved_device)
    encoder = _ENCODER_CACHE.get(cache_key)
    if encoder is None:
        encoder = DenseTextEncoder(model_name=model_name, device=resolved_device)
        _ENCODER_CACHE[cache_key] = encoder
    return encoder


class DenseFaissSearcher:
    def __init__(self, *, corpus_dir: str, nprobe: int = DEFAULT_DENSE_NPROBE, device: str | None = None):
        self.corpus_dir = os.path.abspath(corpus_dir)
        self.metadata = DenseIndexMetadata(**load_json(index_meta_path(self.corpus_dir)))
        self._faiss = require_faiss()
        self.index = self._faiss.read_index(index_path(self.corpus_dir))
        self.row_ids = np.load(row_ids_path(self.corpus_dir), mmap_mode="r")
        if len(self.row_ids) != self.metadata.row_count:
            raise RuntimeError(
                f"Dense row-id count mismatch for {self.corpus_dir}: "
                f"{len(self.row_ids)} row ids vs metadata row_count={self.metadata.row_count}"
            )
        if int(self.index.ntotal) != self.metadata.row_count:
            raise RuntimeError(
                f"Dense FAISS ntotal mismatch for {self.corpus_dir}: "
                f"{self.index.ntotal} vectors vs metadata row_count={self.metadata.row_count}"
            )
        if self.metadata.index_type == "ivf_flat":
            self.index.nprobe = int(nprobe)
        self._device = device

    def search_text(self, query: str, k: int) -> list[int]:
        if k <= 0:
            return []
        encoder = get_shared_dense_text_encoder(self.metadata.model_name, device=self._device)
        query_embedding = encoder.encode_texts(
            [query],
            max_length=DEFAULT_QUERY_MAX_LENGTH,
            batch_size=1,
        ).astype(np.float32, copy=False)
        _scores, indices = self.index.search(query_embedding, int(k))
        result = []
        for idx in indices[0]:
            if idx < 0:
                continue
            result.append(int(self.row_ids[int(idx)]))
        return result

    def batch_search_texts(self, queries: list[str], k: int) -> list[list[int]]:
        """Encode all *queries* in one GPU batch and search FAISS in one call."""
        if k <= 0 or not queries:
            return [[] for _ in queries]
        encoder = get_shared_dense_text_encoder(self.metadata.model_name, device=self._device)
        embeddings = encoder.encode_texts(
            queries,
            max_length=DEFAULT_QUERY_MAX_LENGTH,
        ).astype(np.float32, copy=False)
        _scores, indices = self.index.search(embeddings, int(k))
        results: list[list[int]] = []
        for row in indices:
            results.append([int(self.row_ids[int(idx)]) for idx in row if idx >= 0])
        return results


def get_dense_searcher(corpus_dir: str, *, nprobe: int = DEFAULT_DENSE_NPROBE, device: str | None = None) -> DenseFaissSearcher:
    resolved_dir = os.path.abspath(corpus_dir)
    cache_key = (resolved_dir, int(nprobe), device)
    searcher = _SEARCHER_CACHE.get(cache_key)
    if searcher is None:
        searcher = DenseFaissSearcher(corpus_dir=resolved_dir, nprobe=nprobe, device=device)
        _SEARCHER_CACHE[cache_key] = searcher
    return searcher


def release_dense_resources() -> None:
    """Free all GPU/FAISS resources held in module-level caches.

    Call this after ``precompute_retrieval`` has populated the per-query
    dense caches so that forked DataLoader workers do not inherit the
    FAISS indexes or the GPU-resident encoder model.
    """
    for encoder in _ENCODER_CACHE.values():
        del encoder.model
    _ENCODER_CACHE.clear()
    _SEARCHER_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_dense_corpus(
    corpus_name: str,
    *,
    qa_kilt_source_dir: str = DEFAULT_KILT_SOURCE_DIR,
    qa_kilt_window_dir: str = DEFAULT_KILT_WINDOW_DIR,
) -> DenseCorpus:
    if corpus_name not in DENSE_CORPUS_NAMES:
        raise KeyError(
            f"Unknown dense corpus name: {corpus_name}. "
            f"Expected one of {', '.join(DENSE_CORPUS_NAMES)}"
        )
    if corpus_name == "kilt_windows":
        from recallm.tasks.qa_kilt.corpus import KILTWindowCorpus

        corpus = KILTWindowCorpus(
            source_dir=qa_kilt_source_dir,
            window_dir=qa_kilt_window_dir,
            dense_index_dir=resolve_dense_root(None),
            ensure_bm25=False,
        )
        return DenseCorpus(
            name=corpus_name,
            dataset=corpus.window_dataset,
            source_paths={
                "qa_kilt_source_dir": qa_kilt_source_dir,
                "qa_kilt_window_dir": qa_kilt_window_dir,
            },
            row_id_field="window_id",
        )

    if corpus_name == "hotpotqa_paragraphs":
        data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "hotpotqa")
        )
        from recallm.tasks.qa_kilt.load_hotpotqa import get_hotpotqa

        _questions, documents = get_hotpotqa(data_dir)
        return DenseCorpus(
            name=corpus_name,
            dataset=documents,
            source_paths={"hotpotqa_data_dir": data_dir},
            row_id_field="id",
        )

    if corpus_name == "musique_paragraphs":
        from recallm.tasks.qa_kilt.musique import _get_musique_datasets

        data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "musique")
        )
        _questions, documents = _get_musique_datasets(data_dir)
        return DenseCorpus(
            name=corpus_name,
            dataset=documents,
            source_paths={"musique_data_dir": data_dir},
            row_id_field="id",
        )

    if corpus_name == "2wikimultihopqa_paragraphs":
        from recallm.tasks.qa_kilt.wiki_multi_hop import _get_two_wiki_datasets

        data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "2wikimultihopqa")
        )
        _questions, documents = _get_two_wiki_datasets(data_dir)
        return DenseCorpus(
            name=corpus_name,
            dataset=documents,
            source_paths={"twowiki_data_dir": data_dir},
            row_id_field="id",
        )
