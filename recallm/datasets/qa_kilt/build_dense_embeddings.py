"""
Offline multi-GPU dense embedding builder for qa_kilt corpora.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
import torch.distributed as dist
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from recallm.datasets.qa_kilt import DEFAULT_DENSE_MODEL_NAME
from recallm.datasets.qa_kilt.dense import (
    DEFAULT_DOC_MAX_LENGTH,
    DEFAULT_EMBED_BATCH_SIZE,
    DENSE_CORPUS_NAMES,
    DenseEmbeddingsMetadata,
    corpus_index_dir,
    embeddings_meta_path,
    get_shared_dense_text_encoder,
    load_dense_corpus,
    resolve_dense_root,
    save_json,
    shard_embeddings_path,
    shard_row_ids_path,
)

DEFAULT_NUM_WORKERS = 8
DEFAULT_PREFETCH_FACTOR = 8


def row_bounds_for_rank(total_rows: int, world_size: int, rank: int) -> tuple[int, int]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    start = (int(total_rows) * int(rank)) // int(world_size)
    end = (int(total_rows) * int(rank + 1)) // int(world_size)
    return start, end


def row_indices_for_rank(total_rows: int, world_size: int, rank: int) -> np.ndarray:
    start, end = row_bounds_for_rank(total_rows, world_size, rank)
    return np.arange(start, end, dtype=np.int64)


class DenseShardDataset(Dataset):
    def __init__(self, dense_corpus, start_idx: int, end_idx: int):
        self.dense_corpus = dense_corpus
        self.dataset = dense_corpus.dataset
        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx)

    def __len__(self) -> int:
        return max(0, self.end_idx - self.start_idx)

    def _sample_from_row(self, row: dict) -> dict:
        return {
            "row_id": self.dense_corpus.get_row_id(row),
            "text": self.dense_corpus.get_text(row),
        }

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        row = self.dataset[self.start_idx + int(idx)]
        return self._sample_from_row(row)

    def __getitems__(self, indices: list[int]) -> list[dict]:
        if not indices:
            return []
        global_indices = [self.start_idx + int(idx) for idx in indices]
        batch_rows = self.dataset[global_indices]
        row_count = len(batch_rows[self.dense_corpus.row_id_field])
        return [
            self._sample_from_row({key: values[idx] for key, values in batch_rows.items()})
            for idx in range(row_count)
        ]


class TokenizingCollator:
    def __init__(self, model_name: str, max_length: int):
        self.model_name = model_name
        self.max_length = int(max_length)
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            hf_token = os.environ.get("HF_ACCESS_TOKEN")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token)
        return self._tokenizer

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        tokenizer = self._get_tokenizer()
        texts = [sample["text"] for sample in batch]
        row_ids = torch.as_tensor([sample["row_id"] for sample in batch], dtype=torch.int64)
        tokens = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens["row_ids"] = row_ids
        return tokens


def _distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        kwargs = {"backend": backend}
        if torch.cuda.is_available():
            kwargs["device_id"] = local_rank
        dist.init_process_group(**kwargs)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _barrier_if_needed(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.barrier()

def build_dense_embeddings(args: argparse.Namespace) -> None:
    rank, world_size, local_rank = _distributed_context()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    dense_root = resolve_dense_root(args.dense_dir)
    dense_corpus = load_dense_corpus(
        args.corpus,
        qa_kilt_source_dir=args.qa_kilt_source_dir,
        qa_kilt_window_dir=args.qa_kilt_window_dir,
    )
    corpus_dir = corpus_index_dir(dense_root, args.corpus)
    os.makedirs(corpus_dir, exist_ok=True)

    encoder = get_shared_dense_text_encoder(args.model_name, device=device)
    metadata = DenseEmbeddingsMetadata(
        corpus_name=args.corpus,
        model_name=args.model_name,
        embedding_dim=encoder.embedding_dim,
        row_count=len(dense_corpus.dataset),
        world_size=world_size,
        doc_max_length=args.doc_max_length,
        source_paths={key: os.path.abspath(value) for key, value in dense_corpus.source_paths.items()},
    )
    if rank == 0:
        save_json(embeddings_meta_path(corpus_dir), metadata.__dict__)
    _barrier_if_needed(world_size)

    embedding_path = shard_embeddings_path(corpus_dir, rank)
    row_id_path = shard_row_ids_path(corpus_dir, rank)
    if (
        not args.overwrite
        and os.path.exists(embedding_path)
        and os.path.exists(row_id_path)
    ):
        print(f"[rank {rank}] Reusing existing dense shard {embedding_path}")
        _barrier_if_needed(world_size)
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
        return

    shard_start, shard_end = row_bounds_for_rank(len(dense_corpus.dataset), world_size, rank)
    local_row_count = shard_end - shard_start
    shard_embeddings = open_memmap(
        embedding_path,
        mode="w+",
        dtype=np.float16,
        shape=(local_row_count, encoder.embedding_dim),
    )
    shard_row_ids = np.empty((local_row_count,), dtype=np.int64)

    shard_dataset = DenseShardDataset(dense_corpus, shard_start, shard_end)
    dataloader_kwargs = {
        "dataset": shard_dataset,
        "batch_size": args.embed_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": TokenizingCollator(args.model_name, args.doc_max_length),
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
    data_loader = DataLoader(**dataloader_kwargs)

    write_offset = 0
    total_batches = math.ceil(local_row_count / args.embed_batch_size) if local_row_count else 0
    for batch in tqdm(
        data_loader,
        disable=rank != 0,
        desc=f"Embedding {args.corpus}",
        total=total_batches,
    ):
        batch_row_ids = batch.pop("row_ids").cpu().numpy().astype(np.int64, copy=False)
        tokens = {
            key: value.to(device, non_blocking=torch.cuda.is_available())
            for key, value in batch.items()
        }
        with torch.inference_mode():
            if device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    hidden = encoder.model(**tokens).last_hidden_state[:, 0, :]
            else:
                hidden = encoder.model(**tokens).last_hidden_state[:, 0, :]
        hidden = torch.nn.functional.normalize(hidden.float(), dim=-1)
        embeddings = hidden.cpu().numpy().astype(np.float16, copy=False)
        batch_size = embeddings.shape[0]
        shard_embeddings[write_offset:write_offset + batch_size] = embeddings
        shard_row_ids[write_offset:write_offset + batch_size] = batch_row_ids
        write_offset += batch_size

    np.save(row_id_path, shard_row_ids)
    shard_embeddings.flush()
    print(
        f"[rank {rank}] Saved {len(shard_row_ids):,} dense embeddings to {embedding_path} "
        f"and row ids to {row_id_path}"
    )
    _barrier_if_needed(world_size)
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build dense embedding shards for qa_kilt corpora.")
    parser.add_argument("--corpus", required=True, choices=DENSE_CORPUS_NAMES)
    parser.add_argument("--dense_dir", type=str, default=None,
                        help="Dense index root. Defaults to the qa_kilt gte-modernbert-base root.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_DENSE_MODEL_NAME)
    parser.add_argument("--doc_max_length", type=int, default=DEFAULT_DOC_MAX_LENGTH)
    parser.add_argument("--embed_batch_size", type=int, default=DEFAULT_EMBED_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help="Per-rank DataLoader workers for dense text formatting and tokenization.")
    parser.add_argument("--prefetch_factor", type=int, default=DEFAULT_PREFETCH_FACTOR,
                        help="Per-worker DataLoader prefetch factor.")
    parser.add_argument("--qa_kilt_source_dir", type=str,
                        default=os.environ.get("RECALLM_KILT_SOURCE_DIR"),
                        help="Also settable via RECALLM_KILT_SOURCE_DIR env var.")
    parser.add_argument("--qa_kilt_window_dir", type=str,
                        default=os.environ.get("RECALLM_KILT_WINDOW_DIR"),
                        help="Also settable via RECALLM_KILT_WINDOW_DIR env var.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    build_dense_embeddings(get_parser().parse_args())


if __name__ == "__main__":
    main()
