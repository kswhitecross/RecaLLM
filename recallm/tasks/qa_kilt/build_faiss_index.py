"""
Build a reusable FAISS index from precomputed qa_kilt dense embedding shards.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict

import numpy as np

from recallm.tasks.qa_kilt import DEFAULT_DENSE_MODEL_NAME, DEFAULT_DENSE_NPROBE
from recallm.tasks.qa_kilt.dense import (
    DENSE_CORPUS_NAMES,
    DenseEmbeddingsMetadata,
    DenseIndexMetadata,
    choose_index_type,
    compute_ivf_nlist,
    compute_training_sample_size,
    corpus_index_dir,
    embeddings_meta_path,
    index_meta_path,
    index_path,
    list_embedding_shards,
    load_dense_corpus,
    load_json,
    require_faiss,
    resolve_dense_root,
    row_ids_path,
    save_json,
)

DEFAULT_ADD_BATCH_SIZE = 131072
DEFAULT_SAMPLE_SEED = 0
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def load_shard_views(corpus_dir: str) -> list[dict]:
    shard_views = []
    for rank, emb_path, row_path in list_embedding_shards(corpus_dir):
        embeddings = np.load(emb_path, mmap_mode="r")
        row_ids = np.load(row_path, mmap_mode="r")
        if embeddings.shape[0] != len(row_ids):
            raise RuntimeError(
                f"Dense shard length mismatch for rank {rank}: "
                f"{embeddings.shape[0]} embeddings vs {len(row_ids)} row ids"
            )
        shard_views.append(
            {
                "rank": rank,
                "embeddings": embeddings,
                "row_ids": row_ids,
            }
        )
    return shard_views


def sample_training_embeddings(shard_views: list[dict], sample_size: int, seed: int) -> np.ndarray:
    total_rows = sum(int(shard["embeddings"].shape[0]) for shard in shard_views)
    if total_rows == 0:
        raise RuntimeError("Cannot build FAISS index from empty embedding shards")
    if sample_size >= total_rows:
        return np.concatenate(
            [np.asarray(shard["embeddings"], dtype=np.float32) for shard in shard_views],
            axis=0,
        )

    rng = np.random.default_rng(seed)
    sample_positions = rng.choice(total_rows, size=sample_size, replace=False)
    sample_positions.sort()

    embedding_dim = int(shard_views[0]["embeddings"].shape[1])
    sample = np.empty((sample_size, embedding_dim), dtype=np.float32)
    write_offset = 0
    global_offset = 0
    for shard in shard_views:
        shard_rows = int(shard["embeddings"].shape[0])
        mask = (sample_positions >= global_offset) & (sample_positions < global_offset + shard_rows)
        local_positions = sample_positions[mask] - global_offset
        if len(local_positions) > 0:
            batch = np.asarray(shard["embeddings"][local_positions], dtype=np.float32)
            sample[write_offset:write_offset + len(local_positions)] = batch
            write_offset += len(local_positions)
        global_offset += shard_rows
    return sample


def concatenate_embeddings(shard_views: list[dict]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(shard["embeddings"], dtype=np.float32) for shard in shard_views],
        axis=0,
    )


def concatenate_row_ids(shard_views: list[dict]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(shard["row_ids"], dtype=np.int64) for shard in shard_views],
        axis=0,
    )


def create_faiss_index(faiss, embedding_dim: int, row_count: int, nprobe: int):
    index_type = choose_index_type(row_count)
    if index_type == "flat_ip":
        return faiss.IndexFlatIP(embedding_dim), index_type, None, None

    nlist = compute_ivf_nlist(row_count)
    quantizer = faiss.IndexFlatIP(embedding_dim)
    index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = int(nprobe)
    return index, index_type, nlist, int(nprobe)


def configure_faiss_threads(faiss, requested_threads: int | None) -> int | None:
    configured_threads = None
    if requested_threads is not None:
        configured_threads = max(1, int(requested_threads))
        if hasattr(faiss, "omp_set_num_threads"):
            faiss.omp_set_num_threads(configured_threads)
    elif hasattr(faiss, "omp_get_max_threads"):
        configured_threads = int(faiss.omp_get_max_threads())

    env_summary = ", ".join(f"{name}={os.environ.get(name)}" for name in THREAD_ENV_VARS)
    if configured_threads is None:
        print(f"FAISS thread configuration unavailable ({env_summary})")
    else:
        print(f"FAISS CPU thread target: {configured_threads} ({env_summary})")
    return configured_threads


def _maybe_train_on_gpu(faiss, index):
    if not hasattr(faiss, "get_num_gpus"):
        return index, False, None
    num_gpus = int(faiss.get_num_gpus())
    if num_gpus <= 0:
        return index, False, None
    if not hasattr(faiss, "StandardGpuResources") or not hasattr(faiss, "index_cpu_to_gpu"):
        return index, False, None

    resources = faiss.StandardGpuResources()
    try:
        gpu_index = faiss.index_cpu_to_gpu(resources, 0, index)
    except TypeError:
        gpu_index = faiss.index_cpu_to_gpu(resources, 0, index)
    return gpu_index, True, resources


def _maybe_gpu_clone(faiss, index):
    if not hasattr(faiss, "get_num_gpus"):
        return index, False
    num_gpus = int(faiss.get_num_gpus())
    if num_gpus <= 0 or not hasattr(faiss, "StandardGpuResources"):
        return index, False
    if not hasattr(faiss, "index_cpu_to_all_gpus"):
        return index, False
    options = faiss.GpuMultipleClonerOptions()
    options.shard = True
    try:
        return faiss.index_cpu_to_all_gpus(index, co=options), True
    except TypeError:
        return faiss.index_cpu_to_all_gpus(index), True


def build_faiss_index_for_corpus(args: argparse.Namespace) -> DenseIndexMetadata:
    faiss = require_faiss()
    configure_faiss_threads(faiss, args.faiss_threads)
    dense_root = resolve_dense_root(args.dense_dir)
    dense_corpus = load_dense_corpus(
        args.corpus,
        qa_kilt_source_dir=args.qa_kilt_source_dir,
        qa_kilt_window_dir=args.qa_kilt_window_dir,
    )
    corpus_dir = corpus_index_dir(dense_root, args.corpus)
    meta_path = embeddings_meta_path(corpus_dir)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Dense embeddings metadata not found at {meta_path}. "
            "Run build_dense_embeddings.py first."
        )
    embeddings_meta = DenseEmbeddingsMetadata(**load_json(meta_path))
    if embeddings_meta.model_name != args.model_name:
        raise RuntimeError(
            f"Dense embedding model mismatch for {args.corpus}: "
            f"embeddings were built with {embeddings_meta.model_name}, "
            f"but build_faiss_index was asked to use {args.model_name}"
        )
    shard_views = load_shard_views(corpus_dir)
    if not shard_views:
        raise FileNotFoundError(f"No dense embedding shards found under {corpus_dir}")

    total_rows = sum(int(shard["embeddings"].shape[0]) for shard in shard_views)
    if total_rows != embeddings_meta.row_count:
        raise RuntimeError(
            f"Dense shard row-count mismatch for {args.corpus}: "
            f"{total_rows} rows from shards vs metadata row_count={embeddings_meta.row_count}"
        )

    index, index_type, nlist, nprobe = create_faiss_index(
        faiss,
        int(embeddings_meta.embedding_dim),
        total_rows,
        args.nprobe,
    )
    if index_type == "ivf_flat":
        train_size = compute_training_sample_size(total_rows, nlist)
        sample_start = time.perf_counter()
        training_matrix = sample_training_embeddings(shard_views, train_size, args.sample_seed)
        sample_elapsed = time.perf_counter() - sample_start
        print(
            f"Prepared IVF training matrix with {len(training_matrix):,} sampled embeddings "
            f"in {sample_elapsed:.1f}s"
        )

        train_index, train_on_gpu, _gpu_resources = _maybe_train_on_gpu(faiss, index)
        if train_on_gpu:
            print("Training IVF coarse quantizer on GPU 0")
        else:
            print("Training IVF coarse quantizer on CPU FAISS index")
        train_start = time.perf_counter()
        train_index.train(training_matrix)
        train_elapsed = time.perf_counter() - train_start
        print(f"Finished IVF coarse quantizer training in {train_elapsed:.1f}s")
        index = faiss.index_gpu_to_cpu(train_index) if train_on_gpu else train_index

    add_index, used_gpu = _maybe_gpu_clone(faiss, index)
    if used_gpu:
        print(f"Adding embeddings with FAISS GPU sharding across {faiss.get_num_gpus()} GPUs")
    else:
        print("Adding embeddings on CPU FAISS index")

    combined_row_ids = concatenate_row_ids(shard_views)
    add_start = time.perf_counter()
    if used_gpu and index_type == "flat_ip":
        full_embeddings = concatenate_embeddings(shard_views)
        print(
            f"Using single-pass GPU add for flat_ip with {len(full_embeddings):,} vectors "
            "(FAISS sharded flat indexes do not support chunked successive add() calls)"
        )
        add_index.add(full_embeddings)
    else:
        for shard in shard_views:
            embeddings = shard["embeddings"]
            for start in range(0, embeddings.shape[0], args.add_batch_size):
                batch = np.asarray(embeddings[start:start + args.add_batch_size], dtype=np.float32)
                add_index.add(batch)
    add_elapsed = time.perf_counter() - add_start
    print(f"Finished adding {total_rows:,} vectors in {add_elapsed:.1f}s")

    save_start = time.perf_counter()
    final_index = faiss.index_gpu_to_cpu(add_index) if used_gpu else add_index
    faiss.write_index(final_index, index_path(corpus_dir))
    np.save(row_ids_path(corpus_dir), combined_row_ids)
    save_elapsed = time.perf_counter() - save_start
    print(f"Saved dense FAISS artifacts in {save_elapsed:.1f}s")

    metadata = DenseIndexMetadata(
        corpus_name=args.corpus,
        model_name=embeddings_meta.model_name,
        embedding_dim=embeddings_meta.embedding_dim,
        row_count=total_rows,
        metric="inner_product",
        index_type=index_type,
        source_paths={key: os.path.abspath(value) for key, value in dense_corpus.source_paths.items()},
        nlist=nlist,
        nprobe=nprobe,
    )
    save_json(index_meta_path(corpus_dir), asdict(metadata))
    print(f"Saved dense FAISS index to {index_path(corpus_dir)}")
    return metadata


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a reusable FAISS index for qa_kilt dense retrieval.")
    parser.add_argument("--corpus", required=True, choices=DENSE_CORPUS_NAMES)
    parser.add_argument("--dense_dir", type=str, default=None,
                        help="Dense index root. Defaults to the qa_kilt gte-modernbert-base root.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_DENSE_MODEL_NAME)
    parser.add_argument("--nprobe", type=int, default=DEFAULT_DENSE_NPROBE)
    parser.add_argument("--sample_seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--add_batch_size", type=int, default=DEFAULT_ADD_BATCH_SIZE)
    parser.add_argument("--faiss_threads", type=int, default=None,
                        help="Explicit FAISS/OpenMP CPU thread target. Defaults to env if unset.")
    parser.add_argument("--qa_kilt_source_dir", type=str,
                        default=os.environ.get("RECALLM_KILT_SOURCE_DIR"),
                        help="Also settable via RECALLM_KILT_SOURCE_DIR env var.")
    parser.add_argument("--qa_kilt_window_dir", type=str,
                        default=os.environ.get("RECALLM_KILT_WINDOW_DIR"),
                        help="Also settable via RECALLM_KILT_WINDOW_DIR env var.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = get_parser().parse_args()
    corpus_dir = corpus_index_dir(resolve_dense_root(args.dense_dir), args.corpus)
    if (
        not args.overwrite
        and os.path.exists(index_path(corpus_dir))
        and os.path.exists(index_meta_path(corpus_dir))
        and os.path.exists(row_ids_path(corpus_dir))
    ):
        print(f"Reusing existing dense FAISS index under {corpus_dir}")
        return
    build_faiss_index_for_corpus(args)


if __name__ == "__main__":
    main()
