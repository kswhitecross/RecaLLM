# RecaLLM Dataset Generation

This module generates training and evaluation datasets for RecaLLM across **10 task categories** and **17 dataset types**. Each dataset is output as a VERL-compatible parquet file.

## Pre-Generated Data

Pre-generated training and evaluation datasets are available on HuggingFace:

```python
from datasets import load_dataset

# Load GRPO training data for a single dataset
ds = load_dataset("kswhitecross/RecaLLM-data", "hotpotqa", split="train")

# Load evaluation data at a specific context length
ds = load_dataset("kswhitecross/RecaLLM-data", "hotpotqa", split="val_32k")

# Available splits per config: train, val_4k, val_8k, val_16k, val_32k, val_64k, val_96k, val_128k
```

If you want to regenerate datasets or create custom variants, use the scripts below.

## Dataset Categories

| Category | Datasets | Description | Augmentations |
|----------|----------|-------------|---------------|
| **Retrieval** | `retrieval`, `multi_niah` | Synthetic key-value lookup and multi-needle-in-a-haystack | KV format (lines/JSON/CSV), needle structure, value type |
| **Reasoning Retrieval** | `math_retrieval` | Solve math problem, then retrieve value by computed key | KV format, math difficulty (n_terms) |
| **Short-Context Math** | `dapo_math`, `mcqa_math` | Math problems with no retrieval required (teaches when NOT to recall) | Instruction variants only |
| **In-Context Learning** | `banking77`, `massive` | Many-shot intent classification via demonstrations | Label format (numeric/text), demo layout, label permutation |
| **Long-Document QA** | `quality` | Multiple-choice QA over long literary articles | Choice format, question position |
| **Aggregation** | `majority_vote`, `top_n_vote` | Vote counting / frequency estimation (recall is counterproductive) | Difficulty level, candidate naming scheme |
| **Reranking** | `msmarco_v2` | Passage relevance ranking with ranked ID output | Document format, negative source (BM25/judged/random) |
| **Entity Citation** | `qampari` | Multi-answer QA with inline passage citations | Document format, negative source |
| **Multi-Hop QA** | `hotpotqa`, `musique`, `2wikimultihopqa` | Multi-hop reasoning over multiple passages | Document format, passage type (chunk/paragraph), negative source |
| **Single-Hop QA** | `nq`, `triviaqa` | Single-passage factoid QA | Document format, passage type, negative source |

## Quick Start: Create a Single Dataset

Self-contained datasets (retrieval, math, ICL, aggregation, quality) need no external data:

```bash
python -m recallm.tasks.create_dataset \
    --type retrieval \
    --save_path ./output/retrieval \
    --target_context 8000 \
    --n_examples 2000 \
    --n_eval_examples 200
```

Output: `train.parquet`, `validation.parquet`, `args.json`, and sample `.txt` files in the save path.

## Batch Creation: All Datasets at Multiple Context Lengths

### Training mode (single context length)
```bash
python -m recallm.tasks.create_datasets \
    --save_dir ./output/training \
    --target_context 8000
```

### Evaluation mode (multiple context lengths)
```bash
python -m recallm.tasks.create_datasets \
    --save_dir ./output/evaluation \
    --target_contexts 4096 8192 16384 32768 65536 98304 120704 \
    --n_examples 10 \
    --n_eval_examples 200
```

Use `--only retrieval multi_niah math_retrieval` to create a subset of dataset types.

## Context Fitting

Each dataset uses `TokenizeableExample`, which performs a binary search over the number of context items (K) to find the largest K that fits the target context length. This enables generating the same dataset at different context lengths simply by changing `--target_context`.

For evaluation, pass `--target_contexts` to the batch orchestrator, which creates separate parquet files per context length.

## Shared Augmentations

All long-context datasets share these augmentations (sampled deterministically per example):

- **Instruction template variation**: Multiple paraphrased prompt templates per dataset
- **Question position**: End (most common), beginning, or both ends of the context
- **Gold position randomization**: Gold information placed at uniform-random depth in context

## External Data Dependencies

Some datasets require external corpora. Set the environment variables below and run the preparation scripts before generating those datasets.

| Dataset Type | External Data | Setup |
|-------------|---------------|-------|
| `msmarco_v2` | MSMARCO V2 passage corpus | `python -m recallm.tasks.reranking.prepare_data` |
| `qampari` | QAMPARI DPR retriever data | `python -m recallm.tasks.citation_qa.download_qampari` |
| QA datasets (`hotpotqa`, `musique`, `2wikimultihopqa`, `nq`, `triviaqa`) | KILT knowledge source + BM25/FAISS indexes | See QA setup below |
| `retrieval`, `multi_niah`, `math_retrieval` | None (self-contained) | -- |
| `dapo_math`, `mcqa_math` | None (uses HuggingFace datasets) | -- |
| `banking77`, `massive` | None (uses HuggingFace datasets) | -- |
| `quality` | None (auto-downloads from GitHub) | -- |
| `majority_vote`, `top_n_vote` | None (self-contained) | -- |

### QA Setup (KILT-based datasets)

1. Download and prepare the KILT knowledge source
2. Build BM25 indexes per corpus type
3. (Optional) Build dense FAISS indexes for dense retrieval

```bash
# Build dense embeddings (requires GPU)
python -m recallm.tasks.qa_kilt.build_dense_embeddings \
    --qa_kilt_source_dir /path/to/kilt \
    --qa_kilt_window_dir /path/to/kilt_windows

# Build FAISS indexes
python -m recallm.tasks.qa_kilt.build_faiss_index \
    --qa_kilt_source_dir /path/to/kilt \
    --qa_kilt_window_dir /path/to/kilt_windows
```

## Environment Variables

| Variable | Used By | Description |
|----------|---------|-------------|
| `RECALLM_MSMARCO_DIR` | `msmarco_v2` | Prepared MSMARCO V2 reranking data directory |
| `IR_DATASETS_HOME` | `msmarco_v2` | ir_datasets cache for passage text lookup |
| `RECALLM_QAMPARI_DIR` | `qampari` | Extracted QAMPARI JSONL.gz data directory |
| `RECALLM_KILT_SOURCE_DIR` | QA datasets | Cached KILT source dataset directory |
| `RECALLM_KILT_WINDOW_DIR` | QA datasets | Prepared KILT article/window artifacts |
| `RECALLM_INDEX_DIR` | QA datasets | BM25 index directory |
| `RECALLM_DENSE_INDEX_DIR` | QA datasets | Dense FAISS index directory |

All paths can also be passed as CLI arguments (e.g., `--rerank_data_dir`, `--qampari_data_dir`). CLI arguments take precedence over environment variables.
