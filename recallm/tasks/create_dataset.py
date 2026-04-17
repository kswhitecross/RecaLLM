"""
Create a single dataset and save it as VERL-format parquet.

Usage:
    python -m recallm.tasks.create_dataset \
        --type retrieval \
        --save_path /path/to/output \
        --target_context 8000 \
        --n_examples 2000 \
        --n_eval_examples 200

Output (in save_path/):
    train.parquet, validation.parquet, args.json, sample txt files.
"""

import argparse
import json
import os

from datasets import Dataset, Features, Sequence, Value
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

from .base import normalize_context_range
from .qa_kilt import DEFAULT_DENSE_INDEX_DIR, DEFAULT_DENSE_MODEL_NAME

# Resolve paths relative to the recallm package directory
_PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DATASETS_DIR = os.path.dirname(__file__)
_DEFAULT_SYSTEM_PROMPT = os.path.join(_PACKAGE_DIR, "system_prompts", "strategic_recall.txt")
_DEFAULT_TOKENIZER = "XX-1/RecaLLM-Qwen2.5-7B"
_DEFAULT_FILTER_DIR = os.path.join(_DATASETS_DIR, "filters")
_CONTEXT_RANGE_UNSUPPORTED_TYPES = {"dapo_math", "mcqa_math", "quality"}


# ---------------------------------------------------------------------------
# VERL parquet schema
# ---------------------------------------------------------------------------

OUTPUT_FEATURES = Features({
    "prompt": [{
        "role": Value("string"),
        "content": Value("string"),
    }],
    "data_source": Value("string"),
    "ability": Value("string"),
    "reward_model": {
        "ground_truth": {
            "answer": Value("string"),
            "pos_docs": Sequence(Value("string")),
            "math_answer": Value("string"),
            "neg_answer": Value("string"),
            "relevance_grades": Value("string"),
        }
    },
    "extra_info": {
        "id": Value("string"),
        "question": Value("string"),
        "question_id": Value("string"),
        "settings": Value("string"),
    },
})

DATASET_TYPE_TO_ABILITY = {
    "retrieval": "retrieval",
    "multi_niah": "retrieval",
    "math_retrieval": "math_to_retrieval",
    "hotpotqa": "rag",
    "musique": "rag",
    "2wikimultihopqa": "rag",
    "nq": "rag",
    "triviaqa": "rag",
    "dapo_math": "math",
    "mcqa_math": "math",
    "banking77": "icl",
    "massive": "icl",
    "msmarco_v2": "reranking",
    "qampari": "citation_qa",
    "quality": "long_doc_qa",
    "majority_vote": "aggregation",
    "threshold_filter": "aggregation",
    "top_n_vote": "aggregation",
}


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def example_to_verl(example: dict, data_source: str) -> dict:
    """Convert a raw dataset example dict to VERL parquet schema."""
    ability = DATASET_TYPE_TO_ABILITY[data_source]

    # Replace recall tags with recall tokens in prompt
    prompt = example["prompt"]
    for msg in prompt:
        msg["content"] = (
            msg["content"]
            .replace("<recall>", "<|start_recall|>")
            .replace("</recall>", "<|end_recall|>")
        )

    # Serialize relevance_grades dict as JSON string for parquet storage
    relevance_grades = example.get("relevance_grades")
    relevance_grades_str = json.dumps(relevance_grades) if relevance_grades is not None else None

    return {
        "prompt": prompt,
        "data_source": data_source,
        "ability": ability,
        "reward_model": {
            "ground_truth": {
                "answer": example["answer"],
                "pos_docs": example.get("pos_docs", []),
                "math_answer": example.get("math_answer", None),
                "neg_answer": example.get("neg_answer", None),
                "relevance_grades": relevance_grades_str,
            }
        },
        "extra_info": {
            "id": str(example.get("id", "")),
            "question": example.get("question", ""),
            "question_id": example.get("question_id", ""),
            "settings": example.get("settings", ""),
        },
    }


def example_to_string(example: dict, tokenizer) -> str:
    """Render a raw example dict as a human-readable string."""
    data_type = example.get('type', '')
    lines = [
        f"ID: {example.get('id', '')}",
        f"Type: {data_type}",
        f"Ability: {DATASET_TYPE_TO_ABILITY.get(data_type, '')}",
        f"Question: {example.get('question', '')}",
        f"Answer: {example.get('answer', '')}",
    ]
    if example.get("math_answer"):
        lines.append(f"Math Answer: {example['math_answer']}")
    if example.get("problem_type"):
        lines.append(f"Problem Type: {example['problem_type']}")
    if example.get("n_terms"):
        lines.append(f"N Terms: {example['n_terms']}")
    # ICL-specific metadata
    if example.get("label_format"):
        lines.append(f"Label Format: {example['label_format']}")
    if example.get("num_labels"):
        lines.append(f"Num Labels: {example['num_labels']}")
    if example.get("num_demos"):
        lines.append(f"Num Demos: {example['num_demos']}")
    # Aggregation-specific metadata
    if example.get("difficulty"):
        lines.append(f"Difficulty: {example['difficulty']}")
    if example.get("n_candidates"):
        lines.append(f"N Candidates: {example['n_candidates']}")
    if example.get("winner_share"):
        lines.append(f"Winner Share: {example['winner_share']}")
    if example.get("n_attributes"):
        lines.append(f"N Attributes: {example['n_attributes']}")
    if example.get("n_conditions"):
        lines.append(f"N Conditions: {example['n_conditions']}")
    if example.get("question_type"):
        lines.append(f"Question Type: {example['question_type']}")
    # Reranking-specific metadata
    if example.get("doc_format"):
        lines.append(f"Doc Format: {example['doc_format']}")
    if example.get("corpus_type"):
        lines.append(f"Corpus Type: {example['corpus_type']}")
    if example.get("n_passages") is not None:
        lines.append(f"N Passages: {example['n_passages']}")
    if example.get("n_relevant") is not None:
        lines.append(f"N Relevant: {example['n_relevant']}")
    # NIAH-specific metadata
    if example.get("settings"):
        lines.append(f"Settings: {example['settings']}")
    if example.get("k_total") is not None:
        lines.append(f"K Total: {example['k_total']}")
    if example.get("v_per_key") is not None:
        lines.append(f"V Per Key: {example['v_per_key']}")
    if example.get("k_query") is not None:
        lines.append(f"K Query: {example['k_query']}")
    if example.get("value_type"):
        lines.append(f"Value Type: {example['value_type']}")
    if example.get("question_position"):
        lines.append(f"Question Position: {example['question_position']}")
    if example.get("neg_answer"):
        lines.append(f"Neg Answer: {example['neg_answer']}")
    if example.get("pos_doc_depths"):
        depths_str = ", ".join(f"{d:.3f}" for d in example['pos_doc_depths'])
        lines.append(f"Pos Doc Depths: [{depths_str}]")
    lines.extend([
        f"Format: {example.get('kv_format', example.get('label_format', ''))}",
        f"Instruction Variant: {example.get('instruction_variant', '')}",
        f"K Used: {example.get('k_used', '')}",
        f"Input Length: {example.get('input_length', '')}",
        f"Target Context: {example.get('target_context', '')}",
        "",
        "Positive Documents:",
    ])
    for doc in example.get("pos_docs", []):
        lines.append(doc)
    lines.append("")
    lines.append("Prompt:")
    lines.append(tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dataset instantiation
# ---------------------------------------------------------------------------

def _create_retrieval_dataset(args, tokenizer, system_prompt, seed, **kwargs):
    from .recall.retrieval import RetrievalDataset

    prompts_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "prompts", "recall", "retrieval")
    )
    question_position_weights = {
        "end": args.qp_end,
        "beginning": args.qp_beginning,
        "both": args.qp_both,
    }
    return RetrievalDataset(
        tokenizer=tokenizer,
        n_examples=args.n_examples,
        system_prompt=system_prompt,
        prompts_dir=prompts_dir,
        target_context=args.target_context,
        context_length_max=args._effective_context_length_max,
        key_min=args.key_min,
        key_max=args.key_max,
        value_size=args.value_size,
        seed=seed,
        question_position_weights=question_position_weights,
        kv_formats=args.kv_formats,
        instruction_variants=args.instruction_variants,
    )


def _create_math_retrieval_dataset(args, tokenizer, system_prompt, seed, **kwargs):
    from .reasoning_retrieval.math_retrieval import MathRetrievalDataset

    prompts_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "prompts", "reasoning_retrieval", "math_retrieval")
    )
    question_position_weights = {
        "end": args.qp_end,
        "beginning": args.qp_beginning,
        "both": args.qp_both,
    }
    return MathRetrievalDataset(
        tokenizer=tokenizer,
        n_examples=args.n_examples,
        system_prompt=system_prompt,
        prompts_dir=prompts_dir,
        target_context=args.target_context,
        context_length_max=args._effective_context_length_max,
        seed=seed,
        question_position_weights=question_position_weights,
        kv_formats=args.kv_formats,
        instruction_variants=args.instruction_variants,
        n_terms_min=args.n_terms_min,
        n_terms_max=args.n_terms_max,
        coeff_min=args.coeff_min,
        coeff_max=args.coeff_max,
        dict_val_size=args.value_size,
    )


def _create_icl_dataset(args, tokenizer, system_prompt, seed, dataset_cls, dataset_task_description):
    """Shared factory for ICL datasets (Banking77, MASSIVE)."""
    from .icl.icl import ICLPromptDataset

    prompts_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "prompts", "icl")
    )
    from .icl.icl import _AVG_DEMO_TOKEN_LEN

    # Compute safe upper bound on demos: target_context / avg_demo_len * 2x safety
    _, target_context_max = normalize_context_range(
        args.target_context, args._effective_context_length_max
    )
    max_demos = max(100, int(target_context_max / _AVG_DEMO_TOKEN_LEN * 2))
    raw_dataset = dataset_cls(n_examples=args.n_examples, seed=seed, max_demos=max_demos)
    return ICLPromptDataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        prompts_dir=prompts_dir,
        dataset_task_description=dataset_task_description,
        target_context=args.target_context,
        context_length_max=args._effective_context_length_max,
        seed=seed,
        label_formats=args.label_formats,
        instruction_variants=args.instruction_variants,
    )


def _create_banking77_dataset(args, tokenizer, system_prompt, seed, **kwargs):
    from .icl.banking77 import Banking77Dataset

    return _create_icl_dataset(
        args, tokenizer, system_prompt, seed,
        dataset_cls=Banking77Dataset,
        dataset_task_description="Each text is a banking customer service inquiry. Labels represent the customer's intent.",
    )


def _create_massive_dataset(args, tokenizer, system_prompt, seed, **kwargs):
    from .icl.massive import MassiveDataset

    return _create_icl_dataset(
        args, tokenizer, system_prompt, seed,
        dataset_cls=MassiveDataset,
        dataset_task_description="Each text is a virtual assistant command. Labels represent the user's intent.",
    )


def _create_dapo_math_dataset(args, tokenizer, system_prompt, seed, split="train"):
    from .short_context_math.dapo_math import DAPOMathDataset

    prompts_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "prompts", "short_context_math", "dapo_math")
    )
    return DAPOMathDataset(
        tokenizer=tokenizer,
        n_examples=args.n_examples,
        system_prompt=system_prompt,
        prompts_dir=prompts_dir,
        seed=seed,
        instruction_variants=args.instruction_variants,
        split=split,
    )


def _create_mcqa_math_dataset(args, tokenizer, system_prompt, seed, split="train"):
    from .short_context_math.mcqa_math import MCQAMathDataset

    prompts_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "prompts", "short_context_math", "mcqa_math")
    )
    return MCQAMathDataset(
        tokenizer=tokenizer,
        n_examples=args.n_examples,
        system_prompt=system_prompt,
        prompts_dir=prompts_dir,
        seed=seed,
        instruction_variants=args.instruction_variants,
        min_level=args.mcqa_math_min_level,
        max_level=args.mcqa_math_max_level,
        split=split,
    )


def _create_quality_dataset(args, tokenizer, system_prompt, seed, split="train"):
    from .long_doc_qa.quality import QualityDataset

    prompts_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "prompts", "long_doc_qa", "quality")
    )
    question_position_weights = {
        "end": args.qp_end,
        "beginning": args.qp_beginning,
        "both": args.qp_both,
    }
    return QualityDataset(
        tokenizer=tokenizer,
        n_examples=args.n_examples,
        system_prompt=system_prompt,
        prompts_dir=prompts_dir,
        target_context=args.target_context,
        min_context=args.min_context,
        seed=seed,
        question_position_weights=question_position_weights,
        instruction_variants=args.instruction_variants,
        split=split,
    )


def _create_multi_niah_dataset(args, tokenizer, system_prompt, seed, **kwargs):
    from .recall.multi_niah import MultiNIAHDataset

    prompts_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "prompts", "recall", "multi_niah")
    )
    question_position_weights = {
        "end": args.qp_end,
        "beginning": args.qp_beginning,
        "both": args.qp_both,
    }
    return MultiNIAHDataset(
        tokenizer=tokenizer,
        n_examples=args.n_examples,
        system_prompt=system_prompt,
        prompts_dir=prompts_dir,
        target_context=args.target_context,
        context_length_max=args._effective_context_length_max,
        seed=seed,
        question_position_weights=question_position_weights,
        instruction_variants=args.instruction_variants,
        k_total_range=(args.niah_k_total_min, args.niah_k_total_max),
        v_range=(args.niah_v_min, args.niah_v_max),
        max_total_needles=args.niah_max_total_needles,
        k_query_max=args.niah_k_query_max,
    )


def _create_majority_vote_dataset(args, tokenizer, system_prompt, seed, **kwargs):
    from .aggregation.majority_vote import MajorityVoteDataset

    prompts_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "prompts", "aggregation", "majority_vote")
    )
    question_position_weights = {
        "end": args.qp_end,
        "beginning": args.qp_beginning,
        "both": args.qp_both,
    }
    difficulty_weights = {
        "easy": args.diff_easy,
        "medium": args.diff_medium,
        "hard": args.diff_hard,
    }
    return MajorityVoteDataset(
        tokenizer=tokenizer,
        n_examples=args.n_examples,
        system_prompt=system_prompt,
        prompts_dir=prompts_dir,
        target_context=args.target_context,
        context_length_max=args._effective_context_length_max,
        seed=seed,
        question_position_weights=question_position_weights,
        difficulty_weights=difficulty_weights,
        instruction_variants=args.instruction_variants,
    )


def _create_threshold_filter_dataset(args, tokenizer, system_prompt, seed, **kwargs):
    from .aggregation.threshold_filter import ThresholdFilterDataset

    prompts_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "prompts", "aggregation", "threshold_filter")
    )
    question_position_weights = {
        "end": args.qp_end,
        "beginning": args.qp_beginning,
        "both": args.qp_both,
    }
    difficulty_weights = {
        "easy": args.diff_easy,
        "medium": args.diff_medium,
        "hard": args.diff_hard,
    }
    return ThresholdFilterDataset(
        tokenizer=tokenizer,
        n_examples=args.n_examples,
        system_prompt=system_prompt,
        prompts_dir=prompts_dir,
        target_context=args.target_context,
        context_length_max=args._effective_context_length_max,
        seed=seed,
        question_position_weights=question_position_weights,
        difficulty_weights=difficulty_weights,
        instruction_variants=args.instruction_variants,
    )


def _create_top_n_vote_dataset(args, tokenizer, system_prompt, seed, **kwargs):
    from .aggregation.top_n_vote import TopNVoteDataset

    prompts_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "prompts", "aggregation", "top_n_vote")
    )
    question_position_weights = {
        "end": args.qp_end,
        "beginning": args.qp_beginning,
        "both": args.qp_both,
    }
    difficulty_weights = {
        "easy": args.diff_easy,
        "medium": args.diff_medium,
        "hard": args.diff_hard,
    }
    return TopNVoteDataset(
        tokenizer=tokenizer,
        n_examples=args.n_examples,
        system_prompt=system_prompt,
        prompts_dir=prompts_dir,
        target_context=args.target_context,
        context_length_max=args._effective_context_length_max,
        seed=seed,
        question_position_weights=question_position_weights,
        difficulty_weights=difficulty_weights,
        instruction_variants=args.instruction_variants,
    )


def _create_qampari_dataset(args, tokenizer, system_prompt, seed, split="train"):
    from .citation_qa.qampari import QampariPromptDataset, QampariDataset

    prompts_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "prompts", "citation_qa", "qampari")
    )
    # QAMPARI ignores --qp_* args: citation QA is the one category
    # that places the question BEFORE documents, so we hard-code 60%
    # beginning to match the ALCE citation format.
    question_position_weights = None  # uses qampari DEFAULT_QUESTION_POSITION_WEIGHTS
    # Oversample to compensate for max_gold_ratio filtering
    oversample = int(args.n_examples * 1.3) + 10
    raw_dataset = QampariDataset(
        data_dir=args.qampari_data_dir,
        n_examples=oversample,
        seed=seed,
        split=split,
        min_distractors=args.qampari_min_distractors,
    )
    return QampariPromptDataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        prompts_dir=prompts_dir,
        target_context=args.target_context,
        context_length_max=args._effective_context_length_max,
        seed=seed,
        question_position_weights=question_position_weights,
        instruction_variants=args.instruction_variants,
        max_gold_ratio=args.qampari_max_gold_ratio,
    )


def _create_msmarco_v2_dataset(args, tokenizer, system_prompt, seed, split="train"):
    from .reranking.msmarco_v2 import MSMARCOv2Dataset
    from .reranking.reranking import RerankingPromptDataset

    prompts_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "prompts", "reranking", "msmarco_v2")
    )
    question_position_weights = {
        "end": args.qp_end,
        "beginning": args.qp_beginning,
        "both": args.qp_both,
    }
    neg_source_weights = {
        "judged": args.rerank_neg_judged_weight,
        "bm25": args.rerank_neg_bm25_weight,
        "random": args.rerank_neg_random_weight,
    }
    # Oversample to compensate for degenerate examples (k=0 passages) filtered by Layer 2
    oversample = int(args.n_examples * 1.05) + 5
    raw_dataset = MSMARCOv2Dataset(
        data_dir=args.rerank_data_dir,
        ir_datasets_home=args.ir_datasets_home,
        n_examples=oversample,
        seed=seed,
        split=split,
        pool_size=args.rerank_pool_size,
        neg_source_weights=neg_source_weights,
        pos_ratio_range=(args.rerank_pos_ratio_min, args.rerank_pos_ratio_max),
    )
    return RerankingPromptDataset(
        raw_dataset=raw_dataset,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        prompts_dir=prompts_dir,
        target_context=args.target_context,
        context_length_max=args._effective_context_length_max,
        seed=seed,
        question_position_weights=question_position_weights,
        instruction_variants=args.instruction_variants,
    )


# ---------------------------------------------------------------------------
# QA dataset factories (shared pattern)
# ---------------------------------------------------------------------------

# Dual-corpus QA datasets: multi-hop datasets still mix chunk and paragraph
# modes. Single-hop QA is chunk-only in qa_kilt.
_DUAL_CORPUS_QA_TYPES = {"hotpotqa", "musique", "2wikimultihopqa"}


def _enforce_corpus_ratio(data: list[dict], n_target: int, chunk_weight: float) -> list[dict]:
    """Enforce chunk:paragraph ratio after materialization.

    Splits into chunk and paragraph pools, trims each to target count
    based on chunk_weight. If one side is short, gives slack to the other.
    """
    chunks = [ex for ex in data if ex.get("corpus_type") == "chunk"]
    paras = [ex for ex in data if ex.get("corpus_type") == "paragraph"]

    n_chunk = int(n_target * chunk_weight)
    n_para = n_target - n_chunk

    # Cap to available
    n_chunk = min(n_chunk, len(chunks))
    n_para = min(n_para, len(paras))

    # If one side is short, give the slack to the other
    total = n_chunk + n_para
    if total < n_target:
        shortfall = n_target - total
        extra_chunks = len(chunks) - n_chunk
        extra_paras = len(paras) - n_para
        if extra_chunks > 0:
            add = min(shortfall, extra_chunks)
            n_chunk += add
            shortfall -= add
        if shortfall > 0 and extra_paras > 0:
            n_para += min(shortfall, extra_paras)

    result = chunks[:n_chunk] + paras[:n_para]

    actual_chunk = min(n_chunk, len(chunks))
    actual_para = len(result) - actual_chunk
    total = len(result)
    if total > 0:
        print(f"  Corpus ratio enforced: {actual_chunk} chunk ({actual_chunk/total*100:.1f}%), "
              f"{actual_para} paragraph ({actual_para/total*100:.1f}%) "
              f"(target {chunk_weight*100:.0f}%/{(1-chunk_weight)*100:.0f}%)")

    return result


# Cached corpus instance shared across QA datasets within a run
_kilt_window_corpus_cache = None
_kilt_window_corpus_cache_key = None


def _get_kilt_window_corpus(args):
    global _kilt_window_corpus_cache, _kilt_window_corpus_cache_key
    cache_key = (
        os.path.abspath(args.qa_kilt_source_dir),
        os.path.abspath(args.qa_kilt_window_dir),
        os.path.abspath(args.qa_bm25_index_dir),
        os.path.abspath(args.qa_dense_index_dir),
        int(args.qa_dense_nprobe),
        str(args.qa_bm25_backend),
        int(args.qa_bm25_n_threads),
        int(args.qa_kilt_num_proc),
        bool(args.qa_kilt_rebuild_articles),
        bool(args.qa_kilt_rebuild_windows),
        bool(args.qa_kilt_rebuild_bm25),
    )
    if _kilt_window_corpus_cache is None or _kilt_window_corpus_cache_key != cache_key:
        from .qa_kilt.corpus import KILTWindowCorpus

        _kilt_window_corpus_cache = KILTWindowCorpus(
            source_dir=args.qa_kilt_source_dir,
            window_dir=args.qa_kilt_window_dir,
            bm25_index_dir=args.qa_bm25_index_dir,
            dense_index_dir=args.qa_dense_index_dir,
            dense_nprobe=args.qa_dense_nprobe,
            bm25_backend=args.qa_bm25_backend,
            bm25_n_threads=args.qa_bm25_n_threads,
            num_proc=args.qa_kilt_num_proc,
            rebuild_articles=args.qa_kilt_rebuild_articles,
            rebuild_windows=args.qa_kilt_rebuild_windows,
            rebuild_bm25=args.qa_kilt_rebuild_bm25,
        )
        _kilt_window_corpus_cache_key = cache_key
    return _kilt_window_corpus_cache


def _create_qa_dataset(layer1_cls, args, tokenizer, system_prompt, seed, split="train"):
    """Shared factory for all 5 QA datasets."""
    from .qa_kilt.prompt import QAPromptDataset

    prompts_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "prompts", "qa")
    )
    question_position_weights = {
        "end": args.qp_end,
        "beginning": args.qp_beginning,
        "both": args.qp_both,
    }
    corpus_type_weights = {
        "chunk": args.qa_corpus_chunk_weight,
        "paragraph": 1.0 - args.qa_corpus_chunk_weight,
    }
    neg_type_weights = {
        "dense": args.qa_neg_dense_weight,
        "bm25": args.qa_neg_bm25_weight,
        "random": args.qa_neg_random_weight,
        "mixed": args.qa_neg_mixed_weight,
    }
    kilt_corpus = _get_kilt_window_corpus(args)

    # Load filter IDs if applicable
    filter_ids = None
    if args.qa_filter != "none" and split == "train":
        filter_path = os.path.join(
            args.qa_filter_dir, layer1_cls.DATASET_TYPE, args.qa_filter, "ids.json"
        )
        if os.path.exists(filter_path):
            with open(filter_path) as f:
                filter_ids = json.load(f)["id"]
            print(f"  Loaded {len(filter_ids)} filter IDs from {filter_path}")
    elif args.qa_filter != "none" and split != "train":
        print(f"  Skipping qa_filter for {layer1_cls.DATASET_TYPE} split={split}; using full split")
    max_context = args._effective_context_length_max or args.target_context

    # Build Layer 1 kwargs
    l1_kwargs = dict(
        kilt_corpus=kilt_corpus,
        tokenizer=tokenizer,
        max_context=max_context,
        n_examples=args.n_examples,
        seed=seed,
        neg_type_weights=neg_type_weights,
        filter_ids=filter_ids,
        split=split,
    )

    # Multi-hop datasets have their own BM25 index over HF paragraphs
    if layer1_cls.DATASET_TYPE in ("hotpotqa", "musique", "2wikimultihopqa"):
        l1_kwargs["corpus_type_weights"] = corpus_type_weights
        l1_kwargs["bm25_index_dir"] = args.qa_bm25_index_dir
        l1_kwargs["dense_index_dir"] = args.qa_dense_index_dir
        l1_kwargs["dense_nprobe"] = args.qa_dense_nprobe
        l1_kwargs["bm25_backend"] = args.qa_bm25_backend
        l1_kwargs["bm25_n_threads"] = args.qa_bm25_n_threads
        l1_kwargs["strict_corpus_type"] = args.qa_strict_corpus_type
        l1_kwargs["num_proc"] = args.qa_kilt_num_proc

    raw_dataset = layer1_cls(**l1_kwargs)

    return QAPromptDataset(
        raw_dataset=raw_dataset,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        prompts_dir=prompts_dir,
        target_context=args.target_context,
        context_length_max=args._effective_context_length_max,
        seed=seed,
        question_position_weights=question_position_weights,
        instruction_variants=args.instruction_variants,
    )


def _release_qa_planning_resources(*prompt_datasets) -> None:
    from .qa_kilt.dense import release_dense_resources
    from .qa_kilt.multihop_base import BaseKILTMultiHopDataset

    kilt_corpora = {}
    release_multihop_shared = False
    for dataset in prompt_datasets:
        raw_dataset = getattr(dataset, "raw_dataset", None)
        if raw_dataset is None:
            continue
        raw_dataset.release_planning_resources()
        corpus = getattr(raw_dataset, "kilt_corpus", None)
        if corpus is not None:
            kilt_corpora[id(corpus)] = corpus
        if isinstance(raw_dataset, BaseKILTMultiHopDataset):
            release_multihop_shared = True
    for corpus in kilt_corpora.values():
        corpus.release_planning_resources()
    if release_multihop_shared:
        BaseKILTMultiHopDataset.release_shared_planning_resources()
    release_dense_resources()


def _create_nq_dataset(args, tokenizer, system_prompt, seed, split="train"):
    from .qa_kilt.nq import NQLayer1Dataset

    return _create_qa_dataset(NQLayer1Dataset, args, tokenizer, system_prompt, seed, split=split)


def _create_triviaqa_dataset(args, tokenizer, system_prompt, seed, split="train"):
    from .qa_kilt.triviaqa import TriviaQALayer1Dataset

    return _create_qa_dataset(TriviaQALayer1Dataset, args, tokenizer, system_prompt, seed, split=split)


def _create_hotpotqa_dataset(args, tokenizer, system_prompt, seed, split="train"):
    from .qa_kilt.hotpotqa import HotPotQALayer1Dataset

    return _create_qa_dataset(HotPotQALayer1Dataset, args, tokenizer, system_prompt, seed, split=split)


def _create_musique_dataset(args, tokenizer, system_prompt, seed, split="train"):
    from .qa_kilt.musique import MuSiQueLayer1Dataset

    return _create_qa_dataset(MuSiQueLayer1Dataset, args, tokenizer, system_prompt, seed, split=split)


def _create_2wikimultihopqa_dataset(args, tokenizer, system_prompt, seed, split="train"):
    from .qa_kilt.wiki_multi_hop import TwoWikiMultihopQALayer1Dataset

    return _create_qa_dataset(TwoWikiMultihopQALayer1Dataset, args, tokenizer, system_prompt, seed, split=split)


DATASET_FACTORIES = {
    "retrieval": _create_retrieval_dataset,
    "multi_niah": _create_multi_niah_dataset,
    "math_retrieval": _create_math_retrieval_dataset,
    "banking77": _create_banking77_dataset,
    "massive": _create_massive_dataset,
    "dapo_math": _create_dapo_math_dataset,
    "mcqa_math": _create_mcqa_math_dataset,
    "quality": _create_quality_dataset,
    "majority_vote": _create_majority_vote_dataset,
    "threshold_filter": _create_threshold_filter_dataset,
    "top_n_vote": _create_top_n_vote_dataset,
    "msmarco_v2": _create_msmarco_v2_dataset,
    "qampari": _create_qampari_dataset,
    "hotpotqa": _create_hotpotqa_dataset,
    "musique": _create_musique_dataset,
    "2wikimultihopqa": _create_2wikimultihopqa_dataset,
    "nq": _create_nq_dataset,
    "triviaqa": _create_triviaqa_dataset,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def create_dataset(args: argparse.Namespace):
    set_seed(args.seed)
    normalize_context_range(args.target_context, args.context_length_max)

    effective_context_length_max = args.context_length_max
    if args.type in _CONTEXT_RANGE_UNSUPPORTED_TYPES and effective_context_length_max is not None:
        print(
            f"WARNING: Dataset type '{args.type}' does not support --context_length_max; "
            f"using fixed target_context={args.target_context}."
        )
        effective_context_length_max = None
    args._effective_context_length_max = effective_context_length_max

    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, "args.json"), "w") as f:
        json.dump(
            {k: v for k, v in vars(args).items() if not k.startswith("_")},
            f,
            indent=4,
        )

    # Load tokenizer
    hf_token = os.environ.get("HF_ACCESS_TOKEN", None)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, token=hf_token)

    # Load system prompt
    with open(args.system_prompt, "r") as f:
        system_prompt = f.read()

    # Get factory for this dataset type
    factory = DATASET_FACTORIES[args.type]

    # Create train and eval datasets
    train_dataset = factory(args, tokenizer, system_prompt, seed=args.seed, split="train")
    # Temporarily override n_examples for eval
    orig_n = args.n_examples
    args.n_examples = args.n_eval_examples
    eval_dataset = factory(args, tokenizer, system_prompt, seed=args.seed + 1, split="validation")
    args.n_examples = orig_n
    if args.type in {"hotpotqa", "musique", "2wikimultihopqa", "nq", "triviaqa"}:
        _release_qa_planning_resources(train_dataset, eval_dataset)

    dataloader_workers = args.num_workers

    # Materialize (datasets may return None to skip degenerate examples)
    train_data = []
    n_skipped_train = 0
    for example in tqdm(
        DataLoader(train_dataset, batch_size=1, num_workers=dataloader_workers, shuffle=False, collate_fn=lambda x: x[0]),
        desc="Materializing train",
        total=len(train_dataset),
    ):
        if example is None:
            n_skipped_train += 1
            continue
        train_data.append(example)

    eval_data = []
    n_skipped_eval = 0
    for example in tqdm(
        DataLoader(eval_dataset, batch_size=1, num_workers=dataloader_workers, shuffle=False, collate_fn=lambda x: x[0]),
        desc="Materializing eval",
        total=len(eval_dataset),
    ):
        if example is None:
            n_skipped_eval += 1
            continue
        eval_data.append(example)

    if n_skipped_train or n_skipped_eval:
        print(f"Skipped {n_skipped_train} train, {n_skipped_eval} eval examples (filtered by dataset).")

    # Enforce corpus type ratio for dual-corpus QA datasets, then truncate
    chunk_weight = getattr(args, "qa_corpus_chunk_weight", 0.5)
    is_dual_corpus_qa = (args.type in _DUAL_CORPUS_QA_TYPES
                         and 0 < chunk_weight < 1)
    ratio_preplanned = bool(getattr(getattr(train_dataset, "raw_dataset", None), "_ratio_is_preplanned", False))
    if is_dual_corpus_qa and not ratio_preplanned:
        train_data = _enforce_corpus_ratio(train_data, args.n_examples, chunk_weight)
        eval_data = _enforce_corpus_ratio(eval_data, args.n_eval_examples, chunk_weight)
    else:
        train_data = train_data[:args.n_examples]
        eval_data = eval_data[:args.n_eval_examples]

    if len(train_data) < args.n_examples:
        print(f"WARNING: Only {len(train_data)} train examples survived (requested {args.n_examples}). "
              f"Increase oversampling or relax filters.")

    def _log_target_context_stats(split_name: str, data: list[dict]) -> None:
        target_contexts = [
            int(ex["target_context"])
            for ex in data
            if ex is not None and ex.get("target_context") is not None
        ]
        if not target_contexts:
            return
        unique_contexts = len(set(target_contexts))
        mean_context = sum(target_contexts) / len(target_contexts)
        print(
            f"  {split_name} target_context stats: min={min(target_contexts)}, "
            f"max={max(target_contexts)}, mean={mean_context:.1f}, "
            f"unique={unique_contexts}"
        )

    # Log corpus type stats for QA datasets
    for data, label in [(train_data, "train"), (eval_data, "eval")]:
        _log_target_context_stats(label, data)
        corpus_types = [ex.get("corpus_type", "") for ex in data]
        corpus_types = [ct for ct in corpus_types if ct]
        if corpus_types:
            from collections import Counter
            ct_counts = Counter(corpus_types)
            parts = [f"{ct}: {n}" for ct, n in sorted(ct_counts.items())]
            print(f"  {label} corpus types: {', '.join(parts)}")

    # Verify all pos_docs appear in the user prompt
    for split_name, data in [("train", train_data), ("eval", eval_data)]:
        for ex in data:
            pos_docs = ex.get("pos_docs", [])
            if not pos_docs:
                continue
            user_content = next(msg["content"] for msg in ex["prompt"] if msg["role"] == "user")
            for doc in pos_docs:
                assert doc in user_content, (
                    f"pos_doc not found in prompt for {split_name} example {ex.get('id')}: {doc!r}"
                )
    print("Pos-doc verification passed: all positive documents found in prompts.")

    # Convert to VERL format and save as parquet
    data_source = args.type
    train_verl = [example_to_verl(ex, data_source) for ex in train_data]
    eval_verl = [example_to_verl(ex, data_source) for ex in eval_data]

    train_ds = Dataset.from_list(train_verl).cast(OUTPUT_FEATURES)
    eval_ds = Dataset.from_list(eval_verl).cast(OUTPUT_FEATURES)

    train_ds.to_parquet(os.path.join(args.save_path, "train.parquet"))
    eval_ds.to_parquet(os.path.join(args.save_path, "validation.parquet"))

    # Save one train sample per (instruction_variant, corpus_type) combination
    saved_combos = set()
    for ex in train_data:
        v = ex.get("instruction_variant")
        ct = ex.get("corpus_type", "")
        combo = (v, ct)
        if combo not in saved_combos:
            saved_combos.add(combo)
            suffix = f"_{ct}" if ct else ""
            with open(os.path.join(args.save_path, f"train_variant_{v}{suffix}.txt"), "w") as f:
                f.write(example_to_string(ex, tokenizer))

    # Save first/last validation samples
    with open(os.path.join(args.save_path, "validation_0.txt"), "w") as f:
        f.write(example_to_string(eval_data[0], tokenizer))
    with open(os.path.join(args.save_path, f"validation_{len(eval_data) - 1}.txt"), "w") as f:
        f.write(example_to_string(eval_data[-1], tokenizer))

    print(f"Saved {len(train_data)} train + {len(eval_data)} validation examples to {args.save_path}")
    return train_ds, eval_ds


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a RecaLLM training/evaluation dataset.")

    # Common args
    parser.add_argument("--type", type=str, required=True, choices=list(DATASET_FACTORIES.keys()),
                        help="Dataset type to create.")
    parser.add_argument("--save_path", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--system_prompt", type=str, default=_DEFAULT_SYSTEM_PROMPT,
                        help="Path to the global system prompt file.")
    parser.add_argument("--tokenizer", type=str, default=_DEFAULT_TOKENIZER,
                        help="Tokenizer to use for context length computation.")
    parser.add_argument("--target_context", type=int, default=8000, help="Target prompt length in tokens.")
    parser.add_argument("--context_length_max", type=int, default=None,
                        help="Optional max prompt length in tokens. If set, supported datasets sample "
                             "a per-example target uniformly from [target_context, context_length_max].")
    parser.add_argument("--min_context", type=int, default=None,
                        help="Minimum prompt length in tokens (exclusive). "
                             "Examples at or below this length are excluded. "
                             "Used for bracket-style context filtering (e.g., Quality eval).")
    parser.add_argument("--n_examples", type=int, default=2000, help="Number of train examples.")
    parser.add_argument("--n_eval_examples", type=int, default=200, help="Number of eval examples.")
    parser.add_argument("--num_workers", type=int, default=16, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Retrieval-specific args
    parser.add_argument("--key_min", type=int, default=-10000, help="Minimum key value (retrieval).")
    parser.add_argument("--key_max", type=int, default=10000, help="Maximum key value (retrieval).")
    parser.add_argument("--value_size", type=int, default=10, help="Length of random alphanumeric values (retrieval, math_retrieval).")

    # Math-retrieval-specific args
    parser.add_argument("--n_terms_min", type=int, default=3, help="Min number of terms in math problems (math_retrieval).")
    parser.add_argument("--n_terms_max", type=int, default=5, help="Max number of terms in math problems (math_retrieval).")
    parser.add_argument("--coeff_min", type=int, default=-10, help="Min coefficient in linear equations (math_retrieval).")
    parser.add_argument("--coeff_max", type=int, default=10, help="Max coefficient in linear equations (math_retrieval).")

    # MCQA-math-specific args
    parser.add_argument("--mcqa_math_min_level", type=int, default=2,
                        help="Minimum difficulty level for MCQA Math (1-5).")
    parser.add_argument("--mcqa_math_max_level", type=int, default=4,
                        help="Maximum difficulty level for MCQA Math (1-5).")

    # NIAH-specific args
    parser.add_argument("--niah_k_total_min", type=int, default=1, help="Min total keys (multi_niah).")
    parser.add_argument("--niah_k_total_max", type=int, default=10, help="Max total keys (multi_niah).")
    parser.add_argument("--niah_v_min", type=int, default=1, help="Min values per key (multi_niah).")
    parser.add_argument("--niah_v_max", type=int, default=5, help="Max values per key (multi_niah).")
    parser.add_argument("--niah_max_total_needles", type=int, default=16, help="Max K_total * V (multi_niah).")
    parser.add_argument("--niah_k_query_max", type=int, default=6, help="Max queried keys (multi_niah).")

    # Augmentation args
    parser.add_argument("--qp_end", type=float, default=0.6, help="Question position weight: end. Note: qampari ignores these.")
    parser.add_argument("--qp_beginning", type=float, default=0.2, help="Question position weight: beginning.")
    parser.add_argument("--qp_both", type=float, default=0.2, help="Question position weight: both.")
    parser.add_argument("--kv_formats", type=str, nargs="+", default=None,
                        help="KV formats to use (e.g. lines json csv). Default: all available.")
    parser.add_argument("--label_formats", type=str, nargs="+", default=None,
                        help="Label formats for ICL (e.g. numeric text). Default: all available.")
    parser.add_argument("--instruction_variants", type=int, nargs="+", default=None,
                        help="Instruction variant indices to use (e.g. 0 1 2). Default: all available.")

    # Reranking-specific args
    parser.add_argument("--rerank_data_dir", type=str,
                        default=os.environ.get("RECALLM_MSMARCO_DIR"),
                        help="Directory with prepared reranking data (qrels, queries, bm25_negatives, etc.). "
                             "Also settable via RECALLM_MSMARCO_DIR env var.")
    parser.add_argument("--ir_datasets_home", type=str,
                        default=os.environ.get("IR_DATASETS_HOME"),
                        help="IR_DATASETS_HOME for passage text lookup. Also settable via IR_DATASETS_HOME env var.")
    parser.add_argument("--rerank_pool_size", type=int, default=500,
                        help="Total passages in Layer 1 pool (Layer 2 binary-searches to fit context).")
    parser.add_argument("--rerank_neg_judged_weight", type=float, default=0.4,
                        help="Weight for judged negatives in per-example source selection.")
    parser.add_argument("--rerank_neg_bm25_weight", type=float, default=0.3,
                        help="Weight for BM25 negatives in per-example source selection.")
    parser.add_argument("--rerank_neg_random_weight", type=float, default=0.3,
                        help="Weight for random negatives in per-example source selection.")
    parser.add_argument("--rerank_pos_ratio_min", type=float, default=0.01,
                        help="Minimum positive passage ratio (default 1%%).")
    parser.add_argument("--rerank_pos_ratio_max", type=float, default=0.10,
                        help="Maximum positive passage ratio (default 10%%).")

    # QAMPARI (Citation QA) args
    parser.add_argument("--qampari_data_dir", type=str,
                        default=os.environ.get("RECALLM_QAMPARI_DIR"),
                        help="Directory with extracted QAMPARI JSONL.gz data (full_{split}_data.jsonl.gz). "
                             "Also settable via RECALLM_QAMPARI_DIR env var.")
    parser.add_argument("--qampari_min_distractors", type=int, default=80,
                        help="Minimum distractors per QAMPARI example; BM25S augments if below this.")
    parser.add_argument("--qampari_max_gold_ratio", type=float, default=0.7,
                        help="Max fraction of in-context docs that are gold. Examples above this are skipped.")

    # QA filter args
    parser.add_argument("--qa_filter", type=str, default="high_prec_filter",
                        help="Filter name (subdirectory under filtering/subsets/<dataset>/), or 'none' to disable.")
    parser.add_argument("--qa_filter_dir", type=str, default=_DEFAULT_FILTER_DIR,
                        help="Base directory for filter files.")

    # QA-specific args
    parser.add_argument("--qa_kilt_source_dir", type=str,
                        default=os.environ.get("RECALLM_KILT_SOURCE_DIR"),
                        help="Directory for the cached KILT source dataset. "
                             "Also settable via RECALLM_KILT_SOURCE_DIR env var.")
    parser.add_argument("--qa_kilt_window_dir", type=str,
                        default=os.environ.get("RECALLM_KILT_WINDOW_DIR"),
                        help="Directory for the prepared KILT article/window artifacts. "
                             "Also settable via RECALLM_KILT_WINDOW_DIR env var.")
    parser.add_argument("--qa_bm25_index_dir", type=str,
                        default=os.environ.get("RECALLM_INDEX_DIR"),
                        help="Directory for QA BM25 indexes. "
                             "Also settable via RECALLM_INDEX_DIR env var.")
    parser.add_argument("--qa_bm25_backend", type=str, default="numba", choices=["numba", "numpy", "auto"],
                        help="BM25 backend for QA retrieval planning. `numba` enables the fast threaded path.")
    parser.add_argument("--qa_bm25_n_threads", type=int, default=-1,
                        help="Thread count passed to BM25 retrieval. Use -1 for all available CPU threads.")
    _dense_idx_default = None
    if DEFAULT_DENSE_INDEX_DIR is not None:
        _dense_idx_default = os.path.join(
            DEFAULT_DENSE_INDEX_DIR,
            DEFAULT_DENSE_MODEL_NAME.replace("/", "__"),
        )
    parser.add_argument("--qa_dense_index_dir", type=str,
                        default=os.environ.get("RECALLM_DENSE_INDEX_DIR", _dense_idx_default),
                        help="Directory containing reusable QA dense FAISS indexes. "
                             "Also settable via RECALLM_DENSE_INDEX_DIR env var.")
    parser.add_argument("--qa_dense_nprobe", type=int, default=64,
                        help="FAISS IVF nprobe for QA dense retrieval at runtime.")
    parser.add_argument("--qa_corpus_chunk_weight", type=float, default=0.5,
                        help="Weight for chunk mode (vs paragraph mode) per example.")
    parser.add_argument("--qa_strict_corpus_type", action=argparse.BooleanOptionalAction, default=True,
                        help="Deprecated no-op. QA chunk planning now retries until counts are filled instead of "
                             "falling back during worker materialization.")
    parser.add_argument("--qa_kilt_num_proc", type=int, default=8,
                        help="Worker count for KILT article normalization and window construction.")
    parser.add_argument("--qa_kilt_rebuild_articles", action=argparse.BooleanOptionalAction, default=False,
                        help="Rebuild normalized KILT article artifacts before dataset creation.")
    parser.add_argument("--qa_kilt_rebuild_windows", action=argparse.BooleanOptionalAction, default=False,
                        help="Rebuild rolling KILT window artifacts before dataset creation.")
    parser.add_argument("--qa_kilt_rebuild_bm25", action=argparse.BooleanOptionalAction, default=False,
                        help="Rebuild the KILT window BM25 index before dataset creation.")
    parser.add_argument("--qa_neg_dense_weight", type=float, default=0.45,
                        help="Weight for dense-only negatives per example.")
    parser.add_argument("--qa_neg_bm25_weight", type=float, default=0.30,
                        help="Weight for BM25-only negatives per example.")
    parser.add_argument("--qa_neg_random_weight", type=float, default=0.10,
                        help="Weight for random-only negatives per example.")
    parser.add_argument("--qa_neg_mixed_weight", type=float, default=0.15,
                        help="Weight for mixed (dense + BM25 + random) negatives per example.")

    # Aggregation difficulty weights
    parser.add_argument("--diff_easy", type=float, default=0.45, help="Difficulty weight: easy (aggregation).")
    parser.add_argument("--diff_medium", type=float, default=0.45, help="Difficulty weight: medium (aggregation).")
    parser.add_argument("--diff_hard", type=float, default=0.10, help="Difficulty weight: hard (aggregation).")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    create_dataset(args)
    print("Done!")
