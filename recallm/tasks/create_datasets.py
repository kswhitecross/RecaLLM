"""
Orchestrator: create all datasets for the final training run or evaluation.

Training mode (default):
    Creates one subfolder per dataset type under save_dir/, each containing
    train.parquet, validation.parquet, sample txt files, and args.json.

    python -m recallm.tasks.create_datasets \
        --save_dir /path/to/all_datasets \
        --target_context 8000

Evaluation mode (--target_contexts):
    Creates one subfolder per dataset type per context length:
        save_dir/<dataset>/<context_length>/validation.parquet

    python -m recallm.tasks.create_datasets \
        --save_dir /path/to/eval_datasets \
        --target_contexts 4k 8k 16k 32k 64k 96k 125k \
        --n_examples 10 --n_eval_examples 1000
"""

import argparse
import hashlib
import os

from .create_dataset import _DEFAULT_SYSTEM_PROMPT, _DEFAULT_TOKENIZER, create_dataset, get_parser


def context_str_to_int(s: str) -> int:
    """Convert context size string (e.g., '4k', '16k', '125k') to integer token count."""
    s = s.lower().strip()
    if s.endswith('k'):
        return int(float(s[:-1]) * 1000)
    elif s.endswith('m'):
        return int(float(s[:-1]) * 1_000_000)
    else:
        return int(s)


def stable_dataset_seed(base_seed: int, dataset_name: str) -> int:
    """Return a deterministic per-dataset seed independent of loop order.

    This keeps materialization stable across reruns even when `--only` changes
    the order or subset of datasets being built.
    """
    payload = f"{int(base_seed)}:{dataset_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


# Per-dataset default overrides (n_examples, n_eval_examples, and any dataset-specific args).
# These can be overridden from the CLI.
DATASET_CONFIGS = {
    "retrieval": {
        "n_examples": 2000,
        "n_eval_examples": 200,
    },
    "math_retrieval": {
        "n_examples": 4000,
        "n_eval_examples": 200,
    },
    "banking77": {
        "n_examples": 3000,
        "n_eval_examples": 200,
    },
    "massive": {
        "n_examples": 3000,
        "n_eval_examples": 200,
    },
    "dapo_math": {
        "n_examples": 1000,
        "n_eval_examples": 200,
    },
    "mcqa_math": {
        "n_examples": 1000,
        "n_eval_examples": 200,
    },
    "quality": {
        "n_examples": 2000,
        "n_eval_examples": 200,
    },
    "multi_niah": {
        "n_examples": 2000,
        "n_eval_examples": 200,
    },
    "majority_vote": {
        "n_examples": 2000,
        "n_eval_examples": 200,
    },
    "top_n_vote": {
        "n_examples": 2000,
        "n_eval_examples": 200,
    },
    "msmarco_v2": {
        "n_examples": 4000,
        "n_eval_examples": 200,
    },
    "qampari": {
        "n_examples": 4000,
        "n_eval_examples": 200,
    },
    "hotpotqa": {
        "n_examples": 2000,
        "n_eval_examples": 200,
    },
    "musique": {
        "n_examples": 2000,
        "n_eval_examples": 200,
    },
    "2wikimultihopqa": {
        "n_examples": 2000,
        "n_eval_examples": 200,
    },
    "nq": {
        "n_examples": 2000,
        "n_eval_examples": 200,
    },
    "triviaqa": {
        "n_examples": 2000,
        "n_eval_examples": 200,
    },
}

# Datasets where target_context is meaningless (no context to scale)
_FIXED_SIZE_TYPES = {"dapo_math", "mcqa_math"}

# Datasets with limited max context (article length ceiling)
_MAX_CONTEXT = {
    "quality": 8000,  # articles are 2.6K-8.8K tokens
}

# Datasets where bracket-style min_context filtering applies in multi-context mode.
# These have fixed natural document lengths (not synthetically constructed context).
_BRACKET_FILTER_TYPES = {"quality"}


def _get_context_lengths(ds_type: str, target_contexts: list[str]) -> list[str]:
    """Filter context lengths to those meaningful for a given dataset type."""
    if ds_type in _FIXED_SIZE_TYPES:
        # Context-free: only create at the smallest requested context
        return [target_contexts[0]]
    max_ctx = _MAX_CONTEXT.get(ds_type)
    if max_ctx is not None:
        return [c for c in target_contexts if context_str_to_int(c) <= max_ctx]
    return target_contexts


def main():
    parser = argparse.ArgumentParser(description="Create all final training/evaluation datasets.")
    parser.add_argument("--save_dir", type=str, required=True, help="Root directory for all datasets.")
    parser.add_argument("--system_prompt", type=str, default=_DEFAULT_SYSTEM_PROMPT,
                        help="Path to the global system prompt file.")
    parser.add_argument("--tokenizer", type=str, default=_DEFAULT_TOKENIZER,
                        help="Tokenizer for context length computation.")
    parser.add_argument("--target_context", type=int, default=8000, help="Target prompt length in tokens.")
    parser.add_argument("--context_length_max", type=int, default=None,
                        help="Optional max prompt length in tokens. If set, supported datasets sample "
                             "a per-example target uniformly from [target_context, context_length_max].")
    parser.add_argument("--target_contexts", type=str, nargs="+", default=None,
                        help="Create datasets at multiple context lengths (eval mode). "
                             "E.g. --target_contexts 4k 8k 16k 32k 64k 96k 125k. "
                             "Output: save_dir/<dataset>/<ctx>/. Overrides --target_context.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--num_workers", type=int, default=16, help="DataLoader workers.")
    parser.add_argument("--only", type=str, nargs="+", default=None,
                        help="Only create these dataset types (e.g. --only retrieval multi_niah).")
    parser.add_argument("--n_examples", type=int, default=None,
                        help="Override n_examples for all datasets.")
    parser.add_argument("--n_eval_examples", type=int, default=None,
                        help="Override n_eval_examples for all datasets.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip datasets whose save_path already contains validation.parquet.")
    args = parser.parse_args()

    datasets_to_create = args.only if args.only else list(DATASET_CONFIGS.keys())
    multi_context = args.target_contexts is not None

    # Use the create_dataset parser to build proper Namespace objects
    base_parser = get_parser()

    for ds_type in datasets_to_create:
        if ds_type not in DATASET_CONFIGS:
            print(f"WARNING: No config for dataset type '{ds_type}', skipping.")
            continue
        dataset_seed = stable_dataset_seed(args.seed, ds_type)

        # Determine context lengths to iterate over
        if multi_context:
            ctx_list = _get_context_lengths(ds_type, args.target_contexts)
        else:
            ctx_list = [None]  # single-context mode: use args.target_context

        # Sort ascending for bracket computation (min_context = previous bracket)
        ctx_list_sorted = sorted(ctx_list, key=lambda c: context_str_to_int(c)) if ctx_list[0] is not None else ctx_list
        for ctx_idx, ctx_str in enumerate(ctx_list_sorted):
            config = dict(DATASET_CONFIGS[ds_type])  # copy so we can pop
            if args.n_examples is not None:
                config["n_examples"] = args.n_examples
            if args.n_eval_examples is not None:
                config["n_eval_examples"] = args.n_eval_examples

            if multi_context:
                save_path = os.path.join(args.save_dir, ds_type, ctx_str)
                target_ctx = context_str_to_int(ctx_str)
            else:
                save_path = os.path.join(args.save_dir, ds_type)
                target_ctx = args.target_context

            if args.skip_existing and os.path.exists(os.path.join(save_path, "validation.parquet")):
                print(f"SKIP (exists): {save_path}")
                continue

            # Allow config to override the factory type (e.g. "hotpotqa_chunk" → type "hotpotqa")
            actual_type = config.pop("type", ds_type)

            # Build CLI-like args list
            cli_args = [
                "--type", actual_type,
                "--save_path", save_path,
                "--system_prompt", args.system_prompt,
                "--tokenizer", args.tokenizer,
                "--target_context", str(target_ctx),
                "--seed", str(dataset_seed),
                "--num_workers", str(args.num_workers),
            ]
            # Only pass context_length_max in single-context (training) mode
            if not multi_context and args.context_length_max is not None:
                cli_args.extend(["--context_length_max", str(args.context_length_max)])
            # For bracket-filtered types, set min_context to the previous bracket
            if multi_context and ds_type in _BRACKET_FILTER_TYPES and ctx_idx > 0:
                prev_ctx = context_str_to_int(ctx_list_sorted[ctx_idx - 1])
                cli_args.extend(["--min_context", str(prev_ctx)])
            for key, val in config.items():
                if isinstance(val, bool):
                    if val:
                        cli_args.append(f"--{key}")
                else:
                    cli_args.extend([f"--{key}", str(val)])

            ds_args = base_parser.parse_args(cli_args)

            ctx_label = ctx_str if multi_context else f"{target_ctx}"
            print(f"\n{'='*60}")
            print(f"Creating dataset: {ds_type} @ {ctx_label}")
            print(f"  Save path: {save_path}")
            print(f"  Examples: {ds_args.n_examples} train, {ds_args.n_eval_examples} eval")
            print(f"{'='*60}\n")

            create_dataset(ds_args)

        # For fixed-size or max-context-limited datasets in multi-context mode,
        # symlink the created context dirs to all other requested context lengths
        # so evaluation picks them up.
        if multi_context:
            created = set(ctx_list)
            all_requested = set(args.target_contexts)
            missing = all_requested - created
            if missing and ctx_list:
                # Symlink missing context lengths to the largest created one
                source_ctx = max(ctx_list, key=lambda c: context_str_to_int(c))
                source_path = os.path.abspath(os.path.join(args.save_dir, ds_type, source_ctx))
                for m in sorted(missing, key=lambda c: context_str_to_int(c)):
                    link_path = os.path.join(args.save_dir, ds_type, m)
                    if not os.path.exists(link_path):
                        os.symlink(source_path, link_path)
                        print(f"  SYMLINK: {ds_type}/{m} -> {ds_type}/{source_ctx}")

    print(f"\nAll datasets created in {args.save_dir}")


if __name__ == "__main__":
    main()
