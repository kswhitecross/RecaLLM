"""
Download and merge GRPO training data from HuggingFace.

VeRL requires local .parquet files, but the HuggingFace dataset
(kswhitecross/RecaLLM-data) stores data in 17 separate configs
(one per dataset). This script downloads all train splits and merges
them into a single train.parquet for GRPO training.

Usage:
    python -m recallm.grpo.download_data --output_dir data/grpo
"""
import argparse
import os

from datasets import load_dataset, concatenate_datasets


HF_REPO = "kswhitecross/RecaLLM-data"

DATASET_CONFIGS = [
    "hotpotqa",
    "musique",
    "2wikimultihopqa",
    "nq",
    "triviaqa",
    "retrieval",
    "multi_niah",
    "math_retrieval",
    "dapo_math",
    "mcqa_math",
    "banking77",
    "massive",
    "quality",
    "majority_vote",
    "top_n_vote",
    "msmarco_v2",
    "qampari",
]


def main():
    parser = argparse.ArgumentParser(description="Download and merge GRPO training data from HuggingFace.")
    parser.add_argument("--output_dir", type=str, default="data/grpo",
                        help="Output directory for merged parquet files. Default: data/grpo")
    parser.add_argument("--repo", type=str, default=HF_REPO,
                        help=f"HuggingFace dataset repository. Default: {HF_REPO}")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Download and merge all train splits
    train_datasets = []
    for config in DATASET_CONFIGS:
        print(f"Downloading {config} train split...")
        ds = load_dataset(args.repo, config, split="train")
        train_datasets.append(ds)
        print(f"  {config}: {len(ds)} examples")

    print(f"\nMerging {len(train_datasets)} datasets...")
    merged_train = concatenate_datasets(train_datasets)
    print(f"Total training examples: {len(merged_train)}")

    train_path = os.path.join(args.output_dir, "train.parquet")
    merged_train.to_parquet(train_path)
    print(f"Saved to {train_path}")

    # Download and merge a small validation set (val_4k from each config)
    val_datasets = []
    for config in DATASET_CONFIGS:
        print(f"Downloading {config} val_4k split...")
        try:
            ds = load_dataset(args.repo, config, split="val_4k")
            val_datasets.append(ds)
            print(f"  {config}: {len(ds)} examples")
        except Exception as e:
            print(f"  {config}: skipped ({e})")

    if val_datasets:
        merged_val = concatenate_datasets(val_datasets)
        print(f"Total validation examples: {len(merged_val)}")
        val_path = os.path.join(args.output_dir, "validation.parquet")
        merged_val.to_parquet(val_path)
        print(f"Saved to {val_path}")

    print("\nDone! You can now run GRPO training with the default config.")


if __name__ == "__main__":
    main()
