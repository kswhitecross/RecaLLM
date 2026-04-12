"""
Convert datasets to VeRL parquet format.

Produces parquet files with columns:
- prompt: chat template format prompt
- data_source: dataset name (passed to reward function)
- ability: task category
- reward_model: dict with ground_truth for reward computation
- extra_info: additional metadata per example
"""
from datasets import load_from_disk, Features, Sequence, Value
import argparse
import os

# Explicit output schema so all parquet files have consistent column types.
# Without this, datasets that have no pos_docs get List(null) instead of List(string),
# which prevents concatenation with datasets that do have pos_docs.
OUTPUT_FEATURES = Features({
    'prompt': [{
        'role': Value('string'),
        'content': Value('string'),
    }],
    'data_source': Value('string'),
    'ability': Value('string'),
    'reward_model': {
        'ground_truth': {
            'answer': Value('string'),
            'pos_docs': Sequence(Value('string')),
            'math_answer': Value('string'),
            'neg_answer': Value('string'),
            'relevance_grades': Value('string'),
        }
    },
    'extra_info': {
        'id': Value('string'),
        'question': Value('string'),
        'question_id': Value('string'),
        'settings': Value('string'),
    },
})

DATASET_TYPE_TO_ABILITY = {
    'retrieval': 'retrieval',
    'multi_niah': 'retrieval',
    'retrieval_math': 'math_to_retrieval',
    'math_retrieval': 'math_to_retrieval',
    'mcqa_math': 'math',
    'dapo_math': 'math',
    'hotpotqa': 'rag',
    '2wikimultihopqa': 'rag',
    'musique': 'rag',
    'msmarco_v2': 'reranking',
    'qampari': 'citation_qa',
}


def main(args: argparse.Namespace):
    dataset_dict = load_from_disk(args.load_path)

    if args.change_system_prompt is not None:
        with open(args.change_system_prompt, "r") as f:
            new_system_prompt = f.read()

    def map_fn(example: dict) -> dict:
        prompt = example["prompt"]
        if args.change_system_prompt is not None:
            prompt[0]['content'] = new_system_prompt
        # Replace recall tags with recall tokens in the prompt
        for msg in prompt:
            msg['content'] = msg['content'].replace("<recall>", "<|start_recall|>").replace("</recall>", "<|end_recall|>")

        data_source = example['type']
        ability = DATASET_TYPE_TO_ABILITY[data_source]

        ground_truth = {
            'answer': example['answer'],
            'pos_docs': example.get('pos_docs', []),
            'math_answer': example.get('math_answer', None),
            'neg_answer': example.get('neg_answer', None),
            'relevance_grades': example.get('relevance_grades', None),
        }

        extra_info = {
            'id': example['id'],
            'question': example['question'],
            'question_id': example.get('question_id', None),
            'settings': example.get('settings', ''),
        }

        return {
            'prompt': prompt,
            'data_source': data_source,
            'ability': ability,
            'reward_model': {"ground_truth": ground_truth},
            'extra_info': extra_info,
        }

    os.makedirs(args.save_path, exist_ok=True)
    for split_name, split_dataset in dataset_dict.items():
        if args.downsample_to is not None and split_name == "train":
            n = len(split_dataset)
            if args.downsample_to >= n:
                print(f"  {split_name}: requested {args.downsample_to} but only {n} rows, keeping all")
            else:
                split_dataset = split_dataset.shuffle(seed=args.seed).select(range(args.downsample_to))
                print(f"  {split_name}: downsampled {n} -> {args.downsample_to} rows (seed={args.seed})")
        split_dataset = split_dataset.map(
            map_fn,
            remove_columns=split_dataset.column_names,
            features=OUTPUT_FEATURES,
            num_proc=args.num_workers,
            desc=f"Converting {split_name} split...",
        )
        split_dataset.to_parquet(os.path.join(args.save_path, f"{split_name}.parquet"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert datasets to VeRL parquet format")
    parser.add_argument("load_path", type=str, help="Path of dataset to load from disk")
    parser.add_argument("save_path", type=str, help="Folder to save parquet files to")
    parser.add_argument("--change_system_prompt", type=str, default=None,
                        help="If specified, change the system prompt to the contents of this file.")
    parser.add_argument("--num_workers", type=int, default=24,
                        help="Number of workers to use when mapping over the dataset.")
    parser.add_argument("--downsample_to", type=int, default=None,
                        help="If specified, randomly downsample the train split to this many rows.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for downsampling reproducibility.")
    args = parser.parse_args()
    main(args)
