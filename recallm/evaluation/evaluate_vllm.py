"""
Evaluate RecaLLM models on in-domain validation datasets using vLLM.

Loads data from HuggingFace Hub (kswhitecross/RecaLLM-data) by default,
or from a local .parquet file. Scores completions using final_reward()
from recallm.grpo.rewards.

Usage:
    # Evaluate on HotpotQA at 32k context
    python -m recallm.evaluation.evaluate_vllm \
        --model kswhitecross/RecaLLM-Qwen2.5-7B \
        --dataset hotpotqa --context_length 32k \
        --save_path ./results/my_model/hotpotqa/32k

    # Evaluate from a local parquet file
    python -m recallm.evaluation.evaluate_vllm \
        --model /path/to/model \
        --dataset /path/to/validation.parquet \
        --save_path ./results/my_model/hotpotqa/32k

Requires vllm>=0.8.0 (V1 logits processor API).
"""

import argparse
import json
import math
import os
import re
from typing import Any

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from vllm import SamplingParams, LLM, TokensPrompt
from tqdm.auto import tqdm

from recallm.recallm_vllm import VLLMTokenRecaLLMLogitsProcessor
from recallm.grpo.rewards import final_reward, compute_qampari_answer_coverage
from recallm.grpo.reward_utils import ANSWER_RE


def vllm_tqdm(*args, **kwargs):
    # Force vLLM's bars to render under mine
    kwargs.setdefault("position", 1)
    kwargs.setdefault("leave", False)
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(*args, **kwargs)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def flatten_parquet_dataset(dataset: Dataset) -> Dataset:
    """Flatten VERL-schema parquet into flat columns for uniform downstream use.

    The VERL format nests ground_truth inside reward_model and stores metadata
    in extra_info. This function extracts those into top-level columns.
    """
    def _flatten(batch):
        gts = [rm['ground_truth'] for rm in batch['reward_model']]
        eis = batch['extra_info']
        return {
            'answer': [gt['answer'] for gt in gts],
            'pos_docs': [gt['pos_docs'] for gt in gts],
            'math_answer': [gt['math_answer'] for gt in gts],
            'neg_answer': [gt['neg_answer'] for gt in gts],
            'relevance_grades': [gt['relevance_grades'] for gt in gts],
            'id': [ei['id'] for ei in eis],
            'question': [ei['question'] for ei in eis],
            'question_id': [ei['question_id'] for ei in eis],
            'settings': [ei['settings'] for ei in eis],
        }
    dataset = dataset.map(_flatten, batched=True, batch_size=1000, keep_in_memory=True,
                          desc="Flattening parquet fields...")
    dataset = dataset.remove_columns(['reward_model', 'extra_info'])
    return dataset


def compute_citation_f1_plain(model_answer: str, settings_str: str) -> dict:
    """Citation F1 without recall-backed weighting. Fair baseline comparison."""
    _zero = {"citation_f1_plain": 0.0, "citation_recall_plain": 0.0, "citation_precision_plain": 0.0}
    cited = set(int(x) for x in re.findall(r'\[(\d+)\]', model_answer or ""))
    if not cited:
        return _zero
    if isinstance(settings_str, str) and settings_str:
        try:
            settings = json.loads(settings_str)
        except (ValueError, TypeError):
            settings = {}
    else:
        settings = {}
    gold = set(settings.get('gold_doc_ids', []))
    if not gold:
        return _zero
    correct = cited & gold
    n_correct = len(correct)
    recall_denom = min(5, len(gold))
    n_correct_capped = min(n_correct, recall_denom)
    recall = n_correct_capped / recall_denom
    precision = n_correct / len(cited)
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    return {"citation_f1_plain": f1, "citation_recall_plain": recall, "citation_precision_plain": precision}


def compute_qampari_answer_coverage_plain(
    model_answer: str,
    settings_str: str,
    ground_truth_answer: str,
) -> dict:
    return compute_qampari_answer_coverage(
        model_answer=model_answer,
        extra_info={"settings": settings_str},
        ground_truth={"answer": ground_truth_answer},
    )


def sanitize_for_json(d: dict) -> dict:
    """Replace NaN/inf floats with None for valid JSON serialization."""
    return {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
            for k, v in d.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace):
    # Check if results already exist
    if args.skip_existing and os.path.exists(os.path.join(args.save_path, 'completed.txt')):
        print(f"Evaluation already completed at {args.save_path}, skipping...")
        return

    # Load dataset: HuggingFace hub or local parquet
    if args.dataset.endswith('.parquet'):
        dataset = Dataset.from_parquet(args.dataset)
    else:
        split_name = f"val_{args.context_length}"
        dataset = load_dataset(args.data_repo, args.dataset, split=split_name)

    # Flatten VERL nested schema if present
    if 'reward_model' in dataset.column_names:
        dataset = flatten_parquet_dataset(dataset)

    # Truncate dataset for quick testing
    if args.truncate_dataset is not None:
        dataset = dataset.select(range(min(args.truncate_dataset, len(dataset))))

    # Optionally load a different system prompt
    new_system_prompt = None
    if args.system_prompt_path is not None:
        with open(args.system_prompt_path, 'r') as f:
            new_system_prompt = f.read()

    recall_start = "<|start_recall|>"
    recall_end = "<|end_recall|>"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        prompts = batch['prompt']
        if new_system_prompt is not None:
            updated_prompts = []
            for prompt in prompts:
                if prompt[0]['role'] == 'system':
                    prompt[0]['content'] = new_system_prompt
                updated_prompts.append(prompt)
            prompts = updated_prompts
        if args.merge_system_prompt:
            merged_prompts = []
            for prompt in prompts:
                merged_prompt = [
                    {"role": "user", "content": prompt[0]['content'] + "\n\n" + prompt[1]['content']}
                ]
                merged_prompts.append(merged_prompt)
            prompts = merged_prompts
        chat_template_kwargs = {}
        if args.disable_thinking:
            chat_template_kwargs['enable_thinking'] = False
        input_texts = tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=False, **chat_template_kwargs)
        if args.add_think_prompt:
            input_texts = [text + "<think>\n" for text in input_texts]
        inputs = tokenizer(input_texts, return_length=True, add_special_tokens=False)
        return inputs

    # Tokenize the dataset
    dataset = dataset.map(tokenize_batch, batched=True, batch_size=32, num_proc=args.num_workers, desc="Tokenizing dataset...", keep_in_memory=True)

    max_dataset_len = max(dataset['length'])
    max_seq_len = max_dataset_len + args.max_new_tokens + 1

    # Set up logits processors
    logits_processors = []
    if not args.no_recall_masking:
        logits_processors.append(VLLMTokenRecaLLMLogitsProcessor)

    # Build hf_overrides — for RecaLLM models, override model_type and
    # architectures so vLLM uses the native backend (e.g. Qwen2) instead of
    # falling back to the generic TransformersForCausalLM.
    hf_overrides: dict[str, Any] = {"vllm_disable_recall_masking": True}
    model_config_path = os.path.join(args.model, "config.json")
    if os.path.exists(model_config_path):
        with open(model_config_path) as f:
            model_config = json.load(f)
        model_type = model_config.get("model_type", "")
        if model_type.startswith("recallm_"):
            base_type = model_type[len("recallm_"):]  # e.g. "qwen2"
            base_arch = model_config["architectures"][0].replace("RecaLLM", "")  # e.g. "Qwen2ForCausalLM"
            hf_overrides["model_type"] = base_type
            hf_overrides["architectures"] = [base_arch]

    # Create LLM
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_seq_len,
        logits_processors=logits_processors,
        tensor_parallel_size=args.tensor_parallel_size,
        hf_overrides=hf_overrides,
        trust_remote_code=True,
        safetensors_load_strategy='eager'
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens
    )

    # Create output directory and results file
    os.makedirs(args.save_path, exist_ok=True)

    # Save args
    with open(os.path.join(args.save_path, 'args.json'), 'w', encoding='utf-8') as args_file:
        json.dump(vars(args), args_file, indent=4)

    with open(os.path.join(args.save_path, 'results.jsonl'), 'w', encoding='utf-8') as results_file:

        consecutive_truncated = 0
        total_written = 0
        aborted = False

        # Evaluate the dataset
        for start_idx in tqdm(range(0, len(dataset), args.wave_size), desc="Evaluating dataset..."):
            example_batch = dataset[start_idx : start_idx + args.wave_size]

            # Get input ids
            input_ids = example_batch['input_ids']

            # Pass tokens to vLLM and generate
            token_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in input_ids]
            wave_outputs = llm.generate(token_prompts, sampling_params=sampling_params, use_tqdm=vllm_tqdm)

            # Extract completions
            completions = [request_output.outputs[0].text for request_output in wave_outputs]
            n_output_ids = [len(request_output.outputs[0].token_ids) for request_output in wave_outputs]

            # Score with final_reward()
            batch_size = len(completions)
            results_batch = []
            for i in range(batch_size):
                ground_truth = {
                    'answer': example_batch['answer'][i],
                    'pos_docs': example_batch['pos_docs'][i],
                    'math_answer': example_batch['math_answer'][i],
                    'neg_answer': example_batch['neg_answer'][i],
                    'relevance_grades': example_batch['relevance_grades'][i],
                }
                extra_info = {
                    'id': example_batch['id'][i],
                    'question': example_batch['question'][i],
                    'question_id': example_batch['question_id'][i],
                    'settings': example_batch['settings'][i],
                    'response_length': n_output_ids[i],
                }
                data_source = example_batch['data_source'][i]

                reward_result = final_reward(
                    data_source=data_source,
                    solution_str=completions[i],
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    recall_start_str=recall_start,
                    recall_end_str=recall_end,
                )

                # Math retrieval: strict scoring (no 0.25 partial for math_answer)
                if data_source == 'math_retrieval':
                    model_answer = reward_result.get('model_answer')
                    if model_answer is None:
                        completion_after_think = completions[i].split("</think>")[-1]
                        match = ANSWER_RE.search(completion_after_think)
                        model_answer = match.group(1) if match else ""
                    reward_result['answer_score'] = 1.0 if ground_truth['answer'] in model_answer else 0.0

                # Plain citation F1 (always computed for fair baseline comparison)
                completion_after_think = completions[i].split("</think>")[-1]
                match = ANSWER_RE.search(completion_after_think)
                model_answer_text = match.group(1) if match else ""
                plain_citation = compute_citation_f1_plain(model_answer_text, extra_info.get('settings', ''))
                qampari_coverage = compute_qampari_answer_coverage_plain(
                    model_answer_text,
                    extra_info.get('settings', ''),
                    ground_truth['answer'],
                ) if data_source == 'qampari' else {}

                result = {
                    "completion": completions[i],
                    "n_output_ids": n_output_ids[i],
                    "truncated": n_output_ids[i] >= args.max_new_tokens,
                    "data_source": data_source,
                    "ability": example_batch['ability'][i],
                    "question": extra_info['question'],
                    "answer": ground_truth['answer'],
                    "id": extra_info['id'],
                    "eval_version": 2.0,
                    **reward_result,
                    **plain_citation,
                    **qampari_coverage,
                }
                results_batch.append(sanitize_for_json(result))

            results_file.writelines([json.dumps(result) + "\n" for result in results_batch])
            results_file.flush()
            total_written += len(results_batch)

            # Early termination: abort if too many consecutive examples hit max tokens
            if args.max_consecutive_truncated is not None:
                for r in results_batch:
                    if r.get("truncated", False):
                        consecutive_truncated += 1
                    else:
                        consecutive_truncated = 0
                    if consecutive_truncated >= args.max_consecutive_truncated:
                        aborted = True
                        break
                if aborted:
                    reason = (f"Aborted: {consecutive_truncated} consecutive examples hit max_new_tokens "
                              f"({args.max_new_tokens}). {total_written}/{len(dataset)} examples written.")
                    print(f"\n*** {reason}")
                    with open(os.path.join(args.save_path, 'aborted.txt'), 'w', encoding='utf-8') as af:
                        af.write(reason + "\n")
                    break

        print(f"Saved evaluation results to {args.save_path}/results.jsonl ({total_written}/{len(dataset)} examples)")

    # Save a completed.txt file to indicate completion (even if aborted — partial results are valid)
    with open(os.path.join(args.save_path, 'completed.txt'), 'w', encoding='utf-8') as f:
        pass
    print("Done!" if not aborted else "Done (aborted early).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RecaLLM models on in-domain validation datasets using vLLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset arguments
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path (HuggingFace hub ID or local path)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="HF dataset config name (e.g., 'hotpotqa', 'multi_niah') or path to a local .parquet file")
    parser.add_argument("--context_length", type=str, default="32k",
                        help="Context length split to evaluate (e.g., '4k', '32k', '128k'). "
                             "Only used when --dataset is an HF config name. Default: 32k")
    parser.add_argument("--data_repo", type=str, default="kswhitecross/RecaLLM-data",
                        help="HuggingFace dataset repository. Default: kswhitecross/RecaLLM-data")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save evaluation results")

    # Generation arguments
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Generation temperature. 0.0 means greedy decoding. Default: 0.0")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p sampling value. Default: 0.95")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum number of new tokens to generate. Default: 2048")

    # vLLM arguments
    parser.add_argument("--wave_size", type=int, default=200,
                        help="Number of samples to process in each wave. Default: 200")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95,
                        help="GPU memory utilization for vLLM. Default: 0.95")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size for vLLM. Default: 1")

    # Evaluation control
    parser.add_argument("--truncate_dataset", type=int, default=None,
                        help="Truncate dataset to this many samples for quick testing")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel workers for tokenization. Default: 8")
    parser.add_argument("--no_recall_masking", action="store_true",
                        help="Disable recall masking (constrained decoding within recall spans)")
    parser.add_argument("--add_think_prompt", action="store_true",
                        help="Append '<think>\\n' to prompt to trigger reasoning")
    parser.add_argument("--disable_thinking", action="store_true",
                        help="Pass enable_thinking=False to apply_chat_template (for Qwen3 no-think mode)")
    parser.add_argument("--merge_system_prompt", action="store_true",
                        help="Merge the system prompt into the first user message")
    parser.add_argument("--system_prompt_path", type=str, default=None,
                        help="Path to a file containing a system prompt to override the default")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip evaluation if results already exist at save_path")
    parser.add_argument("--max_consecutive_truncated", type=int, default=None,
                        help="Abort evaluation if this many consecutive examples hit max_new_tokens. "
                             "Useful for detecting runaway generation. Partial results are still saved.")

    args = parser.parse_args()
    main(args)
