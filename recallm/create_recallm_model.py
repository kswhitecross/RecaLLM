"""
Create a RecaLLM model from a base Qwen2 or Llama model.

Adds recall tokens (<|start_recall|>, <|end_recall|>) and thinking tokens
(<think>, </think>), sets up the RecaLLM config, and updates the chat
template to auto-include <think> at the start of assistant turns.

For Qwen2 models, also adds YaRN rope scaling for extended context.

Usage:
    python create_recallm_model.py --base_model Qwen/Qwen2.5-7B-Instruct --output_path ./recallm_qwen2
    python create_recallm_model.py --base_model meta-llama/Llama-3.1-8B-Instruct --output_path ./recallm_llama
"""
import argparse
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AddedToken

from .modelling_recallm import (
    RecaLLMQwen2Config, RecaLLMQwen2ForCausalLM,
    RecaLLMLlamaConfig, RecaLLMLlamaForCausalLM,
)


# Chat template modifications: insert <think>\n after the assistant header
# in the add_generation_prompt block.
QWEN2_GENERATION_PROMPT_OLD = "{{- '<|im_start|>assistant\\n' }}"
QWEN2_GENERATION_PROMPT_NEW = "{{- '<|im_start|>assistant\\n<think>\\n' }}"

LLAMA_GENERATION_PROMPT_OLD = "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
LLAMA_GENERATION_PROMPT_NEW = "{{- '<|start_header_id|>assistant<|end_header_id|>\\n<think>\\n' }}"


def detect_architecture(config) -> str:
    """Detect whether the base model is Qwen2 or Llama from its config."""
    model_type = getattr(config, "model_type", "")
    if "qwen2" in model_type:
        return "qwen2"
    elif "llama" in model_type:
        return "llama"
    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            "Only 'qwen2' and 'llama' architectures are supported."
        )


def update_chat_template(tokenizer, arch: str) -> str:
    """
    Update the chat template to include <think> after the assistant header.

    Returns the updated template string and saves it as a .jinja file.
    """
    template = tokenizer.chat_template
    if template is None:
        raise ValueError("Tokenizer has no chat_template to update.")

    if arch == "qwen2":
        assert QWEN2_GENERATION_PROMPT_OLD in template, \
            f"Could not find expected Qwen2 generation prompt block in chat template"
        updated = template.replace(
            QWEN2_GENERATION_PROMPT_OLD,
            QWEN2_GENERATION_PROMPT_NEW,
        )
    elif arch == "llama":
        assert LLAMA_GENERATION_PROMPT_OLD in template, \
            f"Could not find expected Llama generation prompt block in chat template"
        updated = template.replace(
            LLAMA_GENERATION_PROMPT_OLD,
            LLAMA_GENERATION_PROMPT_NEW,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return updated


def create_recallm_model(base_model: str, output_path: str):
    """Create a RecaLLM model from a base model."""
    hf_token = os.environ.get("HF_ACCESS_TOKEN")

    # Load base model and tokenizer
    print(f"Loading base model: {base_model}")
    original_model = AutoModelForCausalLM.from_pretrained(
        base_model, token=hf_token, device_map="auto", dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)

    # Detect architecture
    arch = detect_architecture(original_model.config)
    print(f"Detected architecture: {arch}")

    # Add special tokens
    recall_start_token_str = "<|start_recall|>"
    recall_end_token_str = "<|end_recall|>"
    start_think_token_str = "<think>"
    end_think_token_str = "</think>"

    tokenizer.add_tokens([
        AddedToken(recall_start_token_str, single_word=False, lstrip=False, rstrip=False, normalized=False, special=False),
        AddedToken(recall_end_token_str, single_word=False, lstrip=False, rstrip=False, normalized=False, special=False),
        AddedToken(start_think_token_str, single_word=False, lstrip=False, rstrip=False, normalized=False, special=False),
        AddedToken(end_think_token_str, single_word=False, lstrip=False, rstrip=False, normalized=False, special=False),
    ], special_tokens=False)

    recall_start_token_id = tokenizer.convert_tokens_to_ids(recall_start_token_str)
    recall_end_token_id = tokenizer.convert_tokens_to_ids(recall_end_token_str)

    # Build RecaLLM config
    config_dict = original_model.config.to_dict()
    recallm_kwargs = dict(
        force_recall=True,
        recall_tag_type="token_id",
        recall_start_str="<recall>",
        recall_end_str="</recall>",
        recall_start_token_id=recall_start_token_id,
        recall_end_token_id=recall_end_token_id,
        use_fast_recall=True,
        max_recall_length=128,
    )

    if arch == "qwen2":
        # Add YaRN rope scaling for extended context (32K -> 131K)
        config_dict["rope_scaling"] = {
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": config_dict.get("max_position_embeddings", 32768),
        }
        recallm_config = RecaLLMQwen2Config.from_dict(config_dict, **recallm_kwargs)
        print("Creating RecaLLM Qwen2 model with YaRN (factor=4.0)")
        recallm_model = RecaLLMQwen2ForCausalLM(recallm_config).to(original_model.device)
    elif arch == "llama":
        recallm_config = RecaLLMLlamaConfig.from_dict(config_dict, **recallm_kwargs)
        print("Creating RecaLLM Llama model")
        recallm_model = RecaLLMLlamaForCausalLM(recallm_config).to(original_model.device)

    # Load base model weights
    print("Loading base model weights into RecaLLM model")
    recallm_model.load_state_dict(original_model.state_dict())

    # Resize embeddings for new tokens
    print("Resizing embeddings for new special tokens")
    with torch.no_grad():
        recallm_model.resize_token_embeddings(len(tokenizer))

    # Update chat template to include <think>
    print("Updating chat template to include <think> token")
    updated_template = update_chat_template(tokenizer, arch)
    tokenizer.chat_template = updated_template

    # Save model and tokenizer
    print(f"Saving RecaLLM model to {output_path}")
    recallm_model.save_pretrained(output_path, max_shard_size="10GB")
    tokenizer.save_pretrained(output_path)

    # Also save chat template as a .jinja file (matching trained model pattern)
    jinja_path = os.path.join(output_path, "chat_template.jinja")
    with open(jinja_path, "w") as f:
        f.write(updated_template)
    print(f"Chat template saved to {jinja_path}")

    print("Done!")
    print(f"  Architecture: {arch}")
    print(f"  Vocab size: {recallm_config.vocab_size}")
    print(f"  Recall start token ID: {recall_start_token_id}")
    print(f"  Recall end token ID: {recall_end_token_id}")
    if arch == "qwen2":
        print(f"  YaRN factor: 4.0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a RecaLLM model from a base model")
    parser.add_argument("--base_model", required=True,
                        help="HuggingFace model name or path (e.g., Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--output_path", required=True,
                        help="Path to save the RecaLLM model")
    args = parser.parse_args()

    create_recallm_model(args.base_model, args.output_path)
