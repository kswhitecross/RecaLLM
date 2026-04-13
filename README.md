# RecaLLM: Addressing the Lost-in-Thought Phenomenon with Explicit In-Context Retrieval

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/collections/kswhitecross/recallm"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Collection-ffd21e.svg" alt="HuggingFace Collection"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
</p>

RecaLLM trains reasoning language models to interleave reasoning with explicit, constrained-decoding **recall spans** that copy evidence verbatim from context. This addresses the **lost-in-thought** phenomenon: reasoning LLMs lose in-context retrieval ability after chain-of-thought generation.

## Quick Start

### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Load model (recall masking is built into the forward pass)
model_name = "kswhitecross/RecaLLM-Qwen2.5-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load an evaluation example (math retrieval at 32k context)
ds = load_dataset("kswhitecross/RecaLLM-data", "math_retrieval", split="val_32k")
example = ds[0]

# The chat template auto-injects the system prompt when none is provided
inputs = tokenizer.apply_chat_template(
    example["prompt"], add_generation_prompt=True, return_tensors="pt",
    return_dict=True,
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=10240, temperature=0.6, top_p=0.95)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
print(response)
```

### vLLM (recommended for long contexts)

```python
from vllm import LLM, SamplingParams
from datasets import load_dataset

model_name = "kswhitecross/RecaLLM-Qwen2.5-7B"

llm = LLM(
    model=model_name,
    trust_remote_code=True,
    max_model_len=49152,
    gpu_memory_utilization=0.95,
    # Use the native Qwen2 backend with RecaLLM's logits processor
    hf_overrides={
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "vllm_disable_recall_masking": True,  # vLLM uses external logits processor
    },
    logits_processors=["recallm.recallm_vllm:VLLMTokenRecaLLMLogitsProcessor"],
)

ds = load_dataset("kswhitecross/RecaLLM-data", "math_retrieval", split="val_32k")
example = ds[0]

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt_text = tokenizer.apply_chat_template(
    example["prompt"], add_generation_prompt=True, tokenize=False
)

outputs = llm.generate([prompt_text], SamplingParams(temperature=0.6, top_p=0.95, max_tokens=10240))
print(outputs[0].outputs[0].text)
```

## Installation

```bash
git clone https://github.com/kswhitecross/RecaLLM.git
cd RecaLLM
pip install -e . -e trl/ -e verl/
```

This installs three packages:
- **recallm** -- core model classes, SFT training, GRPO training, datasets, evaluation
- **trl** -- vendored [TRL](https://github.com/huggingface/trl) fork (v0.21, used for SFT only)
- **verl** -- vendored [VeRL](https://github.com/verl-project/verl) fork (used for GRPO only)

**Key dependency versions** (tested):
- `transformers==4.53.2`
- `vllm>=0.8.0` (for evaluation and vLLM inference)
- `torch>=2.5`
- `deepspeed` (for SFT training)

## Training

RecaLLM uses a two-stage training pipeline: an SFT cold start followed by GRPO reinforcement learning.

### Create RecaLLM Base Model

Add recall tokens (`<|start_recall|>`, `<|end_recall|>`) and thinking tokens (`<think>`, `</think>`) to a base model:

```bash
python -m recallm.create_recallm_model \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --output_path models/recallm_qwen2
```

For Llama:
```bash
python -m recallm.create_recallm_model \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --output_path models/recallm_llama
```

### Stage 1: SFT -- Embedding Training

Train only the new token embeddings (5 epochs, frozen base model). SFT data is automatically downloaded from [kswhitecross/RecaLLM-sft](https://huggingface.co/datasets/kswhitecross/RecaLLM-sft) (1,795 GPT-5.2-annotated reasoning traces with recall spans).

```bash
cd recallm/sft
deepspeed train_sft.py --config configs/qwen2_embed.yaml
```

### Stage 2: SFT -- Full Finetune

Unfreeze all parameters and finetune for 1 epoch at a lower learning rate. Update `model.name` in the config to point to your Stage 1 checkpoint.

```bash
deepspeed train_sft.py --config configs/qwen2_full.yaml
```

See `recallm/sft/configs/` for Llama configs.

### Stage 3: GRPO Reinforcement Learning

GRPO (Group Relative Policy Optimization) training using [VeRL](https://github.com/verl-project/verl).

**Hardware:** 4x A100 80GB GPUs (single node).

#### Download Training Data

```bash
python -m recallm.grpo.download_data --output_dir data/grpo
```

This downloads all 17 dataset configs from [kswhitecross/RecaLLM-data](https://huggingface.co/datasets/kswhitecross/RecaLLM-data) and merges them into `data/grpo/train.parquet` (20K examples) and `data/grpo/validation.parquet`.

#### Launch Training

```bash
cd recallm/grpo
bash run_grpo.sh --config-name configs/qwen2   # 150 steps
bash run_grpo.sh --config-name configs/llama    # 60 steps
```

Override any config value via the command line:
```bash
# Allow more recall spans before density penalty kicks in
bash run_grpo.sh --config-name configs/qwen2 \
    custom_reward_function.reward_kwargs.density_threshold=8 \
    custom_reward_function.reward_kwargs.density_half_life=6
```

#### Merge FSDP Checkpoint

After training, merge the FSDP-sharded checkpoint into a standard HuggingFace model:

```bash
python -m recallm.grpo.merge_checkpoint output/recallm_qwen2/global_step_150 --cpu
```

The merged model is saved to `output/recallm_qwen2/global_step_150/bf16/`.

## Evaluation

Evaluate on in-domain validation datasets (17 datasets x 7 context lengths, loaded from HuggingFace):

```bash
python -m recallm.evaluation.evaluate_vllm \
    --model kswhitecross/RecaLLM-Qwen2.5-7B \
    --dataset math_retrieval \
    --context_length 32k \
    --save_path results/recallm_qwen2/math_retrieval/32k
```

To run all datasets and context lengths:

```bash
cd recallm/evaluation
bash submit_all.sh kswhitecross/RecaLLM-Qwen2.5-7B results/recallm_qwen2
```

By default, `submit_all.sh` prints commands (dry run). Edit the `run_command()` function to submit to your cluster or run locally.

Use `recallm/evaluation/analysis.ipynb` to generate summary tables and plots from results.

## Datasets

All training and evaluation data is available on HuggingFace: [kswhitecross/RecaLLM-data](https://huggingface.co/datasets/kswhitecross/RecaLLM-data).

To generate custom datasets from source corpora:

```bash
python -m recallm.datasets.create_dataset \
    --type math_retrieval \
    --target_context 8000 \
    --n_examples 100 \
    --save_path data/custom/math_retrieval_8k
```

See `recallm/datasets/` for all 10 dataset categories and their source implementations.

## Models

| Model | Base | HuggingFace |
|-------|------|-------------|
| RecaLLM-Qwen2.5-7B | Qwen2.5-7B-Instruct | [kswhitecross/RecaLLM-Qwen2.5-7B](https://huggingface.co/kswhitecross/RecaLLM-Qwen2.5-7B) |
| RecaLLM-Llama-3.1-8B | Llama-3.1-8B-Instruct | [kswhitecross/RecaLLM-Llama-3.1-8B](https://huggingface.co/kswhitecross/RecaLLM-Llama-3.1-8B) |

## Citation

```bibtex
@article{whitecross2026recallm,
  title={RecaLLM: Addressing the Lost-in-Thought Phenomenon with Explicit In-Context Retrieval},
  author={Whitecross, Kyle and Rahimi, Negin},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
