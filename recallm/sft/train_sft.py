"""
RecaLLM SFT training.

Two-stage supervised finetuning to teach a model recall token usage:
  Stage 1 (embedding-only): Freeze all parameters except new token embeddings.
  Stage 2 (full finetune): Unfreeze all parameters, lower learning rate.

Usage (from recallm/sft/):
    deepspeed train_sft.py --config configs/qwen2_embed.yaml
"""
import os
import argparse
from datasets import load_from_disk, load_dataset, concatenate_datasets, Dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
from trl import SFTConfig, SFTTrainer
import torch

from recallm.sft.sft_config import SFT_CONFIG, update_config, finalize_config
from recallm import *  # noqa
from recallm.sft.train_utils import EvalOnFirstStepCallback, sample_by_dataset
from recallm.sft.tokenize_sft_completions import tokenize_recallm_completions_dataset


def load_sft_dataset(dataset_name: str) -> Dataset:
    """Load a dataset from either a local path or HuggingFace Hub."""
    if os.path.exists(dataset_name):
        return load_from_disk(dataset_name)
    else:
        return load_dataset(dataset_name)


def main(args: argparse.Namespace):
    transformers.set_seed(SFT_CONFIG.seed)

    # load the dataset
    print("Loading datasets...")
    train_datasets = []
    validation_datasets = []
    for dataset_name in SFT_CONFIG.dataset.names:
        dataset = load_sft_dataset(dataset_name)

        if 'train' in dataset:
            train_dataset = dataset['train'].shuffle(seed=SFT_CONFIG.seed)
            print(f"Loaded train dataset of size {len(dataset['train'])} from {dataset_name}")

        if 'validation' in dataset:
            validation_dataset = dataset['validation']
            print(f"Loaded validation dataset of size {len(dataset['validation'])} from {dataset_name}")
        elif SFT_CONFIG.dataset.create_validaion_split:
            split_dataset = train_dataset.train_test_split(
                test_size=SFT_CONFIG.dataset.validation_split,
                seed=SFT_CONFIG.dataset.train_val_split_seed
            )
            train_dataset = split_dataset['train']
            validation_dataset = split_dataset['test']
            print(f"Created validation split of size {len(validation_dataset)} from training data in {dataset_name}")
        else:
            validation_dataset = None
            print(f"No validation dataset found in {dataset_name}, and not creating one from training data.")

        train_datasets.append(train_dataset)
        if validation_dataset is not None:
            validation_datasets.append(validation_dataset)

    assert len(train_datasets) > 0, "No train datasets found."
    train_dataset = concatenate_datasets(train_datasets)
    if SFT_CONFIG.dataset.shuffle:
        train_dataset = train_dataset.shuffle(seed=SFT_CONFIG.seed)
    if len(validation_datasets) > 0:
        validation_dataset = concatenate_datasets(validation_datasets)

        if SFT_CONFIG.dataset.sample_validation:
            print("Sampling a fraction of the validation dataset for faster training")
            initial_size = len(validation_dataset)
            if SFT_CONFIG.dataset.validation_sample_stratified:
                validation_dataset = sample_by_dataset(
                    validation_dataset,
                    sample_fraction=SFT_CONFIG.dataset.validation_sample_fraction,
                    seed=SFT_CONFIG.dataset.validation_sample_seed
                )
            else:
                sample_size = int(len(validation_dataset) * SFT_CONFIG.dataset.validation_sample_fraction)
                validation_dataset = validation_dataset.shuffle(
                    seed=SFT_CONFIG.dataset.validation_sample_seed
                ).select(range(sample_size))
            print(f"Validation dataset size reduced from {initial_size} to {len(validation_dataset)}")
    else:
        validation_dataset = None

    if SFT_CONFIG.dataset.reverse_training_data:
        print("Reversing the training dataset...")
        train_dataset = train_dataset.select(range(len(train_dataset)-1, -1, -1))

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset) if validation_dataset is not None else 0}")

    # tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(SFT_CONFIG.model.name)
    train_dataset = tokenize_recallm_completions_dataset(
        train_dataset, tokenizer,
        max_length=SFT_CONFIG.sft.max_length,
        convert_tags_to_tokens=SFT_CONFIG.train.use_recall_tokens,
        combine_system_prompt=SFT_CONFIG.dataset.combine_system_prompt
    )
    if validation_dataset is not None:
        validation_dataset = tokenize_recallm_completions_dataset(
            validation_dataset, tokenizer,
            max_length=SFT_CONFIG.sft.max_length,
            convert_tags_to_tokens=SFT_CONFIG.train.use_recall_tokens,
            combine_system_prompt=SFT_CONFIG.dataset.combine_system_prompt
        )

    # optional liger kernel for throughput
    if SFT_CONFIG.sft.use_liger_kernel:
        from liger_kernel.transformers import apply_liger_kernel_to_llama, apply_liger_kernel_to_qwen2
        print("Manually applying Liger kernel...")
        apply_liger_kernel_to_llama()
        apply_liger_kernel_to_qwen2()

    # logging
    report_to = ['tensorboard']
    if not args.no_wandb:
        report_to.append('wandb')
        if SFT_CONFIG.wandb_project:
            os.environ['WANDB_PROJECT'] = SFT_CONFIG.wandb_project
        if SFT_CONFIG.wandb_entity:
            os.environ['WANDB_ENTITY'] = SFT_CONFIG.wandb_entity
        if SFT_CONFIG.wandb_group:
            os.environ['WANDB_RUN_GROUP'] = SFT_CONFIG.wandb_group
    tensorboard_logdir = os.path.join(SFT_CONFIG.logging_dir, SFT_CONFIG.name)

    # create training arguments
    output_dir = os.path.join(SFT_CONFIG.save_path, 'trainer')
    training_args = SFTConfig(
        output_dir=output_dir,
        run_name=SFT_CONFIG.name,
        logging_dir=tensorboard_logdir,
        report_to=report_to,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=SFT_CONFIG.seed,
        data_seed=SFT_CONFIG.seed,
        dataset_kwargs={"skip_prepare_dataset": True},
        **SFT_CONFIG.sft
    )

    # callbacks
    callbacks = []
    if SFT_CONFIG.train.eval_on_first_step:
        callbacks.append(EvalOnFirstStepCallback())

    # load the model
    model = AutoModelForCausalLM.from_pretrained(
        SFT_CONFIG.model.name,
        trust_remote_code=True
    )

    # Stage 1: freeze all parameters except recall + think token embeddings
    if SFT_CONFIG.train.freeze_base_model:
        for param in model.parameters():
            param.requires_grad = False

        model.get_input_embeddings().weight.requires_grad = True
        model.get_output_embeddings().weight.requires_grad = True

        recall_start_token_id = model.config.recall_start_token_id
        recall_end_token_id = model.config.recall_end_token_id
        assert recall_start_token_id is not None and recall_end_token_id is not None, \
            "Recall token IDs are not set in the model config."

        think_start_token_id = tokenizer.convert_tokens_to_ids('<think>')
        think_end_token_id = tokenizer.convert_tokens_to_ids('</think>')
        assert think_start_token_id is not None and think_end_token_id is not None, \
            "Think token IDs could not be found in the tokenizer."

        vocab_size = model.get_input_embeddings().weight.shape[0]
        keep_tokens_mask = torch.zeros(vocab_size, dtype=torch.bool)
        keep_tokens_mask[recall_start_token_id] = True
        keep_tokens_mask[recall_end_token_id] = True
        keep_tokens_mask[think_start_token_id] = True
        keep_tokens_mask[think_end_token_id] = True

        def mask_non_keep_tokens_hook(grad: torch.Tensor):
            mask = keep_tokens_mask.to(grad.device)
            grad = grad.clone()
            grad[~mask] = 0.0
            return grad

        model.get_input_embeddings().weight.register_hook(mask_non_keep_tokens_hook)
        if model.get_output_embeddings().weight is not model.get_input_embeddings().weight:
            model.get_output_embeddings().weight.register_hook(mask_non_keep_tokens_hook)

    # create the trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        args=training_args,
        callbacks=callbacks
    )

    # configure recall masking
    if SFT_CONFIG.train.recall_masking:
        if isinstance(trainer.model, RecaLLMMixin):
            trainer.model.enable_recall_in_forward_pass()
            print("Recall masking enabled during training.")
    else:
        if isinstance(trainer.model, RecaLLMMixin):
            trainer.model.disable_recall_in_forward_pass()
            print("Recall masking disabled during training.")

    if SFT_CONFIG.train.use_recall_tokens:
        assert isinstance(trainer.model, RecaLLMMixin), \
            "use_recall_tokens requires a RecaLLM model."
        assert trainer.model.config.recall_tag_type == 'token_id', \
            "Only recall_tag_type='token_id' is supported."
        print("Using recall tokens during training.")

    # train
    trainer.train()
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RecaLLM SFT training.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML config file.")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable Weights & Biases logging.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank for distributed training. Used for DeepSpeed compatibility.")

    args, opts = parser.parse_known_args()

    update_config(SFT_CONFIG, args.config, opts, no_run_id=True)
    finalize_config(SFT_CONFIG)

    # create run folder and save the config
    if os.path.exists(SFT_CONFIG.save_path):
        warnings.warn(f"Run save path {SFT_CONFIG.save_path} already exists!")
    os.makedirs(SFT_CONFIG.save_path, exist_ok=True)

    with open(os.path.join(SFT_CONFIG.save_path, 'config.yaml'), 'w') as f:
        f.write(SFT_CONFIG.dump())

    print(SFT_CONFIG.dump())
    main(args)
