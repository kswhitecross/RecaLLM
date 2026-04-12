"""
SFT configuration using yacs CfgNode.

All configuration options are defined here with sensible defaults.
Override via YAML config files and/or command-line arguments.
"""
from yacs.config import CfgNode
from argparse import ArgumentParser
import os


def get_cfg_defaults() -> CfgNode:
    """
    Returns the default configuration settings.
    """
    cfg = CfgNode()

    # ====== General settings ======
    cfg.name = 'sft_run'
    cfg.save_dir = '../output'
    cfg.run_id = None
    cfg.name_id = None
    cfg.save_path = None
    cfg.seed = 42
    cfg.logging_dir = 'logs'
    cfg.wandb_project = 'RecaLLM-SFT'
    cfg.wandb_entity = ''
    cfg.wandb_group = ''

    # ====== Dataset settings ======
    cfg.dataset = CfgNode()
    cfg.dataset.names = []
    cfg.dataset.create_validaion_split = False
    cfg.dataset.validation_split = 0.2
    cfg.dataset.train_val_split_seed = 42
    cfg.dataset.shuffle = False
    cfg.dataset.sample_validation = True
    cfg.dataset.validation_sample_fraction = 0.4
    cfg.dataset.validation_sample_stratified = False
    cfg.dataset.validation_sample_seed = 43
    cfg.dataset.reverse_training_data = False
    cfg.dataset.combine_system_prompt = False

    # ====== Model settings ======
    cfg.model = CfgNode()
    cfg.model.name = ''

    # ====== Training settings ======
    cfg.train = CfgNode()
    cfg.train.eval_on_first_step = True
    cfg.train.recall_masking = True
    cfg.train.use_recall_tokens = False
    cfg.train.freeze_base_model = False

    # ====== SFTConfig kwargs ======
    cfg.sft = CfgNode()
    cfg.sft.eval_strategy = "no"
    cfg.sft.eval_steps = 100
    cfg.sft.max_steps = -1
    cfg.sft.num_train_epochs = 5
    cfg.sft.per_device_train_batch_size = 2
    cfg.sft.per_device_eval_batch_size = 2
    cfg.sft.max_length = 16000
    cfg.sft.gradient_accumulation_steps = 8
    cfg.sft.eval_accumulation_steps = 1
    cfg.sft.gradient_checkpointing = True
    cfg.sft.learning_rate = 5e-5
    cfg.sft.weight_decay = 0.0
    cfg.sft.adam_beta1 = 0.9
    cfg.sft.adam_beta2 = 0.999
    cfg.sft.adam_epsilon = 1e-8
    cfg.sft.bf16 = True
    cfg.sft.logging_steps = 1
    cfg.sft.save_strategy = "best"
    cfg.sft.save_steps = 100
    cfg.sft.metric_for_best_model = "eval_loss"
    cfg.sft.save_only_model = True
    cfg.sft.save_total_limit = 20
    cfg.sft.deepspeed = "ds_config.json"
    cfg.sft.use_liger_kernel = False
    cfg.sft.padding_free = False
    cfg.sft.torch_empty_cache_steps = 1
    cfg.sft.lr_scheduler_type = "cosine_with_min_lr"
    cfg.sft.lr_scheduler_kwargs = CfgNode()
    cfg.sft.lr_scheduler_kwargs.min_lr_rate = 0.1
    cfg.sft.warmup_ratio = 0.1
    cfg.sft.completion_only_loss = True
    cfg.sft.neftune_noise_alpha = None
    cfg.sft.dataloader_drop_last = True
    cfg.sft.logging_nan_inf_filter = False
    cfg.sft.remove_unused_columns = False
    # Skips materializing logits for prompt tokens, major VRAM savings.
    # NOTE: when recall_masking is disabled, this can conflict with liger kernel's
    # fused cross-entropy loss. If you disable recall masking, also disable this.
    cfg.sft.use_logits_to_keep = True

    return cfg


def update_config(p_cfg: CfgNode, config_path, arg_opts: list, run_id: str = None, no_run_id: bool = False):
    """
    Updates the config based on a config file and commandline arguments.
    """
    p_cfg.merge_from_file(config_path)
    p_cfg.merge_from_list(arg_opts)

    if no_run_id:
        p_cfg.run_id = None
        p_cfg.name_id = p_cfg.name
        p_cfg.save_path = os.path.join(p_cfg.save_dir, p_cfg.name_id)
    else:
        if run_id is not None:
            p_cfg.run_id = run_id
        else:
            import uuid
            p_cfg.run_id = uuid.uuid4().hex
        p_cfg.name_id = f'{p_cfg.name}_{p_cfg.run_id}'
        p_cfg.save_path = os.path.join(p_cfg.save_dir, p_cfg.name_id)

    p_cfg.merge_from_list(arg_opts)


def finalize_config(p_cfg: CfgNode):
    """
    Freezes the configuration to prevent further modifications.
    """
    p_cfg.freeze()


SFT_CONFIG = get_cfg_defaults()
