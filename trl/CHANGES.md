# TRL Modifications for RecaLLM

This is a vendored copy of [TRL](https://github.com/huggingface/trl) used for SFT training in RecaLLM.

## Upstream Version

- **Version:** TRL v0.21
- **Commit:** `de27d61` ("Validate `vllm_mode` param in GRPO")
- **License:** Apache 2.0 (Copyright 2020-2025 The HuggingFace Team)

## Modifications

### `trl/trainer/sft_config.py`

Added `use_logits_to_keep` parameter (default: `True`). When enabled, only computes logits for completion tokens (not prompt tokens), resulting in major VRAM savings during SFT training.

### `trl/trainer/sft_trainer.py`

Added `logits_to_keep` optimization in `compute_loss()`. When `use_logits_to_keep=True` and the model supports a `logits_to_keep` forward argument, the trainer:

1. Computes the number of leading prompt tokens (where `labels == -100`)
2. Passes `logits_to_keep` to the model's forward pass, avoiding materializing the full logits tensor
3. Truncates labels to match the reduced logits

This avoids allocating a `(batch_size, seq_len, vocab_size)` tensor for the full sequence, which is the dominant VRAM cost during SFT training of 7B+ models.
