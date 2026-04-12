# RecaLLM modifications to VeRL

**Upstream:** [verl-project/verl](https://github.com/verl-project/verl) commit `e7376432` (Jan 27, 2026)
**License:** Apache-2.0 (same as upstream)

## Modifications

### `verl/utils/torch_functional.py`

- **Flash-attn cross-entropy NaN fix:** RecaLLM's constrained decoding masks out invalid tokens by setting logits to `-inf`. Flash-attention's cross-entropy kernel produces NaN when given `-inf` logits. Added a safeguard that falls back to the v2 kernel when `-inf` is detected.
- **Entropy `-inf` fix:** The entropy calculation `p * log(p)` produces `NaN` when `p=0` and `log(p)=-inf` (since `0 * -inf = NaN`). Fixed by replacing `-inf` logits with `0` before the entropy multiplication.

### `verl/trainer/ppo/metric_utils.py`

- **NaN-safe aggregation:** Replaced `np.mean`/`np.std`/`np.max`/`np.min` with `np.nanmean`/`np.nanstd`/`np.nanmax`/`np.nanmin`. In RecaLLM's multi-task training, per-category metrics are NaN when a category has no examples in a batch. Standard numpy aggregation propagates NaN to the overall metric; nan-safe variants correctly skip missing categories.

### `verl/trainer/ppo/ray_trainer.py`

- **Reward extra info logging:** Log per-step reward component metrics (answer_score, correct_format, etc.) from the reward function's `reward_extra_info` dict.
- **Data source tracking:** Log per-completion data_source (dataset name) for multi-task analysis.
- **Response length logging:** Track response lengths per completion.
- **Advantage logging:** Log masked per-completion mean advantage for training diagnostics.
- **File logger routing:** Route file logger output to the run's save directory via `VERL_FILE_LOGGER_PATH`.
- **`_to_py()` helper:** Convert numpy generic types to Python scalars for JSON-safe logging.

### `verl/workers/rollout/vllm_rollout/vllm_async_server.py`

- **`hf_overrides` merge fix:** RecaLLM passes `hf_overrides` in `engine_kwargs` to override `model_type` and `architectures` (so vLLM loads the standard architecture instead of the custom RecaLLM class). The original code silently overwrote user-supplied `hf_overrides` with internally-built ones (e.g., quantization config). Fixed to merge both dicts, preserving user-supplied overrides.
