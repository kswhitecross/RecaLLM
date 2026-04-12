"""
RecaLLM — Recall-augmented Language Models with constrained decoding.

Provides model classes, configuration mixins, and logits processors for
training and serving language models that interleave reasoning with
explicit, constrained-decoding recall spans.
"""
from .modelling_recallm import (
    RECALL_MASK_VALUE,
    RecallSpan,
    get_prefix_continuations,
    _get_prefix_continuations_incremental,
    RecaLLMConfigMixin,
    RecaLLMMixin,
    RecaLLMLlamaConfig,
    RecaLLMLlamaForCausalLM,
    RecaLLMLlamaModel,
    RecaLLMQwen2Config,
    RecaLLMQwen2ForCausalLM,
    RecaLLMQwen2Model,
)
