"""
RecaLLM model classes and recall masking logic.

Provides the RecaLLM mixin that adds constrained-decoding recall spans to
any causal language model. During a recall span, the model can only generate
tokens that continue a valid prefix match against the input context.

Supports Llama and Qwen2 architectures.
"""
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Model,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
from typing import Optional, Union
import contextlib
import inspect
import threading

# Large finite negative used instead of float('-inf') for recall masking.
# Using -inf triggers VeRL's isinf() guard in logprobs_from_logits(), which
# diverts from the fast Flash Attention Triton kernel to a row-by-row Python
# loop (~60x slower). exp(-1e4) underflows to exactly 0 in bfloat16/float16/float32,
# so softmax behavior is identical, but isinf() returns False.
# -1e4 is the largest safe value across all dtypes (float16 max is 65504;
# -1e5 overflows to -inf in float16).
RECALL_MASK_VALUE = -1e4


from dataclasses import dataclass, field


@dataclass
class RecallSpan:
    """Tracks a single recall span within a sequence during generation."""
    input_idx_start: int
    input_idx_end: int
    logits_start: int
    logits_end: int
    span_start: int
    span_end: int
    allowed_tokens_list: list = field(default_factory=list)

    def get_mask(self, vocab_size: int) -> torch.Tensor:
        """Build a [span_length, vocab_size] bool mask (True = disallowed)."""
        n_positions = len(self.allowed_tokens_list)
        mask = torch.ones(n_positions, vocab_size, dtype=torch.bool)
        for i, allowed in enumerate(self.allowed_tokens_list):
            mask[i, allowed] = False
        return mask


def get_prefix_continuations(
    context: torch.Tensor,
    prefix: torch.Tensor,
) -> torch.Tensor:
    """
    Find all valid next-token continuations of ``prefix`` in ``context``.

    Parameters
    ----------
    context : torch.Tensor
        1-D tensor of token IDs representing the available context.
    prefix : torch.Tensor
        1-D tensor of token IDs representing the current recall span so far.

    Returns
    -------
    torch.Tensor
        1-D tensor of unique token IDs that could validly continue the prefix.
    """
    n = len(prefix)
    if n == 0:
        return context.unique()
    if n >= len(context):
        return torch.tensor([], dtype=context.dtype, device=context.device)

    windows = context.unfold(0, n, 1)
    match_mask = (windows == prefix.unsqueeze(0)).all(dim=1)
    match_indices = match_mask.nonzero(as_tuple=False).squeeze(1)
    continuation_indices = match_indices + n
    valid = continuation_indices < len(context)
    continuation_indices = continuation_indices[valid]

    if len(continuation_indices) == 0:
        return torch.tensor([], dtype=context.dtype, device=context.device)
    return context[continuation_indices].unique()


def _get_prefix_continuations_incremental(
    context: torch.Tensor,
    span_tokens: torch.Tensor,
    recall_start_token_id: int,
    recall_end_token_id: int,
) -> list[torch.Tensor]:
    """
    Compute valid continuations for every prefix length in a recall span.

    Returns R+1 tensors (for the empty prefix through the full span), each
    containing the allowed token IDs at that position. The recall start token
    is excluded from continuations and the recall end token is always appended.

    This is O(C) per span (single pass over context) instead of O(C*R).
    """
    C = len(context)
    R = len(span_tokens)
    device = context.device

    end_tok = torch.tensor([recall_end_token_id], device=device)

    if C == 0:
        return [end_tok] * (R + 1)

    # Build match matrix: match[r][c] = True iff span_tokens[r] == context[c]
    if R > 0:
        match = (span_tokens.unsqueeze(1) == context.unsqueeze(0))
    else:
        match = torch.empty(0, C, dtype=torch.bool, device=device)

    # alive[c] tracks which context positions still match the prefix at each step
    alive = torch.ones(C, dtype=torch.bool, device=device)
    results = []

    # k=0: empty prefix -> any context token is a valid continuation
    conts_0 = context[alive].unique()
    conts_0 = conts_0[conts_0 != recall_start_token_id]
    results.append(torch.cat([conts_0, end_tok]))

    for k in range(R):
        alive_shifted = torch.zeros(C, dtype=torch.bool, device=device)
        if k == 0:
            alive_shifted[1:] = match[0, :-1]
        else:
            prev_alive = alive
            alive_shifted[1:] = prev_alive[:-1] & match[k, :-1]
        alive = alive_shifted

        if not alive.any():
            results.append(end_tok)
        else:
            conts = context[alive].unique()
            conts = conts[conts != recall_start_token_id]
            results.append(torch.cat([conts, end_tok]))

    return results


class RecaLLMGenerateLogitsProcessor(LogitsProcessor):
    """
    HF LogitsProcessor for recall masking during .generate().

    Tracks recall state token-by-token and masks logits so that only valid
    context continuations (or the recall end token) are allowed inside recall
    spans. Must be re-created for each generate() call.
    """
    def __init__(
        self,
        recall_start_token_id: int,
        recall_end_token_id: int,
        max_recall_length: int = 128,
    ):
        self.recall_start_token_id = recall_start_token_id
        self.recall_end_token_id = recall_end_token_id
        self.max_recall_length = max_recall_length

        self.initial_input_ids: Optional[torch.Tensor] = None
        self.context_starts: Optional[torch.Tensor] = None
        self.batch_size: int = 0
        self.prompt_length: int = 0
        self.in_recall: Optional[torch.Tensor] = None
        self.recall_start_pos: Optional[torch.Tensor] = None

    def setup(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Store the prompt context. Call once before generate()."""
        self.initial_input_ids = input_ids
        self.batch_size = input_ids.shape[0]
        self.prompt_length = input_ids.shape[1]
        self.in_recall = torch.zeros(self.batch_size, dtype=torch.bool)
        self.recall_start_pos = torch.zeros(self.batch_size, dtype=torch.long)

        if attention_mask is not None:
            self.context_starts = attention_mask.argmax(dim=-1)
        else:
            self.context_starts = torch.zeros(self.batch_size, dtype=torch.long)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Called at each generation step.

        Parameters
        ----------
        input_ids : (batch_size, sequence_length)
            Full token sequence so far (prompt + generated).
        scores : (batch_size, vocab_size)
            Logits for the next token.
        """
        n_total = input_ids.shape[1]
        # Number of generated tokens (excluding prompt)
        n_generated = n_total - self.prompt_length
        if n_generated < 1:
            return scores

        scores = scores.clone()

        for batch_idx in range(self.batch_size):
            last_token = input_ids[batch_idx, -1].item()

            # Update recall state based on last generated token
            if self.in_recall[batch_idx]:
                if last_token == self.recall_end_token_id:
                    self.in_recall[batch_idx] = False
            else:
                if last_token == self.recall_start_token_id:
                    self.in_recall[batch_idx] = True
                    self.recall_start_pos[batch_idx] = n_total

            if self.in_recall[batch_idx]:
                # Tokens generated inside this recall span
                prefix = input_ids[batch_idx, self.recall_start_pos[batch_idx]:]
                # Context = original prompt content (excluding padding)
                context = self.initial_input_ids[batch_idx, self.context_starts[batch_idx]:]

                continuations = get_prefix_continuations(context, prefix)
                # Exclude recall start token from continuations
                continuations = continuations[continuations != self.recall_start_token_id]

                end_token = torch.tensor([self.recall_end_token_id], device=scores.device)
                recall_span_length = n_total - self.recall_start_pos[batch_idx].item()

                if recall_span_length >= self.max_recall_length:
                    allowed_tokens = end_token
                else:
                    allowed_tokens = torch.cat([continuations.to(scores.device), end_token])

                mask = torch.full_like(scores[batch_idx], fill_value=RECALL_MASK_VALUE)
                mask[allowed_tokens] = 0.0
                scores[batch_idx] += mask

        return scores


class RecaLLMConfigMixin:
    """
    Configuration mixin for RecaLLM models.

    Parameters
    ----------
    force_recall : bool, default True
        If True, mask out logits for tokens that would produce non-viable
        sequences inside recall spans.
    recall_tag_type : str, default "token_id"
        Must be "token_id" — uses single-token delimiters for recall spans.
    recall_start_token_id : int
        Token ID for the start of a recall span.
    recall_end_token_id : int
        Token ID for the end of a recall span.
    recall_start_str : str, default "<recall>"
        String representation of recall start (for display/prompts).
    recall_end_str : str, default "</recall>"
        String representation of recall end (for display/prompts).
    max_recall_length : int, default 128
        Maximum tokens in a recall span before forcing closure.
    use_fast_recall : bool, default True
        Use the optimized background-thread forward pass.
    """
    def __init__(
            self,
            force_recall: bool = True,
            recall_tag_type: str = "token_id",
            recall_start_str: Optional[str] = "<recall>",
            recall_end_str: Optional[str] = "</recall>",
            recall_start_token_id: Optional[int] = None,
            recall_end_token_id: Optional[int] = None,
            record_recall_prob_after_generate: bool = False,
            max_recall_length: int = 128,
            use_fast_recall: bool = True,
            **kwargs):
        super().__init__(**kwargs)
        self.force_recall = force_recall
        self.recall_tag_type = recall_tag_type
        self.recall_start_str = recall_start_str
        self.recall_end_str = recall_end_str
        self.recall_start_token_id = recall_start_token_id
        self.recall_end_token_id = recall_end_token_id
        self.record_recall_prob_after_generate = record_recall_prob_after_generate
        self.max_recall_length = max_recall_length
        self.use_fast_recall = use_fast_recall


class RecaLLMMixin:
    """
    Mixin that adds RecaLLM recall masking to any causal LM.

    Overrides ``forward()`` to mask logits at positions inside recall spans,
    allowing only tokens that continue a valid prefix match against the
    input context. Uses background threading + CUDA stream overlap for
    efficient masking during training.
    """
    def __init__(self, config: RecaLLMConfigMixin, recall_enabled_in_forward_pass: bool = True):
        super().__init__(config)
        self.recall_enabled_in_forward_pass = recall_enabled_in_forward_pass

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.LongTensor] = 0,
            recall_enabled: bool | None = None,
            **kwargs
        ) -> CausalLMOutputWithPast:
        """
        Forward pass with optional recall masking.

        When recall masking is enabled, logits at positions inside recall spans
        are masked so that only valid context continuations are allowed.

        Parameters
        ----------
        recall_enabled : bool or None
            Override for recall masking. None uses ``self.recall_enabled_in_forward_pass``.
        """
        if recall_enabled is None:
            recall_enabled = self.recall_enabled_in_forward_pass

        if recall_enabled:
            assert input_ids is not None, \
                "`input_ids` must be provided when recall masking is enabled."
            assert self.config.recall_tag_type == "token_id", \
                "Only recall_tag_type='token_id' is supported."
            return self._fast_recall_forward(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, labels=labels,
                logits_to_keep=logits_to_keep, **kwargs)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

    @staticmethod
    def _compute_recall_masks(
        input_ids_cpu: torch.Tensor,
        attention_mask_cpu: Optional[torch.Tensor],
        position_ids_cpu: Optional[torch.Tensor],
        sequence_length: int,
        vocab_size: int,
        start_id: int,
        end_id: int,
        eos_ids_list: list[int],
        keep_logit_start: Optional[int],
        logits_to_keep_map: Optional[dict],
    ) -> list[tuple[int, int, int, torch.Tensor]]:
        """
        Compute all recall masks on CPU (runs in a background thread).

        Detects recall spans via a state machine, computes valid prefix
        continuations for each position, and builds boolean masks.

        Returns
        -------
        list of (batch_idx, logits_start, logits_end, bool_mask_cpu) tuples.
        """
        batch_size = input_ids_cpu.shape[0]
        mask_instructions = []

        for batch_idx in range(batch_size):
            seq_cpu = input_ids_cpu[batch_idx]
            attn_cpu = attention_mask_cpu[batch_idx] if attention_mask_cpu is not None else None
            pos_cpu = position_ids_cpu[batch_idx] if position_ids_cpu is not None else None

            # Determine sequence bounds
            if attn_cpu is not None:
                nonzero = attn_cpu.nonzero(as_tuple=False).squeeze(1)
                start_idx = nonzero.min().item()
                end_idx = nonzero.max().item() + 1
            else:
                start_idx = 0
                end_idx = sequence_length

            # Vectorized event detection
            seq_slice = seq_cpu[start_idx:end_idx]

            is_start = (seq_slice == start_id)
            start_positions = is_start.nonzero(as_tuple=False).squeeze(1) + start_idx

            is_end = (seq_slice == end_id)
            for eid in eos_ids_list:
                is_end = is_end | (seq_slice == eid)
            end_positions = is_end.nonzero(as_tuple=False).squeeze(1) + start_idx

            if pos_cpu is not None:
                pos_slice = pos_cpu[start_idx:end_idx]
                is_boundary = torch.zeros(pos_slice.shape[0], dtype=torch.bool)
                if pos_slice.shape[0] > 1:
                    is_boundary[1:] = (pos_slice[1:] == 0)
                boundary_positions = is_boundary.nonzero(as_tuple=False).squeeze(1) + start_idx
            else:
                boundary_positions = torch.tensor([], dtype=torch.long)

            # Build sorted event list
            events = []
            for p in start_positions.tolist():
                events.append((p, 0))  # start recall
            for p in end_positions.tolist():
                events.append((p, 1))  # end recall
            for p in boundary_positions.tolist():
                events.append((p, 2))  # sequence boundary
            events.sort(key=lambda e: (e[0], e[1]))

            # State machine to extract recall spans
            in_recall_mode = False
            context_start = start_idx
            current_content_start = None
            current_ctx_start = None
            raw_spans = []

            for pos, etype in events:
                if etype == 0:
                    if not in_recall_mode:
                        in_recall_mode = True
                        current_content_start = pos + 1
                        current_ctx_start = context_start
                elif etype == 1:
                    if in_recall_mode:
                        in_recall_mode = False
                        raw_spans.append((current_ctx_start, current_content_start, pos))
                        current_content_start = None
                        current_ctx_start = None

                if etype == 2:
                    if in_recall_mode:
                        raw_spans.append((current_ctx_start, current_content_start, pos))
                        in_recall_mode = False
                        current_content_start = None
                        current_ctx_start = None
                    context_start = pos

            if in_recall_mode:
                raw_spans.append((current_ctx_start, current_content_start, end_idx))

            # For each span: prefix match + build bool mask
            for (ctx_start, content_start, content_end) in raw_spans:
                if content_end < content_start:
                    continue

                start_token_pos = content_start - 1

                if keep_logit_start is not None:
                    if content_end - 1 < keep_logit_start:
                        continue
                elif logits_to_keep_map is not None:
                    has_any = False
                    for p in range(start_token_pos, content_end):
                        if p in logits_to_keep_map:
                            has_any = True
                            break
                    if not has_any:
                        continue

                context = seq_cpu[ctx_start:content_start - 1]
                span_tokens = seq_cpu[content_start:content_end]

                allowed_lists = _get_prefix_continuations_incremental(
                    context=context,
                    span_tokens=span_tokens,
                    recall_start_token_id=start_id,
                    recall_end_token_id=end_id,
                )

                # Filter to positions that have logits
                mask_positions = list(range(start_token_pos, content_end))
                filtered_allowed = []
                logits_start = None
                logits_end = None

                for i, abs_pos in enumerate(mask_positions):
                    logits_idx = None
                    if keep_logit_start is not None:
                        if abs_pos >= keep_logit_start:
                            logits_idx = abs_pos - keep_logit_start
                    elif logits_to_keep_map is not None:
                        if abs_pos in logits_to_keep_map:
                            logits_idx = logits_to_keep_map[abs_pos]

                    if logits_idx is not None:
                        filtered_allowed.append(allowed_lists[i])
                        if logits_start is None:
                            logits_start = logits_idx
                        logits_end = logits_idx + 1

                # Build boolean mask on CPU
                if logits_start is not None:
                    N = len(filtered_allowed)
                    bool_mask = torch.ones(N, vocab_size, dtype=torch.bool, pin_memory=True)
                    sizes = torch.tensor([a.numel() for a in filtered_allowed])
                    row_idx = torch.repeat_interleave(torch.arange(N), sizes)
                    col_idx = torch.cat(filtered_allowed)
                    bool_mask[row_idx, col_idx] = False

                    mask_instructions.append((batch_idx, logits_start, logits_end, bool_mask))

        return mask_instructions

    def _fast_recall_forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.LongTensor] = 0,
            **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Optimized forward pass with recall masking (token_id mode only).

        Overlaps CPU-bound mask computation and GPU mask transfer with the
        model forward pass using a background thread + CUDA stream.

        Flow:
          1. Copy input_ids / attention_mask / position_ids to CPU
          2. Background thread: span detection + prefix matching + bool mask
             building (CPU), then transfer masks to GPU on a dedicated stream
          3. Run super().forward() on the default CUDA stream (overlaps with 2)
          4. Join thread, sync streams, apply masks to logits
          5. Compute loss if labels provided
        """
        assert self.config.recall_tag_type == "token_id", \
            "_fast_recall_forward only supports recall_tag_type='token_id'"

        batch_size = input_ids.shape[0]
        sequence_length = input_ids.shape[1]

        # --- Step A: Precompute config values ---
        keep_logit_start = None
        if isinstance(logits_to_keep, int):
            keep_logit_start = 0 if logits_to_keep == 0 else sequence_length - logits_to_keep

        logits_to_keep_map = None
        if isinstance(logits_to_keep, torch.Tensor):
            ltk_cpu = logits_to_keep.cpu()
            logits_to_keep_map = {pos.item(): idx for idx, pos in enumerate(ltk_cpu)}

        start_id = self.config.recall_start_token_id
        end_id = self.config.recall_end_token_id
        eos_id = self.config.eos_token_id
        if isinstance(eos_id, int):
            eos_ids_list = [eos_id]
        elif isinstance(eos_id, list):
            eos_ids_list = eos_id
        else:
            eos_ids_list = []

        vocab_size = self.config.vocab_size

        # --- Step B: Copy inputs to CPU ---
        input_ids_cpu = input_ids.cpu()
        attention_mask_cpu = attention_mask.cpu() if attention_mask is not None else None
        position_ids_cpu = position_ids.cpu() if position_ids is not None else None

        # --- Step C: Background thread for CPU mask computation + GPU transfer ---
        target_device = input_ids.device
        thread_result = [None]
        thread_stream = [None]
        thread_error = [None]

        def _mask_thread():
            try:
                masks_cpu = self._compute_recall_masks(
                    input_ids_cpu=input_ids_cpu,
                    attention_mask_cpu=attention_mask_cpu,
                    position_ids_cpu=position_ids_cpu,
                    sequence_length=sequence_length,
                    vocab_size=vocab_size,
                    start_id=start_id,
                    end_id=end_id,
                    eos_ids_list=eos_ids_list,
                    keep_logit_start=keep_logit_start,
                    logits_to_keep_map=logits_to_keep_map,
                )
                if masks_cpu and target_device.type == 'cuda':
                    stream = torch.cuda.Stream(device=target_device)
                    masks_gpu = []
                    with torch.cuda.stream(stream):
                        for batch_idx, ls, le, bm_cpu in masks_cpu:
                            bm_gpu = bm_cpu.to(target_device, non_blocking=True)
                            masks_gpu.append((batch_idx, ls, le, bm_gpu))
                    thread_result[0] = masks_gpu
                    thread_stream[0] = stream
                else:
                    thread_result[0] = masks_cpu
            except Exception as e:
                thread_error[0] = e

        mask_thread = threading.Thread(target=_mask_thread, daemon=True)
        mask_thread.start()

        # --- Step D: Run base model forward on GPU (overlaps with thread) ---
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
            position_ids=position_ids,
            **kwargs,
        )

        # --- Step E: Wait for background thread ---
        mask_thread.join()

        if thread_error[0] is not None:
            raise thread_error[0]

        mask_instructions = thread_result[0]
        logits_device = outputs.logits.device

        # --- Step F: Apply masks to logits ---
        if thread_stream[0] is not None:
            torch.cuda.current_stream(logits_device).wait_stream(thread_stream[0])

        for batch_idx, logits_start, logits_end, bool_mask in mask_instructions:
            outputs.logits[batch_idx, logits_start:logits_end, :].masked_fill_(
                bool_mask, RECALL_MASK_VALUE
            )

        # --- Step G: Compute loss if labels provided ---
        if labels is not None:
            loss_func_params = set(inspect.signature(self.loss_function).parameters.keys())
            loss_func_kwargs = {key: val for key, val in kwargs.items() if key in loss_func_params}
            outputs["loss"] = self.loss_function(
                logits=outputs.logits, labels=labels,
                vocab_size=self.config.vocab_size, **loss_func_kwargs,
            )

        return outputs

    def generate(self, inputs=None, logits_processor=None, **kwargs):
        """
        Generate with recall masking via a LogitsProcessor.

        Disables recall masking in the forward pass (incompatible with
        KV-cached token-by-token decoding) and instead applies masking
        through a LogitsProcessor that tracks state across steps.
        """
        with self.temp_disable_recall_in_forward_pass():
            input_ids = inputs if inputs is not None else kwargs.get('input_ids')
            attention_mask = kwargs.get('attention_mask')

            processor = RecaLLMGenerateLogitsProcessor(
                recall_start_token_id=self.config.recall_start_token_id,
                recall_end_token_id=self.config.recall_end_token_id,
                max_recall_length=self.config.max_recall_length,
            )
            processor.setup(input_ids=input_ids, attention_mask=attention_mask)

            if logits_processor is None:
                logits_processor = LogitsProcessorList([processor])
            else:
                logits_processor.insert(0, processor)

            return super().generate(
                inputs=inputs, logits_processor=logits_processor, **kwargs
            )

    def enable_recall_in_forward_pass(self):
        self.recall_enabled_in_forward_pass = True

    def disable_recall_in_forward_pass(self):
        self.recall_enabled_in_forward_pass = False

    @contextlib.contextmanager
    def temp_disable_recall_in_forward_pass(self):
        original = self.recall_enabled_in_forward_pass
        self.disable_recall_in_forward_pass()
        try:
            yield
        finally:
            self.recall_enabled_in_forward_pass = original

    def get_base_model(self):
        return super()


# ── Model Registration ──────────────────────────────────────────────────────
# Llama 3 RecaLLM models
class RecaLLMLlamaConfig(RecaLLMConfigMixin, LlamaConfig):
    model_type = "recallm_llama"

class RecaLLMLlamaForCausalLM(RecaLLMMixin, LlamaForCausalLM):
    config_class = RecaLLMLlamaConfig

class RecaLLMLlamaModel(RecaLLMMixin, LlamaModel):
    config_class = RecaLLMLlamaConfig
    _supports_attention_backend = True

AutoConfig.register("recallm_llama", RecaLLMLlamaConfig)
AutoModelForCausalLM.register(RecaLLMLlamaConfig, RecaLLMLlamaForCausalLM)
AutoModel.register(RecaLLMLlamaConfig, RecaLLMLlamaModel)

# Qwen 2 (and 2.5) RecaLLM models
class RecaLLMQwen2Config(RecaLLMConfigMixin, Qwen2Config):
    model_type = "recallm_qwen2"

class RecaLLMQwen2ForCausalLM(RecaLLMMixin, Qwen2ForCausalLM):
    config_class = RecaLLMQwen2Config

class RecaLLMQwen2Model(RecaLLMMixin, Qwen2Model):
    config_class = RecaLLMQwen2Config
    _supports_attention_backend = True

AutoConfig.register("recallm_qwen2", RecaLLMQwen2Config)
AutoModelForCausalLM.register(RecaLLMQwen2Config, RecaLLMQwen2ForCausalLM)
AutoModel.register(RecaLLMQwen2Config, RecaLLMQwen2Model)
