"""
vLLM logits processor for RecaLLM recall masking during inference.

Integrates with vLLM's v1 logits processor API to constrain decoding
inside recall spans to valid context continuations.
"""
from vllm.v1.sample.logits_processor import LogitsProcessor as VLLMLogitsProcessor, BatchUpdate, MoveDirectionality
from vllm.config import VllmConfig
from vllm import SamplingParams
import torch
from typing import Optional
from transformers import AutoTokenizer
from dataclasses import dataclass
from .modelling_recallm import get_prefix_continuations, RECALL_MASK_VALUE


@dataclass
class RecaLLMState:
    in_recall: bool = False
    recall_start_idx: Optional[int] = None


class VLLMTokenRecaLLMLogitsProcessor(VLLMLogitsProcessor):
    """
    vLLM logits processor for RecaLLM using single-token recall delimiters.

    At each decoding step, if the current request is inside a recall span,
    masks out all tokens except valid prefix continuations from the context.

    See: https://docs.vllm.ai/en/latest/features/custom_logitsprocs/
    """
    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
        self.tokenizer = AutoTokenizer.from_pretrained(vllm_config.model_config.model)

        self.prompt_ids: dict[int, list[int]] = {}
        self.output_ids: dict[int, list[int]] = {}
        self.sampling_params: dict[int, SamplingParams] = {}
        self.last_seen_len: dict[int, int] = {}
        self.state: dict[int, RecaLLMState] = {}

        self.recall_start_token_id = self.tokenizer.convert_tokens_to_ids("<|start_recall|>")
        self.recall_end_token_id = self.tokenizer.convert_tokens_to_ids("<|end_recall|>")
        self.max_recall_length = 128

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        extra_args = sampling_params.extra_args

    def is_argmax_invariant(self) -> bool:
        return False

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply recall masking to the batch logits tensor.

        For each request in a recall span, only allow tokens that continue
        a valid prefix match against the context (prompt + generated tokens
        before the recall span).
        """
        for request_idx, recall_state in self.state.items():
            if recall_state.in_recall:
                output_ids = self.output_ids[request_idx]
                prompt_ids = self.prompt_ids[request_idx]

                current_recall_toks = torch.tensor(output_ids[recall_state.recall_start_idx:])
                recall_span_len = len(current_recall_toks)

                # Context = prompt + output before the recall start token
                context = torch.tensor(prompt_ids + output_ids[:max(0, recall_state.recall_start_idx - 1)])

                continuations = get_prefix_continuations(context, current_recall_toks)
                continuations = continuations[continuations != self.recall_start_token_id]

                additional_allowed_tokens = torch.tensor([self.recall_end_token_id])

                if recall_span_len >= self.max_recall_length:
                    allowed_tokens = additional_allowed_tokens
                else:
                    allowed_tokens = torch.concat([continuations, additional_allowed_tokens])

                mask = torch.ones_like(logits[request_idx], dtype=torch.bool)
                mask[allowed_tokens] = False
                mask = mask.to(device=logits.device)

                logits[request_idx][mask] = RECALL_MASK_VALUE

        return logits

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        """
        Process batch updates: removed, added, and moved requests.

        Must be processed in order: removed -> added -> moved.
        Then update recall state for any new output tokens.
        """
        if batch_update:
            for idx in batch_update.removed:
                self.prompt_ids.pop(idx)
                self.output_ids.pop(idx)
                self.sampling_params.pop(idx)
                self.state.pop(idx)
                self.last_seen_len.pop(idx)

            for idx, sampling_params, prompt_tok_ids, output_tok_ids in batch_update.added:
                self.prompt_ids[idx] = prompt_tok_ids
                self.output_ids[idx] = output_tok_ids
                self.sampling_params[idx] = sampling_params
                self.state[idx] = RecaLLMState()
                self.last_seen_len[idx] = 0

            for old_idx, new_idx, directionality in batch_update.moved:
                if directionality == MoveDirectionality.UNIDIRECTIONAL:
                    self.prompt_ids[new_idx] = self.prompt_ids.pop(old_idx)
                    self.output_ids[new_idx] = self.output_ids.pop(old_idx)
                    self.sampling_params[new_idx] = self.sampling_params.pop(old_idx)
                    self.state[new_idx] = self.state.pop(old_idx)
                    self.last_seen_len[new_idx] = self.last_seen_len.pop(old_idx)
                elif directionality == MoveDirectionality.SWAP:
                    (self.prompt_ids[old_idx], self.prompt_ids[new_idx]) = (self.prompt_ids[new_idx], self.prompt_ids[old_idx])
                    (self.output_ids[old_idx], self.output_ids[new_idx]) = (self.output_ids[new_idx], self.output_ids[old_idx])
                    (self.sampling_params[old_idx], self.sampling_params[new_idx]) = (self.sampling_params[new_idx], self.sampling_params[old_idx])
                    (self.state[old_idx], self.state[new_idx]) = (self.state[new_idx], self.state[old_idx])
                    (self.last_seen_len[old_idx], self.last_seen_len[new_idx]) = (self.last_seen_len[new_idx], self.last_seen_len[old_idx])

        # Process new output tokens and update recall state
        for request_idx, output_tok_ids in self.output_ids.items():
            last_len = self.last_seen_len[request_idx]
            recall_state = self.state[request_idx]
            new_len = len(output_tok_ids)
            self.last_seen_len[request_idx] = new_len
            for tok_idx in range(last_len, new_len):
                tok = output_tok_ids[tok_idx]
                if recall_state.in_recall:
                    if tok == self.recall_end_token_id:
                        recall_state.in_recall = False
                        recall_state.recall_start_idx = None
                else:
                    if tok == self.recall_start_token_id:
                        recall_state.in_recall = True
                        recall_state.recall_start_idx = tok_idx + 1
