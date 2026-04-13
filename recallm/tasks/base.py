"""
Shared base classes and utilities for dataset generation.
"""

from abc import ABC, abstractmethod

import numpy as np
from transformers.tokenization_utils import PreTrainedTokenizerBase


_CONTEXT_SAMPLING_SEED_SALT = 0xC0FFEE


def normalize_context_range(target_context: int, context_length_max: int | None = None) -> tuple[int, int]:
    """Return an inclusive [min, max] context range after validation."""
    target_context = int(target_context)
    if context_length_max is None:
        return target_context, target_context

    context_length_max = int(context_length_max)
    if context_length_max < target_context:
        raise ValueError(
            f"context_length_max ({context_length_max}) must be >= target_context ({target_context})"
        )
    return target_context, context_length_max


def sample_target_context(
    seed_base: int,
    target_context: int,
    context_length_max: int | None = None,
) -> int:
    """Sample a deterministic per-example target context from an inclusive range."""
    context_min, context_max = normalize_context_range(target_context, context_length_max)
    if context_min == context_max:
        return context_min

    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed_base), _CONTEXT_SAMPLING_SEED_SALT])
    )
    return int(rng.integers(context_min, context_max + 1))


class TokenizeableExample(ABC):
    """
    Abstract class to represent prompts that will be tokenized, where we need to set k to a value such that the tokenized length will be less a certain
     max length.
    Parameters:
        max_k (int): The maximum value of k to consider.
        max_length (int): The maximum tokenized length allowed.
        step_size (int): The step size to increment k by when searching for the largest valid k.
        k (int): The current value of k. If None, it means k has not been set yet.
    """
    def __init__(self, max_k: int = None):
        self.max_k = max_k
        self.k = 0

    @abstractmethod
    def average_length_per_item(self, tokenizer) -> float:
        pass

    @abstractmethod
    def tokenized_length(self, tokenizer) -> int:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

    def set_largest_k(self, tokenizer: PreTrainedTokenizerBase, target_context: int, initial_step_size: int = 10, min_step_size: int = 1, verbose: bool = False):
        """
        Sets self.k to the largest possible value, such that the tokenized length of the example is less than or equal to max_length.
        """
        import time as _time
        _t0 = _time.perf_counter()
        _tokenize_calls = 0

        self.k = 0
        empty_length = self.tokenized_length(tokenizer)
        _tokenize_calls += 1
        if empty_length > target_context:
            return
        avg_length = self.average_length_per_item(tokenizer)

        # initial guess
        self.k = int((target_context - empty_length) / avg_length)
        initial_k = self.k

        # Scale step size with k so the linear scan phase is bounded to ~20
        # iterations regardless of how far off the initial guess is.
        step_size = max(initial_step_size, self.k // 20)

        # iteratively adjust k till we find the largest valid k
        decreasing = None

        # only recompute the length if k changes, not every time
        length = self.tokenized_length(tokenizer)
        _tokenize_calls += 1
        recompute_length = False

        _iterations = 0
        while True:
            _iterations += 1
            if recompute_length:
                length = self.tokenized_length(tokenizer)
                _tokenize_calls += 1
                recompute_length = False

            if decreasing is None:
                decreasing = length > target_context
            if length == target_context:
                # perfect, we're done
                break
            if decreasing:
                if length < target_context:
                    # the last step got us below the threshold...
                    if step_size == min_step_size:
                        # if we're already at the smallest step size, then we're done
                        break
                    else:
                        # otherwise, reduce the step size and change direction
                        step_size = max(step_size // 2, min_step_size)
                        decreasing = False
                else:
                    # keep going down
                    if self.k == 0:
                        # we can't go any lower, so we're done
                        break
                    self.k = max(self.k - step_size, 0)
                    recompute_length = True

            else:
                if length > target_context:
                    # the last step got us above the threshold...
                    # reduce the step size and change direction
                    step_size = max(step_size // 2, min_step_size)
                    decreasing = True
                else:
                    # keep going up
                    if self.max_k is not None:
                        if self.k == self.max_k:
                            # we can't go any higher, so we're done
                            break
                        self.k = min(self.k + step_size, self.max_k)
                        recompute_length = True
                    else:
                        self.k += step_size
                        recompute_length = True

        _elapsed = _time.perf_counter() - _t0
        if verbose:
            print(
                f"[set_largest_k] target={target_context} initial_k={initial_k} "
                f"final_k={self.k} delta={self.k - initial_k} "
                f"iterations={_iterations} tokenize_calls={_tokenize_calls} "
                f"avg_item_len={avg_length:.1f} elapsed={_elapsed:.3f}s"
            )
        return length


def insert_into_list(lst: list, values: list, depths: list[float]) -> tuple[list, list[int]]:
    """
    Inserts values into list at the specified depths, and returns the updated list and the indexes of the inserted values.
    Depths should be in the range [0, 1], where 0 is the start of the list and 1 is the end of the list.
    """
    if len(values) != len(depths):
        raise ValueError("Values and depths must have the same length.")

    item_depth_pairs = list(zip(depths, values, range(len(values))))
    item_depth_pairs.sort(key=lambda x: x[0])

    inserted_indexes = []
    for depth, value, _ in item_depth_pairs:
        index = int(depth * len(lst))
        lst.insert(index, value)
        inserted_indexes.append(index)

    # inserted_indexes must be in the same order as the original values
    pairs = [(value_index, inserted_index) for (depth, value, value_index), inserted_index in zip(item_depth_pairs, inserted_indexes)]
    pairs.sort(key=lambda x: x[0])  # Sort by original value index
    inserted_indexes = [index for _, index in pairs]
    return lst, inserted_indexes
