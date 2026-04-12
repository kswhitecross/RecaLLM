from transformers import PreTrainedTokenizerBase
import re
from enum import Enum, auto
from datasets import Dataset
from typing import Optional


def find_tuple_idx(tuples_list: list[tuple[int, int]], idx: int) -> int:
    """
    Given a sorted list of (start, end) tuples, and an index, return the index of the tuple containing idx
    """
    if idx < tuples_list[0][0] or idx >= tuples_list[-1][1]:
        raise ValueError(f"Index {idx} out of bounds for tuples {tuples_list}")
    # binary search
    left = 0
    right = len(tuples_list) - 1
    while left < right:
        mid = (left + right) // 2
        start, end = tuples_list[mid]
        if start <= idx < end:
            return mid
        elif idx < start:
            right = mid - 1
        else:
            left = mid + 1
    return left



def tokenize_sft_completion(
        tokenizer: PreTrainedTokenizerBase,
        prompt_messages: list[dict[str, str]],
        completion_text: str,
        recall_start_str: str = "<recall>",
        recall_end_str: str = "</recall>"
) -> tuple[list[int], list[int]]:
    """
    Tokenizes a single SFT completion consisting of prompt messages and completion text.  Makes sure
     that all of the tokens inside recall spans are are valid sequences from the prompt.
    Returns input_ids and a completion mask.
    """
    # tokenize the prompt, and get offsets mapping
    prompt_text = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=False)
    prompt_inputs = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids = prompt_inputs['input_ids']
    offset_mapping = prompt_inputs['offset_mapping']

    # also, tokenize the recall end str, to make sure that it's tokenization in the completion follows the
    # way it was allowed in the recall masking
    recall_end_ids = tokenizer.encode(recall_end_str, add_special_tokens=False)

    # add eos token to completion text, if not already present
    if not completion_text.endswith(tokenizer.eos_token):
        completion_text = completion_text + tokenizer.eos_token
    
    # split the completion text into segments based on recall spans
    split_regex = re.compile(pattern = rf"(?<={re.escape(recall_start_str)})|(?={re.escape(recall_end_str)})|(?<={re.escape(recall_end_str)})")
    chunks = re.split(split_regex, completion_text)
    
    # determine what kind of segment each split is, as they're all tokenized differently
    class ChunkType(Enum):
        NORMAL_CONTENT = auto()
        RECALL_END = auto()
        RECALL_SPAN = auto()
    
    chunk_types: list[ChunkType] = []
    in_recall = False

    for chunk in chunks:
        # if it ends with a start string, it's normal content leading up to a recall span
        if chunk.endswith(recall_start_str):
            chunk_types.append(ChunkType.NORMAL_CONTENT)
            in_recall = True
        # if it ends with an end string, it's just a recall end string
        elif chunk.endswith(recall_end_str):
            chunk_types.append(ChunkType.RECALL_END)
            in_recall = False
        else:
            # otherwise, if we're in recall mode, it's a recall span
            if in_recall:
                chunk_types.append(ChunkType.RECALL_SPAN)
            else:
                # if not, it's normal content (ends with eos or end of completion)
                chunk_types.append(ChunkType.NORMAL_CONTENT)
    
    # tokenize each chunk
    chunk_input_ids: list[list[int]] = []
    for chunk, chunk_type in zip(chunks, chunk_types):
        if chunk_type == ChunkType.RECALL_END:
            chunk_ids = recall_end_ids
        elif chunk_type == ChunkType.NORMAL_CONTENT:
            chunk_ids = tokenizer(chunk, add_special_tokens=False)['input_ids']
        elif chunk_type == ChunkType.RECALL_SPAN:
            # for recall spans, we need to find the corresponing sequence of tokens in the prompt
            prompt_char_start_idx = prompt_text.find(chunk)
            promt_char_end_idx = prompt_char_start_idx + len(chunk)
            if prompt_char_start_idx == -1:
                raise ValueError(f"Recall span '{chunk}' not found in prompt text.")
            
            # the the indices of the tokens that cover this span
            # these tokens may cover more characters than the span itself, but that's ok
            token_start_idx = find_tuple_idx(offset_mapping, prompt_char_start_idx)
            token_end_idx = find_tuple_idx(offset_mapping, promt_char_end_idx - 1) + 1  # end is exclusive
            chunk_ids = prompt_ids[token_start_idx:token_end_idx]
        else:
            raise ValueError(f"Unknown chunk type: {chunk_type}")
        chunk_input_ids.append(chunk_ids)
    
    # concatenate all chunk ids to get final completion ids
    completion_ids = [chunk_id for chunk_ids in chunk_input_ids for chunk_id in chunk_ids]

    # create completion mask (1 for tokens to predict, 0 for tokens from prompt)
    completion_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    input_ids = prompt_ids + completion_ids

    return input_ids, completion_mask


def tags_to_tokens(
        text: str,
        recall_start_str: str = "<recall>",
        recall_end_str: str = "</recall>",
        recall_start_token: str = "<|start_recall|>",
        recall_end_token: str = "<|end_recall|>"
) -> str:
    """
    Replaces recall tags in the text with the corresponding special tokens.
    """
    text = text.replace(recall_start_str, recall_start_token)
    text = text.replace(recall_end_str, recall_end_token)
    return text

def tokenize_recallm_completions_dataset(
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        max_length: Optional[int] = None,
        recall_start_str: str = "<recall>",
        recall_end_str: str = "</recall>",
        convert_tags_to_tokens: bool = False,
        recall_start_token: str = "<|start_recall|>",
        recall_end_token: str = "<|end_recall|>",
        combine_system_prompt: bool = False,
        num_workers: int = 16
) -> Dataset:
    """
    Tokenizes a dataset of SFT completions, adding 'input_ids' and 'completion_mask' fields.
    If max_length is provided, then sequences longer than max_length are truncated from the right.
    If convert_tags_to_tokens is True, then the recall start and end strings are replaced with
     the corresponding special tokens before tokenization.
    """
    def tokenize_example(example):
        prompt_messages = example['prompt']
        completion_text = example['completion']

        # optionally convert recall tags to special tokens
        if convert_tags_to_tokens:
            for prompt_message in prompt_messages:
                prompt_message['content'] = prompt_message['content'].replace(recall_start_str, recall_start_token)
                prompt_message['content'] = prompt_message['content'].replace(recall_end_str, recall_end_token)
            completion_text = completion_text.replace(recall_start_str, recall_start_token)
            completion_text = completion_text.replace(recall_end_str, recall_end_token)
        
        if combine_system_prompt:
            # combine the system prompt into the user prompt
            if len(prompt_messages) > 1 and prompt_messages[0]['role'] == 'system':
                combined_content = prompt_messages[0]['content'] + "\n" + prompt_messages[1]['content']
                prompt_messages = [{'role': 'user', 'content': combined_content}] + prompt_messages[2:]
    
        recall_start = recall_start_str if not convert_tags_to_tokens else recall_start_token
        recall_end = recall_end_str if not convert_tags_to_tokens else recall_end_token

        input_ids, completion_mask = tokenize_sft_completion(
            tokenizer,
            prompt_messages,
            completion_text,
            recall_start_str=recall_start,
            recall_end_str=recall_end
        )

        # optionally truncate to max length
        if max_length is not None and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            completion_mask = completion_mask[:max_length]

        return {
            'input_ids': input_ids,
            'completion_mask': completion_mask
        }
    
    return dataset.map(tokenize_example, remove_columns=dataset.column_names, desc="Tokenizing SFT completions", num_proc=num_workers)