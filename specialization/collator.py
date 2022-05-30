import transformers
import torch
from transformers import PreTrainedTokenizer, BatchEncoding, PreTrainedTokenizerBase
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Callable, Dict, List, NewType, Tuple, Union
import random
import numpy as np
from dataclasses import dataclass, field

@dataclass
class DataCollatorForSeq2SeqMaskLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15

    def __call__(self, examples: List[Union[torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        batch = self._tensorize_batch(examples)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "labels": labels}
        else:
            labels = batch.clone().detach()
            labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": batch, "labels": labels}

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def _check_sentinel(self, tokens, mask, max_sentinels, sentinel_id):
        sentineled_toks = tokens.clone()
        prev_tok_noise = torch.nn.functional.pad(mask[:-1], [1, 0])

        first_noise_toks = torch.logical_and(mask, ~prev_tok_noise)
        subse_noise_toks = torch.logical_and(mask, prev_tok_noise)
        if first_noise_toks.sum().item()<100:
            return True
        else:
            return False
        
    def _noise_span_to_unique_sentinel(self, tokens, mask, max_sentinels, sentinel_id):
        sentineled_toks = tokens.clone()
        prev_tok_noise = torch.nn.functional.pad(mask[:-1], [1, 0])

        first_noise_toks = torch.logical_and(mask, ~prev_tok_noise)
        subse_noise_toks = torch.logical_and(mask, prev_tok_noise)
        
        sentinels = torch.arange(start = sentinel_id, end = sentinel_id - max_sentinels, step = -1)
        sentineled_toks[first_noise_toks] = sentinels[:first_noise_toks.sum().item()]
        return sentineled_toks[~subse_noise_toks]

    def mask_tokens(self, inputs: torch.Tensor, mlm_probability = 0.15, min_span_length = 1, max_span_length = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        # print(inputs)
        device = inputs.device
        inpts = inputs.clone()
        span_lengths = torch.randint(low = min_span_length, high = max_span_length + 1, size = (inpts.shape[0],), device = device)
        periods = torch.round(span_lengths / mlm_probability)
        offsets = torch.tensor([random.randint(0, period.item()) for period in periods], device = device)
        masks = torch.stack([(torch.arange(start = 0, end = inpts.shape[1]) + offset) % period < span for offset, period, span in zip(offsets, periods, span_lengths)])

        if self.tokenizer._pad_token is not None:
            padding_mask = inpts.eq(self.tokenizer.pad_token_id)
            masks.masked_fill_(padding_mask, value = False)
        num_masks = torch.floor_divide(masks.sum(axis = 1), span_lengths)
        new_inpts = []
        lbls = []
        for inpt, mask in zip(inpts, masks):
            if self._check_sentinel(inpt, mask, 100, self.tokenizer.convert_tokens_to_ids(['<extra_id_0>'])[0]):
                new_inpts.append(
                    self._noise_span_to_unique_sentinel(inpt, mask, 100, self.tokenizer.convert_tokens_to_ids(['<extra_id_0>'])[0])
                )
                lbls.append(
                    self._noise_span_to_unique_sentinel(inpt, ~mask, 100, self.tokenizer.convert_tokens_to_ids(['<extra_id_0>'])[0])
                )

        new_inpts = pad_sequence(new_inpts, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        lbls = pad_sequence(lbls, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        return new_inpts, lbls

def shift_tokens_right(input_ids: np.array, pad_token_id: int, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    #shifted_input_ids = pad_sequence(shifted_input_ids, batch_first=True, padding_value=pad_token_id)
    shifted_input_ids = torch.from_numpy(shifted_input_ids)
    return shifted_input_ids

@dataclass
class DataCollatorForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: PreTrainedTokenizerBase
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int
    decoder_start_token_id: int

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        #print(examples)
        # convert list to dict and tensorize input
        batch = BatchEncoding(
            {k: np.array([examples[i][k].detach().numpy() for i in range(len(examples))]) for k, v in examples[0].items()}
        )
        #print(batch["input_ids"])
        #batch["input_ids"] = torch.from_numpy(batch["input_ids"])
        input_ids = batch["input_ids"]
        #print(input_ids)
        #batch_size = input_ids.shape[0]
        #expanded_inputs_length = len(input_ids[0])
        #print(len(input_ids[0]), len(input_ids[1]), len(input_ids[2]), len(input_ids[3]), len(input_ids[4]))
        batch_size, expanded_inputs_length = input_ids.shape
        #print(expanded_inputs_length, self.input_length)
        mask_indices = np.asarray([self.random_spans_noise_mask(expanded_inputs_length) for i in range(batch_size)])
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)
        #print(len(batch["input_ids"][0]), len(batch["input_ids"][1]), len(batch["input_ids"][2]), len(batch["input_ids"][3]), len(batch["input_ids"][4]))
        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but should be {self.target_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be {self.target_length}."
            )

        # to check that tokens are correctly preprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.pad_token_id, self.decoder_start_token_id
        )
        #batch["attention_mask"] = torch.from_numpy(batch["attention_mask"])
       
        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )
        #input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        input_ids = torch.from_numpy(input_ids)
        return input_ids

    def random_spans_noise_mask(self, length):

        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number
        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]