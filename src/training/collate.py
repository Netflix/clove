# Adapted from https://github.com/bryant1410/fitclip/blob/a55ec2b/aligner/data/tokenizer_collate.py
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import TextInput


# Derived from `default_collate`.
def pad_collate(batch: Sequence) -> Any:
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return pad_sequence(batch, batch_first=True)  # noqa
    elif isinstance(elem, Mapping):
        return {k: pad_collate([d[k] for d in batch]) for k in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # NamedTuple
        return elem_type(*(pad_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(e) == elem_size for e in it):
            raise RuntimeError("Each element in sequence of batch should be of equal size.")
        transposed = zip(*batch)
        return [pad_collate(samples) for samples in transposed]
    else:
        return default_collate(batch)


# Derived from `default_collate`.
def batch_tokenize_collate(batch: Sequence, tokenizer: PreTrainedTokenizerBase) -> Any:
    """`DataLoader` collate function that batch-tokenizes the batch."""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, TextInput):
        return tokenizer(batch)  # noqa
    elif isinstance(elem, Mapping):
        return {k: batch_tokenize_collate([d[k] for d in batch], tokenizer) for k in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # NamedTuple
        return elem_type(*(batch_tokenize_collate(samples, tokenizer) for samples in zip(*batch)))
    elif isinstance(elem, Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(e) == elem_size for e in it):
            raise RuntimeError("Each element in sequence of batch should be of equal size.")
        transposed = zip(*batch)
        return [batch_tokenize_collate(samples, tokenizer) for samples in transposed]
    else:
        raise TypeError(f"Batch must contain strings, mappings or sequences; found {elem_type}.")


class DataCollatorWithTokenization(ABC):
    """`DataLoader` collate function (callable) that batch-tokenizes part of the batch.

    The pros of batch-tokenizing during collation are:
    1) We can pad at the same time, based on the longest sequence. If we tokenized in the dataset, we wouldn't know
    what size to take, and we may take a long one, wasting computing and especially memory. If we batch-tokenize when
    iterating through the data_module loader, we are in the main thread and wasting valuable time that could be used for
    the GPU.
    2) The `tokenizers` library is written in Rust and may have some optimizations for batch-tokenizing (apart from
    multi-threading, which is disabled so each data loader worker uses one CPU core.)
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase | Mapping[str, PreTrainedTokenizerBase], *,
                 batch_tokenize_collate_fn: Callable[[Sequence, PreTrainedTokenizerBase], Any] = batch_tokenize_collate,
                 default_collate_fn: Callable[[Sequence], Any] = default_collate) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_tokenize_collate_fn = batch_tokenize_collate_fn
        self.default_collate_fn = default_collate_fn

    @abstractmethod
    def _split_uncollated_batch(self, batch: Sequence) -> tuple[Sequence, Sequence]:
        """Splits the batch into a pair where the first element is going to be processed with the default collate
        function, and each of the elements in the second one is going to be batch-tokenized.
        """
        raise NotImplementedError

    @abstractmethod
    def _join_collated_batch(self, collated_with_default: Any, collated_with_tokenizer: Any) -> Any:
        raise NotImplementedError

    def __call__(self, batch: Sequence) -> Any:
        s1, s2 = self._split_uncollated_batch(batch)
        return self._join_collated_batch(self.default_collate_fn(s1),
                                         # FIXME: a tokenizer mapping isn't working here:
                                         self.batch_tokenize_collate_fn(s2, self.tokenizer))


class MappingDataCollatorWithTokenization(DataCollatorWithTokenization):
    def __init__(self, tokenizer: PreTrainedTokenizerBase | Mapping[str, PreTrainedTokenizerBase],
                 keys_to_tokenize: str | Iterable[str], **kwargs) -> None:
        super().__init__(tokenizer, **kwargs)
        self.keys_to_tokenize = frozenset({keys_to_tokenize} if isinstance(keys_to_tokenize, str) else keys_to_tokenize)

    def _split_uncollated_batch(self, batch: Sequence[Mapping[str, Any]]) -> tuple[Sequence, Sequence]:
        return [{k: v for k, v in d.items() if k not in self.keys_to_tokenize} for d in batch], \
            [{k: v for k, v in d.items() if k in self.keys_to_tokenize} for d in batch]

    def _join_collated_batch(self, collated_with_default: Any, collated_with_tokenizer: Any) -> Any:
        # If the tokenizer is actually composed of many tokenizers, we flatten out the structure.
        if isinstance(self.tokenizer, Mapping):
            collated_with_tokenizer = {f"{k_child}_{k_parent}": v_child
                                       for k_parent, v_parent in collated_with_tokenizer.items()
                                       for k_child, v_child in v_parent.items()}

        return {**collated_with_default, **collated_with_tokenizer}
