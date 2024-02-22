import functools
import html
import os
import random
import string
from collections.abc import Callable, Iterable, Sequence
from typing import Literal

import ftfy
import nltk
import numpy as np
import regex as re
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase, TensorType
from transformers.models.clip.tokenization_clip import whitespace_clean
from transformers.tokenization_utils_base import TextInput
from transformers.utils import PaddingStrategy
from typing_extensions import Self

from open_clip.utils import maybe_nltk_download

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # See https://stackoverflow.com/q/62691279

# A default short max length is nice because text captions are short, and it's memory- and compute-efficient.
# CLIP's default max context length is 77, but we try to use multiples of 8 to use Tensor Cores with amp.
# Even better, a power of 2; and we lose little context.
DEFAULT_CONTEXT_LENGTH = 64


# The "!" is a hack to set the padding token ID to 0, which is the default for CLIP.
CLIP_TOKENIZER_TO_WRAP = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", pad_token="!",
                                                       model_max_length=DEFAULT_CONTEXT_LENGTH)

maybe_nltk_download("punkt", "tokenizers/punkt")
maybe_nltk_download("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger")


def _basic_clean(text: str) -> str:
    return html.unescape(html.unescape(ftfy.fix_text(text))).strip()


def _clean_canonicalize(x: str) -> str:
    return _canonicalize_text(_basic_clean(x))  # basic, remove whitespace, remove punctuation, lower case


def _clean_lower(x: str) -> str:
    return whitespace_clean(_basic_clean(x)).lower()  # basic, remove whitespace, lower case


def _clean_whitespace(x: str) -> str:
    return whitespace_clean(_basic_clean(x))  # basic, remove whitespace


CleanFunctionType = Literal["canonicalize", "lower", "whitespace"]


def _get_clean_fn(type_: CleanFunctionType) -> Callable[[str], str]:
    if type_ == "canonicalize":
        return _clean_canonicalize
    elif type_ == "lower":
        return _clean_lower
    elif type_ == "whitespace":
        return _clean_whitespace
    else:
        raise ValueError(f"Invalid clean function ({type_}).")


def _canonicalize_text(text: str, *, keep_punctuation_exact_string: str | None = None) -> str:
    """Returns canonicalized `text` (lowercase and punctuation removed).
    From:
    https://github.com/google-research/big_vision/blob/53f18ca/big_vision/evaluators/proj/image_text/prompt_engineering.py#L94

    Args:
      text: string to be canonicalized.
      keep_punctuation_exact_string: If provided, then this exact string kept.
        For example, providing "{}" will keep any occurrences of "{}" (but will
        still remove "{" and "}" that appear separately).
    """
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(part.translate(str.maketrans("", "", string.punctuation))
                                                  for part in text.split(keep_punctuation_exact_string))
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _simple_mask_tokenize(texts: TextInput | Iterable[TextInput],
                          tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    context_length = tokenizer.model_max_length

    all_tokens = tokenizer(texts, add_special_tokens=False).input_ids

    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        num_tokens = len(tokens)
        if num_tokens > context_length - 2:  # 2 for the SoT and EoT tokens.
            num_keep = context_length - 2
            start_index = random.randint(0, num_tokens - num_keep)  # High is incl
            tokens = tokens[start_index: start_index + num_keep]
        tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def _random_mask_tokenize(texts: TextInput | Iterable[TextInput], tokenizer: PreTrainedTokenizerBase,
                          shuffle: bool = False) -> torch.Tensor:
    context_length = tokenizer.model_max_length

    all_tokens = tokenizer(texts, add_special_tokens=False).input_ids

    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        tokens = torch.tensor(tokens)
        num_tokens = len(tokens)
        if num_tokens > context_length - 2:  # 2 for the SoT and EoT tokens.
            num_keep = context_length - 2
            indices = torch.randperm(len(tokens))
            indices = indices[:num_keep]
            if not shuffle:
                indices = indices.msort()
            tokens = tokens[indices]
            num_tokens = num_keep
        result[i, 0] = tokenizer.bos_token_id
        result[i, 1:num_tokens + 1] = tokens
        result[i, num_tokens + 1] = tokenizer.eos_token_id

    return result


def _syntax_mask_tokenize(texts: TextInput | Iterable[TextInput],
                          tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    """Returns the tokenized representation of given input string(s). Apply syntax masking before tokenize."""
    def _get_order(x: str) -> int:
        if x.startswith("NN"):
            return 1
        elif x.startswith("JJ"):
            return 2
        elif x.startswith("VB"):
            return 3
        else:
            return 4

    context_length = tokenizer.model_max_length

    # Syntax masking.
    new_texts = []
    for text in texts:
        list_tokens = nltk.tokenize.word_tokenize(text)
        pos_tags = nltk.pos_tag(list_tokens)
        # Sample the words by the `_get_order` method.
        order_list = np.array([_get_order(tag) for _, tag in pos_tags])
        order_list.partition(context_length - 2)  # Need 2 slots for the SoT and EoT tokens.
        sampled_ids = order_list[:context_length - 2]
        sampled_ids.sort()
        sampled_tokens = [list_tokens[i] for i in sampled_ids]  # Sample the tokens.

        new_text = ""
        for token in sampled_tokens:
            new_text += str(token) + " "
        new_text = new_text.strip()
        new_texts.append(new_text)
    texts = new_texts

    all_tokens = tokenizer(texts).input_ids

    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        # Still need to first truncate because some words produce two tokens.
        if len(tokens) > context_length:
            tokens = tokens[:context_length]  # Truncate
            tokens[-1] = tokenizer.eos_token_id
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


ReductionMaskType = Literal["simple", "random", "shuffle", "syntax"]


def _get_reduction_mask_fn(
        type_: ReductionMaskType,
) -> Callable[[TextInput | Iterable[TextInput], PreTrainedTokenizerBase], torch.Tensor]:
    """Choose strategy for dropping (masking) tokens to achieve a target context length."""
    if type_ == "simple":
        return _simple_mask_tokenize  # randomly select block [start:end]
    elif type_ == "random":
        return _random_mask_tokenize  # randomly drop tokens (keep order)
    elif type_ == "shuffle":
        return functools.partial(_random_mask_tokenize, shuffle=True)  # noqa. randomly drop tokens (shuffle order)
    elif type_ == "syntax":
        return _syntax_mask_tokenize  # randomly drop prioritized by syntax
    else:
        raise ValueError(f"Invalid reduction mask type: {type_}")


class HFTokenizer(PreTrainedTokenizerBase):  # We subclass for type hints.
    """A wrapper around a HuggingFace's tokenizer.

    When creating an instance of this class,
    it actually creates an instance of a dynamically created class
    (different from the former) that wraps the input tokenizer and overrides some behavior.
    So the instance is going to have all the attributes of the wrapped tokenizer
    and can be a drop-in replacement for it.
    """

    # For type hints. Should be defined by super, but for some reason it's not, even though it's always present.
    vocab_size: int

    def __new__(cls, tokenizer: PreTrainedTokenizerBase, prefix: str = "", clean: CleanFunctionType = "whitespace",
                reduction_mask: ReductionMaskType | None = None, strip_sep_token: bool = False,
                default_padding: bool | str | PaddingStrategy = "longest") -> Self:  # noqa
        # We create an instance that has the same attributes as the wrapped tokenizer (including all values and
        # functions), but we add/override methods.
        current_attrs = {k: v for k, v in cls.__dict__.items() if k != "__new__"}
        # We need to set `__init__` to an empty one to create the instance of this dynamically created class.
        new_attrs = {**tokenizer.__dict__, **current_attrs, "__init__": lambda self: None}
        class_ = type(cls.__name__, (tokenizer.__class__,), new_attrs)
        instance = class_()

        instance.prefix = prefix
        instance.clean_fn = _get_clean_fn(clean)
        instance.strip_sep_token = strip_sep_token

        instance.reduction_fn = _get_reduction_mask_fn(reduction_mask) if reduction_mask else None
        instance.default_truncation = reduction_mask is None

        instance.default_padding = default_padding

        return instance

    # Note that in `__call__` some args are defined because their default values are different from those in super.
    def __call__(self, texts: TextInput | Iterable[TextInput], use_prefix: bool = True,  # noqa
                 pad_to_multiple_of: int | None = 8, return_tensors: str | TensorType | None = "pt",
                 **kwargs) -> Sequence[int] | torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        texts = [self.clean_fn(text) for text in texts]

        texts = [self.prefix + text for text in texts] if use_prefix else (texts if isinstance(texts, list)
                                                                           else list(texts))

        if self.reduction_fn:
            assert return_tensors == "pt"

            truncation = kwargs.get("truncation", False)
            assert not truncation or truncation == "do_not_truncate"

            # Pass super so we don't enter into a recursion error.
            input_ids = self.reduction_fn(texts, super(self.__class__, self))
        else:
            # Need to pass args to `super` because the defined class (`class HFTokenizer`) is different from
            # `self.__class__` (even though they have the same name),
            # because it was changed dynamically in `__new__`.
            input_ids = super(self.__class__, self).__call__(
                texts, padding=kwargs.get("padding", self.default_padding),
                truncation=kwargs.get("truncation", self.default_truncation), pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors, **kwargs).input_ids

        if self.strip_sep_token:
            assert return_tensors == "pt"
            input_ids = torch.where(input_ids == self.sep_token_id, torch.zeros_like(input_ids), input_ids)

        return input_ids
