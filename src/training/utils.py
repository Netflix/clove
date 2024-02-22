import functools
import random
from collections.abc import Callable, Iterable, Mapping, MutableSequence, Sequence
from typing import Any, TypeVar

import psutil
import torch
from torch.nn.parallel import DistributedDataParallel

from open_clip import unwrap_model
from open_clip.model import ImageTextModel


def argmax_with_random_tiebreaks(x: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    # Inspired from https://stackoverflow.com/a/43103296/1165181
    max_ = x.max(dim=dim, keepdim=True)
    if not isinstance(max_, torch.Tensor):
        max_ = max_[0]  # When `dim=None` it's a tensor, but otherwise it's a tuple.
    # We consider the case that there are NaNs. In that case, the equality check will return False regardless. Then, we
    # want to tell apart the cases in which there's no max because of this. Thus, we want to sort all values, even when
    # they aren't the max. So we assign one plus a random value in [0, 1) when it's max, otherwise zero plus a random
    # value in [0, 1). If there's no max, a random non-max value is going to be selected.
    return ((x == max_) + torch.rand_like(x)).argmax(dim=dim, keepdim=keepdim)


def use_process_no_throw(func: Callable[[], int]) -> int:
    try:
        return func()
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return 0


T = TypeVar("T")


def sample_up_to_k(seq: Sequence[T], k: int) -> Sequence[T]:
    return random.sample(seq, k) if len(seq) > k else seq


# From https://stackoverflow.com/a/43649323/1165181
def weighted_random_sample_without_replacement(population: Sequence[T],
                                               weights: Iterable[float | int] | None = None,
                                               k: int = 1) -> Sequence[T]:
    weights = ((weights if isinstance(weights, MutableSequence) else list(weights)) if weights
               else ([1] * len(population)))
    positions = range(len(population))
    indices = []
    while needed := k - len(indices):
        for i in random.choices(positions, weights, k=needed):
            if weights[i]:
                weights[i] = 0
                indices.append(i)
    return [population[i] for i in indices]


def cache_generator_as_list(func: Callable[..., Iterable[T]]) -> Callable[..., Sequence[T]]:
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Sequence[T]:
        key = (args, frozenset(kwargs.items()))
        if key not in cache:
            cache[key] = list(func(*args, **kwargs))
        return cache[key]

    return wrapper


def non_overlapping_consecutive_pairs(seq: Sequence[T]) -> Iterable[tuple[T, T]]:
    return zip(seq[::2], seq[1::2])


def interleave_seqs(seq1: Sequence[T], seq2: Sequence[T]) -> Sequence[T]:
    assert len(seq1) == len(seq2)
    return [x for pair in zip(seq1, seq2) for x in pair]


def get_state_dict(checkpoint: Mapping[str, Any] | torch.jit.ScriptModule,
                   model: ImageTextModel | DistributedDataParallel | None = None,
                   is_distributed: bool = False) -> Mapping[str, Any]:
    model = None if model is None else unwrap_model(model)

    if isinstance(checkpoint, torch.jit.ScriptModule):
        state_dict = checkpoint.state_dict()
    elif "epoch" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not is_distributed and next(iter(state_dict.items()))[0].startswith("module."):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    # In cases where we're loading an older `CLIP` checkpoint with a longer context length.
    if (model is not None and "positional_embedding" in state_dict
            and model.positional_embedding.shape[0] < state_dict["positional_embedding"].shape[0]):
        state_dict["positional_embedding"] = \
            state_dict["positional_embedding"][:model.positional_embedding.shape[0]]

    return state_dict


def compute_wise_state_dict(state_dict1: Mapping[str, torch.Tensor], state_dict2: Mapping[str, torch.Tensor],
                            weight_for_2: float = 0.5) -> Mapping[str, torch.Tensor]:
    keys1 = frozenset(state_dict1.keys())
    keys2 = frozenset(state_dict2.keys())

    keys_in_1_not_2 = keys1 - keys2
    keys_in_2_not_1 = keys2 - keys1

    if keys_in_1_not_2 or keys_in_2_not_1:
        raise ValueError("State dicts must have the same keys."
                         f" Keys in the first one but not the second one:  {sorted(keys_in_1_not_2)}."
                         f" Keys in the second one but not the first one:  {sorted(keys_in_2_not_1)}.")

    return {k: (1 - weight_for_2) * state_dict1[k] + weight_for_2 * state_dict2[k] for k in state_dict1}


def patch_model(model: ImageTextModel | DistributedDataParallel, state_dict: Mapping[str, torch.Tensor],
                weight_for_state_dict: float = 0.5) -> None:
    """Patches the model with the given state dict by employing WiSE-FT (model patching)."""
    model = unwrap_model(model)
    model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    wise_ft_state_dict = compute_wise_state_dict(model_state_dict, state_dict, weight_for_2=weight_for_state_dict)
    model.load_state_dict(wise_ft_state_dict)
