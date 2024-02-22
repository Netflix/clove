import re
from collections.abc import Callable, Container, Iterable
from itertools import repeat
from typing import Any, TypeVar

import nltk
import torch
import unicodedata
from filelock import FileLock
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torchvision.ops.misc import FrozenBatchNorm2d


def freeze_batch_norm_2d(module: nn.Module, module_match: Container[str] | None = None,
                         name: str = "") -> nn.Module:
    """Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`.
    If `module` is itself an instance of either `BatchNorm2d` or `SyncBatchNorm`,
    it is converted into `FrozenBatchNorm2d` and returned.
    Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module: Any PyTorch module.
        module_match: Dictionary of full module names to freeze (all if empty)
        name: Full module name (prefix)

    Returns: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f8/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    is_match = not module_match or name in module_match
    if is_match and isinstance(module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = ".".join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


T = TypeVar("T")


# From PyTorch internals
def _ntuple(n: int) -> Callable[[T | Iterable[T]], tuple[T, ...] | Iterable[T]]:
    def parse(x: T | Iterable[T]) -> tuple[T, ...] | Iterable[T]:
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)


# TODO: add int8 support for other linear layers including attn and convnets.
def replace_linear(module: nn.Module, linear_replacement: Callable[[int, int, bool], nn.Module],
                   include_modules: Container[str] = frozenset({"c_fc", "c_proj"}),
                   copy_weights: bool = True) -> nn.Module:
    for name, module in module.named_children():
        replace_linear(module, linear_replacement, include_modules, copy_weights)

        if isinstance(module, torch.nn.Linear) and name in include_modules:
            old_module = module._modules[name]
            module._modules[name] = linear_replacement(module.in_features, module.out_features, module.bias is not None)
            if copy_weights:
                module._modules[name].weight.data.copy_(old_module.weight.data)
                if module._modules[name].bias is not None:
                    module._modules[name].bias.data.copy_(old_module.bias)

    return module


def convert_int8_model_to_inference_mode(module: nn.Module) -> None:
    for m in module.modules():
        if hasattr(m, "prepare_for_eval"):
            int8_original_dtype = m.weight.dtype
            m.prepare_for_eval()
            m.int8_original_dtype = int8_original_dtype


M = TypeVar("M", bound=nn.Module)


def unwrap_model(model: M | DistributedDataParallel) -> M:
    return getattr(model, "module", model)


# See https://stackoverflow.com/a/295466/1165181
def slugify(value: Any, allow_unicode: bool = False) -> str:
    """Taken from https://github.com/django/django/blob/main/django/utils/text.py

    Convert to ASCII if 'allow_unicode' is False.
    Convert spaces or repeated dashes to single dashes.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase.
    Also, strip leading and trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def maybe_nltk_download(id_: str, path: str) -> None:
    try:
        nltk.data.find(path)
    except (LookupError, OSError):
        with FileLock(slugify(id_) + ".lock"):
            nltk.download(id_)
