from collections.abc import Callable
from contextlib import AbstractContextManager, suppress
from typing import Literal, get_args

import torch

Autocast = Callable[[], AbstractContextManager]
Precision = Literal["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"]

PRECISION_OPTIONS = get_args(Precision)


def get_autocast(precision: Precision) -> Autocast:
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress
