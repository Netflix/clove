from collections.abc import Callable

import math
from torch.optim import Optimizer

LrScheduler = Callable[[int], float]


def assign_learning_rate(optimizer: Optimizer, new_lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr: float, warmup_length: int, step: int) -> float:
    return base_lr * (step + 1) / warmup_length


def const_lr(optimizer: Optimizer, base_lr: float, warmup_length: int) -> LrScheduler:
    def _lr_adjuster(step: int) -> float:
        lr = _warmup_lr(base_lr, warmup_length, step) if step < warmup_length else base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def const_lr_cooldown(optimizer: Optimizer, base_lr: float, warmup_length: int, steps: int, cooldown_steps: int,
                      cooldown_power: float = 1.0, cooldown_end_lr: float = 0.) -> LrScheduler:
    def _lr_adjuster(step: int) -> float:
        start_cooldown_step = steps - cooldown_steps
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            if step < start_cooldown_step:
                lr = base_lr
            else:
                e = step - start_cooldown_step
                es = steps - start_cooldown_step
                # linear decay if power == 1; polynomial decay otherwise;
                decay = (1 - (e/es)) ** cooldown_power
                lr = decay * (base_lr - cooldown_end_lr) + cooldown_end_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def cosine_lr(optimizer: Optimizer, base_lr: float, warmup_length: int, steps: int) -> LrScheduler:
    def _lr_adjuster(step: int) -> float:
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + math.cos(math.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def lr_finder(optimizer: Optimizer, min_lr: float, max_lr: float, steps: int) -> LrScheduler:
    assert min_lr < max_lr

    factor = (max_lr / min_lr) ** (1 / steps)

    def _lr_adjuster(step: int) -> float:
        lr = min_lr * (factor ** step)
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster
