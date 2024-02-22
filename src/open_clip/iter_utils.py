from collections.abc import Iterable, Sequence
from itertools import islice
from typing import TypeVar

T = TypeVar("T")


def batched(iterable: Iterable[T], n: int | None) -> Iterable[Sequence[T]]:
    """Batch data into lists of length *n*. The last batch may be shorter.

    It's based on the more-itertools impl, to be replaced by Python 3.12 `itertools.batched` impl.
    """
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


# See https://stackoverflow.com/a/31062966/1165181
def mean(it: Iterable[T]) -> T | float | int:
    n = 0
    mean = 0

    for x in it:
        n += 1
        mean += (x - mean) / n

    return float("nan") if n == 0 else mean


def is_empty(it: Iterable) -> bool:
    return all(False for _ in it)
