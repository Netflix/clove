import ast
import collections.abc
import contextlib
import json
import logging
import os
import random
import re
import sys
import tarfile
import uuid
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, MutableMapping, MutableSequence, Sequence
from multiprocessing import Value
from typing import Any, BinaryIO, ContextManager, Generic, Literal, NoReturn, Protocol, TypeVar
from urllib.parse import urlparse

import braceexpand
import webdataset as wds
from cached_path import cached_path
from cached_path.common import PathOrStr
from torch.utils.data import IterableDataset, get_worker_info
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, valid_sample

from training.s3_utils import RefreshableBotoSession

T = TypeVar("T")


def _url_or_file_exist(path: PathOrStr) -> bool:
    try:
        cached_path(path)
    except FileNotFoundError:
        return False

    return True


def get_dataset_size(shards: str | Iterable[str]) -> tuple[int | None, int]:
    shard_list = expand_urls(shards)[0]
    dir_path = os.path.dirname(shard_list[0])

    if urlparse(dir_path).scheme in {"", "file"}:  # Only check if the dir exists if it's a file.
        if not os.path.exists(dir_path):
            raise ValueError(f"The dataset directory does not exist: {dir_path}")

    sizes_file_path = os.path.join(dir_path, "sizes.json")
    len_file_path = os.path.join(dir_path, "__len__")
    if _url_or_file_exist(sizes_file_path):
        with open(cached_path(sizes_file_path)) as file:
            sizes = json.load(file)
        total_size = sum(int(sizes[os.path.basename(shard)]) for shard in shard_list)
    elif _url_or_file_exist(len_file_path):
        # FIXME: this used to be `eval(file.read())` but that seemed rather unsafe.
        with open(cached_path(len_file_path)) as file:
            total_size = ast.literal_eval(file.read())
    else:
        total_size = None  # num samples undefined
        # Some common dataset sizes (at the time of authors' last download):
        # CC3M (train): 2_905_954
        # CC12M: 10_968_539
        # LAION-400M: 407_332_084
        # LAION-2B (english): 2_170_337_258
    num_shards = len(shard_list)
    return total_size, num_shards


def expand_urls(urls: str | Iterable[str],
                weights: str | Sequence[float] | None = None) -> tuple[Sequence[str], Sequence[float] | None]:
    if weights is None:
        return wds.shardlists.expand_urls(urls), None
    if isinstance(urls, str):
        url_list = urls.split("::")
        weights = weights.split("::")
        assert len(weights) == len(url_list), \
            f"Expected the number of data components ({len(url_list)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(url_list, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight] * len(expanded_url)
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        return list(urls), weights


def log_and_continue(exn: Exception) -> Literal[True]:
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def group_by_keys_no_throw(data: Iterator[Mapping[str, Any]], keys: Callable[[str], tuple[str, str]] = base_plus_ext,
                           lcase: bool = True,
                           suffixes: Collection[str] | None = None) -> Iterator[Mapping[str, str]]:
    """Returns an iterator that groups `(key, value)` pairs into samples."""
    current_sample = None
    for file_sample in data:
        assert isinstance(file_sample, dict)
        f_name, value = file_sample["fname"], file_sample["data"]
        prefix, suffix = keys(f_name)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME: the webdataset version of this function throws an exception if `suffix in current_sample`, but this
        #  could potentially happen for us in the current LAION400m dataset if a TAR file ends with same prefix as the
        #  next begins, which is rare, but it could happen since the prefixes aren't unique across the TAR files in this
        #  dataset.
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if current_sample is not None and valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=file_sample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tar_file_iterator_close_file(
        fileobj: tarfile.TarFile,
        skip_meta: str | None = r"__[^/]*__($|/)",
        handler: Callable[[Exception], bool] = wds.reraise_exception,
        select_files: Callable[[str], bool] | None = None,
        rename_files: Callable[[str], str] | None = None,
) -> Iterator[MutableMapping[str, Any]]:
    """Reimplementation of `tar_file_iterator` that closes the file explicitly."""
    try:
        with tarfile.open(fileobj=fileobj, mode="r|*") as stream:
            for tarinfo in stream:
                fname = tarinfo.name
                try:
                    if not tarinfo.isreg():
                        continue
                    if fname is None:
                        continue
                    if (
                            "/" not in fname
                            and fname.startswith(wds.tariterators.meta_prefix)
                            and fname.endswith(wds.tariterators.meta_suffix)
                    ):
                        # skipping metadata for now
                        continue
                    if skip_meta is not None and re.match(skip_meta, fname):
                        continue
                    if rename_files:
                        fname = rename_files(fname)
                    if select_files is not None and not select_files(fname):
                        continue
                    data = stream.extractfile(tarinfo).read()
                    result = dict(fname=fname, data=data)
                    yield result
                    stream.members = []
                except Exception as exn:
                    if hasattr(exn, "add_note"):
                        exn.add_note("@ " + str(fileobj))
                    elif hasattr(exn, "args") and len(exn.args) > 0 and isinstance(exn.args[0], str):
                        exn.args = (exn.args[0] + " @ " + str(fileobj),) + exn.args[1:]
                    elif hasattr(exn, "message"):
                        exn.message += " @ " + str(fileobj)
                    if handler(exn):
                        continue
                    else:
                        break
    finally:
        try:
            fileobj.close()
        except OSError:
            pass  # We don't care about errors when closing the file object, such as "Broken pipe" errors.


def tar_file_expander_close_file(
        data: Iterable[MutableMapping[str, Any]],
        handler: Callable[[Exception], bool] = wds.reraise_exception,
        select_files: Callable[[str], bool] | None = None,
        rename_files: Callable[[str], str] | None = None,
) -> Iterator[MutableMapping[str, Any]]:
    """Reimplementation of `tar_file_expander` that closes the files explicitly."""
    for source in data:
        url = source["url"]
        try:
            assert isinstance(source, dict)
            assert "stream" in source
            for sample in tar_file_iterator_close_file(
                    source["stream"],
                    handler=handler,
                    select_files=select_files,
                    rename_files=rename_files,
            ):
                assert isinstance(sample, dict) and "data" in sample and "fname" in sample
                sample["__url__"] = url
                yield sample
        except Exception as exn:
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                break


def tarfile_to_samples_no_throw_and_close_file(
        src: Iterator[T],
        handler: Callable[[Any], bool] | None = log_and_continue,
        select_files: Callable[[str], bool] | None = None,
        rename_files: Callable[[str], str] | None = None,
) -> Iterator[T]:
    """Reimplementation of `wds.tarfile_to_samples` with a `group_by_keys` that doesn't throw and closes the files."""
    streams = url_opener(src, handler=handler)
    files = tar_file_expander_close_file(streams, handler=handler, select_files=select_files, rename_files=rename_files)
    return group_by_keys_no_throw(files)


class SharedEpoch:
    def __init__(self, epoch: int = 0) -> None:
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch: int) -> None:
        self.shared_epoch.value = epoch

    def get_value(self) -> int:
        return self.shared_epoch.value


def pytorch_worker_seed(increment: int = 0) -> int:
    """Get the data loader worker seed from PyTorch."""
    worker_info = get_worker_info()
    if worker_info:
        # We favor using the seed already created for pytorch dataloader workers if it exists.
        seed = worker_info.seed
        if increment:
            # Space out the seed increments, so they can't overlap across workers in different iterations.
            seed += increment * max(1, worker_info.num_workers)
        return seed

    return wds.utils.pytorch_worker_seed()  # Fallback to wds rank-based seed.


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of URLs."""

    def __init__(self, urls: str | Iterable[str], weights: str | Sequence[float] | None = None,
                 nshards: int = sys.maxsize, worker_seed: Callable[[], int] | None = None,
                 deterministic: bool = False, epoch: int | SharedEpoch = -1) -> None:
        """Sample shards from the shard list with replacement.

        :param urls: A list of URLs as a Python list or brace notation string.
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights), \
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self) -> Iterator[Mapping[str, str]]:
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this epoch tracking is problematic in a multiprocess (dataloader workers or train) situation as
            # different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch

        if self.deterministic:  # Reset seed w/ epoch if deterministic.
            # The PyTorch worker seed should be deterministic due to being init by `arg.seed + rank + worker id`.
            self.rng.seed(pytorch_worker_seed(epoch) if self.worker_seed is None else (self.worker_seed() + epoch))

        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


class detshuffle2(wds.PipelineStage):
    def __init__(self, bufsize: int = 1000, initial: int = 100, seed: int = 0,
                 epoch: int | SharedEpoch = -1) -> None:
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src: Iterator[T]) -> Iterator[T]:
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch

        # If the seed is non-negative, it's deterministic and the same across all nodes and workers in each epoch.
        # Else, we use the worker's seed, which will be different across all nodes/workers.
        seed = (self.seed + epoch) if self.seed >= 0 else pytorch_worker_seed(epoch)

        return _shuffle(src, self.bufsize, self.initial, rng=random.Random(seed))


def _shuffle_all(x: Iterator[T], rng: random.Random) -> MutableSequence[T]:
    # Only use this function is you can afford (in time and space) consuming all the items at once.
    if not isinstance(x, collections.abc.MutableSequence):
        x = list(x)
    rng.shuffle(x)
    return x


shuffle_all = wds.pipelinefilter(_shuffle_all)

# Without a refreshable session, we'd eventually get an error with the credentials.
S3 = RefreshableBotoSession().refreshable_session().resource("s3")


def gopen_s3(url: str, mode: str = "rb", bufsize: int = 8192) -> BinaryIO:  # noqa
    if mode[0] == "r":
        parsed_path = urlparse(url, allow_fragments=False)
        bucket = parsed_path.netloc
        key = parsed_path.path.lstrip("/")
        return S3.Object(bucket, key).get()["Body"]
    elif mode[0] == "w":
        raise NotImplementedError("Writing to S3 is not implemented.")
    else:
        raise ValueError(f"{mode}: unknown mode")


wds.gopen_schemes["s3"] = gopen_s3


class ResumableRepeatedIterable(Generic[T]):
    """
    Iterable whose iterator resumes the last iteration.
    It's recreated on exhaustion.

    If the inner iterator is empty at some point, there's a chance that it's a bug and also that future calls to
    the iterable function are also empty.
    If this happens, if we don't check for this, we'll get stuck in an infinite loop, never returning anything.
    Thus, by default, we stop if the inner iterator is empty at any point,
    and the flag `stop_if_empty` can be used to change this behavior.
    """

    def __init__(self, iterable_fn: Callable[[], Iterable[T]], stop_if_empty: bool = True) -> None:
        self.iterable_fn = iterable_fn
        self.stop_if_empty = stop_if_empty

        self._iterable = None
        self._is_empty = False

    def __iter__(self) -> Iterator[T]:
        if self._iterable is None:
            self._iterable = self.iterable_fn()

        while not self.stop_if_empty or not self._is_empty:
            self._is_empty = True
            for item in self._iterable:
                self._is_empty = False
                yield item
            self._iterable = self.iterable_fn()


class ResumableDataPipeline(wds.DataPipeline):
    """A version of the data pipeline that can be resumed from the last sample it yielded.
    It's useful for continuing where you left off after `self.nsamples` were iterated.
    Once the inner iterator is exhausted, it's recreated to not have an epoch that is shorter than `self.nsamples`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._iter = ResumableRepeatedIterable(super().iterator1)

    def iterator1(self) -> Iterator:
        assert self.nsamples > 0, ("`ResumableDataPipeline` only makes sense when requires `nsamples` to be set"
                                   " (typically with `with_epoch` or `repeat`).")
        yield from self._iter


@contextlib.contextmanager
def datasets_spawn_mode() -> ContextManager[None]:
    """The following is a hack to change the default multiprocessing context used in `map` to spawn one.

    See https://github.com/huggingface/datasets/issues/6357 for context.
    """
    from datasets import arrow_dataset
    Pool_ = arrow_dataset.Pool  # noqa
    try:
        import multiprocess
        arrow_dataset.Pool = multiprocess.get_context("spawn").Pool  # noqa

        yield
    finally:
        arrow_dataset.Pool = Pool_


def PicklableGeneratorRaise(*args, **kwargs) -> NoReturn:  # noqa
    raise AssertionError("Failed to unpickle.")


class GetIntItem(Protocol[T]):
    def __getitem__(self: "GetIntItem", key: int) -> T:
        pass


class PicklableGenerator:
    def __init__(self, generator: Callable[..., Iterable[T] | GetIntItem[T]],
                 generator_id: str | None = None) -> None:
        # Also support objects that implement `__getitem__` because `iter` does, even when they don't have `__iter__`.
        # See https://stackoverflow.com/a/926645/1165181
        self.generator = generator
        self.generator_id = str(uuid.uuid4()) if generator_id is None else generator_id

    def __call__(self, *args, **kwargs) -> Iterable[T] | GetIntItem[T]:
        return self.generator(*kwargs, **kwargs)

    def __reduce__(self) -> tuple[Callable, tuple[str]]:
        return PicklableGeneratorRaise, (self.generator_id,)
