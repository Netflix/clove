import argparse
import functools
import hashlib
import logging
import os
import random
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence, Sized
from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

import datasets
import math
import pandas as pd
import torch
import torchvision.datasets
import webdataset as wds
from PIL import Image, ImageColor
from clip_benchmark.datasets.builder import build_dataset, get_dataset_default_task
from filelock import FileLock
from huggingface_hub import HfFileSystem
from imagenetv2_pytorch import ImageNetV2Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose

from open_clip import HFTokenizer, Transform
from open_clip.tokenizer import DEFAULT_CONTEXT_LENGTH
from training import UniformFrameSampler
from training.aro_datasets import AroCocoOrder, AroFlickr30kOrder, AroVgAttribution, AroVgRelation
from training.collate import MappingDataCollatorWithTokenization
from training.data_utils import ResampledShards2, ResumableDataPipeline, SharedEpoch, detshuffle2, \
    expand_urls, get_dataset_size, log_and_continue, shuffle_all, tarfile_to_samples_no_throw_and_close_file
from training.s3_utils import s3_sync
from training.text_negatives_data import add_random_text_hard_negatives
from training.utils import interleave_seqs
from training.video_datasets import Didemo, Hmdb, MsrVtt, Ucf, YouCook2
from training.zero_shot_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES

try:
    import horovod.torch as hvd
except ImportError:
    hvd = Any if TYPE_CHECKING else None


class CsvDataset(Dataset):
    def __init__(self, input_filename: str, transform: Transform, tokenizer: HFTokenizer, img_key: str,
                 caption_key: str, sep: str = "\t") -> None:
        logging.debug(f"Loading csv data from {input_filename}.")
        df = pd.read_csv(input_filename, sep=sep)
        logging.debug("Done loading data.")

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(Image.open(str(self.images[i]))), self.tokenizer([str(self.captions[i])])[0]


@dataclass
class DataInfo:
    data_loader: DataLoader | wds.WebLoader
    shared_epoch: SharedEpoch | None = None

    def set_epoch(self, epoch: int) -> None:
        if self.shared_epoch:
            self.shared_epoch.set_value(epoch)
        if getattr(self.data_loader, "sampler", None) and isinstance(self.data_loader.sampler, DistributedSampler):
            self.data_loader.sampler.set_epoch(epoch)


_DatasetGetter = Callable[[argparse.Namespace, Transform, HFTokenizer], Any]

_DATASET_GETTER_REGISTRY: MutableMapping[str, _DatasetGetter] = {}


def _eval_dataset(name: str) -> Callable[[_DatasetGetter], _DatasetGetter]:
    def _wrapper(fn: _DatasetGetter) -> _DatasetGetter:
        if name in _DATASET_GETTER_REGISTRY:
            raise ValueError(f"A dataset getter for {name} is already registered")
        _DATASET_GETTER_REGISTRY[name] = fn
        return fn

    return _wrapper


def _get_clip_benchmark_dataset(name: str, args: argparse.Namespace, transform: Transform, *_, **__) -> Any:
    dataset_name_for_url = name.removeprefix("wds/").replace('/', '-')
    dataset = build_dataset(
        dataset_name=name, task=get_dataset_default_task(name), transform=transform,
        root=f"https://huggingface.co/datasets/clip-benchmark/wds_{dataset_name_for_url}/tree/main")
    return _create_data_loader(dataset=dataset, args=args)


def _get_data_one(name: str, args: argparse.Namespace, transform: Transform, tokenizer: HFTokenizer) -> Any:
    if name.startswith("cb/"):
        return _get_clip_benchmark_dataset(name.removeprefix("cb/"), args, transform, tokenizer)
    else:
        return _DATASET_GETTER_REGISTRY[name](args, transform, tokenizer)


def get_registered_datasets() -> Iterable[str]:
    return _DATASET_GETTER_REGISTRY.keys()


@_eval_dataset("imagenet-val")
def get_imagenet_val(args: argparse.Namespace, transform: Transform, *_, **__) -> DataLoader:
    if args.imagenet_val.startswith("s3://"):
        new_path = "/mnt/imagenet/ILSVRC2012_img_val/"

        if not os.path.exists(new_path):
            with FileLock(new_path.rstrip("/") + ".lock"):
                if not os.path.exists(new_path):  # Retry in case the process had been locked.
                    s3_sync(args.imagenet_val.rstrip("/") + "/", new_path.rstrip("/") + "/")

        args.imagenet_val = new_path

    dataset = torchvision.datasets.ImageFolder(args.imagenet_val, transform=transform)

    dataset.name = "ImageNet"
    dataset.classes = IMAGENET_CLASSNAMES
    dataset.templates = OPENAI_IMAGENET_TEMPLATES

    return _create_data_loader(dataset=dataset, args=args)


@_eval_dataset("imagenet-v2")
def get_imagenet_v2(args: argparse.Namespace, transform: Transform, *_, **__) -> DataLoader:
    dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=transform)

    dataset.name = "ImageNet v2"
    dataset.classes = IMAGENET_CLASSNAMES
    dataset.templates = OPENAI_IMAGENET_TEMPLATES

    return _create_data_loader(dataset=dataset, args=args)


@_eval_dataset("winoground")
def get_winoground(args: argparse.Namespace, transform: Transform, tokenizer: HFTokenizer) -> DataLoader:
    image_shape = transform(Image.new(mode="RGB", size=(1, 1), color=ImageColor.getrgb("red"))).shape

    dataset = (datasets.load_dataset("facebook/winoground", split="test", token=True, trust_remote_code=True)
               .map(lambda sample: {k: transform(v).numpy() for k, v in sample.items() if k.startswith("image_")},
                    remove_columns=["id", "tag", "secondary_tag", "num_main_preds", "collapsed_tag"],
                    features=datasets.Features({"image_0": datasets.Array3D(shape=image_shape, dtype="float32"),
                                                "image_1": datasets.Array3D(shape=image_shape, dtype="float32"),
                                                "caption_0": datasets.Value("string"),
                                                "caption_1": datasets.Value("string")}),
                    desc="Preprocessing Winoground")
               .with_format("torch"))

    collator = MappingDataCollatorWithTokenization(tokenizer, keys_to_tokenize=("caption_0", "caption_1"))
    return _create_data_loader(dataset=dataset, args=args, collate_fn=collator)


def _concatenate_tokenized_texts(batch: Mapping[str, Any], key: str, padding_value: int = 0) -> Mapping[str, Any]:
    return {**batch, key: pad_sequence((c.T for c in batch[key]), padding_value=padding_value).transpose(0, 2)}  # noqa


@_eval_dataset("aro")
def get_aro(args: argparse.Namespace, transform: Transform, tokenizer: HFTokenizer) -> Mapping[str, DataLoader]:
    tokenizer_collator = MappingDataCollatorWithTokenization(tokenizer, keys_to_tokenize="caption_options")
    collator = lambda *args, **kwargs: _concatenate_tokenized_texts(tokenizer_collator(*args, **kwargs),  # noqa
                                                                    key="caption_options",
                                                                    padding_value=tokenizer.pad_token_id)

    # To have this data cached, we could do:
    # image_shape = transform(Image.new(mode="RGB", size=(1, 1), color=ImageColor.getrgb("red"))).shape  # noqa
    # dataset=datasets.Dataset.from_generator(
    #     PicklableGenerator(lambda: dataset_factory(lambda x: transform(x).numpy()), generator_id=name),
    #     features=datasets.Features({"image": datasets.Array3D(shape=image_shape, dtype="float32"),
    #                                 "caption_options": datasets.Sequence(datasets.Value("string")),
    #                                 "class_name": datasets.Value("string")})).with_format("torch")
    # However, it takes a considerable time to iterate through the dataset (> 2 mins vs 30s),
    # causing an overhead in the evaluation (even when using the data loader).
    # I'm not sure why it happens.
    return {
        name: _create_data_loader(dataset=dataset_factory(transform), args=args, collate_fn=collator)
        for name, dataset_factory in [("vg-relation", AroVgRelation), ("vg-attribution", AroVgAttribution),
                                      ("coco-order", AroCocoOrder), ("flickr30k-order", AroFlickr30kOrder)]
    }


class ColorDataset(Dataset):
    def __init__(self, transform: Transform) -> None:
        self.colors = list(ImageColor.colormap.items())
        self.transform = transform

    def __getitem__(self, i: int) -> Mapping[str, Any]:
        name, code = self.colors[i]
        image = Image.new(mode="RGB", size=(1, 1), color=ImageColor.getrgb(code))
        return {"image": self.transform(image), "text": f"A {name} color picture."}

    def __len__(self) -> int:
        return len(self.colors)


@_eval_dataset("color")
def get_color_dataset(args: argparse.Namespace, transform: Transform, tokenizer: HFTokenizer) -> DataLoader:
    dataset = ColorDataset(transform=transform)
    collator = MappingDataCollatorWithTokenization(tokenizer, keys_to_tokenize="text")
    return _create_data_loader(dataset=dataset, args=args, collate_fn=collator)


@_eval_dataset("sugar-crepe")
def get_sugar_crepe_dataset(args: argparse.Namespace, transform: Transform,
                            tokenizer: HFTokenizer) -> Mapping[str, DataLoader]:
    tokenizer_collator = MappingDataCollatorWithTokenization(tokenizer, keys_to_tokenize="tested_labels")
    collator = lambda *args, **kwargs: _concatenate_tokenized_texts(tokenizer_collator(*args, **kwargs),  # noqa
                                                                    key="tested_labels",
                                                                    padding_value=tokenizer.pad_token_id)

    image_shape = transform(Image.new(mode="RGB", size=(1, 1), color=ImageColor.getrgb("red"))).shape

    hf_fs = HfFileSystem()

    return {
        k: _create_data_loader(dataset=(
            # Workaround to load the dataset much faster. The issue is that these datasets contain a ton of metadata,
            # which makes loading them and loading the cache version expensive. The reason for the slow load is that
            # the YAML parsing takes time, and can't be easily optimized using LibYAML. The cache loading also takes
            # time because the whole dataset metadata is hashed multiple times.
            datasets.load_dataset(
                "parquet", split="train",
                data_files=os.path.join(f"https://huggingface.co/datasets/HuggingFaceM4/SugarCrepe_{k}/resolve/main",
                                        "data",
                                        # The HF Hub path is different from the actual URL, so we just take the base.
                                        os.path.basename(hf_fs.ls(f"datasets/HuggingFaceM4/SugarCrepe_{k}/data"
                                                                  )[0]["name"])))
            # We could use `num_proc` along with `with datasets_spawn_mode():` to speed up the loading.
            # However, it's slower.
            # Loading the cached data also uses multiple processes and takes time to spawn them and especially to
            # concatenate their data.
            # Contrast this with using a single process to load the data from disk (it's way faster).
            # Note it'd probably also occur if we could manage to use "fork" instead of "spawn" for the data loading,
            # as the concatenation is the slowest part.
            # Also, it'd fork processes to load the data from the cache anyway, which feels overkill.
            .map(lambda sample: {"image": transform(sample["image"]).numpy()}, remove_columns="true_label",
                 features=datasets.Features({"image": datasets.Array3D(shape=image_shape, dtype="float32"),
                                             "tested_labels": datasets.Sequence(datasets.Value("string"))}),
                 desc=f"Preprocessing SugarCrepe/{k.capitalize()}")
            .with_format("torch")
        ), args=args, collate_fn=collator)
        for k in ["replace_obj", "replace_att", "replace_rel", "swap_obj", "swap_att", "add_obj", "add_att"]
    }


@_eval_dataset("svo-probes")
def get_svo_probes(args: argparse.Namespace, transform: Transform,
                   tokenizer: HFTokenizer) -> tuple[DataLoader, DataLoader]:
    dir_ = "/mnt/svo_probes"

    if not os.path.exists(dir_):
        with FileLock(dir_.rstrip("/") + ".lock"):
            if not os.path.exists(dir_):  # Retry in case the process had been locked.
                s3_sync("REPLACE_WITH_S3_PATH", dir_.rstrip("/") + "/")

    image_shape = transform(Image.new(mode="RGB", size=(1, 1), color=ImageColor.getrgb("red"))).shape

    available_filenames = os.listdir(dir_)

    image_dataset = (
        # Use `from_generator` instead of `from_list` to use the cache.
        datasets.Dataset.from_generator(lambda: ({"path": os.path.join(dir_, filename), "id": filename}
                                                 for filename in available_filenames))
        .map(lambda sample: {"id": sample["id"], "image": transform(Image.open(sample["path"]).convert("RGB")).numpy()},
             remove_columns=["path"], desc="Preprocessing SVO-Probes images",
             features=datasets.Features({"id": datasets.Value("string"),
                                         "image": datasets.Array3D(shape=image_shape, dtype="float32")}))
        .with_format("torch")
    )
    image_data_loader = _create_data_loader(dataset=image_dataset, args=args)

    available_filenames_set = set(available_filenames)

    dataset = (
        datasets.load_dataset("MichiganNLP/svo_probes", split="train")
        .map(lambda sample: {"pos_image_id": hashlib.md5(sample["pos_url"].encode("utf-8")).hexdigest(),
                             "neg_image_id": hashlib.md5(sample["neg_url"].encode("utf-8")).hexdigest(),
                             "neg_type": "s" if sample["subj_neg"] else ("v" if sample["verb_neg"] else "o")},
             remove_columns=["pos_triplet", "neg_triplet", "subj_neg", "verb_neg", "obj_neg", "pos_url", "neg_url"],
             desc="Preprocessing the negative type for SVO-Probes",
             features=datasets.Features({"sentence": datasets.Value("string"), "pos_image_id": datasets.Value("string"),
                                         "neg_image_id": datasets.Value("string"),
                                         "neg_type": datasets.Value("string")}))
        .filter(lambda sample: (sample["pos_image_id"] in available_filenames_set
                                and sample["neg_image_id"] in available_filenames_set),
                desc="Filtering SVO-Probes samples")
        .with_format("torch")
    )

    collator = MappingDataCollatorWithTokenization(tokenizer, keys_to_tokenize="sentence")

    data_loader = _create_data_loader(dataset=dataset, args=args, collate_fn=collator)

    return image_data_loader, data_loader


def _create_frame_sampler() -> UniformFrameSampler:
    return UniformFrameSampler(max_frames=4)


@_eval_dataset("didemo")
def get_didemo(args: argparse.Namespace, transform: Transform, tokenizer: HFTokenizer) -> DataLoader:
    dataset = Didemo(tokenizer=tokenizer, frame_sampler=_create_frame_sampler(), transform=transform)
    return _create_data_loader(dataset=dataset, args=args, collate_fn=dataset.collate)


@_eval_dataset("hmdb51")
def get_hmdb51(args: argparse.Namespace, transform: Transform, *_, **__) -> DataLoader:
    dataset = Hmdb(frame_sampler=_create_frame_sampler(), transform=transform)
    dataset.name = "HMDB51"
    return _create_data_loader(dataset=dataset, args=args, collate_fn=dataset.collate)


@_eval_dataset("msrvtt")
def get_msrvtt(args: argparse.Namespace, transform: Transform, tokenizer: HFTokenizer) -> DataLoader:
    dataset = MsrVtt(tokenizer=tokenizer, frame_sampler=_create_frame_sampler(), transform=transform)
    return _create_data_loader(dataset=dataset, args=args, collate_fn=dataset.collate)


@_eval_dataset("ucf101")
def get_ucf101(args: argparse.Namespace, transform: Transform, *_, **__) -> DataLoader:
    dataset = Ucf(frame_sampler=_create_frame_sampler(), transform=transform)
    dataset.name = "UCF101"
    return _create_data_loader(dataset=dataset, args=args, collate_fn=dataset.collate)


@_eval_dataset("youcook2")
def get_youcook2(args: argparse.Namespace, transform: Transform, tokenizer: HFTokenizer) -> DataLoader:
    dataset = YouCook2(tokenizer=tokenizer, frame_sampler=_create_frame_sampler(), transform=transform)
    return _create_data_loader(dataset=dataset, args=args, collate_fn=dataset.collate)


def filter_no_caption_or_no_image(sample: Mapping[str, Any]) -> bool:
    has_caption = "txt" in sample
    has_image = "png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample
    return has_caption and has_image


def _add_extra_caption(sample: MutableMapping[str, Any]) -> Mapping[str, Any]:
    sample["extra_caption"] = sample["json"]["top_caption"]
    return sample


def _replace_with_extra_caption(sample: MutableMapping[str, Any]) -> Mapping[str, Any]:
    sample["txt"] = sample["json"]["top_caption"]
    return sample


def _concat_text_and_extra_caption(
        sample: tuple[torch.Tensor, Sequence[str], Sequence[str]]) -> tuple[torch.Tensor, Sequence[str]]:
    return sample[0], interleave_seqs(sample[1], sample[2])


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


def get_wds_dataset(args: argparse.Namespace, transform: Transform, tokenizer: HFTokenizer, is_train: bool,
                    epoch: int = 0, floor: bool = False) -> DataInfo:
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    "Currently, the number of dataset samples must be specified for the training dataset. "
                    "Please specify it via `--train-num-samples` if no dataset length info is present.")
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or get_dataset_size(input_shards)[0] or 0

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors:
        assert resampled, ("--train_data_upsampling_factors is only supported when sampling with replacement"
                           " (with --dataset-resampled).")

    pipeline = [ResampledShards2(input_shards, weights=args.train_data_upsampling_factors, deterministic=True,
                                 epoch=shared_epoch) if resampled else wds.SimpleShardList(input_shards)]

    # At this point, we have an iterator over all the shards.

    if is_train:
        if not resampled:
            pipeline.extend([
                # Note that the shards aren't that many, so we can shuffle them altogether. This allows us to have more
                # randomness if the number of samples to use is tiny compared to the total. Otherwise, we'd use a sample
                # out of the very first ones.
                #
                # And note that we don't need the epoch info for determinism, because the random object is going to be
                # the same for future epochs, so that the shuffling is going to be deterministically different.
                detshuffle2(bufsize=_SHARD_SHUFFLE_SIZE, initial=_SHARD_SHUFFLE_INITIAL, seed=args.seed,
                            epoch=shared_epoch) if args.shuffle_train_with_buffer
                else shuffle_all(rng=random.Random(args.seed)),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        # At this point, we have an iterator over the shards assigned to each worker at each node.
        pipeline.extend([
            tarfile_to_samples_no_throw_and_close_file,
            wds.shuffle(bufsize=_SAMPLE_SHUFFLE_SIZE, initial=_SAMPLE_SHUFFLE_INITIAL),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # At this point, we have an iterator over the shards assigned to each worker.
            wds.tarfile_to_samples(handler=log_and_continue),
        ])

    batch_size = args.batch_size if is_train else args.eval_batch_size

    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
    ])

    if is_train:
        assert not args.add_extra_caption or not args.replace_with_extra_caption, \
            "Only one of `--add-extra-caption` and `--replace-with-extra-caption` can be specified."

        if args.add_extra_caption:
            pipeline.append(wds.map(_add_extra_caption))
        elif args.replace_with_extra_caption:
            pipeline.append(wds.map(_replace_with_extra_caption))

    pipeline.extend([
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=transform),
        wds.to_tuple(*(("image", "text") + (("extra_caption",) if is_train and args.add_extra_caption else ()))),
        wds.batched(batch_size, partial=not is_train),
    ])

    if is_train and args.add_extra_caption:
        pipeline.append(wds.map(_concat_text_and_extra_caption))

    if is_train and args.add_random_text_hard_negatives:
        # It's better to add the random text hard negatives once we have the batch, so we make sure they are added at
        # the end of it.
        pipeline.append(wds.map_tuple(
            None, lambda t: add_random_text_hard_negatives(t, style=args.add_random_text_hard_negatives)))

    # Image captions are generally short, and we want to be memory-efficient here.
    # So we use a short context length.
    tokenization_fn = (functools.partial(tokenizer, max_length=min(DEFAULT_CONTEXT_LENGTH, tokenizer.model_max_length))
                       if is_train else tokenizer)
    pipeline.append(wds.map_tuple(None, tokenization_fn))

    if is_train and args.add_extra_caption:
        pipeline.append(wds.map_tuple(None, lambda t: t.reshape(-1, 2, t.shape[1])))

    dataset_class = ResumableDataPipeline if is_train and not resampled else wds.DataPipeline
    dataset = dataset_class(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, "number of shards must be >= total workers"
        # Roll over and repeat a few samples to get the same number of full batches on each node.
        round_fn = math.floor if floor else math.ceil
        global_batch_size = batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # Per data loader worker.
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size

        # FIXME: not clear which approach is better, `with_epoch` applied to the dataset or the data loader?
        #  See https://github.com/webdataset/webdataset/issues/169.
        dataset = dataset.with_epoch(num_worker_batches)  # Each worker is iterating over this.
    else:
        # Last batches are partial, the eval is done on a single (primary) node.
        num_batches = math.ceil(num_samples / batch_size)

    # Not using persistent workers with wds because it may cause processes to freeze or reach OOM.
    # See https://github.com/webdataset/webdataset/issues/90
    data_loader = wds.WebLoader(dataset, batch_size=None, num_workers=args.workers, pin_memory=True)

    # Add the metadata to the data loader instance, for convenience.
    data_loader.num_batches = num_batches
    data_loader.num_samples = num_samples

    return DataInfo(data_loader=data_loader, shared_epoch=shared_epoch)


def _create_data_loader(dataset: Dataset, args: argparse.Namespace, is_train: bool = False,
                        collate_fn: Callable[[list], Any] | None = None) -> DataLoader:
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None

    data_loader = DataLoader(dataset, batch_size=args.batch_size if is_train else args.eval_batch_size,
                             shuffle=is_train and sampler is None, num_workers=args.workers, pin_memory=True,
                             persistent_workers=args.workers > 0, sampler=sampler, drop_last=is_train,
                             collate_fn=collate_fn)

    data_loader.num_samples = len(dataset) if isinstance(dataset, Sized) else None
    data_loader.num_batches = len(data_loader) if data_loader.num_samples else None

    return data_loader


def get_csv_dataset(args: argparse.Namespace, transform: Transform, tokenizer: HFTokenizer, is_train: bool,
                    *_, **__) -> DataInfo:
    assert not args.add_random_text_hard_negatives, \
        "Adding random text hard negatives is not supported for the CSV datasets."

    input_filename = args.train_data if is_train else args.val_data
    assert input_filename

    # Image captions are generally short, and we want to be memory-efficient here.
    # So we use a short context length.
    tokenization_fn = (functools.partial(tokenizer, max_length=min(DEFAULT_CONTEXT_LENGTH, tokenizer.model_max_length))
                       if is_train else tokenizer)
    dataset = CsvDataset(input_filename, transform=transform, tokenizer=tokenization_fn, img_key=args.csv_img_key,
                         caption_key=args.csv_caption_key, sep=args.csv_separator)

    return DataInfo(_create_data_loader(dataset=dataset, args=args, is_train=is_train))


class SyntheticDataset(Dataset):
    def __init__(self, transform: Transform, tokenizer: HFTokenizer, image_size: tuple[int, int] = (224, 224),
                 caption: str = "Dummy caption", dataset_size: int = 100) -> None:
        self.transform = transform
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new("RGB", image_size)
        self.dataset_size = dataset_size

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(self.image), self.tokenizer([self.caption])[0]


def get_synthetic_dataset(args: argparse.Namespace, transform: Transform, tokenizer: HFTokenizer, is_train: bool,
                          *_, **__) -> DataInfo:
    assert not args.add_random_text_hard_negatives, \
        "Adding random text hard negatives is not supported for the synthetic datasets."

    assert isinstance(transform, Compose)
    image_size = transform.transforms[0].size

    # Image captions are generally short, and we want to be memory-efficient here.
    # So we use a short context length.
    tokenization_fn = (functools.partial(tokenizer, max_length=min(DEFAULT_CONTEXT_LENGTH, tokenizer.model_max_length))
                       if is_train else tokenizer)
    dataset = SyntheticDataset(transform=transform, tokenizer=tokenization_fn, image_size=image_size,
                               dataset_size=args.train_num_samples)

    return DataInfo(_create_data_loader(dataset=dataset, args=args, is_train=is_train))


def _get_dataset_fn(data_path: str, dataset_type: Literal["auto", "csv", "synthetic", "webdataset"]) -> Callable:
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.rsplit(".", maxsplit=1)[-1].lower()
        if ext in {"csv", "tsv"}:
            return get_csv_dataset
        elif ext in {"tar"}:
            return get_wds_dataset
        else:
            raise ValueError(f"Tried to figure out dataset type, but failed for the extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


@_eval_dataset("val")
def get_val(args: argparse.Namespace, transform: Transform, tokenizer: HFTokenizer) -> DataInfo:
    return _get_dataset_fn(args.val_data, args.dataset_type)(args, transform, tokenizer, is_train=False)


def get_data(args: argparse.Namespace, train_transform: Transform, val_transform: Transform, tokenizer: HFTokenizer,
             epoch: int = 0) -> Mapping[str, Any]:
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = _get_dataset_fn(args.train_data, args.dataset_type)(args=args, transform=train_transform,
                                                                            tokenizer=tokenizer, is_train=True,
                                                                            epoch=epoch)

    for i, name in enumerate(args.eval_benchmarks):
        # TODO: we could support the following two features:
        #  * Computing the time it takes to run each eval dataset,
        #    then load to minimize the max time per rank.
        #  * Support a flag to have nodes that only do evaluation, and some that only do training.
        if i % args.world_size == args.rank:
            try:
                data[name] = _get_data_one(name, args, val_transform, tokenizer)
            except KeyboardInterrupt:
                raise
            except:  # noqa
                logging.error(f"Failed to load the {name} dataset. See the error below", exc_info=1)

    return data
