# Adapted from https://github.com/bryant1410/fitclip/blob/a55ec2b/aligner/data/
import asyncio
import functools
import glob
import json
import logging
import os
import random
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Generic, Literal, TypeVar

import pandas as pd
import rarfile
import torch
from cached_path import cached_path, find_latest_cached
from filelock import FileLock
from torch.utils.data import Dataset, default_collate
from torchvision.transforms.functional import to_pil_image

from open_clip import HFTokenizer, Transform
from training import DecordVideoReader, FrameSampler
from training.collate import MappingDataCollatorWithTokenization, pad_collate
from training.video_utils import get_sorted_videos_in_dir
from training.zero_shot_metadata import OPENAI_UCF101_TEMPLATES

T = TypeVar("T")


def get_filename_without_extension(path: str) -> str:
    return os.path.basename(path).split(".", maxsplit=1)[0]


# TODO: support taking multiple clips per video, where they are chosen according to some strategy.
class VideoDataset(Dataset, Generic[T], ABC):
    def __init__(self, *, video_paths: Iterable[str], frame_sampler: FrameSampler, transform: Transform,
                 video_key_name: str = "image", target_key_name: str = "target", pad_batch: bool = True,
                 cache: bool = False) -> None:
        super().__init__()
        self.video_paths = video_paths if isinstance(video_paths, Sequence) else list(video_paths)
        self.frame_sampler = frame_sampler
        self.transform = transform
        self.video_key_name = video_key_name
        self.target_key_name = target_key_name
        self.pad_batch = pad_batch
        self.cache = cache

    @abstractmethod
    def _get_target(self, video_idx: int) -> T:
        """Returns the target associated with `self.video_paths[video_idx]`."""
        raise NotImplementedError

    @functools.lru_cache
    def _get_video_id(self, video_idx: int) -> str:
        return get_filename_without_extension(self.video_paths[video_idx])

    def _get_times(self, video_idx: int) -> tuple[float | None, float | None]:  # noqa
        """Returns the video clip start and end times for the given video index, if any."""
        return None, None

    @functools.lru_cache(maxsize=None)
    def _cached_get_item(self, video_idx: int) -> Mapping[str, torch.Tensor | str | T]:
        path = self.video_paths[video_idx]
        video_id = self._get_video_id(video_idx)
        video_reader = DecordVideoReader(path)

        start_time, end_time = self._get_times(video_idx)

        start_frame_idx = 0 if start_time is None else video_reader.time_to_indices(start_time).item()
        end_frame_idx = len(video_reader) - 1 if end_time is None else video_reader.time_to_indices(end_time).item()

        indices = self.frame_sampler(start_frame_idx, end_frame_idx, fps=video_reader.get_avg_fps())
        indices = indices if isinstance(indices, Sequence) else list(indices)
        frames = video_reader(indices).permute(0, 3, 1, 2)

        return {
            self.target_key_name: self._get_target(video_idx),
            "video_id": video_id,
            # FIXME: avoid this extra transformation to a PIL image, batch-transform everything,
            #  and make sure all have the same number of frames:
            self.video_key_name: torch.stack([self.transform(to_pil_image(frame)) for frame in frames]),
        }

    def __getitem__(self, video_idx: int) -> Mapping[str, torch.Tensor | str | T]:
        # Note we have to explicitly pass `self` to the wrapped one.
        fn = self._cached_get_item if self.cache else functools.partial(self._cached_get_item.__wrapped__, self)  # noqa
        return fn(video_idx)

    def __len__(self) -> int:
        return len(self.video_paths)

    def collate(self, batch: Sequence) -> Any:
        return (pad_collate if self.pad_batch else default_collate)(batch)


class Hmdb(VideoDataset):
    HMDB_TAGS = {"train": 1, "test": 2}

    def __init__(self, video_dir_or_url: str = "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/"
                                               "hmdb51_org.rar!",
                 splits_dir_or_url: str = "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/"
                                          "test_train_splits.rar!testTrainMulti_7030_splits/",
                 split: Literal[1, 2, 3] = 1, tag_string: Literal["train", "test"] = "test", **kwargs) -> None:
        tag_number = self.HMDB_TAGS[tag_string]

        video_dir = cached_path(video_dir_or_url, extract_archive=True)

        class_names = []

        for path in glob.iglob(os.path.join(video_dir, f"*.rar")):
            filename_without_extension = get_filename_without_extension(path)

            class_names.append(filename_without_extension.replace("_", " "))

            class_dir = os.path.join(video_dir, filename_without_extension)
            if not os.path.exists(class_dir):
                with FileLock(path + ".lock"):
                    if not os.path.exists(class_dir):
                        logging.info(f"Extracting {path}…")
                        with rarfile.RarFile(path) as file:
                            # Note the archive file already has a directory with the same name.
                            file.extractall(video_dir)

        self.class_name_to_idx = {class_name: i for i, class_name in enumerate(sorted(class_names))}
        self.classes = self.class_name_to_idx.keys()

        self.templates = OPENAI_UCF101_TEMPLATES

        video_paths = []
        for path in glob.iglob(os.path.join(cached_path(splits_dir_or_url, extract_archive=True),
                                            f"*_test_split{split}.txt")):
            class_name = os.path.basename(path).rsplit("_", maxsplit=2)[0]
            with open(path) as file:
                for line in file:
                    filename, file_tag = line.strip().split(maxsplit=1)
                    file_tag = int(file_tag)
                    if file_tag == tag_number:
                        video_paths.append(os.path.join(video_dir, class_name, filename))

        super().__init__(video_paths=video_paths, **kwargs)

    def _get_video_id(self, video_idx: int) -> str:
        path = self.video_paths[video_idx]
        dir_path, filename = os.path.split(path)
        dir_name = os.path.basename(dir_path)
        return os.path.join(dir_name, filename)

    def _get_target(self, video_idx: int) -> int:
        video_id = self._get_video_id(video_idx)
        dir_path = os.path.dirname(video_id)
        class_name = dir_path.replace("_", " ")
        return self.class_name_to_idx[class_name]


class Ucf(VideoDataset):
    BASE_URL = "https://www.crcv.ucf.edu/data/UCF101/"
    SPLIT_REL_DIR = "UCF101TrainTestSplits-RecognitionTask.zip!ucfTrainTestlist"

    def __init__(self, class_names_path_or_url: str = os.path.join(BASE_URL, SPLIT_REL_DIR, "classInd.txt"),
                 file_list_path_or_url: str = os.path.join(BASE_URL, SPLIT_REL_DIR, "testlist01.txt"),
                 video_dir_path_or_url: str = os.path.join(BASE_URL, "UCF101.rar!UCF-101"), **kwargs) -> None:
        with open(cached_path(class_names_path_or_url, extract_archive=True)) as file:
            self.class_name_to_idx = {}
            for line in file:
                id_, dir_name = line.strip().split(maxsplit=1)
                self.class_name_to_idx[self._dir_to_class_name(dir_name)] = int(id_) - 1
            assert list(self.class_name_to_idx.values()) == list(range(101))

        self.classes = self.class_name_to_idx.keys()
        self.templates = OPENAI_UCF101_TEMPLATES

        video_dir = cached_path(video_dir_path_or_url, extract_archive=True)
        with open(cached_path(file_list_path_or_url, extract_archive=True)) as file:
            video_ids = (stripped_line for line in file if (stripped_line := line.strip()))
            super().__init__(video_paths=(os.path.join(video_dir, rel_path) for rel_path in video_ids), **kwargs)

    def _get_video_id(self, video_idx: int) -> str:
        path = self.video_paths[video_idx]
        dir_, filename = os.path.split(path)
        dir_dir = os.path.basename(dir_)
        return os.path.join(dir_dir, filename)

    @staticmethod
    def _dir_to_class_name(dir_: str) -> str:
        return " ".join(re.findall(r"[a-zA-Z][^A-Z]*", dir_))

    def _get_target(self, video_idx: int) -> int:
        video_id = self._get_video_id(video_idx)
        class_name = self._dir_to_class_name(os.path.dirname(video_id))
        return self.class_name_to_idx[class_name]


class VideoTextDataset(VideoDataset, ABC):
    def __init__(self, *, tokenizer: HFTokenizer, target_key_name: str = "text", **kwargs) -> None:
        super().__init__(target_key_name=target_key_name, **kwargs)
        self.collate = MappingDataCollatorWithTokenization(tokenizer, target_key_name,
                                                           default_collate_fn=getattr(self, "collate", default_collate))


class Didemo(VideoTextDataset):
    @staticmethod
    async def _cached_path(path_or_url: str) -> str:
        # We only download some videos and not the whole folder.
        # But if it's already cached, we avoid sending a HEAD request.
        # This is an issue if the file was updated, but we assume it won't happen.
        return str(find_latest_cached(path_or_url) or cached_path(path_or_url))

    async def _cached_paths(self, path_or_urls: Iterable[str]) -> Sequence[str]:
        return await asyncio.gather(*(self._cached_path(path_or_url) for path_or_url in path_or_urls))

    def __init__(self, video_dir_or_url: str = "https://multimedia-commons.s3-us-west-2.amazonaws.com/data/videos/mp4/",
                 hash_list_path_or_url: str = "https://raw.githubusercontent.com/LisaAnne/LocalizingMoments/master/"
                                              "data/yfcc100m_hash.txt",
                 annotations_path_or_url: str = "https://raw.githubusercontent.com/LisaAnne/LocalizingMoments/master/"
                                                "data/val_data.json", **kwargs) -> None:
        with open(cached_path(annotations_path_or_url)) as file:
            description_list_by_id = defaultdict(list)
            for video in json.load(file):
                description_list_by_id[video["video"]].append(video["description"])

        self.description_paragraph_by_id = {video_id: " ".join(descriptions)
                                            for video_id, descriptions in description_list_by_id.items()}

        with open(cached_path(hash_list_path_or_url)) as file:
            hash_by_flickr_id = dict(line.strip().split("\t", maxsplit=1) for line in file)

        video_id_to_path_or_url = {}

        for video_id in self.description_paragraph_by_id:
            flickr_id = video_id.split("_")[1]
            hash_ = hash_by_flickr_id[flickr_id]
            video_id_to_path_or_url[video_id] = os.path.join(video_dir_or_url, hash_[:3], hash_[3:6], f"{hash_}.mp4")

        self.video_path_to_id = dict(zip(asyncio.run(self._cached_paths(video_id_to_path_or_url.values())),
                                         video_id_to_path_or_url.keys()))

        super().__init__(video_paths=self.video_path_to_id.keys(), **kwargs)

    def _get_target(self, video_idx: int) -> str:
        video_path = self.video_paths[video_idx]
        video_id = self.video_path_to_id[video_path]
        return self.description_paragraph_by_id[video_id]


class MsrVtt(VideoTextDataset):
    BASE_URL = "https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip!MSRVTT"

    def __init__(self, *, dir_path_or_url: str = os.path.join(BASE_URL, "videos", "all"),
                 file_list_path_or_url: str = os.path.join(BASE_URL, "structured-symlinks",
                                                           "val_list_jsfusion.txt"),  # 1K-A split
                 annotations_path: str = os.path.join(BASE_URL, "annotation", "MSR_VTT.json"),
                 caption_sampling_strategy: Literal["all", "first", "random"] = "first", **kwargs) -> None:
        with open(cached_path(file_list_path_or_url, extract_archive=True)) as file:
            video_ids = {stripped_line for line in file if (stripped_line := line.strip())}  # noqa

        video_paths = (path
                       for path in get_sorted_videos_in_dir(cached_path(dir_path_or_url, extract_archive=True))
                       if os.path.basename(path).split(".", maxsplit=1)[0] in video_ids)

        super().__init__(video_paths=video_paths, **kwargs)

        self.caption_sampling_strategy = caption_sampling_strategy

        with open(cached_path(annotations_path, extract_archive=True)) as file:
            metadata = json.load(file)

        self.video_info = pd.DataFrame(metadata["annotations"])
        self.video_info.set_index("image_id", inplace=True)

    def _get_target(self, video_idx: int) -> str | Sequence[str]:
        video_id = self._get_video_id(video_idx)
        captions = self.video_info.loc[video_id, "caption"]
        if self.caption_sampling_strategy == "all":
            # FIXME: this strategy doesn't work because there's a different number of captions per video,
            #   which makes the collation process to fail.
            #   To make it work, we could pad and carry a mask.
            return captions.tolist()
        elif self.caption_sampling_strategy == "first":
            return captions.iloc[0]
        elif self.caption_sampling_strategy == "random":
            return random.choice(captions)
        else:
            raise ValueError(f"Invalid choice of caption sampling strategy: {self.caption_sampling_strategy}")


class YouCook2(VideoTextDataset):
    def __init__(self, video_info_path_or_url: str = "https://raw.githubusercontent.com/antoine77340/MIL-NCE_HowTo100M/"
                                                     "master/csv/validation_youcook.csv",
                 video_dir_or_url: str = "https://www.rocq.inria.fr/cluster-willow/amiech/Youcook2_val.zip!validation",
                 **kwargs) -> None:
        self.video_info = pd.read_csv(cached_path(video_info_path_or_url), dtype={"task": str})

        video_dir = cached_path(video_dir_or_url, extract_archive=True)
        video_paths = (next(glob.iglob(os.path.join(video_dir, row.task, f"{row.video_id}.*")))
                       for _, row in self.video_info.iterrows())

        super().__init__(video_paths=video_paths, **kwargs)

    def _get_target(self, video_idx: int) -> str:
        return self.video_info.loc[video_idx, "text"]

    def _get_times(self, video_idx: int) -> tuple[float | None, float | None]:
        row = self.video_info.loc[video_idx]
        return row.start, row.end
