# Adapted from https://github.com/bryant1410/fit-clip/blob/285c6b1/aligner/data/video_reader.py
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

import decord
import torch


class VideoReader(ABC):
    def __init__(self, path: str) -> None:  # noqa
        pass

    def __call__(self, indices: Sequence[int]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def time_to_indices(self, time: float) -> torch.Tensor:
        """The input can be a single value or a sequence of values."""
        raise NotImplementedError

    @abstractmethod
    def get_avg_fps(self) -> float:
        raise NotImplementedError


decord.bridge.set_bridge("torch")


class DecordVideoReader(VideoReader):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        # Using `width` and `height` from VideoReader is actually faster because it resizes while decoding.
        # However, it doesn't preserve the aspect ratio (even if setting only one of the two).
        # Using the GPU for decoding may actually be faster,
        # but it isn't trivial how to optimize the whole data loading process so to accomplish it.
        try:
            self.video_reader = decord.VideoReader(path, num_threads=1)
        except decord.DECORDError:
            logging.error(f"An error occurred when trying to load the video with path {path}.")
            self.video_reader = None

    def __call__(self, indices: Sequence[int]) -> torch.Tensor:
        if self.video_reader:
            try:
                return self.video_reader.get_batch(indices)  # noqa
            except decord.DECORDError:
                # FIXME: change the handle for the path? How to get the path?
                logging.error(f"An error occurred when trying to read the video with path {self.video_reader._handle}"
                              f" and indices {indices}.")

        return torch.zeros(len(indices), 256, 256, 3)

    def __len__(self) -> int:
        return len(self.video_reader) if self.video_reader else 1

    def time_to_indices(self, time: float) -> torch.Tensor:
        times = (torch.from_numpy(self.video_reader.get_frame_timestamp(range(len(self)))).mean(dim=-1)
                 if self.video_reader else torch.zeros(1))
        indices = torch.searchsorted(times, time)
        return indices.where((indices == 0) | (times[indices - 1] - time <= time - times[indices - 1]), indices - 1)

    def get_avg_fps(self) -> float:
        return self.video_reader.get_avg_fps() if self.video_reader else 1
