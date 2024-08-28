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
        """Returns the index of the closest frame to the input `time`. The input is expressed in seconds and can be a
        scalar or a 1-D tensor. The output has the same shape as the input.
        """
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

    def time_to_indices(self, time: float | torch.Tensor, end_time_tolerance: float = 1) -> torch.Tensor:
        # Decord provides us with the start and end of each frame. We could simply check where `time` falls in.
        # However, it gets more complex, as `time` can possibly fall in between two frames.
        # The easiest approach I've found is to take the mean, supposing the frame is instantaneous,
        # to then the closest frame index.
        available_times = (torch.from_numpy(self.video_reader.get_frame_timestamp(range(len(self)))).mean(dim=-1)
                           if self.video_reader else torch.zeros(1))
        # `available_times` is the mean of the start and end time for each frame, sorted.

        if (time > available_times[-1] + end_time_tolerance).any():
            logging.warning(f"The sought time(s) {time}s is significantly larger than the last frame time of"
                            f" {available_times[-1]}s.")

        # A batched binary search is the fastest way I could think of to do this.
        indices = torch.searchsorted(available_times, time)
        # `indices` indicates where `time` should be inserted in `available_times` to maintain the order.
        # Thus, `available_times[indices - 1] < time <= available_times[indices]`,
        # as long as `0 < indices < len(available_times)`.

        indices.clamp_(max=len(available_times) - 1)

        # Now, we have to check if `time` is closer to the previous or the next frame.
        # For each element in `time`, we return the closest time between that of `indices` and `indices - 1`.
        return indices.where((indices == 0) | (available_times[indices] - time <= time - available_times[indices - 1]),
                             indices - 1)

    def get_avg_fps(self) -> float:
        return self.video_reader.get_avg_fps() if self.video_reader else 1
